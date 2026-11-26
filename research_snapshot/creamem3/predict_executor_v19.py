"""Score-blind formal prediction executor for the ten-cell MemEye v19 matrix.

The module has two deliberately separate surfaces:

* ``seal_predict_budget`` is zero-provider.  It opens the sealed build graphs,
  counts the worst-case 500-document Qwen batches, and writes the exact static
  ceiling that an independent reviewer must approve.
* ``execute_predict_with_backends`` accepts injected, already-gated providers
  and performs shared query planning/embedding, MGTM-matched retrieval, exact
  306/65 Reader routing, and ten isolated 371-row prediction seals.

``run_matrix_phase`` is the formal adapter used by the main orchestrator.  It
never reads raw MemEye task files, gold answers, historical predictions,
verdicts, judge caches, or calibration artifacts.
"""

from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import importlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from creamem3 import run_memeye_full_arch_matrix_v19 as matrix


PREDICT_EXECUTOR_SCHEMA = "creamem3.memeye_full_arch_predict_executor.v19.0"
PREDICT_BUDGET_SCHEMA = "creamem3.memeye_predict_budget.v19.0"
PREDICT_BUDGET_GATE_SCHEMA = "creamem3.memeye_predict_budget_gate.v19.0"
PREDICTION_CELL_SCHEMA = "creamem3.memeye_prediction_cell.v19.0"
QUERY_EMBEDDING_SCHEMA = "creamem3.memeye_query_embeddings.v19.0"
QWEN_CACHE_SCHEMA = "creamem3.memeye_qwen_formal_cache.v19.0"

SUMMARY_N = 10
DENSE_N = 20
BM25_N = 20
QWEN_MAX_DOCUMENTS = 500
MAX_PROVIDER_ATTEMPTS = 3
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_DIMENSION = 3072

SS_QUERY_AUTHORITY = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\multimodal\ss_query.py"
)
SS_QUERY_SHA256 = "f81a1f5ef321c29d59532932ed90d4147c766d8d3dbae2784d14c0a5ef2dab3c"
STRICT_PLANNER_SHA256 = "ea8ecdacb4107f1f46e7d6b84a25ee051bd44f7847381756e07c8cdb00263da4"
SEMANTIC_PLANNER_SHA256 = "daa68edab125d482dfafd94e050dabfc5c455077fd677f4549d6a4039bffbd36"
G25_RUNTIME_ADAPTER_SHA256 = (
    "1880f0366bbad7db4ce03cd8a7d8e429ad67e78bba90a05a95a3ac5ff5476872"
)
BUILD_EXECUTOR_SHA256 = (
    "a28f7c0397f20f4729c98949cabb2d13efa62361f28010c8ae87658c83aee3c7"
)
QWEN_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)
QWEN_MODEL = "qwen3-rerank"

ARCHITECTURE_TO_BINDING = {
    "mgtm_matched": "mgtm_matched",
    "session_only": "session_only",
    "online_base": "online_base",
    "offline_base": "offline_base",
    "online_operation_context_v3": "online_base",
}


def _ordered_parallel_map(
    function: Callable[[Any], Any],
    values: Sequence[Any],
    *,
    max_workers: int,
) -> list[Any]:
    """Map concurrently while retaining the caller's exact row order.

    The default executor path remains serial.  The isolated runner opts into
    four workers; ``Executor.map`` deliberately returns results in input order
    so prediction artifacts and their hashes are independent of completion
    timing.
    """

    workers = int(max_workers)
    if workers < 1:
        raise PredictExecutorError("prediction max_workers must be positive")
    if workers == 1:
        return [function(value) for value in values]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="memeye-predict"
    ) as pool:
        return list(pool.map(function, values))
RETRIEVAL_ARCHITECTURES = (
    "mgtm_matched",
    "session_only",
    "online_base",
    "offline_base",
)


class PredictExecutorError(RuntimeError):
    """Protocol, artifact, provider, or cache failure."""


class ExplicitQwenProviderError(PredictExecutorError):
    def __init__(self, message: str, *, status_code: int, retryable: bool) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.retryable = bool(retryable)


class IndeterminateQwenProviderError(PredictExecutorError):
    """A request may have reached DashScope; automatic recall is forbidden."""


@dataclass(frozen=True)
class TextReply:
    text: str
    request_sha256: str
    response_sha256: str
    cache_hit: bool = False


@dataclass(frozen=True)
class RerankReply:
    aligned_scores: tuple[float, ...]
    request_sha256: str
    response_sha256: str
    cache_hit: bool = False
    cache_identity: str = ""


class TextBackend(Protocol):
    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        json_mode: bool,
        max_output_tokens: int,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def usage_receipt(self) -> Mapping[str, Any]: ...


class EmbeddingBackend(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        tag: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Sequence[Sequence[float]]: ...

    def usage_receipt(self) -> Mapping[str, Any]: ...


class RerankBackend(Protocol):
    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def usage_receipt(self) -> Mapping[str, Any]: ...


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise PredictExecutorError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PredictExecutorError(f"{label} is not an object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise PredictExecutorError(f"write-once artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validate_vector(vector: Sequence[Any]) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != EMBEDDING_DIMENSION or not all(
        math.isfinite(value) for value in values
    ):
        raise PredictExecutorError("query embedding dimension/value drift")
    return values


def _reply(value: Any) -> TextReply:
    text = str(getattr(value, "text", ""))
    request_sha = str(getattr(value, "request_sha256", ""))
    response_sha = str(getattr(value, "response_sha256", ""))
    if not text.strip() or not request_sha or response_sha != _sha256(text):
        raise PredictExecutorError("text provider reply binding drift")
    return TextReply(text, request_sha, response_sha, bool(getattr(value, "cache_hit", False)))


def _rerank_reply(value: Any, expected: int) -> RerankReply:
    raw_scores = getattr(value, "aligned_scores", value)
    scores = tuple(float(item) for item in raw_scores)
    request_sha = str(getattr(value, "request_sha256", ""))
    response_sha = str(getattr(value, "response_sha256", ""))
    if (
        len(scores) != expected
        or not all(math.isfinite(item) for item in scores)
        or not request_sha
        or not response_sha
    ):
        raise PredictExecutorError("Qwen rerank reply shape/binding drift")
    return RerankReply(
        scores,
        request_sha,
        response_sha,
        bool(getattr(value, "cache_hit", False)),
        str(getattr(value, "cache_identity", "")),
    )


def validate_predict_authorities(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every pure retrieval/Reader dependency before importing it."""

    authorities = protocol["authority_files"]
    rows: dict[str, Any] = {}
    for name in (
        "historical_eval_memeye",
        "focused_reader",
        "structured_fused_reader",
        "historical_mgtm_embedding",
        "historical_qwen_reranker",
        "graph_and_retrieval",
        "online_operation_context",
        "g25_authority_client",
    ):
        item = authorities[name]
        path = matrix._resolve_authority_path(str(item["path"]))
        actual = _file_sha256(path)
        if actual != item["sha256"]:
            raise PredictExecutorError(f"predict authority drift: {name}")
        rows[name] = {"path": str(path), "sha256": actual}
    local = {
        "ss_query": (SS_QUERY_AUTHORITY, SS_QUERY_SHA256),
        "strict_semantic_planner": (
            Path(__file__).with_name("strict_semantic_planner_v8.py"),
            STRICT_PLANNER_SHA256,
        ),
        "semantic_query_planner": (
            Path(__file__).with_name("semantic_query_planner_v6.py"),
            SEMANTIC_PLANNER_SHA256,
        ),
        "official_g25_runtime_adapter": (
            Path(__file__).with_name("official_g25_runtime_adapter_v19.py"),
            G25_RUNTIME_ADAPTER_SHA256,
        ),
        "build_executor": (
            Path(__file__).with_name("build_executor_v19.py"),
            BUILD_EXECUTOR_SHA256,
        ),
    }
    for name, (path, expected) in local.items():
        actual = _file_sha256(path)
        if actual != expected:
            raise PredictExecutorError(f"predict authority drift: {name}")
        rows[name] = {"path": str(path.resolve()), "sha256": actual}
    return {"status": "pass", "rows": rows, "rows_sha256": _sha256(rows)}


def _reader_authority(protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_predict_authorities(protocol)
    authorities = protocol["authority_files"]
    focused_path = matrix._resolve_authority_path(authorities["focused_reader"]["path"])
    fused_path = matrix._resolve_authority_path(
        authorities["structured_fused_reader"]["path"]
    )
    eval_path = matrix._resolve_authority_path(
        authorities["historical_eval_memeye"]["path"]
    )
    focused_source = focused_path.read_text(encoding="utf-8")
    eval_source = eval_path.read_text(encoding="utf-8")
    fused_source = fused_path.read_text(encoding="utf-8")
    focused_prompt, _ = matrix._literal_node(focused_source, "FOCUSED_ANSWER_PROMPT")
    open_prompt, _ = matrix._literal_node(eval_source, "MEMEYE_OPEN_QA_PROMPT")
    fused_prompt, _ = matrix._literal_node(fused_source, "PROMPT")
    instruction, _ = matrix._literal_node(eval_source, "MEMEYE_OPEN_RETRIEVAL_INSTRUCT")
    reader = protocol["reader_protocol"]
    expected = {
        "focused": reader["evidence_prompt_sha256"],
        "open": reader["final_prompt_sha256"],
        "fused": reader["structured_fused_prompt_sha256"],
        "instruction": protocol["provider_availability_protocol"]
        ["reranker_prepredict_smoke"]["historical_instruction_sha256"],
    }
    values = {
        "focused": str(focused_prompt),
        "open": str(open_prompt),
        "fused": str(fused_prompt),
        "instruction": str(instruction),
    }
    if any(_sha256(values[key]) != expected[key] for key in expected):
        raise PredictExecutorError("Reader/retrieval literal authority drift")

    # Execute only the two pinned pure renderer functions.  Their containing
    # file was hash-gated above, and the renderer source hash is in protocol.
    import ast

    module_ast = ast.parse(focused_source)
    wanted = [
        copy.deepcopy(node)
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"list_section", "render_aggregation"}
    ]
    namespace: dict[str, Any] = {"json": json}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(focused_path), "exec"), namespace)

    # The 0.4533 routed Reader selects IDs with ss_query but renders them with
    # eval_memeye's ID-bearing assembler.  Execute only that hash-pinned pure
    # function; importing eval_memeye would initialize unrelated clients.
    eval_ast = ast.parse(eval_source)
    assemble_nodes = [
        copy.deepcopy(node)
        for node in eval_ast.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "assemble_context_with_fact_ids"
    ]
    if len(assemble_nodes) != 1:
        raise PredictExecutorError("eval_memeye native assembler authority is missing")
    eval_namespace: dict[str, Any] = {}
    exec(
        compile(
            ast.Module(body=assemble_nodes, type_ignores=[]),
            str(eval_path),
            "exec",
        ),
        eval_namespace,
    )
    return {
        **values,
        "render": namespace["render_aggregation"],
        "native_assemble": eval_namespace["assemble_context_with_fact_ids"],
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}


def _normalize_matrix(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    if not rows:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    matrix_value = np.asarray([row[field] for row in rows], dtype=np.float32)
    if matrix_value.shape != (len(rows), EMBEDDING_DIMENSION):
        raise PredictExecutorError("sealed graph embedding matrix drift")
    if not np.isfinite(matrix_value).all():
        raise PredictExecutorError("sealed graph contains non-finite embedding")
    norms = np.linalg.norm(matrix_value, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix_value / norms


def _cosine_scores(matrix_value: np.ndarray, query_vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(_validate_vector(query_vector), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return np.zeros(matrix_value.shape[0], dtype=np.float32)
    return matrix_value @ (vector / norm)


def _rank_indices(scores: np.ndarray, count: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:count]


def _score_candidates(
    reranker: RerankBackend,
    *,
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    instruction: str,
    metadata: Mapping[str, Any],
) -> tuple[list[float], list[dict[str, Any]]]:
    scores: list[float] = []
    receipts: list[dict[str, Any]] = []
    documents = [str(item["fact"]["text"]) for item in candidates]
    candidate_ids = [item["fact"]["id"] for item in candidates]
    for start in range(0, len(documents), QWEN_MAX_DOCUMENTS):
        batch = documents[start : start + QWEN_MAX_DOCUMENTS]
        result = _rerank_reply(
            reranker.score(
                question,
                batch,
                instruction=instruction,
                cache_metadata={**dict(metadata), "batch_start": start, "batch_size": len(batch)},
            ),
            len(batch),
        )
        scores.extend(result.aligned_scores)
        receipts.append(
            {
                "batch_start": start,
                "batch_size": len(batch),
                "candidate_fact_ids": candidate_ids[start : start + len(batch)],
                "documents_sha256": _sha256(batch),
                "request_sha256": result.request_sha256,
                "response_sha256": result.response_sha256,
                "cache_identity": result.cache_identity,
                "cache_metadata": {
                    **dict(metadata),
                    "batch_start": start,
                    "batch_size": len(batch),
                },
            }
        )
    if len(scores) != len(candidates):
        raise PredictExecutorError("Qwen candidate score census drift")
    return scores, receipts


def _native_candidate_pool(
    tree: Mapping[str, Any],
    *,
    question: str,
    query_vector: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = list(tree.get("SUMMARIES", []))
    facts = list(tree.get("FACTS", []))
    if not summaries or not facts:
        raise PredictExecutorError("native retrieval tree is empty")
    summary_matrix = _normalize_matrix(summaries, "emb")
    fact_matrix = _normalize_matrix(facts, "emb")
    summary_scores = _cosine_scores(summary_matrix, query_vector)
    fact_scores = _cosine_scores(fact_matrix, query_vector)
    summary_indices = _rank_indices(summary_scores, SUMMARY_N)
    dense_indices = _rank_indices(fact_scores, DENSE_N)
    by_id = {row["id"]: row for row in facts}
    children: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for fact in facts:
        children[fact.get("parent_summary_id")].append(fact)
    pool: dict[Any, Mapping[str, Any]] = {}
    for index in summary_indices:
        for fact in children.get(summaries[index]["id"], []):
            pool[fact["id"]] = fact
    for index in dense_indices:
        pool[facts[index]["id"]] = facts[index]
    graph_memory = importlib.import_module("creamem3.graph_memory_v2")
    bm25 = graph_memory.BM25Index()
    for fact in facts:
        bm25.add(fact["id"], str(fact["text"]))
    lexical = bm25.search(question, top_n=BM25_N)
    for fact_id, _ in lexical:
        if fact_id in by_id:
            pool[fact_id] = by_id[fact_id]
    candidates = [{"fact": fact} for fact in pool.values()]
    if not candidates:
        raise PredictExecutorError("native retrieval candidate pool is empty")
    trace = {
        "policy": {
            "summary_top_n": SUMMARY_N,
            "dense_top_n": DENSE_N,
            "bm25_top_n": BM25_N,
        },
        "top_summaries": [
            {"summary_id": summaries[index]["id"], "score": float(summary_scores[index])}
            for index in summary_indices
        ],
        "candidate_fact_ids": [item["fact"]["id"] for item in candidates],
    }
    return candidates, trace


def _native_retrieve(
    tree: Mapping[str, Any],
    *,
    question: str,
    query_vector: Sequence[float],
    top_k: int,
    reranker: RerankBackend,
    instruction: str,
    metadata: Mapping[str, Any],
    assemble: Callable[[Mapping[str, Any], Sequence[Any]], str]
    | None = None,
) -> tuple[str, list[Any], dict[str, Any]]:
    summaries = list(tree.get("SUMMARIES", []))
    candidates, trace = _native_candidate_pool(
        tree, question=question, query_vector=query_vector
    )
    qwen_scores, qwen_receipts = _score_candidates(
        reranker,
        question=question,
        candidates=candidates,
        instruction=instruction,
        metadata=metadata,
    )
    selected = [
        fact
        for fact, _ in sorted(
            zip((item["fact"] for item in candidates), qwen_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
    ]
    ids = [fact["id"] for fact in selected]
    if assemble is None:
        assemble = _reader_authority(matrix.load_protocol())["native_assemble"]
    context = str(assemble(tree, ids))
    valid_summary_ids = {summary["id"] for summary in summaries}
    rendered_summary_ids: list[Any] = []
    for fact in selected:
        parent = fact.get("parent_summary_id")
        if parent in valid_summary_ids and parent not in rendered_summary_ids:
            rendered_summary_ids.append(parent)
    trace["policy"]["top_k"] = top_k
    trace["rerank"] = {
        "selected_fact_ids": ids,
        "fallback": False,
        "batches": qwen_receipts,
    }
    trace["assembly"] = {
        "rendered_event_view_ids": [],
        "rendered_session_summary_ids": rendered_summary_ids,
    }
    return context, ids, trace


def _creamem_retrieve(
    graph: Mapping[str, Any],
    *,
    question: str,
    query_vector: Sequence[float],
    top_k: int,
    reranker: RerankBackend,
    instruction: str,
    metadata: Mapping[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    graph_memory = importlib.import_module("creamem3.graph_memory_v2")
    facts = list(graph.get("fact_nodes", []))
    index = graph_memory.GraphMemoryIndex(
        graph,
        summary_top_k=SUMMARY_N,
        dense_fact_top_k=DENSE_N,
        bm25_fact_top_k=BM25_N,
        candidate_cap=max(1, len(facts)),
        rerank_top_k=top_k,
    )
    candidates, trace = index._candidate_pool(question, query_vector)
    if not candidates:
        raise PredictExecutorError("CreaMem retrieval candidate pool is empty")
    qwen_scores, qwen_receipts = _score_candidates(
        reranker,
        question=question,
        candidates=candidates,
        instruction=instruction,
        metadata=metadata,
    )
    selected = [
        item
        for item, _ in sorted(
            zip(candidates, qwen_scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
    ]
    context, assembly = index._assemble(selected, trace["top_summaries"])
    ids = [str(item["fact"]["id"]) for item in selected]
    trace["rerank"] = {
        "selected_fact_ids": ids,
        "fallback": False,
        "batches": qwen_receipts,
    }
    trace["assembly"] = assembly
    return context, ids, trace


def _artifact_for(
    build: Mapping[str, Any], backbone: str, architecture: str, task: str
) -> tuple[Path, str]:
    lookup = {
        (str(row["backbone"]), str(row["task"])): row
        for row in build.get("architecture_bindings", [])
        if isinstance(row, Mapping)
    }
    row = lookup.get((backbone, task))
    if row is None:
        raise PredictExecutorError("build architecture binding is missing")
    prefix = ARCHITECTURE_TO_BINDING[architecture]
    path = Path(str(row[f"{prefix}_path"])).resolve()
    expected = str(row[f"{prefix}_sha256"])
    if _file_sha256(path) != expected:
        raise PredictExecutorError("sealed build artifact hash drift")
    return path, expected


def _load_build(output_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(output_root) / "build" / "complete.json"
    value = _read_mapping(path, "BUILD_COMPLETE")
    copy_value = dict(value)
    observed = copy_value.pop("complete_sha256", None)
    if (
        observed != _sha256(copy_value)
        or value.get("complete") is not True
        or value.get("protocol_sha256") != _sha256(protocol)
    ):
        raise PredictExecutorError("BUILD_COMPLETE contract drift")
    executor_path = Path(__file__).with_name("build_executor_v19.py")
    if value.get("executor_sha256") != _file_sha256(executor_path):
        raise PredictExecutorError("BUILD_COMPLETE executor authority drift")
    build_module = importlib.import_module("creamem3.build_executor_v19")
    build_module.validate_build_complete(output_root=output_root, protocol=protocol)
    return value


def _query_payload(output_root: Path) -> dict[str, Any]:
    project = matrix.validate_project(output_root=output_root)
    path = Path(str(project["query_path"]))
    value = _read_mapping(path, "score-free query projection")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise PredictExecutorError("score-free query rows are missing")
    matrix._assert_query_rows_score_free(rows)
    if value.get("rows_sha256") != _sha256(rows):
        raise PredictExecutorError("score-free query row digest drift")
    return value


def compute_predict_budget(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Compute a provider-free upper bound from the sealed final graph sizes."""

    protocol = dict(protocol or matrix.load_protocol())
    validate_predict_authorities(protocol)
    build = _load_build(output_root, protocol)
    queries = _query_payload(output_root)
    rows = list(queries["rows"])
    task_counts = Counter(str(row["task"]) for row in rows)
    artifact_rows: list[dict[str, Any]] = []
    qwen_first = 0
    for backbone in matrix.BACKBONE_IDS:
        for architecture in RETRIEVAL_ARCHITECTURES:
            for task, question_count in sorted(task_counts.items()):
                path, digest = _artifact_for(build, backbone, architecture, task)
                artifact = _read_mapping(path, "sealed retrieval artifact")
                fact_count = len(
                    artifact.get("FACTS", [])
                    if architecture == "mgtm_matched"
                    else artifact.get("fact_nodes", [])
                )
                if fact_count <= 0:
                    raise PredictExecutorError("retrieval artifact has no facts")
                batches_per_query = math.ceil(fact_count / QWEN_MAX_DOCUMENTS)
                calls = int(question_count) * batches_per_query
                qwen_first += calls
                artifact_rows.append(
                    {
                        "backbone": backbone,
                        "retrieval_architecture": architecture,
                        "task": task,
                        "artifact_path": str(path),
                        "artifact_sha256": digest,
                        "facts": fact_count,
                        "questions": int(question_count),
                        "qwen_batches_per_query_hard": batches_per_query,
                        "qwen_first_attempt_hard": calls,
                    }
                )
    route_counts = Counter(str(row["reader_route"]) for row in rows)
    cells = len(matrix.matrix_cell_ids())
    planner_first = len(rows)
    embedding_first = math.ceil(len(rows) / EMBEDDING_BATCH_SIZE)
    focused_first = route_counts["two_stage"] * cells
    answer_first = route_counts["two_stage"] * cells
    fused_first = route_counts["structured_fused"] * cells
    stage_first = {
        "query_embedding": embedding_first,
        "semantic_planner": planner_first,
        "qwen_rerank": qwen_first,
        "focused_reader": focused_first,
        "answer_reader": answer_first,
        "structured_fused_reader": fused_first,
    }
    stage_hard = {
        key: value * MAX_PROVIDER_ATTEMPTS for key, value in stage_first.items()
    }
    value = {
        "schema_version": PREDICT_BUDGET_SCHEMA,
        "protocol_sha256": _sha256(protocol),
        "build_complete_file_sha256": _file_sha256(
            Path(output_root) / "build" / "complete.json"
        ),
        "query_file_sha256": _file_sha256(
            Path(output_root) / "project" / "queries_score_free.json"
        ),
        "matrix_cells": cells,
        "query_rows": len(rows),
        "retrieval_sharing": "online_base_and_operation_context_share_one_qwen_result",
        "qwen_candidate_batch_size": QWEN_MAX_DOCUMENTS,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        "stage_first_attempt_ceilings": stage_first,
        "stage_provider_attempt_hard_ceilings": stage_hard,
        "logical_first_attempt_ceiling": sum(stage_first.values()),
        "provider_attempt_hard_ceiling": sum(stage_hard.values()),
        "artifact_rows": artifact_rows,
        "artifact_rows_sha256": _sha256(artifact_rows),
        "provider_calls": 0,
    }
    value["predict_budget_sha256"] = _sha256(value)
    return value


def seal_predict_budget(
    output_root: Path = matrix.DEFAULT_ROOT,
) -> dict[str, Any]:
    value = compute_predict_budget(output_root=output_root)
    _write_once(Path(output_root) / "predict" / "budget.json", value)
    return value


def validate_predict_budget_gate(
    *, output_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    budget_path = Path(output_root) / "predict" / "budget.json"
    gate_path = Path(output_root) / "predict" / "budget_gate.json"
    budget = compute_predict_budget(output_root=output_root, protocol=protocol)
    stored = _read_mapping(budget_path, "predict budget")
    if stored != budget:
        raise PredictExecutorError("predict budget file drift")
    gate = _read_mapping(gate_path, "predict budget gate")
    copy_gate = dict(gate)
    observed = copy_gate.pop("gate_sha256", None)
    expected = {
        "schema_version": PREDICT_BUDGET_GATE_SCHEMA,
        "decision": "approve",
        "reviewer_role": "independent_protocol_auditor",
        "protocol_sha256": _sha256(protocol),
        "budget_path": str(budget_path.resolve()),
        "budget_file_sha256": _file_sha256(budget_path),
        "predict_budget_sha256": budget["predict_budget_sha256"],
        "approved_provider_attempt_hard_ceiling": budget[
            "provider_attempt_hard_ceiling"
        ],
        "approved_stage_provider_attempt_hard_ceilings": budget[
            "stage_provider_attempt_hard_ceilings"
        ],
        "legacy_cache_readthrough": False,
    }
    if observed != _sha256(copy_gate):
        raise PredictExecutorError("predict budget gate self-hash drift")
    for key, value in expected.items():
        if gate.get(key) != value:
            raise PredictExecutorError(f"predict budget gate binding drift: {key}")
    if not str(gate.get("reviewer", "")).strip():
        raise PredictExecutorError("predict budget gate lacks reviewer")
    return {**gate, "budget": budget, "file_sha256": _file_sha256(gate_path)}


def _provider_calls(receipt: Mapping[str, Any]) -> int:
    if isinstance(receipt.get("provider_calls"), int):
        return int(receipt["provider_calls"])
    stages = receipt.get("stages")
    if isinstance(stages, Mapping):
        return sum(int(row.get("provider_calls", 0)) for row in stages.values())
    return 0


def _durable_gemini_usage(
    receipt: Mapping[str, Any], *, cache_root: Path
) -> dict[str, Any]:
    value = copy.deepcopy(dict(receipt))
    stages = value.get("stages")
    if not isinstance(stages, Mapping):
        raise PredictExecutorError("Gemini usage stages are missing")
    for stage, raw_row in list(stages.items()):
        row = dict(raw_row)
        durable = len(
            list(
                (Path(cache_root) / "attempts" / str(stage)).rglob(
                    "attempt_*_request.json"
                )
            )
        )
        in_process = int(row.get("provider_calls", 0))
        limits = row.get("limits", {})
        row["in_process_provider_calls"] = in_process
        row["durable_provider_attempt_intents"] = durable
        row["provider_calls"] = durable
        row["remaining_live_calls"] = int(limits.get("max_live_calls", 0)) - durable
        stages[stage] = row
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _assert_prediction_score_free(value: Any) -> None:
    forbidden = matrix.FORBIDDEN_PREDICTION_KEYS | {"answer", "answers"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in forbidden:
                raise PredictExecutorError("score-bearing field entered prediction artifact")
            _assert_prediction_score_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_prediction_score_free(child)


def _validate_predict_provider_usage(
    *,
    output_root: Path,
    usage: Mapping[str, Any],
    budget: Mapping[str, Any],
    qwen_first_attempt_floor: int,
) -> int:
    if set(usage) != {"g25", "query_embedding", "qwen_rerank"}:
        raise PredictExecutorError("predict provider usage namespace drift")
    hard = budget["stage_provider_attempt_hard_ceilings"]
    first = budget["stage_first_attempt_ceilings"]
    text = usage["g25"]
    expected_text_stages = {
        "planner": (512, hard["semantic_planner"]),
        "focused": (2600, hard["focused_reader"]),
        "answer": (500, hard["answer_reader"]),
        "structured_fused": (1200, hard["structured_fused_reader"]),
    }
    if (
        not isinstance(text, Mapping)
        or set(text)
        != {
            "schema_version",
            "model",
            "cache_root",
            "allow_model_calls",
            "provider_loaded",
            "provider_factory_calls",
            "stages",
        }
        or text.get("model") != "gemini-2.5-flash-lite"
        or Path(str(text.get("cache_root", ""))).resolve()
        != (Path(output_root) / "provider" / "predict" / "g25_reader").resolve()
        or set(text.get("stages", {})) != set(expected_text_stages)
        or text.get("allow_model_calls") is not True
        or int(text.get("provider_factory_calls", -1)) not in {0, 1}
    ):
        raise PredictExecutorError("G25 Reader usage receipt drift")
    text_calls = 0
    for stage, (tokens, ceiling) in expected_text_stages.items():
        row = text["stages"][stage]
        expected_limits = {
            "max_output_tokens": tokens,
            "max_live_calls": ceiling,
            "max_attempts": MAX_PROVIDER_ATTEMPTS,
        }
        calls = int(row.get("provider_calls", -1))
        first_key = {
            "planner": "semantic_planner",
            "focused": "focused_reader",
            "answer": "answer_reader",
            "structured_fused": "structured_fused_reader",
        }[stage]
        if (
            set(row)
            != {
                "provider_calls",
                "in_process_provider_calls",
                "durable_provider_attempt_intents",
                "cache_hits",
                "usage",
                "limits",
                "remaining_live_calls",
            }
            or row.get("durable_provider_attempt_intents") != calls
            or not 0 <= int(row.get("in_process_provider_calls", -1)) <= calls
            or row.get("limits") != expected_limits
            or not first[first_key] <= calls <= ceiling
        ):
            raise PredictExecutorError(f"G25 Reader stage usage drift: {stage}")
        text_calls += calls

    embedding = usage["query_embedding"]
    if (
        not isinstance(embedding, Mapping)
        or embedding.get("model") != "gemini-embedding-001"
        or embedding.get("dimension") != EMBEDDING_DIMENSION
        or embedding.get("batch_size") != EMBEDDING_BATCH_SIZE
        or embedding.get("provider_visible_character_limit") != 8000
        or embedding.get("cache_identity_uses_complete_untruncated_text") is not True
        or embedding.get("failure_policy")
        != "fail_closed_no_per_item_or_zero_fallback"
        or embedding.get("max_provider_attempts_per_physical_batch")
        != MAX_PROVIDER_ATTEMPTS
        or embedding.get("approved_provider_attempt_hard_ceiling")
        != hard["query_embedding"]
        or embedding.get("task_type") is not None
        or embedding.get("input_type") is not None
        or embedding.get("output_dimensionality") is not None
    ):
        raise PredictExecutorError("query embedding usage receipt drift")
    embedding_calls = int(embedding.get("provider_calls", -1))
    if not first["query_embedding"] <= embedding_calls <= hard["query_embedding"]:
        raise PredictExecutorError("query embedding provider census drift")

    rerank = usage["qwen_rerank"]
    if (
        not isinstance(rerank, Mapping)
        or rerank.get("model") != QWEN_MODEL
        or rerank.get("endpoint") != QWEN_ENDPOINT
        or rerank.get("max_documents") != QWEN_MAX_DOCUMENTS
        or rerank.get("max_provider_attempts") != MAX_PROVIDER_ATTEMPTS
        or rerank.get("approved_provider_attempt_hard_ceiling")
        != hard["qwen_rerank"]
        or rerank.get("fallback") is not False
        or Path(str(rerank.get("cache_root", ""))).resolve()
        != (Path(output_root) / "provider" / "predict" / "qwen3_rerank").resolve()
    ):
        raise PredictExecutorError("Qwen usage receipt drift")
    rerank_calls = int(rerank.get("provider_calls", -1))
    if not qwen_first_attempt_floor <= rerank_calls <= hard["qwen_rerank"]:
        raise PredictExecutorError("Qwen provider census drift")
    return text_calls + embedding_calls + rerank_calls


def _validate_query_embeddings(
    *,
    output_root: Path,
    protocol: Mapping[str, Any],
    queries: Mapping[str, Any],
    gate: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    path = (Path(output_root) / "predict" / "query_embeddings.json").resolve()
    value = _read_mapping(path, "query embedding artifact")
    copy_value = dict(value)
    observed = copy_value.pop("query_embedding_sha256", None)
    rows = value.get("rows")
    batches = value.get("batches")
    expected_keys = {
        "schema_version",
        "protocol_sha256",
        "query_rows_sha256",
        "rows",
        "rows_sha256",
        "batches",
        "batches_sha256",
        "complete",
        "query_embedding_sha256",
    }
    if (
        set(value) != expected_keys
        or observed != _sha256(copy_value)
        or value.get("schema_version") != QUERY_EMBEDDING_SCHEMA
        or value.get("protocol_sha256") != _sha256(protocol)
        or value.get("query_rows_sha256") != queries["rows_sha256"]
        or value.get("complete") is not True
        or not isinstance(rows, list)
        or value.get("rows_sha256") != _sha256(rows)
        or not isinstance(batches, list)
        or value.get("batches_sha256") != _sha256(batches)
    ):
        raise PredictExecutorError("query embedding artifact seal drift")
    query_rows = list(queries["rows"])
    if len(rows) != len(query_rows):
        raise PredictExecutorError("query embedding row census drift")
    vectors: dict[str, list[float]] = {}
    for query, row in zip(query_rows, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "row_id",
            "question_sha256",
            "vector_sha256",
            "vector",
        }:
            raise PredictExecutorError("query embedding row fieldset drift")
        vector = _validate_vector(row["vector"])
        row_id = str(query["row_id"])
        if (
            row.get("row_id") != row_id
            or row.get("question_sha256") != _sha256(query["question"])
            or row.get("vector_sha256") != _sha256(vector)
        ):
            raise PredictExecutorError("query embedding row/order drift")
        vectors[row_id] = vector

    runtime_path = Path(__file__).with_name(
        "memeye_official_embedding_batch_runtime_v19_1.py"
    )
    expected_runtime_sha = (
        "bb782cbba5d24f08c13907f2a9fe386c8bcffa5e15f53b675b69818c4989298b"
    )
    if _file_sha256(runtime_path) != expected_runtime_sha:
        raise PredictExecutorError("official query embedding runtime authority drift")
    runtime = importlib.import_module(
        "creamem3.memeye_official_embedding_batch_runtime_v19_1"
    )
    if Path(runtime.__file__).resolve() != runtime_path.resolve():
        raise PredictExecutorError("official query embedding runtime import drift")
    cache_root = Path(output_root) / "provider" / "predict" / "query_embedding_native_3072"
    audit = runtime.OfficialGoogleBatchEmbeddingRuntime(
        cache_root,
        containment_root=Path(output_root),
        limits=runtime.EmbeddingLimits(
            max_live_calls=budget["stage_provider_attempt_hard_ceilings"][
                "query_embedding"
            ],
            max_attempts=MAX_PROVIDER_ATTEMPTS,
        ),
        allow_model_calls=False,
        budget_binding_sha256=str(gate["gate_sha256"]),
        provider_factory=lambda: (_ for _ in ()).throw(
            PredictExecutorError("provider construction during embedding validation")
        ),
    )
    expected_batch_receipts: list[dict[str, Any]] = []
    questions = [str(row["question"]) for row in query_rows]
    for start in range(0, len(questions), EMBEDDING_BATCH_SIZE):
        batch = questions[start : start + EMBEDDING_BATCH_SIZE]
        metadata = {
            "query_rows_sha256": queries["rows_sha256"],
            "tag": "memeye_v19_predict_query_embedding",
            "batch_start": start,
            "batch_size": len(batch),
        }
        result = audit.embed_batch(batch, cache_metadata=metadata)
        batch_vectors = [list(map(float, vector)) for vector in result.vectors]
        expected_vectors = [
            vectors[str(row["row_id"])]
            for row in query_rows[start : start + len(batch)]
        ]
        if batch_vectors != expected_vectors:
            raise PredictExecutorError("query embedding cache/vector drift")
        expected_batch_receipts.append(
            {
                "tag": "memeye_v19_predict_query_embedding",
                "batch_start": start,
                "batch_size": len(batch),
                "cache_metadata": metadata,
                "request_sha256": result.request_sha256,
                "resume_hash": result.resume_hash,
                "response_sha256": result.response_sha256,
                "vector_sha256s": list(result.vector_sha256s),
                "vectors_sha256": result.vectors_sha256,
                "full_text_sha256s": list(result.full_text_sha256s),
                "visible_text_sha256s": list(result.visible_text_sha256s),
                "cache_path": result.cache_path,
                "cache_file_sha256": _file_sha256(Path(result.cache_path)),
                "successful_attempt_ordinal": result.successful_attempt_ordinal,
            }
        )
    if batches != expected_batch_receipts:
        raise PredictExecutorError("query embedding batch receipt drift")
    expected_resumes = {row["resume_hash"] for row in batches}
    observed_responses = {
        path.stem for path in (cache_root / "responses").glob("*.json")
    } if (cache_root / "responses").is_dir() else set()
    observed_plans = {
        path.stem for path in (cache_root / "plans").glob("*.json")
    } if (cache_root / "plans").is_dir() else set()
    observed_attempts = {
        path.name for path in (cache_root / "attempts").iterdir() if path.is_dir()
    } if (cache_root / "attempts").is_dir() else set()
    if (
        observed_responses != expected_resumes
        or observed_plans != expected_resumes
        or observed_attempts != expected_resumes
        or ((cache_root / "orphans").is_dir() and any((cache_root / "orphans").iterdir()))
        or ((cache_root / "failures").is_dir() and any((cache_root / "failures").iterdir()))
    ):
        raise PredictExecutorError("query embedding cache identity census drift")
    lifecycle_root = cache_root / "lifecycle_locks"
    observed_locks = {
        path.stem for path in lifecycle_root.glob("*.lock") if path.is_file()
    } if lifecycle_root.is_dir() else set()
    if observed_locks != expected_resumes:
        raise PredictExecutorError("query embedding lifecycle-lock census drift")
    for resume_hash_value in sorted(expected_resumes):
        lock_path = lifecycle_root / f"{resume_hash_value}.lock"
        if lock_path.read_bytes() != b"\0":
            raise PredictExecutorError("query embedding lifecycle-lock bytes drift")
        with runtime.acquire_request_lifecycle_lock(
            lock_path, timeout_seconds=0.1, poll_seconds=0.01
        ):
            pass
    budget_lock = cache_root / "budget_lifecycle" / "global.lock"
    if budget_lock.read_bytes() != b"\0":
        raise PredictExecutorError("query embedding budget-lock bytes drift")
    with runtime.acquire_request_lifecycle_lock(
        budget_lock, timeout_seconds=0.1, poll_seconds=0.01
    ):
        pass
    return vectors, {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "query_embedding_sha256": observed,
        "batches": len(batches),
    }


def _validate_g25_cached_call(
    *,
    audit_reader: Any,
    stage: str,
    prompt: str,
    json_mode: bool,
    max_output_tokens: int,
    cache_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure-read exact replay of one G25 plan/attempt/raw/final chain."""

    module = audit_reader._module
    core = audit_reader._runtime
    request = module.build_generate_content_request(
        stage,
        prompt,
        json_mode=json_mode,
        max_output_tokens=max_output_tokens,
        stage_limits=audit_reader.stage_limits,
    )
    request_sha = module.request_identity(
        stage=stage,
        request=request,
        stage_limits=audit_reader.stage_limits,
        cache_metadata=cache_metadata,
    )
    resumable = module.resume_hash(
        stage=stage,
        request_sha256=request_sha,
        stage_limits=audit_reader.stage_limits,
    )
    final_path = Path(audit_reader.cache_root) / "responses" / stage / f"{resumable}.json"
    plan_path = Path(audit_reader.cache_root) / "plans" / stage / f"{resumable}.json"
    final = _read_mapping(final_path, f"G25 {stage} final")
    final_keys = {
        "schema_version",
        "protocol_sha256",
        "stage",
        "model",
        "request_sha256",
        "resume_hash",
        "attempt_ordinal",
        "response_sha256",
        "text",
        "usage",
        "cache_path",
    }
    if (
        set(final) != final_keys
        or final.get("schema_version") != module.CACHE_SCHEMA_VERSION
        or final.get("protocol_sha256")
        != module.protocol_sha256(audit_reader.stage_limits)
        or final.get("stage") != stage
        or final.get("model") != "gemini-2.5-flash-lite"
        or final.get("request_sha256") != request_sha
        or final.get("resume_hash") != resumable
        or final.get("response_sha256") != _sha256(final.get("text", ""))
        or Path(str(final.get("cache_path", ""))).resolve() != final_path.resolve()
    ):
        raise PredictExecutorError(f"G25 {stage} final cache binding drift")
    ordinal = final.get("attempt_ordinal")
    if not isinstance(ordinal, int) or not 1 <= ordinal <= MAX_PROVIDER_ATTEMPTS:
        raise PredictExecutorError(f"G25 {stage} success ordinal drift")
    plan = _read_mapping(plan_path, f"G25 {stage} plan")
    expected_plan = core._plan_record(
        stage=stage,
        request=request,
        prompt=prompt,
        request_sha=request_sha,
        resumable=resumable,
        caption_context_count=None,
        cache_metadata=cache_metadata,
    )
    if plan != expected_plan:
        raise PredictExecutorError(f"G25 {stage} plan reconstruction drift")
    intent_path, raw_path = core._attempt_paths(stage, resumable, ordinal)
    intent = _read_mapping(intent_path, f"G25 {stage} success intent")
    expected_intent = core._attempt_request_record(
        stage=stage,
        request_sha=request_sha,
        resumable=resumable,
        ordinal=ordinal,
        plan=plan,
    )
    raw = _read_mapping(raw_path, f"G25 {stage} raw response")
    raw_keys = {
        "schema_version",
        "protocol_sha256",
        "stage",
        "model",
        "request_sha256",
        "resume_hash",
        "attempt_ordinal",
        "attempt_identity_sha256",
        "outcome_kind",
        "response_sha256",
        "text",
        "usage",
    }
    if (
        intent != expected_intent
        or set(raw) != raw_keys
        or raw.get("attempt_identity_sha256")
        != intent["attempt_identity_sha256"]
        or raw.get("outcome_kind") != "raw_response"
        or raw.get("response_sha256") != _sha256(raw.get("text", ""))
        or raw.get("response_sha256") != final["response_sha256"]
        or raw.get("text") != final["text"]
        or raw.get("usage") != final["usage"]
    ):
        raise PredictExecutorError(f"G25 {stage} raw/final binding drift")
    retry_chain: list[dict[str, Any]] = []
    base_retry_keys = {
        "schema_version",
        "protocol_sha256",
        "stage",
        "model",
        "request_sha256",
        "resume_hash",
        "attempt_ordinal",
        "attempt_identity_sha256",
        "outcome_kind",
        "retryable",
        "status_code",
        "error_type",
        "error_message",
        "error_sha256",
    }
    for previous in range(1, ordinal):
        previous_intent_path, previous_raw_path = core._attempt_paths(
            stage, resumable, previous
        )
        previous_intent = _read_mapping(
            previous_intent_path, f"G25 {stage} previous intent"
        )
        expected_previous_intent = core._attempt_request_record(
            stage=stage,
            request_sha=request_sha,
            resumable=resumable,
            ordinal=previous,
            plan=plan,
        )
        previous_raw = _read_mapping(
            previous_raw_path, f"G25 {stage} previous response"
        )
        if (
            previous_intent != expected_previous_intent
            or (
                set(previous_raw) != base_retry_keys
                and set(previous_raw) != base_retry_keys | {"usage"}
            )
            or previous_raw.get("attempt_identity_sha256")
            != previous_intent["attempt_identity_sha256"]
            or previous_raw.get("outcome_kind") != "explicit_provider_error"
            or previous_raw.get("retryable") is not True
            or previous_raw.get("error_sha256")
            != _sha256(previous_raw.get("error_message", ""))
        ):
            raise PredictExecutorError(f"G25 {stage} retry chain drift")
        retry_chain.append(
            {
                "ordinal": previous,
                "intent_file_sha256": _file_sha256(previous_intent_path),
                "response_file_sha256": _file_sha256(previous_raw_path),
            }
        )
    for later in range(ordinal + 1, MAX_PROVIDER_ATTEMPTS + 1):
        later_intent, later_raw = core._attempt_paths(stage, resumable, later)
        if later_intent.exists() or later_raw.exists():
            raise PredictExecutorError(f"G25 {stage} post-success attempt drift")
    if core._orphan_path(stage, resumable).exists() or core._failure_path(
        stage, resumable
    ).exists():
        raise PredictExecutorError(f"G25 {stage} success/terminal receipt conflict")
    return {
        "stage": stage,
        "text": str(final["text"]),
        "request_sha256": request_sha,
        "response_sha256": str(final["response_sha256"]),
        "resume_hash": resumable,
        "successful_attempt_ordinal": ordinal,
        "retry_chain": retry_chain,
    }


def _validate_g25_cache_census(
    *, audit_reader: Any, expected_by_stage: Mapping[str, set[str]]
) -> dict[str, Any]:
    module = audit_reader._module
    core = audit_reader._runtime
    attempts_total = 0
    for stage in ("planner", "focused", "answer", "structured_fused"):
        expected = set(expected_by_stage.get(stage, set()))
        response_root = Path(audit_reader.cache_root) / "responses" / stage
        plan_root = Path(audit_reader.cache_root) / "plans" / stage
        attempt_root = Path(audit_reader.cache_root) / "attempts" / stage
        lock_root = Path(audit_reader.cache_root) / "lifecycle_locks" / stage
        observed_responses = {
            path.stem for path in response_root.glob("*.json") if path.is_file()
        } if response_root.is_dir() else set()
        observed_plans = {
            path.stem for path in plan_root.glob("*.json") if path.is_file()
        } if plan_root.is_dir() else set()
        observed_attempts = {
            path.name for path in attempt_root.iterdir() if path.is_dir()
        } if attempt_root.is_dir() else set()
        observed_locks = {
            path.stem for path in lock_root.glob("*.lock") if path.is_file()
        } if lock_root.is_dir() else set()
        if not (
            observed_responses
            == observed_plans
            == observed_attempts
            == observed_locks
            == expected
        ):
            raise PredictExecutorError(f"G25 {stage} cache identity census drift")
        for identity in sorted(expected):
            lock_path = lock_root / f"{identity}.lock"
            if lock_path.read_bytes() != b"\0":
                raise PredictExecutorError(f"G25 {stage} lifecycle-lock bytes drift")
            with module.acquire_request_lifecycle_lock(
                lock_path, timeout_seconds=0.1, poll_seconds=0.01
            ):
                pass
            attempt_files = {
                path.resolve() for path in (attempt_root / identity).glob("attempt_*.json")
                if path.is_file()
            }
            final = _read_mapping(
                response_root / f"{identity}.json", f"G25 {stage} census final"
            )
            ordinal = int(final["attempt_ordinal"])
            expected_attempt_files = {
                core._attempt_paths(stage, identity, number)[kind].resolve()
                for number in range(1, ordinal + 1)
                for kind in (0, 1)
            }
            if attempt_files != expected_attempt_files:
                raise PredictExecutorError(f"G25 {stage} attempt file census drift")
            attempts_total += ordinal
        for terminal in ("orphans", "failures"):
            root = Path(audit_reader.cache_root) / terminal / stage
            if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
                raise PredictExecutorError(f"G25 {stage} {terminal} census nonempty")
    global_lease = Path(audit_reader.cache_root) / "formal_global_generation_budget.lock"
    if global_lease.exists():
        raise PredictExecutorError("G25 Reader global budget lease remains")
    durable_observed = sum(
        len(
            list(
                (Path(audit_reader.cache_root) / "attempts" / stage).rglob(
                    "attempt_*_request.json"
                )
            )
        )
        for stage in ("planner", "focused", "answer", "structured_fused")
    )
    if durable_observed != attempts_total:
        raise PredictExecutorError("G25 Reader durable attempt census drift")
    return {
        "identities": sum(len(value) for value in expected_by_stage.values()),
        "durable_attempts": attempts_total,
        "provider_calls_during_validation": 0,
    }


def _prediction_complete(
    output_root: Path,
    *,
    expected_rows: int,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    protocol = dict(protocol or matrix.load_protocol())
    queries = _query_payload(output_root)
    query_rows = list(queries["rows"])
    if len(query_rows) != expected_rows:
        raise PredictExecutorError("prediction/query row census drift")
    gate = validate_predict_budget_gate(output_root=output_root, protocol=protocol)
    budget = gate["budget"]
    build = _load_build(output_root, protocol)
    retrieval_authority = _reader_authority(protocol)
    query_vectors, embedding_receipt = _validate_query_embeddings(
        output_root=output_root,
        protocol=protocol,
        queries=queries,
        gate=gate,
        budget=budget,
    )
    path = Path(output_root) / "predict" / "complete.json"
    value = _read_mapping(path, "prediction completion")
    copy_value = dict(value)
    observed = copy_value.pop("complete_sha256", None)
    expected_complete_keys = {
        "schema_version",
        "complete",
        "protocol_sha256",
        "predict_executor_file_sha256",
        "build_complete_file_sha256",
        "query_file_sha256",
        "predict_budget_sha256",
        "predict_budget_gate_file_sha256",
        "query_embedding_path",
        "query_embedding_file_sha256",
        "query_embedding_sha256",
        "retrieval_requests",
        "retrieval_requests_sha256",
        "cells",
        "provider_usage",
        "provider_calls",
        "provider_attempt_hard_ceiling",
        "score_fields_present",
        "complete_sha256",
    }
    if (
        set(value) != expected_complete_keys
        or
        observed != _sha256(copy_value)
        or value.get("schema_version") != matrix.PREDICTION_COMPLETE_SCHEMA
        or value.get("complete") is not True
        or value.get("protocol_sha256") != _sha256(protocol)
        or value.get("predict_executor_file_sha256")
        != _file_sha256(Path(__file__).resolve())
        or value.get("build_complete_file_sha256")
        != _file_sha256(output_root / "build" / "complete.json")
        or value.get("query_file_sha256")
        != _file_sha256(output_root / "project" / "queries_score_free.json")
        or value.get("predict_budget_sha256") != budget["predict_budget_sha256"]
        or value.get("predict_budget_gate_file_sha256") != gate["file_sha256"]
        or Path(str(value.get("query_embedding_path", ""))).resolve()
        != Path(embedding_receipt["path"]).resolve()
        or value.get("query_embedding_file_sha256")
        != embedding_receipt["file_sha256"]
        or value.get("query_embedding_sha256")
        != embedding_receipt["query_embedding_sha256"]
        or value.get("provider_attempt_hard_ceiling")
        != budget["provider_attempt_hard_ceiling"]
        or value.get("score_fields_present") is not False
    ):
        raise PredictExecutorError("prediction completion self-hash drift")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != len(matrix.matrix_cell_ids()):
        raise PredictExecutorError("prediction completion cell census drift")
    seen: set[str] = set()
    prediction_bindings: dict[tuple[str, str, str], tuple[str, str]] = {}
    prediction_context_bindings: dict[tuple[str, str, str], str] = {}
    sealed_prediction_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise PredictExecutorError("prediction completion cell receipt drift")
        if set(cell) != {
            "cell_id",
            "backbone",
            "architecture",
            "rows",
            "complete",
            "prediction_path",
            "prediction_sha256",
            "cell_sha256",
        }:
            raise PredictExecutorError("prediction cell receipt fieldset drift")
        cell_id = str(cell.get("cell_id", ""))
        if cell_id in seen or cell_id not in matrix.matrix_cell_ids():
            raise PredictExecutorError("prediction completion cell identity drift")
        seen.add(cell_id)
        backbone, architecture = cell_id.split("__", 1)
        path_value = Path(str(cell.get("prediction_path", ""))).resolve()
        expected_path = (
            output_root / "predict" / "cells" / cell_id / "predictions.json"
        ).resolve()
        if (
            path_value != expected_path
            or cell.get("backbone") != backbone
            or cell.get("architecture") != architecture
            or cell.get("rows") != expected_rows
            or cell.get("complete") is not True
            or _file_sha256(path_value) != cell.get("prediction_sha256")
        ):
            raise PredictExecutorError("prediction cell seal drift")
        artifact = _read_mapping(path_value, "prediction cell")
        artifact_copy = dict(artifact)
        artifact_hash = artifact_copy.pop("cell_sha256", None)
        rows = artifact.get("rows")
        if (
            set(artifact)
            != {
                "schema_version",
                "cell_id",
                "backbone",
                "architecture",
                "protocol_sha256",
                "query_rows_sha256",
                "rows",
                "rows_sha256",
                "complete",
                "cell_sha256",
            }
            or
            artifact_hash != _sha256(artifact_copy)
            or artifact_hash != cell.get("cell_sha256")
            or artifact.get("schema_version") != PREDICTION_CELL_SCHEMA
            or artifact.get("cell_id") != cell_id
            or artifact.get("backbone") != backbone
            or artifact.get("architecture") != architecture
            or artifact.get("protocol_sha256") != _sha256(protocol)
            or artifact.get("query_rows_sha256") != queries["rows_sha256"]
            or artifact.get("complete") is not True
            or not isinstance(rows, list)
            or len(rows) != expected_rows
            or artifact.get("rows_sha256") != _sha256(rows)
        ):
            raise PredictExecutorError("prediction cell internal seal drift")
        _assert_prediction_score_free(rows)
        for expected_query, row in zip(query_rows, rows, strict=True):
            if not isinstance(row, Mapping):
                raise PredictExecutorError("prediction row is not an object")
            if set(row) != {
                "row_id",
                "task",
                "question_id",
                "source_ordinal",
                "question",
                "question_sha256",
                "reader_route",
                "top_k",
                "selected_fact_ids",
                "selected_fact_ids_sha256",
                "retrieval_trace_sha256",
                "context_sha256",
                "operation_context_audit_sha256",
                "semantic_plan_sha256",
                "prediction",
                "prediction_sha256",
                "reader_stages",
            }:
                raise PredictExecutorError("prediction row fieldset drift")
            for key in (
                "row_id",
                "task",
                "question_id",
                "source_ordinal",
                "question",
                "reader_route",
                "top_k",
            ):
                if row.get(key) != expected_query.get(key):
                    raise PredictExecutorError(
                        f"prediction/query row order drift: {cell_id}:{key}"
                    )
            prediction = str(row.get("prediction", ""))
            stages = row.get("reader_stages")
            expected_stage_names = (
                ["focused", "answer"]
                if expected_query["reader_route"] == "two_stage"
                else ["structured_fused"]
            )
            if (
                not prediction.strip()
                or row.get("question_sha256") != _sha256(expected_query["question"])
                or row.get("prediction_sha256") != _sha256(prediction)
                or not isinstance(row.get("selected_fact_ids"), list)
                or row.get("selected_fact_ids_sha256")
                != _sha256(row["selected_fact_ids"])
                or not all(
                    _is_sha256(row.get(key))
                    for key in (
                        "retrieval_trace_sha256",
                        "context_sha256",
                        "semantic_plan_sha256",
                    )
                )
                or not isinstance(stages, list)
                or [stage.get("stage") for stage in stages] != expected_stage_names
                or any(
                    not isinstance(stage, Mapping)
                    or set(stage)
                    != {"stage", "request_sha256", "response_sha256"}
                    or not _is_sha256(stage.get("request_sha256"))
                    or not _is_sha256(stage.get("response_sha256"))
                    for stage in stages
                )
                or (
                    architecture == "online_operation_context_v3"
                    and not _is_sha256(row.get("operation_context_audit_sha256"))
                )
                or (
                    architecture != "online_operation_context_v3"
                    and row.get("operation_context_audit_sha256") is not None
                )
            ):
                raise PredictExecutorError("prediction row content/binding drift")
            retrieval_architecture = ARCHITECTURE_TO_BINDING[architecture]
            binding_key = (
                backbone,
                retrieval_architecture,
                str(expected_query["row_id"]),
            )
            binding_value = (
                str(row["retrieval_trace_sha256"]),
                str(row["selected_fact_ids_sha256"]),
            )
            previous_binding = prediction_bindings.get(binding_key)
            if previous_binding is not None and previous_binding != binding_value:
                raise PredictExecutorError("shared retrieval binding drift")
            prediction_bindings[binding_key] = binding_value
            prediction_context_bindings[
                (backbone, architecture, str(expected_query["row_id"]))
            ] = str(row["context_sha256"])
            sealed_prediction_rows[
                (backbone, architecture, str(expected_query["row_id"]))
            ] = row
    if seen != set(matrix.matrix_cell_ids()):
        raise PredictExecutorError("prediction completion misses matrix cells")

    requests = value.get("retrieval_requests")
    if (
        not isinstance(requests, list)
        or value.get("retrieval_requests_sha256") != _sha256(requests)
    ):
        raise PredictExecutorError("retrieval request manifest seal drift")
    request_keys = {
        "backbone",
        "retrieval_architecture",
        "row_id",
        "task",
        "retrieval_trace_sha256",
        "selected_fact_ids",
        "selected_fact_ids_sha256",
        "batch_start",
        "batch_size",
        "candidate_fact_ids",
        "candidate_fact_ids_sha256",
        "question_sha256",
        "documents_sha256",
        "instruction_sha256",
        "request_sha256",
        "response_sha256",
        "cache_identity",
        "cache_metadata",
    }
    query_by_id = {str(row["row_id"]): row for row in query_rows}
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[str] = set()
    artifact_facts: dict[tuple[str, str, str], dict[Any, str]] = {}
    artifact_values: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in requests:
        if not isinstance(row, Mapping) or set(row) != request_keys:
            raise PredictExecutorError("retrieval request manifest fieldset drift")
        key = (
            str(row["backbone"]),
            str(row["retrieval_architecture"]),
            str(row["row_id"]),
        )
        query = query_by_id.get(key[2])
        identity = str(row["cache_identity"])
        artifact_key = (key[0], key[1], str(row["task"]))
        if query is not None and artifact_key not in artifact_facts:
            artifact_path, _ = _artifact_for(
                build, key[0], key[1], str(row["task"])
            )
            artifact_value = _read_mapping(
                artifact_path, "retrieval request graph artifact"
            )
            artifact_values[artifact_key] = artifact_value
            facts = (
                artifact_value.get("FACTS", [])
                if key[1] == "mgtm_matched"
                else artifact_value.get("fact_nodes", [])
            )
            artifact_facts[artifact_key] = {
                fact["id"]: str(fact["text"])
                for fact in facts
                if isinstance(fact, Mapping)
            }
        candidate_ids = row.get("candidate_fact_ids")
        selected_ids = row.get("selected_fact_ids")
        fact_lookup = artifact_facts.get(artifact_key, {})
        documents = (
            [fact_lookup.get(fact_id) for fact_id in candidate_ids]
            if isinstance(candidate_ids, list)
            else []
        )
        expected_metadata = {
            "backbone": key[0],
            "retrieval_architecture": key[1],
            "task": str(row["task"]),
            "row_id": key[2],
            "batch_start": row["batch_start"],
            "batch_size": row["batch_size"],
        }
        if (
            key not in prediction_bindings
            or query is None
            or row["task"] != query["task"]
            or row.get("question_sha256") != _sha256(query["question"])
            or row.get("instruction_sha256")
            != _sha256(retrieval_authority["instruction"])
            or not isinstance(candidate_ids, list)
            or not isinstance(selected_ids, list)
            or row.get("selected_fact_ids_sha256") != _sha256(selected_ids)
            or len(candidate_ids) != row["batch_size"]
            or row.get("candidate_fact_ids_sha256") != _sha256(candidate_ids)
            or any(document is None for document in documents)
            or row.get("documents_sha256") != _sha256(documents)
            or row.get("cache_metadata") != expected_metadata
            or (
                str(row["retrieval_trace_sha256"]),
                str(row["selected_fact_ids_sha256"]),
            )
            != prediction_bindings[key]
            or not isinstance(row["batch_start"], int)
            or not isinstance(row["batch_size"], int)
            or not 1 <= row["batch_size"] <= QWEN_MAX_DOCUMENTS
            or not all(
                _is_sha256(row[field])
                for field in (
                    "retrieval_trace_sha256",
                    "selected_fact_ids_sha256",
                    "request_sha256",
                    "response_sha256",
                    "cache_identity",
                    "candidate_fact_ids_sha256",
                    "question_sha256",
                    "documents_sha256",
                    "instruction_sha256",
                )
            )
            or identity in identities
        ):
            raise PredictExecutorError("retrieval request manifest binding drift")
        identities.add(identity)
        groups[key].append(row)
    if set(groups) != set(prediction_bindings):
        raise PredictExecutorError("retrieval request logical census drift")
    for rows_for_retrieval in groups.values():
        ordered = sorted(rows_for_retrieval, key=lambda row: row["batch_start"])
        next_start = 0
        all_candidate_ids: list[Any] = []
        for index, row in enumerate(ordered):
            if row["batch_start"] != next_start or (
                index < len(ordered) - 1
                and row["batch_size"] != QWEN_MAX_DOCUMENTS
            ):
                raise PredictExecutorError("Qwen batch partition drift")
            next_start += row["batch_size"]
            all_candidate_ids.extend(row["candidate_fact_ids"])
        if len(set(map(str, all_candidate_ids))) != len(all_candidate_ids):
            raise PredictExecutorError("Qwen candidate ID deduplication drift")

    reconstructed_retrieval: dict[
        tuple[str, str, str], dict[str, Any]
    ] = {}
    if expected_rows == matrix.EXPECTED_QUESTIONS:
        cache_validator = DashScopeFormalReranker(
            output_root / "provider" / "predict" / "qwen3_rerank",
            allow_model_calls=False,
            approved_provider_attempt_hard_ceiling=budget[
                "stage_provider_attempt_hard_ceilings"
            ]["qwen_rerank"],
            provider_factory=lambda: (_ for _ in ()).throw(
                PredictExecutorError("provider construction during cache validation")
            ),
        )
        cache_validation = cache_validator.validate_manifest(requests)
        aligned_by_identity = cache_validation["aligned_scores_by_identity"]
        graph_memory = importlib.import_module("creamem3.graph_memory_v2")
        for key, rows_for_retrieval in groups.items():
            backbone, retrieval_architecture, row_id = key
            query = query_by_id[row_id]
            artifact_key = (backbone, retrieval_architecture, str(query["task"]))
            graph = artifact_values[artifact_key]
            if retrieval_architecture == "mgtm_matched":
                candidates, trace = _native_candidate_pool(
                    graph,
                    question=str(query["question"]),
                    query_vector=query_vectors[row_id],
                )
                assembler = retrieval_authority["native_assemble"]
                index = None
            else:
                facts = list(graph.get("fact_nodes", []))
                index = graph_memory.GraphMemoryIndex(
                    graph,
                    summary_top_k=SUMMARY_N,
                    dense_fact_top_k=DENSE_N,
                    bm25_fact_top_k=BM25_N,
                    candidate_cap=max(1, len(facts)),
                    rerank_top_k=int(query["top_k"]),
                )
                candidates, trace = index._candidate_pool(
                    str(query["question"]), query_vectors[row_id]
                )
                assembler = None
            exact_candidate_ids = [item["fact"]["id"] for item in candidates]
            ordered_batches = sorted(
                rows_for_retrieval, key=lambda row: row["batch_start"]
            )
            manifest_candidate_ids = [
                fact_id
                for batch in ordered_batches
                for fact_id in batch["candidate_fact_ids"]
            ]
            if manifest_candidate_ids != exact_candidate_ids:
                raise PredictExecutorError("Qwen exact candidate-pool drift")
            qwen_scores = [
                score_value
                for batch in ordered_batches
                for score_value in aligned_by_identity[batch["cache_identity"]]
            ]
            if len(qwen_scores) != len(candidates):
                raise PredictExecutorError("Qwen aligned score census drift")
            selected = [
                item
                for item, _ in sorted(
                    zip(candidates, qwen_scores),
                    key=lambda pair: pair[1],
                    reverse=True,
                )[: int(query["top_k"])]
            ]
            selected_ids = [item["fact"]["id"] for item in selected]
            if any(batch["selected_fact_ids"] != selected_ids for batch in ordered_batches):
                raise PredictExecutorError("Qwen selected top-k fact drift")
            qwen_receipts = [
                {
                    field: batch[field]
                    for field in (
                        "batch_start",
                        "batch_size",
                        "candidate_fact_ids",
                        "documents_sha256",
                        "request_sha256",
                        "response_sha256",
                        "cache_identity",
                        "cache_metadata",
                    )
                }
                for batch in ordered_batches
            ]
            if retrieval_architecture == "mgtm_matched":
                context = str(assembler(graph, selected_ids))
                summaries = list(graph.get("SUMMARIES", []))
                valid_summary_ids = {summary["id"] for summary in summaries}
                rendered_summary_ids: list[Any] = []
                for item in selected:
                    parent = item["fact"].get("parent_summary_id")
                    if (
                        parent in valid_summary_ids
                        and parent not in rendered_summary_ids
                    ):
                        rendered_summary_ids.append(parent)
                trace["policy"]["top_k"] = int(query["top_k"])
                trace["rerank"] = {
                    "selected_fact_ids": selected_ids,
                    "fallback": False,
                    "batches": qwen_receipts,
                }
                trace["assembly"] = {
                    "rendered_event_view_ids": [],
                    "rendered_session_summary_ids": rendered_summary_ids,
                }
            else:
                context, assembly = index._assemble(
                    selected, trace["top_summaries"]
                )
                trace["rerank"] = {
                    "selected_fact_ids": [str(value) for value in selected_ids],
                    "fallback": False,
                    "batches": qwen_receipts,
                }
                trace["assembly"] = assembly
                selected_ids = [str(value) for value in selected_ids]
            expected_trace_sha = _sha256(trace)
            expected_selected_sha = _sha256(selected_ids)
            if (
                prediction_bindings[key]
                != (expected_trace_sha, expected_selected_sha)
                or prediction_context_bindings[
                    (backbone, retrieval_architecture, row_id)
                ]
                != _sha256(context)
            ):
                raise PredictExecutorError("retrieval top-k/context reconstruction drift")
            reconstructed_retrieval[key] = {
                "context": context,
                "trace": trace,
                "selected_fact_ids": selected_ids,
                "graph": graph,
            }
        adapter_path = Path(__file__).with_name("official_g25_runtime_adapter_v19.py")
        if _file_sha256(adapter_path) != G25_RUNTIME_ADAPTER_SHA256:
            raise PredictExecutorError("G25 Reader adapter drift before audit import")
        g25_adapter = importlib.import_module(
            "creamem3.official_g25_runtime_adapter_v19"
        )
        if Path(g25_adapter.__file__).resolve() != adapter_path.resolve():
            raise PredictExecutorError("G25 Reader adapter audit import drift")
        audit_reader = g25_adapter.OfficialGemini25Runtime(
            output_root / "provider" / "predict" / "g25_reader",
            allow_model_calls=False,
            provider_factory=lambda: (_ for _ in ()).throw(
                PredictExecutorError("provider construction during G25 audit")
            ),
        )
        strict_planner = importlib.import_module("creamem3.strict_semantic_planner_v8")
        expected_g25_identities: dict[str, set[str]] = {
            stage: set()
            for stage in ("planner", "focused", "answer", "structured_fused")
        }
        semantic_plans: dict[str, dict[str, Any]] = {}
        for query in query_rows:
            row_id = str(query["row_id"])
            question = str(query["question"])
            planner_call = _validate_g25_cached_call(
                audit_reader=audit_reader,
                stage="planner",
                prompt=strict_planner.build_planner_prompt(question),
                json_mode=True,
                max_output_tokens=512,
                cache_metadata={
                    "row_id": query["row_id"],
                    "question_sha256": _sha256(question),
                },
            )
            expected_g25_identities["planner"].add(planner_call["resume_hash"])
            parsed_plan = strict_planner.strict_plan_from_raw(
                planner_call["text"]
            ).to_dict()
            sealed_plan = {
                "plan": parsed_plan,
                "request_sha256": planner_call["request_sha256"],
                "response_sha256": planner_call["response_sha256"],
            }
            semantic_plans[row_id] = parsed_plan
            for backbone in matrix.BACKBONE_IDS:
                for architecture in matrix.ARCHITECTURE_IDS:
                    sealed_row = sealed_prediction_rows[
                        (backbone, architecture, row_id)
                    ]
                    if sealed_row.get("semantic_plan_sha256") != _sha256(sealed_plan):
                        raise PredictExecutorError("semantic planner/cache binding drift")
        online_operation = importlib.import_module(
            "creamem3.online_operation_context_v8"
        )
        for backbone in matrix.BACKBONE_IDS:
            for architecture in matrix.ARCHITECTURE_IDS:
                cell_id = f"{backbone}__{architecture}"
                retrieval_architecture = ARCHITECTURE_TO_BINDING[architecture]
                for query in query_rows:
                    row_id = str(query["row_id"])
                    question = str(query["question"])
                    sealed_row = sealed_prediction_rows[(backbone, architecture, row_id)]
                    reconstruction = reconstructed_retrieval[
                        (backbone, retrieval_architecture, row_id)
                    ]
                    context = str(reconstruction["context"])
                    operation_audit: Mapping[str, Any] | None = None
                    if architecture == "online_operation_context_v3":
                        operation = str(
                            semantic_plans[row_id]["plan"]["operation"]
                        )
                        context, operation_audit = (
                            online_operation.materialize_online_operation_context(
                                operation,
                                {
                                    "question": question,
                                    "retrieved": str(reconstruction["context"]),
                                    "retrieval_trace": copy.deepcopy(
                                        reconstruction["trace"]
                                    ),
                                },
                                reconstruction["graph"],
                            )
                        )
                    if (
                        sealed_row.get("context_sha256") != _sha256(context)
                        or sealed_row.get("operation_context_audit_sha256")
                        != (
                            _sha256(operation_audit)
                            if operation_audit is not None
                            else None
                        )
                    ):
                        raise PredictExecutorError(
                            "Reader context/operation-policy reconstruction drift"
                        )
                    stages = list(sealed_row["reader_stages"])
                    if query["reader_route"] == "two_stage":
                        focused_prompt = retrieval_authority["focused"].replace(
                            "{question}", question
                        ).replace("{context}", context)
                        focused_call = _validate_g25_cached_call(
                            audit_reader=audit_reader,
                            stage="focused",
                            prompt=focused_prompt,
                            json_mode=True,
                            max_output_tokens=2600,
                            cache_metadata={
                                "cell_id": cell_id,
                                "row_id": query["row_id"],
                                "context_sha256": _sha256(context),
                            },
                        )
                        expected_g25_identities["focused"].add(
                            focused_call["resume_hash"]
                        )
                        aggregation = _parse_json_object(focused_call["text"])
                        if not aggregation:
                            aggregation = {"_failed": True}
                        qa_context = retrieval_authority["render"](
                            aggregation, context, "candidate_first"
                        )
                        answer_prompt = retrieval_authority["open"].format(
                            context=qa_context, question=question
                        )
                        answer_call = _validate_g25_cached_call(
                            audit_reader=audit_reader,
                            stage="answer",
                            prompt=answer_prompt,
                            json_mode=False,
                            max_output_tokens=500,
                            cache_metadata={
                                "cell_id": cell_id,
                                "row_id": query["row_id"],
                                "qa_context_sha256": _sha256(qa_context),
                            },
                        )
                        expected_g25_identities["answer"].add(
                            answer_call["resume_hash"]
                        )
                        expected_stages = [
                            {
                                "stage": "focused",
                                "request_sha256": focused_call["request_sha256"],
                                "response_sha256": focused_call["response_sha256"],
                            },
                            {
                                "stage": "answer",
                                "request_sha256": answer_call["request_sha256"],
                                "response_sha256": answer_call["response_sha256"],
                            },
                        ]
                        prediction = str(answer_call["text"]).strip()
                    else:
                        fused_prompt = retrieval_authority["fused"].format(
                            question=question, context=context
                        )
                        fused_call = _validate_g25_cached_call(
                            audit_reader=audit_reader,
                            stage="structured_fused",
                            prompt=fused_prompt,
                            json_mode=True,
                            max_output_tokens=1200,
                            cache_metadata={
                                "cell_id": cell_id,
                                "row_id": query["row_id"],
                                "context_sha256": _sha256(context),
                            },
                        )
                        expected_g25_identities["structured_fused"].add(
                            fused_call["resume_hash"]
                        )
                        expected_stages = [
                            {
                                "stage": "structured_fused",
                                "request_sha256": fused_call["request_sha256"],
                                "response_sha256": fused_call["response_sha256"],
                            }
                        ]
                        parsed = _parse_json_object(fused_call["text"])
                        prediction = str(
                            parsed.get("final_answer") or "Not found in memory."
                        ).strip()
                    if (
                        stages != expected_stages
                        or sealed_row.get("prediction") != prediction
                        or sealed_row.get("prediction_sha256") != _sha256(prediction)
                    ):
                        raise PredictExecutorError(
                            "Reader output/durable cache evidence drift"
                        )
        _validate_g25_cache_census(
            audit_reader=audit_reader,
            expected_by_stage=expected_g25_identities,
        )
    actual_provider_calls = _validate_predict_provider_usage(
        output_root=output_root,
        usage=value.get("provider_usage", {}),
        budget=budget,
        qwen_first_attempt_floor=len(requests),
    )
    if value.get("provider_calls") != actual_provider_calls:
        raise PredictExecutorError("prediction provider census binding drift")
    return value


def execute_predict_with_backends(
    *,
    output_root: Path,
    text_backend: TextBackend,
    embedding_backend: EmbeddingBackend,
    rerank_backend: RerankBackend,
    budget_gate: Mapping[str, Any],
    enforce_full_scope: bool = True,
    max_workers: int = 1,
) -> dict[str, Any]:
    workers = int(max_workers)
    if workers < 1:
        raise PredictExecutorError("prediction max_workers must be positive")
    protocol = matrix.load_protocol()
    validate_predict_authorities(protocol)
    build = _load_build(output_root, protocol)
    queries = _query_payload(output_root)
    rows = list(queries["rows"])
    expected_rows = matrix.EXPECTED_QUESTIONS if enforce_full_scope else len(rows)
    complete_path = Path(output_root) / "predict" / "complete.json"
    if complete_path.is_file():
        complete = _prediction_complete(
            output_root, expected_rows=expected_rows, protocol=protocol
        )
        return {
            "status": "resumed_complete",
            "provider_calls": 0,
            "complete_path": str(complete_path.resolve()),
            "complete_sha256": complete["complete_sha256"],
        }
    budget = compute_predict_budget(output_root=output_root, protocol=protocol)
    if budget_gate.get("budget") != budget:
        raise PredictExecutorError("injected predict budget gate drift")
    authority = _reader_authority(protocol)
    questions = [str(row["question"]) for row in rows]
    vectors = list(
        embedding_backend.embed(
            questions,
            tag="memeye_v19_predict_query_embedding",
            cache_metadata={"query_rows_sha256": queries["rows_sha256"]},
        )
    )
    if len(vectors) != len(rows):
        raise PredictExecutorError("query embedding row census drift")
    vector_by_row = {
        str(row["row_id"]): _validate_vector(vector)
        for row, vector in zip(rows, vectors, strict=True)
    }
    if not hasattr(embedding_backend, "batch_receipts"):
        raise PredictExecutorError("formal query embedding backend lacks batch receipts")
    embedding_rows = [
        {
            "row_id": row["row_id"],
            "question_sha256": _sha256(row["question"]),
            "vector_sha256": _sha256(vector_by_row[str(row["row_id"])]),
            "vector": vector_by_row[str(row["row_id"])],
        }
        for row in rows
    ]
    embedding_batches = list(embedding_backend.batch_receipts())
    embedding_value = {
        "schema_version": QUERY_EMBEDDING_SCHEMA,
        "protocol_sha256": _sha256(protocol),
        "query_rows_sha256": queries["rows_sha256"],
        "rows": embedding_rows,
        "rows_sha256": _sha256(embedding_rows),
        "batches": embedding_batches,
        "batches_sha256": _sha256(embedding_batches),
        "complete": True,
    }
    embedding_value["query_embedding_sha256"] = _sha256(embedding_value)
    embedding_path = Path(output_root) / "predict" / "query_embeddings.json"
    _write_once(embedding_path, embedding_value)

    strict_planner = importlib.import_module("creamem3.strict_semantic_planner_v8")

    def plan_one(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        response = _reply(
            text_backend.generate(
                "planner",
                strict_planner.build_planner_prompt(str(row["question"])),
                json_mode=True,
                max_output_tokens=512,
                cache_metadata={"row_id": row["row_id"], "question_sha256": _sha256(row["question"])},
            )
        )
        result = strict_planner.strict_plan_from_raw(response.text)
        return (
            str(row["row_id"]),
            {
                "plan": result.to_dict(),
                "request_sha256": response.request_sha256,
                "response_sha256": response.response_sha256,
            },
        )

    plans = dict(_ordered_parallel_map(plan_one, rows, max_workers=workers))

    artifact_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    artifact_cache_lock = threading.Lock()
    retrieval_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    online_operation = importlib.import_module("creamem3.online_operation_context_v8")
    cell_receipts: list[dict[str, Any]] = []
    for backbone in matrix.BACKBONE_IDS:
        for architecture in matrix.ARCHITECTURE_IDS:
            cell_id = f"{backbone}__{architecture}"
            retrieval_architecture = ARCHITECTURE_TO_BINDING[architecture]

            # A cell owns one unique row_id per query.  Architectures are still
            # evaluated sequentially, so the online-operation cell observes
            # the already sealed online-base retrieval cache exactly as before.
            # Within a cell, distinct row identities make the durable Gemini
            # and Qwen lifecycle paths disjoint and safe for ordered workers.
            def predict_one(row: Mapping[str, Any]) -> dict[str, Any]:
                task = str(row["task"])
                artifact_key = (backbone, retrieval_architecture, task)
                if artifact_key not in artifact_cache:
                    with artifact_cache_lock:
                        if artifact_key not in artifact_cache:
                            artifact_path, _ = _artifact_for(
                                build, backbone, retrieval_architecture, task
                            )
                            artifact_cache[artifact_key] = _read_mapping(
                                artifact_path, "sealed retrieval artifact"
                            )
                graph = artifact_cache[artifact_key]
                retrieval_key = (backbone, retrieval_architecture, str(row["row_id"]))
                if retrieval_key not in retrieval_cache:
                    metadata = {
                        "backbone": backbone,
                        "retrieval_architecture": retrieval_architecture,
                        "task": task,
                        "row_id": row["row_id"],
                    }
                    if retrieval_architecture == "mgtm_matched":
                        context, selected, trace = _native_retrieve(
                            graph,
                            question=str(row["question"]),
                            query_vector=vector_by_row[str(row["row_id"])],
                            top_k=int(row["top_k"]),
                            reranker=rerank_backend,
                            instruction=authority["instruction"],
                            metadata=metadata,
                            assemble=authority["native_assemble"],
                        )
                    else:
                        context, selected, trace = _creamem_retrieve(
                            graph,
                            question=str(row["question"]),
                            query_vector=vector_by_row[str(row["row_id"])],
                            top_k=int(row["top_k"]),
                            reranker=rerank_backend,
                            instruction=authority["instruction"],
                            metadata=metadata,
                        )
                    retrieval_cache[retrieval_key] = {
                        "context": context,
                        "selected_fact_ids": list(selected),
                        "trace": trace,
                    }
                retrieval = retrieval_cache[retrieval_key]
                context = str(retrieval["context"])
                operation_audit: dict[str, Any] | None = None
                if architecture == "online_operation_context_v3":
                    operation = str(
                        plans[str(row["row_id"])]["plan"]["plan"]["operation"]
                    )
                    context, operation_audit = online_operation.materialize_online_operation_context(
                        operation,
                        {
                            "question": str(row["question"]),
                            "retrieved": context,
                            "retrieval_trace": copy.deepcopy(retrieval["trace"]),
                        },
                        graph,
                    )
                stages: list[dict[str, Any]] = []
                if row["reader_route"] == "two_stage":
                    focused_prompt = authority["focused"].replace(
                        "{question}", str(row["question"])
                    ).replace("{context}", context)
                    focused = _reply(
                        text_backend.generate(
                            "focused",
                            focused_prompt,
                            json_mode=True,
                            max_output_tokens=2600,
                            cache_metadata={"cell_id": cell_id, "row_id": row["row_id"], "context_sha256": _sha256(context)},
                        )
                    )
                    aggregation = _parse_json_object(focused.text)
                    if not aggregation:
                        aggregation = {"_failed": True}
                    qa_context = authority["render"](
                        aggregation, context, "candidate_first"
                    )
                    final_prompt = authority["open"].format(
                        context=qa_context, question=str(row["question"])
                    )
                    answer = _reply(
                        text_backend.generate(
                            "answer",
                            final_prompt,
                            json_mode=False,
                            max_output_tokens=500,
                            cache_metadata={"cell_id": cell_id, "row_id": row["row_id"], "qa_context_sha256": _sha256(qa_context)},
                        )
                    )
                    prediction = answer.text.strip()
                    stages.extend(
                        [
                            {"stage": "focused", "request_sha256": focused.request_sha256, "response_sha256": focused.response_sha256},
                            {"stage": "answer", "request_sha256": answer.request_sha256, "response_sha256": answer.response_sha256},
                        ]
                    )
                elif row["reader_route"] == "structured_fused":
                    prompt = authority["fused"].format(
                        question=str(row["question"]), context=context
                    )
                    fused = _reply(
                        text_backend.generate(
                            "structured_fused",
                            prompt,
                            json_mode=True,
                            max_output_tokens=1200,
                            cache_metadata={"cell_id": cell_id, "row_id": row["row_id"], "context_sha256": _sha256(context)},
                        )
                    )
                    parsed = _parse_json_object(fused.text)
                    prediction = str(
                        parsed.get("final_answer") or "Not found in memory."
                    ).strip()
                    stages.append(
                        {"stage": "structured_fused", "request_sha256": fused.request_sha256, "response_sha256": fused.response_sha256}
                    )
                else:
                    raise PredictExecutorError("unknown Reader route")
                if not prediction:
                    raise PredictExecutorError("Reader returned an empty prediction")
                return {
                    "row_id": row["row_id"],
                    "task": task,
                    "question_id": row["question_id"],
                    "source_ordinal": row["source_ordinal"],
                    "question": row["question"],
                    "question_sha256": _sha256(row["question"]),
                    "reader_route": row["reader_route"],
                    "top_k": row["top_k"],
                    "selected_fact_ids": retrieval["selected_fact_ids"],
                    "selected_fact_ids_sha256": _sha256(retrieval["selected_fact_ids"]),
                    "retrieval_trace_sha256": _sha256(retrieval["trace"]),
                    "context_sha256": _sha256(context),
                    "operation_context_audit_sha256": (
                        _sha256(operation_audit) if operation_audit is not None else None
                    ),
                    "semantic_plan_sha256": _sha256(plans[str(row["row_id"])]),
                    "prediction": prediction,
                    "prediction_sha256": _sha256(prediction),
                    "reader_stages": stages,
                }

            prediction_rows = _ordered_parallel_map(
                predict_one, rows, max_workers=workers
            )
            if len(prediction_rows) != expected_rows or len(
                {row["row_id"] for row in prediction_rows}
            ) != expected_rows:
                raise PredictExecutorError("prediction cell row census drift")
            cell_value = {
                "schema_version": PREDICTION_CELL_SCHEMA,
                "cell_id": cell_id,
                "backbone": backbone,
                "architecture": architecture,
                "protocol_sha256": _sha256(protocol),
                "query_rows_sha256": queries["rows_sha256"],
                "rows": prediction_rows,
                "rows_sha256": _sha256(prediction_rows),
                "complete": True,
            }
            cell_value["cell_sha256"] = _sha256(cell_value)
            cell_path = Path(output_root) / "predict" / "cells" / cell_id / "predictions.json"
            _write_once(cell_path, cell_value)
            cell_receipts.append(
                {
                    "cell_id": cell_id,
                    "backbone": backbone,
                    "architecture": architecture,
                    "rows": len(prediction_rows),
                    "complete": True,
                    "prediction_path": str(cell_path.resolve()),
                    "prediction_sha256": _file_sha256(cell_path),
                    "cell_sha256": cell_value["cell_sha256"],
                }
            )
    query_by_id = {str(row["row_id"]): row for row in rows}
    retrieval_requests: list[dict[str, Any]] = []
    for (backbone, retrieval_architecture, row_id), retrieval in sorted(
        retrieval_cache.items()
    ):
        query_row = query_by_id[row_id]
        trace = retrieval["trace"]
        batches = trace.get("rerank", {}).get("batches", [])
        if not isinstance(batches, list) or not batches:
            raise PredictExecutorError("retrieval lacks sealed Qwen batches")
        for batch in batches:
            retrieval_requests.append(
                {
                    "backbone": backbone,
                    "retrieval_architecture": retrieval_architecture,
                    "row_id": row_id,
                    "task": query_row["task"],
                    "retrieval_trace_sha256": _sha256(trace),
                    "selected_fact_ids": retrieval["selected_fact_ids"],
                    "selected_fact_ids_sha256": _sha256(
                        retrieval["selected_fact_ids"]
                    ),
                    "batch_start": batch["batch_start"],
                    "batch_size": batch["batch_size"],
                    "candidate_fact_ids": batch["candidate_fact_ids"],
                    "candidate_fact_ids_sha256": _sha256(
                        batch["candidate_fact_ids"]
                    ),
                    "question_sha256": _sha256(query_row["question"]),
                    "documents_sha256": batch["documents_sha256"],
                    "instruction_sha256": _sha256(authority["instruction"]),
                    "request_sha256": batch["request_sha256"],
                    "response_sha256": batch["response_sha256"],
                    "cache_identity": batch["cache_identity"],
                    "cache_metadata": batch["cache_metadata"],
                }
            )
    text_usage = _durable_gemini_usage(
        text_backend.usage_receipt(),
        cache_root=Path(output_root) / "provider" / "predict" / "g25_reader",
    )
    embedding_usage = dict(embedding_backend.usage_receipt())
    rerank_usage = dict(rerank_backend.usage_receipt())
    provider_calls = sum(
        _provider_calls(value) for value in (text_usage, embedding_usage, rerank_usage)
    )
    if provider_calls > budget["provider_attempt_hard_ceiling"]:
        raise PredictExecutorError("predict provider-attempt hard ceiling exceeded")
    complete = {
        "schema_version": matrix.PREDICTION_COMPLETE_SCHEMA,
        "complete": True,
        "protocol_sha256": _sha256(protocol),
        "predict_executor_file_sha256": _file_sha256(Path(__file__).resolve()),
        "build_complete_file_sha256": budget["build_complete_file_sha256"],
        "query_file_sha256": budget["query_file_sha256"],
        "predict_budget_sha256": budget["predict_budget_sha256"],
        "predict_budget_gate_file_sha256": budget_gate.get("file_sha256"),
        "query_embedding_path": str(embedding_path.resolve()),
        "query_embedding_file_sha256": _file_sha256(embedding_path),
        "query_embedding_sha256": embedding_value["query_embedding_sha256"],
        "retrieval_requests": retrieval_requests,
        "retrieval_requests_sha256": _sha256(retrieval_requests),
        "cells": sorted(cell_receipts, key=lambda row: row["cell_id"]),
        "provider_usage": {
            "g25": text_usage,
            "query_embedding": embedding_usage,
            "qwen_rerank": rerank_usage,
        },
        "provider_calls": provider_calls,
        "provider_attempt_hard_ceiling": budget["provider_attempt_hard_ceiling"],
        "score_fields_present": False,
    }
    complete["complete_sha256"] = _sha256(complete)
    _write_once(complete_path, complete)
    _prediction_complete(output_root, expected_rows=expected_rows, protocol=protocol)
    return {
        "status": "complete",
        "provider_calls": provider_calls,
        "complete_path": str(complete_path.resolve()),
        "complete_sha256": complete["complete_sha256"],
        "cells": len(cell_receipts),
        "rows_per_cell": expected_rows,
    }


class DashScopeFormalReranker:
    """Durable, no-fallback Qwen runtime with a global physical-call ledger."""

    def __init__(
        self,
        cache_root: Path,
        *,
        allow_model_calls: bool,
        approved_provider_attempt_hard_ceiling: int,
        provider_factory: Callable[[], Any] | None = None,
        parallel_safe: bool = False,
    ) -> None:
        if approved_provider_attempt_hard_ceiling <= 0:
            raise PredictExecutorError("Qwen provider ceiling must be positive")
        self.cache_root = Path(cache_root).resolve()
        self.allow_model_calls = bool(allow_model_calls)
        self.ceiling = int(approved_provider_attempt_hard_ceiling)
        self.parallel_safe = bool(parallel_safe)
        self.provider_factory = provider_factory or self._default_provider_factory
        self._provider: Any | None = None
        self._provider_calls_this_process = 0
        self._cache_hits = 0
        self._state_lock = threading.Lock()
        self._budget_thread_lock = threading.Lock()

    @staticmethod
    def _default_provider_factory() -> Any:
        import urllib.error
        import urllib.request

        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise PredictExecutorError("DASHSCOPE_API_KEY is required")

        class Provider:
            def rerank(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
                payload = _canonical_bytes(request)
                http_request = urllib.request.Request(
                    QWEN_ENDPOINT,
                    data=payload,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(http_request, timeout=120) as response:
                        return json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as error:
                    status = int(error.code)
                    raise ExplicitQwenProviderError(
                        f"DashScope rerank returned HTTP {status}",
                        status_code=status,
                        retryable=status in {429, 500, 502, 503, 504},
                    ) from error
                except (TimeoutError, urllib.error.URLError) as error:
                    raise IndeterminateQwenProviderError(
                        "DashScope transport outcome is indeterminate"
                    ) from error

        return Provider()

    def _durable_intents(self) -> int:
        root = self.cache_root / "attempts"
        return len(list(root.rglob("attempt_*_intent.json"))) if root.is_dir() else 0

    def _acquire(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise PredictExecutorError(f"stale Qwen lifecycle lock: {path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def _provider_after_gate(self) -> Any:
        if not self.allow_model_calls:
            raise PredictExecutorError("formal Qwen calls are not authorized")
        with self._state_lock:
            if self._provider is None:
                self._provider = self.provider_factory()
            return self._provider

    @staticmethod
    def _aligned(response: Mapping[str, Any], size: int) -> list[float]:
        items = (response.get("output") or {}).get("results", [])
        if not isinstance(items, list) or len(items) != size:
            raise PredictExecutorError("Qwen response result census drift")
        aligned: list[float | None] = [None] * size
        for item in items:
            if not isinstance(item, Mapping):
                raise PredictExecutorError("Qwen result row is invalid")
            index = int(item["index"])
            score = float(item["relevance_score"])
            if not 0 <= index < size or aligned[index] is not None or not math.isfinite(score):
                raise PredictExecutorError("Qwen response index/score drift")
            aligned[index] = score
        if any(value is None for value in aligned):
            raise PredictExecutorError("Qwen response omitted a document")
        return [float(value) for value in aligned]

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> RerankReply:
        docs = [str(value) for value in documents]
        if not docs or len(docs) > QWEN_MAX_DOCUMENTS or any(not value.strip() for value in docs):
            raise PredictExecutorError("formal Qwen document batch drift")
        request = {
            "model": QWEN_MODEL,
            "input": {"query": str(query), "documents": docs},
            "parameters": {"return_documents": False, "top_n": len(docs), "instruct": str(instruction)},
        }
        identity = _sha256(
            {
                "request": request,
                "cache_metadata": dict(cache_metadata or {}),
                "endpoint": QWEN_ENDPOINT,
            }
        )
        cache_path = self.cache_root / "responses" / identity[:2] / f"{identity}.json"
        attempt_root = self.cache_root / "attempts" / identity
        if cache_path.is_file():
            value = _read_mapping(cache_path, "formal Qwen cache")
            scores = [float(item) for item in value.get("aligned_scores", [])]
            try:
                ordinal = int(value.get("successful_attempt_ordinal"))
            except (TypeError, ValueError) as exc:
                raise PredictExecutorError("formal Qwen success ordinal drift") from exc
            intent_path = attempt_root / f"attempt_{ordinal:03d}_intent.json"
            result_path = attempt_root / f"attempt_{ordinal:03d}_result.json"
            raw_path = attempt_root / f"attempt_{ordinal:03d}_raw_response.json"
            intent = _read_mapping(intent_path, "formal Qwen success intent")
            result = _read_mapping(result_path, "formal Qwen success result")
            raw_receipt = _read_mapping(raw_path, "formal Qwen raw response")
            raw = raw_receipt.get("raw_response")
            recomputed_scores = self._aligned(
                raw if isinstance(raw, Mapping) else {}, len(docs)
            )
            if (
                ordinal < 1
                or ordinal > MAX_PROVIDER_ATTEMPTS
                or
                set(value)
                != {
                    "schema_version",
                    "identity",
                    "request_sha256",
                    "response_sha256",
                    "aligned_scores_sha256",
                    "aligned_scores",
                    "successful_attempt_ordinal",
                    "successful_intent_file_sha256",
                    "successful_result_file_sha256",
                    "raw_response_file_sha256",
                    "explicit_retry_chain",
                }
                or set(intent)
                != {
                    "schema_version",
                    "identity",
                    "ordinal",
                    "request_sha256",
                    "request",
                    "documents_sha256",
                    "instruction_sha256",
                    "cache_metadata",
                    "approved_global_attempt_hard_ceiling",
                }
                or set(result)
                != {
                    "schema_version",
                    "identity",
                    "ordinal",
                    "status",
                    "intent_file_sha256",
                    "aligned_scores_sha256",
                    "raw_response_sha256",
                    "raw_response_file_sha256",
                }
                or set(raw_receipt)
                != {
                    "schema_version",
                    "identity",
                    "ordinal",
                    "request_sha256",
                    "raw_response_sha256",
                    "raw_response",
                }
                or
                value.get("schema_version") != QWEN_CACHE_SCHEMA
                or value.get("identity") != identity
                or value.get("request_sha256") != _sha256(request)
                or len(scores) != len(docs)
                or not all(math.isfinite(item) for item in scores)
                or scores != recomputed_scores
                or value.get("successful_intent_file_sha256")
                != _file_sha256(intent_path)
                or value.get("successful_result_file_sha256")
                != _file_sha256(result_path)
                or value.get("raw_response_file_sha256") != _file_sha256(raw_path)
                or intent.get("schema_version") != QWEN_CACHE_SCHEMA
                or intent.get("identity") != identity
                or intent.get("ordinal") != ordinal
                or intent.get("request_sha256") != _sha256(request)
                or intent.get("request") != request
                or intent.get("documents_sha256") != _sha256(docs)
                or intent.get("instruction_sha256") != _sha256(instruction)
                or intent.get("cache_metadata") != dict(cache_metadata or {})
                or intent.get("approved_global_attempt_hard_ceiling") != self.ceiling
                or result.get("schema_version") != QWEN_CACHE_SCHEMA
                or result.get("identity") != identity
                or result.get("ordinal") != ordinal
                or result.get("status") != "success"
                or result.get("intent_file_sha256") != _file_sha256(intent_path)
                or result.get("raw_response_file_sha256") != _file_sha256(raw_path)
                or result.get("raw_response_sha256") != _sha256(raw)
                or result.get("aligned_scores_sha256") != _sha256(scores)
                or raw_receipt.get("schema_version") != QWEN_CACHE_SCHEMA
                or raw_receipt.get("identity") != identity
                or raw_receipt.get("ordinal") != ordinal
                or raw_receipt.get("request_sha256") != _sha256(request)
                or raw_receipt.get("raw_response_sha256") != _sha256(raw)
                or value.get("aligned_scores_sha256") != _sha256(scores)
                or value.get("response_sha256") != _sha256(scores)
            ):
                raise PredictExecutorError("formal Qwen cache binding drift")
            retry_chain: list[dict[str, Any]] = []
            for previous_ordinal in range(1, ordinal):
                previous_intent_path = (
                    attempt_root
                    / f"attempt_{previous_ordinal:03d}_intent.json"
                )
                previous_result_path = (
                    attempt_root
                    / f"attempt_{previous_ordinal:03d}_result.json"
                )
                previous_intent = _read_mapping(
                    previous_intent_path, "formal Qwen previous intent"
                )
                previous_result = _read_mapping(
                    previous_result_path, "formal Qwen previous result"
                )
                if (
                    set(previous_intent)
                    != {
                        "schema_version",
                        "identity",
                        "ordinal",
                        "request_sha256",
                        "request",
                        "documents_sha256",
                        "instruction_sha256",
                        "cache_metadata",
                        "approved_global_attempt_hard_ceiling",
                    }
                    or set(previous_result)
                    != {
                        "schema_version",
                        "identity",
                        "ordinal",
                        "status",
                        "status_code",
                        "retryable",
                        "intent_file_sha256",
                        "error_sha256",
                    }
                    or
                    previous_intent.get("identity") != identity
                    or previous_intent.get("ordinal") != previous_ordinal
                    or previous_intent.get("request_sha256") != _sha256(request)
                    or previous_intent.get("request") != request
                    or previous_result.get("identity") != identity
                    or previous_result.get("ordinal") != previous_ordinal
                    or previous_result.get("status") != "explicit_error"
                    or previous_result.get("retryable") is not True
                    or previous_result.get("intent_file_sha256")
                    != _file_sha256(previous_intent_path)
                ):
                    raise PredictExecutorError("formal Qwen retry chain drift")
                retry_chain.append(
                    {
                        "ordinal": previous_ordinal,
                        "intent_file_sha256": _file_sha256(previous_intent_path),
                        "result_file_sha256": _file_sha256(previous_result_path),
                        "status_code": previous_result.get("status_code"),
                    }
                )
            if value.get("explicit_retry_chain") != retry_chain:
                raise PredictExecutorError("formal Qwen cache retry-chain binding drift")
            for later_ordinal in range(ordinal + 1, MAX_PROVIDER_ATTEMPTS + 1):
                if any(
                    (
                        attempt_root
                        / f"attempt_{later_ordinal:03d}_{suffix}.json"
                    ).exists()
                    for suffix in ("intent", "result", "raw_response")
                ):
                    raise PredictExecutorError("formal Qwen post-success attempt drift")
            with self._state_lock:
                self._cache_hits += 1
            return RerankReply(
                tuple(scores), _sha256(request), _sha256(scores), True, identity
            )
        lock = self._acquire(self.cache_root / "locks" / f"{identity}.lock", identity)
        try:
            for ordinal in range(1, MAX_PROVIDER_ATTEMPTS + 1):
                intent_path = attempt_root / f"attempt_{ordinal:03d}_intent.json"
                result_path = attempt_root / f"attempt_{ordinal:03d}_result.json"
                raw_path = attempt_root / f"attempt_{ordinal:03d}_raw_response.json"
                if intent_path.exists():
                    if not result_path.exists():
                        raise PredictExecutorError("orphaned Qwen intent forbids recall")
                    prior = _read_mapping(result_path, "formal Qwen attempt result")
                    if prior.get("status") == "success":
                        raise PredictExecutorError("Qwen success lacks sealed cache")
                    if (
                        prior.get("schema_version") != QWEN_CACHE_SCHEMA
                        or prior.get("identity") != identity
                        or prior.get("ordinal") != ordinal
                        or prior.get("intent_file_sha256") != _file_sha256(intent_path)
                        or prior.get("retryable") is not True
                    ):
                        raise PredictExecutorError("nonretryable Qwen result forbids recall")
                    continue
                intent = {
                    "schema_version": QWEN_CACHE_SCHEMA,
                    "identity": identity,
                    "ordinal": ordinal,
                    "request_sha256": _sha256(request),
                    "request": request,
                    "documents_sha256": _sha256(docs),
                    "instruction_sha256": _sha256(instruction),
                    "cache_metadata": dict(cache_metadata or {}),
                    "approved_global_attempt_hard_ceiling": self.ceiling,
                }

                def reserve_attempt() -> None:
                    if self._durable_intents() >= self.ceiling:
                        raise PredictExecutorError(
                            "Qwen physical-call ceiling exhausted"
                        )
                    _write_once(intent_path, intent)
                    with self._state_lock:
                        self._provider_calls_this_process += 1

                held_budget_lock: Path | None = None
                if self.parallel_safe:
                    # The isolated single-process evaluator releases the
                    # global ledger after its durable reservation, allowing
                    # independent identities to overlap network I/O.
                    with self._budget_thread_lock:
                        budget_lock = self._acquire(
                            self.cache_root / "budget.lock", str(self.ceiling)
                        )
                        try:
                            reserve_attempt()
                        finally:
                            budget_lock.unlink(missing_ok=True)
                else:
                    # Preserve the formal runner's historical global lease
                    # duration when parallel execution was not opted into.
                    held_budget_lock = self._acquire(
                        self.cache_root / "budget.lock", str(self.ceiling)
                    )
                    try:
                        reserve_attempt()
                    except Exception:
                        held_budget_lock.unlink(missing_ok=True)
                        raise
                try:
                    try:
                        raw = self._provider_after_gate().rerank(request)
                    except ExplicitQwenProviderError as error:
                        _write_once(
                            result_path,
                            {
                                "schema_version": QWEN_CACHE_SCHEMA,
                                "identity": identity,
                                "ordinal": ordinal,
                                "status": "explicit_error",
                                "status_code": error.status_code,
                                "retryable": error.retryable,
                                "intent_file_sha256": _file_sha256(intent_path),
                                "error_sha256": _sha256(str(error)),
                            },
                        )
                        if not error.retryable:
                            raise
                        continue
                    except Exception as error:
                        # The write-once intent remains deliberately unresolved.
                        raise IndeterminateQwenProviderError(
                            "Qwen provider outcome is indeterminate; recall prohibited"
                        ) from error
                finally:
                    if held_budget_lock is not None:
                        held_budget_lock.unlink(missing_ok=True)
                scores = self._aligned(raw, len(docs))
                _write_once(
                    raw_path,
                    {
                        "schema_version": QWEN_CACHE_SCHEMA,
                        "identity": identity,
                        "ordinal": ordinal,
                        "request_sha256": _sha256(request),
                        "raw_response_sha256": _sha256(raw),
                        "raw_response": raw,
                    },
                )
                _write_once(
                    result_path,
                    {
                        "schema_version": QWEN_CACHE_SCHEMA,
                        "identity": identity,
                        "ordinal": ordinal,
                        "status": "success",
                        "intent_file_sha256": _file_sha256(intent_path),
                        "aligned_scores_sha256": _sha256(scores),
                        "raw_response_sha256": _sha256(raw),
                        "raw_response_file_sha256": _file_sha256(raw_path),
                    },
                )
                _write_once(
                    cache_path,
                    {
                        "schema_version": QWEN_CACHE_SCHEMA,
                        "identity": identity,
                        "request_sha256": _sha256(request),
                        "response_sha256": _sha256(scores),
                        "aligned_scores_sha256": _sha256(scores),
                        "aligned_scores": scores,
                        "successful_attempt_ordinal": ordinal,
                        "successful_intent_file_sha256": _file_sha256(intent_path),
                        "successful_result_file_sha256": _file_sha256(result_path),
                        "raw_response_file_sha256": _file_sha256(raw_path),
                        "explicit_retry_chain": [
                            {
                                "ordinal": previous_ordinal,
                                "intent_file_sha256": _file_sha256(
                                    attempt_root
                                    / f"attempt_{previous_ordinal:03d}_intent.json"
                                ),
                                "result_file_sha256": _file_sha256(
                                    attempt_root
                                    / f"attempt_{previous_ordinal:03d}_result.json"
                                ),
                                "status_code": _read_mapping(
                                    attempt_root
                                    / f"attempt_{previous_ordinal:03d}_result.json",
                                    "formal Qwen previous result",
                                ).get("status_code"),
                            }
                            for previous_ordinal in range(1, ordinal)
                        ],
                    },
                )
                return RerankReply(
                    tuple(scores), _sha256(request), _sha256(scores), False, identity
                )
            raise PredictExecutorError("Qwen exhausted three explicit retryable attempts")
        finally:
            lock.unlink(missing_ok=True)

    def usage_receipt(self) -> Mapping[str, Any]:
        with self._state_lock:
            calls_this_process = self._provider_calls_this_process
            cache_hits = self._cache_hits
        return {
            "provider_calls": self._durable_intents(),
            "provider_calls_this_process": calls_this_process,
            "cache_hits": cache_hits,
            "model": QWEN_MODEL,
            "endpoint": QWEN_ENDPOINT,
            "max_documents": QWEN_MAX_DOCUMENTS,
            "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
            "approved_provider_attempt_hard_ceiling": self.ceiling,
            "cache_root": str(self.cache_root),
            "fallback": False,
        }

    def validate_manifest(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Strictly replay sealed cache reads for a score-free request manifest."""

        expected_identities = [str(row.get("cache_identity", "")) for row in rows]
        if (
            not expected_identities
            or any(not _is_sha256(identity) for identity in expected_identities)
            or len(set(expected_identities)) != len(expected_identities)
        ):
            raise PredictExecutorError("Qwen request manifest identity drift")
        response_root = self.cache_root / "responses"
        actual_identities = {
            path.stem for path in response_root.rglob("*.json")
        } if response_root.is_dir() else set()
        if actual_identities != set(expected_identities):
            raise PredictExecutorError("Qwen formal response-cache census drift")
        attempts_root = self.cache_root / "attempts"
        actual_attempt_identities = {
            path.name for path in attempts_root.iterdir() if path.is_dir()
        } if attempts_root.is_dir() else set()
        if actual_attempt_identities != set(expected_identities):
            raise PredictExecutorError("Qwen formal attempt-root census drift")
        lock_root = self.cache_root / "locks"
        if (
            (lock_root.is_dir() and any(lock_root.iterdir()))
            or (self.cache_root / "budget.lock").exists()
        ):
            raise PredictExecutorError("Qwen formal cache has unresolved lifecycle lock")
        aligned_by_identity: dict[str, list[float]] = {}
        for row in rows:
            identity = str(row["cache_identity"])
            cache_path = response_root / identity[:2] / f"{identity}.json"
            cache = _read_mapping(cache_path, "formal Qwen cache manifest row")
            ordinal = int(cache.get("successful_attempt_ordinal"))
            intent_path = (
                self.cache_root
                / "attempts"
                / identity
                / f"attempt_{ordinal:03d}_intent.json"
            )
            intent = _read_mapping(intent_path, "formal Qwen manifest intent")
            request = intent.get("request")
            if not isinstance(request, Mapping):
                raise PredictExecutorError("Qwen manifest intent request is missing")
            request_input = request.get("input")
            parameters = request.get("parameters")
            if not isinstance(request_input, Mapping) or not isinstance(parameters, Mapping):
                raise PredictExecutorError("Qwen manifest request shape drift")
            query = str(request_input.get("query", ""))
            documents = request_input.get("documents")
            instruction = str(parameters.get("instruct", ""))
            if not isinstance(documents, list):
                raise PredictExecutorError("Qwen manifest documents drift")
            if (
                set(request) != {"model", "input", "parameters"}
                or request.get("model") != QWEN_MODEL
                or set(request_input) != {"query", "documents"}
                or set(parameters) != {"return_documents", "top_n", "instruct"}
                or parameters.get("return_documents") is not False
                or parameters.get("top_n") != len(documents)
                or _sha256(query) != row.get("question_sha256")
                or _sha256(documents) != row.get("documents_sha256")
                or _sha256(instruction) != row.get("instruction_sha256")
                or intent.get("cache_metadata") != row.get("cache_metadata")
            ):
                raise PredictExecutorError("Qwen manifest immutable request drift")
            result = self.score(
                query,
                [str(item) for item in documents],
                instruction=instruction,
                cache_metadata=intent.get("cache_metadata", {}),
            )
            if (
                result.cache_identity != identity
                or result.request_sha256 != row.get("request_sha256")
                or result.response_sha256 != row.get("response_sha256")
                or len(documents) != row.get("batch_size")
            ):
                raise PredictExecutorError("Qwen request manifest/cache binding drift")
            aligned_by_identity[identity] = list(result.aligned_scores)
        return {
            "rows": len(rows),
            "identities_sha256": _sha256(sorted(expected_identities)),
            "durable_intents": self._durable_intents(),
            "aligned_scores_by_identity": aligned_by_identity,
            "provider_calls": 0,
        }


def run_matrix_phase(request: Mapping[str, Any]) -> dict[str, Any]:
    """Formal main-runner entrypoint; first invocation can only seal budget."""

    if request.get("phase") != "predict":
        raise PredictExecutorError("predict executor accepts only phase=predict")
    protocol = matrix.load_protocol(Path(str(request["protocol_path"])))
    if (
        request.get("protocol_sha256") != _sha256(protocol)
        or request.get("legacy_cache_readthrough") is not False
        or Path(str(request["output_root"])).resolve() != matrix.DEFAULT_ROOT.resolve()
    ):
        raise PredictExecutorError("formal predict request contract drift")
    output_root = Path(str(request["output_root"])).resolve()
    validate_predict_authorities(protocol)
    _load_build(output_root, protocol)
    budget_path = output_root / "predict" / "budget.json"
    if not budget_path.is_file():
        budget = seal_predict_budget(output_root)
        return {
            "status": "waiting_predict_budget_gate",
            "provider_calls": 0,
            "predict_budget_path": str(budget_path.resolve()),
            "predict_budget_sha256": budget["predict_budget_sha256"],
            "provider_attempt_hard_ceiling": budget["provider_attempt_hard_ceiling"],
        }
    budget_gate = validate_predict_budget_gate(output_root=output_root, protocol=protocol)
    # Provider-bearing adapters are imported/constructed only below the
    # independent dynamic budget gate and the main runner's smoke gates.
    from creamem3.official_g25_runtime_adapter_v19 import OfficialGemini25Runtime
    from creamem3.build_executor_v19 import OfficialNative3072EmbeddingBackend

    budget = budget_gate["budget"]
    text = OfficialGemini25Runtime(
        output_root / "provider" / "predict" / "g25_reader",
        allow_model_calls=True,
    )
    embedding = OfficialNative3072EmbeddingBackend(
        output_root / "provider" / "predict" / "query_embedding_native_3072",
        containment_root=output_root,
        approved_provider_attempt_hard_ceiling=budget[
            "stage_provider_attempt_hard_ceilings"
        ]["query_embedding"],
        budget_binding_sha256=str(budget_gate["gate_sha256"]),
    )
    reranker = DashScopeFormalReranker(
        output_root / "provider" / "predict" / "qwen3_rerank",
        allow_model_calls=True,
        approved_provider_attempt_hard_ceiling=budget[
            "stage_provider_attempt_hard_ceilings"
        ]["qwen_rerank"],
    )
    return execute_predict_with_backends(
        output_root=output_root,
        text_backend=text,
        embedding_backend=embedding,
        rerank_backend=reranker,
        budget_gate=budget_gate,
        enforce_full_scope=True,
    )


def blocked_formal_release() -> dict[str, Any]:
    protocol = matrix.load_protocol()
    return {
        "schema_version": matrix.RELEASE_SCHEMA,
        "phase": "predict",
        "status": "blocked_pending_official_smokes_dynamic_budget_and_independent_gate",
        "protocol_sha256": _sha256(protocol),
        "module": "creamem3.predict_executor_v19",
        "module_file_sha256": _file_sha256(Path(__file__).resolve()),
        "entrypoint": "run_matrix_phase",
        "logical_call_hard_ceiling": 0,
        "stage_call_hard_ceilings": {},
        "provider_cache_root": str((matrix.DEFAULT_ROOT / "provider" / "predict").resolve()),
        "legacy_cache_readthrough": False,
    }


__all__ = [
    "DashScopeFormalReranker",
    "PREDICT_BUDGET_GATE_SCHEMA",
    "PREDICT_BUDGET_SCHEMA",
    "PREDICTION_CELL_SCHEMA",
    "PredictExecutorError",
    "RerankReply",
    "TextReply",
    "blocked_formal_release",
    "compute_predict_budget",
    "execute_predict_with_backends",
    "run_matrix_phase",
    "seal_predict_budget",
    "validate_predict_authorities",
    "validate_predict_budget_gate",
]

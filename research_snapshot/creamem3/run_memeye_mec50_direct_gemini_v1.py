"""Isolated no-Qwen Gemini harness for MemEye offline selectors.

The selector is deliberately outside this runner.  It must materialize one
ordered ``selected_fact_ids`` list (at most 50 IDs) and one JSON trace per
question.  This harness then:

* opens the accepted, immutable G3.5-low Online build read-only;
* assembles the selected facts with the unchanged Online context assembler;
* runs the unchanged Gemini-2.5 Reader route/prompts;
* judges the predictions with the unchanged Gemini-3.5 prompt/parser/mapping;
* never constructs, imports, or calls a Qwen reranker.

Prediction and Judge caches live below an explicit cache root, separate from
the result root.  Per-row write-once artifacts plus the durable provider
caches make both phases resumable.  Live phases require the explicit
``--allow-model-calls`` flag.  Preflight and audit are zero-network.

Selection manifest (minimal accepted form)::

    {
      "selector_name": "mec50_v1",
      "rows": [
        {
          "row_id": "Brand_Memory_Test::Q1",
          "selected_fact_ids": ["..."],
          "trace": {"...": "selector-owned evidence"}
        }
      ]
    }

Optional top-level/row SHA fields are checked when present.  The row order in
the selection file is not trusted; canonical query order is always used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2 as graph_memory
from creamem3 import memeye_g35_runtime_v19 as g35_runtime
from creamem3 import predict_executor_v19 as predict
from creamem3 import run_memeye_full_arch_matrix_v19 as matrix
from creamem3 import run_memeye_isolated_full_arch_retry5_v19 as retry5
from creamem3 import run_memeye_isolated_full_arch_thinking512_low_v19 as thinking
from creamem3 import run_memeye_thinking_predict_fast_v19 as fast_build
from creamem3 import score_executor_v19 as score


SCHEMA_VERSION = "creamem3.memeye_mec50_direct_gemini.v1.0"
PREDICTION_ROW_SCHEMA = "creamem3.memeye_mec50_prediction_row.v1.0"
PREDICTION_COMPLETE_SCHEMA = "creamem3.memeye_mec50_prediction_complete.v1.0"
SCORE_ROW_SCHEMA = "creamem3.memeye_mec50_score_row.v1.0"
SCORE_COMPLETE_SCHEMA = "creamem3.memeye_mec50_score_complete.v1.0"

MAX_SELECTED_FACTS = 50
MAX_WORKERS = 32
DEFAULT_WORKERS = 8
DEFAULT_SMOKE_ROWS = 3
MAX_GENERATION_ATTEMPTS = 5
CELL_ID = "g35__online_base"
BACKBONE = "g35"
ARCHITECTURE = "online_base"

CANONICAL_ROOT = thinking.DEFAULT_OUTPUT_ROOT.resolve()
SOURCE_PROJECT_ROOT = thinking.DEFAULT_SOURCE_PROJECT_ROOT.resolve()
SCORE_DATA_ROOT = thinking.DEFAULT_SCORE_DATA_ROOT.resolve()
MEMEYE_CACHE_ROOT = retry5.MEMEYE_CACHE_ROOT.resolve()
DEFAULT_OUTPUT_ROOT = (
    MEMEYE_CACHE_ROOT / "mec50_direct_gemini_results_r1"
).resolve()
DEFAULT_CACHE_ROOT = (
    MEMEYE_CACHE_ROOT / "mec50_direct_gemini_provider_cache_r1"
).resolve()

CANONICAL_BUILD_PATH = CANONICAL_ROOT / "build" / "complete.json"
CANONICAL_QUERY_PATH = CANONICAL_ROOT / "project" / "queries_score_free.json"
CANONICAL_EMBEDDING_PATH = CANONICAL_ROOT / "predict" / "query_embeddings.json"
EXPECTED_QUERY_EMBEDDING_FILE_SHA256 = (
    "ca92abc223651ce861c77c78807e76085b5e884f3604df4a3ec888f2a09bb6cb"
)
EXPECTED_QUERY_EMBEDDING_SHA256 = (
    "e925dbcd543b9763135771c43412ed0c0561f35f46ca878228ee185a611d5afd"
)


class MEC50HarnessError(RuntimeError):
    """A selector, frozen-source, cache, provider, or seal invariant failed."""


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


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise MEC50HarnessError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MEC50HarnessError(f"{label} is not an object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise MEC50HarnessError(f"write-once artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _safe_row_name(row: Mapping[str, Any]) -> str:
    ordinal = int(row["source_ordinal"])
    row_id = str(row["row_id"])
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", row_id)[:80]
    return f"{ordinal:04d}__{readable}__{_sha256(row_id)[:12]}.json"


def _validate_roots(output_root: Path, cache_root: Path) -> tuple[Path, Path]:
    output = Path(output_root).resolve()
    cache = Path(cache_root).resolve()
    forbidden = {
        CANONICAL_ROOT,
        SOURCE_PROJECT_ROOT,
        retry5.DEFAULT_OUTPUT_ROOT.resolve(),
        retry5.DEFAULT_CAPTION_ROOT.resolve(),
        matrix.DEFAULT_ROOT.resolve(),
    }
    if output == cache:
        raise MEC50HarnessError("result and provider-cache roots must be separate")
    for label, root in (("output", output), ("cache", cache)):
        if root in forbidden or root == MEMEYE_CACHE_ROOT or MEMEYE_CACHE_ROOT not in root.parents:
            raise MEC50HarnessError(
                f"{label} root must be a dedicated child of {MEMEYE_CACHE_ROOT}"
            )
    return output, cache


def _adjusted_protocol() -> dict[str, Any]:
    """Bind the accepted isolated graph parser repair without mutating source."""

    protocol = copy.deepcopy(matrix.load_protocol())
    graph_item = protocol["authority_files"]["graph_and_retrieval"]
    graph_path = matrix._resolve_authority_path(str(graph_item["path"]))
    actual = _file_sha256(graph_path)
    if (
        graph_path.resolve() != Path(graph_memory.__file__).resolve()
        or actual != retry5.ISOLATED_GRAPH_RETRIEVAL_SHA256
    ):
        raise MEC50HarnessError("accepted isolated graph/retrieval authority drift")
    graph_item["sha256"] = actual
    return protocol


def _reader_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact v19 Reader literals/renderers under accepted authorities."""

    protocol = _adjusted_protocol()
    old_build_sha = predict.BUILD_EXECUTOR_SHA256
    actual_build_sha = _file_sha256(Path(predict.__file__).with_name("build_executor_v19.py"))
    if actual_build_sha != fast_build.EXPECTED_BUILD_EXECUTOR_SHA256:
        raise MEC50HarnessError("accepted build executor authority drift")
    predict.BUILD_EXECUTOR_SHA256 = actual_build_sha
    try:
        authority = predict._reader_authority(protocol)  # type: ignore[attr-defined]
    finally:
        predict.BUILD_EXECUTOR_SHA256 = old_build_sha
    receipt = {
        "protocol_sha256": _sha256(protocol),
        "focused_prompt_sha256": _sha256(authority["focused"]),
        "open_prompt_sha256": _sha256(authority["open"]),
        "structured_fused_prompt_sha256": _sha256(authority["fused"]),
        "predict_executor_file_sha256": _file_sha256(Path(predict.__file__)),
        "g25_runtime_adapter_file_sha256": _file_sha256(
            Path(predict.__file__).with_name("official_g25_runtime_adapter_v19.py")
        ),
        "g25_model": "gemini-2.5-flash-lite",
        "g25_thinking_config": {"thinkingBudget": 0},
    }
    return authority, receipt


def _canonical_sources() -> dict[str, Any]:
    if _file_sha256(CANONICAL_BUILD_PATH) != fast_build.EXPECTED_BUILD_COMPLETE_FILE_SHA256:
        raise MEC50HarnessError("canonical BUILD_COMPLETE byte drift")
    if _file_sha256(CANONICAL_QUERY_PATH) != fast_build.EXPECTED_QUERY_FILE_SHA256:
        raise MEC50HarnessError("canonical score-free query byte drift")
    if _file_sha256(CANONICAL_EMBEDDING_PATH) != EXPECTED_QUERY_EMBEDDING_FILE_SHA256:
        raise MEC50HarnessError("canonical query-embedding byte drift")

    build = _read_object(CANONICAL_BUILD_PATH, "canonical BUILD_COMPLETE")
    build_contract = dict(build)
    build_sha = build_contract.pop("complete_sha256", None)
    if (
        build.get("complete") is not True
        or build_sha != _sha256(build_contract)
        or build_sha != fast_build.EXPECTED_BUILD_COMPLETE_CONTRACT_SHA256
        or build.get("executor_sha256") != fast_build.EXPECTED_BUILD_EXECUTOR_SHA256
    ):
        raise MEC50HarnessError("canonical BUILD_COMPLETE contract drift")

    queries = _read_object(CANONICAL_QUERY_PATH, "canonical score-free queries")
    query_rows = queries.get("rows")
    if (
        not isinstance(query_rows, list)
        or len(query_rows) != matrix.EXPECTED_QUESTIONS
        or queries.get("rows_sha256") != _sha256(query_rows)
    ):
        raise MEC50HarnessError("canonical query census/digest drift")
    matrix._assert_query_rows_score_free(query_rows)

    embeddings = _read_object(CANONICAL_EMBEDDING_PATH, "canonical query embeddings")
    embedding_contract = dict(embeddings)
    embedding_sha = embedding_contract.pop("query_embedding_sha256", None)
    embedding_rows = embeddings.get("rows")
    if (
        embeddings.get("complete") is not True
        or embedding_sha != _sha256(embedding_contract)
        or embedding_sha != EXPECTED_QUERY_EMBEDDING_SHA256
        or not isinstance(embedding_rows, list)
        or len(embedding_rows) != len(query_rows)
        or embeddings.get("rows_sha256") != _sha256(embedding_rows)
        or embeddings.get("query_rows_sha256") != queries.get("rows_sha256")
    ):
        raise MEC50HarnessError("canonical query-embedding seal drift")
    vector_by_id: dict[str, list[float]] = {}
    for query, embedded in zip(query_rows, embedding_rows, strict=True):
        if (
            not isinstance(embedded, Mapping)
            or embedded.get("row_id") != query.get("row_id")
            or embedded.get("question_sha256") != _sha256(query.get("question", ""))
            or embedded.get("vector_sha256") != _sha256(embedded.get("vector"))
        ):
            raise MEC50HarnessError("canonical query-embedding row binding drift")
        vector = predict._validate_vector(embedded.get("vector"))  # type: ignore[attr-defined]
        vector_by_id[str(query["row_id"])] = vector

    return {
        "build": build,
        "queries": queries,
        "query_rows": [dict(row) for row in query_rows],
        "vector_by_id": vector_by_id,
        "receipt": {
            "canonical_root": str(CANONICAL_ROOT),
            "build_path": str(CANONICAL_BUILD_PATH),
            "build_file_sha256": _file_sha256(CANONICAL_BUILD_PATH),
            "build_contract_sha256": build_sha,
            "query_path": str(CANONICAL_QUERY_PATH),
            "query_file_sha256": _file_sha256(CANONICAL_QUERY_PATH),
            "query_rows_sha256": queries["rows_sha256"],
            "query_embedding_path": str(CANONICAL_EMBEDDING_PATH),
            "query_embedding_file_sha256": _file_sha256(CANONICAL_EMBEDDING_PATH),
            "query_embedding_sha256": embedding_sha,
            "immutable_build_reused_read_only": True,
        },
    }


def _selection_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    value = _read_object(path, "offline selector manifest")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise MEC50HarnessError("selector manifest rows are missing")
    if "rows_sha256" in value and value["rows_sha256"] != _sha256(rows):
        raise MEC50HarnessError("selector manifest rows_sha256 drift")
    output: dict[str, dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise MEC50HarnessError("selector row is not an object")
        row_id = str(raw.get("row_id", ""))
        selected = raw.get("selected_fact_ids")
        trace = raw.get("trace", raw.get("selection_trace"))
        if (
            not row_id
            or row_id in output
            or not isinstance(selected, list)
            or not selected
            or len(selected) > MAX_SELECTED_FACTS
            or len(set(map(str, selected))) != len(selected)
            or not isinstance(trace, Mapping)
        ):
            raise MEC50HarnessError(f"invalid selector row: {row_id or '<missing>'}")
        selected_ids = [str(item) for item in selected]
        normalized = {
            "row_id": row_id,
            "selected_fact_ids": selected_ids,
            "trace": copy.deepcopy(dict(trace)),
        }
        expected_selected_sha = raw.get("selected_fact_ids_sha256")
        if expected_selected_sha is not None and expected_selected_sha != _sha256(selected_ids):
            raise MEC50HarnessError(f"selector selected-ID digest drift: {row_id}")
        expected_trace_sha = raw.get("trace_sha256", raw.get("selection_trace_sha256"))
        if expected_trace_sha is not None and expected_trace_sha != _sha256(normalized["trace"]):
            raise MEC50HarnessError(f"selector trace digest drift: {row_id}")
        normalized["selected_fact_ids_sha256"] = _sha256(selected_ids)
        normalized["trace_sha256"] = _sha256(normalized["trace"])
        normalized["row_sha256"] = _sha256(normalized)
        output[row_id] = normalized
        normalized_rows.append(normalized)
    receipt = {
        "selection_path": str(Path(path).resolve()),
        "selection_file_sha256": _file_sha256(path),
        "selector_name": str(value.get("selector_name", "unspecified")),
        "selector_schema_version": value.get("schema_version"),
        "rows": len(normalized_rows),
        "normalized_rows_sha256": _sha256(normalized_rows),
        "max_selected_facts": max(len(row["selected_fact_ids"]) for row in normalized_rows),
    }
    return output, receipt


def _scope_rows(
    query_rows: Sequence[Mapping[str, Any]],
    selection_by_id: Mapping[str, Mapping[str, Any]],
    *,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if int(offset) < 0:
        raise MEC50HarnessError("offset must be nonnegative")
    selected = [dict(row) for row in query_rows if str(row["row_id"]) in selection_by_id]
    unknown = set(selection_by_id) - {str(row["row_id"]) for row in query_rows}
    if unknown:
        raise MEC50HarnessError(f"selector contains unknown row IDs: {sorted(unknown)[:3]}")
    if not selected:
        raise MEC50HarnessError("selector has no rows in the canonical query bank")
    scoped = selected[int(offset) :]
    if limit is not None:
        if int(limit) <= 0:
            raise MEC50HarnessError("limit must be positive")
        scoped = scoped[: int(limit)]
    if not scoped:
        raise MEC50HarnessError("offset/limit produced an empty experiment scope")
    return scoped


def _top_summaries(
    index: graph_memory.GraphMemoryIndex, vector: Sequence[float]
) -> list[dict[str, Any]]:
    scores = index._scores(index.summary_matrix, vector)  # type: ignore[attr-defined]
    indices = index._rank_indices(scores, predict.SUMMARY_N)  # type: ignore[attr-defined]
    return [
        {
            "summary_id": index.summaries[position]["id"],
            "node_type": index.summaries[position]["node_type"],
            "score": float(scores[position]),
            "rank": rank,
        }
        for rank, position in enumerate(indices, start=1)
    ]


def _prepare_rows(
    canonical: Mapping[str, Any],
    selection_by_id: Mapping[str, Mapping[str, Any]],
    scoped_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    build = canonical["build"]
    vectors = canonical["vector_by_id"]
    graph_by_task: dict[str, dict[str, Any]] = {}
    index_by_task: dict[str, graph_memory.GraphMemoryIndex] = {}
    prepared: list[dict[str, Any]] = []
    for query in scoped_rows:
        row_id = str(query["row_id"])
        task = str(query["task"])
        if task not in graph_by_task:
            artifact_path, _ = predict._artifact_for(  # type: ignore[attr-defined]
                build, BACKBONE, ARCHITECTURE, task
            )
            graph = _read_object(artifact_path, f"sealed {task} Online graph")
            graph_by_task[task] = graph
            index_by_task[task] = graph_memory.GraphMemoryIndex(
                graph,
                summary_top_k=predict.SUMMARY_N,
                dense_fact_top_k=predict.DENSE_N,
                bm25_fact_top_k=predict.BM25_N,
                candidate_cap=max(1, len(graph.get("fact_nodes", []))),
                rerank_top_k=MAX_SELECTED_FACTS,
            )
        index = index_by_task[task]
        selected_row = selection_by_id[row_id]
        selected_ids = list(selected_row["selected_fact_ids"])
        missing = [fact_id for fact_id in selected_ids if fact_id not in index.fact_by_id]
        if missing:
            raise MEC50HarnessError(
                f"selector references facts outside {task} graph: {row_id}:{missing[:3]}"
            )
        selected_items = [
            {"fact": index.fact_by_id[fact_id], "selector_rank": rank}
            for rank, fact_id in enumerate(selected_ids, start=1)
        ]
        summaries = _top_summaries(index, vectors[row_id])
        context, assembly = index._assemble(selected_items, summaries)  # type: ignore[attr-defined]
        if assembly.get("rendered_fact_ids") != selected_ids:
            raise MEC50HarnessError(f"context assembler changed selector order: {row_id}")
        selected_facts = [item["fact"] for item in selected_items]
        fact_lengths = sorted(
            len(str(fact.get("text", ""))) for fact in selected_facts
        )
        fact_chars = sum(fact_lengths)
        session_summary_chars = sum(
            len(str(index.summary_by_id[summary_id].get("text", "")))
            for summary_id in assembly.get("rendered_session_summary_ids", [])
        )
        event_view_chars = sum(
            len(str(index.summary_by_id[summary_id].get("text", "")))
            for summary_id in assembly.get("rendered_event_view_ids", [])
        )
        sessions = sorted({int(fact["session_id"]) for fact in selected_facts})
        source_ids = sorted(
            {
                str(source_id)
                for fact in selected_facts
                for source_id in fact.get("source_dia_ids", [])
            }
        )
        prepared.append(
            {
                "query": dict(query),
                "selection": copy.deepcopy(dict(selected_row)),
                "context": context,
                "context_sha256": _sha256(context),
                "assembly": assembly,
                "assembly_sha256": _sha256(assembly),
                "stats": {
                    "selected_facts": len(selected_ids),
                    "selected_fact_chars": fact_chars,
                    "selected_fact_chars_mean": fact_chars / len(fact_lengths),
                    "selected_fact_chars_p50": fact_lengths[
                        (len(fact_lengths) - 1) // 2
                    ],
                    "selected_fact_chars_p90": fact_lengths[
                        math.ceil(0.9 * len(fact_lengths)) - 1
                    ],
                    "selected_fact_chars_max": fact_lengths[-1],
                    "context_chars": len(context),
                    "assembly_overhead_chars": len(context) - fact_chars,
                    "session_summary_text_chars": session_summary_chars,
                    "event_view_text_chars": event_view_chars,
                    "formatting_and_provenance_chars": (
                        len(context)
                        - fact_chars
                        - session_summary_chars
                        - event_view_chars
                    ),
                    "unique_sessions": len(sessions),
                    "session_ids": sessions,
                    "unique_source_rounds": len(source_ids),
                    "source_dia_ids": source_ids,
                    "rendered_session_summaries": len(
                        assembly.get("rendered_session_summary_ids", [])
                    ),
                    "rendered_event_views": len(
                        assembly.get("rendered_event_view_ids", [])
                    ),
                },
            }
        )
    return prepared


def _contract(
    *,
    output_root: Path,
    cache_root: Path,
    selection_receipt: Mapping[str, Any],
    canonical_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
    workers: int,
) -> dict[str, Any]:
    scope = [
        {
            "row_id": row["query"]["row_id"],
            "question_sha256": _sha256(row["query"]["question"]),
            "selection_row_sha256": row["selection"]["row_sha256"],
            "context_sha256": row["context_sha256"],
        }
        for row in prepared
    ]
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": "offline_selector_direct50_then_unchanged_gemini_reader_judge",
        "output_root": str(Path(output_root).resolve()),
        "provider_cache_root": str(Path(cache_root).resolve()),
        "cell_id": CELL_ID,
        "backbone": BACKBONE,
        "architecture": ARCHITECTURE,
        "selection_cap": MAX_SELECTED_FACTS,
        "selection_receipt": dict(selection_receipt),
        "canonical_receipt": dict(canonical_receipt),
        "reader_authority": dict(authority_receipt),
        "judge_authority": {
            "model": g35_runtime.MODEL_ID,
            "thinking_config": {"thinkingLevel": g35_runtime.THINKING_LEVEL},
            "runtime_file_sha256": _file_sha256(Path(g35_runtime.__file__)),
            "prompt_sha256": _adjusted_protocol()["judge_protocol"]["prompt_sha256"],
            "parser_mapping": "score_executor_v19._judge_one unchanged",
        },
        "scope_rows": scope,
        "scope_rows_sha256": _sha256(scope),
        "rows": len(scope),
        "execution_parallelism": {
            "workers_not_semantic": True,
            "allowed_range": [1, MAX_WORKERS],
        },
        "qwen_rerank": {
            "enabled": False,
            "constructed": False,
            "provider_calls": 0,
        },
        "semantic_planner": {
            "called": False,
            "why": "online_base Reader prompt/route does not consume the planner result",
        },
        "formal_release_claimed": False,
    }
    value["contract_sha256"] = _sha256(value)
    return value


def _write_contracts(output_root: Path, cache_root: Path, contract: Mapping[str, Any]) -> None:
    qwen_paths = _qwen_paths(output_root, cache_root)
    if qwen_paths:
        raise MEC50HarnessError(
            f"dedicated no-Qwen roots already contain a Qwen namespace: {qwen_paths[:3]}"
        )
    _write_once(Path(output_root) / "contract.json", contract)
    cache_binding = {
        "schema_version": SCHEMA_VERSION,
        "cache_kind": "dedicated_mec50_direct_gemini_provider_cache",
        "cache_root": str(Path(cache_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "contract_sha256": contract["contract_sha256"],
        "qwen_namespace_created": False,
    }
    cache_binding["cache_binding_sha256"] = _sha256(cache_binding)
    _write_once(Path(cache_root) / "contract.json", cache_binding)


def _ordered_parallel_map(
    function: Callable[[Mapping[str, Any]], dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        return [function(row) for row in rows]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="memeye-mec50"
    ) as pool:
        return list(pool.map(function, rows))


def _reader_budget(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    route_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        route_counts[str(row["query"]["reader_route"])] += 1
    hard = {
        "semantic_planner": 1,
        "focused_reader": max(1, route_counts["two_stage"] * MAX_GENERATION_ATTEMPTS),
        "answer_reader": max(1, route_counts["two_stage"] * MAX_GENERATION_ATTEMPTS),
        "structured_fused_reader": max(
            1, route_counts["structured_fused"] * MAX_GENERATION_ATTEMPTS
        ),
    }
    return {"stage_provider_attempt_hard_ceilings": hard}


def _prediction_path(output_root: Path, prepared: Mapping[str, Any]) -> Path:
    return Path(output_root) / "predict" / "rows" / _safe_row_name(prepared["query"])


def _load_prediction_row(path: Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_object(path, "MEC50 prediction row")
    contract = dict(value)
    observed = contract.pop("row_sha256", None)
    if (
        value.get("schema_version") != PREDICTION_ROW_SCHEMA
        or observed != _sha256(contract)
        or value.get("row_id") != prepared["query"]["row_id"]
        or value.get("question_sha256") != _sha256(prepared["query"]["question"])
        or value.get("selected_fact_ids_sha256")
        != prepared["selection"]["selected_fact_ids_sha256"]
        or value.get("selection_trace_sha256") != prepared["selection"]["trace_sha256"]
        or value.get("context_sha256") != prepared["context_sha256"]
        or value.get("prediction_sha256") != _sha256(value.get("prediction", ""))
    ):
        raise MEC50HarnessError(f"prediction row binding drift: {path}")
    return value


def run_predict(
    *,
    output_root: Path,
    cache_root: Path,
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
    workers: int,
) -> dict[str, Any]:
    _write_contracts(output_root, cache_root, contract)
    complete_path = Path(output_root) / "predict" / "complete.json"
    if complete_path.is_file():
        complete, _rows = _load_prediction_complete(
            output_root, contract, prepared
        )
        return complete
    budget = _reader_budget(prepared)
    # This is the same concurrency adapter used by the accepted v19 run.  It
    # bypasses only the facade's process-global lease; the underlying durable
    # runtime retains per-identity OS locks and locked provider counters.
    reader = retry5._reader_backend(Path(cache_root), budget)  # type: ignore[attr-defined]

    def predict_one(item: Mapping[str, Any]) -> dict[str, Any]:
        path = _prediction_path(output_root, item)
        if path.is_file():
            return _load_prediction_row(path, item)
        query = item["query"]
        context = str(item["context"])
        route = str(query["reader_route"])
        stages: list[dict[str, Any]] = []
        if route == "two_stage":
            focused_prompt = str(authority["focused"]).replace(
                "{question}", str(query["question"])
            ).replace("{context}", context)
            focused = predict._reply(  # type: ignore[attr-defined]
                reader.generate(
                    "focused",
                    focused_prompt,
                    json_mode=True,
                    max_output_tokens=2600,
                    cache_metadata={
                        "cell_id": CELL_ID,
                        "row_id": query["row_id"],
                        "context_sha256": item["context_sha256"],
                    },
                )
            )
            aggregation = predict._parse_json_object(focused.text)  # type: ignore[attr-defined]
            if not aggregation:
                aggregation = {"_failed": True}
            qa_context = authority["render"](aggregation, context, "candidate_first")
            final_prompt = str(authority["open"]).format(
                context=qa_context, question=str(query["question"])
            )
            answer = predict._reply(  # type: ignore[attr-defined]
                reader.generate(
                    "answer",
                    final_prompt,
                    json_mode=False,
                    max_output_tokens=500,
                    cache_metadata={
                        "cell_id": CELL_ID,
                        "row_id": query["row_id"],
                        "qa_context_sha256": _sha256(qa_context),
                    },
                )
            )
            prediction = answer.text.strip()
            stages = [
                {
                    "stage": "focused",
                    "request_sha256": focused.request_sha256,
                    "response_sha256": focused.response_sha256,
                },
                {
                    "stage": "answer",
                    "request_sha256": answer.request_sha256,
                    "response_sha256": answer.response_sha256,
                },
            ]
        elif route == "structured_fused":
            prompt = str(authority["fused"]).format(
                question=str(query["question"]), context=context
            )
            fused = predict._reply(  # type: ignore[attr-defined]
                reader.generate(
                    "structured_fused",
                    prompt,
                    json_mode=True,
                    max_output_tokens=1200,
                    cache_metadata={
                        "cell_id": CELL_ID,
                        "row_id": query["row_id"],
                        "context_sha256": item["context_sha256"],
                    },
                )
            )
            parsed = predict._parse_json_object(fused.text)  # type: ignore[attr-defined]
            prediction = str(parsed.get("final_answer") or "Not found in memory.").strip()
            stages = [
                {
                    "stage": "structured_fused",
                    "request_sha256": fused.request_sha256,
                    "response_sha256": fused.response_sha256,
                }
            ]
        else:
            raise MEC50HarnessError(f"unknown Reader route: {route}")
        if not prediction:
            raise MEC50HarnessError(f"Reader returned an empty prediction: {query['row_id']}")
        value = {
            "schema_version": PREDICTION_ROW_SCHEMA,
            "row_id": query["row_id"],
            "task": query["task"],
            "question_id": query["question_id"],
            "source_ordinal": query["source_ordinal"],
            "question": query["question"],
            "question_sha256": _sha256(query["question"]),
            "reader_route": route,
            "original_query_top_k": query["top_k"],
            "selection_cap": MAX_SELECTED_FACTS,
            "selected_fact_ids": item["selection"]["selected_fact_ids"],
            "selected_fact_ids_sha256": item["selection"]["selected_fact_ids_sha256"],
            "selection_trace_sha256": item["selection"]["trace_sha256"],
            "context_sha256": item["context_sha256"],
            "assembly_sha256": item["assembly_sha256"],
            "context_stats": item["stats"],
            "prediction": prediction,
            "prediction_sha256": _sha256(prediction),
            "reader_stages": stages,
            "qwen_provider_calls": 0,
        }
        value["row_sha256"] = _sha256(value)
        _write_once(path, value)
        return value

    rows = _ordered_parallel_map(predict_one, prepared, workers=workers)
    if len(rows) != len(prepared):
        raise MEC50HarnessError("prediction row census drift")
    usage = predict._durable_gemini_usage(  # type: ignore[attr-defined]
        reader.usage_receipt(),
        cache_root=Path(cache_root) / "provider" / "predict" / "g25_reader",
    )
    provider_calls = sum(
        int(stage.get("provider_calls", 0)) for stage in usage.get("stages", {}).values()
    )
    receipts = [
        {
            "row_id": row["row_id"],
            "path": str(_prediction_path(output_root, item).resolve()),
            "file_sha256": _file_sha256(_prediction_path(output_root, item)),
            "row_sha256": row["row_sha256"],
        }
        for item, row in zip(prepared, rows, strict=True)
    ]
    complete = {
        "schema_version": PREDICTION_COMPLETE_SCHEMA,
        "complete": True,
        "contract_sha256": contract["contract_sha256"],
        "rows": receipts,
        "rows_sha256": _sha256(receipts),
        "reader_usage": usage,
        "reader_provider_calls": provider_calls,
        "qwen_usage": {
            "enabled": False,
            "constructed": False,
            "provider_calls": 0,
        },
        "formal_release_claimed": False,
    }
    complete["complete_sha256"] = _sha256(complete)
    _write_once(complete_path, complete)
    return complete


def _load_prediction_complete(
    output_root: Path,
    contract: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete = _read_object(Path(output_root) / "predict" / "complete.json", "prediction complete")
    check = dict(complete)
    observed = check.pop("complete_sha256", None)
    if (
        complete.get("schema_version") != PREDICTION_COMPLETE_SCHEMA
        or complete.get("complete") is not True
        or observed != _sha256(check)
        or complete.get("contract_sha256") != contract["contract_sha256"]
        or complete.get("qwen_usage", {}).get("provider_calls") != 0
        or len(complete.get("rows", [])) != len(prepared)
    ):
        raise MEC50HarnessError("prediction completion drift")
    rows: list[dict[str, Any]] = []
    for receipt, item in zip(complete["rows"], prepared, strict=True):
        path = _prediction_path(output_root, item)
        if (
            Path(str(receipt.get("path", ""))).resolve() != path.resolve()
            or receipt.get("file_sha256") != _file_sha256(path)
        ):
            raise MEC50HarnessError("prediction receipt/file drift")
        row = _load_prediction_row(path, item)
        if receipt.get("row_sha256") != row["row_sha256"]:
            raise MEC50HarnessError("prediction receipt/row drift")
        rows.append(row)
    return complete, rows


def _gold_rows() -> dict[str, dict[str, Any]]:
    project = retry5._project(SOURCE_PROJECT_ROOT)  # type: ignore[attr-defined]
    receipts = {str(row["task"]): row for row in project["task_receipts"]}
    output: dict[str, dict[str, Any]] = {}
    for task_file in matrix._task_files(SCORE_DATA_ROOT):
        task = task_file.stem.removesuffix("_Open")
        receipt = receipts.get(task)
        if receipt is None or _file_sha256(task_file) != receipt["source_file_sha256"]:
            raise MEC50HarnessError(f"raw score source hash drift: {task}")
        value = _read_object(task_file, f"raw score source {task}")
        qas = value.get("human-annotated QAs")
        if not isinstance(qas, list):
            raise MEC50HarnessError(f"raw score source lacks QAs: {task}")
        for ordinal, qa in enumerate(qas, start=1):
            if not isinstance(qa, Mapping):
                raise MEC50HarnessError("raw QA row is invalid")
            qid = str(qa.get("question_id", f"Q{ordinal}"))
            row_id = f"{task}::{qid}"
            point = qa.get("point")
            x = None
            y = None
            if isinstance(point, list) and len(point) >= 2:
                if isinstance(point[0], list) and point[0]:
                    x = str(point[0][0])
                if isinstance(point[1], list) and point[1]:
                    y = str(point[1][0])
            output[row_id] = {
                "question": str(qa.get("question", "")).strip(),
                "gold": str(qa.get("answer", "")).strip(),
                "task": task,
                "x": x,
                "y": y,
            }
    if len(output) != matrix.EXPECTED_QUESTIONS:
        raise MEC50HarnessError("raw gold row census drift")
    return output


def _judge_backend(cache_root: Path, row_count: int) -> Any:
    hard = max(1, int(row_count) * MAX_GENERATION_ATTEMPTS)
    limits = {
        "caption": g35_runtime.StageLimits(2400, 0, MAX_GENERATION_ATTEMPTS),
        "writer": g35_runtime.StageLimits(16000, 0, MAX_GENERATION_ATTEMPTS),
        "view": g35_runtime.StageLimits(16384, 0, MAX_GENERATION_ATTEMPTS),
        "judge": g35_runtime.StageLimits(400, hard, MAX_GENERATION_ATTEMPTS),
    }
    primary = g35_runtime.Gemini35Runtime(
        Path(cache_root) / "provider" / "score" / "g35_judge",
        allow_model_calls=True,
        provider_factory=lambda: retry5._ReliableRESTProvider(g35_runtime),  # type: ignore[attr-defined]
        stage_limits=limits,
    )
    retry_limits = dict(limits)
    retry_limits["judge"] = g35_runtime.StageLimits(400, hard, 1)
    retry = g35_runtime.Gemini35Runtime(
        Path(cache_root) / "provider" / "score" / "g35_judge",
        allow_model_calls=True,
        provider_factory=lambda: retry5._ReliableRESTProvider(g35_runtime),  # type: ignore[attr-defined]
        stage_limits=retry_limits,
    )
    return retry5._ParseRetryJudgeBackend(primary, retry, g35_runtime)  # type: ignore[attr-defined]


def _score_path(output_root: Path, prediction: Mapping[str, Any]) -> Path:
    return Path(output_root) / "score" / "rows" / _safe_row_name(prediction)


def _load_score_row(
    path: Path, prediction: Mapping[str, Any], gold: Mapping[str, Any]
) -> dict[str, Any]:
    value = _read_object(path, "MEC50 score row")
    contract = dict(value)
    observed = contract.pop("row_sha256", None)
    if (
        value.get("schema_version") != SCORE_ROW_SCHEMA
        or observed != _sha256(contract)
        or value.get("row_id") != prediction["row_id"]
        or value.get("question_sha256") != _sha256(gold["question"])
        or value.get("gold_sha256") != _sha256(gold["gold"])
        or value.get("prediction_sha256") != prediction["prediction_sha256"]
    ):
        raise MEC50HarnessError(f"score row binding drift: {path}")
    return value


def _group_metrics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        groups[str(value) if value is not None else "unknown"].append(row)
    return {
        name: {
            "rows": len(items),
            "raw_mean": sum(float(item["raw_score"]) for item in items) / len(items),
            "mapped_mean": sum(float(item["mapped_score"]) for item in items) / len(items),
            "mapped_percent": 100.0
            * sum(float(item["mapped_score"]) for item in items)
            / len(items),
        }
        for name, items in sorted(groups.items())
    }


def run_score(
    *,
    output_root: Path,
    cache_root: Path,
    contract: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
    workers: int,
) -> dict[str, Any]:
    _write_contracts(output_root, cache_root, contract)
    _prediction_complete, predictions = _load_prediction_complete(
        output_root, contract, prepared
    )
    gold_by_id = _gold_rows()
    complete_path = Path(output_root) / "score" / "complete.json"
    if complete_path.is_file():
        complete = _read_object(complete_path, "MEC50 score complete")
        check = dict(complete)
        observed = check.pop("complete_sha256", None)
        if (
            complete.get("schema_version") != SCORE_COMPLETE_SCHEMA
            or complete.get("complete") is not True
            or observed != _sha256(check)
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("qwen_provider_calls") != 0
            or len(complete.get("rows", [])) != len(predictions)
        ):
            raise MEC50HarnessError("score completion drift")
        for receipt, prediction in zip(complete["rows"], predictions, strict=True):
            row_id = str(prediction["row_id"])
            path = _score_path(output_root, prediction)
            if (
                receipt.get("row_id") != row_id
                or Path(str(receipt.get("path", ""))).resolve() != path.resolve()
                or receipt.get("file_sha256") != _file_sha256(path)
            ):
                raise MEC50HarnessError("score receipt/file drift")
            row = _load_score_row(path, prediction, gold_by_id[row_id])
            if receipt.get("row_sha256") != row["row_sha256"]:
                raise MEC50HarnessError("score receipt/row drift")
        return complete
    judge = _judge_backend(cache_root, len(predictions))
    protocol = _adjusted_protocol()
    score.validate_score_authorities(protocol)

    original_validator = score._validate_runtime_judge_evidence  # type: ignore[attr-defined]
    original_attempts = score.MAX_PROVIDER_ATTEMPTS

    def validate_evidence(**kwargs: Any) -> dict[str, Any]:
        return judge.validate_evidence(
            result=kwargs["result"],
            triple=kwargs["triple"],
            scope=kwargs["scope"],
            triple_sha256=kwargs["triple_sha256"],
            canonical_validator=original_validator,
        )

    score._validate_runtime_judge_evidence = validate_evidence  # type: ignore[attr-defined]
    score.MAX_PROVIDER_ATTEMPTS = MAX_GENERATION_ATTEMPTS

    def score_one(prediction: Mapping[str, Any]) -> dict[str, Any]:
        row_id = str(prediction["row_id"])
        gold = gold_by_id[row_id]
        if gold["question"] != prediction["question"]:
            raise MEC50HarnessError(f"prediction/gold question drift: {row_id}")
        path = _score_path(output_root, prediction)
        if path.is_file():
            return _load_score_row(path, prediction, gold)
        triple = (gold["question"], gold["gold"], str(prediction["prediction"]))
        triple_sha = _sha256(list(triple))
        judged = score._judge_one(  # type: ignore[attr-defined]
            judge,
            triple,
            scope=score.JUDGE_CACHE_SCOPE,
            triple_sha256=triple_sha,
        )
        value = {
            "schema_version": SCORE_ROW_SCHEMA,
            "row_id": row_id,
            "task": gold["task"],
            "x": gold["x"],
            "y": gold["y"],
            "xy": f"{gold['x']}-{gold['y']}",
            "question_sha256": _sha256(gold["question"]),
            "gold_sha256": _sha256(gold["gold"]),
            "prediction_sha256": prediction["prediction_sha256"],
            "exact_triple_sha256": triple_sha,
            **judged,
        }
        value["row_sha256"] = _sha256(value)
        _write_once(path, value)
        return value

    try:
        rows = _ordered_parallel_map(score_one, predictions, workers=workers)
    finally:
        score.MAX_PROVIDER_ATTEMPTS = original_attempts
        score._validate_runtime_judge_evidence = original_validator  # type: ignore[attr-defined]

    raw_mean = sum(float(row["raw_score"]) for row in rows) / len(rows)
    mapped_mean = sum(float(row["mapped_score"]) for row in rows) / len(rows)
    usage = score._durable_gemini_usage(  # type: ignore[attr-defined]
        judge.usage_receipt(),
        cache_root=Path(cache_root) / "provider" / "score" / "g35_judge",
    )
    provider_calls = int(usage["stages"]["judge"]["provider_calls"])
    receipts = [
        {
            "row_id": row["row_id"],
            "path": str(_score_path(output_root, prediction).resolve()),
            "file_sha256": _file_sha256(_score_path(output_root, prediction)),
            "row_sha256": row["row_sha256"],
        }
        for prediction, row in zip(predictions, rows, strict=True)
    ]
    complete = {
        "schema_version": SCORE_COMPLETE_SCHEMA,
        "complete": True,
        "contract_sha256": contract["contract_sha256"],
        "rows": receipts,
        "rows_sha256": _sha256(receipts),
        "raw_mean": raw_mean,
        "mapped_mean": mapped_mean,
        "mapped_percent": mapped_mean * 100.0,
        "by_task": _group_metrics(rows, "task"),
        "by_x": _group_metrics(rows, "x"),
        "by_y": _group_metrics(rows, "y"),
        "by_xy": _group_metrics(rows, "xy"),
        "judge_usage": usage,
        "judge_provider_calls": provider_calls,
        "qwen_provider_calls": 0,
        "calibration_skipped": True,
        "formal_release_claimed": False,
    }
    complete["complete_sha256"] = _sha256(complete)
    _write_once(complete_path, complete)
    return complete


def _length_summary(prepared: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("selected_fact_chars", "context_chars", "assembly_overhead_chars")
    result: dict[str, Any] = {"rows": len(prepared)}
    for field in fields:
        values = sorted(int(row["stats"][field]) for row in prepared)
        result[field] = {
            "mean": sum(values) / len(values),
            "p50": values[(len(values) - 1) // 2],
            "p90": values[math.ceil(0.9 * len(values)) - 1],
            "max": values[-1],
        }
    result["selected_facts_mean"] = sum(
        int(row["stats"]["selected_facts"]) for row in prepared
    ) / len(prepared)
    result["rendered_session_summaries_mean"] = sum(
        int(row["stats"]["rendered_session_summaries"]) for row in prepared
    ) / len(prepared)
    return result


def _qwen_paths(output_root: Path, cache_root: Path) -> list[str]:
    paths: list[str] = []
    for root in (Path(output_root), Path(cache_root)):
        if not root.is_dir():
            continue
        paths.extend(
            str(path.resolve())
            for path in root.rglob("*")
            if "qwen" in path.name.lower()
        )
    return sorted(paths)


def preflight(
    *,
    selection_path: Path,
    output_root: Path,
    cache_root: Path,
    offset: int = 0,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    output, cache = _validate_roots(output_root, cache_root)
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise MEC50HarnessError(f"workers must be in [1,{MAX_WORKERS}]")
    canonical = _canonical_sources()
    authority, authority_receipt = _reader_authority()
    selection, selection_receipt = _selection_rows(selection_path)
    scoped = _scope_rows(
        canonical["query_rows"], selection, offset=int(offset), limit=limit
    )
    prepared = _prepare_rows(canonical, selection, scoped)
    contract = _contract(
        output_root=output,
        cache_root=cache,
        selection_receipt=selection_receipt,
        canonical_receipt=canonical["receipt"],
        authority_receipt=authority_receipt,
        prepared=prepared,
        workers=int(workers),
    )
    first = prepared[0]
    query = first["query"]
    # Provider-free prompt construction catches renderer/placeholder drift.
    if query["reader_route"] == "two_stage":
        sample_prompt = str(authority["focused"]).replace(
            "{question}", str(query["question"])
        ).replace("{context}", str(first["context"]))
    else:
        sample_prompt = str(authority["fused"]).format(
            question=str(query["question"]), context=str(first["context"])
        )
    qwen_paths = _qwen_paths(output, cache)
    if qwen_paths:
        raise MEC50HarnessError(
            f"dedicated no-Qwen roots contain a Qwen namespace: {qwen_paths[:3]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "zero_network_preflight_passed",
        "decision": "approve",
        "output_root": str(output),
        "cache_root": str(cache),
        "contract": contract,
        "prepared": prepared,
        "authority": authority,
        "length_summary": _length_summary(prepared),
        "sample_reader_route": query["reader_route"],
        "sample_prompt_chars": len(sample_prompt),
        "qwen_paths": qwen_paths,
        "provider_calls": 0,
        "network_calls": 0,
        "files_written": 0,
    }


def audit(
    *,
    selection_path: Path,
    output_root: Path,
    cache_root: Path,
    offset: int = 0,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    gate = preflight(
        selection_path=selection_path,
        output_root=output_root,
        cache_root=cache_root,
        offset=offset,
        limit=limit,
        workers=workers,
    )
    output = Path(gate["output_root"])
    qwen_paths = _qwen_paths(output, Path(gate["cache_root"]))
    if qwen_paths:
        raise MEC50HarnessError(f"Qwen namespace unexpectedly exists: {qwen_paths[:3]}")
    prediction = None
    score_value = None
    prediction_path = output / "predict" / "complete.json"
    if prediction_path.is_file():
        prediction, _ = _load_prediction_complete(
            output, gate["contract"], gate["prepared"]
        )
    score_path = output / "score" / "complete.json"
    if score_path.is_file():
        score_value = _read_object(score_path, "MEC50 score complete")
        check = dict(score_value)
        observed = check.pop("complete_sha256", None)
        if (
            observed != _sha256(check)
            or score_value.get("contract_sha256") != gate["contract"]["contract_sha256"]
            or score_value.get("qwen_provider_calls") != 0
        ):
            raise MEC50HarnessError("score completion audit drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "zero_network_audit_passed",
        "rows": len(gate["prepared"]),
        "prediction_complete": prediction is not None,
        "score_complete": score_value is not None,
        "mapped_percent": score_value.get("mapped_percent") if score_value else None,
        "qwen_provider_calls": 0,
        "qwen_paths": [],
        "provider_calls": 0,
        "network_calls": 0,
    }


def _public_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove in-memory callables, vectors, contexts, and traces before print."""

    return {
        key: item
        for key, item in value.items()
        if key not in {"prepared", "authority"}
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "predict", "score", "all", "audit"), default="preflight"
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"run the first {DEFAULT_SMOKE_ROWS} selected rows (conflicts with --limit/--offset)",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--allow-model-calls", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    live = args.phase in {"predict", "score", "all"}
    if live != bool(args.allow_model_calls):
        raise SystemExit("live phases require --allow-model-calls; zero-call phases reject it")
    if args.smoke and (args.limit is not None or args.offset != 0):
        raise SystemExit("--smoke conflicts with --limit and --offset")
    limit = DEFAULT_SMOKE_ROWS if args.smoke else args.limit
    output_root = args.output_root
    cache_root = args.cache_root
    # Keep an accidental default smoke from sealing the later full-run roots.
    if args.smoke and Path(output_root).resolve() == DEFAULT_OUTPUT_ROOT:
        output_root = DEFAULT_OUTPUT_ROOT.with_name(
            DEFAULT_OUTPUT_ROOT.name + f"_smoke{DEFAULT_SMOKE_ROWS}"
        )
    if args.smoke and Path(cache_root).resolve() == DEFAULT_CACHE_ROOT:
        cache_root = DEFAULT_CACHE_ROOT.with_name(
            DEFAULT_CACHE_ROOT.name + f"_smoke{DEFAULT_SMOKE_ROWS}"
        )
    gate = preflight(
        selection_path=args.selection_manifest,
        output_root=output_root,
        cache_root=cache_root,
        offset=args.offset,
        limit=limit,
        workers=args.workers,
    )
    if args.phase == "preflight":
        result = _public_gate(gate)
    elif args.phase == "audit":
        result = audit(
            selection_path=args.selection_manifest,
            output_root=output_root,
            cache_root=cache_root,
            offset=args.offset,
            limit=limit,
            workers=args.workers,
        )
    elif args.phase == "predict":
        result = run_predict(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            authority=gate["authority"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
    elif args.phase == "score":
        result = run_score(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
    else:
        predicted = run_predict(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            authority=gate["authority"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
        scored = run_score(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
        result = {"status": "all_complete", "predict": predicted, "score": scored}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

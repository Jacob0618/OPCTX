"""Run the frozen 100-question LongMemEval Online/OpCtx v3 adaptation.

Default execution is an offline preflight.  Network access is impossible
unless ``--allow-live`` is explicit.  A smoke run and the subsequent 100-item
run can share one root: run ``--limit 1`` first, then repeat with ``--resume``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import math
import os
import pickle
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import online_memory_v3
from creamem3 import semantic_query_planner_v6 as planner
from creamem3 import strict_semantic_planner_v8 as strict_planner
from creamem3.audit_lme_opctx_materializer_v1 import audit as audit_materializer
from creamem3.lme_opctx_v1 import (
    LME_DATASET,
    MGTM_CLIENT_SOURCE,
    MGTM_MEMORY_SOURCE,
    MGTM_PROMPT_SOURCE,
    MGTM_ROOT,
    NonRetryableProviderError,
    PromptContract,
    QwenGraphMemoryIndex,
    assert_live_prompt_contract,
    assert_prompt_contract,
    build_session_graph_from_packets,
    materialize_operation_context,
    render_lme_session,
    render_write_prompt,
    sha256_text,
    write_one_session,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "creamem3" / "audit" / "lme100_stratified_20260826" / "lme100_manifest.json"
MANIFEST_SHA256 = "d4c446b1eb3bfb58e0cd54910b5453e0e352ca325199dae1fb3e26e1342e593c"
DEFAULT_OUTPUT = ROOT / "creamem3" / "results_lme_opctx100_v1_20260826"
DEFAULT_CACHE = Path(r"D:\creamem3_cache\lme_opctx100_v1_20260826")
DEFAULT_PREFLIGHT = ROOT / "creamem3" / "audit" / "lme_opctx100_v1" / "preflight.json"
WORKSPACE_ENV = ROOT / ".env"
RUN_SCHEMA = "creamem3.lme_opctx100.runner.v1"
MAX_GENERATION_ATTEMPTS = 5
FIXED_BACKOFF_S = 2.0
GENERATION_TRANSPORT_RETRIES = 1
RETRIEVAL_TRANSPORT_RETRIES = 8
GENERATION_SEMAPHORE = threading.BoundedSemaphore(8)
QWEN_SEMAPHORE = threading.BoundedSemaphore(4)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return gm.canonical_sha256(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sealed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["checkpoint_payload_sha256"] = _canonical(payload)
    return payload


def _verify_sealed_payload(value: Any, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint must be an object: {path}")
    payload = dict(value)
    recorded = payload.pop("checkpoint_payload_sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical(payload):
        raise RuntimeError(f"checkpoint payload seal mismatch: {path}")
    return payload


def _validate_physical_attempts(
    attempts: Any,
    *,
    same_canonical_body: bool,
    require_nonempty: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(attempts, list) or (require_nonempty and not attempts) or len(attempts) > 5:
        raise RuntimeError("physical attempt evidence must contain 1..5 rows")
    normalized: list[dict[str, Any]] = []
    for ordinal, row in enumerate(attempts, 1):
        if not isinstance(row, dict) or int(row.get("physical_ordinal", -1)) != ordinal:
            raise RuntimeError("physical attempt ordinals are not exact")
        if not isinstance(row.get("canonical_request_sha256"), str):
            raise RuntimeError("physical attempt lacks canonical request hash")
        if not isinstance(row.get("actual_http_body_sha256"), str):
            raise RuntimeError("physical attempt lacks actual HTTP body hash")
        normalized.append(row)
    if same_canonical_body and len({row["canonical_request_sha256"] for row in normalized}) > 1:
        raise RuntimeError("transient retry changed canonical request")
    if same_canonical_body and len({row["actual_http_body_sha256"] for row in normalized}) > 1:
        raise RuntimeError("transient retry changed actual HTTP body")
    return normalized


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(contract: PromptContract) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if _sha_file(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("frozen LME100 manifest SHA mismatch")
    data = _load_json(LME_DATASET)
    manifest = _load_json(MANIFEST)
    if not isinstance(data, list) or len(data) != 500:
        raise RuntimeError("LongMemEval source must contain 500 records")
    if manifest.get("count") != 100 or len(manifest.get("ids", [])) != 100:
        raise RuntimeError("manifest must bind exactly 100 questions")
    selected: list[dict[str, Any]] = []
    for position, binding in enumerate(manifest["items"]):
        source_index = int(binding["source_index"])
        item = data[source_index]
        if str(item["question_id"]) != str(binding["question_id"]):
            raise RuntimeError(f"manifest/source id mismatch at {position}")
        if _canonical(item) != str(binding["record_sha256"]):
            raise RuntimeError(f"source record hash mismatch for {item['question_id']}")
        if len(item["haystack_sessions"]) != int(binding["session_count"]):
            raise RuntimeError(f"session count mismatch for {item['question_id']}")
        if not (
            len(item["haystack_sessions"])
            == len(item["haystack_dates"])
            == len(item["haystack_session_ids"])
        ):
            raise RuntimeError(f"session arrays misaligned for {item['question_id']}")
        selected.append(item)
    if [str(row["question_id"]) for row in selected] != list(manifest["ids"]):
        raise RuntimeError("selected source order differs from frozen manifest")
    if contract.file_sha256["lme_dataset"] != _sha_file(LME_DATASET):
        raise RuntimeError("dataset changed after prompt preflight")
    return data, manifest, selected


def _load_live_credentials() -> None:
    """Load only task-required keys without logging values or overriding exports."""

    allowed = {"GEMINI_API_KEY", "DASHSCOPE_API_KEY"}
    if WORKSPACE_ENV.is_file():
        for raw_line in WORKSPACE_ENV.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            key, value = line.split("=", 1)
            key = key.strip()
            if key in allowed:
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))
    missing = sorted(key for key in allowed if not os.environ.get(key))
    if missing:
        raise RuntimeError("missing required live credential names: " + ", ".join(missing))


_STAMP_RE = re.compile(r"^\s*(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})\s*$")


def _timestamp_key(value: str, source_position: int) -> tuple[int, int, int, int, int, int]:
    match = _STAMP_RE.match(str(value))
    if not match:
        raise ValueError(f"unsupported LongMemEval timestamp: {value!r}")
    return (*map(int, match.groups()), int(source_position))


def _ordered_session_specs(item: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = [
        {
            "source_position": position,
            "source_session_id": str(session_id),
            "date": str(date),
            "turns": list(turns),
        }
        for position, (session_id, date, turns) in enumerate(
            zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])
        )
    ]
    original_positions = [int(row["source_position"]) for row in specs]
    ordered = sorted(specs, key=lambda row: _timestamp_key(str(row["date"]), int(row["source_position"])))
    ordered_positions = [int(row["source_position"]) for row in ordered]
    adjacent_inversions = sum(
        _timestamp_key(str(specs[index]["date"]), index)
        > _timestamp_key(str(specs[index + 1]["date"]), index + 1)
        for index in range(len(specs) - 1)
    )
    return ordered, {
        "policy": "stable_timestamp_then_source_position",
        "source_positions_before": original_positions,
        "source_positions_after": ordered_positions,
        "reordered": original_positions != ordered_positions,
        "adjacent_inversions_repaired": adjacent_inversions,
        "duplicate_source_session_ids": len(specs) - len({row["source_session_id"] for row in specs}),
    }


def _call_estimate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = sum(len(item["haystack_sessions"]) for item in selected)
    questions = len(selected)
    online = sessions - questions
    logical = sessions + online + questions * 3  # planner + QA + judge
    return {
        "questions": questions,
        "physical_sessions": sessions,
        "writer_logical_calls": sessions,
        "online_fusion_logical_calls": online,
        "planner_logical_calls": questions,
        "qa_logical_calls": questions,
        "judge_logical_calls": questions,
        "generative_logical_success_calls": logical,
        "generative_physical_attempt_upper_bound": logical * MAX_GENERATION_ATTEMPTS,
        "qwen_rerank_logical_calls": questions,
        "qwen_physical_attempt_upper_bound": questions * RETRIEVAL_TRANSPORT_RETRIES,
        "embedding_http_calls": "data/cache/view-dependent; each release batch contains <=100 texts",
    }


def _run_config(contract: PromptContract) -> dict[str, Any]:
    from creamem3 import online_operation_context_v8
    from creamem3 import prompts_v2

    source_paths = {
        "runner": Path(__file__).resolve(),
        "adapter": ROOT / "creamem3" / "lme_opctx_v1.py",
        "graph_memory": Path(gm.__file__).resolve(),
        "online_memory": Path(online_memory_v3.__file__).resolve(),
        "semantic_planner": Path(planner.__file__).resolve(),
        "strict_semantic_planner": Path(strict_planner.__file__).resolve(),
        "operation_context": Path(online_operation_context_v8.__file__).resolve(),
        "online_prompts": Path(prompts_v2.__file__).resolve(),
    }
    return {
        "schema_version": RUN_SCHEMA,
        "manifest_sha256": MANIFEST_SHA256,
        "dataset_sha256": contract.file_sha256["lme_dataset"],
        "prompt_contract": contract.receipt(),
        "models": {
            "writer": "gemini-2.5-flash",
            "answerer": "gemini-2.5-flash",
            "judge": "gemini-3-flash-preview",
            "embedding": "gemini-embedding-001",
            "reranker": "qwen3-rerank",
        },
        "architecture": {
            "write_frontier": "one_physical_session",
            "causal_order_policy": "stable_timestamp_then_source_position",
            "online": "creamem3.online_memory_v3",
            "reader": "online_semantic_operation_context_v3",
            "lme_final_k": 30,
            "online_update_prompt_sha256": prompts_v2.PROMPT_SHA256["online_event_update"],
            "planner_system_prompt_sha256": sha256_text(planner.SEMANTIC_QUERY_PLANNER_SYSTEM_PROMPT),
            "planner_schema_version": planner.PLANNER_SCHEMA_VERSION,
            "strict_planner_schema_version": strict_planner.STRICT_SCHEMA_VERSION,
            "operation_mapping_sha256": _canonical(online_operation_context_v8.OPERATION_TO_CONTEXT_POLICY),
        },
        "implementation_source_sha256": {
            name: _sha_file(path) for name, path in source_paths.items()
        },
        "generation_retry": {
            "physical_attempts_max": 5,
            "transient_backoff_seconds": 2.0,
            "provider_internal_retries": 1,
            "non_retryable_http": "all 4xx except 429 stop on first occurrence",
            "retryable_failures": "HTTP 429, HTTP 5xx, and transport failures",
            "same_canonical_body": "writer/QA/fusion transient retries",
            "judge_application_schedule": {
                "max_tokens": [200, 400, 800, 1600, 3200],
                "filled_judge_prompt_fixed": True,
                "one_physical_post_per_application_ordinal": True,
            },
        },
        "retrieval_retry": {
            "provider_internal_retries": 8,
            "policy": "release client exponential+jitter; reported separately from generation",
        },
        "credential_loading": {
            "policy": "exported environment first, then workspace .env setdefault for required key names only",
            "required_key_names": ["DASHSCOPE_API_KEY", "GEMINI_API_KEY"],
            "values_logged": False,
        },
    }


def run_preflight(output: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    contract = assert_prompt_contract()
    _, manifest, selected = _load_inputs(contract)
    parity = audit_materializer()
    if not parity["passed"]:
        raise RuntimeError("LoCoMo 1540-row materializer parity failed")
    order_audit = [_ordered_session_specs(item)[1] for item in selected]
    receipt: dict[str, Any] = {
        "schema_version": "creamem3.lme_opctx100.preflight.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "network_calls": 0,
        "model_calls": 0,
        "manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA256, "count": manifest["count"]},
        "dataset": {"path": str(LME_DATASET), "sha256": contract.file_sha256["lme_dataset"]},
        "prompt_contract": contract.receipt(),
        "materializer_parity": {
            "rows_checked": parity["rows_checked"],
            "mismatch_count": parity["mismatch_count"],
            "receipt_sha256": parity["receipt_sha256"],
        },
        "causal_order": {
            "policy": "stable_timestamp_then_source_position",
            "questions_reordered": sum(bool(row["reordered"]) for row in order_audit),
            "adjacent_inversions_repaired": sum(int(row["adjacent_inversions_repaired"]) for row in order_audit),
            "duplicate_source_session_ids": sum(int(row["duplicate_source_session_ids"]) for row in order_audit),
        },
        "call_estimate": _call_estimate(selected),
        "run_config": _run_config(contract),
        "files": {
            "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha_file(Path(__file__).resolve())},
            "adapter": {"path": str(ROOT / "creamem3" / "lme_opctx_v1.py"), "sha256": _sha_file(ROOT / "creamem3" / "lme_opctx_v1.py")},
        },
        "not_used": {
            "historical_3_session_checkpoint_bank": "main arm requires one physical session per Online causal frontier",
        },
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    _atomic_json(output, receipt)
    return receipt


class _NoGlobalsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"global class forbidden: {module}.{name}")


def _save_graph(
    path: Path,
    graph: Mapping[str, Any],
    config_sha256: str,
    question_binding_sha256: str,
) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".seal.json").exists():
        raise RuntimeError(f"refusing to overwrite sealed graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": RUN_SCHEMA,
        "config_sha256": config_sha256,
        "question_binding_sha256": question_binding_sha256,
        "graph": dict(graph),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(envelope, handle, protocol=5)
    temporary.replace(path)
    _atomic_json(path.with_suffix(path.suffix + ".seal.json"), {
        "file_sha256": _sha_file(path),
        "config_sha256": config_sha256,
        "question_binding_sha256": question_binding_sha256,
        "content_fingerprint": graph.get("content_fingerprint"),
    })


def _load_graph(
    path: Path,
    config_sha256: str,
    question_binding_sha256: str,
) -> dict[str, Any] | None:
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if not path.exists() and not seal_path.exists():
        return None
    if not path.is_file() or not seal_path.is_file():
        raise RuntimeError(f"partial graph checkpoint: {path}")
    seal = _load_json(seal_path)
    if (
        seal.get("config_sha256") != config_sha256
        or seal.get("question_binding_sha256") != question_binding_sha256
        or seal.get("file_sha256") != _sha_file(path)
    ):
        raise RuntimeError(f"graph checkpoint seal mismatch: {path}")
    with path.open("rb") as handle:
        envelope = _NoGlobalsUnpickler(handle).load()
    if (
        envelope.get("config_sha256") != config_sha256
        or envelope.get("question_binding_sha256") != question_binding_sha256
        or not isinstance(envelope.get("graph"), dict)
    ):
        raise RuntimeError(f"graph checkpoint envelope mismatch: {path}")
    graph = envelope["graph"]
    if graph.get("content_fingerprint") != seal.get("content_fingerprint"):
        raise RuntimeError(f"graph fingerprint mismatch: {path}")
    return graph


def _save_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        previous = _verify_sealed_payload(_load_json(path), path)
        if previous.get("status") == "complete":
            raise RuntimeError(f"refusing to overwrite complete result: {path}")
        history = path.with_name("result.error.previous.json")
        if not history.exists():
            path.replace(history)
            old_seal = path.with_suffix(path.suffix + ".seal.json")
            if old_seal.exists():
                old_seal.replace(history.with_suffix(history.suffix + ".seal.json"))
    sealed = _sealed_payload(result)
    _atomic_json(path, sealed)
    _atomic_json(path.with_suffix(path.suffix + ".seal.json"), {
        "file_sha256": _sha_file(path),
        "checkpoint_payload_sha256": sealed["checkpoint_payload_sha256"],
        "binding_sha256": _canonical(result.get("binding")),
    })


def _load_result(path: Path) -> dict[str, Any]:
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if not seal_path.is_file():
        raise RuntimeError(f"result sidecar seal missing: {path}")
    raw = _load_json(path)
    result = _verify_sealed_payload(raw, path)
    seal = _load_json(seal_path)
    if (
        seal.get("file_sha256") != _sha_file(path)
        or seal.get("checkpoint_payload_sha256") != raw.get("checkpoint_payload_sha256")
        or seal.get("binding_sha256") != _canonical(result.get("binding"))
    ):
        raise RuntimeError(f"result sidecar seal mismatch: {path}")
    return result


class _LockedUsage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}

    def add(self, tag: str, *, prompt_tokens: int = 0, output_tokens: int = 0,
            thought_tokens: int = 0, model: str = "") -> None:
        with self._lock:
            row = self._data.setdefault(tag, {
                "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
                "calls": 0, "models": {},
            })
            row["prompt_tokens"] += int(prompt_tokens or 0)
            row["output_tokens"] += int(output_tokens or 0)
            row["thought_tokens"] += int(thought_tokens or 0)
            row["calls"] += 1
            if model:
                row["models"][model] = int(row["models"].get(model, 0)) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data, sort_keys=True))


def _load_release_runtime(contract: PromptContract) -> tuple[ModuleType, ModuleType]:
    # The full file/prompt gate must happen before provider code is imported.
    current = assert_prompt_contract()
    if current.contract_sha256 != contract.contract_sha256:
        raise RuntimeError("release authority changed between preflight and provider import")
    release_root = MGTM_ROOT.parents[2]
    if str(release_root) not in sys.path:
        sys.path.insert(0, str(release_root))
    clients = importlib.import_module("CreaMem2.benchmarks.longmemeval.clients")
    memory = importlib.import_module("CreaMem2.benchmarks.longmemeval.memory")
    if Path(clients.__file__).resolve() != MGTM_CLIENT_SOURCE.resolve():
        raise RuntimeError("release clients imported from an unexpected path")
    if Path(memory.__file__).resolve() != MGTM_MEMORY_SOURCE.resolve():
        raise RuntimeError("release memory imported from an unexpected path")
    return clients, memory


def _client_types(release_clients: ModuleType) -> tuple[type[Any], type[Any]]:
    base = release_clients.APIClients

    class AuditedGenerationClient(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["max_retries"] = GENERATION_TRANSPORT_RETRIES
            super().__init__(*args, **kwargs)
            self.usage = _LockedUsage()
            self._http_local = threading.local()

        def _post(self, url: str, payload: dict[str, Any], *, bearer: str | None = None) -> dict[str, Any]:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._http_local.body_sha256 = hashlib.sha256(body).hexdigest()
            self._http_local.url_class = "dashscope" if bearer else "gemini"
            try:
                return super()._post(url, payload, bearer=bearer)
            except RuntimeError as exc:
                match = re.search(r"\bHTTP\s+(4\d\d)\b", str(exc))
                if match and int(match.group(1)) != 429:
                    raise NonRetryableProviderError(str(exc)) from exc
                raise

        def last_http_evidence(self) -> dict[str, Any]:
            return {
                "actual_http_body_sha256": getattr(self._http_local, "body_sha256", None),
                "endpoint_class": getattr(self._http_local, "url_class", None),
            }

    class StrictRetrievalClient(AuditedGenerationClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.max_retries = RETRIEVAL_TRANSPORT_RETRIES
            self.last_rerank_audit: dict[str, Any] = {}

        def rerank(self, query: str, documents: Sequence[str], instruction: str) -> list[float]:
            if not documents:
                self.last_rerank_audit = {"documents": 0, "cache_hit": False, "strict_indices": True}
                return []
            if len(documents) > 500:
                raise ValueError("Qwen3 reranking accepts at most 500 documents")
            key = self._cache_key(["rerank.v1", self.stack.reranker, instruction, query, list(documents)])
            path = self.rerank_cache / f"{key}.json"
            strict_path = self.rerank_cache / f"{key}.strict.json"
            strict_binding = {
                "model": self.stack.reranker,
                "query_sha256": sha256_text(query),
                "documents_sha256": _canonical(list(documents)),
                "instruction_sha256": sha256_text(instruction),
                "documents": len(documents),
            }
            if path.exists():
                if not strict_path.is_file():
                    raise RuntimeError("strict Qwen cache sidecar missing")
                scores = [float(value) for value in json.loads(path.read_text(encoding="utf-8"))]
                if len(scores) != len(documents) or not all(math.isfinite(value) for value in scores):
                    raise RuntimeError("invalid strict Qwen cache")
                strict = _verify_sealed_payload(_load_json(strict_path), strict_path)
                if strict.get("binding") != strict_binding or strict.get("scores_sha256") != _canonical(scores):
                    raise RuntimeError("strict Qwen cache binding mismatch")
                self.last_rerank_audit = {
                    "documents": len(documents), "cache_hit": True,
                    "strict_indices": True, "cache_sha256": _sha_file(path),
                }
                return scores
            payload = {
                "model": self.stack.reranker,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {
                    "return_documents": False,
                    "top_n": len(documents),
                    "instruct": instruction,
                },
            }
            with QWEN_SEMAPHORE:
                response = self._post(release_clients.DASHSCOPE_RERANK_URL, payload, bearer=self.dashscope_api_key)
            results = response.get("output", {}).get("results")
            if not isinstance(results, list) or len(results) != len(documents):
                raise RuntimeError("Qwen results count must equal document count")
            parsed: dict[int, float] = {}
            for row in results:
                if not isinstance(row, Mapping):
                    raise RuntimeError("Qwen result row must be an object")
                index = row.get("index")
                score = row.get("relevance_score")
                if isinstance(index, bool) or not isinstance(index, int) or index in parsed:
                    raise RuntimeError("Qwen result indices must be unique integers")
                value = float(score)
                if not math.isfinite(value):
                    raise RuntimeError("Qwen score must be finite")
                parsed[index] = value
            expected = set(range(len(documents)))
            if set(parsed) != expected:
                raise RuntimeError("Qwen result indices must be exactly 0..N-1")
            scores = [parsed[index] for index in range(len(documents))]
            self._atomic_json(path, scores)
            _atomic_json(strict_path, _sealed_payload({
                "schema_version": RUN_SCHEMA,
                "binding": strict_binding,
                "scores_sha256": _canonical(scores),
                "response_indices": sorted(parsed),
                "actual_http_body_sha256": self.last_http_evidence()["actual_http_body_sha256"],
            }))
            self.last_rerank_audit = {
                "documents": len(documents), "cache_hit": False,
                "strict_indices": True, "response_indices_sha256": _canonical(sorted(parsed)),
                "actual_http_body_sha256": self.last_http_evidence()["actual_http_body_sha256"],
            }
            return scores

    return AuditedGenerationClient, StrictRetrievalClient


class PhysicalChat:
    """Strict compatibility shim: one invocation equals one physical POST."""

    _ALLOWED = frozenset({
        "model", "temperature", "max_tokens", "thinking_budget", "tag",
        "use_cache", "json_mode", "system",
    })

    def __init__(self, client: Any, contract: PromptContract, *, default_model: str, stage: str) -> None:
        self.client = client
        self.contract = contract
        self.default_model = default_model
        self.stage = stage
        self.attempts: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        unexpected = set(kwargs).difference(self._ALLOWED)
        if unexpected:
            raise TypeError(f"unsupported chat compatibility kwargs: {sorted(unexpected)}")
        if len(self.attempts) >= MAX_GENERATION_ATTEMPTS:
            raise RuntimeError("physical attempt cap already exhausted")
        temperature = float(kwargs.get("temperature", 0.0))
        thinking_budget = int(kwargs.get("thinking_budget", 0))
        if temperature != 0.0 or thinking_budget != 0:
            raise ValueError("frozen generation requires temperature=0 and thinking_budget=0")
        model = str(kwargs.get("model") or self.default_model)
        tag = str(kwargs.get("tag") or self.stage)
        max_tokens = int(kwargs.get("max_tokens", 8192))
        json_mode = bool(kwargs.get("json_mode", False))
        system = kwargs.get("system")
        if system is not None and not isinstance(system, str):
            raise TypeError("system must be a string or None")
        request_contract = {
            "prompt": str(prompt), "system": system, "model": model, "tag": tag,
            "temperature": 0.0, "thinking": (
                {"thinkingBudget": 0} if model.startswith("gemini-2.5")
                else ({"thinkingLevel": "minimal"} if model.startswith("gemini-3") else {"type": "disabled"})
            ),
            "max_tokens": max_tokens, "json_mode": json_mode,
            "generative_cache": False,
            "caller_requested_use_cache": bool(kwargs.get("use_cache", False)),
        }
        canonical_sha = _canonical(request_contract)
        assert_live_prompt_contract(self.contract)
        evidence: dict[str, Any] = {
            "physical_ordinal": len(self.attempts) + 1,
            "canonical_request_sha256": canonical_sha,
            "filled_prompt_sha256": sha256_text(str(prompt)),
            "system_sha256": sha256_text(system) if isinstance(system, str) else None,
            "model": model, "tag": tag, "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        try:
            with GENERATION_SEMAPHORE:
                response = self.client.generate(
                    str(prompt), model=model, tag=tag, system=system,
                    max_tokens=max_tokens, json_mode=json_mode,
                )
        except Exception as exc:
            evidence.update(self.client.last_http_evidence())
            evidence.update({
                "status": (
                    "non_retryable_provider_error"
                    if isinstance(exc, NonRetryableProviderError)
                    else "transport_error"
                ),
                "error_type": type(exc).__name__,
                "elapsed_seconds": time.perf_counter() - started,
            })
            self.attempts.append(evidence)
            raise
        evidence.update(self.client.last_http_evidence())
        evidence.update({
            "status": "response_received", "response_sha256": sha256_text(str(response)),
            "elapsed_seconds": time.perf_counter() - started,
        })
        self.attempts.append(evidence)
        return str(response)


class EmbedCompat:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        unexpected = set(kwargs).difference({"task_type", "tag"})
        if unexpected:
            raise TypeError(f"unsupported embedding compatibility kwargs: {sorted(unexpected)}")
        self.calls.append({
            "texts": len(texts),
            "input_sha256": _canonical(list(texts)),
            "requested_task_type": kwargs.get("task_type"),
            "requested_tag": kwargs.get("tag"),
            "provider_task_type_sent": None,
            "provider_output_dimensionality_sent": None,
        })
        vectors = self.client.embed(list(texts))
        if len(vectors) != len(texts) or any(len(vector) != 3072 for vector in vectors):
            raise RuntimeError("Gemini Embedding 001 must return aligned 3072-d vectors")
        return vectors


def _first_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def _retry_same_request(
    chat: PhysicalChat,
    prompt: str,
    *,
    validator: Callable[[str], tuple[bool, Any, str]],
    sleep_fn: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Any:
    canonical_hashes: list[str] = []
    last_error = "unknown generation failure"
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            raw = chat(prompt, **kwargs)
            canonical_hashes.append(chat.attempts[-1]["canonical_request_sha256"])
            valid, value, error = validator(raw)
            if valid:
                if len(set(canonical_hashes)) != 1:
                    raise RuntimeError("transient retry changed canonical request")
                return value
            last_error = error
            chat.attempts[-1]["validation_error"] = error
        except NonRetryableProviderError:
            raise
        except Exception as exc:
            if chat.attempts:
                canonical_hashes.append(chat.attempts[-1]["canonical_request_sha256"])
            last_error = f"{type(exc).__name__}: {str(exc)[-300:]}"
        if attempt + 1 < MAX_GENERATION_ATTEMPTS:
            if chat.attempts:
                chat.attempts[-1]["backoff_before_next_seconds"] = FIXED_BACKOFF_S
            sleep_fn(FIXED_BACKOFF_S)
    raise RuntimeError(f"{chat.stage} failed after five physical attempts: {last_error}")


def _nonempty(raw: str) -> tuple[bool, Any, str]:
    value = str(raw).strip()
    return bool(value), value, "empty generation response"


def _operations(raw: str) -> tuple[bool, Any, str]:
    value = _first_object(raw)
    valid = isinstance(value, dict) and set(value) == {"operations"} and isinstance(value.get("operations"), list)
    return valid, str(raw), "online response must contain exactly one operations:list field"


def _durable_retry(
    path: Path,
    binding: Mapping[str, Any],
    chat: PhysicalChat,
    prompt: str,
    *,
    validator: Callable[[str], tuple[bool, Any, str]],
    **kwargs: Any,
) -> Any:
    prompt_sha = sha256_text(prompt)
    if path.exists():
        saved = _verify_sealed_payload(_load_json(path), path)
        if saved.get("binding") != dict(binding) or saved.get("filled_prompt_sha256") != prompt_sha:
            raise RuntimeError(f"durable generation checkpoint binding mismatch: {path}")
        if saved.get("status") == "ok":
            response = str(saved.get("response", ""))
            if saved.get("response_sha256") != sha256_text(response):
                raise RuntimeError(f"durable response hash mismatch: {path}")
            valid, value, error = validator(response)
            if not valid:
                raise RuntimeError(f"durable response no longer validates: {error}")
            _validate_physical_attempts(saved.get("attempts"), same_canonical_body=True)
            return value
        attempts = _validate_physical_attempts(saved.get("attempts"), same_canonical_body=True)
        if saved.get("status") == "non_retryable_provider_error":
            if attempts[-1].get("status") != "non_retryable_provider_error":
                raise RuntimeError(f"non-retryable checkpoint evidence mismatch: {path}")
            raise NonRetryableProviderError(
                f"durable non-retryable provider rejection: {path}"
            )
        if len(attempts) != MAX_GENERATION_ATTEMPTS:
            raise RuntimeError(f"exhausted checkpoint does not contain five attempts: {path}")
        raise RuntimeError(f"durable generation checkpoint exhausted: {path}")
    try:
        response = _retry_same_request(chat, prompt, validator=validator, **kwargs)
    except NonRetryableProviderError as exc:
        _atomic_json(path, _sealed_payload({
            "schema_version": RUN_SCHEMA,
            "binding": dict(binding),
            "filled_prompt_sha256": prompt_sha,
            "status": "non_retryable_provider_error",
            "attempts": chat.attempts,
            "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
        }))
        raise
    except Exception as exc:
        if len(chat.attempts) == MAX_GENERATION_ATTEMPTS:
            _atomic_json(path, _sealed_payload({
                "schema_version": RUN_SCHEMA,
                "binding": dict(binding),
                "filled_prompt_sha256": prompt_sha,
                "status": "exhausted",
                "attempts": chat.attempts,
                "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
            }))
        raise
    _validate_physical_attempts(chat.attempts, same_canonical_body=True)
    _atomic_json(path, _sealed_payload({
        "schema_version": RUN_SCHEMA,
        "binding": dict(binding),
        "filled_prompt_sha256": prompt_sha,
        "status": "ok",
        "attempts": chat.attempts,
        "response": response,
        "response_sha256": sha256_text(str(response)),
    }))
    return response


def _session_binding(
    item: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    contract: PromptContract,
) -> dict[str, Any]:
    batch = render_lme_session(str(spec["source_session_id"]), str(spec["date"]), spec["turns"])
    prompt = render_write_prompt(contract, batch)
    return {
        "question_id": str(item["question_id"]),
        "manifest_sha256": MANIFEST_SHA256,
        "record_sha256": str(manifest_binding["record_sha256"]),
        "source_index": int(manifest_binding["source_index"]),
        "source_position": int(spec["source_position"]),
        "source_session_id": str(spec["source_session_id"]),
        "date": str(spec["date"]),
        "turn_count": len(spec["turns"]),
        "session_payload_sha256": _canonical(spec["turns"]),
        "rendered_batch_sha256": sha256_text(batch),
        "filled_write_prompt_sha256": sha256_text(prompt),
        "write_template_sha256": contract.prompt_sha256["write_template"],
        "prompt_contract_sha256": contract.contract_sha256,
    }


def _write_packet(
    *,
    item: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    question_cache: Path,
    contract: PromptContract,
    generation_client_type: type[Any],
    stack: Any,
) -> dict[str, Any]:
    binding = _session_binding(item, manifest_binding, spec, contract)
    position = int(spec["source_position"])
    path = question_cache / "writer_packets" / f"source-{position:04d}.json"
    failure_path = path.with_suffix(".failure.json")
    if path.exists():
        packet = _verify_sealed_payload(_load_json(path), path)
        if packet.get("binding") != binding:
            raise RuntimeError(f"writer packet binding mismatch: {path}")
        if not isinstance(packet.get("batch_summary"), str) or not packet["batch_summary"].strip():
            raise RuntimeError(f"writer packet has empty summary: {path}")
        if not isinstance(packet.get("atomic_facts"), list):
            raise RuntimeError(f"writer packet facts are not a list: {path}")
        attempts = _validate_physical_attempts(
            packet.get("transport_attempts"), same_canonical_body=True
        )
        if int(packet.get("writer_attempts", 0)) != len(attempts):
            raise RuntimeError(f"writer packet attempt count mismatch: {path}")
        if packet.get("write_prompt_sha256") != binding["filled_write_prompt_sha256"]:
            raise RuntimeError(f"writer packet prompt hash mismatch: {path}")
        if packet.get("rendered_batch_sha256") != binding["rendered_batch_sha256"]:
            raise RuntimeError(f"writer packet batch hash mismatch: {path}")
        return packet
    if failure_path.exists():
        failure = _verify_sealed_payload(_load_json(failure_path), failure_path)
        if failure.get("binding") != binding:
            raise RuntimeError(f"writer failure checkpoint binding mismatch: {failure_path}")
        attempts = _validate_physical_attempts(
            failure.get("transport_attempts"), same_canonical_body=True
        )
        if failure.get("status") == "non_retryable_provider_error":
            if attempts[-1].get("status") != "non_retryable_provider_error":
                raise RuntimeError(f"writer non-retryable evidence mismatch: {failure_path}")
            raise NonRetryableProviderError(
                f"writer non-retryable provider rejection already sealed: {failure_path}"
            )
        if len(attempts) != MAX_GENERATION_ATTEMPTS:
            raise RuntimeError(f"writer exhausted checkpoint does not contain five attempts: {failure_path}")
        raise RuntimeError(f"writer physical attempt cap was already exhausted: {failure_path}")
    client = generation_client_type(question_cache / "generation", stack)
    chat = PhysicalChat(
        client, contract, default_model=stack.writer,
        stage=f"writer_source_{position:04d}",
    )
    try:
        packet = write_one_session(
            contract=contract,
            source_session_id=str(spec["source_session_id"]),
            date=str(spec["date"]),
            turns=spec["turns"],
            chat_fn=chat,
            write_model=stack.writer,
        )
    except NonRetryableProviderError as exc:
        _atomic_json(failure_path, _sealed_payload({
            "schema_version": RUN_SCHEMA, "binding": binding,
            "status": "non_retryable_provider_error",
            "transport_attempts": chat.attempts,
            "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
        }))
        raise
    except Exception as exc:
        if len(chat.attempts) == MAX_GENERATION_ATTEMPTS:
            _atomic_json(failure_path, _sealed_payload({
                "schema_version": RUN_SCHEMA, "binding": binding,
                "status": "exhausted", "transport_attempts": chat.attempts,
                "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
            }))
        raise
    if int(packet["writer_attempts"]) != len(chat.attempts):
        raise RuntimeError("writer logical/physical attempt accounting mismatch")
    packet.update({
        "source_position": position,
        "turn_count": len(spec["turns"]),
        "binding": binding,
        "transport_attempts": chat.attempts,
        "usage": client.usage.snapshot(),
    })
    canonical_hashes = {row["canonical_request_sha256"] for row in chat.attempts}
    if len(canonical_hashes) != 1:
        raise RuntimeError("writer retry changed canonical request body")
    _validate_physical_attempts(chat.attempts, same_canonical_body=True)
    _atomic_json(path, _sealed_payload(packet))
    return packet


class ViewCheckpointChat:
    def __init__(
        self,
        *,
        question_cache: Path,
        question_binding: Mapping[str, Any],
        ordered_specs: Sequence[Mapping[str, Any]],
        client: Any,
        contract: PromptContract,
        default_model: str,
    ) -> None:
        self.root = question_cache / "view_responses"
        self.binding = dict(question_binding)
        self.specs = list(ordered_specs)
        self.client = client
        self.contract = contract
        self.default_model = default_model
        self.call_index = 0
        self.fatal = False

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        self.call_index += 1
        frontier = self.call_index + 1
        if self.fatal:
            raise RuntimeError("earlier Online frontier exhausted its physical attempt cap")
        spec = self.specs[frontier - 1]
        binding = {
            **self.binding,
            "frontier_session_id": frontier,
            "source_position": int(spec["source_position"]),
            "source_session_id": str(spec["source_session_id"]),
            "date": str(spec["date"]),
        }
        path = self.root / f"frontier-{frontier:04d}.json"
        chat = PhysicalChat(
            self.client, self.contract, default_model=self.default_model,
            stage=f"online_frontier_{frontier:04d}",
        )
        try:
            return str(_durable_retry(path, binding, chat, prompt, validator=_operations, **kwargs))
        except Exception:
            self.fatal = True
            raise


def _judge(
    *,
    path: Path,
    binding: Mapping[str, Any],
    client: Any,
    contract: PromptContract,
    model: str,
    question: str,
    reference: str,
    predicted: str,
) -> tuple[int, str, list[dict[str, Any]]]:
    prompt = contract.judge_template.format(
        question=question, reference=reference, predicted=predicted
    )
    prompt_sha = sha256_text(prompt)
    if path.exists():
        saved = _verify_sealed_payload(_load_json(path), path)
        if saved.get("binding") != dict(binding) or saved.get("filled_prompt_sha256") != prompt_sha:
            raise RuntimeError("judge checkpoint binding mismatch")
        if saved.get("status") != "ok":
            attempts = _validate_physical_attempts(saved.get("attempts"), same_canonical_body=False)
            if saved.get("status") == "non_retryable_provider_error":
                if attempts[-1].get("status") != "non_retryable_provider_error":
                    raise RuntimeError("judge non-retryable evidence mismatch")
                raise NonRetryableProviderError(
                    "judge non-retryable provider rejection already sealed"
                )
            if len(attempts) != 5:
                raise RuntimeError("judge exhausted checkpoint does not contain five attempts")
            raise RuntimeError("judge checkpoint already exhausted")
        response = str(saved.get("response", ""))
        if saved.get("response_sha256") != sha256_text(response):
            raise RuntimeError("judge response hash mismatch")
        tags = re.findall(r"\[\[(yes|no)\]\]", response.lower())
        if len(tags) != 1 or int(saved.get("correct", -1)) != int(tags[0] == "yes"):
            raise RuntimeError("judge cached verdict semantics mismatch")
        attempts = _validate_physical_attempts(saved.get("attempts"), same_canonical_body=False)
        expected_schedule = [200, 400, 800, 1600, 3200][: len(attempts)]
        if [int(row.get("max_tokens", -1)) for row in attempts] != expected_schedule:
            raise RuntimeError("judge cached max-token schedule mismatch")
        return int(saved["correct"]), response, attempts
    chat = PhysicalChat(client, contract, default_model=model, stage="judge")
    last_output = ""
    try:
        for application_ordinal, max_tokens in enumerate((200, 400, 800, 1600, 3200), 1):
            try:
                last_output = chat(
                    prompt, model=model, tag="judge", temperature=0.0,
                    thinking_budget=0, max_tokens=max_tokens,
                    use_cache=False, json_mode=False,
                ).strip()
                tags = re.findall(r"\[\[(yes|no)\]\]", last_output.lower())
                chat.attempts[-1]["application_ordinal"] = application_ordinal
                if len(tags) == 1:
                    correct = int(tags[0] == "yes")
                    _validate_physical_attempts(chat.attempts, same_canonical_body=False)
                    _atomic_json(path, _sealed_payload({
                        "schema_version": RUN_SCHEMA, "binding": dict(binding),
                        "filled_prompt_sha256": prompt_sha, "status": "ok",
                        "correct": correct, "response": last_output,
                        "response_sha256": sha256_text(last_output), "attempts": chat.attempts,
                        "max_tokens_schedule": [200, 400, 800, 1600, 3200],
                    }))
                    return correct, last_output, chat.attempts
                chat.attempts[-1]["validation_error"] = "judge must return exactly one binary tag"
            except NonRetryableProviderError:
                if chat.attempts:
                    chat.attempts[-1]["application_ordinal"] = application_ordinal
                raise
            except Exception:
                if len(chat.attempts) >= application_ordinal:
                    chat.attempts[-1]["application_ordinal"] = application_ordinal
            if application_ordinal < 5:
                if chat.attempts:
                    chat.attempts[-1]["backoff_before_next_seconds"] = FIXED_BACKOFF_S
                time.sleep(FIXED_BACKOFF_S)
        raise RuntimeError(f"judge did not return exactly one binary tag: {last_output[:200]}")
    except NonRetryableProviderError as exc:
        _atomic_json(path, _sealed_payload({
            "schema_version": RUN_SCHEMA, "binding": dict(binding),
            "filled_prompt_sha256": prompt_sha,
            "status": "non_retryable_provider_error",
            "attempts": chat.attempts,
            "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
        }))
        raise
    except Exception as exc:
        if len(chat.attempts) == MAX_GENERATION_ATTEMPTS:
            _atomic_json(path, _sealed_payload({
                "schema_version": RUN_SCHEMA, "binding": dict(binding),
                "filled_prompt_sha256": prompt_sha, "status": "exhausted",
                "attempts": chat.attempts,
                "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
            }))
        raise


def _require_pass(report: Mapping[str, Any], label: str) -> None:
    if str(report.get("status")) != "pass" or int(report.get("failed", 1)) != 0:
        failed = [row.get("name") for row in report.get("checks", []) if not row.get("passed")]
        raise RuntimeError(f"{label} audit failed: {failed}")


def _validate_session_graph_binding(
    graph: Mapping[str, Any],
    ordered_specs: Sequence[Mapping[str, Any]],
    order_audit: Mapping[str, Any],
) -> None:
    if graph.get("causal_order_audit") != dict(order_audit):
        raise RuntimeError("session graph causal-order audit differs from current source")
    packets = graph.get("session_packets")
    if not isinstance(packets, list) or len(packets) != len(ordered_specs):
        raise RuntimeError("session graph packet count differs from current source")
    for ordinal, (packet, spec) in enumerate(zip(packets, ordered_specs), 1):
        expected = {
            "session_id": ordinal,
            "source_position": int(spec["source_position"]),
            "source_session_id": str(spec["source_session_id"]),
            "timestamp": str(spec["date"]),
            "turn_count": len(spec["turns"]),
        }
        if any(packet.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"session graph packet binding differs at ordinal {ordinal}")
        marker = f"D{ordinal}:0"
        provenance = graph.get("provenance_map", {}).get(marker)
        if not isinstance(provenance, Mapping):
            raise RuntimeError(f"session graph provenance missing {marker}")
        if (
            provenance.get("source_position") != expected["source_position"]
            or provenance.get("source_session_id") != expected["source_session_id"]
            or provenance.get("date") != expected["timestamp"]
        ):
            raise RuntimeError(f"session graph provenance binding differs for {marker}")


def _validate_online_steps(
    online_graph: Mapping[str, Any],
    expected_sessions: int,
) -> None:
    steps = online_graph.get("online_fusion", {}).get("steps", [])
    if not isinstance(steps, list) or len(steps) != expected_sessions:
        raise RuntimeError("Online fusion did not emit one step per causal frontier")
    if sum(int(step.get("model_call_count", 0)) for step in steps) != expected_sessions - 1:
        raise RuntimeError("Online fusion call count differs from N-1")
    for ordinal, step in enumerate(steps, 1):
        if bool(step.get("model_call_made")) != (ordinal > 1):
            raise RuntimeError("Online model_call_made sequence differs from causal contract")
        if bool(step.get("model_call_made")) and step.get("model_error") is not None:
            raise RuntimeError("Online fusion contained a swallowed model/embedding error")
        if any(
            row.get("source_index") is None
            and "must contain only an operations list" in str(row.get("reason", ""))
            for row in step.get("rejected", [])
        ):
            raise RuntimeError("Online fusion accepted a top-level schema failure")


def _checkpoint_attempts(path: Path, *, same_canonical_body: bool) -> list[dict[str, Any]]:
    payload = _verify_sealed_payload(_load_json(path), path)
    return _validate_physical_attempts(
        payload.get("attempts"), same_canonical_body=same_canonical_body
    )


def _process_question(
    *,
    position: int,
    item: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    output_root: Path,
    cache_root: Path,
    config_sha256: str,
    contract: PromptContract,
    release_clients: ModuleType,
    release_memory: ModuleType,
    generation_client_type: type[Any],
    retrieval_client_type: type[Any],
    writer_workers: int,
) -> dict[str, Any]:
    question_id = str(item["question_id"])
    result_dir = output_root / "questions" / f"{position:03d}-{question_id}"
    result_path = result_dir / "result.json"
    result_binding = {
        "position": position,
        "question_id": question_id,
        "record_sha256": str(manifest_binding["record_sha256"]),
        "manifest_sha256": MANIFEST_SHA256,
        "config_sha256": config_sha256,
    }
    if result_path.exists():
        result = _load_result(result_path)
        if result.get("binding") != result_binding:
            raise RuntimeError(f"result binding mismatch: {result_path}")
        if result.get("status") == "complete":
            return result
        raise RuntimeError("canonical result exists but is not complete")

    question_cache = cache_root / "questions" / f"{position:03d}-{question_id}"
    ordered_specs, order_audit = _ordered_session_specs(item)
    stack = release_clients.ModelStack.for_provider("gemini")

    def write(spec: Mapping[str, Any]) -> dict[str, Any]:
        return _write_packet(
            item=item, manifest_binding=manifest_binding, spec=spec,
            question_cache=question_cache, contract=contract,
            generation_client_type=generation_client_type, stack=stack,
        )

    if writer_workers == 1:
        packets = [write(spec) for spec in ordered_specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=writer_workers) as executor:
            packets = list(executor.map(write, ordered_specs))
    if [int(packet["source_position"]) for packet in packets] != [int(spec["source_position"]) for spec in ordered_specs]:
        raise RuntimeError("writer packet order differs from stable causal order")
    session_checkpoint_binding = _canonical({
        "result_binding": result_binding,
        "ordered_session_bindings": [
            _session_binding(item, manifest_binding, spec, contract)
            for spec in ordered_specs
        ],
        "writer_packet_payload_sha256": [_canonical(packet) for packet in packets],
    })

    retrieval_client = retrieval_client_type(question_cache / "retrieval", stack)
    embed = EmbedCompat(retrieval_client)
    session_path = question_cache / "graphs" / "graph_session.pkl"
    session_graph = _load_graph(session_path, config_sha256, session_checkpoint_binding)
    if session_graph is None:
        session_graph = build_session_graph_from_packets(
            packets, embed, contract=contract,
            display_text_fn=release_memory.temporal_display_text,
        )
        session_graph["architecture"]["causal_order_policy"] = "stable_timestamp_then_source_position"
        session_graph["causal_order_audit"] = order_audit
        _validate_session_graph_binding(session_graph, ordered_specs, order_audit)
        graph_audit = gm.audit_graph(
            session_graph,
            legal_dia_ids=session_graph["provenance_map"].keys(),
            expected_sessions=len(ordered_specs),
        )
        _require_pass(graph_audit, "session graph")
        _atomic_json(question_cache / "audits" / "graph_session.audit.json", graph_audit)
        _save_graph(session_path, session_graph, config_sha256, session_checkpoint_binding)
    else:
        _validate_session_graph_binding(session_graph, ordered_specs, order_audit)
        graph_audit = gm.audit_graph(
            session_graph,
            legal_dia_ids=session_graph["provenance_map"].keys(),
            expected_sessions=len(ordered_specs),
        )
        _require_pass(graph_audit, "resumed session graph")

    generation_client = generation_client_type(question_cache / "generation_main", stack)
    online_path = question_cache / "graphs" / "graph_online.pkl"
    online_checkpoint_binding = _canonical({
        "session_checkpoint_binding": session_checkpoint_binding,
        "session_content_fingerprint": session_graph["content_fingerprint"],
        "ordered_source_positions": [int(spec["source_position"]) for spec in ordered_specs],
        "online_update_prompt_sha256": online_memory_v3.PROMPT_SHA256["online_event_update"],
    })
    online_graph = _load_graph(online_path, config_sha256, online_checkpoint_binding)
    if online_graph is None:
        view_chat = ViewCheckpointChat(
            question_cache=question_cache,
            question_binding=result_binding,
            ordered_specs=ordered_specs,
            client=generation_client,
            contract=contract,
            default_model=stack.writer,
        )
        online_graph = online_memory_v3.add_online_event_views(
            session_graph, view_chat, embed
        )
        _validate_online_steps(online_graph, len(ordered_specs))
        online_audit = online_memory_v3.audit_online_graph(session_graph, online_graph)
        _require_pass(online_audit, "online causal graph")
        online_graph_audit = gm.audit_graph(
            online_graph,
            legal_dia_ids=online_graph["provenance_map"].keys(),
            expected_sessions=len(ordered_specs),
        )
        _require_pass(online_graph_audit, "online typed graph")
        _atomic_json(question_cache / "audits" / "graph_online.audit.json", online_graph_audit)
        _atomic_json(question_cache / "audits" / "online_causal.audit.json", online_audit)
        _save_graph(online_path, online_graph, config_sha256, online_checkpoint_binding)
    else:
        _validate_session_graph_binding(online_graph, ordered_specs, order_audit)
        _validate_online_steps(online_graph, len(ordered_specs))
        online_audit = online_memory_v3.audit_online_graph(session_graph, online_graph)
        _require_pass(online_audit, "resumed online causal graph")
        online_graph_audit = gm.audit_graph(
            online_graph,
            legal_dia_ids=online_graph["provenance_map"].keys(),
            expected_sessions=len(ordered_specs),
        )
        _require_pass(online_graph_audit, "resumed online typed graph")

    # Raw benchmark question only: date/type/gold never reach search or planner.
    question = str(item["question"])
    query_vector = embed([question])[0]
    index = QwenGraphMemoryIndex(
        online_graph,
        reranker=retrieval_client.rerank,
        instruction=contract.qwen_instruction,
    )
    retrieved, retrieval_trace = index.search(
        question, query_vector,
        rerank_chat=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Gemini rerank forbidden")),
    )
    expected_selected = min(30, int(retrieval_trace["candidates_after_cap"]))
    selected_ids = retrieval_trace["rerank"]["selected_fact_ids"]
    if len(selected_ids) != expected_selected or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("final Qwen selection is not the exact unique Top30")
    if not retrieval_client.last_rerank_audit.get("strict_indices"):
        raise RuntimeError("Qwen strict index audit missing")

    planner_path = question_cache / "reader" / "planner.json"
    planner_binding = {**result_binding, "raw_question_sha256": sha256_text(question), "labels_visible": False}
    planner_non_retryable: NonRetryableProviderError | None = None

    def durable_planner_chat(prompt: str, **kwargs: Any) -> str:
        nonlocal planner_non_retryable
        if prompt != planner.build_planner_prompt(question):
            raise RuntimeError("strict planner prompt was not built from the raw question only")
        chat = PhysicalChat(
            generation_client, contract, default_model=stack.answerer, stage="semantic_planner"
        )
        try:
            return str(_durable_retry(
                planner_path, planner_binding, chat, prompt,
                validator=_nonempty, **kwargs,
            ))
        except NonRetryableProviderError as exc:
            planner_non_retryable = exc
            raise

    planning, _planner_raw = strict_planner.classify_strict(
        question, model=stack.answerer, chat_fn=durable_planner_chat
    )
    if planner_non_retryable is not None:
        raise planner_non_retryable
    operation = planning.plan.operation
    operation_context, opctx_audit = materialize_operation_context(
        operation=operation,
        question=question,
        base_retrieved=retrieved,
        retrieval_trace=retrieval_trace,
        graph=index.graph,
        selected_top_k=30,
    )
    if opctx_audit["selected_fact_ids"] != selected_ids:
        raise RuntimeError("OpCtx changed the frozen Top30 selection")

    qa_system = contract.qa_template.format(
        context=operation_context,
        qdate=str(item["question_date"]),
        question=question,
    )
    qa_binding = {
        **result_binding,
        "raw_question_sha256": sha256_text(question),
        "question_date_sha256": sha256_text(str(item["question_date"])),
        "context_sha256": sha256_text(operation_context),
        "qa_template_sha256": contract.prompt_sha256["qa_template"],
    }
    qa_chat = PhysicalChat(generation_client, contract, default_model=stack.answerer, stage="qa")
    response = str(_durable_retry(
        question_cache / "reader" / "qa.json",
        qa_binding,
        qa_chat,
        question,
        validator=_nonempty,
        model=stack.answerer,
        tag="read",
        system=qa_system,
        temperature=0.0,
        thinking_budget=0,
        max_tokens=500,
        use_cache=False,
        json_mode=False,
    )).strip()

    reference = str(item["answer"])
    judge_binding = {
        **result_binding,
        "question_sha256": sha256_text(question),
        "reference_sha256": sha256_text(reference),
        "predicted_sha256": sha256_text(response),
        "judge_template_sha256": contract.prompt_sha256["judge_template"],
    }
    correct, judge_output, judge_attempts = _judge(
        path=question_cache / "reader" / "judge.json",
        binding=judge_binding,
        client=generation_client,
        contract=contract,
        model=stack.judge,
        question=question,
        reference=reference,
        predicted=response,
    )

    writer_attempt_rows = [
        attempt
        for packet in packets
        for attempt in packet.get("transport_attempts", [])
    ]
    online_attempt_rows = [
        attempt
        for checkpoint in sorted((question_cache / "view_responses").glob("frontier-*.json"))
        for attempt in _checkpoint_attempts(checkpoint, same_canonical_body=True)
    ]
    planner_attempt_rows = _checkpoint_attempts(planner_path, same_canonical_body=True)
    qa_attempt_rows = _checkpoint_attempts(
        question_cache / "reader" / "qa.json", same_canonical_body=True
    )
    all_generation_attempts = [
        *writer_attempt_rows,
        *online_attempt_rows,
        *planner_attempt_rows,
        *qa_attempt_rows,
        *judge_attempts,
    ]

    result = {
        "schema_version": RUN_SCHEMA,
        "binding": result_binding,
        "status": "complete",
        "question_id": question_id,
        "question": question,
        "question_date": str(item["question_date"]),
        "question_type": str(item["question_type"]),
        "is_abstention": question_id.endswith("_abs"),
        "reference": reference,
        "response": response,
        "correct": correct,
        "judge_output": judge_output,
        "operation": operation,
        "semantic_plan": planning.to_dict(),
        "context_policy": opctx_audit["context_policy"],
        "retrieved_context": operation_context,
        "retrieved_characters": len(operation_context),
        "retrieval_trace": retrieval_trace,
        "operation_context_audit": opctx_audit,
        "qwen_strict_audit": retrieval_client.last_rerank_audit,
        "causal_order_audit": order_audit,
        "online_fusion_diagnostics": {
            "accepted_operations": int(online_graph.get("stats", {}).get("online_accepted_operations", 0)),
            "rejected_operations": int(online_graph.get("stats", {}).get("online_rejected_operations", 0)),
            "event_view_nodes": int(online_graph.get("stats", {}).get("event_view_nodes", 0)),
            "logical_calls": int(online_graph.get("stats", {}).get("online_fusion_calls", 0)),
        },
        "graph": {
            "session_path": str(session_path),
            "session_fingerprint": session_graph["content_fingerprint"],
            "online_path": str(online_path),
            "online_fingerprint": online_graph["content_fingerprint"],
            "session_audit_failed": graph_audit["failed"],
            "online_audit_failed": online_audit["failed"],
        },
        "prompt_evidence": {
            "write_template_sha256": contract.prompt_sha256["write_template"],
            "filled_qa_system_sha256": sha256_text(qa_system),
            "qa_template_sha256": contract.prompt_sha256["qa_template"],
            "filled_judge_prompt_sha256": sha256_text(contract.judge_template.format(
                question=question, reference=reference, predicted=response
            )),
            "judge_template_sha256": contract.prompt_sha256["judge_template"],
        },
        "physical_attempts": {
            "writer": len(writer_attempt_rows),
            "online_fusion": len(online_attempt_rows),
            "strict_planner": len(planner_attempt_rows),
            "qa": len(qa_attempt_rows),
            "judge": len(judge_attempts),
            "total_generation": len(all_generation_attempts),
            "total_generation_elapsed_seconds": sum(
                float(row.get("elapsed_seconds", 0.0)) for row in all_generation_attempts
            ),
        },
        "embedding_compat_calls": embed.calls,
        "generation_usage": generation_client.usage.snapshot(),
        "error": "",
    }
    _save_result(result_path, result)
    return result


def _safe_process(**kwargs: Any) -> dict[str, Any]:
    try:
        return _process_question(**kwargs)
    except Exception as exc:
        item = kwargs["item"]
        binding = kwargs["manifest_binding"]
        position = int(kwargs["position"])
        qid = str(item["question_id"])
        result_binding = {
            "position": position, "question_id": qid,
            "record_sha256": str(binding["record_sha256"]),
            "manifest_sha256": MANIFEST_SHA256,
            "config_sha256": str(kwargs["config_sha256"]),
        }
        result = {
            "schema_version": RUN_SCHEMA, "binding": result_binding,
            "status": "error", "question_id": qid,
            "question_type": str(item["question_type"]),
            "is_abstention": qid.endswith("_abs"),
            "correct": 0, "error": f"{type(exc).__name__}: {str(exc)[-1000:]}",
        }
        error_path = (
            kwargs["output_root"] / "questions" / f"{position:03d}-{qid}"
            / "errors" / f"attempt-{time.time_ns()}.json"
        )
        _atomic_json(error_path, _sealed_payload(result))
        return result


def _aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]

    def metrics(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(group),
            "correct": sum(int(row.get("correct", 0)) for row in group),
            "accuracy": (
                sum(int(row.get("correct", 0)) for row in group) / len(group)
                if group else None
            ),
        }

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_abs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_operation_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cross: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in complete:
        qtype = str(row["question_type"])
        by_type[qtype].append(row)
        by_abs["abstention" if row.get("is_abstention") else "non_abstention"].append(row)
        by_operation[str(row["operation"])].append(row)
        by_policy[str(row["context_policy"])].append(row)
        by_operation_policy[f"{row['operation']} | {row['context_policy']}"].append(row)
        cross[f"{qtype} | {row['operation']} | {row['context_policy']}"].append(row)
    candidates = [int(row["retrieval_trace"]["candidates_after_cap"]) for row in complete]
    selections = [len(row["retrieval_trace"]["rerank"]["selected_fact_ids"]) for row in complete]
    attempt_totals = [int(row["physical_attempts"]["total_generation"]) for row in complete]
    generation_seconds = [
        float(row["physical_attempts"]["total_generation_elapsed_seconds"])
        for row in complete
    ]
    context_chars = [int(row["retrieved_characters"]) for row in complete]
    planner_sources: dict[str, int] = defaultdict(int)
    stage_attempts: dict[str, int] = defaultdict(int)
    stage_seconds: dict[str, float] = defaultdict(float)
    online_totals = {"accepted_operations": 0, "rejected_operations": 0, "event_view_nodes": 0}
    for row in complete:
        planner_sources[str(row["semantic_plan"]["source"])] += 1
        for stage, value in row["physical_attempts"].items():
            if stage in {"total_generation", "total_generation_elapsed_seconds"}:
                continue
            stage_attempts[stage] += int(value)
        for key in online_totals:
            online_totals[key] += int(row["online_fusion_diagnostics"][key])
        stage_seconds["generation_total"] += float(
            row["physical_attempts"]["total_generation_elapsed_seconds"]
        )
    expected_types = {
        "knowledge-update": 16,
        "multi-session": 27,
        "single-session-assistant": 11,
        "single-session-preference": 6,
        "single-session-user": 14,
        "temporal-reasoning": 26,
    }
    observed_type_rows: dict[str, int] = defaultdict(int)
    for row in rows:
        observed_type_rows[str(row.get("question_type", "unknown"))] += 1
    return {
        "schema_version": RUN_SCHEMA,
        "rows": len(rows),
        "complete": len(complete),
        "errors": len(rows) - len(complete),
        "overall_all_rows_errors_as_incorrect": metrics(rows),
        "overall_complete_rows": metrics(complete),
        "by_question_type": {key: metrics(value) for key, value in sorted(by_type.items())},
        "question_type_completion": {
            key: {
                "full_manifest_expected": expected,
                "selected_rows": observed_type_rows.get(key, 0),
                "complete": len(by_type.get(key, [])),
                "error": observed_type_rows.get(key, 0) - len(by_type.get(key, [])),
            }
            for key, expected in expected_types.items()
        },
        "by_abstention": {key: metrics(value) for key, value in sorted(by_abs.items())},
        "by_operation": {key: metrics(value) for key, value in sorted(by_operation.items())},
        "by_context_policy": {key: metrics(value) for key, value in sorted(by_policy.items())},
        "by_operation_policy": {key: metrics(value) for key, value in sorted(by_operation_policy.items())},
        "posthoc_type_operation_policy": {key: metrics(value) for key, value in sorted(cross.items())},
        "diagnostics": {
            "candidate_after_cap_mean": sum(candidates) / len(candidates) if candidates else None,
            "candidate_after_cap_min": min(candidates) if candidates else None,
            "candidate_after_cap_max": max(candidates) if candidates else None,
            "selected_fact_count_mean": sum(selections) / len(selections) if selections else None,
            "selected_fact_count_min": min(selections) if selections else None,
            "selected_fact_count_max": max(selections) if selections else None,
            "context_characters_mean": sum(context_chars) / len(context_chars) if context_chars else None,
            "context_characters_min": min(context_chars) if context_chars else None,
            "context_characters_max": max(context_chars) if context_chars else None,
            "generation_physical_attempts_total": sum(attempt_totals),
            "generation_physical_attempts_mean": sum(attempt_totals) / len(attempt_totals) if attempt_totals else None,
            "generation_elapsed_seconds_total": sum(generation_seconds),
            "physical_attempts_by_stage": dict(sorted(stage_attempts.items())),
            "elapsed_seconds_by_stage": dict(sorted(stage_seconds.items())),
            "planner_source_counts": dict(sorted(planner_sources.items())),
            "online_fusion_totals": online_totals,
            "error_rows": [
                {"question_id": row.get("question_id"), "error": row.get("error")}
                for row in rows if row.get("status") != "complete"
            ],
        },
        "routing_label_visibility": "question_type used only after completion for aggregation",
        "config": config,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true", help="explicitly authorize provider calls")
    parser.add_argument("--resume", action="store_true", help="resume the exact sealed run root")
    parser.add_argument("--limit", type=int, default=100, help="manifest prefix; use 1 for smoke")
    parser.add_argument("--question-workers", type=int, default=4)
    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    if not 1 <= args.question_workers <= 8:
        raise SystemExit("--question-workers must be between 1 and 8")
    if not 1 <= args.writer_workers <= 8:
        raise SystemExit("--writer-workers must be between 1 and 8")
    if args.question_workers * args.writer_workers > 8:
        raise SystemExit("question-workers × writer-workers must not exceed global generation cap 8")

    preflight = run_preflight(args.preflight_output.resolve())
    if not args.allow_live:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    # Re-gate immediately at the live boundary, before importing provider code.
    contract = assert_prompt_contract()
    _, manifest, selected_all = _load_inputs(contract)
    _load_live_credentials()
    release_clients, release_memory = _load_release_runtime(contract)
    generation_client_type, retrieval_client_type = _client_types(release_clients)
    config = _run_config(contract)
    config_sha256 = _canonical(config)
    output_root = args.output.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    config_path = output_root / "run_config.json"

    existing_material = (
        (output_root.exists() and any(output_root.iterdir()))
        or (cache_root.exists() and any(cache_root.iterdir()))
    )
    if existing_material and not args.resume:
        raise SystemExit("output/cache root already contains material; use --resume only for this exact run")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    config_envelope = {"config": config, "config_sha256": config_sha256}
    if config_path.exists():
        if _load_json(config_path) != config_envelope:
            raise SystemExit("existing run_config differs; choose a fresh output/cache root")
    else:
        _atomic_json(config_path, config_envelope)

    selected = selected_all[: args.limit]
    bindings = manifest["items"][: args.limit]
    jobs = [
        {
            "position": position,
            "item": item,
            "manifest_binding": binding,
            "output_root": output_root,
            "cache_root": cache_root,
            "config_sha256": config_sha256,
            "contract": contract,
            "release_clients": release_clients,
            "release_memory": release_memory,
            "generation_client_type": generation_client_type,
            "retrieval_client_type": retrieval_client_type,
            "writer_workers": args.writer_workers,
        }
        for position, (item, binding) in enumerate(zip(selected, bindings))
    ]

    rows_by_position: dict[int, dict[str, Any]] = {}
    if args.question_workers == 1:
        for job in jobs:
            row = _safe_process(**job)
            rows_by_position[int(job["position"])] = row
            print(f"[{len(rows_by_position)}/{len(jobs)}] {row['question_id']} {row['status']}", flush=True)
            if args.fail_fast and row["status"] != "complete":
                raise RuntimeError(row["error"])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.question_workers) as executor:
            future_to_position = {
                executor.submit(_safe_process, **job): int(job["position"])
                for job in jobs
            }
            for future in concurrent.futures.as_completed(future_to_position):
                position = future_to_position[future]
                row = future.result()
                rows_by_position[position] = row
                print(f"[{len(rows_by_position)}/{len(jobs)}] {row['question_id']} {row['status']}", flush=True)
                if args.fail_fast and row["status"] != "complete":
                    raise RuntimeError(row["error"])

    # Seal public rows in frozen manifest order, independent of worker finish order.
    rows = [rows_by_position[position] for position in sorted(rows_by_position)]
    rows_path = output_root / "rows.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(rows_path)
    summary = _aggregate(rows, config)
    summary["manifest_prefix_count"] = args.limit
    summary["rows_sha256"] = _sha_file(rows_path)
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    summary["summary_sha256"] = _canonical(summary)
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps({
        "status": "PASS" if summary["errors"] == 0 else "PARTIAL",
        "rows": summary["rows"], "complete": summary["complete"],
        "errors": summary["errors"], "output": str(output_root),
        "cache_root": str(cache_root), "summary_sha256": summary["summary_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

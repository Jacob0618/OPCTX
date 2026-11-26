#!/usr/bin/env python3
"""Build and evaluate CreaMem3 Online/Offline readers on MemEye Open QA.

Evaluated configurations:

* ``online_v3``: strict-causal Online graph + operation-context v3;
* ``offline_v3``: global session-first Offline graph + operation-context v3;
* ``offline_p06``: the same Offline graph + semantic-subtype p06 renderer.

The two Offline configurations intentionally share one memory graph and one
retrieval result per question.  They differ only in the reader-visible context.
All three configurations independently rerank their own graph evidence.
"""

from __future__ import annotations

import os


WORK_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
JUDGE_MODEL = "openrouter/openai/gpt-5.2"
DEFAULT_CACHE_ROOT = os.environ.get(
    "AGENT_MEMORY_CACHE_ROOT", r"D:\creamem3_cache\memeye_v9\default"
)
os.environ.setdefault("AGENT_MEMORY_CACHE_ROOT", DEFAULT_CACHE_ROOT)
os.environ.setdefault("AGENT_MEMORY_BASE_CACHE", "")
os.environ["BACKBONE_MODEL"] = WORK_MODEL
os.environ["EMBED_MODEL"] = EMBED_MODEL
os.environ["BACKBONE_THINKING"] = "off"
os.environ["LLM_NO_CHAT_CACHE"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


import argparse
import copy
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVAL_DIR = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gemini_client import USAGE, chat, embed  # noqa: E402
from memeye_creamem_reader import (  # noqa: E402
    MEMEYE_JUDGE_PROMPT,
    MEMEYE_OPEN_QA_PROMPT,
    bleu1,
    load_task,
    parse_judge_response,
    qa_points,
    task_files,
)

from creamem3.analysis_offline_semantic_subtype_screen_v8 import (  # noqa: E402
    VARIANTS as OFFLINE_SUBTYPE_VARIANTS,
    _materialize as materialize_offline_subtype_context,
    select_context_policy,
)
from creamem3.graph_memory_v2 import (  # noqa: E402
    GraphMemoryIndex,
    add_offline_event_views,
    audit_graph,
    canonical_sha256,
)
from creamem3.memeye_adapter_v9 import (  # noqa: E402
    DEFAULT_CAPTION_ROOT,
    DEFAULT_DATA_ROOT,
    MEMEYE_OFFLINE_EVENT_PROMPT_SHA256,
    MEMEYE_OFFLINE_EVENT_VIEW_PROMPT,
    MEMEYE_OFFLINE_LARGE_EVENT_PROMPT_SHA256,
    MEMEYE_OFFLINE_LARGE_EVENT_VIEW_PROMPT,
    MEMEYE_ONLINE_EVENT_PROMPT_SHA256,
    MEMEYE_ONLINE_EVENT_UPDATE_PROMPT,
    SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION,
    audit_memeye_graph,
    add_large_memeye_offline_event_views,
    build_memeye_session_graph,
    build_records_from_cached_captions,
    load_mgtm_prompt_sources,
    load_mgtm_evidence_aggregator_prompt,
    load_mgtm_open_retrieval_instruct,
    validate_caption_coverage,
)
from creamem3.offline_operation_context_v8 import (  # noqa: E402
    materialize_offline_operation_context,
)
from creamem3.online_memory_v3 import add_online_event_views, audit_online_graph  # noqa: E402
from creamem3.online_operation_context_v8 import (  # noqa: E402
    BOUNDED_VIEWS_2,
    OPERATION_TO_CONTEXT_POLICY,
    ORIGINAL_FULL,
    ROUTE_ONLY,
)
from creamem3.semantic_context_v8 import (  # noqa: E402
    _candidate_view_ids,
    _grounded_claims,
)
from creamem3.semantic_query_planner_v6 import (  # noqa: E402
    classify_with_gemini,
    plan_query,
)


RESULT_SCHEMA_VERSION = "creamem3.memeye_result.v9.1"
GRAPH_ROOT = Path(r"D:\creamem3_cache\memeye_v9\graphs")
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results_memeye_v9"
DEFAULT_JUDGE_CACHE = Path(r"D:\creamem3_cache\memeye_v9\openrouter_judge")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALLOWED_JUDGE_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}
VARIANTS = ("online_v3", "offline_v3", "offline_p06")
P06 = next(item for item in OFFLINE_SUBTYPE_VARIANTS if item["name"] == "p06_compact_atomic_plus_temporal")
MEMEYE_EVIDENCE_AGGREGATOR_PROMPT = load_mgtm_evidence_aggregator_prompt()
MEMEYE_OPEN_RETRIEVAL_INSTRUCT = load_mgtm_open_retrieval_instruct()
QWEN_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
QWEN_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_CACHE = Path(r"D:\creamem3_cache\memeye_v9\qwen_rerank")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _model_chat(
    prompt: str,
    *,
    tag: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    json_mode: bool = False,
    **_: Any,
) -> str:
    del temperature
    return chat(
        prompt,
        system=system,
        model=WORK_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        thinking_budget=0,
        tag=tag,
        use_cache=True,
        json_mode=json_mode,
    )


def _embed_texts(
    texts: Sequence[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    tag: str = "creamem3_memeye_embed",
    **_: Any,
) -> list[list[float]]:
    return embed(
        list(texts),
        model=EMBED_MODEL,
        tag=tag,
        task_type=task_type,
        dim=768,
        use_cache=True,
    )


def _content_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(value for value in parts if value).strip()
    return str(message_content or "").strip()


class OpenRouterJudge:
    """Judge-only client whose credential is process-local and never serialized."""

    def __init__(self, cache_dir: Path, retries: int = 5) -> None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for MemEye Open QA")
        self._key = key
        self.model = JUDGE_MODEL.removeprefix("openrouter/")
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self._lock = threading.Lock()
        self._usage = Counter()

    def complete(self, prompt: str) -> tuple[str, str]:
        cache_key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "prompt": prompt},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            try:
                payload = _load_json(cache_path)
                text = str(payload.get("response", ""))
                raw, _mapped, _reason = parse_judge_response(text)
                if raw in ALLOWED_JUDGE_SCORES:
                    with self._lock:
                        self._usage["cache_hits"] += 1
                    return text, ""
            except Exception:
                pass
        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        encoded = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        last_error = ""
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    OPENROUTER_URL,
                    data=encoded,
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                        "User-Agent": "CreaMem3-MemEye-Eval/1.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError("OpenRouter response has no choices")
                text = _content_text((choices[0].get("message") or {}).get("content"))
                raw, _mapped, _reason = parse_judge_response(text)
                if raw not in ALLOWED_JUDGE_SCORES:
                    raise ValueError("invalid GPT-5.2 judge score")
                usage = data.get("usage") or {}
                with self._lock:
                    self._usage["calls"] += 1
                    self._usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self._usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    self._usage["total_tokens"] += int(usage.get("total_tokens") or 0)
                _dump_json(
                    cache_path,
                    {
                        "model": self.model,
                        "prompt_sha256": cache_key,
                        "response": text,
                    },
                )
                return text, ""
            except Exception as exc:
                code = getattr(exc, "code", "")
                last_error = f"{type(exc).__name__}" + (f" HTTP {code}" if code else "")
                if attempt + 1 < self.retries:
                    retry_after = 0.0
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            retry_after = float(exc.headers.get("Retry-After", "0"))
                        except (TypeError, ValueError):
                            retry_after = 0.0
                    time.sleep(max(retry_after, min(2**attempt, 12)))
        with self._lock:
            self._usage["failures"] += 1
        return "", last_error or "OpenRouter request failed"

    def usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self._usage)


class QwenReranker:
    """MGTM's qwen3-rerank protocol with an isolated D-drive cache."""

    def __init__(self, cache_dir: Path, retries: int = 5) -> None:
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for MGTM reranking")
        self._key = key
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self._lock = threading.Lock()
        self._usage = Counter()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        key_payload = {
            "model": QWEN_RERANK_MODEL,
            "instruct": MEMEYE_OPEN_RETRIEVAL_INSTRUCT,
            "query": query,
            "documents": list(documents),
        }
        cache_key = _sha256_text(
            json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            try:
                payload = _load_json(cache_path)
                scores = payload.get("scores")
                if (
                    isinstance(scores, list)
                    and len(scores) == len(documents)
                    and all(isinstance(value, (int, float)) for value in scores)
                ):
                    with self._lock:
                        self._usage["cache_hits"] += 1
                    return [float(value) for value in scores]
            except Exception:
                pass
        body: dict[str, Any] = {
            "model": QWEN_RERANK_MODEL,
            "input": {"query": query, "documents": list(documents)},
            "parameters": {
                "return_documents": False,
                "top_n": len(documents),
                "instruct": MEMEYE_OPEN_RETRIEVAL_INSTRUCT,
            },
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last_error = ""
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    QWEN_RERANK_URL,
                    data=encoded,
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                scores = [0.0] * len(documents)
                for item in (data.get("output") or {}).get("results", []):
                    index = int(item["index"])
                    if 0 <= index < len(scores):
                        scores[index] = float(item["relevance_score"])
                if len(scores) != len(documents) or not all(
                    isinstance(value, float) for value in scores
                ):
                    raise ValueError("invalid qwen3-rerank score vector")
                _dump_json(
                    cache_path,
                    {
                        "model": QWEN_RERANK_MODEL,
                        "request_sha256": cache_key,
                        "scores": scores,
                    },
                )
                with self._lock:
                    self._usage["calls"] += 1
                    usage = data.get("usage") or {}
                    self._usage["total_tokens"] += int(
                        usage.get("total_tokens") or 0
                    )
                return scores
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[-300:]}"
                if attempt + 1 < self.retries:
                    time.sleep(min(1.5 * (attempt + 1), 8.0))
        with self._lock:
            self._usage["failures"] += 1
        raise RuntimeError("qwen3-rerank failed: " + last_error)

    def usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self._usage)


def _graph_paths(task_name: str, graph_root: Path) -> dict[str, Path]:
    directory = graph_root / task_name
    return {
        "session": directory / "graph_session.json",
        "online": directory / "graph_online.json",
        "offline": directory / "graph_offline.json",
        "manifest": directory / "manifest.json",
    }


def _task_data(data_root: Path, task_file: str) -> tuple[str, Path, dict[str, Any]]:
    task_name, path, data = load_task(data_root, task_file)
    if not isinstance(data, dict):
        raise ValueError(f"invalid task object: {path}")
    return task_name, path, data


def _build_task(
    task_file: str,
    *,
    data_root: Path,
    caption_root: Path,
    graph_root: Path,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    task_name, task_path, data = _task_data(data_root, task_file)
    paths = _graph_paths(task_name, graph_root)
    if not force and all(paths[key].is_file() for key in ("session", "online", "offline", "manifest")):
        print(f"[{task_name}] graph cache exists", flush=True)
        return _load_json(paths["manifest"])

    captions, caption_report = validate_caption_coverage(
        task_name, data, caption_root
    )
    records, source_map = build_records_from_cached_captions(
        task_name, data, captions
    )
    prompt_sources = load_mgtm_prompt_sources()
    legal_ids = sorted(
        source_map,
        key=lambda value: tuple(int(x) for x in re.findall(r"\d+", value)),
    )
    print(
        f"[{task_name}] build sessions={len(records)} sources={len(legal_ids)}",
        flush=True,
    )
    session_graph = build_memeye_session_graph(
        records,
        _model_chat,
        _embed_texts,
        prompt_sources,
        workers=workers,
    )
    session_graph["memeye_source_map"] = source_map
    session_graph["memeye_caption_report"] = caption_report
    session_graph["memeye_task"] = task_name
    session_audit = audit_memeye_graph(
        session_graph,
        legal_dia_ids=legal_ids,
        expected_sessions=len(records),
    )
    if session_audit["status"] != "pass":
        raise RuntimeError(f"{task_name} session graph audit failed")
    _dump_json(paths["session"], session_graph)

    online_graph = add_online_event_views(
        session_graph,
        _model_chat,
        _embed_texts,
        prompt_template=MEMEYE_ONLINE_EVENT_UPDATE_PROMPT,
        prompt_sha256=MEMEYE_ONLINE_EVENT_PROMPT_SHA256,
        current_fact_cap=16,
        prior_fact_cap=24,
        prune_unclaimed_create_members=True,
    )
    online_graph["memeye_source_map"] = source_map
    online_graph["memeye_caption_report"] = caption_report
    online_graph["memeye_task"] = task_name
    online_audit = audit_online_graph(
        session_graph,
        online_graph,
        expected_prompt_sha256=MEMEYE_ONLINE_EVENT_PROMPT_SHA256,
        expected_current_fact_cap=16,
        expected_prior_fact_cap=24,
        expected_prune_unclaimed_create_members=True,
    )
    graph_online_audit = audit_memeye_graph(
        online_graph,
        legal_dia_ids=legal_ids,
        expected_sessions=len(records),
    )
    if online_audit["status"] != "pass" or graph_online_audit["status"] != "pass":
        raise RuntimeError(f"{task_name} online graph audit failed")
    _dump_json(paths["online"], online_graph)

    large_offline = len(session_graph.get("fact_nodes", [])) > 1000
    if large_offline:
        offline_graph = add_large_memeye_offline_event_views(
            session_graph, _model_chat, _embed_texts
        )
    else:
        offline_graph = add_offline_event_views(
            session_graph,
            _model_chat,
            _embed_texts,
            prompt_template=MEMEYE_OFFLINE_EVENT_VIEW_PROMPT,
            prompt_sha256=MEMEYE_OFFLINE_EVENT_PROMPT_SHA256,
        )
    offline_graph["memeye_source_map"] = source_map
    offline_graph["memeye_caption_report"] = caption_report
    offline_graph["memeye_task"] = task_name
    offline_audit = audit_memeye_graph(
        offline_graph,
        legal_dia_ids=legal_ids,
        expected_sessions=len(records),
    )
    if offline_audit["status"] != "pass":
        raise RuntimeError(f"{task_name} offline graph audit failed")
    _dump_json(paths["offline"], offline_graph)

    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "task": task_name,
        "task_file": str(task_path.resolve()),
        "task_file_sha256": _sha256_file(task_path),
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "caption_report": caption_report,
        "records": len(records),
        "legal_sources": len(legal_ids),
        "graphs": {
            key: {
                "path": str(paths[key]),
                "sha256": _sha256_file(paths[key]),
                "stats": graph.get("stats", {}),
                "content_fingerprint": graph.get("content_fingerprint"),
            }
            for key, graph in (
                ("session", session_graph),
                ("online", online_graph),
                ("offline", offline_graph),
            )
        },
        "audits": {
            "session": session_audit,
            "online_causal": online_audit,
            "online_graph": graph_online_audit,
            "offline": offline_audit,
        },
        "models": {
            "writer_reader_reranker_planner": WORK_MODEL,
            "embedding": EMBED_MODEL,
        },
        "usage": json.loads(USAGE.snapshot()),
    }
    _dump_json(paths["manifest"], manifest)
    return manifest


def _rebuild_online_task(
    task_file: str,
    *,
    data_root: Path,
    graph_root: Path,
) -> dict[str, Any]:
    """Rebuild only the causal Online derivative from a frozen session graph."""

    task_name, _task_path, _data = _task_data(data_root, task_file)
    paths = _graph_paths(task_name, graph_root)
    if not paths["session"].is_file() or not paths["manifest"].is_file():
        raise FileNotFoundError(f"{task_name} lacks a frozen session graph/manifest")
    session_graph = _load_json(paths["session"])
    source_map = session_graph.get("memeye_source_map") or {}
    legal_ids = sorted(
        source_map,
        key=lambda value: tuple(int(x) for x in re.findall(r"\d+", value)),
    )
    online_graph = add_online_event_views(
        session_graph,
        _model_chat,
        _embed_texts,
        prompt_template=MEMEYE_ONLINE_EVENT_UPDATE_PROMPT,
        prompt_sha256=MEMEYE_ONLINE_EVENT_PROMPT_SHA256,
        current_fact_cap=16,
        prior_fact_cap=24,
        prune_unclaimed_create_members=True,
    )
    online_graph["memeye_source_map"] = source_map
    online_graph["memeye_caption_report"] = session_graph.get(
        "memeye_caption_report", {}
    )
    online_graph["memeye_task"] = task_name
    causal_audit = audit_online_graph(
        session_graph,
        online_graph,
        expected_prompt_sha256=MEMEYE_ONLINE_EVENT_PROMPT_SHA256,
        expected_current_fact_cap=16,
        expected_prior_fact_cap=24,
        expected_prune_unclaimed_create_members=True,
    )
    graph_audit = audit_memeye_graph(
        online_graph,
        legal_dia_ids=legal_ids,
        expected_sessions=int(session_graph.get("stats", {}).get("sessions", 0)),
    )
    if causal_audit["status"] != "pass" or graph_audit["status"] != "pass":
        raise RuntimeError(f"{task_name} rebuilt Online graph audit failed")
    _dump_json(paths["online"], online_graph)
    manifest = _load_json(paths["manifest"])
    manifest["updated_utc"] = _utc_now()
    manifest.setdefault("graphs", {})["online"] = {
        "path": str(paths["online"]),
        "sha256": _sha256_file(paths["online"]),
        "stats": online_graph.get("stats", {}),
        "content_fingerprint": online_graph.get("content_fingerprint"),
    }
    manifest.setdefault("audits", {})["online_causal"] = causal_audit
    manifest.setdefault("audits", {})["online_graph"] = graph_audit
    _dump_json(paths["manifest"], manifest)
    print(f"[{task_name}] rebuilt Online graph", flush=True)
    return manifest


def _rebuild_offline_task(
    task_file: str,
    *,
    data_root: Path,
    graph_root: Path,
) -> dict[str, Any]:
    """Rebuild only the global Offline derivative from a frozen session graph."""

    task_name, _task_path, _data = _task_data(data_root, task_file)
    paths = _graph_paths(task_name, graph_root)
    if not paths["session"].is_file() or not paths["manifest"].is_file():
        raise FileNotFoundError(f"{task_name} lacks a frozen session graph/manifest")
    session_graph = _load_json(paths["session"])
    source_map = session_graph.get("memeye_source_map") or {}
    legal_ids = sorted(
        source_map,
        key=lambda value: tuple(int(x) for x in re.findall(r"\d+", value)),
    )
    if len(session_graph.get("fact_nodes", [])) > 1000:
        offline_graph = add_large_memeye_offline_event_views(
            session_graph, _model_chat, _embed_texts
        )
    else:
        offline_graph = add_offline_event_views(
            session_graph,
            _model_chat,
            _embed_texts,
            prompt_template=MEMEYE_OFFLINE_EVENT_VIEW_PROMPT,
            prompt_sha256=MEMEYE_OFFLINE_EVENT_PROMPT_SHA256,
        )
    offline_graph["memeye_source_map"] = source_map
    offline_graph["memeye_caption_report"] = session_graph.get(
        "memeye_caption_report", {}
    )
    offline_graph["memeye_task"] = task_name
    graph_audit = audit_memeye_graph(
        offline_graph,
        legal_dia_ids=legal_ids,
        expected_sessions=int(session_graph.get("stats", {}).get("sessions", 0)),
    )
    if graph_audit["status"] != "pass":
        raise RuntimeError(f"{task_name} rebuilt Offline graph audit failed")
    _dump_json(paths["offline"], offline_graph)
    manifest = _load_json(paths["manifest"])
    manifest["updated_utc"] = _utc_now()
    manifest.setdefault("graphs", {})["offline"] = {
        "path": str(paths["offline"]),
        "sha256": _sha256_file(paths["offline"]),
        "stats": offline_graph.get("stats", {}),
        "content_fingerprint": offline_graph.get("content_fingerprint"),
    }
    manifest.setdefault("audits", {})["offline"] = graph_audit
    _dump_json(paths["manifest"], manifest)
    print(f"[{task_name}] rebuilt Offline graph", flush=True)
    return manifest


def _render_online_from_selected(
    base: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    policy: str,
) -> tuple[str, dict[str, Any]]:
    """Top-K agnostic version of the frozen Online display policy."""

    original = str(base["retrieved"])
    trace = base["retrieval_trace"]
    selected_ids = [str(value) for value in trace["rerank"]["selected_fact_ids"]]
    if policy == ORIGINAL_FULL:
        return original, {
            "policy": policy,
            "selected_fact_ids": selected_ids,
            "rendered_event_view_ids": list(
                trace.get("assembly", {}).get("rendered_event_view_ids", [])
            ),
            "top_k_preserved": True,
        }

    selected_set = set(selected_ids)
    fact_by_id = {
        str(node["id"]): node for node in graph.get("fact_nodes", [])
    }
    session_by_id = {
        int(node["session_ids"][0]): node
        for node in graph.get("summary_nodes", [])
        if node.get("node_type") == "session_packet" and node.get("session_ids")
    }
    view_by_id = {
        str(node["id"]): node
        for node in graph.get("summary_nodes", [])
        if node.get("node_type") == "event_view" and node.get("id")
    }
    lines: list[str] = []
    rendered_views: list[str] = []
    if policy == BOUNDED_VIEWS_2:
        chars = 0
        for view_id in _candidate_view_ids(base, selected_ids, view_by_id):
            if len(rendered_views) >= 2:
                break
            view = view_by_id[view_id]
            header = (
                f"[Selected cross-session claims | {view.get('view_type', '')} | sessions "
                + ", ".join(str(value) for value in view.get("session_ids", []))
                + "] "
            )
            eligible = _grounded_claims(
                view, selected_set, str(base["question"]), max_claim_chars=260
            )
            chosen: list[str] = []
            for claim in eligible:
                candidate = " ".join(chosen + [str(claim["text"])])
                if chars + len(header) + len(candidate) > 600:
                    continue
                chosen.append(str(claim["text"]))
                if len(chosen) >= 3:
                    break
            if chosen:
                rendered = header + " ".join(chosen)
                lines.append(rendered)
                chars += len(rendered)
                rendered_views.append(view_id)
    elif policy != ROUTE_ONLY:
        raise ValueError(f"unsupported Online policy: {policy}")

    by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    session_order: list[int] = []
    for fact_id in selected_ids:
        fact = fact_by_id[fact_id]
        session_id = int(fact["session_id"])
        if session_id not in by_session:
            session_order.append(session_id)
        by_session[session_id].append(fact)
    for session_id in session_order:
        summary = session_by_id[session_id]
        lines.append(
            f"[Session summary | session {session_id} | {summary.get('timestamp', '')}; ISO date {summary.get('iso_date', '')}] {summary.get('text', '')}"
        )
        for fact in by_session[session_id]:
            sources = ", ".join(str(value) for value in fact.get("source_dia_ids", []))
            lines.append(f"  - [{sources}] {fact.get('text', '')}")
    return "\n".join(lines), {
        "policy": policy,
        "selected_fact_ids": selected_ids,
        "rendered_event_view_ids": rendered_views,
        "top_k_preserved": True,
    }


def _planner(question: str) -> dict[str, Any]:
    result = plan_query(
        question,
        classifier=lambda value: classify_with_gemini(
            value, model=WORK_MODEL, chat_fn=_model_chat
        ),
    )
    return result.to_dict(include_raw=False)


def _retrieve(
    graph: Mapping[str, Any],
    question: str,
    *,
    top_k: int,
    reranker: QwenReranker,
) -> tuple[str, dict[str, Any]]:
    index = GraphMemoryIndex(
        graph,
        summary_top_k=10,
        dense_fact_top_k=20,
        bm25_fact_top_k=20,
        candidate_cap=120,
        rerank_top_k=top_k,
    )
    query_vector = _embed_texts(
        [question], task_type="RETRIEVAL_QUERY", tag="creamem3_memeye_query_embed"
    )[0]
    candidates, trace = index._candidate_pool(question, query_vector)  # type: ignore[attr-defined]
    documents = [str(item["fact"]["text"]) for item in candidates]
    scores = reranker.score(question, documents)
    ranked = sorted(
        range(len(candidates)),
        key=lambda idx: (-float(scores[idx]), idx),
    )
    selected = [candidates[idx] for idx in ranked[: min(top_k, len(candidates))]]
    text, assembly_trace = index._assemble(  # type: ignore[attr-defined]
        selected, trace["top_summaries"]
    )
    trace["rerank"] = {
        "provider": QWEN_RERANK_MODEL,
        "fallback": False,
        "error": None,
        "selected_fact_ids": [str(item["fact"]["id"]) for item in selected],
    }
    trace["assembly"] = assembly_trace
    return text, trace


def _answer(context: str, question: str) -> str:
    prompt = MEMEYE_OPEN_QA_PROMPT.format(context=context, question=question)
    return _model_chat(
        prompt,
        tag="creamem3_memeye_read",
        max_tokens=500,
        json_mode=False,
    ).strip()


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _render_aggregation(data: Mapping[str, Any]) -> str:
    sections = [f"QUESTION TYPE: {data.get('question_type', 'other')}"]
    target = data.get("target") or {}
    if target:
        sections.append(
            "TARGET: " + json.dumps(target, ensure_ascii=False, sort_keys=True)
        )
    candidate = str(data.get("candidate_answer") or "").strip()
    if candidate:
        sections.append(f"CANDIDATE ANSWER: {candidate}")
    for title, key in (
        ("DIRECT EVIDENCE:", "direct_evidence"),
        ("COUNT TABLE:", "count_table"),
        ("TIMELINE:", "timeline"),
        ("COMPARISON TABLE:", "comparison_table"),
        ("ANSWER CONSTRAINTS:", "answer_constraints"),
    ):
        values = data.get(key) or []
        if not isinstance(values, list) or not values:
            continue
        sections.append(title)
        for item in values[:16]:
            text = item if isinstance(item, str) else json.dumps(
                item, ensure_ascii=False, sort_keys=True
            )
            sections.append(f"- {text}")
    return "\n".join(sections) if len(sections) > 1 else ""


def _aggregate_evidence_context(
    context: str, question: str
) -> tuple[str, dict[str, Any]]:
    prompt = MEMEYE_EVIDENCE_AGGREGATOR_PROMPT.replace(
        "{question}", question
    ).replace("{context}", context)
    raw = _model_chat(
        prompt,
        tag="creamem3_memeye_evidence_aggregate",
        max_tokens=2000,
        json_mode=True,
    )
    data = _parse_json_response(raw)
    compact = _render_aggregation(data or {})
    if not data or not compact:
        return context, {
            "failed": True,
            "prompt_sha256": _sha256_text(MEMEYE_EVIDENCE_AGGREGATOR_PROMPT),
            "source_context_chars": len(context),
        }
    return (
        "PRIORITIZED AGGREGATED EVIDENCE:\n"
        "This compact evidence is derived only from the retrieved memory. "
        "Use the candidate answer only when supported by the direct evidence.\n"
        + compact,
        {
            "failed": False,
            "prompt_sha256": _sha256_text(MEMEYE_EVIDENCE_AGGREGATOR_PROMPT),
            "source_context_chars": len(context),
            "packet": data,
        },
    )


def _judge(
    client: OpenRouterJudge,
    question: str,
    gold: str,
    prediction: str,
) -> tuple[float | None, float | None, str, str]:
    prompt = MEMEYE_JUDGE_PROMPT.format(
        question=question,
        ground_truth=gold,
        model_output=prediction,
    )
    text, error = client.complete(prompt)
    raw, mapped, reasoning = parse_judge_response(text)
    return raw, mapped, reasoning, error


def _evaluate_task(
    task_file: str,
    *,
    data_root: Path,
    graph_root: Path,
    results_root: Path,
    judge_cache: Path,
    top_k: int,
    workers: int,
    limit_q: int | None,
    force: bool,
) -> dict[str, Any]:
    task_name, task_path, data = _task_data(data_root, task_file)
    result_path = results_root / task_name / f"result_open_top{top_k}.json"
    if result_path.is_file() and not force:
        print(f"[{task_name}] result exists", flush=True)
        return _load_json(result_path)
    paths = _graph_paths(task_name, graph_root)
    online_graph = _load_json(paths["online"])
    offline_graph = _load_json(paths["offline"])
    qas = list(data.get("human-annotated QAs", []))
    if limit_q is not None:
        qas = qas[:limit_q]
    judge_client = OpenRouterJudge(judge_cache / task_name)
    reranker_client = QwenReranker(DEFAULT_RERANK_CACHE / task_name)

    rows: list[dict[str, Any] | None] = [None] * len(qas)

    def work(index: int, qa: Mapping[str, Any]) -> dict[str, Any]:
        question = str(qa.get("question", ""))
        gold = str(qa.get("answer", ""))
        question_id = str(qa.get("question_id", index))
        x, y = qa_points(qa)
        plan_result = _planner(question)
        plan = plan_result["plan"]

        online_retrieved, online_trace = _retrieve(
            online_graph, question, top_k=top_k, reranker=reranker_client
        )
        online_base = {
            "question": question,
            "retrieved": online_retrieved,
            "retrieval_trace": online_trace,
        }
        online_policy = OPERATION_TO_CONTEXT_POLICY[str(plan["operation"])]
        online_context, online_context_audit = _render_online_from_selected(
            online_base, online_graph, policy=online_policy
        )

        offline_retrieved, offline_trace = _retrieve(
            offline_graph, question, top_k=top_k, reranker=reranker_client
        )
        offline_base = {
            "question": question,
            "retrieved": offline_retrieved,
            "retrieval_trace": offline_trace,
        }
        offline_v3_context, offline_v3_audit = materialize_offline_operation_context(
            str(plan["operation"]), offline_base, offline_graph
        )
        p06_policy = select_context_policy(P06, plan)
        offline_p06_context = materialize_offline_subtype_context(
            p06_policy, offline_base, offline_graph
        )

        contexts = {
            "online_v3": online_context,
            "offline_v3": offline_v3_context,
            "offline_p06": offline_p06_context,
        }
        answer_contexts: dict[str, str] = {}
        aggregations: dict[str, dict[str, Any]] = {}
        for variant, context in contexts.items():
            answer_contexts[variant], aggregations[variant] = (
                _aggregate_evidence_context(context, question)
            )
        predictions = {
            variant: _answer(answer_contexts[variant], question)
            for variant in contexts
        }
        scores: dict[str, dict[str, Any]] = {}
        for variant, prediction in predictions.items():
            raw, mapped, reasoning, judge_error = _judge(
                judge_client, question, gold, prediction
            )
            scores[variant] = {
                "prediction": prediction,
                "bleu1": round(float(bleu1(prediction, gold)), 6),
                "llm_judge_raw": raw,
                "llm_judge": mapped,
                "judge_reasoning": reasoning,
                "judge_error": judge_error,
                "context_sha256": _sha256_text(answer_contexts[variant]),
                "context_chars": len(answer_contexts[variant]),
                "source_context_sha256": _sha256_text(contexts[variant]),
                "source_context_chars": len(contexts[variant]),
            }
        return {
            "task": task_name,
            "question_id": question_id,
            "question": question,
            "gold": gold,
            "x": x,
            "y": y,
            "semantic_plan": plan_result,
            "routing": {
                "online_v3": online_policy,
                "offline_v3": offline_v3_audit.get("context_policy"),
                "offline_p06": p06_policy,
            },
            "retrieval": {
                "online_selected_fact_ids": online_trace["rerank"]["selected_fact_ids"],
                "offline_selected_fact_ids": offline_trace["rerank"]["selected_fact_ids"],
                "online_fallback": bool(online_trace["rerank"].get("fallback")),
                "offline_fallback": bool(offline_trace["rerank"].get("fallback")),
            },
            "context_audit": {
                "online": online_context_audit,
                "offline_v3": offline_v3_audit,
                "offline_p06": {"policy": p06_policy},
            },
            "evidence_aggregation": aggregations,
            "scores": scores,
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(work, index, qa): index
            for index, qa in enumerate(qas)
        }
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            rows[index] = future.result()
            done += 1
            if done % 5 == 0 or done == len(qas):
                print(f"  [{task_name} QA] {done}/{len(qas)}", flush=True)
    complete_rows = [row for row in rows if row is not None]
    if len(complete_rows) != len(qas):
        raise RuntimeError(f"{task_name} has missing result rows")
    summary = summarize_rows(complete_rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "task": task_name,
        "task_file": str(task_path.resolve()),
        "task_file_sha256": _sha256_file(task_path),
        "protocol": {
            "work_model": WORK_MODEL,
            "work_model_thinking_budget": 0,
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBED_MODEL,
            "caption_model": WORK_MODEL,
            "caption_source": "frozen MGTM V8 cache",
            "writer_prompt": "MGTM/Mem0 visual extraction semantics adapted only to CreaMem3 provenance schema",
            "top_k": top_k,
            "candidate_cap": 120,
            "summary_top_k": 10,
            "dense_fact_top_k": 20,
            "bm25_fact_top_k": 20,
            "reader_prompt_sha256": _sha256_text(MEMEYE_OPEN_QA_PROMPT),
            "evidence_aggregator": True,
            "evidence_aggregator_prompt_sha256": _sha256_text(
                MEMEYE_EVIDENCE_AGGREGATOR_PROMPT
            ),
            "judge_prompt_sha256": _sha256_text(MEMEYE_JUDGE_PROMPT),
            "independent_rerank": {
                "online": True,
                "offline": True,
                "offline_p06_reuses_offline_retrieval": True,
            },
            "reranker_model": QWEN_RERANK_MODEL,
            "reranker_instruction_sha256": _sha256_text(
                MEMEYE_OPEN_RETRIEVAL_INSTRUCT
            ),
        },
        "graph_manifest_sha256": _sha256_file(paths["manifest"]),
        "summary": summary,
        "rows": complete_rows,
        "usage": {
            "gemini": json.loads(USAGE.snapshot()),
            "openrouter_judge": judge_client.usage(),
            "qwen_reranker": reranker_client.usage(),
        },
    }
    _dump_json(result_path, result)
    return result


def _mean(values: Sequence[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def pack(subset: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
        scores = [row["scores"][variant] for row in subset]
        return {
            "n": len(subset),
            "bleu1": _mean([score["bleu1"] for score in scores]),
            "llm_judge_raw": _mean([score["llm_judge_raw"] for score in scores]),
            "llm_judge": _mean([score["llm_judge"] for score in scores]),
            "judge_failures": sum(bool(score["judge_error"]) for score in scores),
            "mean_context_chars": _mean([score["context_chars"] for score in scores]),
        }

    output: dict[str, Any] = {"overall": {}, "by_task": {}, "by_x": {}, "by_y": {}}
    for variant in VARIANTS:
        output["overall"][variant] = pack(rows, variant)
    for field, target in (("task", "by_task"), ("x", "by_x"), ("y", "by_y")):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        for group, subset in sorted(groups.items()):
            output[target][group] = {
                variant: pack(subset, variant) for variant in VARIANTS
            }
    output["operation_counts"] = dict(
        Counter(str(row["semantic_plan"]["plan"]["operation"]) for row in rows)
    )
    output["routing_counts"] = {
        variant: dict(Counter(str(row["routing"][variant]) for row in rows))
        for variant in VARIANTS
    }
    output["planner_fallbacks"] = sum(
        row["semantic_plan"]["source"] != "gemini" for row in rows
    )
    return output


def _aggregate(results_root: Path, top_k: int, expected_tasks: int = 8) -> dict[str, Any]:
    paths = sorted(results_root.glob(f"*/result_open_top{top_k}.json"))
    if len(paths) != expected_tasks:
        raise RuntimeError(f"expected {expected_tasks} task results, found {len(paths)}")
    payloads = [_load_json(path) for path in paths]
    rows = [row for payload in payloads for row in payload["rows"]]
    output = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "MemEye 8 tasks / 371 Open QA",
        "protocol": copy.deepcopy(payloads[0]["protocol"]),
        "task_results": [
            {
                "task": payload["task"],
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "n": len(payload["rows"]),
            }
            for path, payload in zip(paths, payloads)
        ],
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    output["result_fingerprint"] = canonical_sha256(
        {
            "protocol": output["protocol"],
            "rows": [
                {
                    "task": row["task"],
                    "question_id": row["question_id"],
                    "scores": row["scores"],
                    "routing": row["routing"],
                    "retrieval": row["retrieval"],
                }
                for row in rows
            ],
        }
    )
    aggregate_path = results_root / f"aggregate_open_top{top_k}.json"
    _dump_json(aggregate_path, output)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "build",
            "rebuild-online",
            "rebuild-offline",
            "eval",
            "all",
            "aggregate",
        ),
        default="all",
    )
    parser.add_argument("--task", default="all", help="task filename/stem or all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--caption-root", type=Path, default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--graph-root", type=Path, default=GRAPH_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--judge-cache", type=Path, default=DEFAULT_JUDGE_CACHE)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-q", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _selected_task_files(args: argparse.Namespace) -> list[str]:
    available = task_files(args.data_root, open_only=True)
    if args.task.strip().lower() == "all":
        return available
    token = args.task.strip()
    matches = [
        name
        for name in available
        if name == token
        or Path(name).stem == token
        or Path(name).stem.removesuffix("_Open") == token
    ]
    if len(matches) != 1:
        raise ValueError(f"task selector {token!r} matched {matches}")
    return matches


def main() -> None:
    args = _parse_args()
    if args.top_k < 1 or args.top_k > 120:
        raise ValueError("top-k must be between 1 and candidate cap 120")
    if args.stage == "aggregate":
        aggregate = _aggregate(args.results_root.resolve(), args.top_k)
        for variant, metrics in aggregate["summary"]["overall"].items():
            print(f"METRIC {variant}_judge_raw={metrics['llm_judge_raw']}")
            print(f"METRIC {variant}_judge_mapped={metrics['llm_judge']}")
        return
    selected = _selected_task_files(args)
    for task_file in selected:
        if args.stage == "rebuild-online":
            _rebuild_online_task(
                task_file,
                data_root=args.data_root.resolve(),
                graph_root=args.graph_root.resolve(),
            )
            continue
        if args.stage == "rebuild-offline":
            _rebuild_offline_task(
                task_file,
                data_root=args.data_root.resolve(),
                graph_root=args.graph_root.resolve(),
            )
            continue
        if args.stage in {"build", "all"}:
            _build_task(
                task_file,
                data_root=args.data_root.resolve(),
                caption_root=args.caption_root.resolve(),
                graph_root=args.graph_root.resolve(),
                workers=args.workers,
                force=args.force,
            )
        if args.stage in {"eval", "all"}:
            result = _evaluate_task(
                task_file,
                data_root=args.data_root.resolve(),
                graph_root=args.graph_root.resolve(),
                results_root=args.results_root.resolve(),
                judge_cache=args.judge_cache.resolve(),
                top_k=args.top_k,
                workers=args.workers,
                limit_q=args.limit_q,
                force=args.force,
            )
            for variant, metrics in result["summary"]["overall"].items():
                print(
                    f"METRIC {task_file}:{variant}_judge_raw={metrics['llm_judge_raw']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build and evaluate independent CreaMem3 graphs on one LoCoMo conversation.

Only the frozen QA/judge prompts and the shared audited Gemini transport are
reused.  The writer, graph schema, fusion, and retrieval policy live entirely
inside the CreaMem3 v2 package.
"""

# Keep protocol environment variables above imports that load gemini_client.
import os


FIXED_CACHE_ROOT = os.environ.get(
    "AGENT_MEMORY_CACHE_ROOT", r"D:\creamem3_cache\v2"
)
FIXED_WRITER_READER_MODEL = "gemini-2.5-flash"
FIXED_JUDGE_MODEL = "gemini-3-flash-preview"
FIXED_EMBED_MODEL = "gemini-embedding-001"

os.environ.setdefault("AGENT_MEMORY_CACHE_ROOT", FIXED_CACHE_ROOT)
os.environ["BACKBONE_MODEL"] = FIXED_WRITER_READER_MODEL
os.environ["JUDGE_MODEL"] = FIXED_JUDGE_MODEL
os.environ["EMBED_MODEL"] = FIXED_EMBED_MODEL
os.environ["BACKBONE_THINKING"] = "off"
os.environ["JUDGE_THINKING"] = "off"
os.environ["GEMINI3_THINKING_LEVEL"] = "minimal"
os.environ["LLM_NO_CHAT_CACHE"] = ""
os.environ.setdefault("AGENT_MEMORY_BASE_CACHE", "")
os.environ["BACKBONE_BASE_URL"] = ""
os.environ["BACKBONE_API_KEY"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVAL_DIR = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gemini_client import (  # noqa: E402
    BACKBONE_MODEL,
    EMBED_MODEL,
    JUDGE_MODEL,
    USAGE,
    chat,
    embed,
    pmap,
)
import judge as frozen_judge  # noqa: E402
import prompts as frozen_prompts  # noqa: E402
from judge import judge_one_detailed  # noqa: E402
from prompts import JUDGE_PROMPT, QA_PROMPT  # noqa: E402

from creamem3.graph_memory_v2 import (  # noqa: E402
    BM25_FACT_TOP_K,
    CANDIDATE_CAP,
    DENSE_FACT_TOP_K,
    GRAPH_SCHEMA_VERSION,
    RERANK_TOP_K,
    SUMMARY_TOP_K,
    GraphMemoryError,
    GraphMemoryIndex,
    add_offline_event_views,
    audit_graph,
    build_session_graph,
    canonical_sha256,
)
from creamem3.locomo_all import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    build_manifest,
    file_sha256,
    load_locomo_conversation,
)
from creamem3.online_memory_v3 import (  # noqa: E402
    add_online_event_views,
    audit_online_graph,
)
from creamem3.prompts_v2 import (  # noqa: E402
    PROMPT_SCHEMA_VERSION,
    PROMPT_SHA256,
)


PROTOCOL_SCHEMA_VERSION = "creamem3.locomo_protocol.v3"
RESULT_SCHEMA_VERSION = "creamem3.eval_result.v3"
ALLOWED_VARIANTS = ("session", "online", "offline")
RESULTS_DIR = SCRIPT_DIR / "results_v2"
CURRENT_SAMPLE_ID = "conv-26"
CURRENT_SAMPLE_INDEX = 0
EMBEDDING_DIM = 768
EXPERIMENT_SEED = 20260805
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _usage_snapshot() -> dict[str, Any]:
    return json.loads(USAGE.snapshot())


def _numeric_delta(after: Any, before: Any) -> Any:
    if isinstance(after, Mapping) or isinstance(before, Mapping):
        after_map = after if isinstance(after, Mapping) else {}
        before_map = before if isinstance(before, Mapping) else {}
        return {
            str(key): _numeric_delta(after_map.get(key), before_map.get(key))
            for key in sorted(set(after_map) | set(before_map))
        }
    if isinstance(after, (int, float)) or isinstance(before, (int, float)):
        return (after or 0) - (before or 0)
    return after


def _usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return _numeric_delta(after, before)


def _project_protocol(
    protocol: Mapping[str, Any], *, graph_only: bool
) -> dict[str, Any]:
    graph_keys = (
        "sample_id",
        "sample_index",
        "writer_model",
        "writer_thinking_budget",
        "embedding_model",
        "embedding_dimension",
        "write_unit",
        "writer_prompt_family",
        "mem0_writer_prompt_used",
        "input_scope",
        "memory_prompt_schema_version",
        "memory_prompt_sha256",
        "graph_memory_source_sha256",
        "online_memory_source_sha256",
    )
    eval_keys = (
        "reader_model",
        "reranker_model",
        "reader_thinking_budget",
        "reranker_thinking_budget",
        "judge_model",
        "judge_thinking_level",
        "summary_top_k",
        "dense_fact_top_k",
        "bm25_fact_top_k",
        "candidate_cap",
        "rerank_top_k",
        "context_character_cap",
        "qa_prompt_sha256",
        "judge_prompt_sha256",
        "qa_prompt_source_sha256",
        "judge_source_sha256",
    )
    keys = graph_keys if graph_only else (*graph_keys, *eval_keys)
    return {key: protocol.get(key) for key in keys}


def _graph_content_fingerprint(
    graph: Mapping[str, Any], *, base_graph: Mapping[str, Any] | None = None
) -> str:
    variant = str(graph.get("graph_variant", ""))
    summaries = list(graph.get("summary_nodes", []))
    facts = list(graph.get("fact_nodes", []))
    edges = list(graph.get("edges", []))
    if variant == "session":
        return canonical_sha256(
            {
                "summaries": [
                    {key: value for key, value in node.items() if key != "embedding"}
                    for node in summaries
                ],
                "facts": [
                    {key: value for key, value in node.items() if key != "embedding"}
                    for node in facts
                ],
                "edges": edges,
            }
        )
    if variant == "offline":
        if base_graph is None:
            raise GraphMemoryError("offline fingerprint requires its session graph")
        views = [node for node in summaries if node.get("node_type") == "event_view"]
        return canonical_sha256(
            {
                "base": base_graph.get("content_fingerprint"),
                "views": [
                    {key: value for key, value in view.items() if key != "embedding"}
                    for view in views
                ],
            }
        )
    # The online fingerprint additionally binds the causal log and is checked
    # by audit_online_graph, so it is not duplicated here.
    return str(graph.get("content_fingerprint", ""))


def _model_chat(
    prompt: str,
    *,
    tag: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    json_mode: bool = False,
    **_: Any,
) -> str:
    del temperature
    return chat(
        prompt,
        model=FIXED_WRITER_READER_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        thinking_budget=0,
        tag=tag,
        json_mode=json_mode,
        use_cache=True,
    )


def _embed_texts(
    texts: Sequence[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    tag: str = "creamem3_v2_embed",
    **_: Any,
) -> list[list[float]]:
    return embed(
        list(texts),
        model=FIXED_EMBED_MODEL,
        task_type=task_type,
        tag=tag,
        dim=EMBEDDING_DIM,
        use_cache=True,
    )


def _fixed_protocol() -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "sample_id": CURRENT_SAMPLE_ID,
        "sample_index": CURRENT_SAMPLE_INDEX,
        "writer_model": FIXED_WRITER_READER_MODEL,
        "reader_model": FIXED_WRITER_READER_MODEL,
        "reranker_model": FIXED_WRITER_READER_MODEL,
        "writer_thinking_budget": 0,
        "reader_thinking_budget": 0,
        "reranker_thinking_budget": 0,
        "judge_model": FIXED_JUDGE_MODEL,
        "judge_thinking_level": "minimal",
        "embedding_model": FIXED_EMBED_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "write_unit": "one_complete_physical_session",
        "writer_prompt_family": "short_mgtm_style_session_fact_extraction",
        "mem0_writer_prompt_used": False,
        "input_scope": "dialogue_text_only_no_images_or_captions",
        "summary_top_k": SUMMARY_TOP_K,
        "dense_fact_top_k": DENSE_FACT_TOP_K,
        "bm25_fact_top_k": BM25_FACT_TOP_K,
        "candidate_cap": CANDIDATE_CAP,
        "rerank_top_k": RERANK_TOP_K,
        "context_character_cap": None,
        "cache_root": FIXED_CACHE_ROOT,
        "chat_cache_enabled": True,
        "seed": EXPERIMENT_SEED,
        "qa_prompt_sha256": _sha256_text(QA_PROMPT),
        "judge_prompt_sha256": _sha256_text(JUDGE_PROMPT),
        "qa_prompt_source_path": str((EVAL_DIR / "prompts.py").resolve()),
        "qa_prompt_source_sha256": file_sha256(EVAL_DIR / "prompts.py"),
        "judge_source_path": str((EVAL_DIR / "judge.py").resolve()),
        "judge_source_sha256": file_sha256(EVAL_DIR / "judge.py"),
        "memory_prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "memory_prompt_sha256": dict(PROMPT_SHA256),
        "graph_memory_source_sha256": file_sha256(
            SCRIPT_DIR / "graph_memory_v2.py"
        ),
        "online_memory_source_sha256": file_sha256(
            SCRIPT_DIR / "online_memory_v3.py"
        ),
    }


def _assert_imported_protocol() -> None:
    actual = (BACKBONE_MODEL, JUDGE_MODEL, EMBED_MODEL)
    expected = (
        FIXED_WRITER_READER_MODEL,
        FIXED_JUDGE_MODEL,
        FIXED_EMBED_MODEL,
    )
    if actual != expected:
        raise RuntimeError(
            f"unexpected imported model protocol: actual={actual!r}, expected={expected!r}"
        )
    imported_judge = Path(frozen_judge.__file__).resolve()
    imported_prompts = Path(frozen_prompts.__file__).resolve()
    if imported_judge != (EVAL_DIR / "judge.py").resolve():
        raise RuntimeError(f"unexpected judge module: {imported_judge}")
    if imported_prompts != (EVAL_DIR / "prompts.py").resolve():
        raise RuntimeError(f"unexpected prompt module: {imported_prompts}")


def _parse_variants(raw: str) -> list[str]:
    output: list[str] = []
    for token in raw.split(","):
        variant = token.strip()
        if not variant:
            continue
        if variant not in ALLOWED_VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; choose from {ALLOWED_VARIANTS}"
            )
        if variant not in output:
            output.append(variant)
    if not output:
        raise ValueError("no variants selected")
    return output


def _validate_tag(tag: str) -> str:
    if not TAG_RE.fullmatch(tag):
        raise ValueError("tag contains unsupported characters")
    return tag


def _graph_path(variant: str) -> Path:
    return RESULTS_DIR / f"graph_{variant}.json"


def _result_path(variant: str, tag: str) -> Path:
    return RESULTS_DIR / f"{variant}_{CURRENT_SAMPLE_ID}_{tag}.json"


def _aggregate_path(tag: str) -> Path:
    return RESULTS_DIR / f"aggregate_{tag}.json"


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GraphMemoryError(f"artifact is not a JSON object: {path}")
    return value


def _source_dependency_audit() -> dict[str, Any]:
    files = [
        SCRIPT_DIR / "prompts_v2.py",
        SCRIPT_DIR / "graph_memory_v2.py",
        SCRIPT_DIR / "online_memory_v3.py",
        SCRIPT_DIR / "locomo_all.py",
        SCRIPT_DIR / "run_locomo10_v3.py",
        Path(__file__).resolve(),
    ]
    forbidden_import = re.compile(
        r"^\s*(?:from|import)\s+[^\n]*chronohg", re.IGNORECASE | re.MULTILINE
    )
    matches: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if forbidden_import.search(text):
            matches.append(str(path))
    return {
        "status": "pass" if not matches else "fail",
        "checked_files": [str(path) for path in files],
        "forbidden_import_matches": matches,
        "policy": "CreaMem3 v2 has no imports from the historical comparison architecture",
    }


def _artifact_manifest(
    loaded: Mapping[str, Any],
    *,
    kind: str,
    usage: Mapping[str, Any],
    wall_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": "creamem3.artifact_manifest.v3",
        "artifact_kind": kind,
        "created_utc": _utc_now(),
        "dataset": build_manifest(loaded),
        "protocol": _fixed_protocol(),
        "usage": dict(usage),
        "wall_s": wall_s,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "architecture_dependency_audit": _source_dependency_audit(),
    }


def _validate_graph_artifact(
    graph: Mapping[str, Any],
    loaded: Mapping[str, Any],
    variant: str,
    *,
    session_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise GraphMemoryError(f"{variant} graph has unsupported schema")
    if graph.get("graph_variant") != variant:
        raise GraphMemoryError(f"graph variant mismatch for {variant}")
    manifest = graph.get("runner_manifest")
    if not isinstance(manifest, Mapping):
        raise GraphMemoryError(f"{variant} graph lacks runner_manifest")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("hashes") != loaded.get("hashes"):
        raise GraphMemoryError(f"{variant} graph dataset hash mismatch")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise GraphMemoryError(f"{variant} graph lacks protocol")
    current = _fixed_protocol()
    if _project_protocol(protocol, graph_only=True) != _project_protocol(
        current, graph_only=True
    ):
        raise GraphMemoryError(f"{variant} graph protocol fingerprint mismatch")
    if variant in {"offline", "online"}:
        if session_graph is None:
            session_graph = _load_json(_graph_path("session"))
        base_path = _graph_path("session")
        if manifest.get("base_graph_sha256") != file_sha256(base_path):
            raise GraphMemoryError(f"{variant} graph base file hash mismatch")
        if variant == "offline" and graph.get("content_fingerprint") != _graph_content_fingerprint(
            graph, base_graph=session_graph
        ):
            raise GraphMemoryError("offline graph content fingerprint mismatch")
    elif graph.get("content_fingerprint") != _graph_content_fingerprint(graph):
        raise GraphMemoryError("session graph content fingerprint mismatch")
    audit = audit_graph(
        graph,
        legal_dia_ids=loaded["dia_id_map"],
        expected_sessions=int(loaded["stats"]["sessions"]),
    )
    if audit["status"] != "pass":
        failures = [item["name"] for item in audit["checks"] if not item["passed"]]
        raise GraphMemoryError(
            f"{variant} graph audit failed: {', '.join(failures)}"
        )
    if variant == "online":
        causal_audit = audit_online_graph(session_graph, graph)
        if causal_audit["status"] != "pass":
            failures = [
                item["name"]
                for item in causal_audit["checks"]
                if not item["passed"]
            ]
            raise GraphMemoryError(
                "online causal audit failed: " + ", ".join(failures)
            )
        audit["causal_audit"] = causal_audit
    dependency_audit = manifest.get("architecture_dependency_audit", {})
    if dependency_audit.get("status") != "pass":
        raise GraphMemoryError("architecture dependency audit failed")
    return audit


def build_stage(
    loaded: Mapping[str, Any],
    variants: Sequence[str],
    *,
    workers: int,
    force: bool,
) -> dict[str, dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = _graph_path("session")
    session_rebuilt = force or not session_path.exists()
    if session_rebuilt:
        before = _usage_snapshot()
        started = time.time()
        print(
            f"[build-v3] writing {loaded['stats']['sessions']} complete "
            "text-only sessions",
            flush=True,
        )
        session_graph = build_session_graph(
            loaded["records"], _model_chat, _embed_texts, workers=workers
        )
        after = _usage_snapshot()
        session_graph["runner_manifest"] = _artifact_manifest(
            loaded,
            kind="typed_graph_session_packets",
            usage=_usage_delta(before, after),
            wall_s=time.time() - started,
        )
        session_graph["runner_manifest"]["graph_audit"] = audit_graph(
            session_graph,
            legal_dia_ids=loaded["dia_id_map"],
            expected_sessions=int(loaded["stats"]["sessions"]),
        )
        _validate_graph_artifact(session_graph, loaded, "session")
        _dump_json(session_path, session_graph)
        print(
            f"[build-v2] saved {session_path} facts={len(session_graph['fact_nodes'])}",
            flush=True,
        )
    else:
        session_graph = _load_json(session_path)
        _validate_graph_artifact(session_graph, loaded, "session")
        print(f"[build-v2] reuse {session_path}", flush=True)

    graphs = {"session": session_graph}
    if "online" in variants:
        online_path = _graph_path("online")
        online_rebuilt = force or session_rebuilt or not online_path.exists()
        if online_rebuilt:
            before = _usage_snapshot()
            started = time.time()
            print(
                "[build-v3] deriving strict causal online event views",
                flush=True,
            )
            online_graph = add_online_event_views(
                session_graph, _model_chat, _embed_texts
            )
            after = _usage_snapshot()
            causal_audit = audit_online_graph(session_graph, online_graph)
            online_graph["online_causal_audit"] = causal_audit
            online_graph["runner_manifest"] = _artifact_manifest(
                loaded,
                kind="typed_graph_strict_causal_online_event_views",
                usage=_usage_delta(before, after),
                wall_s=time.time() - started,
            )
            online_graph["runner_manifest"]["base_graph_path"] = str(
                session_path.resolve()
            )
            online_graph["runner_manifest"]["base_graph_sha256"] = file_sha256(
                session_path
            )
            online_graph["runner_manifest"]["graph_audit"] = audit_graph(
                online_graph,
                legal_dia_ids=loaded["dia_id_map"],
                expected_sessions=int(loaded["stats"]["sessions"]),
            )
            online_graph["runner_manifest"]["online_causal_audit"] = causal_audit
            _validate_graph_artifact(
                online_graph, loaded, "online", session_graph=session_graph
            )
            _dump_json(online_path, online_graph)
            print(
                f"[build-v3] saved {online_path} "
                f"views={online_graph['stats']['event_view_nodes']}",
                flush=True,
            )
        else:
            online_graph = _load_json(online_path)
            _validate_graph_artifact(
                online_graph, loaded, "online", session_graph=session_graph
            )
            print(f"[build-v3] reuse {online_path}", flush=True)
        graphs["online"] = online_graph
    if "offline" in variants:
        offline_path = _graph_path("offline")
        offline_rebuilt = force or session_rebuilt or not offline_path.exists()
        if offline_rebuilt:
            before = _usage_snapshot()
            started = time.time()
            print("[build-v2] deriving non-destructive offline event views", flush=True)
            offline_graph = add_offline_event_views(
                session_graph, _model_chat, _embed_texts
            )
            after = _usage_snapshot()
            offline_graph["runner_manifest"] = _artifact_manifest(
                loaded,
                kind="typed_graph_offline_event_views",
                usage=_usage_delta(before, after),
                wall_s=time.time() - started,
            )
            offline_graph["runner_manifest"]["base_graph_path"] = str(
                session_path.resolve()
            )
            offline_graph["runner_manifest"]["base_graph_sha256"] = file_sha256(
                session_path
            )
            offline_graph["runner_manifest"]["graph_audit"] = audit_graph(
                offline_graph,
                legal_dia_ids=loaded["dia_id_map"],
                expected_sessions=int(loaded["stats"]["sessions"]),
            )
            _validate_graph_artifact(offline_graph, loaded, "offline")
            _dump_json(offline_path, offline_graph)
            print(
                f"[build-v2] saved {offline_path} views={offline_graph['stats']['event_view_nodes']}",
                flush=True,
            )
        else:
            offline_graph = _load_json(offline_path)
            _validate_graph_artifact(offline_graph, loaded, "offline")
            print(f"[build-v2] reuse {offline_path}", flush=True)
        graphs["offline"] = offline_graph
    return {variant: graphs[variant] for variant in variants}


def _load_graphs(
    loaded: Mapping[str, Any], variants: Sequence[str]
) -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    for variant in variants:
        path = _graph_path(variant)
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run --stage build first")
        graph = _load_json(path)
        _validate_graph_artifact(graph, loaded, variant)
        graphs[variant] = graph
    return graphs


def _qa_selection(
    loaded: Mapping[str, Any], limit_q: int | None
) -> list[dict[str, Any]]:
    qa = list(loaded["qa"])
    if limit_q is not None:
        qa = qa[:limit_q]
    return qa


def _answer_one(item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item["qa"])
    row.update(
        {
            "sample_id": CURRENT_SAMPLE_ID,
            "conv_id": CURRENT_SAMPLE_INDEX,
            "retrieved": item.get("retrieved", ""),
            "retrieved_chars": len(item.get("retrieved", "")),
            "retrieval_trace": item.get("retrieval_trace", {}),
            "t_search": float(item.get("t_search", 0.0)),
            "retrieval_error": item.get("retrieval_error"),
            "response": "",
            "qa_error": None,
        }
    )
    if row["retrieval_error"]:
        row["qa_error"] = "reader skipped because retrieval failed"
        return row
    prompt = QA_PROMPT.format(
        retrieved_texts=row["retrieved"], question=row["question"]
    )
    try:
        response = chat(
            prompt,
            model=FIXED_WRITER_READER_MODEL,
            temperature=0.0,
            max_tokens=8192,
            thinking_budget=0,
            tag="read",
            use_cache=True,
        )
        if response:
            row["response"] = response
        else:
            row["qa_error"] = "empty reader response"
    except Exception as exc:
        row["qa_error"] = f"{type(exc).__name__}: {str(exc)[-400:]}"
    return row


def _judge_one_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "correct": 0,
            "judge_raw": "",
            "judge_parse_error": False,
            "judge_error": None,
            "judge_skipped": False,
        }
    )
    if output.get("qa_error") or not output.get("response"):
        output["judge_skipped"] = True
        output["judge_parse_error"] = True
        output["judge_error"] = "judge skipped because reader response failed"
        return output
    try:
        verdict, raw, parsed = judge_one_detailed(
            output["question"], output["answer"], output["response"]
        )
        output["correct"] = int(verdict)
        output["judge_raw"] = raw
        output["judge_parse_error"] = not parsed
        if not parsed:
            output["judge_error"] = "judge response lacked one unambiguous tag"
    except Exception as exc:
        output["judge_parse_error"] = True
        output["judge_error"] = f"{type(exc).__name__}: {str(exc)[-400:]}"
    return output


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(int(row.get("correct", 0)) for row in rows)
    valid = [
        row
        for row in rows
        if not row.get("retrieval_error")
        and not row.get("qa_error")
        and not row.get("judge_parse_error")
    ]
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category_name"])].append(row)
    evidence_rows = [row for row in rows if row.get("evidence_dia_ids")]
    retrieved_ids = {
        id(row): set(re.findall(r"\bD\d+:\d+\b", str(row.get("retrieved", ""))))
        for row in evidence_rows
    }
    evidence_total = sum(len(row["evidence_dia_ids"]) for row in evidence_rows)
    evidence_hits = sum(
        source_id in retrieved_ids[id(row)]
        for row in evidence_rows
        for source_id in row["evidence_dia_ids"]
    )
    evidence_full = sum(
        all(source_id in retrieved_ids[id(row)] for source_id in row["evidence_dia_ids"])
        for row in evidence_rows
    )
    cross_session_rows = [
        row
        for row in evidence_rows
        if len({source_id.split(":", 1)[0] for source_id in row["evidence_dia_ids"]}) > 1
    ]
    rerank_traces = [row.get("retrieval_trace", {}).get("rerank", {}) for row in rows]
    summaries: dict[str, Any] = {
        "n": n,
        "correct": correct,
        "acc_overall": 100.0 * correct / n if n else 0.0,
        "protocol_valid_rows": len(valid),
        "protocol_failures": n - len(valid),
        "retrieval_errors": sum(bool(row.get("retrieval_error")) for row in rows),
        "qa_errors": sum(bool(row.get("qa_error")) for row in rows),
        "judge_parse_errors": sum(bool(row.get("judge_parse_error")) for row in rows),
        "judge_skipped": sum(bool(row.get("judge_skipped")) for row in rows),
        "acc_protocol_valid_only": (
            100.0 * sum(int(row["correct"]) for row in valid) / len(valid)
            if valid
            else None
        ),
        "mean_retrieved_chars": (
            sum(int(row.get("retrieved_chars", 0)) for row in rows) / n if n else 0.0
        ),
        "max_retrieved_chars": max(
            (int(row.get("retrieved_chars", 0)) for row in rows), default=0
        ),
        "mean_search_s": (
            sum(float(row.get("t_search", 0.0)) for row in rows) / n if n else 0.0
        ),
        "rerank_fallbacks": sum(bool(trace.get("fallback")) for trace in rerank_traces),
        "mean_candidate_union": (
            sum(
                int(row.get("retrieval_trace", {}).get("union_before_cap", 0))
                for row in rows
            )
            / n
            if n
            else 0.0
        ),
        "mean_selected_facts": (
            sum(
                len(
                    row.get("retrieval_trace", {})
                    .get("rerank", {})
                    .get("selected_fact_ids", [])
                )
                for row in rows
            )
            / n
            if n
            else 0.0
        ),
        "gold_evidence_ids": evidence_total,
        "gold_evidence_id_hits": evidence_hits,
        "gold_evidence_id_recall": evidence_hits / evidence_total if evidence_total else None,
        "gold_evidence_questions": len(evidence_rows),
        "gold_evidence_questions_full_hit": evidence_full,
        "qa_with_unresolved_evidence": sum(
            bool(row.get("evidence_unresolved")) for row in rows
        ),
        "unresolved_evidence_items": sum(
            len(row.get("evidence_unresolved", [])) for row in rows
        ),
        "gold_evidence_question_full_recall": (
            evidence_full / len(evidence_rows) if evidence_rows else None
        ),
        "cross_session_evidence_questions": len(cross_session_rows),
        "acc_cross_session_evidence": (
            100.0
            * sum(int(row.get("correct", 0)) for row in cross_session_rows)
            / len(cross_session_rows)
            if cross_session_rows
            else None
        ),
        "cross_session_evidence_question_full_recall": (
            sum(
                all(
                    source_id in retrieved_ids[id(row)]
                    for source_id in row["evidence_dia_ids"]
                )
                for row in cross_session_rows
            )
            / len(cross_session_rows)
            if cross_session_rows
            else None
        ),
    }
    for category, category_rows in sorted(by_category.items()):
        summaries[f"n_{category}"] = len(category_rows)
        summaries[f"acc_{category}"] = (
            100.0
            * sum(int(row.get("correct", 0)) for row in category_rows)
            / len(category_rows)
        )
    return summaries


def _prepare_retrievals(
    graph: Mapping[str, Any],
    qa: Sequence[Mapping[str, Any]],
    question_vectors: Sequence[Sequence[float]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    index = GraphMemoryIndex(graph)
    tasks = [
        {"qa": question, "query_vector": vector}
        for question, vector in zip(qa, question_vectors)
    ]

    def retrieve_one(item: Mapping[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            context, trace = index.search(
                str(item["qa"]["question"]),
                item["query_vector"],
                _model_chat,
            )
            return {
                "qa": item["qa"],
                "retrieved": context,
                "retrieval_trace": trace,
                "t_search": time.time() - started,
                "retrieval_error": None,
            }
        except Exception as exc:
            return {
                "qa": item["qa"],
                "retrieved": "",
                "retrieval_trace": {},
                "t_search": time.time() - started,
                "retrieval_error": f"{type(exc).__name__}: {str(exc)[-500:]}",
            }

    return pmap(retrieve_one, tasks, workers=workers, desc="retrieve-v2")


def _evaluate_variant(
    variant: str,
    graph: Mapping[str, Any],
    loaded: Mapping[str, Any],
    qa: Sequence[Mapping[str, Any]],
    question_vectors: Sequence[Sequence[float]],
    question_embed_usage: Mapping[str, Any],
    *,
    tag: str,
    workers: int,
    force: bool,
    limit_q: int | None,
) -> tuple[dict[str, Any], Path]:
    path = _result_path(variant, tag)
    if path.exists() and not force:
        raise FileExistsError(f"result exists: {path}; use another tag or --force")
    before = _usage_snapshot()
    started = time.time()
    pending = _prepare_retrievals(
        graph, qa, question_vectors, workers=workers
    )
    rows = pmap(_answer_one, pending, workers=workers, desc=f"answer-{variant}")
    rows = pmap(_judge_one_row, rows, workers=workers, desc=f"judge-{variant}")
    after = _usage_snapshot()
    summary = _summarize_rows(rows)
    summary["wall_s"] = time.time() - started
    graph_path = _graph_path(variant)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "system": f"creamem3_v2_{variant}",
        "dataset": f"locomo_{CURRENT_SAMPLE_ID}",
        "variant": variant,
        "tag": tag,
        "summary": summary,
        "rows": rows,
        "usage": {
            "question_embeddings": dict(question_embed_usage),
            "variant_eval": _usage_delta(before, after),
            "process_snapshot_at_save": after,
        },
        "manifest": {
            "created_utc": _utc_now(),
            "protocol": _fixed_protocol(),
            "dataset_hashes": dict(loaded["hashes"]),
            "qa_selection": {
                "available": len(loaded["qa"]),
                "selected": len(qa),
                "limit_q": limit_q,
                "qa_ids": [row["qa_id"] for row in qa],
                "selection_sha256": canonical_sha256(list(qa)),
            },
            "graph_path": str(graph_path.resolve()),
            "graph_sha256": file_sha256(graph_path),
            "graph_content_fingerprint": graph.get("content_fingerprint"),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "architecture_dependency_audit": _source_dependency_audit(),
        },
    }
    if int(summary["protocol_failures"]) != 0:
        failed_path = path.with_name(path.stem + ".failed.json")
        _dump_json(failed_path, payload)
        raise GraphMemoryError(
            f"{variant} evaluation has {summary['protocol_failures']} protocol "
            f"failures; diagnostic saved to {failed_path}"
        )
    _validate_result_artifact(
        payload,
        variant=variant,
        tag=tag,
        loaded=loaded,
        qa=qa,
        graph=graph,
    )
    _dump_json(path, payload)
    print(
        f"[eval-v2] {variant}: {summary['correct']}/{summary['n']} "
        f"accuracy={summary['acc_overall']:.2f}% evidence_recall="
        f"{summary['gold_evidence_id_recall']:.4f} fallbacks="
        f"{summary['rerank_fallbacks']}",
        flush=True,
    )
    return payload, path


def _validate_result_artifact(
    payload: Mapping[str, Any],
    *,
    variant: str,
    tag: str,
    loaded: Mapping[str, Any],
    qa: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise GraphMemoryError(f"{variant} result has unsupported schema")
    if payload.get("variant") != variant:
        raise GraphMemoryError(f"{variant} result variant mismatch")
    if payload.get("tag") != tag:
        raise GraphMemoryError(f"{variant} result tag mismatch")
    if payload.get("dataset") != f"locomo_{CURRENT_SAMPLE_ID}":
        raise GraphMemoryError(f"{variant} result dataset label mismatch")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise GraphMemoryError(f"{variant} result lacks manifest")
    if manifest.get("dataset_hashes") != loaded.get("hashes"):
        raise GraphMemoryError(f"{variant} result dataset hash mismatch")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or _project_protocol(
        protocol, graph_only=False
    ) != _project_protocol(_fixed_protocol(), graph_only=False):
        raise GraphMemoryError(f"{variant} result protocol fingerprint mismatch")
    expected_ids = [str(row["qa_id"]) for row in qa]
    selection = manifest.get("qa_selection", {})
    rows = payload.get("rows")
    if (
        selection.get("qa_ids") != expected_ids
        or selection.get("selection_sha256") != canonical_sha256(list(qa))
        or not isinstance(rows, list)
    ):
        raise GraphMemoryError(f"{variant} result QA selection mismatch")
    actual_ids = [str(row.get("qa_id", "")) for row in rows]
    if actual_ids != expected_ids:
        raise GraphMemoryError(f"{variant} result row order mismatch")
    immutable_keys = (
        "qa_id",
        "source_index",
        "question",
        "question_occurrence",
        "answer",
        "category",
        "category_name",
        "evidence",
        "evidence_dia_ids",
        "evidence_repairs",
        "evidence_unresolved",
    )
    for result_row, source_row in zip(rows, qa):
        if any(result_row.get(key) != source_row.get(key) for key in immutable_keys):
            raise GraphMemoryError(
                f"{variant} result immutable QA fields mismatch at "
                f"{source_row['qa_id']}"
            )
        if (
            result_row.get("sample_id") != CURRENT_SAMPLE_ID
            or result_row.get("conv_id") != CURRENT_SAMPLE_INDEX
        ):
            raise GraphMemoryError(
                f"{variant} result sample binding mismatch at {source_row['qa_id']}"
            )
        if result_row.get("correct") not in {0, 1}:
            raise GraphMemoryError(
                f"{variant} result has a non-binary verdict at {source_row['qa_id']}"
            )
    graph_path = _graph_path(variant)
    if (
        manifest.get("graph_sha256") != file_sha256(graph_path)
        or manifest.get("graph_content_fingerprint")
        != graph.get("content_fingerprint")
    ):
        raise GraphMemoryError(f"{variant} result graph binding mismatch")
    recomputed = _summarize_rows(rows)
    stored_summary = payload.get("summary")
    if not isinstance(stored_summary, Mapping) or any(
        stored_summary.get(key) != value for key, value in recomputed.items()
    ):
        raise GraphMemoryError(f"{variant} result summary mismatch")
    if int(recomputed["protocol_failures"]) != 0:
        raise GraphMemoryError(f"{variant} result contains protocol failures")


def eval_stage(
    loaded: Mapping[str, Any],
    graphs: Mapping[str, Mapping[str, Any]],
    variants: Sequence[str],
    *,
    tag: str,
    workers: int,
    force: bool,
    limit_q: int | None,
) -> dict[str, Any]:
    qa = _qa_selection(loaded, limit_q)
    results: dict[str, Any] = {}
    paths: dict[str, str] = {}
    missing: list[str] = []
    for variant in variants:
        path = _result_path(variant, tag)
        if path.exists() and not force:
            result = _load_json(path)
            _validate_result_artifact(
                result,
                variant=variant,
                tag=tag,
                loaded=loaded,
                qa=qa,
                graph=graphs[variant],
            )
            results[variant] = result
            paths[variant] = str(path.resolve())
            print(f"[eval-v3] reuse {path}", flush=True)
        else:
            missing.append(variant)

    question_vectors: list[list[float]] = []
    question_embed_usage: dict[str, Any] = {}
    if missing:
        before_embed = _usage_snapshot()
        question_vectors = _embed_texts(
            [str(row["question"]) for row in qa],
            task_type="RETRIEVAL_QUERY",
            tag="creamem3_v2_embed_question",
        )
        after_embed = _usage_snapshot()
        question_embed_usage = _usage_delta(before_embed, after_embed)
    for variant in missing:
        result, path = _evaluate_variant(
            variant,
            graphs[variant],
            loaded,
            qa,
            question_vectors,
            question_embed_usage,
            tag=tag,
            workers=workers,
            force=force,
            limit_q=limit_q,
        )
        results[variant] = result
        paths[variant] = str(path.resolve())
    aggregate = {
        "schema_version": "creamem3.aggregate.v3",
        "tag": tag,
        "created_utc": _utc_now(),
        "protocol": _fixed_protocol(),
        "dataset_hashes": dict(loaded["hashes"]),
        "variants": {
            variant: {
                "result_path": paths[variant],
                "result_sha256": file_sha256(paths[variant]),
                "summary": results[variant]["summary"],
            }
            for variant in variants
        },
    }
    aggregate_path = _aggregate_path(tag)
    _dump_json(aggregate_path, aggregate)
    best = max(
        (float(results[variant]["summary"]["acc_overall"]) for variant in variants),
        default=0.0,
    )
    for variant in variants:
        print(
            f"METRIC {variant}_accuracy="
            f"{float(results[variant]['summary']['acc_overall']):.6f}",
            flush=True,
        )
    print(f"METRIC best_accuracy={best:.6f}", flush=True)
    return aggregate


def audit_stage(
    loaded: Mapping[str, Any], variants: Sequence[str]
) -> dict[str, Any]:
    graphs = _load_graphs(loaded, variants)
    variant_audits: dict[str, Any] = {}
    for variant, graph in graphs.items():
        graph_audit = audit_graph(
            graph,
            legal_dia_ids=loaded["dia_id_map"],
            expected_sessions=int(loaded["stats"]["sessions"]),
        )
        if variant == "online":
            base_graph = graphs.get("session") or _load_json(_graph_path("session"))
            causal_audit = audit_online_graph(base_graph, graph)
            graph_audit["causal_audit"] = causal_audit
            if causal_audit["status"] != "pass":
                graph_audit["status"] = "fail"
        variant_audits[variant] = graph_audit
    payload = {
        "schema_version": "creamem3.audit.v3",
        "created_utc": _utc_now(),
        "architecture_dependency_audit": _source_dependency_audit(),
        "variants": variant_audits,
    }
    path = RESULTS_DIR / "audit_graphs.json"
    _dump_json(path, payload)
    print(f"[audit-v2] saved {path}", flush=True)
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    global CURRENT_SAMPLE_ID, CURRENT_SAMPLE_INDEX, RESULTS_DIR
    parser = argparse.ArgumentParser(
        description="Run CreaMem3 on one validated LoCoMo conversation."
    )
    parser.add_argument(
        "--stage", choices=("build", "eval", "all", "audit"), default="all"
    )
    parser.add_argument(
        "--variants", default=",".join(ALLOWED_VARIANTS)
    )
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit-q", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--conversation", default="conv-26")
    parser.add_argument(
        "--results-root", type=Path, default=SCRIPT_DIR / "results_locomo10_v3"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _assert_imported_protocol()
    variants = _parse_variants(args.variants)
    tag = _validate_tag(args.tag)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit_q is not None and args.limit_q < 1:
        parser.error("--limit-q must be positive")
    loaded = load_locomo_conversation(args.dataset, args.conversation)
    CURRENT_SAMPLE_ID = str(loaded["sample_id"])
    CURRENT_SAMPLE_INDEX = int(loaded["sample_index"])
    RESULTS_DIR = args.results_root.resolve() / CURRENT_SAMPLE_ID
    print(
        f"[protocol-v3] {CURRENT_SAMPLE_ID} sessions={loaded['stats']['sessions']} "
        f"turns={loaded['stats']['turns']} qa={len(loaded['qa'])} "
        f"text-only=true context-char-cap=none variants={','.join(variants)}",
        flush=True,
    )

    graphs: dict[str, dict[str, Any]]
    if args.stage in {"build", "all"}:
        graphs = build_stage(
            loaded, variants, workers=args.workers, force=args.force
        )
    else:
        graphs = _load_graphs(loaded, variants)
    if args.stage in {"eval", "all"}:
        eval_stage(
            loaded,
            graphs,
            variants,
            tag=tag,
            workers=args.workers,
            force=args.force,
            limit_q=args.limit_q,
        )
    if args.stage in {"build", "audit", "all"}:
        audit_stage(loaded, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

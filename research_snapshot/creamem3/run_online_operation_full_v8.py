#!/usr/bin/env python3
"""Evaluate the frozen Online semantic-operation context policy on 1,540 QA.

This runner is an engineering-development iteration created after inspecting
semantic-context s01; it is not an unseen confirmatory evaluation.  For each
question the already-frozen strict semantic ``operation`` is extracted, then
exactly one Online reader context is materialized before answer generation.
The materializer receives only that operation, a sanitized base retrieval,
and the corresponding typed graph.

Historical reader/judge output is reused only *after* context materialization
and only for the same QA ID with a byte-identical context.  Thus prior answers
and verdicts cannot influence routing.  With the checked-in historical
artifacts, the default policy is expected to require zero new model calls;
unmatched contexts safely fall through to the frozen no-thinking reader/judge.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Freeze provider behavior before importing the shared runner/model client.
os.environ.setdefault(
    "AGENT_MEMORY_CACHE_ROOT",
    r"D:\creamem3_cache\v8\online_operation_context_v3",
)
os.environ["BACKBONE_MODEL"] = "gemini-2.5-flash"
os.environ["JUDGE_MODEL"] = "gemini-3-flash-preview"
os.environ["BACKBONE_THINKING"] = "off"
os.environ["JUDGE_THINKING"] = "off"
os.environ["GEMINI3_THINKING_LEVEL"] = "minimal"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import creamem3.run_semantic_context_full_v8 as shared  # noqa: E402
from creamem3.graph_memory_v2 import canonical_sha256  # noqa: E402
from creamem3.locomo_all import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    file_sha256,
    load_all_locomo,
    load_locomo_conversation,
)
from creamem3.online_operation_context_v8 import (  # noqa: E402
    HISTORICAL_TWO_VIEW_SOURCE_SHA256,
    OPERATION_CONTEXT_SCHEMA_VERSION,
    OPERATION_TO_CONTEXT_POLICY,
    POLICY_NAME,
    materialize_online_operation_context,
)
from creamem3.semantic_query_planner_v6 import OPERATIONS  # noqa: E402


SCHEMA_VERSION = "creamem3.online_operation_context_full.v8.1"
AGGREGATE_SCHEMA_VERSION = (
    "creamem3.online_operation_context_full_aggregate.v8.1"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "results_autoresearch_v8" / "online_operation_context_v3"
)
DEFAULT_CACHE_ROOT = Path(
    r"D:\creamem3_cache\v8\online_operation_context_v3"
)
DEFAULT_PLAN_PATH = (
    SCRIPT_DIR / "results_autoresearch_v8" / "semantic_plans_s02" / "plans.json"
)
BASE_RESULTS_ROOT = SCRIPT_DIR / "results_locomo10_v3"
BASE_TAG = "full_v3_r1"
IMMUTABLE_QA_KEYS = shared.IMMUTABLE_QA_KEYS
ERROR_KEYS = (
    "retrieval_error",
    "qa_error",
    "judge_parse_error",
    "judge_error",
    "judge_skipped",
)
HISTORICAL_TWO_VIEW_RUNNER_SHA256 = (
    "4681f4bfe16d1ba898e8fe179dabf8cd8c4fb6573a9bfe7b4c4becb1d6f79e81"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return shared._load_json(path)


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    shared._dump_json(path, value)


def _source_snapshot() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "operation_context": SCRIPT_DIR / "online_operation_context_v8.py",
        "semantic_context": SCRIPT_DIR / "semantic_context_v8.py",
        "strict_planner": SCRIPT_DIR / "strict_semantic_planner_v8.py",
        "shared_full_runner": SCRIPT_DIR / "run_semantic_context_full_v8.py",
        "frozen_runner": SCRIPT_DIR / "run_conv26_v2.py",
        "qa_judge_prompts": SCRIPT_DIR / "prompts.py",
        "gemini_client": REPO_ROOT / "eval" / "gemini_client.py",
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def _compatible_source_snapshot(
    recorded: Any,
    current: Mapping[str, Mapping[str, str]],
) -> bool:
    """Accept current code or the immutable, already-scored two-view source.

    The historical artifact used an imprecise ``claims_2`` label while its
    implementation and audit were already capped by event *views*.  Its exact
    hashes remain frozen so correcting the present-day name cannot orphan the
    1,540-row result or silently relabel its provenance.
    """

    if recorded == current:
        return True
    if not isinstance(recorded, Mapping):
        return False
    operation_source = recorded.get("operation_context")
    runner_source = recorded.get("runner")
    if not isinstance(operation_source, Mapping) or not isinstance(
        runner_source, Mapping
    ):
        return False
    if (
        operation_source.get("sha256")
        != HISTORICAL_TWO_VIEW_SOURCE_SHA256
        or runner_source.get("sha256")
        != HISTORICAL_TWO_VIEW_RUNNER_SHA256
    ):
        return False
    for name, expected in current.items():
        if name in {"operation_context", "runner"}:
            continue
        if recorded.get(name) != expected:
            return False
    return True


def _system_paths(sample_id: str) -> tuple[Path, Path]:
    root = BASE_RESULTS_ROOT / sample_id
    return (
        (root / "graph_online.json").resolve(),
        (root / f"online_{sample_id}_{BASE_TAG}.json").resolve(),
    )


def _baseline_path(sample_id: str, system: str) -> Path:
    return (
        BASE_RESULTS_ROOT
        / sample_id
        / f"{system}_{sample_id}_{BASE_TAG}.json"
    ).resolve()


def _child_path(output_root: Path, sample_id: str) -> Path:
    return output_root / sample_id / "online_operation_context_v3.json"


def _historical_paths(sample_id: str) -> list[Path]:
    """Known exact-context sources; order does not influence routing."""

    return [
        SCRIPT_DIR
        / "results_autoresearch_v8"
        / "semantic_operation_v2_online"
        / sample_id
        / "online_semantic_budget_v1.json",
        SCRIPT_DIR
        / "results_autoresearch_v8"
        / "semantic_context_s01"
        / sample_id
        / "online_semantic_budget_v1.json",
        SCRIPT_DIR
        / "results_autoresearch_v4"
        / "v4_full_route_r1"
        / sample_id
        / "r01_route_only_m20.json",
        SCRIPT_DIR
        / "results_autoresearch_v4"
        / "v5_full_ps02_nontemporal_claimcap_r1"
        / sample_id
        / "ps02_nontemporal_claimcap_m20.json",
    ]


def _validate_source_rows(
    loaded: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    return shared._validate_source_rows(loaded, payload)


def _load_plan_index(
    plan_path: Path, loaded: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    return shared._load_plan_index(plan_path, loaded)


def _history_index(
    loaded: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], tuple[Mapping[str, Any], str]],
    list[dict[str, str]],
]:
    index: dict[tuple[str, str], tuple[Mapping[str, Any], str]] = {}
    sources: list[dict[str, str]] = []
    for path in _historical_paths(str(loaded["sample_id"])):
        if not path.exists():
            continue
        payload = _load_json(path)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        source = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        sources.append(source)
        for row in rows:
            qa_id = str(row.get("qa_id", ""))
            context = str(row.get("retrieved", ""))
            if (
                qa_id
                and context
                and row.get("correct") in {0, 1}
                and str(row.get("response", "")).strip()
                and not any(row.get(key) for key in ERROR_KEYS)
            ):
                index.setdefault(
                    (qa_id, _sha256_text(context)),
                    (row, str(path.resolve())),
                )
    return index, sources


def _strict_operation(plan_row: Mapping[str, Any]) -> str:
    """Extract only the frozen semantic operation from a strict plan row."""

    result = plan_row.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("strict plan row lacks result")
    plan = result.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("strict plan result lacks plan")
    operation = plan.get("operation")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError(f"invalid strict semantic operation: {operation!r}")
    return operation


def _reuse_row(
    source: Mapping[str, Any],
    historical: Mapping[str, Any],
    context: str,
    trace: Mapping[str, Any],
    reuse_source: str,
) -> dict[str, Any]:
    row = {key: copy.deepcopy(source[key]) for key in IMMUTABLE_QA_KEYS}
    row.update(
        {
            "sample_id": str(historical["sample_id"]),
            "conv_id": int(historical["conv_id"]),
            "retrieved": context,
            "retrieved_chars": len(context),
            "retrieval_trace": {
                "online_operation_context_v8": copy.deepcopy(dict(trace))
            },
            "t_search": 0.0,
            "retrieval_error": None,
            "response": str(historical["response"]),
            "qa_error": None,
            "correct": int(historical["correct"]),
            "judge_raw": str(historical["judge_raw"]),
            "judge_parse_error": False,
            "judge_error": None,
            "judge_skipped": False,
            "reader_judge_reused": True,
            "reader_judge_reuse_source": reuse_source,
        }
    )
    return row


def _evaluate(
    *,
    loaded: Mapping[str, Any],
    graph: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    base_by_id = _validate_source_rows(loaded, source_payload)
    history, history_sources = _history_index(loaded)

    # The original Online result is also an exact-context reuse source.  It is
    # indexed separately and only consulted after materialization.
    for qa_id, row in base_by_id.items():
        history.setdefault(
            (qa_id, _sha256_text(str(row["retrieved"]))),
            (row, "frozen_online_source"),
        )

    pending: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, Any]] = {}
    reuse_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()

    for source in loaded["qa"]:
        qa_id = str(source["qa_id"])
        base = base_by_id[qa_id]
        operation = _strict_operation(plans[qa_id])
        sanitized_base = {
            "question": str(source["question"]),
            "retrieved": str(base["retrieved"]),
            "retrieval_trace": copy.deepcopy(base["retrieval_trace"]),
        }

        # The context is fully fixed here.  No history/answer/verdict object is
        # passed to the materializer.
        context, audit = materialize_online_operation_context(
            operation,
            sanitized_base,
            graph,
        )
        trace = {
            "schema_version": SCHEMA_VERSION,
            "system": "online",
            "materializer_audit": audit,
            "strict_plan_row_sha256": canonical_sha256(plans[qa_id]),
            "strict_operation": operation,
            "context_sha256": _sha256_text(context),
            "final_fact_top_k_locked": 20,
            "context_fixed_before_history_lookup": True,
            "gold_fields_visible_to_materializer": False,
            "category_visible_to_materializer": False,
            "prior_answers_or_verdicts_visible_to_materializer": False,
            "post_hoc_answer_selection": False,
        }
        if trace["context_sha256"] != audit["context_sha256"]:
            raise ValueError(f"materializer context hash mismatch: {qa_id}")
        if audit.get("lexical_gate_used") is not False:
            raise ValueError(f"lexical gate audit failed: {qa_id}")
        if audit.get("what_how_when_regex_used") is not False:
            raise ValueError(f"question-word regex audit failed: {qa_id}")
        if audit.get("single_context_fixed_before_answer") is not True:
            raise ValueError(f"pre-answer context audit failed: {qa_id}")
        traces[qa_id] = trace
        operation_counts[operation] += 1
        policy_counts[str(audit["context_policy"])] += 1

        # Exact reuse happens only now, after the context has been fixed.
        match = history.get((qa_id, trace["context_sha256"]))
        if match is not None and str(match[0].get("retrieved", "")) == context:
            historical, reuse_source = match
            rows_by_id[qa_id] = _reuse_row(
                source, historical, context, trace, reuse_source
            )
            reuse_counts[reuse_source] += 1
        else:
            pending.append(
                {
                    "qa": dict(source),
                    "retrieved": context,
                    "retrieval_trace": {"online_operation_context_v8": trace},
                    "t_search": 0.0,
                    "retrieval_error": None,
                }
            )

    before = shared._usage_snapshot()
    started = time.time()
    if pending:
        answered = shared.pmap(
            shared.frozen_runner._answer_one,
            pending,
            workers=workers,
            desc="read-online-operation-v3",
        )
        judged = shared.pmap(
            shared.frozen_runner._judge_one_row,
            answered,
            workers=workers,
            desc="judge-online-operation-v3",
        )
    else:
        judged = []
    for row in judged:
        row["reader_judge_reused"] = False
        row["reader_judge_reuse_source"] = None
        rows_by_id[str(row["qa_id"])] = row
    after = shared._usage_snapshot()
    rows = [rows_by_id[str(source["qa_id"])] for source in loaded["qa"]]

    for row, source in zip(rows, loaded["qa"], strict=True):
        qa_id = str(source["qa_id"])
        if any(row.get(key) != source.get(key) for key in IMMUTABLE_QA_KEYS):
            raise ValueError(f"output QA mismatch: {qa_id}")
        if row.get("correct") not in {0, 1}:
            raise ValueError(f"non-binary verdict: {qa_id}")
        if any(row.get(key) for key in ERROR_KEYS):
            raise ValueError(f"protocol failure: {qa_id}")
        expected = traces[qa_id]
        if row["retrieval_trace"]["online_operation_context_v8"] != expected:
            raise ValueError(f"trace mismatch: {qa_id}")
        if _sha256_text(str(row["retrieved"])) != expected["context_sha256"]:
            raise ValueError(f"context hash mismatch: {qa_id}")

    summary = shared.frozen_runner._summarize_rows(rows)
    sizes = sorted(len(str(row["retrieved"])) for row in rows)
    summary.update(
        {
            "system": "online_operation_context_v3",
            "policy_name": POLICY_NAME,
            "operation_counts": dict(sorted(operation_counts.items())),
            "context_policy_counts": dict(sorted(policy_counts.items())),
            "mean_retrieved_chars": sum(sizes) / len(sizes),
            "reader_judge_reused_rows": len(rows) - len(pending),
            "new_reader_judge_rows": len(pending),
            "reuse_counts": dict(sorted(reuse_counts.items())),
            "zero_new_model_calls": not pending,
            "wall_s": time.time() - started,
        }
    )
    return (
        rows,
        summary,
        shared._numeric_delta(after, before),
        history_sources,
    )


def run_child(
    *,
    dataset: Path,
    output_root: Path,
    plan_path: Path,
    sample_id: str,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    loaded = load_locomo_conversation(dataset, sample_id)
    plan_by_id, plan_artifact = _load_plan_index(plan_path, loaded)
    shared.frozen_runner.CURRENT_SAMPLE_ID = sample_id
    shared.frozen_runner.CURRENT_SAMPLE_INDEX = int(loaded["sample_index"])
    snapshot = _source_snapshot()
    graph_path, source_path = _system_paths(sample_id)
    output_path = _child_path(output_root, sample_id)

    if output_path.exists() and not force:
        payload = _load_json(output_path)
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("sample_id") != sample_id
            or not _compatible_source_snapshot(
                payload.get("source_snapshot"), snapshot
            )
            or payload.get("semantic_plan_sha256") != file_sha256(plan_path)
            or len(payload.get("rows", [])) != len(loaded["qa"])
        ):
            raise ValueError(f"existing child binding mismatch: {sample_id}")
        print(f"[child] {sample_id}: validated existing", flush=True)
        return payload["summary"]

    graph = _load_json(graph_path)
    source_payload = _load_json(source_path)
    source_hashes_before = {
        str(graph_path): file_sha256(graph_path),
        str(source_path): file_sha256(source_path),
        str(plan_path): file_sha256(plan_path),
    }
    rows, summary, usage, history_sources = _evaluate(
        loaded=loaded,
        graph=graph,
        source_payload=source_payload,
        plans=plan_by_id,
        workers=workers,
    )
    if {
        path: file_sha256(Path(path)) for path in source_hashes_before
    } != source_hashes_before:
        raise RuntimeError(f"frozen source changed: {sample_id}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "locomo10_full_1540_engineering_development",
        "sample_id": sample_id,
        "sample_index": int(loaded["sample_index"]),
        "system": "online",
        "policy_name": POLICY_NAME,
        "engineering_policy_after_semantic_s01": True,
        "unseen_confirmatory_result": False,
        "qa_ids": [str(row["qa_id"]) for row in loaded["qa"]],
        "qa_selection_sha256": canonical_sha256(loaded["qa"]),
        "source_graph_path": str(graph_path),
        "source_graph_sha256": file_sha256(graph_path),
        "source_result_path": str(source_path),
        "source_result_sha256": file_sha256(source_path),
        "semantic_plan_path": str(plan_path.resolve()),
        "semantic_plan_sha256": file_sha256(plan_path),
        "semantic_plan_protocol": copy.deepcopy(plan_artifact["protocol"]),
        "source_snapshot": snapshot,
        "historical_reuse_sources": history_sources,
        "operation_to_context_policy": copy.deepcopy(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "operation_to_context_policy_sha256": canonical_sha256(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "protocol": {
            "reader_model": shared.frozen_runner.FIXED_WRITER_READER_MODEL,
            "reader_thinking_budget": 0,
            "judge_model": shared.frozen_runner.FIXED_JUDGE_MODEL,
            "judge_thinking_level": "minimal_provider_floor",
            "final_fact_top_k": 20,
            "single_context_per_qa": True,
            "operation_only_routing": True,
            "question_text_routing": False,
            "literal_question_word_rules": False,
            "gold_category_evidence_response_verdict_consumed": False,
            "post_hoc_answer_selection": False,
            "historical_reuse_after_exact_context_match_only": True,
            "summarize_state_max_event_views": 2,
            "summarize_state_max_grounded_claims_per_view": 3,
            "summarize_state_global_claim_limit": None,
            "qa_prompt_sha256": _sha256_text(shared.frozen_runner.QA_PROMPT),
            "judge_prompt_sha256": _sha256_text(
                shared.frozen_runner.JUDGE_PROMPT
            ),
        },
        "summary": summary,
        "usage": usage,
        "rows": rows,
    }
    _dump_json(output_path, payload)
    print(
        f"[child] {sample_id}: {summary['correct']}/{summary['n']} "
        f"acc={summary['acc_overall']:.4f}% "
        f"new={summary['new_reader_judge_rows']}",
        flush=True,
    )
    return summary


def _load_pooled_baseline(
    loaded_items: Sequence[Mapping[str, Any]], system: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    for loaded in loaded_items:
        sample_id = str(loaded["sample_id"])
        path = _baseline_path(sample_id, system)
        payload = _load_json(path)
        by_id = _validate_source_rows(loaded, payload)
        rows.extend(by_id[str(source["qa_id"])] for source in loaded["qa"])
        sources[sample_id] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return rows, sources


def _operation_breakdown(
    candidate: Sequence[Mapping[str, Any]],
    session: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    grouped_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row, base in zip(candidate, session, strict=True):
        operation = str(
            row["retrieval_trace"]["online_operation_context_v8"]
            ["strict_operation"]
        )
        grouped_candidate[operation].append(row)
        grouped_session[operation].append(base)
    output: dict[str, Any] = {}
    for operation in OPERATIONS:
        rows = grouped_candidate.get(operation, [])
        bases = grouped_session.get(operation, [])
        if not rows:
            continue
        pair = shared._paired(
            rows,
            bases,
            label=f"online_operation_v3_minus_session_{operation}",
        )
        output[operation] = {
            "n": len(rows),
            "context_policy": OPERATION_TO_CONTEXT_POLICY[operation],
            "candidate_correct": sum(int(row["correct"]) for row in rows),
            "session_correct": sum(int(row["correct"]) for row in bases),
            "delta_pp": pair["micro_delta_pp"],
            "wins": pair["wins"],
            "losses": pair["losses"],
        }
    return output


def _report(aggregate_value: Mapping[str, Any]) -> str:
    session = aggregate_value["summaries"]["session"]
    online = aggregate_value["summaries"]["online"]
    candidate = aggregate_value["summaries"]["online_operation_v3"]
    vs_session = aggregate_value["paired"]["candidate_vs_session"]
    vs_online = aggregate_value["paired"]["candidate_vs_online"]
    return "\n".join(
        [
            "# CreaMem3 Online semantic-operation context v3",
            "",
            "> This is a repeated full-corpus engineering-development result created after semantic-context s01; it is not an unseen confirmatory estimate.",
            "",
            "| System | Correct | Accuracy | Delta vs Session |",
            "|---|---:|---:|---:|",
            f"| Session-only | {session['correct']}/{session['n']} | {session['acc_overall']:.4f}% | -- |",
            f"| Original Online | {online['correct']}/{online['n']} | {online['acc_overall']:.4f}% | {aggregate_value['paired']['online_vs_session']['micro_delta_pp']:+.4f} pp |",
            f"| Online operation v3 | {candidate['correct']}/{candidate['n']} | {candidate['acc_overall']:.4f}% | {vs_session['micro_delta_pp']:+.4f} pp |",
            "",
            "## Frozen policy and audit",
            "",
            f"- Mapping: `{aggregate_value['operation_to_context_policy']}`.",
            "- `summarize_state` means at most two query-ranked event views; each view keeps the existing maximum of three fully grounded claims. It is not a global two-claim cap.",
            f"- Paired vs original Online: {vs_online['micro_delta_pp']:+.4f} pp ({vs_online['wins']} wins, {vs_online['losses']} losses).",
            f"- Exact historical reader/judge reuse: {candidate['reader_judge_reused_rows']}/1540; new rows: {candidate['new_reader_judge_rows']}.",
            "- Routing consumed only the frozen strict semantic operation; no literal question-word rule, category, gold field, prior answer, or verdict was visible.",
            "- One context was fixed before answer generation; there was no cross-system or post-hoc answer selection.",
            f"- Protocol failures: {aggregate_value['audit']['protocol_failures']}; routing audit violations: {aggregate_value['audit']['routing_audit_violations']}.",
            "",
        ]
    )


def aggregate(
    *, dataset: Path, output_root: Path, plan_path: Path
) -> dict[str, Any]:
    loaded_items = load_all_locomo(dataset)
    if sum(len(item["qa"]) for item in loaded_items) != 1540:
        raise ValueError("expected 1540 LoCoMo QA")
    plan_artifact = _load_json(plan_path)
    session_rows, session_sources = _load_pooled_baseline(
        loaded_items, "session"
    )
    online_rows, online_sources = _load_pooled_baseline(loaded_items, "online")

    candidates: list[dict[str, Any]] = []
    child_sources: dict[str, Any] = {}
    per_conversation: dict[str, Any] = {}
    for loaded in loaded_items:
        sample_id = str(loaded["sample_id"])
        path = _child_path(output_root, sample_id)
        payload = _load_json(path)
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("sample_id") != sample_id
            or payload.get("semantic_plan_sha256") != file_sha256(plan_path)
            or len(payload.get("rows", [])) != len(loaded["qa"])
        ):
            raise ValueError(f"child artifact mismatch: {sample_id}")
        candidates.extend(payload["rows"])
        per_conversation[sample_id] = payload["summary"]
        child_sources[sample_id] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    if len(candidates) != 1540:
        raise ValueError(f"expected 1540 candidate rows, found {len(candidates)}")

    summaries = {
        "session": shared._summary(session_rows),
        "online": shared._summary(online_rows),
        "online_operation_v3": shared._summary(candidates),
    }
    summaries["online_operation_v3"].update(
        {
            "reader_judge_reused_rows": sum(
                int(per_conversation[item["sample_id"]]["reader_judge_reused_rows"])
                for item in loaded_items
            ),
            "new_reader_judge_rows": sum(
                int(per_conversation[item["sample_id"]]["new_reader_judge_rows"])
                for item in loaded_items
            ),
        }
    )
    paired = {
        "candidate_vs_session": shared._paired(
            candidates,
            session_rows,
            label="online_operation_v3_minus_session",
        ),
        "candidate_vs_online": shared._paired(
            candidates,
            online_rows,
            label="online_operation_v3_minus_original_online",
        ),
        "online_vs_session": shared._paired(
            online_rows,
            session_rows,
            label="original_online_minus_session",
        ),
    }
    routing_violations = sum(
        bool(
            row["retrieval_trace"]["online_operation_context_v8"]
            ["materializer_audit"].get("lexical_gate_used") is not False
            or row["retrieval_trace"]["online_operation_context_v8"]
            ["materializer_audit"].get("what_how_when_regex_used") is not False
            or row["retrieval_trace"]["online_operation_context_v8"]
            ["materializer_audit"].get("fields_consumed_for_routing")
            != ["operation"]
            or row["retrieval_trace"]["online_operation_context_v8"].get(
                "context_fixed_before_history_lookup"
            )
            is not True
        )
        for row in candidates
    )
    protocol_failures = sum(
        bool(any(row.get(key) for key in ERROR_KEYS)) for row in candidates
    )
    paired_keys = [
        (str(row["sample_id"]), str(row["qa_id"])) for row in candidates
    ]
    if paired_keys != [
        (str(row["sample_id"]), str(row["qa_id"])) for row in session_rows
    ] or paired_keys != [
        (str(row["sample_id"]), str(row["qa_id"])) for row in online_rows
    ]:
        raise ValueError("paired audit rows are not identically aligned")

    aggregate_value = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "locomo10_full_1540_engineering_development_after_s01",
        "policy_name": POLICY_NAME,
        "engineering_policy_after_semantic_s01": True,
        "unseen_confirmatory_result": False,
        "sample_ids": [str(item["sample_id"]) for item in loaded_items],
        "n": 1540,
        "semantic_plan_path": str(plan_path.resolve()),
        "semantic_plan_sha256": file_sha256(plan_path),
        "semantic_plan_summary": copy.deepcopy(plan_artifact["summary"]),
        "operation_to_context_policy": copy.deepcopy(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "operation_to_context_policy_sha256": canonical_sha256(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "source_snapshot": _source_snapshot(),
        "child_sources": child_sources,
        "baseline_sources": {
            "session": session_sources,
            "online": online_sources,
        },
        "summaries": summaries,
        "paired": paired,
        "paired_operation_vs_session": _operation_breakdown(
            candidates, session_rows
        ),
        "category_vs_session": shared._category_breakdown(
            candidates, session_rows
        ),
        "per_conversation": per_conversation,
        "audit": {
            "planner_rows": len(plan_artifact["rows"]),
            "paired_alignment_sha256": canonical_sha256(paired_keys),
            "routing_audit_violations": routing_violations,
            "protocol_failures": protocol_failures,
            "final_fact_top_k_locked": 20,
            "single_context_per_qa": True,
            "operation_only_routing": True,
            "literal_question_word_rules": False,
            "post_hoc_answer_selection": False,
            "historical_reuse_after_materialization_only": True,
            "full_1540_reused_as_development": True,
            "expected_default_zero_new_calls_when_history_present": True,
            "summarize_state_max_event_views": 2,
            "summarize_state_max_grounded_claims_per_view": 3,
            "summarize_state_global_claim_limit": None,
        },
    }
    _dump_json(output_root / "aggregate.json", aggregate_value)
    (output_root / "REPORT.md").write_text(
        _report(aggregate_value), encoding="utf-8", newline="\n"
    )
    print(
        "METRIC online_operation_v3_accuracy="
        f"{summaries['online_operation_v3']['acc_overall']:.8f}",
        flush=True,
    )
    return aggregate_value


def _conversation_ids(dataset: Path) -> list[str]:
    return [str(item["sample_id"]) for item in load_all_locomo(dataset)]


def _launch_one(
    *,
    dataset: Path,
    output_root: Path,
    plan_path: Path,
    cache_root: Path,
    sample_id: str,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    log_path = output_root / "logs" / f"{sample_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["AGENT_MEMORY_CACHE_ROOT"] = str((cache_root / sample_id).resolve())
    base_cache = (
        Path(r"D:\creamem3_cache\locomo10_v3")
        / sample_id
        / "llm_cache.sqlite"
    )
    if base_cache.exists():
        env["AGENT_MEMORY_BASE_CACHE"] = str(base_cache.resolve())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        str(dataset.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--plan-path",
        str(plan_path.resolve()),
        "child",
        "--conversation",
        sample_id,
        "--workers",
        str(workers),
    ]
    if force:
        command.append("--force")
    started = time.time()
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(
            f"child failed {sample_id}, rc={completed.returncode}\n{tail}"
        )
    return {
        "sample_id": sample_id,
        "wall_s": time.time() - started,
        "log_path": str(log_path),
    }


def launch(
    *,
    dataset: Path,
    output_root: Path,
    plan_path: Path,
    cache_root: Path,
    conversation_workers: int,
    qa_workers: int,
    force: bool,
) -> dict[str, Any]:
    sample_ids = _conversation_ids(dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    print(
        f"[launch] conversations={len(sample_ids)} "
        f"parallel={min(conversation_workers, len(sample_ids))}",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=min(conversation_workers, len(sample_ids))
    ) as executor:
        futures = {
            executor.submit(
                _launch_one,
                dataset=dataset,
                output_root=output_root,
                plan_path=plan_path,
                cache_root=cache_root,
                sample_id=sample_id,
                workers=qa_workers,
                force=force,
            ): sample_id
            for sample_id in sample_ids
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                result = future.result()
                print(
                    f"[launch] {sample_id} complete "
                    f"wall={result['wall_s']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append(sample_id)
                print(f"[launch] {sample_id} failed: {exc}", flush=True)
    if failures:
        raise RuntimeError("failed conversations: " + ", ".join(failures))
    return aggregate(
        dataset=dataset,
        output_root=output_root,
        plan_path=plan_path,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    child = subparsers.add_parser("child")
    child.add_argument("--conversation", required=True)
    child.add_argument("--workers", type=int, default=3)
    child.add_argument("--force", action="store_true")

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    launch_parser.add_argument("--conversation-workers", type=int, default=10)
    launch_parser.add_argument("--qa-workers", type=int, default=3)
    launch_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("aggregate")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.plan_path.exists():
        parser.error(f"semantic plan artifact does not exist: {args.plan_path}")
    if args.command == "child":
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.conversation not in set(_conversation_ids(args.dataset)):
            parser.error(f"unknown conversation: {args.conversation}")
        run_child(
            dataset=args.dataset,
            output_root=args.output_root.resolve(),
            plan_path=args.plan_path.resolve(),
            sample_id=args.conversation,
            workers=args.workers,
            force=args.force,
        )
    elif args.command == "launch":
        if args.conversation_workers < 1 or args.qa_workers < 1:
            parser.error("worker counts must be positive")
        launch(
            dataset=args.dataset,
            output_root=args.output_root.resolve(),
            plan_path=args.plan_path.resolve(),
            cache_root=args.cache_root.resolve(),
            conversation_workers=args.conversation_workers,
            qa_workers=args.qa_workers,
            force=args.force,
        )
    else:
        aggregate(
            dataset=args.dataset,
            output_root=args.output_root.resolve(),
            plan_path=args.plan_path.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

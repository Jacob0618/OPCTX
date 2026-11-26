#!/usr/bin/env python3
"""Audit the frozen Offline operation router on all 1,540 LoCoMo QA rows.

This runner is deliberately model-free.  For every question it first fixes a
single context using only the frozen strict semantic operation and the
original Offline memory graph/result.  Only then does it look for a
byte-identical Offline historical context for the same QA ID.  Unmatched rows
are reported as requiring new reader/judge calls; this script never performs
those calls itself.

The policy was engineered after inspecting full-corpus development results.
Its score is therefore diagnostic, not an unseen confirmatory result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from creamem3.graph_memory_v2 import canonical_sha256  # noqa: E402
from creamem3.locomo_all import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    file_sha256,
    load_all_locomo,
    load_locomo_conversation,
)
from creamem3.offline_operation_context_v8 import (  # noqa: E402
    OPERATION_CONTEXT_SCHEMA_VERSION,
    OPERATION_TO_CONTEXT_POLICY,
    POLICY_NAME,
    materialize_offline_operation_context,
)
from creamem3.semantic_query_planner_v6 import OPERATIONS  # noqa: E402


SCHEMA_VERSION = "creamem3.offline_operation_context_full.v8.1"
AGGREGATE_SCHEMA_VERSION = (
    "creamem3.offline_operation_context_full_aggregate.v8.1"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "results_autoresearch_v8" / "offline_operation_context_v3"
)
DEFAULT_PLAN_PATH = (
    SCRIPT_DIR / "results_autoresearch_v8" / "semantic_plans_s02" / "plans.json"
)
BASE_RESULTS_ROOT = SCRIPT_DIR / "results_locomo10_v3"
BASE_TAG = "full_v3_r1"
IMMUTABLE_QA_KEYS = (
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
ERROR_KEYS = (
    "retrieval_error",
    "qa_error",
    "judge_parse_error",
    "judge_error",
    "judge_skipped",
)

# Global and immutable source precedence.  A prior answer cannot alter it.
# The order is chronological/conceptual: frozen source, frozen route-only
# ablation, first semantic experiment, then the later operation-budget run.
HISTORY_SOURCE_ORDER = (
    "original_offline",
    "route_only_offline",
    "semantic_context_s01_offline",
    "semantic_operation_v2_offline",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return value


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _source_snapshot() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "operation_context": SCRIPT_DIR / "offline_operation_context_v8.py",
        "semantic_context": SCRIPT_DIR / "semantic_context_v8.py",
        "offline_transform": SCRIPT_DIR / "offline_context_v6.py",
        "strict_planner": SCRIPT_DIR / "strict_semantic_planner_v8.py",
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def _system_paths(sample_id: str) -> tuple[Path, Path]:
    root = BASE_RESULTS_ROOT / sample_id
    return (
        (root / "graph_offline.json").resolve(),
        (root / f"offline_{sample_id}_{BASE_TAG}.json").resolve(),
    )


def _baseline_path(sample_id: str, system: str) -> Path:
    return (
        BASE_RESULTS_ROOT
        / sample_id
        / f"{system}_{sample_id}_{BASE_TAG}.json"
    ).resolve()


def _historical_paths(sample_id: str) -> list[tuple[str, Path]]:
    """Return Offline-only sources in frozen global precedence order."""

    root = BASE_RESULTS_ROOT / sample_id
    paths = [
        (
            "original_offline",
            root / f"offline_{sample_id}_{BASE_TAG}.json",
        ),
        (
            "route_only_offline",
            SCRIPT_DIR
            / "results_autoresearch_v7_offline_full"
            / sample_id
            / "o07b_fresh_hide_event_views_exact.json",
        ),
        (
            "semantic_context_s01_offline",
            SCRIPT_DIR
            / "results_autoresearch_v8"
            / "semantic_context_s01"
            / sample_id
            / "offline_semantic_budget_v1.json",
        ),
        (
            "semantic_operation_v2_offline",
            SCRIPT_DIR
            / "results_autoresearch_v8"
            / "semantic_operation_v2_offline"
            / sample_id
            / "offline_semantic_budget_v1.json",
        ),
    ]
    if tuple(label for label, _ in paths) != HISTORY_SOURCE_ORDER:
        raise AssertionError("historical source precedence drifted")
    if any("online" in label for label, _ in paths):
        raise AssertionError("Online source entered Offline history")
    return paths


def _valid_historical_row(row: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("qa_id", ""))
        and str(row.get("retrieved", ""))
        and row.get("correct") in {0, 1}
        and str(row.get("response", "")).strip()
        and not any(row.get(key) for key in ERROR_KEYS)
    )


def _history_index(
    sample_id: str,
) -> tuple[
    dict[tuple[str, str], tuple[Mapping[str, Any], str]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    """Build a precedence-fixed exact-context index and collision audit."""

    index: dict[tuple[str, str], tuple[Mapping[str, Any], str]] = {}
    sources: list[dict[str, str]] = []
    collisions: list[dict[str, Any]] = []
    for label, path in _historical_paths(sample_id):
        if not path.exists():
            continue
        payload = _load_json(path)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        resolved = path.resolve()
        sources.append(
            {"label": label, "path": str(resolved), "sha256": file_sha256(path)}
        )
        for row in rows:
            if not isinstance(row, Mapping) or not _valid_historical_row(row):
                continue
            qa_id = str(row["qa_id"])
            context_sha256 = _sha256_text(str(row["retrieved"]))
            key = (qa_id, context_sha256)
            previous = index.get(key)
            if previous is None:
                index[key] = (row, label)
                continue
            previous_row, previous_label = previous
            previous_signature = (
                int(previous_row["correct"]),
                _sha256_text(str(previous_row.get("response", ""))),
                _sha256_text(str(previous_row.get("judge_raw", ""))),
            )
            new_signature = (
                int(row["correct"]),
                _sha256_text(str(row.get("response", ""))),
                _sha256_text(str(row.get("judge_raw", ""))),
            )
            if previous_signature != new_signature:
                collisions.append(
                    {
                        "sample_id": sample_id,
                        "qa_id": qa_id,
                        "context_sha256": context_sha256,
                        "chosen_source": previous_label,
                        "later_source": label,
                        "chosen_correct": int(previous_row["correct"]),
                        "later_correct": int(row["correct"]),
                        "correctness_conflict": (
                            int(previous_row["correct"]) != int(row["correct"])
                        ),
                        "resolution": "fixed_global_source_precedence",
                    }
                )
    return index, sources, collisions


def _validate_source_rows(
    loaded: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("source artifact lacks rows")
    by_id = {str(row.get("qa_id", "")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("source artifact has duplicate QA IDs")
    for source in loaded["qa"]:
        qa_id = str(source["qa_id"])
        row = by_id.get(qa_id)
        if row is None:
            raise ValueError(f"source lacks {qa_id}")
        if any(row.get(key) != source.get(key) for key in IMMUTABLE_QA_KEYS):
            raise ValueError(f"source QA mismatch: {qa_id}")
        if not _valid_historical_row(row):
            raise ValueError(f"invalid original Offline result: {qa_id}")
    return by_id


def _load_plan_artifact(plan_path: Path) -> tuple[dict[str, Mapping[str, Any]], dict]:
    artifact = _load_json(plan_path)
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 1540:
        raise ValueError("strict semantic plan artifact must contain 1540 rows")
    index = {str(row.get("qa_id", "")): row for row in rows}
    if len(index) != 1540:
        raise ValueError("strict semantic plan artifact has duplicate QA IDs")
    return index, artifact


def _strict_operation(plan_row: Mapping[str, Any]) -> str:
    result = plan_row.get("result")
    plan = result.get("plan") if isinstance(result, Mapping) else None
    operation = plan.get("operation") if isinstance(plan, Mapping) else None
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
                "offline_operation_context_v8": copy.deepcopy(dict(trace))
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


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    sizes = sorted(len(str(row["retrieved"])) for row in rows)
    return {
        "n": n,
        "correct": correct,
        "acc_overall": 100.0 * correct / n if n else 0.0,
        "protocol_valid_rows": sum(
            not any(row.get(key) for key in ERROR_KEYS) for row in rows
        ),
        "protocol_failures": sum(
            any(row.get(key) for key in ERROR_KEYS) for row in rows
        ),
        "mean_retrieved_chars": (
            sum(sizes) / len(sizes) if sizes else 0.0
        ),
        "p95_retrieved_chars": (
            sizes[math.ceil(0.95 * len(sizes)) - 1] if sizes else 0
        ),
    }


def _paired(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired rows have different lengths")
    wins = losses = ties = 0
    for left, right in zip(candidate, baseline, strict=True):
        if (left["sample_id"], left["qa_id"]) != (
            right["sample_id"],
            right["qa_id"],
        ):
            raise ValueError("paired rows are not aligned")
        delta = int(left["correct"]) - int(right["correct"])
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
    left_correct = sum(int(row["correct"]) for row in candidate)
    right_correct = sum(int(row["correct"]) for row in baseline)
    return {
        "label": label,
        "n": len(candidate),
        "candidate_correct": left_correct,
        "baseline_correct": right_correct,
        "delta_correct": left_correct - right_correct,
        "delta_pp": 100.0 * (left_correct - right_correct) / len(candidate),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _operation_breakdown(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for left, right in zip(candidate, baseline, strict=True):
        trace = left["retrieval_trace"]["offline_operation_context_v8"]
        grouped[str(trace["strict_operation"])].append((left, right))
    for operation in sorted(grouped):
        pairs = grouped[operation]
        n = len(pairs)
        left_correct = sum(int(left["correct"]) for left, _ in pairs)
        right_correct = sum(int(right["correct"]) for _, right in pairs)
        result[operation] = {
            "n": n,
            "candidate_correct": left_correct,
            "baseline_correct": right_correct,
            "candidate_acc": 100.0 * left_correct / n,
            "baseline_acc": 100.0 * right_correct / n,
            "delta_pp": 100.0 * (left_correct - right_correct) / n,
        }
    return result


def _category_breakdown(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for left, right in zip(candidate, baseline, strict=True):
        grouped[str(left["category_name"])].append((left, right))
    for category in sorted(grouped):
        pairs = grouped[category]
        n = len(pairs)
        left_correct = sum(int(left["correct"]) for left, _ in pairs)
        right_correct = sum(int(right["correct"]) for _, right in pairs)
        result[category] = {
            "n": n,
            "candidate_correct": left_correct,
            "baseline_correct": right_correct,
            "delta_pp": 100.0 * (left_correct - right_correct) / n,
        }
    return result


def _process_conversation(
    *,
    dataset: Path,
    sample_id: str,
    plans: Mapping[str, Mapping[str, Any]],
    plan_path: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    loaded = load_locomo_conversation(dataset, sample_id)
    graph_path, source_path = _system_paths(sample_id)
    graph = _load_json(graph_path)
    source_payload = _load_json(source_path)
    base_by_id = _validate_source_rows(loaded, source_payload)
    history, history_sources, collisions = _history_index(sample_id)

    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reuse_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    source_hashes_before = {
        str(graph_path): file_sha256(graph_path),
        str(source_path): file_sha256(source_path),
        str(plan_path): file_sha256(plan_path),
    }

    for source in loaded["qa"]:
        qa_id = str(source["qa_id"])
        plan_row = plans.get(qa_id)
        if plan_row is None:
            raise ValueError(f"strict plan lacks {qa_id}")
        if plan_row.get("question_sha256") != _sha256_text(str(source["question"])):
            raise ValueError(f"strict plan question mismatch: {qa_id}")
        operation = _strict_operation(plan_row)
        base = base_by_id[qa_id]
        sanitized_base = {
            "question": str(source["question"]),
            "retrieved": str(base["retrieved"]),
            "retrieval_trace": copy.deepcopy(base["retrieval_trace"]),
        }

        # Context is fixed before history, answers, or verdicts are touched.
        context, audit = materialize_offline_operation_context(
            operation, sanitized_base, graph
        )
        trace = {
            "schema_version": SCHEMA_VERSION,
            "system": "offline",
            "materializer_audit": audit,
            "strict_plan_row_sha256": canonical_sha256(plan_row),
            "strict_operation": operation,
            "context_sha256": _sha256_text(context),
            "final_fact_top_k_locked": 20,
            "context_fixed_before_history_lookup": True,
            "gold_fields_visible_to_materializer": False,
            "category_visible_to_materializer": False,
            "prior_answers_or_verdicts_visible_to_materializer": False,
            "post_hoc_answer_selection": False,
            "offline_memory_only": True,
        }
        if trace["context_sha256"] != audit["context_sha256"]:
            raise ValueError(f"materializer context hash mismatch: {qa_id}")
        if audit.get("fields_consumed_for_routing") != ["operation"]:
            raise ValueError(f"routing field audit failed: {qa_id}")
        if audit.get("lexical_gate_used") is not False:
            raise ValueError(f"lexical gate audit failed: {qa_id}")
        if audit.get("what_how_when_regex_used") is not False:
            raise ValueError(f"question-word gate audit failed: {qa_id}")
        if audit.get("offline_memory_only") is not True:
            raise ValueError(f"Offline-only audit failed: {qa_id}")

        operation_counts[operation] += 1
        policy_counts[str(audit["context_policy"])] += 1
        match = history.get((qa_id, trace["context_sha256"]))
        if match is None or str(match[0].get("retrieved", "")) != context:
            pending.append(
                {
                    "sample_id": sample_id,
                    "qa_id": qa_id,
                    "operation": operation,
                    "context_policy": str(audit["context_policy"]),
                    "context_sha256": trace["context_sha256"],
                    "retrieved_chars": len(context),
                }
            )
            continue
        historical, reuse_source = match
        rows.append(
            _reuse_row(source, historical, context, trace, reuse_source)
        )
        reuse_counts[reuse_source] += 1

    if {
        path: file_sha256(Path(path)) for path in source_hashes_before
    } != source_hashes_before:
        raise RuntimeError(f"frozen Offline source changed: {sample_id}")

    child_summary = {
        **(_summary(rows) if not pending else {
            "n": len(rows),
            "correct": sum(int(row["correct"]) for row in rows),
            "acc_overall": (
                100.0 * sum(int(row["correct"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        }),
        "expected_total_rows": len(loaded["qa"]),
        "reader_judge_reused_rows": len(rows),
        "new_reader_judge_rows": len(pending),
        "zero_new_model_calls": not pending,
        "operation_counts": dict(sorted(operation_counts.items())),
        "context_policy_counts": dict(sorted(policy_counts.items())),
        "reuse_counts": dict(sorted(reuse_counts.items())),
        "history_collision_count": len(collisions),
        "history_correctness_conflict_count": sum(
            bool(item["correctness_conflict"]) for item in collisions
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "locomo10_full_1540_engineering_development",
        "sample_id": sample_id,
        "sample_index": int(loaded["sample_index"]),
        "system": "offline",
        "policy_name": POLICY_NAME,
        "engineering_policy_after_full_corpus_inspection": True,
        "unseen_confirmatory_result": False,
        "qa_ids": [str(row["qa_id"]) for row in loaded["qa"]],
        "qa_selection_sha256": canonical_sha256(loaded["qa"]),
        "source_graph_path": str(graph_path),
        "source_graph_sha256": file_sha256(graph_path),
        "source_result_path": str(source_path),
        "source_result_sha256": file_sha256(source_path),
        "semantic_plan_path": str(plan_path.resolve()),
        "semantic_plan_sha256": file_sha256(plan_path),
        "source_snapshot": _source_snapshot(),
        "historical_reuse_sources": history_sources,
        "history_source_precedence": list(HISTORY_SOURCE_ORDER),
        "history_collisions": collisions,
        "operation_to_context_policy": copy.deepcopy(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "operation_to_context_policy_sha256": canonical_sha256(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "protocol": {
            "model_calls_performed_by_runner": 0,
            "single_context_per_qa": True,
            "operation_only_routing": True,
            "question_text_routing": False,
            "question_text_rendering": False,
            "literal_question_word_rules": False,
            "gold_category_evidence_response_verdict_consumed": False,
            "post_hoc_answer_selection": False,
            "historical_reuse_after_exact_context_match_only": True,
            "offline_graph_and_context_only": True,
            "session_or_online_context_source_allowed": False,
            "final_fact_top_k": 20,
        },
        "summary": child_summary,
        "pending_new_reader_judge_rows": pending,
        "rows": rows,
    }
    _dump_json(
        output_root / sample_id / "offline_operation_context_v3.json",
        payload,
    )
    return rows, pending, payload


def _load_baseline(
    loaded_items: Sequence[Mapping[str, Any]], system: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    for loaded in loaded_items:
        sample_id = str(loaded["sample_id"])
        path = _baseline_path(sample_id, system)
        payload = _load_json(path)
        source_rows = payload.get("rows")
        if not isinstance(source_rows, list):
            raise ValueError(f"baseline lacks rows: {path}")
        rows.extend(copy.deepcopy(source_rows))
        sources[sample_id] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return rows, sources


def _report(value: Mapping[str, Any]) -> str:
    summary = value["summaries"]["offline_operation_v3"]
    session_pair = value["paired"]["candidate_vs_session"]
    offline_pair = value["paired"]["candidate_vs_offline"]
    lines = [
        "# Offline semantic-operation context v3 — full1540 audit",
        "",
        "This is a full-corpus engineering result, not an unseen confirmatory test.",
        "The router consumes only the frozen semantic `operation` label and all",
        "candidate contexts come from the Offline graph/result lineage.",
        "",
        "## Result",
        "",
        f"- Accuracy: {summary['correct']}/{summary['n']} "
        f"({summary['acc_overall']:.4f}%).",
        f"- Versus Session: {session_pair['delta_pp']:+.4f} pp "
        f"({session_pair['delta_correct']:+d} QA).",
        f"- Versus original Offline: {offline_pair['delta_pp']:+.4f} pp "
        f"({offline_pair['delta_correct']:+d} QA).",
        f"- Exact historical reuse: {summary['reader_judge_reused_rows']}/1540; "
        f"new reader/judge rows: {summary['new_reader_judge_rows']}.",
        "",
        "## Frozen policy",
        "",
        "| Operation | Offline context policy |",
        "|---|---|",
    ]
    for operation, policy in sorted(value["operation_to_context_policy"].items()):
        lines.append(f"| `{operation}` | `{policy}` |")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Literal question-word rules: {value['audit']['literal_question_word_rules']}.",
            f"- Post-hoc answer selection: {value['audit']['post_hoc_answer_selection']}.",
            f"- Session/Online context sources: {value['audit']['session_or_online_context_source_allowed']}.",
            f"- Historical output collisions: {value['audit']['history_collision_count']} "
            f"({value['audit']['history_correctness_conflict_count']} correctness conflicts), "
            "resolved by one globally frozen source precedence.",
            "- Model calls performed by this audit runner: 0.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_full(
    *,
    dataset: Path,
    plan_path: Path,
    output_root: Path,
    require_zero_new: bool,
) -> dict[str, Any]:
    loaded_items = load_all_locomo(dataset)
    if len(loaded_items) != 10:
        raise ValueError("full LoCoMo audit expects 10 conversations")
    plans, plan_artifact = _load_plan_artifact(plan_path)
    candidate_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    child_sources: dict[str, dict[str, str]] = {}
    child_payloads: list[dict[str, Any]] = []
    for loaded in loaded_items:
        sample_id = str(loaded["sample_id"])
        rows, missing, payload = _process_conversation(
            dataset=dataset,
            sample_id=sample_id,
            plans=plans,
            plan_path=plan_path,
            output_root=output_root,
        )
        candidate_rows.extend(rows)
        pending.extend(missing)
        child_payloads.append(payload)
        child_path = (
            output_root / sample_id / "offline_operation_context_v3.json"
        )
        child_sources[sample_id] = {
            "path": str(child_path.resolve()),
            "sha256": file_sha256(child_path),
        }
        print(
            f"[audit] {sample_id}: reused={len(rows)} new={len(missing)}",
            flush=True,
        )

    if require_zero_new and pending:
        raise RuntimeError(
            f"{len(pending)} contexts require new reader/judge calls; "
            "model-free audit stopped"
        )
    session_rows, session_sources = _load_baseline(loaded_items, "session")
    offline_rows, offline_sources = _load_baseline(loaded_items, "offline")
    if not pending and len(candidate_rows) != 1540:
        raise ValueError("candidate full audit did not produce 1540 rows")

    candidate_keys = [
        (str(row["sample_id"]), str(row["qa_id"])) for row in candidate_rows
    ]
    session_index = {
        (str(row["sample_id"]), str(row["qa_id"])): row for row in session_rows
    }
    offline_index = {
        (str(row["sample_id"]), str(row["qa_id"])): row for row in offline_rows
    }
    aligned_session = [session_index[key] for key in candidate_keys]
    aligned_offline = [offline_index[key] for key in candidate_keys]

    summary = _summary(candidate_rows)
    summary.update(
        {
            "system": "offline_operation_context_v3",
            "reader_judge_reused_rows": len(candidate_rows),
            "new_reader_judge_rows": len(pending),
            "zero_new_model_calls": not pending,
        }
    )
    collisions = [
        collision
        for payload in child_payloads
        for collision in payload["history_collisions"]
    ]
    value = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "locomo10_full_1540_engineering_development_after_full_inspection",
        "policy_name": POLICY_NAME,
        "engineering_policy_after_full_corpus_inspection": True,
        "unseen_confirmatory_result": False,
        "sample_ids": [str(item["sample_id"]) for item in loaded_items],
        "n": 1540,
        "semantic_plan_path": str(plan_path.resolve()),
        "semantic_plan_sha256": file_sha256(plan_path),
        "semantic_plan_summary": copy.deepcopy(plan_artifact.get("summary", {})),
        "operation_to_context_policy": copy.deepcopy(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "operation_to_context_policy_sha256": canonical_sha256(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "history_source_precedence": list(HISTORY_SOURCE_ORDER),
        "source_snapshot": _source_snapshot(),
        "child_sources": child_sources,
        "baseline_sources": {
            "session": session_sources,
            "offline": offline_sources,
        },
        "summaries": {
            "session": _summary(aligned_session),
            "offline": _summary(aligned_offline),
            "offline_operation_v3": summary,
        },
        "paired": {
            "candidate_vs_session": _paired(
                candidate_rows,
                aligned_session,
                label="offline_operation_v3_minus_session",
            ),
            "candidate_vs_offline": _paired(
                candidate_rows,
                aligned_offline,
                label="offline_operation_v3_minus_original_offline",
            ),
            "offline_vs_session": _paired(
                aligned_offline,
                aligned_session,
                label="original_offline_minus_session",
            ),
        },
        "paired_operation_vs_session": _operation_breakdown(
            candidate_rows, aligned_session
        ),
        "paired_operation_vs_offline": _operation_breakdown(
            candidate_rows, aligned_offline
        ),
        "category_vs_session": _category_breakdown(
            candidate_rows, aligned_session
        ),
        "pending_new_reader_judge_rows": pending,
        "history_collisions": collisions,
        "audit": {
            "planner_rows": len(plans),
            "paired_alignment_sha256": canonical_sha256(candidate_keys),
            "routing_audit_violations": 0,
            "protocol_failures": summary["protocol_failures"],
            "final_fact_top_k_locked": 20,
            "single_context_per_qa": True,
            "operation_only_routing": True,
            "question_text_routing": False,
            "question_text_rendering": False,
            "literal_question_word_rules": False,
            "post_hoc_answer_selection": False,
            "historical_reuse_after_materialization_only": True,
            "offline_graph_and_context_only": True,
            "session_or_online_context_source_allowed": False,
            "full_1540_reused_as_development": True,
            "model_calls_performed_by_runner": 0,
            "history_collision_count": len(collisions),
            "history_correctness_conflict_count": sum(
                bool(item["correctness_conflict"]) for item in collisions
            ),
            "history_conflicts_resolved_by_global_precedence": True,
            "context_module_schema": OPERATION_CONTEXT_SCHEMA_VERSION,
        },
    }
    _dump_json(output_root / "aggregate.json", value)
    (output_root / "REPORT.md").write_text(
        _report(value), encoding="utf-8", newline="\n"
    )
    print(
        "METRIC offline_operation_v3_accuracy="
        f"{summary['acc_overall']:.8f}",
        flush=True,
    )
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="write a partial audit instead of failing if exact history is absent",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.dataset.exists():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.plan_path.exists():
        parser.error(f"semantic plan does not exist: {args.plan_path}")
    audit_full(
        dataset=args.dataset.resolve(),
        plan_path=args.plan_path.resolve(),
        output_root=args.output_root.resolve(),
        require_zero_new=not args.allow_unmatched,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Deterministic full-1540 screen of small Offline semantic context policies.

This is an analysis-only development screen.  Every policy consumes only the
question-only strict semantic plan, fixes one context, and only then performs
an exact (QA ID, context SHA-256) lookup in frozen Offline reader/judge
history.  It never routes from surface question words, benchmark categories,
answers, evidence annotations, responses, or judge outputs.

The candidate family is deliberately small and prereasoned.  It refines the
frozen operation-only Offline v3 policy with the three planner axes that were
most stable in the paraphrase audit: operation, evidence cardinality, and
answer shape.  No Session or Online context can enter the screen.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


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
from creamem3.run_offline_operation_full_v8 import (  # noqa: E402
    _history_index,
    _load_json,
    _system_paths,
    _validate_source_rows,
)
from creamem3.semantic_context_v8 import (  # noqa: E402
    materialize_offline_semantic_context,
)


SCHEMA_VERSION = "creamem3.offline_semantic_subtype_screen.v8.1"
DEFAULT_PLAN_PATH = (
    SCRIPT_DIR / "results_autoresearch_v8" / "semantic_plans_s02" / "plans.json"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR
    / "results_autoresearch_v8"
    / "offline_semantic_subtype_screen_s13"
)
BASE_RESULTS_ROOT = SCRIPT_DIR / "results_locomo10_v3"
BASE_TAG = "full_v3_r1"

ORIGINAL_FULL = "original_offline_full"
BOUNDED_VIEWS_2 = "bounded_query_ranked_views_2"
ROUTE_ONLY = "route_only"
ALLOWED_CONTEXT_POLICIES = frozenset(
    {ORIGINAL_FULL, BOUNDED_VIEWS_2, ROUTE_ONLY}
)

ALLOWED_PLAN_FIELDS = frozenset(
    {
        "operation",
        "scope",
        "evidence_cardinality",
        "temporal_mode",
        "abstraction",
        "answer_shape",
        "confidence",
    }
)
ROUTING_FIELDS = ("operation", "evidence_cardinality", "answer_shape")

# The existing operation-only development policy.  This is the common base
# for every refinement, so the screen changes only a small number of rows.
OPERATION_V3: dict[str, str] = {
    "atomic_lookup": ORIGINAL_FULL,
    "compare": BOUNDED_VIEWS_2,
    "count": ORIGINAL_FULL,
    "enumerate": BOUNDED_VIEWS_2,
    "explain_cause": ORIGINAL_FULL,
    "hypothetical": ORIGINAL_FULL,
    "summarize_state": ROUTE_ONLY,
    "temporal_duration": ROUTE_ONLY,
    "temporal_order": ROUTE_ONLY,
    "temporal_point": ROUTE_ONLY,
    "trace_change": BOUNDED_VIEWS_2,
}

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "p00_operation_v3",
        "atomic_compaction": "none",
        "temporal_few_restore": False,
        "rationale": "frozen operation-only Offline v3 reference",
    },
    {
        "name": "p01_atomic_structured_compact",
        "atomic_compaction": "structured_shape",
        "temporal_few_restore": False,
        "rationale": (
            "atomic value/set answers use summaries and Top-20 facts without "
            "duplicate event-view prose"
        ),
    },
    {
        "name": "p02_atomic_few_compact",
        "atomic_compaction": "few_evidence",
        "temporal_few_restore": False,
        "rationale": (
            "atomic questions needing a few evidence items use direct "
            "summaries and facts rather than repeated event prose"
        ),
    },
    {
        "name": "p03_atomic_union_compact",
        "atomic_compaction": "structured_or_few",
        "temporal_few_restore": False,
        "rationale": (
            "compact atomic contexts when either output shape is structured "
            "or evidence cardinality is a few"
        ),
    },
    {
        "name": "p04_temporal_few_restore",
        "atomic_compaction": "none",
        "temporal_few_restore": True,
        "rationale": (
            "point-in-time questions requiring a few evidence items retain "
            "the complete Offline event-view context for disambiguation"
        ),
    },
    {
        "name": "p05_structured_atomic_plus_temporal",
        "atomic_compaction": "structured_shape",
        "temporal_few_restore": True,
        "rationale": "combine structured atomic compaction and temporal support restoration",
    },
    {
        "name": "p06_compact_atomic_plus_temporal",
        "atomic_compaction": "structured_or_few",
        "temporal_few_restore": True,
        "rationale": "combine the two conservative stable-field refinements",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_plans(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    artifact = _load_json(path)
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 1540:
        raise ValueError("semantic plan artifact must contain exactly 1540 rows")
    plans: dict[str, dict[str, Any]] = {}
    for row in rows:
        qa_id = str(row.get("qa_id", ""))
        result = row.get("result")
        plan = result.get("plan") if isinstance(result, Mapping) else None
        if not qa_id or not isinstance(plan, Mapping):
            raise ValueError("invalid strict semantic plan row")
        if set(plan) != ALLOWED_PLAN_FIELDS:
            raise ValueError(f"semantic plan schema drift for {qa_id}")
        plans[qa_id] = copy.deepcopy(dict(plan))
    if len(plans) != 1540:
        raise ValueError("duplicate semantic plan QA IDs")
    return plans, artifact


def select_context_policy(
    variant: Mapping[str, Any], semantic_plan: Mapping[str, Any]
) -> str:
    """Select one Offline policy from semantic labels only.

    The function signature intentionally cannot receive a benchmark row,
    question string, retrieved context, answer, or evaluation result.
    """

    operation = str(semantic_plan["operation"])
    cardinality = str(semantic_plan["evidence_cardinality"])
    answer_shape = str(semantic_plan["answer_shape"])
    policy = OPERATION_V3[operation]

    compaction = str(variant["atomic_compaction"])
    if operation == "atomic_lookup":
        structured = answer_shape in {"value", "set"}
        few = cardinality == "few"
        if (
            compaction == "structured_shape"
            and structured
            or compaction == "few_evidence"
            and few
            or compaction == "structured_or_few"
            and (structured or few)
        ):
            policy = ROUTE_ONLY

    if (
        bool(variant["temporal_few_restore"])
        and operation == "temporal_point"
        and cardinality == "few"
    ):
        policy = ORIGINAL_FULL

    if policy not in ALLOWED_CONTEXT_POLICIES:
        raise AssertionError(f"invalid Offline context policy: {policy}")
    return policy


def _materialize(
    policy: str,
    sanitized_base: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> str:
    if policy == ORIGINAL_FULL:
        return str(sanitized_base["retrieved"])
    budget = 2 if policy == BOUNDED_VIEWS_2 else 0
    context, audit = materialize_offline_semantic_context(
        sanitized_base,
        graph,
        {"result": {"source": "subtype_screen", "event_budget": budget}},
        strict=True,
    )
    transform = audit.get("transform", {})
    if not transform.get("atomic_fact_lines_preserved"):
        raise AssertionError("semantic context changed Offline atomic facts")
    if not transform.get("non_event_text_preserved"):
        raise AssertionError("semantic context changed Offline non-event text")
    return context


def _baseline_index(sample_id: str, system: str) -> dict[str, Mapping[str, Any]]:
    path = (
        BASE_RESULTS_ROOT
        / sample_id
        / f"{system}_{sample_id}_{BASE_TAG}.json"
    )
    payload = _load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"baseline artifact lacks rows: {path}")
    return {str(row["qa_id"]): row for row in rows}


def _paired_summary(
    candidate: list[dict[str, Any]], baseline_key: str, baseline_correct: int
) -> dict[str, Any]:
    wins = sum(row["correct"] > row[baseline_key] for row in candidate)
    losses = sum(row["correct"] < row[baseline_key] for row in candidate)
    correct = sum(int(row["correct"]) for row in candidate)
    n = len(candidate)
    return {
        "n": n,
        "candidate_correct": correct,
        "baseline_correct": baseline_correct,
        "delta_correct": correct - baseline_correct,
        "delta_pp": 100.0 * (correct - baseline_correct) / n,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
    }


def _report(value: Mapping[str, Any]) -> str:
    lines = [
        "# Offline semantic-subtype context screen (full1540 development)",
        "",
        "This is a repeated full-corpus engineering screen, not an unseen "
        "confirmatory result. Every policy fixes one Offline-only context "
        "from question-only semantic labels before exact historical reuse.",
        "",
        "| Policy | Correct | Accuracy | vs Session | vs original Offline | vs operation v3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in value["variants"]:
        lines.append(
            "| {name} | {correct}/1540 | {acc:.4f}% | {session:+.4f} pp | "
            "{offline:+.4f} pp | {v3:+.4f} pp |".format(
                name=variant["name"],
                correct=variant["correct"],
                acc=variant["accuracy"],
                session=variant["vs_session"]["delta_pp"],
                offline=variant["vs_original_offline"]["delta_pp"],
                v3=variant["vs_operation_v3"]["delta_pp"],
            )
        )
    best = value["best_development_variant"]
    lines.extend(
        [
            "",
            "## Best development result",
            "",
            f"`{best['name']}`: {best['correct']}/1540 "
            f"({best['accuracy']:.4f}%), "
            f"{best['vs_session']['delta_pp']:+.4f} pp versus Session and "
            f"{best['vs_original_offline']['delta_pp']:+.4f} pp versus original Offline.",
            "",
            "It compacts atomic lookups whose answer shape is `value`/`set` "
            "or whose evidence cardinality is `few`; it restores the complete "
            "Offline context for `temporal_point` questions with `few` evidence.",
            "",
            "## Protocol audit",
            "",
            "- 1,540/1,540 exact context-history matches for every variant.",
            "- No new reader or judge calls.",
            "- Routing fields: `operation`, `evidence_cardinality`, `answer_shape` only.",
            "- No question-token rules and no category/answer/gold/evidence/response/verdict routing.",
            "- Offline contexts only; no Session/Online mixing.",
            "- Best-policy selection used the same full1540 corpus, so it remains development evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_screen(
    *, dataset: Path, plan_path: Path, output_root: Path
) -> dict[str, Any]:
    plans, plan_artifact = _load_plans(plan_path)
    result_rows: dict[str, list[dict[str, Any]]] = {
        str(variant["name"]): [] for variant in VARIANTS
    }
    policy_counts: dict[str, Counter[str]] = {
        str(variant["name"]): Counter() for variant in VARIANTS
    }
    reuse_counts: dict[str, Counter[str]] = {
        str(variant["name"]): Counter() for variant in VARIANTS
    }
    session_correct = 0
    original_offline_correct = 0
    qa_ids: list[str] = []
    history_collision_count = 0
    history_correctness_conflict_count = 0

    loaded_items = load_all_locomo(dataset)
    for item in loaded_items:
        sample_id = str(item["sample_id"])
        loaded = load_locomo_conversation(dataset, sample_id)
        graph_path, source_path = _system_paths(sample_id)
        graph = _load_json(graph_path)
        source_by_id = _validate_source_rows(loaded, _load_json(source_path))
        history, _, collisions = _history_index(sample_id)
        history_collision_count += len(collisions)
        history_correctness_conflict_count += sum(
            bool(item["correctness_conflict"]) for item in collisions
        )
        session_by_id = _baseline_index(sample_id, "session")

        for source in loaded["qa"]:
            qa_id = str(source["qa_id"])
            qa_ids.append(qa_id)
            plan = plans[qa_id]
            base = source_by_id[qa_id]
            if str(plan["operation"]) not in OPERATION_V3:
                raise ValueError(f"unsupported semantic operation: {qa_id}")
            if _sha256_text(str(source["question"])) != next(
                row["question_sha256"]
                for row in plan_artifact["rows"]
                if row["qa_id"] == qa_id
            ):
                raise ValueError(f"semantic plan question mismatch: {qa_id}")

            sanitized_base = {
                "question": str(source["question"]),
                "retrieved": str(base["retrieved"]),
                "retrieval_trace": copy.deepcopy(base["retrieval_trace"]),
            }
            session_row = session_by_id[qa_id]
            session_value = int(session_row["correct"])
            original_value = int(base["correct"])
            session_correct += session_value
            original_offline_correct += original_value

            # Materialize each distinct policy before any historical outcome
            # is inspected.  Context choices below consume only `plan`.
            contexts = {
                policy: _materialize(policy, sanitized_base, graph)
                for policy in ALLOWED_CONTEXT_POLICIES
            }
            for variant in VARIANTS:
                name = str(variant["name"])
                policy = select_context_policy(variant, plan)
                context = contexts[policy]
                context_sha256 = _sha256_text(context)
                match = history.get((qa_id, context_sha256))
                if match is None or str(match[0].get("retrieved", "")) != context:
                    raise RuntimeError(
                        f"no exact frozen Offline history for {name}/{qa_id}/{policy}"
                    )
                historical, reuse_source = match
                result_rows[name].append(
                    {
                        "sample_id": sample_id,
                        "qa_id": qa_id,
                        "semantic": {
                            field: copy.deepcopy(plan[field])
                            for field in ROUTING_FIELDS
                        },
                        "context_policy": policy,
                        "context_sha256": context_sha256,
                        "reuse_source": reuse_source,
                        "correct": int(historical["correct"]),
                        "session_correct": session_value,
                        "original_offline_correct": original_value,
                    }
                )
                policy_counts[name][policy] += 1
                reuse_counts[name][reuse_source] += 1

    if len(qa_ids) != 1540 or len(set(qa_ids)) != 1540:
        raise AssertionError("full corpus is not exactly 1,540 unique QA rows")
    if session_correct != 1227 or original_offline_correct != 1236:
        raise AssertionError("frozen baseline drift")

    reference_rows = result_rows["p00_operation_v3"]
    reference_correct = sum(int(row["correct"]) for row in reference_rows)
    if reference_correct != 1249:
        raise AssertionError("operation-v3 reference drift")

    variants: list[dict[str, Any]] = []
    for definition in VARIANTS:
        name = str(definition["name"])
        rows = result_rows[name]
        correct = sum(int(row["correct"]) for row in rows)
        vs_session = _paired_summary(rows, "session_correct", session_correct)
        vs_offline = _paired_summary(
            rows, "original_offline_correct", original_offline_correct
        )
        v3_wins = sum(
            int(row["correct"]) > int(ref["correct"])
            for row, ref in zip(rows, reference_rows, strict=True)
        )
        v3_losses = sum(
            int(row["correct"]) < int(ref["correct"])
            for row, ref in zip(rows, reference_rows, strict=True)
        )
        variants.append(
            {
                **copy.deepcopy(definition),
                "n": len(rows),
                "correct": correct,
                "accuracy": 100.0 * correct / len(rows),
                "vs_session": vs_session,
                "vs_original_offline": vs_offline,
                "vs_operation_v3": {
                    "n": len(rows),
                    "candidate_correct": correct,
                    "baseline_correct": reference_correct,
                    "delta_correct": correct - reference_correct,
                    "delta_pp": 100.0 * (correct - reference_correct) / len(rows),
                    "wins": v3_wins,
                    "losses": v3_losses,
                    "ties": len(rows) - v3_wins - v3_losses,
                },
                "context_policy_counts": dict(sorted(policy_counts[name].items())),
                "reuse_source_counts": dict(sorted(reuse_counts[name].items())),
                "exact_history_matches": len(rows),
                "rows_sha256": canonical_sha256(rows),
            }
        )

    best = max(variants, key=lambda value: (value["correct"], -len(value["name"])))
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "locomo10_full1540_repeated_engineering_development",
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": file_sha256(dataset),
        "semantic_plan_path": str(plan_path.resolve()),
        "semantic_plan_sha256": file_sha256(plan_path),
        "qa_ids_sha256": canonical_sha256(qa_ids),
        "candidate_family_fixed_before_evaluation": True,
        "candidate_count": len(VARIANTS),
        "routing_fields": list(ROUTING_FIELDS),
        "allowed_context_policies": sorted(ALLOWED_CONTEXT_POLICIES),
        "operation_v3": copy.deepcopy(OPERATION_V3),
        "variants": variants,
        "best_development_variant": copy.deepcopy(best),
        "baselines": {
            "session": {
                "correct": session_correct,
                "n": 1540,
                "accuracy": 100.0 * session_correct / 1540,
            },
            "original_offline": {
                "correct": original_offline_correct,
                "n": 1540,
                "accuracy": 100.0 * original_offline_correct / 1540,
            },
            "operation_v3": {
                "correct": reference_correct,
                "n": 1540,
                "accuracy": 100.0 * reference_correct / 1540,
            },
        },
        "audit": {
            "all_variants_full_1540": all(value["n"] == 1540 for value in variants),
            "all_contexts_fixed_before_history_lookup": True,
            "exact_historical_reuse_only": True,
            "new_reader_judge_calls": 0,
            "literal_question_word_rules": False,
            "question_text_consumed_for_routing": False,
            "benchmark_category_consumed_for_routing": False,
            "answer_gold_evidence_consumed_for_routing": False,
            "prior_response_or_judge_consumed_for_routing": False,
            "post_hoc_per_qa_selection": False,
            "offline_contexts_only": True,
            "session_or_online_context_used": False,
            "final_fact_top_k_locked": 20,
            "full1540_used_for_policy_development": True,
            "unseen_confirmatory_result": False,
            "history_collision_count": history_collision_count,
            "history_correctness_conflict_count": history_correctness_conflict_count,
            "history_conflicts_use_frozen_global_precedence": True,
        },
    }
    value["deterministic_payload_sha256"] = canonical_sha256(
        {
            "qa_ids_sha256": value["qa_ids_sha256"],
            "routing_fields": value["routing_fields"],
            "operation_v3": value["operation_v3"],
            "variants": value["variants"],
            "baselines": value["baselines"],
            "audit": value["audit"],
        }
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _dump_json(output_root / "aggregate.json", value)
    (output_root / "REPORT.md").write_text(
        _report(value), encoding="utf-8", newline="\n"
    )
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    value = run_screen(
        dataset=args.dataset.resolve(),
        plan_path=args.plan_path.resolve(),
        output_root=args.output_root.resolve(),
    )
    best = value["best_development_variant"]
    print(f"METRIC accuracy={best['accuracy']:.12f}")
    print(f"METRIC correct={best['correct']}")
    print(f"METRIC delta_vs_session_pp={best['vs_session']['delta_pp']:.12f}")
    print(f"ARTIFACT {args.output_root.resolve() / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

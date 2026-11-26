"""Operation-routed Offline reader contexts for CreaMem3 v8.

This is a frozen, full-corpus *engineering* policy.  Routing consumes only
the strict semantic planner's ``operation`` label.  It does not inspect the
question text, LoCoMo category, reference answer, evidence annotations,
previous responses, or judge verdicts.

All branches stay inside the Offline memory system and retain the original
Offline Top-20 fact selection.  They either preserve the complete Offline
context, remove only cross-session event-view prose, or keep at most the
first two already query-ranked event-view lines.  Session summaries and
atomic-fact lines are byte-preserved by the deterministic transformer.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from creamem3.graph_memory_v2 import canonical_sha256
from creamem3.semantic_context_v8 import materialize_offline_semantic_context
from creamem3.semantic_query_planner_v6 import OPERATIONS


OPERATION_CONTEXT_SCHEMA_VERSION = "creamem3.offline_operation_context.v8.1"
POLICY_NAME = "offline_semantic_operation_context_v3"

ORIGINAL_FULL = "original_offline_full"
BOUNDED_VIEWS_2 = "bounded_query_ranked_views_2"
ROUTE_ONLY = "route_only"

# Frozen after inspecting full-corpus development results.  This mapping is
# intentionally small, semantic, and operation-only.  It contains no surface
# tokens and no benchmark-specific categories.
OPERATION_TO_CONTEXT_POLICY: dict[str, str] = {
    "atomic_lookup": ORIGINAL_FULL,
    "count": ORIGINAL_FULL,
    "explain_cause": ORIGINAL_FULL,
    "hypothetical": ORIGINAL_FULL,
    "compare": BOUNDED_VIEWS_2,
    "enumerate": BOUNDED_VIEWS_2,
    "trace_change": BOUNDED_VIEWS_2,
    "summarize_state": ROUTE_ONLY,
    "temporal_point": ROUTE_ONLY,
    "temporal_duration": ROUTE_ONLY,
    "temporal_order": ROUTE_ONLY,
}
if set(OPERATION_TO_CONTEXT_POLICY) != set(OPERATIONS):
    raise RuntimeError("Offline operation policy does not cover planner schema")

POLICY_RATIONALE: dict[str, str] = {
    ORIGINAL_FULL: (
        "localized_count_explanation_or_hypothesis_keeps_complete_offline_memory"
    ),
    BOUNDED_VIEWS_2: (
        "comparison_enumeration_or_change_trace_keeps_two_query_ranked_event_hints"
    ),
    ROUTE_ONLY: (
        "broad_state_or_temporal_reasoning_uses_dated_summaries_and_atomic_facts_without_duplicate_event_prose"
    ),
}

_ALLOWED_BASE_KEYS = frozenset({"question", "retrieved", "retrieval_trace"})
_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "answer",
        "category",
        "category_name",
        "correct",
        "evidence",
        "evidence_dia_ids",
        "gold",
        "judge_raw",
        "response",
        "verdict",
    }
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_sanitized_base(base: Mapping[str, Any]) -> None:
    unexpected = set(base).difference(_ALLOWED_BASE_KEYS)
    if unexpected:
        raise ValueError(
            "sanitized base has forbidden/unexpected top-level fields: "
            + ", ".join(sorted(str(value) for value in unexpected))
        )
    missing = _ALLOWED_BASE_KEYS.difference(base)
    if missing:
        raise ValueError(
            "sanitized base is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if not isinstance(base.get("retrieval_trace"), Mapping):
        raise ValueError("sanitized base retrieval_trace must be a mapping")

    stack: list[Any] = [base["retrieval_trace"]]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            bad = {
                str(key).casefold()
                for key in value
                if str(key).casefold() in _FORBIDDEN_FIELD_NAMES
            }
            if bad:
                raise ValueError(
                    "sanitized retrieval trace contains evaluation fields: "
                    + ", ".join(sorted(bad))
                )
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def context_policy_for_operation(operation: str) -> str:
    """Return the frozen Offline context policy for one strict operation."""

    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError(f"invalid strict semantic operation: {operation!r}")
    try:
        return OPERATION_TO_CONTEXT_POLICY[operation]
    except KeyError as exc:  # protects against schema drift
        raise ValueError(f"operation has no Offline context policy: {operation}") from exc


def _synthetic_budget_plan(event_budget: int) -> dict[str, Any]:
    return {
        "result": {
            "source": POLICY_NAME,
            "event_budget": event_budget,
        }
    }


def _original_full_audit(
    sanitized_base: Mapping[str, Any],
) -> dict[str, Any]:
    trace = sanitized_base["retrieval_trace"]
    rerank = trace.get("rerank", {}) if isinstance(trace, Mapping) else {}
    assembly = trace.get("assembly", {}) if isinstance(trace, Mapping) else {}
    selected = (
        [str(value) for value in rerank.get("selected_fact_ids", [])]
        if isinstance(rerank, Mapping)
        else []
    )
    rendered_views = (
        [str(value) for value in assembly.get("rendered_event_view_ids", [])]
        if isinstance(assembly, Mapping)
        else []
    )
    rendered_summaries = (
        [str(value) for value in assembly.get("rendered_session_summary_ids", [])]
        if isinstance(assembly, Mapping)
        else []
    )
    return {
        "selected_fact_ids": selected,
        "selected_fact_count": len(selected),
        "selection_mode": "frozen_original_offline_top20",
        "new_retrieval_or_reranker_called": False,
        "rendered_event_view_ids": rendered_views,
        "rendered_event_view_count": len(rendered_views),
        "rendered_session_summary_ids": rendered_summaries,
        "reused_original_context": True,
        "atomic_fact_lines_preserved": True,
        "non_event_text_preserved": True,
    }


def materialize_offline_operation_context(
    operation: str,
    sanitized_base: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Fix one Offline context before reader execution or history lookup.

    The materializer is deterministic and model-free.  The question field is
    accepted only so every system uses the same sanitized input envelope; it
    is never read by Offline routing or rendering.
    """

    _assert_sanitized_base(sanitized_base)
    policy = context_policy_for_operation(operation)
    original = str(sanitized_base["retrieved"])

    if policy == ORIGINAL_FULL:
        context = original
        branch_audit = _original_full_audit(sanitized_base)
        event_budget: int | None = None
    else:
        event_budget = 2 if policy == BOUNDED_VIEWS_2 else 0
        context, semantic_audit = materialize_offline_semantic_context(
            {
                "question": str(sanitized_base["question"]),
                "retrieved": original,
                "retrieval_trace": copy.deepcopy(
                    sanitized_base["retrieval_trace"]
                ),
            },
            graph,
            _synthetic_budget_plan(event_budget),
            strict=True,
        )
        transform = semantic_audit.get("transform", {})
        branch_audit = {
            **semantic_audit,
            "reused_original_context": False,
            "delegated_offline_event_budget": event_budget,
            "selected_fact_ids": list(transform.get("selected_fact_ids", [])),
            "selected_fact_count": int(transform.get("selected_fact_count", 0)),
            "rendered_event_view_ids": list(
                transform.get("kept_event_view_ids", [])
            ),
            "rendered_event_view_count": int(
                transform.get("kept_event_view_count", 0)
            ),
            "rendered_session_summary_ids": list(
                transform.get("rendered_session_summary_ids", [])
            ),
            "atomic_fact_lines_preserved": bool(
                transform.get("atomic_fact_lines_preserved")
            ),
            "non_event_text_preserved": bool(
                transform.get("non_event_text_preserved")
            ),
            "new_retrieval_or_reranker_called": False,
        }

    audit = {
        **branch_audit,
        "schema_version": OPERATION_CONTEXT_SCHEMA_VERSION,
        "policy_name": POLICY_NAME,
        "system": "offline",
        "operation": operation,
        "operation_sha256": _sha256_text(operation),
        "context_policy": policy,
        "policy_rationale": POLICY_RATIONALE[policy],
        "operation_mapping_sha256": canonical_sha256(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "input_context_sha256": _sha256_text(original),
        "context_sha256": _sha256_text(context),
        "fields_consumed_for_routing": ["operation"],
        "question_text_consumed_for_routing": False,
        "question_text_consumed_for_rendering": False,
        "strict_operation_only": True,
        "lexical_gate_used": False,
        "what_how_when_regex_used": False,
        "temporal_regex_path_called": False,
        "gold_or_category_visible": False,
        "prior_response_or_verdict_visible": False,
        "post_hoc_answer_selection": False,
        "single_context_fixed_before_answer": True,
        "offline_memory_only": True,
        "session_or_online_context_used": False,
        "engineering_policy_after_full_corpus_inspection": True,
        "unseen_confirmatory_result": False,
        "bounded_event_view_limit": event_budget,
    }
    if not audit["atomic_fact_lines_preserved"]:
        raise AssertionError("Offline operation policy changed atomic facts")
    if not audit["non_event_text_preserved"]:
        raise AssertionError("Offline operation policy changed non-event text")
    if policy == BOUNDED_VIEWS_2 and int(
        audit.get("rendered_event_view_count", 0)
    ) > 2:
        raise AssertionError("bounded Offline policy kept more than two views")
    if policy == ROUTE_ONLY and int(
        audit.get("rendered_event_view_count", 0)
    ) != 0:
        raise AssertionError("route-only Offline policy kept an event view")
    return context, audit


__all__ = [
    "BOUNDED_VIEWS_2",
    "OPERATION_CONTEXT_SCHEMA_VERSION",
    "OPERATION_TO_CONTEXT_POLICY",
    "ORIGINAL_FULL",
    "POLICY_NAME",
    "ROUTE_ONLY",
    "context_policy_for_operation",
    "materialize_offline_operation_context",
]

"""Operation-routed Online reader contexts for CreaMem3 v8.

This is the third, frozen engineering policy derived after inspecting the
full-corpus semantic-context s01 result.  Routing consumes exactly one field:
the strict semantic planner's ``operation`` label.  It never reads question
surface forms, benchmark categories, reference answers, evidence labels,
prior responses, or verdicts.

The policy intentionally stays inside the original event-memory retrieval
design.  It either preserves the original Online context byte-for-byte,
renders frozen Top-20 facts without an event view, or exposes at most two
query-ranked event views over those same frozen Top-20 facts.  Each selected
view retains the existing renderer's cap of three fully grounded claims; this
is a two-*view* policy, not a global two-claim policy.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from creamem3.graph_memory_v2 import canonical_sha256
from creamem3.semantic_context_v8 import materialize_online_semantic_context
from creamem3.semantic_query_planner_v6 import OPERATIONS


OPERATION_CONTEXT_SCHEMA_VERSION = "creamem3.online_operation_context.v8.1"
POLICY_NAME = "online_semantic_operation_context_v3"

ORIGINAL_FULL = "original_full"
BOUNDED_VIEWS_2 = "bounded_grounded_views_2"
ROUTE_ONLY = "route_only"
HISTORICAL_TWO_VIEW_SOURCE_SHA256 = (
    "a11b945988a26075dcf8599882502f456e34a8d2ce86d19a401438f7a4a20037"
)

# Frozen before the v3 run.  The mapping is semantic-label based and contains
# no question tokens or benchmark-specific categories.
OPERATION_TO_CONTEXT_POLICY: dict[str, str] = {
    "atomic_lookup": ORIGINAL_FULL,
    "count": ORIGINAL_FULL,
    "explain_cause": ORIGINAL_FULL,
    "summarize_state": BOUNDED_VIEWS_2,
    "enumerate": ROUTE_ONLY,
    "compare": ROUTE_ONLY,
    "temporal_point": ROUTE_ONLY,
    "temporal_duration": ROUTE_ONLY,
    "temporal_order": ROUTE_ONLY,
    "trace_change": ROUTE_ONLY,
    "hypothetical": ROUTE_ONLY,
}
if set(OPERATION_TO_CONTEXT_POLICY) != set(OPERATIONS):
    raise RuntimeError("operation context policy does not cover planner schema exactly")

POLICY_RATIONALE: dict[str, str] = {
    ORIGINAL_FULL: "localized_or_explanatory_request_preserves_full_event_memory",
    BOUNDED_VIEWS_2: "summary_synthesis_receives_up_to_two_compact_grounded_event_views",
    ROUTE_ONLY: "enumeration_temporal_or_relational_request_uses_atomic_facts_without_event_prose",
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
    # Retrieval traces can be nested.  Reject evaluation-like field names at
    # any depth so the routing/materialization boundary is auditable.
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
    """Return the frozen v3 context policy for one strict operation label."""

    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError(f"invalid strict semantic operation: {operation!r}")
    try:
        return OPERATION_TO_CONTEXT_POLICY[operation]
    except KeyError as exc:  # protects against planner/policy schema drift
        raise ValueError(f"operation has no frozen context policy: {operation}") from exc


def _synthetic_budget_plan(event_budget: int) -> dict[str, Any]:
    """Minimal plan accepted by the grounded v8 materializer.

    No actual plan descriptor except the already-routed operation reaches the
    materializer; in particular, no lexical fallback or secondary type rule
    can affect the context.
    """

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
        "selection_mode": "frozen_original_online",
        "new_reranker_called": False,
        "rendered_event_view_ids": rendered_views,
        "rendered_event_view_count": len(rendered_views),
        "rendered_session_summary_ids": rendered_summaries,
        "reused_original_context": True,
        "grounded_claims_only": None,
    }


def materialize_online_operation_context(
    operation: str,
    sanitized_base: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Choose and materialize one Online context before answer generation.

    Parameters are deliberately narrow: one strict semantic operation, one
    sanitized Online source row, and its typed memory graph.  The function is
    deterministic and performs neither model calls nor answer selection.
    """

    _assert_sanitized_base(sanitized_base)
    policy = context_policy_for_operation(operation)
    original = str(sanitized_base["retrieved"])

    if policy == ORIGINAL_FULL:
        context = original
        branch_audit = _original_full_audit(sanitized_base)
    else:
        budget = 2 if policy == BOUNDED_VIEWS_2 else 0
        context, semantic_audit = materialize_online_semantic_context(
            {"question": str(sanitized_base["question"])},
            {
                "retrieved": original,
                "retrieval_trace": copy.deepcopy(
                    sanitized_base["retrieval_trace"]
                ),
            },
            graph,
            _synthetic_budget_plan(budget),
        )
        branch_audit = {
            **semantic_audit,
            "reused_original_context": False,
            "delegated_grounded_event_budget": budget,
        }

    mapping_sha256 = canonical_sha256(OPERATION_TO_CONTEXT_POLICY)
    audit = {
        **branch_audit,
        "schema_version": OPERATION_CONTEXT_SCHEMA_VERSION,
        "policy_name": POLICY_NAME,
        "operation": operation,
        "operation_sha256": _sha256_text(operation),
        "context_policy": policy,
        "policy_rationale": POLICY_RATIONALE[policy],
        "operation_mapping_sha256": mapping_sha256,
        "input_context_sha256": _sha256_text(original),
        "context_sha256": _sha256_text(context),
        "fields_consumed_for_routing": ["operation"],
        "question_text_consumed_for_routing": False,
        "question_text_used_only_for_claim_relevance": (
            policy == BOUNDED_VIEWS_2
        ),
        "strict_operation_only": True,
        "lexical_gate_used": False,
        "what_how_when_regex_used": False,
        "temporal_regex_path_called": False,
        "gold_or_category_visible": False,
        "prior_response_or_verdict_visible": False,
        "post_hoc_answer_selection": False,
        "single_context_fixed_before_answer": True,
        "engineering_policy_after_semantic_s01": True,
        "unseen_confirmatory_result": False,
        "historical_two_view_result_source_sha256": (
            HISTORICAL_TWO_VIEW_SOURCE_SHA256
        ),
        "bounded_event_view_limit": (
            2 if policy == BOUNDED_VIEWS_2 else (0 if policy == ROUTE_ONLY else None)
        ),
        "bounded_claim_limit_per_view": (
            3 if policy == BOUNDED_VIEWS_2 else None
        ),
        "global_bounded_claim_limit": None,
    }
    if audit["context_sha256"] != _sha256_text(context):
        raise AssertionError("operation context hash changed during audit")
    if policy == BOUNDED_VIEWS_2:
        if int(audit.get("rendered_event_view_count", 0)) > 2:
            raise AssertionError("bounded-view policy rendered more than two views")
        if audit.get("grounded_claims_only") is not True:
            raise AssertionError("bounded-view policy rendered an ungrounded claim")
    if policy == ROUTE_ONLY and int(audit.get("rendered_event_view_count", 0)) != 0:
        raise AssertionError("route-only policy rendered an event view")
    return context, audit


__all__ = [
    "BOUNDED_VIEWS_2",
    "HISTORICAL_TWO_VIEW_SOURCE_SHA256",
    "OPERATION_CONTEXT_SCHEMA_VERSION",
    "OPERATION_TO_CONTEXT_POLICY",
    "ORIGINAL_FULL",
    "POLICY_NAME",
    "ROUTE_ONLY",
    "context_policy_for_operation",
    "materialize_online_operation_context",
]

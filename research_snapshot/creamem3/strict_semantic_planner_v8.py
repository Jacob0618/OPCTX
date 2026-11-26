"""Strict dataset-agnostic semantic planning for CreaMem3 v8.

Unlike the historical deterministic fallback, this module never inspects
question words or regular-expression cues.  A valid model plan is mapped to a
small event-evidence budget through four axes that were stable in the neutral
paraphrase screen.  Invalid model output receives one fixed neutral plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from creamem3.semantic_query_planner_v6 import (
    QueryPlan,
    build_planner_prompt,
    parse_query_plan,
)


STRICT_SCHEMA_VERSION = "creamem3.strict_semantic_plan.v8.2"


NEUTRAL_PLAN = QueryPlan(
    operation="atomic_lookup",
    scope="one_entity_state",
    evidence_cardinality="one",
    temporal_mode="none",
    abstraction="concrete",
    answer_shape="value",
    confidence=0.0,
)


OPERATION_WEIGHT = {
    "atomic_lookup": 0,
    "temporal_point": 0,
    "temporal_duration": 1,
    "temporal_order": 1,
    "hypothetical": 1,
    "explain_cause": 2,
    "count": 3,
    "enumerate": 3,
    "compare": 3,
    "trace_change": 4,
    "summarize_state": 4,
}
CARDINALITY_WEIGHT = {"one": 0, "few": 1, "many": 2}
ABSTRACTION_WEIGHT = {"concrete": 0, "relational": 1, "narrative": 2}
ANSWER_SHAPE_WEIGHT = {
    "entity": 0,
    "value": 0,
    "date": 0,
    "interval": 0,
    "explanation": 1,
    "count": 2,
    "set": 2,
    "trajectory": 2,
}


@dataclass(frozen=True)
class StrictPlanningResult:
    plan: QueryPlan
    source: str
    event_need_score: int
    event_budget: int
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRICT_SCHEMA_VERSION,
            "plan": asdict(self.plan),
            "source": self.source,
            "event_need_score": self.event_need_score,
            "event_budget": self.event_budget,
            "error": self.error,
        }


def event_need_score(plan: QueryPlan) -> int:
    """Map only preregistered stable semantic axes to an integer score."""

    return (
        OPERATION_WEIGHT[plan.operation]
        + CARDINALITY_WEIGHT[plan.evidence_cardinality]
        + ABSTRACTION_WEIGHT[plan.abstraction]
        + ANSWER_SHAPE_WEIGHT[plan.answer_shape]
    )


def event_budget_from_score(score: int) -> int:
    if score <= 2:
        return 0
    if score <= 5:
        return 1
    return 2


def strict_plan_from_raw(raw_response: str) -> StrictPlanningResult:
    try:
        plan = parse_query_plan(raw_response)
        score = event_need_score(plan)
        return StrictPlanningResult(
            plan=plan,
            source="gemini",
            event_need_score=score,
            event_budget=event_budget_from_score(score),
        )
    except Exception as exc:
        score = event_need_score(NEUTRAL_PLAN)
        return StrictPlanningResult(
            plan=NEUTRAL_PLAN,
            source="neutral_failure",
            event_need_score=score,
            event_budget=0,
            error=f"{type(exc).__name__}: {str(exc)[-500:]}",
        )


def classify_strict(
    question: str,
    *,
    chat_fn: Callable[..., str],
    model: str = "gemini-2.5-flash",
) -> tuple[StrictPlanningResult, str]:
    """Call the model once; failures use a fixed neutral plan, never regex."""

    try:
        raw = chat_fn(
            build_planner_prompt(question),
            model=model,
            temperature=0.0,
            max_tokens=512,
            thinking_budget=0,
            tag="retrieve",
            use_cache=True,
            json_mode=True,
        )
    except Exception as exc:
        score = event_need_score(NEUTRAL_PLAN)
        return (
            StrictPlanningResult(
                plan=NEUTRAL_PLAN,
                source="neutral_failure",
                event_need_score=score,
                event_budget=0,
                error=f"{type(exc).__name__}: {str(exc)[-500:]}",
            ),
            "",
        )
    return strict_plan_from_raw(raw), raw


__all__ = [
    "NEUTRAL_PLAN",
    "STRICT_SCHEMA_VERSION",
    "StrictPlanningResult",
    "classify_strict",
    "event_budget_from_score",
    "event_need_score",
    "strict_plan_from_raw",
]

"""Semantic-plan-controlled reader contexts for CreaMem3 v8.

This module consumes only the strict semantic planner's small event budget.
It never infers intent from question words and never calls the lexical
temporal gate in :mod:`creamem3.retrieval_ablation_v4`.

Online contexts are rebuilt from the frozen model Top-20 fact IDs.  Budget
zero is route-only; budgets one and two may expose at most that many event
views, and every rendered claim must be fully supported by a selected fact.
Offline contexts retain the first N event-view lines in the already frozen,
query-ranked rendering order.  Every non-event line is byte-preserved.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

from creamem3.graph_memory_v2 import canonical_sha256
from creamem3.offline_context_v6 import (
    OfflineContextPolicy,
    transform_offline_context,
)
from creamem3.retrieval_ablation_v4 import AblationConfig


SEMANTIC_CONTEXT_SCHEMA_VERSION = "creamem3.semantic_context.v8.1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_TEMPORAL_CONFIG_FIELDS = (
    "temporal_selection_mode",
    "temporal_final_k",
    "temporal_session_render",
    "temporal_view_render",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if is_dataclass(value):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return converted
    raise TypeError("semantic_plan must be a mapping or serializable plan result")


def semantic_event_budget(
    semantic_plan: Any,
) -> tuple[int, dict[str, Any], str]:
    """Return ``(budget, normalized_result, result_sha256)``.

    Rows emitted by ``run_semantic_plans_v8.py`` wrap the actual strict result
    under ``result``; direct ``StrictPlanningResult`` values and result
    dictionaries are accepted as well.  Only budgets 0, 1, and 2 are valid.
    """

    outer = _as_mapping(semantic_plan)
    nested = outer.get("result")
    payload = dict(_as_mapping(nested)) if nested is not None else dict(outer)
    raw_budget = payload.get("event_budget")
    if isinstance(raw_budget, bool) or not isinstance(raw_budget, int):
        raise ValueError("semantic plan event_budget must be an integer")
    if raw_budget not in {0, 1, 2}:
        raise ValueError("semantic plan event_budget must be one of 0, 1, or 2")
    return raw_budget, payload, canonical_sha256(payload)


def online_config_for_budget(event_budget: int) -> AblationConfig:
    """Build the frozen-Top20 Online policy for one semantic event budget."""

    if isinstance(event_budget, bool) or event_budget not in {0, 1, 2}:
        raise ValueError("event_budget must be one of 0, 1, or 2")
    view_render = "none" if event_budget == 0 else "selected_claims"
    config = AblationConfig(
        name=f"v8_semantic_online_budget_{event_budget}",
        description=(
            "Strict semantic-plan budget over frozen model Top-20 facts; "
            f"at most {event_budget} grounded event views"
        ),
        selection_mode="model",
        final_k=20,
        session_render="full",
        view_render=view_render,
        original_context=False,
        max_views=event_budget,
        max_view_chars_total=300 * event_budget,
        max_claim_chars=260,
        temporal_selection_mode=None,
        temporal_final_k=None,
        temporal_session_render=None,
        temporal_view_render=None,
    )
    _assert_no_temporal_override(config)
    return config


def _assert_no_temporal_override(config: AblationConfig) -> None:
    configured = {
        field: getattr(config, field) for field in _TEMPORAL_CONFIG_FIELDS
    }
    if any(value is not None for value in configured.values()):
        raise ValueError(
            "semantic context policies forbid all lexical temporal overrides"
        )


def _tokens(text: str) -> set[str]:
    """Content-overlap tokens; never used to infer a question type."""

    return {token.casefold() for token in _TOKEN_RE.findall(str(text))}


def _ordered_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output


def _frozen_top20_ids(
    base_row: Mapping[str, Any], graph: Mapping[str, Any]
) -> tuple[list[str], str]:
    trace = base_row.get("retrieval_trace", {})
    if not isinstance(trace, Mapping):
        raise ValueError("base_row.retrieval_trace must be a mapping")
    rerank = trace.get("rerank", {})
    if not isinstance(rerank, Mapping):
        raise ValueError("base_row retrieval trace lacks a frozen rerank record")
    raw_ids = rerank.get("selected_fact_ids", [])
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("frozen selected_fact_ids must be a sequence")
    ids = _ordered_unique(str(value) for value in raw_ids)[:20]
    fact_ids = {
        str(fact.get("id", ""))
        for fact in graph.get("fact_nodes", [])
        if isinstance(fact, Mapping)
    }
    missing = [fact_id for fact_id in ids if fact_id not in fact_ids]
    if missing:
        raise ValueError("frozen selected facts missing from graph: " + ", ".join(missing))
    if not ids:
        raise ValueError("frozen Top-20 selection is empty")
    source = "frozen_fallback" if bool(rerank.get("fallback")) else "frozen_model"
    return ids, source


def _candidate_view_ids(
    base_row: Mapping[str, Any],
    selected_ids: Sequence[str],
    view_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Use frozen query-ranked summaries, then deterministic support order."""

    selected = set(selected_ids)
    trace = base_row.get("retrieval_trace", {})
    top = trace.get("top_summaries", []) if isinstance(trace, Mapping) else []
    ordered: list[str] = []
    if isinstance(top, Sequence) and not isinstance(top, (str, bytes)):
        for item in top:
            if not isinstance(item, Mapping):
                continue
            view_id = str(item.get("summary_id", ""))
            view = view_by_id.get(view_id)
            if view is None:
                continue
            members = {str(value) for value in view.get("member_fact_ids", [])}
            if selected.intersection(members) and view_id not in ordered:
                ordered.append(view_id)
    remaining = sorted(
        (
            -len(
                selected.intersection(
                    str(value) for value in view.get("member_fact_ids", [])
                )
            ),
            view_id,
        )
        for view_id, view in view_by_id.items()
        if view_id not in ordered
        and len(
            selected.intersection(
                str(value) for value in view.get("member_fact_ids", [])
            )
        )
        >= 2
    )
    ordered.extend(view_id for _, view_id in remaining)
    return ordered


def _grounded_claims(
    view: Mapping[str, Any],
    selected_ids: set[str],
    question: str,
    *,
    max_claim_chars: int,
) -> list[dict[str, Any]]:
    """Return eligible claims ordered by content relevance and support size."""

    query_tokens = _tokens(question)
    ranked: list[tuple[int, int, int, dict[str, Any]]] = []
    claims = view.get("summary_claims", [])
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        return []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            continue
        text = str(claim.get("text", "")).strip()
        raw_support = claim.get("support_fact_ids", [])
        if not isinstance(raw_support, Sequence) or isinstance(
            raw_support, (str, bytes)
        ):
            continue
        support = _ordered_unique(str(value) for value in raw_support)
        support_set = set(support)
        if (
            not text
            or not support
            or not support_set.issubset(selected_ids)
            or len(text) > max_claim_chars
        ):
            continue
        normalized = {"text": text, "support_fact_ids": support, "index": index}
        ranked.append(
            (
                -len(_tokens(text).intersection(query_tokens)),
                -len(support),
                index,
                normalized,
            )
        )
    ranked.sort(key=lambda value: value[:3])
    return [item for _, _, _, item in ranked]


def _render_online_context(
    source_qa: Mapping[str, Any],
    base_row: Mapping[str, Any],
    graph: Mapping[str, Any],
    config: AblationConfig,
) -> tuple[str, dict[str, Any]]:
    """Render without calling any v4 intent gate or selection function."""

    _assert_no_temporal_override(config)
    question = str(source_qa.get("question", ""))
    selected_ids, selection_source = _frozen_top20_ids(base_row, graph)
    selected_set = set(selected_ids)
    fact_by_id = {
        str(fact["id"]): fact
        for fact in graph.get("fact_nodes", [])
        if isinstance(fact, Mapping) and fact.get("id")
    }
    session_summary = {
        int(node["session_ids"][0]): node
        for node in graph.get("summary_nodes", [])
        if isinstance(node, Mapping)
        and node.get("node_type") == "session_packet"
        and node.get("session_ids")
    }
    view_by_id = {
        str(node["id"]): node
        for node in graph.get("summary_nodes", [])
        if isinstance(node, Mapping)
        and node.get("node_type") == "event_view"
        and node.get("id")
    }

    lines: list[str] = []
    rendered_view_ids: list[str] = []
    rendered_claims: list[dict[str, Any]] = []
    rendered_view_chars = 0
    if config.view_render == "selected_claims":
        for view_id in _candidate_view_ids(base_row, selected_ids, view_by_id):
            if len(rendered_view_ids) >= config.max_views:
                break
            view = view_by_id[view_id]
            header = (
                f"[Selected cross-session claims | {view.get('view_type', '')} | "
                f"sessions {', '.join(str(value) for value in view.get('session_ids', []))}] "
            )
            separator_chars = 1 if rendered_view_ids else 0
            remaining = (
                config.max_view_chars_total
                - rendered_view_chars
                - separator_chars
                - len(header)
            )
            if remaining <= 0:
                continue
            eligible = _grounded_claims(
                view,
                selected_set,
                question,
                max_claim_chars=config.max_claim_chars,
            )
            chosen: list[dict[str, Any]] = []
            claim_chars = 0
            for claim in eligible:
                claim_text = str(claim["text"])
                separator = 1 if chosen else 0
                if claim_chars + separator + len(claim_text) > remaining:
                    continue
                chosen.append(claim)
                claim_chars += separator + len(claim_text)
                if len(chosen) >= 3:
                    break
            if not chosen:
                continue
            claim_text = " ".join(str(claim["text"]) for claim in chosen)
            rendered_line = header + claim_text
            lines.append(rendered_line)
            rendered_view_ids.append(view_id)
            rendered_view_chars += separator_chars + len(rendered_line)
            rendered_claims.extend(
                {
                    "view_id": view_id,
                    "claim_index": int(claim["index"]),
                    "text_sha256": _sha256_text(str(claim["text"])),
                    "support_fact_ids": list(claim["support_fact_ids"]),
                }
                for claim in chosen
            )
    elif config.view_render != "none":
        raise ValueError(f"unsupported view rendering: {config.view_render}")

    by_session: dict[int, list[Mapping[str, Any]]] = {}
    session_order: list[int] = []
    for fact_id in selected_ids:
        fact = fact_by_id[fact_id]
        session_id = int(fact["session_id"])
        if session_id not in by_session:
            by_session[session_id] = []
            session_order.append(session_id)
        by_session[session_id].append(fact)

    rendered_summary_ids: list[str] = []
    for session_id in session_order:
        if session_id not in session_summary:
            raise ValueError(f"session packet missing for selected session {session_id}")
        summary = session_summary[session_id]
        timestamp = str(summary.get("timestamp", ""))
        iso_date = str(summary.get("iso_date", ""))
        lines.append(
            f"[Session summary | session {session_id} | {timestamp}; ISO date {iso_date}] "
            f"{summary.get('text', '')}"
        )
        rendered_summary_ids.append(str(summary["id"]))
        for fact in by_session[session_id]:
            sources = ", ".join(str(value) for value in fact.get("source_dia_ids", []))
            lines.append(f"  - [{sources}] {fact.get('text', '')}")

    text = "\n".join(lines)
    grounded = all(
        set(item["support_fact_ids"]).issubset(selected_set)
        and bool(item["support_fact_ids"])
        for item in rendered_claims
    )
    return text, {
        "selected_fact_ids": selected_ids,
        "selected_fact_count": len(selected_ids),
        "selection_source": selection_source,
        "selection_mode": "frozen_model_top20",
        "new_reranker_called": False,
        "rendered_event_view_ids": rendered_view_ids,
        "rendered_event_view_count": len(rendered_view_ids),
        "rendered_event_view_claims": rendered_claims,
        "rendered_event_claim_count": len(rendered_claims),
        "rendered_event_view_chars": rendered_view_chars,
        "rendered_session_summary_ids": rendered_summary_ids,
        "grounded_claims_only": grounded,
    }


def _common_audit(
    *,
    mode: str,
    event_budget: int,
    plan_payload: Mapping[str, Any],
    plan_sha256: str,
    input_context: str,
    output_context: str,
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_CONTEXT_SCHEMA_VERSION,
        "mode": mode,
        "event_budget": event_budget,
        "semantic_plan_sha256": plan_sha256,
        "semantic_plan_source": str(plan_payload.get("source", "unknown")),
        "input_context_sha256": _sha256_text(input_context),
        "context_sha256": _sha256_text(output_context),
        "lexical_gate_used": False,
        "temporal_regex_path_called": False,
        "question_type_source": "strict_semantic_plan",
        "gold_or_category_visible": False,
    }


def materialize_online_semantic_context(
    source_qa: Mapping[str, Any],
    base_row: Mapping[str, Any],
    graph: Mapping[str, Any],
    semantic_plan: Any,
) -> tuple[str, dict[str, Any]]:
    """Build one Online context from a frozen Top-20 and semantic budget."""

    event_budget, plan_payload, plan_sha256 = semantic_event_budget(semantic_plan)
    config = online_config_for_budget(event_budget)
    text, renderer_audit = _render_online_context(
        source_qa, base_row, graph, config
    )
    temporal_fields = {
        field: getattr(config, field) for field in _TEMPORAL_CONFIG_FIELDS
    }
    if any(value is not None for value in temporal_fields.values()):
        raise AssertionError("lexical temporal override reached Online rendering")
    common = _common_audit(
        mode="online",
        event_budget=event_budget,
        plan_payload=plan_payload,
        plan_sha256=plan_sha256,
        input_context=str(base_row.get("retrieved", "")),
        output_context=text,
    )
    return text, {
        **common,
        "config": asdict(config),
        "config_sha256": config.sha256,
        "temporal_override_fields": temporal_fields,
        "temporal_overrides_all_none": True,
        "max_event_views": event_budget,
        **renderer_audit,
    }


def materialize_offline_semantic_context(
    base_row: Mapping[str, Any],
    graph: Mapping[str, Any],
    semantic_plan: Any,
    *,
    strict: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Cap frozen Offline event views by semantic budget, preserving all else."""

    event_budget, plan_payload, plan_sha256 = semantic_event_budget(semantic_plan)
    context = str(base_row.get("retrieved", ""))
    policy = OfflineContextPolicy(
        view_policy="cap",
        max_event_views=event_budget,
        session_summary_policy="all",
    )
    text, transform_audit = transform_offline_context(
        context,
        graph,
        base_row,
        policy,
        strict=strict,
    )
    if not transform_audit["atomic_fact_lines_preserved"]:
        raise AssertionError("Offline semantic cap changed an atomic fact line")
    if not transform_audit["non_event_text_preserved"]:
        raise AssertionError("Offline semantic cap changed non-event text")
    common = _common_audit(
        mode="offline",
        event_budget=event_budget,
        plan_payload=plan_payload,
        plan_sha256=plan_sha256,
        input_context=context,
        output_context=text,
    )
    return text, {
        **common,
        "offline_policy": asdict(policy),
        "offline_policy_sha256": policy.sha256,
        "view_order_source": "frozen_query_ranked_render_order",
        "max_event_views": event_budget,
        "transform": transform_audit,
    }


__all__ = [
    "SEMANTIC_CONTEXT_SCHEMA_VERSION",
    "materialize_offline_semantic_context",
    "materialize_online_semantic_context",
    "online_config_for_budget",
    "semantic_event_budget",
]

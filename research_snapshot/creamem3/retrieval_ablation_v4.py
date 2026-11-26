"""Deterministic retrieval/materialization ablations over frozen CreaMem3 v3 artifacts.

The v3 online graph is treated as immutable.  This module changes only how its
already-saved candidate facts are selected and how derived routing views are
shown to the reader.  In particular, a long event-view claim ledger is never
silently rewritten: route-only experiments hide it, while selected-claim
experiments expose a small extractive subset whose supporting facts are also
present in the final context.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from creamem3.graph_memory_v2 import canonical_sha256


ABLATION_SCHEMA_VERSION = "creamem3.retrieval_ablation.v4.1"
DEV_PANEL_SEED = "creamem3-v4-dev-20260805"
DEV_CONVERSATIONS = ("conv-26", "conv-41", "conv-43", "conv-47", "conv-49")
HOLDOUT_CONVERSATIONS = ("conv-30", "conv-42", "conv-44", "conv-48", "conv-50")
DEV_CATEGORY_QUOTAS = {
    "temporal": 10,
    "multi": 10,
    "open": 3,
    "single": 27,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TEMPORAL_RE = re.compile(
    r"\b(?:when|what\s+date|how\s+long|how\s+many\s+"
    r"(?:days?|weeks?|months?|years?)|before|after|between|ago|"
    r"yesterday|today|tomorrow|last\s+(?:day|week|month|year|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"first|most\s+recent)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    selection_mode: str = "model"
    final_k: int = 20
    session_render: str = "full"
    view_render: str = "none"
    original_context: bool = False
    rrf_summary_weight: float = 1.0
    rrf_dense_weight: float = 2.0
    rrf_bm25_weight: float = 1.0
    mmr_lambda: float = 0.85
    max_views: int = 3
    max_view_chars_total: int = 600
    max_claim_chars: int = 260
    max_session_sentence_chars: int = 320
    temporal_selection_mode: str | None = None
    temporal_final_k: int | None = None
    temporal_session_render: str | None = None
    temporal_view_render: str | None = None

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


BASELINE_CONFIG = AblationConfig(
    name="r00_original_online",
    description="Frozen full_v3_r1 online context and model rerank baseline",
    original_context=True,
    view_render="full",
)


SCREEN_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="r01_route_only_m20",
        description="Keep model Top-20 fact selection but hide event-view text",
    ),
    AblationConfig(
        name="r02_claimcap_m20",
        description="Model Top-20 plus selected grounded view claims capped at 600 chars",
        view_render="selected_claims",
    ),
    AblationConfig(
        name="r03_session_sentence_m20",
        description="Model Top-20, route-only views, one query-selected session sentence",
        session_render="selected_sentence",
    ),
    AblationConfig(
        name="r04_facts_anchor_m20",
        description="Model Top-20 facts only with explicit original and ISO session anchors",
        session_render="none",
    ),
    AblationConfig(
        name="r05_facts_claimcap_m20",
        description="Model Top-20 anchored facts plus bounded selected view claims",
        session_render="none",
        view_render="selected_claims",
    ),
    AblationConfig(
        name="r06_norerank_dense20_route",
        description="Disable model rerank; cosine Top-20 with route-only views",
        selection_mode="dense",
        final_k=20,
    ),
    AblationConfig(
        name="r07_norerank_dense30_route",
        description=(
            "Disable model rerank; frozen candidate-union cosine Top-30 "
            "with route-only views"
        ),
        selection_mode="dense",
        final_k=30,
    ),
    AblationConfig(
        name="r08_norerank_dense40_route",
        description=(
            "Disable model rerank; frozen candidate-union cosine Top-40 "
            "with route-only views"
        ),
        selection_mode="dense",
        final_k=40,
    ),
    AblationConfig(
        name="r09_norerank_dense30_facts",
        description=(
            "Disable model rerank; frozen candidate-union cosine Top-30 "
            "anchored facts only"
        ),
        selection_mode="dense",
        final_k=30,
        session_render="none",
    ),
    AblationConfig(
        name="r10_norerank_dense40_facts",
        description=(
            "Disable model rerank; frozen candidate-union cosine Top-40 "
            "anchored facts only"
        ),
        selection_mode="dense",
        final_k=40,
        session_render="none",
    ),
    AblationConfig(
        name="r11_norerank_rrf30_route",
        description="Disable model rerank; weighted RRF Top-30 with route-only views",
        selection_mode="rrf",
        final_k=30,
    ),
    AblationConfig(
        name="r12_norerank_rrf40_facts",
        description="Disable model rerank; weighted RRF Top-40 anchored facts only",
        selection_mode="rrf",
        final_k=40,
        session_render="none",
    ),
    AblationConfig(
        name="r13_norerank_mmr30_facts",
        description="Disable model rerank; diversity-aware MMR Top-30 anchored facts",
        selection_mode="mmr",
        final_k=30,
        session_render="none",
        mmr_lambda=0.85,
    ),
    AblationConfig(
        name="r14_temporal_dense40_facts",
        description=(
            "Question-text temporal-intent hits use frozen candidate-union "
            "cosine Top-40 anchored facts; others use model Top-20 route-only"
        ),
        temporal_selection_mode="dense",
        temporal_final_k=40,
        temporal_session_render="none",
        temporal_view_render="none",
    ),
    AblationConfig(
        name="r15_temporal_facts_claimcap",
        description=(
            "Question-text temporal-intent hits use frozen candidate-union "
            "cosine Top-40 facts; others use model Top-20 bounded claims"
        ),
        view_render="selected_claims",
        temporal_selection_mode="dense",
        temporal_final_k=40,
        temporal_session_render="none",
        temporal_view_render="none",
    ),
)

POST_SCREEN_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="ps01_route_only_m20_legacy_header",
        description=(
            "Post-screen isolation: frozen model Top-20, no event-view text, "
            "and the exact legacy timestamp-only session header"
        ),
        session_render="legacy_full",
    ),
    AblationConfig(
        name="ps02_nontemporal_claimcap_m20",
        description=(
            "Question-text temporal-intent hits stay route-only; other questions "
            "receive bounded grounded event claims with frozen model Top-20 facts"
        ),
        view_render="selected_claims",
        temporal_view_render="none",
    ),
)

ALL_CONFIGS = (BASELINE_CONFIG, *SCREEN_CONFIGS, *POST_SCREEN_CONFIGS)
CONFIG_BY_NAME = {config.name: config for config in ALL_CONFIGS}


def is_temporal_question(question: str) -> bool:
    """Question-text-only gate; evaluation category labels are never consulted."""

    return bool(_TEMPORAL_RE.search(str(question)))


def select_dev_panel(qa: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the preregistered 50-question stratified panel for one conversation."""

    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for row in qa:
        by_category.setdefault(str(row["category_name"]), []).append(row)
    selected: list[Mapping[str, Any]] = []
    for category, quota in DEV_CATEGORY_QUOTAS.items():
        rows = list(by_category.get(category, []))
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{DEV_PANEL_SEED}|{row['qa_id']}".encode("utf-8")
            ).hexdigest()
        )
        if len(rows) < quota:
            raise ValueError(
                f"dev panel lacks {category}: need {quota}, found {len(rows)}"
            )
        selected.extend(rows[:quota])
    source_order = {str(row["qa_id"]): index for index, row in enumerate(qa)}
    selected.sort(key=lambda row: source_order[str(row["qa_id"])])
    return [dict(row) for row in selected]


def config_manifest(config: AblationConfig) -> dict[str, Any]:
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "config": asdict(config),
        "config_sha256": config.sha256,
        "temporal_gate_uses_question_text_only": True,
        "event_view_claim_policy": (
            "extractive_only_support_subset_of_final_facts_skip_overlong_claims"
        ),
    }


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(text))}


def _candidate_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("retrieval_trace", {}).get("candidates", [])
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        fact_id = str(item.get("fact_id", ""))
        if not fact_id or fact_id in seen:
            continue
        seen.add(fact_id)
        output.append(dict(item))
    return output


def _rrf_score(item: Mapping[str, Any], config: AblationConfig) -> float:
    channels = item.get("channels", {})
    score = 0.0
    summary_ranks = channels.get("summary_ranks", [])
    if summary_ranks:
        score += config.rrf_summary_weight / (60.0 + min(summary_ranks))
    dense_rank = channels.get("dense_rank")
    if isinstance(dense_rank, int) and dense_rank > 0:
        score += config.rrf_dense_weight / (60.0 + dense_rank)
    bm25_rank = channels.get("bm25_rank")
    if isinstance(bm25_rank, int) and bm25_rank > 0:
        score += config.rrf_bm25_weight / (60.0 + bm25_rank)
    return score


def _mmr_ids(
    candidates: Sequence[Mapping[str, Any]],
    fact_by_id: Mapping[str, Mapping[str, Any]],
    *,
    final_k: int,
    lambda_value: float,
) -> list[str]:
    valid = [item for item in candidates if str(item.get("fact_id", "")) in fact_by_id]
    if not valid or final_k <= 0:
        return []
    matrix = np.asarray(
        [fact_by_id[str(item["fact_id"])]["embedding"] for item in valid],
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    relevance = np.asarray(
        [float(item.get("dense_score", 0.0)) for item in valid], dtype=np.float32
    )
    selected: list[int] = []
    remaining = set(range(len(valid)))
    while remaining and len(selected) < final_k:
        best_index = min(remaining)
        best_key: tuple[float, float, int] | None = None
        for index in sorted(remaining):
            redundancy = (
                max(float(matrix[index] @ matrix[chosen]) for chosen in selected)
                if selected
                else 0.0
            )
            mmr = lambda_value * float(relevance[index]) - (1.0 - lambda_value) * redundancy
            key = (mmr, float(relevance[index]), -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)
    return [str(valid[index]["fact_id"]) for index in selected]


def select_fact_ids(
    row: Mapping[str, Any],
    graph: Mapping[str, Any],
    config: AblationConfig,
) -> tuple[list[str], dict[str, Any]]:
    temporal = is_temporal_question(str(row.get("question", "")))
    mode = config.selection_mode
    final_k = config.final_k
    if temporal and config.temporal_selection_mode is not None:
        mode = config.temporal_selection_mode
        final_k = int(config.temporal_final_k or final_k)

    fact_by_id = {str(fact["id"]): fact for fact in graph.get("fact_nodes", [])}
    candidates = _candidate_rows(row)
    if mode == "model":
        frozen_rerank = row.get("retrieval_trace", {}).get("rerank", {})
        ids = [
            str(value)
            for value in frozen_rerank.get("selected_fact_ids", [])
            if str(value) in fact_by_id
        ][:final_k]
        selection_source = (
            "frozen_fallback"
            if bool(frozen_rerank.get("fallback"))
            else "frozen_model"
        )
    elif mode == "dense":
        ids = [str(item["fact_id"]) for item in candidates[:final_k]]
        selection_source = (
            "deterministic_global_dense"
            if final_k <= 20
            else "deterministic_candidate_union_cosine"
        )
    elif mode == "rrf":
        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: (
                -_rrf_score(pair[1], config),
                -float(pair[1].get("dense_score", 0.0)),
                pair[0],
            ),
        )
        ids = [str(item["fact_id"]) for _, item in ranked[:final_k]]
        selection_source = "deterministic_candidate_union_rrf"
    elif mode == "mmr":
        ids = _mmr_ids(
            candidates,
            fact_by_id,
            final_k=final_k,
            lambda_value=config.mmr_lambda,
        )
        selection_source = "deterministic_candidate_union_mmr"
    else:
        raise ValueError(f"unknown selection mode: {mode}")

    deduped: list[str] = []
    for fact_id in ids:
        if fact_id in fact_by_id and fact_id not in deduped:
            deduped.append(fact_id)
    return deduped, {
        "temporal_question_detected": temporal,
        "effective_selection_mode": mode,
        "effective_final_k": final_k,
        "selection_source": selection_source,
        "new_reranker_called": False,
        "model_rerank_enabled": selection_source == "frozen_model",
    }


def _bounded_sentence(text: str, question: str, limit: int) -> str:
    sentences = [item.strip() for item in _SENTENCE_RE.split(str(text)) if item.strip()]
    if not sentences:
        return ""
    query_tokens = _tokens(question)
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (
            -len(_tokens(pair[1]) & query_tokens),
            len(pair[1]) > limit,
            pair[0],
        ),
    )
    for _, sentence in ranked:
        if len(sentence) <= limit:
            return sentence
    return ""


def _view_ids_for_selected(
    row: Mapping[str, Any],
    selected_ids: Sequence[str],
    view_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    selected = set(selected_ids)
    ordered: list[str] = []
    top = row.get("retrieval_trace", {}).get("top_summaries", [])
    for item in top:
        view_id = str(item.get("summary_id", ""))
        view = view_by_id.get(view_id)
        if view is None:
            continue
        if selected.intersection(view.get("member_fact_ids", [])) and view_id not in ordered:
            ordered.append(view_id)
    remaining = sorted(
        (
            (
                -len(selected.intersection(view.get("member_fact_ids", []))),
                view_id,
            )
            for view_id, view in view_by_id.items()
            if view_id not in ordered
            and len(selected.intersection(view.get("member_fact_ids", []))) >= 2
        )
    )
    ordered.extend(view_id for _, view_id in remaining)
    return ordered


def _selected_claim_text(
    view: Mapping[str, Any],
    selected_ids: set[str],
    question: str,
    *,
    max_claim_chars: int,
    remaining_chars: int,
) -> str:
    query_tokens = _tokens(question)
    eligible: list[tuple[int, int, int, str]] = []
    for index, claim in enumerate(view.get("summary_claims", [])):
        if not isinstance(claim, Mapping):
            continue
        text = str(claim.get("text", "")).strip()
        support = {str(value) for value in claim.get("support_fact_ids", [])}
        if (
            not text
            or not support
            or not support.issubset(selected_ids)
            or len(text) > max_claim_chars
            or len(text) > remaining_chars
        ):
            continue
        eligible.append(
            (
                -len(_tokens(text) & query_tokens),
                -len(support),
                index,
                text,
            )
        )
    eligible.sort()
    chosen: list[str] = []
    used = 0
    for _, _, _, text in eligible:
        separator = 1 if chosen else 0
        if used + separator + len(text) > remaining_chars:
            continue
        chosen.append(text)
        used += separator + len(text)
        if len(chosen) >= 3:
            break
    return " ".join(chosen)


def materialize_context(
    row: Mapping[str, Any],
    graph: Mapping[str, Any],
    config: AblationConfig,
) -> tuple[str, dict[str, Any]]:
    if config.original_context:
        rerank_trace = row.get("retrieval_trace", {}).get("rerank", {})
        selected = list(
            rerank_trace.get("selected_fact_ids", [])
        )
        selection_source = (
            "frozen_fallback"
            if bool(rerank_trace.get("fallback"))
            else "frozen_model"
        )
        view_by_id = {
            str(node["id"]): node
            for node in graph.get("summary_nodes", [])
            if node.get("node_type") == "event_view"
        }
        rendered_view_ids = list(
            row.get("retrieval_trace", {})
            .get("assembly", {})
            .get("rendered_event_view_ids", [])
        )
        rendered_view_lines: list[str] = []
        rendered_view_claim_chars = 0
        for view_id in rendered_view_ids:
            view = view_by_id.get(str(view_id))
            if view is None:
                continue
            view_text = str(view.get("text", ""))
            rendered_view_claim_chars += len(view_text)
            rendered_view_lines.append(
                f"[Cross-session event view | {view.get('view_type', '')} | sessions "
                + ", ".join(str(value) for value in view.get("session_ids", []))
                + f"] {view_text}"
            )
        rendered_event_view_chars = len("\n".join(rendered_view_lines))
        selected_set = set(selected)
        selected_sessions = {
            int(fact["session_id"])
            for fact in graph.get("fact_nodes", [])
            if str(fact.get("id", "")) in selected_set
        }
        return str(row.get("retrieved", "")), {
            "config_sha256": config.sha256,
            "selected_fact_ids": selected,
            "effective_selection_mode": "model",
            "effective_final_k": len(selected),
            "selection_source": selection_source,
            "new_reranker_called": False,
            "model_rerank_enabled": selection_source == "frozen_model",
            "temporal_question_detected": is_temporal_question(str(row.get("question", ""))),
            "rendered_event_view_ids": rendered_view_ids,
            "rendered_session_summary_ids": list(
                row.get("retrieval_trace", {})
                .get("assembly", {})
                .get("rendered_session_summary_ids", [])
            ),
            "session_header_policy": "legacy_timestamp_only",
            "iso_header_chars": 0,
            "counterfactual_r01_iso_header_chars": 21 * len(selected_sessions),
            "rendered_view_claim_chars": rendered_view_claim_chars,
            "rendered_event_view_chars": rendered_event_view_chars,
            "rendered_chars": len(str(row.get("retrieved", ""))),
            "reused_original_context": True,
        }

    selected_ids, selection_trace = select_fact_ids(row, graph, config)
    temporal = bool(selection_trace["temporal_question_detected"])
    session_render = (
        config.temporal_session_render
        if temporal and config.temporal_session_render is not None
        else config.session_render
    )
    view_render = (
        config.temporal_view_render
        if temporal and config.temporal_view_render is not None
        else config.view_render
    )
    fact_by_id = {str(fact["id"]): fact for fact in graph.get("fact_nodes", [])}
    session_summary = {
        int(node["session_ids"][0]): node
        for node in graph.get("summary_nodes", [])
        if node.get("node_type") == "session_packet"
    }
    view_by_id = {
        str(node["id"]): node
        for node in graph.get("summary_nodes", [])
        if node.get("node_type") == "event_view"
    }

    lines: list[str] = []
    rendered_view_lines: list[str] = []
    rendered_view_ids: list[str] = []
    view_claim_chars = 0
    view_chars = 0
    if view_render == "selected_claims":
        for view_id in _view_ids_for_selected(row, selected_ids, view_by_id):
            if len(rendered_view_ids) >= config.max_views:
                break
            view = view_by_id[view_id]
            header = (
                f"[Selected cross-session claims | {view.get('view_type', '')} | "
                f"sessions {', '.join(str(value) for value in view.get('session_ids', []))}] "
            )
            separator_chars = 1 if rendered_view_lines else 0
            remaining = (
                config.max_view_chars_total
                - view_chars
                - separator_chars
                - len(header)
            )
            if remaining <= 0:
                continue
            claim_text = _selected_claim_text(
                view,
                set(selected_ids),
                str(row.get("question", "")),
                max_claim_chars=config.max_claim_chars,
                remaining_chars=remaining,
            )
            if not claim_text:
                continue
            rendered_line = header + claim_text
            proposed_chars = view_chars + separator_chars + len(rendered_line)
            if proposed_chars > config.max_view_chars_total:
                continue
            rendered_view_lines.append(rendered_line)
            rendered_view_ids.append(view_id)
            view_claim_chars += len(claim_text)
            view_chars = proposed_chars
    elif view_render != "none":
        raise ValueError(f"unsupported non-original view rendering: {view_render}")
    lines.extend(rendered_view_lines)

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
    iso_header_chars = 0
    for session_id in session_order:
        summary = session_summary[session_id]
        timestamp = str(summary.get("timestamp", ""))
        iso_date = str(summary.get("iso_date", ""))
        if session_render == "full":
            lines.append(
                f"[Session summary | session {session_id} | {timestamp}; ISO date {iso_date}] "
                f"{summary['text']}"
            )
            rendered_summary_ids.append(str(summary["id"]))
            iso_header_chars += 21
        elif session_render == "legacy_full":
            lines.append(
                f"[Session summary | session {session_id} | {timestamp}] "
                f"{summary['text']}"
            )
            rendered_summary_ids.append(str(summary["id"]))
        elif session_render == "selected_sentence":
            sentence = _bounded_sentence(
                str(summary.get("text", "")),
                str(row.get("question", "")),
                config.max_session_sentence_chars,
            )
            lines.append(
                f"[Session evidence | session {session_id} | {timestamp}; ISO date {iso_date}]"
                + (f" {sentence}" if sentence else "")
            )
            if sentence:
                rendered_summary_ids.append(str(summary["id"]))
            iso_header_chars += 21
        elif session_render == "none":
            lines.append(
                f"[Session evidence | session {session_id} | {timestamp}; ISO date {iso_date}]"
            )
            iso_header_chars += 21
        else:
            raise ValueError(f"unknown session rendering: {session_render}")
        for fact in by_session[session_id]:
            sources = ", ".join(str(value) for value in fact.get("source_dia_ids", []))
            lines.append(f"  - [{sources}] {fact['text']}")

    text = "\n".join(lines)
    return text, {
        "config_sha256": config.sha256,
        "selected_fact_ids": selected_ids,
        **selection_trace,
        "session_render": session_render,
        "view_render": view_render,
        "rendered_event_view_ids": rendered_view_ids,
        "rendered_session_summary_ids": rendered_summary_ids,
        "session_header_policy": (
            "legacy_timestamp_only"
            if session_render == "legacy_full"
            else "explicit_iso_date"
        ),
        "iso_header_chars": iso_header_chars,
        "counterfactual_r01_iso_header_chars": 21 * len(session_order),
        "rendered_view_claim_chars": view_claim_chars,
        "rendered_event_view_chars": view_chars,
        "rendered_chars": len(text),
        "reused_original_context": False,
    }


def rematerialize_from_frozen(
    source_qa: Mapping[str, Any],
    base_row: Mapping[str, Any],
    graph: Mapping[str, Any],
    config: AblationConfig,
) -> tuple[str, dict[str, Any]]:
    """Rebuild one context without exposing gold fields to retrieval/rendering."""

    trusted_row = {
        "question": str(source_qa["question"]),
        "retrieved": str(base_row.get("retrieved", "")),
        "retrieval_trace": base_row.get("retrieval_trace", {}),
    }
    context, trace = materialize_context(trusted_row, graph, config)
    return context, {
        **trace,
        "base_row_trace_sha256": canonical_sha256(base_row["retrieval_trace"]),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }


def source_ids_for_facts(
    graph: Mapping[str, Any], fact_ids: Sequence[str]
) -> set[str]:
    fact_by_id = {str(fact["id"]): fact for fact in graph.get("fact_nodes", [])}
    output: set[str] = set()
    for fact_id in fact_ids:
        fact = fact_by_id.get(str(fact_id))
        if fact is not None:
            output.update(str(value) for value in fact.get("source_dia_ids", []))
    return output


__all__ = [
    "ABLATION_SCHEMA_VERSION",
    "ALL_CONFIGS",
    "BASELINE_CONFIG",
    "CONFIG_BY_NAME",
    "DEV_CATEGORY_QUOTAS",
    "DEV_CONVERSATIONS",
    "DEV_PANEL_SEED",
    "HOLDOUT_CONVERSATIONS",
    "POST_SCREEN_CONFIGS",
    "SCREEN_CONFIGS",
    "AblationConfig",
    "config_manifest",
    "is_temporal_question",
    "materialize_context",
    "rematerialize_from_frozen",
    "select_dev_panel",
    "source_ids_for_facts",
]

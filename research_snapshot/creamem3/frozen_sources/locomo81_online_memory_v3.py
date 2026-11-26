"""Strictly causal online event-view construction for CreaMem3.

The session writer and immutable atomic facts live in ``graph_memory_v2``.
This module consumes that session-only graph one session at a time and creates
or monotonically extends derived routing views.  At frontier session ``t`` the
model sees only the complete current packet, selected earlier facts, and
selected active views from the already-built prefix.

No storage or update logic from a comparison architecture is imported here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from creamem3.graph_memory_v2 import (
    BM25Index,
    GRAPH_SCHEMA_VERSION,
    GraphMemoryError,
    canonical_sha256,
)
from creamem3.prompts_v2 import ONLINE_EVENT_UPDATE_PROMPT, PROMPT_SHA256


ONLINE_SCHEMA_VERSION = "creamem3.strict_causal_online.v3"
SUMMARY_TOP_K = 5
DENSE_FACT_TOP_K = 40
BM25_FACT_TOP_K = 20
ACTIVE_VIEW_TOP_K = 10
PRIOR_FACT_CAP = 80
MAX_OPERATIONS = 12

_DIA_ID_RE = re.compile(r"D(\d+):(\d+)\Z")
_DIA_ID_SCAN_RE = re.compile(r"\bD(\d+):(\d+)\b")
_SPACE_RE = re.compile(r"\s+")
_ALLOWED_VIEW_TYPES = {"same_event", "continuing_process", "update_chain"}

ChatFn = Callable[..., str]
EmbedFn = Callable[..., list[list[float]]]


def _normalize_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _parse_json_object(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _without_embedding(node: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in node.items() if key != "embedding"}


def _checked_matrix(
    nodes: Sequence[Mapping[str, Any]], label: str, expected_dim: int | None = None
) -> np.ndarray:
    if not nodes:
        width = expected_dim or 0
        return np.empty((0, width), dtype=np.float32)
    try:
        matrix = np.asarray([node.get("embedding", []) for node in nodes], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise GraphMemoryError(f"{label} embeddings are ragged or non-numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise GraphMemoryError(f"{label} embeddings are missing or ragged")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise GraphMemoryError(
            f"{label} embedding dimension {matrix.shape[1]} != {expected_dim}"
        )
    if not np.isfinite(matrix).all():
        raise GraphMemoryError(f"{label} embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _embed_views(
    texts: Sequence[str], embed_fn: EmbedFn, *, expected_dim: int
) -> list[list[float]]:
    if not texts:
        return []
    vectors = embed_fn(
        list(texts),
        task_type="RETRIEVAL_DOCUMENT",
        tag="creamem3_v3_embed_online_views",
    )
    if len(vectors) != len(texts):
        raise GraphMemoryError(
            f"online view embedding count mismatch: {len(vectors)} != {len(texts)}"
        )
    checked: list[list[float]] = []
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        if len(values) != expected_dim or not all(math.isfinite(value) for value in values):
            raise GraphMemoryError(f"online view embedding {index} is invalid")
        checked.append(values)
    return checked


def _cosine_scores(
    candidates: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    *,
    expected_dim: int,
) -> list[float]:
    if not candidates:
        return []
    if not queries:
        return [0.0] * len(candidates)
    candidate_matrix = _checked_matrix(candidates, "candidate", expected_dim)
    query_matrix = _checked_matrix(queries, "query", expected_dim)
    return [float(value) for value in (candidate_matrix @ query_matrix.T).max(axis=1)]


def _base_parts(session_graph: Mapping[str, Any]) -> dict[str, Any]:
    if session_graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise GraphMemoryError("online builder received an unsupported base graph schema")
    if session_graph.get("graph_variant") != "session":
        raise GraphMemoryError("online builder requires the session-only base graph")

    summaries = list(session_graph.get("summary_nodes", []))
    facts = list(session_graph.get("fact_nodes", []))
    edges = list(session_graph.get("edges", []))
    if not facts:
        raise GraphMemoryError("session-only base graph contains no facts")
    if any(node.get("node_type") != "session_packet" for node in summaries):
        raise GraphMemoryError("session-only base graph unexpectedly contains event views")
    if any(edge.get("edge_type") != "SUPPORTS" for edge in edges):
        raise GraphMemoryError("session-only base graph has a non-SUPPORTS edge")

    session_ids = sorted(int(node["session_ids"][0]) for node in summaries)
    if len(session_ids) != len(set(session_ids)):
        raise GraphMemoryError("base graph contains duplicate session summaries")
    summary_by_session = {int(node["session_ids"][0]): node for node in summaries}
    facts_by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    fact_by_id: dict[str, Mapping[str, Any]] = {}
    fact_order: dict[str, int] = {}
    embedding_dim: int | None = None
    for index, fact in enumerate(facts):
        fact_id = str(fact.get("id", ""))
        session_id = int(fact.get("session_id"))
        if not fact_id or fact_id in fact_by_id or session_id not in summary_by_session:
            raise GraphMemoryError(f"invalid base fact identity {fact_id!r}")
        for source_id in fact.get("source_dia_ids", []):
            match = _DIA_ID_RE.fullmatch(str(source_id))
            if match is None or int(match.group(1)) != session_id:
                raise GraphMemoryError(f"fact {fact_id} has a cross-session source ID")
        vector = fact.get("embedding", [])
        if not isinstance(vector, list) or not vector:
            raise GraphMemoryError(f"fact {fact_id} lacks an embedding")
        embedding_dim = embedding_dim or len(vector)
        if len(vector) != embedding_dim:
            raise GraphMemoryError("base fact embedding dimensions are inconsistent")
        fact_by_id[fact_id] = fact
        fact_order[fact_id] = index
        facts_by_session[session_id].append(fact)

    summary_by_id = {str(node["id"]): node for node in summaries}
    support_by_summary: dict[str, list[str]] = defaultdict(list)
    support_edges_by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        source_id = str(edge.get("source_id", ""))
        target_id = str(edge.get("target_id", ""))
        if source_id not in summary_by_id or target_id not in fact_by_id:
            raise GraphMemoryError("base SUPPORTS edge references an unknown node")
        session_id = int(fact_by_id[target_id]["session_id"])
        if int(summary_by_id[source_id]["session_ids"][0]) != session_id:
            raise GraphMemoryError("base SUPPORTS edge crosses sessions")
        support_by_summary[source_id].append(target_id)
        support_edges_by_session[session_id].append(edge)

    packets = list(session_graph.get("session_packets", []))
    packet_by_session = {int(packet["session_id"]): packet for packet in packets}
    if set(packet_by_session) != set(session_ids):
        raise GraphMemoryError("session packet IDs do not match session summaries")

    return {
        "session_ids": session_ids,
        "summaries": summaries,
        "summary_by_session": summary_by_session,
        "facts": facts,
        "fact_by_id": fact_by_id,
        "fact_order": fact_order,
        "facts_by_session": facts_by_session,
        "edges": edges,
        "support_by_summary": support_by_summary,
        "support_edges_by_session": support_edges_by_session,
        "packet_by_session": packet_by_session,
        "embedding_dim": int(embedding_dim or 0),
    }


def _ranked_ids(
    nodes: Sequence[Mapping[str, Any]], scores: Sequence[float], top_k: int
) -> list[str]:
    ranked = sorted(
        range(len(nodes)),
        key=lambda index: (-float(scores[index]), str(nodes[index]["id"])),
    )[:top_k]
    return [str(nodes[index]["id"]) for index in ranked]


def _select_candidates(
    *,
    current_summary: Mapping[str, Any],
    current_facts: Sequence[Mapping[str, Any]],
    prior_summaries: Sequence[Mapping[str, Any]],
    prior_facts: Sequence[Mapping[str, Any]],
    active_views: Mapping[str, Mapping[str, Any]],
    support_by_summary: Mapping[str, Sequence[str]],
    fact_order: Mapping[str, int],
    embedding_dim: int,
) -> dict[str, Any]:
    summary_scores = _cosine_scores(
        prior_summaries, [current_summary], expected_dim=embedding_dim
    )
    summary_ids = _ranked_ids(prior_summaries, summary_scores, SUMMARY_TOP_K)

    query_nodes = [current_summary, *current_facts]
    dense_scores = _cosine_scores(
        prior_facts, query_nodes, expected_dim=embedding_dim
    )
    dense_ids = _ranked_ids(prior_facts, dense_scores, DENSE_FACT_TOP_K)
    dense_score_by_id = {
        str(node["id"]): float(score) for node, score in zip(prior_facts, dense_scores)
    }

    lexical_query = " ".join(
        [str(current_summary.get("text", ""))]
        + [str(fact.get("text", "")) for fact in current_facts]
    )
    bm25 = BM25Index()
    for fact in prior_facts:
        bm25.add(str(fact["id"]), str(fact.get("text", "")))
    bm25_rows = bm25.search(lexical_query, BM25_FACT_TOP_K)
    bm25_ids = [fact_id for fact_id, _ in bm25_rows]

    view_nodes = [active_views[key] for key in sorted(active_views)]
    view_scores = _cosine_scores(
        view_nodes, query_nodes, expected_dim=embedding_dim
    )
    view_ids = _ranked_ids(view_nodes, view_scores, ACTIVE_VIEW_TOP_K)

    channels: dict[str, dict[str, Any]] = defaultdict(dict)
    for rank, summary_id in enumerate(summary_ids, 1):
        for fact_id in support_by_summary.get(summary_id, []):
            channels[fact_id].setdefault("summary_ids", []).append(summary_id)
            channels[fact_id].setdefault("summary_ranks", []).append(rank)
    for rank, fact_id in enumerate(dense_ids, 1):
        channels[fact_id]["dense_rank"] = rank
    for rank, (fact_id, score) in enumerate(bm25_rows, 1):
        channels[fact_id]["bm25_rank"] = rank
        channels[fact_id]["bm25_score"] = float(score)

    prior_ids = {str(fact["id"]) for fact in prior_facts}
    union_ids = [fact_id for fact_id in channels if fact_id in prior_ids]
    union_ids.sort(
        key=lambda fact_id: (
            -dense_score_by_id.get(fact_id, -1.0),
            int(fact_order[fact_id]),
        )
    )
    union_before_cap = len(union_ids)
    selected_fact_ids = union_ids[:PRIOR_FACT_CAP]

    return {
        "summary_ids": summary_ids,
        "dense_fact_ids": dense_ids,
        "bm25_rows": bm25_rows,
        "active_view_ids": view_ids,
        "prior_fact_ids": selected_fact_ids,
        "union_before_cap": union_before_cap,
        "fact_channels": {fact_id: channels[fact_id] for fact_id in selected_fact_ids},
        "summary_scores": {
            str(node["id"]): float(score)
            for node, score in zip(prior_summaries, summary_scores)
            if str(node["id"]) in summary_ids
        },
        "view_scores": {
            str(node["id"]): float(score)
            for node, score in zip(view_nodes, view_scores)
            if str(node["id"]) in view_ids
        },
    }


def _prompt_fact(fact: Mapping[str, Any], channels: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output = {
        "fact_id": str(fact["id"]),
        "session_id": int(fact["session_id"]),
        "session_date": str(fact.get("iso_date", "")),
        "text": str(fact.get("text", "")),
        "source_dia_ids": list(fact.get("source_dia_ids", [])),
    }
    if channels is not None:
        output["retrieval_channels"] = copy.deepcopy(dict(channels))
    return output


def _prompt_view(view: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "view_id": str(view["id"]),
        "view_type": str(view["view_type"]),
        "summary_claims": copy.deepcopy(list(view.get("summary_claims", []))),
        "member_fact_ids": list(view.get("member_fact_ids", [])),
        "session_ids": list(view.get("session_ids", [])),
    }


def _build_prompt_input(
    session_id: int,
    *,
    current_summary: Mapping[str, Any],
    current_facts: Sequence[Mapping[str, Any]],
    prior_summaries: Sequence[Mapping[str, Any]],
    prior_facts: Sequence[Mapping[str, Any]],
    active_views: Mapping[str, Mapping[str, Any]],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _select_candidates(
        current_summary=current_summary,
        current_facts=current_facts,
        prior_summaries=prior_summaries,
        prior_facts=prior_facts,
        active_views=active_views,
        support_by_summary=base["support_by_summary"],
        fact_order=base["fact_order"],
        embedding_dim=int(base["embedding_dim"]),
    )
    candidate_fact_ids = list(selected["prior_fact_ids"])
    candidate_view_ids = list(selected["active_view_ids"])
    current_fact_ids = [str(fact["id"]) for fact in current_facts]
    return {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "frontier_session_id": session_id,
        "policy": {
            "summary_dense_top_k": SUMMARY_TOP_K,
            "fact_dense_top_k": DENSE_FACT_TOP_K,
            "bm25_top_k": BM25_FACT_TOP_K,
            "active_view_top_k": ACTIVE_VIEW_TOP_K,
            "prior_fact_cap": PRIOR_FACT_CAP,
            "membership_update": "monotonic_create_or_extend",
        },
        "current_session": {
            "session_id": session_id,
            "timestamp": str(current_summary.get("timestamp", "")),
            "iso_date": str(current_summary.get("iso_date", "")),
            "summary_id": str(current_summary["id"]),
            "summary": str(current_summary.get("text", "")),
            "facts": [_prompt_fact(fact) for fact in current_facts],
        },
        "prior_candidate_facts": [
            _prompt_fact(
                base["fact_by_id"][fact_id], selected["fact_channels"].get(fact_id, {})
            )
            for fact_id in candidate_fact_ids
        ],
        "candidate_active_views": [
            _prompt_view(active_views[view_id]) for view_id in candidate_view_ids
        ],
        "allowed_ids": {
            "current_fact_ids": current_fact_ids,
            "prior_candidate_fact_ids": candidate_fact_ids,
            "candidate_active_view_ids": candidate_view_ids,
        },
        "selection_trace": {
            "top_prior_summary_ids": list(selected["summary_ids"]),
            "dense_prior_fact_ids": list(selected["dense_fact_ids"]),
            "bm25_prior_facts": [
                {"fact_id": fact_id, "score": float(score)}
                for fact_id, score in selected["bm25_rows"]
            ],
            "top_active_view_ids": candidate_view_ids,
            "union_before_cap": int(selected["union_before_cap"]),
            "prior_facts_after_cap": len(candidate_fact_ids),
            "top_prior_summary_scores": dict(selected["summary_scores"]),
            "top_active_view_scores": dict(selected["view_scores"]),
        },
    }


def _view_edges(active_views: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"edge_type": "VIEW_SUPPORTS", "source_id": view_id, "target_id": fact_id}
        for view_id in sorted(active_views)
        for fact_id in active_views[view_id].get("member_fact_ids", [])
    ]


def _state_hash(
    session_summaries: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    support_edges: Sequence[Mapping[str, Any]],
    active_views: Mapping[str, Mapping[str, Any]],
) -> str:
    return canonical_sha256(
        {
            "session_summary_node_hashes": [canonical_sha256(node) for node in session_summaries],
            "fact_node_hashes": [canonical_sha256(node) for node in facts],
            "support_edges": sorted(
                (dict(edge) for edge in support_edges),
                key=lambda edge: (str(edge["source_id"]), str(edge["target_id"])),
            ),
            "active_view_node_hashes": {
                view_id: canonical_sha256(active_views[view_id])
                for view_id in sorted(active_views)
            },
            "view_edges": _view_edges(active_views),
        }
    )


def _normalize_claims(
    raw_claims: Any,
    *,
    membership: Sequence[str],
    fact_order: Mapping[str, int],
) -> list[dict[str, Any]]:
    if not isinstance(raw_claims, list) or not raw_claims:
        raise GraphMemoryError("summary_claims must be a non-empty list")
    member_set = set(membership)
    supported: set[str] = set()
    claims: list[dict[str, Any]] = []
    for index, claim in enumerate(raw_claims):
        if not isinstance(claim, dict) or set(claim) != {"text", "support_fact_ids"}:
            raise GraphMemoryError(f"summary_claims[{index}] has an invalid shape")
        text = _normalize_text(claim.get("text"))
        support = claim.get("support_fact_ids")
        if not text or not isinstance(support, list) or not support:
            raise GraphMemoryError(f"summary_claims[{index}] is empty")
        unique_support: list[str] = []
        for fact_id in support:
            if not isinstance(fact_id, str) or fact_id not in member_set:
                raise GraphMemoryError(
                    f"summary_claims[{index}] cites a non-member fact {fact_id!r}"
                )
            if fact_id not in unique_support:
                unique_support.append(fact_id)
        unique_support.sort(key=fact_order.__getitem__)
        supported.update(unique_support)
        claims.append({"text": text, "support_fact_ids": unique_support})
    if supported != member_set:
        missing = sorted(member_set - supported, key=fact_order.__getitem__)
        raise GraphMemoryError(f"member facts lack a summary claim: {missing}")
    return claims


def _make_view(
    *,
    view_id: str,
    view_type: str,
    claims: Sequence[Mapping[str, Any]],
    membership: Sequence[str],
    fact_by_id: Mapping[str, Mapping[str, Any]],
    created_at_session: int,
    updated_at_session: int,
    revision: int,
) -> dict[str, Any]:
    session_ids = sorted({int(fact_by_id[fact_id]["session_id"]) for fact_id in membership})
    source_ids: list[str] = []
    for fact_id in membership:
        for source_id in fact_by_id[fact_id].get("source_dia_ids", []):
            if source_id not in source_ids:
                source_ids.append(str(source_id))
    return {
        "id": view_id,
        "node_type": "event_view",
        "view_type": view_type,
        "text": " ".join(str(claim["text"]) for claim in claims),
        "summary_claims": copy.deepcopy(list(claims)),
        "member_fact_ids": list(membership),
        "session_ids": session_ids,
        "source_dia_ids": source_ids,
        "created_at_session": created_at_session,
        "last_updated_at_session": updated_at_session,
        "revision": revision,
        "embedding": [],
    }


def _validate_operation(
    raw: Any,
    *,
    session_id: int,
    prompt_input: Mapping[str, Any],
    working_views: Mapping[str, Mapping[str, Any]],
    touched_view_ids: set[str],
    fact_by_id: Mapping[str, Mapping[str, Any]],
    fact_order: Mapping[str, int],
    next_view_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {
        "operation",
        "target_view_id",
        "view_type",
        "summary_claims",
        "member_fact_ids",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise GraphMemoryError("operation has an invalid shape")
    operation = raw.get("operation")
    target_view_id = raw.get("target_view_id")
    view_type = raw.get("view_type")
    raw_members = raw.get("member_fact_ids")
    if operation not in {"create", "extend"}:
        raise GraphMemoryError("operation must be create or extend")
    if not isinstance(target_view_id, str):
        raise GraphMemoryError("target_view_id must be a string")
    if view_type not in _ALLOWED_VIEW_TYPES:
        raise GraphMemoryError("view_type is invalid")
    if not isinstance(raw_members, list) or len(raw_members) < 2:
        raise GraphMemoryError("member_fact_ids must contain at least two facts")

    member_ids: list[str] = []
    for fact_id in raw_members:
        if not isinstance(fact_id, str) or fact_id not in fact_by_id:
            raise GraphMemoryError(f"unknown member fact {fact_id!r}")
        if fact_id not in member_ids:
            member_ids.append(fact_id)
    member_ids.sort(key=fact_order.__getitem__)
    if len(member_ids) != len(raw_members):
        raise GraphMemoryError("member_fact_ids contains duplicates")

    allowed = prompt_input["allowed_ids"]
    current_ids = set(allowed["current_fact_ids"])
    prior_candidate_ids = set(allowed["prior_candidate_fact_ids"])
    candidate_view_ids = set(allowed["candidate_active_view_ids"])
    member_set = set(member_ids)
    member_sessions = {int(fact_by_id[fact_id]["session_id"]) for fact_id in member_ids}
    if any(value > session_id for value in member_sessions):
        raise GraphMemoryError("operation references a future-session fact")
    if not member_set.intersection(current_ids):
        raise GraphMemoryError("operation does not touch the current session")

    if operation == "create":
        if target_view_id:
            raise GraphMemoryError("create target_view_id must be empty")
        if not member_set.issubset(current_ids | prior_candidate_ids):
            raise GraphMemoryError("create uses a fact outside the prompt allowlist")
        if not member_set.intersection(prior_candidate_ids) or len(member_sessions) < 2:
            raise GraphMemoryError("create does not connect current and earlier sessions")
        view_id = next_view_id
        created_at = session_id
        revision = 1
    else:
        if target_view_id not in candidate_view_ids or target_view_id not in working_views:
            raise GraphMemoryError("extend target is not an exposed active view")
        if target_view_id in touched_view_ids:
            raise GraphMemoryError("an active view may be extended at most once per step")
        existing = working_views[target_view_id]
        existing_members = set(existing.get("member_fact_ids", []))
        if not existing_members.issubset(member_set):
            raise GraphMemoryError("extend removes an existing membership")
        additions = member_set - existing_members
        if not additions.intersection(current_ids):
            raise GraphMemoryError("extend adds no current-session fact")
        if not additions.issubset(current_ids | prior_candidate_ids):
            raise GraphMemoryError("extend adds a fact outside the prompt allowlist")
        if view_type != existing.get("view_type"):
            raise GraphMemoryError("extend changes the supplied view_type")
        view_id = target_view_id
        created_at = int(existing["created_at_session"])
        revision = int(existing["revision"]) + 1

    claims = _normalize_claims(
        raw.get("summary_claims"), membership=member_ids, fact_order=fact_order
    )
    view = _make_view(
        view_id=view_id,
        view_type=str(view_type),
        claims=claims,
        membership=member_ids,
        fact_by_id=fact_by_id,
        created_at_session=created_at,
        updated_at_session=session_id,
        revision=revision,
    )
    membership_tuple = tuple(member_ids)
    for other_id, other in working_views.items():
        if other_id != view_id and tuple(other.get("member_fact_ids", [])) == membership_tuple:
            raise GraphMemoryError(f"membership duplicates active view {other_id}")

    normalized = {
        "operation": operation,
        "target_view_id": target_view_id,
        "view_type": str(view_type),
        "summary_claims": claims,
        "member_fact_ids": member_ids,
    }
    return normalized, view


def _apply_model_response(
    value: Mapping[str, Any] | None,
    *,
    session_id: int,
    prompt_input: Mapping[str, Any],
    active_views: Mapping[str, Mapping[str, Any]],
    fact_by_id: Mapping[str, Mapping[str, Any]],
    fact_order: Mapping[str, int],
    next_view_number: int,
    embed_fn: EmbedFn,
    embedding_dim: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    working = {key: copy.deepcopy(dict(value)) for key, value in active_views.items()}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if value is None or set(value) != {"operations"} or not isinstance(value.get("operations"), list):
        rejected.append(
            {
                "source_index": None,
                "operation": None,
                "reason": "model response must contain only an operations list",
            }
        )
        return working, accepted, rejected, next_view_number

    touched_view_ids: set[str] = set()
    pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for index, raw_operation in enumerate(value["operations"]):
        if index >= MAX_OPERATIONS:
            rejected.append(
                {
                    "source_index": index,
                    "operation": copy.deepcopy(raw_operation),
                    "reason": f"operation exceeds the maximum of {MAX_OPERATIONS}",
                }
            )
            continue
        proposed_id = f"OV{next_view_number:04d}"
        try:
            normalized, view = _validate_operation(
                raw_operation,
                session_id=session_id,
                prompt_input=prompt_input,
                working_views=working,
                touched_view_ids=touched_view_ids,
                fact_by_id=fact_by_id,
                fact_order=fact_order,
                next_view_id=proposed_id,
            )
            if normalized["operation"] == "create":
                next_view_number += 1
            else:
                touched_view_ids.add(normalized["target_view_id"])
            working[view["id"]] = view
            accepted_entry = {
                "source_index": index,
                **normalized,
                "resulting_view": copy.deepcopy(view),
            }
            accepted.append(accepted_entry)
            pending.append((view["id"], view, accepted_entry))
        except Exception as exc:
            rejected.append(
                {
                    "source_index": index,
                    "operation": copy.deepcopy(raw_operation),
                    "reason": f"{type(exc).__name__}: {str(exc)[-500:]}",
                }
            )

    vectors = _embed_views(
        [view["text"] for _, view, _ in pending],
        embed_fn,
        expected_dim=embedding_dim,
    )
    for (view_id, view, accepted_entry), vector in zip(pending, vectors):
        view["embedding"] = vector
        working[view_id] = copy.deepcopy(view)
        accepted_entry["resulting_view"] = copy.deepcopy(view)
    return working, accepted, rejected, next_view_number


def _response_hash(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return _sha256_text(raw)


def _chain_hash(previous_chain: str, step: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "previous_chain_sha256": previous_chain,
            "session_id": step["session_id"],
            "pre_state_sha256": step["pre_state_sha256"],
            "post_append_state_sha256": step["post_append_state_sha256"],
            "post_state_sha256": step["post_state_sha256"],
            "prompt_input_sha256": step["prompt_input_sha256"],
            "model_call_made": step["model_call_made"],
            "model_response_sha256": step.get("model_response_sha256"),
            "model_error": step.get("model_error"),
            "accepted": step["accepted"],
            "rejected": step["rejected"],
        }
    )


def add_online_event_views(
    session_graph: Mapping[str, Any],
    chat_fn: ChatFn,
    embed_fn: EmbedFn,
) -> dict[str, Any]:
    """Create causal event views with exactly one chat call for every t >= 2."""

    base = _base_parts(session_graph)
    visible_summaries: list[Mapping[str, Any]] = []
    visible_facts: list[Mapping[str, Any]] = []
    visible_support_edges: list[Mapping[str, Any]] = []
    active_views: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    next_view_number = 1
    genesis = canonical_sha256(
        {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "base_content_fingerprint": session_graph.get("content_fingerprint"),
            "session_order": base["session_ids"],
            "prompt_sha256": PROMPT_SHA256["online_event_update"],
        }
    )
    previous_chain = genesis

    for ordinal, session_id in enumerate(base["session_ids"]):
        pre_state = _state_hash(
            visible_summaries, visible_facts, visible_support_edges, active_views
        )
        current_summary = base["summary_by_session"][session_id]
        current_facts = list(base["facts_by_session"][session_id])
        prior_summaries = list(visible_summaries)
        prior_facts = list(visible_facts)
        visible_summaries.append(current_summary)
        visible_facts.extend(current_facts)
        visible_support_edges.extend(base["support_edges_by_session"][session_id])
        post_append_state = _state_hash(
            visible_summaries, visible_facts, visible_support_edges, active_views
        )
        prompt_input = _build_prompt_input(
            session_id,
            current_summary=current_summary,
            current_facts=current_facts,
            prior_summaries=prior_summaries,
            prior_facts=prior_facts,
            active_views=active_views,
            base=base,
        )
        prompt_input_sha = canonical_sha256(prompt_input)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        raw_response: Any = None
        model_error: str | None = None
        model_call_made = ordinal > 0
        if model_call_made:
            prompt = ONLINE_EVENT_UPDATE_PROMPT.format(
                frontier_session_id=session_id,
                online_input_json=json.dumps(
                    prompt_input, ensure_ascii=False, separators=(",", ":")
                ),
            )
            try:
                raw_response = chat_fn(
                    prompt,
                    tag="creamem3_v3_online_fuse",
                    temperature=0.0,
                    max_tokens=16384,
                    json_mode=True,
                )
                value = _parse_json_object(raw_response)
                active_views, accepted, rejected, next_view_number = _apply_model_response(
                    value,
                    session_id=session_id,
                    prompt_input=prompt_input,
                    active_views=active_views,
                    fact_by_id=base["fact_by_id"],
                    fact_order=base["fact_order"],
                    next_view_number=next_view_number,
                    embed_fn=embed_fn,
                    embedding_dim=int(base["embedding_dim"]),
                )
            except Exception as exc:
                model_error = f"{type(exc).__name__}: {str(exc)[-500:]}"
                rejected.append(
                    {
                        "source_index": None,
                        "operation": None,
                        "reason": "online fusion call or response processing failed",
                    }
                )
        post_state = _state_hash(
            visible_summaries, visible_facts, visible_support_edges, active_views
        )
        response_sha = _response_hash(raw_response)
        step: dict[str, Any] = {
            "step_id": f"T{session_id:03d}",
            "session_id": session_id,
            "ordinal": ordinal + 1,
            "visible_session_ids": list(base["session_ids"][: ordinal + 1]),
            "model_call_made": model_call_made,
            "model_call_count": int(model_call_made),
            "prompt_input": prompt_input,
            "allowed_ids": copy.deepcopy(prompt_input["allowed_ids"]),
            "prompt_input_sha256": prompt_input_sha,
            "model_response_sha256": response_sha,
            "model_error": model_error,
            "accepted": accepted,
            "rejected": rejected,
            "pre_state_sha256": pre_state,
            "post_append_state_sha256": post_append_state,
            "post_state_sha256": post_state,
            "previous_chain_sha256": previous_chain,
        }
        step["chain_sha256"] = _chain_hash(previous_chain, step)
        previous_chain = step["chain_sha256"]
        steps.append(step)

    graph = copy.deepcopy(dict(session_graph))
    final_views = [active_views[view_id] for view_id in sorted(active_views)]
    graph["graph_variant"] = "online"
    graph["architecture"] = dict(graph.get("architecture", {}))
    graph["architecture"].update(
        {
            "online_update_policy": "strict_causal_prefix_only_create_or_extend",
            "event_view_membership": "monotonic_non_destructive",
        }
    )
    graph["prompt_sha256"] = dict(graph.get("prompt_sha256", {}))
    graph["prompt_sha256"]["online_event_update"] = PROMPT_SHA256[
        "online_event_update"
    ]
    graph["summary_nodes"] = copy.deepcopy(base["summaries"]) + copy.deepcopy(final_views)
    graph["fact_nodes"] = copy.deepcopy(base["facts"])
    graph["edges"] = copy.deepcopy(base["edges"]) + _view_edges(active_views)
    graph["stats"] = dict(graph.get("stats", {}))
    graph["stats"].update(
        {
            "event_view_nodes": len(final_views),
            "summary_nodes": len(graph["summary_nodes"]),
            "edges": len(graph["edges"]),
            "cross_session_view_edges": sum(
                len(view["member_fact_ids"]) for view in final_views
            ),
            "online_fusion_calls": sum(step["model_call_count"] for step in steps),
            "online_accepted_operations": sum(len(step["accepted"]) for step in steps),
            "online_rejected_operations": sum(len(step["rejected"]) for step in steps),
        }
    )
    graph["online_fusion"] = {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "policy": "strict_causal_prefix_only_create_or_extend",
        "prompt_sha256": PROMPT_SHA256["online_event_update"],
        "session_order": list(base["session_ids"]),
        "candidate_policy": {
            "summary_dense_top_k": SUMMARY_TOP_K,
            "fact_dense_top_k": DENSE_FACT_TOP_K,
            "bm25_top_k": BM25_FACT_TOP_K,
            "active_view_top_k": ACTIVE_VIEW_TOP_K,
            "prior_fact_cap": PRIOR_FACT_CAP,
        },
        "base_content_fingerprint": session_graph.get("content_fingerprint"),
        "base_fact_identity_sha256": canonical_sha256(base["facts"]),
        "genesis_chain_sha256": genesis,
        "final_chain_sha256": previous_chain,
        "steps": steps,
    }
    graph["content_fingerprint"] = canonical_sha256(
        {
            "base": session_graph.get("content_fingerprint"),
            "online_schema": ONLINE_SCHEMA_VERSION,
            "active_views": [_without_embedding(view) for view in final_views],
            "final_chain_sha256": previous_chain,
        }
    )
    return graph


def _scan_future_dia_ids(value: Any, frontier: int) -> list[str]:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return sorted(
        {match.group(0) for match in _DIA_ID_SCAN_RE.finditer(text) if int(match.group(1)) > frontier}
    )


def audit_online_graph(
    session_graph: Mapping[str, Any], online_graph: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay the online log and verify causal, identity, and hash invariants."""

    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    try:
        base = _base_parts(session_graph)
    except Exception as exc:
        return {
            "status": "fail",
            "checks": [
                {
                    "name": "base_graph_valid",
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {str(exc)[-500:]}",
                }
            ],
            "warnings": [],
            "passed": 0,
            "failed": 1,
        }

    fusion = online_graph.get("online_fusion")
    check("online_graph_variant", online_graph.get("graph_variant") == "online")
    check("online_schema_version", isinstance(fusion, Mapping) and fusion.get("schema_version") == ONLINE_SCHEMA_VERSION)
    if not isinstance(fusion, Mapping):
        failures = [item for item in checks if not item["passed"]]
        return {
            "status": "fail",
            "checks": checks,
            "warnings": warnings,
            "passed": len(checks) - len(failures),
            "failed": len(failures),
        }

    check("session_order", fusion.get("session_order") == base["session_ids"])
    check(
        "online_prompt_hash",
        fusion.get("prompt_sha256") == PROMPT_SHA256["online_event_update"]
        and online_graph.get("prompt_sha256", {}).get("online_event_update")
        == PROMPT_SHA256["online_event_update"],
    )
    check(
        "base_content_fingerprint",
        fusion.get("base_content_fingerprint")
        == session_graph.get("content_fingerprint"),
    )
    check(
        "base_fact_identity_manifest",
        fusion.get("base_fact_identity_sha256") == canonical_sha256(base["facts"]),
    )
    check("facts_byte_identical", online_graph.get("fact_nodes") == session_graph.get("fact_nodes"))
    check("session_packets_byte_identical", online_graph.get("session_packets") == session_graph.get("session_packets"))
    online_session_summaries = [
        node for node in online_graph.get("summary_nodes", []) if node.get("node_type") == "session_packet"
    ]
    check(
        "summary_node_partition",
        all(
            node.get("node_type") in {"session_packet", "event_view"}
            for node in online_graph.get("summary_nodes", [])
        ),
    )
    check("session_summaries_byte_identical", online_session_summaries == base["summaries"])
    online_support_edges = [
        edge for edge in online_graph.get("edges", []) if edge.get("edge_type") == "SUPPORTS"
    ]
    check("base_support_edges_byte_identical", online_support_edges == base["edges"])

    steps = fusion.get("steps")
    check("one_step_per_session", isinstance(steps, list) and len(steps) == len(base["session_ids"]))
    if not isinstance(steps, list):
        steps = []

    visible_summaries: list[Mapping[str, Any]] = []
    visible_facts: list[Mapping[str, Any]] = []
    visible_edges: list[Mapping[str, Any]] = []
    active_views: dict[str, dict[str, Any]] = {}
    next_view_number = 1
    expected_genesis = canonical_sha256(
        {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "base_content_fingerprint": session_graph.get("content_fingerprint"),
            "session_order": base["session_ids"],
            "prompt_sha256": PROMPT_SHA256["online_event_update"],
        }
    )
    check("genesis_chain", fusion.get("genesis_chain_sha256") == expected_genesis)
    previous_chain = expected_genesis
    replay_ok = True
    prompt_prefix_ok = True
    accepted_prefix_ok = True
    call_count_ok = True
    hash_chain_ok = True
    rejected_future_references = 0

    for ordinal, session_id in enumerate(base["session_ids"]):
        if ordinal >= len(steps):
            replay_ok = False
            break
        step = steps[ordinal]
        if not isinstance(step, Mapping):
            replay_ok = False
            hash_chain_ok = False
            break
        pre_state = _state_hash(visible_summaries, visible_facts, visible_edges, active_views)
        current_summary = base["summary_by_session"][session_id]
        current_facts = list(base["facts_by_session"][session_id])
        prior_summaries = list(visible_summaries)
        prior_facts = list(visible_facts)
        visible_summaries.append(current_summary)
        visible_facts.extend(current_facts)
        visible_edges.extend(base["support_edges_by_session"][session_id])
        post_append = _state_hash(visible_summaries, visible_facts, visible_edges, active_views)
        expected_input = _build_prompt_input(
            session_id,
            current_summary=current_summary,
            current_facts=current_facts,
            prior_summaries=prior_summaries,
            prior_facts=prior_facts,
            active_views=active_views,
            base=base,
        )
        actual_input = step.get("prompt_input")
        prompt_prefix_ok = prompt_prefix_ok and actual_input == expected_input
        prompt_prefix_ok = prompt_prefix_ok and not _scan_future_dia_ids(actual_input, session_id)
        expected_call = ordinal > 0
        recorded_call_count = step.get("model_call_count")
        call_count_ok = (
            call_count_ok
            and bool(step.get("model_call_made")) == expected_call
            and isinstance(recorded_call_count, int)
            and recorded_call_count == int(expected_call)
        )
        if (
            step.get("session_id") != session_id
            or step.get("visible_session_ids") != base["session_ids"][: ordinal + 1]
            or step.get("pre_state_sha256") != pre_state
            or step.get("post_append_state_sha256") != post_append
            or step.get("prompt_input_sha256") != canonical_sha256(actual_input)
            or step.get("allowed_ids") != expected_input["allowed_ids"]
        ):
            replay_ok = False

        touched: set[str] = set()
        accepted_rows = step.get("accepted")
        rejected_rows = step.get("rejected")
        if not isinstance(accepted_rows, list) or not isinstance(rejected_rows, list):
            replay_ok = False
            accepted_prefix_ok = False
            hash_chain_ok = False
            break
        for accepted in accepted_rows:
            try:
                operation_payload = {
                    key: copy.deepcopy(accepted[key])
                    for key in (
                        "operation",
                        "target_view_id",
                        "view_type",
                        "summary_claims",
                        "member_fact_ids",
                    )
                }
                proposed_id = f"OV{next_view_number:04d}"
                normalized, expected_view = _validate_operation(
                    operation_payload,
                    session_id=session_id,
                    prompt_input=expected_input,
                    working_views=active_views,
                    touched_view_ids=touched,
                    fact_by_id=base["fact_by_id"],
                    fact_order=base["fact_order"],
                    next_view_id=proposed_id,
                )
                actual_view = accepted.get("resulting_view")
                if not isinstance(actual_view, Mapping):
                    raise GraphMemoryError("accepted operation lacks resulting_view")
                if normalized["operation"] == "create":
                    next_view_number += 1
                else:
                    touched.add(normalized["target_view_id"])
                if _without_embedding(actual_view) != _without_embedding(expected_view):
                    raise GraphMemoryError("resulting_view does not match replay")
                vector = actual_view.get("embedding", [])
                if (
                    not isinstance(vector, list)
                    or len(vector) != base["embedding_dim"]
                    or not all(math.isfinite(float(value)) for value in vector)
                ):
                    raise GraphMemoryError("resulting_view embedding is invalid")
                if any(int(base["fact_by_id"][fact_id]["session_id"]) > session_id for fact_id in actual_view["member_fact_ids"]):
                    raise GraphMemoryError("accepted resulting_view contains a future fact")
                if _scan_future_dia_ids(actual_view, session_id):
                    raise GraphMemoryError("accepted resulting_view contains a future Dia ID")
                active_views[str(actual_view["id"])] = copy.deepcopy(dict(actual_view))
            except Exception:
                accepted_prefix_ok = False
                replay_ok = False

        for rejected in rejected_rows:
            rejected_future_references += len(_scan_future_dia_ids(rejected, session_id))
            raw_operation = rejected.get("operation") if isinstance(rejected, Mapping) else None
            if isinstance(raw_operation, Mapping):
                for fact_id in raw_operation.get("member_fact_ids", []):
                    fact = base["fact_by_id"].get(fact_id)
                    if fact is not None and int(fact["session_id"]) > session_id:
                        rejected_future_references += 1

        post_state = _state_hash(visible_summaries, visible_facts, visible_edges, active_views)
        if step.get("post_state_sha256") != post_state:
            replay_ok = False
        if step.get("previous_chain_sha256") != previous_chain:
            hash_chain_ok = False
        try:
            expected_chain = _chain_hash(previous_chain, step)
        except Exception:
            hash_chain_ok = False
            replay_ok = False
            break
        if step.get("chain_sha256") != expected_chain:
            hash_chain_ok = False
        previous_chain = expected_chain

    if rejected_future_references:
        warnings.append(
            {
                "name": "rejected_future_reference_attempts",
                "count": rejected_future_references,
                "detail": "Future-looking references occurred only in rejected model operations.",
            }
        )

    final_views = [active_views[key] for key in sorted(active_views)]
    graph_views = [
        node for node in online_graph.get("summary_nodes", []) if node.get("node_type") == "event_view"
    ]
    expected_edges = base["edges"] + _view_edges(active_views)
    check("exactly_one_model_call_per_t_after_first", call_count_ok)
    check("prompt_inputs_are_exact_prefix_candidates", prompt_prefix_ok)
    check("accepted_operations_have_no_future_information", accepted_prefix_ok)
    check("accepted_resulting_views_replay", replay_ok)
    check("continuous_hash_chain", hash_chain_ok and fusion.get("final_chain_sha256") == previous_chain)
    check("final_active_views_match_replay", graph_views == final_views)
    check("final_edges_match_replay", online_graph.get("edges") == expected_edges)
    check(
        "all_final_views_cross_session",
        all(len(set(view.get("session_ids", []))) >= 2 for view in graph_views),
    )
    check(
        "all_view_embeddings_present",
        all(bool(view.get("embedding")) for view in graph_views),
    )
    expected_content_fingerprint = canonical_sha256(
        {
            "base": session_graph.get("content_fingerprint"),
            "online_schema": ONLINE_SCHEMA_VERSION,
            "active_views": [_without_embedding(view) for view in final_views],
            "final_chain_sha256": previous_chain,
        }
    )
    check(
        "content_fingerprint",
        online_graph.get("content_fingerprint") == expected_content_fingerprint,
    )
    stats = online_graph.get("stats", {})
    check(
        "online_stats_consistent",
        isinstance(stats, Mapping)
        and stats.get("event_view_nodes") == len(final_views)
        and stats.get("summary_nodes") == len(online_graph.get("summary_nodes", []))
        and stats.get("edges") == len(online_graph.get("edges", []))
        and stats.get("online_fusion_calls") == max(0, len(base["session_ids"]) - 1),
    )

    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "warnings": warnings,
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "replayed_sessions": min(len(steps), len(base["session_ids"])),
        "replayed_active_views": len(final_views),
    }


__all__ = [
    "ACTIVE_VIEW_TOP_K",
    "BM25_FACT_TOP_K",
    "DENSE_FACT_TOP_K",
    "MAX_OPERATIONS",
    "ONLINE_SCHEMA_VERSION",
    "PRIOR_FACT_CAP",
    "SUMMARY_TOP_K",
    "add_online_event_views",
    "audit_online_graph",
]

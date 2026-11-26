"""Frontier-causal Online event views with compact prefix bookkeeping.

Version 4 keeps the v3 create/extend semantics, while treating each session
packet as a logical writer frontier.  Immutable node hashes and normalized
dense matrices are computed once, BM25 grows incrementally, and unchanged
views/embeddings are structurally shared instead of deep-copied per step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from creamem3 import online_memory_v3 as legacy
from creamem3.graph_memory_v2 import BM25Index, GraphMemoryError, canonical_sha256


ONLINE_SCHEMA_VERSION = "creamem3.strict_causal_online.v4_frontier"
SUMMARY_TOP_K = legacy.SUMMARY_TOP_K
DENSE_FACT_TOP_K = legacy.DENSE_FACT_TOP_K
BM25_FACT_TOP_K = legacy.BM25_FACT_TOP_K
ACTIVE_VIEW_TOP_K = legacy.ACTIVE_VIEW_TOP_K
PRIOR_FACT_CAP = legacy.PRIOR_FACT_CAP
MAX_OPERATIONS = legacy.MAX_OPERATIONS

ChatFn = Callable[..., str]
EmbedFn = Callable[..., list[list[float]]]


ONLINE_FRONTIER_UPDATE_PROMPT = r"""You update causal cross-frontier routing views after receiving dialogue frontier {frontier_id}. A frontier is one writer batch containing one to three consecutive physical dialogue sessions.

The input is strictly limited to the complete current frontier, selected facts from earlier frontiers, and selected active views created from that prefix. Atomic facts are immutable and are the source of truth. Return no operation when the evidence does not describe the same concrete event, a continuing process or goal, or an explicit update chain. Do not group items merely because they share a person, broad topic, hobby, place, or repeated activity.

Output STRICT JSON with exactly this shape:
{{
  "operations": [
    {{
      "operation": "create|extend",
      "target_view_id": "",
      "view_type": "same_event|continuing_process|update_chain",
      "summary_claims": [
        {{"text": "<one grounded claim>", "support_fact_ids": ["F0001"]}}
      ],
      "member_fact_ids": ["F0001", "F0002"]
    }}
  ]
}}

Rules:
1. Copy every fact ID and target view ID exactly from the input. Do not invent IDs or facts.
2. Every create operation must include at least one current-frontier fact and at least one earlier-frontier fact, spanning at least two frontiers. Its target_view_id must be an empty string.
3. Every extend operation must target one supplied active view, preserve all of its existing member facts, and add at least one current-frontier fact. Keep the supplied view_type.
4. Each summary claim must cite supporting member facts. Every member fact must support at least one claim. Preserve dates, uncertainty, corrections, and distinct occurrences.
5. Membership may only grow. Never rewrite, delete, merge, or duplicate atomic facts. Return at most 12 operations.
6. Return one JSON object only, without Markdown or commentary. An empty operations list is valid and preferred to a weak grouping.

Causal frontier-prefix input JSON:
{online_input_json}
"""
ONLINE_FRONTIER_PROMPT_SHA256 = hashlib.sha256(
    ONLINE_FRONTIER_UPDATE_PROMPT.encode("utf-8")
).hexdigest()


def _normalized_matrix(nodes: Sequence[Mapping[str, Any]], *, expected_dim: int) -> np.ndarray:
    if not nodes:
        return np.empty((0, expected_dim), dtype=np.float32)
    matrix = np.asarray([node.get("embedding", []) for node in nodes], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim or not np.isfinite(matrix).all():
        raise GraphMemoryError("frontier prefix embeddings are invalid")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise GraphMemoryError("frontier prefix contains a zero embedding")
    return matrix / norms


class _PrefixSelector:
    """Exact v3 candidate policy without rebuilding matrices/BM25 each step."""

    def __init__(self, base: Mapping[str, Any]) -> None:
        self.base = base
        self.dim = int(base["embedding_dim"])
        self.summaries = list(base["summaries"])
        self.facts = list(base["facts"])
        self.summary_matrix = _normalized_matrix(self.summaries, expected_dim=self.dim)
        self.fact_matrix = _normalized_matrix(self.facts, expected_dim=self.dim)
        self.summary_index = {str(node["id"]): i for i, node in enumerate(self.summaries)}
        self.fact_index = {str(node["id"]): i for i, node in enumerate(self.facts)}
        self.visible_summary_count = 0
        self.visible_fact_count = 0
        self.bm25 = BM25Index()

    @staticmethod
    def _rank(nodes: Sequence[Mapping[str, Any]], scores: Sequence[float], top_k: int) -> list[str]:
        order = sorted(
            range(len(nodes)),
            key=lambda index: (-float(scores[index]), str(nodes[index]["id"])),
        )[:top_k]
        return [str(nodes[index]["id"]) for index in order]

    def select(
        self,
        *,
        current_summary: Mapping[str, Any],
        current_facts: Sequence[Mapping[str, Any]],
        active_views: Mapping[str, Mapping[str, Any]],
        prior_fact_cap: int,
    ) -> dict[str, Any]:
        prior_summaries = self.summaries[: self.visible_summary_count]
        prior_facts = self.facts[: self.visible_fact_count]
        current_nodes = [current_summary, *current_facts]
        query_matrix = _normalized_matrix(current_nodes, expected_dim=self.dim)
        if prior_summaries:
            summary_scores_array = (
                self.summary_matrix[: self.visible_summary_count] @ query_matrix[:1].T
            ).reshape(-1)
        else:
            summary_scores_array = np.empty((0,), dtype=np.float32)
        if prior_facts:
            dense_scores_array = (
                self.fact_matrix[: self.visible_fact_count] @ query_matrix.T
            ).max(axis=1)
        else:
            dense_scores_array = np.empty((0,), dtype=np.float32)
        summary_scores = [float(value) for value in summary_scores_array]
        dense_scores = [float(value) for value in dense_scores_array]
        summary_ids = self._rank(prior_summaries, summary_scores, SUMMARY_TOP_K)
        dense_ids = self._rank(prior_facts, dense_scores, DENSE_FACT_TOP_K)
        dense_score_by_id = {
            str(node["id"]): float(score) for node, score in zip(prior_facts, dense_scores)
        }

        lexical_query = " ".join(
            [str(current_summary.get("text", ""))]
            + [str(fact.get("text", "")) for fact in current_facts]
        )
        bm25_rows = self.bm25.search(lexical_query, BM25_FACT_TOP_K)

        view_nodes = [active_views[key] for key in sorted(active_views)]
        view_scores = legacy._cosine_scores(
            view_nodes, current_nodes, expected_dim=self.dim
        )
        view_ids = self._rank(view_nodes, view_scores, ACTIVE_VIEW_TOP_K)

        channels: dict[str, dict[str, Any]] = defaultdict(dict)
        for rank, summary_id in enumerate(summary_ids, 1):
            for fact_id in self.base["support_by_summary"].get(summary_id, []):
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
                int(self.base["fact_order"][fact_id]),
            )
        )
        selected_ids = union_ids[:prior_fact_cap]
        return {
            "summary_ids": summary_ids,
            "dense_fact_ids": dense_ids,
            "bm25_rows": bm25_rows,
            "active_view_ids": view_ids,
            "prior_fact_ids": selected_ids,
            "union_before_cap": len(union_ids),
            "fact_channels": {fact_id: channels[fact_id] for fact_id in selected_ids},
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

    def append(self, summary: Mapping[str, Any], facts: Sequence[Mapping[str, Any]]) -> None:
        if self.summary_index[str(summary["id"])] != self.visible_summary_count:
            raise GraphMemoryError("summary prefix is not contiguous")
        for fact in facts:
            if self.fact_index[str(fact["id"])] != self.visible_fact_count:
                raise GraphMemoryError("fact prefix is not contiguous")
            self.bm25.add(str(fact["id"]), str(fact.get("text", "")))
            self.visible_fact_count += 1
        self.visible_summary_count += 1


def _build_prompt_input(
    frontier_id: int,
    *,
    current_summary: Mapping[str, Any],
    current_facts: Sequence[Mapping[str, Any]],
    active_views: Mapping[str, Mapping[str, Any]],
    base: Mapping[str, Any],
    selector: _PrefixSelector,
    prior_fact_cap: int,
) -> dict[str, Any]:
    selected = selector.select(
        current_summary=current_summary,
        current_facts=current_facts,
        active_views=active_views,
        prior_fact_cap=prior_fact_cap,
    )
    candidate_fact_ids = list(selected["prior_fact_ids"])
    candidate_view_ids = list(selected["active_view_ids"])
    current_fact_ids = [str(fact["id"]) for fact in current_facts]
    packet = base["packet_by_session"][frontier_id]
    return {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "frontier_id": frontier_id,
        "policy": {
            "summary_dense_top_k": SUMMARY_TOP_K,
            "fact_dense_top_k": DENSE_FACT_TOP_K,
            "bm25_top_k": BM25_FACT_TOP_K,
            "active_view_top_k": ACTIVE_VIEW_TOP_K,
            "prior_fact_cap": prior_fact_cap,
            "membership_update": "monotonic_create_or_extend",
        },
        "current_frontier": {
            "frontier_id": frontier_id,
            "physical_sessions": copy.deepcopy(list(packet.get("physical_sessions", []))),
            "timestamp": str(current_summary.get("timestamp", "")),
            "iso_date_range": str(current_summary.get("iso_date", "")),
            "summary_id": str(current_summary["id"]),
            "summary": str(current_summary.get("text", "")),
            "facts": [legacy._prompt_fact(fact) for fact in current_facts],
        },
        "prior_candidate_facts": [
            legacy._prompt_fact(
                base["fact_by_id"][fact_id], selected["fact_channels"].get(fact_id, {})
            )
            for fact_id in candidate_fact_ids
        ],
        "candidate_active_views": [
            legacy._prompt_view(active_views[view_id]) for view_id in candidate_view_ids
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
            "current_facts_before_cap": len(current_facts),
            "current_facts_after_cap": len(current_facts),
            "current_fact_cap": None,
            "top_prior_summary_scores": dict(selected["summary_scores"]),
            "top_active_view_scores": dict(selected["view_scores"]),
        },
    }


class _HashState:
    """Cache hashes of immutable vectors while preserving the canonical state."""

    def __init__(self, base: Mapping[str, Any]) -> None:
        self.summary_hash = {
            str(node["id"]): canonical_sha256(node) for node in base["summaries"]
        }
        self.fact_hash = {str(node["id"]): canonical_sha256(node) for node in base["facts"]}
        self.summary_hashes: list[str] = []
        self.fact_hashes: list[str] = []
        self.support_edges: list[Mapping[str, Any]] = []
        self.active_views: dict[str, Mapping[str, Any]] = {}
        self.active_view_hashes: dict[str, str] = {}

    def append(
        self,
        summary: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
    ) -> None:
        self.summary_hashes.append(self.summary_hash[str(summary["id"])])
        self.fact_hashes.extend(self.fact_hash[str(fact["id"])] for fact in facts)
        self.support_edges.extend(edges)

    def replace_views(self, views: Mapping[str, Mapping[str, Any]]) -> None:
        for view_id, view in views.items():
            if self.active_views.get(view_id) is not view:
                self.active_view_hashes[view_id] = canonical_sha256(view)
        self.active_views = dict(views)
        self.active_view_hashes = {
            key: self.active_view_hashes[key] for key in self.active_views
        }

    def digest(self) -> str:
        return canonical_sha256(
            {
                "session_summary_node_hashes": self.summary_hashes,
                "fact_node_hashes": self.fact_hashes,
                "support_edges": sorted(
                    (dict(edge) for edge in self.support_edges),
                    key=lambda edge: (str(edge["source_id"]), str(edge["target_id"])),
                ),
                "active_view_node_hashes": {
                    view_id: self.active_view_hashes[view_id]
                    for view_id in sorted(self.active_views)
                },
                "view_edges": legacy._view_edges(self.active_views),
            }
        )


def _normalize_view_vectors(
    texts: Sequence[str],
    embed_fn: EmbedFn,
    *,
    expected_dim: int,
) -> list[list[float]]:
    if not texts:
        return []
    vectors = embed_fn(
        list(texts),
        task_type="RETRIEVAL_DOCUMENT",
        tag="creamem3_v4_embed_online_views",
    )
    if len(vectors) != len(texts):
        raise GraphMemoryError("online view embedding count mismatch")
    normalized: list[list[float]] = []
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        if len(values) != expected_dim or not all(math.isfinite(value) for value in values):
            raise GraphMemoryError(f"online view embedding {index} is invalid")
        norm = math.sqrt(math.fsum(value * value for value in values))
        if not math.isfinite(norm) or norm <= 0.0:
            raise GraphMemoryError(f"online view embedding {index} has invalid norm")
        normalized.append([value / norm for value in values])
    return normalized


def _apply_model_response(
    value: Mapping[str, Any] | None,
    *,
    frontier_id: int,
    prompt_input: Mapping[str, Any],
    active_views: Mapping[str, Mapping[str, Any]],
    fact_by_id: Mapping[str, Mapping[str, Any]],
    fact_order: Mapping[str, int],
    next_view_number: int,
    embed_fn: EmbedFn,
    embedding_dim: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    """Apply operations with structural sharing of every unchanged view."""

    working: dict[str, dict[str, Any]] = {
        key: value if isinstance(value, dict) else dict(value)
        for key, value in active_views.items()
    }
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

    touched: set[str] = set()
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
        try:
            normalized, view = legacy._validate_operation(
                raw_operation,
                session_id=frontier_id,
                prompt_input=prompt_input,
                working_views=working,
                touched_view_ids=touched,
                fact_by_id=fact_by_id,
                fact_order=fact_order,
                next_view_id=f"OV{next_view_number:04d}",
            )
            if normalized["operation"] == "create":
                next_view_number += 1
            else:
                touched.add(str(normalized["target_view_id"]))
            # ``view`` is a new small object.  Existing values in ``working``
            # retain their original identity and 768-float list.
            working[str(view["id"])] = view
            accepted_entry = {
                "source_index": index,
                **normalized,
                "resulting_view": legacy._without_embedding(view),
            }
            accepted.append(accepted_entry)
            pending.append((str(view["id"]), view, accepted_entry))
        except Exception as exc:
            rejected.append(
                {
                    "source_index": index,
                    "operation": copy.deepcopy(raw_operation),
                    "reason": f"{type(exc).__name__}: {str(exc)[-500:]}",
                }
            )

    vectors = _normalize_view_vectors(
        [view["text"] for _, view, _ in pending],
        embed_fn,
        expected_dim=embedding_dim,
    )
    for (view_id, view, accepted_entry), vector in zip(pending, vectors, strict=True):
        materialized = {**view, "embedding": vector}
        working[view_id] = materialized
        accepted_entry["resulting_view"] = copy.deepcopy(materialized)
    return working, accepted, rejected, next_view_number


def _chain_hash(previous_chain: str, step: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "previous_chain_sha256": previous_chain,
            "frontier_id": step["frontier_id"],
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
    *,
    prompt_template: str | None = None,
    prompt_sha256: str | None = None,
    prior_fact_cap: int = PRIOR_FACT_CAP,
) -> dict[str, Any]:
    """Create one logged step per frontier and one chat call after the first."""

    resolved_prompt = prompt_template or ONLINE_FRONTIER_UPDATE_PROMPT
    resolved_prompt_sha = prompt_sha256 or hashlib.sha256(resolved_prompt.encode("utf-8")).hexdigest()
    base = legacy._base_parts(session_graph)
    selector = _PrefixSelector(base)
    state = _HashState(base)
    active_views: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    next_view_number = 1
    genesis = canonical_sha256(
        {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "base_content_fingerprint": session_graph.get("content_fingerprint"),
            "frontier_order": base["session_ids"],
            "prompt_sha256": resolved_prompt_sha,
        }
    )
    previous_chain = genesis
    previous_post = state.digest()

    for ordinal, frontier_id in enumerate(base["session_ids"]):
        pre_state = previous_post
        current_summary = base["summary_by_session"][frontier_id]
        current_facts = list(base["facts_by_session"][frontier_id])
        current_edges = list(base["support_edges_by_session"][frontier_id])
        prompt_input = _build_prompt_input(
            frontier_id,
            current_summary=current_summary,
            current_facts=current_facts,
            active_views=active_views,
            base=base,
            selector=selector,
            prior_fact_cap=prior_fact_cap,
        )
        prompt_input_sha = canonical_sha256(prompt_input)
        state.append(current_summary, current_facts, current_edges)
        post_append = state.digest()
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        raw_response: Any = None
        model_error: str | None = None
        model_call_made = ordinal > 0
        if model_call_made:
            prompt = resolved_prompt.format(
                frontier_id=frontier_id,
                online_input_json=json.dumps(prompt_input, ensure_ascii=False, separators=(",", ":")),
            )
            # The checkpointing chat has already exhausted its bounded retry
            # policy before it raises. Transport/provider and view-embedding
            # failures are fatal and must propagate immediately. Malformed
            # individual operations are rejected inside _apply_model_response.
            raw_response = chat_fn(
                prompt,
                tag="creamem3_v4_online_frontier_fuse",
                temperature=0.0,
                max_tokens=16384,
                json_mode=True,
            )
            value = legacy._parse_json_object(raw_response)
            active_views, accepted, rejected, next_view_number = _apply_model_response(
                value,
                frontier_id=frontier_id,
                prompt_input=prompt_input,
                active_views=active_views,
                fact_by_id=base["fact_by_id"],
                fact_order=base["fact_order"],
                next_view_number=next_view_number,
                embed_fn=embed_fn,
                embedding_dim=int(base["embedding_dim"]),
            )
        if accepted:
            state.replace_views(active_views)
            post_state = state.digest()
        else:
            post_state = post_append
        selector.append(current_summary, current_facts)
        response_sha = legacy._response_hash(raw_response)
        step: dict[str, Any] = {
            "step_id": f"T{frontier_id:03d}",
            "frontier_id": frontier_id,
            "session_id": frontier_id,
            "ordinal": ordinal + 1,
            "visible_frontier_ids": list(base["session_ids"][: ordinal + 1]),
            "model_call_made": model_call_made,
            "model_call_count": int(model_call_made),
            "prompt_input": prompt_input,
            "allowed_ids": copy.deepcopy(prompt_input["allowed_ids"]),
            "prompt_input_sha256": prompt_input_sha,
            "model_response_sha256": response_sha,
            "model_error": model_error,
            "accepted": accepted,
            "rejected": rejected,
            "response_normalizations": [],
            "pre_state_sha256": pre_state,
            "post_append_state_sha256": post_append,
            "post_state_sha256": post_state,
            "previous_chain_sha256": previous_chain,
        }
        step["chain_sha256"] = _chain_hash(previous_chain, step)
        previous_chain = step["chain_sha256"]
        previous_post = post_state
        steps.append(step)

    final_views = [active_views[view_id] for view_id in sorted(active_views)]
    # Structural sharing is safe: base nodes are immutable after construction.
    graph = dict(session_graph)
    graph["graph_variant"] = "online"
    graph["architecture"] = {
        **dict(graph.get("architecture", {})),
        "online_update_policy": "strict_causal_frontier_prefix_only_create_or_extend",
        "event_view_membership": "monotonic_non_destructive",
        "online_engine": "creamem3.online_memory_v4",
    }
    graph["prompt_sha256"] = {
        **dict(graph.get("prompt_sha256", {})),
        "online_frontier_update": resolved_prompt_sha,
    }
    graph["summary_nodes"] = list(base["summaries"]) + final_views
    graph["fact_nodes"] = list(base["facts"])
    graph["edges"] = list(base["edges"]) + legacy._view_edges(active_views)
    graph["stats"] = {
        **dict(graph.get("stats", {})),
        "event_view_nodes": len(final_views),
        "summary_nodes": len(graph["summary_nodes"]),
        "edges": len(graph["edges"]),
        "cross_session_view_edges": sum(len(view["member_fact_ids"]) for view in final_views),
        "online_fusion_calls": sum(step["model_call_count"] for step in steps),
        "online_accepted_operations": sum(len(step["accepted"]) for step in steps),
        "online_rejected_operations": sum(len(step["rejected"]) for step in steps),
    }
    graph["online_fusion"] = {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "policy": "strict_causal_frontier_prefix_only_create_or_extend",
        "prompt_sha256": resolved_prompt_sha,
        "frontier_order": list(base["session_ids"]),
        "session_order": list(base["session_ids"]),
        "candidate_policy": {
            "summary_dense_top_k": SUMMARY_TOP_K,
            "fact_dense_top_k": DENSE_FACT_TOP_K,
            "bm25_top_k": BM25_FACT_TOP_K,
            "active_view_top_k": ACTIVE_VIEW_TOP_K,
            "prior_fact_cap": prior_fact_cap,
            "current_fact_cap": None,
        },
        "performance_policy": {
            "immutable_node_hash_cache": True,
            "incremental_bm25": True,
            "pre_normalized_dense_prefix": True,
            "structural_embedding_sharing": True,
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
            "active_views": [legacy._without_embedding(view) for view in final_views],
            "final_chain_sha256": previous_chain,
        }
    )
    return graph


def audit_online_graph(
    session_graph: Mapping[str, Any],
    online_graph: Mapping[str, Any],
    *,
    expected_prompt_sha256: str | None = None,
    expected_prior_fact_cap: int = PRIOR_FACT_CAP,
) -> dict[str, Any]:
    """Replay frontier inputs and verify causal/hash/final-graph invariants."""

    prompt_sha = expected_prompt_sha256 or ONLINE_FRONTIER_PROMPT_SHA256
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    try:
        base = legacy._base_parts(session_graph)
        fusion = online_graph.get("online_fusion", {})
        steps = list(fusion.get("steps", [])) if isinstance(fusion, Mapping) else []
        check("online_graph_variant", online_graph.get("graph_variant") == "online")
        check("online_schema_version", fusion.get("schema_version") == ONLINE_SCHEMA_VERSION)
        check("frontier_order", fusion.get("frontier_order") == base["session_ids"])
        check("session_order_alias", fusion.get("session_order") == base["session_ids"])
        check(
            "online_prompt_hash",
            fusion.get("prompt_sha256") == prompt_sha
            and online_graph.get("prompt_sha256", {}).get("online_frontier_update")
            == prompt_sha,
        )
        check("base_content_fingerprint", fusion.get("base_content_fingerprint") == session_graph.get("content_fingerprint"))
        check(
            "base_fact_identity_manifest",
            fusion.get("base_fact_identity_sha256") == canonical_sha256(base["facts"]),
        )
        check(
            "candidate_policy",
            fusion.get("candidate_policy")
            == {
                "summary_dense_top_k": SUMMARY_TOP_K,
                "fact_dense_top_k": DENSE_FACT_TOP_K,
                "bm25_top_k": BM25_FACT_TOP_K,
                "active_view_top_k": ACTIVE_VIEW_TOP_K,
                "prior_fact_cap": expected_prior_fact_cap,
                "current_fact_cap": None,
            },
        )
        check(
            "embedding_contract",
            session_graph.get("embedding_contract")
            == {
                "model": "gemini-embedding-001",
                "output_dimensionality": 768,
                "task_type_sent": None,
                "normalization": "l2_unit_v1",
            },
        )
        check(
            "online_embedding_contract_matches_base",
            online_graph.get("embedding_contract")
            == session_graph.get("embedding_contract"),
        )
        check("facts_byte_identical", online_graph.get("fact_nodes") == session_graph.get("fact_nodes"))
        check("packets_byte_identical", online_graph.get("session_packets") == session_graph.get("session_packets"))
        check(
            "session_summaries_byte_identical",
            [
                node
                for node in online_graph.get("summary_nodes", [])
                if node.get("node_type") == "session_packet"
            ]
            == base["summaries"],
        )
        embedding_manifest = {
            str(node["id"]): canonical_sha256(node["embedding"])
            for node in [*base["summaries"], *base["facts"]]
        }
        check(
            "embedding_manifest",
            session_graph.get("embedding_manifest_sha256")
            == canonical_sha256(embedding_manifest),
        )
        check(
            "online_embedding_manifest_matches_base",
            online_graph.get("embedding_manifest_sha256")
            == session_graph.get("embedding_manifest_sha256"),
        )
        check("one_step_per_frontier", len(steps) == len(base["session_ids"]))

        selector = _PrefixSelector(base)
        state = _HashState(base)
        active_views: dict[str, dict[str, Any]] = {}
        next_view_number = 1
        genesis = canonical_sha256(
            {
                "schema_version": ONLINE_SCHEMA_VERSION,
                "base_content_fingerprint": session_graph.get("content_fingerprint"),
                "frontier_order": base["session_ids"],
                "prompt_sha256": prompt_sha,
            }
        )
        check("genesis_chain", fusion.get("genesis_chain_sha256") == genesis)
        previous_chain = genesis
        previous_post = state.digest()
        replay_ok = len(steps) == len(base["session_ids"])
        call_ok = True
        prefix_ok = True
        hash_ok = True
        rejected_future_references = 0
        for ordinal, frontier_id in enumerate(base["session_ids"]):
            if ordinal >= len(steps) or not isinstance(steps[ordinal], Mapping):
                replay_ok = False
                break
            step = steps[ordinal]
            summary = base["summary_by_session"][frontier_id]
            facts = list(base["facts_by_session"][frontier_id])
            edges = list(base["support_edges_by_session"][frontier_id])
            expected_input = _build_prompt_input(
                frontier_id,
                current_summary=summary,
                current_facts=facts,
                active_views=active_views,
                base=base,
                selector=selector,
                prior_fact_cap=expected_prior_fact_cap,
            )
            actual_input = step.get("prompt_input")
            prefix_ok = prefix_ok and actual_input == expected_input
            prefix_ok = prefix_ok and not legacy._scan_future_dia_ids(actual_input, frontier_id)
            expected_call = ordinal > 0
            call_ok = call_ok and bool(step.get("model_call_made")) == expected_call and step.get("model_call_count") == int(expected_call)
            if (
                step.get("frontier_id") != frontier_id
                or step.get("visible_frontier_ids") != base["session_ids"][: ordinal + 1]
                or step.get("pre_state_sha256") != previous_post
                or step.get("prompt_input_sha256") != canonical_sha256(actual_input)
                or step.get("allowed_ids") != expected_input["allowed_ids"]
            ):
                replay_ok = False
            state.append(summary, facts, edges)
            post_append = state.digest()
            if step.get("post_append_state_sha256") != post_append:
                replay_ok = False

            touched: set[str] = set()
            accepted_rows = step.get("accepted", [])
            if not isinstance(accepted_rows, list):
                accepted_rows = []
                replay_ok = False
            for accepted in accepted_rows:
                try:
                    operation = {
                        key: copy.deepcopy(accepted[key])
                        for key in ("operation", "target_view_id", "view_type", "summary_claims", "member_fact_ids")
                    }
                    normalized, expected_view = legacy._validate_operation(
                        operation,
                        session_id=frontier_id,
                        prompt_input=expected_input,
                        working_views=active_views,
                        touched_view_ids=touched,
                        fact_by_id=base["fact_by_id"],
                        fact_order=base["fact_order"],
                        next_view_id=f"OV{next_view_number:04d}",
                    )
                    actual_view = accepted.get("resulting_view")
                    if not isinstance(actual_view, Mapping) or legacy._without_embedding(actual_view) != legacy._without_embedding(expected_view):
                        raise GraphMemoryError("accepted view replay mismatch")
                    vector = actual_view.get("embedding", [])
                    if len(vector) != base["embedding_dim"] or not all(math.isfinite(float(value)) for value in vector):
                        raise GraphMemoryError("accepted view embedding invalid")
                    if abs(
                        math.sqrt(
                            math.fsum(float(value) * float(value) for value in vector)
                        )
                        - 1.0
                    ) > 1e-5:
                        raise GraphMemoryError("accepted view embedding is not unit-normalized")
                    if any(
                        int(base["fact_by_id"][fact_id]["session_id"])
                        > frontier_id
                        for fact_id in actual_view.get("member_fact_ids", [])
                    ):
                        raise GraphMemoryError("accepted view contains a future-frontier fact")
                    if legacy._scan_future_dia_ids(actual_view, frontier_id):
                        raise GraphMemoryError("accepted view contains a future Dia ID")
                    if normalized["operation"] == "create":
                        next_view_number += 1
                    else:
                        touched.add(str(normalized["target_view_id"]))
                    active_views[str(actual_view["id"])] = dict(actual_view)
                except Exception:
                    replay_ok = False
            rejected_rows = step.get("rejected", [])
            if not isinstance(rejected_rows, list):
                replay_ok = False
                rejected_rows = []
            for rejected in rejected_rows:
                rejected_future_references += len(
                    legacy._scan_future_dia_ids(rejected, frontier_id)
                )
                raw_operation = (
                    rejected.get("operation") if isinstance(rejected, Mapping) else None
                )
                if isinstance(raw_operation, Mapping):
                    for fact_id in raw_operation.get("member_fact_ids", []):
                        fact = base["fact_by_id"].get(fact_id)
                        if fact is not None and int(fact["session_id"]) > frontier_id:
                            rejected_future_references += 1
            if accepted_rows:
                state.replace_views(active_views)
                post_state = state.digest()
            else:
                post_state = post_append
            if step.get("post_state_sha256") != post_state:
                replay_ok = False
            if step.get("previous_chain_sha256") != previous_chain:
                hash_ok = False
            expected_chain = _chain_hash(previous_chain, step)
            if step.get("chain_sha256") != expected_chain:
                hash_ok = False
            previous_chain = expected_chain
            previous_post = post_state
            selector.append(summary, facts)

        final_views = [active_views[key] for key in sorted(active_views)]
        graph_views = [node for node in online_graph.get("summary_nodes", []) if node.get("node_type") == "event_view"]
        check("exactly_one_model_call_after_first_frontier", call_ok)
        check(
            "no_swallowed_model_or_embedding_errors",
            all(step.get("model_error") is None for step in steps),
        )
        check("prompt_inputs_are_exact_frontier_prefix", prefix_ok)
        check("accepted_operations_replay", replay_ok)
        check("continuous_hash_chain", hash_ok and fusion.get("final_chain_sha256") == previous_chain)
        check("final_active_views_match", graph_views == final_views)
        check("final_edges_match", online_graph.get("edges") == base["edges"] + legacy._view_edges(active_views))
        check("all_views_cross_frontier", all(len(set(view.get("session_ids", []))) >= 2 for view in graph_views))
        if rejected_future_references:
            warnings.append(
                {
                    "name": "rejected_future_reference_attempts",
                    "count": rejected_future_references,
                    "detail": "Future references occurred only inside rejected operations.",
                }
            )
        all_nodes = [*online_graph.get("summary_nodes", []), *online_graph.get("fact_nodes", [])]
        check(
            "all_embeddings_768_finite_unit_norm",
            all(
                len(node.get("embedding", [])) == 768
                and all(math.isfinite(float(value)) for value in node.get("embedding", []))
                and abs(
                    math.sqrt(
                        math.fsum(float(value) * float(value) for value in node.get("embedding", []))
                    )
                    - 1.0
                )
                <= 1e-5
                for node in all_nodes
            ),
        )
        expected_fp = canonical_sha256(
            {
                "base": session_graph.get("content_fingerprint"),
                "online_schema": ONLINE_SCHEMA_VERSION,
                "active_views": [legacy._without_embedding(view) for view in final_views],
                "final_chain_sha256": previous_chain,
            }
        )
        check("content_fingerprint", online_graph.get("content_fingerprint") == expected_fp)
        stats = online_graph.get("stats", {})
        check(
            "online_stats_consistent",
            isinstance(stats, Mapping)
            and stats.get("event_view_nodes") == len(graph_views)
            and stats.get("summary_nodes")
            == len(online_graph.get("summary_nodes", []))
            and stats.get("edges") == len(online_graph.get("edges", []))
            and stats.get("online_fusion_calls")
            == max(0, len(base["session_ids"]) - 1)
            and stats.get("online_accepted_operations")
            == sum(len(step.get("accepted", [])) for step in steps)
            and stats.get("online_rejected_operations")
            == sum(len(step.get("rejected", [])) for step in steps)
            and stats.get("cross_session_view_edges")
            == sum(len(view.get("member_fact_ids", [])) for view in graph_views),
        )
    except Exception as exc:
        check("audit_exception", False, f"{type(exc).__name__}: {str(exc)[-500:]}")

    failures = [row for row in checks if not row["passed"]]
    return {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "warnings": warnings,
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "replayed_frontiers": len(session_graph.get("session_packets", [])),
    }


__all__ = [
    "ACTIVE_VIEW_TOP_K",
    "BM25_FACT_TOP_K",
    "DENSE_FACT_TOP_K",
    "MAX_OPERATIONS",
    "ONLINE_FRONTIER_PROMPT_SHA256",
    "ONLINE_FRONTIER_UPDATE_PROMPT",
    "ONLINE_SCHEMA_VERSION",
    "PRIOR_FACT_CAP",
    "SUMMARY_TOP_K",
    "add_online_event_views",
    "audit_online_graph",
]

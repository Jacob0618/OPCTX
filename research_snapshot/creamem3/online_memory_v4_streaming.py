"""Persistent one-frontier-at-a-time execution for :mod:`online_memory_v4`.

The batch v4 builder receives a completed session graph and then replays it in
frontier order.  This module exposes the same transition as a transactional
``advance`` operation.  A state instance owns only the prefix already written;
future packets are neither accepted by the constructor nor needed by a step.

The live checkpoint chain is causal and can therefore be sealed before the
final graph exists.  ``to_graph`` recomputes the legacy v4 chain anchor from the
completed base graph, so a run with identical packets/model responses is
byte-semantically compatible with ``online_memory_v4.add_online_event_views``.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from creamem3 import online_memory_v3 as legacy
from creamem3 import online_memory_v4 as batch_v4
from creamem3.graph_memory_v2 import (
    BM25Index,
    GRAPH_SCHEMA_VERSION,
    GraphMemoryError,
    canonical_sha256,
)


STREAMING_SCHEMA_VERSION = "creamem3.strict_causal_online.v4_streaming.v1"
CHECKPOINT_SCHEMA_VERSION = f"{STREAMING_SCHEMA_VERSION}.checkpoint"
CHECKPOINT_SEAL_SCHEMA_VERSION = "creamem3.streaming_checkpoint_seal.v1"

_COLLECTION_KEYS = {
    "session_packets",
    "provenance_map",
    "summary_nodes",
    "fact_nodes",
    "edges",
    "stats",
    "content_fingerprint",
}
_DELTA_KEYS = {
    "session_packet",
    "summary_node",
    "fact_nodes",
    "support_edges",
    "provenance_map",
}


def _without_embedding(node: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in node.items() if key != "embedding"}


def _seal(payload: Mapping[str, Any], kind: str = "streaming_online_checkpoint") -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    value.pop("seal", None)
    value["seal"] = {
        "schema_version": CHECKPOINT_SEAL_SCHEMA_VERSION,
        "kind": kind,
        "payload_sha256": canonical_sha256(value),
    }
    return value


def _verify_seal(value: Mapping[str, Any], kind: str = "streaming_online_checkpoint") -> None:
    seal = value.get("seal")
    if not isinstance(seal, Mapping):
        raise GraphMemoryError("streaming checkpoint lacks a seal")
    payload = copy.deepcopy(dict(value))
    payload.pop("seal", None)
    expected = {
        "schema_version": CHECKPOINT_SEAL_SCHEMA_VERSION,
        "kind": kind,
        "payload_sha256": canonical_sha256(payload),
    }
    if dict(seal) != expected:
        raise GraphMemoryError("streaming checkpoint seal mismatch")


def _checked_vector(value: Any, *, dimension: int, label: str) -> list[float]:
    if not isinstance(value, list):
        raise GraphMemoryError(f"{label} embedding must be a list")
    vector = [float(item) for item in value]
    if len(vector) != dimension or not all(math.isfinite(item) for item in vector):
        raise GraphMemoryError(f"{label} embedding is not finite {dimension}D")
    norm = math.sqrt(math.fsum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise GraphMemoryError(f"{label} embedding has zero/invalid norm")
    return vector


class _StreamingPrefixSelector:
    """Incremental equivalent of ``online_memory_v4._PrefixSelector``."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.summaries: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.summary_matrix = np.empty((0, dimension), dtype=np.float32)
        self.fact_matrix = np.empty((0, dimension), dtype=np.float32)
        self.bm25 = BM25Index()

    @staticmethod
    def _rank(
        nodes: Sequence[Mapping[str, Any]], scores: Sequence[float], top_k: int
    ) -> list[str]:
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
        support_by_summary: Mapping[str, Sequence[str]],
        fact_order: Mapping[str, int],
    ) -> dict[str, Any]:
        current_nodes = [current_summary, *current_facts]
        query_matrix = batch_v4._normalized_matrix(
            current_nodes, expected_dim=self.dimension
        )
        if self.summaries:
            summary_scores_array = (self.summary_matrix @ query_matrix[:1].T).reshape(-1)
        else:
            summary_scores_array = np.empty((0,), dtype=np.float32)
        if self.facts:
            dense_scores_array = (self.fact_matrix @ query_matrix.T).max(axis=1)
        else:
            dense_scores_array = np.empty((0,), dtype=np.float32)
        summary_scores = [float(item) for item in summary_scores_array]
        dense_scores = [float(item) for item in dense_scores_array]
        summary_ids = self._rank(
            self.summaries, summary_scores, batch_v4.SUMMARY_TOP_K
        )
        dense_ids = self._rank(self.facts, dense_scores, batch_v4.DENSE_FACT_TOP_K)
        dense_score_by_id = {
            str(node["id"]): float(score)
            for node, score in zip(self.facts, dense_scores, strict=True)
        }
        lexical_query = " ".join(
            [str(current_summary.get("text", ""))]
            + [str(fact.get("text", "")) for fact in current_facts]
        )
        bm25_rows = self.bm25.search(lexical_query, batch_v4.BM25_FACT_TOP_K)
        view_nodes = [active_views[key] for key in sorted(active_views)]
        view_scores = legacy._cosine_scores(
            view_nodes, current_nodes, expected_dim=self.dimension
        )
        view_ids = self._rank(view_nodes, view_scores, batch_v4.ACTIVE_VIEW_TOP_K)

        channels: dict[str, dict[str, Any]] = defaultdict(dict)
        for rank, summary_id in enumerate(summary_ids, 1):
            for fact_id in support_by_summary.get(summary_id, []):
                channels[str(fact_id)].setdefault("summary_ids", []).append(summary_id)
                channels[str(fact_id)].setdefault("summary_ranks", []).append(rank)
        for rank, fact_id in enumerate(dense_ids, 1):
            channels[fact_id]["dense_rank"] = rank
        for rank, (fact_id, score) in enumerate(bm25_rows, 1):
            channels[str(fact_id)]["bm25_rank"] = rank
            channels[str(fact_id)]["bm25_score"] = float(score)
        prior_ids = {str(fact["id"]) for fact in self.facts}
        union_ids = [fact_id for fact_id in channels if fact_id in prior_ids]
        union_ids.sort(
            key=lambda fact_id: (
                -dense_score_by_id.get(fact_id, -1.0),
                int(fact_order[fact_id]),
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
                for node, score in zip(self.summaries, summary_scores, strict=True)
                if str(node["id"]) in summary_ids
            },
            "view_scores": {
                str(node["id"]): float(score)
                for node, score in zip(view_nodes, view_scores, strict=True)
                if str(node["id"]) in view_ids
            },
        }

    def append(
        self, summary: Mapping[str, Any], facts: Sequence[Mapping[str, Any]]
    ) -> None:
        summary_node = copy.deepcopy(dict(summary))
        fact_nodes = [copy.deepcopy(dict(fact)) for fact in facts]
        summary_vector = batch_v4._normalized_matrix(
            [summary_node], expected_dim=self.dimension
        )
        fact_vectors = batch_v4._normalized_matrix(
            fact_nodes, expected_dim=self.dimension
        )
        self.summary_matrix = np.vstack((self.summary_matrix, summary_vector))
        if fact_nodes:
            self.fact_matrix = np.vstack((self.fact_matrix, fact_vectors))
        self.summaries.append(summary_node)
        for fact in fact_nodes:
            self.facts.append(fact)
            self.bm25.add(str(fact["id"]), str(fact.get("text", "")))


class _StreamingHashState:
    """Incrementally produce the exact visible-state digest used by v4."""

    def __init__(self) -> None:
        self.summary_hashes: list[str] = []
        self.fact_hashes: list[str] = []
        self.support_edges: list[dict[str, Any]] = []
        self.active_views: dict[str, dict[str, Any]] = {}
        self.active_view_hashes: dict[str, str] = {}

    def append(
        self,
        summary: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
    ) -> None:
        self.summary_hashes.append(canonical_sha256(summary))
        self.fact_hashes.extend(canonical_sha256(fact) for fact in facts)
        self.support_edges.extend(copy.deepcopy(dict(edge)) for edge in edges)

    def replace_views(self, views: Mapping[str, Mapping[str, Any]]) -> None:
        copied = {str(key): copy.deepcopy(dict(value)) for key, value in views.items()}
        self.active_views = copied
        self.active_view_hashes = {
            view_id: canonical_sha256(view) for view_id, view in copied.items()
        }

    def digest(self) -> str:
        return canonical_sha256(
            {
                "session_summary_node_hashes": self.summary_hashes,
                "fact_node_hashes": self.fact_hashes,
                "support_edges": sorted(
                    (copy.deepcopy(edge) for edge in self.support_edges),
                    key=lambda edge: (str(edge["source_id"]), str(edge["target_id"])),
                ),
                "active_view_node_hashes": {
                    view_id: self.active_view_hashes[view_id]
                    for view_id in sorted(self.active_views)
                },
                "view_edges": legacy._view_edges(self.active_views),
            }
        )


class StreamingOnlineState:
    """A transactional, serializable v4 Online prefix state."""

    def __init__(
        self,
        *,
        graph_header: Mapping[str, Any],
        stream_id: str,
        embedding_dim: int = 768,
        prior_fact_cap: int = batch_v4.PRIOR_FACT_CAP,
        prompt_template: str | None = None,
        prompt_sha256: str | None = None,
    ) -> None:
        if embedding_dim != 768:
            raise GraphMemoryError("LME streaming Online requires 768D embeddings")
        if not isinstance(prior_fact_cap, int) or prior_fact_cap < 1:
            raise GraphMemoryError("prior_fact_cap must be a positive integer")
        header = copy.deepcopy(dict(graph_header))
        leaked = {key for key in _COLLECTION_KEYS if header.get(key) not in (None, [], {})}
        if leaked:
            raise GraphMemoryError(
                f"streaming graph header contains future-capable collections: {sorted(leaked)}"
            )
        for key in _COLLECTION_KEYS:
            header.pop(key, None)
        if header.get("schema_version") != GRAPH_SCHEMA_VERSION:
            raise GraphMemoryError("streaming graph header has an unsupported schema")
        if header.get("graph_variant") not in (None, "session"):
            raise GraphMemoryError("streaming graph header must describe a session graph")
        header["graph_variant"] = "session"
        resolved_prompt = prompt_template or batch_v4.ONLINE_FRONTIER_UPDATE_PROMPT
        computed_prompt_sha = hashlib.sha256(resolved_prompt.encode("utf-8")).hexdigest()
        if prompt_sha256 is not None and prompt_sha256 != computed_prompt_sha:
            raise GraphMemoryError("streaming Online prompt hash mismatch")

        self.graph_header = header
        self.stream_id = str(stream_id)
        self.embedding_dim = embedding_dim
        self.prior_fact_cap = prior_fact_cap
        self.prompt_template = resolved_prompt
        self.prompt_sha256 = computed_prompt_sha
        self._deltas: list[dict[str, Any]] = []
        self._session_packets: list[dict[str, Any]] = []
        self._summaries: list[dict[str, Any]] = []
        self._facts: list[dict[str, Any]] = []
        self._edges: list[dict[str, Any]] = []
        self._provenance: dict[str, Any] = {}
        self._active_views: dict[str, dict[str, Any]] = {}
        self._steps: list[dict[str, Any]] = []
        self._next_view_number = 1
        self._selector = _StreamingPrefixSelector(embedding_dim)
        self._hash_state = _StreamingHashState()
        self._stream_genesis = canonical_sha256(
            {
                "schema_version": STREAMING_SCHEMA_VERSION,
                "stream_id": self.stream_id,
                "graph_header_sha256": canonical_sha256(self.graph_header),
                "embedding_dim": embedding_dim,
                "prior_fact_cap": prior_fact_cap,
                "prompt_sha256": self.prompt_sha256,
            }
        )
        self._stream_chain = self._stream_genesis

    @property
    def processed_frontiers(self) -> int:
        return len(self._deltas)

    @property
    def active_view_count(self) -> int:
        return len(self._active_views)

    def _validate_delta(self, delta: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(delta, Mapping) or set(delta) != _DELTA_KEYS:
            raise GraphMemoryError(
                f"frontier delta must contain exactly {sorted(_DELTA_KEYS)}"
            )
        value = copy.deepcopy(dict(delta))
        packet = value["session_packet"]
        summary = value["summary_node"]
        facts = value["fact_nodes"]
        edges = value["support_edges"]
        provenance = value["provenance_map"]
        if not isinstance(packet, dict) or not isinstance(summary, dict):
            raise GraphMemoryError("frontier delta packet/summary must be objects")
        if not isinstance(facts, list) or not all(isinstance(row, dict) for row in facts):
            raise GraphMemoryError("frontier delta facts must be objects")
        if not isinstance(edges, list) or not all(isinstance(row, dict) for row in edges):
            raise GraphMemoryError("frontier delta edges must be objects")
        if not isinstance(provenance, dict):
            raise GraphMemoryError("frontier delta provenance must be an object")
        session_ids = summary.get("session_ids")
        if not isinstance(session_ids, list) or len(session_ids) != 1:
            raise GraphMemoryError("frontier summary must bind exactly one session")
        session_id = int(session_ids[0])
        expected_session = self.processed_frontiers + 1
        if session_id != expected_session or int(packet.get("session_id", -1)) != session_id:
            raise GraphMemoryError("streaming frontier IDs must be contiguous and aligned")
        if summary.get("node_type") != "session_packet":
            raise GraphMemoryError("frontier summary has the wrong node type")
        _checked_vector(
            summary.get("embedding"), dimension=self.embedding_dim, label="summary"
        )
        existing_node_ids = {
            str(node["id"]) for node in [*self._summaries, *self._facts]
        }
        summary_id = str(summary.get("id", ""))
        if not summary_id or summary_id in existing_node_ids:
            raise GraphMemoryError("frontier summary ID is empty or duplicated")
        fact_ids: list[str] = []
        current_source_ids: set[str] = set()
        for fact in facts:
            fact_id = str(fact.get("id", ""))
            if (
                not fact_id
                or fact_id in existing_node_ids
                or fact_id in fact_ids
                or int(fact.get("session_id", -1)) != session_id
                or fact.get("node_type") != "atomic_fact"
            ):
                raise GraphMemoryError("frontier fact identity/session is invalid")
            _checked_vector(
                fact.get("embedding"), dimension=self.embedding_dim, label=fact_id
            )
            sources = [str(source) for source in fact.get("source_dia_ids", [])]
            if not sources:
                raise GraphMemoryError(f"frontier fact {fact_id} lacks current provenance")
            current_source_ids.update(sources)
            fact_ids.append(fact_id)
        edge_pairs: list[tuple[str, str]] = []
        for edge in edges:
            if edge.get("edge_type") != "SUPPORTS":
                raise GraphMemoryError("frontier delta contains a non-SUPPORTS edge")
            edge_pairs.append((str(edge.get("source_id", "")), str(edge.get("target_id", ""))))
        expected_pairs = [(summary_id, fact_id) for fact_id in fact_ids]
        if edge_pairs != expected_pairs:
            raise GraphMemoryError("frontier SUPPORTS edges do not exactly match current facts")
        if any(source not in provenance for source in current_source_ids):
            raise GraphMemoryError("frontier fact cites provenance outside the current delta")
        if set(provenance).intersection(self._provenance):
            raise GraphMemoryError("frontier provenance IDs overlap the existing prefix")
        if str(summary.get("text", "")).strip() != str(packet.get("batch_summary", "")).strip():
            raise GraphMemoryError("frontier packet and summary text differ")
        return value

    def _base_with(self, delta: Mapping[str, Any]) -> dict[str, Any]:
        summary = delta["summary_node"]
        facts = list(delta["fact_nodes"])
        edges = list(delta["support_edges"])
        packet = delta["session_packet"]
        summaries = [*self._summaries, summary]
        all_facts = [*self._facts, *facts]
        all_edges = [*self._edges, *edges]
        session_ids = [int(node["session_ids"][0]) for node in summaries]
        summary_by_session = {int(node["session_ids"][0]): node for node in summaries}
        fact_by_id = {str(node["id"]): node for node in all_facts}
        fact_order = {str(node["id"]): index for index, node in enumerate(all_facts)}
        facts_by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for fact in all_facts:
            facts_by_session[int(fact["session_id"])].append(fact)
        support_by_summary: dict[str, list[str]] = defaultdict(list)
        support_edges_by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for edge in all_edges:
            support_by_summary[str(edge["source_id"])].append(str(edge["target_id"]))
            support_edges_by_session[int(fact_by_id[str(edge["target_id"])]["session_id"])].append(edge)
        return {
            "session_ids": session_ids,
            "summaries": summaries,
            "summary_by_session": summary_by_session,
            "facts": all_facts,
            "fact_by_id": fact_by_id,
            "fact_order": fact_order,
            "facts_by_session": facts_by_session,
            "edges": all_edges,
            "support_by_summary": support_by_summary,
            "support_edges_by_session": support_edges_by_session,
            "packet_by_session": {
                int(row["session_id"]): row for row in [*self._session_packets, packet]
            },
            "embedding_dim": self.embedding_dim,
        }

    def _prompt_input(
        self, delta: Mapping[str, Any], base: Mapping[str, Any]
    ) -> dict[str, Any]:
        summary = delta["summary_node"]
        session_id = int(summary["session_ids"][0])
        selected = self._selector.select(
            current_summary=summary,
            current_facts=delta["fact_nodes"],
            active_views=self._active_views,
            prior_fact_cap=self.prior_fact_cap,
            support_by_summary=base["support_by_summary"],
            fact_order=base["fact_order"],
        )
        candidate_fact_ids = list(selected["prior_fact_ids"])
        candidate_view_ids = list(selected["active_view_ids"])
        current_fact_ids = [str(fact["id"]) for fact in delta["fact_nodes"]]
        packet = delta["session_packet"]
        return {
            "schema_version": batch_v4.ONLINE_SCHEMA_VERSION,
            "frontier_id": session_id,
            "policy": {
                "summary_dense_top_k": batch_v4.SUMMARY_TOP_K,
                "fact_dense_top_k": batch_v4.DENSE_FACT_TOP_K,
                "bm25_top_k": batch_v4.BM25_FACT_TOP_K,
                "active_view_top_k": batch_v4.ACTIVE_VIEW_TOP_K,
                "prior_fact_cap": self.prior_fact_cap,
                "membership_update": "monotonic_create_or_extend",
            },
            "current_frontier": {
                "frontier_id": session_id,
                "physical_sessions": copy.deepcopy(list(packet.get("physical_sessions", []))),
                "timestamp": str(summary.get("timestamp", "")),
                "iso_date_range": str(summary.get("iso_date", "")),
                "summary_id": str(summary["id"]),
                "summary": str(summary.get("text", "")),
                "facts": [legacy._prompt_fact(fact) for fact in delta["fact_nodes"]],
            },
            "prior_candidate_facts": [
                legacy._prompt_fact(
                    base["fact_by_id"][fact_id],
                    selected["fact_channels"].get(fact_id, {}),
                )
                for fact_id in candidate_fact_ids
            ],
            "candidate_active_views": [
                legacy._prompt_view(self._active_views[view_id])
                for view_id in candidate_view_ids
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
                "current_facts_before_cap": len(delta["fact_nodes"]),
                "current_facts_after_cap": len(delta["fact_nodes"]),
                "current_fact_cap": None,
                "top_prior_summary_scores": dict(selected["summary_scores"]),
                "top_active_view_scores": dict(selected["view_scores"]),
            },
        }

    def _commit_delta(
        self,
        delta: Mapping[str, Any],
        *,
        working_views: Mapping[str, Mapping[str, Any]],
        working_hash_state: _StreamingHashState,
        step: Mapping[str, Any],
        next_view_number: int,
    ) -> None:
        copied = copy.deepcopy(dict(delta))
        self._deltas.append(copied)
        self._session_packets.append(copied["session_packet"])
        self._summaries.append(copied["summary_node"])
        self._facts.extend(copied["fact_nodes"])
        self._edges.extend(copied["support_edges"])
        self._provenance.update(copied["provenance_map"])
        self._selector.append(copied["summary_node"], copied["fact_nodes"])
        self._active_views = {
            str(key): copy.deepcopy(dict(value)) for key, value in working_views.items()
        }
        self._hash_state = working_hash_state
        self._steps.append(copy.deepcopy(dict(step)))
        self._next_view_number = next_view_number
        self._stream_chain = str(step["chain_sha256"])

    def advance(
        self,
        delta: Mapping[str, Any],
        chat_fn: Any,
        embed_fn: Any,
    ) -> dict[str, Any]:
        """Atomically append and process exactly one new frontier.

        No state is committed if prompt parsing, the model call, operation
        validation, or view embedding raises.
        """

        current = self._validate_delta(delta)
        base = self._base_with(current)
        summary = current["summary_node"]
        facts = current["fact_nodes"]
        edges = current["support_edges"]
        session_id = int(summary["session_ids"][0])
        ordinal = self.processed_frontiers
        prompt_input = self._prompt_input(current, base)
        prompt_input_sha = canonical_sha256(prompt_input)
        pre_state = self._hash_state.digest()
        working_hash = copy.deepcopy(self._hash_state)
        working_hash.append(summary, facts, edges)
        post_append = working_hash.digest()
        working_views = {
            key: copy.deepcopy(value) for key, value in self._active_views.items()
        }
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        raw_response: Any = None
        model_call_made = ordinal > 0
        next_view_number = self._next_view_number
        if model_call_made:
            prompt = self.prompt_template.format(
                frontier_id=session_id,
                online_input_json=json.dumps(
                    prompt_input, ensure_ascii=False, separators=(",", ":")
                ),
            )
            raw_response = chat_fn(
                prompt,
                tag="creamem3_v4_online_frontier_fuse",
                temperature=0.0,
                max_tokens=16384,
                json_mode=True,
            )
            parsed = legacy._parse_json_object(raw_response)
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"operations"}
                or not isinstance(parsed.get("operations"), list)
            ):
                raise GraphMemoryError(
                    "streaming Online response must contain only an operations list"
                )
            working_views, accepted, rejected, next_view_number = batch_v4._apply_model_response(
                parsed,
                frontier_id=session_id,
                prompt_input=prompt_input,
                active_views=working_views,
                fact_by_id=base["fact_by_id"],
                fact_order=base["fact_order"],
                next_view_number=next_view_number,
                embed_fn=embed_fn,
                embedding_dim=self.embedding_dim,
            )
        if accepted:
            working_hash.replace_views(working_views)
        post_state = working_hash.digest() if accepted else post_append
        step: dict[str, Any] = {
            "step_id": f"T{session_id:03d}",
            "frontier_id": session_id,
            "session_id": session_id,
            "ordinal": ordinal + 1,
            "visible_frontier_ids": [
                *[int(row["session_id"]) for row in self._session_packets],
                session_id,
            ],
            "model_call_made": model_call_made,
            "model_call_count": int(model_call_made),
            "prompt_input": prompt_input,
            "allowed_ids": copy.deepcopy(prompt_input["allowed_ids"]),
            "prompt_input_sha256": prompt_input_sha,
            "model_response_sha256": legacy._response_hash(raw_response),
            "model_error": None,
            "accepted": accepted,
            "rejected": rejected,
            "response_normalizations": [],
            "pre_state_sha256": pre_state,
            "post_append_state_sha256": post_append,
            "post_state_sha256": post_state,
            "previous_chain_sha256": self._stream_chain,
        }
        step["chain_sha256"] = batch_v4._chain_hash(self._stream_chain, step)
        self._commit_delta(
            current,
            working_views=working_views,
            working_hash_state=working_hash,
            step=step,
            next_view_number=next_view_number,
        )
        return copy.deepcopy(step)

    def route_active_views(
        self, query_vector: Sequence[Any], *, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Deterministically rank only views already present in this prefix."""

        if not isinstance(top_k, int) or top_k < 0:
            raise GraphMemoryError("top_k must be a non-negative integer")
        query = _checked_vector(
            list(query_vector), dimension=self.embedding_dim, label="writer route query"
        )
        view_nodes = [self._active_views[key] for key in sorted(self._active_views)]
        scores = legacy._cosine_scores(
            view_nodes,
            [{"id": "CURRENT_SESSION_QUERY", "embedding": query}],
            expected_dim=self.embedding_dim,
        )
        order = sorted(
            range(len(view_nodes)),
            key=lambda index: (-float(scores[index]), str(view_nodes[index]["id"])),
        )[:top_k]
        return [
            {
                "rank": rank,
                "view_id": str(view_nodes[index]["id"]),
                "score": float(scores[index]),
            }
            for rank, index in enumerate(order, 1)
        ]

    def hydrate_views(self, view_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Return complete, prefix-only member facts for selected active views."""

        fact_by_id = {str(fact["id"]): fact for fact in self._facts}
        fact_order = {str(fact["id"]): index for index, fact in enumerate(self._facts)}
        summary_by_session = {
            int(summary["session_ids"][0]): summary for summary in self._summaries
        }
        packages: list[dict[str, Any]] = []
        seen_views: set[str] = set()
        for raw_view_id in view_ids:
            view_id = str(raw_view_id)
            if view_id in seen_views or view_id not in self._active_views:
                raise GraphMemoryError("hydrate_views received an unknown/duplicate view ID")
            seen_views.add(view_id)
            view = self._active_views[view_id]
            member_ids = sorted(
                [str(value) for value in view.get("member_fact_ids", [])],
                key=fact_order.__getitem__,
            )
            if any(fact_id not in fact_by_id for fact_id in member_ids):
                raise GraphMemoryError("active view refers outside the written prefix")
            source_sessions = sorted(
                {int(fact_by_id[fact_id]["session_id"]) for fact_id in member_ids}
            )
            packages.append(
                {
                    "view": _without_embedding(view),
                    "member_fact_ids": member_ids,
                    "member_facts_chronological": [
                        _without_embedding(fact_by_id[fact_id]) for fact_id in member_ids
                    ],
                    "source_session_summaries": [
                        _without_embedding(summary_by_session[session_id])
                        for session_id in source_sessions
                    ],
                    "provenance": {
                        source_id: copy.deepcopy(self._provenance[source_id])
                        for fact_id in member_ids
                        for source_id in fact_by_id[fact_id].get("source_dia_ids", [])
                    },
                }
            )
        return packages

    def to_session_graph(self) -> dict[str, Any]:
        graph = copy.deepcopy(self.graph_header)
        graph["graph_variant"] = "session"
        graph["session_packets"] = copy.deepcopy(self._session_packets)
        graph["provenance_map"] = copy.deepcopy(self._provenance)
        graph["summary_nodes"] = copy.deepcopy(self._summaries)
        graph["fact_nodes"] = copy.deepcopy(self._facts)
        graph["edges"] = copy.deepcopy(self._edges)
        embedding_manifest = {
            str(node["id"]): canonical_sha256(node["embedding"])
            for node in [*self._summaries, *self._facts]
        }
        graph["embedding_manifest_sha256"] = canonical_sha256(embedding_manifest)
        previous_stats = self.graph_header.get("stats", {})
        graph["stats"] = {
            **(copy.deepcopy(dict(previous_stats)) if isinstance(previous_stats, Mapping) else {}),
            "physical_sessions": sum(
                len(packet.get("physical_sessions", [])) for packet in self._session_packets
            ),
            "frontiers": len(self._summaries),
            "sessions": len(self._summaries),
            "session_summary_nodes": len(self._summaries),
            "event_view_nodes": 0,
            "facts": len(self._facts),
            "edges": len(self._edges),
            "writer_calls": sum(
                int(packet.get("writer_attempts", 1)) for packet in self._session_packets
            ),
            "writer_retries": sum(
                max(0, int(packet.get("writer_attempts", 1)) - 1)
                for packet in self._session_packets
            ),
            "embedding_dim": self.embedding_dim,
        }
        graph["content_fingerprint"] = canonical_sha256(
            {
                "summaries": [_without_embedding(node) for node in self._summaries],
                "facts": [_without_embedding(node) for node in self._facts],
                "edges": self._edges,
                "provenance_map": self._provenance,
                "prompt_contract_sha256": graph.get("prompt_contract_sha256"),
                "embedding_contract": graph.get("embedding_contract"),
                "embedding_manifest_sha256": graph["embedding_manifest_sha256"],
            }
        )
        return graph

    def _validate_session_graph(self, session_graph: Mapping[str, Any]) -> dict[str, Any]:
        graph = copy.deepcopy(dict(session_graph))
        if (
            graph.get("session_packets") != self._session_packets
            or graph.get("summary_nodes") != self._summaries
            or graph.get("fact_nodes") != self._facts
            or graph.get("edges") != self._edges
            or graph.get("provenance_map") != self._provenance
        ):
            raise GraphMemoryError("final session graph differs from the streamed prefix")
        legacy._base_parts(graph)
        return graph

    def to_graph(
        self, session_graph: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Materialize a legacy-v4-compatible final Online graph."""

        base_graph = (
            self._validate_session_graph(session_graph)
            if session_graph is not None
            else self._validate_session_graph(self.to_session_graph())
        )
        base = legacy._base_parts(base_graph)
        genesis = canonical_sha256(
            {
                "schema_version": batch_v4.ONLINE_SCHEMA_VERSION,
                "base_content_fingerprint": base_graph.get("content_fingerprint"),
                "frontier_order": base["session_ids"],
                "prompt_sha256": self.prompt_sha256,
            }
        )
        previous = genesis
        legacy_steps: list[dict[str, Any]] = []
        for stored in self._steps:
            step = copy.deepcopy(stored)
            step["previous_chain_sha256"] = previous
            step["chain_sha256"] = batch_v4._chain_hash(previous, step)
            previous = step["chain_sha256"]
            legacy_steps.append(step)
        final_views = [self._active_views[key] for key in sorted(self._active_views)]
        graph = copy.deepcopy(base_graph)
        graph["graph_variant"] = "online"
        graph["architecture"] = {
            **dict(graph.get("architecture", {})),
            "online_update_policy": "strict_causal_frontier_prefix_only_create_or_extend",
            "event_view_membership": "monotonic_non_destructive",
            "online_engine": "creamem3.online_memory_v4",
        }
        graph["prompt_sha256"] = {
            **dict(graph.get("prompt_sha256", {})),
            "online_frontier_update": self.prompt_sha256,
        }
        graph["summary_nodes"] = copy.deepcopy(base["summaries"]) + copy.deepcopy(final_views)
        graph["fact_nodes"] = copy.deepcopy(base["facts"])
        graph["edges"] = copy.deepcopy(base["edges"]) + legacy._view_edges(self._active_views)
        graph["stats"] = {
            **dict(graph.get("stats", {})),
            "event_view_nodes": len(final_views),
            "summary_nodes": len(graph["summary_nodes"]),
            "edges": len(graph["edges"]),
            "cross_session_view_edges": sum(
                len(view["member_fact_ids"]) for view in final_views
            ),
            "online_fusion_calls": sum(step["model_call_count"] for step in legacy_steps),
            "online_accepted_operations": sum(len(step["accepted"]) for step in legacy_steps),
            "online_rejected_operations": sum(len(step["rejected"]) for step in legacy_steps),
        }
        graph["online_fusion"] = {
            "schema_version": batch_v4.ONLINE_SCHEMA_VERSION,
            "policy": "strict_causal_frontier_prefix_only_create_or_extend",
            "prompt_sha256": self.prompt_sha256,
            "frontier_order": list(base["session_ids"]),
            "session_order": list(base["session_ids"]),
            "candidate_policy": {
                "summary_dense_top_k": batch_v4.SUMMARY_TOP_K,
                "fact_dense_top_k": batch_v4.DENSE_FACT_TOP_K,
                "bm25_top_k": batch_v4.BM25_FACT_TOP_K,
                "active_view_top_k": batch_v4.ACTIVE_VIEW_TOP_K,
                "prior_fact_cap": self.prior_fact_cap,
                "current_fact_cap": None,
            },
            "performance_policy": {
                "immutable_node_hash_cache": True,
                "incremental_bm25": True,
                "pre_normalized_dense_prefix": True,
                "structural_embedding_sharing": True,
            },
            "base_content_fingerprint": base_graph.get("content_fingerprint"),
            "base_fact_identity_sha256": canonical_sha256(base["facts"]),
            "genesis_chain_sha256": genesis,
            "final_chain_sha256": previous,
            "steps": legacy_steps,
        }
        graph["content_fingerprint"] = canonical_sha256(
            {
                "base": base_graph.get("content_fingerprint"),
                "online_schema": batch_v4.ONLINE_SCHEMA_VERSION,
                "active_views": [legacy._without_embedding(view) for view in final_views],
                "final_chain_sha256": previous,
            }
        )
        return graph

    def to_checkpoint(self) -> dict[str, Any]:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config": {
                "graph_header": copy.deepcopy(self.graph_header),
                "stream_id": self.stream_id,
                "embedding_dim": self.embedding_dim,
                "prior_fact_cap": self.prior_fact_cap,
                "prompt_template": self.prompt_template,
                "prompt_sha256": self.prompt_sha256,
            },
            "deltas": copy.deepcopy(self._deltas),
            "steps": copy.deepcopy(self._steps),
            "final_active_views": copy.deepcopy(self._active_views),
            "next_view_number": self._next_view_number,
            "stream_genesis_chain_sha256": self._stream_genesis,
            "stream_final_chain_sha256": self._stream_chain,
        }
        return _seal(payload)

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "StreamingOnlineState":
        _verify_seal(checkpoint)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise GraphMemoryError("streaming checkpoint schema mismatch")
        config = checkpoint.get("config")
        if not isinstance(config, Mapping):
            raise GraphMemoryError("streaming checkpoint config is invalid")
        state = cls(
            graph_header=config["graph_header"],
            stream_id=str(config["stream_id"]),
            embedding_dim=int(config["embedding_dim"]),
            prior_fact_cap=int(config["prior_fact_cap"]),
            prompt_template=str(config["prompt_template"]),
            prompt_sha256=str(config["prompt_sha256"]),
        )
        deltas = checkpoint.get("deltas")
        steps = checkpoint.get("steps")
        if not isinstance(deltas, list) or not isinstance(steps, list) or len(deltas) != len(steps):
            raise GraphMemoryError("streaming checkpoint delta/step list is invalid")
        for delta, step in zip(deltas, steps, strict=True):
            state._restore_step(delta, step)
        if (
            state._active_views != checkpoint.get("final_active_views")
            or state._next_view_number != checkpoint.get("next_view_number")
            or state._stream_genesis != checkpoint.get("stream_genesis_chain_sha256")
            or state._stream_chain != checkpoint.get("stream_final_chain_sha256")
        ):
            raise GraphMemoryError("streaming checkpoint derived state mismatch")
        report = state.audit()
        if report["status"] != "pass":
            raise GraphMemoryError("restored streaming checkpoint failed deterministic audit")
        return state

    def _restore_step(self, delta: Mapping[str, Any], stored_step: Mapping[str, Any]) -> None:
        current = self._validate_delta(delta)
        if not isinstance(stored_step, Mapping):
            raise GraphMemoryError("streaming checkpoint step is not an object")
        step = copy.deepcopy(dict(stored_step))
        base = self._base_with(current)
        expected_input = self._prompt_input(current, base)
        summary = current["summary_node"]
        facts = current["fact_nodes"]
        edges = current["support_edges"]
        session_id = int(summary["session_ids"][0])
        ordinal = self.processed_frontiers
        expected_call = ordinal > 0
        if (
            step.get("frontier_id") != session_id
            or step.get("session_id") != session_id
            or step.get("ordinal") != ordinal + 1
            or step.get("visible_frontier_ids")
            != [*[int(row["session_id"]) for row in self._session_packets], session_id]
            or bool(step.get("model_call_made")) != expected_call
            or step.get("model_call_count") != int(expected_call)
            or step.get("prompt_input") != expected_input
            or step.get("allowed_ids") != expected_input["allowed_ids"]
            or step.get("prompt_input_sha256") != canonical_sha256(expected_input)
            or step.get("model_error") is not None
        ):
            raise GraphMemoryError("streaming checkpoint prompt/call replay mismatch")
        if step.get("pre_state_sha256") != self._hash_state.digest():
            raise GraphMemoryError("streaming checkpoint pre-state mismatch")
        working_hash = copy.deepcopy(self._hash_state)
        working_hash.append(summary, facts, edges)
        post_append = working_hash.digest()
        if step.get("post_append_state_sha256") != post_append:
            raise GraphMemoryError("streaming checkpoint post-append mismatch")
        working_views = {
            key: copy.deepcopy(value) for key, value in self._active_views.items()
        }
        touched: set[str] = set()
        next_view_number = self._next_view_number
        accepted = step.get("accepted")
        if not isinstance(accepted, list):
            raise GraphMemoryError("streaming checkpoint accepted list is invalid")
        for accepted_row in accepted:
            if not isinstance(accepted_row, Mapping):
                raise GraphMemoryError("streaming checkpoint accepted operation is invalid")
            operation = {
                key: copy.deepcopy(accepted_row[key])
                for key in (
                    "operation",
                    "target_view_id",
                    "view_type",
                    "summary_claims",
                    "member_fact_ids",
                )
            }
            normalized, expected_view = legacy._validate_operation(
                operation,
                session_id=session_id,
                prompt_input=expected_input,
                working_views=working_views,
                touched_view_ids=touched,
                fact_by_id=base["fact_by_id"],
                fact_order=base["fact_order"],
                next_view_id=f"OV{next_view_number:04d}",
            )
            resulting = accepted_row.get("resulting_view")
            if (
                not isinstance(resulting, Mapping)
                or any(accepted_row.get(key) != value for key, value in normalized.items())
                or legacy._without_embedding(resulting)
                != legacy._without_embedding(expected_view)
            ):
                raise GraphMemoryError("streaming checkpoint accepted-view replay mismatch")
            vector = _checked_vector(
                resulting.get("embedding"),
                dimension=self.embedding_dim,
                label=str(resulting.get("id", "view")),
            )
            norm = math.sqrt(math.fsum(value * value for value in vector))
            if abs(norm - 1.0) > 1e-5:
                raise GraphMemoryError("streaming checkpoint view embedding is not unit-normalized")
            working_views[str(resulting["id"])] = copy.deepcopy(dict(resulting))
            if normalized["operation"] == "create":
                next_view_number += 1
            else:
                touched.add(str(normalized["target_view_id"]))
        if accepted:
            working_hash.replace_views(working_views)
        post_state = working_hash.digest() if accepted else post_append
        if step.get("post_state_sha256") != post_state:
            raise GraphMemoryError("streaming checkpoint post-state mismatch")
        if step.get("previous_chain_sha256") != self._stream_chain:
            raise GraphMemoryError("streaming checkpoint chain predecessor mismatch")
        expected_chain = batch_v4._chain_hash(self._stream_chain, step)
        if step.get("chain_sha256") != expected_chain:
            raise GraphMemoryError("streaming checkpoint chain mismatch")
        self._commit_delta(
            current,
            working_views=working_views,
            working_hash_state=working_hash,
            step=step,
            next_view_number=next_view_number,
        )

    def restore_committed_step(
        self, delta: Mapping[str, Any], stored_step: Mapping[str, Any]
    ) -> None:
        """Restore one externally sealed durable commit without model calls.

        Callers must feed commits in frontier order.  The same deterministic
        validation used by full-checkpoint restore verifies the delta, hashes,
        accepted operations, embeddings, and chain before mutating state.
        """
        self._restore_step(copy.deepcopy(dict(delta)), copy.deepcopy(dict(stored_step)))

    def audit(self, session_graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def check(name: str, passed: bool, detail: str = "") -> None:
            checks.append({"name": name, "passed": bool(passed), "detail": detail})

        try:
            checkpoint = self.to_checkpoint()
            _verify_seal(checkpoint)
            check("checkpoint_seal", True)
            check("one_step_per_frontier", len(self._steps) == len(self._deltas))
            check(
                "exact_call_sequence",
                all(
                    bool(step.get("model_call_made")) == (index > 0)
                    and step.get("model_call_count") == int(index > 0)
                    for index, step in enumerate(self._steps)
                ),
            )
            check(
                "stream_chain",
                (self._steps[-1]["chain_sha256"] if self._steps else self._stream_genesis)
                == self._stream_chain,
            )
            graph = self.to_graph(session_graph)
            base_graph = (
                self._validate_session_graph(session_graph)
                if session_graph is not None
                else self.to_session_graph()
            )
            legacy_report = batch_v4.audit_online_graph(
                base_graph,
                graph,
                expected_prompt_sha256=self.prompt_sha256,
                expected_prior_fact_cap=self.prior_fact_cap,
            )
            check("legacy_v4_deterministic_replay", legacy_report.get("status") == "pass")
        except Exception as exc:
            legacy_report = None
            check("audit_exception", False, f"{type(exc).__name__}: {str(exc)[-500:]}")
        failures = [row for row in checks if not row["passed"]]
        return {
            "schema_version": f"{STREAMING_SCHEMA_VERSION}.audit",
            "status": "pass" if not failures else "fail",
            "processed_frontiers": self.processed_frontiers,
            "active_views": self.active_view_count,
            "checks": checks,
            "failed": len(failures),
            "legacy_v4_audit": legacy_report,
        }


def audit_streaming_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Restore and audit a serialized checkpoint without any model calls."""

    try:
        state = StreamingOnlineState.from_checkpoint(checkpoint)
        return state.audit()
    except Exception as exc:
        return {
            "schema_version": f"{STREAMING_SCHEMA_VERSION}.checkpoint_audit",
            "status": "fail",
            "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
        }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "STREAMING_SCHEMA_VERSION",
    "StreamingOnlineState",
    "audit_streaming_checkpoint",
]

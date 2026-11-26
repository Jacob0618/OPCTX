"""Independent typed-graph memory used by the CreaMem3 v2 pilot.

Atomic facts are immutable evidence nodes.  Session summaries and optional
cross-session event views are routing nodes connected to facts by typed edges.
This many-to-many representation avoids making a generated summary the owner
of a fact and allows later views to overlap without destructive tree merges.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from creamem3.prompts_v2 import (
    FACT_RERANK_PROMPT,
    OFFLINE_EVENT_VIEW_PROMPT,
    PROMPT_SCHEMA_VERSION,
    PROMPT_SHA256,
    SESSION_PACKET_PROMPT,
)


GRAPH_SCHEMA_VERSION = "creamem3.typed_event_graph.v2"
SUMMARY_TOP_K = 10
DENSE_FACT_TOP_K = 20
BM25_FACT_TOP_K = 20
CANDIDATE_CAP = 120
RERANK_TOP_K = 20

_DIA_ID_RE = re.compile(r"D(\d+):(\d+)\Z")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_SPACE_RE = re.compile(r"\s+")

ChatFn = Callable[..., str]
EmbedFn = Callable[..., list[list[float]]]


class GraphMemoryError(RuntimeError):
    """Raised when a model output or graph artifact violates the protocol."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_json_object(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def iso_date_from_timestamp(timestamp: str) -> str:
    """Parse LoCoMo's human timestamp without changing its original wording."""

    text = str(timestamp or "").strip()
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %B %d, %Y",
        "%d %B, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", text)
    if match:
        try:
            return datetime.strptime(match.group(0), "%d %B, %Y").date().isoformat()
        except ValueError:
            pass
    return ""


def render_text_only_session(record: Mapping[str, Any]) -> str:
    """Render only speaker text and identifiers; image metadata is never read."""

    session_id = int(record["session_id"])
    timestamp = str(record.get("timestamp", "") or "")
    iso_date = iso_date_from_timestamp(timestamp)
    header = f"[Session {session_id}"
    if timestamp:
        header += f" at {timestamp}"
    if iso_date:
        header += f"; ISO date {iso_date}"
    lines = [header + "]"]
    for turn in record.get("turns", []):
        dia_id = str(turn.get("dia_id", "") or "")
        match = _DIA_ID_RE.fullmatch(dia_id)
        if match is None or int(match.group(1)) != session_id:
            raise GraphMemoryError(
                f"session {session_id} has invalid source dia_id {dia_id!r}"
            )
        speaker = _normalize_text(turn.get("speaker", ""))
        # Intentionally access only the dialogue text field.  In particular,
        # blip_caption and img_url are outside this experiment's input scope.
        text = _normalize_text(turn.get("text", ""))
        lines.append(f"{speaker} ({dia_id}): {text}")
    return "\n".join(lines)


def _fact_guidance(turn_count: int) -> str:
    low = max(1, math.ceil(turn_count / 2))
    high = max(3, math.ceil(turn_count * 1.5))
    return f"approximately {low}-{high} atomic facts"


def _validate_session_packet(
    value: Mapping[str, Any], legal_ids: Sequence[str]
) -> tuple[str, list[dict[str, Any]]]:
    allowed_top = {"batch_summary", "atomic_facts"}
    extra = set(value) - allowed_top
    if extra:
        raise GraphMemoryError(f"writer returned unexpected top-level keys: {sorted(extra)}")
    summary = value.get("batch_summary")
    if not isinstance(summary, str) or not _normalize_text(summary):
        raise GraphMemoryError("writer returned an empty or non-string batch_summary")
    raw_facts = value.get("atomic_facts")
    if not isinstance(raw_facts, list) or not raw_facts:
        raise GraphMemoryError("writer returned no atomic_facts list")

    legal = set(legal_ids)
    legal_order = {dia_id: index for index, dia_id in enumerate(legal_ids)}
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, item in enumerate(raw_facts):
        if not isinstance(item, dict) or set(item) != {"fact", "source_dia_ids"}:
            raise GraphMemoryError(
                f"atomic_facts[{index}] must contain exactly fact and source_dia_ids"
            )
        fact = item.get("fact")
        source_ids = item.get("source_dia_ids")
        if not isinstance(fact, str) or not _normalize_text(fact):
            raise GraphMemoryError(f"atomic_facts[{index}] has invalid fact text")
        if not isinstance(source_ids, list) or not source_ids:
            raise GraphMemoryError(f"atomic_facts[{index}] lacks source_dia_ids")
        normalized_ids: list[str] = []
        for source_id in source_ids:
            if not isinstance(source_id, str) or source_id not in legal:
                raise GraphMemoryError(
                    f"atomic_facts[{index}] cites unknown source {source_id!r}"
                )
            if source_id not in normalized_ids:
                normalized_ids.append(source_id)
        normalized_ids.sort(key=legal_order.__getitem__)
        fact_text = _normalize_text(fact)
        key = fact_text.casefold()
        if key not in merged:
            merged[key] = {"fact": fact_text, "source_dia_ids": normalized_ids}
            order.append(key)
        else:
            existing = merged[key]["source_dia_ids"]
            merged[key]["source_dia_ids"] = sorted(
                set(existing) | set(normalized_ids), key=legal_order.__getitem__
            )
    return _normalize_text(summary), [merged[key] for key in order]


def extract_session_packet(
    record: Mapping[str, Any],
    chat_fn: ChatFn,
    *,
    max_attempts: int = 3,
) -> dict[str, Any]:
    session_id = int(record["session_id"])
    timestamp = str(record.get("timestamp", "") or "")
    turns = list(record.get("turns", []))
    legal_ids = [str(turn.get("dia_id", "") or "") for turn in turns]
    batch = render_text_only_session(record)
    prompt = SESSION_PACKET_PROMPT.format(
        turn_count=len(turns),
        fact_guidance=_fact_guidance(len(turns)),
        session_id=session_id,
        timestamp=timestamp,
        iso_date=iso_date_from_timestamp(timestamp) or "unknown",
        batch=batch,
    )
    last_error = "unknown writer error"
    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                "\n\nThe previous response failed validation: "
                + last_error
                + "\nReturn a fresh object that follows the exact JSON schema."
            )
        try:
            raw = chat_fn(
                attempt_prompt,
                tag="creamem3_v2_write",
                temperature=0.0,
                max_tokens=16384,
                json_mode=True,
            )
            value = _parse_json_object(raw)
            if value is None:
                raise GraphMemoryError("writer returned no valid JSON object")
            summary, facts = _validate_session_packet(value, legal_ids)
            return {
                "session_id": session_id,
                "timestamp": timestamp,
                "iso_date": iso_date_from_timestamp(timestamp),
                "turn_count": len(turns),
                "source_dia_ids": legal_ids,
                "batch_summary": summary,
                "atomic_facts": facts,
                "writer_attempts": attempt + 1,
                "writer_prompt_sha256": PROMPT_SHA256["session_packet"],
                "text_only_input_sha256": hashlib.sha256(
                    batch.encode("utf-8")
                ).hexdigest(),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-500:]}"
    raise GraphMemoryError(
        f"session {session_id} writer failed after {max_attempts} attempts: {last_error}"
    )


def _embed_checked(
    texts: Sequence[str],
    embed_fn: EmbedFn,
    *,
    tag: str,
) -> list[list[float]]:
    vectors = embed_fn(texts, task_type="RETRIEVAL_DOCUMENT", tag=tag)
    if len(vectors) != len(texts):
        raise GraphMemoryError(
            f"embedding count mismatch: expected {len(texts)}, got {len(vectors)}"
        )
    output: list[list[float]] = []
    dimension: int | None = None
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        if not values or not all(math.isfinite(value) for value in values):
            raise GraphMemoryError(f"embedding {index} is empty or non-finite")
        dimension = dimension or len(values)
        if len(values) != dimension:
            raise GraphMemoryError("embedding dimensions are inconsistent")
        output.append(values)
    return output


def build_session_graph(
    records: Sequence[Mapping[str, Any]],
    chat_fn: ChatFn,
    embed_fn: EmbedFn,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Preprocess every complete session once and build an immutable fact graph."""

    ordered_records = sorted(records, key=lambda item: int(item["session_id"]))
    packets: list[dict[str, Any] | None] = [None] * len(ordered_records)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_index = {
            executor.submit(extract_session_packet, record, chat_fn): index
            for index, record in enumerate(ordered_records)
        }
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            packets[index] = future.result()
            completed += 1
            print(
                f"  [creamem3-v2 write] {completed}/{len(ordered_records)} sessions",
                flush=True,
            )
    normalized_packets = [packet for packet in packets if packet is not None]
    if len(normalized_packets) != len(ordered_records):
        raise GraphMemoryError("one or more session packets are missing")

    summary_nodes: list[dict[str, Any]] = []
    fact_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    fact_counter = 0
    for packet in normalized_packets:
        session_id = int(packet["session_id"])
        summary_id = f"S{session_id:03d}"
        packet_fact_ids: list[str] = []
        summary_source_ids: list[str] = []
        for fact in packet["atomic_facts"]:
            fact_counter += 1
            fact_id = f"F{fact_counter:04d}"
            packet_fact_ids.append(fact_id)
            for source_id in fact["source_dia_ids"]:
                if source_id not in summary_source_ids:
                    summary_source_ids.append(source_id)
            fact_nodes.append(
                {
                    "id": fact_id,
                    "node_type": "atomic_fact",
                    "text": fact["fact"],
                    "source_dia_ids": list(fact["source_dia_ids"]),
                    "session_id": session_id,
                    "timestamp": packet["timestamp"],
                    "iso_date": packet["iso_date"],
                    "embedding": [],
                }
            )
            edges.append(
                {
                    "edge_type": "SUPPORTS",
                    "source_id": summary_id,
                    "target_id": fact_id,
                }
            )
        summary_nodes.append(
            {
                "id": summary_id,
                "node_type": "session_packet",
                "text": packet["batch_summary"],
                "session_ids": [session_id],
                "timestamp": packet["timestamp"],
                "iso_date": packet["iso_date"],
                "source_dia_ids": summary_source_ids,
                "embedding": [],
            }
        )

    texts = [node["text"] for node in summary_nodes] + [
        node["text"] for node in fact_nodes
    ]
    vectors = _embed_checked(texts, embed_fn, tag="creamem3_v2_embed_write")
    split = len(summary_nodes)
    for node, vector in zip(summary_nodes, vectors[:split]):
        node["embedding"] = vector
    for node, vector in zip(fact_nodes, vectors[split:]):
        node["embedding"] = vector

    graph = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "graph_variant": "session",
        "architecture": {
            "representation": "typed_many_to_many_event_graph",
            "fact_policy": "immutable_atomic_facts_are_source_of_truth",
            "summary_policy": "derived_routing_views_do_not_own_facts",
            "write_unit": "one_complete_physical_session",
            "input_scope": "dialogue_text_only",
        },
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "prompt_sha256": dict(PROMPT_SHA256),
        "session_packets": normalized_packets,
        "summary_nodes": summary_nodes,
        "fact_nodes": fact_nodes,
        "edges": edges,
        "stats": {
            "sessions": len(normalized_packets),
            "session_summary_nodes": len(summary_nodes),
            "event_view_nodes": 0,
            "facts": len(fact_nodes),
            "edges": len(edges),
            "writer_calls": sum(int(packet["writer_attempts"]) for packet in normalized_packets),
            "writer_retries": sum(max(0, int(packet["writer_attempts"]) - 1) for packet in normalized_packets),
            "min_facts_per_session": min(
                (len(packet["atomic_facts"]) for packet in normalized_packets),
                default=0,
            ),
            "max_facts_per_session": max(
                (len(packet["atomic_facts"]) for packet in normalized_packets),
                default=0,
            ),
        },
    }
    graph["content_fingerprint"] = canonical_sha256(
        {
            "summaries": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in summary_nodes
            ],
            "facts": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in fact_nodes
            ],
            "edges": edges,
        }
    )
    return graph


def _fusion_input(graph: Mapping[str, Any]) -> str:
    facts_by_session: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fact in graph["fact_nodes"]:
        facts_by_session[int(fact["session_id"])].append(
            {
                "fact_id": fact["id"],
                "text": fact["text"],
                "source_dia_ids": fact["source_dia_ids"],
            }
        )
    packets = []
    for summary in graph["summary_nodes"]:
        if summary["node_type"] != "session_packet":
            continue
        session_id = int(summary["session_ids"][0])
        packets.append(
            {
                "summary_id": summary["id"],
                "session_id": session_id,
                "session_date": summary.get("iso_date", ""),
                "summary": summary["text"],
                "facts": facts_by_session[session_id],
            }
        )
    return json.dumps(packets, ensure_ascii=False, separators=(",", ":"))


def _validate_event_views(
    value: Mapping[str, Any], graph: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if set(value) != {"event_views"}:
        raise GraphMemoryError("fusion output must contain only event_views")
    raw_views = value.get("event_views")
    if not isinstance(raw_views, list):
        raise GraphMemoryError("fusion event_views is not a list")
    facts = {fact["id"]: fact for fact in graph["fact_nodes"]}
    allowed_types = {"same_event", "continuing_process", "update_chain"}
    seen_memberships: set[tuple[str, ...]] = set()
    views: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(raw_views):
        try:
            expected = {"view_id", "view_type", "summary", "member_fact_ids"}
            if not isinstance(item, dict) or set(item) != expected:
                raise GraphMemoryError("invalid shape")
            view_type = item.get("view_type")
            summary = item.get("summary")
            member_ids = item.get("member_fact_ids")
            if view_type not in allowed_types:
                raise GraphMemoryError("invalid view_type")
            if not isinstance(summary, str) or not _normalize_text(summary):
                raise GraphMemoryError("empty summary")
            if not isinstance(member_ids, list):
                raise GraphMemoryError("member_fact_ids is not a list")
            unique_ids: list[str] = []
            for fact_id in member_ids:
                if not isinstance(fact_id, str) or fact_id not in facts:
                    raise GraphMemoryError(f"unknown fact {fact_id!r}")
                if fact_id not in unique_ids:
                    unique_ids.append(fact_id)
            sessions = sorted(
                {int(facts[fact_id]["session_id"]) for fact_id in unique_ids}
            )
            if len(unique_ids) < 2 or len(sessions) < 2:
                raise GraphMemoryError("view does not span at least two sessions")
            membership = tuple(sorted(unique_ids))
            if membership in seen_memberships:
                raise GraphMemoryError("duplicate membership set")
            seen_memberships.add(membership)
            source_ids: list[str] = []
            for fact_id in unique_ids:
                for source_id in facts[fact_id]["source_dia_ids"]:
                    if source_id not in source_ids:
                        source_ids.append(source_id)
            views.append(
                {
                    "id": f"V{len(views) + 1:03d}",
                    "node_type": "event_view",
                    "view_type": view_type,
                    "text": _normalize_text(summary),
                    "session_ids": sessions,
                    "source_dia_ids": source_ids,
                    "member_fact_ids": unique_ids,
                    "embedding": [],
                }
            )
        except GraphMemoryError as exc:
            source_view_id = item.get("view_id") if isinstance(item, dict) else None
            rejected.append(
                {
                    "source_index": index,
                    "source_view_id": source_view_id,
                    "reason": str(exc),
                }
            )
    return views, rejected


def add_offline_event_views(
    session_graph: Mapping[str, Any],
    chat_fn: ChatFn,
    embed_fn: EmbedFn,
    *,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Add global cross-session routing views without mutating raw facts."""

    graph = copy.deepcopy(dict(session_graph))
    prompt = OFFLINE_EVENT_VIEW_PROMPT.format(
        memory_packets_json=_fusion_input(graph)
    )
    last_error = "unknown fusion error"
    views: list[dict[str, Any]] | None = None
    rejected_views: list[dict[str, Any]] = []
    attempts = 0
    for attempt in range(max_attempts):
        attempts = attempt + 1
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                "\n\nThe previous response failed validation: "
                + last_error
                + "\nReturn a fresh object that follows the exact JSON schema."
            )
        try:
            raw = chat_fn(
                attempt_prompt,
                tag="creamem3_v2_fuse",
                temperature=0.0,
                max_tokens=16384,
                json_mode=True,
            )
            value = _parse_json_object(raw)
            if value is None:
                raise GraphMemoryError("fusion returned no valid JSON object")
            views, rejected_views = _validate_event_views(value, graph)
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-500:]}"
    if views is None:
        raise GraphMemoryError(
            f"offline fusion failed after {max_attempts} attempts: {last_error}"
        )

    if views:
        vectors = _embed_checked(
            [view["text"] for view in views],
            embed_fn,
            tag="creamem3_v2_embed_views",
        )
        for view, vector in zip(views, vectors):
            view["embedding"] = vector
            graph["summary_nodes"].append(view)
            for fact_id in view["member_fact_ids"]:
                graph["edges"].append(
                    {
                        "edge_type": "VIEW_SUPPORTS",
                        "source_id": view["id"],
                        "target_id": fact_id,
                    }
                )
    graph["graph_variant"] = "offline"
    graph["stats"] = dict(graph["stats"])
    graph["stats"].update(
        {
            "event_view_nodes": len(views),
            "summary_nodes": len(graph["summary_nodes"]),
            "edges": len(graph["edges"]),
            "cross_session_view_edges": sum(len(view["member_fact_ids"]) for view in views),
            "fusion_calls": attempts,
            "fusion_retries": max(0, attempts - 1),
            "fusion_rejected_views": len(rejected_views),
        }
    )
    graph["offline_fusion"] = {
        "policy": "non_destructive_global_cross_session_views",
        "prompt_sha256": PROMPT_SHA256["offline_event_view"],
        "attempts": attempts,
        "view_ids": [view["id"] for view in views],
        "rejected_views": rejected_views,
    }
    graph["content_fingerprint"] = canonical_sha256(
        {
            "base": session_graph.get("content_fingerprint"),
            "views": [
                {key: value for key, value in view.items() if key != "embedding"}
                for view in views
            ],
        }
    )
    return graph


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


class BM25Index:
    """Small BM25 implementation matching MGTM's k1=1.5, b=0.75 path."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.doc_freq: Counter[str] = Counter()
        self.avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokens(text)
        previous_n = len(self.docs)
        self.docs.append(tokens)
        self.doc_ids.append(doc_id)
        for token in set(tokens):
            self.doc_freq[token] += 1
        self.avgdl = (
            (self.avgdl * previous_n + len(tokens)) / (previous_n + 1)
            if previous_n + 1
            else 0.0
        )

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        n_docs = len(self.docs)
        if not n_docs:
            return []
        scores = [0.0] * n_docs
        for term in _tokens(query):
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for index, tokens in enumerate(self.docs):
                tf = tokens.count(term)
                if not tf:
                    continue
                dl = len(tokens)
                denom = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avgdl, 1)
                )
                scores[index] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [
            (self.doc_ids[index], score)
            for index, score in ranked[:top_n]
            if score > 0
        ]


class GraphMemoryIndex:
    """Three-channel MGTM-style retrieval over the CreaMem3 typed graph."""

    def __init__(
        self,
        graph: Mapping[str, Any],
        *,
        summary_top_k: int = SUMMARY_TOP_K,
        dense_fact_top_k: int = DENSE_FACT_TOP_K,
        bm25_fact_top_k: int = BM25_FACT_TOP_K,
        candidate_cap: int = CANDIDATE_CAP,
        rerank_top_k: int = RERANK_TOP_K,
    ) -> None:
        self.graph = graph
        self.summary_top_k = summary_top_k
        self.dense_fact_top_k = dense_fact_top_k
        self.bm25_fact_top_k = bm25_fact_top_k
        self.candidate_cap = candidate_cap
        self.rerank_top_k = rerank_top_k
        self.summaries = list(graph.get("summary_nodes", []))
        self.facts = list(graph.get("fact_nodes", []))
        self.summary_by_id = {node["id"]: node for node in self.summaries}
        self.fact_by_id = {node["id"]: node for node in self.facts}
        self.fact_order = {node["id"]: index for index, node in enumerate(self.facts)}
        self.summary_order = {
            node["id"]: index for index, node in enumerate(self.summaries)
        }
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in graph.get("edges", []):
            if edge.get("edge_type") not in {"SUPPORTS", "VIEW_SUPPORTS"}:
                continue
            source_id = str(edge.get("source_id", ""))
            target_id = str(edge.get("target_id", ""))
            if source_id in self.summary_by_id and target_id in self.fact_by_id:
                self.adjacency[source_id].append(target_id)
        self.summary_matrix = self._matrix(self.summaries, "summary")
        self.fact_matrix = self._matrix(self.facts, "fact")
        self.embedding_dim = (
            int(self.fact_matrix.shape[1]) if self.fact_matrix.ndim == 2 else 0
        )
        self.bm25 = BM25Index()
        for fact in self.facts:
            self.bm25.add(fact["id"], fact["text"])

    @staticmethod
    def _matrix(nodes: Sequence[Mapping[str, Any]], label: str) -> np.ndarray:
        if not nodes:
            return np.empty((0, 0), dtype=np.float32)
        matrix = np.asarray([node.get("embedding", []) for node in nodes], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise GraphMemoryError(f"{label} embeddings are missing or ragged")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _scores(self, matrix: np.ndarray, query_vector: Sequence[float]) -> np.ndarray:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != vector.shape[0]:
            raise GraphMemoryError(
                f"query embedding dimension {vector.shape} does not match {matrix.shape}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        return matrix @ (vector / norm)

    @staticmethod
    def _rank_indices(scores: np.ndarray, n: int) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:n]

    def _candidate_pool(
        self, question: str, query_vector: Sequence[float]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        summary_scores = self._scores(self.summary_matrix, query_vector)
        fact_scores = self._scores(self.fact_matrix, query_vector)
        summary_indices = self._rank_indices(summary_scores, self.summary_top_k)
        dense_indices = self._rank_indices(fact_scores, self.dense_fact_top_k)
        lexical = self.bm25.search(question, self.bm25_fact_top_k)

        channels: dict[str, dict[str, Any]] = defaultdict(dict)
        top_summaries: list[dict[str, Any]] = []
        for rank, index in enumerate(summary_indices, 1):
            node = self.summaries[index]
            top_summaries.append(
                {
                    "summary_id": node["id"],
                    "node_type": node["node_type"],
                    "score": float(summary_scores[index]),
                    "rank": rank,
                }
            )
            for fact_id in self.adjacency.get(node["id"], []):
                previous = channels[fact_id].get("summary_ranks", [])
                channels[fact_id]["summary_ranks"] = [*previous, rank]
                channels[fact_id]["summary_ids"] = [
                    *channels[fact_id].get("summary_ids", []),
                    node["id"],
                ]
        for rank, index in enumerate(dense_indices, 1):
            fact_id = self.facts[index]["id"]
            channels[fact_id]["dense_rank"] = rank
        for rank, (fact_id, score) in enumerate(lexical, 1):
            channels[fact_id]["bm25_rank"] = rank
            channels[fact_id]["bm25_score"] = float(score)

        wanted_ids = set(channels)
        candidates: list[dict[str, Any]] = []
        for fact in self.facts:
            fact_id = fact["id"]
            if fact_id not in wanted_ids:
                continue
            fact_index = self.fact_order[fact_id]
            candidates.append(
                {
                    "fact": fact,
                    "dense_score": float(fact_scores[fact_index]),
                    "channels": channels[fact_id],
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["dense_score"],
                self.fact_order[item["fact"]["id"]],
            )
        )
        union_before_cap = len(candidates)
        candidates = candidates[: self.candidate_cap]
        trace = {
            "policy": {
                "summary_top_k": self.summary_top_k,
                "dense_fact_top_k": self.dense_fact_top_k,
                "bm25_fact_top_k": self.bm25_fact_top_k,
                "candidate_cap": self.candidate_cap,
                "rerank_top_k": self.rerank_top_k,
                "character_cap": None,
            },
            "top_summaries": top_summaries,
            "dense_fact_ids": [self.facts[index]["id"] for index in dense_indices],
            "bm25_facts": [
                {"fact_id": fact_id, "score": float(score)}
                for fact_id, score in lexical
            ],
            "union_before_cap": union_before_cap,
            "candidates_after_cap": len(candidates),
            "candidates": [
                {
                    "fact_id": item["fact"]["id"],
                    "dense_score": item["dense_score"],
                    "channels": item["channels"],
                }
                for item in candidates
            ],
        }
        return candidates, trace

    def _rerank(
        self,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        rerank_chat: ChatFn,
        *,
        max_attempts: int = 2,
    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        target = min(self.rerank_top_k, len(candidates))
        if target == 0:
            return [], {"attempts": 0, "fallback": False, "error": None}
        candidate_map = {item["fact"]["id"]: item for item in candidates}
        candidate_json = json.dumps(
            [
                {"fact_id": item["fact"]["id"], "text": item["fact"]["text"]}
                for item in candidates
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = FACT_RERANK_PROMPT.format(
            top_k=target,
            question=question,
            candidates_json=candidate_json,
        )
        last_error = "unknown reranker error"
        for attempt in range(max_attempts):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\n\nThe previous response failed validation: "
                    + last_error
                    + "\nReturn exactly the required number of unique candidate IDs."
                )
            try:
                raw = rerank_chat(
                    attempt_prompt,
                    tag="creamem3_v2_rerank",
                    temperature=0.0,
                    max_tokens=4096,
                    json_mode=True,
                )
                value = _parse_json_object(raw)
                if value is None or set(value) != {"ranked_fact_ids"}:
                    raise GraphMemoryError("reranker returned an invalid object")
                ids = value.get("ranked_fact_ids")
                if not isinstance(ids, list) or len(ids) != target:
                    raise GraphMemoryError(
                        f"reranker returned {len(ids) if isinstance(ids, list) else 'non-list'} IDs; expected {target}"
                    )
                if len(set(ids)) != len(ids):
                    raise GraphMemoryError("reranker returned duplicate IDs")
                if any(not isinstance(fact_id, str) or fact_id not in candidate_map for fact_id in ids):
                    raise GraphMemoryError("reranker returned an unknown fact ID")
                return [candidate_map[fact_id] for fact_id in ids], {
                    "attempts": attempt + 1,
                    "fallback": False,
                    "error": None,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[-400:]}"
        # Fail open is explicit in the trace.  The deterministic fallback is
        # the already-specified cosine pre-rank, never a hidden heuristic.
        return list(candidates[:target]), {
            "attempts": max_attempts,
            "fallback": True,
            "error": last_error,
        }

    def _assemble(
        self,
        selected: Sequence[Mapping[str, Any]],
        top_summaries: Sequence[Mapping[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        selected_ids = {item["fact"]["id"] for item in selected}
        view_ids: list[str] = []
        for item in top_summaries:
            summary_id = item["summary_id"]
            summary = self.summary_by_id[summary_id]
            if summary["node_type"] != "event_view":
                continue
            if selected_ids.intersection(self.adjacency.get(summary_id, [])):
                view_ids.append(summary_id)
        # Also expose any view supported by at least two final facts.  These
        # views are routing context only; the facts remain rendered once below.
        for summary in self.summaries:
            if summary["node_type"] != "event_view" or summary["id"] in view_ids:
                continue
            if len(selected_ids.intersection(self.adjacency.get(summary["id"], []))) >= 2:
                view_ids.append(summary["id"])

        lines: list[str] = []
        for view_id in view_ids:
            view = self.summary_by_id[view_id]
            lines.append(
                f"[Cross-session event view | {view['view_type']} | sessions "
                + ", ".join(str(value) for value in view["session_ids"])
                + f"] {view['text']}"
            )

        by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        session_order: list[int] = []
        for item in selected:
            fact = item["fact"]
            session_id = int(fact["session_id"])
            if session_id not in by_session:
                session_order.append(session_id)
            by_session[session_id].append(fact)
        session_summary = {
            int(node["session_ids"][0]): node
            for node in self.summaries
            if node["node_type"] == "session_packet"
        }
        rendered_summary_ids: list[str] = []
        for session_id in session_order:
            summary = session_summary[session_id]
            rendered_summary_ids.append(summary["id"])
            lines.append(
                f"[Session summary | session {session_id} | {summary.get('timestamp', '')}] "
                f"{summary['text']}"
            )
            for fact in by_session[session_id]:
                sources = ", ".join(fact["source_dia_ids"])
                lines.append(f"  - [{sources}] {fact['text']}")
        text = "\n".join(lines)
        return text, {
            "rendered_event_view_ids": view_ids,
            "rendered_session_summary_ids": rendered_summary_ids,
            "rendered_fact_ids": [item["fact"]["id"] for item in selected],
            "rendered_chars": len(text),
        }

    def search(
        self,
        question: str,
        query_vector: Sequence[float],
        rerank_chat: ChatFn,
    ) -> tuple[str, dict[str, Any]]:
        candidates, trace = self._candidate_pool(str(question), query_vector)
        selected, rerank_trace = self._rerank(
            str(question), candidates, rerank_chat
        )
        text, assembly_trace = self._assemble(selected, trace["top_summaries"])
        trace["rerank"] = {
            **rerank_trace,
            "selected_fact_ids": [item["fact"]["id"] for item in selected],
        }
        trace["assembly"] = assembly_trace
        return text, trace


def audit_graph(
    graph: Mapping[str, Any],
    *,
    legal_dia_ids: Iterable[str],
    expected_sessions: int,
) -> dict[str, Any]:
    """Return machine-readable invariant checks for a saved graph."""

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append(
            {"name": name, "passed": bool(condition), "detail": detail}
        )

    summaries = list(graph.get("summary_nodes", []))
    facts = list(graph.get("fact_nodes", []))
    edges = list(graph.get("edges", []))
    legal = set(legal_dia_ids)
    summary_ids = [str(node.get("id", "")) for node in summaries]
    fact_ids = [str(node.get("id", "")) for node in facts]
    session_summaries = [
        node for node in summaries if node.get("node_type") == "session_packet"
    ]
    event_views = [node for node in summaries if node.get("node_type") == "event_view"]
    check("schema_version", graph.get("schema_version") == GRAPH_SCHEMA_VERSION)
    check("typed_graph_representation", graph.get("architecture", {}).get("representation") == "typed_many_to_many_event_graph")
    check("text_only_scope", graph.get("architecture", {}).get("input_scope") == "dialogue_text_only")
    check("one_session_summary_each", len(session_summaries) == expected_sessions)
    check("facts_nonempty", bool(facts))
    check("unique_summary_ids", len(summary_ids) == len(set(summary_ids)))
    check("unique_fact_ids", len(fact_ids) == len(set(fact_ids)))
    check(
        "fact_sources_legal",
        all(
            isinstance(fact.get("source_dia_ids"), list)
            and bool(fact["source_dia_ids"])
            and set(fact["source_dia_ids"]).issubset(legal)
            for fact in facts
        ),
    )
    check(
        "fact_sources_match_session",
        all(
            all(
                int(source_id[1:].split(":", 1)[0]) == int(fact["session_id"])
                for source_id in fact.get("source_dia_ids", [])
            )
            for fact in facts
        ),
    )
    valid_edge_types = {"SUPPORTS", "VIEW_SUPPORTS"}
    check(
        "edge_references_valid",
        all(
            edge.get("edge_type") in valid_edge_types
            and edge.get("source_id") in set(summary_ids)
            and edge.get("target_id") in set(fact_ids)
            for edge in edges
        ),
    )
    check(
        "event_views_cross_session",
        all(len(set(node.get("session_ids", []))) >= 2 for node in event_views),
    )
    check(
        "flat_node_storage",
        all("atomic_facts" not in node and "children" not in node for node in summaries)
        and all("parent_summary_id" not in node for node in facts),
    )
    check(
        "embeddings_present",
        all(bool(node.get("embedding")) for node in summaries + facts),
    )
    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "passed": len(checks) - len(failures),
        "failed": len(failures),
    }


__all__ = [
    "BM25_FACT_TOP_K",
    "CANDIDATE_CAP",
    "DENSE_FACT_TOP_K",
    "GRAPH_SCHEMA_VERSION",
    "GraphMemoryError",
    "GraphMemoryIndex",
    "RERANK_TOP_K",
    "SUMMARY_TOP_K",
    "add_offline_event_views",
    "audit_graph",
    "build_session_graph",
    "canonical_sha256",
    "extract_session_packet",
    "iso_date_from_timestamp",
    "render_text_only_session",
]

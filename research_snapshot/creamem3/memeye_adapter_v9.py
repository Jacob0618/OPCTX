"""MemEye-to-CreaMem3 adapter using MGTM's frozen visual extraction rules.

The source MemEye reader, caption cache, and extraction instructions live in
``CreaMem-AutoResearch``.  This module reads them without modifying that
workspace.  MGTM's visual rules are retained, while only the output contract is
adapted from a tree-specific/Mem0 object to CreaMem3's provenance-bound
``session_packet -> atomic_fact`` graph.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from creamem3 import graph_memory_v2 as gm


SCHEMA_VERSION = "creamem3.memeye_adapter.v9.1"
DEFAULT_SOURCE_ROOT = Path(r"C:\Users\86139\Desktop\CreaMem-AutoResearch")
DEFAULT_DATA_ROOT = DEFAULT_SOURCE_ROOT / "external_memeye_data"
DEFAULT_CAPTION_ROOT = (
    DEFAULT_SOURCE_ROOT
    / "multimodal"
    / "cache"
    / "memeye_captions_memeye_v8_recent9_summary2_context_b10"
)
SOURCE_PROMPT_FILE = DEFAULT_SOURCE_ROOT / "multimodal" / "eval_memeye.py"
SOURCE_MEM0_FILE = (
    DEFAULT_SOURCE_ROOT / "multimodal" / "vendor_mem0_prompts_official.py"
)
EXPECTED_CAPTION_MODEL = "gemini-2.5-flash-lite"
EXPECTED_CAPTION_PROMPT = "memeye_v8_recent9_summary2_context_b10"

ChatFn = Callable[..., str]
EmbedFn = Callable[..., list[list[float]]]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _literal_constant(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str):
                        raise TypeError(f"{name} is not a string")
                    return value
    raise KeyError(f"missing string constant {name} in {path}")


def load_mgtm_prompt_sources(
    source_prompt_file: Path = SOURCE_PROMPT_FILE,
    source_mem0_file: Path = SOURCE_MEM0_FILE,
) -> dict[str, str]:
    """Load exact MGTM prompt constants and bind their content hashes."""

    values = {
        "summary_prompt": _literal_constant(source_prompt_file, "SUMMARY_PROMPT"),
        "visual_custom_instructions": _literal_constant(
            source_prompt_file, "MEM0_VISUAL_CUSTOM_INSTRUCTIONS"
        ),
        "visual_system_override": _literal_constant(
            source_prompt_file, "MEM0_VISUAL_SYSTEM_OVERRIDE"
        ),
        "mem0_additive_prompt": _literal_constant(
            source_mem0_file, "ADDITIVE_EXTRACTION_PROMPT"
        ),
    }
    values["source_prompt_file_sha256"] = gm.canonical_sha256(
        {"path": str(source_prompt_file.resolve()), "bytes": source_prompt_file.read_bytes().hex()}
    )
    values["source_mem0_file_sha256"] = gm.canonical_sha256(
        {"path": str(source_mem0_file.resolve()), "bytes": source_mem0_file.read_bytes().hex()}
    )
    values["prompt_bundle_sha256"] = gm.canonical_sha256(
        {key: value for key, value in values.items() if not key.endswith("file_sha256")}
    )
    return values


CREAMEM3_OUTPUT_OVERRIDE = r"""CREAMEM3 OUTPUT ADAPTATION (highest priority)

Retain all MGTM/Mem0 visual extraction and high-recall evidence rules above.
Only the output schema changes because CreaMem3 stores immutable atomic facts
with explicit provenance instead of a Mem0/MGTM tree leaf object.

Return exactly one JSON object and no prose:
{
  "atomic_facts": [
    {"fact": "complete anchored answerable sentence", "source_dia_ids": ["D1:1"]}
  ]
}

Rules for source_dia_ids:
- cite one or more identifiers from LEGAL_SOURCE_DIA_IDS below;
- cite every source turn directly supporting the fact;
- never invent, shorten, or rewrite an identifier;
- a cross-round comparison or change fact must cite all compared source turns.

The earlier Mem0 instruction to emit {"memory": ...} is superseded only for
serialization.  Its extraction semantics and the complete MGTM visual rules
remain binding.
"""


MEMEYE_ONLINE_EVENT_UPDATE_PROMPT = r"""You update compact cross-session event routing views after receiving MemEye session {frontier_session_id}.

The input is limited to the current session, selected earlier facts, and selected active views from the causal prefix. Atomic facts are immutable evidence. MemEye sessions contain many visual state snapshots; sharing a task, game, person, object, or activity is not enough to make snapshots one event. Connect only facts that describe the same concrete occurrence, a continuing process, or an explicit state/update chain across sessions.

Output STRICT JSON:
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
1. Copy all IDs exactly. A create spans current and earlier sessions; an extend targets one supplied active view, preserves all its members, and adds a current-session fact.
2. Include only the smallest decisive evidence set, normally 4-12 facts and never more than 24 facts per operation. Do not copy every snapshot from a session.
3. Every member fact must appear in at least one summary_claim support list. Every claim must be fully supported by its cited member facts.
4. Keep separate game rounds, visits, dates, attempts, and repeated activities separate unless the evidence explicitly makes them one continuing occurrence or update chain.
5. Membership only grows. Never rewrite, delete, merge, or duplicate atomic facts. Return at most 8 operations.
6. For create, target_view_id MUST be exactly the empty string "". For extend, it MUST be one supplied active view ID.
7. Before returning, verify that every create contains at least one current_session fact ID and at least one prior_candidate_facts fact ID. The cross_session_candidate_pairs are similarity hints, not proof; use only pairs whose texts support the claimed relation.
8. Prefer an empty operations list over a broad thematic group. Return JSON only.

Causal prefix input JSON:
{online_input_json}
"""


MEMEYE_OFFLINE_EVENT_VIEW_PROMPT = r"""You create compact non-destructive cross-session event routing views for MemEye memory.

You can inspect all session summaries and immutable atomic facts together. Discover non-adjacent evidence that belongs to the same concrete occurrence, continuing process, or explicit update chain, then create derived views. MemEye sessions contain many visual state snapshots; sharing a task, game, person, object, or repeated activity is not enough to make snapshots one event.

Output STRICT JSON:
{{
  "event_views": [
    {{
      "view_id": "V1",
      "view_type": "same_event|continuing_process|update_chain",
      "summary": "<2-4 evidence-grounded sentences with distinguishing dates, rounds, or stages>",
      "member_fact_ids": ["F0001", "F0002"]
    }}
  ]
}}

Rules:
1. Copy fact IDs exactly. Each view must contain facts from at least two sessions.
2. Include only the smallest decisive evidence set, normally 4-12 facts and never more than 24 facts per view. Do not copy every snapshot from a session.
3. The summary may state only claims supported by members. Preserve exact dates, rounds, uncertainty, corrections, and state order.
4. Keep separate games, visits, attempts, dates, and repeated activities separate unless evidence explicitly establishes one continuing occurrence or update chain.
5. A fact may support overlapping views only when scopes genuinely overlap. Prefer no view over a broad thematic group.
6. Return JSON only.

Memory packets:
{memory_packets_json}
"""


MEMEYE_ONLINE_EVENT_PROMPT_SHA256 = _sha256_text(
    MEMEYE_ONLINE_EVENT_UPDATE_PROMPT
)
MEMEYE_OFFLINE_EVENT_PROMPT_SHA256 = _sha256_text(
    MEMEYE_OFFLINE_EVENT_VIEW_PROMPT
)
MEMEYE_OFFLINE_LARGE_EVENT_VIEW_PROMPT = r"""You perform global event grouping for a large MemEye long-term memory.

Inspect ALL supplied session summaries and atomic facts together. Discover non-adjacent sessions that describe the same concrete occurrence, a continuing process, or an explicit update chain. The full facts are evidence for grouping, but the output deliberately names sessions rather than copying long fact-id lists; a deterministic evidence projection stage will select compact atomic support afterward.

Output STRICT JSON:
{{
  "event_groups": [
    {{
      "view_type": "same_event|continuing_process|update_chain",
      "summary": "<1-2 concise evidence-grounded sentences with dates/stages>",
      "member_session_ids": [1, 3]
    }}
  ]
}}

Rules:
1. Every group must contain 2-8 distinct session IDs copied from the input.
2. Return at most 16 highest-confidence groups.
3. Do not group sessions merely because they share a broad character, show, topic, location, or activity. Keep distinct occurrences separate.
4. Preserve dates, corrections, uncertainty, and temporal order. The summary may state only claims supported somewhere in the named sessions' supplied facts.
5. The umbrella brand-redesign/cartoon-analysis project is explicitly NOT a valid event group. Never put all or most sessions in one group. If that umbrella is the only relationship, return an empty event_groups list.
6. Mechanically verify before output that every member_session_ids list has length 2 through 8. Any longer list is invalid.
7. Do not output fact IDs, Markdown, explanations, or any field outside the schema. Prefer no group over a weak thematic group.

Complete memory packets:
{memory_packets_json}
"""
MEMEYE_OFFLINE_LARGE_EVENT_PROMPT_SHA256 = _sha256_text(
    MEMEYE_OFFLINE_LARGE_EVENT_VIEW_PROMPT
)


def load_mgtm_evidence_aggregator_prompt(
    source_prompt_file: Path = SOURCE_PROMPT_FILE,
) -> str:
    """Load MGTM's exact MemEye evidence-aggregation prompt."""

    return _literal_constant(source_prompt_file, "EVIDENCE_AGGREGATOR_PROMPT")


def load_mgtm_open_retrieval_instruct(
    source_prompt_file: Path = SOURCE_PROMPT_FILE,
) -> str:
    """Load the exact Open-QA Qwen reranker instruction used by MGTM."""

    return _literal_constant(source_prompt_file, "MEMEYE_OPEN_RETRIEVAL_INSTRUCT")


def add_large_memeye_offline_event_views(
    session_graph: Mapping[str, Any],
    chat_fn: ChatFn,
    embed_fn: EmbedFn,
    *,
    max_attempts: int = 4,
    member_fact_cap: int = 12,
) -> dict[str, Any]:
    """Group a full large graph by session, then project compact fact support.

    The grouping model still receives every session packet and atomic fact.  It
    emits only a bounded session-level plan so its JSON cannot be dominated by
    thousands of fact IDs.  Existing fact embeddings then select a balanced,
    auditable support set without rewriting any fact.
    """

    memory_packets = gm._fusion_input(session_graph)  # type: ignore[attr-defined]
    base_prompt = MEMEYE_OFFLINE_LARGE_EVENT_VIEW_PROMPT.format(
        memory_packets_json=memory_packets
    )
    value: Mapping[str, Any] | None = None
    attempts = 0
    last_error = "unknown large Offline grouping error"
    for attempt in range(max_attempts):
        attempts = attempt + 1
        prompt = base_prompt
        if attempt:
            prompt += (
                f"\n\nRetry ordinal {attempt + 1}. Previous validation error: "
                + last_error
                + "\nReturn a shorter fresh JSON object with no extra fields."
            )
        try:
            raw = chat_fn(
                prompt,
                tag="creamem3_memeye_large_offline_group",
                temperature=0.0,
                max_tokens=4096,
                json_mode=True,
            )
            parsed = _parse_json(raw)
            if set(parsed) != {"event_groups"} or not isinstance(
                parsed.get("event_groups"), list
            ):
                raise gm.GraphMemoryError("invalid event_groups object")
            value = parsed
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-500:]}"
    if value is None:
        raise gm.GraphMemoryError(
            f"large Offline grouping failed after {max_attempts} attempts: {last_error}"
        )

    facts = list(session_graph.get("fact_nodes", []))
    facts_by_session: dict[int, list[Mapping[str, Any]]] = {}
    for fact in facts:
        facts_by_session.setdefault(int(fact["session_id"]), []).append(fact)
    legal_sessions = set(facts_by_session)
    accepted_groups: list[dict[str, Any]] = []
    rejected_groups: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for index, item in enumerate(value.get("event_groups", [])[:16]):
        try:
            if not isinstance(item, Mapping) or set(item) != {
                "view_type",
                "summary",
                "member_session_ids",
            }:
                raise gm.GraphMemoryError("invalid event group shape")
            view_type = str(item.get("view_type", ""))
            if view_type not in {"same_event", "continuing_process", "update_chain"}:
                raise gm.GraphMemoryError("invalid event group type")
            summary = " ".join(str(item.get("summary", "")).split())
            raw_sessions = item.get("member_session_ids")
            if not summary or not isinstance(raw_sessions, list):
                raise gm.GraphMemoryError("empty event group")
            session_ids: list[int] = []
            for raw_session in raw_sessions:
                if not isinstance(raw_session, int) or raw_session not in legal_sessions:
                    raise gm.GraphMemoryError("unknown event group session")
                if raw_session not in session_ids:
                    session_ids.append(raw_session)
            session_ids.sort()
            if len(session_ids) < 2 or len(session_ids) > 8:
                raise gm.GraphMemoryError("event group must span 2-8 sessions")
            identity = (view_type, tuple(session_ids))
            if identity in seen:
                raise gm.GraphMemoryError("duplicate event group")
            seen.add(identity)
            accepted_groups.append(
                {
                    "view_type": view_type,
                    "summary": summary,
                    "session_ids": session_ids,
                }
            )
        except gm.GraphMemoryError as exc:
            rejected_groups.append(
                {"source_index": index, "reason": str(exc)}
            )

    if accepted_groups:
        query_vectors = embed_fn(
            [group["summary"] for group in accepted_groups],
            tag="creamem3_memeye_large_offline_group_embed",
            task_type="RETRIEVAL_QUERY",
        )
    else:
        query_vectors = []
    standard_views: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for ordinal, (group, query_vector) in enumerate(
        zip(accepted_groups, query_vectors), 1
    ):
        query = np.asarray(query_vector, dtype=np.float64)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scored: list[tuple[float, Mapping[str, Any]]] = []
        per_session_best: list[tuple[float, Mapping[str, Any]]] = []
        for session_id in group["session_ids"]:
            session_scored: list[tuple[float, Mapping[str, Any]]] = []
            for fact in facts_by_session[session_id]:
                vector = np.asarray(fact.get("embedding", []), dtype=np.float64)
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                session_scored.append((float(vector @ query), fact))
            session_scored.sort(
                key=lambda row: (-row[0], str(row[1]["id"]))
            )
            if session_scored:
                per_session_best.append(session_scored[0])
                scored.extend(session_scored)
        selected_ids = [str(row[1]["id"]) for row in per_session_best]
        scored.sort(key=lambda row: (-row[0], str(row[1]["id"])))
        for _score, fact in scored:
            fact_id = str(fact["id"])
            if fact_id not in selected_ids:
                selected_ids.append(fact_id)
            if len(selected_ids) >= member_fact_cap:
                break
        standard_views.append(
            {
                "view_id": f"LG{ordinal}",
                "view_type": group["view_type"],
                "summary": group["summary"],
                "member_fact_ids": selected_ids,
            }
        )
        projection_rows.append(
            {
                "source_group_index": ordinal - 1,
                "session_ids": group["session_ids"],
                "member_fact_ids": selected_ids,
            }
        )

    response = json.dumps(
        {"event_views": standard_views}, ensure_ascii=False, separators=(",", ":")
    )
    graph = gm.add_offline_event_views(
        session_graph,
        lambda *_args, **_kwargs: response,
        embed_fn,
        max_attempts=1,
        prompt_template=MEMEYE_OFFLINE_LARGE_EVENT_VIEW_PROMPT,
        prompt_sha256=MEMEYE_OFFLINE_LARGE_EVENT_PROMPT_SHA256,
    )
    graph["offline_fusion"].update(
        {
            "policy": "global_full_context_session_group_then_balanced_fact_projection",
            "grouping_attempts": attempts,
            "group_count_raw": len(value.get("event_groups", [])),
            "group_count_accepted": len(accepted_groups),
            "group_rejections": rejected_groups,
            "fact_projection": {
                "policy": "embedding_top_with_one_fact_per_member_session",
                "member_fact_cap": member_fact_cap,
                "rows": projection_rows,
            },
        }
    )
    return graph


def _parse_json(raw: str) -> Mapping[str, Any]:
    value = gm._parse_json_object(raw)  # type: ignore[attr-defined]
    if not isinstance(value, Mapping):
        raise gm.GraphMemoryError("MemEye writer returned no JSON object")
    return value


def _iso_date(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return gm.iso_date_from_timestamp(text)


def _session_batch(record: Mapping[str, Any]) -> str:
    lines = [
        "## Chronological Session Evidence",
        "All turns below are ordered from earliest to latest. Preserve exact image, round, date, state, and order anchors.",
    ]
    for turn in record.get("turns", []):
        lines.append(
            f"{turn.get('speaker', '')} ({turn.get('dia_id', '')}): {turn.get('text', '')}"
        )
    return "\n\n".join(lines)


def _summary(
    record: Mapping[str, Any],
    prompt_sources: Mapping[str, str],
    chat_fn: ChatFn,
    *,
    max_attempts: int,
) -> tuple[str, int]:
    batch = _session_batch(record)
    prompt = str(prompt_sources["summary_prompt"]).replace("{batch}", batch)
    last_error = "unknown summary error"
    for attempt in range(max_attempts):
        try:
            raw = chat_fn(
                prompt,
                tag="creamem3_memeye_summary",
                temperature=0.0,
                max_tokens=1600,
                json_mode=True,
            )
            value = _parse_json(raw)
            summary = str(value.get("batch_summary", "")).strip()
            if not summary:
                raise gm.GraphMemoryError("empty batch_summary")
            return summary, attempt + 1
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-500:]}"
            prompt += (
                "\n\nThe previous response failed validation: "
                + last_error
                + '\nReturn only {"batch_summary":"..."}.'
            )
    raise gm.GraphMemoryError(
        f"session {record['session_id']} summary failed: {last_error}"
    )


def _fact_user_prompt(
    record: Mapping[str, Any],
    summary: str,
    prompt_sources: Mapping[str, str],
) -> str:
    batch = _session_batch(record)
    timestamp = str(record.get("timestamp", "") or "")
    legal_ids = [str(turn["dia_id"]) for turn in record.get("turns", [])]
    sections = [
        f"## Summary\n{summary}",
        "## Last k Messages\n",
        "## Recently Extracted Memories\n[]",
        "## Existing Memories\n[]",
        f"## New Messages\n{batch}",
        f"## Observation Date\n{timestamp}",
        f"## Current Date\n{date.today().isoformat()}",
        "## Custom Instructions\n"
        + str(prompt_sources["visual_custom_instructions"]),
        "## LEGAL_SOURCE_DIA_IDS\n"
        + json.dumps(legal_ids, ensure_ascii=False),
        CREAMEM3_OUTPUT_OVERRIDE,
        "# Output:",
    ]
    return "\n\n".join(sections)


def extract_memeye_session_packet(
    record: Mapping[str, Any],
    chat_fn: ChatFn,
    prompt_sources: Mapping[str, str],
    *,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Run MGTM-style visual extraction for one physical MemEye session."""

    summary, summary_attempts = _summary(
        record, prompt_sources, chat_fn, max_attempts=max_attempts
    )
    legal_ids = [str(turn["dia_id"]) for turn in record.get("turns", [])]
    system = (
        str(prompt_sources["mem0_additive_prompt"])
        + "\n\n"
        + str(prompt_sources["visual_system_override"])
        + "\n\n"
        + CREAMEM3_OUTPUT_OVERRIDE
    )
    prompt = _fact_user_prompt(record, summary, prompt_sources)
    last_error = "unknown fact extraction error"
    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                f"\n\nRetry ordinal {attempt + 1}. The previous response failed validation: "
                + last_error
                + "\nReturn a fresh object following the exact CreaMem3 schema."
            )
        try:
            raw = chat_fn(
                attempt_prompt,
                system=system,
                tag="creamem3_memeye_write",
                temperature=0.0,
                max_tokens=16384,
                json_mode=True,
            )
            value = _parse_json(raw)
            if set(value) != {"atomic_facts"}:
                raise gm.GraphMemoryError(
                    "writer output must contain only atomic_facts"
                )
            combined = {
                "batch_summary": summary,
                "atomic_facts": value["atomic_facts"],
            }
            normalized_summary, facts = gm._validate_session_packet(  # type: ignore[attr-defined]
                combined, legal_ids
            )
            batch = _session_batch(record)
            return {
                "session_id": int(record["session_id"]),
                "source_session_id": str(record.get("source_session_id", "")),
                "timestamp": str(record.get("timestamp", "") or ""),
                "iso_date": _iso_date(str(record.get("timestamp", "") or "")),
                "turn_count": len(legal_ids),
                "source_dia_ids": legal_ids,
                "batch_summary": normalized_summary,
                "atomic_facts": facts,
                "summary_attempts": summary_attempts,
                "writer_attempts": attempt + 1,
                "writer_prompt_sha256": str(
                    prompt_sources["prompt_bundle_sha256"]
                ),
                "input_sha256": _sha256_text(batch),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-700:]}"
    raise gm.GraphMemoryError(
        f"session {record['session_id']} writer failed: {last_error}"
    )


def build_memeye_session_graph(
    records: Sequence[Mapping[str, Any]],
    chat_fn: ChatFn,
    embed_fn: EmbedFn,
    prompt_sources: Mapping[str, str],
    *,
    workers: int = 3,
    max_attempts: int = 6,
) -> dict[str, Any]:
    """Build the standard CreaMem3 graph with the adapted MGTM writer."""

    ordered = sorted(records, key=lambda item: int(item["session_id"]))
    packets: list[dict[str, Any] | None] = [None] * len(ordered)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                extract_memeye_session_packet,
                record,
                chat_fn,
                prompt_sources,
                max_attempts=max_attempts,
            ): index
            for index, record in enumerate(ordered)
        }
        done = 0
        for future in as_completed(futures):
            packets[futures[future]] = future.result()
            done += 1
            print(f"  [MemEye write] {done}/{len(ordered)} sessions", flush=True)
    normalized = [packet for packet in packets if packet is not None]
    if len(normalized) != len(ordered):
        raise gm.GraphMemoryError("one or more MemEye session packets are missing")

    summaries: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    fact_counter = 0
    for packet in normalized:
        session_id = int(packet["session_id"])
        summary_id = f"S{session_id:03d}"
        summary_sources: list[str] = []
        for fact in packet["atomic_facts"]:
            fact_counter += 1
            fact_id = f"F{fact_counter:05d}"
            for source_id in fact["source_dia_ids"]:
                if source_id not in summary_sources:
                    summary_sources.append(source_id)
            facts.append(
                {
                    "id": fact_id,
                    "node_type": "atomic_fact",
                    "text": str(fact["fact"]),
                    "source_dia_ids": list(fact["source_dia_ids"]),
                    "session_id": session_id,
                    "source_session_id": packet["source_session_id"],
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
        summaries.append(
            {
                "id": summary_id,
                "node_type": "session_packet",
                "text": packet["batch_summary"],
                "session_ids": [session_id],
                "source_session_id": packet["source_session_id"],
                "timestamp": packet["timestamp"],
                "iso_date": packet["iso_date"],
                "source_dia_ids": summary_sources,
                "embedding": [],
            }
        )

    texts = [node["text"] for node in summaries] + [node["text"] for node in facts]
    vectors = gm._embed_checked(  # type: ignore[attr-defined]
        texts, embed_fn, tag="creamem3_memeye_embed_write"
    )
    split = len(summaries)
    for node, vector in zip(summaries, vectors[:split]):
        node["embedding"] = vector
    for node, vector in zip(facts, vectors[split:]):
        node["embedding"] = vector

    graph: dict[str, Any] = {
        "schema_version": gm.GRAPH_SCHEMA_VERSION,
        "graph_variant": "session",
        "architecture": {
            "representation": "typed_many_to_many_event_graph",
            "fact_policy": "immutable_atomic_facts_are_source_of_truth",
            "summary_policy": "derived_routing_views_do_not_own_facts",
            "write_unit": "one_complete_physical_memeye_session",
            "input_scope": "dialogue_plus_frozen_gemini_image_captions",
            "source_extraction_semantics": "mgtm_mem0_visual_high_recall",
            "source_output_adaptation": "creamem3_fact_plus_provenance_schema_only",
        },
        "prompt_schema_version": SCHEMA_VERSION,
        "prompt_sha256": {
            **dict(gm.PROMPT_SHA256),
            "memeye_mgtm_bundle": prompt_sources["prompt_bundle_sha256"],
            "memeye_output_override": _sha256_text(CREAMEM3_OUTPUT_OVERRIDE),
        },
        "session_packets": normalized,
        "summary_nodes": summaries,
        "fact_nodes": facts,
        "edges": edges,
        "stats": {
            "sessions": len(normalized),
            "session_summary_nodes": len(summaries),
            "event_view_nodes": 0,
            "facts": len(facts),
            "edges": len(edges),
            "summary_calls": sum(int(p["summary_attempts"]) for p in normalized),
            "writer_calls": sum(int(p["writer_attempts"]) for p in normalized),
            "writer_retries": sum(
                max(0, int(p["writer_attempts"]) - 1) for p in normalized
            ),
            "min_facts_per_session": min(
                (len(p["atomic_facts"]) for p in normalized), default=0
            ),
            "max_facts_per_session": max(
                (len(p["atomic_facts"]) for p in normalized), default=0
            ),
        },
    }
    graph["content_fingerprint"] = gm.canonical_sha256(
        {
            "summaries": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in summaries
            ],
            "facts": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in facts
            ],
            "edges": edges,
        }
    )
    return graph


def build_records_from_cached_captions(
    task_name: str,
    data: Mapping[str, Any],
    captions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Convert MemEye sessions to CreaMem3 records with synthetic legal IDs."""

    raw_sessions = data.get("multi_session_dialogues", [])
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"{task_name} has no multi_session_dialogues")
    records: list[dict[str, Any]] = []
    source_map: dict[str, dict[str, Any]] = {}
    for numeric_id, session in enumerate(raw_sessions, start=1):
        source_session_id = str(session.get("session_id", f"S{numeric_id}"))
        timestamp = str(session.get("date", "") or "")
        turns: list[dict[str, str]] = []
        for round_index, rd in enumerate(session.get("dialogues", []), start=1):
            round_id = str(rd.get("round") or f"{source_session_id}:R{round_index}")
            anchor = f"[{timestamp} | {source_session_id} | {round_id}]"

            def add_turn(speaker: str, text: str, source_kind: str, extra: Mapping[str, Any] | None = None) -> None:
                if not str(text).strip():
                    return
                dia_id = f"D{numeric_id}:{len(turns) + 1}"
                turns.append(
                    {"speaker": speaker, "dia_id": dia_id, "text": f"{anchor}\n{text}"}
                )
                source_map[dia_id] = {
                    "task": task_name,
                    "session_id": source_session_id,
                    "date": timestamp,
                    "round": round_id,
                    "source_kind": source_kind,
                    **dict(extra or {}),
                }

            add_turn("User", str(rd.get("user", "") or ""), "user")
            add_turn("Assistant", str(rd.get("assistant", "") or ""), "assistant")
            rels = rd.get("input_image") or []
            rels = rels if isinstance(rels, list) else [rels]
            image_ids = rd.get("image_id") or []
            image_ids = image_ids if isinstance(image_ids, list) else [image_ids]
            hints = rd.get("image_caption") or []
            hints = hints if isinstance(hints, list) else [hints]
            for image_index, rel in enumerate(rels):
                rel = str(rel)
                image_id = str(image_ids[image_index]) if image_index < len(image_ids) else ""
                hint = str(hints[image_index]) if image_index < len(hints) else ""
                label = f"Image {rel}"
                if image_id:
                    label += f" ({image_id})"
                text = f"{label}: {captions.get(rel, '')}"
                if hint:
                    text += f"\nDataset hint for {rel}: {hint}"
                add_turn(
                    "VisualMemory",
                    text,
                    "image_caption",
                    {"image": rel, "image_id": image_id},
                )
        if not turns:
            dia_id = f"D{numeric_id}:1"
            turns.append(
                {
                    "speaker": "MemoryRecord",
                    "dia_id": dia_id,
                    "text": f"[{timestamp} | {source_session_id}] Empty session record.",
                }
            )
            source_map[dia_id] = {
                "task": task_name,
                "session_id": source_session_id,
                "date": timestamp,
                "round": "",
                "source_kind": "empty_session",
            }
        records.append(
            {
                "session_id": numeric_id,
                "source_session_id": source_session_id,
                "timestamp": timestamp,
                "turns": turns,
            }
        )
    return records, source_map


def validate_caption_coverage(
    task_name: str,
    data: Mapping[str, Any],
    caption_root: Path = DEFAULT_CAPTION_ROOT,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load every referenced frozen caption and enforce source metadata."""

    images: list[str] = []
    for session in data.get("multi_session_dialogues", []):
        for rd in session.get("dialogues", []):
            rels = rd.get("input_image") or []
            rels = rels if isinstance(rels, list) else [rels]
            for rel in rels:
                rel = str(rel)
                if rel and rel not in images:
                    images.append(rel)
    captions: dict[str, str] = {}
    metadata_rows: list[dict[str, str]] = []
    issues: list[str] = []
    for rel in images:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
        path = caption_root / task_name / f"{safe}.json"
        if not path.is_file():
            issues.append(f"missing:{rel}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        caption = str(payload.get("caption", "")).strip()
        model = str(payload.get("model", ""))
        prompt_version = str(payload.get("prompt_version", ""))
        if not caption:
            issues.append(f"empty:{rel}")
        if model != EXPECTED_CAPTION_MODEL:
            issues.append(f"model:{rel}:{model}")
        if prompt_version != EXPECTED_CAPTION_PROMPT:
            issues.append(f"prompt:{rel}:{prompt_version}")
        if payload.get("error"):
            issues.append(f"error:{rel}")
        captions[rel] = caption
        metadata_rows.append(
            {
                "image": rel,
                "model": model,
                "prompt_version": prompt_version,
                "caption_sha256": _sha256_text(caption),
            }
        )
    if issues:
        raise ValueError("caption validation failed: " + "; ".join(issues[:20]))
    return captions, {
        "task": task_name,
        "images": len(images),
        "captions": len(captions),
        "model": EXPECTED_CAPTION_MODEL,
        "prompt_version": EXPECTED_CAPTION_PROMPT,
        "manifest_sha256": gm.canonical_sha256(metadata_rows),
    }


def audit_memeye_graph(
    graph: Mapping[str, Any],
    *,
    legal_dia_ids: Sequence[str],
    expected_sessions: int,
) -> dict[str, Any]:
    """Run the core graph audit with an explicit multimodal input-scope gate."""

    audit = copy.deepcopy(
        gm.audit_graph(
            graph,
            legal_dia_ids=legal_dia_ids,
            expected_sessions=expected_sessions,
        )
    )
    for check in audit.get("checks", []):
        if check.get("name") == "text_only_scope":
            check["name"] = "dialogue_plus_frozen_caption_scope"
            check["passed"] = (
                graph.get("architecture", {}).get("input_scope")
                == "dialogue_plus_frozen_gemini_image_captions"
            )
            check["detail"] = "MemEye uses frozen Gemini caption evidence by design"
    failures = [check for check in audit.get("checks", []) if not check.get("passed")]
    audit["passed"] = len(audit.get("checks", [])) - len(failures)
    audit["failed"] = len(failures)
    audit["status"] = "pass" if not failures else "fail"
    audit["audit_adapter"] = "creamem3_memeye_input_scope_v1"
    return audit


__all__ = [
    "CREAMEM3_OUTPUT_OVERRIDE",
    "DEFAULT_CAPTION_ROOT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "EXPECTED_CAPTION_MODEL",
    "EXPECTED_CAPTION_PROMPT",
    "SCHEMA_VERSION",
    "build_memeye_session_graph",
    "build_records_from_cached_captions",
    "audit_memeye_graph",
    "extract_memeye_session_packet",
    "load_mgtm_prompt_sources",
    "validate_caption_coverage",
]

"""768-dimensional, three-session-frontier LongMemEval adapter.

This module deliberately leaves :mod:`creamem3.lme_opctx_v1` untouched.  It
uses the same frozen MGTM write/QA/judge prompt authority, but restores the
reference LongMemEval write unit of one to three consecutive physical
sessions.  One such write batch is one *logical causal frontier*.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import lme_opctx_v1 as legacy


SCHEMA_VERSION = "creamem3.lme_online_opctx.v2"
FRONTIER_SIZE = 3
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_TASK_TYPE_SENT: str | None = None
EMBEDDING_NORMALIZATION = "l2_unit_v1"

# Prompt/data authorities and read-side behavior are intentionally unchanged.
LME_DATASET = legacy.LME_DATASET
MGTM_ROOT = legacy.MGTM_ROOT
MGTM_PROMPT_SOURCE = legacy.MGTM_PROMPT_SOURCE
MGTM_MEMORY_SOURCE = legacy.MGTM_MEMORY_SOURCE
MGTM_CLIENT_SOURCE = legacy.MGTM_CLIENT_SOURCE
MGTM_RUNNER_SOURCE = legacy.MGTM_RUNNER_SOURCE
NonRetryableProviderError = legacy.NonRetryableProviderError
PromptContract = legacy.PromptContract
PromptContractError = legacy.PromptContractError
QwenGraphMemoryIndex = legacy.QwenGraphMemoryIndex
assert_prompt_contract = legacy.assert_prompt_contract
assert_live_prompt_contract = legacy.assert_live_prompt_contract
materialize_operation_context = legacy.materialize_operation_context
render_lme_session = legacy.render_lme_session
sha256_text = legacy.sha256_text

WRITE_MAX_ATTEMPTS = legacy.WRITE_MAX_ATTEMPTS
WRITE_RETRY_BACKOFF_S = legacy.WRITE_RETRY_BACKOFF_S
WRITE_MAX_TOKENS = legacy.WRITE_MAX_TOKENS

ChatFn = Callable[..., str]
EmbedFn = Callable[..., Sequence[Sequence[float]]]
DisplayTextFn = Callable[[str], tuple[str, bool]]


def group_session_specs(
    ordered_specs: Sequence[Mapping[str, Any]],
    frontier_size: int = FRONTIER_SIZE,
) -> list[dict[str, Any]]:
    """Chunk a stable chronological stream without padding or duplication."""

    if isinstance(frontier_size, bool) or not isinstance(frontier_size, int) or frontier_size <= 0:
        raise ValueError("frontier_size must be a positive integer")
    materialized = [dict(spec) for spec in ordered_specs]
    groups: list[dict[str, Any]] = []
    for start in range(0, len(materialized), frontier_size):
        members = materialized[start : start + frontier_size]
        groups.append(
            {
                "frontier_id": len(groups) + 1,
                "frontier_size": len(members),
                "members": members,
            }
        )
    flattened = [member for group in groups for member in group["members"]]
    if flattened != materialized:
        raise RuntimeError("frontier plan does not preserve the ordered physical sessions")
    if groups and any(group["frontier_size"] != frontier_size for group in groups[:-1]):
        raise RuntimeError("only the final frontier may be shorter")
    return groups


def render_lme_frontier(frontier: Mapping[str, Any]) -> str:
    """Byte-match the release ``LongMemEvalMGTM.render_sessions`` batch."""

    members = list(frontier.get("members", []))
    if not 1 <= len(members) <= FRONTIER_SIZE:
        raise ValueError("a write frontier must contain one to three sessions")
    return "\n".join(
        render_lme_session(str(spec["source_session_id"]), str(spec["date"]), spec["turns"])
        for spec in members
    )


def render_write_prompt(
    contract: PromptContract,
    batch: str,
    *,
    unit_count: int,
) -> str:
    if not 1 <= int(unit_count) <= FRONTIER_SIZE:
        raise ValueError("unit_count must be between one and three")
    return contract.write_template.format(
        unit_count=int(unit_count),
        unit_label="session" if int(unit_count) == 1 else "sessions",
        batch=str(batch),
    )


def write_one_frontier(
    *,
    contract: PromptContract,
    frontier: Mapping[str, Any],
    chat_fn: ChatFn,
    write_model: str = "gemini-2.5-flash",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run the unchanged frozen writer request over one 1--3 session batch."""

    assert_live_prompt_contract(contract)
    members = [dict(spec) for spec in frontier.get("members", [])]
    if not 1 <= len(members) <= FRONTIER_SIZE:
        raise gm.GraphMemoryError("writer frontier must contain one to three sessions")
    batch = render_lme_frontier(frontier)
    prompt = render_write_prompt(contract, batch, unit_count=len(members))
    last_error = "unknown writer failure"
    request_hashes: list[str] = []
    physical_attempts: list[dict[str, Any]] = []
    for attempt in range(WRITE_MAX_ATTEMPTS):
        request_hashes.append(sha256_text(prompt))
        try:
            raw = chat_fn(
                prompt,
                model=write_model,
                temperature=0.0,
                max_tokens=WRITE_MAX_TOKENS,
                thinking_budget=0,
                tag="write",
                use_cache=False,
                json_mode=True,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-300:]}"
            physical_attempts.append(
                {
                    "ordinal": attempt + 1,
                    "canonical_request_sha256": request_hashes[-1],
                    "status": "transport_error",
                    "error_type": type(exc).__name__,
                }
            )
            if isinstance(exc, NonRetryableProviderError):
                raise
        else:
            physical_attempts.append(
                {
                    "ordinal": attempt + 1,
                    "canonical_request_sha256": request_hashes[-1],
                    "status": "response_received",
                    "response_sha256": sha256_text(str(raw or "")),
                }
            )
            value = legacy._parse_writer_json(raw)
            if value is None:
                last_error = "writer returned no valid JSON object"
            else:
                summary = str(value.get("batch_summary", "") or "").strip()
                raw_facts = value.get("atomic_facts")
                if not isinstance(raw_facts, list):
                    last_error = "writer returned atomic_facts with a non-list type"
                else:
                    facts = [legacy._fact_text(item) for item in raw_facts]
                    facts = [fact for fact in facts if fact]
                    if summary:
                        return {
                            "frontier_id": int(frontier["frontier_id"]),
                            "physical_sessions": [
                                {
                                    "source_position": int(spec["source_position"]),
                                    "source_session_id": str(spec["source_session_id"]),
                                    "date": str(spec["date"]),
                                    "turn_count": len(spec["turns"]),
                                }
                                for spec in members
                            ],
                            "batch_summary": summary,
                            "atomic_facts": facts,
                            "writer_attempts": attempt + 1,
                            "write_prompt_sha256": sha256_text(prompt),
                            "write_prompt_contract_sha256": contract.contract_sha256,
                            "request_sha256s": request_hashes,
                            "physical_attempts": physical_attempts,
                            "rendered_batch_sha256": sha256_text(batch),
                        }
                    last_error = "writer returned an empty batch_summary"
        if attempt + 1 < WRITE_MAX_ATTEMPTS:
            sleep_fn(WRITE_RETRY_BACKOFF_S)
    raise gm.GraphMemoryError(
        f"MGTM-compatible frontier write failed after {WRITE_MAX_ATTEMPTS} calls: {last_error}"
    )


def _date_range(physical_sessions: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    dates = [str(row.get("date", "")) for row in physical_sessions]
    iso_dates = [legacy._iso_date(value) for value in dates]
    iso_dates = [value for value in iso_dates if value]
    timestamp = dates[0] if len(dates) == 1 else f"{dates[0]} .. {dates[-1]}"
    iso_range = (
        iso_dates[0]
        if len(iso_dates) == 1
        else (f"{iso_dates[0]}..{iso_dates[-1]}" if iso_dates else "")
    )
    return timestamp, iso_range


def _checked_unit_vector(vector: Sequence[Any], index: int) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != EMBEDDING_DIM or not all(math.isfinite(value) for value in values):
        raise gm.GraphMemoryError(f"invalid 768-dimensional embedding at position {index}")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise gm.GraphMemoryError(f"zero or invalid embedding at position {index}")
    # The transport normally performs this normalization.  Re-normalizing here
    # makes custom/test embed functions obey the same stored-vector contract.
    return [value / norm for value in values]


def build_frontier_graph_from_packets(
    packets: Sequence[Mapping[str, Any]],
    embed_fn: EmbedFn,
    *,
    contract: PromptContract,
    display_text_fn: DisplayTextFn | None = None,
) -> dict[str, Any]:
    """Build one logical session packet per three-session writer frontier."""

    summaries: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    normalized_packets: list[dict[str, Any]] = []
    provenance_map: dict[str, Any] = {}
    fact_counter = 0
    physical_total = 0
    for ordinal, packet in enumerate(packets, 1):
        frontier_id = int(packet.get("frontier_id", ordinal))
        if frontier_id != ordinal:
            raise gm.GraphMemoryError("frontier packet order is not contiguous")
        physical = [dict(row) for row in packet.get("physical_sessions", [])]
        if not 1 <= len(physical) <= FRONTIER_SIZE:
            raise gm.GraphMemoryError("frontier packet has an invalid physical-session count")
        physical_total += len(physical)
        marker = f"D{ordinal}:0"
        summary_id = f"S{ordinal:03d}"
        timestamp, iso_range = _date_range(physical)
        provenance_map[marker] = {
            "kind": "complete_session_frontier",
            "frontier_id": ordinal,
            "physical_sessions": physical,
            "mapping_policy": "writer facts inherit their complete 1_to_3_session input frontier",
        }
        packet_facts: list[dict[str, Any]] = []
        for text in packet.get("atomic_facts", []):
            fact_text = str(text).strip()
            if not fact_text:
                continue
            fact_counter += 1
            fact_id = f"F{fact_counter:04d}"
            display_text, temporally_grounded = (
                display_text_fn(fact_text) if display_text_fn is not None else (fact_text, False)
            )
            packet_facts.append({"fact": fact_text, "source_dia_ids": [marker]})
            facts.append(
                {
                    "id": fact_id,
                    "node_type": "atomic_fact",
                    "text": fact_text,
                    "raw_text": fact_text,
                    "display_text": str(display_text),
                    "temporally_grounded": bool(temporally_grounded),
                    "source_dia_ids": [marker],
                    "session_id": ordinal,
                    "frontier_id": ordinal,
                    "physical_session_ids": [row["source_session_id"] for row in physical],
                    "timestamp": timestamp,
                    "iso_date": iso_range,
                    "embedding": [],
                }
            )
            edges.append({"edge_type": "SUPPORTS", "source_id": summary_id, "target_id": fact_id})
        summaries.append(
            {
                "id": summary_id,
                "node_type": "session_packet",
                "text": str(packet["batch_summary"]).strip(),
                "session_ids": [ordinal],
                "frontier_id": ordinal,
                "physical_session_ids": [row["source_session_id"] for row in physical],
                "timestamp": timestamp,
                "iso_date": iso_range,
                "source_dia_ids": [marker],
                "embedding": [],
            }
        )
        normalized_packets.append(
            {
                "session_id": ordinal,
                "frontier_id": ordinal,
                "frontier_size": len(physical),
                "physical_sessions": physical,
                "timestamp": timestamp,
                "iso_date": iso_range,
                "source_dia_ids": [marker],
                "batch_summary": str(packet["batch_summary"]).strip(),
                "atomic_facts": packet_facts,
                "writer_attempts": int(packet.get("writer_attempts", 1)),
                "writer_prompt_sha256": str(packet["write_prompt_sha256"]),
                "write_prompt_contract_sha256": str(packet["write_prompt_contract_sha256"]),
                "rendered_batch_sha256": str(packet["rendered_batch_sha256"]),
            }
        )

    texts = [node["text"] for node in summaries] + [node["raw_text"] for node in facts]
    vectors = list(embed_fn(texts))
    if len(vectors) != len(texts):
        raise gm.GraphMemoryError("embedding count mismatch while building frontier graph")
    checked = [_checked_unit_vector(vector, index) for index, vector in enumerate(vectors)]
    split = len(summaries)
    for node, vector in zip(summaries, checked[:split], strict=True):
        node["embedding"] = vector
    for node, vector in zip(facts, checked[split:], strict=True):
        node["embedding"] = vector

    embedding_contract = {
        "model": EMBEDDING_MODEL,
        "output_dimensionality": EMBEDDING_DIM,
        "task_type_sent": EMBEDDING_TASK_TYPE_SENT,
        "normalization": EMBEDDING_NORMALIZATION,
    }
    embedding_manifest = {
        str(node["id"]): gm.canonical_sha256(node["embedding"])
        for node in [*summaries, *facts]
    }
    graph: dict[str, Any] = {
        "schema_version": gm.GRAPH_SCHEMA_VERSION,
        "adapter_schema_version": SCHEMA_VERSION,
        "graph_variant": "session",
        "architecture": {
            "representation": "typed_many_to_many_event_graph",
            "input_scope": "dialogue_text_only",
            "fact_policy": "immutable_atomic_facts_are_source_of_truth",
            "summary_policy": "derived_routing_views_do_not_own_facts",
            "write_unit": "one_to_three_complete_consecutive_physical_sessions",
            "causal_unit": "writer_frontier",
            "causality_granularity": "frontier_not_physical_session",
            "write_prompt_environment": "mgtm_7935_lme_literal_and_direct_format",
            "provenance_adaptation": "single_complete_frontier_marker_Dordinal_0",
        },
        "embedding_contract": embedding_contract,
        "embedding_manifest_sha256": gm.canonical_sha256(embedding_manifest),
        "prompt_sha256": {
            "mgtm_write_template": contract.prompt_sha256["write_template"],
            "mgtm_qa_template": contract.prompt_sha256["qa_template"],
            "mgtm_judge_template": contract.prompt_sha256["judge_template"],
        },
        "prompt_contract_sha256": contract.contract_sha256,
        "session_packets": normalized_packets,
        "provenance_map": provenance_map,
        "summary_nodes": summaries,
        "fact_nodes": facts,
        "edges": edges,
        "stats": {
            "physical_sessions": physical_total,
            "frontiers": len(summaries),
            "sessions": len(summaries),
            "session_summary_nodes": len(summaries),
            "event_view_nodes": 0,
            "facts": len(facts),
            "edges": len(edges),
            "writer_calls": sum(int(row["writer_attempts"]) for row in normalized_packets),
            "writer_retries": sum(max(0, int(row["writer_attempts"]) - 1) for row in normalized_packets),
            "embedding_dim": EMBEDDING_DIM,
        },
    }
    graph["content_fingerprint"] = gm.canonical_sha256(
        {
            "summaries": [{k: v for k, v in node.items() if k != "embedding"} for node in summaries],
            "facts": [{k: v for k, v in node.items() if k != "embedding"} for node in facts],
            "edges": edges,
            "provenance_map": provenance_map,
            "prompt_contract_sha256": contract.contract_sha256,
            "embedding_contract": embedding_contract,
            "embedding_manifest_sha256": graph["embedding_manifest_sha256"],
        }
    )
    return graph


__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EMBEDDING_NORMALIZATION",
    "FRONTIER_SIZE",
    "LME_DATASET",
    "NonRetryableProviderError",
    "PromptContract",
    "QwenGraphMemoryIndex",
    "SCHEMA_VERSION",
    "assert_live_prompt_contract",
    "assert_prompt_contract",
    "build_frontier_graph_from_packets",
    "group_session_specs",
    "materialize_operation_context",
    "render_lme_frontier",
    "render_lme_session",
    "render_write_prompt",
    "sha256_text",
    "write_one_frontier",
]

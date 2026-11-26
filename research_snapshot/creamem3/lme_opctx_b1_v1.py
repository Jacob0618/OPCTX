"""Single-physical-session LongMemEval OpCtx adapter.

This is the strict B1 counterpart to :mod:`creamem3.lme_opctx_v2`.  It keeps
the frozen MGTM Writer/QA/Judge authority and the 768-dimensional normalized
embedding contract, but makes every physical session its own Writer and
Online causal frontier.  No Writer-output count is imposed by this adapter;
the unchanged MGTM prompt remains the sole 10--70 atomic-fact instruction.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import lme_opctx_v2 as base


SCHEMA_VERSION = "creamem3.lme_online_opctx.b1_v1"
FRONTIER_SIZE = 1

# Keep every non-frontier authority byte-identical to the validated v2 stack.
EMBEDDING_DIM = base.EMBEDDING_DIM
EMBEDDING_MODEL = base.EMBEDDING_MODEL
EMBEDDING_TASK_TYPE_SENT = base.EMBEDDING_TASK_TYPE_SENT
EMBEDDING_NORMALIZATION = base.EMBEDDING_NORMALIZATION
LME_DATASET = base.LME_DATASET
MGTM_ROOT = base.MGTM_ROOT
MGTM_PROMPT_SOURCE = base.MGTM_PROMPT_SOURCE
MGTM_MEMORY_SOURCE = base.MGTM_MEMORY_SOURCE
MGTM_CLIENT_SOURCE = base.MGTM_CLIENT_SOURCE
MGTM_RUNNER_SOURCE = base.MGTM_RUNNER_SOURCE
NonRetryableProviderError = base.NonRetryableProviderError
PromptContract = base.PromptContract
PromptContractError = base.PromptContractError
QwenGraphMemoryIndex = base.QwenGraphMemoryIndex
assert_prompt_contract = base.assert_prompt_contract
assert_live_prompt_contract = base.assert_live_prompt_contract
materialize_operation_context = base.materialize_operation_context
render_lme_session = base.render_lme_session
sha256_text = base.sha256_text
legacy = base.legacy


def _members(frontier: Mapping[str, Any]) -> list[dict[str, Any]]:
    members = [dict(row) for row in frontier.get("members", [])]
    if len(members) != 1:
        raise ValueError("B1 frontier must contain exactly one physical session")
    return members


def group_session_specs(
    ordered_specs: Sequence[Mapping[str, Any]],
    frontier_size: int = FRONTIER_SIZE,
) -> list[dict[str, Any]]:
    """Make a one-to-one, ordered mapping from physical sessions to frontiers."""

    if isinstance(frontier_size, bool) or frontier_size != FRONTIER_SIZE:
        raise ValueError("B1 frontier_size is frozen at 1")
    groups = base.group_session_specs(ordered_specs, frontier_size=FRONTIER_SIZE)
    if any(len(group["members"]) != 1 for group in groups):
        raise RuntimeError("B1 grouping did not preserve one session per frontier")
    return groups


def render_lme_frontier(frontier: Mapping[str, Any]) -> str:
    """Render exactly one session with the unchanged MGTM serialization."""

    member = _members(frontier)[0]
    return render_lme_session(
        str(member["source_session_id"]),
        str(member["date"]),
        member["turns"],
    )


def render_write_prompt(
    contract: PromptContract,
    batch: str,
    *,
    unit_count: int,
) -> str:
    """Fill the frozen writer template in its original one-session form."""

    if isinstance(unit_count, bool) or int(unit_count) != 1:
        raise ValueError("B1 unit_count is frozen at 1")
    return contract.write_template.format(
        unit_count=1,
        unit_label="session",
        batch=str(batch),
    )


def write_one_frontier(**kwargs: Any) -> dict[str, Any]:
    """Run the base writer transport after enforcing the B1 frontier shape."""

    _members(kwargs["frontier"])
    # For a one-member frontier the base renderer produces the exact same
    # frozen prompt as ``render_write_prompt`` above.  Reusing its transport
    # retains retry, parsing, temperature, thinking, and token contracts.
    return base.write_one_frontier(**kwargs)


def _content_fingerprint(graph: Mapping[str, Any]) -> str:
    return gm.canonical_sha256(
        {
            "summaries": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in graph["summary_nodes"]
            ],
            "facts": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in graph["fact_nodes"]
            ],
            "edges": graph["edges"],
            "provenance_map": graph["provenance_map"],
            "prompt_contract_sha256": graph["prompt_contract_sha256"],
            "embedding_contract": graph["embedding_contract"],
            "embedding_manifest_sha256": graph["embedding_manifest_sha256"],
        }
    )


def build_frontier_graph_from_packets(
    packets: Sequence[Mapping[str, Any]],
    embed_fn: base.EmbedFn,
    *,
    contract: PromptContract,
    display_text_fn: base.DisplayTextFn | None = None,
) -> dict[str, Any]:
    """Build a graph whose logical frontier is exactly one physical session."""

    for packet in packets:
        physical = list(packet.get("physical_sessions", []))
        if len(physical) != 1:
            raise gm.GraphMemoryError(
                "B1 graph packet must bind exactly one physical session"
            )
    graph = base.build_frontier_graph_from_packets(
        packets,
        embed_fn,
        contract=contract,
        display_text_fn=display_text_fn,
    )
    graph["adapter_schema_version"] = SCHEMA_VERSION
    graph["architecture"] = {
        **dict(graph.get("architecture", {})),
        "write_unit": "one_complete_physical_session",
        "causal_unit": "physical_session_writer_frontier",
        "causality_granularity": "physical_session",
        "provenance_adaptation": "single_complete_physical_session_marker_Dordinal_0",
        "writer_fact_count_policy": "unchanged_mgtm_soft_instruction_10_to_70_no_posthoc_cap",
    }
    for marker in graph["provenance_map"].values():
        marker["kind"] = "complete_physical_session"
        marker["mapping_policy"] = (
            "writer facts inherit their complete single physical-session input"
        )
    for packet in graph["session_packets"]:
        if packet.get("frontier_size") != 1 or len(packet["physical_sessions"]) != 1:
            raise gm.GraphMemoryError("B1 normalized packet lost its one-session binding")
    graph["content_fingerprint"] = _content_fingerprint(graph)
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

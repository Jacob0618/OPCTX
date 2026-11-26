"""Empty-session-only correction for the frozen B1 and scaled-B3 Writers.

Physical B3 boundaries are established *before* removing literal ``turns=[]``
members.  Retained frontiers are renumbered for the old graph/Online protocol,
while original session identities and original frontier ordinals are audited.
Writer prompts, transport, model summary, facts, and reader policy are delegated
unchanged.  No history is supplied to either Writer.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import lme_opctx_b1_v1 as frozen_b1
from creamem3 import lme_opctx_v2 as frozen_base
from creamem3 import lme_opctx_v2_b3_scaled_facts as frozen_b3


SCHEMA_VERSION = "creamem3.lme_clean_writer.v1"
EMPTY_SESSION_POLICY = "skip_literal_empty_turn_list_preserve_original_batch_boundaries_v1"
_DELEGATES: dict[str, ModuleType] = {"b1": frozen_b1, "b3_scaled": frozen_b3}


def _delegate(variant: str) -> ModuleType:
    if variant not in _DELEGATES:
        raise ValueError("variant must be 'b1' or 'b3_scaled'")
    return _DELEGATES[variant]


def _session_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_position": int(spec["source_position"]),
        "source_session_id": str(spec["source_session_id"]),
        "date": str(spec["date"]),
        "turn_count": len(spec["turns"]),
    }


def audit_and_group(
    ordered_specs: Sequence[Mapping[str, Any]], variant: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return nonempty frontiers and a deterministic source-to-frontier audit.

    Missing/null/non-list turns fail closed rather than being treated as an
    empty session.  A nonempty list containing blank messages is deliberately
    retained: broadening that cleaning rule would be another intervention.
    """

    source = _delegate(variant)
    specs = [dict(spec) for spec in ordered_specs]
    for spec in specs:
        if not isinstance(spec.get("turns"), list):
            raise ValueError("session turns must be a list; only literal [] is skipped")
        _session_identity(spec)  # Fail before any write if source metadata is absent.
    positions = [int(spec["source_position"]) for spec in specs]
    if len(positions) != len(set(positions)):
        raise ValueError("source_position must be unique within a question")

    # Importantly, this chunks the original physical sequence, not the filtered one.
    original_groups = source.group_session_specs(specs)
    groups: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for original in original_groups:
        original_id = int(original["frontier_id"])
        members = [spec for spec in original["members"] if spec["turns"] != []]
        skipped_members = [
            {
                **_session_identity(spec),
                "original_frontier_id": original_id,
                "reason": "literal_empty_turn_list",
            }
            for spec in original["members"]
            if spec["turns"] == []
        ]
        skipped.extend(skipped_members)
        clean_id = len(groups) + 1 if members else None
        row = {
            "original_frontier_id": original_id,
            "frontier_id": clean_id,
            "original_frontier_size": len(original["members"]),
            "retained_physical_sessions": [_session_identity(spec) for spec in members],
            "skipped_physical_sessions": skipped_members,
            "status": "retained" if members else "skipped_all_empty",
        }
        mapping.append(row)
        if members:
            groups.append(
                {
                    "frontier_id": clean_id,
                    "frontier_size": len(members),
                    "members": members,
                    "original_frontier_id": original_id,
                    "original_frontier_size": len(original["members"]),
                    "skipped_physical_sessions": skipped_members,
                    "empty_session_policy": EMPTY_SESSION_POLICY,
                }
            )

    retained = [spec for group in groups for spec in group["members"]]
    if retained != [spec for spec in specs if spec["turns"] != []]:
        raise RuntimeError("clean frontier plan did not preserve nonempty source order")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "empty_session_policy": EMPTY_SESSION_POLICY,
        "physical_frontier_size": source.FRONTIER_SIZE,
        "physical_sessions": len(specs),
        "nonempty_physical_sessions": len(retained),
        "skipped_empty_sessions": len(skipped),
        "original_frontiers": len(original_groups),
        "retained_frontiers": len(groups),
        "skipped_all_empty_frontiers": len(original_groups) - len(groups),
        "skipped_physical_sessions": skipped,
        "frontier_mapping": mapping,
        "source_positions_preserved": True,
        "original_batch_boundaries_preserved": True,
        "group_sha256": gm.canonical_sha256(groups),
    }
    audit["audit_sha256"] = gm.canonical_sha256(audit)
    return groups, audit


class CleanWriterAdapter:
    """Module-compatible delegate without mutating frozen module globals."""

    __file__ = __file__

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.frozen_adapter = _delegate(variant)
        self.FRONTIER_SIZE = self.frozen_adapter.FRONTIER_SIZE
        self.SCHEMA_VERSION = f"{SCHEMA_VERSION}.{variant}"
        self.EMPTY_SESSION_POLICY = EMPTY_SESSION_POLICY

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.frozen_adapter, name)
        except AttributeError:
            # B1 does not re-export the base Writer token/attempt constants.
            return getattr(frozen_base, name)

    def audit_and_group(
        self, ordered_specs: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return audit_and_group(ordered_specs, self.variant)

    def group_session_specs(
        self,
        ordered_specs: Sequence[Mapping[str, Any]],
        frontier_size: int | None = None,
    ) -> list[dict[str, Any]]:
        if frontier_size is not None and (
            isinstance(frontier_size, bool)
            or not isinstance(frontier_size, int)
            or frontier_size != self.FRONTIER_SIZE
        ):
            raise ValueError(f"{self.variant} frontier_size is frozen at {self.FRONTIER_SIZE}")
        return self.audit_and_group(ordered_specs)[0]

    def _validate_frontier(self, frontier: Mapping[str, Any]) -> None:
        members = list(frontier.get("members", []))
        if not 1 <= len(members) <= self.FRONTIER_SIZE:
            raise ValueError("clean frontier has an invalid nonempty member count")
        if frontier.get("frontier_size", len(members)) != len(members):
            raise ValueError("clean frontier_size does not match its member count")
        for member in members:
            if not isinstance(member.get("turns"), list) or member["turns"] == []:
                raise ValueError("empty or malformed session must not reach Writer")

    def render_lme_frontier(self, frontier: Mapping[str, Any]) -> str:
        self._validate_frontier(frontier)
        return self.frozen_adapter.render_lme_frontier(frontier)

    def write_one_frontier(self, **kwargs: Any) -> dict[str, Any]:
        frontier = kwargs["frontier"]
        self._validate_frontier(frontier)
        packet = self.frozen_adapter.write_one_frontier(**kwargs)
        packet.update(
            {
                "original_frontier_id": int(
                    frontier.get("original_frontier_id", frontier["frontier_id"])
                ),
                "original_frontier_size": int(
                    frontier.get("original_frontier_size", len(frontier["members"]))
                ),
                "skipped_physical_sessions": list(frontier.get("skipped_physical_sessions", [])),
                "empty_session_policy": EMPTY_SESSION_POLICY,
            }
        )
        return packet

    def build_frontier_graph_from_packets(
        self,
        packets: Sequence[Mapping[str, Any]],
        embed_fn: Any,
        *,
        contract: Any,
        display_text_fn: Any = None,
    ) -> dict[str, Any]:
        materialized = list(packets)
        for packet in materialized:
            physical = list(packet.get("physical_sessions", []))
            if not 1 <= len(physical) <= self.FRONTIER_SIZE:
                raise gm.GraphMemoryError("clean packet has an invalid physical-session count")
            for member in physical:
                count = member.get("turn_count")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise gm.GraphMemoryError("empty or unauditable session cannot enter clean graph")

        # The all-empty case is a true no-op, including the embedding transport.
        def nonempty_embed(texts: Any) -> Any:
            return embed_fn(texts) if texts else []

        graph = self.frozen_adapter.build_frontier_graph_from_packets(
            materialized,
            nonempty_embed,
            contract=contract,
            display_text_fn=display_text_fn,
        )
        audit_rows = []
        for normalized, packet in zip(graph["session_packets"], materialized, strict=True):
            clean_metadata = {
                "original_frontier_id": int(packet.get("original_frontier_id", packet["frontier_id"])),
                "original_frontier_size": int(
                    packet.get("original_frontier_size", len(packet["physical_sessions"]))
                ),
                "skipped_physical_sessions": list(packet.get("skipped_physical_sessions", [])),
                "empty_session_policy": EMPTY_SESSION_POLICY,
            }
            normalized.update(clean_metadata)
            audit_rows.append({"frontier_id": int(packet["frontier_id"]), **clean_metadata})
        graph["adapter_schema_version"] = self.SCHEMA_VERSION
        graph["architecture"]["empty_session_policy"] = EMPTY_SESSION_POLICY
        graph["clean_writer_audit"] = {
            "variant": self.variant,
            "empty_session_policy": EMPTY_SESSION_POLICY,
            "retained_frontier_mapping": audit_rows,
            "all_empty_frontier_audit_location": "question_input_audit_not_graph",
        }
        graph["content_fingerprint"] = gm.canonical_sha256(
            {
                "frozen_graph_content_fingerprint": graph["content_fingerprint"],
                "clean_writer_audit": graph["clean_writer_audit"],
            }
        )
        return graph


def make_adapter(variant: str) -> CleanWriterAdapter:
    return CleanWriterAdapter(variant)


__all__ = ["CleanWriterAdapter", "EMPTY_SESSION_POLICY", "SCHEMA_VERSION", "audit_and_group", "make_adapter"]

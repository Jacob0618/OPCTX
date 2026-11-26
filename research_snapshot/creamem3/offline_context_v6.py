"""Deterministic, retrieval-frozen Offline context filters for CreaMem3 v6.

The functions in this module operate on an already retrieved reader context.
They never search, rerank, rewrite facts, or call a model.  The original
``retrieval_trace`` supplies the final selected fact IDs and the rendering
order; the frozen graph supplies view membership and session-node metadata.

Every public transform returns ``(text, audit)``.  The audit contains input,
output, and policy hashes together with the exact IDs retained or removed.
The default strict mode fails closed when the trace cannot be aligned with the
rendered text, preventing a context ablation from silently changing the wrong
line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


_EVENT_PREFIX = "[Cross-session event view |"
_FACT_PREFIX = "  - ["
_SESSION_LINE_RE = re.compile(
    r"^(?P<header>\[Session summary \| session (?P<session_id>\d+) \| [^\]\r\n]*\])"
    r"(?P<body>[ \t]+[^\r\n]*)?(?P<eol>\r\n|\n|\r)?$"
)


class OfflineContextTransformError(ValueError):
    """Raised when frozen metadata cannot be aligned with rendered context."""


@dataclass(frozen=True)
class OfflineContextPolicy:
    """Policy for one deterministic reader-context transformation.

    ``view_policy`` is one of ``all``, ``none``, ``support_min``, or ``cap``.
    ``session_summary_policy`` is either ``all`` or ``top10_header_only``.
    Public convenience wrappers below each change only one of these axes.
    """

    view_policy: str = "all"
    min_view_support: int = 2
    max_event_views: int = 2
    session_summary_policy: str = "all"
    top_summary_k: int = 10

    def validate(self) -> None:
        if self.view_policy not in {"all", "none", "support_min", "cap"}:
            raise ValueError(f"unsupported view_policy: {self.view_policy}")
        if self.session_summary_policy not in {"all", "top10_header_only"}:
            raise ValueError(
                "unsupported session_summary_policy: "
                f"{self.session_summary_policy}"
            )
        if self.min_view_support < 0:
            raise ValueError("min_view_support must be non-negative")
        if self.max_event_views < 0:
            raise ValueError("max_event_views must be non-negative")
        if self.top_summary_k < 1:
            raise ValueError("top_summary_k must be positive")

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return _sha256_text(payload)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _trace_payload(trace_or_row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = trace_or_row.get("retrieval_trace")
    return nested if isinstance(nested, Mapping) else trace_or_row


def _summary_nodes(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nodes = graph.get("summary_nodes", [])
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise OfflineContextTransformError("graph.summary_nodes must be a list")
    return [node for node in nodes if isinstance(node, Mapping)]


def _graph_indexes(
    graph: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, set[str]],
    dict[int, str],
    dict[str, str],
]:
    nodes = _summary_nodes(graph)
    by_id = {str(node.get("id", "")): node for node in nodes if node.get("id")}

    view_members: dict[str, set[str]] = {}
    event_line_to_id: dict[str, str] = {}
    session_id_to_summary_id: dict[int, str] = {}
    for node_id, node in by_id.items():
        node_type = str(node.get("node_type", ""))
        if node_type == "event_view":
            members = node.get("member_fact_ids", [])
            if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
                view_members[node_id] = {str(value) for value in members}
            else:
                view_members[node_id] = set()
            rendered = (
                f"[Cross-session event view | {node.get('view_type', '')} | sessions "
                + ", ".join(str(value) for value in node.get("session_ids", []))
                + f"] {node.get('text', '')}"
            )
            if rendered not in event_line_to_id:
                event_line_to_id[rendered] = node_id
            else:
                # An empty marker makes an ambiguous text fallback fail closed.
                event_line_to_id[rendered] = ""
        elif node_type == "session_packet":
            session_ids = node.get("session_ids", [])
            if isinstance(session_ids, Sequence) and not isinstance(
                session_ids, (str, bytes)
            ) and session_ids:
                session_id_to_summary_id[int(session_ids[0])] = node_id

    edges = graph.get("edges", [])
    if isinstance(edges, Sequence) and not isinstance(edges, (str, bytes)):
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            source_id = str(edge.get("source_id", edge.get("source", "")))
            target_id = str(edge.get("target_id", edge.get("target", "")))
            if source_id in view_members and target_id:
                edge_type = str(edge.get("edge_type", ""))
                if edge_type in {"VIEW_SUPPORTS", "SUPPORTS", ""}:
                    view_members[source_id].add(target_id)

    return by_id, view_members, session_id_to_summary_id, event_line_to_id


def _selected_fact_ids(trace: Mapping[str, Any]) -> list[str]:
    rerank = trace.get("rerank", {})
    if isinstance(rerank, Mapping):
        values = rerank.get("selected_fact_ids", [])
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            selected = _ordered_unique(str(value) for value in values)
            if selected:
                return selected
    assembly = trace.get("assembly", {})
    if isinstance(assembly, Mapping):
        values = assembly.get("rendered_fact_ids", [])
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return _ordered_unique(str(value) for value in values)
    return []


def _assembly_ids(trace: Mapping[str, Any], key: str) -> list[str]:
    assembly = trace.get("assembly", {})
    if not isinstance(assembly, Mapping):
        return []
    values = assembly.get(key, [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value) for value in values]


def _top_summary_ids(trace: Mapping[str, Any], top_k: int) -> set[str]:
    items = trace.get("top_summaries", [])
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return set()
    ranked: list[tuple[int, int, str]] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, Mapping) or not item.get("summary_id"):
            continue
        raw_rank = item.get("rank", position)
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            rank = position
        ranked.append((rank, position, str(item["summary_id"])))
    ranked.sort(key=lambda value: (value[0], value[1]))
    return {summary_id for rank, _, summary_id in ranked if rank <= top_k}


def _line_without_eol(line: str) -> str:
    return line[:-2] if line.endswith("\r\n") else line[:-1] if line.endswith(("\n", "\r")) else line


def _event_line_positions(lines: Sequence[str]) -> list[int]:
    return [
        index
        for index, line in enumerate(lines)
        if _line_without_eol(line).startswith(_EVENT_PREFIX)
    ]


def _session_line_positions(lines: Sequence[str]) -> list[tuple[int, re.Match[str]]]:
    result: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = _SESSION_LINE_RE.match(line)
        if match:
            result.append((index, match))
    return result


def _resolve_rendered_view_ids(
    lines: Sequence[str],
    positions: Sequence[int],
    trace: Mapping[str, Any],
    event_line_to_id: Mapping[str, str],
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    trace_ids = _assembly_ids(trace, "rendered_event_view_ids")
    if len(trace_ids) == len(positions):
        graph_ids = [
            event_line_to_id.get(_line_without_eol(lines[position]), "")
            for position in positions
        ]
        if graph_ids and all(graph_ids) and graph_ids != trace_ids:
            message = (
                "event-view order mismatch between rendered text and trace: "
                f"text={graph_ids}, trace={trace_ids}"
            )
            if strict:
                raise OfflineContextTransformError(message)
            warnings.append(message)
        return trace_ids, warnings

    fallback: list[str] = []
    for position in positions:
        line = _line_without_eol(lines[position])
        node_id = event_line_to_id.get(line, "")
        if not node_id:
            fallback = []
            break
        fallback.append(node_id)
    if positions and len(fallback) == len(positions):
        warnings.append(
            "rendered_event_view_ids did not align; IDs recovered from exact graph text"
        )
        return fallback, warnings

    message = (
        "event-view line/trace mismatch: "
        f"text={len(positions)}, trace={len(trace_ids)}"
    )
    if strict:
        raise OfflineContextTransformError(message)
    warnings.append(message)
    # Positional placeholders allow policies that do not require graph support
    # (none/cap) to proceed in explicitly non-strict diagnostic mode.
    return [f"__unresolved_view_{index}" for index in range(len(positions))], warnings


def _resolve_rendered_session_ids(
    session_lines: Sequence[tuple[int, re.Match[str]]],
    trace: Mapping[str, Any],
    session_id_to_summary_id: Mapping[int, str],
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    trace_ids = _assembly_ids(trace, "rendered_session_summary_ids")
    if len(trace_ids) == len(session_lines):
        graph_ids = [
            session_id_to_summary_id.get(int(match.group("session_id")), "")
            for _, match in session_lines
        ]
        if graph_ids and all(graph_ids) and graph_ids != trace_ids:
            message = (
                "session-summary order mismatch between rendered text and trace: "
                f"text={graph_ids}, trace={trace_ids}"
            )
            if strict:
                raise OfflineContextTransformError(message)
            warnings.append(message)
        return trace_ids, warnings

    fallback = [
        session_id_to_summary_id.get(int(match.group("session_id")), "")
        for _, match in session_lines
    ]
    if session_lines and all(fallback) and len(fallback) == len(session_lines):
        warnings.append(
            "rendered_session_summary_ids did not align; IDs recovered from session headers"
        )
        return fallback, warnings

    message = (
        "session-summary line/trace mismatch: "
        f"text={len(session_lines)}, trace={len(trace_ids)}"
    )
    if strict:
        raise OfflineContextTransformError(message)
    warnings.append(message)
    return [
        node_id or f"__unresolved_session_{index}"
        for index, node_id in enumerate(fallback)
    ], warnings


def _hash_selected_lines(lines: Sequence[str], prefix: str) -> str:
    return _sha256_text("".join(line for line in lines if line.startswith(prefix)))


def transform_offline_context(
    context: str,
    graph: Mapping[str, Any],
    trace_or_row: Mapping[str, Any],
    policy: OfflineContextPolicy,
    *,
    strict: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Apply a deterministic policy to an existing frozen Offline context.

    Line endings and every non-target line are retained byte-for-byte at the
    Python-string level.  In particular, facts are never regenerated from the
    graph.  They are copied from ``context`` unchanged.
    """

    if not isinstance(context, str):
        raise TypeError("context must be a string")
    if not isinstance(graph, Mapping):
        raise TypeError("graph must be a mapping")
    if not isinstance(trace_or_row, Mapping):
        raise TypeError("trace_or_row must be a mapping")
    policy.validate()

    trace = _trace_payload(trace_or_row)
    _, view_members, session_id_to_summary_id, event_line_to_id = _graph_indexes(
        graph
    )
    selected_ids = _selected_fact_ids(trace)
    selected_set = set(selected_ids)

    lines = context.splitlines(keepends=True)
    event_positions = _event_line_positions(lines)
    session_lines = _session_line_positions(lines)
    rendered_view_ids, view_warnings = _resolve_rendered_view_ids(
        lines,
        event_positions,
        trace,
        event_line_to_id,
        strict=strict,
    )
    rendered_session_ids, session_warnings = _resolve_rendered_session_ids(
        session_lines,
        trace,
        session_id_to_summary_id,
        strict=strict,
    )

    support_counts = {
        view_id: len(selected_set.intersection(view_members.get(view_id, set())))
        for view_id in rendered_view_ids
    }
    if policy.view_policy == "all":
        kept_view_ids = list(rendered_view_ids)
    elif policy.view_policy == "none":
        kept_view_ids = []
    elif policy.view_policy == "support_min":
        missing = [view_id for view_id in rendered_view_ids if view_id not in view_members]
        if missing and strict:
            raise OfflineContextTransformError(
                "missing graph membership for event views: " + ", ".join(missing)
            )
        kept_view_ids = [
            view_id
            for view_id in rendered_view_ids
            if support_counts[view_id] >= policy.min_view_support
        ]
    else:  # cap
        kept_view_ids = rendered_view_ids[: policy.max_event_views]
    kept_view_set = set(kept_view_ids)

    top_summary_ids = _top_summary_ids(trace, policy.top_summary_k)
    redacted_session_ids: list[str] = []
    kept_session_body_ids: list[str] = []
    replacements: dict[int, str] = {}
    for (line_index, match), summary_id in zip(session_lines, rendered_session_ids):
        if (
            policy.session_summary_policy == "top10_header_only"
            and summary_id not in top_summary_ids
        ):
            replacements[line_index] = match.group("header") + (match.group("eol") or "")
            redacted_session_ids.append(summary_id)
        else:
            kept_session_body_ids.append(summary_id)

    event_position_to_id = dict(zip(event_positions, rendered_view_ids))
    output_lines: list[str] = []
    for index, line in enumerate(lines):
        view_id = event_position_to_id.get(index)
        if view_id is not None and view_id not in kept_view_set:
            continue
        output_lines.append(replacements.get(index, line))
    output = "".join(output_lines)

    output_split = output.splitlines(keepends=True)
    non_event_before = "".join(
        line for line in lines if not _line_without_eol(line).startswith(_EVENT_PREFIX)
    )
    non_event_after = "".join(
        line
        for line in output_split
        if not _line_without_eol(line).startswith(_EVENT_PREFIX)
    )
    fact_hash_before = _hash_selected_lines(lines, _FACT_PREFIX)
    fact_hash_after = _hash_selected_lines(output_split, _FACT_PREFIX)

    audit: dict[str, Any] = {
        "schema_version": "creamem3.offline_context_transform.v1",
        "policy": asdict(policy),
        "policy_sha256": policy.sha256,
        "strict": strict,
        "input_sha256": _sha256_text(context),
        "output_sha256": _sha256_text(output),
        "input_chars": len(context),
        "output_chars": len(output),
        "removed_chars": len(context) - len(output),
        "changed": output != context,
        "selected_fact_ids": selected_ids,
        "selected_fact_count": len(selected_ids),
        "rendered_event_view_ids": rendered_view_ids,
        "kept_event_view_ids": kept_view_ids,
        "removed_event_view_ids": [
            view_id for view_id in rendered_view_ids if view_id not in kept_view_set
        ],
        "view_support_counts": support_counts,
        "rendered_event_view_count": len(rendered_view_ids),
        "kept_event_view_count": len(kept_view_ids),
        "rendered_session_summary_ids": rendered_session_ids,
        "kept_session_summary_body_ids": kept_session_body_ids,
        "redacted_session_summary_body_ids": redacted_session_ids,
        "top_summary_ids": sorted(top_summary_ids),
        "atomic_fact_lines_sha256_input": fact_hash_before,
        "atomic_fact_lines_sha256_output": fact_hash_after,
        "atomic_fact_lines_preserved": fact_hash_before == fact_hash_after,
        "non_event_text_sha256_input": _sha256_text(non_event_before),
        "non_event_text_sha256_output": _sha256_text(non_event_after),
        "non_event_text_preserved": non_event_before == non_event_after,
        "warnings": view_warnings + session_warnings,
    }
    return output, audit


def hide_all_event_views(
    context: str,
    graph: Mapping[str, Any],
    trace_or_row: Mapping[str, Any],
    *,
    strict: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Remove all rendered cross-session event-view lines only."""

    return transform_offline_context(
        context,
        graph,
        trace_or_row,
        OfflineContextPolicy(view_policy="none"),
        strict=strict,
    )


def keep_event_views_with_min_support(
    context: str,
    graph: Mapping[str, Any],
    trace_or_row: Mapping[str, Any],
    *,
    min_support: int = 2,
    strict: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Keep a rendered event view only if final selected facts support it."""

    return transform_offline_context(
        context,
        graph,
        trace_or_row,
        OfflineContextPolicy(
            view_policy="support_min", min_view_support=min_support
        ),
        strict=strict,
    )


def cap_event_views(
    context: str,
    graph: Mapping[str, Any],
    trace_or_row: Mapping[str, Any],
    *,
    max_views: int = 2,
    strict: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Keep only the first ``max_views`` event views in frozen render order."""

    return transform_offline_context(
        context,
        graph,
        trace_or_row,
        OfflineContextPolicy(view_policy="cap", max_event_views=max_views),
        strict=strict,
    )


def header_only_for_non_top10_session_summaries(
    context: str,
    graph: Mapping[str, Any],
    trace_or_row: Mapping[str, Any],
    *,
    top_k: int = 10,
    strict: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Strip only non-TopK session-summary bodies; retain their exact headers."""

    return transform_offline_context(
        context,
        graph,
        trace_or_row,
        OfflineContextPolicy(
            session_summary_policy="top10_header_only", top_summary_k=top_k
        ),
        strict=strict,
    )


__all__ = [
    "OfflineContextPolicy",
    "OfflineContextTransformError",
    "cap_event_views",
    "header_only_for_non_top10_session_summaries",
    "hide_all_event_views",
    "keep_event_views_with_min_support",
    "transform_offline_context",
]

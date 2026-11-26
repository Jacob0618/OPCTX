"""Isolated MEC-50 Gemini experiment with routing-only summaries.

This entry point intentionally reuses the frozen MEC-50 selector, canonical
G3.5 Online graph, original Gemini Reader/Judge prompts, parsers, and provider
runtimes from :mod:`run_memeye_mec50_direct_gemini_v1`.  The sole semantic
change is the Reader context renderer:

* selected fact IDs and their order are unchanged;
* every ``fact.text`` is rendered byte-for-byte unchanged;
* session packets and cross-session event views remain routing/index data but
  are not expanded into Reader context;
* one short structural header is rendered per selected session;
* Qwen is never constructed.

Fresh output/cache roots make the experiment reversible and prevent it from
modifying or reusing the completed direct-context v1 result contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import run_memeye_mec50_direct_gemini_v1 as base


SCHEMA_VERSION = "creamem3.memeye_mec50_routing_only_raw_gemini.v1.0"
CONTEXT_MODE = "routing_only_raw_facts"
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\mec50_routing_only_raw_gemini_v1_r1_results"
)
DEFAULT_CACHE_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\mec50_routing_only_raw_gemini_v1_r1_provider"
)


def _routing_only_assemble(
    selected: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Render short session headers plus immutable raw fact text only."""

    by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    session_order: list[int] = []
    for item in selected:
        fact = item["fact"]
        session_id = int(fact["session_id"])
        if session_id not in by_session:
            session_order.append(session_id)
        by_session[session_id].append(fact)

    lines: list[str] = []
    header_rows: list[dict[str, Any]] = []
    for session_id in session_order:
        facts = by_session[session_id]
        first = facts[0]
        source_session_id = str(first.get("source_session_id", f"S{session_id}"))
        timestamp = str(first.get("iso_date") or first.get("timestamp") or "")
        header = f"[Session {source_session_id} | {timestamp}]"
        lines.append(header)
        header_rows.append(
            {
                "session_id": session_id,
                "source_session_id": source_session_id,
                "timestamp": timestamp,
                "header": header,
            }
        )
        for fact in facts:
            # Do not normalize, strip, truncate, or otherwise modify fact.text.
            lines.append(f"  - {fact['text']}")

    context = "\n".join(lines)
    return context, {
        "context_mode": CONTEXT_MODE,
        "routing_only_summary_nodes": True,
        "raw_fact_text_unchanged": True,
        "rendered_event_view_ids": [],
        "rendered_session_summary_ids": [],
        "rendered_session_headers": header_rows,
        "rendered_fact_ids": [item["fact"]["id"] for item in selected],
        "rendered_chars": len(context),
    }


def _prepare_rows(
    canonical: Mapping[str, Any],
    selection_by_id: Mapping[str, Mapping[str, Any]],
    scoped_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    build = canonical["build"]
    graph_by_task: dict[str, dict[str, Any]] = {}
    fact_by_task: dict[str, dict[str, Mapping[str, Any]]] = {}
    prepared: list[dict[str, Any]] = []

    for query in scoped_rows:
        row_id = str(query["row_id"])
        task = str(query["task"])
        if task not in graph_by_task:
            artifact_path, _ = base.predict._artifact_for(  # type: ignore[attr-defined]
                build, base.BACKBONE, base.ARCHITECTURE, task
            )
            graph = base._read_object(artifact_path, f"sealed {task} Online graph")
            graph_by_task[task] = graph
            fact_by_task[task] = {
                str(fact["id"]): fact for fact in graph.get("fact_nodes", [])
            }

        selected_row = selection_by_id[row_id]
        selected_ids = list(selected_row["selected_fact_ids"])
        fact_by_id = fact_by_task[task]
        missing = [fact_id for fact_id in selected_ids if fact_id not in fact_by_id]
        if missing:
            raise base.MEC50HarnessError(
                f"selector references facts outside {task} graph: {row_id}:{missing[:3]}"
            )
        selected_items = [
            {"fact": fact_by_id[fact_id], "selector_rank": rank}
            for rank, fact_id in enumerate(selected_ids, start=1)
        ]
        context, assembly = _routing_only_assemble(selected_items)
        if assembly["rendered_fact_ids"] != selected_ids:
            raise base.MEC50HarnessError(
                f"routing-only assembler changed selector order: {row_id}"
            )

        selected_facts = [item["fact"] for item in selected_items]
        fact_lengths = sorted(len(str(fact["text"])) for fact in selected_facts)
        fact_chars = sum(fact_lengths)
        sessions = sorted({int(fact["session_id"]) for fact in selected_facts})
        source_ids = sorted(
            {
                str(source_id)
                for fact in selected_facts
                for source_id in fact.get("source_dia_ids", [])
            }
        )
        raw_fact_bindings = [
            {
                "fact_id": str(fact["id"]),
                "text_sha256": base._sha256(str(fact["text"])),
            }
            for fact in selected_facts
        ]
        prepared.append(
            {
                "query": dict(query),
                "selection": copy.deepcopy(dict(selected_row)),
                "context": context,
                "context_sha256": base._sha256(context),
                "assembly": assembly,
                "assembly_sha256": base._sha256(assembly),
                "stats": {
                    "context_mode": CONTEXT_MODE,
                    "selected_facts": len(selected_ids),
                    "selected_fact_chars": fact_chars,
                    "selected_fact_chars_mean": fact_chars / len(fact_lengths),
                    "selected_fact_chars_p50": fact_lengths[(len(fact_lengths) - 1) // 2],
                    "selected_fact_chars_p90": fact_lengths[
                        math.ceil(0.9 * len(fact_lengths)) - 1
                    ],
                    "selected_fact_chars_max": fact_lengths[-1],
                    "context_chars": len(context),
                    "assembly_overhead_chars": len(context) - fact_chars,
                    "session_summary_text_chars": 0,
                    "event_view_text_chars": 0,
                    "formatting_and_provenance_chars": len(context) - fact_chars,
                    "unique_sessions": len(sessions),
                    "session_ids": sessions,
                    "unique_source_rounds": len(source_ids),
                    "source_dia_ids": source_ids,
                    "rendered_session_summaries": 0,
                    "rendered_event_views": 0,
                    "rendered_session_headers": len(
                        assembly["rendered_session_headers"]
                    ),
                    "raw_fact_text_bindings_sha256": base._sha256(raw_fact_bindings),
                },
            }
        )
    return prepared


def _contract(
    *,
    output_root: Path,
    cache_root: Path,
    selection_receipt: Mapping[str, Any],
    canonical_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
    workers: int,
) -> dict[str, Any]:
    value = base._contract(
        output_root=output_root,
        cache_root=cache_root,
        selection_receipt=selection_receipt,
        canonical_receipt=canonical_receipt,
        authority_receipt=authority_receipt,
        prepared=prepared,
        workers=workers,
    )
    value.pop("contract_sha256", None)
    value["schema_version"] = SCHEMA_VERSION
    value["experiment_kind"] = (
        "offline_selector_direct50_routing_only_raw_facts_then_unchanged_gemini_reader_judge"
    )
    value["context_mode"] = {
        "name": CONTEXT_MODE,
        "selected_fact_ids_unchanged": True,
        "selected_fact_order_unchanged": True,
        "fact_text_unchanged": True,
        "session_summary_reader_rendered": False,
        "event_view_reader_rendered": False,
        "short_header_granularity": "one_per_selected_session",
    }
    value["contract_sha256"] = base._sha256(value)
    return value


def preflight(
    *,
    selection_path: Path,
    output_root: Path,
    cache_root: Path,
    offset: int = 0,
    limit: int | None = None,
    workers: int = base.DEFAULT_WORKERS,
) -> dict[str, Any]:
    output, cache = base._validate_roots(output_root, cache_root)
    if not 1 <= int(workers) <= base.MAX_WORKERS:
        raise base.MEC50HarnessError(
            f"workers must be in [1,{base.MAX_WORKERS}]"
        )
    canonical = base._canonical_sources()
    authority, authority_receipt = base._reader_authority()
    selection, selection_receipt = base._selection_rows(selection_path)
    scoped = base._scope_rows(
        canonical["query_rows"], selection, offset=int(offset), limit=limit
    )
    prepared = _prepare_rows(canonical, selection, scoped)
    contract = _contract(
        output_root=output,
        cache_root=cache,
        selection_receipt=selection_receipt,
        canonical_receipt=canonical["receipt"],
        authority_receipt=authority_receipt,
        prepared=prepared,
        workers=int(workers),
    )

    first = prepared[0]
    query = first["query"]
    if query["reader_route"] == "two_stage":
        sample_prompt = str(authority["focused"]).replace(
            "{question}", str(query["question"])
        ).replace("{context}", str(first["context"]))
    else:
        sample_prompt = str(authority["fused"]).format(
            question=str(query["question"]), context=str(first["context"])
        )
    qwen_paths = base._qwen_paths(output, cache)
    if qwen_paths:
        raise base.MEC50HarnessError(
            f"dedicated no-Qwen roots contain a Qwen namespace: {qwen_paths[:3]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "zero_network_preflight_passed",
        "decision": "approve",
        "context_mode": CONTEXT_MODE,
        "output_root": str(output),
        "cache_root": str(cache),
        "contract": contract,
        "prepared": prepared,
        "authority": authority,
        "length_summary": base._length_summary(prepared),
        "sample_reader_route": query["reader_route"],
        "sample_prompt_chars": len(sample_prompt),
        "qwen_paths": qwen_paths,
        "provider_calls": 0,
        "network_calls": 0,
        "files_written": 0,
    }


def audit(**kwargs: Any) -> dict[str, Any]:
    # base.audit resolves base.preflight dynamically.  Temporarily bind this
    # module's renderer so every reconstructed context is contract-identical.
    original = base.preflight
    base.preflight = preflight  # type: ignore[assignment]
    try:
        value = base.audit(**kwargs)
    finally:
        base.preflight = original  # type: ignore[assignment]
    value["schema_version"] = SCHEMA_VERSION
    value["context_mode"] = CONTEXT_MODE
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "predict", "score", "all", "audit"),
        default="preflight",
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=base.DEFAULT_WORKERS)
    parser.add_argument("--allow-model-calls", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    live = args.phase in {"predict", "score", "all"}
    if live != bool(args.allow_model_calls):
        raise SystemExit(
            "live phases require --allow-model-calls; zero-call phases reject it"
        )
    if args.smoke and (args.limit is not None or args.offset != 0):
        raise SystemExit("--smoke conflicts with --limit and --offset")
    limit = base.DEFAULT_SMOKE_ROWS if args.smoke else args.limit
    output_root = args.output_root
    cache_root = args.cache_root
    if args.smoke and Path(output_root).resolve() == DEFAULT_OUTPUT_ROOT:
        output_root = DEFAULT_OUTPUT_ROOT.with_name(
            DEFAULT_OUTPUT_ROOT.name + f"_smoke{base.DEFAULT_SMOKE_ROWS}"
        )
    if args.smoke and Path(cache_root).resolve() == DEFAULT_CACHE_ROOT:
        cache_root = DEFAULT_CACHE_ROOT.with_name(
            DEFAULT_CACHE_ROOT.name + f"_smoke{base.DEFAULT_SMOKE_ROWS}"
        )

    gate = preflight(
        selection_path=args.selection_manifest,
        output_root=output_root,
        cache_root=cache_root,
        offset=args.offset,
        limit=limit,
        workers=args.workers,
    )
    if args.phase == "preflight":
        result = base._public_gate(gate)
    elif args.phase == "audit":
        result = audit(
            selection_path=args.selection_manifest,
            output_root=output_root,
            cache_root=cache_root,
            offset=args.offset,
            limit=limit,
            workers=args.workers,
        )
    elif args.phase == "predict":
        result = base.run_predict(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            authority=gate["authority"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
    elif args.phase == "score":
        result = base.run_score(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
    else:
        predicted = base.run_predict(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            authority=gate["authority"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
        scored = base.run_score(
            output_root=Path(gate["output_root"]),
            cache_root=Path(gate["cache_root"]),
            contract=gate["contract"],
            prepared=gate["prepared"],
            workers=args.workers,
        )
        result = {"status": "all_complete", "predict": predicted, "score": scored}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Full371 fixed-MEC50 physical-observation order, offline preparation only.

Complete structured (task,source session,round) SET equality defines grouping.
No entity inference, multi-round splitting, caption expansion or new text.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from creamem3 import run_memeye_mec50_direct_gemini_v1 as direct
from creamem3 import run_memeye_mec50_routing_only_raw_gemini_v1 as raw

ROOT = Path(r"D:\creamem3_cache\memeye_v19\full_budget_20260907_r1")
SCOPE = ROOT / "observation_order"
PREREGISTRATION = ROOT / "observation_preregistration.json"
SELECTION = Path(r"D:\creamem3_cache\memeye_v19\mec50_soft_bundle_v2_r1\mec50_soft_bundle_v2_manifest.json")
SELECTION_SHA256 = "604deb42d0a737abe5cbb1bba744f2db4272fa45cc0fb867d7027b94e01821a6"
BASE_RESULTS = Path(r"D:\creamem3_cache\memeye_v19\mec50_soft_bundle_v2_routing_only_gemini_r1_results")
BASE_CACHE = Path(r"D:\creamem3_cache\memeye_v19\mec50_soft_bundle_v2_routing_only_gemini_r1_provider")
SCHEMA = "memeye.full371.physical_observation_order.v1"
MODE = "within_session_complete_physical_round_set_stable_grouping"


class ObservationContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ObservationContractError(message)


def checked_seal(path: Path, field: str) -> dict[str, Any]:
    value = direct._read_object(path, "observation-order bound input")
    body = dict(value)
    digest = body.pop(field, None)
    require(digest == direct._sha256(body), f"input seal drift: {path.name}")
    return value


def source_index(source_map: Mapping[str, Any], *, expected_task: str | None = None) -> dict[str, tuple[str, str, str]]:
    """Only explicit source-map fields; no parsing of IDs, paths, or fact text."""
    require(isinstance(source_map, Mapping) and bool(source_map), "missing structured source map")
    index, dates = {}, {}
    for source_id, meta in source_map.items():
        require(isinstance(source_id, str) and bool(source_id) and isinstance(meta, Mapping), "invalid source mapping")
        require(all(isinstance(meta.get(k), str) and bool(meta[k]) for k in ("task", "session_id", "round")),
                "source lacks explicit task/session/physical-round identity")
        require(expected_task is None or meta["task"] == expected_task, "source-map task drift")
        require(meta.get("source_kind") in {"user", "assistant", "image_caption"}, "unknown or empty source kind")
        require(isinstance(meta.get("date"), str), "source date must be explicit string")
        if meta["source_kind"] == "image_caption":
            require(isinstance(meta.get("image"), str) and bool(meta["image"]), "caption lacks exact structured image path")
            require(isinstance(meta.get("image_id"), str), "caption image_id must be an explicit string (may be empty)")
        key = (meta["task"], meta["session_id"], meta["round"])
        require(key not in dates or dates[key] == meta["date"], "physical round has conflicting dates")
        dates[key] = meta["date"]
        index[source_id] = key
    return index


def observation_signature(fact: Mapping[str, Any], index: Mapping[str, tuple[str, str, str]]) -> tuple[tuple[str, str, str], ...]:
    sources = fact.get("source_dia_ids")
    require(isinstance(sources, list) and bool(sources)
            and all(isinstance(s, str) and bool(s) and s in index for s in sources), "unknown/empty fact sources")
    return tuple(sorted({index[source_id] for source_id in sources}))


def assemble(selected: Sequence[Mapping[str, Any]], source_map: Mapping[str, Any], *, expected_task: str | None = None):
    ids = [str(item["fact"]["id"]) for item in selected]
    require(bool(ids) and len(ids) <= 50 and len(ids) == len(set(ids)), "invalid fixed selected fact IDs")
    index = source_index(source_map, expected_task=expected_task)
    baseline, baseline_receipt = raw._routing_only_assemble(selected)
    sessions = {}
    for item in selected:
        sessions.setdefault(int(item["fact"]["session_id"]), []).append(item)
    reordered, baseline_order, audit_groups = [], [], []
    for session, items in sessions.items():
        groups = {}
        baseline_order.extend(str(item["fact"]["id"]) for item in items)
        for item in items:
            key = observation_signature(item["fact"], index)
            groups.setdefault(key, []).append(item)
        for signature, members in groups.items():
            reordered.extend(members)
            audit_groups.append({"stored_session_id": session, "observation_signature": [list(v) for v in signature],
                "fact_ids": [str(m["fact"]["id"]) for m in members],
                "distinct_full_source_signatures": len({tuple(sorted(set(m["fact"]["source_dia_ids"]))) for m in members}),
                "multi_round_group": len(signature) > 1})
    context, receipt = raw._routing_only_assemble(reordered)
    body_ids = [str(item["fact"]["id"]) for item in reordered]
    require(Counter(body_ids) == Counter(ids), "selected fact census changed")
    require(len(context) == len(baseline), "context character length changed")
    require(Counter(context.split("\n")) == Counter(baseline.split("\n")), "complete raw line multiset changed")
    require(receipt["rendered_session_headers"] == baseline_receipt["rendered_session_headers"], "original headers changed")
    return context, {"schema": SCHEMA, "context_mode": MODE, "selected_fact_ids_input_order": ids,
        "baseline_body_fact_ids": baseline_order, "rendered_fact_ids": body_ids,
        "body_order_changed": body_ids != baseline_order, "observation_groups": audit_groups,
        "bridge_groups": sum(g["distinct_full_source_signatures"] > 1 for g in audit_groups),
        "multi_round_selected_facts": sum(len(observation_signature(i["fact"], index)) > 1 for i in selected),
        "source_map_sha256": direct._sha256(source_map),
        "baseline_context_sha256": direct._sha256(baseline), "context_sha256": direct._sha256(context),
        "rendered_chars": len(context), "raw_fact_text_unchanged": True,
        "selected_fact_input_ranks_unchanged": True, "exact_characters_unchanged": True,
        "complete_line_multiset_unchanged": True, "multi_round_facts_not_split_or_duplicated": True,
        "rendered_session_headers": receipt["rendered_session_headers"],
        "rendered_session_summary_ids": [], "rendered_event_view_ids": [],
        "semantic_entity_or_complete_proof_claimed": False}


def prepare_rows(canonical, selections, scope, *, baseline_prepared=None):
    prepared = raw._prepare_rows(canonical, selections, scope) if baseline_prepared is None else baseline_prepared
    graphs, graph_receipts = {}, {}
    for row in prepared:
        task = str(row["query"]["task"])
        if task not in graphs:
            path, digest = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
            graphs[task] = direct._read_object(path, "frozen G35 Online graph")
            graph_receipts[task] = {"path": str(path), "file_sha256": digest}
        graph = graphs[task]
        by_id = {str(f["id"]): f for f in graph["fact_nodes"]}
        items = [{"fact": by_id[fid]} for fid in row["selection"]["selected_fact_ids"]]
        context, assembly = assemble(items, graph["memeye_source_map"], expected_task=task)
        require(assembly["baseline_context_sha256"] == row["context_sha256"], "old context independent reconstruction drift")
        row.update(context=context, context_sha256=direct._sha256(context), assembly=assembly,
                   assembly_sha256=direct._sha256(assembly))
        row["stats"].update(context_mode=MODE, body_order_changed=assembly["body_order_changed"],
                             physical_observation_groups=len(assembly["observation_groups"]),
                             bridge_groups=assembly["bridge_groups"])
    return prepared, graph_receipts


def contract(**kwargs):
    value = direct._contract(**kwargs)
    value.pop("contract_sha256", None)
    value.update(schema_version=SCHEMA, experiment_kind="fixed_MEC50_full371_physical_observation_order_only",
                 renderer_file_sha256=direct._file_sha256(Path(__file__)),
                 legacy_raw_renderer_file_sha256=direct._file_sha256(Path(raw.__file__)),
                 legacy_direct_runner_file_sha256=direct._file_sha256(Path(direct.__file__)),
                 context_mode={"name": MODE, "same_selected_ids_input_ranks_facts_and_chars": True,
                               "no_new_text": True, "group_identity": "entire structured(task,session_id,round) set",
                               "stable_group_order": "first original rank", "within_group_order": "original rank",
                               "no_inferred_entity_or_semantic_proof": True})
    value["contract_sha256"] = direct._sha256(value)
    return value


def preflight(*, workers=4):
    require(direct._file_sha256(SELECTION) == SELECTION_SHA256, "frozen MEC50 full selection drift")
    output, cache = direct._validate_roots(SCOPE / "results", SCOPE / "provider_cache")
    require(isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= direct.MAX_WORKERS, "invalid worker count")
    canonical = direct._canonical_sources()
    authority, authority_receipt = direct._reader_authority()
    selections, selection_receipt = direct._selection_rows(SELECTION)
    scope = direct._scope_rows(canonical["query_rows"], selections, offset=0, limit=None)
    require(len(scope) == len(selections) == 371, "full371 panel drift")
    baseline = raw._prepare_rows(canonical, selections, scope)
    old_contract = checked_seal(BASE_RESULTS / "contract.json", "contract_sha256")
    reconstructed = raw._contract(output_root=BASE_RESULTS, cache_root=BASE_CACHE,
        selection_receipt=selection_receipt, canonical_receipt=canonical["receipt"],
        authority_receipt=authority_receipt, prepared=baseline, workers=workers)
    require(old_contract == reconstructed, "historical baseline Reader/input/render contract drift")
    pcomplete, predictions = direct._load_prediction_complete(BASE_RESULTS, old_contract, baseline)
    score_complete = checked_seal(BASE_RESULTS / "score" / "complete.json", "complete_sha256")
    require(score_complete["complete"] is True and len(score_complete["rows"]) == 371
            and score_complete["contract_sha256"] == old_contract["contract_sha256"], "historical baseline completion drift")
    for item, pred in zip(baseline, predictions, strict=True):
        require(item["selection"]["selected_fact_ids"] == pred["selected_fact_ids"]
                and item["context_sha256"] == pred["context_sha256"], "not exact old selected facts/context")
    prepared, graph_receipts = prepare_rows(canonical, selections, scope, baseline_prepared=baseline)
    new_contract = contract(output_root=output, cache_root=cache, selection_receipt=selection_receipt,
        canonical_receipt=canonical["receipt"], authority_receipt=authority_receipt, prepared=prepared, workers=workers)
    require(not direct._qwen_paths(output, cache), "unexpected Qwen namespace")
    baseline_receipt = {"output_root": str(BASE_RESULTS), "cache_root": str(BASE_CACHE),
        "contract_sha256": old_contract["contract_sha256"],
        "contract_file_sha256": direct._file_sha256(BASE_RESULTS / "contract.json"),
        "prediction_complete_file_sha256": direct._file_sha256(BASE_RESULTS / "predict" / "complete.json"),
        "score_complete_file_sha256": direct._file_sha256(BASE_RESULTS / "score" / "complete.json"),
        "rows": 371, "mapped_percent": score_complete["mapped_percent"], "raw_mean": score_complete["raw_mean"],
        "comparison_is_asynchronous_historical": True, "gold_used_for_selection_or_order": False}
    return {"status": "zero_network_full371_preflight_passed", "contract": new_contract,
        "prepared": prepared, "authority": authority, "graph_receipts": graph_receipts,
        "baseline_receipt": baseline_receipt, "output_root": str(output), "cache_root": str(cache),
        "length_summary": direct._length_summary(prepared), "network_calls": 0}


def freeze_full(*, workers=4):
    gate = preflight(workers=workers)
    prereg = {"schema": SCHEMA, "method_file_sha256": direct._file_sha256(Path(__file__)),
        "selection_file_sha256": SELECTION_SHA256, "baseline": gate["baseline_receipt"],
        "graph_receipts": gate["graph_receipts"], "contract_sha256": gate["contract"]["contract_sha256"],
        "scope": "all371_original_order_no_topic_or_gold_filter", "only_change": MODE,
        "not_semantic_entity_binding_or_verified_proof": True, "network_calls": 0}
    prereg["preregistration_sha256"] = direct._sha256(prereg)
    direct._write_once(PREREGISTRATION, prereg)
    rows = [{"row_id": r["query"]["row_id"], "selected_fact_ids": r["selection"]["selected_fact_ids"],
        "context": r["context"], "context_sha256": r["context_sha256"],
        "assembly": r["assembly"], "assembly_sha256": r["assembly_sha256"]} for r in gate["prepared"]]
    artifact = {"contract": gate["contract"], "rows": rows, "rows_sha256": direct._sha256(rows),
                "preregistration_sha256": prereg["preregistration_sha256"]}
    artifact["artifact_sha256"] = direct._sha256(artifact)
    direct._write_once(SCOPE / "context_freeze" / "contexts.json", artifact)
    summary = {"status": "offline_full371_contexts_sealed", "rows": 371,
        "body_order_changed": sum(r["assembly"]["body_order_changed"] for r in gate["prepared"]),
        "rows_with_bridge_groups": sum(r["assembly"]["bridge_groups"] > 0 for r in gate["prepared"]),
        "bridge_group_occurrences": sum(r["assembly"]["bridge_groups"] for r in gate["prepared"]),
        "multi_round_fact_occurrences_preserved_whole": sum(r["assembly"]["multi_round_selected_facts"] for r in gate["prepared"]),
        "selected_counts": dict(Counter(len(r["selection"]["selected_fact_ids"]) for r in gate["prepared"])),
        "all_ids_input_ranks_rawtext_line_multisets_and_chars_unchanged": True,
        "length_summary": gate["length_summary"], "baseline": gate["baseline_receipt"],
        "contract_sha256": gate["contract"]["contract_sha256"], "artifact_sha256": artifact["artifact_sha256"],
        "network_calls": 0, "gold_used_for_selection_or_order": False, "scope": str(SCOPE)}
    direct._write_once(SCOPE / "context_freeze" / "complete.json", summary)
    return summary


def frozen_gate(*, workers=4):
    prereg = checked_seal(PREREGISTRATION, "preregistration_sha256")
    require(prereg["method_file_sha256"] == direct._file_sha256(Path(__file__)), "frozen observation method drift")
    artifact = checked_seal(SCOPE / "context_freeze" / "contexts.json", "artifact_sha256")
    require(artifact["preregistration_sha256"] == prereg["preregistration_sha256"]
            and artifact["rows_sha256"] == direct._sha256(artifact["rows"]), "frozen context binding drift")
    gate = preflight(workers=workers)
    require(gate["contract"] == artifact["contract"] and gate["graph_receipts"] == prereg["graph_receipts"]
            and gate["baseline_receipt"] == prereg["baseline"], "frozen experiment provenance drift")
    require(len(gate["prepared"]) == len(artifact["rows"]) == 371, "frozen scope count drift")
    for row, frozen in zip(gate["prepared"], artifact["rows"], strict=True):
        require(row["query"]["row_id"] == frozen["row_id"] and row["context"] == frozen["context"]
                and row["assembly"] == frozen["assembly"] and row["selection"]["selected_fact_ids"] == frozen["selected_fact_ids"],
                "frozen rendered context/selected IDs drift")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="write new frozen contexts, never call models")
    args = parser.parse_args()
    if args.freeze:
        result = freeze_full()
    else:
        gate = preflight()
        result = {k: v for k, v in gate.items() if k not in {"prepared", "authority"}}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

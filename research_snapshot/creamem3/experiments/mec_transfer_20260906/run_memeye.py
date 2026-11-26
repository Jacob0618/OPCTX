"""Score-blind MemEye adapter for a fixed 24-question, no-reranker pilot.

Only ``prepare`` runs the offline selector. Prediction/scoring use the unchanged
legacy raw-fact renderer and Gemini runtimes in dedicated arm directories.
No evaluator data is read before all arm manifests have been sealed.
"""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2
from creamem3 import memeye_mec50_offline_v1 as legacy_selector
from creamem3 import run_memeye_mec50_direct_gemini_v1 as direct
from creamem3 import run_memeye_mec50_routing_only_raw_gemini_v1 as raw

SCHEMA = "creamem3.memeye_budget_transfer_adapter.v1"
PILOT_PREFIX = "mec-transfer-20260906:"
PILOT_SIZE = 24
MAX_FACTS = 50
ARMS = ("global_dense50", "budget_evidence50", "v2_historical")
LIVE_ARMS = ARMS[:2]
DEFAULT_ROOT = Path(r"D:\creamem3_cache\memeye_v19\mec_transfer_20260906_r2")
V2_MANIFEST = Path(r"D:\creamem3_cache\memeye_v19\mec50_soft_bundle_v2_r1\mec50_soft_bundle_v2_manifest.json")
FORBIDDEN = frozenset({"gold", "answer", "clue", "point", "x", "y", "category", "score", "correct"})


class AdapterError(RuntimeError):
    pass


def pilot_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Question IDs are the sole sampling input; no task/type stratification."""
    if len(rows) < PILOT_SIZE:
        raise AdapterError("fewer than 24 source questions")
    ids = [str(row["row_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise AdapterError("duplicate source question IDs")
    if any(FORBIDDEN.intersection(str(key).lower() for key in row) for row in rows):
        raise AdapterError("evaluator-only source query fields")
    return [dict(row) for row in sorted(rows, key=lambda row: (
        direct._sha256(PILOT_PREFIX + str(row["row_id"])), str(row["row_id"])
    ))[:PILOT_SIZE]]


def session_header(fact: Mapping[str, Any]) -> str:
    session_id = int(fact["session_id"])
    source = str(fact.get("source_session_id", f"S{session_id}"))
    timestamp = str(fact.get("iso_date") or fact.get("timestamp") or "")
    return f"[Session {source} | {timestamp}]"


def adapt_facts(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Preserve exact source IDs (not round-only identity) and render costs."""
    facts: list[dict[str, Any]] = []
    headers: dict[str, str] = {}
    ids: set[str] = set()
    source_map = graph.get("memeye_source_map", {})
    for raw_fact in graph.get("fact_nodes", []):
        fact_id = str(raw_fact["id"])
        if not fact_id or fact_id in ids or not isinstance(raw_fact.get("text"), str):
            raise AdapterError("invalid or duplicated source fact")
        ids.add(fact_id)
        group = str(int(raw_fact["session_id"]))
        header = session_header(raw_fact)
        if group in headers and headers[group] != header:
            raise AdapterError("same-session headers differ: additive cost would be inexact")
        headers[group] = header
        sources: set[str] = set()
        for dia_id in raw_fact.get("source_dia_ids", []):
            source = source_map.get(str(dia_id))
            if not isinstance(source, Mapping) or not isinstance(source.get("round"), str) or not source["round"]:
                raise AdapterError("source physical round missing from structured source map")
            sources.add(str(dia_id))
        if not sources:
            raise AdapterError("fact lacks structured source-round provenance")
        facts.append({"id": fact_id, "text": raw_fact["text"], "sources": sorted(sources),
                      "group": group, "cost": len(raw_fact["text"]) + 5})
    if not facts:
        raise AdapterError("empty fact pool")
    return facts, {group: len(header) + 1 for group, header in headers.items()}


def render_selection(graph: Mapping[str, Any], selected_ids: Sequence[str]) -> tuple[str, dict[str, Any]]:
    by_id = {str(fact["id"]): fact for fact in graph["fact_nodes"]}
    selected = list(selected_ids)
    if (not selected or len(selected) > MAX_FACTS or len(set(selected)) != len(selected)
            or any(fact_id not in by_id for fact_id in selected)):
        raise AdapterError("invalid selected IDs")
    text, receipt = raw._routing_only_assemble([{"fact": by_id[fact_id]} for fact_id in selected])
    # Legacy receipt lists input ranks, while the body groups sessions. Expose
    # the actual body order honestly without changing the frozen renderer.
    groups = list(dict.fromkeys(int(by_id[fact_id]["session_id"]) for fact_id in selected))
    body_ids = [fact_id for group in groups for fact_id in selected
                if int(by_id[fact_id]["session_id"]) == group]
    return text, {"context_sha256": direct._sha256(text), "context_chars": len(text),
                  "input_selected_ids": selected, "actual_body_fact_ids": body_ids,
                  "body_regrouped_by_session": body_ids != selected,
                  "raw_fact_text_unchanged": receipt["raw_fact_text_unchanged"],
                  "rendered_event_view_ids": receipt["rendered_event_view_ids"],
                  "rendered_session_summary_ids": receipt["rendered_session_summary_ids"]}


def additive_cost(facts: Sequence[Mapping[str, Any]], groups: Mapping[str, int], ids: Sequence[str]) -> int:
    if not ids:
        return 0
    by_id = {str(fact["id"]): fact for fact in facts}
    return (sum(int(by_id[fact_id]["cost"]) for fact_id in ids)
            + sum(int(groups[group]) for group in {str(by_id[fact_id]["group"]) for fact_id in ids}) - 1)


def select_row(*, graph: Mapping[str, Any], question: str, dense_scores: Sequence[float],
               bm25_scores: Sequence[float], selector: Callable[..., Mapping[str, Any]],
               time_limit: float = 5.0) -> dict[str, Any]:
    facts, group_costs = adapt_facts(graph)
    if len(dense_scores) != len(facts) or len(bm25_scores) != len(facts):
        raise AdapterError("score vector / fact order mismatch")
    if not all(math.isfinite(float(value)) for value in [*dense_scores, *bm25_scores]):
        raise AdapterError("non-finite retrieval scores")
    order = sorted(range(len(facts)), key=lambda i: (-float(dense_scores[i]), facts[i]["id"]))
    dense_ids = [facts[i]["id"] for i in order[:MAX_FACTS]]
    baseline_text, baseline_render = render_selection(graph, dense_ids)
    budget = len(baseline_text)
    if additive_cost(facts, group_costs, dense_ids) != budget:
        raise AdapterError("baseline additive cost / renderer mismatch")
    result = dict(selector(facts=facts, question=question, dense_scores=list(map(float, dense_scores)),
                           bm25_scores=list(map(float, bm25_scores)), max_facts=MAX_FACTS,
                           char_budget=budget + 1, group_costs=group_costs, time_limit=time_limit))
    selected = result.get("selected_ids")
    if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
        raise AdapterError("core selected_ids contract is missing")
    selected_text, selected_render = render_selection(graph, selected)
    if additive_cost(facts, group_costs, selected) != len(selected_text):
        raise AdapterError("selected additive cost / renderer mismatch")
    if len(selected_text) > budget or len(selected) > len(dense_ids):
        raise AdapterError("new selector exceeds dense baseline character/fact budget")
    return {"pool_size": len(facts), "pool_sha256": direct._sha256(facts),
            "score_arrays_sha256": direct._sha256({"dense": list(map(float, dense_scores)), "bm25": list(map(float, bm25_scores))}),
            "char_budget": budget, "core_additive_budget": budget + 1,
            "global_dense50": {"selected_ids": dense_ids, "render": baseline_render},
            "budget_evidence50": {"selected_ids": selected, "render": selected_render, "core": result}}


def _manifest(arm: str, rows: Sequence[Mapping[str, Any]], prereg_sha: str) -> dict[str, Any]:
    value = {"schema_version": SCHEMA, "selector_name": arm, "score_free": True,
             "zero_reranker": True, "preregistration_sha256": prereg_sha, "rows": list(rows)}
    value["rows_sha256"] = direct._sha256(value["rows"])
    value["manifest_sha256"] = direct._sha256(value)
    return value


def _scope_root(root: Path, smoke: bool) -> Path:
    root = Path(root).resolve()
    # Keep every experiment artifact under an explicit dedicated MemEye root.
    if root.parent != direct.MEMEYE_CACHE_ROOT or not root.name.startswith("mec_transfer_20260906"):
        raise AdapterError("root must be a dedicated mec_transfer_20260906 child of the MemEye cache root")
    direct._validate_roots(root / "_output_validation", root / "_cache_validation")
    if root == direct.MEMEYE_CACHE_ROOT or root == direct.CANONICAL_ROOT:
        raise AdapterError("unsafe experiment root")
    return root / ("smoke4" if smoke else "pilot24")


def prepare(*, root: Path = DEFAULT_ROOT, smoke: bool = False, time_limit: float = 5.0) -> dict[str, Any]:
    scope = _scope_root(root, smoke)
    if (scope / "selection_complete.json").exists():
        return validate_bundle(scope)
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise AdapterError("invalid solver stage time limit")
    core = importlib.import_module("creamem3.budget_evidence_v1")
    canonical = direct._canonical_sources()
    pilot = pilot_rows(canonical["query_rows"])
    selected_queries = pilot[:4] if smoke else pilot
    v2 = direct._read_object(V2_MANIFEST, "frozen V2 score-free selection manifest")
    legacy_selector._validate_mec50_manifest(v2)
    v2_by_id = {str(row["row_id"]): row for row in v2["rows"]}
    prereg = {"schema_version": SCHEMA, "score_free": True,
              "sampling": "sha256('mec-transfer-20260906:'+row_id) ascending; first24; no labels/task quotas",
              "pilot_ids_hash_order": [row["row_id"] for row in pilot],
              "scope_ids_hash_order": [row["row_id"] for row in selected_queries],
              "smoke": smoke, "max_facts": MAX_FACTS, "solver_time_limit_per_stage_seconds": time_limit,
              "candidate_pool": "every fact in the query task's sealed g35 Online graph; no preliminary cap",
              "dense_ties": "fact ID ascending", "budget_rule": "Bq=len(same_raw_renderer(global_dense50)); new selection <= Bq and <=50 facts",
              "cost_model": "sum(len(raw_fact.text)+5)+sum(unique_session_header_chars+1)-1; core budget=Bq+1",
              "budget_unit": "Unicode characters, NOT tokens; per-question dense-matched, NOT global constant",
              "sources": "exact source_dia_ids; source_map entries and rounds validated but different sources in one round remain distinct; group=session_id",
              "rendering": "unchanged legacy raw renderer for all arms; body regrouped by session, not global rank order",
              "v2_historical": "frozen IDs only; not recomputed/truncated, not scheduled for new model calls; may exceed Bq",
              "live_arms": list(LIVE_ARMS), "gold_read_before_manifest_seal": False,
              "no_retuning_from_pilot_scores": True, "canonical_sources": canonical["receipt"],
              "adapter_file_sha256": direct._file_sha256(Path(__file__)),
              "core_file_sha256": direct._file_sha256(Path(core.__file__)),
              "renderer_file_sha256": direct._file_sha256(Path(raw.__file__)),
              "v2_manifest_path": str(V2_MANIFEST), "v2_manifest_file_sha256": direct._file_sha256(V2_MANIFEST)}
    prereg["preregistration_sha256"] = direct._sha256(prereg)
    # This immutable preregistration exists before any arm selection is made.
    direct._write_once(scope / "preregistration.json", prereg)
    graphs: dict[str, Any] = {}
    indices: dict[str, Any] = {}
    graph_receipts: dict[str, Any] = {}
    output_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    diagnostics: list[dict[str, Any]] = []
    for ordinal, query in enumerate(selected_queries, 1):
        task, row_id = str(query["task"]), str(query["row_id"])
        if task not in graphs:
            path, digest = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
            graph = direct._read_object(path, "sealed Online graph")
            graphs[task] = graph
            indices[task] = graph_memory_v2.GraphMemoryIndex(graph, summary_top_k=10, dense_fact_top_k=20,
                bm25_fact_top_k=20, candidate_cap=len(graph["fact_nodes"]), rerank_top_k=50)
            graph_receipts[task] = {"path": str(path), "file_sha256": digest, "fact_count": len(graph["fact_nodes"])}
        index, graph = indices[task], graphs[task]
        vector = canonical["vector_by_id"][row_id]
        dense = list(map(float, index._scores(index.fact_matrix, vector)))
        bm25_map = {str(fact_id): float(value) for fact_id, value in index.bm25.search(str(query["question"]), len(index.facts))}
        bm25 = [bm25_map.get(str(fact["id"]), 0.0) for fact in index.facts]
        if [str(fact["id"]) for fact in index.facts] != [str(fact["id"]) for fact in graph["fact_nodes"]]:
            raise AdapterError("index reordered source fact pool")
        selected = select_row(graph=graph, question=str(query["question"]), dense_scores=dense,
                              bm25_scores=bm25, selector=core.select_evidence, time_limit=time_limit)
        for arm in LIVE_ARMS:
            trace = {"pool_sha256": selected["pool_sha256"], "pool_size": selected["pool_size"],
                     "score_arrays_sha256": selected["score_arrays_sha256"], "char_budget": selected["char_budget"],
                     **selected[arm]}
            ids = selected[arm]["selected_ids"]
            output_rows[arm].append({"row_id": row_id, "selected_fact_ids": ids,
                "selected_fact_ids_sha256": direct._sha256(ids), "trace": trace, "trace_sha256": direct._sha256(trace)})
        historical = v2_by_id[row_id]
        if historical.get("question") != query["question"] or historical.get("task") != task:
            raise AdapterError("frozen V2 question binding mismatch")
        v2_ids = list(historical["selected_fact_ids"])
        _, v2_render = render_selection(graph, v2_ids)
        v2_trace = {"frozen_source_manifest_file_sha256": prereg["v2_manifest_file_sha256"],
                    "not_recomputed": True, "not_scheduled_for_live_calls": True,
                    "render": v2_render, "char_budget_diagnostic": selected["char_budget"],
                    "over_dense_char_budget": v2_render["context_chars"] > selected["char_budget"]}
        output_rows["v2_historical"].append({"row_id": row_id, "selected_fact_ids": v2_ids,
            "selected_fact_ids_sha256": direct._sha256(v2_ids), "trace": v2_trace,
            "trace_sha256": direct._sha256(v2_trace)})
        diagnostics.append({"row_id": row_id, "pool_size": selected["pool_size"],
            "char_budget": selected["char_budget"], "new_chars": selected["budget_evidence50"]["render"]["context_chars"],
            "new_facts": len(selected["budget_evidence50"]["selected_ids"]),
            "certified_optimal": selected["budget_evidence50"]["core"].get("certified_optimal"),
            "v2_chars": v2_render["context_chars"], "v2_over_budget": v2_trace["over_dense_char_budget"]})
        print(f"offline selection {ordinal}/{len(selected_queries)}: {row_id}", flush=True)
    manifest_hashes = {}
    for arm in ARMS:
        path = scope / "manifests" / f"{arm}.json"
        direct._write_once(path, _manifest(arm, output_rows[arm], prereg["preregistration_sha256"]))
        manifest_hashes[arm] = direct._file_sha256(path)
    complete = {"schema_version": SCHEMA, "complete": True, "rows": len(selected_queries),
                "preregistration_sha256": prereg["preregistration_sha256"],
                "manifest_file_sha256": manifest_hashes, "graph_receipts": graph_receipts,
                "rows_diagnostic": diagnostics, "provider_calls": 0, "qwen_calls": 0,
                "evaluator_data_read": False}
    complete["complete_sha256"] = direct._sha256(complete)
    direct._write_once(scope / "selection_complete.json", complete)
    return validate_bundle(scope)


def validate_bundle(scope: Path) -> dict[str, Any]:
    complete = direct._read_object(scope / "selection_complete.json", "sealed selector bundle")
    body = dict(complete)
    digest = body.pop("complete_sha256", None)
    if digest != direct._sha256(body) or complete.get("complete") is not True:
        raise AdapterError("selector bundle seal drift")
    prereg = direct._read_object(scope / "preregistration.json", "selector preregistration")
    body = dict(prereg)
    digest = body.pop("preregistration_sha256", None)
    if digest != direct._sha256(body) or digest != complete["preregistration_sha256"]:
        raise AdapterError("preregistration seal drift")
    if direct._file_sha256(Path(__file__)) != prereg["adapter_file_sha256"]:
        raise AdapterError("adapter source drift after preregistration")
    core = importlib.import_module("creamem3.budget_evidence_v1")
    if direct._file_sha256(Path(core.__file__)) != prereg["core_file_sha256"]:
        raise AdapterError("core source drift after preregistration")
    if direct._file_sha256(Path(raw.__file__)) != prereg["renderer_file_sha256"]:
        raise AdapterError("renderer source drift after preregistration")
    canonical = direct._canonical_sources()
    if canonical["receipt"] != prereg["canonical_sources"]:
        raise AdapterError("frozen query/build/embedding receipt drift")
    for task, graph_receipt in complete["graph_receipts"].items():
        path, digest = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
        if str(path) != graph_receipt["path"] or digest != graph_receipt["file_sha256"]:
            raise AdapterError("sealed graph binding drift")
    for arm in ARMS:
        path = scope / "manifests" / f"{arm}.json"
        if direct._file_sha256(path) != complete["manifest_file_sha256"][arm]:
            raise AdapterError("arm manifest byte drift")
        value = direct._read_object(path, "arm manifest")
        body = dict(value)
        digest = body.pop("manifest_sha256", None)
        if digest != direct._sha256(body) or value["preregistration_sha256"] != complete["preregistration_sha256"]:
            raise AdapterError("arm manifest seal drift")
        if [row["row_id"] for row in value["rows"]] != prereg["scope_ids_hash_order"]:
            raise AdapterError("arm scope mismatch")
        direct._selection_rows(path)
    return complete


def run_phase(*, root: Path, smoke: bool, phase: str, arm: str | None, workers: int,
              allow_model_calls: bool = False) -> dict[str, Any]:
    live = phase in {"predict", "score", "all"}
    if live != allow_model_calls:
        raise AdapterError("live phases require explicit allow_model_calls; offline phases reject it")
    if live and arm not in LIVE_ARMS:
        raise AdapterError("only dense/new arms are authorized for live execution")
    scope = _scope_root(root, smoke)
    complete = validate_bundle(scope)  # All arms must be sealed before scoring can read gold.
    results = {}
    for current in ([arm] if arm else list(ARMS)):
        if current not in ARMS:
            raise AdapterError("unknown arm")
        selection_path = scope / "manifests" / f"{current}.json"
        output, cache = scope / current / "results", scope / current / "provider_cache"
        gate = raw.preflight(selection_path=selection_path, output_root=output, cache_root=cache, workers=workers)
        if len(gate["prepared"]) != complete["rows"]:
            raise AdapterError("reader scope count mismatch")
        for prepared in gate["prepared"]:
            trace = prepared["selection"]["trace"]
            if prepared["context_sha256"] != trace["render"]["context_sha256"]:
                raise AdapterError("reader context differs from sealed selector rendering")
        if phase == "preflight":
            results[current] = {"status": gate["status"], "rows": gate["contract"]["rows"],
                "contract_sha256": gate["contract"]["contract_sha256"], "length_summary": gate["length_summary"],
                "reader_authority": gate["contract"]["reader_authority"], "judge_authority": gate["contract"]["judge_authority"],
                "network_calls": 0, "files_written": 0}
        elif phase == "audit":
            results[current] = raw.audit(selection_path=selection_path, output_root=output, cache_root=cache, workers=workers)
        else:
            output = Path(gate["output_root"])
            cache = Path(gate["cache_root"])
            phases = {}
            if phase in {"predict", "all"}:
                phases["predict"] = direct.run_predict(output_root=output, cache_root=cache, contract=gate["contract"],
                    authority=gate["authority"], prepared=gate["prepared"], workers=workers)
            if phase in {"score", "all"}:
                phases["score"] = direct.run_score(output_root=output, cache_root=cache, contract=gate["contract"],
                    prepared=gate["prepared"], workers=workers)
            results[current] = phases
    return {"phase": phase, "scope": str(scope), "arms": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "preflight", "predict", "score", "all", "audit"), default="preflight")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--smoke", action="store_true", help="first4 of the frozen hash-selected24, isolated scope")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--solver-stage-seconds", type=float, default=5.0)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        if args.allow_model_calls or args.arm is not None:
            parser.error("prepare is offline and freezes all arms")
        result = prepare(root=args.root, smoke=args.smoke, time_limit=args.solver_stage_seconds)
    else:
        result = run_phase(root=args.root, smoke=args.smoke, phase=args.phase, arm=args.arm,
                           workers=args.workers, allow_model_calls=args.allow_model_calls)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

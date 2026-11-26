"""Frozen full371 MemEye rounds, unchanged Reader/Judge prompts and models.

Only `live --allow-model-calls` invokes configured providers. Preparation and
audit are offline. Contexts are reconstructed from original facts or explicit
nonoverlapping contiguous character spans; no arbitrary context prose is used.
No git operation, old-cache mutation, or experiment-ledger update is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from creamem3 import official_g25_runtime_adapter_v19 as g25
from creamem3.experiments.full_budget_20260907 import run_memeye_math as baseline
from creamem3.experiments.full_budget_20260907 import observation_order as observation
from creamem3.experiments.full_budget_20260907 import summarize_memeye_math as full_audit
from creamem3.experiments.mec_transfer_20260906 import summarize_memeye as previous

direct, raw = baseline.direct, baseline.raw
ROOT = Path(r"D:\creamem3_cache\memeye_v19\iterative_memory_20260907")
SCHEMA = "memeye.iterative.context_manifest.v1"
CAPS = {"focused_max_output_tokens": 2600, "fused_max_output_tokens": 1200}
FORBIDDEN = {"gold", "ground_truth", "gold_answer", "answer", "expected_answer", "clue", "point", "x", "y", "xy",
             "question_type", "category", "score", "mapped_score", "raw_score", "correct", "correctness"}
ORIGIN_ARMS = ("math50", "dense50")  # Priority is fixed, never score-dependent.
read, require = previous.read, previous.require


def sealed(value: Mapping[str, Any], field: str = "seal_sha256") -> dict[str, Any]:
    result = dict(value)
    require(field not in result, f"cannot seal an already sealed {field}")
    result[field] = direct._sha256(result)
    return result


def checked(path: Path, field: str = "seal_sha256") -> dict[str, Any]:
    value = read(path)
    previous.seal(value, field)
    return value


def round_root(round_id: str) -> Path:
    require(isinstance(round_id, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", round_id) is not None,
            "round_id must be a simple dedicated directory name")
    root = (ROOT / round_id).resolve()
    require(root.parent == ROOT.resolve(), "round path escaped dedicated root")
    return root


def reject_evaluator_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        require(not (set(str(k).casefold() for k in value) & FORBIDDEN), "evaluator fields forbidden at preparation boundary")
        for item in value.values():
            reject_evaluator_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_evaluator_fields(item)


def reader_config(value: Any) -> dict[str, int]:
    require(isinstance(value, Mapping) and not (set(value) - set(CAPS)), "unknown Reader configuration")
    result = {**CAPS, **value}
    require(all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 65536 for v in result.values()),
            "invalid explicitly declared output cap")
    return result


def validate_manifest(manifest: Mapping[str, Any], canonical_rows: Sequence[Mapping[str, Any]], round_id: str) -> dict[str, Any]:
    reject_evaluator_fields(manifest)
    allowed = {"schema", "round_id", "method", "reader", "render_order", "rows", "workers"}
    require(not (set(manifest) - allowed), "unknown manifest field")
    require(manifest.get("schema") == SCHEMA and manifest.get("round_id") == round_id, "manifest schema/round drift")
    method = manifest.get("method")
    require(isinstance(method, Mapping) and isinstance(method.get("name"), str) and bool(method["name"].strip()), "method name required")
    rows = manifest.get("rows")
    require(isinstance(rows, list) and len(rows) == len(canonical_rows) == 371, "full371 required; sampling is not an effective round")
    require([r.get("row_id") for r in rows] == [r["row_id"] for r in canonical_rows]
            and len({r["row_id"] for r in rows}) == 371, "full371 canonical order/uniqueness drift")
    for row, query in zip(rows, canonical_rows, strict=True):
        require(all(row.get(k) == query[k] for k in ("row_id", "question", "task")), "canonical query binding drift")
        allowed_row = {"row_id", "question", "task", "selected_fact_ids", "units", "selection_provenance",
                       "expected_context_sha256", "expected_context_chars"}
        require(not (set(row) - allowed_row), "unknown manifest row field")
        require(isinstance(row.get("selection_provenance"), Mapping) and bool(row["selection_provenance"]), "selection provenance required")
        require(("selected_fact_ids" in row) != ("units" in row), "choose exactly one facts/units representation")
    config = reader_config(manifest.get("reader", {}))
    order = manifest.get("render_order", "original")
    require(order in {"original", "observation"}, "render order must be explicitly supported")
    require(order == "original" or not any("units" in r for r in rows), "observation order currently accepts whole facts only")
    workers = manifest.get("workers", 8)
    require(isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= 12, "workers must be1..12")
    return {**dict(manifest), "reader": config, "render_order": order, "workers": workers}


def materialize_units(row: Mapping[str, Any], graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Every single contiguous slice counts as one unit; overlap is rejected."""
    by_id = {str(f["id"]): f for f in graph["fact_nodes"]}
    sources = observation.source_index(graph["memeye_source_map"], expected_task=row["task"])
    if "selected_fact_ids" in row:
        ids = row["selected_fact_ids"]
        require(isinstance(ids, list) and 1 <= len(ids) <= 50 and len(set(ids)) == len(ids), "1..50 unique full facts required")
        specs = [{"unit_id": str(fid), "fact_id": str(fid), "start": 0,
                  "end": len(by_id[str(fid)]["text"])} for fid in ids if str(fid) in by_id]
        require(len(specs) == len(ids), "unknown source fact")
    else:
        specs = row.get("units")
        require(isinstance(specs, list) and 1 <= len(specs) <= 50, "1..50 explicit contiguous units required")
    spans, unit_ids, items, provenance = {}, set(), [], []
    for spec in specs:
        require(isinstance(spec, Mapping) and "spans" not in spec, "one continuous start/end per unit; multislice packing forbidden")
        fid, uid = spec.get("fact_id"), spec.get("unit_id")
        require(isinstance(fid, str) and fid in by_id and isinstance(uid, str) and bool(uid) and uid not in unit_ids, "unknown fact/duplicate unit ID")
        parent = by_id[fid]
        start, end = spec.get("start"), spec.get("end")
        require(all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end))
                and 0 <= start < end <= len(parent["text"]), "unit offset outside source text")
        require(not any(start < old_end and old_start < end for old_start, old_end in spans.get(fid, [])), "overlapping same-fact spans are forbidden")
        text = parent["text"][start:end]
        require(bool(text.strip()), "empty/whitespace-only unit")
        require("text" not in spec or spec["text"] == text, "extractive unit text does not exactly match source slice")
        source_ids = sorted(set(parent["source_dia_ids"]))
        signature = [list(s) for s in observation.observation_signature(parent, sources)]
        require("source_dia_ids" not in spec or sorted(set(spec["source_dia_ids"])) == source_ids, "unit changed inherited fact sources")
        require("observation_signature" not in spec or [list(s) for s in spec["observation_signature"]] == signature,
                "unit changed inherited observation signature")
        parent_sha = direct._sha256(parent["text"])
        require("parent_text_sha256" not in spec or spec["parent_text_sha256"] == parent_sha, "parent text SHA drift")
        unit_ids.add(uid)
        spans.setdefault(fid, []).append((start, end))
        materialized = dict(parent)
        materialized.update(id=uid, text=text)
        items.append({"fact": materialized})
        provenance.append({"unit_id": uid, "fact_id": fid, "start": start, "end": end,
            "text_sha256": direct._sha256(text), "parent_text_sha256": parent_sha,
            "source_dia_ids": source_ids, "observation_signature": signature,
            "provenance_scope": "inherited_parent_fact_not_clause_resolved"})
    return items, provenance, list(dict.fromkeys(p["fact_id"] for p in provenance))


def compile_row(query: Mapping[str, Any], row: Mapping[str, Any], graph: Mapping[str, Any], order: str) -> dict[str, Any]:
    items, unit_provenance, fact_ids = materialize_units(row, graph)
    if order == "observation":
        require("selected_fact_ids" in row, "whole facts required for observation order")
        context, assembly = observation.assemble(items, graph["memeye_source_map"], expected_task=query["task"])
    else:
        require(order == "original", "unsupported render order")
        context, assembly = raw._routing_only_assemble(items)
    whole = "selected_fact_ids" in row
    assembly = {**assembly, "original_fact_text_unchanged": whole, "units_exact_source_slices": True,
                "selected_unit_ids": [p["unit_id"] for p in unit_provenance], "unit_provenance_sha256": direct._sha256(unit_provenance)}
    if not whole:
        assembly["raw_fact_text_unchanged"] = False
        assembly["context_mode"] = "routing_only_extractive_units"
    require("expected_context_sha256" not in row or row["expected_context_sha256"] == direct._sha256(context), "declared context hash drift")
    require("expected_context_chars" not in row or row["expected_context_chars"] == len(context), "declared context cost drift")
    lengths = sorted(len(i["fact"]["text"]) for i in items)
    source_ids = sorted({s for p in unit_provenance for s in p["source_dia_ids"]})
    sessions = sorted({int(i["fact"]["session_id"]) for i in items})
    trace = {"selection_provenance": copy.deepcopy(row["selection_provenance"]), "units": unit_provenance,
             "render_order": order, "context_sha256": direct._sha256(context), "actual_cost": {"units": len(items),
                "distinct_parent_facts": len(fact_ids), "unit_text_chars": sum(lengths), "context_chars": len(context)},
             "raw_fact_text_unchanged": whole}
    selection = sealed({"row_id": query["row_id"], "selected_fact_ids": fact_ids,
        "selected_fact_ids_sha256": direct._sha256(fact_ids), "trace": trace, "trace_sha256": direct._sha256(trace)}, "row_sha256")
    return {"query": dict(query), "selection": selection, "context": context, "context_sha256": direct._sha256(context),
        "assembly": assembly, "assembly_sha256": direct._sha256(assembly), "stats": {
            "context_mode": assembly["context_mode"], "selected_facts": len(fact_ids), "selected_units": len(items),
            "selected_fact_chars": sum(lengths), "selected_fact_chars_mean": sum(lengths) / len(lengths),
            "selected_fact_chars_p50": lengths[(len(lengths) - 1) // 2], "selected_fact_chars_p90": lengths[math.ceil(.9 * len(lengths)) - 1],
            "selected_fact_chars_max": max(lengths), "context_chars": len(context), "assembly_overhead_chars": len(context) - sum(lengths),
            "session_summary_text_chars": 0, "event_view_text_chars": 0, "formatting_and_provenance_chars": len(context) - sum(lengths),
            "unique_sessions": len(sessions), "session_ids": sessions, "source_dia_ids": source_ids,
            "unique_source_rounds": len({tuple(s) for p in unit_provenance for s in p["observation_signature"]}),
            "rendered_session_summaries": 0, "rendered_event_views": 0,
            "rendered_session_headers": len(assembly["rendered_session_headers"])}}


def source_hashes() -> dict[str, str]:
    return {str(Path(m.__file__).resolve()): direct._file_sha256(Path(m.__file__))
            for m in (direct, direct.predict, direct.retry5, direct.score, direct.g35_runtime, g25, raw, observation, full_audit, previous)} | {
                str(Path(__file__).resolve()): direct._file_sha256(Path(__file__))}


def baseline_receipts() -> dict[str, Any]:
    return {arm: {str(path): direct._file_sha256(path) for path in (
        baseline.ROOT / arm / "results/contract.json", baseline.ROOT / arm / "results/predict/complete.json",
        baseline.ROOT / arm / "results/score/complete.json")} for arm in ORIGIN_ARMS}


def _build_gate(manifest: Mapping[str, Any], round_id: str) -> dict[str, Any]:
    canonical = direct._canonical_sources()
    clean = validate_manifest(manifest, canonical["query_rows"], round_id)
    graphs, graph_receipts, prepared = {}, {}, []
    for query, row in zip(canonical["query_rows"], clean["rows"], strict=True):
        task = query["task"]
        if task not in graphs:
            path, sha = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
            graphs[task] = read(path)
            graph_receipts[task] = {"path": str(path), "sha256": sha}
        prepared.append(compile_row(query, row, graphs[task], clean["render_order"]))
    authority, receipt = direct._reader_authority()
    root = round_root(round_id)
    output, cache = direct._validate_roots(root / "results", root / "provider_cache")
    contract = direct._contract(output_root=output, cache_root=cache,
        selection_receipt={"manifest_sha256": direct._sha256(clean)}, canonical_receipt=canonical["receipt"],
        authority_receipt=receipt, prepared=prepared, workers=clean["workers"])
    contract.pop("contract_sha256")
    contract.update(iterative_schema=SCHEMA, round_id=round_id, declared_reader_output_caps=clean["reader"],
        context_derivation="original_facts_or_verified_contiguous_extracts", render_order=clean["render_order"],
        method=clean["method"], judge_reuse_policy="exact_triple_same_prompt_model_math50_then_dense50_never_choose_score")
    contract = sealed(contract, "contract_sha256")
    return {"manifest": clean, "prepared": prepared, "authority": authority, "contract": contract,
        "graph_receipts": graph_receipts, "canonical_receipt": canonical["receipt"], "output_root": str(output),
        "cache_root": str(cache), "workers": clean["workers"], "reader": clean["reader"]}


def prepare(round_id: str, manifest_path: Path) -> dict[str, Any]:
    """No gold or score parser is called; baseline files are only hash-bound."""
    root = round_root(round_id)
    raw_manifest = read(manifest_path)
    gate = _build_gate(raw_manifest, round_id)
    method_sources = gate["manifest"]["method"].get("source_hashes", {})
    require(isinstance(method_sources, Mapping), "method.source_hashes must map explicit path to SHA256")
    for path, sha in method_sources.items():
        require(direct._file_sha256(Path(path)) == sha, "declared method implementation changed")
    manifest = gate["manifest"]
    direct._write_once(root / "manifest.json", manifest)
    bundle = sealed({"contract": gate["contract"], "prepared": gate["prepared"], "graph_receipts": gate["graph_receipts"],
        "canonical_receipt": gate["canonical_receipt"], "manifest_sha256": direct._sha256(manifest)})
    direct._write_once(root / "prepared.json", bundle)
    registration = sealed({"schema": SCHEMA, "round_id": round_id, "rows": len(gate["prepared"]),
        "method": manifest["method"], "reader": manifest["reader"], "render_order": manifest["render_order"],
        "manifest_sha256": direct._sha256(manifest), "prepared_sha256": bundle["seal_sha256"],
        "source_hashes": source_hashes(), "baseline_receipts": baseline_receipts(), "judge_priority": list(ORIGIN_ARMS),
        "no_gold_at_context_preparation": True, "network_calls": 0})
    direct._write_once(root / "preregistration.json", registration)
    return {"phase": "frozen", "round_id": round_id, "rows": len(gate["prepared"]), "network_calls": 0,
        "preregistration_sha256": registration["seal_sha256"], "context_char_total": sum(len(r["context"]) for r in gate["prepared"])}


def frozen_gate(round_id: str) -> dict[str, Any]:
    root = round_root(round_id)
    reg, bundle = checked(root / "preregistration.json"), checked(root / "prepared.json")
    require(reg["source_hashes"] == source_hashes(), "frozen runtime/assembler/audit code drift")
    require(reg["baseline_receipts"] == baseline_receipts(), "frozen baseline receipt drift")
    for path, sha in reg["method"].get("source_hashes", {}).items():
        require(direct._file_sha256(Path(path)) == sha, "method source drift")
    gate = _build_gate(read(root / "manifest.json"), round_id)
    require(bundle["manifest_sha256"] == reg["manifest_sha256"] == direct._sha256(gate["manifest"]), "manifest seal drift")
    require(bundle["seal_sha256"] == reg["prepared_sha256"] and bundle["prepared"] == gate["prepared"]
            and bundle["contract"] == gate["contract"] and bundle["graph_receipts"] == gate["graph_receipts"]
            and bundle["canonical_receipt"] == gate["canonical_receipt"], "frozen full context/contract drift")
    return {**gate, "registration": reg}


class DeclaredCapReader:
    """Explicit registered output-cap variant; prompt/json route is unchanged."""
    def __init__(self, delegate: Any, config: Mapping[str, int]):
        self.delegate, self.config = delegate, dict(config)

    def generate(self, stage: str, prompt: str, *, json_mode: bool, max_output_tokens: int, cache_metadata=None):
        defaults = {"focused": 2600, "answer": 500, "structured_fused": 1200}
        require(stage in defaults and max_output_tokens == defaults[stage], "base Reader call contract unexpectedly changed")
        cap = self.config["focused_max_output_tokens"] if stage == "focused" else self.config["fused_max_output_tokens"] if stage == "structured_fused" else 500
        return self.delegate.generate(stage, prompt, json_mode=json_mode, max_output_tokens=cap, cache_metadata=cache_metadata)

    def usage_receipt(self):
        return self.delegate.usage_receipt()


def _reader_backend(cache: Path, budget: Mapping[str, Any], config: Mapping[str, int]):
    source = g25._load_source_runtime()
    hard = budget["stage_provider_attempt_hard_ceilings"]
    limits = {"planner": source.StageLimits(512, hard["semantic_planner"], 5),
        "focused": source.StageLimits(config["focused_max_output_tokens"], hard["focused_reader"], 5),
        "answer": source.StageLimits(500, hard["answer_reader"], 5),
        "structured_fused": source.StageLimits(config["fused_max_output_tokens"], hard["structured_fused_reader"], 5)}
    facade = g25.OfficialGemini25Runtime(cache / "provider/predict/g25_reader", allow_model_calls=True,
        provider_factory=lambda: direct.retry5._ReliableRESTProvider(source), stage_limits=limits)
    return DeclaredCapReader(direct.retry5._ConcurrentG25ReaderBackend(facade), config)


@contextmanager
def declared_reader_factory(config: Mapping[str, int]):
    original = direct.retry5._reader_backend
    direct.retry5._reader_backend = lambda root, budget: _reader_backend(root, budget, config)
    try:
        yield
    finally:
        direct.retry5._reader_backend = original


def choose_judge_origins(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Input order is trusted fixed arm priority + canonical row order, not score."""
    chosen, alternatives = {}, {}
    for item in candidates:
        key = item["exact_triple_sha256"]
        alternatives.setdefault(key, []).append(dict(item))
        chosen.setdefault(key, dict(item))
    return {key: {"chosen": chosen[key], "alternatives": items,
        "mapped_conflict": len({float(i["native_mapped_score"]) for i in items}) > 1,
        "raw_conflict": len({float(i["native_raw_score"]) for i in items}) > 1} for key, items in alternatives.items()}


def judge_lookup(gold: Mapping[str, Any], judge_authority: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluator-only: executed after current full371 contexts and QA are frozen."""
    candidates, origin_rows, baselines = [], {}, {}
    for arm in ORIGIN_ARMS:
        root = baseline.ROOT / arm / "results"
        cache = baseline.ROOT / arm / "provider_cache"
        contract = checked(root / "contract.json", "contract_sha256")
        require(contract["judge_authority"] == judge_authority, "historical Judge model/prompt/parser differs")
        complete = checked(root / "score/complete.json", "complete_sha256")
        require(complete["complete"] is True and len(complete["rows"]) == 371, "historical Judge arm incomplete")
        baselines[arm] = {"native_mapped_sum": 0.0, "native_raw_sum": 0.0, "triples": []}
        for receipt in complete["rows"]:
            path = Path(receipt["path"])
            require(receipt["file_sha256"] == direct._file_sha256(path), "historical score receipt SHA drift")
            prediction = read(root / "predict/rows" / path.name)
            previous.seal(prediction, "row_sha256")
            row = direct._load_score_row(path, prediction, gold[receipt["row_id"]])
            require(row["row_sha256"] == receipt["row_sha256"], "historical score row seal drift")
            final = read(Path(row["cache_path"]))
            branch = cache / "provider/score/g35_judge"
            full_audit._judge_binding(row, prediction, gold[receipt["row_id"]], branch,
                {("judge", row["request_sha256"]): final})
            plan = read(branch / "plans/judge" / f"{row['resume_hash']}.json")
            expected_prompt = direct.g35_runtime.build_mgtm_judge_prompt(gold[receipt["row_id"]]["question"], gold[receipt["row_id"]]["gold"], prediction["prediction"])
            require(plan["prompt_sha256"] == direct._sha256(expected_prompt) and plan["model"] == judge_authority["model"], "historical exact Judge prompt/model drift")
            require(plan["cache_metadata"]["exact_triple_sha256"] == row["exact_triple_sha256"], "historical plan triple drift")
            item = {"arm": arm, "row_id": row["row_id"], "exact_triple_sha256": row["exact_triple_sha256"],
                "score_path": str(path), "score_file_sha256": direct._file_sha256(path), "score_row_sha256": row["row_sha256"],
                "prediction_path": str(root / "predict/rows" / path.name), "prediction_file_sha256": direct._file_sha256(root / "predict/rows" / path.name),
                "cache_branch": str(branch), "cache_path": row["cache_path"], "cache_file_sha256": row["cache_file_sha256"],
                "native_mapped_score": row["mapped_score"], "native_raw_score": row["raw_score"]}
            candidates.append(item)
            origin_rows[str(path)] = row
            baselines[arm]["native_mapped_sum"] += row["mapped_score"]
            baselines[arm]["native_raw_sum"] += row["raw_score"]
            baselines[arm]["triples"].append(row["exact_triple_sha256"])
    entries = choose_judge_origins(candidates)
    for item in baselines.values():
        item["canonical_mapped_sum"] = sum(entries[key]["chosen"]["native_mapped_score"] for key in item["triples"])
        item["canonical_raw_sum"] = sum(entries[key]["chosen"]["native_raw_score"] for key in item["triples"])
        item.pop("triples")
    return {"entries": entries, "origin_rows": origin_rows, "baselines": baselines,
            "priority": list(ORIGIN_ARMS), "authority": dict(judge_authority)}


def score_with_reuse(gate: Mapping[str, Any]) -> dict[str, Any]:
    root, cache = Path(gate["output_root"]), Path(gate["cache_root"])
    _, predictions = direct._load_prediction_complete(root, gate["contract"], gate["prepared"])
    gold = direct._gold_rows()
    lookup = judge_lookup(gold, gate["contract"]["judge_authority"])
    plan_rows = []
    for prediction in predictions:
        rid = prediction["row_id"]
        key = direct._sha256([gold[rid]["question"], gold[rid]["gold"], prediction["prediction"]])
        entry = lookup["entries"].get(key)
        plan_rows.append({"row_id": rid, "exact_triple_sha256": key, "kind": "fixed_historical_exact_triple" if entry else "new_round_provider",
                          "origin": entry})
    plan = sealed({"rows": plan_rows, "contract_sha256": gate["contract"]["contract_sha256"],
        "priority": list(ORIGIN_ARMS), "baseline_native_and_canonical": lookup["baselines"]})
    direct._write_once(root.parent / "judge_plan.json", plan)
    for prediction, planned in zip(predictions, plan_rows, strict=True):
        if planned["origin"] is None:
            continue
        origin = planned["origin"]["chosen"]
        original = lookup["origin_rows"][origin["score_path"]]
        rid, g = prediction["row_id"], gold[prediction["row_id"]]
        row = dict(original)
        row.pop("row_sha256")
        row.update(row_id=rid, task=g["task"], x=g["x"], y=g["y"], xy=f"{g['x']}-{g['y']}",
            question_sha256=direct._sha256(g["question"]), gold_sha256=direct._sha256(g["gold"]),
            prediction_sha256=prediction["prediction_sha256"], exact_triple_sha256=planned["exact_triple_sha256"],
            judge_origin={"policy": "math50_then_dense50_fixed_priority", "receipt": origin,
                          "judge_plan_sha256": plan["seal_sha256"], "new_provider_call": False})
        direct._write_once(direct._score_path(root, prediction), sealed(row, "row_sha256"))
    return direct.run_score(output_root=root, cache_root=cache, contract=gate["contract"], prepared=gate["prepared"], workers=gate["workers"])


def provider_audit(cache: Path) -> dict[str, Any]:
    """Original physical audit, including all-cache-hit (no Judge directory) runs.

    The imported audit is executed in a private namespace with only its branch
    enumeration made empty-directory-safe. No sealed file or live module is
    modified. Its original function source remains hash-bound by registration.
    """
    import inspect
    source = inspect.getsource(previous.provider_audit)
    old = 'for stage_dir in sorted((branch / "attempts").iterdir()):'
    new = 'for stage_dir in sorted((branch / "attempts").iterdir()) if (branch / "attempts").exists() else []:'
    require(source.count(old) == 1, "provider audit adaptation anchor drift")
    namespace = dict(vars(previous))
    exec(compile(source.replace(old, new), "<readonly-empty-judge-audit>", "exec"), namespace)
    return namespace["provider_audit"](cache)


def _reader_payload_audit(predictions, prepared, authority, finals, cache, config):
    """Rebuild actual payload SHA for explicit caps, not merely prompt SHA."""
    source = g25._load_source_runtime()
    # Payload does not contain concurrency ceilings; caps are explicit below.
    limits = {name: source.StageLimits(cap, 10000, 5) for name, cap in (
        ("focused", config["focused_max_output_tokens"]), ("answer", 500),
        ("structured_fused", config["fused_max_output_tokens"]), ("planner", 512))}
    for prediction, item in zip(predictions, prepared, strict=True):
        question, context = item["query"]["question"], item["context"]
        stages = prediction["reader_stages"]
        prompts = {}
        if prediction["reader_route"] == "two_stage":
            prompts["focused"] = str(authority["focused"]).replace("{question}", question).replace("{context}", context)
            parsed = direct.predict._parse_json_object(finals[("focused", stages[0]["request_sha256"])]["text"]) or {"_failed": True}
            qa_context = authority["render"](parsed, context, "candidate_first")
            prompts["answer"] = str(authority["open"]).format(context=qa_context, question=question)
        else:
            prompts["structured_fused"] = str(authority["fused"]).format(question=question, context=context)
        for stage in stages:
            name = stage["stage"]
            final = finals[(name, stage["request_sha256"])]
            plan = read(cache / "provider/predict/g25_reader/plans" / name / f"{final['resume_hash']}.json")
            request = source.build_generate_content_request(name, prompts[name], json_mode=name != "answer",
                max_output_tokens=limits[name].max_output_tokens, stage_limits=limits)
            require(plan["payload_sha256"] == direct._sha256(request), "actual Reader payload/output cap drift")


def audit(round_id: str, *, write: bool = False, gate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    gate = frozen_gate(round_id) if gate is None else gate
    root, cache = Path(gate["output_root"]), Path(gate["cache_root"])
    complete = checked(root / "score/complete.json", "complete_sha256")
    require(complete["complete"] is True and len(complete["rows"]) == 371, "incomplete/failed round cannot be effective")
    predicted, predictions = direct._load_prediction_complete(root, gate["contract"], gate["prepared"])
    require(complete["contract_sha256"] == gate["contract"]["contract_sha256"], "score contract drift")
    require(complete["rows_sha256"] == direct._sha256(complete["rows"]), "score receipt list drift")
    gold = direct._gold_rows()
    lookup = judge_lookup(gold, gate["contract"]["judge_authority"])
    plan = checked(root.parent / "judge_plan.json")
    require(plan["contract_sha256"] == gate["contract"]["contract_sha256"], "Judge reuse plan contract drift")
    physical = provider_audit(cache)
    finals = physical.pop("_final_by_request")
    reader_links, row_data, origins = set(), {}, Counter()
    for prediction, item, receipt, planned in zip(predictions, gate["prepared"], complete["rows"], plan["rows"], strict=True):
        rid = prediction["row_id"]
        require(receipt["row_id"] == planned["row_id"] == rid, "score/reuse-plan census drift")
        path = direct._score_path(root, prediction)
        require(Path(receipt["path"]).resolve() == path.resolve() and receipt["file_sha256"] == direct._file_sha256(path), "score file binding drift")
        row = direct._load_score_row(path, prediction, gold[rid])
        require(row["row_sha256"] == receipt["row_sha256"], "score row receipt drift")
        key = direct._sha256([gold[rid]["question"], gold[rid]["gold"], prediction["prediction"]])
        require(planned["exact_triple_sha256"] == key and planned["origin"] == lookup["entries"].get(key), "fixed-priority exact Judge reuse changed")
        reader_links.update(full_audit._reader_bindings(prediction, item, gate["authority"], finals, cache / "provider/predict/g25_reader"))
        if planned["origin"] is not None:
            origin = planned["origin"]["chosen"]
            require(row["judge_origin"] == {"policy": "math50_then_dense50_fixed_priority", "receipt": origin,
                "judge_plan_sha256": plan["seal_sha256"], "new_provider_call": False}, "Judge origin receipt drift")
            branch = Path(origin["cache_branch"])
            final = read(Path(origin["cache_path"]))
            full_audit._judge_binding(row, prediction, gold[rid], branch, {("judge", row["request_sha256"]): final})
            origins[origin["arm"]] += 1
        else:
            require("judge_origin" not in row, "unplanned historical origin")
            full_audit._judge_binding(row, prediction, gold[rid], cache / "provider/score/g35_judge", finals)
            origins["new_round_provider"] += 1
        row_data[rid] = {"row_id": rid, "task": row["task"], "question": prediction["question"], "prediction": prediction["prediction"],
            "x": row["x"], "y": row["y"], "xy": row["xy"], "mapped_score": row["mapped_score"], "raw_score": row["raw_score"],
            "score_origin": planned["kind"], "context_chars": len(item["context"]), "selected_fact_ids": prediction["selected_fact_ids"],
            "selected_units": item["stats"]["selected_units"], "score_path": str(path)}
    _reader_payload_audit(predictions, gate["prepared"], gate["authority"], finals, cache, gate["reader"])
    physical.update(full_audit._plan_census(cache, reader_links, gold, predictions))
    for completion, field in ((predicted, "reader_usage"), (complete, "judge_usage")):
        for stage, receipt in completion[field]["stages"].items():
            actual = physical["by_stage"].get(stage, {"physical_attempts": 0, "usage": {k: 0 for k in previous.TOKENS}})
            require(actual["physical_attempts"] == receipt["provider_calls"] and actual["usage"] == receipt["usage"], "durable physical/token census drift")
    mapped, raw_sum = sum(r["mapped_score"] for r in row_data.values()), sum(r["raw_score"] for r in row_data.values())
    require(len(row_data) == 371 and math.isclose(mapped / 371, complete["mapped_mean"], abs_tol=1e-12)
        and math.isclose(raw_sum / 371, complete["raw_mean"], abs_tol=1e-12) and complete["qwen_provider_calls"] == 0,
        "final aggregate/full371/no-rerank drift")
    require(len(list((root / "predict/rows").glob("*.json"))) == len(list((root / "score/rows").glob("*.json"))) == 371,
            "extra or missing materialized rows")
    result = sealed({"schema": "memeye.iterative.full371_audit.v1", "status": "PASS", "valid_full_round": True,
        "round_id": round_id, "rows": 371, "mapped_sum": mapped, "mapped_percent": mapped / 371 * 100,
        "raw_mean": raw_sum / 371, "score_policy": "fixed_exact_triple_canonical_priority_math50_then_dense50_else_new",
        "baseline_native_and_canonical": lookup["baselines"], "judge_reuse_rows": dict(origins),
        "reader_config": gate["reader"], "provider": physical, "row_data": row_data,
        "mean_context_chars": sum(r["context_chars"] for r in row_data.values()) / 371,
        "mean_units": sum(r["selected_units"] for r in row_data.values()) / 371,
        "by_x": direct._group_metrics(list(row_data.values()), "x"), "by_y": direct._group_metrics(list(row_data.values()), "y"),
        "by_xy": direct._group_metrics(list(row_data.values()), "xy"),
        "preregistration_sha256": gate["registration"]["seal_sha256"], "judge_plan_sha256": plan["seal_sha256"],
        "score_complete_file_sha256": direct._file_sha256(root / "score/complete.json"),
        "new_reranker_calls": 0, "audit_network_calls": 0,
        "qualification": "Complete audited benchmark round, not independent held-out generalization or answer-optimality proof."})
    if write:
        direct._write_once(root.parent / "analysis/complete_audit.json", result)
    return result


def live(round_id: str, *, allow_model_calls: bool = False) -> dict[str, Any]:
    require(allow_model_calls is True, "live requires explicit --allow-model-calls")
    gate = frozen_gate(round_id)
    with declared_reader_factory(gate["reader"]):
        direct.run_predict(output_root=Path(gate["output_root"]), cache_root=Path(gate["cache_root"]),
            contract=gate["contract"], authority=gate["authority"], prepared=gate["prepared"], workers=gate["workers"])
    score_with_reuse(gate)
    return audit(round_id, write=True, gate=gate)


def status(round_id: str) -> dict[str, Any]:
    root = round_root(round_id)
    return {"round_id": round_id, "prediction_rows": len(list((root / "results/predict/rows").glob("*.json"))),
        "score_rows": len(list((root / "results/score/rows").glob("*.json"))),
        "score_complete": (root / "results/score/complete.json").is_file(),
        "valid_audit_complete": (root / "analysis/complete_audit.json").is_file(), "network_calls": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "live", "audit", "status"))
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args(argv)
    require(args.allow_model_calls == (args.phase == "live"), "live alone requires --allow-model-calls")
    if args.phase == "prepare":
        require(args.manifest is not None, "prepare requires --manifest")
        result = prepare(args.round_id, args.manifest)
    elif args.phase == "preflight":
        gate = frozen_gate(args.round_id)
        result = {"phase": "frozen_preflight_passed", "rows": len(gate["prepared"]), "network_calls": 0,
                  "contract_sha256": gate["contract"]["contract_sha256"]}
    elif args.phase == "status":
        result = status(args.round_id)
    elif args.phase == "audit":
        result = audit(args.round_id, write=True)
    else:
        result = live(args.round_id, allow_model_calls=args.allow_model_calls)
    public = {k: v for k, v in result.items() if k not in {"row_data", "provider"}}
    print(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

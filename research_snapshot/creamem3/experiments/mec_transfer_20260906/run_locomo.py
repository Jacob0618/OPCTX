"""Offline, gold-free LoCoMo selector pilot over the original frozen candidate pool.

No model/client is imported by this command. Source JSON files do contain
evaluator fields; only whitelisted question/graph/trace fields reach selection.
This is NOT an exact reproduction of the historical 81.75% OpCtx renderer.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "creamem3.mec_transfer.locomo.offline.v1"
SEED = "mec-transfer-20260906:"
RENDERER = "plain_facts_session_date.v1"
DATA_SHA = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
BEST_SHA = "66231c750ad8502a1e2e8c9fd0a641d34b9a5c1f2d19a57d2c4922faa6f8f66b"
FROZEN_CODE_SHA = "6e60aa6f8a4123f848221ba180f703f12c540268d0e17981f435d8a81947b8ee"
PLANS_SHA = "c2f964ab24f1c04080b823bd60cd0bdcf63b647ded2d43698802770936ebe0e7"
QA_SHA = "e84efdf3d7d115b04f547261a6683c60014606e39f41e33c13566c4edc71dfb5"
JUDGE_SHA = "9a2fb4e839be00c5c20a5c4f78782382b6695d740237698aa4c6706115a693ca"
GRAPH_SHA = {
    "conv-26": "ce8cfc53c63731bbbd6bd56fff81d24268ea5a61ef5500b0f1b009cd7cdc4fc7",
    "conv-30": "099d87ca3c59fca67960ad67e511500ebdc4052562d05f77e9a216e42d25911e",
    "conv-41": "be7f06805a62c62faf7bd5316b51023a5dd8eeee373b06651858a37108918db4",
    "conv-42": "a90fdc669af52ba23811b1ded3fea10dda01e786fddddbf866c04bd15183805c",
    "conv-43": "61287a2dfde01993e746f61c756cf9be1c8b230c4f9b013f6c912ddc490c4465",
    "conv-44": "f9f38c4cd93866fcc9b5e4bdbbeccf455bfcca507acd2764a65a2231a0303f19",
    "conv-47": "499a72562d8243ec358991acfab0c0d46ac1544c6ac83589a5d96e7fefbdcd9a",
    "conv-48": "ba5706093c0bc6ed28bca04efd40ba7198045d10c09a9cf8b38f2cfcc574ad79",
    "conv-49": "593f82dee8c436fb8ed92e7b2748638b7b65badd10f1ef43096bfadcbe1ea112",
    "conv-50": "c95da42532c6f205454da092ea583b906cb85b63cf29deceda7ed08f609dbfcd",
}
RESULT_SHA = {
    "conv-26": "9d877bc5e1fd562b318332edd7df7f321615ca910e3cb3e1798a842ad5363405",
    "conv-30": "556475141c0d1e102201661f21bdd58d6edf6c89718180b56f4c38cbf7983c1c",
    "conv-41": "87cbb29c48e6e03238ab5cb1872dd1028a2c17a9c313487bf57c4562416e4791",
    "conv-42": "234bf563e43d0dd6a1961793c357cd3d5139faae3a147bf8974aa15d701af0aa",
    "conv-43": "6f5e2f7b00d02543837c31d4562595cb98561512e9bd741a160a9f42cb41880f",
    "conv-44": "bbcd238395ea667f0812332421e84bd239115e8596c8dcc92969a64063608c15",
    "conv-47": "7af9f1ab86f328b6ebf4c9aa4a999a384076c63ae0bc7adfa7d46cefe4c1492c",
    "conv-48": "06f5c112012e39baa052e38db2edaf6e5b2eb3348523428a11dec423e768cff4",
    "conv-49": "ebf393b9ca86aa03d036b58518a613a5cedd9f9bb629c7772708bb8e96310fe8",
    "conv-50": "bb7fec2e17f3f02328e0fcadcb01cbf747b51cc35a7a67a18f4e301882fff0ae",
}


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_json(path: Path, expected: str | None = None) -> tuple[Any, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"frozen artifact hash drift: {path}")
    return json.loads(raw), digest


def frozen_bm25(root: Path = ROOT) -> type:
    path = root / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"
    if file_sha(path) != FROZEN_CODE_SHA:
        raise ValueError("frozen graph code drift")
    spec = importlib.util.spec_from_file_location("_mec_locomo_frozen_graph", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen BM25 implementation")
    module = importlib.util.module_from_spec(spec)
    # This frozen module contains pure graph code, not a provider client.
    spec.loader.exec_module(module)
    return module.BM25Index


def frozen_prompts(root: Path = ROOT) -> dict[str, str]:
    """Read literal prompt constants without importing the eval client stack."""
    tree = ast.parse((root / "eval/prompts.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"QA_PROMPT", "JUDGE_PROMPT"}:
                    values[target.id] = ast.literal_eval(node.value)
    for name, expected in (("QA_PROMPT", QA_SHA), ("JUDGE_PROMPT", JUDGE_SHA)):
        if text_sha(values.get(name, "")) != expected:
            raise ValueError(f"frozen prompt drift: {name}")
    return values


def pilot_ids(rows: Sequence[Mapping[str, Any]], limit: int = 24) -> list[str]:
    """Sampling has access only to row_id, never category/answer/correctness."""
    ids = [str(row["row_id"]) for row in rows]
    if len(ids) != len(set(ids)) or limit < 0:
        raise ValueError("duplicate row IDs or negative pilot size")
    return sorted(ids, key=lambda row_id: (text_sha(SEED + row_id), row_id))[:limit]


def group_header(fact: Mapping[str, Any]) -> str:
    date = str(fact.get("iso_date") or fact.get("timestamp") or "unknown")
    return f"[Session {int(fact['session_id'])}; date {date}]\n"


def fact_line(fact: Mapping[str, Any]) -> str:
    return f"[{fact['id']}] {fact['text']}\n"


def make_query_input(sample_id: str, row: Mapping[str, Any],
                     graph: Mapping[str, Any], lexical_scores: Mapping[str, float]) -> dict[str, Any]:
    """Strict whitelist: no gold, category, original responses, views or rerank IDs."""
    node_list = graph["fact_nodes"]
    nodes = {str(f["id"]): f for f in node_list}
    if len(nodes) != len(node_list):
        raise ValueError("duplicate graph fact ID")
    order = {str(f["id"]): i for i, f in enumerate(node_list)}
    trace = row["retrieval_trace"]
    if trace["policy"].get("candidate_cap") != 120:
        raise ValueError("not the historical candidate-cap120 protocol")
    candidates = trace["candidates"]
    if not 1 <= len(candidates) <= 120:
        raise ValueError("invalid frozen candidate count")
    ids = [str(c["fact_id"]) for c in candidates]
    if len(ids) != len(set(ids)) or any(fid not in nodes for fid in ids):
        raise ValueError("duplicate or foreign candidate ID")
    dense = {str(c["fact_id"]): float(c["dense_score"]) for c in candidates}
    if any(not math.isfinite(score) for score in dense.values()):
        raise ValueError("nonfinite frozen dense score")
    expected_order = sorted(ids, key=lambda fid: (-dense[fid], order[fid]))
    if ids != expected_order:
        raise ValueError("frozen candidate ordering drift")
    headers, facts, render_nodes = {}, [], {}
    for fid in ids:
        node = nodes[fid]
        sources = node.get("source_dia_ids")
        if not isinstance(sources, list) or not sources or any(not isinstance(s, str) for s in sources):
            raise ValueError(f"fact lacks source dialogue IDs: {fid}")
        if not isinstance(node.get("text"), str) or not node["text"].strip():
            raise ValueError(f"empty fact text: {fid}")
        if any(not s.startswith(f"D{int(node['session_id'])}:") for s in sources):
            raise ValueError(f"cross-session fact source mismatch: {fid}")
        group = f"S{int(node['session_id']):03d}"
        header = group_header(node)
        if group in headers and headers[group] != header:
            raise ValueError("inconsistent date within a session")
        headers[group] = header
        facts.append({"id": fid, "text": node["text"], "sources": list(sources),
                      "group": group, "cost": len(fact_line(node))})
        render_nodes[fid] = {"id": fid, "text": node["text"], "group": group}
    scores = [float(lexical_scores.get(fid, 0.0)) for fid in ids]
    if any(not math.isfinite(s) or s < 0 for s in scores):
        raise ValueError("invalid BM25 scores")
    return {"row_id": str(row["qa_id"]), "sample_id": sample_id,
            "question": str(row["question"]), "question_sha256": text_sha(str(row["question"])),
            "facts": facts, "dense_scores": [dense[fid] for fid in ids],
            "bm25_scores": scores, "group_costs": {g: len(h) for g, h in headers.items()},
            "group_headers": headers, "render_nodes": render_nodes,
            "candidate_ids": ids, "renderer": RENDERER}


def render_context(item: Mapping[str, Any], selected_ids: Sequence[str]) -> str:
    selected = set(selected_ids)
    if len(selected) != len(selected_ids) or not selected <= set(item["candidate_ids"]):
        raise ValueError("duplicate or out-of-pool selected ID")
    grouped: dict[str, list[str]] = defaultdict(list)
    # Display order is fixed to session number then frozen dense order for all arms.
    for fid in item["candidate_ids"]:
        if fid in selected:
            node = item["render_nodes"][fid]
            grouped[node["group"]].append(fact_line(node))
    return "".join(item["group_headers"][g] + "".join(grouped[g]) for g in sorted(grouped))


def additive_cost(item: Mapping[str, Any], selected_ids: Sequence[str]) -> int:
    lookup = {f["id"]: f for f in item["facts"]}
    return (sum(lookup[fid]["cost"] for fid in selected_ids)
            + sum(item["group_costs"][g] for g in {lookup[fid]["group"] for fid in selected_ids}))


def load_inputs(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset, data_hash = verified_json(root / "data/locomo/locomo10.json", DATA_SHA)
    best_hash = file_sha(root / "creamem3/results_autoresearch_v8/online_operation_context_v3/aggregate.json")
    if best_hash != BEST_SHA:
        raise ValueError("historical best aggregate drift")
    plans, plans_hash = verified_json(root / "creamem3/results_autoresearch_v8/semantic_plans_s02/plans.json", PLANS_SHA)
    plan_hashes = {r["qa_id"]: r["question_sha256"] for r in plans["rows"]}
    BM25Index = frozen_bm25(root)
    prompts = frozen_prompts(root)
    source_samples = {str(s["sample_id"]): s for s in dataset}
    inputs, bindings, fallback_ids = [], {}, []
    for sample_id, expected_hash in GRAPH_SHA.items():
        directory = root / "creamem3/results_locomo10_v3" / sample_id
        graph, graph_hash = verified_json(directory / "graph_online.json", expected_hash)
        result_path = directory / f"online_{sample_id}_full_v3_r1.json"
        result, result_hash = verified_json(result_path, RESULT_SHA[sample_id])
        source = source_samples[sample_id]
        valid_sources = {t["dia_id"] for key, turns in source["conversation"].items()
                         if re.fullmatch(r"session_\d+", key) for t in turns}
        for f in graph["fact_nodes"]:
            if not set(f["source_dia_ids"]) <= valid_sources:
                raise ValueError(f"graph references absent source dialogue: {sample_id}/{f['id']}")
        bm25 = BM25Index()
        for fact in graph["fact_nodes"]:
            bm25.add(fact["id"], fact["text"])
        for row in result["rows"]:
            qid, question = str(row["qa_id"]), str(row["question"])
            index = int(row["source_index"])
            if qid != f"{sample_id}-q{index:03d}" or source["qa"][index]["question"] != question:
                raise ValueError("source question binding drift")
            if plan_hashes.get(qid) != text_sha(question):
                raise ValueError("frozen plan/question binding drift")
            lexical = dict(bm25.search(question, len(graph["fact_nodes"])))
            # Verify the full BM25 recomputation against every stored top20 score.
            for hit in row["retrieval_trace"].get("bm25_facts", []):
                if not math.isclose(lexical.get(hit["fact_id"], 0), float(hit["score"]), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("frozen BM25 score drift")
            inputs.append(make_query_input(sample_id, row, graph, lexical))
            if row["retrieval_trace"].get("rerank", {}).get("fallback"):
                fallback_ids.append(qid)
        bindings[sample_id] = {"graph_path": str(directory / "graph_online.json"), "graph_sha256": graph_hash,
                               "result_path": str(result_path), "result_sha256": result_hash}
    if len(inputs) != 1540 or {r["row_id"] for r in inputs} != set(plan_hashes):
        raise ValueError("not the exact frozen 1540-question population")
    metadata = {"dataset_sha256": data_hash, "best_aggregate_sha256": best_hash,
                "plans_sha256": plans_hash, "frozen_graph_code_sha256": FROZEN_CODE_SHA,
                "bindings": bindings, "historical_rerank_fallback_ids": fallback_ids,
                "qa_prompt_sha256": text_sha(prompts["QA_PROMPT"]),
                "judge_prompt_sha256": text_sha(prompts["JUDGE_PROMPT"]),
                "candidate_count_distribution": dict(Counter(len(q["facts"]) for q in inputs)),
                "population_sha256": sha(inputs), "population_n": len(inputs)}
    return inputs, metadata


def selection_record(item: Mapping[str, Any], selected_ids: Sequence[str], *,
                     arm: str, k: int, budget: int, solver: Mapping[str, Any] | None = None) -> dict[str, Any]:
    context = render_context(item, selected_ids)
    actual_cost = len(context)
    if actual_cost != additive_cost(item, selected_ids):
        raise ValueError("renderer/additive cost mismatch")
    if len(selected_ids) > k or actual_cost > budget:
        raise ValueError("selector violated the hard fact/character budget")
    selected = set(selected_ids)
    fs = [f for f in item["facts"] if f["id"] in selected]
    from creamem3.budget_evidence_v1 import STOPWORDS, tokens
    requirements = tokens(item["question"]) - STOPWORDS
    pool_terms = set().union(*(tokens(f["text"]) for f in item["facts"]))
    selected_terms = set().union(*(tokens(f["text"]) for f in fs))
    return {"row_id": item["row_id"], "question": item["question"],
            "question_sha256": item["question_sha256"], "arm": arm, "k": k,
            "char_budget": budget, "actual_chars": actual_cost, "selected_n": len(selected_ids),
            "selected_ids": list(selected_ids), "selected_sources": {f["id"]: f["sources"] for f in fs},
            "selected_sessions": len({f["group"] for f in fs}), "context": context,
            "context_sha256": text_sha(context), "input_sha256": sha(item),
            "literal_diagnostic": {"requirements": sorted(requirements),
                "covered": sorted(requirements & selected_terms),
                "missing": sorted(requirements - selected_terms),
                "unsupported_in_pool": sorted(requirements - pool_terms),
                "answer_completeness_claim": False},
            "solver": dict(solver) if solver is not None else None}


def run_offline(inputs: Sequence[Mapping[str, Any]], *, limit: int = 24,
                time_limit: float = 5.0, selector: Callable[..., Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    if selector is None:
        from creamem3.budget_evidence_v1 import select_evidence
        selector = select_evidence
    by_id = {r["row_id"]: r for r in inputs}
    records = []
    for row_id in pilot_ids(inputs, limit):
        item = by_id[row_id]
        for k in (20, 50):
            dense_ids = list(item["candidate_ids"][:k])
            budget = len(render_context(item, dense_ids))
            records.append(selection_record(item, dense_ids, arm="dense", k=k, budget=budget))
            result = selector(facts=item["facts"], question=item["question"], dense_scores=item["dense_scores"],
                              bm25_scores=item["bm25_scores"], max_facts=k, char_budget=budget,
                              group_costs=item["group_costs"], time_limit=time_limit)
            records.append(selection_record(item, result["selected_ids"], arm="budget_evidence", k=k,
                                            budget=budget, solver=result))
    return records


def future_qa_request(record: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Provider-neutral request specification; does NOT execute a model call."""
    if text_sha(str(record["context"])) != record["context_sha256"]:
        raise ValueError("sealed context drift")
    if text_sha(str(record["question"])) != record["question_sha256"]:
        raise ValueError("sealed question drift")
    return {"prompt": frozen_prompts(root)["QA_PROMPT"].format(retrieved_texts=record["context"], question=record["question"]),
            "model": "gemini-2.5-flash", "temperature": 0.0, "max_tokens": 8192,
            "thinking_budget": 0, "tag": "read", "use_cache": True}


def future_judge_request(question: str, answer: str, response: str, root: Path = ROOT) -> dict[str, Any]:
    """Gold is accepted only by this separate, future post-QA judge function."""
    return {"prompt": frozen_prompts(root)["JUDGE_PROMPT"].format(question=question, answer=answer, response=response),
            "model": "gemini-3-flash-preview", "temperature": 0.0, "max_tokens": 2048,
            "thinking_budget": 0, "tag": "judge", "use_cache": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--limit", type=int, default=24, help="Score-free hash panel size; 0 only constructs all 1540 inputs")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Seconds per solver stage, not whole question")
    args = parser.parse_args(argv)
    if not 0 <= args.limit <= 1540:
        parser.error("--limit must lie in [0,1540]")
    if args.output.exists():
        parser.error("output already exists; use a new directory")
    adapter_hash = file_sha(Path(__file__))
    selector_path = ROOT / "creamem3/budget_evidence_v1.py"
    selector_hash = file_sha(selector_path)
    inputs, metadata = load_inputs()
    print(f"Validated {len(inputs)} frozen inputs; optimizing {args.limit} score-free questions", flush=True)
    records = run_offline(inputs, limit=args.limit, time_limit=args.time_limit)
    if file_sha(Path(__file__)) != adapter_hash or file_sha(selector_path) != selector_hash:
        raise ValueError("adapter/selector changed during run; refusing to seal artifacts")
    metadata.update({"schema": SCHEMA, "renderer": RENDERER, "pilot_seed": SEED,
                     "pilot_ids": pilot_ids(inputs, args.limit), "qa_executed": False,
                     "api_calls": 0, "time_limit_per_stage": args.time_limit,
                     "adapter_sha256": adapter_hash, "selector_sha256": selector_hash,
                     "label_boundary": "Source JSON may contain evaluator fields; only whitelisted question/graph/trace fields are passed to selector. No labels are used to sample, optimize or score this offline pilot.",
                     "selection_manifest_sha256": sha(records),
                     "interpretation": "Within frozen cap120; plain facts and matched characters, not old OpCtx renderer"})
    metadata["summary"] = {f"{arm}_k{k}": {"n": len(part),
        "mean_selected": sum(r["selected_n"] for r in part) / max(len(part), 1),
        "mean_chars": sum(r["actual_chars"] for r in part) / max(len(part), 1),
        "mean_literal_covered": sum(len(r["literal_diagnostic"]["covered"]) for r in part) / max(len(part), 1),
        "mean_literal_missing": sum(len(r["literal_diagnostic"]["missing"]) for r in part) / max(len(part), 1),
        "certified_optimal": sum(bool((r.get("solver") or {}).get("certified_optimal")) for r in part),
        "solver_seconds": sum(float((r.get("solver") or {}).get("seconds", 0)) for r in part)}
        for arm in ("dense", "budget_evidence") for k in (20, 50)
        for part in [[r for r in records if r["arm"] == arm and r["k"] == k]]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("inputs.jsonl", inputs), ("selections.jsonl", records)):
        with (args.output / name).open("x", encoding="utf-8", newline="\n") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

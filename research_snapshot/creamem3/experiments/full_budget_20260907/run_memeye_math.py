"""Full371, unchanged raw MemEye Reader, frozen Dense vs math selection <=50."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from creamem3 import budget_evidence_v1 as core
from creamem3 import graph_memory_v2 as memory
from creamem3.experiments.mec_transfer_20260906 import run_memeye as old

direct, raw = old.direct, old.raw
ROOT = Path(r"D:\creamem3_cache\memeye_v19\full_budget_20260907_r1\math50")
PROTOCOL = Path(__file__).with_name("autoresearch.md")
ARMS = ("dense50", "math50")
CORE_HASH = "475e1c379b8a264ff4e3e6613948e984498a7d16d4a0304630c97a0b56fa52dc"


def seal(path, value):
    result = dict(value)
    result["seal_sha256"] = direct._sha256(result)
    direct._write_once(path, result)
    return result


def read_seal(path):
    result = direct._read_object(path, "full-corpus experiment seal")
    body = dict(result)
    sha = body.pop("seal_sha256", None)
    if sha != direct._sha256(body):
        raise RuntimeError("artifact seal mismatch")
    return result


def full_rows(rows):
    if len(rows) != 371 or len({r["row_id"] for r in rows}) != 371:
        raise RuntimeError("require the complete371 unique questions; sampling forbidden")
    if any(old.FORBIDDEN.intersection(str(k).lower() for k in row) for row in rows):
        raise RuntimeError("evaluator fields at selector boundary")
    return [dict(r) for r in rows]


def hashes():
    if direct._file_sha256(Path(core.__file__)) != CORE_HASH:
        raise RuntimeError("original mathematical method changed")
    paths = [Path(__file__), PROTOCOL, Path(core.__file__), Path(old.__file__),
             Path(raw.__file__), Path(direct.__file__), Path(memory.__file__),
             Path(direct.retry5.__file__)]
    return {str(p): direct._file_sha256(p) for p in paths}


def register():
    direct._validate_roots(ROOT / "validation_results", ROOT / "validation_cache")
    canonical = direct._canonical_sources()
    reg = seal(ROOT / "preregistration.json", {"schema": "full_memeye_math.v1",
        "source_hashes": hashes(), "canonical_sources": canonical["receipt"],
        "rows": full_rows(canonical["query_rows"]), "arms": list(ARMS),
        "complete_dataset": True, "rows_total": 371, "max_facts": 50,
        "char_budget_rule": "actual same-row raw-rendered globalDense50 length",
        "core_budget": "renderedDense50+1 due final newline accounting",
        "selection_time_limit_per_stage": 5.0, "reranker_calls": 0,
        "new_caption_writer_online_embedding_calls": 0, "no_gold_tuning": True})
    return reg, canonical


def validate():
    reg = read_seal(ROOT / "preregistration.json")
    canonical = direct._canonical_sources()
    if (reg["source_hashes"] != hashes() or reg["canonical_sources"] != canonical["receipt"]
            or reg["rows"] != full_rows(canonical["query_rows"])):
        raise RuntimeError("frozen method or full input contract changed")
    return reg, canonical


def prepare():
    if (ROOT / "preregistration.json").is_file():
        reg, canonical = validate()
    else:
        reg, canonical = register()
    if (ROOT / "selection_complete.json").is_file():
        return validate_selection()[0]
    graphs, indices, graph_receipts = {}, {}, {}
    output = {a: [] for a in ARMS}
    diagnostics = []
    for ordinal, query in enumerate(reg["rows"], 1):
        task, row_id = str(query["task"]), str(query["row_id"])
        if task not in graphs:
            path, sha = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
            graphs[task] = direct._read_object(path, "unchanged original Online graph")
            indices[task] = memory.GraphMemoryIndex(graphs[task], candidate_cap=len(graphs[task]["fact_nodes"]), rerank_top_k=50)
            graph_receipts[task] = {"path": str(path), "sha256": sha}
        graph, index = graphs[task], indices[task]
        dense = list(map(float, index._scores(index.fact_matrix, canonical["vector_by_id"][row_id])))
        bm = {str(fid): float(score) for fid, score in index.bm25.search(query["question"], len(index.facts))}
        lexical = [bm.get(str(f["id"]), 0.0) for f in index.facts]
        if [f["id"] for f in index.facts] != [f["id"] for f in graph["fact_nodes"]]:
            raise RuntimeError("graph/index column drift")
        identity = {"row_id": row_id, "question_sha256": direct._sha256(query["question"]),
                    "graph": graph_receipts[task], "score_sha256": direct._sha256([dense, lexical]),
                    "preregistration_sha256": reg["seal_sha256"]}
        row_path = ROOT / "selection_rows" / (direct._sha256(row_id) + ".json")
        if row_path.is_file():
            saved = read_seal(row_path)
            if saved["identity"] != identity:
                raise RuntimeError("selection row resume binding mismatch")
        else:
            selection = old.select_row(graph=graph, question=query["question"], dense_scores=dense,
                bm25_scores=lexical, selector=core.select_evidence, time_limit=5.0)
            saved = seal(row_path, {"identity": identity, "selection": selection})
        chosen = saved["selection"]
        for arm, source in (("dense50", "global_dense50"), ("math50", "budget_evidence50")):
            item = chosen[source]
            ids = item["selected_ids"]
            context, render = old.render_selection(graph, ids)
            if render != item["render"] or len(context) > chosen["char_budget"]:
                raise RuntimeError("context or hard budget mismatch")
            trace = {"pool_sha256": chosen["pool_sha256"], "score_arrays_sha256": chosen["score_arrays_sha256"],
                "char_budget": chosen["char_budget"], "render": render, "core": item.get("core"),
                "full_selection_row_sha256": saved["seal_sha256"]}
            output[arm].append({"row_id": row_id, "selected_fact_ids": ids,
                "selected_fact_ids_sha256": direct._sha256(ids), "trace": trace,
                "trace_sha256": direct._sha256(trace)})
        diagnostics.append({"row_id": row_id, "dense_facts": len(chosen["global_dense50"]["selected_ids"]),
            "new_facts": len(chosen["budget_evidence50"]["selected_ids"]), "budget": chosen["char_budget"],
            "new_chars": chosen["budget_evidence50"]["render"]["context_chars"],
            "optimal": chosen["budget_evidence50"]["core"]["certified_optimal"],
            "solver_seconds": chosen["budget_evidence50"]["core"]["seconds"]})
        if ordinal % 25 == 0 or ordinal == 371:
            print(f"sealed selection {ordinal}/371", flush=True)
    manifest_hashes = {}
    for arm in ARMS:
        path = ROOT / "manifests" / f"{arm}.json"
        direct._write_once(path, old._manifest(arm, output[arm], reg["seal_sha256"]))
        manifest_hashes[arm] = direct._file_sha256(path)
    return seal(ROOT / "selection_complete.json", {"rows_total": 371,
        "manifest_hashes": manifest_hashes, "graphs": graph_receipts, "diagnostics": diagnostics,
        "preregistration_sha256": reg["seal_sha256"], "network_calls": 0})


def validate_selection():
    reg, canonical = validate()
    complete = read_seal(ROOT / "selection_complete.json")
    if complete["preregistration_sha256"] != reg["seal_sha256"] or complete["rows_total"] != 371:
        raise RuntimeError("complete scope mismatch")
    for task, receipt in complete["graphs"].items():
        path, sha = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
        if receipt != {"path": str(path), "sha256": sha}:
            raise RuntimeError("graph drift")
    for arm in ARMS:
        path = ROOT / "manifests" / f"{arm}.json"
        if direct._file_sha256(path) != complete["manifest_hashes"][arm]:
            raise RuntimeError("manifest byte drift")
        body = direct._read_object(path, "full371 selection")
        sha = body.pop("manifest_sha256")
        if (sha != direct._sha256(body) or body["preregistration_sha256"] != reg["seal_sha256"]
                or [r["row_id"] for r in body["rows"]] != [r["row_id"] for r in reg["rows"]]):
            raise RuntimeError("manifest panel/contract drift")
        direct._selection_rows(path)
    return complete, reg


def gate(arm, workers):
    complete, reg = validate_selection()
    value = raw.preflight(selection_path=ROOT / "manifests" / f"{arm}.json",
        output_root=ROOT / arm / "results", cache_root=ROOT / arm / "provider_cache", workers=workers)
    if (len(value["prepared"]) != 371 or {r["query"]["row_id"] for r in value["prepared"]}
            != {r["row_id"] for r in reg["rows"]}):
        raise RuntimeError("reader is not full371")
    for row in value["prepared"]:
        if row["context_sha256"] != row["selection"]["trace"]["render"]["context_sha256"]:
            raise RuntimeError("QA did not receive frozen context")
    return value


def run(arm, workers):
    value = gate(arm, workers)
    out, cache = Path(value["output_root"]), Path(value["cache_root"])
    prediction = direct.run_predict(output_root=out, cache_root=cache, contract=value["contract"],
        authority=value["authority"], prepared=value["prepared"], workers=workers)
    score = direct.run_score(output_root=out, cache_root=cache, contract=value["contract"],
        prepared=value["prepared"], workers=workers)
    return {"arm": arm, "prediction_rows": prediction.get("rows_total", 371),
            "score": score["mapped_mean"], "complete": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "preflight", "live"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--allow-model-calls", action="store_true")
    args = p.parse_args()
    if args.allow_model_calls != (args.phase == "live") or not 1 <= args.workers <= 12:
        p.error("live alone needs explicit allow-model-calls; workers1..12")
    if args.phase == "prepare":
        result = prepare()
        print("SELECTION_COMPLETE", result["rows_total"], result["seal_sha256"], flush=True)
    else:
        if args.arm is None: p.error("arm required")
        if args.phase == "preflight":
            value = gate(args.arm, args.workers)
            result = {"rows": len(value["prepared"]), "lengths": value["length_summary"], "network_calls": 0}
        else: result = run(args.arm, args.workers)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__": main()

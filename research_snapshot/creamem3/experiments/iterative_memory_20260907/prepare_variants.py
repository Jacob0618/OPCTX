"""Score-blind full371 feature preparation and initial frozen proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from creamem3 import graph_memory_v2 as memory
from creamem3.experiments.full_budget_20260907 import run_memeye_math as previous

direct = previous.direct
ROOT = Path(r"D:\creamem3_cache\memeye_v19\iterative_memory_20260907")


def seal(path: Path, value: dict) -> dict:
    out = dict(value)
    out["seal_sha256"] = direct._sha256(value)
    direct._write_once(path, out)
    return out


def read_seal(path: Path) -> dict:
    out = direct._read_object(path, "iteration feature seal")
    body = dict(out)
    sha = body.pop("seal_sha256")
    if sha != direct._sha256(body):
        raise RuntimeError("feature seal drift")
    return out


def initial_proposals() -> list[str]:
    _, reg = previous.validate_selection()
    selection_path = previous.ROOT / "manifests" / "math50.json"
    source = direct._read_object(selection_path, "prior full371 math manifest")
    selections = {r["row_id"]: r for r in source["rows"]}
    paths = []
    for rid, method, focused, fused, order in (
        ("r01", "math50_reader_output8192", 8192, 8192, "original"),
        ("r02", "math50_physical_observation_order", 2600, 1200, "observation"),
    ):
        rows = []
        for q in reg["rows"]:
            selected = selections[q["row_id"]]
            rows.append({"row_id": q["row_id"], "question": q["question"], "task": q["task"],
                "selected_fact_ids": selected["selected_fact_ids"],
                "selection_provenance": {"selector": "previous_frozen_math50",
                    "source_manifest_sha256": direct._file_sha256(selection_path),
                    "source_selection_sha256": selected["selected_fact_ids_sha256"]}})
        manifest = {"schema": "memeye.iterative.context_manifest.v1", "round_id": rid,
            "method": {"name": method, "description": "Same frozen math50 selections; only the named Reader ceiling or observation-order intervention changes."},
            "reader": {"focused_max_output_tokens": focused, "fused_max_output_tokens": fused},
            "render_order": order, "rows": rows}
        path = ROOT / "proposals" / f"{rid}.json"
        direct._write_once(path, manifest)
        paths.append(str(path))
    return paths


def prepare_features() -> dict:
    canonical = direct._canonical_sources()
    queries = previous.full_rows(canonical["query_rows"])
    complete_path = ROOT / "features" / "complete.json"
    if complete_path.exists():
        ready = read_seal(complete_path)
        if ready["canonical_receipt"] != canonical["receipt"]:
            raise RuntimeError("canonical source drift")
        for rec in ready["files"]:
            if direct._file_sha256(Path(rec["path"])) != rec["sha256"]:
                raise RuntimeError("feature file drift")
        return ready
    indices, files = {}, []
    graph_receipts = {}
    for ordinal, q in enumerate(queries, 1):
        task, qid = q["task"], q["row_id"]
        if task not in indices:
            path, sha = direct.predict._artifact_for(canonical["build"], direct.BACKBONE, direct.ARCHITECTURE, task)
            graph = direct._read_object(path, "original graph, no evaluations")
            index = memory.GraphMemoryIndex(graph, candidate_cap=len(graph["fact_nodes"]), rerank_top_k=50)
            indices[task] = index
            graph_receipts[task] = {"path": str(path), "sha256": sha}
        index = indices[task]
        dense = list(map(float, index._scores(index.fact_matrix, canonical["vector_by_id"][qid])))
        lexical = {str(fid): float(score) for fid, score in index.bm25.search(q["question"], len(index.facts))}
        data = {"query": q, "graph": graph_receipts[task],
            "fact_ids": [str(f["id"]) for f in index.facts],
            "dense_scores": dense, "bm25_scores": [lexical.get(str(f["id"]), 0.0) for f in index.facts],
            "dense_definition": "original cached3072 graph and query embeddings; original cosine implementation",
            "new_provider_calls": 0}
        path = ROOT / "features" / "queries" / f"{direct._sha256(qid)}.json"
        seal(path, data)
        files.append({"row_id": qid, "path": str(path), "sha256": direct._file_sha256(path)})
        if ordinal % 50 == 0 or ordinal == 371:
            print(f"feature rows {ordinal}/371", flush=True)
    return seal(complete_path, {"schema": "memeye.iterative.full_features.v1", "population_n": 371,
        "canonical_receipt": canonical["receipt"], "queries": queries, "graphs": graph_receipts,
        "files": files, "new_provider_calls": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("initial", "features"))
    args = parser.parse_args()
    result = initial_proposals() if args.phase == "initial" else prepare_features()
    print(json.dumps(result if isinstance(result, list) else {"population_n": result["population_n"], "seal_sha256": result["seal_sha256"]}))


if __name__ == "__main__":
    main()

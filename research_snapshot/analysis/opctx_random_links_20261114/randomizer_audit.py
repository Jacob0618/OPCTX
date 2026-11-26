"""Offline structural comparison only; never runs selection, QA, or any API.

The approved experiment continues to use the historical global permutation.
Independent draws and double-edge swaps below are descriptive alternatives,
fixed before QA and not selected according to answer quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[2]
SEED = 92620
PERMUTATIONS = Path("D:/opctx_review26/main_align/locomo_r2/permutations.json")
ACCEPTANCE = PERMUTATIONS.with_name("acceptance.json")
FROZEN = WORKSPACE / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"
GRAPH_ROOT = WORKSPACE / "creamem3/results_locomo10_v3"
DEFAULT_OUTPUT = Path("D:/opctx_random_links_20261114/randomizer_audit_r1.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_for(sid):
    # Registered explicitly; this derivation is for the two offline alternatives,
    # not a reconstruction of the already-recorded historical permutation.
    salt = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(np.random.SeedSequence([SEED, salt]))


def summarize(original, randomized, fact_ids):
    views = list(original)
    pairs = list(itertools.combinations(views, 2))
    old_degree = Counter(f for members in original.values() for f in members)
    new_degree = Counter(f for members in randomized.values() for f in members)
    n_links = sum(map(len, original.values()))
    retained = sum(len(original[v] & randomized[v]) for v in views)
    return {
        "views": len(views), "links": n_links,
        "retained_original_edge_pairs": retained,
        "replaced_original_edge_pairs": n_links - retained,
        "unchanged_view_member_sets": sum(original[v] == randomized[v] for v in views),
        "per_view_degree_preserved": all(len(original[v]) == len(randomized[v]) for v in views),
        "each_labeled_fact_degree_preserved": old_degree == new_degree,
        "unlabeled_fact_degree_histogram_preserved":
            Counter(old_degree[f] for f in fact_ids) == Counter(new_degree[f] for f in fact_ids),
        "labeled_fact_degrees_changed": sum(old_degree[f] != new_degree[f] for f in fact_ids),
        "view_pairs": len(pairs),
        "view_pair_overlap_sizes_changed": sum(
            len(original[a] & original[b]) != len(randomized[a] & randomized[b]) for a, b in pairs),
        "pairwise_overlap_total": sum(len(randomized[a] & randomized[b]) for a, b in pairs),
        "distinct_linked_facts": len(set().union(*randomized.values())),
    }


def swaps(original, sid):
    """Fixed 20E accepted bipartite switches; not claimed exactly uniform/mixed."""
    randomized = {v: set(fs) for v, fs in original.items()}
    edges = [(v, f) for v in original for f in sorted(original[v])]
    rng = rng_for(sid)
    target, accepted, attempts = 20 * len(edges), 0, 0
    while accepted < target and attempts < 1000 * len(edges):
        i, j = map(int, rng.choice(len(edges), size=2, replace=False))
        attempts += 1
        v, f = edges[i]
        w, h = edges[j]
        if v == w or f == h or h in randomized[v] or f in randomized[w]:
            continue
        randomized[v].remove(f)
        randomized[v].add(h)
        randomized[w].remove(h)
        randomized[w].add(f)
        edges[i], edges[j] = (v, h), (w, f)
        accepted += 1
    if accepted != target:
        raise RuntimeError(f"Fixed swap count not reached in {sid}")
    return randomized, {"accepted_swaps": accepted, "proposal_attempts": attempts}


def audit():
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    spec = importlib.util.spec_from_file_location("_randomizer_frozen", FROZEN)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    permutations = read(PERMUTATIONS)
    expected = {k.replace("\\", "/"): v for k, v in read(ACCEPTANCE)["source_hashes"].items()}
    hashes = {str(PERMUTATIONS): sha(PERMUTATIONS), str(FROZEN): sha(FROZEN),
              str(ACCEPTANCE): sha(ACCEPTANCE)}
    rows = []
    for gp in sorted(GRAPH_ROOT.glob("conv-*/graph_online.json")):
        graph_hash = sha(gp)
        assert graph_hash == expected[str(gp).replace("\\", "/")]
        hashes[str(gp)] = graph_hash
        sid, graph = gp.parent.name, read(gp)
        index = frozen.GraphMemoryIndex(graph)
        facts = list(index.fact_by_id)
        views = [v["id"] for v in index.summaries if v.get("node_type") == "event_view"]
        original = {v: set(index.adjacency[v]) for v in views}
        perm = permutations[sid][str(SEED)]
        assert set(perm) == set(perm.values()) == set(facts)
        global_permutation = {v: {perm[f] for f in fs} for v, fs in original.items()}
        rng = rng_for(sid)
        independent = {v: set(rng.choice(facts, size=len(original[v]), replace=False).tolist())
                       for v in views}
        switched, switch_stats = swaps(original, sid)
        methods = {"original": summarize(original, original, facts),
                   "global_permutation": summarize(original, global_permutation, facts),
                   "independent_per_view": summarize(original, independent, facts),
                   "double_edge_swaps": summarize(original, switched, facts) | switch_stats}
        methods["global_permutation"]["fixed_fact_permutation_points"] = sum(perm[f] == f for f in facts)
        methods["global_permutation"]["pointwise_edge_endpoints_unchanged"] = sum(
            perm[f] == f for members in original.values() for f in members)
        methods["global_permutation"]["pointwise_edge_endpoints_changed"] = sum(
            perm[f] != f for members in original.values() for f in members)
        for values in methods.values():
            assert values["per_view_degree_preserved"]
        assert methods["global_permutation"]["view_pair_overlap_sizes_changed"] == 0
        assert methods["global_permutation"]["unlabeled_fact_degree_histogram_preserved"]
        assert methods["double_edge_swaps"]["each_labeled_fact_degree_preserved"]
        assert sha(gp) == graph_hash
        rows.append({"conversation": sid, "facts": len(facts), "methods": methods})
    assert len(rows) == 10
    aggregate = {}
    for name in rows[0]["methods"]:
        keys = rows[0]["methods"][name]
        aggregate[name] = {
            key: (all(row["methods"][name][key] for row in rows)
                  if isinstance(keys[key], bool)
                  else sum(row["methods"][name][key] for row in rows))
            for key in keys
        }
        aggregate[name]["original_edge_replacement_percent"] = (
            100 * aggregate[name]["replaced_original_edge_pairs"] / aggregate[name]["links"])
    gp = aggregate["global_permutation"]
    assert (gp["views"], gp["links"], gp["pointwise_edge_endpoints_changed"],
            gp["retained_original_edge_pairs"]) == (243, 5567, 5558, 619)
    return {
        "status": "PASS", "schema": "opctx.randomizer.structural_audit.v1",
        "model_calls": 0, "qa_accuracies_measured": False,
        "approved_qa_intervention": "historical_global_permutation_only",
        "seed": SEED, "conversations": len(rows), "facts": sum(r["facts"] for r in rows),
        "alternative_rng": "numpy.default_rng(SeedSequence([92620, int(sha256(sid UTF-8)[:8],16)]))",
        "alternative_ordering": "historical summary-node order, historical fact-node order; switch edges sorted by fact ID within view",
        "independent_sampling": "separate uniform without-replacement draw of the original degree from all same-conversation facts for each view",
        "switch_sampling": "20E accepted simple bipartite double-edge switches per conversation; maximum1000E proposals; no mixing/uniformity certificate claimed",
        "scope": "route endpoints only; no graph or Reader/provenance modifications, no new QA, no score-dependent seed selection",
        "distinction": "Pointwise movement counts whether a particular original endpoint moved; retained edge pairs can survive when another endpoint lands on an already-member fact.",
        "source_sha256": hashes, "code_sha256": sha(__file__),
        "aggregate": aggregate, "per_conversation": rows,
    }


def no_network(*args, **kwargs):
    raise AssertionError("Offline audit attempted network access")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-default", action="store_true")
    args = parser.parse_args()
    destination = DEFAULT_OUTPUT if args.write_default else args.output
    if destination is not None and destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    with patch("socket.socket.connect", no_network), patch("socket.create_connection", no_network):
        report = audit()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": report["status"], "model_calls": 0,
                      "output": str(destination) if destination else None,
                      "aggregate": report["aggregate"]}, indent=2))

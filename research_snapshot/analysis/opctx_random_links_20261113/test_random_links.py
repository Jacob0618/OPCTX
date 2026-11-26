"""Offline, fail-closed audit for the fixed-seed LoCoMo routing intervention.

This module never calls a model, modifies a historical file, or writes a report.
Without --runner it audits the independent reference routing-index constructor.
With --runner PATH it audits PATH.make_routing_index(graph, permutation) instead.
The Reader always receives the original, unmodified graph in this experiment.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import itertools
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

SEED = "92620"
OLD = Path(r"D:\opctx_review26\main_align\locomo_r2")
GRAPH_ROOT = ROOT / "creamem3/results_locomo10_v3"
FINAL_ROOT = ROOT / "creamem3/results_autoresearch_v8/online_operation_context_v3"
FROZEN = ROOT / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fail_network(*args, **kwargs):
    raise AssertionError("Offline audit attempted a network connection")


def reference_routing_index(graph, permutation):
    """Change only runtime routing adjacency, not stored membership/provenance."""
    cls = load_module("_random_links_reference_graph", FROZEN).GraphMemoryIndex
    index = cls(graph)
    for view in index.summaries:
        if view.get("node_type") == "event_view":
            index.adjacency[view["id"]] = [
                permutation[fid] for fid in index.adjacency[view["id"]]
            ]
    return index


def assert_equal(left, right, label):
    if left != right:
        raise AssertionError(label)


def audit(factory=None):
    """Return JSON-serializable evidence; raises on every contract violation."""
    from creamem3.online_operation_context_v8 import materialize_online_operation_context

    factory = factory or reference_routing_index
    frozen = load_module("_random_links_audit_frozen_graph", FROZEN)
    permutations = read(OLD / "permutations.json")
    acceptance = read(OLD / "acceptance.json")
    expected_hashes = {str(k).replace("\\", "/"): v for k, v in acceptance["source_hashes"].items()}
    graph_paths = sorted(GRAPH_ROOT.glob("conv-*/graph_online.json"))
    assert_equal(len(graph_paths), 10, "expected ten original conversation graphs")
    totals = collections.Counter()
    conversations = []
    policy_counts = collections.Counter()
    policy_label_changes = collections.Counter()

    for graph_path in graph_paths:
        sid = graph_path.parent.name
        graph = read(graph_path)
        graph_before = digest(graph)
        expected = expected_hashes[str(graph_path).replace("\\", "/")]
        assert_equal(hashlib.sha256(graph_path.read_bytes()).hexdigest(), expected, sid + " graph SHA drift")
        original = frozen.GraphMemoryIndex(graph)
        perm = permutations[sid][SEED]
        fact_ids = set(original.fact_by_id)
        assert_equal(set(perm), fact_ids, sid + " permutation domain")
        assert_equal(set(perm.values()), fact_ids, sid + " permutation codomain")
        randomized = factory(graph, perm)
        assert_equal(digest(graph), graph_before, sid + " factory mutated source graph")
        assert_equal(digest(randomized.graph), graph_before, sid + " routing index altered stored graph")
        assert_equal(digest(randomized.summaries), digest(original.summaries), sid + " summaries/provenance changed")
        assert_equal(digest(randomized.facts), digest(original.facts), sid + " facts changed")
        np.testing.assert_array_equal(randomized.summary_matrix, original.summary_matrix)
        np.testing.assert_array_equal(randomized.fact_matrix, original.fact_matrix)
        for attr in ("summary_top_k", "dense_fact_top_k", "bm25_fact_top_k", "candidate_cap", "rerank_top_k"):
            assert_equal(getattr(randomized, attr), getattr(original, attr), sid + " changed " + attr)
        assert_equal((original.summary_top_k, original.dense_fact_top_k, original.bm25_fact_top_k,
                      original.candidate_cap, original.rerank_top_k), (10, 20, 20, 120, 20), sid + " original limits")
        views = [v for v in original.summaries if v.get("node_type") == "event_view"]
        old_sets, new_sets = {}, {}
        for node in original.summaries:
            nid = node["id"]
            old_members = original.adjacency[nid]
            if node.get("node_type") == "event_view":
                expected_members = [perm[fid] for fid in old_members]
                old_sets[nid] = set(old_members)
                new_sets[nid] = set(randomized.adjacency[nid])
                assert_equal(len(expected_members), len(set(expected_members)), sid + " duplicate random links")
                totals["event_links"] += len(old_members)
                assert_equal(len(old_members), len(randomized.adjacency[nid]), sid + " per-view degree")
            else:
                expected_members = old_members
            assert_equal(randomized.adjacency[nid], expected_members, sid + " wrong routing adjacency " + nid)
        for left, right in itertools.combinations(old_sets, 2):
            assert_equal(len(old_sets[left] & old_sets[right]), len(new_sets[left] & new_sets[right]),
                         sid + " pairwise view-overlap changed")
            totals["pairwise_overlap_checks"] += 1

        # Every view supplies an artificial query vector, independent of gold QA.
        # These are local routing fixtures, never submitted as evaluation questions.
        seen_views = set()
        changed_candidate_fixtures = 0
        question = "Caroline Melanie family community education employment activity"
        for view in views:
            _, normal_trace = original._candidate_pool(question, view["embedding"])
            _, random_trace = randomized._candidate_pool(question, view["embedding"])
            for key in ("policy", "top_summaries", "dense_fact_ids", "bm25_facts"):
                assert_equal(normal_trace[key], random_trace[key], sid + " changed independent retrieval branch " + key)
            seen_views.update(n["summary_id"] for n in normal_trace["top_summaries"] if n["node_type"] == "event_view")
            expected_union = set(random_trace["dense_fact_ids"]) | {r["fact_id"] for r in random_trace["bm25_facts"]}
            for routed in random_trace["top_summaries"]:
                expected_union.update(randomized.adjacency[routed["summary_id"]])
            scores = original._scores(original.fact_matrix, view["embedding"])
            expected_ids = sorted(expected_union, key=lambda fid: (-float(scores[original.fact_order[fid]]), original.fact_order[fid]))[:120]
            actual_ids = [r["fact_id"] for r in random_trace["candidates"]]
            assert_equal(actual_ids, expected_ids, sid + " random candidate union/order")
            assert len(actual_ids) <= 120
            assert len(actual_ids) >= 20
            changed_candidate_fixtures += actual_ids != [r["fact_id"] for r in normal_trace["candidates"]]
            totals["candidate_fixtures"] += 1
        assert_equal(seen_views, set(old_sets), sid + " not every event view exercised by synthetic queries")
        assert changed_candidate_fixtures > 0, sid + " no routing fixture changed candidates"

        # Independent cold replay: both raw assembly and operation Reader context.
        source = {r["qa_id"]: r for r in read(GRAPH_ROOT / sid / ("online_" + sid + "_full_v3_r1.json"))["rows"]}
        finals = read(FINAL_ROOT / sid / "online_operation_context_v3.json")["rows"]
        replay_count = 0
        for result in finals:
            if result["category_name"] != "multi":
                continue
            row = source[result["qa_id"]]
            trace = copy.deepcopy(row["retrieval_trace"])
            selected = [{"fact": original.fact_by_id[fid]} for fid in trace["rerank"]["selected_fact_ids"]]
            assert_equal(len(selected), 20, result["qa_id"] + " selected fact budget")
            assembled, _ = original._assemble(selected, trace["top_summaries"])
            assert_equal(assembled, row["retrieved"], result["qa_id"] + " raw original assembly parity")
            history = result["retrieval_trace"]["online_operation_context_v8"]
            operation = history["strict_operation"]
            sanitized = {"question": row["question"], "retrieved": assembled, "retrieval_trace": trace}
            context, current_audit = materialize_online_operation_context(operation, sanitized, graph)
            assert_equal(context, result["retrieved"], result["qa_id"] + " exact Reader context parity")
            old_audit = history["materializer_audit"]
            assert_equal(current_audit["selected_fact_ids"], old_audit["selected_fact_ids"], result["qa_id"] + " Reader selected IDs/order")
            assert_equal(current_audit["rendered_event_view_ids"], old_audit["rendered_event_view_ids"], result["qa_id"] + " Reader rendered views")
            if old_audit.get("rendered_event_view_claims") is not None:
                assert_equal(current_audit["rendered_event_view_claims"], old_audit["rendered_event_view_claims"], result["qa_id"] + " Reader rendered claim provenance")
            policy_counts[old_audit["context_policy"]] += 1
            if old_audit["context_policy"] != current_audit["context_policy"]:
                policy_label_changes[old_audit["context_policy"] + " -> " + current_audit["context_policy"]] += 1
            replay_count += 1
        assert_equal(digest(graph), graph_before, sid + " graph changed during candidate/Reader checks")
        totals["graphs"] += 1
        totals["event_views"] += len(views)
        totals["writer_facts"] += len(fact_ids)
        totals["reader_exact_replays"] += replay_count
        totals["changed_candidate_fixtures"] += changed_candidate_fixtures
        conversations.append({"conversation": sid, "views": len(views), "links": sum(map(len, old_sets.values())),
                              "synthetic_queries": len(views), "changed_candidate_fixtures": changed_candidate_fixtures,
                              "reader_exact_replays": replay_count})

    assert_equal(totals["reader_exact_replays"], 282, "all multi-hop Reader replays required")
    assert_equal(totals["event_views"], 243, "expected original event-view inventory")
    assert_equal(totals["event_links"], 5567, "expected original routing-link inventory")
    assert_equal(policy_counts["bounded_grounded_claims_2"], 33, "expected 33 historical bounded cases")
    return {"status": "PASS", "seed": int(SEED), "model_calls": 0, "historical_files_modified": False,
            "factory": factory.__module__ + "." + factory.__name__, "totals": dict(totals),
            "historical_reader_policy_counts": dict(policy_counts), "audit_label_changes_with_exact_context_parity": dict(policy_label_changes),
            "conversations": conversations,
            "scope": "Runtime routing adjacency only; original graph, claims, support provenance, assembly and Reader remain unchanged."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, help="Optional runner whose make_routing_index is audited")
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", fail_network), patch.object(socket.socket, "connect_ex", fail_network), patch.object(socket, "create_connection", fail_network):
        factory = None
        if args.runner:
            factory = load_module("_random_links_runner_under_test", args.runner.resolve()).make_routing_index
        print(json.dumps(audit(factory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

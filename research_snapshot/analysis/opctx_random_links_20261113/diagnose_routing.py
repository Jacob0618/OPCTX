"""Offline stagewise routing diagnosis; no provider/client imports or model calls.

Candidate sets are replayed from frozen facts, shared query vectors and the
registered permutation before loading gold source annotations for measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from creamem3.experiments.mec_transfer_20260906 import run_locomo as frozen

ARMS = ("normal", "random")
STAGES = ("backbone", "event_expansion", "session_expansion", "summary_expansion",
          "mixed_union", "cap120", "dense20", "rerank20_valid", "rerank20_system")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dist(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None)
    a = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(a.mean()), median=float(np.median(a)),
                minimum=float(a.min()), maximum=float(a.max()))


def overlap(a, b):
    a, b = set(a), set(b)
    common = len(a & b)
    return dict(n_normal=len(a), n_random=len(b), intersection=common,
                union=len(a | b), jaccard=common/len(a | b) if a | b else 1.0,
                overlap_fraction_of_smaller=common/min(len(a), len(b)) if a and b else float(not (a or b)),
                same_set=a == b)


def source_ids(values):
    return set(re.findall(r"D\d+:\d+", str(values)))


def measure(ids, facts, gold):
    if ids is None:
        return None
    recovered = set().union(*(source_ids(facts[fid].get("source_dia_ids", [])) for fid in ids))
    return dict(fact_count=len(ids), source_count=len(recovered),
                recall=len(recovered & gold)/len(gold) if gold else None,
                all_sources=gold <= recovered if gold else None,
                gold_sources_recovered=sorted(recovered & gold),
                gold_sources_missed=sorted(gold-recovered))


def aggregate(rows):
    result = dict(n=len(rows), arms={}, paired={}, branch_counts={}, overlaps={})
    for arm in ARMS:
        result["arms"][arm] = {}
        result["branch_counts"][arm] = {}
        for stage in STAGES:
            values = [r["measurements"][arm][stage] for r in rows
                      if r["measurements"][arm][stage] is not None]
            scored = [v for v in values if v["recall"] is not None]
            result["arms"][arm][stage] = dict(n=len(values), gold_n=len(scored),
                fact_counts=dist(v["fact_count"] for v in values),
                source_recall_percent=100*float(np.mean([v["recall"] for v in scored])) if scored else None,
                all_sources=sum(bool(v["all_sources"]) for v in scored))
        if rows:
            for name in rows[0]["branches"][arm]:
                result["branch_counts"][arm][name] = dist(r["branches"][arm][name] for r in rows
                                                         if r["branches"][arm][name] is not None)
    for stage in STAGES:
        paired = [r for r in rows if r["measurements"]["normal"][stage] is not None
                  and r["measurements"]["random"][stage] is not None
                  and r["measurements"]["normal"][stage]["recall"] is not None
                  and r["measurements"]["random"][stage]["recall"] is not None]
        if paired:
            left = np.asarray([r["measurements"]["normal"][stage]["recall"] for r in paired])
            right = np.asarray([r["measurements"]["random"][stage]["recall"] for r in paired])
            result["paired"][stage] = dict(n=len(paired), normal_source_recall_percent=100*float(left.mean()),
                random_source_recall_percent=100*float(right.mean()), delta_pp=100*float((left-right).mean()),
                normal_better=int((left > right).sum()), random_better=int((left < right).sum()),
                ties=int((left == right).sum()))
        else:
            result["paired"][stage] = dict(n=0)
        vv = [r["overlaps"][stage] for r in rows if r["overlaps"][stage] is not None]
        result["overlaps"][stage] = dict(n=len(vv), same_set=sum(v["same_set"] for v in vv),
            intersection=dist(v["intersection"] for v in vv),
            jaccard_percent=dist(100*v["jaccard"] for v in vv),
            overlap_fraction_of_smaller_percent=dist(100*v["overlap_fraction_of_smaller"] for v in vv))
    valid_rerank_pairs = [r for r in rows if all(r["stage_ids"][a]["rerank20_valid"] is not None for a in ARMS)]
    score_pairs = [r for r in valid_rerank_pairs if r["gold_sources"]]
    if score_pairs:
        steps = {}
        for stage in ("summary_expansion", "mixed_union", "cap120", "dense20", "rerank20_valid"):
            left = np.asarray([r["measurements"]["normal"][stage]["recall"] for r in score_pairs])
            right = np.asarray([r["measurements"]["random"][stage]["recall"] for r in score_pairs])
            steps[stage] = dict(n=len(score_pairs), normal_percent=100*float(left.mean()),
                random_percent=100*float(right.mean()), normal_minus_random_pp=100*float((left-right).mean()))
        before = steps["dense20"]["normal_minus_random_pp"]
        after = steps["rerank20_valid"]["normal_minus_random_pp"]
        result["rerank_gap_change_same_valid_pairs"] = dict(stages=steps,
            signed_gap_after_minus_before_pp=after-before,
            absolute_gap_reduction_pp=abs(before)-abs(after),
            interpretation="Positive absolute_gap_reduction means closer source-recall means after reranking; this is not a causal attribution of QA accuracy.")
    result["per_arm_rerank_vs_dense20"] = {}
    for arm in ARMS:
        rr = [r for r in rows if r["measurements"][arm]["rerank20_valid"] is not None and r["gold_sources"]]
        deltas = [r["measurements"][arm]["rerank20_valid"]["recall"]-
                  r["measurements"][arm]["dense20"]["recall"] for r in rr]
        result["per_arm_rerank_vs_dense20"][arm] = dict(n=len(rr),
            source_recall_delta_pp=100*float(np.mean(deltas)) if deltas else None,
            improved=sum(d > 0 for d in deltas), worsened=sum(d < 0 for d in deltas), tied=sum(d == 0 for d in deltas),
            same_selected_set=sum(set(r["stage_ids"][arm]["dense20"]) == set(r["stage_ids"][arm]["rerank20_valid"]) for r in rr))
    result["rerank_status_counts"] = {a:dict(Counter(r["actual_status"][a] for r in rows)) for a in ARMS}
    return result


def diagnose(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Diagnostic destination must be new")
    graph_source = ROOT / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"
    assert sha(graph_source) == frozen.FROZEN_CODE_SHA
    spec = importlib.util.spec_from_file_location("_routing_diagnosis_frozen_graph", graph_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    questions = read(root / "questions.json")
    vectors_raw = read(root / "query_vectors.json")
    vectors = dict(zip(vectors_raw["ids"], vectors_raw["vectors"], strict=True))
    permutations = read(root / "fixed_permutations.json")
    assert len(questions) == len(vectors) == 282
    assert set(vectors) == {q["qa_id"] for q in questions}
    graphs, facts, indices, graph_hashes = {}, {}, {}, {}
    for sid in sorted({q["sample_id"] for q in questions}):
        path = ROOT / "creamem3/results_locomo10_v3" / sid / "graph_online.json"
        assert sha(path) == frozen.GRAPH_SHA[sid]
        graph_hashes[str(path)] = sha(path)
        graph = read(path)
        graphs[sid] = graph
        facts[sid] = {f["id"]:f for f in graph["fact_nodes"]}
        permutation = permutations[sid]
        assert set(permutation) == set(permutation.values()) == set(facts[sid])
        normal = module.GraphMemoryIndex(graph)
        random = module.GraphMemoryIndex(graph)
        for node in random.summaries:
            if node.get("node_type") == "event_view":
                random.adjacency[node["id"]] = [permutation[fid] for fid in random.adjacency[node["id"]]]
        indices[sid] = dict(normal=normal, random=random)
    rows, replay_checks = [], Counter()
    # First construct/replay all stage sets without consulting gold annotations.
    for question in questions:
        qid, sid, text = question["qa_id"], question["sample_id"], question["question"]
        vector = vectors[qid]
        stage_ids, branches, traces, actual_status = {}, {}, {}, {}
        for arm in ARMS:
            index = indices[sid][arm]
            capped, trace = index._candidate_pool(text, vector)
            traces[arm] = trace
            dense = set(trace["dense_fact_ids"])
            bm25 = {item["fact_id"] for item in trace["bm25_facts"]}
            backbone = dense | bm25
            event_ids, session_ids, link_count, event_views = set(), set(), 0, []
            for summary in trace["top_summaries"]:
                members = index.adjacency[summary["summary_id"]]
                if summary["node_type"] == "event_view":
                    event_ids.update(members)
                    event_views.append(summary["summary_id"])
                    link_count += len(members)
                else:
                    session_ids.update(members)
            summaries = event_ids | session_ids
            union = summaries | backbone
            dense_scores = index._scores(index.fact_matrix, vector)
            ordered = sorted(union, key=lambda fid:(-float(dense_scores[index.fact_order[fid]]),index.fact_order[fid]))
            cap_ids = [x["fact"]["id"] for x in capped]
            assert cap_ids == ordered[:120]
            assert len(union) == trace["union_before_cap"]
            actual = read(root / "rows" / arm / (qid + ".json"))
            assert actual["candidate_fact_ids"] == cap_ids, (arm,qid,"candidate drift")
            actual_status[arm] = actual["status"]
            chosen = actual.get("selected_fact_ids")
            if actual["status"] == "ok":
                assert isinstance(chosen,list) and len(chosen) == len(set(chosen)) == 20
                assert set(chosen) <= set(cap_ids)
                valid = chosen
                replay_checks["valid_rerank_rows"] += 1
            elif actual["status"] == "policy_zero":
                assert actual["policy_zero_reason"] == "rerank_invalid_after_two_attempts" and chosen == []
                valid = None
                replay_checks["policy_zero_rows"] += 1
            else:
                raise ValueError(f"Unsettled row {arm}/{qid}; do not silently zero it")
            replay_checks["candidate_exact_replays"] += 1
            stage_ids[arm] = dict(backbone=sorted(backbone), event_expansion=sorted(event_ids),
                session_expansion=sorted(session_ids), summary_expansion=sorted(summaries),
                mixed_union=ordered, cap120=cap_ids, dense20=cap_ids[:20],
                rerank20_valid=valid, rerank20_system=chosen)
            event_exclusive = event_ids - backbone - session_ids
            branches[arm] = dict(event_views_hit=len(event_views), event_link_occurrences=link_count,
                event_unique_facts=len(event_ids), session_unique_facts=len(session_ids),
                dense_facts=len(dense), bm25_facts=len(bm25), dense_bm25_intersection=len(dense & bm25),
                shared_backbone_size=len(backbone), event_backbone_overlap=len(event_ids & backbone),
                event_session_overlap=len(event_ids & session_ids),
                event_unique_increment_beyond_backbone_and_session=len(event_exclusive),
                event_facts_surviving_cap=len(event_ids & set(cap_ids)),
                event_exclusive_surviving_cap=len(event_exclusive & set(cap_ids)),
                event_exclusive_surviving_dense20=len(event_exclusive & set(cap_ids[:20])),
                event_exclusive_surviving_rerank20=len(event_exclusive & set(valid)) if valid is not None else None)
        for key in ("policy","top_summaries","dense_fact_ids","bm25_facts"):
            assert traces["normal"][key] == traces["random"][key], (qid,key)
        assert branches["normal"]["event_unique_facts"] == branches["random"]["event_unique_facts"]
        assert branches["normal"]["event_link_occurrences"] == branches["random"]["event_link_occurrences"]
        replay_checks["unchanged_backbone_and_summary_rankings"] += 1
        rows.append(dict(qa_id=qid,sample_id=sid,stage_ids=stage_ids,branches=branches,actual_status=actual_status))
    # Gold enters only now, after all stage selections are fixed.
    dataset = ROOT / "data/locomo/locomo10.json"
    assert sha(dataset) == frozen.DATA_SHA
    gold = {}
    for conversation in read(dataset):
        for i,q in enumerate(conversation["qa"]):
            gold[f"{conversation['sample_id']}-q{i:03d}"] = source_ids(q.get("evidence",[]))
    for row in rows:
        evidence = gold[row["qa_id"]]
        row["gold_sources"] = sorted(evidence)
        row["measurements"] = {a:{s:measure(row["stage_ids"][a][s],facts[row["sample_id"]],evidence)
                                   for s in STAGES} for a in ARMS}
        row["overlaps"] = {s:overlap(row["stage_ids"]["normal"][s],row["stage_ids"]["random"][s])
                           if all(row["stage_ids"][a][s] is not None for a in ARMS) else None for s in STAGES}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PASS", model_calls=0,
        experiment_root=str(root), seed=92620, questions=282,
        source_hashes={str(p):sha(p) for p in (root/"questions.json",root/"query_vectors.json",root/"fixed_permutations.json",graph_source,dataset,Path(__file__))},
        graph_hashes=graph_hashes,replay_checks=dict(replay_checks),summary=aggregate(rows),
        by_conversation={sid:aggregate([r for r in rows if r["sample_id"]==sid]) for sid in sorted(graphs)},rows=rows,
        boundaries=["Gold annotations used only after every stage set is fixed.",
                    "Dense20 is an offline no-model-rerank selection counterfactual; its QA accuracy has not been measured.",
                    "Source recall measures annotated dialogue provenance, not sufficient answer evidence.",
                    "rerank20_valid excludes explicit invalid-rerank policy failures; rerank20_system treats those no-output paths as empty selected evidence.",
                    "Retrieval routing only was perturbed; the Reader uses original source/claim associations."])
    lines=["# Offline routing-stage diagnosis", "", "282 LoCoMo multi-hop questions; fixed seed92620; zero model calls.", "",
           "## Source recall at each stage", "", "| Stage | Paired n | Normal (%) | Random (%) | Normal−random (pp) |", "|---|---:|---:|---:|---:|"]
    summary=result["summary"]
    for stage in STAGES:
        x=summary["paired"][stage]
        if x["n"]:
            lines.append(f"| {stage} | {x['n']} | {x['normal_source_recall_percent']:.2f} | {x['random_source_recall_percent']:.2f} | {x['delta_pp']:+.2f} |")
    lines += ["", "## Set overlap", "", "| Stage | Pairs | Identical sets | Mean intersection | Mean Jaccard (%) |", "|---|---:|---:|---:|---:|"]
    for stage in ("event_expansion","summary_expansion","mixed_union","cap120","dense20","rerank20_valid"):
        x=summary["overlaps"][stage]
        lines.append(f"| {stage} | {x['n']} | {x['same_set']} | {x['intersection']['mean']:.2f} | {x['jaccard_percent']['mean']:.2f} |")
    lines += ["", "## Branch sizes (means)", "", "| Metric | Normal | Random |", "|---|---:|---:|"]
    for metric in summary["branch_counts"]["normal"]:
        a=summary["branch_counts"]["normal"][metric]["mean"]
        b=summary["branch_counts"]["random"][metric]["mean"]
        lines.append(f"| {metric} | {a:.2f} | {b:.2f} |")
    gap=summary.get("rerank_gap_change_same_valid_pairs")
    if gap:
        lines += ["", "## Does reranking narrow the source-recall gap?", "",
                  "This comparison uses exactly the same question pairs with valid reranker outputs in both arms."]
        for stage in ("mixed_union","cap120","dense20","rerank20_valid"):
            x=gap["stages"][stage]
            lines.append(f"- {stage}: normal {x['normal_percent']:.2f}%, random {x['random_percent']:.2f}%; gap {x['normal_minus_random_pp']:+.2f} pp (n={x['n']}).")
        lines += [f"- Absolute gap reduction from Dense20 to rerank20: {gap['absolute_gap_reduction_pp']:+.2f} pp."]
    lines += ["", "## Scope", "", *["- "+x for x in result["boundaries"]], ""]
    output.mkdir(parents=True,exist_ok=False)
    with (output/"results.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write("\n")
    with (output/"report.md").open("x",encoding="utf-8") as f:
        f.write("\n".join(lines))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.root,args.output)
    print(json.dumps(dict(status=result["status"],output=str(args.output),checks=result["replay_checks"])))

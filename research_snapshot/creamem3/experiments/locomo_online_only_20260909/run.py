"""Frozen-control Online-only LoCoMo experiments. No historical writes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from creamem3.experiments.locomo_online_only_20260909.runtime import (
    Runtime, StopRun, BACKBONE, JUDGE, read, seal, sha, file_sha, now)
from creamem3.experiments.locomo_online_only_20260909.prompts import VARIANTS
from creamem3.experiments.locomo_g35lite_20260908.run import Lease
from creamem3.locomo_all import load_all_locomo, DEFAULT_DATASET_PATH
from creamem3.online_memory_v3 import add_online_event_views, audit_online_graph
from creamem3.online_operation_context_v8 import materialize_online_operation_context
from eval.prompts import QA_PROMPT, JUDGE_PROMPT

RUN = Path(r"D:\creamem3_cache\locomo_online_only_20260909_r1")
HISTORY = ROOT / "creamem3/results_locomo10_v3"
BASE_CONTEXT = Path(r"D:\creamem3_cache\full_budget_20260907\locomo_opctx20\baseline")
VECTOR_FILE = Path(r"D:\creamem3_cache\locomo_g35lite_full1540_20260908_r2\question_vectors.json")
FROZEN = ROOT / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"
spec = importlib.util.spec_from_file_location("locomo_online_research_frozen", FROZEN)
graph_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_api)
PRINT_LOCK = threading.Lock()

def emit(message):
    with PRINT_LOCK:
        print(f"{now()} {message}", flush=True)

def old_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def select_dev(rows, n=100):
    """Only ID/category enter this function; no question, reference or score."""
    groups = {}
    for row in rows:
        groups.setdefault(row["category_name"], []).append(row["qa_id"])
    quotas = {k: len(v)*n/len(rows) for k, v in groups.items()}
    counts = {k: math.floor(v) for k, v in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k]-counts[k]), k))[:n-sum(counts.values())]:
        counts[k] += 1
    selected = []
    for category, ids in sorted(groups.items()):
        ordered = sorted(ids, key=lambda qid: hashlib.sha256(
            ("locomo-online-only-20260909-dev100-v1:"+qid).encode()).hexdigest())
        selected.extend(ordered[:counts[category]])
    return sorted(selected), counts

def prepare(root=RUN):
    loaded = load_all_locomo()
    rows = [q for conv in loaded for q in conv["qa"]]
    if len(rows) != 1540:
        raise StopRun("Population drift")
    ids, quotas = select_dev([{k: q[k] for k in ["qa_id", "category_name"]} for q in rows])
    sources = {}
    for conv in loaded:
        for name in ["graph_session.json", "graph_online.json"]:
            p = HISTORY / conv["sample_id"] / name
            sources[str(p)] = file_sha(p)
    protocol = {"schema": "locomo.online_only.v1", "backbone": BACKBONE, "judge": JUDGE,
                "temperature": 0, "thinking_budget": 0, "judge_thinking": "minimal",
                "judge_votes": 1, "qa_max_tokens": 8192, "judge_initial_max_tokens": 2048,
                "dataset_sha256": file_sha(DEFAULT_DATASET_PATH), "historical_graphs": sources,
                "query_vector_file": str(VECTOR_FILE), "query_vector_file_sha256": file_sha(VECTOR_FILE),
                "dev_ids": ids, "dev_quota": quotas, "all_ids": [q["qa_id"] for q in rows],
                "candidate_cap": 120, "final_facts": 20, "qa_prompt_sha256": sha(QA_PROMPT),
                "judge_prompt_sha256": sha(JUDGE_PROMPT), "frozen_graph_code_sha256": file_sha(FROZEN),
                "runtime_sha256": file_sha(Path(__file__).with_name("runtime.py")),
                "online_renderer_sha256": file_sha(ROOT/"creamem3/online_operation_context_v8.py"),
                "semantic_renderer_sha256": file_sha(ROOT/"creamem3/semantic_context_v8.py"),
                "writer_reused_immutable": True, "full_corpus_previously_developed": True,
                "summary_prose_is_extra_evidence": True}
    seal(root/"protocol.json", protocol)
    return loaded, protocol

def register_variant(root, variant):
    prompt = VARIANTS[variant]
    manifest = {"variant": variant, "prompt": prompt, "prompt_sha256": sha(prompt),
                "online_source_sha256": file_sha(ROOT/"creamem3/online_memory_v3.py"),
                "invariant": "frozen Writer facts; causal prefix; cap80 history;10 views;12ops",
                "changes": "Online update prompt only"}
    seal(root/"variants"/(variant+".json"), manifest)
    return manifest

@lru_cache(maxsize=10)
def legal_sources(sample):
    conv=next(c for c in load_all_locomo() if c["sample_id"]==sample)
    return tuple(turn["dia_id"] for record in conv["records"] for turn in record["turns"]),len(conv["records"])

def graph_structure_audit(graph,sample):
    ids,n=legal_sources(sample)
    return graph_api.audit_graph(graph,legal_dia_ids=ids,expected_sessions=n)

def get_source_graph(sample, arm, protocol):
    path = HISTORY/sample/("graph_session.json" if arm == "session_strong" else "graph_online.json")
    if file_sha(path) != protocol["historical_graphs"][str(path)]:
        raise StopRun("Historical graph changed: " + str(path))
    graph = old_json(path)
    audit = graph_structure_audit(graph,sample)
    if audit["status"] != "pass":
        raise StopRun("Historical graph fails audit: " + sample)
    return graph

def build_variant(root, variant, loaded, protocol, runtime, conv_workers=10):
    registration = register_variant(root, variant)
    def one(conv):
        sample = conv["sample_id"]
        dest = root/"graphs"/variant/(sample+".json")
        base = get_source_graph(sample, "session_strong", protocol)
        identity = {"variant_sha256": sha(registration), "session_sha256": sha(base)}
        if dest.exists():
            record = read(dest)
            if record["identity"] != identity:
                raise StopRun("Variant graph identity drift")
            return record["stats"]
        emit(f"BUILD START {variant} {sample}")
        began = time.perf_counter()
        graph = add_online_event_views(base, runtime.chat_fn(variant+"/"+sample), runtime.embed,
                                       prompt_template=registration["prompt"])
        audit = audit_online_graph(base, graph, expected_prompt_sha256=hashlib.sha256(registration["prompt"].encode()).hexdigest())
        standard_audit = graph_structure_audit(graph,sample)
        if audit["status"] != "pass" or standard_audit["status"] != "pass" or base["fact_nodes"] != graph["fact_nodes"]:
            raise StopRun("New graph causal/immutable audit failed: " + sample)
        steps = graph["online_fusion"]["steps"]
        if any(s["model_error"] for s in steps):
            raise StopRun("Online logical failure: " + sample)
        stats = {"sample": sample, "sessions": len(steps), "views": graph["stats"]["event_view_nodes"],
                 "accepted": sum(len(s["accepted"]) for s in steps),
                 "rejected": sum(len(s["rejected"]) for s in steps),
                 "seconds": time.perf_counter()-began}
        seal(dest, {"identity": identity, "graph": graph, "audit": audit, "standard_audit": standard_audit, "stats": stats})
        runtime.log({"kind": "graph_complete", "variant": variant, **stats})
        emit(f"BUILD COMPLETE {variant} {sample} {stats}")
        return stats
    with ThreadPoolExecutor(max_workers=conv_workers) as pool:
        results = [f.result() for f in as_completed([pool.submit(one, conv) for conv in loaded])]
    seal(root/"graphs"/variant/"complete.json", {"variant": variant, "conversations": sorted(results, key=lambda r:r["sample"])})
    return results

def facts_only(graph, selected):
    by_id = {f["id"]: f for f in graph["fact_nodes"]}
    facts = [by_id[fid] for fid in selected]
    return "\n".join(f"[Session {f['session_id']} | {f.get('iso_date', '')} | {', '.join(f['source_dia_ids'])}] {f['text']}" for f in facts)

def judge(runtime, scope, q, response):
    prompt = JUDGE_PROMPT.format(question=q["question"], answer=q["answer"], response=response)
    for attempt, max_tokens in enumerate([2048, 4096, 8192], 1):
        raw = runtime.chat(scope+f"/parse{attempt}", "judge", prompt, max_tokens=max_tokens)
        tags = re.findall(r"\[\[(yes|no)\]\]", raw.lower())
        if len(tags) == 1:
            return int(tags[0] == "yes"), raw
    raise StopRun("Judge parse exhausted for " + scope)

def evaluate(root, arms, loaded, protocol, runtime, *, full=False, qa_workers=16):
    wanted = set(protocol["all_ids"] if full else protocol["dev_ids"])
    selected_q = [(conv["sample_id"], q) for conv in loaded for q in conv["qa"] if q["qa_id"] in wanted]
    vector_record = read(VECTOR_FILE)
    vectors = dict(zip(vector_record["ids"], vector_record["vectors"]))
    graphs, indexes, hashes = {}, {}, {}
    for sample in [conv["sample_id"] for conv in loaded]:
        for arm in arms:
            actual_arm = "session_strong" if arm == "session_facts_only" else arm
            key = (sample, actual_arm)
            if key in graphs:
                continue
            if actual_arm in {"session_strong", "opctx_original"}:
                graph = get_source_graph(sample, actual_arm, protocol)
            else:
                record = read(root/"graphs"/actual_arm/(sample+".json"))
                registration = read(root/"variants"/(actual_arm+".json"))
                if record["identity"]["variant_sha256"] != sha(registration):
                    raise StopRun("Registration mismatch")
                graph = record["graph"]
            graphs[key] = graph
            hashes[key] = sha(graph)
            indexes[key] = graph_api.GraphMemoryIndex(graph)

    def one(sample, q, arm):
        qid = q["qa_id"]
        actual_arm = "session_strong" if arm == "session_facts_only" else arm
        key = (sample, actual_arm)
        graph = graphs[key]
        identity = {"protocol_sha256": sha(protocol), "qa_sha256": sha(q), "arm": arm,
                    "graph_sha256": hashes[key], "vector_sha256": sha(vectors[qid])}
        path = root/"rows"/arm/(qid+".json")
        if path.exists():
            result = read(path)
            if result["identity"] != identity:
                raise StopRun("Existing QA row drift")
            return result
        scope = arm+"/"+qid
        began = time.perf_counter()
        runtime.log({"kind":"row_start", "scope":scope})
        retrieval_path = root/"retrieval"/actual_arm/(qid+".json")
        # session_facts_only is evaluated after session_strong; no same-path races.
        if retrieval_path.exists():
            retrieval = read(retrieval_path)
            if retrieval["graph_sha256"] != hashes[key] or retrieval["vector_sha256"] != identity["vector_sha256"]:
                raise StopRun("Retrieval graph/vector identity drift")
        else:
            base, trace = indexes[key].search(q["question"], vectors[qid], runtime.chat_fn(scope))
            ids = trace["rerank"]["selected_fact_ids"]
            if len(ids) != 20 or len(set(ids)) != 20 or trace["candidates_after_cap"] > 120:
                raise StopRun("Retrieval budget/selection invalid")
            # Cached operation is a query-only historical semantic plan, never a label from gold.
            operation = old_json(BASE_CONTEXT/(qid+".json"))["operation"]
            if actual_arm == "session_strong":
                context, materializer = base, None
            else:
                context, materializer = materialize_online_operation_context(operation,
                    {"question":q["question"], "retrieved":base, "retrieval_trace":copy.deepcopy(trace)}, graph)
            retrieval = {"question_sha256":sha(q["question"]), "graph_sha256":hashes[key],
                         "vector_sha256":identity["vector_sha256"], "context":context, "trace":trace,
                         "operation":operation, "materializer":materializer}
            seal(retrieval_path, retrieval)
        ids = retrieval["trace"]["rerank"]["selected_fact_ids"]
        context = facts_only(graph, ids) if arm == "session_facts_only" else retrieval["context"]
        response = runtime.chat(scope, "qa", QA_PROMPT.format(retrieved_texts=context, question=q["question"]), max_tokens=8192)
        correct, raw_judge = judge(runtime, scope, q, response)
        fact_map = indexes[key].fact_by_id
        evidence = set(q.get("evidence_dia_ids", []))
        hits = set(s for fid in ids for s in fact_map[fid]["source_dia_ids"]) & evidence
        row = {"identity":identity, "qa_id":qid, "sample_id":sample, "question":q["question"],
               "answer":q["answer"], "category_name":q["category_name"], "correct":correct,
               "response":response, "judge_raw":raw_judge, "context":context, "context_chars":len(context),
               "context_sha256":sha(context), "selected_fact_ids":ids, "evidence_n":len(evidence),
               "evidence_hit_n":len(hits), "operation":retrieval["operation"],
               "rerank_fallback":retrieval["trace"]["rerank"]["fallback"], "elapsed_s":time.perf_counter()-began}
        seal(path,row)
        runtime.log({"kind":"row_complete", "scope":scope, "correct":correct, "elapsed_s":row["elapsed_s"]})
        emit(f"ROW {arm} {qid} correct={correct} chars={len(context)} seconds={row['elapsed_s']:.1f}")
        return row
    for arm in arms:
        with ThreadPoolExecutor(max_workers=qa_workers) as pool:
            results = [f.result() for f in as_completed([pool.submit(one,sample,q,arm) for sample,q in selected_q])]
        report = summarize(root,arm,protocol,full=full)
        seal(root/"summaries"/(arm+("_full" if full else "_dev")+".json"),report)
        emit(f"METRIC {arm} accuracy={report['accuracy']:.4f} correct={report['correct']}/{report['n']} chars={report['mean_context_chars']:.1f}")
    return [summarize(root,arm,protocol,full=full) for arm in arms]

def summarize(root, arm, protocol, *, full=False):
    expected = set(protocol["all_ids"] if full else protocol["dev_ids"])
    rows = [read(p) for p in sorted((root/"rows"/arm).glob("*.json")) if p.stem in expected]
    n = len(rows)
    report = {"arm":arm, "n":n, "expected":len(expected), "complete":n==len(expected),
              "correct":sum(r["correct"] for r in rows), "accuracy":100*sum(r["correct"] for r in rows)/n if n else None,
              "mean_context_chars":statistics.mean(r["context_chars"] for r in rows) if n else None,
              "rerank_fallbacks":sum(r["rerank_fallback"] for r in rows),
              "source_evidence_hits":sum(r["evidence_hit_n"] for r in rows), "source_evidence_total":sum(r["evidence_n"] for r in rows),
              "by_category":{}, "by_operation":{}, "paired":{}}
    for field, dest in [("category_name","by_category"),("operation","by_operation")]:
        for group in sorted({r[field] for r in rows}):
            sub = [r for r in rows if r[field]==group]
            report[dest][group] = {"n":len(sub), "correct":sum(r["correct"] for r in sub)}
    for control in ["session_strong", "opctx_original"]:
        paired = [(r,read(root/"rows"/control/(r["qa_id"]+".json"))) for r in rows if (root/"rows"/control/(r["qa_id"]+".json")).exists()]
        if not paired:
            continue
        report["paired"][control] = {"n":len(paired), "wins":sum(a["correct"]>b["correct"] for a,b in paired),
            "losses":sum(a["correct"]<b["correct"] for a,b in paired),
            "mean_top20_overlap":statistics.mean(len(set(a["selected_fact_ids"])&set(b["selected_fact_ids"])) for a,b in paired),
            "same_context":sum(a["context_sha256"]==b["context_sha256"] for a,b in paired)}
    if full:
        other = [r for r in rows if r["qa_id"] not in set(protocol["dev_ids"])]
        report["excluding_dev"] = {"n":len(other), "correct":sum(r["correct"] for r in other),
                                   "accuracy":100*sum(r["correct"] for r in other)/len(other) if other else None}
    return report

def status(root):
    if not (root/"protocol.json").exists():
        emit("Not prepared")
        return
    protocol = read(root/"protocol.json")
    for directory in sorted((root/"rows").glob("*")):
        if directory.is_dir():
            r=summarize(root,directory.name,protocol,full=False)
            emit(json.dumps(r,ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare","smoke","baseline","build","eval","campaign","status","audit"])
    parser.add_argument("--root",type=Path,default=RUN)
    parser.add_argument("--variant",default="relational_delta",choices=list(VARIANTS))
    parser.add_argument("--arms",default="session_strong,opctx_original,session_facts_only")
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--concurrency",type=int,default=10)
    parser.add_argument("--qa-workers",type=int,default=16)
    parser.add_argument("--conv-workers",type=int,default=10)
    args=parser.parse_args()
    if args.command=="status":
        status(args.root);return
    if args.command=="audit":
        from creamem3.experiments.locomo_online_only_20260909.audit import audit
        print(json.dumps(audit(args.root),ensure_ascii=False));return
    loaded,protocol=prepare(args.root)
    if args.command=="prepare":
        print(json.dumps({"dev_ids":protocol["dev_ids"],"quotas":protocol["dev_quota"]},ensure_ascii=False));return
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT/".env",override=False)
    except ImportError:
        pass
    runtime=Runtime(args.root/"calls",allow_model_calls=True,concurrency=args.concurrency,timeout=180)
    with Lease(args.root):
        seal(args.root/("invocation_"+str(time.time_ns())+".json"),{"utc":now(),"argv":sys.argv,"code_sha256":file_sha(__file__)})
        try:
            if args.command=="smoke":
                response=runtime.chat("connectivity","qa","Reply with exactly the word READY.",max_tokens=32)
                verdict,raw=judge(runtime,"connectivity",{"question":"Which word was requested?","answer":"READY"},response)
                if response.strip()!="READY" or verdict!=1:
                    raise StopRun("Connectivity smoke unexpected response")
                seal(args.root/"smoke.json",{"qa":response,"judge":raw,"passed":True})
                emit("SMOKE PASS:2.5Flash +3FlashPreview with original thinking settings")
            elif args.command in {"baseline","eval"}:
                arms=args.arms.split(",") if args.command=="baseline" else [args.variant]
                evaluate(args.root,arms,loaded,protocol,runtime,full=args.full,qa_workers=args.qa_workers)
            elif args.command=="build":
                build_variant(args.root,args.variant,loaded,protocol,runtime,args.conv_workers)
            elif args.command=="campaign":
                register_variant(args.root,args.variant)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    builds=pool.submit(build_variant,args.root,args.variant,loaded,protocol,runtime,args.conv_workers)
                    bases=pool.submit(evaluate,args.root,["session_strong","opctx_original","session_facts_only"],loaded,protocol,runtime,qa_workers=args.qa_workers)
                    bases.result();builds.result()
                evaluate(args.root,[args.variant],loaded,protocol,runtime,qa_workers=args.qa_workers)
        except BaseException as exc:
            runtime.stopped.set()
            seal(args.root/("stop_"+str(time.time_ns())+".json"),{"utc":now(),"type":type(exc).__name__,"message":str(exc)[:500]})
            emit("STOP "+type(exc).__name__+": "+str(exc)[:300])
            raise

if __name__=="__main__":
    main()

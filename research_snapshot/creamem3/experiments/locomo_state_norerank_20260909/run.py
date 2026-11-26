"""No-reranker controls and support-complete Online retrieval. All writes on D."""
from __future__ import annotations
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from creamem3.experiments.locomo_online_only_20260909 import run as old
from creamem3.experiments.locomo_state_norerank_20260909.runtime import (
    Runtime, StopRun, read, seal, sha, file_sha, now)
from creamem3.experiments.locomo_state_norerank_20260909.selector import normalize, choose

ROOT = Path(r"D:\creamem3_cache\locomo_state_norerank_20260909_r1")
PACKAGE = Path(__file__).parent
ARMS = {
    "session_cosine_facts": ("session", "cosine", "facts"),
    "session_cosine_rich": ("session", "cosine", "rich"),
    "original_cosine_facts": ("original", "cosine", "facts"),
    "original_cosine_rich": ("original", "cosine", "rich"),
    "original_flat_facts": ("original", "flat", "facts"),
    "original_closed_facts": ("original", "closed", "facts"),
    "original_closed_supported": ("original", "closed", "supported"),
    "revision_flat_facts": ("revision", "flat", "facts"),
    "revision_closed_facts": ("revision", "closed", "facts"),
    "revision_closed_supported": ("revision", "closed", "supported"),
}


def prepare():
    parent = read(old.RUN / "protocol.json")
    loaded = old.load_all_locomo()
    if file_sha(old.DEFAULT_DATASET_PATH) != parent["dataset_sha256"]:
        raise StopRun("Dataset drift")
    if [q["qa_id"] for c in loaded for q in c["qa"]] != parent["all_ids"]:
        raise StopRun("Population/order drift")
    if file_sha(old.VECTOR_FILE) != parent["query_vector_file_sha256"]:
        raise StopRun("Query embedding drift")
    sources = [PACKAGE/n for n in ("run.py", "runtime.py", "selector.py")]
    sources += [Path(old.__file__)]
    sources += [old.FROZEN, old.ROOT/"creamem3/online_operation_context_v8.py",
                old.ROOT/"creamem3/semantic_context_v8.py", old.ROOT/"eval/prompts.py",
                old.ROOT/"creamem3/experiments/locomo_g35lite_20260908/runtime.py",
                old.ROOT/"creamem3/experiments/locomo_online_only_20260909/runtime.py"]
    protocol = {"schema": "locomo.state.norerank.v1", "parent_sha256": sha(parent),
        "source_sha256": {str(p):file_sha(p) for p in sources}, "dev_ids":parent["dev_ids"],
        "all_ids":parent["all_ids"], "dev_quota":parent["dev_quota"],
        "backbone":"gemini-2.5-flash", "judge":"gemini-3-flash-preview", "single_judge":True,
        "model_rerank_allowed":False, "final_facts":20, "local_pool":"all immutable Writer facts",
        "new_writer_calls":0, "thinking_budget":0, "temperature":0,
        "facts_renderer_sha256":file_sha(old.__file__), "query_embedding_sha256":file_sha(old.VECTOR_FILE),
        "claim_embedding":"query-independent gemini-embedding-001 768D RETRIEVAL_DOCUMENT",
        "promotion":"Strict gain over matched dev control;freeze each arm before remaining1440 evaluation",
        "previously_developed_dataset":True, "selection_sees_gold_or_category":False}
    seal(ROOT/"protocol.json", protocol)
    return loaded, parent, protocol


def graph_for(sample, source, parent):
    if source in {"session", "original"}:
        return old.get_source_graph(sample, "session_strong" if source=="session" else "opctx_original", parent)
    record = read(ROOT/"graphs"/source/(sample+".json"))
    if record["base_sha256"] != sha(old.get_source_graph(sample,"session_strong",parent)):
        raise StopRun("Revision base graph drift")
    return record["graph"]


def units_from_graph(graph):
    fact_ids = {f["id"] for f in graph["fact_nodes"]}
    units = []
    for view in graph["summary_nodes"]:
        if view.get("node_type") != "event_view":
            continue
        for i, claim in enumerate(view.get("summary_claims", [])):
            members = sorted(set(claim["support_fact_ids"]))
            if not members or not set(members) <= fact_ids:
                raise StopRun("Unknown claim support")
            if len(members) > 20:
                continue
            text = " ".join(claim["text"].split())
            if not text:
                raise StopRun("Empty relation text")
            units.append({"id":view["id"]+":c"+str(i), "text":text,
                          "support_fact_ids":members, "view_id":view["id"]})
    unique = {}
    for unit in units:
        key = sha({"text":unit["text"],"support":unit["support_fact_ids"]})
        unique.setdefault(key, unit)
    return list(unique.values())


def embed_units(source, loaded, parent, runtime):
    def one(conv):
        sample = conv["sample_id"]
        graph = graph_for(sample, source, parent)
        units = units_from_graph(graph)
        path = ROOT/"units"/source/(sample+".json")
        identity = {"graph_sha256":sha(graph),"units_sha256":sha(units),
                    "source":source,"query_independent":True}
        if path.exists():
            record = read(path)
            if record["identity"] != identity:
                raise StopRun("Unit embedding identity drift")
            return
        vectors = runtime.embed([u["text"] for u in units]) if units else []
        vectors = normalize(vectors).tolist() if vectors else []
        seal(path,{"identity":identity,"units":units,"vectors":vectors})
        old.emit(f"UNITS {source} {sample} n={len(units)}")
    with ThreadPoolExecutor(max_workers=10) as pool:
        for result in as_completed([pool.submit(one,c) for c in loaded]): result.result()


def summary(arm, protocol, full=False):
    ids = protocol["all_ids"] if full else protocol["dev_ids"]
    rows = [read(ROOT/"rows"/arm/(q+".json")) for q in ids]
    result = {"arm":arm,"n":len(rows),"correct":sum(r["correct"] for r in rows),
        "accuracy":100*sum(r["correct"] for r in rows)/len(rows),
        "mean_context_chars":statistics.mean(r["context_chars"] for r in rows),
        "mean_packages":statistics.mean(len(r["selected_unit_ids"]) for r in rows),
        "by_category":{},"paired":{},"model_rerank_calls":0}
    for category in sorted({r["category_name"] for r in rows}):
        sub=[r for r in rows if r["category_name"]==category]
        result["by_category"][category]={"n":len(sub),"correct":sum(r["correct"] for r in sub)}
    for control in ARMS:
        if control == arm: continue
        pairs=[(r,read(ROOT/"rows"/control/(r["qa_id"]+".json"))) for r in rows
               if (ROOT/"rows"/control/(r["qa_id"]+".json")).exists()]
        if len(pairs) == len(rows):
            result["paired"][control]={"wins":sum(a["correct"]>b["correct"] for a,b in pairs),
                "losses":sum(a["correct"]<b["correct"] for a,b in pairs),
                "same_context":sum(a["context"]==b["context"] for a,b in pairs),
                "mean_top20_overlap":statistics.mean(len(set(a["selected_fact_ids"])&set(b["selected_fact_ids"])) for a,b in pairs)}
    if full:
        remaining=[r for r in rows if r["qa_id"] not in set(protocol["dev_ids"])]
        result["remaining1440"]={"n":len(remaining),"correct":sum(r["correct"] for r in remaining)}
    return result


def evaluate(arms, loaded, parent, protocol, runtime, full=False, workers=24):
    wanted=set(protocol["all_ids"] if full else protocol["dev_ids"])
    qrows=[(c["sample_id"],q) for c in loaded for q in c["qa"] if q["qa_id"] in wanted]
    vector_record=read(old.VECTOR_FILE)
    vectors=dict(zip(vector_record["ids"],vector_record["vectors"]))
    for arm in arms:
        source,mode,reader=ARMS[arm]
        registration={"arm":arm,"source":source,"mode":mode,"reader":reader,
                      "protocol_sha256":sha(protocol),"phase":"fixed before evaluation"}
        seal(ROOT/"arms"/(arm+".json"),registration)
        graphs={c["sample_id"]:graph_for(c["sample_id"],source,parent) for c in loaded}
        indexes={s:old.graph_api.GraphMemoryIndex(g) for s,g in graphs.items()}
        unit_data={}
        if mode != "cosine":
            for sample,g in graphs.items():
                unit_data[sample]=read(ROOT/"units"/source/(sample+".json"))
                if unit_data[sample]["identity"]["graph_sha256"] != sha(g):
                    raise StopRun("Unit/graph identity drift")
        def one(sample,q):
            graph=graphs[sample];index=indexes[sample];qid=q["qa_id"]
            units=unit_data.get(sample,{}).get("units",[])
            identity={"protocol_sha256":sha(protocol),"arm_sha256":sha(registration),
                "graph_sha256":sha(graph),"qa_sha256":sha(q),"vector_sha256":sha(vectors[qid]),
                "units_sha256":sha(unit_data.get(sample,{}))}
            path=ROOT/"rows"/arm/(qid+".json")
            if path.exists():
                row=read(path)
                if row["identity"]!=identity:raise StopRun("Existing row drift")
                return
            started=time.perf_counter()
            runtime.log({"kind":"row_start","arm":arm,"qa_id":qid})
            fact_scores=index._scores(index.fact_matrix,vectors[qid])
            unit_scores=np.asarray(unit_data[sample]["vectors"]) @ normalize(vectors[qid]) if units else []
            indexed=[{**u,"indices":[index.fact_order[f] for f in u["support_fact_ids"]]} for u in units]
            selected=choose(fact_scores,indexed,unit_scores,mode=mode)
            ids=[index.facts[i]["id"] for i in selected["indices"]]
            selected_units=[units[i] for i in selected["unit_indices"]]
            if reader=="rich":
                _,trace=index._candidate_pool(q["question"],vectors[qid])
                items=[{"fact":index.fact_by_id[f]} for f in ids]
                context,assembly=index._assemble(items,trace["top_summaries"])
                trace["assembly"]=assembly
                if source!="session":
                    operation=old.old_json(old.BASE_CONTEXT/(qid+".json"))["operation"]
                    trace["rerank"]={"selected_fact_ids":ids,"fallback":False,"attempts":0,"model_used":False}
                    context,_=old.materialize_online_operation_context(operation,
                        {"question":q["question"],"retrieved":context,"retrieval_trace":copy.deepcopy(trace)},graph)
            else:
                context=old.facts_only(graph,ids)
                if reader=="supported":
                    for unit in selected_units:
                        if not set(unit["support_fact_ids"])<=set(ids):raise StopRun("Incomplete displayed support")
                    text="\n".join("[Derived relation; support="+",".join(u["support_fact_ids"])+"] "+u["text"] for u in selected_units)
                    if text:context+="\n"+text
            scope=arm+"/"+qid
            response=runtime.chat(scope,"qa",old.QA_PROMPT.format(retrieved_texts=context,question=q["question"]),max_tokens=8192)
            correct,verdict=old.judge(runtime,scope,q,response)
            evidence=set(q.get("evidence_dia_ids",[]))
            hit=evidence&{dia for fid in ids for dia in index.fact_by_id[fid]["source_dia_ids"]}
            row={"identity":identity,"qa_id":qid,"sample_id":sample,"question":q["question"],
                "answer":q["answer"],"category_name":q["category_name"],"correct":correct,
                "response":response,"judge_raw":verdict,"context":context,"context_sha256":sha(context),
                "context_chars":len(context),"selected_fact_ids":ids,"selected_unit_ids":[u["id"] for u in selected_units],
                "selection":selected,"evidence_n":len(evidence),"evidence_hit_n":len(hit),
                "elapsed_s":time.perf_counter()-started,"model_rerank_calls":0}
            seal(path,row)
            runtime.log({"kind":"row_complete","arm":arm,"qa_id":qid,"correct":correct})
            old.emit(f"ROW {arm} {qid} correct={correct} chars={len(context)}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(one,s,q) for s,q in qrows]):future.result()
        report=summary(arm,protocol,full)
        seal(ROOT/"summaries"/(arm+("_full" if full else "_dev")+".json"),report)
        old.emit(f"METRIC {arm} accuracy={report['accuracy']:.4f} correct={report['correct']}/{report['n']}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare","baseline","embed","eval"])
    parser.add_argument("--source",default="original",choices=["original","revision"])
    parser.add_argument("--arms",default="session_cosine_facts,original_cosine_facts,session_cosine_rich,original_cosine_rich")
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    loaded,parent,protocol=prepare()
    if args.command=="prepare":return
    from dotenv import load_dotenv
    load_dotenv(old.ROOT/".env",override=False)
    with old.Lease(ROOT):
        runtime=Runtime(ROOT/"calls",allow_model_calls=True,concurrency=24,timeout=180)
        try:
            if args.command=="embed":embed_units(args.source,loaded,parent,runtime)
            else:evaluate(args.arms.split(","),loaded,parent,protocol,runtime,full=args.full)
        except BaseException as exc:
            runtime.stopped.set()
            seal(ROOT/("stop_"+str(time.time_ns())+".json"),{"utc":now(),"type":type(exc).__name__,"message":str(exc)[:500]})
            raise


if __name__=="__main__":main()

"""Full LoCoMo: shared fresh Writer, Session-only vs original Online OpCtx."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from creamem3.experiments.locomo_g35lite_20260908.runtime import (
    BACKBONE, EMBED, JUDGE, Runtime, StopRun, file_sha, now, read, seal, sha)
from creamem3.locomo_all import DEFAULT_DATASET_PATH, load_all_locomo
from creamem3.online_memory_v3 import add_online_event_views, audit_online_graph
from creamem3.online_operation_context_v8 import materialize_online_operation_context
from creamem3.strict_semantic_planner_v8 import classify_strict
from creamem3.prompts_v2 import PROMPT_SHA256
from eval.prompts import QA_PROMPT, JUDGE_PROMPT

RUN_ROOT = Path(r"D:\creamem3_cache\locomo_g35lite_full1540_20260908_r1")
FROZEN = ROOT / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"
spec = importlib.util.spec_from_file_location("locomo_g35_frozen_graph", FROZEN)
graph_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_api)
CONSOLE_LOCK = threading.Lock()

def emit(message):
    with CONSOLE_LOCK:
        print(f"{now()} {message}", flush=True)

def source_manifest():
    files = [Path(__file__), Path(__file__).with_name("runtime.py"), FROZEN,
             ROOT/"creamem3/online_memory_v3.py", ROOT/"creamem3/online_operation_context_v8.py",
             ROOT/"creamem3/semantic_context_v8.py", ROOT/"creamem3/semantic_query_planner_v6.py",
             ROOT/"creamem3/strict_semantic_planner_v8.py", ROOT/"creamem3/prompts_v2.py",
             ROOT/"creamem3/graph_memory_v2.py", ROOT/"creamem3/locomo_all.py",
             ROOT/"eval/prompts.py", Path(__file__).with_name("protocol.md")]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in files}

def prepare(root, *, smoke=False):
    loaded = load_all_locomo()
    selected = loaded[:1] if smoke else loaded
    manifest = {"schema": "locomo.g35lite.full_backbone.20260908.v1",
                "backbone": BACKBONE, "judge": JUDGE, "embedding": EMBED, "embedding_dim": 768,
                "thinking_level": "minimal", "temperature_sent": False, "qa_max_output": 8192,
                "judge_initial_max_output": 2048, "judge_vote_count": 1,
                "source_hashes": source_manifest(), "dataset_sha256": file_sha(DEFAULT_DATASET_PATH),
                "qa_ids": [q["qa_id"] for x in selected for q in (x["qa"][:1] if smoke else x["qa"])],
                "qa_count_per_arm": 1 if smoke else sum(len(x["qa"]) for x in selected),
                "arms": ["session_only", "online_opctx"], "shared_fresh_writer": True,
                "historical_model_cache_reuse": False, "writer_unit": "one_physical_session",
                "smoke_only": smoke, "summary_top_k": 10, "dense_top_k": 20,
                "bm25_top_k": 20, "candidate_cap": 120, "final_top_k": 20,
                "online_implementation": "original strict prefix; no Hydrate or Offline feedback",
                "population": [{"sample_id": x["sample_id"], "sessions": len(x["records"]),
                                "empty_sessions": sum(not r["turns"] for r in x["records"]),
                                "n": len(x["qa"])} for x in selected]}
    if not smoke and manifest["qa_count_per_arm"] != 1540:
        raise StopRun("Original population no longer has1540 questions")
    if any(x["empty_sessions"] for x in manifest["population"]):
        raise StopRun("Unexpected empty session; requires inspection, no silent dropping")
    seal(root/"manifest.json", manifest)
    return loaded, manifest

class Lease:
    def __init__(self, root):
        self.root = root
    def __enter__(self):
        import msvcrt
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/"runner.lock").open("a+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            raise StopRun("Another runner owns this run directory")
        seal(self.root/f"launch_{time.time_ns()}.json",
             {"pid": os.getpid(), "utc": now(), "args": sys.argv, "script_sha256": file_sha(__file__)})
        return self
    def __exit__(self, *_):
        import msvcrt
        self.handle.seek(0)
        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()

def build_one(loaded, root, runtime, manifest_sha, *, smoke=False):
    sample = loaded["sample_id"]
    directory = root/"graphs"/sample
    records = loaded["records"][:2] if smoke else loaded["records"]
    legal_ids = [turn["dia_id"] for r in records for turn in r["turns"]]
    session_path = directory/"session.json"
    online_path = directory/"online.json"
    began = time.perf_counter()
    emit(f"BUILD {sample}: {len(records)} sessions")
    if session_path.exists():
        session = read(session_path)["graph"]
    else:
        session = graph_api.build_session_graph(records, runtime.chat_fn(sample), runtime.embed, workers=2)
        seal(session_path, {"manifest_sha256": manifest_sha, "graph": session})
    if read(session_path)["manifest_sha256"] != manifest_sha:
        raise StopRun("Session graph manifest drift")
    session_audit = graph_api.audit_graph(session, legal_dia_ids=legal_ids, expected_sessions=len(records))
    if session_audit["status"] != "pass":
        raise StopRun(f"Session graph audit failed: {sample}")
    if online_path.exists():
        online = read(online_path)["graph"]
    else:
        online = add_online_event_views(session, runtime.chat_fn(sample), runtime.embed)
        seal(online_path, {"manifest_sha256": manifest_sha, "graph": online})
    if read(online_path)["manifest_sha256"] != manifest_sha:
        raise StopRun("Online graph manifest drift")
    online_audit = graph_api.audit_graph(online, legal_dia_ids=legal_ids, expected_sessions=len(records))
    causal_audit = audit_online_graph(session, online)
    if online_audit["status"] != "pass" or causal_audit["status"] != "pass":
        raise StopRun(f"Online or causal graph audit failed: {sample}")
    if session["fact_nodes"] != online["fact_nodes"]:
        raise StopRun(f"Online changed shared Writer facts: {sample}")
    steps = online["online_fusion"]["steps"]
    if any(step["model_error"] for step in steps):
        raise StopRun(f"Unrecovered Online call/parse errors: {sample}")
    audit = {"session": session_audit, "online": online_audit, "causal": causal_audit,
             "same_facts": True, "session_sha256": sha(session), "online_sha256": sha(online),
             "online_rejected_operations": sum(len(s["rejected"]) for s in steps),
             "online_accepted_operations": sum(len(s["accepted"]) for s in steps)}
    seal(directory/"audit.json", audit)
    runtime.log({"kind": "graph_complete", "sample": sample, "sessions": len(records),
                 "facts": len(session["fact_nodes"]), "views": online["stats"]["event_view_nodes"],
                 "elapsed_s": time.perf_counter()-began, "rejected": audit["online_rejected_operations"]})
    emit(f"GRAPH COMPLETE {sample}; facts={len(session['fact_nodes'])}, views={online['stats']['event_view_nodes']}")
    return sample, {"session_only": session, "online_opctx": online}

def judge(runtime, scope, question, reference, answer):
    prompt = JUDGE_PROMPT.format(question=question, answer=reference, response=answer)
    raw = ""
    for attempt, tokens in enumerate([2048, 4096, 8192], 1):
        raw = runtime.chat(f"{scope}/parse{attempt}", "judge", prompt, model=JUDGE, max_tokens=tokens)
        tags = re.findall(r"\[\[(yes|no)\]\]", raw.lower())
        if len(tags) == 1:
            return int(tags[0] == "yes"), raw, attempt
    raise StopRun(f"Invalid exact-tag Judge after three output budgets: {scope}")

def question_pair(qa, sample, graphs, indices, vector, root, runtime, manifest_sha, graph_hashes):
    # Only question text enters planner/retriever; reference/category are used below by evaluator.
    qid = qa["qa_id"]
    output = []
    for arm in ["session_only", "online_opctx"]:
        path = root/"rows"/arm/f"{qid}.json"
        graph = graphs[arm]
        identity = {"manifest_sha256": manifest_sha, "qa_sha256": sha(qa),
                    "graph_sha256": graph_hashes[arm], "arm": arm, "vector_sha256": sha(vector)}
        if path.exists():
            row = read(path)
            if row["identity"] != identity:
                raise StopRun(f"Result binding drift: {qid}/{arm}")
            output.append(row)
            continue
        scope = f"{sample}/{arm}/{qid}"
        runtime.log({"kind": "row_start", "scope": scope})
        started = time.perf_counter()
        base, trace = indices[arm].search(qa["question"], vector, runtime.chat_fn(scope))
        selected = trace["rerank"]["selected_fact_ids"]
        if len(selected) != 20 or len(set(selected)) != 20:
            raise StopRun(f"Final selection not exactly20: {scope}")
        if trace["candidates_after_cap"] > 120:
            raise StopRun(f"Candidate cap exceeded: {scope}")
        plan = None
        rendering = None
        if arm == "online_opctx":
            planning, raw_plan = classify_strict(qa["question"], chat_fn=runtime.chat_fn(scope), model=BACKBONE)
            plan = {"result": planning.to_dict(), "raw": raw_plan}
            context, rendering = materialize_online_operation_context(
                planning.plan.operation,
                {"question": qa["question"], "retrieved": base, "retrieval_trace": copy.deepcopy(trace)},
                graph)
        else:
            context = base
            if trace["assembly"]["rendered_event_view_ids"]:
                raise StopRun("Session-only rendered event views")
        seal(root/"retrieval"/arm/f"{qid}.json",
             {"identity": identity, "question": qa["question"], "trace": trace, "plan": plan,
              "context": context, "context_sha256": sha(context), "materializer": rendering})
        prompt = QA_PROMPT.format(retrieved_texts=context, question=qa["question"])
        response = runtime.chat(scope, "qa", prompt, max_tokens=8192)
        correct, raw_judge, judge_attempts = judge(runtime, scope, qa["question"], qa["answer"], response)
        row = {"identity": identity, "qa_id": qid, "sample_id": sample, "question": qa["question"],
               "answer": qa["answer"], "category": qa["category"], "category_name": qa["category_name"],
               "response": response, "correct": correct, "judge_raw": raw_judge,
               "judge_parse_attempts": judge_attempts, "selected_fact_ids": selected,
               "retrieved_chars": len(context), "context_sha256": sha(context),
               "rerank_fallback": trace["rerank"]["fallback"],
               "planner_fallback": plan is not None and plan["result"]["source"] != "gemini",
               "elapsed_s": time.perf_counter()-started}
        seal(path, row)
        runtime.log({"kind": "row_complete", "scope": scope, "correct": correct,
                     "elapsed_s": row["elapsed_s"], "rerank_fallback": row["rerank_fallback"]})
        output.append(row)
    return output

def summarize(root, expected_ids, *, complete=False):
    arms = {}
    by_id = {}
    for arm in ["session_only", "online_opctx"]:
        rows = [read(p) for p in sorted((root/"rows"/arm).glob("*.json"))]
        by_id[arm] = {r["qa_id"]: r for r in rows}
        if set(by_id[arm])-set(expected_ids) or len(by_id[arm]) != len(rows):
            raise StopRun("Unexpected/duplicate result IDs")
        groups = {}
        for category in sorted({r["category_name"] for r in rows}):
            sub = [r for r in rows if r["category_name"] == category]
            groups[category] = {"n": len(sub), "correct": sum(r["correct"] for r in sub),
                                "accuracy": 100*sum(r["correct"] for r in sub)/len(sub)}
        arms[arm] = {"complete": len(rows), "expected": len(expected_ids),
                     "correct": sum(r["correct"] for r in rows),
                     "accuracy_completed": 100*sum(r["correct"] for r in rows)/len(rows) if rows else None,
                     "by_category": groups, "rerank_fallbacks": sum(r["rerank_fallback"] for r in rows),
                     "planner_fallbacks": sum(r["planner_fallback"] for r in rows),
                     "mean_context_chars": sum(r["retrieved_chars"] for r in rows)/len(rows) if rows else None}
    common = sorted(set(by_id["session_only"]) & set(by_id["online_opctx"]))
    comparisons = [{"qa_id": key, "session_correct": by_id["session_only"][key]["correct"],
                    "online_correct": by_id["online_opctx"][key]["correct"],
                    "top20_overlap": len(set(by_id["session_only"][key]["selected_fact_ids"]) &
                                         set(by_id["online_opctx"][key]["selected_fact_ids"]))} for key in common]
    wins = sum(x["online_correct"] > x["session_correct"] for x in comparisons)
    losses = sum(x["online_correct"] < x["session_correct"] for x in comparisons)
    result = {"utc": now(), "complete": complete, "arms": arms, "paired_n": len(common),
              "online_only_correct": wins, "session_only_correct": losses,
              "both_correct": sum(x["online_correct"] and x["session_correct"] for x in comparisons),
              "both_wrong": sum(not x["online_correct"] and not x["session_correct"] for x in comparisons),
              "mean_top20_overlap": sum(x["top20_overlap"] for x in comparisons)/len(common) if common else None,
              "paired_rows": comparisons}
    if complete:
        if any(set(rows) != set(expected_ids) for rows in by_id.values()):
            raise StopRun("Attempted complete summary before all paired rows")
        seal(root/"summary.json", result)
    return result

def run(root, *, smoke=False, allow=False):
    loaded, manifest = prepare(root, smoke=smoke)
    runtime = Runtime(root, allow_model_calls=allow, concurrency=8)
    runtime.log({"kind": "run_start", "pid": os.getpid(), "smoke": smoke})
    manifest_sha = sha(manifest)
    selected = loaded[:1] if smoke else loaded
    graphs = {}
    with ThreadPoolExecutor(max_workers=1 if smoke else 4) as pool:
        pending = [pool.submit(build_one, item, root, runtime, manifest_sha, smoke=smoke) for item in selected]
        try:
            for future in as_completed(pending):
                sample, pair = future.result()
                graphs[sample] = pair
        except BaseException:
            runtime.stopped.set()
            for future in pending:
                future.cancel()
            raise
    indices = {sample: {arm: graph_api.GraphMemoryIndex(graph) for arm, graph in pair.items()}
               for sample, pair in graphs.items()}
    graph_hashes = {sample: {arm: sha(graph) for arm, graph in pair.items()}
                    for sample, pair in graphs.items()}
    flat = [(item["sample_id"], row) for item in selected for row in (item["qa"][:1] if smoke else item["qa"])]
    vectors = runtime.embed([qa["question"] for _, qa in flat], task_type="RETRIEVAL_QUERY")
    seal(root/"question_vectors.json", {"manifest_sha256": manifest_sha,
                                        "ids": [qa["qa_id"] for _,qa in flat], "vectors": vectors})
    completed = 0
    with ThreadPoolExecutor(max_workers=1 if smoke else 8) as pool:
        pending = [pool.submit(question_pair, qa, sample, graphs[sample], indices[sample], vector,
                               root, runtime, manifest_sha, graph_hashes[sample])
                   for (sample, qa), vector in zip(flat, vectors)]
        try:
            for future in as_completed(pending):
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(flat):
                    emit(f"PAIRS COMPLETE {completed}/{len(flat)}")
        except BaseException:
            runtime.stopped.set()
            for future in pending:
                future.cancel()
            raise
    summary = summarize(root, manifest["qa_ids"], complete=True)
    runtime.log({"kind": "run_complete", "arms": summary["arms"]})
    emit(json.dumps({k:v for k,v in summary.items() if k!="paired_rows"}, ensure_ascii=False))
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "full", "status"])
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    root = args.run_root / "smoke" if args.command == "smoke" else args.run_root
    if args.command == "status":
        manifest = read(root/"manifest.json")
        summary = read(root/"summary.json") if (root/"summary.json").exists() else summarize(root, manifest["qa_ids"])
        print(json.dumps({k:v for k,v in summary.items() if k!="paired_rows"}, ensure_ascii=False, indent=2))
        return
    if args.command == "prepare":
        _, manifest = prepare(root)
        emit(f"PREPARED {manifest['qa_count_per_arm']} paired questions; no provider calls")
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT/".env", override=False)
    if args.command == "full":
        smoke_path = root/"smoke"/"summary.json"
        if not smoke_path.exists() or not read(smoke_path)["complete"]:
            raise StopRun("Complete isolated contract smoke required before full dispatch")
        if read(root/"smoke"/"manifest.json")["source_hashes"] != source_manifest():
            raise StopRun("Source changed after smoke; rerun a new isolated contract smoke")
        if (root/"summary.json").exists():
            prepare(root)
            emit("Already complete; no model calls")
            return
    with Lease(root):
        try:
            run(root, smoke=args.command=="smoke", allow=args.allow_model_calls)
        except BaseException as exc:
            seal(root/f"stop_{time.time_ns()}.json",
                 {"utc": now(), "pid": os.getpid(), "type": type(exc).__name__, "message": str(exc)[:1000]})
            emit(f"STOP {type(exc).__name__}: {exc}")
            raise

if __name__ == "__main__":
    main()

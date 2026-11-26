"""Read-only source checks; write only new derived audit artifacts."""
import copy
import json
from pathlib import Path
from collections import Counter
from creamem3.experiments.locomo_online_only_20260909 import run as old
from creamem3.experiments.locomo_online_only_20260909 import audit as binding
from creamem3.experiments.locomo_online_only_20260909.runtime import read, seal, sha, response_text, StopRun
from creamem3.experiments.locomo_joint_revision_20260909 import run as joint

ROOT = Path(r"D:\creamem3_cache\locomo_joint_followup_20260909_r1")
CACHED = Path(r"D:\creamem3_cache\locomo_state_norerank_20260909_r1")


def literal_search(index, question, vector):
    candidates, trace = index._candidate_pool(question, vector)
    selected = candidates[:20]
    context, assembly = index._assemble(selected, trace["top_summaries"])
    trace["assembly"] = assembly
    trace["rerank"] = {"selected_fact_ids": [x["fact"]["id"] for x in selected],
                       "attempts": 0, "fallback": False, "model_used": False}
    return context, trace


def check_bound(root, q, row):
    identity = binding.generation_identity("qa", old.QA_PROMPT.format(question=q["question"], retrieved_texts=row["context"]), 8192)
    if binding.bound_text(root, identity) != row["response"]:
        raise StopRun("QA receipt mismatch")
    prompt = old.JUDGE_PROMPT.format(question=q["question"], answer=q["answer"], response=row["response"])
    for budget in [2048, 4096, 8192]:
        identity = binding.generation_identity("judge", prompt, budget)
        if (root/"calls/provider/judge"/sha(identity)/"final.json").exists():
            actual = binding.bound_text(root, identity)
            tags = old.re.findall(r"\[\[(yes|no)\]\]", actual.lower())
            if len(tags) == 1:
                if row["judge_raw"] != actual or row["correct"] != int(tags[0] == "yes"):
                    raise StopRun("Judge receipt mismatch")
                return
    raise StopRun("No valid Judge receipt")


def literal_equivalence():
    loaded, parent = old.prepare()
    vector_record = read(old.VECTOR_FILE)
    vectors = dict(zip(vector_record["ids"], vector_record["vectors"]))
    prior_protocol = read(CACHED/"protocol.json")
    records = {"session_strong": [], "opctx_original": []}
    rows = {}
    for arm, cached_arm in [("session_strong", "session_cosine_rich"), ("opctx_original", "original_cosine_rich")]:
        for conv in loaded:
            sample = conv["sample_id"]
            graph = old.get_source_graph(sample, arm, parent)
            index = joint.OriginalIndex(graph)
            for q in conv["qa"]:
                qid = q["qa_id"]
                if qid not in parent["dev_ids"]:
                    continue
                context, trace = literal_search(index, q["question"], vectors[qid])
                if arm != "session_strong":
                    operation = old.old_json(old.BASE_CONTEXT/(qid+".json"))["operation"]
                    context, _ = old.materialize_online_operation_context(operation,
                        {"question":q["question"], "retrieved":context, "retrieval_trace":copy.deepcopy(trace)}, graph)
                source_path = CACHED/"rows"/cached_arm/(qid+".json")
                row = read(source_path)
                if (row["context"] != context or row["selected_fact_ids"] != trace["rerank"]["selected_fact_ids"]
                        or row["identity"]["qa_sha256"] != sha(q) or row["identity"]["graph_sha256"] != sha(graph)
                        or row["identity"]["protocol_sha256"] != sha(prior_protocol)):
                    raise StopRun("Literal joint-no-rerank differs from cached reader: " + arm + "/" + qid)
                check_bound(CACHED, q, row)
                dense = [index.facts[i]["id"] for i in index._rank_indices(index._scores(index.fact_matrix, vectors[qid]),20)]
                if dense != row["selected_fact_ids"]:
                    raise StopRun("Expected dense dominance identity does not hold")
                records[arm].append({"qa_id":qid, "source_path":str(source_path), "source_sha256":sha(row),
                    "context_sha256":sha(context), "selection":row["selected_fact_ids"], "correct":row["correct"],
                    "union_before_cap":trace["union_before_cap"], "top_summaries":trace["top_summaries"]})
                rows[(arm,qid)] = row
    pairs = [(rows[("opctx_original",q)],rows[("session_strong",q)]) for q in parent["dev_ids"]]
    result = {"status":"pass", "new_model_calls":0, "n":len(pairs), "exact_bytewise_context_matches":200,
        "physical_qa_judge_bindings":400, "protocol_sha256":sha(parent), "proof":
        "Union contains global Dense20; union is sorted by same dense scores/ties; selecting its first20 equals Dense20",
        "correct":{a:sum(r["correct"] for r in values) for a,values in records.items()},
        "wins":sum(a["correct"]>b["correct"] for a,b in pairs),
        "losses":sum(a["correct"]<b["correct"] for a,b in pairs),
        "same_selected_ids":sum(a["selected_fact_ids"]==b["selected_fact_ids"] for a,b in pairs),
        "same_context":sum(a["context"]==b["context"] for a,b in pairs), "rows":records}
    seal(ROOT/"analysis/literal_joint_no_rerank.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False))


def failed_rerank():
    root = joint.RUN
    protocol = read(root/"protocol.json")
    done = {p.stem for p in (root/"rows/session_strong").glob("*.json")}
    events = [json.loads(s) for s in (root/"calls/events.jsonl").read_text(encoding="utf-8").splitlines()]
    scopes = {}
    for event in events:
        if event.get("kind") == "logical_request" and event.get("stage") == "rerank":
            scopes[event["body_sha256"]] = event["scope"]
    bad = []
    for path in (root/"calls/provider/rerank").glob("*/request.json"):
        request = read(path)
        final = read(path.with_name("final.json"))
        response = read(path.with_name("attempt_%02d.response.json" % final["attempt"]))
        raw = response_text(response["response"])
        value = old.graph_api._parse_json_object(raw)
        ids = value.get("ranked_fact_ids") if isinstance(value,dict) else None
        if not isinstance(ids,list) or len(ids)!=20 or len(set(ids))!=20:
            bad.append({"scope":scopes.get(sha(request["body"])), "request":sha(request),
                "n":len(ids) if isinstance(ids,list) else None, "unique":len(set(ids)) if isinstance(ids,list) else None,
                "finish":response["response"].get("candidates",[{}])[0].get("finishReason"), "raw":raw})
    result = {"status":"diagnosed", "completed_rows":len(done), "missing":sorted(set(protocol["dev_ids"])-done),
              "physical_errors":len(list((root/"calls/provider").glob("*/*/*.error.json"))), "invalid_outputs":bad}
    seal(ROOT/"analysis/stopped_rerank.json", result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    failed_rerank()
    literal_equivalence()

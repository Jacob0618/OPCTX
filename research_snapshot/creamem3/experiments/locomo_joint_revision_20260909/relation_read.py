"""Expose grounded cross-session relations to selection, not just isolated facts."""
import argparse
import copy
import json
import time
from creamem3.experiments.locomo_joint_revision_20260909 import followup_eval as base
from creamem3.experiments.locomo_joint_revision_20260909.followup_audit import ROOT

RUN = ROOT/"relation_aware_rerank"
old, read, seal, sha, file_sha, StopRun = base.old, base.read, base.seal, base.sha, base.file_sha, base.StopRun

INSTRUCTION = """
The following records are derived cross-session memory relations, each with its
complete supporting fact IDs from the SAME candidate facts above. They are NOT
extra observations. Topic similarity alone is not evidence of causation. Use a
relation only when its cited facts actually establish the relation and help this
question. Historical states are not current states; a later mention can recall an
earlier event. Preserve the difference between a plan and an accomplished event.

Select exactly {target} unique candidate fact IDs. When a relation contributes to
your selection, include its relation_id in selected_relation_ids AND include ALL
its support_fact_ids among your ranked_fact_ids. Independent facts are also
allowed; do not waste the budget on an irrelevant relation just to complete it.
Do not output an answer. Output STRICT JSON with exactly these two fields:
ranked_fact_ids (exactly {target} IDs), selected_relation_ids (possibly empty).

Derived relations with fully available support:
{packets}
"""


def packets_for(index, candidates):
    allowed = {c["fact"]["id"] for c in candidates}
    routed = {}
    for candidate in candidates:
        channels = candidate["channels"]
        for view_id, rank in zip(channels.get("summary_ids",[]),channels.get("summary_ranks",[])):
            routed[view_id] = rank
    packets = []
    seen = set()
    for view_id in sorted(routed,key=lambda v:(routed[v],v)):
        view = index.summary_by_id[view_id]
        if view["node_type"] != "event_view":continue
        for claim in view.get("summary_claims",[]):
            support = sorted(set(claim["support_fact_ids"]))
            if not 2 <= len(support) <= index.rerank_top_k or not set(support)<=allowed:continue
            if len({index.fact_by_id[f]["session_id"] for f in support})<2:continue
            value = {"text":claim["text"],"support_fact_ids":support}
            key = sha(value)
            if key in seen:continue
            seen.add(key)
            packets.append({"relation_id":"P"+key[:20],**value})
    if len({p["relation_id"] for p in packets})!=len(packets):raise StopRun("Packet ID collision")
    return packets


def validate(value, by_id, packets, target):
    if not isinstance(value,dict) or set(value)!={"ranked_fact_ids","selected_relation_ids"}:
        raise ValueError("Expected ranked_fact_ids and selected_relation_ids")
    ids,chosen = value["ranked_fact_ids"], value["selected_relation_ids"]
    if (not isinstance(ids,list) or any(not isinstance(v,str) for v in ids) or len(ids)!=target
            or len(set(ids))!=target or not set(ids)<=set(by_id)):
        raise ValueError("Expected exactly twenty unique supplied fact IDs")
    if (not isinstance(chosen,list) or any(not isinstance(v,str) for v in chosen)
            or len(set(chosen))!=len(chosen) or not set(chosen)<=set(packets)):
        raise ValueError("Unknown/duplicate selected relation IDs")
    for rid in chosen:
        missing = sorted(set(packets[rid]["support_fact_ids"])-set(ids))
        if missing:raise ValueError("Selected relation "+rid+" missing support facts "+json.dumps(missing))
    return ids,chosen


class RelationIndex(base.RepairedIndex):
    def _rerank(self, question, candidates, rerank_chat, *, max_attempts=2):
        packets = packets_for(self,candidates)
        if not packets:
            return super()._rerank(question,candidates,rerank_chat,max_attempts=max_attempts)
        target = min(self.rerank_top_k,len(candidates))
        by_id = {x["fact"]["id"]:x for x in candidates}
        packet_map = {p["relation_id"]:p for p in packets}
        prompt = old.graph_api.FACT_RERANK_PROMPT.format(top_k=target,question=question,
            candidates_json=json.dumps([{"fact_id":x["fact"]["id"],"text":x["fact"]["text"]} for x in candidates],
                                       ensure_ascii=False,separators=(",",":")))
        addition = INSTRUCTION.format(target=target,packets=json.dumps(packets,ensure_ascii=False,separators=(",",":")))
        prompt += addition
        schema = copy.deepcopy(base.output_schema(target))
        schema["properties"]["selected_relation_ids"] = {"type":"ARRAY","items":{"type":"STRING"}}
        schema["required"].append("selected_relation_ids")
        error = None
        for attempt in range(2):
            current = prompt if error is None else prompt+"\nValidation failed: "+error+"\nReturn a valid selection within the unchanged fact budget."
            raw = rerank_chat(current,tag="creamem3_v2_rerank",response_schema=schema)
            value = old.graph_api._parse_json_object(raw)
            try:
                ids, chosen = validate(value,by_id,packet_map,target)
            except ValueError as exc:
                error = str(exc);continue
            return [by_id[f] for f in ids],{"attempts":attempt+1,"fallback":False,"error":None,
                "relation_aware":True,"available_packets":packets,"selected_relation_ids":chosen,
                "selected_support_complete":True,"additional_prompt_chars":len(addition),
                "no_additional_candidate_facts":True,"qa_renderer_unchanged":True}
        raise StopRun("Relation-aware selection failed support/budget validation: "+str(error))


def register():
    registration = {"source_sha256":file_sha(__file__),"base_eval_source_sha256":file_sha(base.__file__),
        "hypothesis":"Exposing Online relations and requiring complete selected support avoids breaking multi-premise evidence at final20",
        "changes":"Rerank input and output relationship contract only;same graphs/candidates120/final20/QA renderer/Judge",
        "no_new_online_or_writer":True,"relation_source":"Already routed event views only",
        "eligibility":"Cross-session claims with all support in120 and support size<=20;exact duplicates removed",
        "no_fitted_weights_or_gold":True,"packet_count_cap":None,"extra_rerank_input_cost_must_be_reported":True,
        "not_a_pure_write_only_ablation":True,"registered_before_new_repaired_variant_scores_read":True,
        "instruction":INSTRUCTION,"baseline_root":str(base.RUN)}
    seal(RUN/"relation_registration.json",registration)
    return registration


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["register","run"]);args=parser.parse_args()
    registration=register()
    if args.command=="register":return
    completed=read(base.RUN/"complete.json")
    if not completed["audit"]["pass"]:raise StopRun("Plain rerank experiment is not valid and complete")
    with old.Lease(RUN):
        loaded,protocol=base.prepare(RUN,"relation_aware:"+sha(registration))
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/".env",override=False)
        old.graph_api.GraphMemoryIndex=RelationIndex
        runtime=base.Runtime(RUN/"calls",parents=(base.RUN,base.joint.RUN),allow_model_calls=True,concurrency=24,timeout=180)
        try:
            old.evaluate(RUN,base.ARMS,loaded,protocol,runtime,qa_workers=24)
            base.finish(RUN,loaded,protocol,index_class=RelationIndex)
        except BaseException as exc:
            runtime.stopped.set()
            seal(RUN/("stop_"+str(time.time_ns())+".json"),{"utc":base.now(),"type":type(exc).__name__,"message":str(exc)[:500]})
            raise


if __name__=="__main__":main()

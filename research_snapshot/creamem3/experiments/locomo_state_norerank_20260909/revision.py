"""Grounded, retractable derived-state operations over immutable Writer facts.

This implements explicit local revision and dependency invalidation, NOT a
proof of natural-language entailment or a general AGM/minimum-edit solver.
"""
from __future__ import annotations
import argparse
import copy
import json
import re
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from creamem3 import online_memory_v3 as legacy
from creamem3.experiments.locomo_state_norerank_20260909 import run
from creamem3.experiments.locomo_state_norerank_20260909.runtime import Runtime, read, seal, sha, file_sha, now, StopRun
from creamem3.experiments.locomo_state_norerank_20260909.selector import normalize

PROMPT = """Maintain a causal, revisable event memory using ONLY the supplied evidence.
Writer facts are immutable evidence. Your derived claims, event membership and
current-state assertions can be corrected; historical evidence must not be erased.
You receive the current session, selected prior evidence, and selected existing
claims WITH their complete supporting facts and current-state dependencies.

Return {"operations":[...]} with at most12 operations. Each operation contains:
action: add | replace | retract | dispute;
target_view_id: an exposed event ID to continue, or empty to create a distinct event;
old_claim_ids: exposed existing claim IDs affected by this operation;
revision_kind: none | supersede | correct;
effective_time: explicit YYYY-MM-DD event date when known, otherwise empty;
reason_fact_ids: supplied evidence explaining the revision, including a CURRENT fact;
reason: a concise grounded explanation;
new_claims: zero or more precise new cross-session claims.
Each claim has kind (relation or state), text, support_fact_ids, state_key,
state_value, valid_from, valid_to, depends_on_current_claim_ids.

ADD: old_claim_ids empty, revision_kind none; add a genuinely new relation/state.
REPLACE: cite old claims and supply corrected/new claims. supersede is ONLY for
a genuine change of a state: the old state remains historical with a dated end.
correct withdraws an erroneous derived assertion, not its underlying evidence.
RETRACT: old claims were unsupported/wrong; no new claims, revision_kind correct.
DISPUTE: evidence is incompatible or its temporal order uncertain; preserve the
conflict as unresolved instead of arbitrarily selecting a current value.

Every NEW claim must cite both current and earlier-session facts indispensable
to the relationship. Use the smallest COMPLETE support, not a topical bag.
Do not generate a person's general profile, copy unrelated historical claims,
or confuse co-occurrence with causation. Keep exact names, quantities, roles,
negation, intention versus completion and absolute/relative event time.

For kind=state use a stable subject|single-valued-property state_key and an exact
state_value. Only use state for a genuinely exclusive current property. Hobbies,
friends, pets and lists may coexist; describe them as relations, not exclusive slots.
If a known active single-valued state changes, explicitly REPLACE it; do not ADD
a contradictory active value. A later conversation may recall an earlier event:
last mention is NOT automatically latest event time. Do not invent precise dates.
valid_from/valid_to are explicit YYYY-MM-DD or empty. Historical facts remain
available for historical questions after a state is superseded.
For supersede, each old state must have a replacement state with the SAME
state_key, a genuinely changed state_value, and valid_from=effective_time.

depends_on_current_claim_ids lists only supplied claims whose CONTINUING validity
is logically necessary for the new assertion. A historical citation is not a
current-state dependency. The system marks downstream current dependencies
unresolved when their basis is retracted/superseded, and preserves their evidence.
An empty operations list is correct when no grounded change is justified.
Never use outside knowledge or anticipate future sessions. Change only affected
derived claims; all unmentioned records are retained by the system.

INPUT:
"""

STR = {"type":"STRING"}
IDS = {"type":"ARRAY","items":STR}
CLAIM = {"type":"OBJECT","properties":{
    "kind":{"type":"STRING","enum":["relation","state"]},"text":STR,
    "support_fact_ids":IDS,"state_key":STR,"state_value":STR,"valid_from":STR,"valid_to":STR,
    "depends_on_current_claim_ids":IDS},
    "required":["kind","text","support_fact_ids","state_key","state_value","valid_from","valid_to","depends_on_current_claim_ids"]}
OPERATION = {"type":"OBJECT","properties":{
    "action":{"type":"STRING","enum":["add","replace","retract","dispute"]},
    "target_view_id":STR,"old_claim_ids":IDS,
    "revision_kind":{"type":"STRING","enum":["none","supersede","correct"]},
    "effective_time":STR,"reason_fact_ids":IDS,"reason":STR,
    "new_claims":{"type":"ARRAY","maxItems":12,"items":CLAIM}},
    "required":["action","target_view_id","old_claim_ids","revision_kind","effective_time","reason_fact_ids","reason","new_claims"]}
SCHEMA = {"type":"OBJECT","properties":{"operations":{"type":"ARRAY","maxItems":12,"items":OPERATION}},"required":["operations"]}


def empty_state():
    return {"claims":{},"next_claim":1,"next_view":1,"ledger":[]}


def text(value):
    if not isinstance(value,str):raise ValueError("Expected text")
    return " ".join(value.split())


def ids(value):
    if not isinstance(value,list) or any(not isinstance(v,str) for v in value):raise ValueError("Invalid IDs")
    if len(value)!=len(set(value)):raise ValueError("Duplicate ID")
    return list(value)


def date(value):
    from datetime import date as Date
    value=text(value)
    if value:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}",value):raise ValueError("Date must be explicit ISO or empty")
        Date.fromisoformat(value)
    return value


def invalidate_dependencies(claims, invalidated):
    pending=set(invalidated);changed=[]
    while pending:
        target=pending.pop()
        for cid,claim in claims.items():
            if claim["status"]=="active" and target in claim["depends_on_current_claim_ids"]:
                claim["status"]="unresolved"
                claim["invalidated_by"]=sorted(set(claim.get("invalidated_by",[]))|{target})
                pending.add(cid);changed.append(cid)
    return sorted(set(changed))


def apply_one(state, operation, exposed, facts, session_id):
    if not isinstance(operation,dict) or set(operation)!=set(OPERATION["required"]):raise ValueError("Invalid operation shape")
    result=copy.deepcopy(state);claims=result["claims"]
    action=operation["action"];kind=operation["revision_kind"]
    old_ids=ids(operation["old_claim_ids"]);target=text(operation["target_view_id"])
    current=set(exposed["current_fact_ids"]);allowed=set(exposed["fact_ids"])
    old_set=set(exposed["claim_ids"]);views=set(exposed["view_ids"])
    reasons=ids(operation["reason_fact_ids"]);effective=date(operation["effective_time"])
    if not set(reasons)<=allowed or not set(reasons)&current:raise ValueError("Revision reason lacks exposed current evidence")
    if not text(operation["reason"]):raise ValueError("Missing reason")
    if not set(old_ids)<=old_set or any(i not in claims for i in old_ids):raise ValueError("Unexposed revision target")
    if target and target not in views:raise ValueError("Unexposed event target")
    new=operation["new_claims"]
    if not isinstance(new,list) or len(new)>12:raise ValueError("Invalid new claims")
    if action=="add":
        if old_ids or kind!="none" or not new:raise ValueError("Invalid add")
    elif action=="replace":
        if not old_ids or not new or kind not in {"correct","supersede"}:raise ValueError("Invalid replace")
    elif action=="retract":
        if not old_ids or new or kind!="correct":raise ValueError("Invalid retract")
    elif action=="dispute":
        if not old_ids or kind!="none":raise ValueError("Invalid dispute")
    else:raise ValueError("Unknown action")
    if kind=="supersede" and any(claims[i]["kind"]!="state" for i in old_ids):raise ValueError("Only states can be superseded")
    if kind=="supersede" and not effective:raise ValueError("Unknown transition time: use dispute instead of forced chronology")
    for cid in old_ids:
        claim=claims[cid]
        if claim["status"] not in {"active","unresolved"}:raise ValueError("Target is no longer active/unresolved")
        if kind=="supersede":
            if claim["valid_from"] and effective<claim["valid_from"]:raise ValueError("Backwards state interval")
            claim["status"]="historical";claim["valid_to"]=effective
        elif action=="dispute":claim["status"]="unresolved"
        else:claim["status"]="retracted"
        claim["changed_at_session"]=session_id
    dependent_changes=invalidate_dependencies(claims,old_ids)
    if new and not target:
        target=f"RV{result['next_view']:04d}";result["next_view"]+=1
    created=[]
    for value in new:
        if not isinstance(value,dict) or set(value)!=set(CLAIM["required"]):raise ValueError("Invalid claim shape")
        support=ids(value["support_fact_ids"]);support_set=set(support)
        if not support_set<=allowed or not support_set<=facts.keys():raise ValueError("Unknown/unexposed support")
        if not support_set&current or not any(int(facts[f]["session_id"])<session_id for f in support):raise ValueError("Claim must connect current and earlier evidence")
        if any(int(facts[f]["session_id"])>session_id for f in support):raise ValueError("Future support")
        claim_text=text(value["text"]);ckind=value["kind"]
        if not claim_text or ckind not in {"state","relation"}:raise ValueError("Invalid claim text/kind")
        key=text(value["state_key"]).casefold();state_value=text(value["state_value"])
        if ckind=="state" and (not key or not state_value):raise ValueError("State lacks key/value")
        if ckind=="relation" and (key or state_value):raise ValueError("Relations must not claim exclusive slots")
        start=date(value["valid_from"]);end=date(value["valid_to"])
        if start and end and start>end:raise ValueError("Invalid state interval")
        if ckind=="state" and end:raise ValueError("New active state has a closed interval")
        deps=ids(value["depends_on_current_claim_ids"])
        if not set(deps)<=old_set or any(claims[d]["status"]!="active" for d in deps):raise ValueError("Current dependency unavailable")
        if ckind=="state" and action!="dispute":
            if any(c["status"]=="active" and c["kind"]=="state" and c["state_key"]==key
                   and c["state_value"].casefold()!=state_value.casefold() for c in claims.values()):
                raise ValueError("Conflicting current slot requires explicit replacement or dispute")
        canonical={"kind":ckind,"text":claim_text,"support_fact_ids":sorted(support),
            "state_key":key,"state_value":state_value,"valid_from":start,"valid_to":end,
            "depends_on_current_claim_ids":sorted(deps)}
        if any(all(c[k]==v for k,v in canonical.items()) and c["status"]=="active" for c in claims.values()):
            continue
        cid=f"RC{result['next_claim']:05d}";result["next_claim"]+=1
        claims[cid]={**canonical,"id":cid,"view_id":target,"status":"unresolved" if action=="dispute" else "active",
            "created_at_session":session_id,"changed_at_session":session_id}
        created.append(cid)
    if kind=="supersede":
        for old_id in old_ids:
            old_claim=claims[old_id]
            replacements=[claims[cid] for cid in created if claims[cid]["kind"]=="state"
                          and claims[cid]["state_key"]==old_claim["state_key"]]
            if not replacements or any(c["valid_from"]!=effective or c["state_value"].casefold()==old_claim["state_value"].casefold()
                                       for c in replacements):
                raise ValueError("Supersession requires a same-slot changed value at the effective time")
    ledger={"session_id":session_id,"operation":copy.deepcopy(operation),"created_claim_ids":created,
            "invalidated_dependency_ids":dependent_changes}
    result["ledger"].append(ledger)
    return result,ledger


def claim_text(claim):
    if claim["status"]=="historical":
        interval=f"from {claim['valid_from'] or 'unknown start'} until before {claim['valid_to'] or 'unknown end'}"
        return f"Historical state (valid {interval}; not asserted as current): {claim['text']}"
    return claim["text"]


def views_from_state(state, fact_by_id, include_history=False):
    groups=defaultdict(list)
    for claim in state["claims"].values():
        if claim["status"]=="active":groups[claim["view_id"]].append(claim)
        elif include_history and claim["status"]=="historical":groups["HV"+claim["id"]].append(claim)
    views={}
    for view_id,claims in sorted(groups.items()):
        support=sorted({f for c in claims for f in c["support_fact_ids"]})
        view=legacy._make_view(view_id=view_id,view_type="update_chain" if any(c["kind"]=="state" for c in claims) else "continuing_process",
            claims=[{"text":claim_text(c),"support_fact_ids":c["support_fact_ids"]} for c in claims],
            membership=support,fact_by_id=fact_by_id,created_at_session=min(c["created_at_session"] for c in claims),
            updated_at_session=max(c["changed_at_session"] for c in claims),revision=1)
        vec=normalize([fact_by_id[f]["embedding"] for f in support]).sum(axis=0)
        view["embedding"]=normalize(vec).tolist()
        views[view_id]=view
    return views


def build_input(base, state, session_id):
    fact_by_id=base["fact_by_id"];current=base["facts_by_session"][session_id]
    prior=[f for f in base["facts"] if int(f["session_id"])<session_id]
    views=views_from_state(state,fact_by_id)
    selected=legacy._select_candidates(current_summary=base["summary_by_session"][session_id],current_facts=current,
        prior_summaries=[s for s in base["summaries"] if int(s["session_ids"][0])<session_id],prior_facts=prior,
        active_views=views,support_by_summary=base["support_by_summary"],fact_order=base["fact_order"],
        embedding_dim=base["embedding_dim"])
    # Route claims through Top10 views, hydrate complete support + current dependencies
    # inside the same80 historical-fact budget; fill unused slots with old dense routes.
    prior_ids=set();claim_ids=set();selected_view_ids=[];skipped=[]
    for view_id in selected["active_view_ids"]:
        group=[c for c in state["claims"].values() if c["view_id"]==view_id and c["status"]=="active"]
        for claim in group:
            closure={claim["id"]};queue=[claim["id"]]
            while queue:
                cid=queue.pop()
                for dependency in state["claims"][cid]["depends_on_current_claim_ids"]:
                    if dependency not in closure:closure.add(dependency);queue.append(dependency)
            support={f for cid in closure for f in state["claims"][cid]["support_fact_ids"]}
            if len(prior_ids|support)>80:
                skipped.append(claim["id"]);continue
            prior_ids.update(support);claim_ids.update(closure)
            if view_id not in selected_view_ids:selected_view_ids.append(view_id)
    for fid in selected["prior_fact_ids"]:
        if len(prior_ids)<80:prior_ids.add(fid)
    ordered=sorted(prior_ids,key=base["fact_order"].__getitem__)
    exposed={"current_fact_ids":[f["id"] for f in current],"fact_ids":ordered+[f["id"] for f in current],
             "claim_ids":sorted(claim_ids),"view_ids":sorted({state['claims'][c]['view_id'] for c in claim_ids})}
    input_value={"frontier_session_id":session_id,"current_summary":base["summary_by_session"][session_id]["text"],
        "current_session_date":base["summary_by_session"][session_id].get("iso_date",""),
        "current_facts":[legacy._prompt_fact(f) for f in current],
        "prior_facts":[legacy._prompt_fact(fact_by_id[f]) for f in ordered],
        "existing_claims":[state["claims"][cid] for cid in sorted(claim_ids)],"allowed_ids":exposed}
    trace={"prior_facts":len(ordered),"exposed_claims":len(claim_ids),"skipped_incomplete_budget":skipped,
           "top_view_ids":selected["active_view_ids"],"no_model_rerank":True}
    return input_value,exposed,trace


def apply_response(state, response, exposed, facts, sid):
    if not isinstance(response,dict) or set(response)!={"operations"} or not isinstance(response["operations"],list):
        raise ValueError("Invalid response envelope")
    if len(response["operations"])>12:raise ValueError("Too many operations")
    accepted=[];rejected=[]
    for index,operation in enumerate(response["operations"]):
        try:
            state,ledger=apply_one(state,operation,exposed,facts,sid)
            accepted.append({"index":index,**ledger})
        except ValueError as exc:
            rejected.append({"index":index,"reason":str(exc),"operation":operation})
    return state,accepted,rejected


def register():
    _,_,protocol=run.prepare()
    record={"variant":"revision","source_sha256":file_sha(__file__),"prompt":PROMPT,"schema":SCHEMA,
        "protocol_sha256":sha(protocol),"legacy_route_sha256":file_sha(legacy.__file__),
        "changes":"Explicit add/replace/retract/dispute;historical state retention;current-dependency invalidation;support-complete input",
        "unchanged":"Writerfacts,QA,Judge,no-reranker selector", "prior_fact_budget":80,
        "online_max_output":16384,"initial_session_skipped":True,
        "limitations":"ID/causal/slot checks are not semantic entailment proofs;not an AGM or general minimum-edit solver",
        "no_questions_or_gold_in_build":True}
    seal(run.ROOT/"arms"/"revision_build.json",record)
    return record


def build_one(sample, parent, registration, runtime):
    base_graph=run.old.get_source_graph(sample,"session_strong",parent)
    base=legacy._base_parts(base_graph);state=empty_state();steps=[]
    identity={"base_sha256":sha(base_graph),"registration_sha256":sha(registration)}
    for ordinal,sid in enumerate(base["session_ids"]):
        input_value,exposed,trace=build_input(base,state,sid)
        before=sha(state);path=run.ROOT/"state_build/checkpoints"/sample/(f"{sid:04d}.json")
        step_identity={**identity,"session_id":sid,"input_sha256":sha(input_value),"before_sha256":before}
        if path.exists():
            cached=read(path)
            if cached["identity"]!=step_identity:raise StopRun("Revision checkpoint drift")
            response=cached["response"]
        else:
            response={"operations":[]} if ordinal==0 else runtime.structured_online("revision/"+sample+"/"+str(sid),PROMPT+json.dumps(input_value,ensure_ascii=False,separators=(",",":")),SCHEMA)
        state,accepted,rejected=apply_response(state,response,exposed,base["fact_by_id"],sid)
        checkpoint={"identity":step_identity,"input":input_value,"response":response,"accepted":accepted,
                    "rejected":rejected,"state":state,"state_sha256":sha(state),"trace":trace}
        seal(path,checkpoint);steps.append({"session_id":sid,"checkpoint_sha256":sha(checkpoint)})
        run.old.emit(f"REVISION {sample} session={sid} active={sum(c['status']=='active' for c in state['claims'].values())} accepted={len(accepted)} rejected={len(rejected)}")
    views=views_from_state(state,base["fact_by_id"],include_history=True)
    graph=copy.deepcopy(base_graph);graph["graph_variant"]="online_revision"
    graph["summary_nodes"]+=list(views.values());graph["edges"]+=legacy._view_edges(views)
    graph["stats"]={**graph.get("stats",{}),"event_view_nodes":len(views)}
    graph["revision_memory"]=state
    audit=run.old.graph_structure_audit(graph,sample)
    if graph["fact_nodes"]!=base_graph["fact_nodes"] or audit["status"]!="pass":raise StopRun("Revision structural audit failed")
    counts=Counter(c["status"] for c in state["claims"].values())
    operations=Counter(step["operation"]["action"] for step in state["ledger"])
    result={**identity,"graph":graph,"audit":audit,"step_digests":steps,"status_counts":dict(counts),"operation_counts":dict(operations)}
    seal(run.ROOT/"graphs/revision"/(sample+".json"),result)
    return {"sample":sample,"status_counts":dict(counts),"operation_counts":dict(operations),"views":len(views)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--register-only",action="store_true");args=parser.parse_args()
    loaded,parent,_=run.prepare();registration=register()
    if args.register_only:return
    from dotenv import load_dotenv
    load_dotenv(run.old.ROOT/".env",override=False)
    with run.old.Lease(run.ROOT/"state_build"):
        runtime=Runtime(run.ROOT/"state_build/calls",allow_model_calls=True,concurrency=10,timeout=180)
        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                completed=[future.result() for future in as_completed([pool.submit(build_one,c["sample_id"],parent,registration,runtime) for c in loaded])]
            seal(run.ROOT/"state_build/complete.json",{"registration_sha256":sha(registration),"conversations":sorted(completed,key=lambda x:x["sample"])})
        except BaseException as exc:
            runtime.stopped.set()
            seal(run.ROOT/"state_build"/("stop_"+str(time.time_ns())+".json"),{"utc":now(),"type":type(exc).__name__,"message":str(exc)[:500]})
            raise


if __name__=="__main__":main()

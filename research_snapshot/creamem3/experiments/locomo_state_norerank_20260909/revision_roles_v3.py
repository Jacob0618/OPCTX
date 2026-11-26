"""Role-separated evidence contract; no relaxation of the original support guard."""
import argparse
import copy
import importlib.util
import types
from creamem3.experiments.locomo_state_norerank_20260909 import run,revision
from creamem3.experiments.locomo_state_norerank_20260909.revision_compact_schema import compact_schema
from creamem3.experiments.locomo_state_norerank_20260909.runtime import Runtime,read,seal,sha,file_sha

BUILD_ROOT=run.ROOT/"revision_evidence_roles_v3"
SCHEMA=compact_schema(revision.SCHEMA)
CLAIM_SCHEMA=SCHEMA["properties"]["operations"]["items"]["properties"]["new_claims"]["items"]
CLAIM_SCHEMA["properties"].pop("support_fact_ids")
for field in ["current_fact_ids","earlier_fact_ids"]:
    CLAIM_SCHEMA["properties"][field]={"type":"ARRAY","minItems":1,"items":{"type":"STRING"}}
CLAIM_SCHEMA["required"]=[f for f in CLAIM_SCHEMA["required"] if f!="support_fact_ids"]+["current_fact_ids","earlier_fact_ids"]

PROMPT="""You are the cross-session UPDATE component, NOT the fact-extraction Writer.
All current facts are already stored. Never output a claim merely repeating one
current fact or one earlier fact. Return operations=[] if there is no genuinely
new relationship to prior evidence or no justified correction of existing memory.

Each new claim must express ONE precise relationship that actually needs BOTH:
current_fact_ids: one or more IDs from allowed_ids.current_fact_ids;
earlier_fact_ids: one or more IDs from prior_facts, NOT current IDs.
Only these cited texts may establish the relation. Topic similarity, proximity,
or mentioning the same person does not by itself establish causation or identity.
The compiler will reject any missing, misplaced or unknown evidence ID.

Generic example (NOT part of this input): Earlier E1 says Dana owned a red bicycle.
Current C1 says Dana reports that bicycle was stolen on June4. A grounded relation
is 'The red bicycle Dana previously owned was reported stolen on June4', citing
current_fact_ids=[C1],earlier_fact_ids=[E1]. Merely copying 'a bicycle was stolen'
with C1 alone is NOT an Online update. If current C1 is about Dana's bicycle and
earlier E1 is about an unrelated recipe, return no operation for that pair.

Output a JSON object with operations (maximum12). Each operation uses:
action add/replace/retract/dispute; target_view_id (exposed event ID or empty for
a new event); old_claim_ids; revision_kind none/supersede/correct; effective_time;
reason_fact_ids; a nonempty reason; new_claims (maximum12 per operation).
Even ADD requires reason_fact_ids containing a CURRENT ID and a concise reason
why the new cross-session connection is supported. For ADD use old_claim_ids=[],
revision_kind=none. Continue only a truly matching event, not a broad life profile.

Each claim uses kind relation/state, text, current_fact_ids,earlier_fact_ids,
state_key,state_value,valid_from,valid_to,depends_on_current_claim_ids.
For a relation, state_key and state_value are empty. Preserve exact names,
roles, quantities, chronology, negation, and planned versus completed actions.

Use STATE only for a genuinely exclusive current attribute, with a stable
subject|property key and explicit value. Hobbies/friends/pets/lists may coexist:
they are not exclusive slots. A later mention may recall an earlier event.
Do not impose last-mention-wins or invent precise dates.
REPLACE correct withdraws wrong derived claims and adds evidence-supported claims.
REPLACE supersede changes a state: cite old claims and create same-key states
with DIFFERENT values and valid_from=effective_time. Dates must be explicit
YYYY-MM-DD; when effective time is unknown use DISPUTE rather than force a date.
Old states remain available historically. RETRACT uses correct, no new claims.
DISPUTE uses none and marks conflicting claims unresolved. Revisions require
exposed old_claim_ids and CURRENT reason evidence. All unmentioned claims persist.

depends_on_current_claim_ids lists ONLY exposed claims whose CONTINUING validity
is necessary for the new claim, not mere historical citations. The system marks
dependent claims unresolved if their current premise is withdrawn. Original
facts and their dates are never deleted. Do not anticipate future sessions.

Check before answering: every new claim has evidence in BOTH named lists, and
its text expresses the actual connection between them, rather than another copy
of a single-session fact. No justified connection => empty operations.

ACTUAL INPUT:
"""


def normalize_operation(operation,exposed):
    result=copy.deepcopy(operation)
    current=set(exposed["current_fact_ids"]);earlier=set(exposed["fact_ids"])-current
    if not isinstance(result,dict) or not isinstance(result.get("new_claims"),list):raise ValueError("Invalid role operation")
    for claim in result["new_claims"]:
        if not isinstance(claim,dict) or "support_fact_ids" in claim:raise ValueError("Expected split evidence roles")
        cids=revision.ids(claim.pop("current_fact_ids",None));pids=revision.ids(claim.pop("earlier_fact_ids",None))
        if not cids or not pids or not set(cids)<=current or not set(pids)<=earlier:
            raise ValueError("Missing or incorrectly assigned current/earlier evidence")
        claim["support_fact_ids"]=cids+pids
    return result


def apply_response(state,response,exposed,facts,sid):
    if not isinstance(response,dict) or set(response)!={"operations"} or not isinstance(response["operations"],list) or len(response["operations"])>12:
        raise ValueError("Invalid role response envelope")
    accepted=[];rejected=[]
    for i,operation in enumerate(response["operations"]):
        try:
            normalized=normalize_operation(operation,exposed)
            state,ledger=revision.apply_one(state,normalized,exposed,facts,sid)
            accepted.append({"index":i,**ledger})
        except ValueError as exc:rejected.append({"index":i,"reason":str(exc),"operation":operation})
    return state,accepted,rejected


def register(protocol):
    record={"variant":"revision_v3","source_sha256":file_sha(__file__),"base_compiler_sha256":file_sha(revision.__file__),
        "prompt":PROMPT,"provider_schema":SCHEMA,"protocol_sha256":sha(protocol),
        "changes":"Explicit current/earlier evidence fields and clearer delta-only task;local causal/state/reason guards unchanged",
        "generic_examples":"Bicycle ownership/theft and unrelated recipe;no benchmark questions or gold used",
        "prior_fact_budget":80,"max_operations":12,"max_new_claims_per_operation":12,
        "no_model_rerank":True,"not_a_score_based_revision":True}
    seal(BUILD_ROOT/"registration.json",record);return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--probe",action="store_true");args=parser.parse_args()
    loaded,parent,protocol=run.prepare();record=register(protocol)
    if args.probe:
        if (BUILD_ROOT/"schema_probe_complete.json").exists():return
        from dotenv import load_dotenv
        load_dotenv(run.old.ROOT/".env",override=False)
        with run.old.Lease(BUILD_ROOT/"probe"):
            runtime=Runtime(BUILD_ROOT/"probe/calls",allow_model_calls=True,concurrency=1)
            value=runtime.structured_online("schema_probe","Return an empty operations list. No memory input is provided.",SCHEMA)
            if value!={"operations":[]}:raise RuntimeError("Unexpected schema probe response")
            seal(BUILD_ROOT/"schema_probe_complete.json",{"response":value,"schema_sha256":sha(SCHEMA)})
        return
    if read(BUILD_ROOT/"schema_probe_complete.json")["schema_sha256"]!=sha(SCHEMA):raise RuntimeError("Role schema not validated")
    spec=importlib.util.spec_from_file_location("isolated_role_revision_v3",revision.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    adapter=types.SimpleNamespace(**{k:getattr(run,k) for k in dir(run) if not k.startswith("__")})
    adapter.ROOT=BUILD_ROOT;module.run=adapter;module.PROMPT=PROMPT;module.SCHEMA=SCHEMA
    module.apply_response=apply_response;module.register=lambda:record
    module.main()
    for conv in loaded:
        sample=conv["sample_id"];source=BUILD_ROOT/"graphs/revision"/(sample+".json")
        graph_record=read(source)
        if graph_record["registration_sha256"]!=sha(record):raise RuntimeError("Revision v3 registration drift")
        seal(run.ROOT/"graphs/revision_v3"/(sample+".json"),{**graph_record,"source_path":str(source),"source_sha256":sha(graph_record)})
    seal(run.ROOT/"revision_v3_ready.json",{"registration_sha256":sha(record),"conversations":len(loaded)})


if __name__=="__main__":main()

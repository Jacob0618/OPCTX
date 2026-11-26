"""Resume successful contracts in a new root; repair only invalid rerank shape."""
import argparse
import copy
import json
from pathlib import Path
import time
from creamem3.experiments.locomo_joint_revision_20260909 import run as joint
from creamem3.experiments.locomo_joint_revision_20260909.followup_audit import ROOT, check_bound
from creamem3.experiments.locomo_online_only_20260909 import run as old
from creamem3.experiments.locomo_online_only_20260909 import audit as binding
from creamem3.experiments.locomo_online_only_20260909.runtime import read, seal, sha, file_sha, StopRun, response_text, now
from creamem3.experiments.locomo_online_only_20260909.result_analysis import pair_summary

RUN = ROOT / "rerank_contract_repair"
ARMS = joint.ARMS


def output_schema(n):
    return {"type":"OBJECT", "properties":{"ranked_fact_ids":{"type":"ARRAY", "minItems":n, "maxItems":n,
            "items":{"type":"STRING"}}}, "required":["ranked_fact_ids"]}


class Runtime(joint.Runtime):
    def __init__(self, *args, parents=(joint.RUN,), **kwargs):
        super().__init__(*args, **kwargs)
        self.parents = tuple(Path(p) for p in parents)

    def post(self, stage, scope, model, method, body):
        expected = {"qa":"gemini-2.5-flash", "rerank":"gemini-2.5-flash", "judge":"gemini-3-flash-preview"}
        if expected.get(stage) != model:
            self._halt("Unexpected generation stage/model")
        identity = {"stage":stage,"scope":scope,"model":model,"method":method,"body":body}
        key = sha(identity)
        own = self.root/"provider"/stage/key
        # Copy sealed completed receipts, not regenerated outputs, into this isolated cache.
        # The import manifest distinguishes reused responses from new physical calls.
        with self.mutex:
            if not (own/"request.json").exists():
                for parent in self.parents:
                    source = parent/"calls/provider"/stage/key
                    if not (source/"final.json").exists():
                        continue
                    request = read(source/"request.json")
                    final = read(source/"final.json")
                    raw_name = "attempt_%02d.response.json" % final["attempt"]
                    raw = read(source/raw_name)
                    if request!=identity or final["identity"]!=identity or final["request_sha256"]!=key or final["response_sha256"]!=sha(raw):
                        self._halt("Completed cache import drift")
                    seal(self.root/"imports"/stage/(key+".json"), {"source":str(source), "identity_sha256":key,
                        "source_final_sha256":sha(final), "source_response_sha256":sha(raw), "new_physical_call":False})
                    seal(own/"request.json", request)
                    seal(own/raw_name, raw)
                    seal(own/"final.json", final)
                    break
        return super().post(stage, scope, model, method, body)

    def chat_fn(self, scope, stage=None):
        original = super().chat_fn(scope, stage)
        def call(prompt, **kwargs):
            schema = kwargs.get("response_schema")
            if schema is None:
                return original(prompt, **kwargs)
            if kwargs.get("tag") != "creamem3_v2_rerank":
                self._halt("Structured contract is only for rerank IDs")
            body = {"contents":[{"role":"user","parts":[{"text":str(prompt)}]}], "generationConfig":{
                "temperature":0.0,"maxOutputTokens":4096,"thinkingConfig":{"thinkingBudget":0},
                "responseMimeType":"application/json","responseSchema":schema}}
            physical_scope = "shared:"+sha(body)
            self.log({"kind":"logical_request","scope":scope,"physical_scope":physical_scope,
                "stage":"rerank","model":"gemini-2.5-flash","body_sha256":sha(body),"structured_contract":True})
            response = self.post("rerank",physical_scope,"gemini-2.5-flash","generateContent",body)
            candidates = response.get("candidates",[])
            if len(candidates)!=1 or candidates[0].get("finishReason")!="STOP":
                self._halt("Structured rerank response did not finish")
            return response_text(response)
        return call


class RepairedIndex(joint.OriginalIndex):
    def _rerank(self, question, candidates, rerank_chat, *, max_attempts=2):
        selected, trace = super()._rerank(question, candidates, rerank_chat, max_attempts=max_attempts)
        if not trace["fallback"]:
            return selected, trace
        n = min(self.rerank_top_k,len(candidates))
        prompt = old.graph_api.FACT_RERANK_PROMPT.format(top_k=n,question=question,
            candidates_json=json.dumps([{"fact_id":x["fact"]["id"],"text":x["fact"]["text"]} for x in candidates],
                                       ensure_ascii=False,separators=(",",":")))
        raw = rerank_chat(prompt, tag="creamem3_v2_rerank",response_schema=output_schema(n))
        value = old.graph_api._parse_json_object(raw)
        ids = value.get("ranked_fact_ids") if isinstance(value,dict) else None
        by_id = {x["fact"]["id"]:x for x in candidates}
        if (not isinstance(value,dict) or set(value)!={"ranked_fact_ids"} or not isinstance(ids,list)
                or len(ids)!=n or len(set(ids))!=n or not set(ids)<=set(by_id)):
            raise StopRun("Structured rerank repair failed; do not fill with cosine")
        return [by_id[f] for f in ids], {"attempts":max_attempts+1,"fallback":False,"error":None,
            "format_repair":True,"previous_validation_error":trace["error"]}


def prepare(root=RUN, mode="repaired"):
    loaded, parent = old.prepare()
    previous = read(joint.RUN/"protocol.json")
    if previous["dev_ids"]!=parent["dev_ids"]:
        raise StopRun("Population drift")
    registration = {"mode":mode, "source_sha256":file_sha(__file__),"parent_protocol_sha256":sha(previous),
        "joint_source_sha256":file_sha(joint.__file__),"reader_source_sha256":file_sha(old.__file__),
        "frozen_graph_sha256":file_sha(old.FROZEN),"rerank_prompt_sha256":sha(old.graph_api.FACT_RERANK_PROMPT),
        "qa_prompt_sha256":sha(old.QA_PROMPT),"judge_prompt_sha256":sha(old.JUDGE_PROMPT),
        "arms":ARMS,"candidate_cap":120,"final_facts":20,"format_repair":
        "Same original two attempts, then one constrained twenty-ID request on invalid output only; successful old calls unchanged",
        "cache_policy":"Only exact completed request bodies; explicit source receipts; no in-flight adoption",
        "input_candidate_channels_unchanged":True,"qa_and_judge_unchanged":True,
        "new_writer_or_online_calls":False,"full_corpus_requires_separate_gate":True}
    seal(root/"registration.json",registration)
    protocol = {**parent,"followup_registration_sha256":sha(registration)}
    seal(root/"protocol.json",protocol)
    source_variant = read(joint.RUN/"variants/revision_v3.json")
    variant = {**source_variant,"followup_registration_sha256":sha(registration)}
    seal(root/"variants/revision_v3.json",variant)
    for conv in loaded:
        source = read(joint.RUN/"graphs/revision_v3"/(conv["sample_id"]+".json"))
        if source["identity"]["variant_sha256"]!=sha(source_variant) or source["audit"]["status"]!="pass":
            raise StopRun("Unverified source Online graph")
        seal(root/"graphs/revision_v3"/(conv["sample_id"]+".json"), {**source,
            "identity":{**source["identity"],"variant_sha256":sha(variant)},"source_record_sha256":sha(source)})
    return loaded, protocol


def finish(root, loaded, protocol, index_class=RepairedIndex):
    checks = binding.audit(root)
    if not checks["pass"]:
        raise StopRun("Physical audit failed: "+json.dumps(checks))
    vector_record = read(old.VECTOR_FILE)
    vectors = dict(zip(vector_record["ids"],vector_record["vectors"]))
    replay_runtime = Runtime(root/"calls",allow_model_calls=False)
    row_maps = {}
    for arm in ARMS:
        row_maps[arm] = {}
        for conv in loaded:
            sample = conv["sample_id"]
            graph = read(root/"graphs/revision_v3"/(sample+".json"))["graph"] if arm=="revision_v3" else old.get_source_graph(sample,arm,protocol)
            index = index_class(graph)
            graph_sha = sha(graph)
            for q in conv["qa"]:
                qid = q["qa_id"]
                if qid not in protocol["dev_ids"]:continue
                row = read(root/"rows"/arm/(qid+".json"))
                retrieved = read(root/"retrieval"/arm/(qid+".json"))
                _, trace = index.search(q["question"],vectors[qid],replay_runtime.chat_fn(arm+"/"+qid))
                if trace!=retrieved["trace"] or row["identity"]["qa_sha256"]!=sha(q) or row["identity"]["graph_sha256"]!=graph_sha:
                    raise StopRun("Read-side contract replay mismatch: "+arm+"/"+qid)
                check_bound(root,q,row)
                row_maps[arm][qid] = row
    pairs = {arm+" vs "+control:pair_summary([(row_maps[arm][qid],row_maps[control][qid]) for qid in protocol["dev_ids"]])
             for arm in ARMS[1:] for control in ARMS[:2] if arm!=control}
    reports = {arm:old.summarize(root,arm,protocol) for arm in ARMS}
    imports = len(list((root/"calls/imports").glob("*/*.json")))
    result = {"reports":reports,"pairs":pairs,"audit":checks,"retrieval_replays":300,"imported_exact_requests":imports,
              "full_promoted":False,"one_hundred_development_questions_not_holdout":True}
    seal(root/"complete.json",result)
    old.emit("FOLLOWUP COMPLETE "+json.dumps({a:r["correct"] for a,r in reports.items()}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=["run","prepare","audit"])
    args = parser.parse_args()
    with old.Lease(RUN):
        loaded,protocol = prepare()
        if args.command=="prepare":return
        if args.command=="audit":finish(RUN,loaded,protocol);return
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/".env",override=False)
        old.graph_api.GraphMemoryIndex = RepairedIndex
        runtime = Runtime(RUN/"calls",allow_model_calls=True,concurrency=24,timeout=180)
        try:
            old.evaluate(RUN,ARMS,loaded,protocol,runtime,qa_workers=24)
            finish(RUN,loaded,protocol)
        except BaseException as exc:
            runtime.stopped.set()
            seal(RUN/("stop_"+str(time.time_ns())+".json"),{"utc":now(),"type":type(exc).__name__,"message":str(exc)[:500]})
            raise


if __name__=="__main__":main()

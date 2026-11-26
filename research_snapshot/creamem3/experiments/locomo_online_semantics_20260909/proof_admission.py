"""Transactional admission of valid nested proofs in a new isolated run.

Malformed API/top-level payloads remain fatal. An atomic semantic proposal with
an invalid local reference is rejected, never broadened to the whole event.
The old stopped strict run remains intact. Exact completed physical responses
may be reused, with explicit lineage, under this newly registered compiler.
"""
import argparse
from pathlib import Path
from creamem3.experiments.locomo_online_semantics_20260909 import proof_factorization as original

base,old,read,seal,sha,file_sha,StopRun=original.base,original.old,original.read,original.seal,original.sha,original.file_sha,original.StopRun
repair=original.repair
PARENT=original.RUN
RUN=repair.priority_reader.ROOT/'nested_proof_factorization_guarded'
ARM,PROMPT,canonical_proofs=original.ARM,original.PROMPT,original.canonical_proofs
_compile=original.compile_annotations
_runtime=repair.RepairRuntime


def compile_annotations(view,facts,value):
    if not isinstance(value,dict) or set(value)!={'claims'} or not isinstance(value['claims'],list):
        raise ValueError('Malformed top-level proof transaction')
    claims=[];skipped=[];rejected=[];seen=set()
    for i,item in enumerate(value['claims']):
        try:part,proof=_compile(view,facts,{'claims':[item]})
        except (ValueError,TypeError,KeyError) as exc:
            rejected.append({'proposal_index':i,'proposal_sha256':sha(item),'reason':str(exc)});continue
        skipped.extend(proof['skipped'])
        for c in part:
            if c['claim_id'] not in seen:claims.append(c);seen.add(c['claim_id'])
    return claims,{'proposed':len(value['claims']),'retained':len(claims),'skipped':skipped,
                   'rejected_proposals':rejected,'only_parent_cited_facts':True,
                   'invalid_proposal_does_not_mutate_old_memory':True,
                   'semantic_sufficiency_not_mechanically_proven':True}


class ReuseRuntime(_runtime):
    def post(self,stage,scope,model,method,body):
        identity={'stage':stage,'scope':scope,'model':model,'method':method,'body':body};key=sha(identity)
        own=self.root/'provider'/stage/key
        source=PARENT/'variant_calls'/ARM/'provider'/stage/key
        with self.mutex:
            if not (own/'request.json').exists() and (source/'final.json').exists():
                request=read(source/'request.json');final=read(source/'final.json');name='attempt_%02d.response.json'%final['attempt'];raw=read(source/name)
                if request!=identity or final['identity']!=identity or final['request_sha256']!=key or final['response_sha256']!=sha(raw):self._halt('Completed proof response import mismatch')
                seal(self.root/'imports'/stage/(key+'.json'),{'source':str(source),'source_final_sha256':sha(final),'source_response_sha256':sha(raw),
                     'prior_generated_proposal_may_have_failed_old_stricter_compiler':True,'new_physical_call':False})
                seal(own/'request.json',request);seal(own/name,raw);seal(own/'final.json',final)
        return super().post(stage,scope,model,method,body)


def prepare():
    loaded,protocol=old.prepare(RUN)
    parent=read(PARENT/'variants'/(ARM+'.json'))
    if parent['source_sha256']!=file_sha(original.__file__):raise StopRun('Frozen factorizer changed')
    manifest={**parent,'protocol_sha256':sha(protocol),'admission_source_sha256':file_sha(__file__),
              'parent_strict_run':str(PARENT),'parent_registration_sha256':sha(parent),
              'change':'Individually reject invalid atomic proof proposals;never broaden a claim source allowlist;retain original graph',
              'old_stopped_run_preserved':True,'whole_payload_and_transport_failures_still_fatal':True,
              'completed_exact_response_reuse_with_new_compiler_explicit':True,'no_question_or_score_in_admission':True}
    seal(RUN/'variants'/(ARM+'.json'),manifest)
    return loaded,protocol,manifest


def build():
    original.RUN=RUN;original.prepare=prepare;original.compile_annotations=compile_annotations
    original.repair.RepairRuntime=ReuseRuntime
    original.build()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','build']);a=p.parse_args()
    prepare() if a.command=='prepare' else build()

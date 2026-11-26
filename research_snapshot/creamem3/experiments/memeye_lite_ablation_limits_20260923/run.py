"""One explicit cap amendment; original method functions otherwise reused."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import psutil
from ..memeye_lite_ablation_schema_20260923 import run as driver
from ..memeye_lite_ablation_schema_ordered_20260923 import run as ordered

base,b,rt=driver.base,driver.b,driver.rt
HERE=Path(__file__).resolve().parent
REC=base.ROOT/'limits20_qa4096_amendment_r3'
NEW_MODULE=__package__+'.run'
BaseRuntime=rt.Runtime
OriginalBody=base.v8.body
engine=base.v8.engine_v8
OriginalApply=engine.apply_frontier
SCHEMA=deepcopy(ordered.SCHEMA)
SCHEMA['properties']['operations'].pop('maxItems')
STRICT=deepcopy(SCHEMA)
STRICT['properties']['operations']['maxItems']=20


def hashes():
    return ordered.hashes() | {str(p):b.file_sha(p) for p in sorted([*HERE.glob('*.py'),HERE/'protocol.md'])}


def register():
    prior=ordered.validate()
    protected={}
    for pattern in ('*/smoke/tasks/*/predictions/*/*.json','*/smoke/tasks/*/scores/*/*/*.json',
                    '*/full/tasks/*/predictions/*/*.json','*/full/tasks/*/scores/*/*/*.json'):
        for p in base.ROOT.glob(pattern):protected[str(p)]=b.file_sha(p)
    return b.save(REC/'registration.json',dict(schema='memeye.v8_20.qa4096.v3',
        parent_sha256=prior['seal_sha256'],code_hashes=hashes(),response_schema=SCHEMA,
        local_contract=STRICT,models=rt.MODELS,judges=rt.JUDGES,protected=protected,
        authorized='2026-09-23: QA足够空间，v8不超过20操作',qa_cap=4096,v8_operation_cap=20,
        v8_prompt_before_sha256=b.sha(base.v8.PROMPT),
        v8_prompt_after_sha256=b.sha(base.v8.PROMPT.replace('at most 12 operations','at most 20 operations')),
        max_task_workers=4,max_qa_workers=4,successful_outputs_preserved=True,
        semantic_algorithm_unchanged_except_explicit_operation_cap=True,
        service_schema_count_bound_removed_local_count_guard_20=True))


def validate():
    prior=ordered.validate();reg=b.read(REC/'registration.json')
    base.require(reg['parent_sha256']==prior['seal_sha256'],'Parent amendment drift')
    base.require(reg['code_hashes']==hashes() and reg['response_schema']==SCHEMA,'Limits amendment drift')
    for p,h in reg['protected'].items():base.require(b.file_sha(p)==h,'Protected answer changed '+p)
    return reg


def body(value,repair=None):
    result=OriginalBody(value,repair)
    text=result['contents'][0]['parts'][0]['text']
    base.require(text.startswith(base.v8.PROMPT),'Unexpected v8 prompt builder')
    result['contents'][0]['parts'][0]['text']=text.replace('at most 12 operations','at most 20 operations',1)
    return result


def apply_frontier(memory,response,facts,allowed,exposed,frontier):
    """Same v8 transaction implementation, sole change is max20 vs max12."""
    if not isinstance(response,dict) or set(response)!={'operations'} or not isinstance(response['operations'],list):
        raise ValueError('Expected exactly {operations: [...]}')
    if len(response['operations'])>20 or frontier<=memory['frontier']:
        raise ValueError('Operation cap/increasing frontier violated (max20)')
    if not set(allowed)<=facts.keys():raise ValueError('Unknown exposed evidence')
    out=deepcopy(memory);receipts=[]
    for proposed in response['operations']:
        op=engine.normalize(proposed,out,facts,allowed,exposed)
        if op['action']=='refine':
            out,receipt=engine.refine(out,op,facts,allowed,exposed,frontier)
        else:
            op,expanded=engine.prior.close_date_quote(op,facts,allowed)
            op,inherited=engine.prior.prior.close_previous_support(out,op,facts,allowed,exposed)
            out,receipt=engine.prior.prior.engine.apply_one(out,op,facts,allowed,exposed,frontier)
            receipt.update(inherited_previous_support_fact_ids=inherited,expanded_same_source_date_quotes=expanded)
        receipt['source_id_proposal_sha256']=engine.old.digest(proposed)
        receipt['quotes_materialized_from_visible_sources']=True
        receipts.append(receipt)
    out['frontier']=frontier
    return out,receipts


class Runtime(BaseRuntime):
    def generate(self,stage,prompt,**kw):
        if stage=='answer':
            # Protect a completed original response even if prediction save was interrupted.
            wire,ref=rt.request(self.model,stage,prompt,kw.get('system'),500,kw.get('json_mode',False))
            manifest=dict(model=self.model,stage=stage,prompt=prompt,system=kw.get('system'),image=ref,
                generation_config=wire['generationConfig'],wire_sha256=b.sha(wire),
                metadata=kw.get('cache_metadata') or {},caption_context_count=kw.get('caption_context_count'))
            key=b.sha(manifest);p=self.root/key[:2]/key/'response.json'
            if p.exists():
                old=dict(kw,max_output_tokens=500)
                return super().generate(stage,prompt,**old)
            kw['max_output_tokens']=4096
            kw['cache_metadata']=dict(kw.get('cache_metadata') or {},
                limits_amendment_sha256=b.read(REC/'registration.json')['seal_sha256'])
        return super().generate(stage,prompt,**kw)


def active_workers():
    modules={driver.OLD_MODULE,'creamem3.experiments.memeye_lite_ablation_schema_20260923.run',
        'creamem3.experiments.memeye_lite_ablation_schema_ordered_20260923.run',NEW_MODULE}
    out=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            cmd=p.info['cmdline'] or []
            if '-m' not in cmd:continue
            i=cmd.index('-m')
            if len(cmd)>i+2 and cmd[i+1] in modules and cmd[i+2] in ('live','probe'):out.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return out


def build_v8(root,graph,gen,embed):
    reg=b.read(REC/'registration.json')
    sources=[root,root/'schema_v8',root/'schema_ordered_v8']
    for old in sources:
        if (old/'v8/complete.json').exists():
            b.save(root/'v8_limits20_provenance.json',dict(mode='retain_successful_pre_amendment_update',
                complete_path=str(old/'v8/complete.json'),sha256=b.file_sha(old/'v8/complete.json'),
                amendment_sha256=reg['seal_sha256']))
            return driver.ORIGINAL_BUILD_V8(old,graph,gen,embed)
    branches=[]
    for old in sources:
        prefix=[]
        for step in graph['online_fusion']['steps']:
            p=old/'v8/checkpoints'/f"{step['session_id']}.json"
            if not p.exists():break
            prefix.append(p)
        branches.append(prefix)
    selected=max(branches,key=len)
    dest=root/'limits20_v8';inherited={}
    for p in selected:
        data=b.read(p);data.pop('seal_sha256',None)
        b.save(dest/'v8/checkpoints'/p.name,data);inherited[str(p)]=b.file_sha(p)
    b.save(dest/'provenance.json',dict(amendment_sha256=reg['seal_sha256'],
        inherited_successful_prefix=inherited,source_selection='longest contiguous successful prefix, no scores',
        original_root=str(root),schema_sha256=b.sha(SCHEMA),local_operation_cap=20))
    scoped=driver.SchemaRuntime(dest/'calls',gen.model,gen.allow,reg['seal_sha256'])
    return driver.ORIGINAL_BUILD_V8(dest,graph,scoped,embed)


def configure():
    driver.REC=REC;driver.SCHEMA=SCHEMA;driver.NEW_MODULE=NEW_MODULE
    driver.register=register;driver.validate=validate
    driver.active_workers=active_workers;driver.build_v8=build_v8
    base.v8.body=body;engine.apply_frontier=apply_frontier;rt.Runtime=Runtime


if __name__=='__main__':configure();driver.main()

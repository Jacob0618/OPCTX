"""Format-only v8 continuation without mutating the registered experiment."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import jsonschema
import psutil
from dotenv import load_dotenv
from filelock import FileLock
from ..memeye_lite_ablation_20260923 import run as base

b, rt = base.b, base.rt
HERE = Path(__file__).resolve().parent
REC = base.ROOT/'v8_schema_amendment_r1'
OLD_MODULE = 'creamem3.experiments.memeye_lite_ablation_20260923.run'
NEW_MODULE = __package__+'.run'
SPAWN_LOCK = threading.Lock()
ORIGINAL_BUILD_V8 = base.build_v8


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


STR = dict(type='string')
ARR = dict(type='array', items=STR)
CLAIM = obj(dict(kind=dict(type='string', enum=['state','relation']), subject=STR,
    attribute=STR, scope=STR, value=STR, text=STR, support_fact_ids=ARR,
    valid_from=STR, current_dependencies=ARR))
OPERATION = obj(dict(action=dict(type='string', enum=['add','confirm','refine','supersede',
    'correct','dispute','retract','detach']), event_id=STR, old_claim_ids=ARR,
    new_claim=dict(anyOf=[CLAIM,dict(type='null')]), reason=STR, reason_fact_ids=ARR))
SCHEMA = obj(dict(operations=dict(type='array', items=OPERATION, maxItems=12)))


def hashes():
    return {str(p):b.file_sha(p) for p in sorted([*HERE.glob('*.py'),HERE/'protocol.md'])}


def register():
    parent = base.validate()
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    base.require(set(OPERATION['properties']) == base.v8.engine_v8.OP_KEYS, 'Operation fields differ')
    base.require(set(CLAIM['properties']) == base.v8.engine_v8.CLAIM_KEYS, 'Claim fields differ')
    protected = {}
    for pattern in ('*/smoke/tasks/*/v8/rejected/*.json', '*/smoke/tasks/*/predictions/*/*.json',
                    '*/smoke/tasks/*/scores/*/*/*.json', '*/full/tasks/*/predictions/*/*.json',
                    '*/full/tasks/*/scores/*/*/*.json'):
        for p in base.ROOT.glob(pattern): protected[str(p)] = b.file_sha(p)
    return b.save(REC/'registration.json',dict(schema='memeye.v8.format_only.amendment.v1',
        parent_sha256=parent['seal_sha256'], code_hashes=hashes(), response_schema=SCHEMA,
        authorized='2026-09-23 user: 允许，统一加输出格式约束', models=rt.MODELS,
        stages=['online_revision','online_revision_repair'], max_operations=12,
        unchanged=['models','prompt_body','output_caps','thinking','transaction_engine',
            'selector','Reader','QA','Judge','retrieval_budgets'], protected=protected,
        legacy_workers=[35184,7052,2224,31488], legacy_dispatcher_stopped=56836,
        preserve_successful_pre_amendment_outputs=True, max_total_workers=4,
        grandfathered_inflight_workers_explicit=True, possible_duplicate_billing=True))


def validate():
    reg = b.read(REC/'registration.json')
    base.require(reg['parent_sha256']==base.validate()['seal_sha256'],'Base contract drift')
    base.require(reg['code_hashes']==hashes() and reg['response_schema']==SCHEMA,'Amendment drift')
    for p,h in reg['protected'].items():
        base.require(b.file_sha(p)==h,'Protected output changed '+p)
    return reg


def wire_for(model,stage,prompt,system,cap):
    base.require(stage in ('online_revision','online_revision_repair'),'Wrong schema stage')
    base.require(model in rt.MODELS.values() and cap==8192,'v8 model/cap drift')
    wire,ref=rt.request(model,stage,prompt,system,cap,True)
    wire['generationConfig']['responseJsonSchema']=deepcopy(SCHEMA)
    return wire,ref


class SchemaRuntime(rt.Runtime):
    def __init__(self,root,model,allow,amendment):
        super().__init__(root,model,allow)
        self.amendment=amendment

    def generate(self,stage,prompt,*,system=None,json_mode=False,max_output_tokens=500,
                 max_provider_attempts=3,cache_metadata=None,image_path=None,caption_context_count=None):
        if stage not in ('online_revision','online_revision_repair'):
            raise RuntimeError('Schema runtime restricted to v8 only')
        base.require(json_mode and image_path is None,'Unexpected v8 request')
        wire,ref=wire_for(self.model,stage,prompt,system,max_output_tokens)
        manifest=dict(model=self.model,stage=stage,prompt=prompt,system=system,image=ref,
            generation_config=wire['generationConfig'],wire_sha256=b.sha(wire),
            metadata=cache_metadata or {},caption_context_count=caption_context_count,
            amendment_sha256=self.amendment)
        def send():
            if not self.allow: raise RuntimeError('Model calls disabled')
            return rt.prior.n.VerifiedProvider().generate_content(model=self.model,request=wire)
        def check(raw):
            cs=raw.get('candidates') or []
            base.require(len(cs)==1 and cs[0].get('finishReason')=='STOP','Non-STOP v8 schema generation')
            text=rt.prior.n.g._normalize_text(raw).strip()
            base.require(bool(text),'Empty v8 schema generation')
            served=raw.get('modelVersion','')
            base.require(not served or served.startswith(self.model),'Served model mismatch')
            jsonschema.validate(json.loads(text),SCHEMA)
        result,hit=rt.prior.durable(self.root,manifest,send,check,min(3,max_provider_attempts))
        raw=result['response']
        return rt.prior.Reply(rt.prior.n.g._normalize_text(raw).strip(),result['request_sha256'],
            result['response_sha256'],self.model,raw.get('modelVersion',''),
            asdict(rt.prior.n.g._normalize_usage(raw)),hit)


def build_v8(root,graph,gen,embed):
    amendment=b.read(REC/'registration.json')
    if (root/'v8/complete.json').exists():
        # Original builder replays every transaction and returns cached embeddings.
        b.save(root/'v8_schema_provenance.json',dict(mode='retained_complete_pre_amendment',
            original_complete_sha256=b.file_sha(root/'v8/complete.json'),
            amendment_sha256=amendment['seal_sha256']))
        return ORIGINAL_BUILD_V8(root,graph,gen,embed)
    dest=root/'schema_v8'
    imported={}
    for step in graph['online_fusion']['steps']:
        p=root/'v8/checkpoints'/f"{step['session_id']}.json"
        if not p.exists(): break
        data=b.read(p); data.pop('seal_sha256',None)
        b.save(dest/'v8/checkpoints'/p.name,data)
        imported[str(p)]=b.file_sha(p)
    b.save(dest/'provenance.json',dict(amendment_sha256=amendment['seal_sha256'],
        inherited_successful_prefix=imported,original_root=str(root),new_requests_schema_sha256=b.sha(SCHEMA)))
    scoped=SchemaRuntime(dest/'calls',gen.model,gen.allow,amendment['seal_sha256'])
    return ORIGINAL_BUILD_V8(dest,graph,scoped,embed)


def live(bb,task,smoke,allow):
    validate()
    root=base.task_root(bb,task,smoke)
    if (root/'done.json').exists():
        b.save(REC/'retained'/bb/('smoke' if smoke else 'full')/(task+'.json'),
            dict(done_path=str(root/'done.json'),sha256=b.file_sha(root/'done.json')))
        return
    base.build_v8=build_v8
    base.live(bb,task,smoke,allow)


def probe(bb,allow):
    """Exercise approved schema once without re-answering a successful smoke."""
    amendment=validate();task='Cartoon_Entertainment_Companion'
    root=base.task_root(bb,task,True);dest=REC/'format_probe'/bb
    with FileLock(str(root/'task.lock'),timeout=1):
        if (dest/'passed.json').exists(): return
        prior=root/'v8/checkpoints/1.json'
        checkpoint=b.read(prior)
        prompt=base.v8.body(checkpoint['input'])['contents'][0]['parts'][0]['text']
        gen=SchemaRuntime(dest/'calls',rt.MODELS[bb],allow,amendment['seal_sha256'])
        out=gen.generate('online_revision',prompt,json_mode=True,max_output_tokens=8192,
            cache_metadata=dict(task=task,frontier=1,purpose='schema_compatibility_no_qa'))
        proposal=json.loads(out.text)
        # Semantic transaction validation needs original source facts, not gold.
        reg=base.validate()
        _,_,_,graph=base.prepare_memory(root,bb,task,reg,True,False)
        facts={f['id']:f for f in graph['fact_nodes']}
        exposed=checkpoint['exposure']
        state,transactions=base.v8.engine_v8.apply_frontier(base.v8.engine_v8.empty(task),proposal,
            facts,exposed['allowed_fact_ids'],exposed['exposed_claim_ids'],1)
        b.save(dest/'passed.json',dict(model=gen.model,operations=len(proposal['operations']),
            generation=asdict(out),amendment_sha256=amendment['seal_sha256'],
            prompt_sha256=b.sha(prompt),state_sha256=b.sha(state),transactions=transactions,
            qa_calls=0,judge_calls=0))
        print('SCHEMA PROBE PASSED',bb,len(proposal['operations']),flush=True)


def active_workers():
    out=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            cmd=p.info['cmdline'] or []
            if '-m' not in cmd: continue
            i=cmd.index('-m')
            if len(cmd)<=i+2 or cmd[i+1] not in (OLD_MODULE,NEW_MODULE): continue
            if cmd[i+2] not in ('live','probe'): continue
            out.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied): continue
    return out


def subprocess_job(bb,task=None,smoke=False,probe_only=False):
    root=(REC/'format_probe'/bb if probe_only else base.task_root(bb,task,smoke))
    done=root/('passed.json' if probe_only else 'done.json')
    waited=0
    while True:
        with SPAWN_LOCK:
            workers=active_workers()
            same=[w for w in workers if '--backbone' in w['cmdline'] and
                w['cmdline'][w['cmdline'].index('--backbone')+1]==bb and
                ((probe_only and 'probe' in w['cmdline']) or (not probe_only and
                  '--task' in w['cmdline'] and w['cmdline'][w['cmdline'].index('--task')+1]==task and
                  ('--smoke' in w['cmdline'])==smoke))]
            if not same and done.exists():
                b.save(REC/'retained'/bb/('probe' if probe_only else ('smoke' if smoke else 'full'))/
                    (('format_probe' if probe_only else task)+'.json'),
                    dict(done_path=str(done),sha256=b.file_sha(done)))
                return 0
            if not same and len(workers)<base.TASK_WORKERS:
                rt.prior.disk_guard(base.ROOT)
                cmd=[sys.executable,'-X','utf8','-B','-m',NEW_MODULE,
                    'probe' if probe_only else 'live','--backbone',bb,'--allow-model-calls']
                if not probe_only:
                    cmd+=['--task',task]
                    if smoke: cmd.append('--smoke')
                root.mkdir(parents=True,exist_ok=True)
                log=root/f'schema_run_{time.time_ns()}.log'
                temp=base.ROOT/'temp'/bb;temp.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
                with log.open('ab') as handle:
                    child=subprocess.Popen(cmd,cwd=b.REPO,stdout=handle,stderr=subprocess.STDOUT,
                        env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                b.save(REC/'launches'/f'{child.pid}_{time.time_ns()}.json',
                    dict(pid=child.pid,command=cmd,log=str(log),time=time.time()))
                print('LAUNCHED SCHEMA',bb,task,'probe' if probe_only else ('smoke' if smoke else 'full'),child.pid,flush=True)
                break
        if waited%12==0: print('WAIT SAFE SLOT',bb,task,'active',len(workers),flush=True)
        time.sleep(5);waited+=1
    return child.wait()


def dispatch():
    amendment=validate();reg=base.validate();failed=[]
    with FileLock(str(base.ROOT/'dispatcher.lock'),timeout=1):
        for bb in rt.MODELS:
            for task in ('Cartoon_Entertainment_Companion','Outdoor_Navigation_Route_Memory_Assistant'):
                if subprocess_job(bb,task,True):
                    failed.append(dict(backbone=bb,task=task,stage='smoke'));break
            else:
                # g25lite smokes necessarily use schema after the original failure.
                if bb=='g35lite' and subprocess_job(bb,probe_only=True):
                    failed.append(dict(backbone=bb,stage='schema_probe'));continue
                with ThreadPoolExecutor(max_workers=base.TASK_WORKERS) as pool:
                    jobs={pool.submit(subprocess_job,bb,t):t for t in reg['tasks']}
                    for f in as_completed(jobs):
                        if f.result(): failed.append(dict(backbone=bb,task=jobs[f],stage='full'))
                print('PANEL STATUS',json.dumps(base.summarize(bb)),flush=True)
                continue
            print('PANEL SMOKE BLOCKED',bb,flush=True)
        results=[base.summarize(bb) for bb in rt.MODELS]
        b.save(REC/'outcomes'/f'{time.time_ns()}.json',dict(failures=failed,results=results))
        base.require(not failed and all(v['n']==371 for p in results for j in p['judges'].values() for v in j.values()),'Panel incomplete')
        b.save(REC/'complete.json',dict(amendment_sha256=amendment['seal_sha256'],results=results))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('register','live','probe','run','validate'))
    p.add_argument('--backbone',choices=rt.MODELS);p.add_argument('--task');p.add_argument('--smoke',action='store_true')
    p.add_argument('--allow-model-calls',action='store_true');a=p.parse_args()
    load_dotenv(b.REPO/'.env',override=False)
    if (a.phase in ('live','probe','run'))!=a.allow_model_calls: p.error('Explicit live flag required')
    try:
        if a.phase=='register': print('AMENDMENT',register()['seal_sha256'])
        elif a.phase=='validate': print('VALID',validate()['seal_sha256'])
        elif a.phase=='live': live(a.backbone,a.task,a.smoke,a.allow_model_calls)
        elif a.phase=='probe': probe(a.backbone,a.allow_model_calls)
        else: dispatch()
    except Exception as exc:
        b.save(REC/'stops'/f'{time.time_ns()}_{os.getpid()}.json',dict(phase=a.phase,backbone=a.backbone,
            task=a.task,smoke=a.smoke,error_type=type(exc).__name__,message=str(exc)[:1500],time=time.time()))
        raise


if __name__=='__main__': main()

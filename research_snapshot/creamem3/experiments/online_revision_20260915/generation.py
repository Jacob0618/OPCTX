"""Causal source-only generation and independently replayable state checkpoints."""
import argparse
import json
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from dotenv import load_dotenv
from . import engine,transport
from .probe import ROOT
from creamem3.experiments.memeye_online_state_20260914 import build as oldbuild,intake,state
from creamem3.experiments.memeye_evidence_20260914.verify_evidence import generic

io=transport.io
RUN=ROOT/'generation_v2'
ADDENDUM='''
Additional local-revision action: detach. A group was wrongly merged into a broad
event: event_id is that exposed event; old_claim_ids are its exposed active or
unresolved assertions to move together to a NEW separate event; new_claim=null.
This changes membership only, not source facts or assertion text. Quote current
evidence justifying the split. Never detach solely because a view is long.
For supersede, the new value AND effective date must be quoted in current facts.
An unresolved same-slot assertion requires explicit resolution, not a new add.
Do not output schema notation. Output concrete JSON with all the required fields.
'''


def hashes():
    modules=(engine,transport,intake,state,oldbuild,io)
    return {str(Path(m.__file__).resolve()):io.file_sha(m.__file__) for m in modules} | {
        str(Path(__file__).resolve()):io.file_sha(__file__)}


def body(value, repair=None):
    text=intake.PROMPT.replace('INPUT:\n',ADDENDUM+'\nINPUT:\n')+io.canonical(value)
    if repair is not None:
        text+='\nA prior proposal was REJECTED without committing any change. Correct the source/format error.\n'+io.canonical(repair)
    result=io.request_body(text,8192,True)
    result['generationConfig']['thinkingConfig']={'thinkingLevel':'low'}
    return result


def sources():
    original=oldbuild.baseline.checked(oldbuild.baseline.ROOT/'registration.json')
    refs={k.split('::',1)[1]:r for k,r in original['source_graphs'].items() if k.startswith('online::')}
    graphs={}
    for task,r in refs.items():
        if io.file_sha(r['path'])!=r['sha256']:raise RuntimeError('Frozen Writer graph drift')
        graph=json.loads(Path(r['path']).read_text(encoding='utf-8'))
        if graph['task']!=task:raise RuntimeError('Task binding drift')
        graphs[task]=graph
    reg=dict(schema='online_revision_generation.v2',source_graphs=refs,code_hashes=hashes(),
        model=io.BACKBONE,thinking='low',max_output_tokens=8192,server_schema=False,
        local_strict_schema=True,semantic_repair_limit=2,event_top_k=3,prior_fact_cap=80,
        new_raw_writer_or_image_calls=0,qa_judge_rerank_calls=0,engine=engine.VERSION,
        tasks=sorted(graphs),frontiers=sum(len(g['online_fusion']['steps']) for g in graphs.values()),
        current_session_only_plus_causal_history=True,whole_frontier_transaction=True,
        plan='smoke first two frontiers of every task, then all frontiers; exact cache reuse')
    io.seal(RUN/'registration.json',reg)
    return reg,graphs


def parse_response(response):
    if [c.get('finishReason') for c in response.get('candidates',[])]!=['STOP']:
        raise ValueError('Model output unfinished')
    try:return json.loads(io.response_text(response))
    except json.JSONDecodeError:raise ValueError('Model output is not complete JSON') from None


def run(allow=False,smoke=True,workers=4):
    reg,graphs=sources()
    if not allow:
        print('GENERATION PREFLIGHT',len(graphs),'tasks',reg['frontiers'],'frontiers; no model calls',flush=True);return
    load_dotenv(Path(__file__).resolve().parents[3]/'.env',override=False)
    runtime=transport.Runtime(RUN/'calls',allow_model_calls=True,concurrency=workers,timeout=180)
    def one(task,graph):
        facts={f['id']:f for f in graph['fact_nodes']}; memory=engine.empty(task)
        steps=graph['online_fusion']['steps'][:2] if smoke else graph['online_fusion']['steps']
        for step in steps:
            sid=step['session_id']
            value,exposure=intake.make_input(graph,memory,sid,step['allowed_ids']['prior_candidate_fact_ids'])
            ident=dict(registration_sha256=io.sha(reg),task=task,frontier=sid,
                       input_sha256=io.sha(value),before_state_sha256=io.sha(memory))
            path=RUN/'checkpoints'/task/f'{sid}.json'
            if path.exists():
                saved=io.read(path)
                if saved['identity']!=ident:raise RuntimeError('Checkpoint binding drift')
                memory=saved['state'];continue
            attempts=[];repair=None
            for number in range(3):
                if runtime.stopped.is_set():raise io.StopRun('Generation dispatch stopped')
                request=body(value,repair);stage='online_revision' if number==0 else 'online_revision_repair'
                scope=f'{task}/{sid}/{number}'
                response=runtime.post(stage,scope,io.BACKBONE,'generateContent',request)
                provider_identity=dict(stage=stage,scope=scope,model=io.BACKBONE,method='generateContent',body=request)
                attempt=dict(identity=provider_identity,provider_dir=str(RUN/'calls/provider'/stage/io.sha(provider_identity)),response_sha256=io.sha(response))
                try:
                    packet=parse_response(response)
                    candidate,receipts=engine.apply_frontier(memory,packet,facts,exposure['allowed_fact_ids'],exposure['exposed_claim_ids'],sid)
                except ValueError as exc:
                    attempt['validation_error']=str(exc);attempts.append(attempt)
                    io.seal(RUN/'rejected'/task/f'{sid}_{number}.json',dict(identity=ident,attempt=attempt,
                        unchanged_before_state_sha256=io.sha(memory)))
                    if number==2:raise RuntimeError('Unrecovered local validation '+task+'/'+str(sid)+': '+str(exc))
                    repair=dict(error=str(exc),rejected_response=io.response_text(response));continue
                attempts.append(attempt)
                io.seal(path,dict(identity=ident,input=value,exposure=exposure,attempts=attempts,
                    transactions=receipts,state=candidate,state_sha256=io.sha(candidate)))
                memory=candidate
                print('ONLINE REVISION',task,sid,'claims',len(memory['claims']),'actions',
                      [r['action'] for r in receipts],'repairs',number,flush=True)
                break
        return dict(task=task,frontiers=len(steps),claims=len(memory['claims']),
                    actions=dict(Counter(x['action'] for x in memory['ledger'])))
    results=[]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(one,t,g) for t,g in sorted(graphs.items())]
            for future in as_completed(futures):
                try:results.append(future.result())
                except BaseException:runtime.stopped.set();raise
        if hashes()!=reg['code_hashes']:raise RuntimeError('Generation code drift')
        io.seal(RUN/('smoke_complete.json' if smoke else 'generation_complete.json'),dict(
            complete=True,smoke_only=smoke,stats=sorted(results,key=lambda r:r['task']),
            registration_sha256=io.sha(reg),no_qa_score=True,requires_independent_replay=True))
        verify(smoke)
    except BaseException as exc:
        runtime.stopped.set()
        io.seal(RUN/f'failure_{time.time_ns()}.json',dict(type=type(exc).__name__,message=str(exc)[:1000],
            raw_writer_unchanged=True,smoke_only=smoke))
        raise


def verify(smoke=True):
    reg,graphs=sources();links=set();usage=Counter();statuses=Counter();actions=Counter();repairs=0;count=0
    for task,g in sorted(graphs.items()):
        fs={f['id']:f for f in g['fact_nodes']};memory=engine.empty(task)
        steps=g['online_fusion']['steps'][:2] if smoke else g['online_fusion']['steps']
        for step in steps:
            sid=step['session_id'];saved=io.read(RUN/'checkpoints'/task/f'{sid}.json')
            value,exposure=intake.make_input(g,memory,sid,step['allowed_ids']['prior_candidate_fact_ids'])
            expected=dict(registration_sha256=io.sha(reg),task=task,frontier=sid,input_sha256=io.sha(value),before_state_sha256=io.sha(memory))
            if saved['identity']!=expected or value!=saved['input'] or exposure!=saved['exposure']:raise RuntimeError('Causal input reconstruction mismatch')
            repair=None;out=None;receipts=None
            for number,attempt in enumerate(saved['attempts']):
                stage='online_revision' if number==0 else 'online_revision_repair'
                identity=dict(stage=stage,scope=f'{task}/{sid}/{number}',model=io.BACKBONE,method='generateContent',body=body(value,repair))
                if identity!=attempt['identity']:raise RuntimeError('Model request replay mismatch')
                response=generic(attempt['provider_dir'],identity)
                if io.sha(response)!=attempt['response_sha256']:raise RuntimeError('Model response mismatch')
                links.add(attempt['provider_dir']);usage.update({k:v for k,v in response.get('usageMetadata',{}).items() if isinstance(v,(int,float))})
                try:
                    packet=parse_response(response)
                    out,receipts=engine.apply_frontier(memory,packet,fs,exposure['allowed_fact_ids'],exposure['exposed_claim_ids'],sid)
                except ValueError as exc:
                    if attempt.get('validation_error')!=str(exc) or number==len(saved['attempts'])-1:raise RuntimeError('Rejected transaction replay mismatch')
                    repair=dict(error=str(exc),rejected_response=io.response_text(response));repairs+=1
                else:
                    if 'validation_error' in attempt or number!=len(saved['attempts'])-1:raise RuntimeError('Unexpected accepted transaction')
            if out!=saved['state'] or io.sha(out)!=saved['state_sha256'] or receipts!=saved['transactions']:raise RuntimeError('State transaction mismatch')
            memory=out;count+=1
        statuses.update(c['status'] for c in memory['claims'].values());actions.update(x['action'] for x in memory['ledger'])
    io.seal(RUN/('smoke_replay.json' if smoke else 'independent_replay.json'),dict(status='PASS',frontiers=count,
        linked_calls=len(links),validation_repairs=repairs,usage=dict(usage),claim_statuses=dict(statuses),actions=dict(actions),
        query_gold_future_free=True,raw_writer_unchanged=True,semantic_truth_or_qa_gain_not_established=True))
    print('REPLAY PASS',count,'frontiers',dict(actions),'validation repairs',repairs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','smoke','full','verify-smoke','verify-full']);p.add_argument('--allow-model-calls',action='store_true');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command.startswith('verify'):verify(a.command=='verify-smoke')
    else:run(a.allow_model_calls and a.command!='preflight',a.command!='full',a.workers)

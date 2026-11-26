"""Five fixed full-MemEye ablations, one generative backbone, two judges."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from dotenv import load_dotenv
from filelock import FileLock
import numpy as np
from . import runtime as rt
from creamem3 import build_executor_v19 as build
from creamem3 import graph_memory_v2 as memory
from ..iterative_memory_20260907 import build_multichannel_round_v2 as selection
from ..iterative_memory_20260907 import candidate_context_policy as display
from ..online_revision_20260915 import generation_v8 as v8
from ..online_revision_20260915 import evidence_units

b = rt.b
ROOT = Path(r'D:\me_ablate23\r1')
HERE = Path(__file__).resolve().parent
SOURCE = Path(r'D:\me_backbone21\r1')
ARMS = ('original_reader', 'no_reader', 'enhanced_reader', 'session_only', 'online_v8')
TASK_WORKERS, QA_WORKERS = 4, 4


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def code_hashes():
    modules = (rt, rt.prior, rt.prior.n, rt.prior.n.g, build, build.ingest, build.adapter,
               build.online, memory, selection, selection.selector, selection.selector.base,
               selection.selector.numerical, selection.adapter, selection.adapter.raw,
               selection.qp, display, v8, v8.base, v8.engine_v8, v8.engine_v4,
               v8.engine_v3, v8.engine, v8.intake, v8.state, evidence_units,
               evidence_units.read_api, evidence_units.read_api.reader_bridge,
               build.ingest.legacy, selection.adapter.direct, selection.adapter.direct.predict,
               selection.adapter.direct.retry5, selection.adapter.direct.g35_runtime)
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.update(HERE.glob('*.py')); paths.add(HERE/'protocol.md')
    # Include original Reader/QA authority sources without changing them.
    paths.update(Path(p) for p in b.authorities()[3].get('source_hashes', {}))
    return {str(p): b.file_sha(p) for p in sorted(paths)}


def register():
    old = b.read(SOURCE/'registration.json')
    qs = b.read(SOURCE/'queries_open.json')['rows']
    require(len(qs) == 371 and len({q['row_id'] for q in qs}) == 371, 'Need full371')
    require(all(set(q) == {'row_id','task','question'} for q in qs), 'Non-score-free queries')
    paths = [SOURCE/'queries_open.json', SOURCE/'scoring/open.json']
    paths += [Path(v['path']) for v in old['tasks'].values()]
    for t, info in old['tasks'].items():
        require(b.file_sha(info['path']) == info['sha256'], 'History drift '+t)
        require(not (set(b.read(info['path']))-{'character_profile','multi_session_dialogues','seal_sha256'}), 'Gold in history')
    routes = {q['row_id']: build.matrix.reader_route_for_task(q['task']) for q in qs}
    require(routes == old['opctx_routes'], 'Original task-level Reader routes differ')
    authority = b.authorities()[2]
    model_controls = {m: {s: rt.config(m,s,c,True) for s,c in
        [('caption',2400),('writer',16000),('view',8192),('focused',2600),('answer',500),('planner',2048)]}
        for m in rt.MODELS.values()}
    reg = dict(schema='memeye.single_backbone.five_ablation.dual_judge.v1', models=rt.MODELS,
        judges=rt.JUDGES, arms=list(ARMS), n=371, questions=qs, tasks=old['tasks'], routes=routes,
        code_hashes=code_hashes(), source_hashes={str(p):b.file_sha(p) for p in paths},
        model_controls=model_controls, source_generation_reuse=False, output_caps_unchanged=True,
        prompt_hashes={k:b.sha(authority[k]) for k in ('open','focused','fused')} |
            {'caption':b.sha(build._load_g35_caption_system_prompt()),
             'writer_bundle':build.ingest.load_prompt_bundle()['prompt_bundle_sha256'],
             'v8':b.sha(v8.PROMPT), 'query_projection':b.sha(selection.qp.PROMPT_PREFIX)},
        max_facts=30, char_budget='own-panel raw Dense50 rendered length, shared across arms',
        embedding='gemini-embedding-001 native3072', task_workers=TASK_WORKERS, qa_workers=QA_WORKERS,
        scope='five paper component rows only; open questions; no rerank/baselines/MCQ',
        same_answer_dual_judge=True, judge_retry_on_valid_score=False)
    return b.save(ROOT/'registration.json', reg)


def validate():
    reg = b.read(ROOT/'registration.json')
    require(reg['code_hashes'] == code_hashes(), 'Registered code changed')
    for p, digest in reg['source_hashes'].items():
        require(b.file_sha(p) == digest, 'Source changed '+p)
    return reg


def task_root(bb, task, smoke=False):
    return ROOT/bb/('smoke' if smoke else 'full')/'tasks'/task


def prepare_memory(root, bb, task, reg, smoke, allow):
    model = rt.MODELS[bb]
    build.ingest.WORK_MODELS[bb] = model
    gen = rt.Runtime(root/'calls', model, allow)
    history = b.read(reg['tasks'][task]['path'])
    history = {k:history[k] for k in ('character_profile','multi_session_dialogues')}
    if smoke:
        history = copy.deepcopy(history)
        history['multi_session_dialogues'] = history['multi_session_dialogues'][:2]
    images = build._chronological_unique_images(task, history)
    row = dict(task=task,memory=history,memory_sha256=b.sha(history),images=images,
               sessions=len(history['multi_session_dialogues']))
    b.save(root/'input.json', dict(history=history,images=images,registration_sha256=reg['seal_sha256']))
    captions, previous = {}, []
    for i, image in enumerate(images, 1):
        p = root/'captions'/f'{i:04d}.json'
        prompt = build.matrix.build_v7_reconstruction_user_text(task,image,sequence_index=i,
            total_images=len(images),previous=previous[-2:])
        if p.exists():
            saved = b.read(p)
            require(saved['model']==model and saved['prompt_sha256']==b.sha(prompt), 'Caption resume drift')
        else:
            out = gen.generate('caption',prompt,system=build._load_g35_caption_system_prompt(),
                image_path=build._resolve_image(b.IMAGE_ROOT,image),max_output_tokens=2400,
                caption_context_count=min(i-1,2),cache_metadata=dict(task=task,image=image,sequence_index=i))
            saved = b.save(p,dict(image=image,caption=out.text,model=model,prompt_sha256=b.sha(prompt),generation=asdict(out)))
        captions[image] = saved['caption']; previous.append(dict(image=image,caption=saved['caption']))
        print('CAPTION',bb,task,i,len(images),flush=True)
    embed = build.OfficialNative3072EmbeddingBackend(root/'embedding',containment_root=root,
        approved_provider_attempt_hard_ceiling=6000,budget_binding_sha256=reg['seal_sha256'],
        allow_model_calls=allow,provider_factory=rt.prior.VerifiedEmbeddingProvider)
    session, session_ref = build.build_session_graph_for_task(row,captions,backbone=bb,
        generation=gen,embedding=embed,output_root=root)
    print('WRITER',bb,task,len(session['fact_nodes']),flush=True)
    online, online_ref = build.build_online_graph_for_task(row,session,backbone=bb,
        generation=gen,embedding=embed,output_root=root)
    require(online['fact_nodes']==session['fact_nodes'], 'Online changed source facts')
    b.save(root/'memory_complete.json',dict(model=model,images=len(images),sessions=row['sessions'],
        facts=len(session['fact_nodes']),session=session_ref,online=online_ref,registration_sha256=reg['seal_sha256']))
    return gen, embed, session, online


def select_context(graph, q, channels, budget):
    facts, groups = selection.adapter.adapt_facts(graph)
    result = selection.selector.select_multichannel(facts,q['question'],channels,max_units=30,
        char_budget=budget+1,group_costs=groups,time_limit=5.0)
    ids = result['selected_ids']
    text, render = selection.adapter.render_selection(graph,ids)
    require(0<len(ids)<=30 and len(ids)==len(set(ids)), 'Fact cap/uniqueness')
    require(len(text)<=budget and result['cost']-1==len(text), 'Actual context budget drift')
    require(not render['rendered_event_view_ids'] and not render['rendered_session_summary_ids'], 'Extra unbudgeted prose')
    return dict(query=q,context=text,context_sha256=b.sha(text),selected_fact_ids=ids,
                budget=budget,optimization=result,render=render)


def channels_for(index, facts, queries, vectors):
    out = []
    for q in queries:
        dense = list(map(float,index._scores(index.fact_matrix,vectors[q['query']])))
        lexical = dict(index.bm25.search(q['query'],len(facts)))
        out.append(dict(query_id=q['query_id'],query=q['query'],dense_scores=dense,
                        bm25_scores=[float(lexical.get(f['id'],0.)) for f in facts]))
    return out


def prepare_raw(root, graph, qs, gen, embed):
    facts, _ = selection.adapter.adapt_facts(graph)
    ids = [f['id'] for f in facts]
    index = memory.GraphMemoryIndex(graph,candidate_cap=len(facts),rerank_top_k=50)
    require(ids==[f['id'] for f in index.facts], 'Fact vector alignment')
    contexts = {arm:[] for arm in ARMS[:-1]}
    for q in qs:
        key = b.sha(q['row_id']); pp = root/'projections'/(key+'.json')
        if pp.exists():
            projection = b.read(pp)['projection']
        else:
            out = gen.generate('planner',selection.qp.build_prompt(q['question']),json_mode=True,
                max_output_tokens=2048,cache_metadata=dict(row_id=q['row_id'],purpose='enhanced_query_projection'))
            projection = selection.qp.validate_response(q['question'],out.text)
            b.save(pp,dict(projection=projection,generation=asdict(out)))
        desired = selection.query_texts(q['question'],projection,'q0_plus_anchored_attributes')
        vectors = dict(zip([x['query'] for x in desired], embed.embed([x['query'] for x in desired],
            tag='query',cache_metadata=dict(row_id=q['row_id'],purpose='query_channels')),strict=True))
        channels = channels_for(index,facts,desired,vectors)
        dense50 = selection.display_ids(ids,ids,channels[0]['dense_scores'])[:50]
        baseline, _ = selection.adapter.render_selection(graph,dense50)
        for arm in ('original_reader','enhanced_reader'):
            p = root/'contexts'/arm/(key+'.json')
            if p.exists():
                item = b.read(p)
                require(item['query']==q and item['budget']==len(baseline), 'Context resume mismatch')
            else:
                item = b.save(p,select_context(graph,q,channels if arm=='enhanced_reader' else channels[:1],len(baseline)))
            contexts[arm].append(item)
        raw = contexts['original_reader'][-1]
        for arm in ('no_reader','session_only'):
            item = dict(raw); item.pop('seal_sha256',None)
            contexts[arm].append(b.save(root/'contexts'/arm/(key+'.json'),item))
        b.save(root/'query_vectors'/(key+'.json'),dict(row_id=q['row_id'],vector=vectors[q['question']]))
        print('SELECT RAW',gen.model,q['row_id'],flush=True)
    return contexts


def build_v8(root, graph, gen, embed):
    facts = {f['id']:f for f in graph['fact_nodes']}
    state = v8.engine_v8.empty(graph['task'])
    for step in graph['online_fusion']['steps']:
        sid = step['session_id']
        value, exposure = v8.intake.make_input(graph,state,sid,step['allowed_ids']['prior_candidate_fact_ids'])
        ident = dict(task=graph['task'],frontier=sid,input_sha256=b.sha(value),before_sha256=b.sha(state))
        p = root/'v8/checkpoints'/f'{sid}.json'
        if p.exists():
            saved = b.read(p); require(saved['identity']==ident,'V8 checkpoint drift')
            replay, transactions = v8.engine_v8.apply_frontier(state,saved['proposal'],facts,
                exposure['allowed_fact_ids'],exposure['exposed_claim_ids'],sid)
            require(replay==saved['state'] and transactions==saved['transactions'],'V8 replay failed')
            state = replay; continue
        repair = None
        for attempt in range(3):
            body = v8.body(value,repair)
            prompt = body['contents'][0]['parts'][0]['text']
            out = gen.generate('online_revision' if attempt==0 else 'online_revision_repair',
                prompt,json_mode=True,max_output_tokens=8192,
                cache_metadata=dict(task=graph['task'],frontier=sid,semantic_attempt=attempt))
            try:
                proposal = json.loads(out.text)
                candidate, transactions = v8.engine_v8.apply_frontier(state,proposal,facts,
                    exposure['allowed_fact_ids'],exposure['exposed_claim_ids'],sid)
            except ValueError as exc:
                b.save(root/'v8/rejected'/f'{sid}_{attempt}.json',dict(identity=ident,error=str(exc),generation=asdict(out)))
                if attempt==2:
                    raise RuntimeError('V8 validation exhausted '+str(sid)+': '+str(exc)) from exc
                repair = dict(error=str(exc),rejected_response=out.text); continue
            b.save(p,dict(identity=ident,input=value,exposure=exposure,proposal=proposal,
                generation=asdict(out),state=candidate,state_sha256=b.sha(candidate),transactions=transactions))
            state = candidate
            print('V8',gen.model,graph['task'],sid,'claims',len(state['claims']),flush=True)
            break
    bundle = evidence_units.build(graph,'revised_online',state)
    added = []
    for u in bundle['units']:
        if u['kind']!='derived': continue
        c = state['claims'][u['id'][2:]]
        if c['status']!='active': continue
        require(not c['current_dependencies'],'New v8 dependency case requires audit; no implicit protocol change')
        support = bundle['support_requires'][u['id']]
        require(support and set(support)<=facts.keys(),'Unknown derived support')
        anchor = max((facts[f] for f in support),key=lambda f:(f['session_id'],f['id']))
        fact = {k:anchor[k] for k in ('session_id','source_session_id','timestamp','iso_date')}
        fact.update(id=u['id'],node_type='online_derived_fact',text=u['text'],
            source_dia_ids=u['sources'],support_fact_ids=support,claim_id=c['id'])
        added.append(fact)
    if added:
        vectors = embed.embed([f['text'] for f in added],tag='fact',cache_metadata=dict(task=graph['task'],purpose='v8_overlay'))
        for f,vec in zip(added,vectors,strict=True):
            require(len(vec)==3072 and np.isfinite(vec).all(),'Invalid overlay embedding')
            f['embedding']=vec
    b.save(root/'v8/complete.json',dict(state_sha256=b.sha(state),added=added,
        rejected=bundle['rejected_derived_candidates'],frontiers=len(graph['online_fusion']['steps'])))
    return dict(graph,fact_nodes=graph['fact_nodes']+added)


def prepare_v8(root, graph, controls):
    facts, _ = selection.adapter.adapt_facts(graph)
    index = memory.GraphMemoryIndex(dict(graph,summary_nodes=[],edges=[]),candidate_cap=len(facts),rerank_top_k=50)
    contexts = []
    for control in controls:
        q = control['query']; key = b.sha(q['row_id']); p=root/'contexts/online_v8'/(key+'.json')
        if p.exists():
            item=b.read(p)
        else:
            vec=b.read(root/'query_vectors'/(key+'.json'))['vector']
            channels=channels_for(index,facts,[dict(query_id='q0',query=q['question'])],{q['question']:vec})
            item=b.save(p,select_context(graph,q,channels,control['budget']))
        contexts.append(item)
    return contexts


def predict(root, arm, item, gen, reg):
    q=item['query']; rid=q['row_id']; key=b.sha(rid); p=root/'predictions'/arm/(key+'.json')
    if p.exists():
        saved=b.read(p)
        require(saved['model']==gen.model and saved['context_sha256']==item['context_sha256'],'Prediction drift')
        return saved
    authority=b.authorities()[2]; context=item['context']; stages=[]
    route=reg['routes'][rid] if arm in ('original_reader','enhanced_reader') else 'direct'
    meta=dict(arm=arm,row_id=rid,context_sha256=item['context_sha256'],registration_sha256=reg['seal_sha256'])
    if route=='two_stage':
        prompt=authority['focused'].replace('{question}',q['question']).replace('{context}',context)
        out=gen.generate('focused',prompt,json_mode=True,max_output_tokens=2600,cache_metadata=meta)
        parsed=selection.adapter.direct.predict._parse_json_object(out.text) or {'_failed':True}
        stages.append(dict(stage='focused',generation=asdict(out)))
        if arm=='enhanced_reader':
            context=display.rendered_context(authority['render'],parsed,context,'candidate_first','candidate_last')
        else:
            context=authority['render'](parsed,context,'candidate_first')
    if route=='structured_fused':
        prompt=authority['fused'].format(context=context,question=q['question'])
        out=gen.generate('structured_fused',prompt,json_mode=True,max_output_tokens=1200,cache_metadata=meta)
        parsed=selection.adapter.direct.predict._parse_json_object(out.text)
        answer=str(parsed.get('final_answer') or 'Not found in memory.').strip()
    else:
        prompt=authority['open'].format(context=context,question=q['question'])
        out=gen.generate('answer',prompt,max_output_tokens=500,cache_metadata=meta)
        answer=out.text.strip()
    require(bool(answer),'Empty answer')
    stages.append(dict(stage='structured_fused' if route=='structured_fused' else 'answer',generation=asdict(out)))
    require(all(s['generation']['model']==gen.model for s in stages),'Mixed prediction backbone')
    return b.save(p,dict(row_id=rid,task=q['task'],arm=arm,model=gen.model,prediction=answer,
        context_sha256=item['context_sha256'],qa_prompt_sha256=b.sha(prompt),route=route,stages=stages,
        registration_sha256=reg['seal_sha256']))


def evaluate(root, arm, item, gen, reg, allow):
    pred=predict(root,arm,item,gen,reg)
    q=item['query']; key=b.sha(q['row_id'])
    # Evaluator-only data loaded only after the answer is durably frozen.
    gold=b.read(SOURCE/'scoring/open.json')['rows'][q['row_id']]
    require(gold['question']==q['question'],'Gold question mismatch')
    for name, model in rt.JUDGES.items():
        p=root/'scores'/name/arm/(key+'.json')
        if p.exists():
            require(b.read(p)['prediction_sha256']==pred['seal_sha256'],'Judge prediction drift'); continue
        judge=rt.Runtime(root/'judge_calls'/name,model,allow)
        out,verdict=judge.judge(q['question'],gold['gold'],pred['prediction'],
            dict(arm=arm,row_id=q['row_id'],prediction_sha256=pred['seal_sha256']))
        b.save(p,dict(row_id=q['row_id'],task=q['task'],arm=arm,x=gold['x'],y=gold['y'],xy=gold['x']+'-'+gold['y'],
            raw_score=verdict.raw_score,mapped_score=verdict.mapped_score,reasoning=verdict.reasoning,
            judge=model,generation=asdict(out),prediction_sha256=pred['seal_sha256']))
    print('SCORED BOTH',gen.model,arm,q['row_id'],flush=True)


def evaluate_set(root, contexts, gen, reg, allow):
    failed=[]
    with ThreadPoolExecutor(max_workers=QA_WORKERS) as pool:
        futures={pool.submit(evaluate,root,arm,item,gen,reg,allow):(arm,item['query']['row_id'])
                 for arm,items in contexts.items() for item in items}
        for f in as_completed(futures):
            arm,rid=futures[f]
            try:f.result()
            except Exception as exc:
                receipt=dict(arm=arm,row_id=rid,error_type=type(exc).__name__,message=str(exc)[:1500],time=time.time())
                b.save(root/'errors'/f'{time.time_ns()}.json',receipt); failed.append(receipt)
                print('ROW FAILED',arm,rid,repr(exc),flush=True)
    return failed


def live(bb, task, smoke, allow):
    reg=validate();root=task_root(bb,task,smoke);root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'task.lock'),timeout=1):
        if (root/'done.json').exists():return
        rt.prior.disk_guard(ROOT)
        gen,embed,session,online=prepare_memory(root,bb,task,reg,smoke,allow)
        qs=[q for q in reg['questions'] if q['task']==task]
        if smoke:qs=qs[:1]
        contexts=prepare_raw(root,session,qs,gen,embed)
        require(all(a['context']==s['context'] for a,s in zip(contexts['no_reader'],contexts['session_only'],strict=True)),
                'Raw-only controls unexpectedly differ')
        failures=evaluate_set(root,contexts,gen,reg,allow)
        overlay=build_v8(root,online,gen,embed)
        failures+=evaluate_set(root,{'online_v8':prepare_v8(root,overlay,contexts['session_only'])},gen,reg,allow)
        require(not failures,'Unresolved rows; successful results preserved')
        b.save(root/'done.json',dict(backbone=bb,model=gen.model,task=task,smoke=smoke,n=len(qs),
            arms=list(ARMS),judges=rt.JUDGES,registration_sha256=reg['seal_sha256']))


def summarize(bb):
    reg=b.read(ROOT/'registration.json');root=ROOT/bb/'full';out={}
    for name in rt.JUDGES:
        out[name]={}
        for arm in ARMS:
            rows=[b.read(p) for p in root.glob('tasks/*/scores/'+name+'/'+arm+'/*.json')]
            result=dict(n=len(rows),expected=371)
            if len(rows)==371:
                require({r['row_id'] for r in rows}=={q['row_id'] for q in reg['questions']},'Score ID mismatch')
                result['overall']=100*statistics.mean(r['mapped_score'] for r in rows)
                for axis in ('x','y','xy'):
                    result['by_'+axis]={v:100*statistics.mean(r['mapped_score'] for r in rows if r[axis]==v)
                        for v in sorted({r[axis] for r in rows})}
            out[name][arm]=result
    return dict(backbone=bb,model=rt.MODELS[bb],judges=out)


def subprocess_job(bb,task,smoke):
    root=task_root(bb,task,smoke);root.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-X','utf8','-B','-m',__package__+'.run','live','--backbone',bb,'--task',task,'--allow-model-calls']
    if smoke:cmd.append('--smoke')
    log=root/f'run_{time.time_ns()}.log'
    temp=ROOT/'temp'/bb;temp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    with log.open('ab') as handle:
        p=subprocess.Popen(cmd,cwd=b.REPO,stdout=handle,stderr=subprocess.STDOUT,env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    b.save(ROOT/'launches'/f'{p.pid}_{time.time_ns()}.json',dict(pid=p.pid,command=cmd,log=str(log),time=time.time()))
    print('LAUNCHED',bb,task,'smoke' if smoke else 'full',p.pid,flush=True)
    return p.wait()


def dispatch():
    reg=validate();failed=[]
    with FileLock(str(ROOT/'dispatcher.lock'),timeout=1):
        for bb in rt.MODELS:
            # Smoke both Reader routes and multi-image input, score-independent.
            for task in ('Cartoon_Entertainment_Companion','Outdoor_Navigation_Route_Memory_Assistant'):
                if subprocess_job(bb,task,True):
                    failed.append(dict(backbone=bb,task=task,stage='smoke'))
                    break
            else:
                with ThreadPoolExecutor(max_workers=TASK_WORKERS) as pool:
                    jobs={pool.submit(subprocess_job,bb,t,False):t for t in reg['tasks']}
                    for f in as_completed(jobs):
                        if f.result():failed.append(dict(backbone=bb,task=jobs[f],stage='full'))
                print('PANEL STATUS',json.dumps(summarize(bb)),flush=True)
                continue
            print('PANEL SMOKE BLOCKED',bb,flush=True)
        results=[summarize(bb) for bb in rt.MODELS]
        b.save(ROOT/'dispatch_outcomes'/f'{time.time_ns()}.json',dict(failures=failed,results=results))
        require(not failed and all(v['n']==371 for p in results for j in p['judges'].values() for v in j.values()),'Panel incomplete')
        b.save(ROOT/'complete.json',dict(registration_sha256=reg['seal_sha256'],results=results))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('register','live','run','summary'))
    p.add_argument('--backbone',choices=rt.MODELS);p.add_argument('--task');p.add_argument('--smoke',action='store_true')
    p.add_argument('--allow-model-calls',action='store_true');a=p.parse_args()
    load_dotenv(b.REPO/'.env',override=False)
    if (a.phase in ('live','run'))!=a.allow_model_calls:p.error('Explicit live flag required')
    try:
        if a.phase=='register':print('REGISTERED',register()['seal_sha256'])
        elif a.phase=='live':live(a.backbone,a.task,a.smoke,a.allow_model_calls)
        elif a.phase=='run':dispatch()
        else:print(json.dumps([summarize(bb) for bb in rt.MODELS],ensure_ascii=False,indent=2))
    except Exception as exc:
        b.save(ROOT/'stops'/f'{time.time_ns()}_{os.getpid()}.json',dict(phase=a.phase,backbone=a.backbone,
            task=a.task,smoke=a.smoke,error_type=type(exc).__name__,message=str(exc)[:1500],time=time.time()))
        raise


if __name__=='__main__':main()

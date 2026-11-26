"""Full1540 Offline schedules and fixed-selection reader control."""
import argparse
import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from . import offline_engine_v2 as engine, runtime_v2 as rt, transport

p=transport.prior
read,seal,sha,file_sha,StopRun=p.read,p.seal,p.sha,p.file_sha,p.StopRun
BASE=Path(r'D:\creamem3_cache\locomo_mgtm480_top5_plus_all_full1540_20260910_r2')
ROOT=Path(r'D:\creamem3_cache\component_ablations_20260911_r3\locomo')
ARMS=('end_only','feedback6','session_render')
SOURCES=(p.original.RUN/'calls',p.full.SOURCE/'calls',BASE/'conditions/narrow/calls',
    Path(r'D:\creamem3_cache\component_ablations_20260911_r1\locomo\calls'),
    Path(r'D:\creamem3_cache\component_ablations_20260911_r1\locomo\smoke7\calls'),
    Path(r'D:\creamem3_cache\component_ablations_20260911_r2\locomo\calls'),
    Path(r'D:\creamem3_cache\component_ablations_20260911_r2\locomo\smoke7\calls'))


def source_data():
    parent,protocol,questions,samples,vectors,graphs,hashes,indexes,operations,_,frozen=p.source_check()
    registration=read(BASE/'registration.json'); completed=read(BASE/'complete.json')
    if completed['registration_sha256']!=sha(registration) or not completed['audit_pass']:
        raise StopRun('Narrow baseline is not fully audited')
    for path,digest in registration['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Frozen parent code drift '+path)
    baselines={}
    for qid,q in questions.items():
        path=BASE/'conditions/narrow/rows/opctx_original'/(qid+'.json')
        retpath=BASE/'conditions/narrow/retrieval/opctx_original'/(qid+'.json')
        row,ret=read(path),read(retpath)
        if row['context']!=ret['context'] or row['context_sha256']!=sha(ret['context']):
            raise StopRun('Baseline row/context drift')
        if row['selected_fact_ids']!=ret['trace']['rerank']['selected_fact_ids'] or len(row['selected_fact_ids'])!=5:
            raise StopRun('Baseline selection drift')
        p.base.check_bound(BASE/'conditions/narrow',q,row)
        baselines[qid]=row; frozen[str(path)]=file_sha(path); frozen[str(retpath)]=file_sha(retpath)
    if sum(r['correct'] for r in baselines.values())!=1217: raise StopRun('Original Online score drift')
    own=[Path(__file__),Path(engine.__file__),Path(rt.__file__),Path(transport.__file__),
         Path(engine.old.__file__),Path(rt.original.__file__),Path(__file__).with_name('locomo.py'),Path(__file__).with_name('protocol.md')]
    reg={'schema':'component.locomo.offline.v1','source_registration_sha256':sha(registration),
         'source_hashes':{**registration['source_hashes'],**{str(x):file_sha(x) for x in own}},
         'source_snapshot_sha256':sha(frozen),'all_ids':sorted(questions),'arms':ARMS,
         'config':p.CONFIGS['narrow'],'writer_immutable':True,'memory_model':'gemini-2.5-flash',
         'qa_model':'gemini-2.5-flash','judge_model':'gemini-3-flash-preview','judge_votes':1,
         'format_policy':'deterministic_EOF_closers_then_original_string_preserving_repair',
         'interval':6,'neighbors':2,'pack_char_cap':24000,'memory_caps':rt.CAPS,
         'synchronous_publication':True,'asynchronous_concurrency_claimed':False,
         'historical_online_correct':1217,'n':1540,'not_token_or_maintenance_compute_matched':True}
    seal(ROOT/'source_snapshot.json',frozen); seal(ROOT/'registration.json',reg)
    return dict(registration=reg,questions=questions,samples=samples,vectors=vectors,graphs=graphs,
                graph_hashes=hashes,indexes=indexes,operations=operations,baselines=baselines)


def check_frozen(data):
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Code changed after registration '+path)
    for path,digest in read(ROOT/'source_snapshot.json').items():
        if file_sha(path)!=digest: raise StopRun('Source artifact changed '+path)


def validate_graph(session,graph):
    legal=[d for f in session['fact_nodes'] for d in f['source_dia_ids']]
    audit=p.old.graph_api.audit_graph(graph,legal_dia_ids=legal,expected_sessions=len(session['session_packets']))
    if audit['status']!='pass': raise StopRun('Graph structural failure '+str(audit))
    if graph['fact_nodes']!=session['fact_nodes'] or graph['session_packets']!=session['session_packets']:
        raise StopRun('Writer identity drift')
    for view in graph['summary_nodes']:
        vector=view['embedding']
        if len(vector)!=768 or not all(math.isfinite(v) for v in vector) or not sum(v*v for v in vector)>0:
            raise StopRun('Invalid view embedding')
    return audit


def preflight(data):
    control=transport.ConnectionControl(live=False)
    runtime=rt.Runtime(ROOT/'preflight_calls',control=control,allow_model_calls=False,sources=SOURCES)
    reports={}
    for sample in sorted(set(data['samples'].values())):
        session=data['graphs'][(sample,'session_strong')]
        baseline=data['graphs'][(sample,'opctx_original')]
        graph=engine.build_graph(session,sample,ROOT/'preflight',runtime,sha(data['registration']),mode='online')
        for key in ('fact_nodes','session_packets','summary_nodes','edges'):
            if graph[key]!=baseline[key]: raise StopRun('Original Online equivalence failed '+sample+'/'+key)
        for new,old in zip(graph['offline_feedback']['steps'],baseline['online_fusion']['steps'],strict=True):
            if new['prompt_input_sha256']!=old['prompt_input_sha256'] or new['online_accepted']!=old['accepted']:
                raise StopRun('Baseline transition mismatch')
        reports[sample]=validate_graph(session,graph)
        p.old.emit('COMPONENT BASELINE REPLAY '+sample)
    result={'registration_sha256':sha(data['registration']),'samples':len(reports),'new_model_calls':0,
            'all_original_frontiers_reconstructed':True,'pass':True}
    seal(ROOT/'preflight.json',result)
    return result


def build_one(data,sample,mode,runtime,*,smoke=False):
    session=data['graphs'][(sample,'session_strong')]
    root=ROOT/'smoke7' if smoke else ROOT
    if smoke:
        session=copy.deepcopy(session)
        seen=set(engine.online._base_parts(session)['session_ids'][:7])
        session['summary_nodes']=[s for s in session['summary_nodes'] if s['session_ids'][0] in seen]
        session['fact_nodes']=[f for f in session['fact_nodes'] if f['session_id'] in seen]
        ids={f['id'] for f in session['fact_nodes']}
        session['edges']=[e for e in session['edges'] if e['target_id'] in ids]
        session['session_packets']=[s for s in session['session_packets'] if s['session_id'] in seen]
        session['content_fingerprint']=sha({'prefix':sorted(seen),'source':session['content_fingerprint']})
    graph=engine.build_graph(session,sample,root,runtime,sha(data['registration']),mode=mode)
    audit=validate_graph(session,graph)
    seal(root/'graphs'/mode/(sample+'.json'),{'registration_sha256':sha(data['registration']),'graph':graph,'audit':audit})
    p.old.emit('COMPONENT GRAPH '+mode+'/'+sample+' views='+str(graph['stats']['event_view_nodes']))
    return graph


def retrieve(data,arm,qid,graph,index,runtime):
    q=data['questions'][qid]
    if arm=='session_render':
        baseline=data['baselines'][qid]; ids=baseline['selected_fact_ids']
        byid={f['id']:f for f in graph['fact_nodes']}
        context,assembly=index._assemble([{'fact':byid[i]} for i in ids],[])
        if assembly['rendered_event_view_ids'] or assembly['rendered_fact_ids']!=ids:
            raise StopRun('Session renderer changed selection/exposed view')
        return {'context':context,'ids':ids,'trace':assembly,'candidate_count':baseline['candidate_count'],
                'fixed_selection_source_sha256':sha(baseline),'operation':data['operations'][qid]}
    original,trace=index.search(q['question'],data['vectors'][qid],runtime.chat_fn(arm+'/'+qid))
    candidates,expected=index._candidate_pool(q['question'],data['vectors'][qid])
    if expected!={k:trace[k] for k in expected}: raise StopRun('Candidate replay drift')
    p.validate_selection(trace['rerank']['selected_fact_ids'],candidates)
    context,render=p.render(index,candidates,trace,graph,q,'opctx_original',data['operations'][qid])
    return {'context':context,'ids':trace['rerank']['selected_fact_ids'],'trace':trace,'materializer':render,
            'candidate_count':trace['candidates_after_cap'],'operation':data['operations'][qid]}


def evaluate_one(data,arm,qid,graph,index,runtime,*,smoke=False):
    root=ROOT/'smoke7' if smoke else ROOT
    path=root/'rows'/arm/(qid+'.json')
    graph_digest=data.get('component_hashes',{}).get((data['samples'][qid],arm)) if not smoke else None
    identity={'registration_sha256':sha(data['registration']),'qa_sha256':sha(data['questions'][qid]),
              'graph_sha256':graph_digest or sha(graph),'vector_sha256':sha(data['vectors'][qid]),'arm':arm,'smoke':smoke}
    if path.exists():
        row=read(path)
        if row['identity']!=identity: raise StopRun('Row resume binding drift')
        return row
    runtime.local.requests=[]; started=time.perf_counter()
    runtime.log({'kind':'row_start','arm':arm,'qid':qid})
    result=retrieve(data,arm,qid,graph,index,runtime)
    seal(root/'retrieval'/arm/(qid+'.json'),{'identity':identity,**result})
    q=data['questions'][qid]
    answer=runtime.chat(qid,'qa',p.old.QA_PROMPT.format(question=q['question'],retrieved_texts=result['context']),max_tokens=8192)
    correct,judge=p.old.judge(runtime,qid,q,answer)
    requests=p.unique_requests(runtime.local.requests)
    qa=[r for r in requests if r['stage']=='qa']
    if len(qa)!=1: raise StopRun('Expected exactly one effective QA')
    row={'identity':identity,'qa_id':qid,'sample_id':data['samples'][qid],'arm':arm,'question':q['question'],
         'answer':q['answer'],'category_name':q['category_name'],'response':answer,'correct':correct,'judge_raw':judge,
         'context':result['context'],'context_sha256':sha(result['context']),'context_chars':len(result['context']),
         'selected_fact_ids':result['ids'],'candidate_count':result['candidate_count'],
         'qa_prompt_tokens':qa[0]['usage'].get('promptTokenCount'),'logical_request_receipts':requests,
         'elapsed_s':time.perf_counter()-started}
    seal(path,row); runtime.log({'kind':'row_complete','arm':arm,'qid':qid,'correct':correct})
    return row


def smoke(data,control):
    runtime=rt.Runtime(ROOT/'smoke7/calls',control=control,allow_model_calls=True,sources=SOURCES+(ROOT/'preflight_calls',))
    graph=build_one(data,'conv-30','feedback6',runtime,smoke=True)
    steps=graph['offline_feedback']['steps']
    if len(steps)!=7 or not steps[5]['offline'] or steps[6]['checkpoint_due']:
        raise StopRun('Smoke did not traverse scheduled consolidation and next update')
    # First question in a fixed lexical ID ordering: pipeline check, not a score gate.
    qid=min(q for q in data['questions'] if data['samples'][q]=='conv-30')
    evaluate_one(data,'feedback6',qid,graph,p.source.enum_repair.Index(graph,**p.CONFIGS['narrow']),runtime,smoke=True)
    seal(ROOT/'smoke_complete.json',{'registration_sha256':sha(data['registration']),'pass':True,
          'qa_id':qid,'prefix_sessions':7,'accuracy_used_as_gate':False})


def summary(rows):
    vals=list(rows.values()); out={'n':len(vals),'correct':sum(r['correct'] for r in vals),'by_category':{}}
    out['accuracy']=100*out['correct']/len(vals)
    for cat in sorted({r['category_name'] for r in vals}):
        sub=[r for r in vals if r['category_name']==cat]; c=sum(r['correct'] for r in sub)
        out['by_category'][cat]={'n':len(sub),'correct':c,'accuracy':100*c/len(sub)}
    out['mean_context_chars']=sum(r['context_chars'] for r in vals)/len(vals)
    known=[r['qa_prompt_tokens'] for r in vals if r['qa_prompt_tokens'] is not None]
    out['mean_qa_prompt_tokens']=sum(known)/len(known) if known else None
    return out


def provider_audit(root):
    from collections import Counter
    counts=Counter(); stages={}
    for path in sorted((root/'provider').glob('*/*/request.json')):
        request,final,_,raw=p.full.recover.completed_receipt(path.parent)
        stage=request['stage']; key=sha(request)
        if transport.MODELS.get(stage)!=request['model']: raise StopRun('Model drift in completed receipt')
        group=stages.setdefault(stage,Counter()); group['logical_completed']+=1
        origin=root/'imports'/stage/(key+'.json'); imported={}
        if origin.exists():
            marker=read(origin); imported=marker['files']
            for name,digest in imported.items():
                if file_sha(path.parent/name)!=digest or file_sha(Path(marker['source'])/name)!=digest:
                    raise StopRun('Imported receipt or origin changed')
        for intent in path.parent.glob('*.intent.json'):
            prefix=intent.name.removesuffix('.intent.json')
            if read(intent)['request_sha256']!=key or sum((path.parent/(prefix+s)).exists() for s in ('.response.json','.error.json'))!=1:
                raise StopRun('Orphan/duplicate physical receipt')
            bucket='imported_physical_attempts' if intent.name in imported else 'new_physical_attempts'
            counts[bucket]+=1; group[bucket]+=1
        for error in path.parent.glob('*.error.json'):
            if not read(error)['retryable']: raise StopRun('Nonretryable error receipt')
            counts['recovered_errors']+=1
        for k,v in raw['response'].get('usageMetadata',{}).items():
            if isinstance(v,int): group[k]+=v
    if list(root.rglob('*.pending')): raise StopRun('Pending durable artifact')
    return {'counts':dict(counts),'stages':stages,'unfinished':0,'orphan_count':0,'pass':True,
            'failed_attempt_billing_unknown':True}


def finish(data,graphs):
    runtime=rt.Runtime(ROOT/'calls',control=transport.ConnectionControl(live=False),allow_model_calls=False)
    rows={}
    for arm in ARMS:
        rows[arm]={}
        if {f.stem for f in (ROOT/'rows'/arm).glob('*.json')}!=set(data['questions']): raise StopRun('Incomplete population')
        for qid,q in data['questions'].items():
            sample=data['samples'][qid]; graph=graphs[(sample,arm)]
            index=data['component_indexes'][(sample,arm)]
            row=read(ROOT/'rows'/arm/(qid+'.json')); saved=read(ROOT/'retrieval'/arm/(qid+'.json'))
            replay=retrieve(data,arm,qid,graph,index,runtime)
            if replay!={k:saved[k] for k in replay} or row['context']!=replay['context'] or row['selected_fact_ids']!=replay['ids']:
                raise StopRun('Final retrieval replay drift')
            p.base.check_bound(ROOT,q,row)
            if row['identity']['graph_sha256']!=data['component_hashes'][(sample,arm)] or row['identity']['registration_sha256']!=sha(data['registration']):
                raise StopRun('Final row identity drift')
            for entry in row['logical_request_receipts']:
                _,_,_,raw=p.full.recover.completed_receipt(ROOT/'calls/provider'/entry['stage']/entry['key'])
                if entry['usage']!=raw['response'].get('usageMetadata',{}): raise StopRun('Usage binding drift')
            rows[arm][qid]=row
        p.old.emit('COMPONENT FINAL AUDIT '+arm)
    check_frozen(data)
    paired={a:p.paired(rows[a],data['baselines'],sorted(data['questions'])) for a in ARMS}
    metrics={}
    for mode in ('end_only','feedback6'):
        steps=[s for sample in sorted(set(data['samples'].values())) for s in graphs[(sample,mode)]['offline_feedback']['steps']]
        metrics[mode]={'frontiers':len(steps),'offline_calls':sum(len(s['offline']) for s in steps),
            'accepted_corrections':sum(len(o['accepted']) for s in steps for o in s['offline']),
            'rejected_corrections':sum(len(o['rejected']) for s in steps for o in s['offline']),
            'frontiers_exposing_corrected_views':sum(bool(s['exposed_offline_view_ids']) for s in steps),
            'extensions_of_corrected_views':sum(len(s['extended_offline_view_ids']) for s in steps),
            'deferred_packages':sum(len(s['deferred_view_ids']) for s in steps)}
    result={'registration_sha256':sha(data['registration']),'complete':True,'audit_pass':True,
            'reports':{'original_online':summary(data['baselines']),**{a:summary(rows[a]) for a in ARMS}},
            'paired_vs_original_online':paired,'feedback_vs_end_only':p.paired(rows['feedback6'],rows['end_only'],sorted(data['questions'])),
            'maintenance':metrics,'physical_transport':provider_audit(ROOT/'calls'),
            'smoke_transport':provider_audit(ROOT/'smoke7/calls'),'not_token_matched':True,'old_scores_unchanged':True}
    seal(ROOT/'complete.json',result); p.old.emit('COMPONENT COMPLETE '+json.dumps(result['reports']))
    return result


def full(data,control):
    if read(ROOT/'smoke_complete.json')['registration_sha256']!=sha(data['registration']): raise StopRun('Smoke drift')
    runtime=rt.Runtime(ROOT/'calls',control=control,allow_model_calls=True,
        sources=SOURCES+(ROOT/'preflight_calls',ROOT/'smoke7/calls'))
    samples=sorted(set(data['samples'].values())); graphs={}
    with p.source.enum_repair.FailFastPool(runtime,max_workers=10) as pool:
        futures={pool.submit(build_one,data,sample,mode,runtime):(sample,mode) for sample in samples for mode in ('end_only','feedback6')}
        for future in as_completed(futures): graphs[futures[future]]=future.result()
    for sample in samples: graphs[(sample,'session_render')]=data['graphs'][(sample,'session_strong')]
    indexes={key:p.source.enum_repair.Index(graph,**p.CONFIGS['narrow']) for key,graph in graphs.items()}
    data['component_indexes']=indexes
    data['component_hashes']={key:sha(graph) for key,graph in graphs.items()}
    with p.source.enum_repair.FailFastPool(runtime,max_workers=16) as pool:
        jobs=[pool.submit(evaluate_one,data,arm,qid,graphs[(data['samples'][qid],arm)],indexes[(data['samples'][qid],arm)],runtime)
              for qid in sorted(data['questions']) for arm in ARMS]
        for n,future in enumerate(as_completed(jobs),1):
            future.result()
            if n%50==0: p.old.emit(f'COMPONENT QA {n}/{len(jobs)}')
    return finish(data,graphs)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','preflight','pipeline'])
    args=parser.parse_args()
    with p.old.Lease(ROOT):
        try:
            data=source_data()
            if args.command=='prepare': return
            preflight(data)
            if args.command=='preflight': return
            from dotenv import load_dotenv
            load_dotenv(p.old.ROOT/'.env',override=False)
            control=transport.ConnectionControl(live=True)
            smoke(data,control); full(data,control)
        except BaseException as exc:
            seal(ROOT/f'stop_{time.time_ns()}.json',{'utc':p.base.now(),'type':type(exc).__name__,'message':str(exc)[:1000]})
            raise


if __name__=='__main__': main()



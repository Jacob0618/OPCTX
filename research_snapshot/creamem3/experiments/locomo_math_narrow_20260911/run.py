"""Four fixed full1540 mathematical-selector arms, no model reranker."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time

from . import core
from creamem3.experiments.component_ablations_20260911 import transport, runtime_v2 as rt, locomo_v3 as component

p=transport.prior
read,seal,sha,file_sha,StopRun=p.read,p.seal,p.sha,p.file_sha,p.StopRun
ROOT=Path(r'D:\creamem3_cache\locomo_math_narrow_full1540_20260911_r1')
BASE=component.BASE
ARMS=('session_strong','opctx_original')
CONFIGS={'previous_2_5_5':dict(summary_top_k=2,dense_fact_top_k=5,bm25_fact_top_k=5,candidate_cap=120,rerank_top_k=5),
         'narrower_1_5_3':dict(summary_top_k=1,dense_fact_top_k=5,bm25_fact_top_k=3,candidate_cap=120,rerank_top_k=5)}
WORKERS=12
ALLOWED={'qa':'gemini-2.5-flash','judge':'gemini-3-flash-preview',
         'query_projection':'gemini-2.5-flash','embedding':'gemini-embedding-001'}
SOURCES=(BASE/'conditions/narrow/calls',component.ROOT/'calls')


class Runtime(rt.Runtime):
    def __init__(self,*args,**kwargs):
        # Extend only this new process's dispatch table, never edit frozen transport.
        transport.MODELS['query_projection']=ALLOWED['query_projection']
        super().__init__(*args,**kwargs)
        self.forbid_rerank=True

    def post(self,stage,scope,model,method,body):
        if ALLOWED.get(stage)!=model: self._halt('No maintenance/reranker calls permitted')
        response=super().post(stage,scope,model,method,body)
        if stage in {'qa','query_projection'} and rt.original.finish_reason(response)!='STOP':
            self._halt('Incomplete '+stage+' response; no silent truncation accepted')
        return response


def prepare():
    parent,protocol,questions,samples,vectors,graphs,hashes,_,operations,_,frozen=p.source_check()
    complete,reference=read(BASE/'complete.json'),read(BASE/'registration.json')
    if not complete['audit_pass'] or complete['registration_sha256']!=sha(reference): raise StopRun('Reference not fully audited')
    baselines={a:{} for a in ARMS}
    for a in ARMS:
        for qid,q in questions.items():
            path=BASE/'conditions/narrow/rows'/a/(qid+'.json')
            row=read(path); p.base.check_bound(BASE/'conditions/narrow',q,row)
            baselines[a][qid]=row; frozen[str(path)]=file_sha(path)
    if [sum(r['correct'] for r in baselines[a].values()) for a in ARMS]!=[1178,1217]: raise StopRun('Historical reference score drift')
    for vector in vectors.values(): core.vector_check(vector)
    own=[Path(__file__),Path(core.__file__),Path(__file__).with_name('protocol.md'),
         Path(core.solver.__file__),Path(core.solver.base.__file__),Path(core.solver.numerical.__file__),
         Path(core.solver.lexical.__file__),Path(core.qp.__file__),
         Path(__file__).parents[1]/'iterative_memory_20260907/build_multichannel_round_v2.py',
         Path(rt.__file__),Path(rt.original.__file__),Path(transport.__file__)]
    code={**parent['source_hashes'],**{str(x):file_sha(x) for x in own}}
    frozen[str(BASE/'registration.json')]=file_sha(BASE/'registration.json')
    frozen[str(BASE/'complete.json')]=file_sha(BASE/'complete.json')
    registration={'schema':'locomo.math.narrow.v1','n':1540,'total_rows':6160,'arms':list(ARMS),
        'configs':CONFIGS,'source_hashes':code,'source_snapshot_sha256':sha(frozen),'all_ids':sorted(questions),
        'models':ALLOWED,'projection_prompt_sha256':sha(core.qp.PROMPT_PREFIX),
        'qa_prompt_sha256':sha(p.old.QA_PROMPT),'judge_prompt_sha256':sha(p.old.JUDGE_PROMPT),
        'query_projections':'q0_plus_anchored_attributes','embedding_dimension':768,'query_embedding_task':'RETRIEVAL_QUERY',
        'max_final_facts':5,'candidate_expansion_from_projections':False,'candidate_cap':120,
        'budget':'same-question ordinary globalDense5 source/session characters; extra Online presentation excluded from cap',
        'solver_time_seconds_per_stage':5,'not_strict_token_matched':True,'query_planner_sees_candidates':False,
        'historical_exact_response_reuse_roots':[str(x) for x in SOURCES],
        'historical_correct':{'session_strong':1178,'opctx_original':1217},'workers':WORKERS,
        'fresh_reranker_calls_permitted':0,'writer_and_event_graphs_frozen':True,'judge_votes':1}
    seal(ROOT/'source_snapshot.json',frozen); seal(ROOT/'registration.json',registration)
    indexes={(c,s,a):p.source.enum_repair.Index(graphs[(s,a)],**cfg)
             for c,cfg in CONFIGS.items() for s in sorted(set(samples.values())) for a in ARMS}
    data=dict(registration=registration,questions=questions,samples=samples,vectors=vectors,graphs=graphs,
              graph_hashes=hashes,indexes=indexes,operations=operations,baselines=baselines)
    # Entire score-free manifest; the planner never receives category/answer.
    seal(ROOT/'questions_score_free.json',{'score_free':True,'rows':[{'qid':q,'question':questions[q]['question']} for q in sorted(questions)]})
    seal(ROOT/'preflight.json',{'pass':True,'all_ids':sorted(questions),'new_model_calls':0,
                               'source_registration_sha256':sha(parent),'registration_sha256':sha(registration)})
    return data


def identity(data,c,a,qid):
    return {'registration_sha256':sha(data['registration']),'qa_sha256':sha(data['questions'][qid]),
            'graph_sha256':data['graph_hashes'][(data['samples'][qid],a)],'condition':c,'arm':a,
            'vector_sha256':sha(data['vectors'][qid])}


def query_bank(question,q0,qid,runtime,registration):
    path=ROOT/'query_bank'/(qid+'.json')
    binding={'registration_sha256':sha(registration),'question_sha256':sha(question),'q0_vector_sha256':sha(q0)}
    if path.exists():
        bank=read(path)
        if bank['identity']!=binding or bank['question']!=question or bank['vectors'][question]!=q0: raise StopRun('Query bank resume drift')
        return bank
    runtime.local.requests=[]
    raw=runtime.chat('query-only:'+sha(question),'query_projection',core.qp.build_prompt(question),max_tokens=2048,json_mode=True)
    projection=core.qp.validate_response(question,raw)
    queries=core.query_texts(question,projection,'q0_plus_anchored_attributes')
    texts=[q['query'] for q in queries[1:]]
    values=runtime.embed(texts,task_type='RETRIEVAL_QUERY') if texts else []
    for vector in values: core.vector_check(vector)
    vectors={question:q0,**dict(zip(texts,values,strict=True))}
    bank={'identity':binding,'question':question,'projection':projection,'queries':queries,
          'vectors':vectors,'raw_projection':raw,'logical_request_receipts':p.unique_requests(runtime.local.requests)}
    seal(path,bank); return bank


def get_selection(data,c,a,qid,bank):
    path=ROOT/'selection'/c/a/(qid+'.json'); q=data['questions'][qid]; s=data['samples'][qid]
    binding={'row_identity':identity(data,c,a,qid),'bank_sha256':sha(bank)}
    if path.exists():
        saved=read(path)
        if saved['identity']!=binding: raise StopRun('Selection resume drift')
        return saved
    index=data['indexes'][(c,s,a)]; session_index=data['indexes'][(c,s,'session_strong')]
    saved={'identity':binding,**core.select(index,session_index,q['question'],bank)}
    seal(path,saved); return saved


def evaluate(data,c,a,qid,bank,runtime):
    root=ROOT; path=root/'rows'/c/a/(qid+'.json'); binding=identity(data,c,a,qid)
    if path.exists():
        row=read(path)
        if row['identity']!=binding: raise StopRun('Existing QA row drift')
        return row
    runtime.local.requests=[]; runtime.log({'kind':'row_start','condition':c,'arm':a,'qid':qid})
    started=time.perf_counter(); q=data['questions'][qid]; s=data['samples'][qid]
    saved=get_selection(data,c,a,qid,bank)
    index=data['indexes'][(c,s,a)]; graph=data['graphs'][(s,a)]
    context,trace,display=core.render(index,graph,q['question'],saved,a,data['operations'][qid],p.old.materialize_online_operation_context)
    seal(root/'retrieval'/c/a/(qid+'.json'),{'identity':binding,'selection_sha256':sha(saved),
         'context':context,'trace':trace,'materializer':display})
    answer=runtime.chat(qid,'qa',p.old.QA_PROMPT.format(question=q['question'],retrieved_texts=context),max_tokens=8192)
    correct,judge=p.old.judge(runtime,qid,q,answer)
    requests=p.unique_requests(runtime.local.requests); qa=[r for r in requests if r['stage']=='qa']
    if len(qa)!=1 or any(r['stage'] not in {'qa','judge'} for r in requests): raise StopRun('Read-side forbidden call')
    row={'identity':binding,'qa_id':qid,'sample_id':s,'condition':c,'arm':a,'question':q['question'],
         'answer':q['answer'],'category_name':q['category_name'],'response':answer,'correct':correct,'judge_raw':judge,
         'context':context,'context_sha256':sha(context),'context_chars':len(context),
         'selected_fact_ids':saved['selected_fact_ids'],'selected_count':len(saved['selected_fact_ids']),
         'candidate_count':len(saved['candidate_fact_ids']),'union_before_cap':trace['union_before_cap'],
         'plain_context_chars':saved['plain_chars'],'plain_budget':saved['plain_budget'],
         'qa_prompt_tokens':qa[0]['usage'].get('promptTokenCount'),'logical_request_receipts':requests,
         'selection_sha256':sha(saved),'bank_sha256':sha(bank),'elapsed_s':time.perf_counter()-started}
    seal(path,row); runtime.log({'kind':'row_complete','condition':c,'arm':a,'qid':qid,'correct':correct})
    p.old.emit(f'MATH ROW {c}/{a}/{qid} correct={correct} selected={row["selected_count"]}')
    return row


def question_job(data,qid,runtime):
    try:
        if runtime.stopped.is_set(): raise StopRun('Dispatch stopped')
        bank=query_bank(data['questions'][qid]['question'],data['vectors'][qid],qid,runtime,data['registration'])
        # Produce every arm's selection before exposing this question's QA outcomes.
        for c in CONFIGS:
            for a in ARMS: get_selection(data,c,a,qid,bank)
        for c in CONFIGS:
            for a in ARMS: evaluate(data,c,a,qid,bank,runtime)
    except BaseException as exc:
        runtime.record_failure(exc); raise


def verify_one(data,c,a,qid,bank):
    q=data['questions'][qid]; s=data['samples'][qid]; index=data['indexes'][(c,s,a)]
    session=data['indexes'][(c,s,'session_strong')]
    saved=read(ROOT/'selection'/c/a/(qid+'.json')); row=read(ROOT/'rows'/c/a/(qid+'.json'))
    stored=read(ROOT/'retrieval'/c/a/(qid+'.json'))
    if saved['identity']!={'row_identity':identity(data,c,a,qid),'bank_sha256':sha(bank)}: raise StopRun('Selection source identity drift')
    candidates,trace=index._candidate_pool(q['question'],data['vectors'][qid])
    if trace!=saved['candidate_trace'] or [c['fact']['id'] for c in candidates]!=saved['candidate_fact_ids']: raise StopRun('Candidate replay drift')
    channels=core.build_channels(index,candidates,q['question'],bank)
    if channels!=saved['channels']: raise StopRun('Channel score drift')
    ids=saved['selected_fact_ids']; legal={x['fact']['id'] for x in candidates}
    if not 0<len(ids)<=5 or len(set(ids))!=len(ids) or not set(ids)<=legal: raise StopRun('Invalid mathematical selection')
    text,assembly=core.plain_render(session,ids)
    units,groups=core.adapt_candidates(candidates,session.graph)
    chosen=[i for i,u in enumerate(units) if u['id'] in ids]
    if core.solver.base.selected_cost(units,chosen,groups)!=len(text)+1 or text!=saved['plain_context'] or len(text)>saved['plain_budget']:
        raise StopRun('Ordinary rendering budget drift')
    expected=sorted(ids,key=lambda i:(-float(channels[0]['dense_scores'][saved['candidate_fact_ids'].index(i)]),i))
    if expected!=ids: raise StopRun('Mathematical display order drift')
    context,final_trace,display=core.render(index,data['graphs'][(s,a)],q['question'],saved,a,data['operations'][qid],p.old.materialize_online_operation_context)
    if stored!={'identity':identity(data,c,a,qid),'selection_sha256':sha(saved),'context':context,'trace':final_trace,'materializer':display}:
        raise StopRun('Final display replay drift')
    if row['identity']!=identity(data,c,a,qid) or row['context']!=context or row['context_sha256']!=sha(context) or row['selection_sha256']!=sha(saved) or row['selected_fact_ids']!=ids:
        raise StopRun('QA selection/context binding drift')
    p.base.check_bound(ROOT,q,row)
    for receipt in row['logical_request_receipts']:
        _,_,_,raw=p.full.recover.completed_receipt(ROOT/'calls/provider'/receipt['stage']/receipt['key'])
        if raw['response'].get('usageMetadata',{})!=receipt['usage']: raise StopRun('Usage drift')
    return row


def verify_bank(data,qid):
    bank=read(ROOT/'query_bank'/(qid+'.json')); question=data['questions'][qid]['question']
    if bank['question']!=question or bank['vectors'][question]!=data['vectors'][qid]: raise StopRun('Question/vector drift')
    expected=core.qp.validate_response(question,bank['raw_projection'])
    if expected!=bank['projection'] or core.query_texts(question,expected,'q0_plus_anchored_attributes')!=bank['queries']: raise StopRun('Projection replay drift')
    entries=bank['logical_request_receipts']; planners=[r for r in entries if r['stage']=='query_projection']
    if len(planners)!=1: raise StopRun('Expected one question-only planner receipt')
    for e in entries:
        req,_,_,raw=p.full.recover.completed_receipt(ROOT/'calls/provider'/e['stage']/e['key'])
        if req['stage']=='query_projection':
            if req['body']['contents'][0]['parts']!=[{'text':core.qp.build_prompt(question)}]: raise StopRun('Planner input not question-only')
            from creamem3.experiments.locomo_online_only_20260909.runtime import response_text
            if response_text(raw['response'])!=bank['raw_projection']: raise StopRun('Projection response drift')
        elif req['stage']=='embedding':
            for request,value in zip(req['body']['requests'],raw['response']['embeddings'],strict=True):
                text=request['content']['parts'][0]['text']
                if request['taskType']!='RETRIEVAL_QUERY' or request['outputDimensionality']!=768 or value['values']!=bank['vectors'][text]: raise StopRun('Embedding body/vector drift')
        else: raise StopRun('Forbidden planner bank stage')
    for v in bank['vectors'].values(): core.vector_check(v)
    return bank


def finish(data,ids,smoke=False):
    banks={qid:verify_bank(data,qid) for qid in ids}; rows={c:{a:{} for a in ARMS} for c in CONFIGS}
    selection_stats={}
    for c in CONFIGS:
        for a in ARMS:
            if not smoke and {f.stem for f in (ROOT/'rows'/c/a).glob('*.json')}!=set(ids): raise StopRun('Incomplete final population')
            for qid in ids: rows[c][a][qid]=verify_one(data,c,a,qid,banks[qid])
            opts=[read(ROOT/'selection'/c/a/(q+'.json'))['optimization'] for q in ids]
            selection_stats[c+'/'+a]={'certified_optimal':sum(o['certified_optimal'] for o in opts),
                 'mean_seconds':sum(o['seconds'] for o in opts)/len(opts),'selected_counts':dict(Counter(o['selected_count'] for o in opts))}
            p.old.emit(f'MATH AUDIT {c}/{a} {len(ids)}')
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Registered code drift '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path)!=digest: raise StopRun('Frozen source artifact drift '+path)
    transport_report=component.provider_audit(ROOT/'calls')
    if set(transport_report['stages'])-set(ALLOWED): raise StopRun('Forbidden physical stage')
    reports={c:{a:component.summary(rows[c][a]) for a in ARMS} for c in CONFIGS}
    result={'registration_sha256':sha(data['registration']),'audit_pass':True,'smoke_only':smoke,'n':len(ids),
            'reports':reports,'paired_online_vs_session':{c:p.paired(rows[c]['opctx_original'],rows[c]['session_strong'],ids) for c in CONFIGS},
            'paired_math_vs_previous_gemini_rerank':{a:p.paired(rows['previous_2_5_5'][a],data['baselines'][a],ids) for a in ARMS},
            'paired_narrower_vs_previous':{a:p.paired(rows['narrower_1_5_3'][a],rows['previous_2_5_5'][a],ids) for a in ARMS},
            'selection_stats':selection_stats,'projection_statuses':dict(Counter(b['projection']['status'] for b in banks.values())),
            'mean_query_channels':sum(len(b['queries']) for b in banks.values())/len(ids),
            'fresh_candidate_model_reranker_calls':0,'physical_transport':transport_report,
            'not_token_matched':True,'previous_reference_has_no_source_char_cap':True}
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'),result)
    p.old.emit('MATH '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','audit'])
    command=parser.parse_args().command
    with p.old.Lease(ROOT):
        data=prepare()
        if command=='prepare': return
        if command=='audit':
            transport.MODELS['query_projection']=ALLOWED['query_projection']
            finish(data,sorted(data['questions']));return
        from dotenv import load_dotenv
        load_dotenv(p.old.ROOT/'.env',override=False)
        control=transport.ConnectionControl(live=True)
        runtime=Runtime(ROOT/'calls',control=control,allow_model_calls=True,sources=SOURCES)
        try:
            smoke_ids=p.smoke_ids(data)
            if not (ROOT/'smoke_complete.json').exists():
                for qid in smoke_ids: question_job(data,qid,runtime)
                finish(data,smoke_ids,smoke=True)
            else:
                old=read(ROOT/'smoke_complete.json')
                if not old['audit_pass'] or old['registration_sha256']!=sha(data['registration']): raise StopRun('Smoke resume drift')
            # Every submitted question completes all4 arms; methods never see scores.
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                fs=[pool.submit(question_job,data,qid,runtime) for qid in sorted(data['questions'])]
                for future in as_completed(fs): future.result()
            finish(data,sorted(data['questions']))
        except BaseException as exc:
            runtime.record_failure(exc)
            seal(ROOT/('stop_%d.json'%time.time_ns()),{'type':type(exc).__name__,'message':str(exc)[:600],'utc':p.base.now()})
            raise


if __name__=='__main__': main()

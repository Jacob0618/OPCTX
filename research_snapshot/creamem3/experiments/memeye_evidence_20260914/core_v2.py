"""New-only full371 runs; score-blind preparations and immutable request receipts."""
from __future__ import annotations
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from collections import Counter,defaultdict
from creamem3 import run_memeye_mec50_direct_gemini_v1 as direct
from creamem3.experiments.iterative_memory_20260907 import run_round as frozen
from creamem3.experiments.memeye_failure_online_20260914 import run as legacy
from creamem3.experiments.locomo_g35lite_20260908 import runtime as tr
from . import ledger
from . import core as v1

ROOT=Path(r'D:\creamem3_cache\memeye_evidence_20260914_r2')
BASE=Path(r'D:\creamem3_cache\memeye_v19\component_ablations_20260911_r1\full')
R34=Path(r'D:\creamem3_cache\memeye_v19\iterative_memory_20260907\selection_runs\r34\rows')
sha,file_sha=direct._sha256,direct._file_sha256


def checked(path):
    d=dict(frozen.checked(path)); d.pop('seal_sha256'); return d


def put(path,data): return direct._write_once(Path(path),frozen.sealed(data))


def source_data():
    bundle=checked(BASE/'prepared.json'); graphs={}
    for task,receipt in bundle['graph_receipts'].items():
        if file_sha(Path(receipt['path']))!=receipt['sha256']: raise RuntimeError('Frozen graph drift')
        graphs[task]=direct._read_object(Path(receipt['path']),'graph')
    return bundle,graphs


def code_hashes():
    return {str(Path(m.__file__).resolve()):file_sha(Path(m.__file__)) for m in
            (ledger,legacy,frozen,direct,tr)}|{str(Path(__file__).resolve()):file_sha(Path(__file__))}


def prepare_base():
    bundle,graphs=source_data(); rows=[]
    for item in bundle['prepared']:
        qid=item['query']['row_id']; channel=checked(R34/(sha(qid)+'.json'))
        if channel['query']!=item['query']: raise RuntimeError('Query drift')
        rows.append({'query':item['query'],'context':item['context'],'context_sha256':item['context_sha256'],
            'selected_fact_ids':item['selection']['selected_fact_ids'],'budget':channel['baseline_chars'],
            'old_selection':True})
    for mode in ('control','ledger'):
        put(ROOT/'prepared'/(mode+'.json'),{'mode':mode,'reader':'legacy' if mode=='control' else 'ledger',
            'rows':rows,'source_prepared_sha256':file_sha(BASE/'prepared.json'),'graph_receipts':bundle['graph_receipts'],
            'score_free':True,'code_hashes':code_hashes()})
    print('BASE PREPARED',len(rows),flush=True)


def generate(item,kind,old_reader,runtime,authority,graph):
    if kind=='legacy': return legacy.generate_answer(item,old_reader,authority)
    question=item['query']['question']; qid=item['query']['row_id']; text=item['context']
    cap=2600 if item['query']['reader_route']=='two_stage' else 1200
    body=ledger.body(question,text,cap)
    response=runtime.post('ledger',qid,ledger.MODEL,'generateContent',body)
    raw=tr.response_text(response); packet=direct.predict._parse_json_object(raw)
    format_fallback=None
    try:
        if [c.get('finishReason') for c in response.get('candidates',[])]!=['STOP']:
            raise ValueError('unfinished_ledger')
        if not packet: raise ValueError('invalid_json')
        final_context,receipt=ledger.compile_packet(packet,text,ledger.image_keys(graph))
    except ValueError as exc:
        format_fallback=str(exc)
        final_context=text
        receipt={'input_rows':None,'retained_rows':0,'rejected':[],'unique_image_count':None,
                 'raw_fallback':True,'complete_after_validation':False,'format_fallback':format_fallback}
    if item['query']['reader_route']=='two_stage' or format_fallback:
        answer=old_reader.generate('answer',authority['open'].format(context=final_context,question=question),
            json_mode=False,max_output_tokens=500,cache_metadata={'cell_id':direct.CELL_ID,'row_id':qid,'qa_context_sha256':sha(final_context)})
        prediction=answer.text.strip(); stages=[legacy.reply_meta(answer)]
    else:
        prediction=str(receipt['unique_image_count']) if receipt['unique_image_count'] is not None else str(packet['candidate_answer']).strip()
        stages=[]
    if not prediction: raise RuntimeError('Empty ledger answer')
    identity={'stage':'ledger','scope':qid,'model':ledger.MODEL,'method':'generateContent','body':body}
    return {'prediction':prediction,'reader_route':item['query']['reader_route'],'reader_stages':stages,
        'ledger_provider_dir':str(runtime.root/'provider/ledger'/tr.sha(identity)),
        'ledger_response_sha256':tr.sha(response),'ledger_validation':receipt,'ledger_format_fallback':format_fallback,
        'ledger_finish_reason':[c.get('finishReason') for c in response.get('candidates',[])],
        'actual_final_context_sha256':sha(final_context),'actual_final_context_chars':len(final_context)}


def run(modes,allow,smoke=False):
    prepared={m:checked(ROOT/'prepared'/(m+'.json')) for m in modes}
    for m,p in prepared.items():
        if p['code_hashes']!=code_hashes(): raise RuntimeError('Code drift before '+m)
        if len(p['rows'])!=371 or len({r['query']['row_id'] for r in p['rows']})!=371: raise RuntimeError('Full371 required')
        for item in p['rows']:
            if len(item['context'])>item['budget'] or sha(item['context'])!=item['context_sha256'] or not 0<len(item['selected_fact_ids'])<=50:
                raise RuntimeError('Context budget/hash/count violation')
        if [r['query'] for r in p['rows']]!=[r['query'] for r in prepared[modes[0]]['rows']]:
            raise RuntimeError('Cross-arm population/order drift')
    label='__'.join(modes)
    reg={'modes':modes,'prepared_files':{m:file_sha(ROOT/'prepared'/(m+'.json')) for m in modes},
         'code_hashes':code_hashes(),'n':371,'qa_model':ledger.MODEL,'judge_model':direct.g35_runtime.MODEL_ID,
         'no_rerank':True,'max_facts':50,'same_per_question_character_allowance':True,
         'reader_caps':{'focused_or_ledger':2600,'fused_or_ledger':1200,'final_answer':500},
         'new_ledger_schema':ledger.SCHEMA,'no_historical_prediction_or_score_imports':True,
         'format_policy':'Unfinished or invalid Ledger -> original raw selected context + original QA prompt cap500; no new retrieval; fused may add one call',
         'provider_cache_reuse':'r1 exact requests only; control QA and identical judge triples may coalesce'}
    put(ROOT/'registrations'/(label+'.json'),reg)
    if not allow: print('LIVE PREFLIGHT',label,flush=True); return
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3]/'.env',override=False)
    authority,_=direct._reader_authority(); _,graphs=source_data()
    # Labels are opened only after all selection artifacts have sealed.
    gold=direct._gold_rows()
    old=frozen._reader_backend(v1.ROOT/'shared_reader',{'stage_provider_attempt_hard_ceilings':
        {'semantic_planner':0,'focused_reader':10000,'answer_reader':10000,'structured_fused_reader':10000}},
        {'focused_max_output_tokens':2600,'fused_max_output_tokens':1200})
    runtime=tr.Runtime(v1.ROOT/'structured_provider',allow_model_calls=True,concurrency=8,timeout=300)
    gm=direct.g35_runtime
    limits={s:gm.StageLimits(400 if s=='judge' else 16384,10000 if s=='judge' else 0,5) for s in gm.STAGES}
    judge=gm.Gemini35Runtime(v1.ROOT/'shared_judge',allow_model_calls=True,
        provider_factory=lambda:direct.retry5._ReliableRESTProvider(gm),stage_limits=limits)
    stop=threading.Event()
    def job(i):
        for mode in modes:
            if stop.is_set(): raise RuntimeError('Stopped after failure')
            p=prepared[mode]; item=p['rows'][i]; qid=item['query']['row_id']; path=ROOT/'rows'/mode/(sha(qid)+'.json')
            identity={'mode':mode,'prepared_sha256':reg['prepared_files'][mode],'context_sha256':item['context_sha256']}
            if path.exists():
                if checked(path)['identity']!=identity: raise RuntimeError('Resume row drift')
                continue
            try:
                graph=graphs[item['query']['task']]
                result=generate(item,p['reader'],old,runtime,authority,graph)
                put(ROOT/'predictions'/mode/(sha(qid)+'.json'),{'row_id':qid,'identity':identity,**result})
                g=gold[qid]; prompt=gm.build_mgtm_judge_prompt(g['question'],g['gold'],result['prediction'])
                j=judge.generate('judge',prompt,json_mode=True,max_output_tokens=400,
                    cache_metadata={'purpose':'evidence_factorial_shared_triple','row_id':qid,'triple_sha256':sha([g['question'],g['gold'],result['prediction']])})
                verdict=gm.parse_mgtm_judge_response(j.text)
                put(path,{'row_id':qid,'identity':identity,'x':g['x'],'y':g['y'],**result,
                    'mapped_score':verdict.mapped_score,'raw_score':verdict.raw_score,'reasoning':verdict.reasoning,
                    'judge_response':legacy.reply_meta(j),'selected_fact_ids':item['selected_fact_ids'],'context_chars':len(item['context'])})
            except BaseException as exc:
                stop.set(); runtime.stopped.set(); put(ROOT/'failures'/(sha([mode,qid])+'.json'),
                    {'mode':mode,'row_id':qid,'type':type(exc).__name__,'message':str(exc)}); raise
        return qid
    ids=range(2) if smoke else range(371)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(job,i) for i in ids]
        for n,f in enumerate(as_completed(futures),1):
            qid=f.result()
            if n%10==0 or n==len(ids): print('EVIDENCE QA',label,n,len(ids),qid,flush=True)
    results={}
    if code_hashes()!=reg['code_hashes']: raise RuntimeError('Code drift during execution')
    for mode in modes:
        rows=[checked(ROOT/'rows'/mode/(sha(prepared[mode]['rows'][i]['query']['row_id'])+'.json')) for i in ids]
        results[mode]=legacy.summary(rows)
    put(ROOT/('smoke' if smoke else 'complete')/(label+'.json'),
        {'complete':True,'smoke_only':smoke,'n':len(ids),'reports':results,'registration_sha256':sha(reg),'needs_independent_audit':True})
    print('EVIDENCE COMPLETE',results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','run']); p.add_argument('--modes',nargs='+',default=['control','ledger'])
    p.add_argument('--allow-model-calls',action='store_true'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    prepare_base() if a.command=='prepare' else run(a.modes,a.allow_model_calls,a.smoke)

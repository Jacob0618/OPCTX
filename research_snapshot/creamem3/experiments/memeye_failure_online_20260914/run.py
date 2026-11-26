"""Frozen Reader/Judge, shared exact-request caches, four registered arms."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import threading
from collections import defaultdict,Counter
from creamem3 import run_memeye_mec50_direct_gemini_v1 as direct
from creamem3.experiments.iterative_memory_20260907 import run_round as frozen
from creamem3.experiments.iterative_memory_20260907 import candidate_context_policy as policy
from . import selection as config


def reply_meta(reply):
    return {k:getattr(reply,k) for k in ('stage','request_sha256','response_sha256','cache_path','cache_hit')}


def generate_answer(item,reader,authority):
    query=item['query']; qid=query['row_id']; text=item['context']; ctx=direct._sha256(text)
    metadata={'cell_id':direct.CELL_ID,'row_id':qid,'context_sha256':ctx}; stages=[]
    if query['reader_route']=='two_stage':
        focused=reader.generate('focused',authority['focused'].replace('{question}',query['question']).replace('{context}',text),
            json_mode=True,max_output_tokens=2600,cache_metadata=metadata)
        packet=direct.predict._parse_json_object(focused.text) or {'_failed':True}
        final_context=policy.rendered_context(authority['render'],packet,text,'candidate_first','candidate_last')
        answer=reader.generate('answer',authority['open'].format(context=final_context,question=query['question']),
            json_mode=False,max_output_tokens=500,cache_metadata={'cell_id':direct.CELL_ID,'row_id':qid,'qa_context_sha256':direct._sha256(final_context)})
        prediction=answer.text.strip(); stages=[reply_meta(focused),reply_meta(answer)]
    elif query['reader_route']=='structured_fused':
        answer=reader.generate('structured_fused',authority['fused'].format(question=query['question'],context=text),
            json_mode=True,max_output_tokens=1200,cache_metadata=metadata)
        packet=direct.predict._parse_json_object(answer.text)
        prediction=str(packet.get('final_answer') or 'Not found in memory.').strip()
        final_context=text; stages=[reply_meta(answer)]
    else: raise RuntimeError('Unknown original Reader route')
    if not prediction: raise RuntimeError('Empty prediction')
    return {'prediction':prediction,'reader_stages':stages,'actual_final_context_sha256':direct._sha256(final_context),
            'actual_final_context_chars':len(final_context),'reader_route':query['reader_route']}


def summary(rows):
    cats=defaultdict(list)
    for row in rows:
        for key in ('x','y'): cats[row[key]].append(row['mapped_score'])
    return {'n':len(rows),'mapped_sum':sum(r['mapped_score'] for r in rows),
        'mapped_percent':100*sum(r['mapped_score'] for r in rows)/len(rows),
        'by_type':{k:{'n':len(v),'mapped_percent':100*sum(v)/len(v)} for k,v in cats.items()}}


def run(allow=False,smoke=False):
    root=config.ROOT; reg=frozen.checked(root/'factorial_registration.json')
    if reg['source_hashes']!=config.source_hashes(): raise RuntimeError('Registered code drift')
    complete=frozen.checked(root/'selection_complete.json')
    if not complete['complete'] or complete['registration_sha256']!=config.sha(reg): raise RuntimeError('Selection not accepted')
    prepared={m:frozen.checked(root/'prepared'/(m+'.json')) for m in config.MODES}
    for m,data in prepared.items():
        if data['registration_sha256']!=config.sha(reg) or [r['query']['row_id'] for r in data['rows']]!=reg['all_ids']:
            raise RuntimeError('Prepared binding/population drift')
        for item in data['rows']:
            if direct._sha256(item['context'])!=item['context_sha256'] or len(item['context'])>item['budget'] or len(item['selected_fact_ids'])>50:
                raise RuntimeError('Prepared context/cap violation')
    if not allow: print('FACTORIAL LIVE PREFLIGHT PASS',flush=True); return
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3]/'.env',override=False)
    authority,_=direct._reader_authority()
    budget={'stage_provider_attempt_hard_ceilings':{'semantic_planner':0,'focused_reader':7420,'answer_reader':7420,'structured_fused_reader':7420}}
    reader=frozen._reader_backend(root/'factorial_shared',budget,reg['reader'])
    # Labels are opened only after score-blind selection preparation has sealed.
    gold=direct._gold_rows(); gm=direct.g35_runtime
    limits={s:gm.StageLimits(400 if s=='judge' else 16384,7420 if s=='judge' else 0,5) for s in gm.STAGES}
    judge=gm.Gemini35Runtime(root/'factorial_judge',allow_model_calls=True,
        provider_factory=lambda:direct.retry5._ReliableRESTProvider(gm),stage_limits=limits)
    stop=threading.Event()
    def question_job(i):
        for mode in config.MODES:
            if stop.is_set(): raise RuntimeError('Dispatch stopped after failure')
            item=prepared[mode]['rows'][i]; qid=item['query']['row_id']; path=root/'rows'/mode/(config.sha(qid)+'.json')
            identity={'mode':mode,'registration_sha256':config.sha(reg),'context_sha256':item['context_sha256'],'selection_file_sha256':item['selection_file_sha256']}
            if path.exists():
                row=frozen.checked(path)
                if row['identity']!=identity: raise RuntimeError('Resume row drift')
                continue
            try:
                result=generate_answer(item,reader,authority)
                config.put(root/'predictions'/mode/(config.sha(qid)+'.json'),{'identity':identity,'row_id':qid,**result})
                g=gold[qid]; prompt=gm.build_mgtm_judge_prompt(g['question'],g['gold'],result['prediction'])
                verdict_raw=judge.generate('judge',prompt,json_mode=True,max_output_tokens=400,
                    cache_metadata={'purpose':'factorial_shared_exact_triple','row_id':qid,'triple_sha256':config.sha([g['question'],g['gold'],result['prediction']])})
                verdict=gm.parse_mgtm_judge_response(verdict_raw.text)
                row={'identity':identity,'row_id':qid,'question':g['question'],'x':g['x'],'y':g['y'],**result,
                    'raw_score':verdict.raw_score,'mapped_score':verdict.mapped_score,'reasoning':verdict.reasoning,
                    'judge_response':reply_meta(verdict_raw),'judge_usage':asdict(verdict_raw.usage),
                    'selected_fact_ids':item['selected_fact_ids'],'context_chars':len(item['context'])}
                config.put(path,row)
            except BaseException as exc:
                stop.set(); config.put(root/'failures'/(config.sha([mode,qid])+'.json'),
                    {'mode':mode,'row_id':qid,'exception_type':type(exc).__name__,'message':str(exc)}); raise
        return qid
    ids=list(range(2)) if smoke else list(range(371))
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(question_job,i) for i in ids]
        for n,job in enumerate(as_completed(jobs),1):
            qid=job.result()
            if n%10==0 or n==len(jobs): print('FACTORIAL QA',n,len(jobs),qid,flush=True)
    if reg['source_hashes']!=config.source_hashes(): raise RuntimeError('Code drift during run')
    rows={m:[frozen.checked(root/'rows'/m/(config.sha(reg['all_ids'][i])+'.json')) for i in ids] for m in config.MODES}
    reports={m:summary(rs) for m,rs in rows.items()}; pairs={}
    for method in ('dense','math'):
        a,b=rows['session_'+method],rows['online_'+method]
        pairs[method]={'improved':sum(y['mapped_score']>x['mapped_score'] for x,y in zip(a,b)),
            'regressed':sum(y['mapped_score']<x['mapped_score'] for x,y in zip(a,b)),
            'net_pp':reports['online_'+method]['mapped_percent']-reports['session_'+method]['mapped_percent']}
    config.put(root/('factorial_smoke_complete.json' if smoke else 'factorial_complete.json'),
        {'complete':True,'smoke_only':smoke,'n':len(ids),'total_rows':4*len(ids),'registration_sha256':config.sha(reg),
         'reports':reports,'paired_online_minus_session':pairs,'shared_exact_request_cache':True,
         'event_prose_rendered':False,'new_protocol_not_original5283':True,'requires_independent_final_audit':True})
    print('FACTORIAL COMPLETE',reports,pairs,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--allow-model-calls',action='store_true'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); run(args.allow_model_calls,args.smoke)

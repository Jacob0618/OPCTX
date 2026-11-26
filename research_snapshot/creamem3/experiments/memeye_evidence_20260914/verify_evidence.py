"""Read-only, no-model reconstruction of executed evidence experiments."""
import argparse
import json
from collections import Counter
from pathlib import Path
from . import core,core_v2,ledger
from creamem3.experiments.memeye_failure_online_20260914 import verify as previous
from creamem3.experiments.iterative_memory_20260907 import candidate_context_policy as policy
from creamem3.experiments.mec_transfer_20260906 import run_memeye as adapter
from creamem3.experiments.locomo_g35lite_20260908 import runtime as tr

require=previous.require

def generic(directory,identity):
    directory=Path(directory); request=tr.read(directory/'request.json'); final=tr.read(directory/'final.json')
    key=tr.sha(identity)
    require(request==identity==final['identity'] and key==directory.name==final['request_sha256'],'Generic request binding')
    raw=tr.read(directory/f"attempt_{final['attempt']:02d}.response.json")
    require(tr.sha(raw)==final['response_sha256'],'Generic response binding')
    for p in directory.glob('attempt_*.intent.json'):
        intent=tr.read(p); require(intent['request_sha256']==key,'Generic intent binding')
        response=p.with_name(p.name.replace('.intent.json','.response.json')); error=p.with_name(p.name.replace('.intent.json','.error.json'))
        require(response.exists()!=error.exists(),'Generic missing/ambiguous terminal receipt')
        if error.exists(): require(tr.read(error)['retryable'],'Generic nonretryable error')
        else: tr.read(response)
    return raw['response']

def verify(modes,write=False):
    root=core_v2.ROOT; label='__'.join(modes); completion=core.checked(root/'complete'/(label+'.json'))
    reg=core.checked(root/'registrations'/(label+'.json'))
    require(completion['complete'] and completion['n']==371 and not completion['smoke_only'],'Not full371')
    require(core.sha(reg)==completion['registration_sha256'],'Completion binding')
    require(reg['code_hashes']==core_v2.code_hashes(),'Runner code changed')
    _,graphs=core.source_data(); authority,_=core.direct._reader_authority(); gold=core.direct._gold_rows()
    rm=core.frozen.g25._load_source_runtime(); gm=core.direct.g35_runtime
    rlimits={s:rm.StageLimits(cap,0 if s=='planner' else 10000,5) for s,cap in
             [('planner',512),('focused',2600),('answer',500),('structured_fused',1200)]}
    jlimits={s:gm.StageLimits(400 if s=='judge' else 16384,10000 if s=='judge' else 0,5) for s in gm.STAGES}
    rb=core.ROOT/'shared_reader/provider/predict/g25_reader'; jb=core.ROOT/'shared_judge'
    reports={}; allrows={}; stats={}; rlinks=set(); jlinks=set(); glinks=set()
    for mode in modes:
        path=root/'prepared'/(mode+'.json'); p=core.checked(path)
        require(core.file_sha(path)==reg['prepared_files'][mode],'Prepared binding')
        require(len(p['rows'])==371 and len(list((root/'rows'/mode).glob('*.json')))==371,'Population census')
        rows=[]; usage=Counter(); fallback=Counter(); rejected=0; kept=0; rowcount=0
        def old_call(meta,prompt,metadata):
            response,key=previous.request_check(rb,meta,prompt,rm,rlimits,metadata); rlinks.add(key)
            usage.update(response['usage']); usage['logical_reader_calls']+=1
            return response
        for item in p['rows']:
            q=item['query']; qid=q['row_id']; text=item['context']; ctx=core.sha(text); question=q['question']
            row=core.checked(root/'rows'/mode/(core.sha(qid)+'.json')); pred=core.checked(root/'predictions'/mode/(core.sha(qid)+'.json'))
            identity={'mode':mode,'prepared_sha256':reg['prepared_files'][mode],'context_sha256':ctx}
            require(row['identity']==pred['identity']==identity,'Row identity')
            require(qid==row['row_id']==pred['row_id'] and question==gold[qid]['question'],'Question binding')
            ids=item['selected_fact_ids']; graph=graphs[q['task']]
            require(ids==row['selected_fact_ids'] and 0<len(ids)<=50 and len(ids)==len(set(ids)),'Selection binding')
            render,_=adapter.render_selection(graph,ids)
            require(text==render and ctx==item['context_sha256'] and len(text)<=item['budget'],'Raw context and budget')
            stages=row['reader_stages']; meta={'cell_id':core.direct.CELL_ID,'row_id':qid,'context_sha256':ctx}
            if p['reader']=='legacy':
                if q['reader_route']=='two_stage':
                    require([s['stage'] for s in stages]==['focused','answer'],'Control stage route')
                    first=old_call(stages[0],authority['focused'].replace('{question}',question).replace('{context}',text),meta)
                    packet=core.direct.predict._parse_json_object(first['text']) or {'_failed':True}
                    if packet.get('_failed'): fallback['legacy_focused_invalid_json']+=1
                    finalctx=policy.rendered_context(authority['render'],packet,text,'candidate_first','candidate_last')
                    final=old_call(stages[1],authority['open'].format(context=finalctx,question=question),
                        {'cell_id':core.direct.CELL_ID,'row_id':qid,'qa_context_sha256':core.sha(finalctx)})
                    answer=final['text'].strip()
                else:
                    require([s['stage'] for s in stages]==['structured_fused'],'Control fused route')
                    finalctx=text; final=old_call(stages[0],authority['fused'].format(context=text,question=question),meta)
                    packet=core.direct.predict._parse_json_object(final['text'])
                    if not packet: fallback['legacy_fused_invalid_json']+=1
                    answer=str(packet.get('final_answer') or 'Not found in memory.').strip()
            else:
                body=ledger.body(question,text,2600 if q['reader_route']=='two_stage' else 1200)
                identity2={'stage':'ledger','scope':qid,'model':ledger.MODEL,'method':'generateContent','body':body}
                response=generic(row['ledger_provider_dir'],identity2); glinks.add(row['ledger_provider_dir'])
                require(tr.sha(response)==row['ledger_response_sha256'],'Ledger result binding')
                u=response.get('usageMetadata',{}); usage['prompt_tokens']+=u.get('promptTokenCount',0)
                usage['output_tokens']+=u.get('candidatesTokenCount',0); usage['thought_tokens']+=u.get('thoughtsTokenCount',0)
                usage['total_tokens']+=u.get('totalTokenCount',0); usage['logical_reader_calls']+=1
                packet=core.direct.predict._parse_json_object(tr.response_text(response)); format_fallback=None
                try:
                    if [c.get('finishReason') for c in response.get('candidates',[])]!=['STOP']: raise ValueError('unfinished_ledger')
                    if not packet: raise ValueError('invalid_json')
                    finalctx,receipt=ledger.compile_packet(packet,text,ledger.image_keys(graph))
                except ValueError as exc:
                    format_fallback=str(exc); finalctx=text
                    receipt={'input_rows':None,'retained_rows':0,'rejected':[],'unique_image_count':None,
                             'raw_fallback':True,'complete_after_validation':False,'format_fallback':format_fallback}
                require(receipt==row['ledger_validation'] and format_fallback==row['ledger_format_fallback'],'Ledger validation reconstruction')
                if format_fallback: fallback[format_fallback]+=1
                if receipt['raw_fallback']: fallback['all_raw_fallback']+=1
                rejected+=len(receipt['rejected']); kept+=receipt['retained_rows']; rowcount+=receipt['input_rows'] or 0
                if q['reader_route']=='two_stage' or format_fallback:
                    require([s['stage'] for s in stages]==['answer'],'Ledger final route')
                    final=old_call(stages[0],authority['open'].format(context=finalctx,question=question),
                        {'cell_id':core.direct.CELL_ID,'row_id':qid,'qa_context_sha256':core.sha(finalctx)})
                    answer=final['text'].strip()
                else:
                    require(not stages,'Ledger fused extra call')
                    answer=str(receipt['unique_image_count']) if receipt['unique_image_count'] is not None else str(packet['candidate_answer']).strip()
            require(answer==row['prediction']==pred['prediction'],'Answer extraction')
            require(core.sha(finalctx)==row['actual_final_context_sha256'] and len(finalctx)==row['actual_final_context_chars'],'Final QA context')
            prompt=gm.build_mgtm_judge_prompt(question,gold[qid]['gold'],answer)
            judged,key=previous.request_check(jb,row['judge_response'],prompt,gm,jlimits,
                {'purpose':'evidence_factorial_shared_triple','row_id':qid,'triple_sha256':core.sha([question,gold[qid]['gold'],answer])})
            jlinks.add(key); verdict=gm.parse_mgtm_judge_response(judged['text'])
            require(row['mapped_score']==verdict.mapped_score and row['raw_score']==verdict.raw_score,'Judge score')
            require(row['x']==gold[qid]['x'] and row['y']==gold[qid]['y'],'X/Y labels')
            rows.append(row)
        result=core.legacy.summary(rows); require(result==completion['reports'][mode],'Aggregation')
        reports[mode]=result; allrows[mode]={r['row_id']:r for r in rows}
        stats[mode]={'reader_token_totals':dict(usage),'format_and_raw_fallbacks':dict(fallback),
            'ledger_rows':rowcount,'kept_ledger_rows':kept,'rejected_ledger_rows':rejected,
            'mean_selected':sum(len(r['selected_fact_ids']) for r in rows)/371,'mean_input_chars':sum(r['context_chars'] for r in rows)/371,
            'mean_final_context_chars':sum(r['actual_final_context_chars'] for r in rows)/371}
    paired={}
    if len(modes)==2:
        a,b=(allrows[m] for m in modes); flips=[]
        for qid in a:
            if a[qid]['mapped_score']!=b[qid]['mapped_score']:
                flips.append({'row_id':qid,'before':a[qid]['mapped_score'],'after':b[qid]['mapped_score'],
                    'prediction_before':a[qid]['prediction'],'prediction_after':b[qid]['prediction']})
        paired={'improved':sum(f['after']>f['before'] for f in flips),'regressed':sum(f['after']<f['before'] for f in flips),
            'net_pp':reports[modes[1]]['mapped_percent']-reports[modes[0]]['mapped_percent'],'flips':flips}
    out={'status':'PASS','n':371,'modes':modes,'reports':reports,'stats':stats,'paired':paired,
         'all_row_payloads_reconstructed':True,'source_bound_quotes_not_semantic_correctness_proof':True,
         'unique_linked_legacy_reader_calls':len(rlinks),'unique_linked_ledger_calls':len(glinks),
         'unique_linked_judge_calls':len(jlinks),'audit_model_calls':0,
         'registration_sha256':core.sha(reg),'audit_script_sha256':core.file_sha(Path(__file__))}
    if write: core.put(root/'audit'/(label+'.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k!='paired'},ensure_ascii=False),flush=True)
    print('PAIRED',json.dumps({k:v for k,v in paired.items() if k!='flips'}),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--modes',nargs='+',default=['control','ledger']); p.add_argument('--write',action='store_true'); a=p.parse_args()
    verify(a.modes,a.write)

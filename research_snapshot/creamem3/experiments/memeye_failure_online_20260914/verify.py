"""Independent read-only payload, selection, lifecycle and score verification."""
from __future__ import annotations
import argparse
import inspect
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from . import audit,selection,run,run_compat
from creamem3.experiments.iterative_memory_20260907 import candidate_context_policy as policy

ROOT=audit.ROOT
sha=audit.sha


def checked(path):
    result=dict(audit.frozen.checked(path)); result.pop('seal_sha256'); return result


def require(value,message):
    if not value: raise AssertionError(message)


def physical(branch):
    # Reuse the independent historical receipt checker on one explicit branch.
    # Correct its row-ID logical counter for four contexts per question.
    previous=audit.frozen.previous
    src=inspect.getsource(previous.provider_audit)
    old='for branch in (cache / "provider" / "predict" / "g25_reader", cache / "provider" / "score" / "g35_judge"):'
    require(src.count(old)==1,'audit branch anchor drift')
    src=src.replace(old,'for branch in (cache,):')
    old='logical_key = str(metadata.get("row_id") or metadata.get("exact_triple_sha256") or plan["prompt_sha256"])'
    require(src.count(old)==1,'audit logical identity anchor drift')
    src=src.replace(old,'logical_key = plan["request_sha256"]')
    namespace=dict(vars(previous)); exec(compile(src,'<readonly-single-branch-physical-audit>','exec'),namespace)
    return namespace['provider_audit'](branch)


def request_check(branch,meta,prompt,module,limits,cache_metadata):
    final=audit.read(meta['cache_path'],'response'); stage=meta['stage']
    plan=audit.read(branch/'plans'/stage/(final['resume_hash']+'.json'),'plan')
    payload=module.build_generate_content_request(stage,prompt,json_mode=stage!='answer',
                max_output_tokens=limits[stage].max_output_tokens,stage_limits=limits)
    require(plan['payload_sha256']==sha(payload),'payload/cap mismatch')
    require(plan['prompt_sha256']==sha(prompt),'prompt mismatch')
    require(plan['cache_metadata']==cache_metadata,'metadata mismatch')
    require(plan['model']==final['model']==module.MODEL_ID,'model mismatch')
    for field in ('request_sha256','response_sha256'):
        require(meta[field]==final[field],'stage binding '+field)
    require(final['response_sha256']==sha(final['text']),'text integrity')
    request_id=module.request_identity(stage=stage,request=payload,stage_limits=limits,cache_metadata=cache_metadata)
    require(request_id==plan['request_sha256']==final['request_sha256'],'request identity mismatch')
    resume=module.resume_hash(stage=stage,request_sha256=request_id,stage_limits=limits)
    require(resume==plan['resume_hash']==final['resume_hash'],'resume identity mismatch')
    return final,(stage,request_id)


def factorial_verify():
    registration=checked(ROOT/'factorial_registration.json')
    completion=checked(ROOT/'factorial_complete.json')
    require(registration['source_hashes']==selection.source_hashes(),'registered source drift')
    compat=checked(ROOT/'runner_compat_registration.json')
    require(compat['base_runner_sha256']==audit.direct._file_sha256(Path(run.__file__)),'base runner drift')
    require(compat['compat_runner_sha256']==audit.direct._file_sha256(Path(run_compat.__file__)),'compat runner drift')
    for path,digest in registration['channel_files'].items():
        require(audit.direct._file_sha256(Path(path))==digest,'frozen channel drift')
    graphs={}
    for key,receipt in registration['source_graphs'].items():
        require(audit.direct._file_sha256(Path(receipt['path']))==receipt['sha256'],'graph drift')
        graphs[key]=audit.read(Path(receipt['path']),'graph')
    for key,g in graphs.items():
        if key.startswith('session::'):
            online=graphs[key.replace('session::','online::')]
            require(g['fact_nodes']==online['fact_nodes'] and g['memeye_source_map']==online['memeye_source_map'],'unequal Writer inputs')
    require(completion['n']==371 and completion['total_rows']==1484,'incomplete')
    require(completion['registration_sha256']==sha(registration),'completion binding')
    reader_branch=ROOT/'factorial_shared/provider/predict/g25_reader'
    judge_branch=ROOT/'factorial_judge'
    reader_audit=physical(reader_branch); judge_audit=physical(judge_branch)
    reader_finals=reader_audit.pop('_final_by_request'); judge_finals=judge_audit.pop('_final_by_request')
    gm=audit.direct.g35_runtime; rm=audit.frozen.g25._load_source_runtime()
    rc=registration['reader']
    rlimits={s:rm.StageLimits(cap,0 if s=='planner' else 7420,5) for s,cap in
             [('planner',512),('focused',rc['focused_max_output_tokens']),('answer',500),('structured_fused',rc['fused_max_output_tokens'])]}
    jlimits={s:gm.StageLimits(400 if s=='judge' else 16384,7420 if s=='judge' else 0,5) for s in gm.STAGES}
    authority,_=audit.direct._reader_authority(); gold=audit.direct._gold_rows()
    prepared={}; rows={}; sel={}; rlinks=set(); jlinks=set(); stats={}
    for mode in selection.MODES:
        bundle=checked(ROOT/'prepared'/(mode+'.json')); prepared[mode]=bundle['rows']
        require(bundle['registration_sha256']==sha(registration),'prepared binding')
        require([r['query']['row_id'] for r in bundle['rows']]==registration['all_ids'],'population/order drift')
        require(len(list((ROOT/'rows'/mode).glob('*.json')))==371,'row census')
        rows[mode]=[]; sel[mode]=[]; optstats=Counter()
        for item in bundle['rows']:
            qid=item['query']['row_id']; query=item['query']['question']; context=item['context']; ctx=sha(context)
            selection_path=ROOT/'selection'/mode/(sha(qid)+'.json'); choice=checked(selection_path)
            row=checked(ROOT/'rows'/mode/(sha(qid)+'.json')); prediction=checked(ROOT/'predictions'/mode/(sha(qid)+'.json'))
            require(row['row_id']==prediction['row_id']==qid and row['question']==gold[qid]['question'],'row question binding')
            require(row['identity']==prediction['identity']=={'mode':mode,'registration_sha256':sha(registration),
                    'context_sha256':ctx,'selection_file_sha256':audit.direct._file_sha256(selection_path)},'row identity')
            require(ctx==item['context_sha256'],'context hash')
            ids=item['selected_fact_ids']
            require(0<len(ids)<=50 and len(ids)==len(set(ids)) and set(ids)<=set(choice['pool_ids']),'invalid subset')
            require(ids==choice['selected_fact_ids']==row['selected_fact_ids'],'subset binding')
            require(len(context)<=item['budget']==choice['budget'],'budget mismatch')
            graph=graphs['session::'+item['query']['task']]
            rerender,_=selection.adapter.render_selection(graph,ids)
            require(rerender==context,'raw-fact renderer mismatch')
            require(len(context)==choice['optimization']['cost']-1,'solver/render cost mismatch')
            require(row['reader_route']==item['query']['reader_route'],'reader route mismatch')
            base_metadata={'cell_id':audit.direct.CELL_ID,'row_id':qid,'context_sha256':ctx}
            stages=row['reader_stages']
            if row['reader_route']=='two_stage':
                require([s['stage'] for s in stages]==['focused','answer'],'wrong two-stage route')
                prompt=authority['focused'].replace('{question}',query).replace('{context}',context)
                focus,key=request_check(reader_branch,stages[0],prompt,rm,rlimits,base_metadata); rlinks.add(key)
                packet=audit.direct.predict._parse_json_object(focus['text']) or {'_failed':True}
                final_context=policy.rendered_context(authority['render'],packet,context,'candidate_first','candidate_last')
                prompt=authority['open'].format(context=final_context,question=query)
                final,key=request_check(reader_branch,stages[1],prompt,rm,rlimits,
                    {'cell_id':audit.direct.CELL_ID,'row_id':qid,'qa_context_sha256':sha(final_context)})
                rlinks.add(key); answer=final['text'].strip()
            else:
                require([s['stage'] for s in stages]==['structured_fused'],'wrong fused route')
                final_context=context; prompt=authority['fused'].format(question=query,context=context)
                final,key=request_check(reader_branch,stages[0],prompt,rm,rlimits,base_metadata); rlinks.add(key)
                answer=str(audit.direct.predict._parse_json_object(final['text']).get('final_answer') or 'Not found in memory.').strip()
            require(answer==row['prediction']==prediction['prediction'],'answer extraction mismatch')
            require(sha(final_context)==row['actual_final_context_sha256'] and len(final_context)==row['actual_final_context_chars'],'actual QA context drift')
            prompt=gm.build_mgtm_judge_prompt(query,gold[qid]['gold'],answer)
            judged,key=request_check(judge_branch,row['judge_response'],prompt,gm,jlimits,
                {'purpose':'factorial_shared_exact_triple','row_id':qid,'triple_sha256':sha([query,gold[qid]['gold'],answer])})
            jlinks.add(key); verdict=gm.parse_mgtm_judge_response(judged['text'])
            require(verdict.mapped_score==row['mapped_score'] and verdict.raw_score==row['raw_score'],'Judge extraction mismatch')
            require(row['x']==gold[qid]['x'] and row['y']==gold[qid]['y'],'category mismatch')
            optstats[str(choice['optimization'].get('optimal',choice['optimization'].get('status')))]+=1
            rows[mode].append(row); sel[mode].append(choice)
        actual=run.summary(rows[mode]); require(actual==completion['reports'][mode],'aggregate mismatch')
        stats[mode]={'mean_pool':sum(len(r['pool_ids']) for r in sel[mode])/371,
                     'mean_selected':sum(len(r['selected_fact_ids']) for r in rows[mode])/371,
                     'mean_chars':sum(r['context_chars'] for r in rows[mode])/371,'solver_status':dict(optstats)}
    require(rlinks==set(reader_finals) and jlinks==set(judge_finals),'unreferenced/missing physical generation')
    for i,qid in enumerate(registration['all_ids']):
        require(len({prepared[m][i]['budget'] for m in selection.MODES})==1,'cross-arm unequal budget')
        for kind in ('session','online'):
            require(sel[kind+'_dense'][i]['pool_ids']==sel[kind+'_math'][i]['pool_ids'],'unequal algorithm pool')
        for method in ('dense','math'):
            a,b=rows['session_'+method][i],rows['online_'+method][i]
            if a['identity']['context_sha256']==b['identity']['context_sha256']:
                require(a['prediction']==b['prediction'] and a['mapped_score']==b['mapped_score'],'identical contexts were not shared')
    return {'status':'PASS','n':371,'rows':1484,'all_payloads_reconstructed':True,
            'reader_provider':reader_audit,'judge_provider':judge_audit,'reports':completion['reports'],
            'paired_online_minus_session':completion['paired_online_minus_session'],'selection_stats':stats,
            'no_writer_or_online_or_embedding_or_rerank_calls':True,'audit_model_calls':0,
            'scope':'event-view routing only, not original52.83 or event-prose effect'}


def judge_verify():
    sources=audit.load_sources(); old=sources[0]['row_data']; gold=sources[2]; reports={}; comparisons={}; data={}; branches={}
    gm=audit.direct.g35_runtime
    limits={s:gm.StageLimits(2400 if s=='caption' else 16384 if s in ('writer','view') else 400,371*5 if s=='judge' else 0,5) for s in gm.STAGES}
    for rep in ('j1','j2'):
        branch=ROOT/'audit_cache'/rep; phys=physical(branch); finals=phys.pop('_final_by_request'); links=set(); rows={}
        for qid in old:
            r=checked(ROOT/'judge_repeats'/rep/(sha(qid)+'.json'))
            require(r['row_id']==qid and r['fixed_prediction_sha256']==sha(old[qid]['prediction']),'fixed prediction drift')
            prompt=gm.build_mgtm_judge_prompt(gold[qid]['question'],gold[qid]['gold'],old[qid]['prediction'])
            final=audit.read(r['response_path'],'response')
            metadata={**final,'cache_path':r['response_path']}
            response,key=request_check(branch,metadata,prompt,gm,limits,{'purpose':'fixed_answer_judge_repeat','replicate':rep,'row_id':qid})
            links.add(key); verdict=gm.parse_mgtm_judge_response(response['text'])
            require(verdict.mapped_score==r['mapped_score'] and r['original_mapped']==old[qid]['mapped_score'],'Judge score drift')
            rows[qid]=r
        require(links==set(finals) and len(rows)==371,'Judge call/population census')
        data[rep]=rows; branches[rep]=phys
        reports[rep]={'mapped_sum':sum(r['mapped_score'] for r in rows.values()),
            'mapped_percent':100*sum(r['mapped_score'] for r in rows.values())/371,
            'improved':sum(r['mapped_score']>r['original_mapped'] for r in rows.values()),
            'regressed':sum(r['mapped_score']<r['original_mapped'] for r in rows.values())}
    changed=[]
    for qid in old:
        values=[old[qid]['mapped_score'],data['j1'][qid]['mapped_score'],data['j2'][qid]['mapped_score']]
        if len(set(values))>1: changed.append({'row_id':qid,'scores':values,'question':gold[qid]['question'],
            'gold':gold[qid]['gold'],'prediction':old[qid]['prediction']})
    return {'status':'PASS','n':371,'new_judges':742,'original_sum':196,'original_percent':196/371*100,
            'repeats':reports,'any_grade_changed':len(changed),'changed_rows':changed,'provider':branches,
            'systematic_reference_correctness_not_proved_by_repeats':True,'audit_model_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['judge','factorial']); p.add_argument('--write',action='store_true'); args=p.parse_args()
    result=judge_verify() if args.mode=='judge' else factorial_verify()
    if args.write: audit.put(ROOT/(args.mode+'_independent_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

"""Cached diagnostics plus new-run paired analysis. Never writes source data."""
from __future__ import annotations
from collections import defaultdict, Counter
import gzip
from itertools import product
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import spearmanr
from . import run

b, OUT = run.b, run.OUT
SEED, NBOOT = 92601, 10000
INPUTS = {}


def read(path):
    path=Path(path);raw=path.read_bytes();INPUTS[str(path)]=b.sha(raw)
    obj=json.loads(raw)
    if isinstance(obj,dict) and set(obj)=={'payload','sha256'}:
        assert b.sha(obj['payload'])==obj['sha256'];return obj['payload']
    if isinstance(obj,dict) and 'seal_sha256' in obj:
        assert b.sha({k:v for k,v in obj.items() if k!='seal_sha256'})==obj['seal_sha256']
    return obj


def stats(values):
    a=np.array([x for x in values if x is not None and np.isfinite(x)],dtype=float)
    if not len(a):return {'n':0}
    return dict(n=len(a),mean=float(a.mean()),min=float(a.min()),p25=float(np.quantile(a,.25)),
                median=float(np.median(a)),p75=float(np.quantile(a,.75)),p95=float(np.quantile(a,.95)),max=float(a.max()))


def paired(left,right):
    assert set(left)==set(right)
    groups=defaultdict(list)
    for rid in sorted(left):
        a,c=left[rid],right[rid];assert a['task']==c['task']
        groups[a['task']].append((a['mapped_score'],c['mapped_score']))
    names=sorted(groups);count=np.array([len(groups[g]) for g in names]);sums=np.array([np.array(groups[g]).sum(axis=0) for g in names])
    ds=sums[:,0]-sums[:,1];rng=np.random.default_rng(SEED)
    ix=rng.integers(0,len(names),size=(NBOOT,len(names)))
    draws=ds[ix].sum(axis=1)/count[ix].sum(axis=1)*100
    diffs=[left[r]['mapped_score']-right[r]['mapped_score'] for r in sorted(left)]
    lohi=np.quantile(draws,[.025,.975]).tolist()
    signs=np.array(list(product((-1,1),repeat=len(names))))
    return dict(n=int(count.sum()),clusters=len(names),left_percent=float(sums[:,0].sum()/count.sum()*100),
        right_percent=float(sums[:,1].sum()/count.sum()*100),delta_pp=float(ds.sum()/count.sum()*100),
        cluster_bootstrap95_pp=lohi,gain=sum(x>0 for x in diffs),loss=sum(x<0 for x in diffs),tie=sum(x==0 for x in diffs),
        exploratory_signflip_p=float(np.mean(np.abs(signs@ds)>=abs(ds.sum())-1e-12)),
        leave_one_cluster_out_pp=[float((ds.sum()-ds[i])/(count.sum()-count[i])*100) for i in range(len(names))],
        cluster_details=[dict(task=g,n=int(count[i]),left=float(sums[i,0]/count[i]*100),right=float(sums[i,1]/count[i]*100),delta_pp=float(ds[i]/count[i]*100)) for i,g in enumerate(names)])


def cert(s):
    a=s.get('numerical_audit',{})
    return bool(s.get('optimal') and s.get('status')==0 and s.get('feasible_incumbent') and
                a.get('certified_optimal') and a.get('numerical_checks_pass') and not a.get('check_failures'))


def score_rows(arm):
    scores={};predictions={};contexts={}
    for p in (run.OLD/'g35lite/full/tasks').glob('*/scores/g35flash/'+arm+'/*.json'):
        s=read(p);root=p.parents[3];key=p.name
        pred=read(root/'predictions'/arm/key);ctx=read(root/'contexts'/arm/key)
        assert s['prediction_sha256']==pred['seal_sha256'] and pred['context_sha256']==ctx['context_sha256']==b.sha(ctx['context'])
        scores[s['row_id']]=s;predictions[s['row_id']]=pred;contexts[s['row_id']]=ctx
    assert len(scores)==371
    return scores,predictions,contexts


def current_stats():
    scores={};preds={};contexts={}
    for arm in ('enhanced_reader','original_reader','no_reader','session_only'):
        scores[arm],preds[arm],contexts[arm]=score_rows(arm)
    contrasts={}
    for a,c in [('enhanced_reader','no_reader'),('original_reader','no_reader'),('enhanced_reader','original_reader'),('no_reader','session_only')]:
        x=paired(scores[a],scores[c]);x['same_context_count']=sum(contexts[a][k]['context_sha256']==contexts[c][k]['context_sha256'] for k in scores[a])
        x['scope']='same-input QA-repeat variability; NOT Online intervention' if c=='session_only' else ('Reader plus query-selection changes' if a=='enhanced_reader' else 'fixed-fact Reader removal')
        contrasts[a+'_minus_'+c]=x
    return dict(contrasts=contrasts,seed=SEED,nboot=NBOOT,grouping='eight whole scenes, question-weighted mean; frozen developed histories'),scores,preds,contexts


def solver_audit(scores,contexts):
    graphs={};rows=[]
    for rid,c in contexts.items():
        task=c['query']['task']
        if task not in graphs:
            m=read(run.oldroot(task)/'memory_complete.json');g=read(m['session']['artifact_path'])
            assert INPUTS[m['session']['artifact_path']]==m['session']['artifact_sha256'];graphs[task]=g
        g=graphs[task];o=c['optimization'];fs,groups=run.sel.adapter.adapt_facts(g)
        selected=c['selected_fact_ids'];lookup={f['id']:f for f in fs}
        txt,_=run.sel.adapter.render_selection(g,selected)
        assert txt==c['context'] and len(txt)<=c['budget'] and len(selected)<=30
        assert run.sel.adapter.additive_cost(fs,groups,selected)==len(txt)==o['cost']-1
        assert len(set(selected))==len(selected)
        assert len({(tuple(sorted(lookup[f]['sources'])),lookup[f]['text']) for f in selected})==len(selected)
        stages={x['name']:x for x in o['solver_stages']}
        ideals=[stages.get(k+'_ideal',{}) for k in o['channel_ideal_values']]
        exact_ideals=all(cert(s) for s in ideals)
        ratios={k:o['channel_utility'][k]/v for k,v in o['channel_ideal_values'].items() if v>0}
        assert all(abs(v-o['channel_ratios'][k])<1e-6 for k,v in ratios.items())
        coverage=cert(stages['original_query_literal_coverage'])
        minimum=min(ratios.values()) if exact_ideals and coverage and ratios else None
        mm=stages.get('max_min_normalized_channel_utility',{})
        ub=mm.get('maximization_upper_bound')
        rows.append(dict(row_id=rid,task=task,mapped_score=scores[rid]['mapped_score'],
            candidate_count=len(fs),unique_source_text_count=len({(tuple(sorted(f['sources'])),f['text']) for f in fs}),
            selected_count=len(selected),characters=len(txt),character_budget=c['budget'],
            active_channels=len(o['channel_ideal_values']),has_dense=any(k.endswith(':dense') for k in ratios),has_bm25=any(k.endswith(':bm25') for k in ratios),
            coverage_certified=coverage,ideals_certified=exact_ideals,minimax_certified=cert(mm),
            all_stages_certified=o['certified_optimal'],minimum_retention=minimum,
            core_absolute_suboptimality_bound=max(0.,ub-minimum) if minimum is not None and ub is not None else None,
            stage_seconds=sum(s['seconds'] for s in o['solver_stages']),
            terminal_stage=o['solver_stages'][-1]['name'],stage_statuses=[{k:s.get(k) for k in ('name','status','optimal','mip_gap','maximization_upper_bound','objective','seconds')} for s in o['solver_stages']]))
    valid=[r for r in rows if r['minimum_retention'] is not None]
    corr=spearmanr([r['minimum_retention'] for r in valid],[r['mapped_score'] for r in valid])
    # Within-scene correlation is descriptive; no independent-question p value.
    scene_corr={}
    for task in graphs:
        rr=[r for r in valid if r['task']==task]
        rho=spearmanr([r['minimum_retention'] for r in rr],[r['mapped_score'] for r in rr]).statistic
        scene_corr[task]=float(rho) if np.isfinite(rho) else None
    summary=dict(n=len(rows),hard_constraint_failures=0,solver='scipy.optimize.milp / HiGHS; saved finite-precision certificates; 5 seconds per stage; presolve=False',
        certificates={k:sum(r[k] for r in rows) for k in ('coverage_certified','ideals_certified','minimax_certified','all_stages_certified')},
        both_channel_types=sum(r['has_dense'] and r['has_bm25'] for r in rows),
        distributions={k:stats(r[k] for r in rows) for k in ('candidate_count','unique_source_text_count','selected_count','characters','character_budget','active_channels','minimum_retention','stage_seconds','core_absolute_suboptimality_bound')},
        retention_qa_spearman=float(corr.statistic),retention_qa_spearman_by_scene=scene_corr,
        score_by_certificate={str(c):dict(n=sum(r['all_stages_certified']==c for r in rows),mapped_percent=float(np.mean([r['mapped_score'] for r in rows if r['all_stages_certified']==c])*100)) for c in (True,False)},
        partial_terminal_counts=dict(Counter(r['terminal_stage'] for r in rows if not r['all_stages_certified'])),
        interpretation='Certificate groups/correlation are descriptive, NOT a causal solver sensitivity experiment. Retention is rank utility, not gold-evidence recall.')
    return dict(summary=summary,rows=rows)


def token_summary(items):
    return dict(calls=len(items),**{k:sum((x.get('usage',{}).get(k) or 0) for x in items) for k in ('prompt_tokens','output_tokens','thought_tokens','total_tokens')},
                missing_usage=sum(not x.get('usage') for x in items))


def cost_audit(preds):
    query=defaultdict(list)
    for p in preds.values():
        for st in p['stages']:query[st['stage']].append(st['generation'])
    build=defaultdict(list);times=defaultdict(list);calls=[];scenes=[]
    for task in sorted({p['task'] for p in preds.values()}):
        root=run.oldroot(task);m=read(root/'memory_complete.json')
        scenes.append(dict(task=task,sessions=m['sessions'],facts=m['facts'],images=m['images'],
            session_graph_bytes=Path(m['session']['artifact_path']).stat().st_size,
            online_graph_bytes=Path(m['online']['artifact_path']).stat().st_size))
        for path in (root/'calls').glob('*/*/request.json.gz'):
            blob=path.read_bytes();manifest=json.loads(gzip.decompress(blob));stage=manifest['stage']
            if stage not in ('caption','writer','view','planner'):continue
            INPUTS[str(path)]=b.sha(blob);folder=path.parent;physical=list(folder.glob('attempt_*_request.json'))
            response=folder/'response.json'
            entry=dict(task=task,stage=stage,key=folder.name,physical_attempts=len(physical),has_success=response.exists())
            if response.exists():
                response_value=read(response);raw=response_value['response'];u=raw.get('usageMetadata',{})
                usage=dict(prompt_tokens=u.get('promptTokenCount'),output_tokens=u.get('candidatesTokenCount'),thought_tokens=u.get('thoughtsTokenCount'),total_tokens=u.get('totalTokenCount'))
                build[stage].append(dict(usage=usage));entry['usage']=usage
            for intent in physical:
                outcome=intent.with_name(intent.name.replace('_request.json','_outcome.json'))
                if outcome.exists():
                    st,fn=read(intent),read(outcome)
                    if st.get('time') is not None and fn.get('time') is not None:times[stage].append(fn['time']-st['time'])
            calls.append(entry)
    summary=dict(query_stages={k:token_summary(v) for k,v in query.items()},
        construction_and_planner={k:token_summary(v) for k,v in build.items()},
        service_attempt_seconds={k:stats(v) for k,v in times.items()},
        historical_success_requests=len([x for x in calls if x['has_success']]),
        historical_physical_attempts=sum(x['physical_attempts'] for x in calls),
        failed_logical_requests=sum(not x['has_success'] for x in calls),
        sessions=sum(s['sessions'] for s in scenes),facts=sum(s['facts'] for s in scenes),
        note='Logical deployment ledger: shared build and projection once, enhanced reading once. Excludes v8/other ablations/Judges. Construction rejected outputs are included if request succeeded. Embedding latency/token billing and failed-request billing are unclosed; timings are service durations, not parallel end-to-end latency.',
        scene_storage=scenes)
    qin=sum(v['prompt_tokens'] for v in summary['query_stages'].values());qout=sum(v['output_tokens'] for v in summary['query_stages'].values())
    bin_=sum(v['prompt_tokens'] for v in summary['construction_and_planner'].values());bout=sum(v['output_tokens'] for v in summary['construction_and_planner'].values())
    summary['known_llm_input_per_question']=(qin+bin_)/371;summary['known_llm_output_per_question']=(qout+bout)/371
    if 'view' in summary['construction_and_planner']:
        v=summary['construction_and_planner']['view'];summary['online_input_per_session']=v['prompt_tokens']/summary['sessions'];summary['online_calls_per_session']=v['calls']/summary['sessions']
    return dict(summary=summary,request_ledger=calls)


def event_pack():
    inventory=[];samples=[];ops=[]
    for task in sorted({q['task'] for q in read(run.OLD/'registration.json')['questions']}):
        m=read(run.oldroot(task)/'memory_complete.json');g=read(m['online']['artifact_path']);fs={f['id']:f for f in g['fact_nodes']}
        views=[v for v in g['summary_nodes'] if v['node_type']=='event_view'];steps=g['online_fusion']['steps']
        inventory.append(dict(task=task,views=len(views),claims=sum(len(v.get('summary_claims',[])) for v in views),
            frontiers=len(steps),called_frontiers=sum(s['model_call_made'] for s in steps),accepted_operations=sum(len(s.get('accepted',[])) for s in steps),rejected_operations=sum(len(s.get('rejected',[])) for s in steps),
            source_fact_identity_equal=g['fact_nodes']==read(m['session']['artifact_path'])['fact_nodes']))
        for v in sorted(views,key=lambda v:b.sha('92601:'+task+':'+v['id']))[:8]:
            members=v['member_fact_ids'];claims=v.get('summary_claims',[])
            assert set(members)<=fs.keys()
            assert all(set(c['support_fact_ids'])<=set(members) for c in claims)
            samples.append(dict(task=task,view_id=v['id'],view_text=v['text'],claims=claims,
                members=[{k:fs[f].get(k) for k in ('id','text','session_id','source_dia_ids')} for f in members],
                annotation=dict(same_event_or_process=None,false_merge=None,unsupported_claim_indices=None,
                    contradicted_or_superseded_claim_indices=None,notes=None),
                labels_status='UNANNOTATED; no semantic accuracy claimed'))
        for s in steps:
            for op in s.get('accepted',[]):ops.append(dict(task=task,session_id=s['session_id'],operation=op))
    return dict(inventory=inventory,samples=samples,accepted_operations=ops,
                sample_rule='up to eight hashed-ID views per scene; label-blind; stratified descriptive sample, not weighted population estimate',
                semantic_precision=None,missed_link_recall=None,annotation_required=True)


def historical_locomo():
    p=Path(r'D:\creamem3_cache\paper_review5_20260917\round1\paired_analysis_complete.json')
    old=read(p);out={}
    for name,v in old['comparisons'].items():
        if not name.startswith('locomo_'):continue
        left={r['id']:dict(task=r['cluster'],mapped_score=r['left']) for r in v['rows']}
        right={r['id']:dict(task=r['cluster'],mapped_score=r['right']) for r in v['rows']}
        out[name]=paired(left,right)
    return dict(comparisons=out,source=str(p),scope='historical narrow-rerank5 ablations, NOT main-table 81.75 or newer math20; complete frozen paired ledger reused')


def static():
    out=OUT/'audit_static'
    ci,scores,preds,contexts=current_stats()
    b.save(out/'paired_current.json',ci)
    solver=solver_audit(scores['enhanced_reader'],contexts['enhanced_reader']);b.save(out/'solver_current30.json',solver)
    resources=cost_audit(preds['enhanced_reader']);b.save(out/'cost_current.json',resources)
    events=event_pack();b.save(out/'event_annotation_pack.json',events)
    b.save(out/'locomo_historical_cluster_reanalysis.json',historical_locomo())
    b.save(out/'provenance.json',dict(inputs=INPUTS,code_sha256=b.file_sha(__file__),model_calls=0))
    print(json.dumps({'CI':ci,'solver':solver['summary'],'cost':resources['summary'],'event_inventory':events['inventory'],'sample_count':len(events['samples'])},ensure_ascii=True,indent=2))


def final():
    reg=run.validate();scores={};preds={};missing={};usages={}
    for arm in run.ARMS:
        scores[arm]={};preds[arm]={};missing[arm]=[];stages=defaultdict(list)
        for ref in reg['rows']:
            key=b.sha(ref['row_id']);root=OUT/'runs'/arm
            prepared=read(ref['path'])
            if arm in ('full_refresh','dense30') and prepared['route']=='structured_fused':
                # The original supplement's unescaped-template branch is invalid,
                # even when its answer happened to be correct. Never score-pick.
                root=OUT/'fused_prompt_repair/runs'/arm
            p=root/'predictions'/f'{key}.json';s=root/'scores'/f'{key}.json'
            if not s.exists():missing[arm].append(ref['row_id']);continue
            pred=read(p);score=read(s);assert score['prediction_sha256']==pred['seal_sha256']
            scores[arm][ref['row_id']]=score;preds[arm][ref['row_id']]=pred
            for stage in pred['stages']:stages[stage['stage']].append(stage['generation'])
        usages[arm]={k:token_summary(v) for k,v in stages.items()}
    result=dict(n_by_arm={a:len(x) for a,x in scores.items()},missing=missing,query_costs=usages,comparisons={})
    if all(not v for v in missing.values()):
        for control in ('dense30','generic_reader'):result['comparisons']['full_refresh_minus_'+control]=paired(scores['full_refresh'],scores[control])
        old,_,_=score_rows('enhanced_reader');result['comparisons']['full_refresh_minus_cached_full']=paired(scores['full_refresh'],old)
        result['scores']={a:float(np.mean([r['mapped_score'] for r in rows.values()])*100) for a,rows in scores.items()}
        # Validate input/call-count matching before any attribution.
        result['generic_same_input']=sum(preds['full_refresh'][r]['context_sha256']==preds['generic_reader'][r]['context_sha256'] for r in old)
        result['generic_same_call_count']=sum(len(preds['full_refresh'][r]['stages'])==len(preds['generic_reader'][r]['stages']) for r in old)
        assert result['generic_same_input']==result['generic_same_call_count']==371
    b.save(OUT/'analysis_snapshots'/f'{time.time_ns()}.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='static':static()
    else:final()

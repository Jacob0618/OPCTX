"""Frozen-memory Online prefilter/reranker control, isolated write-once root."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
from pathlib import Path
import statistics
import time

from . import core
from creamem3.experiments.aligned_selector_ablations_20260913 import run as aligned
from creamem3.experiments.component_ablations_20260911 import runtime_v2 as rt

p,previous,source=aligned.p,aligned.previous,aligned.source
transport,component=aligned.transport,aligned.component
read,seal,sha,file_sha,StopRun=aligned.read,aligned.seal,aligned.sha,aligned.file_sha,aligned.StopRun
ROOT=Path(r'D:\creamem3_cache\locomo_prefilter50_rerank5_full1540_20260913_r1')
BASE=previous.BASE
ARM='opctx_original'
MODES=core.MODES
CONFIG=dict(summary_top_k=10,dense_fact_top_k=20,bm25_fact_top_k=20,candidate_cap=120,rerank_top_k=5)
SOURCES=(BASE/'conditions/wide/calls',p.RUN/'conditions/wide/calls')
ALLOWED={k:transport.MODELS[k] for k in ('rerank','qa','judge')}
WORKERS=12


class Runtime(rt.Runtime):
    def post(self,stage,scope,model,method,body):
        if ALLOWED.get(stage)!=model: self._halt('Only rerank/QA/Judge allowed')
        identity=dict(stage=stage,scope=scope,model=model,method=method,body=body)
        if getattr(self.local,'cached_only',False):
            self.import_exact(identity)
            if not (self.root/'provider'/stage/sha(identity)/'final.json').exists():
                self._halt('Missing frozen reference/audit response; no fresh reference call')
        response=super().post(stage,scope,model,method,body)
        if stage in {'qa','rerank'} and rt.original.finish_reason(response)!='STOP':
            self._halt('Incomplete '+stage+' response')
        return response


def prepare():
    loader=aligned.load_private(source.__package__+'._prefilter50_inputs',source.__file__,source.__package__)
    captured={}
    loader.seal=lambda path,value:captured.__setitem__(Path(path).name,copy.deepcopy(value))
    data=loader.prepare()
    reg0,done=read(BASE/'registration.json'),read(BASE/'complete.json')
    if not done['audit_pass'] or done['smoke_only'] or done['registration_sha256']!=sha(reg0):
        raise StopRun('Wide reference incomplete')
    frozen=captured['source_snapshot.json']
    for name in ('registration.json','complete.json'): frozen[str(BASE/name)]=file_sha(BASE/name)
    hashes={**data['registration']['source_hashes'],**reg0['source_hashes']}
    for path,digest in hashes.items():
        if file_sha(path)!=digest: raise StopRun('Frozen code drift '+path)
    data['wide_indexes']={s:p.source.enum_repair.Index(data['graphs'][(s,ARM)],**CONFIG) for s in sorted(set(data['samples'].values()))}
    refs={}; pools={}; channels={}; counts=[]
    for qid,q in sorted(data['questions'].items()):
        s=data['samples'][qid]; index=data['wide_indexes'][s]
        paths={k:BASE/'conditions/wide'/k/ARM/(qid+'.json') for k in ('rows','retrieval')}
        row,ret=read(paths['rows']),read(paths['retrieval'])
        expected=dict(registration_sha256=sha(reg0),qa_sha256=sha(q),graph_sha256=data['graph_hashes'][(s,ARM)],
            vector_sha256=sha(data['vectors'][qid]),condition='wide',arm=ARM)
        if row['identity']!=expected or ret['identity']!=expected: raise StopRun('Reference identity drift')
        candidates,trace=index._candidate_pool(q['question'],data['vectors'][qid])
        if trace!={k:ret['trace'][k] for k in trace}: raise StopRun('Historical wide pool changed')
        p.validate_selection(row['selected_fact_ids'],candidates)
        context,audit=p.render(index,candidates,ret['trace'],data['graphs'][(s,ARM)],q,ARM,data['operations'][qid])
        if context!=row['context'] or context!=ret['context'] or audit!=ret['materializer'] or sha(context)!=row['context_sha256']:
            raise StopRun('Historical wide renderer changed')
        channel=core.old.build_channels(index,candidates,q['question'],data['banks'][qid])
        feasible=core.choose(candidates,data['graphs'][(s,'session_strong')],q['question'],channel,'dense50')
        if feasible['rerank_input_count']!=50: raise StopRun('Exact50 infeasible')
        refs[qid]=row; pools[qid]=trace; channels[qid]=channel; counts.append(len(candidates))
        frozen.update({str(path):file_sha(path) for path in paths.values()})
    if len(refs)!=1540 or sum(r['correct'] for r in refs.values())!=1242: raise StopRun('Wide source population drift')
    own=[Path(__file__),Path(core.__file__),Path(__file__).with_name('protocol.md'),Path(__file__).with_name('test_core.py')]
    reg=dict(schema='locomo.prefilter50.rerank5.v1',n=1540,total_rows=4620,arms=[ARM],modes=list(MODES),config=CONFIG,
        all_ids=sorted(data['questions']),source_hashes={**hashes,**{str(x):file_sha(x) for x in own}},
        source_snapshot_sha256=sha(frozen),models=ALLOWED,workers=WORKERS,wire_concurrency=8,
        rerank_input_counts={'full_pool':'complete capped120 pool','dense50':50,'math50':50},final_facts=5,
        count_matched=True,token_matched=False,character_cap=None,query_bank_cached=True,
        exact_request_sources=[str(x) for x in SOURCES],full_pool_cached_only=True,judge_votes=1,
        qa_prompt_sha256=sha(p.old.QA_PROMPT),judge_prompt_sha256=sha(p.old.JUDGE_PROMPT),
        rerank_prompt_sha256=sha(p.old.graph_api.FACT_RERANK_PROMPT))
    seal(ROOT/'registration.json',reg); seal(ROOT/'source_snapshot.json',frozen)
    seal(ROOT/'questions_score_free.json',captured['questions_score_free.json'])
    seal(ROOT/'preflight.json',dict(pass_=True,n=1540,registration_sha256=sha(reg),all_exact50_feasible=True,
        mean_candidates=statistics.mean(counts),min_candidates=min(counts),max_candidates=max(counts),new_model_calls=0))
    data.update(registration=reg,wide_references=refs,candidate_traces=pools,channels=channels)
    p.old.emit('PREFILTER50 PREFLIGHT PASS all1540, three Online conditions, exact50 feasible')
    return data


def identity(data,mode,qid):
    s=data['samples'][qid]
    return dict(registration_sha256=sha(data['registration']),qa_sha256=sha(data['questions'][qid]),
        graph_sha256=data['graph_hashes'][(s,ARM)],vector_sha256=sha(data['vectors'][qid]),condition=mode,arm=ARM)


def selection(data,mode,qid):
    path=ROOT/'selection'/mode/(qid+'.json'); s=data['samples'][qid]
    binding=dict(row_identity=identity(data,mode,qid),bank_sha256=sha(data['banks'][qid]))
    if path.exists(): saved=read(path)
    else:
        saved=dict(identity=binding,**core.select(data['wide_indexes'][s],data['graphs'][(s,'session_strong')],
            data['questions'][qid]['question'],data['banks'][qid],mode))
        seal(path,saved)
    if saved['identity']!=binding or saved['mode']!=mode or saved['candidate_trace']!=data['candidate_traces'][qid] or saved['channels']!=data['channels'][qid]:
        raise StopRun('Prefilter binding/universe drift')
    return saved


def retrieve(data,mode,qid,runtime):
    saved=selection(data,mode,qid); s=data['samples'][qid]; q=data['questions'][qid]
    idx=data['wide_indexes'][s]; pool,trace=idx._candidate_pool(q['question'],data['vectors'][qid])
    ids=saved['prefilter_fact_ids']; by_id={c['fact']['id']:c for c in pool}
    candidates=[by_id[i] for i in ids]
    picked,detail=idx._rerank(q['question'],candidates,runtime.chat_fn(ARM+'/'+qid))
    chosen=[c['fact']['id'] for c in picked]; p.validate_selection(chosen,candidates)
    if detail['fallback']: raise StopRun('Rerank fallback forbidden')
    _,assembly=idx._assemble(picked,trace['top_summaries'])
    trace.update(rerank=dict(**detail,selected_fact_ids=chosen),assembly=assembly)
    context,audit=p.render(idx,pool,trace,data['graphs'][(s,ARM)],q,ARM,data['operations'][qid])
    return dict(identity=identity(data,mode,qid),selection_sha256=sha(saved),context=context,trace=trace,
        materializer=audit,prefilter_fact_ids=ids,rerank_input_count=len(ids))


def evaluate(data,mode,qid,runtime):
    path=ROOT/'rows'/mode/(qid+'.json'); binding=identity(data,mode,qid)
    if path.exists():
        row=read(path)
        if row['identity']!=binding: raise StopRun('Resume row drift')
        return row
    runtime.local.requests=[]; runtime.local.cached_only=mode=='full_pool'
    t=time.perf_counter(); q=data['questions'][qid]
    ret=retrieve(data,mode,qid,runtime); seal(ROOT/'retrieval'/mode/(qid+'.json'),ret)
    answer=runtime.chat(qid,'qa',p.old.QA_PROMPT.format(question=q['question'],retrieved_texts=ret['context']),max_tokens=8192)
    correct,judge=p.old.judge(runtime,qid,q,answer)
    receipts=p.unique_requests(runtime.local.requests)
    qa=[r for r in receipts if r['stage']=='qa']; rr=[r for r in receipts if r['stage']=='rerank']
    if len(qa)!=1 or not rr or any(r['stage'] not in ALLOWED for r in receipts): raise StopRun('Logical stage drift')
    row=dict(identity=binding,qa_id=qid,sample_id=data['samples'][qid],arm=ARM,condition=mode,
        question=q['question'],answer=q['answer'],category_name=q['category_name'],response=answer,correct=correct,judge_raw=judge,
        context=ret['context'],context_sha256=sha(ret['context']),context_chars=len(ret['context']),
        selected_fact_ids=ret['trace']['rerank']['selected_fact_ids'],candidate_count=len(selection(data,mode,qid)['candidate_fact_ids']),
        prefilter_fact_ids=ret['prefilter_fact_ids'],rerank_input_count=ret['rerank_input_count'],
        selection_sha256=ret['selection_sha256'],qa_prompt_tokens=qa[0]['usage'].get('promptTokenCount'),
        rerank_prompt_tokens=sum(r['usage'].get('promptTokenCount',0) for r in rr),
        rerank_output_tokens=sum(r['usage'].get('candidatesTokenCount',0) for r in rr),
        logical_request_receipts=receipts,elapsed_s=time.perf_counter()-t,reference_reused=mode=='full_pool')
    if mode=='full_pool':
        old=data['wide_references'][qid]
        if any(row[k]!=old[k] for k in ('response','correct','judge_raw','context','selected_fact_ids')):
            raise StopRun('Full reference did not reproduce exact frozen outcome')
    seal(path,row)
    p.old.emit(f'PREFILTER50 ROW {mode}/{qid} correct={correct} rerank_input={ret["rerank_input_count"]}')
    return row


def question_job(data,qid,runtime):
    try:
        if runtime.stopped.is_set(): raise StopRun('Dispatch stopped')
        for mode in MODES: selection(data,mode,qid)
        for mode in MODES: evaluate(data,mode,qid,runtime)
    except BaseException as exc:
        runtime.record_failure(exc); raise


def finish(data,ids,smoke=False):
    replay=Runtime(ROOT/'calls',control=transport.ConnectionControl(live=False),allow_model_calls=False,sources=())
    replay.local.cached_only=True
    rows={m:{} for m in MODES}; stats={}
    for mode in MODES:
        if not smoke and {p.stem for p in (ROOT/'rows'/mode).glob('*.json')}!=set(ids): raise StopRun('Incomplete population')
        for qid in ids:
            saved=selection(data,mode,qid); row=read(ROOT/'rows'/mode/(qid+'.json'))
            ret=read(ROOT/'retrieval'/mode/(qid+'.json')); actual=retrieve(data,mode,qid,replay)
            if actual!=ret or row['identity']!=identity(data,mode,qid) or row['context']!=ret['context'] or row['context_sha256']!=sha(ret['context']) or row['selection_sha256']!=sha(saved) or row['selected_fact_ids']!=ret['trace']['rerank']['selected_fact_ids']:
                raise StopRun('Final rerank/context binding drift')
            if row['prefilter_fact_ids']!=saved['prefilter_fact_ids'] or row['rerank_input_count']!=len(saved['prefilter_fact_ids']): raise StopRun('Row prefilter drift')
            s=data['samples'][qid]; idx=data['wide_indexes'][s]
            candidates,_=idx._candidate_pool(data['questions'][qid]['question'],data['vectors'][qid])
            feasible=core.choose(candidates,data['graphs'][(s,'session_strong')],data['questions'][qid]['question'],saved['channels'],'dense50')
            if mode in {'dense50','full_pool'}:
                wanted=feasible['prefilter_fact_ids'] if mode=='dense50' else [c['fact']['id'] for c in candidates]
                if saved['prefilter_fact_ids']!=wanted: raise StopRun('Deterministic prefilter drift')
            else:
                opt=saved['optimization']; wanted=set(opt['selected_ids'])
                if len(wanted)!=50 or set(saved['prefilter_fact_ids'])!=wanted or not opt['coverage_first'] or not opt['max_min_balance'] or opt['min_units']!=50 or opt['max_units']!=50 or opt['char_budget']!=feasible['nonbinding_character_bound']:
                    raise StopRun('Mathematical count/objective drift')
            p.base.check_bound(ROOT,data['questions'][qid],row)
            for rec in row['logical_request_receipts']:
                _,_,_,raw=p.full.recover.completed_receipt(ROOT/'calls/provider'/rec['stage']/rec['key'])
                if raw['response'].get('usageMetadata',{})!=rec['usage']: raise StopRun('Usage binding drift')
            rows[mode][qid]=row
        selected=[selection(data,mode,q) for q in ids]
        stats[mode]=dict(mean_seconds=statistics.mean(s['optimization']['seconds'] for s in selected),
            certified_optimal=sum(s['optimization'].get('certified_optimal') is True for s in selected) if mode=='math50' else None,
            rerank_input_counts=dict(Counter(s['rerank_input_count'] for s in selected)),
            mean_old_rerank5_retained=statistics.mean(len(set(s['prefilter_fact_ids'])&set(data['wide_references'][q]['selected_fact_ids'])) for s,q in zip(selected,ids)),
            old_rerank5_retention_is_not_gold_recall=True)
        p.old.emit(f'PREFILTER50 AUDIT {mode} {len(ids)}')
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Registered code drift '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path)!=digest: raise StopRun('Frozen input drift '+path)
        for qid in ids: previous.verify_bank(data,qid)
    audit=component.provider_audit(ROOT/'calls')
    if set(audit['stages'])-set(ALLOWED): raise StopRun('Forbidden fresh stage')
    reports={m:component.summary(rows[m]) for m in MODES}
    costs={m:{'mean_'+k:statistics.mean(r[k] for r in rows[m].values()) for k in ('rerank_input_count','rerank_prompt_tokens','rerank_output_tokens')} for m in MODES}
    result=dict(audit_pass=True,smoke_only=smoke,n=len(ids),total_rows=len(ids)*3,registration_sha256=sha(data['registration']),
        reports=reports,read_costs=costs,selection_stats=stats,paired_math_minus_dense=p.paired(rows['math50'],rows['dense50'],ids),
        paired_vs_full={m:p.paired(rows[m],rows['full_pool'],ids) for m in ('dense50','math50')},physical_transport=audit,
        token_matched=False,character_cap=None,projection_cost_source=str(previous.ROOT/'query_bank'),
        no_new_projection_embedding_writer_online_calls=True,full_pool_is_frozen_reference=True)
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'),result)
    p.old.emit('PREFILTER50 '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run']); args=parser.parse_args()
    with p.old.Lease(ROOT):
        data=prepare()
        if args.command=='prepare': return
        from dotenv import load_dotenv
        load_dotenv(p.old.ROOT/'.env',override=False)
        runtime=Runtime(ROOT/'calls',control=transport.ConnectionControl(live=True),allow_model_calls=True,sources=SOURCES)
        try:
            bycat={}
            for qid,q in sorted(data['questions'].items()): bycat.setdefault(q['category_name'],qid)
            ids=sorted(set(p.smoke_ids(data))|set(bycat.values()))
            if not (ROOT/'smoke_complete.json').exists():
                for qid in ids: question_job(data,qid,runtime)
                finish(data,ids,smoke=True)
            else:
                smoke=read(ROOT/'smoke_complete.json')
                if not smoke['audit_pass'] or smoke['registration_sha256']!=sha(data['registration']): raise StopRun('Smoke drift')
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs=[pool.submit(question_job,data,qid,runtime) for qid in sorted(data['questions'])]
                for job in as_completed(jobs): job.result()
            finish(data,sorted(data['questions']))
        except BaseException as exc:
            runtime.record_failure(exc)
            seal(ROOT/f'stop_{time.time_ns()}.json',dict(utc=p.base.now(),type=type(exc).__name__,message=str(exc)[:600]))
            raise


if __name__=='__main__': main()

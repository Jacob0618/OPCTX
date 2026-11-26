"""Six fixed Online arms, with fresh-reference and controlled-cache separation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
from pathlib import Path
import statistics
import time

from . import core
from creamem3.experiments.locomo_prefilter50_rerank5_20260913 import run as prior

p,previous,source,aligned=prior.p,prior.previous,prior.source,prior.aligned
transport,component=prior.transport,prior.component
read,seal,sha,file_sha,StopRun=prior.read,prior.seal,prior.sha,prior.file_sha,prior.StopRun
ROOT=Path(r'D:\creamem3_cache\locomo_narrow_rerank_ablations_full1540_20260913_r1')
BASE=prior.BASE
ARM=prior.ARM
MODES=core.MODES
CONFIG=dict(summary_top_k=2,dense_fact_top_k=5,bm25_fact_top_k=5,candidate_cap=120,rerank_top_k=5)
SOURCES=(BASE/'conditions/narrow/calls',p.RUN/'conditions/narrow/calls')
WORKERS=12
ALLOWED=prior.ALLOWED


class FreshRuntime(prior.Runtime):
    def __init__(self,*args,**kwargs):
        if kwargs.get('sources'): raise ValueError('Fresh rerun forbids imported responses')
        kwargs['sources']=()
        super().__init__(*args,**kwargs)

    def import_exact(self,identity):
        if self.sources: self._halt('Repeat import-source drift')
        return super().import_exact(identity)


def call_root(mode): return ROOT/('fresh_repeat' if mode=='rerank_repeat' else 'shared')


def prepare():
    loader=aligned.load_private(source.__package__+'._narrow_ablations_inputs',source.__file__,source.__package__)
    captured={}; loader.seal=lambda path,value:captured.__setitem__(Path(path).name,copy.deepcopy(value))
    data=loader.prepare(); reg0=read(BASE/'registration.json'); done=read(BASE/'complete.json')
    if not done['audit_pass'] or done['smoke_only'] or done['registration_sha256']!=sha(reg0): raise StopRun('Reference not complete')
    frozen=captured['source_snapshot.json']; hashes={**data['registration']['source_hashes'],**reg0['source_hashes']}
    for path,digest in hashes.items():
        if file_sha(path)!=digest: raise StopRun('Frozen code drift '+path)
    for name in ('registration.json','complete.json'): frozen[str(BASE/name)]=file_sha(BASE/name)
    indexes={s:p.source.enum_repair.Index(data['graphs'][(s,ARM)],**CONFIG) for s in sorted(set(data['samples'].values()))}
    refs={}; pools={}; channels={}; affected=[]
    for qid,q in sorted(data['questions'].items()):
        s=data['samples'][qid]; idx=indexes[s]
        paths={k:BASE/'conditions/narrow'/k/ARM/(qid+'.json') for k in ('rows','retrieval')}
        row,ret=read(paths['rows']),read(paths['retrieval'])
        expected=dict(registration_sha256=sha(reg0),qa_sha256=sha(q),graph_sha256=data['graph_hashes'][(s,ARM)],
            vector_sha256=sha(data['vectors'][qid]),condition='narrow',arm=ARM)
        if row['identity']!=expected or ret['identity']!=expected: raise StopRun('Historical identity drift')
        candidates,trace=idx._candidate_pool(q['question'],data['vectors'][qid])
        if trace!={k:ret['trace'][k] for k in trace}: raise StopRun('Narrow pool drift')
        context,audit=p.render(idx,candidates,ret['trace'],data['graphs'][(s,ARM)],q,ARM,data['operations'][qid])
        if context!=row['context'] or context!=ret['context'] or audit!=ret['materializer'] or sha(context)!=row['context_sha256']:
            raise StopRun('Narrow rendering drift')
        p.validate_selection(row['selected_fact_ids'],candidates)
        ch=core.old.build_channels(idx,candidates,q['question'],data['banks'][qid])
        core.choose(candidates,data['graphs'][(s,'session_strong')],q['question'],ch,'dense50')
        if len(candidates)>50: affected.append(qid)
        refs[qid]=row; pools[qid]=trace; channels[qid]=ch
        frozen.update({str(path):file_sha(path) for path in paths.values()})
    if len(refs)!=1540 or len(affected)!=185 or sum(r['correct'] for r in refs.values())!=1217: raise StopRun('Population drift')
    own=[Path(__file__),Path(core.__file__),Path(prior.__file__),Path(core.parent.__file__),Path(__file__).with_name('protocol.md'),Path(__file__).with_name('test_core.py')]
    # prior.Runtime has frozen helper imports, but no live prior result is consumed.
    reg=dict(schema='locomo.narrow.rerank.ablations.v1',n=1540,total_rows=9240,arms=[ARM],modes=list(MODES),config=CONFIG,
        all_ids=sorted(refs),affected185=affected,source_hashes={**hashes,**{str(x):file_sha(x) for x in own}},
        source_snapshot_sha256=sha(frozen),models=ALLOWED,workers=WORKERS,wire_concurrency=8,final_facts=5,
        prefilter_max_facts=50,short_pool='unchanged pass-through',character_cap=None,token_matched=False,
        exact_request_sources=[str(x) for x in SOURCES],fresh_repeat_imports=False,judge_votes=1,
        qa_prompt_sha256=sha(p.old.QA_PROMPT),judge_prompt_sha256=sha(p.old.JUDGE_PROMPT),
        rerank_prompt_sha256=sha(p.old.graph_api.FACT_RERANK_PROMPT))
    seal(ROOT/'registration.json',reg); seal(ROOT/'source_snapshot.json',frozen)
    seal(ROOT/'questions_score_free.json',captured['questions_score_free.json'])
    seal(ROOT/'preflight.json',dict(pass_=True,n=1540,affected185=len(affected),bypass1355=1540-len(affected),
        registration_sha256=sha(reg),new_model_calls=0))
    data.update(registration=reg,narrow_references=refs,narrow_indexes=indexes,candidate_traces=pools,channels=channels)
    p.old.emit('NARROW ABLATIONS PREFLIGHT PASS 6x1540,185 prefilter-active questions')
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
        saved=dict(identity=binding,**core.select(data['narrow_indexes'][s],data['graphs'][(s,'session_strong')],
            data['questions'][qid]['question'],data['banks'][qid],mode)); seal(path,saved)
    if saved['identity']!=binding or saved['mode']!=mode or saved['candidate_trace']!=data['candidate_traces'][qid] or saved['channels']!=data['channels'][qid]:
        raise StopRun('Prefilter identity/universe drift')
    return saved


def retrieve(data,mode,qid,runtime):
    sel=selection(data,mode,qid); s=data['samples'][qid]; q=data['questions'][qid]; idx=data['narrow_indexes'][s]
    pool,trace=idx._candidate_pool(q['question'],data['vectors'][qid]); by_id={c['fact']['id']:c for c in pool}
    candidates=[by_id[i] for i in sel['prefilter_fact_ids']]
    picked,detail=idx._rerank(q['question'],candidates,runtime.chat_fn(ARM+'/'+qid))
    ids=[c['fact']['id'] for c in picked]; p.validate_selection(ids,candidates)
    if detail['fallback']: raise StopRun('Rerank fallback forbidden')
    _,assembly=idx._assemble(picked,trace['top_summaries']); trace.update(rerank=dict(**detail,selected_fact_ids=ids),assembly=assembly)
    context,audit=p.render(idx,pool,trace,data['graphs'][(s,ARM)],q,ARM,data['operations'][qid])
    return dict(identity=identity(data,mode,qid),selection_sha256=sha(sel),context=context,trace=trace,
        materializer=audit,prefilter_fact_ids=sel['prefilter_fact_ids'],rerank_input_count=len(candidates))


def evaluate(data,mode,qid,runtime):
    path=ROOT/'rows'/mode/(qid+'.json'); binding=identity(data,mode,qid)
    if path.exists():
        row=read(path)
        if row['identity']!=binding: raise StopRun('Resume row drift')
        return row
    runtime.local.requests=[]; start=time.perf_counter(); q=data['questions'][qid]
    ret=retrieve(data,mode,qid,runtime); seal(ROOT/'retrieval'/mode/(qid+'.json'),ret)
    answer=runtime.chat(qid,'qa',p.old.QA_PROMPT.format(question=q['question'],retrieved_texts=ret['context']),max_tokens=8192)
    correct,judge=p.old.judge(runtime,qid,q,answer); receipts=p.unique_requests(runtime.local.requests)
    qa=[r for r in receipts if r['stage']=='qa']; rr=[r for r in receipts if r['stage']=='rerank']
    if len(qa)!=1 or not rr or any(r['stage'] not in ALLOWED for r in receipts): raise StopRun('Read stages drift')
    row=dict(identity=binding,qa_id=qid,sample_id=data['samples'][qid],arm=ARM,condition=mode,
        question=q['question'],answer=q['answer'],category_name=q['category_name'],response=answer,correct=correct,judge_raw=judge,
        context=ret['context'],context_sha256=sha(ret['context']),context_chars=len(ret['context']),
        selected_fact_ids=ret['trace']['rerank']['selected_fact_ids'],prefilter_fact_ids=ret['prefilter_fact_ids'],
        candidate_count=len(selection(data,mode,qid)['candidate_fact_ids']),rerank_input_count=ret['rerank_input_count'],
        selection_sha256=ret['selection_sha256'],qa_prompt_tokens=qa[0]['usage'].get('promptTokenCount'),
        rerank_prompt_tokens=sum(r['usage'].get('promptTokenCount',0) for r in rr),
        rerank_output_tokens=sum(r['usage'].get('candidatesTokenCount',0) for r in rr),
        logical_request_receipts=receipts,provider_root=str(call_root(mode)/'calls'),elapsed_s=time.perf_counter()-start)
    if mode!='rerank_repeat' and row['candidate_count']<=50:
        old=data['narrow_references'][qid]
        if any(row[k]!=old[k] for k in ('response','correct','judge_raw','context','selected_fact_ids')):
            raise StopRun('Bypass must reproduce exact frozen source outcome')
    seal(path,row); p.old.emit(f'NARROW ROW {mode}/{qid} correct={correct} rerank_input={ret["rerank_input_count"]}')
    return row


def question_job(data,qid,runtimes):
    try:
        if runtimes['shared'].stopped.is_set(): raise StopRun('Dispatch stopped')
        for mode in MODES: selection(data,mode,qid)
        for mode in MODES: evaluate(data,mode,qid,runtimes['fresh_repeat' if mode=='rerank_repeat' else 'shared'])
    except BaseException as exc:
        runtimes['shared'].record_failure(exc); raise


def finish(data,ids,smoke=False):
    rows={m:{} for m in MODES}; stats={}; replays={k:prior.Runtime(ROOT/k/'calls',
        control=transport.ConnectionControl(live=False),allow_model_calls=False,sources=()) for k in ('shared','fresh_repeat')}
    for r in replays.values(): r.local.cached_only=True
    for mode in MODES:
        if not smoke and {p.stem for p in (ROOT/'rows'/mode).glob('*.json')}!=set(ids): raise StopRun('Incomplete population')
        opts=[]; bypass=0
        for qid in ids:
            saved=selection(data,mode,qid); row=read(ROOT/'rows'/mode/(qid+'.json')); ret=read(ROOT/'retrieval'/mode/(qid+'.json'))
            actual=retrieve(data,mode,qid,replays['fresh_repeat' if mode=='rerank_repeat' else 'shared'])
            if actual!=ret or row['identity']!=identity(data,mode,qid) or row['context']!=ret['context'] or sha(row['context'])!=row['context_sha256'] or row['selection_sha256']!=sha(saved) or row['selected_fact_ids']!=ret['trace']['rerank']['selected_fact_ids']:
                raise StopRun('Final rerank/context binding drift')
            if row['provider_root']!=str(call_root(mode)/'calls') or row['prefilter_fact_ids']!=saved['prefilter_fact_ids'] or row['rerank_input_count']!=len(saved['prefilter_fact_ids']): raise StopRun('Cache namespace/selection drift')
            s=data['samples'][qid]; idx=data['narrow_indexes'][s]
            candidates,_=idx._candidate_pool(data['questions'][qid]['question'],data['vectors'][qid])
            core.verify(saved,candidates,data['graphs'][(s,'session_strong')],data['questions'][qid]['question'],data['channels'][qid])
            p.base.check_bound(call_root(mode),data['questions'][qid],row)
            for rec in row['logical_request_receipts']:
                _,_,_,raw=p.full.recover.completed_receipt(call_root(mode)/'calls/provider'/rec['stage']/rec['key'])
                if raw['response'].get('usageMetadata',{})!=rec['usage']: raise StopRun('Usage drift')
            rows[mode][qid]=row; opts.append(saved['optimization']); bypass+=saved['bypass']
        stats[mode]=dict(bypass_count=bypass,mean_selector_seconds=statistics.mean(o['seconds'] for o in opts),
            certified_optimal=sum(o.get('certified_optimal') is True for o in opts),
            termination_counts=dict(Counter(o['termination'] for o in opts)))
        p.old.emit('NARROW AUDIT '+mode+' '+str(len(ids)))
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Code drift '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path)!=digest: raise StopRun('Source drift '+path)
        for qid in ids: previous.verify_bank(data,qid)
    physical={k:component.provider_audit(ROOT/k/'calls') for k in ('shared','fresh_repeat')}
    if physical['fresh_repeat']['counts'].get('imported_physical_attempts',0) or list((ROOT/'fresh_repeat/calls/imports').rglob('*.json')):
        raise StopRun('Independent repeat imported history')
    if any(set(a['stages'])-set(ALLOWED) for a in physical.values()): raise StopRun('Forbidden physical stage')
    reports={m:component.summary(rows[m]) for m in MODES}; refs=data['narrow_references']
    affected=sorted(set(ids)&set(data['registration']['affected185']))
    result=dict(audit_pass=True,smoke_only=smoke,n=len(ids),total_rows=6*len(ids),registration_sha256=sha(data['registration']),
        reports=reports,historical79=component.summary({q:refs[q] for q in ids}),
        affected185_reports={m:component.summary({q:rows[m][q] for q in affected}) for m in MODES} if affected else {},
        paired_vs_historical79={m:p.paired(rows[m],refs,ids) for m in MODES},
        paired_control_minus_math={m:p.paired(rows[m],rows['math50'],ids) for m in ('dense50','q0_only','no_coverage','utility_sum')},
        read_costs={m:{'mean_'+k:statistics.mean(r[k] for r in rows[m].values()) for k in ('rerank_input_count','rerank_prompt_tokens','rerank_output_tokens')} for m in MODES},
        selection_stats=stats,physical_transport=physical,character_cap=None,token_matched=False,
        repeat_has_fresh_rerank_qa_judge=True,math_full_is_not_historical79=True)
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'),result)
    p.old.emit('NARROW '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run']); args=parser.parse_args()
    with p.old.Lease(ROOT):
        data=prepare()
        if args.command=='prepare': return
        from dotenv import load_dotenv
        load_dotenv(p.old.ROOT/'.env',override=False)
        control=transport.ConnectionControl(live=True)
        runtimes=dict(shared=prior.Runtime(ROOT/'shared/calls',control=control,allow_model_calls=True,sources=SOURCES),
            fresh_repeat=FreshRuntime(ROOT/'fresh_repeat/calls',control=control,allow_model_calls=True))
        try:
            bycat={}
            for qid,q in sorted(data['questions'].items()): bycat.setdefault(q['category_name'],qid)
            affected=set(data['registration']['affected185'])
            smoke_ids=sorted(set(p.smoke_ids(data))|set(bycat.values())|{min(affected),min(set(data['questions'])-affected)})
            if not (ROOT/'smoke_complete.json').exists():
                for qid in smoke_ids: question_job(data,qid,runtimes)
                finish(data,smoke_ids,smoke=True)
            else:
                smoke=read(ROOT/'smoke_complete.json')
                if not smoke['audit_pass'] or smoke['registration_sha256']!=sha(data['registration']): raise StopRun('Smoke binding drift')
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs=[pool.submit(question_job,data,qid,runtimes) for qid in sorted(data['questions'])]
                for job in as_completed(jobs): job.result()
            finish(data,sorted(data['questions']))
        except BaseException as exc:
            runtimes['shared'].record_failure(exc)
            seal(ROOT/f'stop_{time.time_ns()}.json',dict(utc=p.base.now(),type=type(exc).__name__,message=str(exc)[:600])); raise


if __name__=='__main__': main()

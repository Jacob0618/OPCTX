"""Six matched-budget cumulative controls; two immutable Full references."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time

from creamem3.experiments.locomo_math20_10_20260917 import run as full

p, prior = full.p, full.prior
read, seal, sha, file_sha, StopRun = full.read, full.seal, full.sha, full.file_sha, full.StopRun
ROOT = Path(r'D:\creamem3_cache\locomo_cumulative20_full1540_20260917_r1')
MODES = {f'{line}_{step}': dict(rerank=line=='rerank5', reader=step=='wo_selector',
         online=step!='wo_selector_reader_online')
         for line in ('direct','rerank5')
         for step in ('wo_selector','wo_selector_reader','wo_selector_reader_online')}
SOURCES = (full.ROOT/'calls', prior.BASE/'conditions/narrow/calls',
           Path(r'D:\creamem3_cache\component_ablations_20260911_r3\locomo\calls'))
WORKERS = 12


def choose(pool):
    ids = [c['fact']['id'] for c in pool]
    if not ids or len(ids)!=len(set(ids)):
        raise StopRun('Invalid mixed candidate universe')
    if any(pool[i]['dense_score'] < pool[i+1]['dense_score'] for i in range(len(pool)-1)):
        raise StopRun('Candidate order is not original cosine ranking')
    return ids[:20]


def arm(mode):
    return 'opctx_original' if MODES[mode]['online'] else 'session_strong'


def identity(data, mode, qid):
    a, s = arm(mode), data['samples'][qid]
    return dict(registration_sha256=sha(data['registration']), qa_sha256=sha(data['questions'][qid]),
                graph_sha256=data['graph_hashes'][(s,a)], vector_sha256=sha(data['vectors'][qid]), condition=mode, arm=a)


def prepare():
    loader=prior.aligned.load_private(prior.__package__+'._cumulative20_inputs', prior.__file__, prior.__package__)
    captured={}
    loader.seal=lambda path,value: captured.__setitem__(Path(path).name,copy.deepcopy(value))
    data=loader.prepare()
    oldreg, done = read(full.ROOT/'registration.json'), read(full.ROOT/'complete.json')
    if not done['audit_pass'] or done['smoke_only'] or done['n']!=1540 or done['registration_sha256']!=sha(oldreg):
        raise StopRun('Full references not audited')
    frozen=captured['source_snapshot.json']
    for name in ('registration.json','complete.json','source_snapshot.json'):
        frozen[str(full.ROOT/name)]=file_sha(full.ROOT/name)
    frozen.update(read(full.ROOT/'source_snapshot.json'))
    hashes={**data['registration']['source_hashes'], **oldreg['source_hashes']}
    for path,digest in hashes.items():
        if file_sha(path)!=digest: raise StopRun('Source code drift '+path)
    refs={}
    for mode in ('math20_direct','math20_rerank5'):
        refs[mode]={}
        for qid,q in sorted(data['questions'].items()):
            path=full.ROOT/'rows'/mode/(qid+'.json')
            row=read(path); ret=read(full.ROOT/'retrieval'/mode/(qid+'.json'))
            expected=dict(registration_sha256=sha(oldreg),qa_sha256=sha(q),
                graph_sha256=data['graph_hashes'][(data['samples'][qid],'opctx_original')],
                vector_sha256=sha(data['vectors'][qid]),condition=mode,arm='opctx_original')
            if row['identity']!=expected or row['context']!=ret['context'] or row['context_sha256']!=sha(row['context']):
                raise StopRun('Full row binding drift')
            refs[mode][qid]=row; frozen[str(path)]=file_sha(path)
            frozen[str(full.ROOT/'retrieval'/mode/(qid+'.json'))]=file_sha(full.ROOT/'retrieval'/mode/(qid+'.json'))
        if full.component.summary(refs[mode]) != done['reports'][mode]:
            raise StopRun('Full summary drift')
    own=[Path(__file__),Path(__file__).with_name('protocol.md'),Path(__file__).with_name('test_core.py')]
    reg=dict(schema='locomo.cumulative20.v1',n=1540,new_rows=9240,modes=MODES,
        all_ids=sorted(data['questions']),config=prior.CONFIG,max_shortlist=20,judge_votes=1,
        source_hashes={**hashes,**{str(x):file_sha(x) for x in own}},source_snapshot_sha256=sha(frozen),
        source_full_root=str(full.ROOT),source_full_scores={m:done['reports'][m] for m in refs},
        source_call_roots=[str(x) for x in SOURCES],models=prior.ALLOWED,workers=WORKERS,wire_concurrency=8,
        cumulative_order=['selector','reader','online'],selector_control='original cosine Top20 within identical hybrid union',
        character_cap=None,token_matched=False,new_write_calls=False,
        qa_prompt_sha256=sha(p.old.QA_PROMPT),judge_prompt_sha256=sha(p.old.JUDGE_PROMPT),
        rerank_prompt_sha256=sha(p.old.graph_api.FACT_RERANK_PROMPT))
    seal(ROOT/'registration.json',reg); seal(ROOT/'source_snapshot.json',frozen)
    data.update(registration=reg,full_refs=refs)
    data['control_indexes']={(s,a):p.source.enum_repair.Index(data['graphs'][(s,a)],**prior.CONFIG)
        for s in sorted(set(data['samples'].values())) for a in ('session_strong','opctx_original')}
    for qid in sorted(data['questions']):
        for a in ('session_strong','opctx_original'): selection(data,a,qid)
    protected=set()
    for name in list(frozen)+[str(x) for x in SOURCES]+[str(ROOT),str(full.ROOT)]:
        path=Path(name)
        if str(path).lower().startswith('d:\\creamem3_cache\\'):
            protected.add(str(Path(*path.parts[:3])))
    seal(ROOT/'retention.json',dict(protected_roots=sorted(protected),reason='Current cumulative experiment inputs and exact-cache sources'))
    seal(ROOT/'preflight.json',dict(pass_=True,registration_sha256=sha(reg),n=1540,selections=3080,new_model_calls=0))
    p.old.emit('CUMULATIVE20 PREFLIGHT PASS 1540 questions, 6 controls, 2 frozen references')
    return data


def selection(data,a,qid):
    idx=data['control_indexes'][(data['samples'][qid],a)]
    pool,trace=idx._candidate_pool(data['questions'][qid]['question'],data['vectors'][qid])
    if a=='opctx_original' and trace!=data['candidate_traces'][qid]:
        raise StopRun('Online common pool drift')
    saved=dict(registration_sha256=sha(data['registration']),qa_id=qid,arm=a,candidate_trace=trace,
               prefilter_fact_ids=choose(pool),selection_rule='q0_cosine_top20',no_character_truncation=True)
    path=ROOT/'selection'/a/(qid+'.json')
    if path.exists():
        if read(path)!=saved: raise StopRun('Frozen shortlist drift')
    else: seal(path,saved)
    return saved,pool,trace


def retrieve(data,mode,qid,runtime):
    cfg=MODES[mode]; a=arm(mode); sample=data['samples'][qid]; q=data['questions'][qid]
    sel,pool,trace=selection(data,a,qid)
    idx=data['control_indexes'][(sample,a)]; byid={c['fact']['id']:c for c in pool}
    candidates=[byid[i] for i in sel['prefilter_fact_ids']]
    if cfg['rerank']:
        picked,detail=idx._rerank(q['question'],candidates,runtime.chat_fn(a+'/'+qid))
    else:
        picked=candidates; detail=dict(attempts=0,fallback=False,error=None,method='q0_cosine_top20_no_rerank',model_called=False)
    ids=[c['fact']['id'] for c in picked]
    p.validate_selection(ids,candidates,n=5 if cfg['rerank'] else len(candidates))
    if detail['fallback']: raise StopRun('Rerank fallback forbidden')
    _,assembly=idx._assemble(picked,trace['top_summaries'])
    trace.update(rerank=dict(**detail,selected_fact_ids=ids),assembly=assembly)
    if cfg['reader']:
        context,audit=p.render(idx,pool,trace,data['graphs'][(sample,a)],q,a,data['operations'][qid])
    else:
        plain=data['control_indexes'][(sample,'session_strong')]
        facts={f['id']:f for f in data['graphs'][(sample,'session_strong')]['fact_nodes']}
        context,audit=plain._assemble([{'fact':facts[i]} for i in ids],[])
        if audit['rendered_event_view_ids'] or audit['rendered_fact_ids']!=ids:
            raise StopRun('Reader ablation leaked views/changed fact IDs')
    return dict(identity=identity(data,mode,qid),context=context,trace=trace,materializer=audit,
        selection_sha256=sha(sel),prefilter_fact_ids=sel['prefilter_fact_ids'],final_fact_count=len(ids),
        rerank_input_count=len(candidates) if cfg['rerank'] else 0,reader_enabled=cfg['reader'])


def evaluate(data,mode,qid,runtime):
    path=ROOT/'rows'/mode/(qid+'.json'); binding=identity(data,mode,qid)
    if path.exists():
        row=read(path)
        if row['identity']!=binding: raise StopRun('Resume identity drift')
        return row
    runtime.local.requests=[]; start=time.perf_counter(); q=data['questions'][qid]
    ret=retrieve(data,mode,qid,runtime); seal(ROOT/'retrieval'/mode/(qid+'.json'),ret)
    answer=runtime.chat(qid,'qa',p.old.QA_PROMPT.format(question=q['question'],retrieved_texts=ret['context']),max_tokens=8192)
    correct,judge=p.old.judge(runtime,qid,q,answer); receipts=p.unique_requests(runtime.local.requests)
    qa=[x for x in receipts if x['stage']=='qa']; rr=[x for x in receipts if x['stage']=='rerank']
    if len(qa)!=1 or bool(rr)!=MODES[mode]['rerank'] or any(x['stage'] not in prior.ALLOWED for x in receipts):
        raise StopRun('Unexpected read stages')
    row=dict(identity=binding,qa_id=qid,sample_id=data['samples'][qid],arm=arm(mode),condition=mode,
        question=q['question'],answer=q['answer'],category_name=q['category_name'],response=answer,correct=correct,judge_raw=judge,
        context=ret['context'],context_sha256=sha(ret['context']),context_chars=len(ret['context']),
        selected_fact_ids=ret['trace']['rerank']['selected_fact_ids'],prefilter_fact_ids=ret['prefilter_fact_ids'],
        candidate_count=len(ret['trace']['candidates']),selection_sha256=ret['selection_sha256'],
        rerank_input_count=ret['rerank_input_count'],final_fact_count=ret['final_fact_count'],
        qa_prompt_tokens=qa[0]['usage'].get('promptTokenCount'),rerank_prompt_tokens=sum(x['usage'].get('promptTokenCount',0) for x in rr),
        logical_request_receipts=receipts,provider_root=str(ROOT/'calls'),elapsed_s=time.perf_counter()-start)
    seal(path,row); p.old.emit(f'CUMULATIVE20 ROW {mode}/{qid} correct={correct}')
    return row


def question_job(data,qid,runtime):
    try:
        for mode in MODES:
            if runtime.stopped.is_set(): raise StopRun('Dispatch stopped')
            evaluate(data,mode,qid,runtime)
    except BaseException as exc:
        runtime.record_failure(exc); raise


def finish(data,ids,smoke=False):
    replay=prior.prior.Runtime(ROOT/'calls',control=full.transport.ConnectionControl(live=False),allow_model_calls=False,sources=())
    replay.local.cached_only=True; rows={m:{} for m in MODES}
    for mode in MODES:
        if not smoke and {x.stem for x in (ROOT/'rows'/mode).glob('*.json')}!=set(ids): raise StopRun('Incomplete population')
        for qid in ids:
            row=read(ROOT/'rows'/mode/(qid+'.json')); ret=read(ROOT/'retrieval'/mode/(qid+'.json'))
            actual=retrieve(data,mode,qid,replay)
            if actual!=ret or row['identity']!=identity(data,mode,qid) or row['context']!=ret['context'] or row['context_sha256']!=sha(ret['context']):
                raise StopRun('Final renderer/identity drift')
            if row['selected_fact_ids']!=ret['trace']['rerank']['selected_fact_ids'] or row['prefilter_fact_ids']!=ret['prefilter_fact_ids']:
                raise StopRun('Final ID drift')
            p.base.check_bound(ROOT,data['questions'][qid],row)
            rows[mode][qid]=row
        p.old.emit('CUMULATIVE20 AUDIT '+mode+' '+str(len(ids)))
    for line in ('direct','rerank5'):
        for qid in ids:
            a,b=(rows[line+'_'+step][qid] for step in ('wo_selector','wo_selector_reader'))
            if a['selected_fact_ids']!=b['selected_fact_ids'] or a['prefilter_fact_ids']!=b['prefilter_fact_ids']:
                raise StopRun('Reader removal changed selection')
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Code drift '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path)!=digest: raise StopRun('Input drift '+path)
    reports={m:full.component.summary(rows[m]) for m in MODES}
    paired={}
    for line,base in (('direct','math20_direct'),('rerank5','math20_rerank5')):
        chain=[data['full_refs'][base]]+[rows[line+'_'+s] for s in ('wo_selector','wo_selector_reader','wo_selector_reader_online')]
        for i in range(3): paired[line+'_step'+str(i+1)]=p.paired(chain[i+1],chain[i],ids)
    result=dict(audit_pass=True,smoke_only=smoke,n=len(ids),registration_sha256=sha(data['registration']),reports=reports,
        full_references={m:full.component.summary({q:rr[q] for q in ids}) for m,rr in data['full_refs'].items()},
        paired_cumulative_step=paired,physical_transport=full.component.provider_audit(ROOT/'calls'),token_matched=False,
        costs={m:{'mean_'+k:statistics.mean(r[k] for r in rr.values()) for k in ('final_fact_count','qa_prompt_tokens','context_chars','rerank_prompt_tokens')} for m,rr in rows.items()})
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'),result)
    p.old.emit('CUMULATIVE20 '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','run','status']); args=parser.parse_args()
    if args.command=='status':
        print(json.dumps(dict(root=str(ROOT),rows={m:len(list((ROOT/'rows'/m).glob('*.json'))) for m in MODES},
            smoke=(ROOT/'smoke_complete.json').exists(),complete=(ROOT/'complete.json').exists(),stops=[str(x) for x in ROOT.glob('stop_*.json')]))); return
    with p.old.Lease(ROOT):
        runtime=None
        try:
            data=prepare()
            if args.command=='prepare': return
            from dotenv import load_dotenv
            load_dotenv(p.old.ROOT/'.env',override=False)
            runtime=prior.prior.Runtime(ROOT/'calls',control=full.transport.ConnectionControl(live=True),allow_model_calls=True,sources=SOURCES)
            bycat={}
            for qid,q in sorted(data['questions'].items()): bycat.setdefault(q['category_name'],qid)
            smoke=set(p.smoke_ids(data))|set(bycat.values())
            for eligible in (True,False):
                matches=[q for q,t in sorted(data['candidate_traces'].items()) if (len(t['candidates'])>20)==eligible]
                if matches: smoke.add(matches[0])
            if not (ROOT/'smoke_complete.json').exists():
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    for job in as_completed([pool.submit(question_job,data,q,runtime) for q in sorted(smoke)]): job.result()
                finish(data,sorted(smoke),smoke=True)
            else:
                proof=read(ROOT/'smoke_complete.json')
                if not proof['audit_pass'] or proof['registration_sha256']!=sha(data['registration']): raise StopRun('Smoke drift')
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for job in as_completed([pool.submit(question_job,data,q,runtime) for q in sorted(data['questions'])]): job.result()
            finish(data,sorted(data['questions']))
        except BaseException as exc:
            if runtime: runtime.record_failure(exc)
            seal(ROOT/f'stop_{time.time_ns()}.json',dict(utc=p.base.now(),type=type(exc).__name__,message=str(exc)[:600])); raise


if __name__=='__main__': main()

"""Four fixed read-side arms; immutable cache, shared saved20/10 selections."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import statistics
import time

from . import core
from creamem3.experiments.locomo_narrow_rerank_ablations_20260913 import run as prior

p, transport, component = prior.p, prior.transport, prior.component
read, seal, sha, file_sha, StopRun = prior.read, prior.seal, prior.sha, prior.file_sha, prior.StopRun
ROOT = Path(r'D:\creamem3_cache\locomo_math20_10_full1540_20260917_r1')
ARM = prior.ARM
MODES = core.MODES
WORKERS = 12


def prepare():
    loader = prior.aligned.load_private(prior.__package__+'._math20_10_inputs', prior.__file__, prior.__package__)
    captured = {}
    loader.seal = lambda path, value: captured.__setitem__(Path(path).name, copy.deepcopy(value))
    data = loader.prepare()
    oldreg, done = read(prior.ROOT/'registration.json'), read(prior.ROOT/'complete.json')
    if not done['audit_pass'] or done['smoke_only'] or done['registration_sha256'] != sha(oldreg):
        raise StopRun('Incomplete narrow50 source')
    frozen = captured['source_snapshot.json']
    for name in ('registration.json', 'complete.json'):
        frozen[str(prior.ROOT/name)] = file_sha(prior.ROOT/name)
    hashes = {**oldreg['source_hashes'], **data['registration']['source_hashes']}
    for path, digest in hashes.items():
        if file_sha(path) != digest:
            raise StopRun('Frozen code drift '+path)
    counts = [len(t['candidates']) for t in data['candidate_traces'].values()]
    own = [Path(__file__), Path(core.__file__), Path(__file__).with_name('protocol.md'),
           Path(__file__).with_name('test_core.py')]
    reg = dict(schema='locomo.math20_10.fourarm.v1', n=1540, total_rows=6160,
        arms=[ARM], modes={m:dict(limit=k, rerank=r) for m,(k,r) in MODES.items()},
        config=prior.CONFIG, all_ids=sorted(data['questions']), workers=WORKERS,
        wire_concurrency=transport.WIRE_CONCURRENCY, models=prior.ALLOWED,
        source_hashes={**hashes, **{str(x):file_sha(x) for x in own}},
        source_snapshot_sha256=sha(frozen), judge_votes=1, historical_response_imports=False,
        qa_prompt_sha256=sha(p.old.QA_PROMPT), judge_prompt_sha256=sha(p.old.JUDGE_PROMPT),
        rerank_prompt_sha256=sha(p.old.graph_api.FACT_RERANK_PROMPT),
        character_cap=None, text_truncation=False, token_matched=False,
        shared_saved_selection_per_limit=True, fresh_read_namespace=True,
        candidate_stats=dict(mean=statistics.mean(counts), median=statistics.median(counts),
            min=min(counts), max=max(counts), active20=sum(x>20 for x in counts), active10=sum(x>10 for x in counts)))
    seal(ROOT/'registration.json', reg)
    seal(ROOT/'source_snapshot.json', frozen)
    seal(ROOT/'questions_score_free.json', captured['questions_score_free.json'])
    data['registration'] = reg
    # Every selection is frozen before the first new answer; no label-guided changes.
    def select_question(qid):
        for limit in core.LIMITS:
            selection(data, limit, qid)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [pool.submit(select_question, q) for q in sorted(data['questions'])]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            if i % 100 == 0:
                p.old.emit(f'MATH20_10 SELECT {i}/1540')
    seal(ROOT/'preflight.json', dict(pass_=True, n=1540, selections=3080,
        registration_sha256=sha(reg), new_model_calls=0, candidate_stats=reg['candidate_stats']))
    p.old.emit('MATH20_10 PREFLIGHT PASS '+json.dumps(reg['candidate_stats']))
    return data


def identity(data, mode, qid):
    sample = data['samples'][qid]
    return dict(registration_sha256=sha(data['registration']), qa_sha256=sha(data['questions'][qid]),
        graph_sha256=data['graph_hashes'][(sample, ARM)], vector_sha256=sha(data['vectors'][qid]),
        condition=mode, arm=ARM)


def selection(data, limit, qid):
    path = ROOT/'selection'/str(limit)/(qid+'.json')
    sample = data['samples'][qid]
    binding = dict(registration_sha256=sha(data['registration']), qa_id=qid,
        question_sha256=sha(data['questions'][qid]['question']), bank_sha256=sha(data['banks'][qid]),
        graph_sha256=data['graph_hashes'][(sample, ARM)], limit=limit)
    candidates, trace = data['narrow_indexes'][sample]._candidate_pool(data['questions'][qid]['question'], data['vectors'][qid])
    if trace != data['candidate_traces'][qid]:
        raise StopRun('Frozen candidate replay drift')
    if path.exists():
        saved = read(path)
    else:
        saved = dict(identity=binding, candidate_trace=trace, **core.choose(candidates,
            data['graphs'][(sample, 'session_strong')], data['questions'][qid]['question'], data['channels'][qid], limit))
        seal(path, saved)
    if saved['identity'] != binding or saved['candidate_trace'] != trace:
        raise StopRun('Selection identity/trace drift')
    core.verify(saved, candidates, data['graphs'][(sample, 'session_strong')],
        data['questions'][qid]['question'], data['channels'][qid], limit)
    return saved


def retrieve(data, mode, qid, runtime):
    limit, rerank = MODES[mode]
    sel = selection(data, limit, qid)
    sample, q = data['samples'][qid], data['questions'][qid]
    index = data['narrow_indexes'][sample]
    pool, trace = index._candidate_pool(q['question'], data['vectors'][qid])
    by_id = {c['fact']['id']:c for c in pool}
    candidates = [by_id[i] for i in sel['prefilter_fact_ids']]
    if rerank:
        picked, detail = index._rerank(q['question'], candidates, runtime.chat_fn(ARM+'/'+qid))
        expected_count = 5
    else:
        picked = candidates
        detail = dict(attempts=0, fallback=False, error=None, method='mathematical_shortlist_no_model_rerank', model_called=False)
        expected_count = len(candidates)
    ids = [c['fact']['id'] for c in picked]
    p.validate_selection(ids, candidates, n=expected_count)
    if detail['fallback']:
        raise StopRun('Rerank fallback forbidden')
    _, assembly = index._assemble(picked, trace['top_summaries'])
    trace.update(rerank=dict(**detail, selected_fact_ids=ids), assembly=assembly)
    context, audit = p.render(index, pool, trace, data['graphs'][(sample, ARM)], q, ARM, data['operations'][qid])
    return dict(identity=identity(data, mode, qid), selection_sha256=sha(sel), context=context,
        trace=trace, materializer=audit, prefilter_fact_ids=sel['prefilter_fact_ids'],
        rerank_input_count=len(candidates) if rerank else 0, final_fact_count=len(ids), no_model_rerank=not rerank)


def evaluate(data, mode, qid, runtime):
    path = ROOT/'rows'/mode/(qid+'.json')
    binding = identity(data, mode, qid)
    if path.exists():
        row = read(path)
        if row['identity'] != binding:
            raise StopRun('Resume row drift')
        return row
    runtime.local.requests = []
    start = time.perf_counter()
    q = data['questions'][qid]
    ret = retrieve(data, mode, qid, runtime)
    seal(ROOT/'retrieval'/mode/(qid+'.json'), ret)
    answer = runtime.chat(qid, 'qa', p.old.QA_PROMPT.format(question=q['question'], retrieved_texts=ret['context']), max_tokens=8192)
    correct, judge = p.old.judge(runtime, qid, q, answer)
    receipts = p.unique_requests(runtime.local.requests)
    qa = [r for r in receipts if r['stage']=='qa']
    rr = [r for r in receipts if r['stage']=='rerank']
    if len(qa)!=1 or bool(rr)!=MODES[mode][1] or any(r['stage'] not in prior.ALLOWED for r in receipts):
        raise StopRun('Read stage drift')
    row = dict(identity=binding, qa_id=qid, sample_id=data['samples'][qid], arm=ARM, condition=mode,
        question=q['question'], answer=q['answer'], category_name=q['category_name'], response=answer,
        correct=correct, judge_raw=judge, context=ret['context'], context_sha256=sha(ret['context']),
        context_chars=len(ret['context']), selected_fact_ids=ret['trace']['rerank']['selected_fact_ids'],
        prefilter_fact_ids=ret['prefilter_fact_ids'], selection_sha256=ret['selection_sha256'],
        candidate_count=len(data['candidate_traces'][qid]['candidates']),
        rerank_input_count=ret['rerank_input_count'], final_fact_count=ret['final_fact_count'],
        qa_prompt_tokens=qa[0]['usage'].get('promptTokenCount'),
        rerank_prompt_tokens=sum(r['usage'].get('promptTokenCount',0) for r in rr),
        rerank_output_tokens=sum(r['usage'].get('candidatesTokenCount',0) for r in rr),
        logical_request_receipts=receipts, provider_root=str(ROOT/'calls'), elapsed_s=time.perf_counter()-start)
    seal(path, row)
    p.old.emit(f'MATH20_10 ROW {mode}/{qid} correct={correct} facts={ret["final_fact_count"]}')
    return row


def question_job(data, qid, runtime):
    try:
        for mode in MODES:
            if runtime.stopped.is_set():
                raise StopRun('Dispatch stopped')
            evaluate(data, mode, qid, runtime)
    except BaseException as exc:
        runtime.record_failure(exc)
        raise


def finish(data, ids, smoke=False):
    replay = prior.prior.Runtime(ROOT/'calls', control=transport.ConnectionControl(live=False),
        allow_model_calls=False, sources=())
    replay.local.cached_only = True
    rows = {m:{} for m in MODES}
    for mode in MODES:
        if not smoke and {x.stem for x in (ROOT/'rows'/mode).glob('*.json')} != set(ids):
            raise StopRun('Incomplete population')
        for qid in ids:
            row = read(ROOT/'rows'/mode/(qid+'.json'))
            ret = read(ROOT/'retrieval'/mode/(qid+'.json'))
            actual = retrieve(data, mode, qid, replay)
            if actual != ret or row['identity'] != identity(data,mode,qid):
                raise StopRun('Retrieval/identity drift')
            if row['context'] != ret['context'] or row['context_sha256'] != sha(ret['context']):
                raise StopRun('QA context drift')
            if row['selected_fact_ids'] != ret['trace']['rerank']['selected_fact_ids'] or row['selection_sha256'] != ret['selection_sha256']:
                raise StopRun('Final selection drift')
            if row['prefilter_fact_ids'] != ret['prefilter_fact_ids'] or row['final_fact_count'] != ret['final_fact_count'] or row['rerank_input_count'] != ret['rerank_input_count']:
                raise StopRun('Cardinality binding drift')
            if not MODES[mode][1] and any(r['stage']=='rerank' for r in row['logical_request_receipts']):
                raise StopRun('Direct arm called reranker')
            p.base.check_bound(ROOT, data['questions'][qid], row)
            for rec in row['logical_request_receipts']:
                _,_,_,raw = p.full.recover.completed_receipt(ROOT/'calls/provider'/rec['stage']/rec['key'])
                if raw['response'].get('usageMetadata',{}) != rec['usage']:
                    raise StopRun('Usage binding drift')
            rows[mode][qid] = row
        p.old.emit(f'MATH20_10 AUDIT {mode} {len(ids)}')
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path) != digest:
            raise StopRun('Registered source code changed '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path) != digest:
                raise StopRun('Frozen input changed '+path)
        for qid in ids:
            prior.previous.verify_bank(data, qid)
    physical = component.provider_audit(ROOT/'calls')
    if set(physical['stages']) - set(prior.ALLOWED) or physical['counts'].get('imported_physical_attempts',0):
        raise StopRun('Unexpected stage/import')
    if list((ROOT/'calls/imports').rglob('*.json')):
        raise StopRun('Historical response import forbidden')
    reports = {m:component.summary(rows[m]) for m in MODES}
    pairs = [('math20_rerank5','math10_rerank5'), ('math20_direct','math10_direct'),
             ('math20_rerank5','math20_direct'), ('math10_rerank5','math10_direct')]
    result = dict(audit_pass=True, smoke_only=smoke, n=len(ids), total_rows=4*len(ids),
        registration_sha256=sha(data['registration']), reports=reports,
        paired={a+'_minus_'+b:p.paired(rows[a],rows[b],ids) for a,b in pairs},
        paired_vs_historical79={m:p.paired(rows[m],data['narrow_references'],ids) for m in MODES},
        read_costs={m:{'mean_'+k:statistics.mean(r[k] for r in rows[m].values()) for k in
            ('rerank_input_count','final_fact_count','context_chars','qa_prompt_tokens','rerank_prompt_tokens','rerank_output_tokens')} for m in MODES},
        physical_transport=physical, historical_imports=False, token_matched=False,
        candidate_stats=data['registration']['candidate_stats'])
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'), result)
    p.old.emit('MATH20_10 '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))
    return result


def status():
    counts = {m:sum(1 for _ in (ROOT/'rows'/m).glob('*.json')) for m in MODES}
    stops = [str(x) for x in ROOT.glob('stop_*.json')]
    print(json.dumps(dict(root=str(ROOT), rows=counts, total=sum(counts.values()),
        preflight=(ROOT/'preflight.json').exists(), smoke=(ROOT/'smoke_complete.json').exists(),
        complete=(ROOT/'complete.json').exists(), stops=stops)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','run','status'])
    args=parser.parse_args()
    if args.command=='status':
        return status()
    with p.old.Lease(ROOT):
        runtime=None
        try:
            data=prepare()
            if args.command=='prepare':
                return
            from dotenv import load_dotenv
            load_dotenv(p.old.ROOT/'.env',override=False)
            runtime=prior.FreshRuntime(ROOT/'calls',control=transport.ConnectionControl(live=True),allow_model_calls=True)
            bycat={}
            for qid,q in sorted(data['questions'].items()):
                bycat.setdefault(q['category_name'],qid)
            smoke_ids=set(p.smoke_ids(data)) | set(bycat.values())
            for limit in core.LIMITS:
                active=[q for q in sorted(data['questions']) if len(data['candidate_traces'][q]['candidates'])>limit]
                bypass=[q for q in sorted(data['questions']) if len(data['candidate_traces'][q]['candidates'])<=limit]
                smoke_ids.update(x[0] for x in (active,bypass) if x)
            if not (ROOT/'smoke_complete.json').exists():
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    for job in as_completed([pool.submit(question_job,data,q,runtime) for q in sorted(smoke_ids)]):
                        job.result()
                finish(data,sorted(smoke_ids),smoke=True)
            else:
                smoke=read(ROOT/'smoke_complete.json')
                if not smoke['audit_pass'] or smoke['registration_sha256']!=sha(data['registration']):
                    raise StopRun('Smoke binding drift')
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs=[pool.submit(question_job,data,q,runtime) for q in sorted(data['questions'])]
                for job in as_completed(jobs):
                    job.result()
            finish(data,sorted(data['questions']))
        except BaseException as exc:
            if runtime is not None:
                runtime.record_failure(exc)
            seal(ROOT/f'stop_{time.time_ns()}.json',dict(utc=p.base.now(),type=type(exc).__name__,message=str(exc)[:600]))
            raise


if __name__=='__main__':
    main()

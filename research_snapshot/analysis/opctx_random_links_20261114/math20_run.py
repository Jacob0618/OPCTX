"""Fixed-seed link routing, original full mathematical20, no model reranker.

Separate explicit phases: prepare_bank (paid, question-only), freeze (offline),
run (paid, frozen contexts). This file never starts calls when imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from analysis.opctx_random_links_20261113 import run as previous
from analysis.opctx_random_links_20261113 import transport as wire
from creamem3.experiments.locomo_math20_10_20260917 import core
from creamem3.experiments.locomo_math_narrow_20260911 import core as channels_core
from creamem3.online_operation_context_v8 import materialize_online_operation_context

SHARED = Path('D:/opctx_random_links_20261113/r1')
ARMS = ('normal', 'random')
MODEL, JUDGE = previous.MODEL, previous.JUDGE
read, file_sha, save = previous.read, previous.file_sha, wire.save
digest = wire.digest
PROTOCOL = Path(__file__).with_name('protocol.md')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_or_check(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError('Immutable artifact differs: ' + str(path))
    else:
        save(path, value)


def code_sources(inputs):
    modules = [core, core.parent, core.parent.parent, core.parent.old,
               channels_core, channels_core.solver, channels_core.solver.base,
               channels_core.solver.numerical, channels_core.solver.lexical,
               channels_core.qp, wire]
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.update((Path(__file__).resolve(), PROTOCOL.resolve(),
        Path(inspect.getsourcefile(channels_core.query_texts)).resolve(),
        ROOT/'creamem3/experiments/locomo_math_narrow_20260911/run.py',
        ROOT/'creamem3/experiments/memeye_resume_20260921/network.py'))
    return {**inputs['metadata']['source_hashes'], **{str(p): file_sha(p) for p in paths}}


def load_shared():
    """Read the old score-free q0/operation artifacts, never its QA answers."""
    inputs = previous.load_inputs()
    questions = read(SHARED/'questions.json')
    operations = read(SHARED/'question_operations.json')
    shared = read(SHARED/'query_vectors.json')
    if questions != inputs['rows'] or operations != inputs['plans']:
        raise RuntimeError('Shared question/operation identity drift')
    ids = [q['qa_id'] for q in questions]
    if shared['ids'] != ids or shared['model'] != 'gemini-embedding-001':
        raise RuntimeError('Shared query embedding identity/model drift')
    vectors = dict(zip(ids, shared['vectors'], strict=True))
    for value in vectors.values():
        channels_core.vector_check(value)
    if len(vectors) != 282:
        raise RuntimeError('Expected all282 original query vectors')
    return inputs, vectors


def register(output, workers):
    inputs, vectors = load_shared()
    prompts = previous.frozen.frozen_prompts()
    shared_hashes = {str(SHARED/n): file_sha(SHARED/n) for n in
        ('questions.json', 'question_operations.json', 'query_vectors.json')}
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        registration = dict(schema='locomo.random_links.math20.v1', created_utc=stamp(),
            source_hashes=code_sources(inputs), shared_hashes=shared_hashes,
            seed=92620, arms=list(ARMS), questions=282, expected=564, workers=workers,
            qa_model=MODEL, judge_model=JUDGE, embedding='gemini-embedding-001',
            summaries=10, dense=20, bm25=20, candidate_cap=120, final_fact_count=20,
            total_event_views=243, total_event_links=5567,
            intervention='event-view routing adjacency only; original graph for Reader and support',
            selector='locomo_math20_10_20260917.core.choose(limit=20)',
            coverage_first=True, max_min_balance=True, solver_seconds_per_stage=5.0,
            character_bound='sum of all candidate fact costs plus all candidate session group costs; nonbinding',
            selection_order='unchanged candidate q0 dense order', rerank_calls=0,
            writer_calls=0, online_calls=0, operation_planner_calls=0,
            projection_bank='fresh shared reconstruction; historical bank deleted',
            projection_policy='q0_plus_anchored_attributes', projection_max_tokens=2048,
            projection_temperature=0.0, projection_thinking_budget=0,
            projection_semantic_attempts=1,
            projection_fallback='unchanged validate_response; explicit extractive-validation fallback only',
            q0_source=str(SHARED/'query_vectors.json'), embedding_dim=768,
            embedding_task='RETRIEVAL_QUERY', historical_qa_reuse=False,
            not_strict_token_matched=True, parent_outputs=[],
            qa_prompt_sha256=digest(prompts['QA_PROMPT']),
            judge_prompt_sha256=digest(prompts['JUDGE_PROMPT']),
            projection_prompt_sha256=digest(channels_core.qp.PROMPT_PREFIX))
        save(output/'registration.json', registration)
        save(output/'questions.json', inputs['rows'])
        save(output/'question_operations.json', inputs['plans'])
        save(output/'query_vectors.json', read(SHARED/'query_vectors.json'))
        save(output/'fixed_permutations.json', {s:read(previous.PERMS)[s][previous.SEED]
                                              for s in inputs['graphs']})
    registration = read(output/'registration.json')
    if (registration['source_hashes'] != code_sources(inputs)
            or registration['shared_hashes'] != shared_hashes
            or registration['workers'] != workers):
        raise RuntimeError('Registered source/shared artifact/worker contract changed')
    if read(output/'questions.json') != inputs['rows'] or read(output/'question_operations.json') != inputs['plans']:
        raise RuntimeError('Registered questions/operations changed')
    return inputs, vectors, registration, prompts


def verify_registered(output):
    registration = read(output/'registration.json')
    for mapping in ('source_hashes', 'shared_hashes'):
        for path, expected in registration[mapping].items():
            if file_sha(path) != expected:
                raise RuntimeError('Registered dependency changed: ' + path)
    return register(output, registration['workers'])


def clients(output):
    # Add ledger labels only in this new interpreter. The original chat/embed
    # bodies, payload construction, limits, and transport source are unchanged.
    wire.STAGES.update(query_projection='query_projection',
                       embed_projection='projection_embedding')
    os.environ['JUDGE_THINKING'] = 'off'
    os.environ['GEMINI3_THINKING_LEVEL'] = 'minimal'
    output.mkdir(parents=True, exist_ok=True)
    transport = wire.AuditTransport(output)
    chat, embed = wire.make_clients(transport)
    return transport, chat, embed


def bank_binding(question, q0, registration):
    return dict(registration_sha256=digest(registration), question_sha256=digest(question),
                q0_vector_sha256=digest(q0))


def checked_bank(output, question, q0, qid, registration):
    bank = read(output/'query_bank'/(qid+'.json'))
    if (bank['identity'] != bank_binding(question, q0, registration)
            or bank['question'] != question or bank['vectors'][question] != q0):
        raise RuntimeError('Question/projection bank identity drift')
    projection = channels_core.qp.validate_response(question, bank['raw_projection'])
    queries = channels_core.query_texts(question, projection, 'q0_plus_anchored_attributes')
    if bank['projection'] != projection or bank['queries'] != queries:
        raise RuntimeError('Projection validation/query construction drift')
    if set(bank['vectors']) != {q['query'] for q in queries}:
        raise RuntimeError('Projection vector census mismatch')
    for vector in bank['vectors'].values():
        channels_core.vector_check(vector)
    return bank


def prepare_bank(output, workers):
    """The original question_bank generation sequence, once per question."""
    from filelock import FileLock
    inputs, vectors, registration, _ = register(output, workers)
    with FileLock(str(output/'dispatcher.lock'), timeout=0):
        if (output/'bank_complete.json').exists():
            verify_bank_complete(output, inputs, vectors, registration)
            return read(output/'bank_complete.json')
        transport, chat, embed = clients(output/'bank_runtime')
        began = time.perf_counter()

        def one(question):
            qid, text = question['qa_id'], question['question']
            if transport.stop.is_set():
                return None
            path = output/'query_bank'/(qid+'.json')
            if path.exists():
                return checked_bank(output, text, vectors[qid], qid, registration)
            transport.local.scope = 'query_projection/' + qid
            try:
                # Original lines90-107: one question-only plan, unchanged
                # validator, anchored queries, then only additional embeddings.
                raw = chat(channels_core.qp.build_prompt(text), model=MODEL,
                    max_tokens=2048, json_mode=True, temperature=0.0,
                    thinking_budget=0, tag='query_projection')
                projection = channels_core.qp.validate_response(text, raw)
                queries = channels_core.query_texts(text, projection, 'q0_plus_anchored_attributes')
                texts = [q['query'] for q in queries[1:]]
                values = embed(texts, model='gemini-embedding-001', dim=768,
                    task_type='RETRIEVAL_QUERY', tag='embed_projection') if texts else []
                for vector in values:
                    channels_core.vector_check(vector)
                bank = dict(identity=bank_binding(text, vectors[qid], registration),
                    question=text, projection=projection, queries=queries,
                    vectors={text:vectors[qid], **dict(zip(texts, values, strict=True))},
                    raw_projection=raw, provenance='fresh question-only shared bank',
                    physical_call_scope=transport.local.scope)
                save(path, bank)
                print(json.dumps(dict(phase='bank', qa_id=qid, status=projection['status'],
                                      queries=len(queries))), flush=True)
                return bank
            except BaseException as error:
                transport.stop.set()
                save(output/'bank_errors'/(qid+'.json'), dict(qa_id=qid,
                    error_type=type(error).__name__, message='Inspect bank_runtime calls; no semantic fallback for provider failures'))
                raise

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                banks = list(pool.map(one, inputs['rows']))
            if any(b is None for b in banks) or len(banks) != 282:
                raise RuntimeError('Incomplete shared projection bank')
            result = dict(status='complete', questions=282, created_utc=stamp(),
                registration_sha256=digest(registration), seconds=time.perf_counter()-began,
                projection_status=dict(Counter(b['projection']['status'] for b in banks)),
                fallback_reasons=dict(Counter(b['projection']['fallback_reason'] for b in banks
                                              if b['projection']['status']=='fallback')),
                query_count=dict(Counter(len(b['queries']) for b in banks)),
                files={str(output/'query_bank'/(q['qa_id']+'.json')):
                       file_sha(output/'query_bank'/(q['qa_id']+'.json')) for q in inputs['rows']})
            save(output/'bank_complete.json', result)
            return result
        finally:
            calls, usage = transport.snapshots()
            save(output/'bank_runtime'/('summary_'+str(time.time_ns())+'.json'),
                 dict(calls=calls, usage=usage, stopped=transport.stop.is_set()))


def verify_bank_complete(output, inputs, vectors, registration):
    complete = read(output/'bank_complete.json')
    if complete['status'] != 'complete' or complete['registration_sha256'] != digest(registration):
        raise RuntimeError('Shared bank incomplete or registration drift')
    expected_paths = {str(output/'query_bank'/(q['qa_id']+'.json')) for q in inputs['rows']}
    if set(complete['files']) != expected_paths:
        raise RuntimeError('Bank complete manifest census drift')
    for path, expected in complete['files'].items():
        if file_sha(path) != expected:
            raise RuntimeError('Sealed projection bank changed')
    return {q['qa_id']:checked_bank(output,q['question'],vectors[q['qa_id']],q['qa_id'],registration)
            for q in inputs['rows']}


def select_one(inputs, q, arm, vector, bank):
    """Pure selection/assembly; no labels or gold enter the mathematical code."""
    qid, sid, text = q['qa_id'], q['sample_id'], q['question']
    origin, routing = inputs['indices'][sid]['normal'], inputs['indices'][sid][arm]
    timings = {}
    tick = time.perf_counter()
    candidates, trace = routing._candidate_pool(text, vector)
    _, ordinary_trace = origin._candidate_pool(text, vector)
    for key in ('policy', 'top_summaries', 'dense_fact_ids', 'bm25_facts'):
        if trace[key] != ordinary_trace[key]:
            raise RuntimeError('Unrandomized retrieval branch changed: '+key)
    if not 20 <= len(candidates) <= 120:
        raise RuntimeError('Unexpected candidate count')
    timings['candidate_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter()
    channels = channels_core.build_channels(origin, candidates, text, bank)
    if channels[0]['dense_scores'] != [c['dense_score'] for c in candidates]:
        raise RuntimeError('Frozen q0 score mismatch')
    timings['channel_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter()
    selection = core.choose(candidates, inputs['graphs'][sid], text, channels, 20)
    timings['selector_seconds'] = time.perf_counter()-tick
    ids = selection['prefilter_fact_ids']
    selected = [c for c in candidates if c['fact']['id'] in set(ids)]
    trace['rerank'] = dict(selected_fact_ids=ids, attempts=0, fallback=False, error=None,
        method='mathematical_coverage_maxmin_no_model_rerank', model_called=False)
    tick = time.perf_counter()
    base, assembly = origin._assemble(selected, trace['top_summaries'])
    trace['assembly'] = assembly
    context, reader = materialize_online_operation_context(inputs['plans'][qid],
        dict(question=text, retrieved=base, retrieval_trace=trace), inputs['graphs'][sid])
    timings['reader_seconds'] = time.perf_counter()-tick
    if reader['selected_fact_ids'] != ids or reader['selected_fact_count'] != 20:
        raise RuntimeError('Reader selection drift')
    return dict(**q, arm=arm, candidate_fact_ids=[c['fact']['id'] for c in candidates],
        selected_fact_ids=ids, selection=selection, retrieval_trace=trace,
        reader_audit=reader, retrieved=context, context_chars=len(context),
        selected_fact_chars=sum(len(c['fact']['text']) for c in selected),
        base_context_chars=len(base), bank_sha256=digest(bank), timings=timings)


def distribution(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max()), p95=float(np.percentile(values,95)))


def freeze(output):
    from filelock import FileLock
    inputs, vectors, registration, _ = verify_registered(output)
    banks = verify_bank_complete(output, inputs, vectors, registration)
    with FileLock(str(output/'dispatcher.lock'), timeout=0):
        if (output/'freeze_summary.json').exists():
            return verify_frozen(output, inputs, vectors, registration, banks)
        if (output/'qa_runtime').exists() or (output/'rows').exists():
            raise RuntimeError('Cannot select after answer generation has started')
        started = time.perf_counter()
        rows, files = {}, {}
        # One deterministic pass, never resolve a saved selection based on QA.
        for q in inputs['rows']:
            qid = q['qa_id']
            for arm in ARMS:
                path = output/'selections'/arm/(qid+'.json')
                if path.exists():
                    row = read(path)
                    verify_selection(inputs,q,arm,vectors[qid],banks[qid],row)
                else:
                    row = select_one(inputs,q,arm,vectors[qid],banks[qid])
                    save(path,row)
                rows[(arm,qid)] = row
                files[str(path)] = file_sha(path)
            print(json.dumps(dict(phase='freeze',qa_id=qid,completed=len(rows))),flush=True)
        paired = [(rows[('normal',q['qa_id'])], rows[('random',q['qa_id'])]) for q in inputs['rows']]
        chars = {a:distribution([rows[(a,q['qa_id'])]['context_chars'] for q in inputs['rows']]) for a in ARMS}
        ratio = chars['random']['mean']/chars['normal']['mean']
        summary = dict(status='complete', selections=564, created_utc=stamp(),
            registration_sha256=digest(registration), bank_complete_sha256=file_sha(output/'bank_complete.json'),
            files=files, seconds=time.perf_counter()-started, context_chars=chars,
            mean_random_normal_character_ratio=ratio,
            context_imbalance_flag=abs(ratio-1)>0.1,
            character_ratio_flag_is_diagnostic_not_a_new_selection_rule=True,
            actual_tokens='provider usage after QA; characters are not tokens',
            same_selected_ids=sum(a['selected_fact_ids']==b['selected_fact_ids'] for a,b in paired),
            same_reader_context=sum(a['retrieved']==b['retrieved'] for a,b in paired),
            selected_overlap=distribution([len(set(a['selected_fact_ids'])&set(b['selected_fact_ids'])) for a,b in paired]),
            contexts_review_required_before_qa=True, new_provider_calls=0)
        save(output/'freeze_summary.json',summary)
        return summary


def verify_selection(inputs,q,arm,vector,bank,row):
    qid,sid=q['qa_id'],q['sample_id']
    if any(row[k]!=q[k] for k in ('qa_id','sample_id','question')) or row['arm']!=arm or row['bank_sha256']!=digest(bank):
        raise RuntimeError('Frozen selection binding drift')
    origin=inputs['indices'][sid]['normal']
    candidates,trace=inputs['indices'][sid][arm]._candidate_pool(q['question'],vector)
    channels=channels_core.build_channels(origin,candidates,q['question'],bank)
    core.verify(row['selection'],candidates,inputs['graphs'][sid],q['question'],channels,20)
    ids=row['selection']['prefilter_fact_ids']
    if row['candidate_fact_ids']!=[c['fact']['id'] for c in candidates] or row['selected_fact_ids']!=ids:
        raise RuntimeError('Frozen pool/selection drift')
    base,assembly=origin._assemble([c for c in candidates if c['fact']['id'] in set(ids)],trace['top_summaries'])
    trace['assembly']=assembly
    trace['rerank']=dict(selected_fact_ids=ids,attempts=0,fallback=False,error=None,
        method='mathematical_coverage_maxmin_no_model_rerank',model_called=False)
    context,audit=materialize_online_operation_context(inputs['plans'][qid],
        dict(question=q['question'],retrieved=base,retrieval_trace=trace),inputs['graphs'][sid])
    if (row['retrieval_trace']!=trace or row['retrieved']!=context or row['reader_audit']!=audit
            or row['context_chars']!=len(context)):
        raise RuntimeError('Frozen original-graph Reader context changed')


def verify_frozen(output, inputs, vectors, registration, banks):
    summary=read(output/'freeze_summary.json')
    if (summary['status']!='complete' or summary['selections']!=564
            or summary['registration_sha256']!=digest(registration)
            or summary['bank_complete_sha256']!=file_sha(output/'bank_complete.json')):
        raise RuntimeError('Frozen manifest incomplete or changed')
    expected={str(output/'selections'/a/(q['qa_id']+'.json')) for q in inputs['rows'] for a in ARMS}
    if set(summary['files'])!=expected:
        raise RuntimeError('Frozen manifest census mismatch')
    for q in inputs['rows']:
        for arm in ARMS:
            path=output/'selections'/arm/(q['qa_id']+'.json')
            if file_sha(path)!=summary['files'][str(path)]:
                raise RuntimeError('Frozen selection file changed')
            verify_selection(inputs,q,arm,vectors[q['qa_id']],banks[q['qa_id']],read(path))
    return summary


def run(output, accepted_manifest):
    from filelock import FileLock
    inputs,vectors,registration,prompts=verify_registered(output)
    banks=verify_bank_complete(output,inputs,vectors,registration)
    verify_frozen(output,inputs,vectors,registration,banks)
    if accepted_manifest!=file_sha(output/'freeze_summary.json'):
        raise RuntimeError('Review and pass the exact frozen-context manifest SHA before QA')
    with FileLock(str(output/'dispatcher.lock'),timeout=0):
        if (output/'qa_runtime').exists() or (output/'rows').exists():
            raise RuntimeError('Already launched; inspect preserved output, never automatically restart')
        transport,chat,_=clients(output/'qa_runtime')
        save(output/'context_review_acceptance.json',dict(manifest_sha256=accepted_manifest,utc=stamp()))
        ns=dict(Any=Any,Mapping=Mapping,Sequence=Sequence,chat=chat,
            FIXED_WRITER_READER_MODEL=MODEL,CURRENT_SAMPLE_ID='shared_multihop',
            CURRENT_SAMPLE_INDEX=-1,QA_PROMPT=prompts['QA_PROMPT'])
        previous.load_function(ROOT/'creamem3/run_conv26_v2.py','_model_chat',ns)
        answer_one=previous.load_function(ROOT/'creamem3/run_conv26_v2.py','_answer_one',ns)
        jns=dict(chat=chat,os=os,re=re,JUDGE_PROMPT=prompts['JUDGE_PROMPT'],JUDGE_MODEL=JUDGE)
        for name in ('_thinking_budget','parse_verdict_detailed','judge_one_detailed'):
            previous.load_function(ROOT/'eval/judge.py',name,jns)
        judge=jns['judge_one_detailed']
        statuses=[]
        mutex=threading.Lock()
        started=time.perf_counter()

        def job(task):
            q,arm=task
            if transport.stop.is_set():
                return None
            qid=q['qa_id']
            transport.local.scope=arm+'/'+qid
            result=read(output/'selections'/arm/(qid+'.json'))
            result['status']='started'
            begin=time.perf_counter()
            try:
                save(output/'checkpoints'/arm/qid/'retrieval.json',result)
                tick=time.perf_counter()
                answer=answer_one(dict(qa=dict(qa_id=qid,question=q['question']),retrieved=result['retrieved']))
                if answer.get('qa_error') or not answer.get('response'):
                    raise RuntimeError('No valid QA response')
                result.update(prediction=answer['response'],status='answered')
                result['timings']['qa_seconds']=time.perf_counter()-tick
                save(output/'checkpoints'/arm/qid/'answer.json',result)
                tick=time.perf_counter()
                verdict,raw,parsed=judge(q['question'],inputs['gold'][qid]['answer'],result['prediction'])
                if not parsed:
                    raise RuntimeError('Invalid Judge output, not a zero')
                result.update(correct=int(verdict),judge_raw=raw,status='ok')
                result['timings']['judge_seconds']=time.perf_counter()-tick
            except BaseException as error:
                transport.stop.set()
                result.update(status='failed',error_type=type(error).__name__,
                    error='Inspect preserved calls/checkpoints; no policy zeros or automatic restart')
            finally:
                result['timings']['question_seconds']=time.perf_counter()-begin
                save(output/'rows'/arm/(qid+'.json'),result)
                with mutex:
                    statuses.append((arm,qid,result['status']))
                    print(json.dumps(dict(completed=len(statuses),arm=arm,qa_id=qid,status=result['status'])),flush=True)
            return result

        failure=None
        try:
            tasks=[(q,a) for q in inputs['rows'] for a in ARMS]
            with ThreadPoolExecutor(max_workers=registration['workers']) as pool:
                smoke=list(pool.map(job,tasks[:6]))
                if not all(r and r['status']=='ok' for r in smoke):
                    raise RuntimeError('First3 paired smoke has a missing/invalid output')
                save(output/'smoke_acceptance.json',dict(status='PASS',rows=6,
                    selection='first3 qa_id both arms; validity only, never judged correctness'))
                list(pool.map(job,tasks[6:]))
        except BaseException as error:
            transport.stop.set()
            failure=dict(error_type=type(error).__name__,message='No automatic restart; no missing output scored')
        calls,usage=transport.snapshots()
        summary=dict(status='complete' if len(statuses)==564 and all(s[2]=='ok' for s in statuses) else 'incomplete',
            rows_written=len(statuses),ok=sum(s[2]=='ok' for s in statuses),expected=564,
            elapsed_seconds=time.perf_counter()-started,failure=failure,calls=calls,usage=usage,
            physical_call_directories=[str(output/'bank_runtime/calls'),str(output/'qa_runtime/calls')])
        save(output/'run_summary.json',summary)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','prepare_bank','freeze','run'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--allow-model-calls',action='store_true')
    parser.add_argument('--accept-frozen-contexts')
    args=parser.parse_args()
    if not 1<=args.workers<=4:
        parser.error('workers must be1..4')
    if args.phase=='preflight':
        inputs,vectors=load_shared()
        result=dict(status='PASS',questions=len(inputs['rows']),q0_vectors=len(vectors),
                    source_hashes=code_sources(inputs),new_provider_calls=0)
    else:
        if not args.output:
            parser.error('--output required')
        output=args.output.resolve()
        if args.phase in ('prepare_bank','run') and not args.allow_model_calls:
            parser.error('paid phase requires --allow-model-calls and main-thread authorization')
        if args.phase=='prepare_bank':
            result=prepare_bank(output,args.workers)
        elif args.phase=='freeze':
            result=freeze(output)
        else:
            if not args.accept_frozen_contexts:
                parser.error('--accept-frozen-contexts manifest SHA required after length review')
            result=run(output,args.accept_frozen_contexts)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','calls','source_hashes')},indent=2),flush=True)
    return 0 if result['status'] in ('PASS','complete') else 2


if __name__=='__main__':
    raise SystemExit(main())

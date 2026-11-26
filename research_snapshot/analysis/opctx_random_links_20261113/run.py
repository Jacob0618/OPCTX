"""One fixed-seed routing intervention; original retrieval/Reader/QA functions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
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
SEED = '92620'
MODEL = 'gemini-2.5-flash'
JUDGE = 'gemini-3-flash-preview'
PERMS = Path(r'D:\opctx_review26\main_align\locomo_r2\permutations.json')
BASE = ROOT / 'creamem3/results_locomo10_v3'
FINAL = ROOT / 'creamem3/results_autoresearch_v8/online_operation_context_v3'

from creamem3.experiments.mec_transfer_20260906 import run_locomo as frozen
from analysis.opctx_reviewer_20261111.e2e.locomo_e2e import load_function, private_module
from creamem3.online_operation_context_v8 import materialize_online_operation_context


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


GRAPH_PATH = ROOT / 'creamem3/frozen_sources/locomo81_graph_memory_v2.py'
assert file_sha(GRAPH_PATH) == frozen.FROZEN_CODE_SHA
gm = private_module(GRAPH_PATH, '_random_links_original_graph')


def make_routing_index(graph, permutation):
    index = gm.GraphMemoryIndex(graph)
    for node in index.summaries:
        if node.get('node_type') == 'event_view':
            index.adjacency[node['id']] = [permutation[fid] for fid in index.adjacency[node['id']]]
    return index


def load_inputs():
    graphs, indices, rows, plans, gold, historical, hashes = {}, {}, [], {}, {}, {}, {}
    permutations = read(PERMS)
    assert file_sha(FINAL / 'aggregate.json') == frozen.BEST_SHA
    dataset = ROOT / 'data/locomo/locomo10.json'
    assert file_sha(dataset) == frozen.DATA_SHA
    for sid in sorted(frozen.GRAPH_SHA):
        gp = BASE / sid / 'graph_online.json'
        assert file_sha(gp) == frozen.GRAPH_SHA[sid]
        graph = read(gp)
        graphs[sid] = graph
        indices[sid] = {'normal': gm.GraphMemoryIndex(graph),
                        'random': make_routing_index(graph, permutations[sid][SEED])}
        final_path = FINAL / sid / 'online_operation_context_v3.json'
        original = read(final_path)
        hashes[str(gp)] = file_sha(gp)
        hashes[str(final_path)] = file_sha(final_path)
        for row in original['rows']:
            if row['category_name'] != 'multi':
                continue
            qid = row['qa_id']
            rows.append(dict(qa_id=qid, sample_id=sid, question=row['question']))
            plans[qid] = row['retrieval_trace']['online_operation_context_v8']['strict_operation']
            gold[qid] = dict(answer=row['answer'], evidence=row.get('evidence_dia_ids', row.get('evidence', [])))
            historical[qid] = dict(correct=row['correct'], selected_fact_ids=row['retrieval_trace']['online_operation_context_v8']['materializer_audit']['selected_fact_ids'])
    rows.sort(key=lambda r:r['qa_id'])
    assert len(rows) == len(plans) == len(gold) == 282
    assert sum(x['correct'] for x in historical.values()) == 208
    for p in [GRAPH_PATH, PERMS, ROOT / 'eval/gemini_client.py', ROOT / 'eval/judge.py',
              ROOT / 'creamem3/run_conv26_v2.py', ROOT / 'creamem3/online_operation_context_v8.py',
              ROOT / 'creamem3/semantic_context_v8.py', Path(__file__), Path(__file__).with_name('protocol.md')]:
        hashes[str(p)] = file_sha(p)
    return dict(graphs=graphs, indices=indices, rows=rows, plans=plans, gold=gold,
                historical=historical, metadata=dict(source_hashes=hashes, seed=int(SEED)))


def preflight():
    import test_random_links
    inputs = load_inputs()
    audit = test_random_links.audit(make_routing_index)
    prompts = frozen.frozen_prompts()
    assert audit['status'] == 'PASS'
    return inputs, prompts, audit


def run(output, workers):
    from transport import AuditTransport, make_clients
    from filelock import FileLock
    inputs, prompts, audit = preflight()
    output.mkdir(parents=True, exist_ok=False)
    mutex = threading.Lock()
    start = time.perf_counter()
    os.environ['JUDGE_THINKING'] = 'off'
    os.environ['GEMINI3_THINKING_LEVEL'] = 'minimal'
    registration = dict(**inputs['metadata'], created_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), workers=workers, arms=['normal','random'], questions=282,
        qa_model=MODEL, judge_model=JUDGE, embedding='gemini-embedding-001',
        intervention='event-view retrieval-routing adjacency only; original graph used for Reader',
        total_event_views=243, total_event_links=5567, candidate_cap=120, final_fact_count=20,
        summaries=10, dense=20, bm25=20, source_reuse='frozen memory and question-only operation labels',
        query_vectors='fresh shared RETRIEVAL_QUERY 768', historical_multi_reference=208/282,
        source_models='historical frozen memory, not rebuilt', semantic_retry_limit=2,
        network_extra_retry_limit=1, score_based_retries=False, writer_calls=0, online_calls=0, planner_calls=0)
    registration['source_hashes'][str(Path(__file__).with_name('transport.py'))] = file_sha(Path(__file__).with_name('transport.py'))
    save(output / 'registration.json', registration)
    save(output / 'offline_audit.json', audit)
    save(output / 'questions.json', inputs['rows'])
    save(output / 'question_operations.json', inputs['plans'])
    save(output / 'historical_reference.json', inputs['historical'])
    save(output / 'fixed_permutations.json', {sid:read(PERMS)[sid][SEED] for sid in inputs['graphs']})
    transport = AuditTransport(output)
    chat, embed = make_clients(transport)
    ns = dict(Any=Any, Mapping=Mapping, Sequence=Sequence, chat=chat, FIXED_WRITER_READER_MODEL=MODEL,
              CURRENT_SAMPLE_ID='shared_multihop', CURRENT_SAMPLE_INDEX=-1, QA_PROMPT=prompts['QA_PROMPT'])
    model_chat = load_function(ROOT/'creamem3/run_conv26_v2.py', '_model_chat', ns)
    answer_one = load_function(ROOT/'creamem3/run_conv26_v2.py', '_answer_one', ns)
    jns = dict(chat=chat, os=os, re=re, JUDGE_PROMPT=prompts['JUDGE_PROMPT'], JUDGE_MODEL=JUDGE)
    for name in ('_thinking_budget','parse_verdict_detailed','judge_one_detailed'):
        load_function(ROOT/'eval/judge.py', name, jns)
    judge = jns['judge_one_detailed']
    statuses = []

    def job(task):
        question, arm, vector = task
        qid, sid = question['qa_id'], question['sample_id']
        transport.local.scope = arm + '/' + qid
        result = dict(**question, arm=arm, status='started', timings={})
        if transport.stop.is_set():
            return None
        begin = time.perf_counter()
        try:
            origin = inputs['indices'][sid]['normal']
            routing = inputs['indices'][sid][arm]
            tick = time.perf_counter()
            candidates, trace = routing._candidate_pool(question['question'], vector)
            _, normal_trace = origin._candidate_pool(question['question'], vector)
            for key in ('policy','top_summaries','dense_fact_ids','bm25_facts'):
                assert trace[key] == normal_trace[key], key
            result['candidate_fact_ids'] = [x['fact']['id'] for x in candidates]
            assert 20 <= len(candidates) <= 120
            result['timings']['candidate_seconds'] = time.perf_counter()-tick
            tick = time.perf_counter()
            selected, rr = origin._rerank(question['question'], candidates, model_chat)
            result['timings']['rerank_seconds'] = time.perf_counter()-tick
            if rr.get('fallback'):
                raise RuntimeError('Original reranker exhausted validation; stop rather than silently accept fallback')
            result['selected_fact_ids'] = [x['fact']['id'] for x in selected]
            assert len(set(result['selected_fact_ids'])) == 20
            trace['rerank'] = dict(**rr, selected_fact_ids=result['selected_fact_ids'])
            tick = time.perf_counter()
            base, assembly = origin._assemble(selected, trace['top_summaries'])
            trace['assembly'] = assembly
            context, ra = materialize_online_operation_context(inputs['plans'][qid],
                dict(question=question['question'], retrieved=base, retrieval_trace=trace), inputs['graphs'][sid])
            result['timings']['reader_seconds'] = time.perf_counter()-tick
            assert ra['selected_fact_ids'] == result['selected_fact_ids']
            result.update(reader_audit=ra, retrieval_trace=trace, retrieved=context)
            save(output/'checkpoints'/arm/qid/'retrieval.json', result)
            tick = time.perf_counter()
            answer = answer_one(dict(qa=dict(qa_id=qid,question=question['question']), retrieved=context))
            if answer.get('qa_error') or not answer.get('response'):
                raise RuntimeError('QA has no valid answer')
            result.update(prediction=answer['response'], status='answered')
            result['timings']['qa_seconds'] = time.perf_counter()-tick
            save(output/'checkpoints'/arm/qid/'answer.json', result)
            tick = time.perf_counter()
            verdict, raw, parsed = judge(question['question'], inputs['gold'][qid]['answer'], result['prediction'])
            if not parsed:
                raise RuntimeError('Invalid judge output; do not score as zero')
            result.update(correct=int(verdict), judge_raw=raw, status='ok')
            result['timings']['judge_seconds'] = time.perf_counter()-tick
        except BaseException as exc:
            transport.stop.set()
            result.update(status='failed', error_type=type(exc).__name__, error='See preserved physical call outcomes; no automatic runner restart')
        finally:
            result['timings']['question_seconds'] = time.perf_counter()-begin
            save(output/'rows'/arm/(qid+'.json'), result)
            with mutex:
                statuses.append((arm,qid,result['status']))
                print(json.dumps(dict(completed=len(statuses),arm=arm,qa_id=qid,status=result['status'])), flush=True)
        return result

    failure = None
    try:
        with FileLock(str(output/'dispatcher.lock'), timeout=0):
            transport.local.scope = 'shared_query_embeddings'
            tick = time.perf_counter()
            vectors = embed([q['question'] for q in inputs['rows']], model='gemini-embedding-001',
                tag='embed_query', task_type='RETRIEVAL_QUERY', dim=768)
            assert len(vectors) == 282 and all(len(v)==768 for v in vectors)
            save(output/'query_vectors.json', dict(ids=[q['qa_id'] for q in inputs['rows']],vectors=vectors,
                seconds=time.perf_counter()-tick, model='gemini-embedding-001'))
            tasks = [(q, arm, v) for q,v in zip(inputs['rows'],vectors) for arm in ('normal','random')]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                smoke = list(pool.map(job,tasks[:6]))
                if not all(r and r['status']=='ok' for r in smoke):
                    raise RuntimeError('Smoke incomplete; remaining questions not dispatched')
                save(output/'smoke_acceptance.json', dict(status='PASS',rows=6,selection='first3 qa_id, both arms; not selected by outcomes'))
                list(pool.map(job,tasks[6:]))
    except BaseException as exc:
        transport.stop.set()
        failure = dict(error_type=type(exc).__name__, message='Inspect preserved outcomes; no missing rows are scored')
    calls, usage = transport.snapshots()
    save(output/'run_summary.json', dict(status='complete' if len(statuses)==564 and all(s[2]=='ok' for s in statuses) else 'incomplete',
        rows_written=len(statuses),ok=sum(s[2]=='ok' for s in statuses),expected=564,
        elapsed_seconds=time.perf_counter()-start,failure=failure,usage=usage,calls=calls))
    print('FINISHED',len(statuses),'ok',sum(s[2]=='ok' for s in statuses),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('workers must be 1..8')
    if args.run:
        if not args.output:
            p.error('--output required')
        run(args.output.resolve(),args.workers)
    else:
        _,_,audit = preflight()
        print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()

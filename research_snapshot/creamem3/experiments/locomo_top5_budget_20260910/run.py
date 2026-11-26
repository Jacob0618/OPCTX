"""Four authorized Top5 arms; fixed memory and write-once read-side receipts."""
import argparse
from concurrent.futures import as_completed
import copy
import json
from pathlib import Path
import statistics
import threading
import time

from creamem3.experiments.locomo_mgtm480_writer_20260909 import enum_resume as source
from creamem3 import online_operation_context_v8 as materializer

full, old, base, original = source.full, source.old, source.base, source.original
read, seal, sha, file_sha, StopRun = source.read, source.seal, source.sha, source.file_sha, source.StopRun
SOURCE = source.RUN
RUN = SOURCE.parent / 'locomo_mgtm480_top5_budgets_full1540_20260910_r1'
ARMS = ('session_strong', 'opctx_original')
CONFIGS = {
    'wide': dict(summary_top_k=10, dense_fact_top_k=20, bm25_fact_top_k=20, candidate_cap=120, rerank_top_k=5),
    'narrow': dict(summary_top_k=2, dense_fact_top_k=5, bm25_fact_top_k=5, candidate_cap=120, rerank_top_k=5),
}


class Runtime(source.enum_repair.Runtime):
    def __init__(self, root, *, allow_model_calls, stopped=None):
        super().__init__(root, allow_model_calls=allow_model_calls)
        self.parents = ()
        self.local = threading.local()
        if stopped is not None:
            self.stopped = stopped

    def post(self, stage, scope, model, method, body):
        result = super().post(stage, scope, model, method, body)
        entry = {'stage': stage, 'key': sha(dict(stage=stage, scope=scope, model=model, method=method, body=body)),
                 'usage': result.get('usageMetadata', {})}
        if hasattr(self.local, 'requests'):
            self.local.requests.append(entry)
        return result


def condition_root(condition):
    return RUN / 'conditions' / condition


def validate_selection(ids, candidates, n=5):
    legal = {c['fact']['id'] for c in candidates}
    if len(ids) != n or len(set(ids)) != n or not set(ids) <= legal:
        raise StopRun('Wrong selection count, duplicate or out-of-candidate ID')


def render(index, candidates, trace, graph, q, arm, operation):
    ids = trace['rerank']['selected_fact_ids']
    by_id = {c['fact']['id']: c for c in candidates}
    if trace['rerank']['fallback'] or len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise StopRun('Invalid stored selection')
    context, assembly = index._assemble([by_id[i] for i in ids], trace['top_summaries'])
    if assembly != trace['assembly']:
        raise StopRun('Assembly trace mismatch')
    audit = None
    if arm == 'opctx_original':
        context, audit = old.materialize_online_operation_context(operation,
            {'question': q['question'], 'retrieved': context, 'retrieval_trace': copy.deepcopy(trace)}, graph)
        if audit['selected_fact_ids'] != ids or audit['selected_fact_count'] != len(ids):
            raise StopRun('Materializer changed actual selection')
    return context, audit


def source_check():
    completed = read(SOURCE/'complete.json')
    parent = read(SOURCE/'registration.json')
    protocol = read(SOURCE/'protocol.json')
    if (completed['registration_sha256'] != sha(parent) or not completed['audit']['pass']
            or completed['selection_replays'] != 3080 or completed['qa_judge_bindings'] != 6160):
        raise StopRun('Source full run is not fully audited')
    for path, digest in parent['source_hashes'].items():
        if file_sha(path) != digest:
            raise StopRun('Source code drift: ' + path)
    for path, key in [(old.DEFAULT_DATASET_PATH, 'dataset_sha256'), (old.VECTOR_FILE, 'query_vector_file_sha256')]:
        if file_sha(path) != protocol[key]:
            raise StopRun('Source input drift: ' + str(path))
    loaded = old.load_all_locomo()
    questions = {q['qa_id']: q for c in loaded for q in c['qa']}
    samples = {q['qa_id']: c['sample_id'] for c in loaded for q in c['qa']}
    if len(questions) != 1540 or set(questions) != set(protocol['all_ids']):
        raise StopRun('Question population drift')
    vr = read(old.VECTOR_FILE)
    if len(vr['ids']) != 1540 or set(vr['ids']) != set(questions):
        raise StopRun('Question vector population drift')
    vectors = dict(zip(vr['ids'], vr['vectors']))
    graphs, graph_hashes, indexes = {}, {}, {}
    paths = [SOURCE/n for n in ('complete.json', 'registration.json', 'protocol.json')]
    paths += [Path(old.DEFAULT_DATASET_PATH), Path(old.VECTOR_FILE)]
    for conv in loaded:
        for arm in ARMS:
            key = (conv['sample_id'], arm)
            graph = full.get_graph(SOURCE, *key, parent['source_registration_sha256'])
            graphs[key] = graph
            graph_hashes[key] = sha(graph)
            for condition, config in CONFIGS.items():
                indexes[(condition, *key)] = source.enum_repair.Index(graph, **config)
            paths += [original.graph_path(SOURCE, *key), SOURCE/'memory_imports'/key[0]/(arm+'.json')]
        sg, og = (graphs[(conv['sample_id'], a)] for a in ARMS)
        if sg['fact_nodes'] != og['fact_nodes'] or sg['session_packets'] != og['session_packets']:
            raise StopRun('Shared Writer equality failed')
    operations = {}
    for qid in sorted(questions):
        p = old.BASE_CONTEXT/(qid+'.json')
        operations[qid] = old.old_json(p)['operation']
        materializer.context_policy_for_operation(operations[qid])
        paths.append(p)
    baselines = {a: {} for a in ARMS}
    for arm in ARMS:
        if {p.stem for p in (SOURCE/'rows'/arm).glob('*.json')} != set(questions):
            raise StopRun('Source rows incomplete')
        for qid, q in questions.items():
            row_path, retrieval_path = SOURCE/'rows'/arm/(qid+'.json'), SOURCE/'retrieval'/arm/(qid+'.json')
            row, retrieval = read(row_path), read(retrieval_path)
            sample = samples[qid]
            expected = {'protocol_sha256': sha(protocol), 'qa_sha256': sha(q), 'arm': arm,
                        'graph_sha256': graph_hashes[(sample, arm)], 'vector_sha256': sha(vectors[qid])}
            if row['identity'] != expected:
                raise StopRun('Baseline identity drift')
            index = indexes[('wide', sample, arm)]
            candidates, trace = index._candidate_pool(q['question'], vectors[qid])
            trace['policy']['rerank_top_k'] = 20
            if trace != {k: retrieval['trace'][k] for k in trace}:
                raise StopRun('Baseline candidate replay drift')
            validate_selection(row['selected_fact_ids'], candidates, 20)
            context, audit = render(index, candidates, retrieval['trace'], graphs[(sample, arm)], q, arm, operations[qid])
            if (context != row['context'] or context != retrieval['context'] or audit != retrieval['materializer']
                    or row['selected_fact_ids'] != retrieval['trace']['rerank']['selected_fact_ids']
                    or row['context_sha256'] != sha(context)):
                raise StopRun('Baseline context replay drift')
            base.check_bound(SOURCE, q, row)
            baselines[arm][qid] = row
            paths += [row_path, retrieval_path]
        old.emit('TOP5 BASELINE_REPLAY ' + arm + ' 1540 rows')
    frozen = {str(p): file_sha(p) for p in sorted(set(paths))}
    return parent, protocol, questions, samples, vectors, graphs, graph_hashes, indexes, operations, baselines, frozen


def prepare():
    (parent, protocol, questions, samples, vectors, graphs, graph_hashes, indexes, operations, baselines, frozen) = source_check()
    code = [Path(__file__), Path(__file__).with_name('protocol.md')]
    registration = {'source_root': str(SOURCE), 'source_snapshot_sha256': sha(frozen),
        'source_hashes': {**parent['source_hashes'], **{str(p): file_sha(p) for p in code}},
        'configurations': CONFIGS, 'arms': list(ARMS), 'all_ids': sorted(questions),
        'questions_per_arm': 1540, 'total_rows': 6160, 'workers_total': 20,
        'source_baseline_correct': {a: sum(r['correct'] for r in baselines[a].values()) for a in ARMS},
        'qa_model': 'gemini-2.5-flash', 'rerank_model': 'gemini-2.5-flash',
        'judge_model': 'gemini-3-flash-preview', 'judge_votes': 1,
        'qa_prompt_sha256': sha(old.QA_PROMPT), 'judge_prompt_sha256': sha(old.JUDGE_PROMPT),
        'rerank_prompt_sha256': sha(old.graph_api.FACT_RERANK_PROMPT),
        'new_memory_or_embedding_or_planner_calls': False, 'historical_response_imports': False,
        'shared_identical_requests_within_condition': True, 'separate_condition_caches': True,
        'actual_final_fact_count': 5, 'legacy_materializer_labels_may_say_top20': True,
        'not_token_matched': True, 'candidate_cap_unchanged': 120,
        'no_score_based_promotion_or_unbounded_search': True}
    seal(RUN/'source_snapshot.json', frozen)
    seal(RUN/'registration.json', registration)
    seal(RUN/'preflight.json', {'registration_sha256': sha(registration), 'baseline_context_replays': 3080,
         'baseline_qa_judge_bindings': 6160, 'model_calls': 0})
    return dict(registration=registration, questions=questions, samples=samples, vectors=vectors,
                graphs=graphs, graph_hashes=graph_hashes, indexes=indexes, operations=operations, baselines=baselines)


def row_identity(data, condition, arm, qid):
    sample = data['samples'][qid]
    return {'registration_sha256': sha(data['registration']), 'qa_sha256': sha(data['questions'][qid]),
            'graph_sha256': data['graph_hashes'][(sample, arm)], 'vector_sha256': sha(data['vectors'][qid]),
            'condition': condition, 'arm': arm}


def retrieve(data, condition, arm, qid, runtime):
    q, sample = data['questions'][qid], data['samples'][qid]
    index, graph = data['indexes'][(condition, sample, arm)], data['graphs'][(sample, arm)]
    context, trace = index.search(q['question'], data['vectors'][qid], runtime.chat_fn(arm+'/'+qid))
    candidates, expected = index._candidate_pool(q['question'], data['vectors'][qid])
    if expected != {k: trace[k] for k in expected}:
        raise StopRun('Candidate replay drift')
    validate_selection(trace['rerank']['selected_fact_ids'], candidates)
    context, audit = render(index, candidates, trace, graph, q, arm, data['operations'][qid])
    return {'identity': row_identity(data, condition, arm, qid), 'context': context, 'trace': trace,
            'materializer': audit, 'operation': data['operations'][qid], 'actual_final_fact_count': 5,
            'candidate_document_chars': sum(len(c['fact']['text']) for c in candidates)}


def unique_requests(requests):
    found = {}
    for entry in requests:
        key = (entry['stage'], entry['key'])
        if key in found and found[key] != entry:
            raise StopRun('Request usage drift')
        found[key] = entry
    return [found[key] for key in sorted(found)]


def evaluate_one(data, condition, arm, qid, runtime):
    root = condition_root(condition)
    path = root/'rows'/arm/(qid+'.json')
    identity = row_identity(data, condition, arm, qid)
    if path.exists():
        row = read(path)
        if row['identity'] != identity:
            raise StopRun('Existing row identity drift')
        return row
    runtime.local.requests = []
    runtime.log({'kind': 'row_start', 'scope': arm+'/'+qid, 'condition': condition})
    started = time.perf_counter()
    retrieval = retrieve(data, condition, arm, qid, runtime)
    seal(root/'retrieval'/arm/(qid+'.json'), retrieval)
    q = data['questions'][qid]
    answer = runtime.chat(qid, 'qa', old.QA_PROMPT.format(question=q['question'], retrieved_texts=retrieval['context']), max_tokens=8192)
    correct, judge = old.judge(runtime, qid, q, answer)
    requests = unique_requests(runtime.local.requests)
    qa_requests = [r for r in requests if r['stage'] == 'qa']
    if len(qa_requests) != 1:
        raise StopRun('Expected one QA request')
    row = {'identity': identity, 'qa_id': qid, 'sample_id': data['samples'][qid], 'arm': arm, 'condition': condition,
           'question': q['question'], 'answer': q['answer'], 'category_name': q['category_name'],
           'response': answer, 'correct': correct, 'judge_raw': judge,
           'context': retrieval['context'], 'context_sha256': sha(retrieval['context']),
           'context_chars': len(retrieval['context']), 'selected_fact_ids': retrieval['trace']['rerank']['selected_fact_ids'],
           'candidate_count': retrieval['trace']['candidates_after_cap'], 'union_before_cap': retrieval['trace']['union_before_cap'],
           'candidate_document_chars': retrieval['candidate_document_chars'],
           'qa_prompt_tokens': qa_requests[0]['usage'].get('promptTokenCount'),
           'logical_request_receipts': requests, 'elapsed_s': time.perf_counter()-started}
    seal(path, row)
    runtime.log({'kind': 'row_complete', 'scope': arm+'/'+qid, 'condition': condition, 'correct': correct})
    old.emit(f'TOP5 ROW {condition}/{arm}/{qid} correct={correct} candidates={row["candidate_count"]}')
    return row


def smoke_ids(data):
    selected = {}
    for qid in sorted(data['questions']):
        policy = materializer.context_policy_for_operation(data['operations'][qid])
        selected.setdefault(policy, qid)
    if set(selected) != {materializer.ORIGINAL_FULL, materializer.BOUNDED_VIEWS_2, materializer.ROUTE_ONLY}:
        raise StopRun('Smoke does not cover three renderer branches')
    return sorted(selected.values())


def check_unchanged(data):
    for path, digest in read(RUN/'source_snapshot.json').items():
        if file_sha(path) != digest:
            raise StopRun('Source artifact changed: ' + path)
    for path, digest in data['registration']['source_hashes'].items():
        if file_sha(path) != digest:
            raise StopRun('Frozen code changed: ' + path)


def summarize(rows):
    values = list(rows.values())
    result = {'n': len(values), 'correct': sum(r['correct'] for r in values), 'by_category': {}}
    result['accuracy'] = 100*result['correct']/len(values)
    for category in sorted({r['category_name'] for r in values}):
        group = [r for r in values if r['category_name'] == category]
        correct = sum(r['correct'] for r in group)
        result['by_category'][category] = {'n': len(group), 'correct': correct, 'accuracy': 100*correct/len(group)}
    for key in ('context_chars', 'candidate_count', 'union_before_cap', 'candidate_document_chars', 'qa_prompt_tokens'):
        known = [r[key] for r in values if r[key] is not None]
        result['mean_'+key] = statistics.mean(known) if known else None
        result[key+'_observed_rows'] = len(known)
    return result


def paired(a, b, ids):
    result = base.pair_summary([(a[q], b[q]) for q in ids])
    result['mean_selected_id_overlap'] = result.pop('mean_top20_overlap')
    return result


def finish(data, ids, smoke=False):
    rows, transport = {}, {}
    for condition in CONFIGS:
        root = condition_root(condition)
        runtime = Runtime(root/'calls', allow_model_calls=False)
        rows[condition] = {}
        for arm in ARMS:
            rows[condition][arm] = {}
            if not smoke and {p.stem for p in (root/'rows'/arm).glob('*.json')} != set(ids):
                raise StopRun('Final row population incomplete')
            for qid in ids:
                row = read(root/'rows'/arm/(qid+'.json'))
                stored = read(root/'retrieval'/arm/(qid+'.json'))
                runtime.local.requests = []
                replay = retrieve(data, condition, arm, qid, runtime)
                if (row['identity'] != row_identity(data, condition, arm, qid) or replay != stored
                        or row['context'] != stored['context'] or row['context_sha256'] != sha(row['context'])
                        or row['selected_fact_ids'] != replay['trace']['rerank']['selected_fact_ids']
                        or row['candidate_count'] != replay['trace']['candidates_after_cap']
                        or row['union_before_cap'] != replay['trace']['union_before_cap']
                        or row['candidate_document_chars'] != replay['candidate_document_chars']
                        or row['context_chars'] != len(row['context'])):
                    raise StopRun('Final retrieval/row replay mismatch')
                base.check_bound(root, data['questions'][qid], row)
                for entry in row['logical_request_receipts']:
                    _, _, _, raw = full.recover.completed_receipt(root/'calls/provider'/entry['stage']/entry['key'])
                    if raw['response'].get('usageMetadata', {}) != entry['usage']:
                        raise StopRun('Per-row usage binding failed')
                expected_qa = [e for e in row['logical_request_receipts'] if e['stage'] == 'qa']
                qa_identity = base.binding.generation_identity('qa', old.QA_PROMPT.format(
                    question=data['questions'][qid]['question'], retrieved_texts=row['context']), 8192)
                if (len(expected_qa) != 1 or expected_qa[0]['key'] != sha(qa_identity)
                        or row['qa_prompt_tokens'] != expected_qa[0]['usage'].get('promptTokenCount')):
                    raise StopRun('QA token binding failed')
                rows[condition][arm][qid] = row
            old.emit(f'TOP5 AUDIT {condition}/{arm} {len(ids)} rows')
        if (root/'calls/imports').exists():
            raise StopRun('Historical response imports forbidden')
        transport[condition] = full.transport_audit(root)
    check_unchanged(data)
    result = {'registration_sha256': sha(data['registration']), 'smoke_only': smoke,
              'rows_per_arm': len(ids), 'selection_replays': 4*len(ids), 'qa_judge_bindings': 8*len(ids),
              'audit_pass': True, 'reports': {c: {a: summarize(rows[c][a]) for a in ARMS} for c in CONFIGS},
              'paired_online_vs_session': {c: paired(rows[c]['opctx_original'], rows[c]['session_strong'], ids) for c in CONFIGS},
              'paired_narrow_vs_wide': {a: paired(rows['narrow'][a], rows['wide'][a], ids) for a in ARMS},
              'paired_vs_original_top20': {c: {a: paired(rows[c][a], data['baselines'][a], ids) for a in ARMS} for c in CONFIGS},
              'physical_transport': transport, 'not_token_matched': True,
              'old_scores_unchanged': True, 'legacy_top20_renderer_labels_not_actual_selection_count': True}
    seal(RUN/('smoke_complete.json' if smoke else 'complete.json'), result)
    old.emit('TOP5 '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(result['reports']))
    if not smoke:
        for condition in CONFIGS:
            for arm in ARMS:
                old.emit(f'METRIC {condition}_{arm}_accuracy={result["reports"][condition][arm]["accuracy"]:.8f}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'smoke', 'run', 'audit'])
    command = parser.parse_args().command
    with old.Lease(RUN):
        data = prepare()
        if command == 'prepare':
            return
        if command == 'audit':
            finish(data, sorted(data['questions']))
            return
        if command == 'run':
            smoke = read(RUN/'smoke_complete.json')
            if smoke['registration_sha256'] != sha(data['registration']) or not smoke['audit_pass']:
                raise StopRun('Smoke contract mismatch')
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/'.env', override=False)
        stopped = threading.Event()
        runtimes = {c: Runtime(condition_root(c)/'calls', allow_model_calls=True, stopped=stopped) for c in CONFIGS}
        control = Runtime(RUN/'control_calls', allow_model_calls=False, stopped=stopped)
        previous = original.adapter.install(control.log)
        ids = smoke_ids(data) if command == 'smoke' else sorted(data['questions'])
        try:
            # Interleave conditions and memory arms; no score-dependent order.
            with source.enum_repair.FailFastPool(control, max_workers=20) as pool:
                futures = [pool.submit(evaluate_one, data, c, a, q, runtimes[c])
                           for q in ids for c in CONFIGS for a in ARMS]
                for future in as_completed(futures):
                    future.result()
            finish(data, ids, smoke=command == 'smoke')
        except BaseException as exc:
            control.record_failure(exc)
            seal(RUN/('stop_%d.json' % time.time_ns()), {'utc': base.now(), 'type': type(exc).__name__, 'message': str(exc)[:600]})
            raise
        finally:
            original.adapter.restore(previous)


if __name__ == '__main__':
    main()

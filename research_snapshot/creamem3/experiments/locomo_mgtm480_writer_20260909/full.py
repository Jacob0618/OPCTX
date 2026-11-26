"""Fixed full1540 comparison, promoting dev100 without rebuilding memory."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

from creamem3.experiments.locomo_mgtm480_writer_20260909 import recover

original, old, base = recover.original, recover.old, recover.base
read, seal, sha, file_sha, StopRun = recover.read, recover.seal, recover.sha, recover.file_sha, recover.StopRun
SOURCE = recover.RUN
RUN = SOURCE.parent/'locomo_mgtm480_writer_full1540_20260910_r1'
ARMS = original.ARMS


class Runtime(base.Runtime):
    def __init__(self, root, *, allow_model_calls):
        super().__init__(root, parents=(SOURCE,), allow_model_calls=allow_model_calls, concurrency=20, timeout=240)


def population(protocol):
    all_ids, dev_ids = protocol['all_ids'], protocol['dev_ids']
    if len(all_ids) != 1540 or len(set(all_ids)) != 1540 or len(dev_ids) != 100 or len(set(dev_ids)) != 100:
        raise StopRun('Unexpected full/dev population size or duplicate IDs')
    if not set(dev_ids) <= set(all_ids): raise StopRun('Dev IDs not a subset of full population')
    return set(all_ids), set(dev_ids)


def rebind_row(row, parent_protocol, protocol):
    if row['identity']['protocol_sha256'] != sha(parent_protocol): raise StopRun('Source row protocol drift')
    result = copy.deepcopy(row)
    result['identity']['protocol_sha256'] = sha(protocol)
    return result


def snapshot(protocol):
    files = [SOURCE/name for name in ('complete.json', 'registration.json', 'protocol.json', 'prompt_bundle.json',
                                    'source_sessions.json', 'source_snapshot.json')]
    for name in ('graphs', 'rows', 'retrieval'):
        files.extend(sorted((SOURCE/name).rglob('*.json')))
    for stage in ('qa', 'judge', 'rerank'):
        files.extend(sorted((SOURCE/'calls/provider'/stage).rglob('*.json')))
    files.extend(old.BASE_CONTEXT/(qid+'.json') for qid in protocol['all_ids'])
    return {str(p): file_sha(p) for p in files}


def get_graph(root, sample, arm, parent_sha):
    record = read(original.graph_path(root, sample, arm))
    if record['identity']['registration_sha256'] != parent_sha: raise StopRun('Imported graph registration drift')
    origin = read(root/'memory_imports'/sample/(arm+'.json'))
    if origin['record_sha256'] != sha(record): raise StopRun('Imported graph content drift')
    return record['graph']


def prepare(root=RUN):
    parent = read(SOURCE/'registration.json'); parent_protocol = read(SOURCE/'protocol.json')
    complete = read(SOURCE/'complete.json'); _, dev = population(parent_protocol)
    if complete['registration_sha256'] != sha(parent) or not complete['audit']['pass']:
        raise StopRun('Source dev100 is not completely audited')
    if complete['selection_replays'] != 200 or complete['qa_judge_bindings'] != 400:
        raise StopRun('Incomplete source read-side audit')
    for path, digest in parent['source_hashes'].items():
        if file_sha(path) != digest: raise StopRun('Frozen source code changed: ' + path)
    loaded, control = old.prepare(root/'original_control_protocol')
    if sha(control) != parent['original_control_protocol_sha256']: raise StopRun('Control protocol drift')
    vectors = read(old.VECTOR_FILE)
    if set(vectors['ids']) != set(control['all_ids']) or len(vectors['ids']) != 1540:
        raise StopRun('Missing/duplicate full question vectors')
    for arm in ARMS:
        paths = list((SOURCE/'rows'/arm).glob('*.json'))
        if {p.stem for p in paths} != dev: raise StopRun('Source rows differ from complete dev set')
    frozen = snapshot(parent_protocol); seal(root/'source_snapshot.json', frozen)
    registration = {'source_root': str(SOURCE), 'source_registration_sha256': sha(parent),
        'source_complete_sha256': sha(complete), 'source_snapshot_sha256': sha(frozen),
        'source_hashes': {**parent['source_hashes'], str(Path(__file__)): file_sha(__file__),
                         str(Path(__file__).with_name('full_protocol.md')): file_sha(Path(__file__).with_name('full_protocol.md'))},
        'arms': list(ARMS), 'questions_per_arm': 1540, 'reused_dev_rows_per_arm': 100,
        'new_questions_per_arm': 1440, 'no_new_memory_or_planner_calls': True,
        'qa_workers': 20, 'provider_concurrency': 20, 'timeout_s': 240,
        'same_retrieval_qa_judge_and_shared_writer': True, 'not_equal_token_budget': True,
        'memory_is_mixed_cache_cap_recovery': True, 'separately_report_remaining': True}
    seal(root/'registration.json', registration)
    protocol = {**parent_protocol, 'full_promotion_registration_sha256': sha(registration)}
    seal(root/'protocol.json', protocol)
    for conv in loaded:
        sample = conv['sample_id']; audit = read(SOURCE/'graphs'/sample/'audit.json')
        if any(c['status'] != 'pass' for c in audit['checks'].values()): raise StopRun('Source graph audit failed')
        for arm, digest_name in (('session_strong', 'session_sha256'), ('opctx_original', 'online_sha256')):
            p = original.graph_path(SOURCE, sample, arm); record = read(p)
            if record['identity']['registration_sha256'] != sha(parent) or sha(record['graph']) != audit[digest_name]:
                raise StopRun('Source graph hash/registration mismatch')
            seal(original.graph_path(root, sample, arm), record)
            seal(root/'memory_imports'/sample/(arm+'.json'), {'source': str(p), 'record_sha256': sha(record),
                'audit_sha256': sha(audit), 'graph_unchanged': True})
    # Import only this dev run's exact read-side receipts, including any invalid
    # format attempts that preceded a successful bounded rerank/Judge repair.
    importer = Runtime(root/'calls', allow_model_calls=False)
    for stage in ('qa', 'judge', 'rerank'):
        for path in sorted((SOURCE/'calls/provider'/stage).glob('*/request.json')):
            request, _, _, _ = recover.completed_receipt(path.parent)
            importer.post(**request)
    for arm in ARMS:
        for qid in sorted(dev):
            path = SOURCE/'rows'/arm/(qid+'.json'); row = read(path)
            imported = rebind_row(row, parent_protocol, protocol)
            seal(root/'retrieval'/arm/(qid+'.json'), read(SOURCE/'retrieval'/arm/(qid+'.json')))
            seal(root/'rows'/arm/(qid+'.json'), imported)
            seal(root/'row_imports'/arm/(qid+'.json'), {'source': str(path), 'source_sha256': sha(row),
                'imported_sha256': sha(imported), 'only_protocol_binding_changed': True})
    return loaded, registration, protocol


def check_rows(root, loaded, registration, protocol, wanted):
    vector_record = read(old.VECTOR_FILE); vectors = dict(zip(vector_record['ids'], vector_record['vectors']))
    runtime = Runtime(root/'calls', allow_model_calls=False); counts = Counter(); rows = {a: {} for a in ARMS}
    for conv in loaded:
        sample = conv['sample_id']
        for arm in ARMS:
            graph = get_graph(root, sample, arm, registration['source_registration_sha256'])
            index = base.RepairedIndex(graph); graph_sha = sha(graph)
            for q in conv['qa']:
                qid = q['qa_id']
                if qid not in wanted: continue
                row = read(root/'rows'/arm/(qid+'.json')); retrieval = read(root/'retrieval'/arm/(qid+'.json'))
                expected = {'protocol_sha256': sha(protocol), 'qa_sha256': sha(q), 'arm': arm,
                            'graph_sha256': graph_sha, 'vector_sha256': sha(vectors[qid])}
                if row['identity'] != expected: raise StopRun('Row identity drift: ' + qid)
                context, trace = index.search(q['question'], vectors[qid], runtime.chat_fn('full-replay/'+arm+'/'+qid))
                if trace != retrieval['trace'] or trace['rerank']['fallback']: raise StopRun('Selection replay/fallback failure')
                operation = old.old_json(old.BASE_CONTEXT/(qid+'.json'))['operation']
                materializer = None
                if arm == 'opctx_original':
                    context, materializer = old.materialize_online_operation_context(operation,
                        {'question': q['question'], 'retrieved': context, 'retrieval_trace': copy.deepcopy(trace)}, graph)
                if (context != row['context'] or context != retrieval['context'] or sha(context) != row['context_sha256']
                        or len(context) != row['context_chars'] or materializer != retrieval['materializer']
                        or operation != retrieval['operation'] or trace['rerank']['selected_fact_ids'] != row['selected_fact_ids']):
                    raise StopRun('Context/selection materialization drift: ' + qid)
                base.check_bound(root, q, row)
                marker = root/'row_imports'/arm/(qid+'.json')
                if marker.exists():
                    origin = read(marker); source_row = read(Path(origin['source']))
                    if sha(source_row) != origin['source_sha256'] or sha(row) != origin['imported_sha256']:
                        raise StopRun('Imported row changed')
                rows[arm][qid] = row; counts['selection_replays'] += 1; counts['qa_judge_bindings'] += 2
            old.emit('FULL REPLAY ' + sample + '/' + arm + ' checked=' + str(len(rows[arm])))
    if any(set(values) != wanted for values in rows.values()): raise StopRun('Missing audited row')
    return rows, counts


def transport_audit(root):
    counts = Counter(); usage = {}; imported_keys = set()
    for path in (root/'calls/provider').glob('*/*/request.json'):
        request, final, _, raw = recover.completed_receipt(path.parent)
        stage = request['stage']
        if stage not in {'qa', 'judge', 'rerank'}: raise StopRun('Unexpected memory model call')
        counts[stage+'_requests'] += 1
        marker = root/'calls/imports'/stage/(path.parent.name+'.json')
        if marker.exists():
            origin = read(marker); _, source_final, _, source_raw = recover.completed_receipt(Path(origin['source']), request)
            if (source_final != final or source_raw != raw or origin['source_final_sha256'] != sha(final)
                    or origin['source_response_sha256'] != sha(raw) or origin['identity_sha256'] != sha(request)):
                raise StopRun('Imported provider receipt drift')
            imported_keys.add((stage, path.parent.name)); counts['imported_'+stage] += 1
            if list(path.parent.glob('*.intent.json')): raise StopRun('Imported response also has new dispatch')
        else:
            intents = list(path.parent.glob('*.intent.json'))
            if not intents: raise StopRun('Unattributed provider response')
            for intent in intents:
                prefix = intent.name.removesuffix('.intent.json')
                if sum(intent.with_name(prefix+s).exists() for s in ('.response.json', '.error.json')) != 1:
                    raise StopRun('Orphan or duplicate physical receipt')
                if read(intent)['request_sha256'] != path.parent.name: raise StopRun('Physical intent identity drift')
                counts['new_physical_attempts'] += 1
            errors = list(path.parent.glob('*.error.json')); counts['new_recovered_errors'] += len(errors)
            if any(not read(p)['retryable'] for p in errors): raise StopRun('Nonretryable error')
            stage_usage = usage.setdefault(stage, Counter())
            for k, v in raw['response'].get('usageMetadata', {}).items():
                if isinstance(v, int): stage_usage[k] += v
    if imported_keys != {(p.parent.name, p.stem) for p in (root/'calls/imports').glob('*/*.json')}:
        raise StopRun('Dangling import receipt')
    if list(root.rglob('*.pending')): raise StopRun('Unfinished durable artifact')
    return {'counts': counts, 'new_observed_final_usage': usage, 'pass': True,
            'failed_attempt_billing_without_usage_unknown': True}


def finish(root, loaded, registration, protocol):
    wanted, dev = population(protocol)
    for arm in ARMS:
        if {p.stem for p in (root/'rows'/arm).glob('*.json')} != wanted: raise StopRun('Full row set is incomplete')
    rows, checks = check_rows(root, loaded, registration, protocol, wanted)
    if snapshot(protocol) != read(root/'source_snapshot.json'): raise StopRun('Source snapshot drift')
    for path, digest in registration['source_hashes'].items():
        if file_sha(path) != digest: raise StopRun('Source code drift at final audit')
    reports = {arm: old.summarize(root, arm, protocol, full=True) for arm in ARMS}
    paired = {name: base.pair_summary([(rows['opctx_original'][q], rows['session_strong'][q]) for q in sorted(ids)])
              for name, ids in [('full1540', wanted), ('dev100_reused', dev), ('remaining1440', wanted-dev)]}
    result = {'registration_sha256': sha(registration), 'reports': reports, 'paired': paired,
        'audit': transport_audit(root), **checks, 'reused_dev100_per_arm': True, 'memory_rebuilt': False,
        'not_token_matched': True, 'newly_scored_questions_per_arm': 1440}
    seal(root/'complete.json', result); old.emit('MGTM480 FULL COMPLETE ' + json.dumps(reports))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'audit'])
    args = parser.parse_args()
    with old.Lease(RUN):
        loaded, registration, protocol = prepare()
        if args.command == 'prepare': return
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/'.env', override=False)
        runtime = Runtime(RUN/'calls', allow_model_calls=args.command == 'run')
        previous = original.adapter.install(runtime.log)
        try:
            if args.command == 'run':
                _, checks = check_rows(RUN, loaded, registration, protocol, set(protocol['dev_ids']))
                seal(RUN/'preflight.json', {'registration_sha256': sha(registration), **checks, 'new_model_calls': 0})
                evaluation = original.loader.loader.load_isolated('mgtm480_full_evaluation', old.__file__)
                evaluation.graph_api.GraphMemoryIndex = base.RepairedIndex
                evaluation.get_source_graph = lambda sample, arm, unused: get_graph(
                    RUN, sample, arm, registration['source_registration_sha256'])
                evaluation.evaluate(RUN, ARMS, loaded, protocol, runtime, full=True, qa_workers=20)
            finish(RUN, loaded, registration, protocol)
        except BaseException as exc:
            runtime.stopped.set()
            seal(RUN/('stop_%d.json' % time.time_ns()), {'utc': base.now(), 'type': type(exc).__name__, 'message': str(exc)[:600]})
            raise
        finally:
            original.adapter.restore(previous)


if __name__ == '__main__': main()

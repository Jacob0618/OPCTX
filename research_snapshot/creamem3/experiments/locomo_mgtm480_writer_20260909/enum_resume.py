"""Continue the incomplete full comparison with bounded candidate-ID repair."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import time
from creamem3.experiments.locomo_mgtm480_writer_20260909 import full, enum_repair, recheck_id

SOURCE = full.RUN
RUN = SOURCE.parent/'locomo_mgtm480_writer_full1540_enum_20260910_r2'
read, seal, sha, file_sha, StopRun = full.read, full.seal, full.sha, full.file_sha, full.StopRun
old, base, original = full.old, full.base, full.original


def helpers():
    module = original.loader.loader.load_isolated('mgtm480_enum_helpers', full.__file__)
    module.base = SimpleNamespace(**{**vars(base), 'RepairedIndex': enum_repair.Index})
    module.Runtime = enum_repair.Runtime
    return module


def snapshot(protocol):
    # Include the complete ancestor manifest, plus this stopped run's artifacts.
    files = [SOURCE/n for n in ('registration.json', 'protocol.json', 'source_snapshot.json', 'preflight.json')]
    for name in ('graphs', 'memory_imports', 'rows', 'row_imports', 'retrieval', 'calls/provider', 'calls/imports'):
        files.extend(sorted((SOURCE/name).rglob('*.json')))
    files.extend(SOURCE.glob('stop_*.json'))
    frozen = dict(read(SOURCE/'source_snapshot.json'))
    frozen.update({str(p): file_sha(p) for p in files})
    return frozen


def audit_source():
    counts = Counter(); unfinished = []
    for path in (SOURCE/'calls/provider').glob('*/*/request.json'):
        request = read(path); key = sha(request)
        if path.parent.name != key: raise StopRun('Source request hash drift')
        intents = list(path.parent.glob('*.intent.json')); errors = list(path.parent.glob('*.error.json'))
        for intent in intents:
            prefix = intent.name.removesuffix('.intent.json')
            if sum(intent.with_name(prefix+s).exists() for s in ('.response.json', '.error.json')) != 1:
                raise StopRun('Source orphan/duplicate response must not be retried blindly')
            if read(intent)['request_sha256'] != key: raise StopRun('Source intent drift')
        if any(not read(p)['retryable'] for p in errors): raise StopRun('Source nonretryable failure')
        counts['physical_attempts'] += len(intents); counts['physical_errors'] += len(errors)
        if (path.parent/'final.json').exists():
            full.recover.completed_receipt(path.parent)
            counts['completed_requests'] += 1
        else:
            if not intents or len(errors) != len(intents) or list(path.parent.glob('*.response.json')):
                raise StopRun('Source unfinished request has ambiguous acceptance receipt')
            unfinished.append({'source': str(path.parent), 'request_sha256': key, 'stage': request['stage'],
                               'recorded_error_attempts': len(errors), 'retryable': True})
    if list(SOURCE.rglob('*.pending')): raise StopRun('Source has unfinished durable artifacts')
    return {'counts': counts, 'unfinished': unfinished, 'no_orphans': True, 'pass': True}


def prepare():
    parent = read(SOURCE/'registration.json'); parent_protocol = read(SOURCE/'protocol.json')
    for path, digest in parent['source_hashes'].items():
        if file_sha(path) != digest: raise StopRun('Source code changed: ' + path)
    if full.snapshot(parent_protocol) != read(SOURCE/'source_snapshot.json'):
        raise StopRun('Full-run ancestor source drift')
    loaded, control = old.prepare(RUN/'original_control_protocol')
    for k, v in control.items():
        if parent_protocol.get(k) != v: raise StopRun('Evaluation control changed: ' + k)
    all_ids, dev = full.population(parent_protocol)
    frozen = snapshot(parent_protocol); seal(RUN/'source_snapshot.json', frozen)
    source_audit = audit_source(); seal(RUN/'source_audit.json', source_audit)
    imported_ids = {arm: sorted(p.stem for p in (SOURCE/'rows'/arm).glob('*.json')) for arm in full.ARMS}
    if {a: len(v) for a, v in imported_ids.items()} != {'session_strong': 1338, 'opctx_original': 100}:
        raise StopRun('Source row inventory differs from audited interruption')
    if any(not dev <= set(v) <= all_ids for v in imported_ids.values()): raise StopRun('Source population mismatch')
    paths = [Path(__file__), Path(enum_repair.__file__), Path(recheck_id.__file__), Path(__file__).with_name('enum_resume_protocol.md')]
    registration = {**parent, 'source_hashes': {**parent['source_hashes'], **{str(p): file_sha(p) for p in paths}},
        'resume_source': str(SOURCE), 'resume_source_registration_sha256': sha(parent),
        'resume_source_snapshot_sha256': sha(frozen), 'imported_row_ids': imported_ids,
        'enum_repair_after_existing_failure_only': True, 'enum_max_attempts': 2,
        'new_questions_this_continuation': {a: 1540-len(v) for a, v in imported_ids.items()},
        'unfinished_source_requests': len(source_audit['unfinished']), 'fail_fast_worker_signal': True}
    seal(RUN/'registration.json', registration)
    protocol = {**parent_protocol, 'candidate_enum_resume_registration_sha256': sha(registration)}
    seal(RUN/'protocol.json', protocol)
    for conv in loaded:
        for arm in full.ARMS:
            p = original.graph_path(SOURCE, conv['sample_id'], arm)
            record = read(p); full.get_graph(SOURCE, conv['sample_id'], arm, parent['source_registration_sha256'])
            seal(original.graph_path(RUN, conv['sample_id'], arm), record)
            marker = read(SOURCE/'memory_imports'/conv['sample_id']/(arm+'.json'))
            seal(RUN/'memory_imports'/conv['sample_id']/(arm+'.json'), {**marker, 'source': str(p)})
    importer = enum_repair.Runtime(RUN/'calls', allow_model_calls=False)
    for path in sorted((SOURCE/'calls/provider').glob('*/*/request.json')):
        if (path.parent/'final.json').exists(): importer.post(**read(path))
    for arm in full.ARMS:
        for path in sorted((SOURCE/'retrieval'/arm).glob('*.json')):
            # Retrieval identity has no protocol hash. Preserve unfinished rows'
            # already completed retrieval as well as scored rows' retrieval.
            seal(RUN/'retrieval'/arm/path.name, read(path))
        for qid in imported_ids[arm]:
            path = SOURCE/'rows'/arm/(qid+'.json'); row = read(path)
            imported = full.rebind_row(row, parent_protocol, protocol)
            seal(RUN/'rows'/arm/(qid+'.json'), imported)
            seal(RUN/'row_imports'/arm/(qid+'.json'), {'source': str(path), 'source_sha256': sha(row),
                'imported_sha256': sha(imported), 'only_protocol_binding_changed': True})
    old.emit('ENUM PREPARED ' + json.dumps({a: len(v) for a, v in imported_ids.items()}))
    return loaded, registration, protocol


def smoke(loaded, registration, runtime):
    conv = next(c for c in loaded if c['sample_id'] == 'conv-42')
    q = next(q for q in conv['qa'] if q['qa_id'] == 'conv-42-q014')
    graph = full.get_graph(RUN, conv['sample_id'], 'session_strong', registration['source_registration_sha256'])
    vectors = read(old.VECTOR_FILE); vector = vectors['vectors'][vectors['ids'].index(q['qa_id'])]
    index = enum_repair.Index(graph)
    context, trace = index.search(q['question'], vector, runtime.chat_fn('enum-smoke/'+q['qa_id']))
    if not trace['rerank'].get('candidate_enum_repair') or trace['rerank']['fallback']:
        raise StopRun('Smoke did not exercise the observed enum repair')
    selected = trace['rerank']['selected_fact_ids']
    candidates, _ = index._candidate_pool(q['question'], vector)
    if len(selected) != 20 or len(set(selected)) != 20 or not set(selected) <= {c['fact']['id'] for c in candidates}:
        raise StopRun('Enum smoke returned invalid candidate IDs')
    report = {'registration_sha256': sha(registration), 'qa_id': q['qa_id'], 'selected_ids': selected,
        'trace': trace, 'context_sha256': sha(context), 'no_qa_or_judge': True, 'pass': True}
    seal(RUN/'enum_smoke.json', report); old.emit('ENUM SMOKE PASS ' + json.dumps(trace['rerank']))


def preflight(module, loaded, registration, protocol):
    # The imported populations are unequal. Audit each arm's exact imported set
    # separately, using empty QA lists for the other arm via an isolated helper.
    total = Counter()
    for arm in full.ARMS:
        prior_arms = module.ARMS; module.ARMS = (arm,)
        try:
            _, checks = module.check_rows(RUN, loaded, registration, protocol, set(registration['imported_row_ids'][arm]))
            total.update(checks)
        finally:
            module.ARMS = prior_arms
    seal(RUN/'preflight.json', {'registration_sha256': sha(registration), **total, 'new_model_calls': 0})
    old.emit('ENUM PREFLIGHT PASS ' + json.dumps(total))


def finish(module, loaded, registration, protocol):
    wanted, dev = full.population(protocol)
    for arm in full.ARMS:
        if {p.stem for p in (RUN/'rows'/arm).glob('*.json')} != wanted: raise StopRun('Incomplete full result')
    rows, checks = module.check_rows(RUN, loaded, registration, protocol, wanted)
    if snapshot(protocol) != read(RUN/'source_snapshot.json'): raise StopRun('Resume source snapshot drift')
    for path, digest in registration['source_hashes'].items():
        if file_sha(path) != digest: raise StopRun('Final source code drift')
    transport = module.transport_audit(RUN)
    recovered = []
    for source in read(RUN/'source_audit.json')['unfinished']:
        p = RUN/'calls/provider'/source['stage']/source['request_sha256']
        full.recover.completed_receipt(p, read(Path(source['source'])/'request.json'))
        recovered.append(source['request_sha256'])
    enum_rows = {arm: [p.stem for p in (RUN/'retrieval'/arm).glob('*.json')
                       if read(p)['trace']['rerank'].get('candidate_enum_repair')] for arm in full.ARMS}
    reports = {arm: old.summarize(RUN, arm, protocol, full=True) for arm in full.ARMS}
    pairs = {name: base.pair_summary([(rows['opctx_original'][q], rows['session_strong'][q]) for q in sorted(ids)])
             for name, ids in [('full1540', wanted), ('dev100_reused', dev), ('remaining1440', wanted-dev)]}
    result = {'registration_sha256': sha(registration), 'reports': reports, 'paired': pairs, **checks,
        'audit': transport, 'source_audit': read(RUN/'source_audit.json'), 'recovered_source_requests': recovered,
        'enum_repaired_rows': enum_rows, 'imported_rows': {a: len(v) for a, v in registration['imported_row_ids'].items()},
        'not_token_matched': True, 'memory_rebuilt': False}
    seal(RUN/'complete.json', result); old.emit('ENUM FULL COMPLETE ' + json.dumps(reports))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'smoke', 'run', 'audit'])
    args = parser.parse_args()
    with old.Lease(RUN):
        loaded, registration, protocol = prepare(); module = helpers()
        if args.command == 'prepare': return
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/'.env', override=False)
        runtime = enum_repair.Runtime(RUN/'calls', allow_model_calls=args.command != 'audit')
        previous = original.adapter.install(runtime.log)
        try:
            if args.command == 'smoke': smoke(loaded, registration, runtime); return
            if read(RUN/'enum_smoke.json')['registration_sha256'] != sha(registration):
                raise StopRun('Matching candidate-enum smoke required')
            if args.command == 'run':
                preflight(module, loaded, registration, protocol)
                evaluation = original.loader.loader.load_isolated('mgtm480_enum_evaluation', old.__file__)
                evaluation.graph_api.GraphMemoryIndex = enum_repair.Index
                evaluation.get_source_graph = lambda sample, arm, unused: full.get_graph(
                    RUN, sample, arm, registration['source_registration_sha256'])
                evaluation.ThreadPoolExecutor = lambda *a, **k: enum_repair.FailFastPool(runtime, *a, **k)
                evaluation.evaluate(RUN, full.ARMS, loaded, protocol, runtime, full=True, qa_workers=20)
            finish(module, loaded, registration, protocol)
        except BaseException as exc:
            runtime.record_failure(exc)
            seal(RUN/('stop_%d.json' % time.time_ns()), {'utc': base.now(), 'type': type(exc).__name__, 'message': str(exc)[:600]})
            raise
        finally:
            original.adapter.restore(previous)


if __name__ == '__main__': main()

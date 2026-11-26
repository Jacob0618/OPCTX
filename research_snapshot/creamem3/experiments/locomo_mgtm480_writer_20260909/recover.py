"""Write-once successor: reuse completed memory and recover only output truncation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import time

from creamem3.experiments.locomo_mgtm480_writer_20260909 import run as original

old, base, writer = original.old, original.base, original.writer
read, seal, sha, file_sha, StopRun = original.read, original.seal, original.sha, original.file_sha, original.StopRun
SOURCE = original.RUN
RUN = SOURCE.parent / 'locomo_mgtm480_writer_dev100_20260910_cap24k_r2'
CAPS = (24576, 32768)
BAD_KEY = '586c3fcbfaa499e8bccf24d0bcda80cb36f1255e9b537798b7f971eb5231ce75'


def finish_reason(response):
    candidates = response.get('candidates') or []
    return candidates[0].get('finishReason') if len(candidates) == 1 else 'INVALID_CANDIDATE_COUNT'


def completed_receipt(directory, identity=None):
    request = read(directory/'request.json'); final = read(directory/'final.json')
    raw_name = 'attempt_%02d.response.json' % final['attempt']; raw = read(directory/raw_name)
    key = sha(request)
    if (directory.name != key or final['identity'] != request or final['request_sha256'] != key
            or final['response_sha256'] != sha(raw) or (identity is not None and request != identity)):
        raise StopRun('Completed receipt contract drift: ' + str(directory))
    return request, final, raw_name, raw


class Runtime(base.Runtime):
    def __init__(self, root, *, allow_model_calls, source=SOURCE):
        super().__init__(root, parents=(), allow_model_calls=allow_model_calls, concurrency=20, timeout=240)
        self.source = Path(source)

    def import_exact(self, identity):
        key = sha(identity); stage = identity['stage']
        source = self.source/'calls/provider'/stage/key
        if not (source/'final.json').exists():
            return None
        request, final, raw_name, raw = completed_receipt(source, identity)
        if stage == 'online' and finish_reason(raw['response']) != 'STOP':
            reason = finish_reason(raw['response'])
            seal(self.root/'rejected_source'/(key+'.json'), {
                'source': str(source), 'source_response_sha256': sha(raw), 'reason': reason,
                'accepted_into_memory': False})
            if reason != 'MAX_TOKENS':
                self._halt('Source Online ended abnormally: ' + str(reason))
            return None
        own = self.root/'provider'/stage/key
        with self.mutex:
            if not (own/'final.json').exists():
                seal(self.root/'imports'/stage/(key+'.json'), {'source': str(source),
                    'identity_sha256': key, 'source_final_sha256': sha(final),
                    'source_response_sha256': sha(raw), 'new_physical_call': False})
                seal(own/'request.json', request); seal(own/raw_name, raw); seal(own/'final.json', final)
        return raw['response']

    def post(self, stage, scope, model, method, body):
        if stage == 'writer':
            self._halt('Recovery forbids Writer dispatch; all 272 packets already exist')
        if stage not in {'online', 'embedding'}:
            return super().post(stage, scope, model, method, body)
        expected = original.EMBED if stage == 'embedding' else 'gemini-2.5-flash'
        if model != expected:
            self._halt('Unexpected recovery model')
        identity = {'stage': stage, 'scope': scope, 'model': model, 'method': method, 'body': body}
        prior = self.import_exact(identity)
        if stage == 'embedding':
            return original.GenerationRuntime.post(self, stage, scope, model, method, body)
        key = sha(identity)
        if body.get('generationConfig', {}).get('maxOutputTokens') != 16384:
            self._halt('Unexpected upstream Online output limit')
        if prior is not None:
            selected = key; response = prior; cap = 16384
        else:
            for cap in CAPS:
                expanded = copy.deepcopy(body)
                expanded['generationConfig']['maxOutputTokens'] = cap
                selected = sha({**identity, 'body': expanded})
                response = original.GenerationRuntime.post(self, stage, scope, model, method, expanded)
                reason = finish_reason(response)
                if reason == 'STOP':
                    break
                if reason != 'MAX_TOKENS':
                    self._halt('Online generation ended abnormally: ' + str(reason))
                seal(self.root/'truncations'/(selected+'.json'), {'request_sha256': selected,
                    'logical_identity_sha256': key, 'cap': cap, 'scope': scope,
                    'accepted_into_memory': False, 'next_cap': CAPS[1] if cap == CAPS[0] else None})
                self.log({'kind': 'online_output_truncated', 'request': selected, 'scope': scope,
                          'cap': cap, 'retry_at': CAPS[1] if cap == CAPS[0] else None})
            else:
                self._halt('Online remained truncated at 32768; bounded recovery exhausted')
        # No partial JSON ever reaches the legacy memory updater.
        text = base.response_text(response)
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            self._halt('Finished Online response is not a complete JSON object')
        if not isinstance(value, dict) or set(value) != {'operations'} or not isinstance(value['operations'], list):
            self._halt('Finished Online response violates outer operations schema')
        seal(self.root/'online_resolutions'/(key+'.json'), {'original_identity': identity,
            'selected_request_sha256': selected, 'effective_output_limit': cap, 'finish_reason': 'STOP',
            'reused_original': prior is not None})
        return response


def source_snapshot():
    files = [SOURCE/name for name in ('registration.json', 'protocol.json', 'prompt_bundle.json', 'source_sessions.json')]
    for name in ('writer_packets', 'graphs', 'calls/provider'):
        files.extend(sorted((SOURCE/name).rglob('*.json')))
    return {str(p): file_sha(p) for p in files}


def source_transport_audit():
    counts = Counter(); finishes = Counter()
    for path in (SOURCE/'calls/provider').glob('*/*/request.json'):
        request, _, _, raw = completed_receipt(path.parent)
        stage = request['stage']; counts[stage+'_requests'] += 1
        if stage in {'writer', 'online'}:
            reason = finish_reason(raw['response']); finishes[stage+':'+str(reason)] += 1
            if reason != 'STOP' and not (stage == 'online' and path.parent.name == BAD_KEY and reason == 'MAX_TOKENS'):
                raise StopRun('Unexpected incomplete source generation')
        for intent in path.parent.glob('*.intent.json'):
            prefix = intent.name.removesuffix('.intent.json')
            if sum(intent.with_name(prefix+s).exists() for s in ('.response.json', '.error.json')) != 1:
                raise StopRun('Source orphan or duplicate physical receipt')
            counts['physical_attempts'] += 1
        errors = list(path.parent.glob('*.error.json')); counts['recovered_transport_errors'] += len(errors)
        if any(not read(p)['retryable'] for p in errors):
            raise StopRun('Nonretryable source failure')
    if list(SOURCE.rglob('*.pending')):
        raise StopRun('Source has unfinished durable artifacts')
    return {'counts': counts, 'finishes': finishes, 'known_truncated_response_excluded': BAD_KEY, 'pass': True}


def prepare(root=RUN):
    parent = read(SOURCE/'registration.json'); source_protocol = read(SOURCE/'protocol.json')
    if parent['source_hashes'] != original.source_hashes():
        raise StopRun('Frozen original source code changed')
    sources = read(SOURCE/'source_sessions.json')
    if sha(sources) != parent['sources_sha256'] or sha(writer.bundle()) != parent['prompt_bundle_sha256']:
        raise StopRun('Source population or Writer prompt drift')
    loaded, control = old.prepare(root/'original_control_protocol')
    if sha(control) != parent['original_control_protocol_sha256']:
        raise StopRun('Reader/control protocol drift')
    snapshot_path = root/'source_snapshot.json'
    snapshot = source_snapshot()
    seal(snapshot_path, snapshot)
    source_audit = source_transport_audit(); seal(root/'source_transport_audit.json', source_audit)
    additions = [Path(__file__), Path(__file__).with_name('recovery_protocol.md')]
    registration = {**parent, 'source_hashes': {**parent['source_hashes'], **{str(p): file_sha(p) for p in additions}},
        'recovery': {'source': str(SOURCE), 'source_registration_sha256': sha(parent),
            'source_snapshot_sha256': sha(snapshot), 'uncached_online_caps': list(CAPS),
            'exact_successful_cache_reuse': True, 'writer_dispatch_forbidden': True,
            'mixed_cache_continuation_not_all_fresh_cap24k': True, 'timeout_s': 240},
        'no_old_model_answer_imports': True}
    seal(root/'registration.json', registration)
    seal(root/'prompt_bundle.json', read(SOURCE/'prompt_bundle.json')); seal(root/'source_sessions.json', sources)
    protocol = {**source_protocol, 'mgtm480_registration_sha256': sha(registration),
                'writer_reused_immutable': True, 'recovery_source_protocol_sha256': sha(source_protocol)}
    seal(root/'protocol.json', protocol)
    packet_count = 0; graph_count = Counter()
    for source in sources:
        sample = source['sample_id']
        for record in source['records']:
            sid = record['session_id']; p = SOURCE/'writer_packets'/sample/('%04d.json' % sid)
            packet = read(p)
            expected = {'registration_sha256': sha(parent), 'sample': sample, 'session_id': sid,
                        'record_sha256': sha(record), 'prompt_bundle_sha256': parent['prompt_bundle_sha256']}
            if packet['identity'] != expected:
                raise StopRun('Source Writer packet identity drift')
            seal(root/'writer_packets'/sample/p.name, packet); packet_count += 1
        for arm in original.ARMS:
            source_path = original.graph_path(SOURCE, sample, arm)
            if not source_path.exists():
                if arm == 'session_strong': raise StopRun('Missing completed source Writer graph')
                continue
            record = read(source_path)
            if record['identity'] != {'registration_sha256': sha(parent), 'source_sha256': sha(source)}:
                raise StopRun('Source graph identity drift')
            imported = {**record, 'identity': {'registration_sha256': sha(registration), 'source_sha256': sha(source)}}
            seal(original.graph_path(root, sample, arm), imported)
            seal(root/'memory_imports'/sample/(arm+'.json'), {'source': str(source_path),
                'source_record_sha256': sha(record), 'graph_sha256': sha(record['graph']),
                'new_graph_record_sha256': sha(imported), 'graph_unchanged': True})
            graph_count[arm] += 1
    if packet_count != 272 or graph_count != {'session_strong': 10, 'opctx_original': 6}:
        raise StopRun('Source completion inventory differs from inspected stop')
    seal(root/'memory_reuse.json', {'writer_packets': packet_count, 'graphs': graph_count,
        'packets_retain_original_identity': True, 'source_registration_sha256': sha(parent)})
    return loaded, sources, registration, protocol


def transport_audit(root):
    counts = Counter(); usage = {}; imports = 0
    for path in (root/'calls/provider').glob('*/*/request.json'):
        request, final, _, raw = completed_receipt(path.parent)
        stage = request['stage']; counts[stage+'_requests'] += 1
        marker = root/'calls/imports'/stage/(path.parent.name+'.json')
        if marker.exists():
            imported = read(marker); _, source_final, _, source_raw = completed_receipt(Path(imported['source']), request)
            if (source_final != final or source_raw != raw or imported['identity_sha256'] != sha(request)
                    or imported['source_final_sha256'] != sha(final) or imported['source_response_sha256'] != sha(raw)):
                raise StopRun('Imported receipt drift')
            imports += 1; counts['imported_'+stage] += 1
        else:
            for intent in path.parent.glob('*.intent.json'):
                prefix = intent.name.removesuffix('.intent.json')
                if sum(intent.with_name(prefix+s).exists() for s in ('.response.json', '.error.json')) != 1:
                    raise StopRun('New orphan or duplicate physical receipt')
                if read(intent)['request_sha256'] != path.parent.name:
                    raise StopRun('Physical intent identity drift')
                counts['new_physical_attempts'] += 1
            errors = list(path.parent.glob('*.error.json')); counts['new_recovered_errors'] += len(errors)
            if any(not read(p)['retryable'] for p in errors): raise StopRun('New nonretryable error')
            stage_usage = usage.setdefault(stage, Counter())
            for k, v in raw['response'].get('usageMetadata', {}).items():
                if isinstance(v, int): stage_usage[k] += v
        if stage == 'online' and finish_reason(raw['response']) != 'STOP':
            if finish_reason(raw['response']) != 'MAX_TOKENS' or not (root/'calls/truncations'/(path.parent.name+'.json')).exists():
                raise StopRun('Untracked abnormal Online generation')
    for path in (root/'calls/online_resolutions').glob('*.json'):
        resolution = read(path); request, _, _, raw = completed_receipt(
            root/'calls/provider/online'/resolution['selected_request_sha256'])
        expected = copy.deepcopy(resolution['original_identity'])
        expected['body']['generationConfig']['maxOutputTokens'] = resolution['effective_output_limit']
        if path.stem != sha(resolution['original_identity']) or expected != request or finish_reason(raw['response']) != 'STOP':
            raise StopRun('Online resolution changed more than output cap')
    if source_snapshot() != read(root/'source_snapshot.json') or list(root.rglob('*.pending')):
        raise StopRun('Source mutation or unfinished artifact')
    if list((root/'calls/provider/writer').glob('*/request.json')):
        raise StopRun('Unexpected new Writer call')
    return {'counts': counts, 'new_observed_final_usage': usage, 'imported_requests': imports,
            'original_source_audit': read(root/'source_transport_audit.json'), 'pass': True,
            'failed_attempt_billing_without_usage_unknown': True}


def smoke(root, runtime, registration):
    request, _, _, raw = completed_receipt(SOURCE/'calls/provider/online'/BAD_KEY)
    if finish_reason(raw['response']) != 'MAX_TOKENS': raise StopRun('Smoke is not the observed failure')
    response = runtime.post(**request)
    value = json.loads(base.response_text(response))
    if not 0 <= len(value['operations']) <= 12: raise StopRun('Smoke operations count exceeds contract')
    resolution = read(root/'calls/online_resolutions'/(BAD_KEY+'.json'))
    report = {'registration_sha256': sha(registration), 'source_bad_request': BAD_KEY,
        'resolution': resolution, 'operations': len(value['operations']), 'complete_json': True,
        'whole_graph_validation_still_required': True, 'not_a_QA_score': True}
    seal(root/'recovery_smoke.json', report)
    old.emit('RECOVERY SMOKE PASS cap=' + str(resolution['effective_output_limit']))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'smoke', 'run', 'audit'])
    args = parser.parse_args()
    with old.Lease(RUN):
        loaded, sources, registration, protocol = prepare()
        if args.command == 'prepare': return
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/'.env', override=False)
        runtime = Runtime(RUN/'calls', allow_model_calls=args.command != 'audit')
        previous = original.adapter.install(runtime.log)
        try:
            if args.command == 'smoke': smoke(RUN, runtime, registration); return
            if read(RUN/'recovery_smoke.json')['registration_sha256'] != sha(registration):
                raise StopRun('Matching recovery smoke required')
            with ThreadPoolExecutor(max_workers=10) as pool:
                for task in as_completed([pool.submit(original.build_one, RUN, s, registration, runtime) for s in sources]):
                    task.result()
            evaluation = original.loader.loader.load_isolated('mgtm480_recovery_eval', old.__file__)
            evaluation.graph_api.GraphMemoryIndex = base.RepairedIndex
            def get_graph(sample, arm, unused):
                record = read(original.graph_path(RUN, sample, arm))
                if record['identity']['registration_sha256'] != sha(registration): raise StopRun('Evaluation graph drift')
                return record['graph']
            evaluation.get_source_graph = get_graph
            if args.command == 'run': evaluation.evaluate(RUN, original.ARMS, loaded, protocol, runtime, qa_workers=20)
            finalizer = original.loader.loader.load_isolated('mgtm480_recovery_finalizer', original.__file__)
            finalizer.transport_audit = transport_audit
            finalizer.finish_eval(RUN, loaded, registration, protocol, Runtime(RUN/'calls', allow_model_calls=False))
        except BaseException as exc:
            runtime.stopped.set()
            seal(RUN/('stop_%d.json' % time.time_ns()), {'utc': base.now(), 'type': type(exc).__name__, 'message': str(exc)[:600]})
            raise
        finally:
            original.adapter.restore(previous)


if __name__ == '__main__': main()

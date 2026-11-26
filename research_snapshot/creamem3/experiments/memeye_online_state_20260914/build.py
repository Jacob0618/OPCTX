"""Source-only smoke for the NEW derived-state path, isolated from all QA runs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from copy import deepcopy
import argparse
import json
import time
from . import state, intake, reader_bridge
from .audit import ROOT, baseline
from creamem3.experiments.locomo_g35lite_20260908 import runtime as transport

RUN = ROOT / 'source_smoke_v1'
TEXT = {'type': 'STRING'}
IDLIST = {'type': 'ARRAY', 'items': TEXT}
QUOTE = {'type': 'OBJECT', 'properties': {'fact_id': TEXT, 'quote': TEXT}, 'required': ['fact_id', 'quote']}
CLAIM = {'type': 'OBJECT', 'nullable': True, 'properties': {
    'kind': {'type': 'STRING', 'enum': ['state', 'relation']}, 'subject': TEXT, 'attribute': TEXT,
    'scope': TEXT, 'value': TEXT, 'text': TEXT, 'support_fact_ids': IDLIST,
    'quotes': {'type': 'ARRAY', 'items': QUOTE}, 'valid_from': TEXT, 'current_dependencies': IDLIST},
    'required': sorted(state.CLAIM_KEYS)}
OP = {'type': 'OBJECT', 'properties': {'action': {'type': 'STRING', 'enum': sorted(state.ACTIONS)},
    'event_id': TEXT, 'old_claim_ids': IDLIST, 'new_claim': CLAIM, 'reason': TEXT,
    'reason_quotes': {'type': 'ARRAY', 'items': QUOTE}}, 'required': sorted(state.OP_KEYS)}
SCHEMA = {'type': 'OBJECT', 'properties': {'operations': {'type': 'ARRAY', 'maxItems': 12, 'items': OP}},
          'required': ['operations']}


def hashes():
    return {str(Path(m.__file__).resolve()): baseline.file_sha(Path(m.__file__))
            for m in (state, intake, reader_bridge, transport)} | {str(Path(__file__).resolve()): baseline.file_sha(Path(__file__))}


def body(value):
    payload = transport.request_body(intake.prompt(value), 8192, True)
    payload['generationConfig']['thinkingConfig'] = {'thinkingLevel': 'low'}
    payload['generationConfig']['responseSchema'] = deepcopy(SCHEMA)
    return payload


def sources():
    original = baseline.checked(baseline.ROOT / 'registration.json')
    receipts = {k.split('::', 1)[1]: r for k, r in original['source_graphs'].items() if k.startswith('online::')}
    graphs = {}
    for task, r in receipts.items():
        if baseline.file_sha(Path(r['path'])) != r['sha256']:
            raise RuntimeError('Frozen Writer graph drift')
        graphs[task] = json.loads(Path(r['path']).read_text(encoding='utf-8'))
    reg = {'schema': 'memeye.derived_state_source_smoke.v1', 'source_graphs': receipts,
           'source_hashes': hashes(), 'task_selection': 'all eight tasks, first two physical sessions each',
           'expected_frontiers': 16, 'model': transport.BACKBONE, 'thinking': 'low', 'output_cap': 8192,
           'event_top_k': 3, 'prior_fact_cap': 80, 'writer_changed': False, 'new_embedding_calls': 0,
           'qa_judge_rerank_calls': 0, 'prompt_sha256': state.digest(intake.PROMPT),
           'source_only_not_an_accuracy_experiment': True, 'whole_frontier_transaction': True}
    baseline.put(RUN / 'registration.json', reg)
    return reg, graphs


def execute(allow):
    reg, graphs = sources()
    if not allow:
        print('NEW ONLINE SOURCE PREFLIGHT', len(graphs), 'tasks; no model calls', flush=True)
        return
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / '.env', override=False)
    runtime = transport.Runtime(RUN / 'calls', allow_model_calls=True, concurrency=4, timeout=180)
    def task_run(task, graph):
        facts = {f['id']: f for f in graph['fact_nodes']}
        current = state.empty(task)
        steps = graph['online_fusion']['steps'][:2]
        if len(steps) != 2:
            raise RuntimeError('Smoke population differs')
        for step in steps:
            sid = step['session_id']
            seeds = step['allowed_ids']['prior_candidate_fact_ids']
            value, exposure = intake.make_input(graph, current, sid, seeds, event_top_k=3, prior_fact_cap=80)
            identity = {'registration_sha256': state.digest(reg), 'task': task, 'frontier': sid,
                        'input_sha256': state.digest(value), 'before_state_sha256': state.digest(current)}
            path = RUN / 'checkpoints' / task / (str(sid) + '.json')
            if path.exists():
                saved = baseline.checked(path)
                if saved['identity'] != identity:
                    raise RuntimeError('New-state checkpoint drift')
                response = saved['response']
            else:
                request = body(value)
                response = runtime.post('online_state', task + '/' + str(sid), transport.BACKBONE, 'generateContent', request)
            if [c.get('finishReason') for c in response.get('candidates', [])] != ['STOP']:
                raise RuntimeError('Unfinished online-state output')
            packet = baseline.direct.predict._parse_json_object(transport.response_text(response))
            try:
                next_state, transactions = state.apply_frontier(current, packet, facts,
                    exposure['allowed_fact_ids'], exposure['exposed_claim_ids'], sid)
            except ValueError as exc:
                baseline.put(RUN / 'rejected' / task / (str(sid) + '.json'),
                    {'identity': identity, 'input': value, 'response': response, 'reason': str(exc),
                     'unmodified_before_state': current, 'whole_frontier_not_committed': True})
                raise
            baseline.put(path, {'identity': identity, 'input': value, 'exposure': exposure,
                         'response': response, 'transactions': transactions, 'state': next_state,
                         'state_sha256': state.digest(next_state)})
            current = next_state
            print('NEW ONLINE SMOKE', task, sid, 'claims', len(current['claims']),
                  'actions', [r['action'] for r in transactions], flush=True)
        return {'task': task, 'frontiers': 2, 'claims': len(current['claims']),
                'actions': dict(Counter(r['action'] for r in current['ledger']))}
    stats = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(task_run, t, g) for t, g in sorted(graphs.items())]
            for f in as_completed(futures):
                try:
                    stats.append(f.result())
                except BaseException:
                    runtime.stopped.set()
                    raise
        if hashes() != reg['source_hashes']:
            raise RuntimeError('Source code changed during smoke')
        baseline.put(RUN / 'generation_complete.json', {'complete': True, 'frontiers': 16,
            'stats': sorted(stats, key=lambda r: r['task']), 'registration_sha256': state.digest(reg),
            'requires_independent_replay': True, 'no_qa_accuracy_claim': True})
        verify()
    except BaseException as exc:
        baseline.put(RUN / ('failure_' + str(time.time_ns()) + '.json'),
                     {'type': type(exc).__name__, 'message': str(exc)[:500], 'baseline_unchanged': True})
        raise


def verify():
    reg, graphs = sources()
    from creamem3.experiments.memeye_evidence_20260914.verify_evidence import generic
    count = 0; links = set(); usage = Counter(); statuses = Counter()
    for task, graph in sorted(graphs.items()):
        facts = {f['id']: f for f in graph['fact_nodes']}; current = state.empty(task)
        for step in graph['online_fusion']['steps'][:2]:
            sid = step['session_id']
            saved = baseline.checked(RUN / 'checkpoints' / task / (str(sid) + '.json'))
            value, exposure = intake.make_input(graph, current, sid, step['allowed_ids']['prior_candidate_fact_ids'])
            identity = {'stage': 'online_state', 'scope': task + '/' + str(sid), 'model': transport.BACKBONE,
                        'method': 'generateContent', 'body': body(value)}
            directory = RUN / 'calls/provider/online_state' / transport.sha(identity)
            response = generic(directory, identity); links.add(str(directory.resolve()))
            if value != saved['input'] or exposure != saved['exposure'] or response != saved['response']:
                raise RuntimeError('Request/response/exposure replay mismatch')
            expected_identity = {'registration_sha256': state.digest(reg), 'task': task, 'frontier': sid,
                                 'input_sha256': state.digest(value), 'before_state_sha256': state.digest(current)}
            if expected_identity != saved['identity']:
                raise RuntimeError('Checkpoint identity mismatch')
            packet = baseline.direct.predict._parse_json_object(transport.response_text(response))
            current, transactions = state.apply_frontier(current, packet, facts, exposure['allowed_fact_ids'], exposure['exposed_claim_ids'], sid)
            if current != saved['state'] or state.digest(current) != saved['state_sha256'] or transactions != saved['transactions']:
                raise RuntimeError('State transaction replay mismatch')
            usage.update({k: v for k, v in response.get('usageMetadata', {}).items() if isinstance(v, (int, float))})
            count += 1
        statuses.update(c['status'] for c in current['claims'].values())
    actual = {str(p.parent.resolve()) for p in (RUN / 'calls/provider/online_state').glob('*/final.json')}
    if count != 16 or actual != links:
        raise RuntimeError('Physical/causal population mismatch')
    baseline.put(RUN / 'independent_replay.json', {'status': 'PASS', 'frontiers': count,
                 'linked_model_calls': len(links), 'usage': dict(usage), 'claim_statuses': dict(statuses),
                 'raw_writer_unchanged': True, 'query_and_gold_free': True,
                 'semantic_truth_and_qa_gain_not_established': True, 'audit_model_calls': 0})
    print('NEW ONLINE SOURCE SMOKE REPLAY PASS', count, dict(statuses), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['preflight', 'smoke', 'verify'])
    p.add_argument('--allow-model-calls', action='store_true'); args = p.parse_args()
    if args.command == 'verify':
        verify()
    else:
        execute(args.command == 'smoke' and args.allow_model_calls)

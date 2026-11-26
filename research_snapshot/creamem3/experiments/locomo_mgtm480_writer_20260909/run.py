"""Fresh matched memory build and original-reader evaluation with MGTM480 Writer."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time
from creamem3.experiments.locomo_mgtm480_writer_20260909 import writer
from creamem3.experiments.locomo_joint_revision_20260909 import followup_eval as base
from creamem3.experiments.locomo_online_semantics_20260909 import incomplete_http_read_adapter as adapter
from creamem3.experiments.locomo_online_semantics_20260909 import stable_state_intake_v4 as loader
from creamem3.experiments.locomo_online_only_20260909.runtime import Runtime as GenerationRuntime
from creamem3.experiments.locomo_g35lite_20260908.runtime import EMBED
from creamem3 import online_memory_v3 as online

old, read, seal, sha, file_sha, StopRun = writer.old, writer.read, writer.seal, writer.sha, writer.file_sha, writer.StopRun
RUN = Path(r'D:\creamem3_cache\locomo_mgtm480_writer_dev100_20260909_r1')
SMOKE = RUN.parent / 'locomo_mgtm480_writer_smoke7_20260909_r1'
ARMS = ('session_strong', 'opctx_original')


class Runtime(base.Runtime):
    def __init__(self, root, *, allow_model_calls):
        super().__init__(root, parents=(), allow_model_calls=allow_model_calls, concurrency=20, timeout=180)

    def post(self, stage, scope, model, method, body):
        if stage in {'writer', 'online', 'embedding'}:
            expected = EMBED if stage == 'embedding' else 'gemini-2.5-flash'
            if model != expected: self._halt('Unexpected fresh-memory model')
            result = GenerationRuntime.post(self, stage, scope, model, method, body)
            if stage != 'embedding':
                candidates = result.get('candidates', [])
                if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
                    self._halt('Incomplete Writer/Online generation; no partial memory accepted')
            return result
        return super().post(stage, scope, model, method, body)


def source_hashes():
    paths = [Path(__file__), Path(writer.__file__), Path(__file__).with_name('protocol.md'),
             old.FROZEN, Path(old.__file__), Path(online.__file__), Path(base.__file__),
             Path(adapter.__file__), old.ROOT/'creamem3/prompts_v2.py',
             old.ROOT/'creamem3/online_operation_context_v8.py', old.ROOT/'creamem3/semantic_context_v8.py',
             old.ROOT/'creamem3/locomo_all.py', old.ROOT/'eval/prompts.py',
             old.ROOT/'creamem3/experiments/locomo_online_only_20260909/runtime.py',
             old.ROOT/'creamem3/experiments/locomo_g35lite_20260908/runtime.py',
             old.ROOT/'creamem3/experiments/locomo_joint_revision_20260909/run.py']
    return {str(p): file_sha(p) for p in paths}


def prepare(root, smoke):
    loaded, control = old.prepare(root/'original_control_protocol')
    historical = writer.bundle()
    sources = [{'sample_id': c['sample_id'], 'records': [writer.source_record(r) for r in c['records']]} for c in loaded]
    if smoke:
        sources = [next(c for c in sources if c['sample_id'] == 'conv-30')]
        sources[0]['records'] = sources[0]['records'][:7]
    registration = {'source_hashes': source_hashes(), 'prompt_bundle_sha256': sha(historical),
        'original_control_protocol_sha256': sha(control), 'dataset_sha256': file_sha(old.DEFAULT_DATASET_PATH),
        'sources_sha256': sha(sources), 'smoke_only': smoke, 'source_sessions': sum(len(c['records']) for c in sources),
        'arms': list(ARMS), 'writer_is_new_and_shared_exactly': True, 'writer_receives_no_online_feedback': True,
        'writer_max_output': 16384, 'writer_json_attempts': 3, 'provider_retry_ceiling': 4,
        'summary_top_k': 10, 'dense_top_k': 20, 'bm25_top_k': 20, 'candidate_cap': 120, 'final_entries': 20,
        'qa_model': 'gemini-2.5-flash', 'judge_model': 'gemini-3-flash-preview', 'judge_votes': 1,
        'memory_model': 'gemini-2.5-flash', 'embedding_model': EMBED, 'embedding_dim': 768,
        'qa_prompt_sha256': sha(old.QA_PROMPT), 'judge_prompt_sha256': sha(old.JUDGE_PROMPT),
        'no_old_model_answer_imports': True, 'existing_online_and_reader_unchanged': True,
        'historical_original_controls': {'session_correct': 74, 'online_correct': 75, 'n': 100},
        'budget_note': '20contextual entries,not20atomic claims or matched tokens; report all size/cost differences',
        'no_new_source_qualifier_offline_or_writer_history_feedback': True,
        'dev_ids': control['dev_ids'], 'category_counts': control['dev_quota'],
        'previously_developed_not_unseen_test': True, 'conversations_parallel': 10, 'provider_parallel': 20}
    seal(root/'prompt_bundle.json', historical); seal(root/'registration.json', registration)
    seal(root/'source_sessions.json', sources)
    protocol = {**control, 'writer_reused_immutable': False, 'shared_fresh_writer': True,
                'mgtm480_registration_sha256': sha(registration)}
    seal(root/'protocol.json', protocol)
    return loaded, sources, registration, protocol


def graph_path(root, sample, arm): return root/'graphs'/sample/(arm+'.json')


def build_one(root, source, registration, runtime):
    sample, records = source['sample_id'], source['records']
    api = writer.make_graph_api(root, sample, runtime, registration)
    graph_identity = {'registration_sha256': sha(registration), 'source_sha256': sha(source)}
    sp = graph_path(root, sample, 'session_strong'); op = graph_path(root, sample, 'opctx_original')
    if sp.exists():
        record = read(sp)
        if record['identity'] != graph_identity: raise StopRun('Session graph identity drift')
        session = record['graph']
    else:
        session = api.build_session_graph(records, runtime.chat_fn(sample), runtime.embed, workers=2)
        session['architecture']['fact_granularity'] = 'contextually_rich_memory; atomic_fact is a legacy node label'
        seal(sp, {'identity': graph_identity, 'graph': session})
    if op.exists():
        record = read(op)
        if record['identity'] != graph_identity: raise StopRun('Online graph identity drift')
        graph = record['graph']
    else:
        graph = online.add_online_event_views(session, runtime.chat_fn(sample), runtime.embed)
        seal(op, {'identity': graph_identity, 'graph': graph})
    legal = [t['dia_id'] for r in records for t in r['turns']]
    checks = {'session': api.audit_graph(session, legal_dia_ids=legal, expected_sessions=len(records)),
              'online': api.audit_graph(graph, legal_dia_ids=legal, expected_sessions=len(records)),
              'causal': online.audit_online_graph(session, graph)}
    if any(v['status'] != 'pass' for v in checks.values()): raise StopRun('New graph structural/causal audit failed')
    if session['fact_nodes'] != graph['fact_nodes'] or session['session_packets'] != graph['session_packets']:
        raise StopRun('Online changed shared fresh Writer')
    if any(step['model_error'] for step in graph['online_fusion']['steps']): raise StopRun('Online step model failure')
    facts = session['fact_nodes']; old_source = old.old_json(old.HISTORY/sample/'graph_session.json')
    historical_facts = [f for f in old_source['fact_nodes'] if f['session_id'] in {r['session_id'] for r in records}]
    stats = {'sessions': len(records), 'new_facts': len(facts), 'old_facts': len(historical_facts),
        'new_mean_fact_chars': statistics.mean(len(f['text']) for f in facts),
        'old_mean_fact_chars': statistics.mean(len(f['text']) for f in historical_facts),
        'new_total_fact_chars': sum(len(f['text']) for f in facts),
        'old_total_fact_chars': sum(len(f['text']) for f in historical_facts),
        'online_views': graph['stats']['event_view_nodes'], 'shared_exact_writer': True,
        'accepted_operations': graph['stats']['online_accepted_operations'],
        'rejected_operations': graph['stats']['online_rejected_operations']}
    seal(root/'graphs'/sample/'audit.json', {'identity': graph_identity, 'checks': checks, 'stats': stats,
        'session_sha256': sha(session), 'online_sha256': sha(graph)})
    old.emit('MGTM480 GRAPH ' + sample + ' ' + json.dumps(stats))
    return stats


def transport_audit(root):
    counts = Counter(); usage = {}
    for path in (root/'calls/provider').glob('*/*/request.json'):
        request = read(path); final = read(path.parent/'final.json')
        raw = read(path.parent/('attempt_%02d.response.json' % final['attempt']))
        if path.parent.name != sha(request) or final['identity'] != request or final['response_sha256'] != sha(raw):
            raise StopRun('Fresh build/evaluation receipt binding drift')
        stage = request['stage']; counts[stage+'_requests'] += 1
        for intent in path.parent.glob('*.intent.json'):
            prefix = intent.name.removesuffix('.intent.json')
            if sum(intent.with_name(prefix+s).exists() for s in ('.response.json','.error.json')) != 1:
                raise StopRun('Orphan or duplicate physical receipt')
            counts['physical_attempts'] += 1
        errors = list(path.parent.glob('*.error.json')); counts['errors'] += len(errors)
        counts['recovered_requests'] += bool(errors)
        if any(not read(p)['retryable'] for p in errors): raise StopRun('Nonretryable physical error')
        stage_usage = usage.setdefault(stage, Counter())
        for k, v in raw['response'].get('usageMetadata', {}).items():
            if isinstance(v, int): stage_usage[k] += v
    if list((root/'calls/imports').glob('**/*.json')): raise StopRun('Unexpected historical model import')
    if list(root.glob('**/*.pending')): raise StopRun('Unfinished durable artifact')
    return {'counts': counts, 'observed_final_usage': usage, 'pass': True,
            'failed_attempt_billing_without_usage_unknown': True}


def finish_eval(root, loaded, registration, protocol, runtime):
    vectors_record = read(old.VECTOR_FILE); vectors = dict(zip(vectors_record['ids'], vectors_record['vectors']))
    rows = {arm: {} for arm in ARMS}; graphs = {}
    for conv in loaded:
        sample = conv['sample_id']
        for arm in ARMS:
            graph = read(graph_path(root, sample, arm))['graph']; index = base.RepairedIndex(graph)
            graphs[sample, arm] = graph
            for q in conv['qa']:
                qid = q['qa_id']
                if qid not in protocol['dev_ids']: continue
                row = read(root/'rows'/arm/(qid+'.json')); trace_record = read(root/'retrieval'/arm/(qid+'.json'))
                if row['identity']['graph_sha256'] != sha(graph) or row['identity']['protocol_sha256'] != sha(protocol):
                    raise StopRun('Scored graph/protocol identity drift')
                _, trace = index.search(q['question'], vectors[qid], runtime.chat_fn('replay/'+arm+'/'+qid))
                if trace != trace_record['trace'] or trace['rerank']['fallback']: raise StopRun('Selection replay/fallback failure')
                base.check_bound(root, q, row); rows[arm][qid] = row
    reports = {a: old.summarize(root, a, protocol) for a in ARMS}
    pairs = base.pair_summary([(rows['opctx_original'][q], rows['session_strong'][q]) for q in protocol['dev_ids']])
    comparisons = {a: base.pair_summary([(rows[a][q], read(base.RUN/'rows'/a/(q+'.json'))) for q in protocol['dev_ids']]) for a in ARMS}
    result = {'registration_sha256': sha(registration), 'reports': reports, 'new_online_vs_new_session': pairs,
        'versus_historical_original_writer': comparisons, 'audit': transport_audit(root),
        'qa_judge_bindings': 400, 'selection_replays': 200, 'only_dev100_not_full': True,
        'writer_packet_shared_exactly': True, 'not_token_matched': True}
    seal(root/'complete.json', result); old.emit('MGTM480 COMPLETE ' + json.dumps(reports))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare','smoke','run','audit'])
    args = parser.parse_args(); smoke = args.command == 'smoke'; root = SMOKE if smoke else RUN
    with old.Lease(root):
        loaded, sources, registration, protocol = prepare(root, smoke)
        if args.command == 'prepare': return
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/'.env', override=False)
        runtime = Runtime(root/'calls', allow_model_calls=args.command != 'audit')
        original_urlopen = adapter.install(runtime.log)
        try:
            if not smoke:
                smoke_complete = read(SMOKE/'complete.json')
                if not smoke_complete['audit']['pass'] or smoke_complete['source_hashes'] != registration['source_hashes']:
                    raise StopRun('Matching seven-session smoke is required before formal run')
            with ThreadPoolExecutor(max_workers=10) as pool:
                built = [f.result() for f in as_completed([pool.submit(build_one,root,s,registration,runtime) for s in sources])]
            if smoke:
                result = {'registration_sha256': sha(registration), 'source_hashes': registration['source_hashes'],
                          'stats': built, 'audit': transport_audit(root), 'not_a_QA_score': True}
                seal(root/'complete.json', result); old.emit('MGTM480 SMOKE COMPLETE ' + json.dumps(built)); return
            evaluation = loader.loader.load_isolated('mgtm480_evaluation', old.__file__)
            evaluation.graph_api.GraphMemoryIndex = base.RepairedIndex
            def get_graph(sample, arm, unused_protocol):
                record = read(graph_path(root,sample,arm))
                if record['identity']['registration_sha256'] != sha(registration): raise StopRun('Evaluation graph registration')
                return record['graph']
            evaluation.get_source_graph = get_graph
            if args.command == 'run': evaluation.evaluate(root, ARMS, loaded, protocol, runtime, qa_workers=20)
            replay = Runtime(root/'calls', allow_model_calls=False)
            finish_eval(root, loaded, registration, protocol, replay)
        except BaseException as exc:
            runtime.stopped.set()
            seal(root/('stop_%d.json' % time.time_ns()), {'utc': base.now(), 'type': type(exc).__name__, 'message': str(exc)[:600]})
            raise
        finally: adapter.restore(original_urlopen)


if __name__ == '__main__': main()

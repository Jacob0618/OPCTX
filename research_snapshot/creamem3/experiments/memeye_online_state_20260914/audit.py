"""Audit four delivery stages without provider calls or changes to old runs.

Support closure is structural, not answer relevance/entailment. Explicit fact-ID
mentions in a generated packet are a traceability metric, not semantic recall.
"""
from collections import Counter
from pathlib import Path
import json
import re
from creamem3.experiments.memeye_budget_online_20260914 import config as baseline
from creamem3.experiments.iterative_memory_20260907 import candidate_context_policy

ROOT = Path(r'D:\creamem3_cache\memeye_online_state_20260914_r1')


def plain(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run():
    reg = baseline.checked(baseline.ROOT / 'registration.json')
    done = baseline.checked(baseline.ROOT / 'independent_audit.json')
    if done['status'] != 'PASS' or done['registration_sha256'] != baseline.sha(reg):
        raise RuntimeError('Baseline audit/registration mismatch')
    authority, _ = baseline.direct._reader_authority()
    graphs = {}
    graph_stats = {}
    for key, receipt in reg['source_graphs'].items():
        if baseline.file_sha(Path(receipt['path'])) != receipt['sha256']:
            raise RuntimeError('Source graph drift')
        graphs[key] = plain(receipt['path'])
    for key, g in graphs.items():
        if not key.startswith('online::'):
            continue
        task = key.split('::', 1)[1]
        s = graphs['session::' + task]
        if s['fact_nodes'] != g['fact_nodes']:
            raise RuntimeError('Writer differs between arms')
        views = [v for v in g['summary_nodes'] if v['node_type'] == 'event_view']
        ops = Counter()
        for step in g['online_fusion']['steps']:
            for accepted in step.get('accepted', []):
                op = accepted.get('operation', {})
                ops[op.get('operation', 'unknown') if isinstance(op, dict) else str(op)] += 1
        graph_stats[task] = {'identical_raw_facts': True, 'facts': len(g['fact_nodes']),
                             'event_views': len(views), 'operations': dict(ops)}
    reports = {}
    for mode in baseline.MODES:
        if not mode.endswith('_online'):
            continue
        counts = Counter()
        query_rows = []
        prepared = baseline.checked(baseline.ROOT / 'prepared' / (mode + '.json'))
        for item in prepared['rows']:
            q = item['query']; qid = q['row_id']; task = q['task']
            g = graphs['online::' + task]
            views = {v['id']: v for v in g['summary_nodes'] if v['node_type'] == 'event_view'}
            pool = baseline.checked(Path(item['pool_file']))
            selected = set(item['selected_fact_ids'])
            routed = [r['summary_id'] for r in pool['trace']['top_summaries'] if r['node_type'] == 'event_view']
            routed_claims = []
            for vid in routed:
                v = views[vid]; members = set(v['member_fact_ids'])
                counts['routed_view_occurrences'] += 1
                counts['routed_views_with_any_selected_member'] += bool(members & selected)
                counts['routed_views_with_all_selected_members'] += members <= selected
                for i, claim in enumerate(v.get('summary_claims', [])):
                    support = set(claim['support_fact_ids'])
                    if not support or not support <= set(pool['pool_ids']):
                        raise RuntimeError('Routed claim support absent from expanded pool')
                    status = 'full' if support <= selected else 'partial' if support & selected else 'none'
                    counts['routed_claims_' + status] += 1
                    counts['routed_claims_larger_than_final_cap'] += len(support) > item['max_units']
                    routed_claims.append({'view_id': vid, 'claim_index': i, 'text': claim['text'],
                                         'support': sorted(support), 'selected_support': sorted(support & selected),
                                         'structural_support_status': status})
            counts['questions'] += 1
            counts['questions_with_event_route'] += bool(routed)
            counts['questions_with_any_full_routed_claim'] += any(r['structural_support_status'] == 'full' for r in routed_claims)
            counts['rendered_event_prose_questions'] += bool(item['render']['rendered_event_view_ids'])
            raw_visible_ids = set(re.findall(r'\bF\d{5}\b', item['context'])) & selected
            counts['questions_with_explicit_selected_fact_ids_in_reader_input'] += bool(raw_visible_ids)
            row = baseline.checked(baseline.ROOT / 'rows' / mode / (baseline.sha(qid) + '.json'))
            reader = {'route': q['reader_route']}
            if q['reader_route'] == 'two_stage':
                counts['two_stage_questions'] += 1
                meta = row['reader_stages'][0]
                response = plain(meta['cache_path'])
                if baseline.sha(response['text']) != meta['response_sha256']:
                    raise RuntimeError('Reader text receipt mismatch')
                packet = baseline.direct.predict._parse_json_object(response['text']) or {'_failed': True}
                context = candidate_context_policy.rendered_context(authority['render'], packet,
                            item['context'], 'candidate_first', 'candidate_last')
                if baseline.sha(context) != row['actual_final_context_sha256']:
                    raise RuntimeError('Final QA context mismatch')
                failed = bool(packet.get('_failed'))
                counts['focused_invalid_json'] += failed
                counts['raw_context_fallback'] += context == item['context']
                counts['nonempty_candidate_answers'] += bool(packet.get('candidate_answer'))
                cited = set(re.findall(r'\bF\d{5}\b', context))
                counts['questions_with_explicit_selected_fact_id_mentions'] += bool(cited & selected)
                reader.update(focused_parse_failed=failed, raw_context_fallback=context == item['context'],
                              explicit_selected_fact_ids=sorted(cited & selected),
                              candidate_answer=packet.get('candidate_answer', ''),
                              response_path=meta['cache_path'])
            else:
                counts['fused_questions'] += 1
            session_mode = mode.replace('_online', '_session')
            session_pool = baseline.checked(baseline.ROOT / 'pools' /
                            (mode.split('_')[0] + '_session') / (baseline.sha(qid) + '.json'))
            unique = set(pool['pool_ids']) - set(session_pool['pool_ids'])
            counts['online_only_candidate_facts'] += len(unique)
            counts['online_only_selected_facts'] += len(unique & selected)
            query_rows.append({'row_id': qid, 'routed_views': routed, 'claims': routed_claims,
                               'online_only_pool_ids': sorted(unique), 'online_only_selected_ids': sorted(unique & selected),
                               'reader': reader})
        reports[mode] = {'counts': dict(counts), 'rows': query_rows}
        print('FOUR-STAGE AUDIT', mode, dict(counts), flush=True)
    result = {'schema': 'memeye.four_stage_delivery_audit.v2', 'model_calls': 0,
              'baseline_registration_sha256': baseline.sha(reg), 'source_hashes': reg['source_graphs'],
              'graphs': graph_stats, 'reports': reports,
              'caveats': ['Structural support completeness does not establish claim relevance or truth.',
                          'The legacy raw renderer does not expose fact IDs: output ID absence cannot measure semantic evidence loss.',
                          'No gold-derived evidence sets or answer-based optimization.']}
    baseline.put(ROOT / 'four_stage_audit_v2.json', result)
    print('AUDIT COMPLETE', ROOT / 'four_stage_audit_v2.json', flush=True)


if __name__ == '__main__':
    run()

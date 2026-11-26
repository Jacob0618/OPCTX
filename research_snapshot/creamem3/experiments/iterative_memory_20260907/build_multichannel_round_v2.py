"""Full371 no-API multi-query selection with the original dense display order."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from creamem3 import graph_memory_v2 as memory
from creamem3.experiments.iterative_memory_20260907 import prepare_variants as data
from creamem3.experiments.iterative_memory_20260907 import multichannel_selector_v2 as selector
from creamem3.experiments.iterative_memory_20260907 import query_projections as qp
from creamem3.experiments.iterative_memory_20260907 import projection_embeddings as pe
from creamem3.experiments.mec_transfer_20260906 import run_memeye as adapter

ROOT = data.ROOT
PANEL_FILE_SHA = '842489f1b3bc7bb3faca43f55e0be903b8ae7010ff72384974a9ce427edd4bc1'
METHODS = {'r33': 'q0_plus_all_spans', 'r34': 'q0_plus_anchored_attributes', 'r35': 'q0_only'}


def query_texts(question, projection, method):
    if method not in METHODS.values():
        raise ValueError('unknown structural query-set policy')
    if projection['question'] != question or projection['subqueries'][0] != question:
        raise RuntimeError('question/projection binding drift')
    output = [{'query_id': 'q0', 'query': question, 'origin': 'original_complete_question'}]
    if method == 'q0_only':
        return output
    if projection['status'] != 'valid':
        if projection['status'] not in ('fallback', 'provider_failure_original_only'):
            raise RuntimeError('unknown preserved planner status')
        if projection['subqueries'] != [question]:
            raise RuntimeError('fallback has nonoriginal queries')
        return output
    if method == 'q0_plus_all_spans':
        candidates = [p for p in projection['projections'] if p['kind'] == 'all_extracted_roles']
    else:
        roles = projection['role_quotes']
        anchors = roles['entity'] + roles['relation'] + roles['scope']
        candidates = [qp.combine_spans(question, anchors + [attribute], f'anchored_attribute_{j}')
                      for j, attribute in enumerate(roles['attribute'])]
        # The planner deduplicates an attribute identical to the all-spans
        # bundle. Reconstruct its same exact spans here, not a new model plan.
    seen = {question}
    for p in candidates:
        if p['text'] not in projection['subqueries']:
            raise RuntimeError('projection query has no sealed embedding text')
        if p['text'] not in seen:
            seen.add(p['text'])
            output.append({'query_id': f'q{len(output)}', 'query': p['text'], 'origin': p['kind']})
    return output


def load_projection_panel():
    path = pe.ROOT / 'embedded_panel.json'
    if qp.file_sha(path) != PANEL_FILE_SHA:
        raise RuntimeError('sealed native3072 projection bank drift')
    embedded = qp.read_sealed(path)
    panel = qp.read_sealed(pe.ROOT / 'panel.json')
    if embedded['panel_sha256'] != qp.sha(panel) or len(embedded['rows']) != 371:
        raise RuntimeError('downstream projection panel binding drift')
    q0 = pe.load_q0_vectors(panel['rows'])
    by_id = {}
    for source, row in zip(panel['rows'], embedded['rows'], strict=True):
        if source['row_id'] != row['row_id'] or source['question'] != row['question']:
            raise RuntimeError('projection population order drift')
        vectors = {}
        for item in row['projections']:
            vector = item['vector']
            if len(vector) != 3072 or qp.sha(vector) != item['vector_sha256'] or not all(math.isfinite(v) for v in vector):
                raise RuntimeError('projection vector shape/hash drift')
            if item['kind'] == 'original' and vector != q0[row['row_id']]:
                raise RuntimeError('original q0 vector differs from original cached vector')
            vectors[item['text']] = vector
        by_id[row['row_id']] = {'source': source, 'vectors': vectors}
    return by_id, embedded


def display_ids(ids, fact_ids, dense):
    score = dict(zip(fact_ids, dense, strict=True))
    if len(set(ids)) != len(ids) or any(fid not in score for fid in ids):
        raise RuntimeError('selected ID census/drift')
    return sorted(ids, key=lambda fid: (-float(score[fid]), fid))


def build(round_id):
    if round_id not in METHODS:
        raise ValueError('this numerical v2 adapter supports only declared r33/r34/r35')
    method = METHODS[round_id]
    features = data.read_seal(ROOT / 'features/complete.json')
    projections, embedded = load_projection_panel()
    if features['population_n'] != 371 or len(projections) != 371:
        raise RuntimeError('full371 population required')
    code_paths = [Path(__file__), Path(selector.__file__), Path(selector.base.__file__), Path(selector.lexical.__file__),
                  Path(adapter.__file__), Path(adapter.raw.__file__), Path(data.__file__), Path(memory.__file__),
                  Path(qp.__file__), Path(pe.__file__), Path(selector.numerical.__file__)]
    run_root = ROOT / 'selection_runs' / round_id
    config = data.seal(run_root / 'preregistration.json', {'round_id': round_id, 'method': method,
        'source_hashes': {str(p): qp.file_sha(p) for p in code_paths},
        'numerical_solver_policy': selector.numerical.VERSION, 'presolve': False,
        'old_selections_modified': False, 'numerical_failure_policy': 'return independently feasible partial and stop later objectives; never certify an inconsistent raw/reconstructed/dual receipt',
        'feature_complete_sha256': qp.file_sha(ROOT / 'features/complete.json'), 'embedded_panel_file_sha256': PANEL_FILE_SHA,
        'projection_panel_status_counts': embedded['planner_status_counts'], 'planner_model_successes': 370,
        'planner_failures_preserved': 1, 'population_n': 371, 'max_units': 50, 'whole_raw_facts': True,
        'query_policy': method, 'q0_always_retained': True, 'exact_duplicate_queries_and_utility_arrays_merged': True,
        'objective': 'maximum q0 literal coverage locked, same-domain channel ideals, max-min normalized channel sum utility, normalized sum, minimum shared rendering cost',
        'display_order': 'q0 Dense score descending, ID ascending ties, then unchanged original renderer session grouping; NOT ID sorting',
        'budget_rule': 'original globalDense50 raw-rendered Unicode characters; core additive budget=actual+1',
        'reader': 'unchanged original Reader/Judge prompts/models/ceilings; original raw renderer',
        'time_limit_seconds_per_stage': 5.0, 'new_provider_calls': 0, 'gold_or_xy_inputs': False})
    graphs, pools, indices = {}, {}, {}
    rows, diagnostics = [], []
    for ordinal, reference in enumerate(features['files'], 1):
        if qp.file_sha(Path(reference['path'])) != reference['sha256']:
            raise RuntimeError('original score-free feature file drift')
        item = data.read_seal(Path(reference['path']))
        query = item['query']
        task, row_id = query['task'], query['row_id']
        projection = projections[row_id]
        if projection['source']['question'] != query['question']:
            raise RuntimeError('feature/question mismatch')
        if task not in graphs:
            graph_path = Path(item['graph']['path'])
            if qp.file_sha(graph_path) != item['graph']['sha256']:
                raise RuntimeError('original immutable graph drift')
            graph = data.direct._read_object(graph_path, 'original raw graph')
            graphs[task] = graph
            pools[task] = adapter.adapt_facts(graph)
            indices[task] = memory.GraphMemoryIndex(graph, candidate_cap=len(graph['fact_nodes']), rerank_top_k=50)
        graph, index = graphs[task], indices[task]
        facts, group_costs = pools[task]
        if [f['id'] for f in facts] != item['fact_ids'] or [f['id'] for f in index.facts] != item['fact_ids']:
            raise RuntimeError('fact/score/embedding column drift')
        dense, bm25 = item['dense_scores'], item['bm25_scores']
        original_text = query['question']
        reconstructed_q0 = list(map(float, index._scores(index.fact_matrix, projection['vectors'][original_text])))
        if reconstructed_q0 != dense:
            raise RuntimeError('q0 cosine bytes differ from frozen original features')
        baseline_ids = display_ids(item['fact_ids'], item['fact_ids'], dense)[:50]
        baseline_context, _ = adapter.render_selection(graph, baseline_ids)
        budget = len(baseline_context)
        desired = query_texts(original_text, projection['source']['projection'], method)
        query_channels = []
        for q in desired:
            if q['query_id'] == 'q0':
                ds, bs = dense, bm25
            else:
                ds = list(map(float, index._scores(index.fact_matrix, projection['vectors'][q['query']])))
                bm = {str(fid): float(v) for fid, v in index.bm25.search(q['query'], len(facts))}
                bs = [bm.get(str(f['id']), 0.0) for f in facts]
            query_channels.append({k: q[k] for k in ('query_id', 'query')} | {'dense_scores': ds, 'bm25_scores': bs})
        input_identity = {'feature_file_sha256': reference['sha256'], 'query_channels_sha256': qp.sha(query_channels),
                          'preregistration_sha256': config['seal_sha256'], 'actual_char_budget': budget}
        save_path = run_root / 'rows' / (qp.sha(row_id) + '.json')
        if save_path.exists():
            saved = data.read_seal(save_path)
            if saved['input_identity'] != input_identity:
                raise RuntimeError('row resume contract drift')
        else:
            result = selector.select_multichannel(facts, original_text, query_channels,
                      max_units=50, char_budget=budget+1, group_costs=group_costs, time_limit=5.0)
            ids = display_ids(result['selected_ids'], item['fact_ids'], dense)
            if ids != result['selected_ids'] or not ids or len(ids) > 50:
                raise RuntimeError('selector display/count drift; no outside fill/sort repair')
            context, render = adapter.render_selection(graph, ids)
            if len(context) > budget or result['cost']-1 != len(context):
                raise RuntimeError('actual context budget or shared-cost reconstruction drift')
            saved = data.seal(save_path, {'query': query, 'input_identity': input_identity, 'query_channels': query_channels,
                'query_origins': desired, 'plan_status': projection['source']['projection']['status'],
                'selected_fact_ids': ids, 'render': render, 'context_chars': len(context), 'baseline_chars': budget,
                'optimization': result})
        context, render = adapter.render_selection(graph, saved['selected_fact_ids'])
        if render != saved['render'] or display_ids(saved['selected_fact_ids'], item['fact_ids'], dense) != saved['selected_fact_ids']:
            raise RuntimeError('saved renderer or original Dense display order drift')
        rows.append({'row_id': row_id, 'question': original_text, 'task': task, 'selected_fact_ids': saved['selected_fact_ids'],
                     'expected_context_sha256': render['context_sha256'], 'expected_context_chars': len(context),
                     'selection_provenance': {'selector': method, 'preregistration_sha256': config['seal_sha256'],
                        'selection_row_sha256': saved['seal_sha256'], 'baseline_char_budget': budget}})
        diagnostics.append({'row_id': row_id, 'units': len(saved['selected_fact_ids']), 'chars': len(context), 'budget': budget,
            'queries': len(query_channels), 'channels': len(saved['optimization']['channel_ideal_values']),
            'literal_coverage': saved['optimization']['literal_coverage'], 'optimal': saved['optimization']['certified_optimal'],
            'seconds': saved['optimization']['seconds'], 'plan_status': saved['plan_status']})
        if ordinal % 25 == 0 or ordinal == 371:
            print(f'{round_id}: offline selection {ordinal}/371', flush=True)
    manifest = {'schema': 'memeye.iterative.context_manifest.v1', 'round_id': round_id,
        'method': {'name': method, 'description': config['display_order'], 'selection_preregistration_sha256': config['seal_sha256']},
        'reader': {}, 'workers': 8, 'render_order': 'original', 'rows': rows}
    path = ROOT / 'proposals' / f'{round_id}.json'
    data.direct._write_once(path, manifest)
    done = data.seal(run_root / 'complete.json', {'round_id': round_id, 'population_n': len(rows), 'manifest_path': str(path),
        'manifest_sha256': qp.file_sha(path), 'diagnostics': diagnostics, 'new_provider_calls': 0,
        'preregistration_sha256': config['seal_sha256']})
    return {'round_id': round_id, 'population_n': len(rows), 'manifest_path': str(path), 'seal_sha256': done['seal_sha256'],
            'certified_optimal_rows': sum(d['optimal'] for d in diagnostics), 'mean_context_chars': sum(d['chars'] for d in diagnostics)/len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round-id', choices=list(METHODS), required=True)
    print(json.dumps(build(parser.parse_args().round_id), indent=2))

from itertools import combinations
from types import SimpleNamespace
from scipy.optimize import OptimizeResult
import random

import numpy as np
import pytest

from creamem3 import budget_evidence_v1 as old
from creamem3.experiments.iterative_memory_20260907 import multichannel_selector_v2 as m


def channel(question, dense, bm25, qid='q0'):
    return {'query_id': qid, 'query': question, 'dense_scores': dense, 'bm25_scores': bm25}


def brute(units, question, channels, budget, k, gc, min_units=0):
    data = m.base.validate_inputs(units, channels[0]['dense_scores'], channels[0]['bm25_scores'],
            max_units=k, min_units=min_units, char_budget=budget, group_costs=gc)
    fs = data['units']; n = len(fs)
    utility = m.prepare_channels(question, channels, n, data['original_indices'])['utilities']
    req = m.lexical.tokens(question) - m.lexical.STOPWORDS
    feasible = []
    for size in range(min_units, min(n, k)+1):
        for subset in combinations(range(n), size):
            chosen = set(subset)
            if any(i in chosen and j in chosen for i, j in data['overlap_pairs']):
                continue
            if any(len(chosen.intersection(g)) > 1 for g in data['duplicate_groups']):
                continue
            cost = m.base.selected_cost(fs, subset, gc)
            if cost <= budget:
                covered = req.intersection(set().union(*(m.lexical.tokens(fs[i]['text']) for i in subset)))
                feasible.append((subset, len(covered), cost))
    coverage = max(row[1] for row in feasible)
    domain = [row for row in feasible if row[1] == coverage]
    ideals = {key: max(float(u[list(row[0])].sum()) for row in domain) for key, u in utility.items()}
    ideals = {key: value for key, value in ideals.items() if value > m.TOL}
    def objective(row):
        ratios = [float(utility[key][list(row[0])].sum()) / value for key, value in ideals.items()]
        return (row[1], min(ratios, default=0), sum(ratios), -row[2])
    expected = max(map(objective, domain))
    return fs, utility, ideals, expected


@pytest.mark.parametrize('seed', range(36))
def test_three_query_objective_against_all_subsets(seed):
    rng = random.Random(seed)
    n = 6
    units = [{'id': f'f{n-i}', 'text': rng.choice(['red blue', 'blue', 'red', 'green', 'none']),
              'group': str(i % 2), 'cost': rng.randint(1, 4), 'sources': [f's{i//2}']} for i in range(n)]
    question = 'red blue green?'
    channels = [channel(question, [rng.random() for _ in range(n)], [rng.randrange(4) for _ in range(n)])]
    channels += [channel(f'projection {j}', [rng.random() for _ in range(n)], [rng.randrange(4) for _ in range(n)], f'q{j}') for j in [1, 2]]
    gc = {'0': 2, '1': 1}
    fs, utilities, ideals, expected = brute(units, question, channels, 9, 3, gc)
    result = m.select_multichannel(units, question, channels, max_units=3, char_budget=9, group_costs=gc)
    selected = [i for i, f in enumerate(fs) if f['id'] in result['selected_ids']]
    ratios = [float(utilities[key][selected].sum())/ideal for key, ideal in ideals.items()]
    actual = (result['literal_coverage'], min(ratios, default=0), sum(ratios), -result['cost'])
    assert actual == pytest.approx(expected, abs=1e-6)
    assert result['channel_ideal_values'] == pytest.approx(ideals)
    # The exhaustive oracle above verifies the returned objective separately.
    # A solver raw integer/constraint residual above the declared 1e-7 gate
    # may conservatively withhold a certificate even for this exact optimum.
    if not result['certified_optimal']:
        assert result['termination'] == 'partial_stage_incumbent'
        assert result['solver_stages'][-1]['feasible_incumbent']
        assert result['solver_stages'][-1]['numerical_audit']['check_failures']
    dense = dict(zip([u['id'] for u in units], channels[0]['dense_scores']))
    assert result['selected_ids'] == sorted(result['selected_ids'], key=lambda fid: (-dense[fid], fid))


def test_q0_only_matches_frozen_lexical_baseline_objectives_and_display():
    facts = [{'id': f'f{i}', 'text': text, 'cost': i+1, 'group': str(i % 2), 'sources': [str(i)]}
             for i, text in enumerate(['red', 'blue', 'red blue', 'green', 'other'])]
    ds, bs = [1, 3, 2, 4, 5], [1, 2, 4, 0, 3]
    result = m.select_multichannel(facts, 'red blue', [channel('red blue', ds, bs)], max_units=3, char_budget=10, group_costs={'0': 1, '1': 1})
    prior = old.select_evidence(facts, 'red blue', ds, bs, max_facts=3, char_budget=10, group_costs={'0': 1, '1': 1})
    assert result['selected_ids'] == prior['selected_ids']
    assert result['cost'] == prior['cost']
    assert result['literal_coverage'] == prior['objectives']['literal_coverage']


def test_identical_query_and_identical_objective_channels_merge():
    qs = [channel('red blue', [1, 2, 3], [2, 4, 6]), channel('red blue', [1, 2, 3], [2, 4, 6], 'duplicate'),
          channel('blue', [100, 200, 300], [0, 0, 0], 'q1')]
    result = m.prepare_channels('red blue', qs, 3)
    assert result['query_aliases'] == {'duplicate': 'q0'}
    assert list(result['utilities']) == ['q0:dense']
    assert result['channel_aliases'] == {'q0:bm25': 'q0:dense', 'q1:dense': 'q0:dense'}
    assert result['inactive_channels'] == ['q1:bm25']


def test_duplicate_query_inconsistent_arrays_rejected():
    qs = [channel('q', [1, 2], [0, 1]), channel('q', [1, 3], [0, 1], 'q1')]
    with pytest.raises(ValueError, match='inconsistent'):
        m.prepare_channels('q', qs, 2)


@pytest.mark.parametrize('change', ['not_q0', 'shortened_original', 'unknown_field', 'wrong_length', 'nan', 'negative_bm25'])
def test_input_boundary(change):
    q = channel('complete original', [1, 2], [0, 1])
    if change == 'not_q0': q['query_id'] = 'first'
    if change == 'shortened_original': q['query'] = 'original'
    if change == 'unknown_field': q['gold'] = 'red'
    if change == 'wrong_length': q['dense_scores'] = [1]
    if change == 'nan': q['dense_scores'] = [float('nan'), 2]
    if change == 'negative_bm25': q['bm25_scores'] = [0, -1]
    with pytest.raises(ValueError):
        m.prepare_channels('complete original', [q], 2)


def test_overlap_sources_and_shared_header_with_exact_count():
    units = [{'id': 'a', 'text': 'red', 'cost': 3, 'group': 'g', 'sources': ['s'], 'span': {'fact_id': 'p', 'start': 0, 'end': 3}},
             {'id': 'b', 'text': 'blue', 'cost': 4, 'group': 'g', 'sources': ['s'], 'span': {'fact_id': 'p', 'start': 2, 'end': 6}},
             {'id': 'c', 'text': 'red', 'cost': 3, 'group': 'g', 'sources': ['s']},
             {'id': 'd', 'text': 'green', 'cost': 5, 'group': 'h', 'sources': ['other']}]
    qs = [channel('red blue green', [4, 3, 2, 1], [2, 3, 2, 1])]
    _, _, _, expected = brute(units, 'red blue green', qs, 14, 2, {'g': 2, 'h': 3}, 2)
    result = m.select_multichannel(units, 'red blue green', qs, max_units=2, min_units=2, char_budget=14, group_costs={'g': 2, 'h': 3})
    assert result['literal_coverage'] == expected[0]
    assert result['selected_count'] == 2
    assert not {'a', 'b'} <= set(result['selected_ids'])
    assert not {'a', 'c'} <= set(result['selected_ids'])
    assert result['cost'] <= 14


def test_unsupported_literal_terms_remain_visible_and_flat_channels_minimize_cost():
    units = [{'id': 'z', 'text': 'red', 'cost': 5, 'group': 'g'}, {'id': 'a', 'text': 'red', 'cost': 1, 'group': 'g'}]
    result = m.select_multichannel(units, 'red cyan', [channel('red cyan', [0, 0], [0, 0])], char_budget=20, group_costs={'g': 2})
    assert result['selected_ids'] == ['a']
    assert result['unsupported_requirements'] == ['cyan']
    assert result['cost'] == 3


def test_timeout_returns_feasible_partial_not_optimal(monkeypatch):
    monkeypatch.setattr(m.numerical, 'scipy_milp', lambda *a, **k: OptimizeResult(x=None, fun=None, status=1, message='time limit', mip_gap=None, mip_dual_bound=None))
    units = [{'id': 'a', 'text': 'red', 'cost': 1, 'group': 'g'}]
    result = m.select_multichannel(units, 'red', [channel('red', [1], [1])], char_budget=1, group_costs={'g': 0})
    assert result['selected_ids'] == []
    assert result['termination'] == 'partial_stage_incumbent'
    assert not result['certified_optimal']
    with pytest.raises(RuntimeError, match='no feasible incumbent'):
        m.select_multichannel(units, 'red', [channel('red', [1], [1])], max_units=1, min_units=1, char_budget=1, group_costs={'g': 0})

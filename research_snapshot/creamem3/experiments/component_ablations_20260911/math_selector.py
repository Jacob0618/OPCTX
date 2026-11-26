"""Literal coverage then multi-query Pareto evidence utility; no model calls.

An optimal certificate concerns this finite surrogate, not answer recall.
The original question defines literal requirements and the display order.
Every channel ideal is solved on the same coverage-optimal feasible domain.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from creamem3 import budget_evidence_v1 as lexical
from creamem3.experiments.iterative_memory_20260907 import numerical_solver_v2 as numerical
milp = numerical.milp
from creamem3.experiments.iterative_memory_20260907 import set_selectors as base

VERSION = 'multichannel.component_ablations.v1'
TOL = 1e-7


def prepare_channels(question: str, query_channels: Sequence[Mapping], n: int, order=None):
    if not isinstance(question, str) or not question.strip() or not query_channels:
        raise ValueError('nonempty original question and q0 channel required')
    if query_channels[0].get('query_id') != 'q0' or query_channels[0].get('query') != question:
        raise ValueError('first channel must be q0 with the complete original question')
    if len({q['query_id'] for q in query_channels}) != len(query_channels):
        raise ValueError('duplicate query ID')
    order = list(range(n)) if order is None else list(order)
    unique, query_aliases, seen_text = [], {}, {}
    for query in query_channels:
        if set(query) != {'query_id', 'query', 'dense_scores', 'bm25_scores'}:
            raise ValueError('query channel accepts only query ID/text and Dense/BM25 arrays')
        if not isinstance(query['query_id'], str) or not query['query_id'] or not isinstance(query['query'], str) or not query['query'].strip():
            raise ValueError('query ID/text must be nonempty strings')
        ds, bs = np.asarray(query['dense_scores'], dtype=float), np.asarray(query['bm25_scores'], dtype=float)
        if ds.shape != (n,) or bs.shape != (n,) or not np.isfinite(ds).all() or not np.isfinite(bs).all() or (bs < 0).any():
            raise ValueError('channel shape/nonfinite/negative BM25 drift')
        if query['query'] in seen_text:
            prior = seen_text[query['query']]
            if not np.array_equal(ds, prior['dense_raw']) or not np.array_equal(bs, prior['bm25_raw']):
                raise ValueError('identical query text has inconsistent scores')
            query_aliases[query['query_id']] = prior['query_id']
            continue
        item = {'query_id': query['query_id'], 'query': query['query'], 'dense_raw': ds, 'bm25_raw': bs}
        seen_text[query['query']] = item
        unique.append(item)
    utilities, channel_aliases, inactive = {}, {}, []
    for query in unique:
        for kind in ('dense', 'bm25'):
            key = query['query_id'] + ':' + kind
            utility = base.percentile_utility(query[kind + '_raw'][order], positive_only=kind == 'bm25')
            if not np.any(utility):
                inactive.append(key)
                continue
            # Equal raw channels necessarily merge. Identical empirical utility
            # arrays merge as well: they are the same finite objective even if
            # raw score scales differ, and must not double the sum tiebreak.
            duplicate = next((k for k, u in utilities.items() if np.array_equal(u, utility)), None)
            if duplicate is not None:
                channel_aliases[key] = duplicate
            else:
                utilities[key] = utility
    return {'utilities': utilities, 'query_aliases': query_aliases, 'channel_aliases': channel_aliases,
            'inactive_channels': inactive, 'unique_queries': [q['query_id'] for q in unique],
            'q0_dense': unique[0]['dense_raw'][order]}


def select_multichannel(units: Sequence[Mapping[str, Any]], question: str,
                        query_channels: Sequence[Mapping[str, Any]], *, max_units: int = 50,
                        char_budget: int, group_costs: Mapping[str, int], min_units: int = 0,
                        time_limit: float = 5.0, coverage_first: bool = True, balance: bool = True):
    if not isinstance(coverage_first, bool) or not isinstance(balance, bool):
        raise ValueError('component switches must be booleans')
    if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('positive finite per-stage time limit required')
    if not query_channels:
        raise ValueError('q0 channel required')
    began = time.perf_counter()
    data = base.validate_inputs(units, query_channels[0]['dense_scores'], query_channels[0]['bm25_scores'],
               max_units=max_units, char_budget=char_budget, group_costs=group_costs, min_units=min_units)
    fs, groups, gc = data['units'], data['groups'], data['group_costs']
    channels = prepare_channels(question, query_channels, len(fs), data['original_indices'])
    utilities, dense = channels['utilities'], channels['q0_dense']
    labels = sorted(lexical.tokens(question) - lexical.STOPWORDS)
    f_tokens = [lexical.tokens(f['text']) for f in fs]
    supports = [[i for i, ts in enumerate(f_tokens) if term in ts] for term in labels]
    n, nr, ng = len(fs), len(labels), len(groups)
    y0, z0, tau = n, n + nr, n + nr + ng
    nv = tau + 1
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[:n] = 1
    rows, lows, highs = [], [], []
    def add(row, lo=-np.inf, hi=np.inf):
        rows.append(dict(row)); lows.append(float(lo)); highs.append(float(hi))
    add({i: 1 for i in range(n)}, lo=min_units, hi=max_units)
    members = defaultdict(list)
    costs = np.zeros(nv)
    for i, fact in enumerate(fs):
        members[fact['group']].append(i)
        costs[i] = fact['cost']
    group_indices = {g: z0+j for j, g in enumerate(groups)}
    for group, indices in members.items():
        z = group_indices[group]
        costs[z] = gc[group]
        for i in indices:
            add({i: 1, z: -1}, hi=0)
        add({z: 1, **{i: -1 for i in indices}}, hi=0)
    add({i: value for i, value in enumerate(costs) if value}, hi=char_budget)
    for j, indices in enumerate(supports):
        add({y0+j: 1, **{i: -1 for i in indices}}, hi=0)
        for i in indices:
            add({i: 1, y0+j: -1}, hi=0)
    for i, j in data['overlap_pairs']:
        add({i: 1, j: 1}, hi=1)
    for indices in data['duplicate_groups']:
        add({i: 1 for i in indices}, hi=1)
    stages, ideals = [], {}
    current = np.zeros(nv) if min_units == 0 else None
    stopped = False
    returned_from = 'known_empty_feasible_set' if current is not None else None
    coverage_optimum = None

    def solve(name, objective, lock=False, integer=False):
        nonlocal current, stopped, returned_from
        rr, cc, vv = [], [], []
        for j, row in enumerate(rows):
            for i, value in row.items():
                rr.append(j); cc.append(i); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), nv)).tocsc()
        start = time.perf_counter()
        result = milp(-objective, integrality=integrality, bounds=Bounds(0, 1),
               constraints=LinearConstraint(matrix, lows, highs),
               options={'time_limit': float(time_limit), 'mip_rel_gap': 0.0, 'presolve': False})
        reconstructed, violation = None, None
        feasible = result.x is not None
        if feasible:
            x = np.asarray(result.x, dtype=float)
            feasible = bool(np.isfinite(x).all() and np.max(abs(x[:n] - np.rint(x[:n])), initial=0) < 1e-5)
            if feasible:
                reconstructed = np.zeros(nv)
                reconstructed[:n] = np.rint(x[:n])
                chosen = set(np.flatnonzero(reconstructed[:n] > .5))
                for j, ss in enumerate(supports):
                    reconstructed[y0+j] = bool(chosen.intersection(ss))
                for g, ss in members.items():
                    reconstructed[group_indices[g]] = bool(chosen.intersection(ss))
                indices = list(chosen)
                reconstructed[tau] = min([1.] + [float(utilities[k][indices].sum()/v) for k, v in ideals.items()]) if ideals else 0.
                actual = matrix @ reconstructed
                violation = max(0., float(np.max(np.asarray(lows)-actual, initial=0)), float(np.max(actual-np.asarray(highs), initial=0)))
                feasible = bool(violation <= 1e-10 and np.all(reconstructed >= 0) and np.all(reconstructed <= 1))
        numerical_audit = numerical.audit_reconstruction(result, objective, reconstructed, feasible, integer_objective=integer)
        optimal = numerical_audit['certified_optimal']
        upper, gap = getattr(result, 'mip_dual_bound', None), getattr(result, 'mip_gap', None)
        stages.append({'name': name, 'status': int(result.status), 'message': str(result.message),
            'feasible_incumbent': feasible, 'optimal': optimal, 'seconds': time.perf_counter()-start,
            'objective': float(objective @ reconstructed) if feasible else None,
            'maximization_upper_bound': -float(upper) if upper is not None and math.isfinite(float(upper)) else None,
            'mip_gap': float(gap) if gap is not None and math.isfinite(float(gap)) else None,
            'reconstructed_max_constraint_violation': violation, 'numerical_audit': numerical_audit})
        if feasible:
            current, returned_from = reconstructed, name
        if not optimal:
            stopped = True
            return False
        if lock:
            value = float(objective @ current)
            if integer:
                add({i: v for i, v in enumerate(objective) if v}, lo=round(value), hi=round(value))
            else:
                add({i: v for i, v in enumerate(objective) if v}, lo=value-TOL)
        return True

    objective = np.zeros(nv); objective[y0:y0+nr] = 1
    if coverage_first and solve('original_query_literal_coverage', objective, lock=True, integer=True):
        coverage_optimum = round(float(objective @ current))
    if not stopped:
        for channel, utility in utilities.items():
            objective = np.zeros(nv); objective[:n] = utility
            if not solve(channel + '_ideal', objective):
                break
            value = float(objective @ current)
            if value > TOL:
                ideals[channel] = value
    if not stopped and ideals:
        normalized = np.zeros(nv)
        for channel, ideal in ideals.items():
            ratio = utilities[channel] / ideal
            add({tau: 1, **{i: -v for i, v in enumerate(ratio) if v}}, hi=0)
            normalized[:n] += ratio
        objective = np.zeros(nv); objective[tau] = 1
        if balance:
            solve('max_min_normalized_channel_utility', objective, lock=True)
        if not stopped:
            solve('sum_normalized_channel_utility', normalized, lock=True)
    if not stopped:
        solve('minimum_render_characters', -costs, integer=True)
    if current is None:
        raise RuntimeError('no feasible incumbent; no selection returned')
    chosen = list(np.flatnonzero(current[:n] > .5))
    chosen_set = set(chosen)
    cost = base.selected_cost(fs, chosen, gc)
    if not min_units <= len(chosen) <= max_units or cost > char_budget:
        raise RuntimeError('hard count/cost violation')
    if any(i in chosen_set and j in chosen_set for i, j in data['overlap_pairs']) or any(len(chosen_set.intersection(g)) > 1 for g in data['duplicate_groups']):
        raise RuntimeError('overlap/duplicate violation')
    covered = [labels[j] for j, ss in enumerate(supports) if chosen_set.intersection(ss)]
    if coverage_optimum is not None and len(covered) != coverage_optimum:
        raise RuntimeError('literal coverage optimum was not preserved')
    ordered = sorted(chosen, key=lambda i: (-float(dense[i]), fs[i]['id']))
    return {'version': VERSION, 'selected_ids': [fs[i]['id'] for i in ordered], 'selected_count': len(chosen),
            'cost': cost, 'char_budget': char_budget, 'max_units': max_units, 'min_units': min_units,
            'display_order': 'original q0 dense descending, fact ID ascending ties; caller may only apply frozen session grouping',
            'literal_coverage': len(covered), 'literal_coverage_optimum': coverage_optimum,
            'covered_requirements': covered, 'missing_requirements': [t for t in labels if t not in covered],
            'unsupported_requirements': [labels[j] for j, ss in enumerate(supports) if not ss],
            'channel_ideal_values': ideals, 'channel_utility': {k: float(u[chosen].sum()) for k, u in utilities.items()},
            'channel_ratios': {k: float(utilities[k][chosen].sum()/v) for k, v in ideals.items()},
            'query_aliases': channels['query_aliases'], 'channel_aliases': channels['channel_aliases'],
            'unique_queries': channels['unique_queries'], 'inactive_channels': channels['inactive_channels'],
            'ideal_domain': 'coverage_optimal' if coverage_first else 'unrestricted_original_feasible_domain',
            'coverage_first': coverage_first, 'max_min_balance': balance,
            'objectives': ('coverage -> ' if coverage_first else '') + 'same-domain channel ideals -> ' + ('max-min -> ' if balance else '') + 'normalized sum -> minimum true additive cost',
            'certified_optimal': not stopped, 'termination': 'partial_stage_incumbent' if stopped else 'all_stages_optimal',
            'returned_incumbent_from_stage': returned_from, 'solver_stages': stages,
            'certificate_scope': 'finite surrogate, floating arithmetic and recorded lexicographic tolerance; not semantic answer completeness',
            'lexicographic_tolerance': TOL, 'time_limit_seconds_per_stage': float(time_limit),
            'overlap_conflict_pairs': len(data['overlap_pairs']), 'exact_duplicate_classes': len(data['duplicate_groups']),
            'numerical_solver_policy': numerical.VERSION, 'presolve': False,
            'semantic_completeness_claim': False, 'new_model_calls': 0, 'seconds': time.perf_counter()-began}

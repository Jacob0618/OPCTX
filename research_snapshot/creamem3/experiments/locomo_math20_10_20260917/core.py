"""Frozen old objective; only the user-requested shortlist limit changes."""
from creamem3.experiments.locomo_narrow_rerank_ablations_20260913 import core as parent

LIMITS = (20, 10)
MODES = {'math20_rerank5': (20, True), 'math10_rerank5': (10, True),
         'math20_direct': (20, False), 'math10_direct': (10, False)}


def choose(candidates, graph, question, channels, limit):
    if limit not in LIMITS:
        raise ValueError('Undeclared shortlist limit')
    result = parent.choose(candidates, graph, question, channels, 'math50', limit=limit)
    result.update(mode='math'+str(limit), limit=limit, channels=channels)
    if result['bypass']:
        result['optimization']['termination'] = 'pool_at_or_below_limit'
    verify(result, candidates, graph, question, channels, limit)
    return result


def verify(saved, candidates, graph, question, channels, limit):
    ids = [c['fact']['id'] for c in candidates]
    selected = saved['prefilter_fact_ids']
    if saved['mode'] != 'math'+str(limit) or saved['limit'] != limit:
        raise ValueError('Limit/mode drift')
    if saved['candidate_fact_ids'] != ids or saved['channels'] != channels:
        raise ValueError('Pool/channel drift')
    if len(selected) != min(limit, len(ids)) or len(set(selected)) != len(selected):
        raise ValueError('Selection count/duplicates')
    if not set(selected) <= set(ids) or selected != [i for i in ids if i in set(selected)]:
        raise ValueError('Selection membership/order drift')
    if saved['rerank_input_count'] != len(selected) or saved['character_truncation']:
        raise ValueError('Count/truncation drift')
    units, groups = parent.old.adapt_candidates(candidates, graph)
    bound = sum(u['cost'] for u in units)+sum(groups.values())
    opt = saved['optimization']
    if saved['nonbinding_character_bound'] != bound or opt['char_budget'] != bound:
        raise ValueError('Character bound drift')
    if set(opt['selected_ids']) != set(selected):
        raise ValueError('Optimization binding drift')
    if len(ids) <= limit:
        if not saved['bypass'] or selected != ids:
            raise ValueError('Bypass drift')
        return
    if saved['bypass'] or opt['max_units'] != limit or opt['min_units'] != limit:
        raise ValueError('Active cardinality drift')
    if not opt['coverage_first'] or not opt['max_min_balance'] or opt['cost'] > bound:
        raise ValueError('Objective/feasibility drift')
    checked = parent.old.solver.base.validate_inputs(
        units, channels[0]['dense_scores'], channels[0]['bm25_scores'],
        max_units=limit, min_units=limit, char_budget=bound, group_costs=groups)
    chosen = {i for i, u in enumerate(checked['units']) if u['id'] in set(selected)}
    if any(a in chosen and b in chosen for a, b in checked['overlap_pairs']):
        raise ValueError('Hard overlap violation')
    if any(len(chosen.intersection(g)) > 1 for g in checked['duplicate_groups']):
        raise ValueError('Hard duplicate violation')

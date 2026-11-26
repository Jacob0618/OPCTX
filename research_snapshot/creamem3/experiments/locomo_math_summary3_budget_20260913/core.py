"""Reuse the frozen selector and renderer; vary only the declared domain."""
from creamem3.experiments.locomo_math_narrow_20260911 import core as prior

solver, qp, query_texts = prior.solver, prior.qp, prior.query_texts
vector_check, adapt_candidates = prior.vector_check, prior.adapt_candidates
plain_render, build_channels, render = prior.plain_render, prior.build_channels, prior.render


def select(index, session_index, question, bank, *, budget_policy):
    if budget_policy == 'dense5':
        return prior.select(index, session_index, question, bank)
    if budget_policy != 'nonbinding':
        raise ValueError('Unknown registered budget policy')
    vector = bank['vectors'][question]
    candidates, trace = index._candidate_pool(question, vector)
    if not candidates:
        raise ValueError('Empty candidate pool')
    ds = session_index._scores(session_index.fact_matrix, vector)
    dense5 = sorted(session_index.facts,
        key=lambda f: (-float(ds[session_index.fact_order[f['id']]]), f['id']))[:5]
    dense_ids = [f['id'] for f in dense5]
    reference_text, _ = plain_render(session_index, dense_ids)
    units, groups = adapt_candidates(candidates, session_index.graph)
    # This finite bound admits every subset of the candidate pool. It does not
    # alter the K<=5 rule, overlap constraints, objective, or minimum-cost tie.
    bound = sum(u['cost'] for u in units) + sum(groups.values())
    channels = build_channels(index, candidates, question, bank)
    if channels[0]['dense_scores'] != [c['dense_score'] for c in candidates]:
        raise ValueError('Frozen q0 score mismatch')
    opt = solver.select_multichannel(units, question, channels, max_units=5,
        char_budget=bound, group_costs=groups, time_limit=5.0)
    ids = opt['selected_ids']
    if not 0 < len(ids) <= 5 or len(set(ids)) != len(ids) or not set(ids) <= {u['id'] for u in units}:
        raise ValueError('Invalid selected facts; no filling permitted')
    plain, assembly = plain_render(session_index, ids)
    if opt['cost'] - 1 != len(plain) or len(plain) >= bound or assembly['rendered_event_view_ids']:
        raise ValueError('Ordinary exact additive cost mismatch')
    return {'candidate_trace': trace, 'candidate_fact_ids': [u['id'] for u in units],
        'candidate_document_chars': sum(len(c['fact']['text']) for c in candidates),
        'channels': channels, 'dense5_ids': dense_ids, 'dense5_reference_chars': len(reference_text),
        'plain_budget': bound - 1, 'budget_policy': budget_policy, 'plain_context': plain,
        'plain_chars': len(plain), 'selected_fact_ids': ids, 'optimization': opt}

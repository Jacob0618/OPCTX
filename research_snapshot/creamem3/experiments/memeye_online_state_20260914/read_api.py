"""Causal guards for the optional derived-fact reading path.

Kept separate from the frozen, failed HTTP-400 generation registration. The
legacy benchmarks and their mathematical scoring policy remain unchanged.
"""
from . import state, reader_bridge


def validate_visible(memory, facts, raw_ids, claim_ids):
    for fid in raw_ids:
        if fid not in facts or facts[fid]['session_id'] > memory['frontier']:
            raise ValueError('Unknown or future raw read evidence')
    for cid in claim_ids:
        if cid not in memory['claims']:
            raise ValueError('Unknown derived read evidence')
        claim = memory['claims'][cid]
        if claim['created_at_frontier'] > memory['frontier']:
            raise ValueError('Future derived record')
        for fid in state.required_support(claim):
            if fid not in facts or facts[fid]['session_id'] > memory['frontier']:
                raise ValueError('Future derived support')


def render_package(memory, facts, raw_ids, claim_ids, *, max_items, char_budget):
    validate_visible(memory, facts, raw_ids, claim_ids)
    return state.render_package(memory, facts, raw_ids, claim_ids,
                                max_items=max_items, char_budget=char_budget)


def select_packages(memory, facts, ranked_options, *, max_items, char_budget):
    validate_visible(memory, facts, [o['id'] for o in ranked_options if o['kind'] == 'raw'],
                      [o['id'] for o in ranked_options if o['kind'] == 'derived'])
    text, receipt = state.select_packages(memory, facts, ranked_options,
                                          max_items=max_items, char_budget=char_budget)
    validate_visible(memory, facts, receipt['raw_fact_ids'], receipt['derived_fact_ids'])
    return text, receipt


def final_context(memory, facts, raw_ids, claim_ids, aggregate_context, *, max_items, char_budget):
    validate_visible(memory, facts, raw_ids, claim_ids)
    return reader_bridge.final_context(memory, facts, raw_ids, claim_ids, aggregate_context,
                                      max_items=max_items, char_budget=char_budget)


def export_candidates(memory, facts):
    """First-class derived candidates, not extra uncharged event prose.

    Semantic ranking is deliberately not invented here. A later registered
    retrieval experiment must score raw and derived text and account for source
    closure using the same budget; this API does not replace the old selector.
    """
    result = []
    for cid, c in memory['claims'].items():
        if c['status'] == 'retracted':
            continue
        support = state.required_support(c)
        validate_visible(memory, facts, support, [cid])
        text, receipt = select_packages(memory, facts, [{'kind': 'derived', 'id': cid}],
                                        max_items=len(facts) + len(memory['claims']), char_budget=10**9)
        if cid not in receipt['derived_fact_ids']:
            raise ValueError('Cannot export an invalid or unsupported derived package: ' + cid)
        result.append({'id': cid, 'kind': 'derived', 'text': c['text'], 'status': c['status'],
                       'support_fact_ids': support, 'minimum_package_context': text,
                       'minimum_package_raw_ids': receipt['raw_fact_ids'],
                       'minimum_package_derived_ids': receipt['derived_fact_ids'],
                       'minimum_package_items': receipt['items'], 'minimum_package_chars': receipt['chars']})
    return result

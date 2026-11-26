"""Source-ID proposals, mechanically grounded quotes and non-destructive refinement."""
from copy import deepcopy
from . import engine_v4 as prior

old = prior.prior.engine.old
VERSION = 'source_grounded_event_revision.detail_refine.v8'
OP_KEYS = (old.OP_KEYS - {'reason_quotes'}) | {'reason_fact_ids'}
CLAIM_KEYS = old.CLAIM_KEYS - {'quotes'}


def empty(task):
    m = prior.empty(task); m['schema'] = VERSION
    return m


def normalize(operation, memory, facts, allowed, exposed):
    if not isinstance(operation, dict) or set(operation) != OP_KEYS:
        raise ValueError('Required operation fields: ' + ', '.join(sorted(OP_KEYS)))
    op = deepcopy(operation)
    reason_ids = old.ids(op.pop('reason_fact_ids'))
    if not reason_ids or not set(reason_ids) <= set(allowed):
        raise ValueError('Reason IDs must be nonempty visible evidence')
    op['reason_quotes'] = [{'fact_id': f, 'quote': facts[f]['text']} for f in reason_ids]
    c = op.get('new_claim')
    if c is not None:
        if not isinstance(c, dict) or set(c) != CLAIM_KEYS:
            raise ValueError('Required new_claim fields: ' + ', '.join(sorted(CLAIM_KEYS)))
        support = old.ids(c['support_fact_ids'])
        if not support or not set(support) <= set(allowed):
            raise ValueError('Claim support IDs must be nonempty visible evidence')
        c['quotes'] = [{'fact_id': f, 'quote': facts[f]['text']} for f in support]
    return op


def refine(memory, op, facts, allowed, exposed, frontier):
    targets = old.ids(op['old_claim_ids'])
    if len(targets) != 1 or targets[0] not in exposed or targets[0] not in memory['claims']:
        raise ValueError('Refine requires one exposed active target')
    previous = memory['claims'][targets[0]]; new = op['new_claim']
    if previous['status'] != 'active' or not isinstance(new, dict):
        raise ValueError('Refine extends an active interpretation, not an unresolved conflict')
    if previous['kind'] != new['kind'] or old.slot(previous) != old.slot(new):
        raise ValueError('Refine preserves exact kind/subject/attribute/occurrence scope')
    if previous['kind'] == 'state' and previous['value'].casefold() != new['value'].casefold():
        raise ValueError('A different state value is supersede/dispute, not refine')
    if new['text'].strip() == previous['text'].strip():
        raise ValueError('Same text needs confirm, not refine')
    if not set(old.required_support(previous)) <= set(allowed):
        raise ValueError('Refine requires full old source support')
    inherited = [f for f in old.required_support(previous) if f not in new['support_fact_ids']]
    new['support_fact_ids'] += inherited
    new['quotes'] += [{'fact_id': f, 'quote': facts[f]['text']} for f in inherited]
    # Reuse strict target, current-evidence, dependency and schema guards. Then
    # retain the previous interpretation as history, NOT as an erroneous fact.
    effective = deepcopy(op); effective['action'] = 'correct'
    out, receipt = prior.prior.engine.apply_one(memory, effective, facts, allowed, exposed, frontier)
    out['claims'][targets[0]]['status'] = 'historical'
    receipt['action'] = 'refine'
    receipt['inherited_previous_support_fact_ids'] = inherited
    receipt['after_claims_sha256'] = old.digest(out['claims'])
    return out, receipt


def apply_frontier(memory, response, facts, allowed, exposed, frontier):
    if not isinstance(response, dict) or set(response) != {'operations'} or not isinstance(response['operations'], list):
        raise ValueError('Expected exactly {operations: [...]}')
    if len(response['operations']) > 12 or frontier <= memory['frontier']:
        raise ValueError('Operation cap/increasing frontier violated')
    if not set(allowed) <= facts.keys():
        raise ValueError('Unknown exposed evidence')
    out = deepcopy(memory); receipts = []
    for proposed in response['operations']:
        op = normalize(proposed, out, facts, allowed, exposed)
        if op['action'] == 'refine':
            out, receipt = refine(out, op, facts, allowed, exposed, frontier)
        else:
            op, expanded = prior.close_date_quote(op, facts, allowed)
            op, inherited = prior.prior.close_previous_support(out, op, facts, allowed, exposed)
            out, receipt = prior.prior.engine.apply_one(out, op, facts, allowed, exposed, frontier)
            receipt.update(inherited_previous_support_fact_ids=inherited, expanded_same_source_date_quotes=expanded)
        receipt['source_id_proposal_sha256'] = old.digest(proposed)
        receipt['quotes_materialized_from_visible_sources'] = True
        receipts.append(receipt)
    out['frontier'] = frontier
    return out, receipts

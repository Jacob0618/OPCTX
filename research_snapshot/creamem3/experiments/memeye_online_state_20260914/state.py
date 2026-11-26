"""Query-independent, source-grounded derived facts above immutable evidence.

No old files are changed. Quotation/schema checks establish provenance, NOT
natural-language entailment, identity correctness, or benchmark improvement.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json

VERSION = 'memeye.versioned_derived_facts.v1'
ACTIONS = {'add', 'confirm', 'supersede', 'correct', 'retract', 'dispute'}
CLAIM_KEYS = {'kind', 'subject', 'attribute', 'scope', 'value', 'text',
              'support_fact_ids', 'quotes', 'valid_from', 'current_dependencies'}
OP_KEYS = {'action', 'event_id', 'old_claim_ids', 'new_claim', 'reason', 'reason_quotes'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def empty(task):
    return {'schema': VERSION, 'task': task, 'frontier': 0, 'claims': {},
            'next_claim': 1, 'next_event': 1, 'ledger': []}


def string(value, empty_ok=False):
    if not isinstance(value, str) or (not empty_ok and not value.strip()):
        raise ValueError('Expected nonempty string')
    return value.strip()


def ids(value):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError('Expected ID list')
    if len(value) != len(set(value)):
        raise ValueError('Duplicate ID')
    return list(value)


def iso(value):
    value = string(value, True)
    if value:
        if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
            raise ValueError('Date must be YYYY-MM-DD or empty')
    return value


def quoted(value, facts, allowed, frontier):
    if not isinstance(value, list) or not value:
        raise ValueError('Exact source quotes are required')
    out = []
    for q in value:
        if not isinstance(q, dict) or set(q) != {'fact_id', 'quote'}:
            raise ValueError('Invalid quote shape')
        fid = string(q['fact_id']); text = string(q['quote'])
        if fid not in allowed or fid not in facts or facts[fid]['session_id'] > frontier:
            raise ValueError('Unknown, unexposed, or future evidence')
        if text not in facts[fid]['text']:
            raise ValueError('Quote is not a verbatim substring of its fact')
        out.append({'fact_id': fid, 'quote': text})
    return out


def slot(claim):
    return tuple(' '.join(claim[k].casefold().split()) for k in ('subject', 'attribute', 'scope'))


def required_support(claim):
    return list(dict.fromkeys(claim['support_fact_ids'] + claim.get('conflict_fact_ids', [])))


def validate_claim(value, state, facts, allowed, frontier):
    if not isinstance(value, dict) or set(value) != CLAIM_KEYS:
        raise ValueError('Invalid derived fact shape')
    c = deepcopy(value)
    for key in ('kind', 'subject', 'attribute', 'scope', 'text'):
        c[key] = string(c[key])
    if c['kind'] not in {'state', 'relation'}:
        raise ValueError('Unknown claim kind')
    c['value'] = string(c['value'], c['kind'] == 'relation')
    c['valid_from'] = iso(c['valid_from'])
    c['support_fact_ids'] = ids(c['support_fact_ids'])
    c['quotes'] = quoted(c['quotes'], facts, allowed, frontier)
    supports = set(c['support_fact_ids'])
    if not supports or supports != {q['fact_id'] for q in c['quotes']}:
        raise ValueError('Every supporting fact needs a quote and vice versa')
    if not any(facts[f]['session_id'] == frontier for f in supports):
        raise ValueError('New derived fact needs current-frontier evidence')
    if c['kind'] == 'relation' and not any(facts[f]['session_id'] < frontier for f in supports):
        raise ValueError('New cross-session relation needs an earlier endpoint')
    if c['kind'] == 'state' and not any(c['value'].casefold() in q['quote'].casefold() for q in c['quotes']):
        raise ValueError('State value must be explicitly quoted, not guessed')
    if c['valid_from'] and not any(c['valid_from'] in q['quote'] for q in c['quotes']):
        raise ValueError('Effective date must be quoted in exact ISO form')
    c['current_dependencies'] = ids(c['current_dependencies'])
    if any(d not in state['claims'] or state['claims'][d]['status'] != 'active'
           for d in c['current_dependencies']):
        raise ValueError('Missing active dependency')
    return c


def invalidate(claims, changed):
    pending = set(changed); result = set()
    while pending:
        cid = pending.pop()
        for did, c in claims.items():
            if c['status'] == 'active' and cid in c['current_dependencies']:
                c['status'] = 'unresolved'; result.add(did); pending.add(did)
    return sorted(result)


def apply_one(state, operation, facts, allowed_fact_ids, exposed_claim_ids, frontier):
    """Transactional single operation; invalid output cannot partly mutate state."""
    if not isinstance(operation, dict) or set(operation) != OP_KEYS:
        raise ValueError('Invalid operation shape')
    if frontier <= state['frontier']:
        raise ValueError('Cannot revise an already committed frontier')
    allowed = set(ids(allowed_fact_ids)); exposed = set(ids(exposed_claim_ids))
    if not allowed <= facts.keys() or any(facts[f]['session_id'] > frontier for f in allowed):
        raise ValueError('Exposed facts contain future/unknown data')
    action = operation['action']
    if action not in ACTIONS:
        raise ValueError('Unknown action')
    reason = string(operation['reason'])
    reason_quotes = quoted(operation['reason_quotes'], facts, allowed, frontier)
    if not any(facts[q['fact_id']]['session_id'] == frontier for q in reason_quotes):
        raise ValueError('Operation reason needs current evidence')
    old_ids = ids(operation['old_claim_ids'])
    if not set(old_ids) <= exposed or any(cid not in state['claims'] for cid in old_ids):
        raise ValueError('Revision target is not exposed')
    old = [state['claims'][cid] for cid in old_ids]
    if any(c['status'] not in {'active', 'unresolved'} for c in old):
        raise ValueError('Target is historical/retracted; preserve its history')
    if any(not set(required_support(c)) <= allowed for c in old):
        raise ValueError('Full old support must be hydrated before revision')
    event_id = string(operation['event_id'], True)
    if event_id and event_id not in {state['claims'][x]['event_id'] for x in exposed if x in state['claims']}:
        raise ValueError('Unexposed event')
    if old and (not event_id or any(c['event_id'] != event_id for c in old)):
        raise ValueError('Revision crosses event identity')
    if action == 'add' and old_ids:
        raise ValueError('Add cannot invalidate previous claims')
    if action != 'add' and not old_ids:
        raise ValueError('Revision needs an old claim')
    if action in {'confirm', 'supersede', 'correct'} and len(old) != 1:
        raise ValueError('One explicit target per version change')
    needs_new = action in {'add', 'supersede', 'correct'}
    if needs_new:
        new = validate_claim(operation['new_claim'], state, facts, allowed, frontier)
        if not set(new['current_dependencies']) <= exposed:
            raise ValueError('Current dependency was not exposed')
        if set(new['current_dependencies']) & set(old_ids):
            raise ValueError('New version cannot depend on invalidated version')
    elif operation['new_claim'] is not None:
        raise ValueError('This action must not introduce a new claim')
    else:
        new = None
    if action == 'supersede':
        previous = old[0]
        if previous['kind'] != 'state' or new['kind'] != 'state' or slot(previous) != slot(new):
            raise ValueError('Supersession requires the same subject/attribute/occurrence scope')
        if previous['value'].casefold() == new['value'].casefold():
            raise ValueError('Same value is confirmation, not a state change')
        if not new['valid_from'] or (previous['valid_from'] and new['valid_from'] < previous['valid_from']):
            raise ValueError('Unknown/backward effective time must remain unresolved')
        # Supersession is itself a two-endpoint memory, not a last-value overwrite.
        if not set(previous['support_fact_ids']) <= set(new['support_fact_ids']):
            raise ValueError('Supersession must preserve previous-state support')
    if action == 'confirm':
        if old[0]['status'] != 'active':
            raise ValueError('An unresolved state needs explicit resolution, not confirmation')
        if old[0]['kind'] == 'state' and not any(old[0]['value'].casefold() in q['quote'].casefold() for q in reason_quotes):
            raise ValueError('Confirmation needs the same quoted state value')
    if new and new['kind'] == 'state':
        for cid, c in state['claims'].items():
            if cid not in old_ids and c['status'] == 'active' and c['kind'] == 'state' and slot(c) == slot(new):
                raise ValueError('Existing same-scope state requires confirm/supersede/dispute')
    out = deepcopy(state)
    if not event_id:
        event_id = f"E{out['next_event']:04d}"; out['next_event'] += 1
    changed = []
    for cid in old_ids:
        c = out['claims'][cid]
        c['status'] = {'confirm': 'historical', 'supersede': 'historical',
                       'correct': 'retracted', 'retract': 'retracted', 'dispute': 'unresolved'}[action]
        if action == 'supersede':
            c['valid_to'] = new['valid_from']
        if action == 'dispute':
            c['conflict_fact_ids'] = list(dict.fromkeys(c.get('conflict_fact_ids', []) + [q['fact_id'] for q in reason_quotes]))
            c['conflict_quotes'] = deepcopy(reason_quotes)
        c['changed_at_frontier'] = frontier; changed.append(cid)
    if action == 'confirm':
        previous = old[0]
        new = {k: deepcopy(previous[k]) for k in CLAIM_KEYS}
        new['support_fact_ids'] = list(dict.fromkeys(new['support_fact_ids'] + [q['fact_id'] for q in reason_quotes]))
        new['quotes'] += [q for q in reason_quotes if q not in new['quotes']]
    dependent_changes = invalidate(out['claims'], changed)
    created = None
    if new:
        created = f"DF{out['next_claim']:05d}"; out['next_claim'] += 1
        out['claims'][created] = dict(new, id=created, event_id=event_id, status='active',
            valid_to='', previous_claim_ids=old_ids, version=(old[0]['version'] + 1 if old else 1),
            created_at_frontier=frontier, changed_at_frontier=frontier)
        # Confirmation carries the same proposition: transfer CURRENT references
        # without rewriting their original source support; unlike a changed value.
        if action == 'confirm':
            for cid in dependent_changes:
                c = out['claims'][cid]
                c['current_dependencies'] = [created if d in old_ids else d for d in c['current_dependencies']]
                if all(out['claims'][d]['status'] == 'active' for d in c['current_dependencies']):
                    c['status'] = 'active'
    receipt = {'frontier': frontier, 'action': action, 'event_id': event_id,
               'old_claim_ids': old_ids, 'created_claim_id': created,
               'invalidated_dependencies': dependent_changes, 'reason': reason,
               'reason_quotes': reason_quotes, 'before_sha256': digest(state),
               'after_claims_sha256': digest(out['claims'])}
    out['ledger'].append(receipt)
    return out, receipt


def apply_frontier(state, response, facts, allowed_fact_ids, exposed_claim_ids, frontier):
    if not isinstance(response, dict) or set(response) != {'operations'} or not isinstance(response['operations'], list):
        raise ValueError('Invalid model response envelope')
    if len(response['operations']) > 12:
        raise ValueError('Operation limit exceeded')
    if frontier <= state['frontier']:
        raise ValueError('Frontier is not increasing')
    out = deepcopy(state); receipts = []
    # Commit the entire frontier or none: a rejected operation is not silently
    # recorded as a successful memory update.
    for operation in response['operations']:
        out, receipt = apply_one(out, operation, facts, allowed_fact_ids, exposed_claim_ids, frontier)
        receipts.append(receipt)
    out['frontier'] = frontier
    return out, receipts


def render_package(state, facts, raw_ids, claim_ids, *, max_items, char_budget):
    """Derived facts and their source supports all consume budget, no free prose."""
    raw_ids = ids(raw_ids); claim_ids = ids(claim_ids)
    if len(raw_ids) + len(claim_ids) > max_items:
        raise ValueError('Derived records are not free fact slots')
    if any(fid not in facts for fid in raw_ids):
        raise ValueError('Unknown source fact')
    selected = set(raw_ids)
    sections = []
    for cid in claim_ids:
        if cid not in state['claims']:
            raise ValueError('Unknown derived fact')
        c = state['claims'][cid]
        if not set(required_support(c)) <= selected:
            raise ValueError('Incomplete relation support')
        if not set(c['current_dependencies']) <= set(claim_ids):
            raise ValueError('Current dependencies must be included')
        status = c['status']
        if status == 'retracted':
            raise ValueError('Retracted claims are audit history, not answer evidence')
        sections.append(f"[{cid}; {status}; scope={c['scope']}; valid_from={c['valid_from'] or 'unknown'}; "
                        f"valid_to={c['valid_to'] or 'open'}; supports={','.join(c['support_fact_ids'])}; "
                        f"conflict_evidence={','.join(c.get('conflict_fact_ids', []))}] {c['text']}")
    sections += [f"[{fid}] {facts[fid]['text']}" for fid in raw_ids]
    context = '\n'.join(sections)
    if len(context) > char_budget:
        raise ValueError('Complete rendered evidence exceeds character budget')
    return context, {'raw_fact_ids': raw_ids, 'derived_fact_ids': claim_ids,
                     'items': len(raw_ids) + len(claim_ids), 'chars': len(context),
                     'support_closed': True, 'context_sha256': digest(context)}


def select_packages(state, facts, ranked_options, *, max_items, char_budget):
    """Deterministic admission in externally supplied relevance order.

    A packing adapter, not a new learned/math scorer or an optimality claim.
    Each option is a raw fact or one derived fact with complete support/dependencies.
    An unaffordable package is skipped whole, never clipped to a misleading half.
    """
    raw, claims, skipped = [], [], []
    for option in ranked_options:
        if not isinstance(option, dict) or set(option) != {'kind', 'id'}:
            raise ValueError('Invalid ranked evidence option')
        nr, nc = list(raw), list(claims)
        if option['kind'] == 'raw':
            if option['id'] not in facts:
                raise ValueError('Unknown raw option')
            if option['id'] not in nr:
                nr.append(option['id'])
        elif option['kind'] == 'derived':
            pending = [option['id']]; visited = set()
            while pending:
                cid = pending.pop()
                if cid in visited:
                    continue
                visited.add(cid)
                if cid not in state['claims']:
                    raise ValueError('Unknown derived option')
                c = state['claims'][cid]
                if cid not in nc:
                    nc.append(cid)
                nr += [fid for fid in required_support(c) if fid not in nr]
                pending += c['current_dependencies']
        else:
            raise ValueError('Unknown option kind')
        try:
            render_package(state, facts, nr, nc, max_items=max_items, char_budget=char_budget)
        except ValueError as exc:
            skipped.append({'option': option, 'reason': str(exc)})
            continue
        raw, claims = nr, nc
    text, receipt = render_package(state, facts, raw, claims, max_items=max_items, char_budget=char_budget)
    return text, dict(receipt, skipped=skipped)

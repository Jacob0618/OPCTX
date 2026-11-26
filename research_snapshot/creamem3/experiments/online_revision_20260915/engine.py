"""Local revision above immutable facts; reuses tested v1 guards, adds detach."""
from copy import deepcopy
from creamem3.experiments.memeye_online_state_20260914 import state as old

VERSION = 'source_grounded_event_revision.v2'
ACTIONS = old.ACTIONS | {'detach'}


def empty(task):
    result = old.empty(task); result['schema'] = VERSION
    return result


def detach(memory, op, facts, allowed_ids, exposed_ids, frontier):
    if set(op) != old.OP_KEYS or op['new_claim'] is not None:
        raise ValueError('Detach moves existing claims; no new assertion')
    allowed, exposed = set(old.ids(allowed_ids)), set(old.ids(exposed_ids))
    old_ids = old.ids(op['old_claim_ids']); eid = old.string(op['event_id'])
    if not old_ids or not set(old_ids) <= exposed:
        raise ValueError('Detach targets must be exposed')
    if not allowed <= facts.keys() or any(facts[f]['session_id']>frontier for f in allowed):
        raise ValueError('Unknown/future exposed fact')
    quotes = old.quoted(op['reason_quotes'],facts,allowed,frontier)
    if not any(facts[q['fact_id']]['session_id']==frontier for q in quotes):
        raise ValueError('Detach needs current evidence')
    for cid in old_ids:
        if cid not in memory['claims']:
            raise ValueError('Unknown detach target')
        c = memory['claims'][cid]
        if c['event_id'] != eid or c['status'] not in {'active','unresolved'}:
            raise ValueError('Detach must target one visible current event')
        if not set(old.required_support(c)) <= allowed:
            raise ValueError('Full support required to detach')
    out = deepcopy(memory); target = f"E{out['next_event']:04d}"; out['next_event'] += 1
    for cid in old_ids:
        c = out['claims'][cid]
        c['event_id'] = target; c['changed_at_frontier'] = frontier
    receipt = dict(frontier=frontier,action='detach',event_id=eid,target_event_id=target,
        old_claim_ids=old_ids,created_claim_id=None,invalidated_dependencies=[],reason=old.string(op['reason']),
        reason_quotes=quotes,before_sha256=old.digest(memory),after_claims_sha256=old.digest(out['claims']))
    out['ledger'].append(receipt)
    return out,receipt


def apply_one(memory, op, facts, allowed, exposed, frontier):
    if not isinstance(op,dict) or op.get('action') not in ACTIONS:
        raise ValueError('Unknown/invalid revision operation')
    if op['action']=='detach':
        return detach(memory,op,facts,allowed,exposed,frontier)
    new = op.get('new_claim')
    if isinstance(new,dict) and new.get('kind')=='state' and set(new)==old.CLAIM_KEYS:
        # An unresolved value cannot be silently bypassed by adding a new winner.
        for cid,c in memory['claims'].items():
            if cid not in op.get('old_claim_ids',[]) and c['kind']=='state' and c['status']=='unresolved' and old.slot(c)==old.slot(new):
                raise ValueError('Unresolved same-scope state needs explicit correction, not a new winner')
        if op['action']=='supersede':
            current_quotes=[q for q in new.get('quotes',[]) if q.get('fact_id') in facts and facts[q['fact_id']]['session_id']==frontier]
            if not any(new['value'].casefold() in q.get('quote','').casefold() and new['valid_from'] in q.get('quote','') for q in current_quotes):
                raise ValueError('New state value and time need current-frontier quotation')
    out,receipt=old.apply_one(memory,op,facts,allowed,exposed,frontier)
    if op['action']=='confirm':
        # v1 restores direct dependencies but can strand a deeper chain depending
        # on ID iteration order. Restore only nodes invalidated by this confirm.
        candidates=set(receipt['invalidated_dependencies'])
        for _ in range(len(candidates)):
            changed=False
            for cid in sorted(candidates):
                c=out['claims'][cid]
                if c['status']=='unresolved' and all(out['claims'][d]['status']=='active' for d in c['current_dependencies']):
                    c['status']='active';changed=True
            if not changed:break
        receipt['after_claims_sha256']=old.digest(out['claims'])
    return out,receipt


def apply_frontier(memory, response, facts, allowed, exposed, frontier):
    if not isinstance(response,dict) or set(response)!={'operations'} or not isinstance(response['operations'],list):
        raise ValueError('Invalid response envelope')
    if len(response['operations'])>12 or frontier<=memory['frontier']:
        raise ValueError('Operation cap/increasing frontier violated')
    out=deepcopy(memory); receipts=[]
    for op in response['operations']:
        out,receipt=apply_one(out,op,facts,allowed,exposed,frontier);receipts.append(receipt)
    out['frontier']=frontier
    return out,receipts

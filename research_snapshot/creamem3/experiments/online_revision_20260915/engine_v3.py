"""Carry immutable prior support in code, never ask a model to recopy history."""
from copy import deepcopy
from . import engine

VERSION='source_grounded_event_revision.v3'


def empty(task):
    value=engine.empty(task);value['schema']=VERSION;return value


def close_previous_support(memory,operation,facts,allowed,exposed):
    op=deepcopy(operation)
    if not isinstance(op,dict) or op.get('action')!='supersede':return op,[]
    targets=op.get('old_claim_ids')
    if not isinstance(targets,list) or len(targets)!=1 or targets[0] not in exposed or targets[0] not in memory['claims']:
        return op,[]
    previous=memory['claims'][targets[0]];new=op.get('new_claim')
    if not isinstance(new,dict) or not isinstance(new.get('support_fact_ids'),list) or not isinstance(new.get('quotes'),list):return op,[]
    # No unexposed evidence is admitted under the guise of automatic closure.
    if not set(engine.old.required_support(previous))<=set(allowed):return op,[]
    for quote in previous['quotes']:
        fid=quote['fact_id']
        if fid not in facts or quote['quote'] not in facts[fid]['text']:
            raise ValueError('Historical quotation drift during source inheritance')
    inherited=[]
    for fid in previous['support_fact_ids']:
        if fid not in new['support_fact_ids']:
            new['support_fact_ids'].append(fid);inherited.append(fid)
        # Also close quotation coverage when a model listed the old ID but did
        # not recopy the already-validated prior quotation.
        if not any(isinstance(q,dict) and q.get('fact_id')==fid for q in new['quotes']):
            new['quotes'] += [deepcopy(q) for q in previous['quotes'] if q['fact_id']==fid]
    return op,inherited


def apply_frontier(memory,response,facts,allowed,exposed,frontier):
    if not isinstance(response,dict) or set(response)!={'operations'} or not isinstance(response['operations'],list):
        raise ValueError('Invalid response envelope')
    if len(response['operations'])>12 or frontier<=memory['frontier']:raise ValueError('Operation cap/increasing frontier violated')
    out=deepcopy(memory);receipts=[]
    for proposed in response['operations']:
        normalized,inherited=close_previous_support(out,proposed,facts,allowed,exposed)
        out,receipt=engine.apply_one(out,normalized,facts,allowed,exposed,frontier)
        receipt['proposed_operation_sha256']=engine.old.digest(proposed)
        receipt['effective_operation_sha256']=engine.old.digest(normalized)
        receipt['inherited_previous_support_fact_ids']=inherited
        receipts.append(receipt)
    out['frontier']=frontier
    return out,receipts

"""Source witnesses may be expanded within an already-cited fact, never invented."""
from copy import deepcopy
from . import engine_v3 as prior

VERSION='source_grounded_event_revision.v4'


def empty(task):
    m=prior.empty(task);m['schema']=VERSION;return m


def close_date_quote(operation,facts,allowed):
    op=deepcopy(operation);expanded=[]
    if not isinstance(op,dict):return op,expanded
    claim=op.get('new_claim')
    if not isinstance(claim,dict):return op,expanded
    date=claim.get('valid_from');quotes=claim.get('quotes');ids=claim.get('support_fact_ids')
    if not isinstance(date,str) or not date or not isinstance(quotes,list) or not isinstance(ids,list):return op,expanded
    if any(isinstance(q,dict) and date in q.get('quote','') for q in quotes):return op,expanded
    # A quote's short span may omit the explicit date at the beginning of the
    # same cited source fact. Expand that SAME source; no new IDs, no conversion,
    # no use of session metadata, no invention of an effective date.
    for fid in ids:
        if fid not in facts or fid not in allowed:continue
        text=facts[fid]['text']
        short=[q for q in quotes if isinstance(q,dict) and q.get('fact_id')==fid]
        if date in text and short and all(isinstance(q.get('quote'),str) and q['quote'] in text for q in short):
            quotes.append({'fact_id':fid,'quote':text});expanded.append(fid);break
    return op,expanded


def apply_frontier(memory,response,facts,allowed,exposed,frontier):
    if not isinstance(response,dict) or set(response)!={'operations'} or not isinstance(response['operations'],list):
        raise ValueError('Invalid response envelope')
    if len(response['operations'])>12 or frontier<=memory['frontier']:raise ValueError('Operation cap/increasing frontier violated')
    out=deepcopy(memory);receipts=[]
    for proposed in response['operations']:
        dated,expanded=close_date_quote(proposed,facts,allowed)
        normalized,inherited=prior.close_previous_support(out,dated,facts,allowed,exposed)
        out,receipt=prior.engine.apply_one(out,normalized,facts,allowed,exposed,frontier)
        receipt.update(proposed_operation_sha256=prior.engine.old.digest(proposed),
            effective_operation_sha256=prior.engine.old.digest(normalized),
            inherited_previous_support_fact_ids=inherited,expanded_same_source_date_quotes=expanded)
        receipts.append(receipt)
    out['frontier']=frontier
    return out,receipts

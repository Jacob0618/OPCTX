"""Controlled narrow prefilters; no-op below50, same universe above50."""
from creamem3.experiments.locomo_prefilter50_rerank5_20260913 import core as parent
old=parent.old
MODES=('rerank_repeat','dense50','math50','q0_only','no_coverage','utility_sum')


def choose(candidates,graph,question,channels,mode,limit=50):
    if mode not in MODES: raise ValueError('Undeclared mode')
    ids=[c['fact']['id'] for c in candidates]
    if not ids or len(ids)!=len(set(ids)): raise ValueError('Invalid universe')
    units,groups=old.adapt_candidates(candidates,graph)
    bound=sum(u['cost'] for u in units)+sum(groups.values())
    bypass=mode=='rerank_repeat' or len(ids)<=limit
    if bypass:
        opt=dict(selected_ids=ids,selected_count=len(ids),seconds=0.,certified_optimal=None,
            termination='full_pool_repeat' if mode=='rerank_repeat' else 'pool_at_or_below50',char_budget=bound)
    elif mode=='dense50':
        opt=old.dense_greedy(units,channels,max_units=limit,char_budget=bound,group_costs=groups)
    else:
        effective=channels[:1] if mode=='q0_only' else channels
        opt=old.solver.select_multichannel(units,question,effective,max_units=limit,min_units=limit,
            char_budget=bound,group_costs=groups,time_limit=5.,coverage_first=mode!='no_coverage',balance=mode!='utility_sum')
    selected=set(opt['selected_ids']); count=len(ids) if bypass else limit
    if len(selected)!=count or len(opt['selected_ids'])!=count or not selected<=set(ids): raise ValueError('Count/membership drift')
    return dict(mode=mode,candidate_fact_ids=ids,prefilter_fact_ids=[i for i in ids if i in selected],
        optimization=opt,bypass=bypass,nonbinding_character_bound=bound,character_truncation=False,
        rerank_input_count=count,rerank_input_fact_chars=sum(len(c['fact']['text']) for c in candidates if c['fact']['id'] in selected))


def select(index,graph,question,bank,mode):
    candidates,trace=index._candidate_pool(question,bank['vectors'][question])
    channels=old.build_channels(index,candidates,question,bank)
    return dict(**choose(candidates,graph,question,channels,mode),candidate_trace=trace,channels=channels)


def verify(saved,candidates,graph,question,channels):
    mode=saved['mode']; ids=[c['fact']['id'] for c in candidates]; selected=saved['prefilter_fact_ids']
    if mode not in MODES or saved['rerank_input_count']!=len(selected) or saved['character_truncation']:
        raise ValueError('Mode/count/truncation drift')
    if saved['candidate_fact_ids']!=ids or saved['channels']!=channels or len(selected)!=len(set(selected)) or not set(selected)<=set(ids):
        raise ValueError('Verification pool/channel/membership drift')
    if selected!=[i for i in ids if i in set(selected)]: raise ValueError('Rerank input order drift')
    if mode=='rerank_repeat' or len(ids)<=50:
        if selected!=ids or not saved['bypass']: raise ValueError('Bypass changed pool')
        expected=choose(candidates,graph,question,channels,mode)
        if any(saved[k]!=expected[k] for k in expected): raise ValueError('Bypass metadata drift')
        return
    if saved['bypass'] or len(selected)!=50: raise ValueError('Expected exact50')
    units,groups=old.adapt_candidates(candidates,graph); bound=sum(u['cost'] for u in units)+sum(groups.values())
    checked=old.solver.base.validate_inputs(units,channels[0]['dense_scores'],channels[0]['bm25_scores'],
        max_units=50,min_units=50,char_budget=bound,group_costs=groups)
    chosen={i for i,u in enumerate(checked['units']) if u['id'] in set(selected)}
    if any(a in chosen and b in chosen for a,b in checked['overlap_pairs']) or any(len(chosen.intersection(g))>1 for g in checked['duplicate_groups']):
        raise ValueError('Hard duplicate/overlap violation')
    opt=saved['optimization']
    if opt['char_budget']!=bound or saved['nonbinding_character_bound']!=bound or set(opt['selected_ids'])!=set(selected): raise ValueError('Budget/optimization drift')
    if opt['cost']>bound: raise ValueError('Cost exceeds nonbinding bound')
    if mode=='dense50':
        actual=choose(candidates,graph,question,channels,mode)
        if selected!=actual['prefilter_fact_ids']: raise ValueError('Dense replay drift')
    else:
        if opt['min_units']!=50 or opt['max_units']!=50 or opt['coverage_first']!=(mode!='no_coverage') or opt['max_min_balance']!=(mode!='utility_sum'):
            raise ValueError('Mathematical objective drift')
        if mode=='q0_only' and opt['unique_queries']!=['q0']: raise ValueError('Projection was not removed')

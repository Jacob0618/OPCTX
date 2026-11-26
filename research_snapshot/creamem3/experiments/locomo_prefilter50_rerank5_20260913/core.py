"""Same wide universe, exactly50 candidates; never pass answers to selection."""
from creamem3.experiments.aligned_selector_ablations_20260913 import core as old

MODES=('full_pool','dense50','math50')


def choose(candidates, session_graph, question, channels, mode, limit=50):
    if mode not in MODES: raise ValueError('Undeclared prefilter')
    ids=[c['fact']['id'] for c in candidates]
    if len(ids)!=len(set(ids)) or len(ids)<limit: raise ValueError('Invalid/short universe')
    units,groups=old.adapt_candidates(candidates,session_graph)
    bound=sum(u['cost'] for u in units)+sum(groups.values())
    dense=old.dense_greedy(units,channels,max_units=limit,char_budget=bound,group_costs=groups)
    if len(dense['selected_ids'])!=limit: raise ValueError('Fewer than50 feasible distinct facts')
    if mode=='full_pool':
        opt=dict(selected_ids=ids,selected_count=len(ids),seconds=0.,certified_optimal=None,
            termination='uncompressed_historical_reference',char_budget=bound)
    elif mode=='dense50': opt=dense
    else:
        opt=old.solver.select_multichannel(units,question,channels,max_units=limit,min_units=limit,
            char_budget=bound,group_costs=groups,time_limit=5.,coverage_first=True,balance=True)
    selected=set(opt['selected_ids'])
    ordered=[i for i in ids if i in selected]
    if len(selected)!=len(opt['selected_ids']) or not selected<=set(ids): raise ValueError('Illegal selection')
    if mode!='full_pool' and len(ordered)!=limit: raise ValueError('Prefilter count must match')
    return dict(mode=mode,candidate_fact_ids=ids,prefilter_fact_ids=ordered,optimization=opt,
        nonbinding_character_bound=bound,character_truncation=False,rerank_input_count=len(ordered),
        rerank_input_fact_chars=sum(len(c['fact']['text']) for c in candidates if c['fact']['id'] in selected))


def select(index,session_graph,question,bank,mode):
    candidates,trace=index._candidate_pool(question,bank['vectors'][question])
    channels=old.build_channels(index,candidates,question,bank)
    result=choose(candidates,session_graph,question,channels,mode)
    return dict(**result,candidate_trace=trace,channels=channels)

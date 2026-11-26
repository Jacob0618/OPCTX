"""Same candidate pool, same cost cap, four explicit selector replacements."""
import time
from creamem3.experiments.locomo_math_narrow_20260911 import core as prior

solver, qp, query_texts = prior.solver, prior.qp, prior.query_texts
vector_check, adapt_candidates = prior.vector_check, prior.adapt_candidates
plain_render, build_channels = prior.plain_render, prior.build_channels
MODES = ('dense', 'q0_only', 'no_coverage', 'utility_sum')


def render(index,graph,question,saved,arm,operation,materializer):
    context,trace,audit = prior.render(index,graph,question,saved,arm,operation,materializer)
    trace['rerank']['method'] = 'dense_feasible_greedy' if saved['mode']=='dense' else 'mathematical_'+saved['mode']
    return context,trace,audit


def dense_greedy(units, channels, *, max_units, char_budget, group_costs):
    """Descending q0 cosine; skip infeasible entries; preserve hard conflicts."""
    start = time.perf_counter()
    data = solver.base.validate_inputs(units, channels[0]['dense_scores'], channels[0]['bm25_scores'],
        max_units=max_units, char_budget=char_budget, group_costs=group_costs)
    facts, groups = data['units'], data['group_costs']
    order = sorted(range(len(facts)), key=lambda i:(-float(data['dense'][i]), facts[i]['id']))
    selected, skipped = [], []
    for i in order:
        if len(selected) >= max_units: break
        proposed = set(selected) | {i}
        conflict = any(a in proposed and b in proposed for a,b in data['overlap_pairs']) or any(len(proposed.intersection(g)) > 1 for g in data['duplicate_groups'])
        if conflict:
            skipped.append({'id':facts[i]['id'], 'reason':'same hard duplicate/overlap constraint'}); continue
        if solver.base.selected_cost(facts, proposed, groups) > char_budget:
            skipped.append({'id':facts[i]['id'], 'reason':'character cap'}); continue
        selected.append(i)
    if not selected: raise ValueError('No feasible Dense fact; no truncation/fill')
    return {'version':'aligned.dense.greedy.v1', 'selected_ids':[facts[i]['id'] for i in selected],
        'selected_count':len(selected), 'cost':solver.base.selected_cost(facts,selected,groups),
        'char_budget':char_budget, 'max_units':max_units, 'min_units':0,
        'certified_optimal':False, 'certificate_applicable':False, 'termination':'deterministic_dense_greedy',
        'solver_stages':[], 'new_model_calls':0, 'seconds':time.perf_counter()-start, 'skipped':skipped,
        'description':'q0 cosine ranking within unchanged union; feasible admission only, no coverage/max-min/sum optimization'}


def choose(units, question, channels, *, mode, max_units, char_budget, group_costs):
    if mode not in (*MODES, 'full'): raise ValueError('Undeclared component mode')
    if mode == 'dense':
        return dense_greedy(units,channels,max_units=max_units,char_budget=char_budget,group_costs=group_costs)
    effective = channels[:1] if mode == 'q0_only' else channels
    return solver.select_multichannel(units,question,effective,max_units=max_units,char_budget=char_budget,
        group_costs=group_costs,time_limit=5.,coverage_first=mode!='no_coverage',balance=mode!='utility_sum')


def select(index, session_index, question, bank, *, mode):
    candidates, trace = index._candidate_pool(question,bank['vectors'][question])
    if not candidates: raise ValueError('Empty frozen joint candidate pool')
    ds = session_index._scores(session_index.fact_matrix,bank['vectors'][question])
    dense5 = sorted(session_index.facts,key=lambda f:(-float(ds[session_index.fact_order[f['id']]]),f['id']))[:5]
    dense_ids = [f['id'] for f in dense5]
    reference,_ = plain_render(session_index,dense_ids)
    units,groups = adapt_candidates(candidates,session_index.graph)
    channels = build_channels(index,candidates,question,bank)
    if channels[0]['dense_scores'] != [c['dense_score'] for c in candidates]: raise ValueError('Q0 score drift')
    opt = choose(units,question,channels,mode=mode,max_units=5,char_budget=len(reference)+1,group_costs=groups)
    ids = opt['selected_ids']
    if not 0 < len(ids) <= 5 or len(set(ids)) != len(ids) or not set(ids) <= {u['id'] for u in units}:
        raise ValueError('Selection count/membership drift')
    text,assembly = plain_render(session_index,ids)
    if opt['cost']-1 != len(text) or len(text)>len(reference) or assembly['rendered_event_view_ids']:
        raise ValueError('Exact rendering/budget drift')
    return {'mode':mode,'candidate_trace':trace,'candidate_fact_ids':[u['id'] for u in units],
        'candidate_document_chars':sum(len(c['fact']['text']) for c in candidates),'channels':channels,
        'effective_query_ids':[q['query_id'] for q in (channels[:1] if mode in {'dense','q0_only'} else channels)],
        'dense5_ids':dense_ids,'plain_budget':len(reference),'plain_context':text,'plain_chars':len(text),
        'selected_fact_ids':ids,'optimization':opt}

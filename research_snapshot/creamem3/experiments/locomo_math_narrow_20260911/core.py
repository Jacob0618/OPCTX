"""Score-blind adapter: narrow candidates -> frozen MemEye set objective."""
import copy
import math
from creamem3.experiments.component_ablations_20260911 import math_selector as solver
from creamem3.experiments.iterative_memory_20260907 import query_projections as qp
from creamem3.experiments.iterative_memory_20260907.build_multichannel_round_v2 import query_texts


def vector_check(vector):
    if len(vector)!=768 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in vector) or sum(v*v for v in vector)<=0:
        raise ValueError('Native768 finite nonzero query vector required')


def adapt_candidates(candidates, session_graph):
    summaries={int(n['session_ids'][0]):n for n in session_graph['summary_nodes'] if n['node_type']=='session_packet'}
    units,groups=[],{}
    for c in candidates:
        f=c['fact']; sid=int(f['session_id']); s=summaries[sid]
        header=f"[Session summary | session {sid} | {s.get('timestamp','')}] {s['text']}"
        line=f"  - [{', '.join(f['source_dia_ids'])}] {f['text']}"
        units.append(dict(id=f['id'],text=f['text'],sources=list(f['source_dia_ids']),group=str(sid),cost=len(line)+1))
        groups[str(sid)]=len(header)+1
    return units,groups


def plain_render(session_index, ids):
    return session_index._assemble([{'fact':session_index.fact_by_id[i]} for i in ids],[])


def build_channels(index, candidates, question, bank):
    if bank['question']!=question: raise ValueError('Query bank mismatch')
    desired=query_texts(question,bank['projection'],'q0_plus_anchored_attributes')
    ids=[c['fact']['id'] for c in candidates]
    result=[]
    for q in desired:
        vector=bank['vectors'][q['query']]; vector_check(vector)
        ds=index._scores(index.fact_matrix,vector)
        bm=dict(index.bm25.search(q['query'],len(index.facts)))
        result.append({k:q[k] for k in ('query_id','query')} | {
            'dense_scores':[float(ds[index.fact_order[i]]) for i in ids],
            'bm25_scores':[float(bm.get(i,0)) for i in ids]})
    return result


def select(index, session_index, question, bank):
    """No answer/category/operation input enters set selection."""
    vector=bank['vectors'][question]
    candidates,trace=index._candidate_pool(question,vector)
    if not candidates: raise ValueError('Empty candidate pool')
    ds=session_index._scores(session_index.fact_matrix,vector)
    dense5=sorted(session_index.facts,key=lambda f:(-float(ds[session_index.fact_order[f['id']]]),f['id']))[:5]
    dense_ids=[f['id'] for f in dense5]
    dense_context,_=plain_render(session_index,dense_ids)
    budget=len(dense_context)
    units,groups=adapt_candidates(candidates,session_index.graph)
    channels=build_channels(index,candidates,question,bank)
    if channels[0]['dense_scores'] != [c['dense_score'] for c in candidates]: raise ValueError('Frozen q0 score mismatch')
    opt=solver.select_multichannel(units,question,channels,max_units=5,char_budget=budget+1,group_costs=groups,time_limit=5.0)
    ids=opt['selected_ids']
    if not 0<len(ids)<=5 or len(set(ids))!=len(ids) or not set(ids)<={c['fact']['id'] for c in candidates}:
        raise ValueError('Selected ID/count violation; no fallback filling')
    plain,assembly=plain_render(session_index,ids)
    if opt['cost']-1!=len(plain) or len(plain)>budget or assembly['rendered_event_view_ids']:
        raise ValueError('Exact source/session additive cost mismatch')
    return {'candidate_trace':trace,'candidate_fact_ids':[c['fact']['id'] for c in candidates],
            'candidate_document_chars':sum(len(c['fact']['text']) for c in candidates),
            'channels':channels,'dense5_ids':dense_ids,'plain_budget':budget,'plain_context':plain,
            'plain_chars':len(plain),'selected_fact_ids':ids,'optimization':opt}


def render(index, graph, question, saved, arm, operation, materializer):
    ids=saved['selected_fact_ids']; candidates=[{'fact':index.fact_by_id[i]} for i in ids]
    trace=copy.deepcopy(saved['candidate_trace'])
    context,assembly=index._assemble(candidates,trace['top_summaries'])
    # Legacy materializer reads this key. Its metadata explicitly denies a model reranker.
    trace['rerank']={'selected_fact_ids':ids,'attempts':0,'fallback':False,'error':None,
                     'method':'mathematical_coverage_maxmin','model_called':False}
    trace['assembly']=assembly
    audit=None
    if arm=='opctx_original':
        context,audit=materializer(operation,{'question':question,'retrieved':context,'retrieval_trace':trace},graph)
        if audit['selected_fact_ids']!=ids or audit['selected_fact_count']!=len(ids): raise ValueError('Display selection drift')
    if arm=='session_strong' and context!=saved['plain_context']: raise ValueError('Session renderer drift')
    return context,trace,audit

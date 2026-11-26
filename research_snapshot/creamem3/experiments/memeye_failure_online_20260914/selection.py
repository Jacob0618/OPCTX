"""Score-blind full371 Session/Online x Dense/math routing preparation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
from creamem3 import run_memeye_mec50_direct_gemini_v1 as direct
from creamem3.experiments.iterative_memory_20260907 import run_round as frozen
from creamem3.experiments.mec_transfer_20260906 import run_memeye as adapter
from creamem3.experiments.component_ablations_20260911 import math_selector as solver
from creamem3.experiments.aligned_selector_ablations_20260913.core import dense_greedy

ROOT=Path(r'D:\creamem3_cache\memeye_failure_online_20260914_r1')
BASE=Path(r'D:\creamem3_cache\memeye_v19\component_ablations_20260911_r1\full')
R34=Path(r'D:\creamem3_cache\memeye_v19\iterative_memory_20260907\selection_runs\r34\rows')
MODES=('session_dense','session_math','online_dense','online_math')
sha,file_sha=direct._sha256,direct._file_sha256


def put(path,value): return direct._write_once(path,frozen.sealed(value))


def source_hashes():
    import inspect
    from . import run
    from creamem3.experiments.iterative_memory_20260907 import candidate_context_policy as policy
    files=[Path(__file__),Path(run.__file__),Path(solver.__file__),Path(solver.base.__file__),
           Path(solver.numerical.__file__),Path(adapter.__file__),Path(adapter.raw.__file__),
           Path(inspect.getfile(dense_greedy)),Path(direct.__file__),Path(frozen.__file__),Path(policy.__file__),
           Path(direct.graph_memory.__file__),Path(direct.g35_runtime.__file__),Path(direct.retry5.__file__)]
    return {str(p.resolve()):file_sha(p) for p in files}


def select_one(job):
    p=ROOT/'selection'/job['mode']/(sha(job['query']['row_id'])+'.json')
    binding=sha(job)
    if p.exists():
        saved=frozen.checked(p)
        if saved['input_sha256']!=binding: raise RuntimeError('Selection input drift')
        return saved
    if job['mode'].endswith('dense'):
        opt=dense_greedy(job['units'],job['channels'],max_units=50,char_budget=job['budget']+1,group_costs=job['groups'])
    else:
        opt=solver.select_multichannel(job['units'],job['query']['question'],job['channels'],max_units=50,
            char_budget=job['budget']+1,group_costs=job['groups'],time_limit=5.,coverage_first=True,balance=True)
    ids=opt['selected_ids']
    if not 0<len(ids)<=50 or not set(ids)<=set(job['pool_ids']): raise RuntimeError('Membership/count drift')
    result={'mode':job['mode'],'row_id':job['query']['row_id'],'input_sha256':binding,
        'query':job['query'],'selected_fact_ids':ids,'pool_ids':job['pool_ids'],'routing_trace':job['trace'],
        'routing_graph':job['routing_graph'],'budget':job['budget'],'optimization':opt}
    put(p,result); return result


def prepare():
    canonical=direct._canonical_sources()
    source=frozen.checked(BASE/'prepared.json')
    template=direct._read_object(BASE/'manifest.json','frozen Reader config')
    query_rows=canonical['query_rows']
    if len(query_rows)!=371: raise RuntimeError('Full371 required')
    bindings={b['task']:b for b in canonical['build']['architecture_bindings'] if b['backbone']=='g35'}
    indexes,graphs,units,groups,receipts={},{},{},{},{}
    for task,b in bindings.items():
        for kind,key in [('session','session_only'),('online','online_base')]:
            path=Path(b[key+'_path'])
            if file_sha(path)!=b[key+'_sha256']: raise RuntimeError('Source graph seal drift')
            graph=direct._read_object(path,'frozen graph'); graphs[kind,task]=graph
            indexes[kind,task]=direct.graph_memory.GraphMemoryIndex(graph,summary_top_k=10,dense_fact_top_k=20,
                bm25_fact_top_k=20,candidate_cap=len(graph['fact_nodes']),rerank_top_k=50)
            receipts[kind+'::'+task]={'path':str(path),'sha256':file_sha(path)}
        if graphs['session',task]['fact_nodes']!=graphs['online',task]['fact_nodes'] or graphs['session',task]['memeye_source_map']!=graphs['online',task]['memeye_source_map']:
            raise RuntimeError('Writer facts/provenance must match between architecture arms')
        units[task],groups[task]=adapter.adapt_facts(graphs['session',task])
    channels_by_id={}; source_refs={}; jobs=[]
    for query in query_rows:
        qid,task=query['row_id'],query['task']; cp=R34/(sha(qid)+'.json'); channel=frozen.checked(cp)
        if channel['query']!=query: raise RuntimeError('Projection question drift')
        source_refs[str(cp)]=file_sha(cp)
        all_ids=[f['id'] for f in graphs['session',task]['fact_nodes']]; pos={x:i for i,x in enumerate(all_ids)}
        budget=channel['baseline_chars']; vector=canonical['vector_by_id'][qid]
        for kind in ('session','online'):
            candidates,trace=indexes[kind,task]._candidate_pool(query['question'],vector)
            ids=[c['fact']['id'] for c in candidates]; take=[pos[x] for x in ids]
            channels=[{**{k:q[k] for k in ('query_id','query')},
                'dense_scores':[q['dense_scores'][i] for i in take],
                'bm25_scores':[q['bm25_scores'][i] for i in take]} for q in channel['query_channels']]
            for method in ('dense','math'):
                jobs.append({'mode':kind+'_'+method,'query':query,'units':[units[task][i] for i in take],
                    'groups':groups[task],'channels':channels,'budget':budget,'pool_ids':ids,'trace':trace,
                    'routing_graph':receipts[kind+'::'+task]})
    reg={'schema':'memeye.online.routing.factorial.v1','n':371,'modes':list(MODES),'source_hashes':source_hashes(),
        'source_graphs':receipts,'channel_files':source_refs,'reader':template['reader'],'qa_context_policy':'candidate_last',
        'source_reader_contract_sha256':file_sha(BASE/'results/contract.json'),
        'retrieval':{'summary_top_k':10,'dense_top_k':20,'bm25_top_k':20,'candidate_cap':'all stored facts'},
        'max_facts':50,'character_budget':'frozen per-question raw-all-facts Dense50 context allowance',
        'new_writer_online_embedding_projection_calls':0,'rerank':False,'event_prose_rendered':False,
        'scope':'event-view routing, not event-prose presentation','new_protocol_not_original5283':True,
        'no_evaluator_inputs_at_selection':True,'all_ids':[q['row_id'] for q in query_rows],
        'reader_models':{'qa':'gemini-2.5-flash-lite','judge':'gemini-3.5-flash-lite'},
        'live_namespaces':'shared exact-payload caches across arms; no historical responses imported'}
    put(ROOT/'factorial_registration.json',reg)
    outputs={m:{} for m in MODES}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(select_one,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            result=f.result(); outputs[result['mode']][result['row_id']]=result
            if i%40==0: print('FACTORIAL SELECTION',i,len(jobs),flush=True)
    prepared={}; counts={}
    for mode in MODES:
        rows=[]; event_rows=0
        for query in query_rows:
            selected=outputs[mode][query['row_id']]; graph=graphs['online',query['task']]
            text,render=adapter.render_selection(graph,selected['selected_fact_ids'])
            if len(text)>selected['budget'] or len(text)!=selected['optimization']['cost']-1:
                raise RuntimeError('True rendered character budget violated')
            event_rows+=any(s['node_type']=='event_view' for s in selected['routing_trace']['top_summaries'])
            rows.append({'query':query,'selected_fact_ids':selected['selected_fact_ids'],'context':text,
                'context_sha256':sha(text),'selection_file_sha256':file_sha(ROOT/'selection'/mode/(sha(query['row_id'])+'.json')),
                'budget':selected['budget'],'render':render})
        if len(rows)!=371: raise RuntimeError('Population mismatch')
        put(ROOT/'prepared'/ (mode+'.json'),{'mode':mode,'registration_sha256':sha(reg),'rows':rows})
        prepared[mode]=rows; counts[mode]={'rows':371,'event_route_exposed_rows':event_rows}
    comparisons={}
    for method in ('dense','math'):
        a,b=prepared['session_'+method],prepared['online_'+method]
        comparisons[method]={'identical_contexts':sum(x['context']==y['context'] for x,y in zip(a,b)),
                            'different_contexts':sum(x['context']!=y['context'] for x,y in zip(a,b))}
    put(ROOT/'selection_complete.json',{'complete':True,'n':371,'total_rows':1484,'registration_sha256':sha(reg),
        'counts':counts,'architecture_context_comparison':comparisons,'new_provider_calls':0})
    print('FACTORIAL PREFLIGHT COMPLETE',counts,comparisons,flush=True)


if __name__=='__main__': prepare()

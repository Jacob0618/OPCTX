"""Main-table original Top20 source recall and fixed-seed routing negative control."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import importlib.util
import json
import re
import numpy as np
from creamem3.experiments.review_fast_20260926 import audit as a
from creamem3.experiments.mec_transfer_20260906 import run_locomo as frozen

b=a.b
OUT=Path(r'D:\opctx_review26\main_align\locomo_r2')
BASE=Path('creamem3/results_locomo10_v3').resolve()
MAIN=Path('creamem3/results_autoresearch_v8/online_operation_context_v3').resolve()
SEEDS=list(range(92620,92640))

def norm(values):
    return {d for v in values for d in re.findall(r'D\d+:\d+',str(v))}

def measure(ids,facts,evidence):
    sources=norm([d for fid in ids for d in facts[fid]['source_dia_ids']])
    return dict(recall=len(sources&evidence)/len(evidence) if evidence else None,
                all_sources=bool(evidence) and evidence<=sources,any_source=bool(evidence&sources),
                selected_count=len(ids),source_count=len(sources))

def summary(rows):
    ev=[x for x in rows if x['gold_sources']]
    out=dict(n=len(rows),evidence_n=len(ev),same_selected=sum(x['same_selected'] for x in rows))
    for arm in ('online','session'):
        out[arm]=dict(qa_percent=100*float(np.mean([x['data'][arm]['correct'] for x in rows])),
                     correct=sum(x['data'][arm]['correct'] for x in rows))
        for stage in ('writer','candidate','selected'):
            out[arm][stage]=dict(mean_source_recall_percent=100*float(np.mean([x['data'][arm][stage]['recall'] for x in ev])),
                all_sources=sum(x['data'][arm][stage]['all_sources'] for x in ev),
                mean_count=float(np.mean([x['data'][arm][stage]['selected_count'] for x in rows])))
    vals=[100*float(np.mean([x['shuffled'][str(s)]['recall'] for x in ev])) for s in SEEDS]
    out['shuffled']=dict(mean_percent=float(np.mean(vals)),range_percent=[min(vals),max(vals)],seed_means=vals)
    out['paired']={}
    for stage in ('candidate','selected','qa','shuffled'):
        rr=rows if stage=='qa' else ev
        def score(row,arm):
            if stage=='qa':return row['data'][arm]['correct']
            if stage=='shuffled':
                return row['data']['online']['candidate']['recall'] if arm=='online' else float(np.mean([v['recall'] for v in row['shuffled'].values()]))
            return row['data'][arm][stage]['recall']
        left={x['row_id']:dict(task=x['task'],mapped_score=score(x,'online')) for x in rr}
        right={x['row_id']:dict(task=x['task'],mapped_score=score(x,'session')) for x in rr}
        out['paired'][stage]=a.paired(left,right)
    return out

def main():
    code=Path('creamem3/frozen_sources/locomo81_graph_memory_v2.py').resolve()
    assert b.file_sha(code)==frozen.FROZEN_CODE_SHA
    assert b.file_sha(MAIN/'aggregate.json')==frozen.BEST_SHA
    spec=importlib.util.spec_from_file_location('_mainalign_frozen_graph',code)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ds=Path('data/locomo/locomo10.json').resolve();assert b.file_sha(ds)==frozen.DATA_SHA
    raw=a.read(ds);gold={f"{c['sample_id']}-q{i:03d}":q for c in raw for i,q in enumerate(c['qa'])}
    vp=Path(r'D:\creamem3_cache\locomo_g35lite_full1540_20260908_r2\question_vectors.json')
    v=a.read(vp);vectors=dict(zip(v['ids'],v['vectors'],strict=True))
    b.save(OUT/'registration.json',dict(code_sha256=b.file_sha(__file__),protocol_sha256=b.file_sha(Path(__file__).with_name('protocol.md')),
        model_calls=0,seeds=SEEDS,main_aggregate_sha256=frozen.BEST_SHA,query_vector_path=str(vp),query_vector_sha256=b.file_sha(vp),
        stages=['writer','candidate_cap120','selected20','cached_qa'],intervention='event membership permutation only',
        query_gold_not_used_for_intervention=True,candidate_replay_required=1540))
    rows=[];permutations={};replays=0
    for sid in sorted(frozen.GRAPH_SHA):
        gp=BASE/sid/'graph_online.json';assert b.file_sha(gp)==frozen.GRAPH_SHA[sid]
        graphs={'online':a.read(gp),'session':a.read(BASE/sid/'graph_session.json')}
        assert graphs['online']['fact_nodes']==graphs['session']['fact_nodes'],sid
        op=BASE/sid/f'online_{sid}_full_v3_r1.json';assert b.file_sha(op)==frozen.RESULT_SHA[sid]
        sources={arm:{r['qa_id']:r for r in a.read(p)['rows']} for arm,p in
                 [('online',op),('session',BASE/sid/f'session_{sid}_full_v3_r1.json')]}
        final={r['qa_id']:r for r in a.read(MAIN/sid/'online_operation_context_v3.json')['rows']}
        assert set(final)==set(sources['online'])==set(sources['session'])
        idx=m.GraphMemoryIndex(graphs['online']);ids=[f['id'] for f in idx.facts]
        facts=idx.fact_by_id;perms={}
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(b.sha(sid)[:8],16)]))
            perms[seed]=dict(zip(ids,[ids[i] for i in rng.permutation(len(ids))],strict=True))
        permutations[sid]=perms
        for qid,on in sorted(sources['online'].items()):
            off=sources['session'][qid];last=final[qid];gt=gold[qid]
            assert on['question']==off['question']==last['question']==gt['question']
            saved=on['retrieval_trace'];_,trace=idx._candidate_pool(on['question'],vectors[qid])
            for k in ('policy','top_summaries','dense_fact_ids','bm25_facts','candidates'):
                assert trace[k]==saved[k],(qid,k,'replay drift')
            replays+=1
            chosen=saved['rerank']['selected_fact_ids']
            final_audit=last['retrieval_trace']['online_operation_context_v8']['materializer_audit']
            assert chosen==final_audit['selected_fact_ids'] and len(chosen)==20
            dense=idx._scores(idx.fact_matrix,vectors[qid]);base=set(trace['dense_fact_ids'])|{x['fact_id'] for x in trace['bm25_facts']}
            views=[]
            for node in trace['top_summaries']:
                members=idx.adjacency[node['summary_id']]
                if node['node_type']=='event_view':views.append(members)
                else:base.update(members)
            shuffled={}
            for seed,perm in perms.items():
                union=base|{perm[f] for members in views for f in members}
                shuffled[seed]=sorted(union,key=lambda f:(-float(dense[idx.fact_order[f]]),idx.fact_order[f]))[:120]
            # Gold labels used only after all candidate sets are fixed.
            evidence=norm(gt.get('evidence',[]));data={}
            for arm,row in [('online',on),('session',off)]:
                t=row['retrieval_trace'];selids=t['rerank']['selected_fact_ids'];assert len(selids)==20
                data[arm]=dict(writer=measure(ids,facts,evidence),candidate=measure([x['fact_id'] for x in t['candidates']],facts,evidence),
                    selected=measure(selids,facts,evidence),selected_ids=selids,correct=(last if arm=='online' else row)['correct'],
                    candidate_ids=[x['fact_id'] for x in t['candidates']],rerank_fallback=t['rerank'].get('fallback'),
                    event_view_ids=[n['summary_id'] for n in t['top_summaries'] if n['node_type']=='event_view'])
            rows.append(dict(row_id=qid,task=sid,category=last['category_name'],gold_sources=sorted(evidence),data=data,
                same_selected=set(chosen)==set(data['session']['selected_ids']),
                shuffled={str(s):measure(ff,facts,evidence) for s,ff in shuffled.items()}))
        print('ALIGNED',sid,len(rows),flush=True)
    assert len(rows)==replays==1540
    assert sum(x['data']['online']['correct'] for x in rows)==1259
    assert sum(x['data']['session']['correct'] for x in rows)==1227
    a.SEED=92603
    result=dict(overall=summary(rows),by_category={c:summary([x for x in rows if x['category']==c]) for c in sorted({x['category'] for x in rows})})
    b.save(OUT/'results.json',dict(summary=result,rows=rows))
    b.save(OUT/'permutations.json',permutations)
    b.save(OUT/'acceptance.json',dict(status='PASS',rows=1540,exact_candidate_replays=replays,same_writer_conversations=10,
        source_hashes={p:b.file_sha(p) for p in a.INPUTS},code_sha256=b.file_sha(__file__),model_calls=0,
        boundary='Source provenance recall, not semantic sufficiency. Perturbation ends before rerank and QA.'))
    print(json.dumps({k:{z:v for z,v in vv.items() if z!='paired'} for k,vv in result['by_category'].items()},indent=2),flush=True)

if __name__=='__main__':main()

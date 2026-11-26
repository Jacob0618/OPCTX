"""Routing-only randomized event-link negative control,20 preregistered seeds."""
from pathlib import Path
import json
import numpy as np
from . import run as r
from . import locomo_evidence_audit as e
from creamem3.experiments.review_fast_20260926 import audit as a

SEEDS=list(range(92620,92640))

def main():
    out=r.ROOT/'locomo_view_perturbation'
    r.b.save(out/'registration.json',dict(code_sha256=r.b.file_sha(__file__),seeds=SEEDS,
        scope='All1540 historical narrow rerank5 queries; candidate routing only, no rerank/QA/new embeddings.',
        intervention='One global fact-ID permutation per conversation/seed, applied only to event-view adjacency. Preserve each view size and inter-view overlaps. Session adjacency, summary ranks, dense/BM25 branches, candidate cap120 and dense ordering fixed.',
        precondition='Original exact candidate order must replay for every query before comparison.',
        endpoint='Gold dialogue-source recall at candidate stage; not semantic precision and not downstream QA robustness.',
        selection_on_outcomes=False,model_calls=0))
    v=a.read(Path(r'D:\creamem3_cache\locomo_g35lite_full1540_20260908_r2\question_vectors.json'))
    vectors=dict(zip(v['ids'],v['vectors'],strict=True));records=a.read(r.ROOT/'locomo_evidence/results.json')['rows']
    grouped={}
    for row in records:grouped.setdefault(row['task'],[]).append(row)
    rows=[];maps={};replayed=0
    for sid,rr in sorted(grouped.items()):
        graph=a.read(e.GRAPHS/sid/'opctx_original.json')['graph']
        index=r.source.memory.GraphMemoryIndex(graph,summary_top_k=2,dense_fact_top_k=5,bm25_fact_top_k=5,candidate_cap=120,rerank_top_k=5)
        ids=[f['id'] for f in index.facts];perms={};maps[sid]={}
        for seed in SEEDS:
            # Same global permutation for every query in the conversation.
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(r.b.sha(sid)[:8],16)]))
            order=rng.permutation(len(ids));perms[seed]=dict(zip(ids,[ids[i] for i in order],strict=True));maps[sid][str(seed)]=perms[seed]
        for record in rr:
            qid=record['row_id'];saved=a.read(e.LOC/'conditions/narrow/retrieval/opctx_original'/f'{qid}.json')
            source=a.read(e.LOC/'conditions/narrow/rows/opctx_original'/f'{qid}.json')
            _,trace=index._candidate_pool(source['question'],vectors[qid])
            assert [c['fact_id'] for c in trace['candidates']]==[c['fact_id'] for c in saved['trace']['candidates']],qid
            assert trace['top_summaries']==saved['trace']['top_summaries'],qid
            replayed+=1;fact_scores=index._scores(index.fact_matrix,vectors[qid])
            base=set(trace['dense_fact_ids'])|{f['fact_id'] for f in trace['bm25_facts']}
            selected_views=[]
            for node in trace['top_summaries']:
                members=index.adjacency[node['summary_id']]
                if node['node_type']=='event_view':selected_views.append(members)
                else:base.update(members)
            candidates={}
            for seed,perm in perms.items():
                union=base|{perm[f] for members in selected_views for f in members}
                ordered=sorted(union,key=lambda f:(-float(fact_scores[index.fact_order[f]]),index.fact_order[f]))[:120]
                candidates[seed]=ordered
            # Endpoint labels are used only after the candidate sets are fixed.
            evidence=set(record['gold_sources']);measures={}
            for seed,selected in candidates.items():
                sources=e.normalized([d for fid in selected for d in index.fact_by_id[fid]['source_dia_ids']])
                measures[str(seed)]=dict(source_recall=len(evidence&sources)/len(evidence) if evidence else None,
                    all_sources=bool(evidence) and evidence<=sources,candidate_count=len(selected))
            rows.append(dict(row_id=qid,task=sid,category=record['category'],has_evidence=bool(evidence),
                event_routed=bool(selected_views),original=record['data']['opctx_original']['candidate'],perturbed=measures))
        print('PERTURB',sid,len(rows),flush=True)
    def summary(rr):
        ev=[x for x in rr if x['has_evidence']]
        vals=[100*np.mean([x['perturbed'][str(seed)]['source_recall'] for x in ev]) for seed in SEEDS]
        return dict(n=len(rr),evidence_n=len(ev),original_recall_percent=100*np.mean([x['original']['recall'] for x in ev]),
            shuffled_mean_percent=float(np.mean(vals)),seed_range_percent=[float(min(vals)),float(max(vals))],seed_means_percent=vals)
    answer=dict(all=summary(rows),event_routed=summary([x for x in rows if x['event_routed']]),
        by_category={c:summary([x for x in rows if x['category']==c]) for c in sorted({x['category'] for x in rows})},
        exact_original_replays=replayed,model_calls=0)
    a.SEED=92602
    valid=[x for x in rows if x['has_evidence']]
    left={x['row_id']:dict(task=x['task'],mapped_score=x['original']['recall']) for x in valid}
    right={x['row_id']:dict(task=x['task'],mapped_score=float(np.mean([v['source_recall'] for v in x['perturbed'].values()]))) for x in valid}
    answer['paired_source_recall']=a.paired(left,right)
    r.b.save(out/'results.json',dict(summary=answer,rows=rows))
    r.b.save(out/'permutations.json',maps)
    r.b.save(out/'provenance.json',dict(inputs=a.INPUTS,code_sha256=r.b.file_sha(__file__)))
    print(json.dumps(answer,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()

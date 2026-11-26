"""Offline source-recall diagnostics for four already-fixed routing variants.

Does not run a mathematical selector, reranker, Reader, QA, Judge, or network.
The alternatives do not replace the registered global-permutation experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from analysis.opctx_random_links_20261113 import run as prior
from analysis.opctx_random_links_20261114 import randomizer_audit as structure

import numpy as np

SHARED = Path('D:/opctx_random_links_20261113/r1')
STRUCTURAL = Path('D:/opctx_random_links_20261114/randomizer_audit_r1.json')
METHODS = ('original', 'global_permutation', 'independent_per_view', 'double_edge_swaps')
STAGES = ('event_only', 'hybrid_union', 'cap120')
DEFAULT_OUTPUT = Path('D:/opctx_random_links_20261114/randomizer_recall_r1.json')


def index_variants(graph, sid, permutation):
    normal = prior.gm.GraphMemoryIndex(graph)
    facts = list(normal.fact_by_id)
    views = [v['id'] for v in normal.summaries if v.get('node_type') == 'event_view']
    original = {v:set(normal.adjacency[v]) for v in views}
    global_perm = {v:{permutation[f] for f in fs} for v,fs in original.items()}
    rng = structure.rng_for(sid)
    independent = {v:set(rng.choice(facts,size=len(original[v]),replace=False).tolist()) for v in views}
    switched, switch_stats = structure.swaps(original,sid)
    maps = dict(original=original,global_permutation=global_perm,
                independent_per_view=independent,double_edge_swaps=switched)
    indices, summaries = {}, {}
    for method, adjacency in maps.items():
        index = prior.gm.GraphMemoryIndex(graph)
        for vid in views:
            index.adjacency[vid] = sorted(adjacency[vid])
        indices[method] = index
        summaries[method] = structure.summarize(original,adjacency,facts)
    summaries['double_edge_swaps'].update(switch_stats)
    return indices,summaries


def retrieve(index, question, vector):
    candidates, trace = index._candidate_pool(question,vector)
    event_ids = [x['summary_id'] for x in trace['top_summaries'] if x['node_type']=='event_view']
    event_only = set().union(*(set(index.adjacency[v]) for v in event_ids)) if event_ids else set()
    expanded = set().union(*(set(index.adjacency[x['summary_id']]) for x in trace['top_summaries']))
    hybrid = expanded | set(trace['dense_fact_ids']) | {x['fact_id'] for x in trace['bm25_facts']}
    capped = [c['fact']['id'] for c in candidates]
    if len(hybrid)!=trace['union_before_cap'] or len(capped)!=min(120,len(hybrid)):
        raise RuntimeError('Hybrid union or cap reconstruction mismatch')
    return dict(event_only=sorted(event_only),hybrid_union=sorted(hybrid),cap120=capped,
                selected_event_view_ids=event_ids,shared_branches={k:trace[k] for k in
                ('policy','top_summaries','dense_fact_ids','bm25_facts')})


def measure(ids, facts, targets):
    recovered=set().union(*(facts[f] for f in ids)) if ids else set()
    return dict(fact_count=len(ids),source_count=len(recovered),gold_source_count=len(targets),
                gold_sources_recovered=len(recovered&targets),
                recall=len(recovered&targets)/len(targets) if targets else None,
                all_sources=bool(targets) and targets<=recovered,
                any_source=bool(targets&recovered))


def summarize(rows):
    result={}
    for method in METHODS:
        stages={}
        for stage in STAGES:
            values=[r['methods'][method][stage] for r in rows]
            usable=[v for v in values if v['recall'] is not None]
            stages[stage]=dict(n=len(values),evidence_n=len(usable),
                mean_source_recall_percent=100*float(np.mean([v['recall'] for v in usable])) if usable else None,
                micro_source_recall_percent=100*sum(v['gold_sources_recovered'] for v in usable)/sum(v['gold_source_count'] for v in usable) if usable else None,
                all_sources=sum(v['all_sources'] for v in usable),
                any_source=sum(v['any_source'] for v in usable),
                mean_fact_count=float(np.mean([v['fact_count'] for v in values])),
                min_fact_count=min(v['fact_count'] for v in values),
                max_fact_count=max(v['fact_count'] for v in values))
        result[method]=stages
    return result


def audit():
    source=prior.load_inputs()
    questions=prior.read(SHARED/'questions.json')
    vectors=prior.read(SHARED/'query_vectors.json')
    if questions!=source['rows'] or vectors['ids']!=[q['qa_id'] for q in questions] or len(questions)!=282:
        raise RuntimeError('Shared282 question/q0 identity mismatch')
    if vectors['model']!='gemini-embedding-001' or any(len(v)!=768 for v in vectors['vectors']):
        raise RuntimeError('Query vector model/dimension mismatch')
    structural=prior.read(STRUCTURAL)
    if structural['code_sha256']!=prior.file_sha(structure.__file__):
        raise RuntimeError('Previously fixed randomizer code changed')
    expected={r['conversation']:r['methods'] for r in structural['per_conversation']}
    perms=prior.read(prior.PERMS)
    indices={}
    for sid,graph in source['graphs'].items():
        variants,summary=index_variants(graph,sid,perms[sid][str(structure.SEED)])
        for method in METHODS:
            for key,value in summary[method].items():
                if expected[sid][method][key]!=value:
                    raise RuntimeError('Alternative adjacency not equal to fixed structural audit')
        indices[sid]=variants

    # Finish all routing decisions before consuming gold for diagnostic scores.
    routing=[]
    for question,vector in zip(questions,vectors['vectors'],strict=True):
        sid=question['sample_id']
        methods={m:retrieve(indices[sid][m],question['question'],vector) for m in METHODS}
        if any(methods[m]['shared_branches']!=methods['original']['shared_branches'] for m in METHODS):
            raise RuntimeError('A non-event retrieval branch changed')
        routing.append(dict(**question,methods=methods))

    dataset=WORKSPACE/'data/locomo/locomo10.json'
    gold={}
    for conversation in prior.read(dataset):
        for number,q in enumerate(conversation['qa']):
            gold[f"{conversation['sample_id']}-q{number:03d}"]=set(re.findall(r'D\d+:\d+',str(q.get('evidence',[]))))
    source_sets={sid:{f['id']:set(re.findall(r'D\d+:\d+',str(f.get('source_dia_ids',[]))))
                      for f in graph['fact_nodes']} for sid,graph in source['graphs'].items()}
    rows=[]
    for item in routing:
        qid,sid=item['qa_id'],item['sample_id']
        targets=gold[qid]
        methods={}
        for method,r in item['methods'].items():
            methods[method]={s:measure(r[s],source_sets[sid],targets) for s in STAGES}
            methods[method]['selected_event_view_ids']=r['selected_event_view_ids']
            methods[method]['fact_id_sets']={s:r[s] for s in STAGES}
        rows.append(dict(qa_id=qid,sample_id=sid,gold_sources=sorted(targets),methods=methods))
    result=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        seed=structure.SEED,n=282,model_calls=0,new_math_selections=0,qa_measured=False,
        approved_qa_intervention='historical_global_permutation_only',
        event_only_definition='union of event-view members among the unchanged joint Top10 summary/event nodes; not Top10 event views',
        hybrid_union_definition='Top10 summary/event expansion plus global Dense20 plus BM25-20, before cap',
        cap120_definition='the original dense-ordered hybrid candidate cap120',
        metric='mean per-question annotated dialogue-source recall; not semantic completeness or QA accuracy',
        diagnostic_variants_fixed_before_gold=True,source_hashes={**source['metadata']['source_hashes'],
            **{str(p):prior.file_sha(p) for p in (Path(__file__),Path(structure.__file__),
              STRUCTURAL,SHARED/'questions.json',SHARED/'query_vectors.json',dataset)}},
        overall=summarize(rows),by_conversation={sid:summarize([r for r in rows if r['sample_id']==sid]) for sid in indices},
        rows=rows)
    for method in METHODS[1:]:
        result['overall'][method]['delta_pp_vs_original']={s:
            result['overall'][method][s]['mean_source_recall_percent']-result['overall']['original'][s]['mean_source_recall_percent']
            for s in STAGES}
    return result


def no_network(*args,**kwargs):
    raise RuntimeError('This offline diagnostic forbids network access')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite diagnostic output')
    with patch('socket.socket.connect',no_network),patch('socket.create_connection',no_network):
        result=audit()
    prior.save(args.output,result)
    print(json.dumps(dict(status=result['status'],n=result['n'],model_calls=0,
                         overall=result['overall'],output=str(args.output)),indent=2))


if __name__=='__main__':
    main()

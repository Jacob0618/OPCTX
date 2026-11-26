"""Whole1540 source-evidence routing diagnosis; no generation or score selection."""
from collections import defaultdict,Counter
from pathlib import Path
import json
import re
import hashlib
import numpy as np
from . import run as r
from creamem3.experiments.review_fast_20260926 import audit as a

LOC=Path(r'D:\creamem3_cache\locomo_mgtm480_top5_plus_all_full1540_20260910_r2')
COMP=Path(r'D:\creamem3_cache\component_ablations_20260911_r3\locomo')
GRAPHS=Path(r'D:\creamem3_cache\locomo_mgtm480_writer_full1540_enum_20260910_r2\graphs')

def normalized(values):
    return {x for v in values for x in re.findall(r'D\d+:\d+',str(v))}

def main():
    out=r.ROOT/'locomo_evidence';rows=[];graphs={}
    ds=Path('data/locomo/locomo10.json').resolve();raw=json.loads(ds.read_text('utf-8'));a.INPUTS[str(ds)]=r.b.file_sha(ds)
    gold={}
    for conv in raw:
        for i,q in enumerate(conv['qa']):gold[f"{conv['sample_id']}-q{i:03d}"]=q
    for pp in sorted((LOC/'conditions/narrow/rows/opctx_original').glob('*.json')):
        on=a.read(pp);qid=on['qa_id'];sid=on['sample_id'];gt=gold[qid]
        assert on['question']==gt['question']
        off=a.read(LOC/'conditions/narrow/rows/session_strong'/pp.name)
        plain=a.read(COMP/'rows/session_render'/pp.name)
        assert off['question']==plain['question']==on['question']
        if sid not in graphs:
            graphs[sid]={arm:a.read(GRAPHS/sid/f'{arm}.json')['graph'] for arm in ('opctx_original','session_strong')}
            assert graphs[sid]['opctx_original']['fact_nodes']==graphs[sid]['session_strong']['fact_nodes']
        evidence=normalized(gt.get('evidence',[]));data={}
        for arm,row in [('opctx_original',on),('session_strong',off)]:
            ret=a.read(LOC/'conditions/narrow/retrieval'/arm/pp.name)
            context_digest=hashlib.sha256(json.dumps(row['context'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert ret['context']==row['context'] and context_digest==row['context_sha256']
            graph=graphs[sid][arm];facts={f['id']:f for f in graph['fact_nodes']};trace=ret['trace']
            candidate_ids=[f['fact_id'] for f in trace['candidates']]
            selected=row['selected_fact_ids']
            def measure(ids):
                sources=normalized([s for f in ids for s in facts[f]['source_dia_ids']])
                hits=sources&evidence
                return dict(hits=sorted(hits),recall=len(hits)/len(evidence) if evidence else None,
                    all_gold_sources=bool(evidence) and evidence<=sources,any_gold_source=bool(hits),source_count=len(sources))
            materializer=ret.get('materializer') or {}
            data[arm]=dict(writer=measure(list(facts)),candidate=measure(candidate_ids),selected=measure(selected),
                candidate_count=len(candidate_ids),selected_count=len(selected),correct=row['correct'],
                event_routed=any(s['node_type']=='event_view' for s in trace['top_summaries']),
                event_view_ids=[s['summary_id'] for s in trace['top_summaries'] if s['node_type']=='event_view'],
                event_prose_count=materializer.get('rendered_event_view_count',0),
                operation=ret.get('operation'),selected_ids=selected)
        assert plain['selected_fact_ids']==on['selected_fact_ids']
        rows.append(dict(row_id=qid,task=sid,category=on['category_name'],gold_sources=sorted(evidence),
            data=data,plain_correct=plain['correct'],same_selected=set(on['selected_fact_ids'])==set(off['selected_fact_ids']),
            same_context=on['context']==off['context']))
    assert len(rows)==1540
    def summarize(rr):
        if not rr:return dict(n=0)
        v=dict(n=len(rr),with_gold_evidence=sum(bool(x['gold_sources']) for x in rr),
            online_percent=100*np.mean([x['data']['opctx_original']['correct'] for x in rr]),
            plain_online_percent=100*np.mean([x['plain_correct'] for x in rr]),
            session_percent=100*np.mean([x['data']['session_strong']['correct'] for x in rr]),
            same_selected=sum(x['same_selected'] for x in rr),same_context=sum(x['same_context'] for x in rr))
        ev=[x for x in rr if x['gold_sources']]
        for arm in ('opctx_original','session_strong'):
            v[arm]={}
            for stage in ('writer','candidate','selected'):
                v[arm][stage]=dict(mean_source_recall=float(np.mean([x['data'][arm][stage]['recall'] for x in ev])),
                    all_sources=sum(x['data'][arm][stage]['all_gold_sources'] for x in ev),
                    any_source=sum(x['data'][arm][stage]['any_gold_source'] for x in ev)) if ev else {}
        return v
    summary=dict(overall=summarize(rows),by_category={c:summarize([x for x in rows if x['category']==c]) for c in sorted({x['category'] for x in rows})},
        by_event_routing={str(v):summarize([x for x in rows if x['data']['opctx_original']['event_routed']==v]) for v in (True,False)},
        by_event_prose={str(v):summarize([x for x in rows if bool(x['data']['opctx_original']['event_prose_count'])==v]) for v in (True,False)},
        by_same_selected={str(v):summarize([x for x in rows if x['same_selected']==v]) for v in (True,False)},
        scope='Historical narrow rerank5; not81.75 main configuration. Gold dialogue-source IDs assess provenance recall, not fact semantic sufficiency. Summary/view prose may contain evidence beyond selected-fact source IDs. Subgroups are descriptive, not randomized effects.',
        new_model_calls=0)
    r.b.save(out/'results.json',dict(summary=summary,rows=rows))
    r.b.save(out/'provenance.json',dict(inputs=a.INPUTS,code_sha256=r.b.file_sha(__file__)))
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()

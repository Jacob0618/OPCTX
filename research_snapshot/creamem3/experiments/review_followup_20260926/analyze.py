"""Complete-population paired statistics; current progress contains no scores."""
from collections import defaultdict
from pathlib import Path
import json
import sys
import time
import numpy as np
from . import run as r
from creamem3.experiments.review_fast_20260926 import audit as a

def load(arm,refs,parent=False):
    scores={};predictions={};stages=defaultdict(list);missing=[];failures=[]
    for ref in refs:
        key=r.b.sha(ref['row_id']);item=r.b.read(ref['path'])
        root=(r.old.OUT if parent else r.ROOT)/'runs'/arm
        if parent and item['route']=='structured_fused':root=r.old.OUT/'fused_prompt_repair/runs'/arm
        pp=root/'predictions'/f'{key}.json';sp=root/'scores'/f'{key}.json';fp=root/'failures'/f'{key}.json'
        if not sp.exists():
            missing.append(ref['row_id'])
            if fp.exists():failures.append(r.b.read(fp))
            continue
        p=r.b.read(pp);s=r.b.read(sp);assert s['prediction_sha256']==p['seal_sha256']
        scores[ref['row_id']]=s;predictions[ref['row_id']]=p
        for st in p['stages']:stages[st['stage']].append(st['generation'])
    return scores,predictions,{k:a.token_summary(v) for k,v in stages.items()},missing,failures

def main(final=False):
    reg=r.b.read(r.ROOT/'registration.json');scores={};preds={};usage={};missing={};failures={}
    for arm in r.ARMS:scores[arm],preds[arm],usage[arm],missing[arm],failures[arm]=load(arm,reg['rows'])
    if not final:
        print(json.dumps(dict(n={k:len(v) for k,v in scores.items()},failures=failures),ensure_ascii=True));return
    assert all(not x for x in missing.values()),missing;r.validate()
    base,basep,baseu,miss,_=load('full_refresh',reg['rows'],True);assert not miss
    a.SEED=92602;comparisons={arm:a.paired(base,scores[arm]) for arm in r.ARMS}
    names=sorted(comparisons,key=lambda arm:comparisons[arm]['exploratory_signflip_p']);running=0.
    for i,arm in enumerate(names):
        running=max(running,min(1.,(len(names)-i)*comparisons[arm]['exploratory_signflip_p']))
        comparisons[arm]['exploratory_holm4_p']=running
    generic=preds['generic_reasoning'];assert all(generic[k]['context_sha256']==basep[k]['context_sha256'] for k in generic)
    assert all(len(generic[k]['stages'])==len(basep[k]['stages']) for k in generic)
    by_route={}
    for route in ('two_stage','structured_fused'):
        ids=[k for k,p in basep.items() if p['route']==route]
        by_route[route]=a.paired({k:base[k] for k in ids},{k:scores['generic_reasoning'][k] for k in ids})
    selections={arm:[] for arm in r.ARMS}
    for ref in reg['rows']:
        item=r.b.read(ref['path']);prior=r.b.read(item['source_prepared'])
        for arm in r.ARMS:
            s=item['selections'][arm];p=preds[arm][ref['row_id']]
            assert p['selected_fact_ids']==s['ids'] and p['context_sha256']==s['context_sha256']
            assert len(s['ids'])<=30 and len(s['context'])<=item['budget']
            selections[arm].append(dict(n=len(s['ids']),chars=len(s['context']),budget=item['budget'],
                overlap=len(set(s['ids'])&set(prior['full_ids'])),same_set=set(s['ids'])==set(prior['full_ids'])))
    result=dict(scores={arm:float(np.mean([x['mapped_score'] for x in scores[arm].values()])*100) for arm in r.ARMS},
        full_reference_percent=float(np.mean([s['mapped_score'] for s in base.values()])*100),comparisons=comparisons,
        costs=usage|{'full_refresh':baseu},generic_route_details=by_route,
        selection_summary={arm:{k:a.stats(x[k] for x in rows) for k in ('n','chars','budget','overlap')}|
            {'same_set':sum(x['same_set'] for x in rows)} for arm,rows in selections.items()},
        n=371,clusters=8,bootstrap_n=10000,bootstrap_seed=92602,
        generic_same_inputs=371,generic_same_calls=371,missing=missing,
        note='Same frozen memories and developed371; only one downstream realization. Same budgets do not imply equal consumed compute. All4 predeclared controls reported.',
        code_sha256=r.b.file_sha(__file__))
    r.b.save(r.ROOT/'analysis'/f'{time.time_ns()}.json',result)
    print(json.dumps({k:result[k] for k in ('scores','full_reference_percent','comparisons','costs')},ensure_ascii=True,indent=2))

if __name__=='__main__':main(len(sys.argv)>1 and sys.argv[1]=='final')

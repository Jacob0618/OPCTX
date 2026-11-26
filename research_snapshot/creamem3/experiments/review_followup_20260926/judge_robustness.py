"""Existing dual-Judge scores on exactly the same frozen answers; no API."""
from collections import Counter
import numpy as np
from . import run as r
from creamem3.experiments.review_fast_20260926 import audit as a

def main():
    scores={};summaries={};a.SEED=92602
    for arm in ('enhanced_reader','original_reader','no_reader','session_only'):
        scores[arm]={judge:{} for judge in ('g35flash','g35lite')}
        for ref in r.b.read(r.ROOT/'registration.json')['rows']:
            root=r.old.oldroot(ref['task']);key=r.b.sha(ref['row_id'])+'.json'
            pred=a.read(root/'predictions'/arm/key)
            for judge in scores[arm]:
                path=root/'scores'/judge/arm/key
                if not path.exists():continue
                s=a.read(path);assert s['prediction_sha256']==pred['seal_sha256']
                scores[arm][judge][ref['row_id']]=s
        left,right=scores[arm]['g35flash'],scores[arm]['g35lite']
        common=sorted(set(left)&set(right));conf=Counter((left[k]['raw_score'],right[k]['raw_score']) for k in common)
        summaries[arm]=dict(counts={j:len(s) for j,s in scores[arm].items()},paired_n=len(common),
            score_disagreements=sum(left[k]['mapped_score']!=right[k]['mapped_score'] for k in common),
            raw_confusion={str(pair):n for pair,n in conf.items()},
            judge_delta=a.paired({k:left[k] for k in common},{k:right[k] for k in common}))
    contrasts={}
    for judge in ('g35flash','g35lite'):
        left,right=scores['original_reader'][judge],scores['no_reader'][judge]
        assert len(left)==len(right)==371
        contrasts[judge]=a.paired(left,right)
    r.b.save(r.ROOT/'judge_robustness/results.json',dict(summaries=summaries,
        original_reader_minus_direct=contrasts,model_calls=0,
        note='Scores already existed on identical predictions. This tests Judge model sensitivity, not independent regeneration or objective human correctness. originalReader/direct use identical selected contexts.'))
    r.b.save(r.ROOT/'judge_robustness/provenance.json',dict(inputs=a.INPUTS,code_sha256=r.b.file_sha(__file__)))
    import json
    print(json.dumps(dict(summaries=summaries,reader=contrasts),ensure_ascii=True,indent=2))

if __name__=='__main__':main()

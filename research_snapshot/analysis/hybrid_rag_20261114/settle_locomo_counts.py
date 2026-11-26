"""Final LoCoMo settlement after two specifically authorized count failures.

No API calls and no writes to original run or original policy evidence.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

WORKSPACE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORKSPACE))
from analysis.hybrid_rag_20261114.data import load_dataset

RUN=Path('D:/hybrid_rag_20261114/locomo_r1')
PRIOR_POLICY=Path('D:/hybrid_rag_20261114/locomo_policy_zero_20261114_r1')
OUTPUT=Path('D:/hybrid_rag_20261114/locomo_final_20261114_r1')
AUTHORIZED={'conv-43-q002':7,'conv-48-q127':1}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2)
        handle.write('\n')


def main():
    if (OUTPUT/'summary.json').exists():
        print(json.dumps(read(OUTPUT/'summary.json'),ensure_ascii=False))
        return
    config=read(RUN/'config.json')
    if config['dataset']!='locomo' or config['expected']!=1540:
        raise RuntimeError('Unexpected frozen evaluation population')
    for path,expected in config['source_sha256'].items():
        if sha(path)!=expected:
            raise RuntimeError('Frozen production source changed')
    qmap={q['question_id']:{**q,'context_id':c['context_id']}
          for c in load_dataset('locomo') for q in c['questions']}
    if set(qmap)!=set(config['question_ids']):
        raise RuntimeError('Dataset and frozen question IDs differ')
    prior=read(PRIOR_POLICY/'policy.json')
    if prior['source_config_sha256']!=sha(RUN/'config.json'):
        raise RuntimeError('Prior policy belongs to another configuration')
    policy=dict(policy_id='locomo_two_count_failures_zero_20261114',
        authorization='User explicitly chose only the four newly identified failures; these are the two LoCoMo cases.',
        source_run=str(RUN),source_config_sha256=sha(RUN/'config.json'),
        question_ids=list(AUTHORIZED),returned_counts=AUTHORIZED,expected_count=20,
        correct=0,judge_called=False,new_model_calls=0,
        does_not_authorize_future_other_failures=True)
    save(OUTPUT/'new_policy.json',policy)
    rows=[]
    for path in sorted((RUN/'questions').glob('*/result.json')):
        r=read(path)
        if r['question_id'] not in qmap or r['correct'] not in (0,1,False,True):
            raise RuntimeError('Invalid actual Judge row')
        rows.append(dict(question_id=r['question_id'],context_id=r['context_id'],
            category_name=qmap[r['question_id']]['category_name'],correct=int(r['correct']),
            score_origin='actual_judge',prediction=r['prediction'],
            source=str(path),source_sha256=sha(path)))
    for path in sorted((PRIOR_POLICY/'decisions').glob('*.json')):
        r=read(path)
        if (r['question_id'] not in qmap or r['correct']!=0 or r['judge_called'] is not False
                or sha(r['source_failure'])!=r['source_failure_sha256']):
            raise RuntimeError('Invalid prior policy decision')
        rows.append(dict(question_id=r['question_id'],context_id=qmap[r['question_id']]['context_id'],
            category_name=qmap[r['question_id']]['category_name'],correct=0,
            score_origin='authorized_invalid_id_failure_zero',prediction=None,
            source=str(path),source_sha256=sha(path)))
    for qid,count in AUTHORIZED.items():
        folder=RUN/'questions'/qid
        f=read(folder/'failure.json')
        audit=f.get('audit') or {}
        if (f['question_id']!=qid or f['error_type']!='StageFailure'
                or audit.get('attempts')!=2 or audit.get('fallback') is not True
                or audit.get('error')!=f'GraphMemoryError: reranker returned {count} IDs; expected 20'
                or any((folder/name).exists() for name in ('prediction.json','inference.json','result.json'))):
            raise RuntimeError('Specific authorized count failure does not match')
        r=dict(question_id=qid,context_id=qmap[qid]['context_id'],
            category_name=qmap[qid]['category_name'],correct=0,
            score_origin='authorized_count_failure_zero',prediction=None,
            source=str(folder/'failure.json'),source_sha256=sha(folder/'failure.json'),
            model_judge_called=False)
        save(OUTPUT/'new_decisions'/(qid+'.json'),r)
        rows.append(r)
    ids=[r['question_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(config['question_ids']):
        raise RuntimeError('Evaluation is incomplete or has overlapping settlements')
    groups=defaultdict(list)
    for r in rows:
        groups[r['category_name']].append(r)
    metric=lambda rr:dict(n=len(rr),correct=sum(r['correct'] for r in rr),
                          accuracy_percent=100*sum(r['correct'] for r in rr)/len(rr))
    summary=dict(dataset='locomo',complete=True,overall=metric(rows),
        categories={k:metric(groups[k]) for k in ('single','multi','temporal','open')},
        actual_judge_count=sum(r['score_origin']=='actual_judge' for r in rows),
        invalid_id_policy_zero_count=sum(r['score_origin']=='authorized_invalid_id_failure_zero' for r in rows),
        count_policy_zero_count=len(AUTHORIZED),
        failures_included_in_denominator=True,source_config_sha256=sha(RUN/'config.json'),
        created_utc=datetime.now(timezone.utc).isoformat())
    save(OUTPUT/'settled_rows.json',sorted(rows,key=lambda r:r['question_id']))
    save(OUTPUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

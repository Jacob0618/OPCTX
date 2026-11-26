"""User-authorized zero scores for exhausted invalid-ID LoCoMo reranking.

Independent overlay only: never changes a frozen runner, prediction, request,
failure, or Judge output. No network, model calls, credentials, or retries.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('D:/hybrid_rag_20261114/locomo_r1')
OUTPUT=Path('D:/hybrid_rag_20261114/locomo_policy_zero_20261114_r1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2)
        handle.write('\n')


def eligible(failure):
    audit=failure.get('audit') or {}
    return (failure.get('error_type')=='StageFailure'
        and failure.get('error')=='Original Gemini reranker exhausted validation'
        and audit.get('attempts')==2 and audit.get('fallback') is True
        and 'unknown fact ID' in str(audit.get('error',''))
        and failure.get('transport_stopped') is False)


def apply_policy(root=ROOT,output=OUTPUT):
    root,output=Path(root),Path(output)
    config=read(root/'config.json')
    if config['dataset']!='locomo':
        raise RuntimeError('This policy is not authorized for another dataset')
    expected=set(config['question_ids'])
    policy=dict(policy_id='locomo_invalid_rerank_id_zero_20261114',
        authorization='User answered 判错 to the LoCoMo invalid-rerank-ID recovery question',
        dataset='locomo',source_run=str(root.resolve()),
        source_config_sha256=sha(root/'config.json'),
        eligibility='Original two-attempt Gemini reranker exhausted; unknown fact ID; no valid QA/result',
        score=0,judge_called=False,extra_model_calls=0,
        applies_to_current_and_later_same_failure_in_this_frozen_run=True,
        excluded='QA truncation, safety/network errors, LME failures, other error types',
        retains_all_original_evidence=True)
    pp=output/'policy.json'
    if pp.exists():
        if read(pp)!=policy:
            raise RuntimeError('Policy registration changed')
    else:
        save(pp,policy)
    other=[]
    for path in sorted((root/'questions').glob('*/failure.json')):
        failure=read(path)
        qid=failure['question_id']
        if qid not in expected or path.parent.name!=qid:
            raise RuntimeError('Failure identity is outside frozen population')
        if not eligible(failure):
            other.append(dict(question_id=qid,error_type=failure.get('error_type')))
            continue
        if any(path.with_name(name).exists() for name in
               ('prediction.json','inference.json','result.json')):
            raise RuntimeError('Never replace a successful prediction/result with a policy zero')
        retrieval=read(path.with_name('retrieval.json'))
        if retrieval.get('question_id')!=qid or len(retrieval['candidate_ids'])!=40:
            raise RuntimeError('Unexpected fixed candidate contract')
        target=output/'decisions'/(qid+'.json')
        decision=dict(question_id=qid,correct=0,status='user_policy_failure_zero',
            prediction=None,judge=None,judge_called=False,
            source_failure=str(path),source_failure_sha256=sha(path),
            source_retrieval_sha256=sha(path.with_name('retrieval.json')),
            reason='Gemini reranker returned candidate-external IDs after original two attempts',
            policy_id=policy['policy_id'])
        if target.exists():
            if read(target)!=decision:
                raise RuntimeError('Original failed evidence changed')
        else:
            save(target,decision)
    zero=[read(p) for p in (output/'decisions').glob('*.json')]
    results=[read(p) for p in (root/'questions').glob('*/result.json')]
    inferred=[p.parent.name for p in (root/'questions').glob('*/inference.json')]
    ids=[r['question_id'] for r in results]+[r['question_id'] for r in zero]
    if len(ids)!=len(set(ids)) or not set(ids)<=expected:
        raise RuntimeError('Overlapping/foreign settled question IDs')
    correct=sum(r['correct'] for r in results)
    complete=set(ids)==expected
    summary=dict(policy_id=policy['policy_id'],expected=len(expected),
        valid_scored=len(results),policy_zero=len(zero),settled=len(ids),
        inference_saved=len(inferred),missing_or_unscored=len(expected)-len(ids),
        other_failures=other,complete=complete,correct=correct,
        overall_accuracy_percent=100*correct/len(expected) if complete else None,
        note='No full-population accuracy until every expected ID is scored or explicitly policy-settled',
        generated_utc=datetime.now(timezone.utc).isoformat())
    save(output/'summaries'/(str(time.time_ns())+'.json'),summary)
    if complete:
        rows=[{**r,'status':'actual_judge'} for r in results]+zero
        target=output/'final_settled_results.json'
        if not target.exists():
            save(target,dict(summary=summary,rows=sorted(rows,key=lambda r:r['question_id'])))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch-pid',type=int)
    args=parser.parse_args()
    while True:
        summary=apply_policy()
        print(json.dumps({k:summary[k] for k in ('valid_scored','policy_zero','inference_saved','complete') }),flush=True)
        if not args.watch_pid:
            break
        import psutil
        try:
            proc=psutil.Process(args.watch_pid)
            command=' '.join(proc.cmdline()).replace('\\','/')
            active='hybrid_rag_20261114/run.py' in command and '--dataset locomo ' in command
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            active=False
        if not active:
            break
        time.sleep(15)


if __name__=='__main__':
    main()

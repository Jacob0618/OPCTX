"""Isolate confirmed QA MAX_TOKENS=500 to one missing question, never a zero.

--prepare is offline. --run --allow-model-calls resumes the frozen pipeline.
Runtime layout: runtime/questions/<qid>/{calls,qwen_calls,logical_calls}.
All other provider/safety/permission/unknown failures still stop the whole run.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from filelock import FileLock
from analysis.hybrid_rag_20261114 import run as original, stages, transport
from analysis.hybrid_rag_20261114 import lme_resume_policy_zero as r2
from analysis.hybrid_rag_20261114 import lme_resume_two_zeros as r3

PARENTS=(r3.ANCESTOR,r3.PARENT,r3.OUTPUT)
OUTPUT=Path('D:/hybrid_rag_20261114/lme_r4')
KNOWN_MISSING={'afdc33df':'000021_qa','195a1a1b':'000023_qa'}
POLICY_IDS=r3.POLICY_IDS
read,save,digest,file_sha,bound=r3.read,r3.save,r3.digest,r3.file_sha,r3.bound


class QuestionMaxTokens(transport.ExecutionStop):
    """Only a confirmed, fully preserved QA500 truncation; never a score."""
    def __init__(self,evidence):
        super().__init__('Question-local QA MAX_TOKENS=500; preserved as missing')
        self.evidence=evidence


def is_qa500(req,out,raw,qid):
    c=raw.get('candidates') or []
    return (out.get('scope')==qid and out.get('stage')=='qa'
        and req.get('kind')=='generation' and req.get('model')=='gemini-2.5-flash'
        and raw.get('modelVersion')=='gemini-2.5-flash' and not raw.get('error')
        and out.get('physical_calls')==1 and out.get('http_status')==200
        and out.get('model')=='gemini-2.5-flash' and out.get('model_version')=='gemini-2.5-flash'
        and out.get('finish_reason')=='MAX_TOKENS'
        and req.get('request',{}).get('generationConfig',{}).get('maxOutputTokens')==500
        and raw.get('usageMetadata',{}).get('candidatesTokenCount')==500
        and len(c)==1 and c[0].get('finishReason')=='MAX_TOKENS'
        and not raw.get('promptFeedback',{}).get('blockReason')
        and not any(r.get('blocked') for r in raw.get('promptFeedback',{}).get('safetyRatings',[]))
        and not any(r.get('blocked') for r in c[0].get('safetyRatings',[])))


def last_parent_ledger(parent):
    validator=object.__new__(transport.original.AuditTransport)
    entries=[]
    failures=set()
    for kind,dirname in (('gemini','calls'),('qwen','qwen_calls')):
        for folder in sorted((parent/'runtime'/dirname).glob('*')):
            if not folder.is_dir():
                continue
            req,out,raw=read(folder/'request.json'),read(folder/'outcome.json'),read(folder/'response.json')
            model=req['model'] if kind=='gemini' else 'qwen3-rerank'
            key=digest([model,req['request']])
            if key!=out['request_key'] or digest(raw)!=out['response_sha256']:
                raise RuntimeError('r3 physical receipt binding mismatch')
            if out['status']=='ok':
                if kind=='gemini':
                    validator._validate_response(model,req['kind'],req['request'],raw)
                else:
                    rows=raw.get('output',{}).get('results',[])
                    n=len(req['request']['input']['documents'])
                    ids=[v.get('index') for v in rows]
                    if (req['endpoint']!=transport.QWEN_URL or req['request']['model']!='qwen3-rerank'
                            or len(rows)!=n or any(type(i) is not int for i in ids)
                            or set(ids)!=set(range(n))
                            or any(not math.isfinite(float(v['relevance_score'])) for v in rows)):
                        raise RuntimeError('Invalid parent Qwen output')
            else:
                qid=out['scope']
                if not (qid in KNOWN_MISSING and folder.name==KNOWN_MISSING[qid] and is_qa500(req,out,raw,qid)):
                    raise RuntimeError('Unresolved non-QA500 parent failure')
                failures.add(qid)
            entries.append(dict(kind=kind,key=key,request=req,outcome=out,raw=raw,folder=folder))
    if failures!=set(KNOWN_MISSING):
        raise RuntimeError('Expected exactly the two known missing questions')
    return entries


def validate_registration(parents,output,registration):
    if registration['entry_sha256']!=file_sha(Path(__file__)) or registration['parents']!=list(map(str,parents)) or registration['output']!=str(output):
        raise RuntimeError('Recovery source/root registration changed')
    r3.verify_registration(*parents,read(parents[-1]/'recovery_registration.json'))
    for relative,expected in registration['parent_files_sha256'].items():
        if file_sha(bound(parents[-1],parents[-1]/relative))!=expected:
            raise RuntimeError('Immutable r3 artifact changed: '+relative)
    for relative,expected in registration['copied_files_sha256'].items():
        if file_sha(bound(output,output/relative))!=expected:
            raise RuntimeError('Imported checkpoint changed: '+relative)
    for qid,expected in registration['known_missing_sha256'].items():
        if file_sha(output/'questions'/qid/'missing.json')!=expected:
            raise RuntimeError('Known missing record changed')


def prepare(parents,output):
    parents=tuple(p.resolve() for p in parents)
    output=output.resolve()
    roots=(*parents,output)
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):
        raise RuntimeError('Four disjoint sibling roots required')
    output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for root in roots:
            stack.enter_context(FileLock(str(root/'runner.lock'),timeout=0))
        r3.verify_registration(*parents,read(parents[-1]/'recovery_registration.json'))
        adapter=stages.make_stages(lambda *_a,**_k:(_ for _ in ()).throw(RuntimeError('Offline only')))
        contexts,config=original.frozen_prepare(SimpleNamespace(output=parents[-1],dataset='lme',workers=4),adapter)
        entries=r2.ledger(parents[0])+r3.parent_ledger(parents[1])+last_parent_ledger(parents[2])
        registration_path=output/'recovery_registration.json'
        if registration_path.exists():
            registration=read(registration_path)
            validate_registration(parents,output,registration)
            return registration,entries,contexts,config
        parent=parents[-1]
        if len(list((parent/'questions').glob('*/prediction.json')))!=135:
            raise RuntimeError('Expected 135 durable parent predictions')
        qmap={q['question_id']:q for c in contexts for q in c['questions']}
        sources=[parent/'config.json',parent/'offline_inventory.json']
        for folder in (parent/'indices').glob('*'):
            if not folder.is_dir():
                continue
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json')
                for name,key in (('chunks.json','chunks_sha256'),('vectors.npy','vectors_sha256')):
                    if file_sha(folder/name)!=done[key]:
                        raise RuntimeError('Parent index hash mismatch')
            sources.extend(p for p in folder.iterdir() if p.name in {'chunks.json','vectors.npy','complete.json'})
        for folder in (parent/'questions').glob('*'):
            if folder.name not in qmap:
                raise RuntimeError('Unknown parent question')
            for p in folder.iterdir():
                if p.name not in r2.CHECKPOINT_NAMES:
                    continue
                if folder.name in KNOWN_MISSING and p.name in {'prediction.json','inference.json','result.json'}:
                    raise RuntimeError('Known missing question unexpectedly has a valid output')
                value=read(p)
                if p.name in {'result.json','inference.json'} and value['question_id']!=folder.name:
                    raise RuntimeError('Question checkpoint identity mismatch')
                sources.append(p)
        if shutil.disk_usage(output.anchor).free<sum(p.stat().st_size for p in sources)+15*1024**3:
            raise RuntimeError('Insufficient safe disk space')
        copied={}
        for source in sources:
            relative=bound(parent,source).relative_to(parent).as_posix()
            copied[relative]=r2.checked_copy(source,bound(output,output/relative))
        missing_hashes={}
        for qid in KNOWN_MISSING:
            failure=next(e for e in entries if e['outcome']['scope']==qid and e['outcome']['status']!='ok')
            value=dict(question_id=qid,status='missing',correct=None,judge=None,model_judge_called=False,
                reason='Known QA MAX_TOKENS=500; no retry or zero authorized.',
                original_failure=str(failure['folder']),original_response_sha256=failure['outcome']['response_sha256'])
            path=output/'questions'/qid/'missing.json'
            if path.exists():
                if read(path)!=value:
                    raise RuntimeError('Known missing marker differs')
            else:
                save(path,value)
            missing_hashes[qid]=file_sha(path)
        preserved=set(sources)|{parent/'recovery_registration.json'}
        for sub in ('runtime','executions','execution_timing'):
            preserved.update(p for p in (parent/sub).rglob('*.json') if p.is_file())
        preserved.update((parent/'questions').glob('*/failure.json'))
        registration=dict(schema='hybrid_rag.isolated_qa500_missing.v1',entry=str(Path(__file__).resolve()),
            entry_sha256=file_sha(Path(__file__)),parents=list(map(str,parents)),output=str(output),
            parent_files_sha256={bound(parent,p).relative_to(parent).as_posix():file_sha(p) for p in sorted(preserved)},
            copied_files_sha256=copied,known_missing_sha256=missing_hashes,inherited_predictions=135,
            policy_zero_ids=sorted(POLICY_IDS),known_missing_ids=sorted(KNOWN_MISSING),evaluation_population=500,
            runtime_layout='per_question_isolated',workers=4,qa_max_tokens=500,
            scheduler_change='Only confirmed QA500 MAX_TOKENS stops its question; every other failure stops globally.',
            source_contract='All 500 IDs and the four frozen sources verified by original.frozen_prepare.',
            policy_change=False,new_missing_scored_zero=False,retry_known_failure=False,
            cost_roots=[str(p/'runtime') for p in roots],stop_file=str(output/'STOP'),
            created_utc=datetime.now(timezone.utc).isoformat())
        save(registration_path,registration)
        validate_registration(parents,output,registration)
        return registration,entries,contexts,config


class SharedNative:
    """Share only exact-response cache/attempt locks, never the per-question stop."""
    def __init__(self,entries):
        self.lock=threading.RLock()
        self.request_locks={}
        self.stops=[]
        self.attempts,self.qwen_attempts,self.success,self.qwen_success={},{},{},{}
        self.blocked={e['key'] for e in entries if e['outcome']['status']!='ok'}
        for e in entries:
            attempts=self.attempts if e['kind']=='gemini' else self.qwen_attempts
            cache=self.success if e['kind']=='gemini' else self.qwen_success
            attempts[e['key']]=attempts.get(e['key'],0)+int(e['outcome']['physical_calls'])
            if e['outcome']['status']=='ok':
                if e['key'] in cache and digest(cache[e['key']])!=digest(e['raw']):
                    raise RuntimeError('Conflicting inherited native response')
                cache[e['key']]=e['raw']


def stop_all(shared,global_stop):
    global_stop.set()
    with shared.lock:
        for stop in shared.stops:
            stop.set()


def transport_class(shared,global_stop,output):
    base=transport.AuditTransport
    class IsolatedTransport(base):
        def __init__(self,runtime):
            super().__init__(runtime)
            with shared.lock:
                for local,total in ((self.attempts,shared.attempts),(self.qwen_attempts,shared.qwen_attempts)):
                    for key,n in local.items():
                        total[key]=total.get(key,0)+n
                for local,total in ((self.raw_success,shared.success),(self.qwen_success,shared.qwen_success)):
                    for key,raw in local.items():
                        if key in total and digest(total[key])!=digest(raw):
                            raise transport.ExecutionStop('Recovery native response collision')
                        total[key]=raw
            self.lock=shared.lock
            self.request_locks=shared.request_locks
            self.attempts,self.qwen_attempts=shared.attempts,shared.qwen_attempts
            self.raw_success,self.qwen_success=shared.success,shared.qwen_success
            with shared.lock:
                shared.stops.append(self.stop)
                if global_stop.is_set():
                    self.stop.set()
        def _request_lock(self,key):
            with shared.lock:
                return shared.request_locks.setdefault(key,threading.RLock())
        def guard(self):
            qid=str(getattr(self.local,'scope','shared')).split('/')[0]
            if global_stop.is_set() or (output/'STOP').exists() or qid in POLICY_IDS or qid in KNOWN_MISSING:
                stop_all(shared,global_stop)
                raise transport.ExecutionStop('Global/manual stop or excluded question')
        def request(self,url,payload):
            self.guard()
            model,_,_=self._endpoint(url,payload)
            key=digest([model,payload])
            try:
                with self._request_lock(key):
                    self.guard()
                    if key in shared.blocked:
                        raise transport.ExecutionStop('Known failed exact request cannot be replayed')
                    return super().request(url,payload)
            except BaseException:
                qid=str(getattr(self.local,'scope','shared'))
                match=next((c for c in reversed(self.calls) if c.get('request_key')==key),None)
                if match:
                    folder=self.output/'calls'/f"{match['id']:06d}_{match['stage']}"
                    if (folder/'response.json').exists():
                        req,raw=read(folder/'request.json'),read(folder/'response.json')
                        if (digest([req.get('model'),req.get('request')])==key
                                and digest(raw)==match.get('response_sha256')
                                and is_qa500(req,match,raw,qid)):
                            shared.blocked.add(key)
                            raise QuestionMaxTokens(dict(call=str(folder),request_key=key,
                                response_sha256=digest(raw),usage=match.get('usage'))) from None
                stop_all(shared,global_stop)
                raise
        def qwen_post(self,endpoint,payload):
            self.guard()
            try:
                with self._request_lock(digest(['qwen3-rerank',payload])):
                    self.guard()
                    return super().qwen_post(endpoint,payload)
            except BaseException:
                stop_all(shared,global_stop)
                raise
    return IsolatedTransport


def save_missing(output,qid,evidence):
    path=output/'questions'/qid/'missing.json'
    value=dict(question_id=qid,status='missing',correct=None,judge=None,model_judge_called=False,
        reason='QA MAX_TOKENS at unchanged 500-token cap; not retried and not scored zero.',evidence=evidence)
    if path.exists():
        if read(path)!=value:
            raise RuntimeError('Existing missing evidence differs')
    else:
        save(path,value)


def recover_missing_markers(output):
    """Repair only bookkeeping after a crash, never retry a recorded truncation."""
    for outpath in (output/'runtime/questions').glob('*/calls/*/outcome.json'):
        out=read(outpath)
        if out['status']=='ok':
            continue
        folder=outpath.parent
        qid=folder.parent.parent.name
        req,raw=read(folder/'request.json'),read(folder/'response.json')
        if (digest([req.get('model'),req.get('request')])!=out.get('request_key')
                or digest(raw)!=out.get('response_sha256') or not is_qa500(req,out,raw,qid)):
            raise transport.ExecutionStop('Unresolved prior non-QA500 failure')
        save_missing(output,qid,dict(call=str(folder),request_key=out['request_key'],
            response_sha256=digest(raw),usage=out.get('usage')))


def summary(parents,output,config,global_stop,elapsed):
    rows=[read(p) for p in (output/'questions').glob('*/result.json')]
    missing=[read(p) for p in (output/'questions').glob('*/missing.json')]
    ids={r['question_id'] for r in rows}
    mids={r['question_id'] for r in missing}
    expected=set(config['question_ids'])
    if ids&mids or not ids|mids<=expected or len(ids)!=len(rows):
        raise RuntimeError('Result/missing identity inconsistency')
    group=defaultdict(list)
    for row in rows:
        group[row['category_name']].append(row)
    def counts(rr):
        return dict(n=len(rr),correct=sum(r['correct'] for r in rr),
            accuracy_percent=100*sum(r['correct'] for r in rr)/len(rr) if rr else None)
    calls=[]
    for root in (*parents,output):
        calls.extend(read(p) for p in (root/'runtime').rglob('outcome.json'))
    stats=defaultdict(lambda:dict(physical_calls=0,known_input_tokens=0,known_output_tokens=0,
        thought_tokens=0,cached_input_tokens=0,usage_missing_calls=0,service_seconds=0.0))
    generation=standard=qwen=0.0
    for call in calls:
        s=stats[call['stage']]
        s['physical_calls']+=call.get('physical_calls',0)
        s['service_seconds']+=call.get('service_seconds',0)
        u=call.get('usage')
        if not u:
            s['usage_missing_calls']+=call.get('physical_calls',0)
            continue
        tokens=int(u.get('promptTokenCount',u.get('input_tokens',u.get('total_tokens',0))) or 0)
        out=int(u.get('candidatesTokenCount',u.get('output_tokens',0)) or 0)
        thought=int(u.get('thoughtsTokenCount',0) or 0)
        cached=int(u.get('cachedContentTokenCount',0) or 0)
        s['known_input_tokens']+=tokens;s['known_output_tokens']+=out
        s['thought_tokens']+=thought;s['cached_input_tokens']+=cached
        if call['stage']!='judge' and call['model']=='gemini-2.5-flash':
            generation+=((tokens-cached)*.30+cached*.03+(out+thought)*2.50)/1e6
            standard+=(tokens*.30+(out+thought)*2.50)/1e6
        elif call['stage']!='judge' and call['model']=='qwen3-rerank':
            qwen+=tokens*.50/1e6
    for s in stats.values():
        s['input_tokens']=None if s['usage_missing_calls'] else s['known_input_tokens']
        s['output_tokens']=None if s['usage_missing_calls'] else s['known_output_tokens']
    result=dict(dataset='lme',expected=500,complete=ids==expected,
        all_questions_accounted=(ids|mids)==expected,transport_stopped=global_stop.is_set(),
        overall=counts(rows),accuracy_denominator='Scored questions only, including exactly 3 authorized zeros; missing excluded.',
        scored_n=len(rows),valid_judged_n=sum(r.get('judge') is not None for r in rows),
        policy_zero_ids=sorted(POLICY_IDS),policy_zero_count=3,missing_n=len(missing),missing=missing,
        unscored_remaining=sorted(expected-ids-mids),coverage_percent=100*len(ids)/500,
        categories={k:counts(v) for k,v in group.items()},runtime_layout='per_question_isolated',
        physical_stage_usage=dict(stats),physical_inference_cost=dict(gemini_generation_usd=generation,
            gemini_generation_standard_input_usd=standard,qwen_rerank_cny=qwen,embedding_usd=None,
            embedding_token_cost_not_estimated=True,all_failed_call_usage_included=True,
            judge_cost_excluded=True,judge_usage_reported_separately=True,pricing_convention_date='2026-11-11'),
        current_execution_wall_seconds_including_evaluation=elapsed,cost_roots=list(map(str,(*parents,output))))
    save(output/'recovery_summaries'/(str(time.time_ns())+'.json'),result)
    print(json.dumps({k:result[k] for k in ('overall','scored_n','valid_judged_n','missing_n','all_questions_accounted','transport_stopped')}),flush=True)
    return result


def run_live(parents,output,registration,entries,contexts,config):
    with FileLock(str(output/'runner.lock'),timeout=0):
        validate_registration(parents,output,registration)
        if (output/'STOP').exists():
            raise RuntimeError('STOP file exists')
        recover_missing_markers(output)
        args=SimpleNamespace(output=output,dataset='lme',workers=4)
        offline=stages.make_stages(lambda *_a,**_k:(_ for _ in ()).throw(RuntimeError('Offline only')))
        contexts,config=original.frozen_prepare(args,offline)  # all 500, genuine unchanged-source check
        transport.preflight(output,include_qwen=True)
        global_stop=threading.Event()
        shared=SharedNative(entries)
        cls=transport_class(shared,global_stop,output)
        wires={}
        errors=[]
        def clients(qid):
            if qid not in wires:
                directory=output/'runtime/questions'/original.safe_id(qid)
                directory.mkdir(parents=True,exist_ok=True)
                wire=cls(directory)
                chat,embed=transport.make_clients(wire)
                wires[qid]=(wire,embed,stages.make_stages(chat,qwen_post=wire.qwen_post))
            return wires[qid]
        def infer(context):
            q=context['questions'][0];qid=q['question_id'];directory=output/'questions'/qid
            if global_stop.is_set() or qid in POLICY_IDS or (directory/'missing.json').exists() or (directory/'inference.json').exists():
                return
            try:
                wire,embed,adapter=clients(qid)
                index=original.build_index(context,args,wire,embed)
                original.process_question(context,q,index,args,wire,embed,adapter,score=False)
                print(json.dumps(dict(question_id=qid,status='prediction_saved')),flush=True)
            except QuestionMaxTokens as exc:
                save_missing(output,qid,exc.evidence)
                print(json.dumps(dict(question_id=qid,status='missing_qa500_no_retry')),flush=True)
            except BaseException as exc:
                stop_all(shared,global_stop)
                errors.append(dict(question_id=qid,stage='inference',error_type=type(exc).__name__))
                raise
        def score(row):
            if global_stop.is_set():
                return
            qid=row['question_id']
            try:
                wire,_,adapter=clients(qid)
                original.score_inference(row,args,wire,adapter)
            except BaseException as exc:
                stop_all(shared,global_stop);errors.append(dict(question_id=qid,stage='judge',error_type=type(exc).__name__))
                raise
        start=time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(infer,c) for c in contexts]
                for future in as_completed(futures):
                    try:future.result()
                    except BaseException:stop_all(shared,global_stop)
            save(output/'execution_timing'/(str(time.time_ns())+'_inference.json'),
                dict(wall_seconds=time.perf_counter()-start,includes_indexing=True,includes_judge=False))
            if not global_stop.is_set():
                judge_start=time.perf_counter()
                pending=[read(p) for p in (output/'questions').glob('*/inference.json')
                         if not p.with_name('result.json').exists() and not p.with_name('missing.json').exists()]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures=[pool.submit(score,row) for row in pending]
                    for future in as_completed(futures):
                        try:future.result()
                        except BaseException:stop_all(shared,global_stop)
                save(output/'execution_timing'/(str(time.time_ns())+'_judge.json'),
                    dict(wall_seconds=time.perf_counter()-judge_start,includes_judge=True))
        finally:
            result=summary(parents,output,config,global_stop,time.perf_counter()-start)
            save(output/'executions'/(str(time.time_ns())+'.json'),dict(errors=errors,
                transport_stopped=global_stop.is_set(),all_questions_accounted=result['all_questions_accounted']))
        if global_stop.is_set() or not result['all_questions_accounted']:
            raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--allow-model-calls',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    parents=tuple(p.resolve() for p in PARENTS);output=args.output.resolve()
    registration,entries,contexts,config=prepare(parents,output)
    if not args.run:
        print(json.dumps(dict(status='offline_prepared',output=str(output),inherited_predictions=135,
            known_missing_ids=sorted(KNOWN_MISSING),policy_zero_ids=sorted(POLICY_IDS),runtime_layout='per_question_isolated',model_calls=0)))
        return
    if not args.allow_model_calls:
        raise RuntimeError('Live continuation requires --allow-model-calls')
    run_live(parents,output,registration,entries,contexts,config)


if __name__=='__main__':main()

"""Judge-only completion: immutable answers, serial AST compilation, 4 workers.

No Writer, embedding, retrieval, reranker or QA calls are allowed by transport.
Only pre-request SystemError may remain unscored while other questions proceed.
All native/provider/permission/safety/unknown failures remain fail-closed.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from filelock import FileLock
from analysis.hybrid_rag_20261114 import run as original,stages,transport
from analysis.hybrid_rag_20261114 import lme_isolated_continue_r5 as previous

PARENTS=(*previous.PARENTS,previous.OUTPUT)
OUTPUT=Path('D:/hybrid_rag_20261114/lme_judge_finish_r6')
POLICY_IDS=previous.frozen.POLICY_IDS
QUEUED=('01493427','001be529','00ca467f')
read,save,digest,file_sha=original.read,original.save,original.digest,original.file_sha


def redact(text):
    for name in ('GEMINI_API_KEY','DASHSCOPE_API_KEY'):
        value=os.environ.get(name)
        if value:text=text.replace(value,'[REDACTED]')
    return text


def local_system_error_only(exc,before,after,wire_stopped):
    return type(exc) is SystemError and before==after and not wire_stopped


def judge_receipts(roots):
    values=[]
    for root in roots:
        # Path filtering does not deserialize or retain any embedding response.
        paths=list((root/'runtime/calls').glob('*_judge/outcome.json'))
        paths+=list((root/'runtime/questions').glob('*/calls/*_judge/outcome.json'))
        paths+=list((root/'runtime').glob('worker_*/calls/*_judge/outcome.json'))
        for path in paths:values.append((path,read(path)))
    return values


def verify_parent_judge_requests(roots):
    count=0
    for root in roots:
        folders=list((root/'runtime/calls').glob('*_judge'))
        folders+=list((root/'runtime/questions').glob('*/calls/*_judge'))
        folders+=list((root/'runtime').glob('worker_*/calls/*_judge'))
        for folder in folders:
            if not (folder/'request.json').exists() or not (folder/'outcome.json').exists():
                raise RuntimeError('Unresolved parent Judge request: '+str(folder))
            count+=1
    return count


def compile_judge():
    path=Path(stages.legacy.__file__)
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_judge')
    code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec')
    return code,digest(ast.dump(node,include_attributes=False))


def install_compiled_loader(code):
    def load(path,name,namespace):
        if Path(path).resolve()!=Path(stages.legacy.__file__).resolve() or name!='_judge':
            raise RuntimeError('Unexpected runtime compilation request in Judge-only entry')
        # The exact original function body, independent globals/captures each time.
        exec(code,namespace)
        return namespace[name]
    stages.load_function=load


def verify(output,registration):
    if registration['entry_sha256']!=file_sha(Path(__file__)) or registration['output']!=str(output):
        raise RuntimeError('Judge-only entry/root changed')
    source=PARENTS[-1].resolve()
    for relative,expected in registration['source_files_sha256'].items():
        if file_sha(source/relative)!=expected:raise RuntimeError('Immutable source changed: '+relative)
    for relative,expected in registration['copied_files_sha256'].items():
        if file_sha(output/relative)!=expected:raise RuntimeError('Inherited question checkpoint changed: '+relative)
    for path,expected in registration['implementation_sha256'].items():
        if file_sha(Path(path))!=expected:raise RuntimeError('Frozen implementation changed: '+path)


def prepare(output):
    output=output.resolve();source=PARENTS[-1].resolve()
    if source==output or source.is_relative_to(output) or output.is_relative_to(source):
        raise RuntimeError('Separate Judge-only root required')
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(source/'runner.lock'),timeout=0),FileLock(str(output/'runner.lock'),timeout=0):
        # Source contract validation is real; no old digest is replaced or forged.
        adapter=stages.make_stages(lambda *_a,**_kw:(_ for _ in ()).throw(RuntimeError('Offline only')))
        contexts,config=original.frozen_prepare(SimpleNamespace(output=source,dataset='lme',workers=4),adapter)
        regpath=output/'recovery_registration.json'
        if regpath.exists():
            registration=read(regpath);verify(output,registration);return registration,config
        inferences={p.parent.name:read(p) for p in (source/'questions').glob('*/inference.json')}
        results={p.parent.name:read(p) for p in (source/'questions').glob('*/result.json')}
        missing={p.parent.name:read(p) for p in (source/'questions').glob('*/missing.json')}
        judged={qid for qid,row in results.items() if row.get('judge') is not None}
        if (len(inferences)!=483 or len(judged)!=2 or len(missing)!=14
                or set(results)-judged!=set(POLICY_IDS)
                or set(inferences)|set(POLICY_IDS)|set(missing)!=set(config['question_ids'])
                or set(inferences)&set(missing)):
            raise RuntimeError('Expected source483 answers +3 policy zeros +14 missing, with2 valid Judge results')
        qmap={q['question_id']:q for c in contexts for q in c['questions']}
        for qid,row in inferences.items():
            prediction=read(source/'questions'/qid/'prediction.json')['prediction']
            if (row['question_id']!=qid or row['prediction']!=prediction
                    or row['question']!=qmap[qid]['raw_question'] or row['reference']!=qmap[qid]['answer']):
                raise RuntimeError('Immutable inference/prediction/dataset binding mismatch')
        parent_judge_request_count=verify_parent_judge_requests(PARENTS)
        prior_judge=judge_receipts(PARENTS)
        for path,out in prior_judge:
            if out['scope'] in QUEUED:raise RuntimeError('Queued failure actually has a Judge physical request')
            req,raw=read(path.parent/'request.json'),read(path.parent/'response.json')
            if out['status']!='ok' or digest([req['model'],req['request']])!=out['request_key'] or digest(raw)!=out['response_sha256']:
                raise RuntimeError('Prior native Judge receipt invalid')
            object.__new__(transport.original.AuditTransport)._validate_response(req['model'],req['kind'],req['request'],raw)
        sources=[source/'config.json']
        for name in ('prediction.json','inference.json','result.json','missing.json'):
            sources.extend((source/'questions').glob('*/'+name))
        copied={}
        for path in sources:
            relative=path.relative_to(source).as_posix()
            copied[relative]=previous.frozen.r2.checked_copy(path,output/relative)
        preserved=set(sources)|{source/'recovery_registration.json'}
        queued={}
        for qid in QUEUED:
            path=source/'questions'/qid/'failure.json'
            value=read(path)
            if value.get('stage')!='judge' or value['error_type'] not in ('SystemError','ExecutionStop'):
                raise RuntimeError('Unexpected queued failure category')
            preserved.add(path);queued[qid]=dict(path=str(path),sha256=file_sha(path),error_type=value['error_type'])
        _,ast_sha=compile_judge()
        implementation={str(Path(__file__).resolve()):file_sha(Path(__file__))}
        implementation.update(config['source_sha256']);implementation.update(config['model_contract']['source_hashes'])
        implementation[str(Path(transport.original.__file__))]=file_sha(Path(transport.original.__file__))
        registration=dict(schema='hybrid_rag.judge_only_finish_r6.v1',entry=str(Path(__file__).resolve()),
            entry_sha256=file_sha(Path(__file__)),output=str(output),parents=list(map(str,PARENTS)),
            source_output=str(source),index_source_output=str(source),runtime_layout='judge_only_worker_isolated',
            source_files_sha256={p.relative_to(source).as_posix():file_sha(p) for p in sorted(preserved)},
            copied_files_sha256=copied,implementation_sha256=implementation,
            source_failures_queued=queued,inherited_predictions=483,inherited_valid_judged=2,
            prior_judge_requests_complete=True,parent_native_judge_request_count=parent_judge_request_count,
            policy_zero_ids=sorted(POLICY_IDS),known_missing_ids=sorted(missing),pending_judge_count=481,
            evaluation_population=500,workers=4,model=stages.JUDGE_MODEL,max_tokens_schedule=[200,400,800,1600,3200],
            compiled_judge_ast_sha256=ast_sha,in_memory_injection='stages.load_function resolves only precompiled original _judge; fresh namespace per invocation.',
            no_embedding_cache_loaded=True,no_generation_calls=True,
            local_error_policy='Only SystemError with no additional native request remains unscored without retry; all other errors stop globally.',
            diagnosis=dict(confirmed='Pre-Judge-request SystemError; no provider call for01493427.',
                offline_same_input_serial=10,offline_same_input_four_thread=80,reproduced=False,
                root_cause='Not established; AST concurrency or memory pressure remain unconfirmed hypotheses.'),
            physical_cost_sources=[str(p/'runtime') for p in (*PARENTS,output)],judge_cost_separate=True,
            created_utc=datetime.now(timezone.utc).isoformat())
        save(regpath,registration);verify(output,registration);return registration,config


def report(output,config,stopped,local_errors,elapsed):
    rows=[read(p) for p in (output/'questions').glob('*/result.json')]
    groups=defaultdict(list)
    for row in rows:groups[row['category_name']].append(row)
    counts=lambda rr:dict(n=len(rr),correct=sum(r['correct'] for r in rr),
        accuracy_percent=100*sum(r['correct'] for r in rr)/len(rr) if rr else None)
    def usage(receipts):
        value=dict(physical_calls=0,known_input_tokens=0,known_output_tokens=0,thought_tokens=0,unknown_usage_calls=0,service_seconds=0.0)
        for _,out in receipts:
            value['physical_calls']+=out.get('physical_calls',0);value['service_seconds']+=out.get('service_seconds',0)
            u=out.get('usage')
            if not u:value['unknown_usage_calls']+=out.get('physical_calls',0);continue
            value['known_input_tokens']+=int(u.get('promptTokenCount',0) or 0)
            value['known_output_tokens']+=int(u.get('candidatesTokenCount',0) or 0)
            value['thought_tokens']+=int(u.get('thoughtsTokenCount',0) or 0)
        return value
    done={r['question_id'] for r in rows};missing={p.parent.name for p in (output/'questions').glob('*/missing.json')}
    expected=set(config['question_ids'])
    result=dict(expected=500,runtime_layout='judge_only_worker_isolated',overall=counts(rows),
        accuracy_denominator='Scored only: valid Judge plus exactly3 authorized zeros;14missing excluded.',
        scored_n=len(rows),valid_judged_n=sum(r.get('judge') is not None for r in rows),coverage_percent=100*len(rows)/500,
        policy_zero_ids=sorted(POLICY_IDS),policy_zero_count=3,missing_n=len(missing),known_missing_ids=sorted(missing),
        complete=done==expected,all_questions_accounted=done|missing==expected,
        unscored_remaining=sorted(expected-done-missing),categories={k:counts(v) for k,v in groups.items()},
        transport_stopped=stopped.is_set(),local_unscored_errors=local_errors,
        judge_usage_this_run=usage(judge_receipts((output,))),judge_usage_including_parents=usage(judge_receipts((*PARENTS,output))),
        generation_cost_source_outputs=list(map(str,PARENTS)),judge_cost_separate=True,
        current_execution_wall_seconds=elapsed)
    save(output/'recovery_summaries'/(str(time.time_ns())+'.json'),result)
    print(json.dumps({k:result[k] for k in ('overall','valid_judged_n','missing_n','unscored_remaining','transport_stopped')}),flush=True)
    return result


def run_live(output,registration,config):
    with FileLock(str(output/'runner.lock'),timeout=0):
        verify(output,registration)
        if (output/'STOP').exists():raise RuntimeError('STOP file exists')
        code,ast_sha=compile_judge()
        if ast_sha!=registration['compiled_judge_ast_sha256']:raise RuntimeError('Original Judge AST changed')
        contract=stages.lme.assert_prompt_contract()
        stopped=threading.Event();wires=[];workers=queue.Queue();local=threading.local();lock=threading.RLock()
        def stop_all():
            stopped.set()
            with lock:
                for wire in wires:wire.stop.set()
        class JudgeTransport(transport.AuditTransport):
            def request(self,url,payload):
                if stopped.is_set() or (output/'STOP').exists():
                    stop_all();raise transport.ExecutionStop('Judge completion stopped')
                model,kind,_=self._endpoint(url,payload)
                if model!=stages.JUDGE_MODEL or kind!='generation' or getattr(self.local,'stage',None)!='judge':
                    stop_all();raise transport.ExecutionStop('Judge-only transport forbids generation/retrieval stages')
                try:return super().request(url,payload)
                except BaseException:stop_all();raise
            def qwen_post(self,*args,**kwargs):
                stop_all();raise transport.ExecutionStop('Judge-only transport forbids reranking')
        # Native client AST compilation happens serially here, never in workers.
        for i in range(4):
            path=output/'runtime'/f'worker_{i}';path.mkdir(parents=True,exist_ok=True)
            wire=JudgeTransport(path);wires.append(wire)
            chat,_=transport.make_clients(wire)
            adapter=object.__new__(stages.StageAdapter);adapter.chat=chat;adapter.contract=contract
            workers.put((wire,adapter))
        install_compiled_loader(code)
        # Existing audited preflight checks the configured models without any fallback credential.
        transport.preflight(output,include_qwen=False)
        pending=[read(p) for p in (output/'questions').glob('*/inference.json')
                 if not p.with_name('result.json').exists() and not p.with_name('missing.json').exists()
                 and not p.with_name('local_judge_failure.json').exists()]
        args=SimpleNamespace(output=output,dataset='lme');local_errors=[]
        def initialize():local.worker=workers.get_nowait()
        def score(row):
            if stopped.is_set():return
            wire,adapter=local.worker;qid=row['question_id']
            before=set((wire.output/'calls').glob('*/request.json'))
            try:
                original.score_inference(row,args,wire,adapter)
                print(json.dumps(dict(question_id=qid,status='scored')),flush=True)
            except BaseException as exc:
                after=set((wire.output/'calls').glob('*/request.json'))
                evidence=dict(question_id=qid,error_type=type(exc).__name__,message=redact(str(exc)),
                    traceback=redact(traceback.format_exc()),native_request_added=bool(after-before),
                    new_native_requests=[str(p) for p in sorted(after-before)],
                    source_prediction_sha256=file_sha(output/'questions'/qid/'prediction.json'))
                safe_local=local_system_error_only(exc,before,after,wire.stop.is_set())
                evidence['local_unscored_no_retry']=safe_local
                save(output/'diagnostics'/(qid+'_'+str(time.time_ns())+'.json'),evidence)
                if safe_local:
                    save(output/'questions'/qid/'local_judge_failure.json',evidence)
                    local_errors.append(qid)
                    print(json.dumps(dict(question_id=qid,status='local_system_error_unscored')),flush=True)
                    return
                stop_all();raise
        start=time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=4,initializer=initialize) as pool:
                futures=[pool.submit(score,row) for row in pending]
                for future in as_completed(futures):
                    try:future.result()
                    except BaseException:stop_all()
        finally:
            result=report(output,config,stopped,local_errors,time.perf_counter()-start)
            save(output/'executions'/(str(time.time_ns())+'.json'),dict(transport_stopped=stopped.is_set(),
                attempted_judge_rows=len(pending),local_unscored_errors=local_errors,all_questions_accounted=result['all_questions_accounted']))
        if stopped.is_set() or not result['all_questions_accounted']:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--allow-model-calls',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();output=args.output.resolve();registration,config=prepare(output)
    if not args.run:
        print(json.dumps(dict(status='offline_prepared',predictions=483,pending_judge=481,missing=14,policy_zero=3,model_calls=0)));return
    if not args.allow_model_calls:raise RuntimeError('Judge calls require --allow-model-calls')
    run_live(output,registration,config)


if __name__=='__main__':main()

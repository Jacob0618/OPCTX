"""Independent r3 recovery: two newly authorized zeros, no general zero policy.

--prepare is offline. --run --allow-model-calls uses the unchanged frozen RAG.
Preserves all r1/r2 evidence, 133 valid predictions, and the existing policy zero.
New or unrelated failures still stop; OUTPUT/STOP prevents new requests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from filelock import FileLock
from analysis.hybrid_rag_20261114 import run as original, stages, transport
from analysis.hybrid_rag_20261114 import lme_resume_policy_zero as prior

ANCESTOR=Path('D:/hybrid_rag_20261114/lme_r1')
PARENT=Path('D:/hybrid_rag_20261114/lme_r2')
OUTPUT=Path('D:/hybrid_rag_20261114/lme_r3')
NEW_FAILURES={'06878be2':'000382_qa','75832dbd':'000383_qa'}
POLICY_IDS=frozenset((prior.POLICY_QID,*NEW_FAILURES))
read,save,digest,file_sha=prior.read,prior.save,prior.digest,prior.file_sha
bound,checked_copy=prior.bound,prior.checked_copy


def parent_ledger(parent):
    validator=object.__new__(transport.original.AuditTransport)
    entries=[]
    observed=set()
    for kind,dirname in (('gemini','calls'),('qwen','qwen_calls')):
        for folder in sorted((parent/'runtime'/dirname).glob('*')):
            if not folder.is_dir():
                continue
            req,out=read(folder/'request.json'),read(folder/'outcome.json')
            model=req['model'] if kind=='gemini' else 'qwen3-rerank'
            key=digest([model,req['request']])
            raw=read(folder/'response.json')
            if key!=out['request_key'] or digest(raw)!=out['response_sha256']:
                raise RuntimeError('Parent request/response receipt hash mismatch')
            if out['status']=='ok':
                if kind=='gemini':
                    validator._validate_response(model,req['kind'],req['request'],raw)
                else:
                    n=len(req['request']['input']['documents'])
                    results=raw.get('output',{}).get('results',[])
                    ids=[r.get('index') for r in results]
                    if (req['endpoint']!=transport.QWEN_URL or req['request']['model']!='qwen3-rerank'
                            or len(results)!=n or any(type(i) is not int for i in ids)
                            or set(ids)!=set(range(n))
                            or any(not math.isfinite(float(r['relevance_score'])) for r in results)):
                        raise RuntimeError('Parent Qwen response contract mismatch')
            else:
                qid=out['scope']
                if not (kind=='gemini' and qid in NEW_FAILURES and folder.name==NEW_FAILURES[qid]
                        and out['stage']=='qa' and model=='gemini-2.5-flash'
                        and out['model_version']=='gemini-2.5-flash'
                        and out['finish_reason']=='MAX_TOKENS'
                        and raw['candidates'][0]['finishReason']=='MAX_TOKENS'
                        and req['request']['generationConfig']['maxOutputTokens']==500
                        and out['physical_calls']==1):
                    raise RuntimeError('New/unapproved failure cannot be policy-scored: '+str(folder))
                observed.add(qid)
            entries.append(dict(kind=kind,folder=folder,request=req,outcome=out,raw=raw,key=key))
    if observed!=set(NEW_FAILURES):
        raise RuntimeError('Expected exactly the two explicitly authorized new failures')
    cancelled=[e for e in entries if e['outcome']['scope'].split('/')[0]=='35a27287']
    if not cancelled or any(e['outcome']['stage']!='embedding' or e['outcome']['status']!='ok' for e in cancelled):
        raise RuntimeError('35a27287 is not exclusively a successful partial index; inspect again')
    return entries


def verify_registration(ancestor,parent,output,registration):
    if (registration['entry_sha256']!=file_sha(Path(__file__))
            or registration['ancestor']!=str(ancestor) or registration['parent']!=str(parent)
            or registration['output']!=str(output)):
        raise RuntimeError('Recovery entry or roots changed after registration')
    prior.verify_manifest(ancestor,parent,read(parent/'recovery_registration.json'))
    for rootname,root in (('ancestor',ancestor),('parent',parent)):
        for relative,expected in registration[rootname+'_files_sha256'].items():
            if file_sha(bound(root,root/relative))!=expected:
                raise RuntimeError('Prior artifact changed: '+str(root/relative))
    for relative,expected in registration['copied_files_sha256'].items():
        if file_sha(bound(output,output/relative))!=expected:
            raise RuntimeError('Imported checkpoint changed: '+relative)
    for qid,expected in registration['new_policy_results_sha256'].items():
        if file_sha(output/'questions'/qid/'result.json')!=expected:
            raise RuntimeError('Explicit policy result changed')


def prepare(ancestor,parent,output):
    ancestor,parent,output=ancestor.resolve(),parent.resolve(),output.resolve()
    roots=(ancestor,parent,output)
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):
        raise RuntimeError('Recovery requires three distinct sibling roots')
    output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for root in roots:
            stack.enter_context(FileLock(str(root/'runner.lock'),timeout=0))
        old_registration=read(parent/'recovery_registration.json')
        prior.verify_manifest(ancestor,parent,old_registration)
        offline=stages.make_stages(lambda *_a,**_kw:(_ for _ in ()).throw(RuntimeError('Offline only')))
        contexts,config=original.frozen_prepare(SimpleNamespace(output=parent,dataset='lme',workers=4),offline)
        entries=prior.ledger(ancestor)+parent_ledger(parent)
        target=output/'recovery_registration.json'
        if target.exists():
            registration=read(target)
            verify_registration(ancestor,parent,output,registration)
            return registration,entries
        qmap={q['question_id']:q for c in contexts for q in c['questions']}
        sources=[parent/'config.json',parent/'offline_inventory.json']
        for folder in (parent/'indices').glob('*'):
            if not folder.is_dir():
                continue
            if (folder/'complete.json').exists():
                receipt=read(folder/'complete.json')
                for name,field in (('chunks.json','chunks_sha256'),('vectors.npy','vectors_sha256')):
                    if file_sha(folder/name)!=receipt[field]:
                        raise RuntimeError('Index checkpoint hash mismatch')
            sources.extend(p for p in folder.iterdir() if p.name in {'chunks.json','vectors.npy','complete.json'})
        for folder in (parent/'questions').glob('*'):
            if folder.name not in qmap:
                raise RuntimeError('Question outside frozen dataset')
            for p in folder.iterdir():
                if p.name not in prior.CHECKPOINT_NAMES:
                    continue
                if folder.name in NEW_FAILURES and p.name in {'prediction.json','inference.json','result.json'}:
                    raise RuntimeError('New policy target has a valid output; do not overwrite it')
                value=read(p)
                if p.name in {'inference.json','result.json'} and value['question_id']!=folder.name:
                    raise RuntimeError('Question checkpoint identity mismatch')
                sources.append(p)
        inherited=len(list((parent/'questions').glob('*/prediction.json')))
        if inherited!=133 or old_registration['policy_zero_ids']!=[prior.POLICY_QID]:
            raise RuntimeError('Parent population changed; audit registration again')
        if shutil.disk_usage(output.anchor).free<sum(p.stat().st_size for p in sources)+15*1024**3:
            raise RuntimeError('Insufficient safe free space')
        copied={}
        for source in sources:
            relative=bound(parent,source).relative_to(parent).as_posix()
            copied[relative]=checked_copy(source,bound(output,output/relative))
        new_policy_hashes={}
        for qid in NEW_FAILURES:
            q=qmap[qid]
            failure=next(e for e in entries if e['outcome']['scope']==qid and e['outcome']['status']!='ok')
            context=read(output/'questions'/qid/'context.json')
            retrieval=read(output/'questions'/qid/'retrieval.json')
            policy=dict(question_id=qid,context_id=qid,category=q['category'],category_name=q['category_name'],
                question=q['raw_question'],reference=q['answer'],prediction=None,correct=0,judge=None,
                status='policy_zero',score_origin='user_explicit_two_question_policy_2026-11-14',
                policy_reason='User explicitly authorized this additional MAX_TOKENS=500 QA failure to count as incorrect.',
                model_judge_called=False,generation_valid=False,retry_authorized=False,
                original_failure=str(failure['folder']),original_response_sha256=failure['outcome']['response_sha256'],
                selected_ids=context['selected_ids'],context_chars=context['context_chars'],
                candidate_count=len(retrieval['candidate_ids']),
                timing=read(parent/'questions'/qid/'failure.json')['timing'],
                completed_utc=datetime.now(timezone.utc).isoformat())
            path=output/'questions'/qid/'result.json'
            if path.exists():
                old=read(path)
                if {k:v for k,v in old.items() if k!='completed_utc'}!={k:v for k,v in policy.items() if k!='completed_utc'}:
                    raise RuntimeError('Existing policy result differs')
            else:
                save(path,policy)
            new_policy_hashes[qid]=file_sha(path)
        ancestor_files=dict(old_registration['parent_files_sha256'])
        parent_files=set(sources)|{parent/'recovery_registration.json'}
        for sub in ('runtime','executions','execution_timing'):
            parent_files.update(p for p in (parent/sub).rglob('*.json') if p.is_file())
        parent_files.update((parent/'questions').glob('*/failure.json'))
        registration=dict(schema='hybrid_rag.two_more_policy_zeros.v1',ancestor=str(ancestor),parent=str(parent),output=str(output),
            entry=str(Path(__file__).resolve()),entry_sha256=file_sha(Path(__file__)),
            prior_recovery_entry=str(Path(prior.__file__)),prior_recovery_entry_sha256=file_sha(Path(prior.__file__)),
            ancestor_files_sha256=ancestor_files,
            parent_files_sha256={bound(parent,p).relative_to(parent).as_posix():file_sha(p) for p in sorted(parent_files)},
            copied_files_sha256=copied,new_policy_results_sha256=new_policy_hashes,
            inherited_predictions=133,existing_policy_zero_ids=[prior.POLICY_QID],
            new_policy_zero_ids=sorted(NEW_FAILURES),policy_zero_ids=sorted(POLICY_IDS),
            evaluation_population=500,generation_question_count=497,
            source_contract='Original frozen_prepare validates all 500 IDs and all original source/model/prompt hashes.',
            failure_policy='Only named three IDs have explicit zero policy; all new/unrelated failures stop.',
            transport_reuse='Only exact successful native responses; cumulative physical attempts across all roots.',
            cost_roots=[str(r/'runtime') for r in roots],stop_file=str(output/'STOP'),
            created_utc=datetime.now(timezone.utc).isoformat())
        save(target,registration)
        verify_registration(ancestor,parent,output,registration)
        return registration,entries


def recovery_transport(entries,output):
    base=transport.AuditTransport
    blocked={e['key'] for e in entries if e['outcome']['status']!='ok'}
    class RecoveryTransport(base):
        def __init__(self,runtime):
            super().__init__(runtime)
            for e in entries:
                attempts=self.attempts if e['kind']=='gemini' else self.qwen_attempts
                cache=self.raw_success if e['kind']=='gemini' else self.qwen_success
                key,out=e['key'],e['outcome']
                attempts[key]=attempts.get(key,0)+int(out['physical_calls'])
                if out['status']=='ok':
                    if key in cache and digest(cache[key])!=digest(e['raw']):
                        raise transport.ExecutionStop('Conflicting successful response for identical request')
                    cache[key]=e['raw']  # immutable parent object; request() returns its own deep copy
        def guard(self):
            if (output/'STOP').exists() or str(getattr(self.local,'scope','shared')).split('/')[0] in POLICY_IDS:
                self.stop.set()
                raise transport.ExecutionStop('Explicit STOP or policy-excluded question')
        def request(self,url,payload):
            self.guard()
            model,_,_=self._endpoint(url,payload)
            if digest([model,payload]) in blocked:
                self.stop.set()
                raise transport.ExecutionStop('Authorized failed exact request must not be replayed')
            return super().request(url,payload)
        def qwen_post(self,endpoint,payload):
            self.guard()
            return super().qwen_post(endpoint,payload)
    return RecoveryTransport


def combined_summary(roots,base):
    calls=[]
    seen=set()
    for root in roots:
        for dirname in ('calls','qwen_calls'):
            for p in (root/'runtime'/dirname).glob('*/outcome.json'):
                if str(p.resolve()) not in seen:
                    seen.add(str(p.resolve()))
                    calls.append(read(p))
    stats=defaultdict(lambda:dict(physical_calls=0,known_input_tokens=0,known_output_tokens=0,
        thought_tokens=0,cached_input_tokens=0,usage_missing_calls=0,service_seconds=0.0))
    generation=standard=qwen=0.0
    for c in calls:
        s=stats[c['stage']]
        s['physical_calls']+=c.get('physical_calls',0)
        s['service_seconds']+=c.get('service_seconds',0)
        u=c.get('usage')
        if not u:
            s['usage_missing_calls']+=c.get('physical_calls',0)
            continue
        tokens=int(u.get('promptTokenCount',u.get('input_tokens',u.get('total_tokens',0))) or 0)
        out=int(u.get('candidatesTokenCount',u.get('output_tokens',0)) or 0)
        thought=int(u.get('thoughtsTokenCount',0) or 0)
        cached=int(u.get('cachedContentTokenCount',0) or 0)
        s['known_input_tokens']+=tokens
        s['known_output_tokens']+=out
        s['thought_tokens']+=thought
        s['cached_input_tokens']+=cached
        if c['stage']!='judge' and c['model']=='gemini-2.5-flash':
            generation+=((tokens-cached)*.30+cached*.03+(out+thought)*2.50)/1e6
            standard+=(tokens*.30+(out+thought)*2.50)/1e6
        elif c['stage']!='judge' and c['model']=='qwen3-rerank':
            qwen+=tokens*.50/1e6
    for s in stats.values():
        s['input_tokens']=None if s['usage_missing_calls'] else s['known_input_tokens']
        s['output_tokens']=None if s['usage_missing_calls'] else s['known_output_tokens']
    result={**base,'report_scope':'All r1+r2+r3 physical events; absolute receipt identity deduplication only.',
        'parent_outputs':[str(r) for r in roots[:-1]],'policy_zero_ids':sorted(POLICY_IDS),'policy_zero_count':3,
        'physical_stage_usage':dict(stats),
        'physical_inference_cost':dict(gemini_generation_usd=generation,
            gemini_generation_standard_input_usd=standard,qwen_rerank_cny=qwen,
            embedding_usd=None,embedding_token_cost_not_estimated=True,three_truncated_failures_included=True,
            judge_cost_excluded=True,judge_usage_reported_separately=True,pricing_convention_date='2026-11-11')}
    save(roots[-1]/'recovery_summaries'/(str(time.time_ns())+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--allow-model-calls',action='store_true')
    parser.add_argument('--ancestor',type=Path,default=ANCESTOR)
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    ancestor,parent,output=args.ancestor.resolve(),args.parent.resolve(),args.output.resolve()
    registration,entries=prepare(ancestor,parent,output)
    if not args.run:
        import json
        print(json.dumps(dict(status='offline_prepared',output=str(output),inherited_predictions=133,
            policy_zero_ids=sorted(POLICY_IDS),model_calls=0)))
        return
    if not args.allow_model_calls or (output/'STOP').exists():
        raise RuntimeError('Live recovery requires --allow-model-calls and no STOP file')
    base_prepare,base_summary=original.frozen_prepare,original.summarize
    def registered_prepare(runargs,adapter):
        contexts,config=base_prepare(runargs,adapter)
        verify_registration(ancestor,parent,output,registration)
        return [c for c in contexts if c['context_id'] not in POLICY_IDS],config
    original.frozen_prepare=registered_prepare
    original.summarize=lambda out,config,wall,phase:combined_summary(
        (ancestor,parent,out),base_summary(out,config,wall,phase))
    transport.AuditTransport=recovery_transport(entries,output)
    sys.argv=[str(Path(original.__file__)),'--dataset','lme','--output',str(output),
              '--phase','full','--allow-model-calls','--workers','4']
    original.main()


if __name__=='__main__':
    main()

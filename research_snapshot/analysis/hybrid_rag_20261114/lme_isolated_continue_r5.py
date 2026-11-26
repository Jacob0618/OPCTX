"""Registered r5 adapter: native MAX_TOKENS is authoritative, not usage==500.

Reuses the frozen r4 scheduler, transport isolation, QA/Judge and all original
sources. Only the local-truncation classifier and recovery registration change.
No output cap, answer, model, prompt, retrieval setting or score is changed.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from filelock import FileLock
from analysis.hybrid_rag_20261114 import lme_isolated_continue as frozen

PARENTS=(*frozen.PARENTS,frozen.OUTPUT)
OUTPUT=Path('D:/hybrid_rag_20261114/lme_r5')
read,save,digest,file_sha,bound=frozen.read,frozen.save,frozen.digest,frozen.file_sha,frozen.bound
_PARENT_VALIDATE=frozen.validate_registration
_PARENT_SAVE_MISSING=frozen.save_missing


def is_qa500(req,out,raw,qid):
    """A 500-token request may natively finish MAX_TOKENS with usage <500."""
    candidates=raw.get('candidates') or []
    usage=raw.get('usageMetadata') or {}
    if not isinstance(usage,dict):return False
    tokens=usage.get('candidatesTokenCount')
    valid_count=tokens is None or (type(tokens) is int and tokens>=0)
    return (out.get('scope')==qid and out.get('stage')=='qa'
        and req.get('kind')=='generation' and req.get('model')=='gemini-2.5-flash'
        and raw.get('modelVersion')=='gemini-2.5-flash' and not raw.get('error')
        and out.get('physical_calls')==1 and out.get('http_status')==200
        and out.get('model')=='gemini-2.5-flash' and out.get('model_version')=='gemini-2.5-flash'
        and out.get('finish_reason')=='MAX_TOKENS'
        and req.get('request',{}).get('generationConfig',{}).get('maxOutputTokens')==500
        and valid_count and len(candidates)==1 and candidates[0].get('finishReason')=='MAX_TOKENS'
        and not raw.get('promptFeedback',{}).get('blockReason')
        and not any(v.get('blocked') for v in raw.get('promptFeedback',{}).get('safetyRatings',[]))
        and not any(v.get('blocked') for v in candidates[0].get('safetyRatings',[])))


def save_missing(output,qid,evidence):
    """Usage accounting anomalies cannot turn a native truncation into a score."""
    enriched=dict(evidence)
    tokens=(evidence.get('usage') or {}).get('candidatesTokenCount')
    if type(tokens) is int and tokens>500:
        enriched['usage_anomaly']=dict(reason='reported_output_tokens_exceed_requested_cap',
            requested_max_output_tokens=500,reported_output_tokens=tokens)
    _PARENT_SAVE_MISSING(output,qid,enriched)


def current_parent_ledger(parent):
    entries=[]
    validator=object.__new__(frozen.transport.original.AuditTransport)
    for reqpath in sorted((parent/'runtime/questions').glob('*/*/*/request.json')):
        folder=reqpath.parent
        req,out,raw=read(reqpath),read(folder/'outcome.json'),read(folder/'response.json')
        kind='qwen' if folder.parent.name=='qwen_calls' else 'gemini'
        model=req['model'] if kind=='gemini' else 'qwen3-rerank'
        key=digest([model,req['request']])
        if key!=out['request_key'] or digest(raw)!=out['response_sha256']:
            raise RuntimeError('r4 native request/response binding mismatch')
        if out['status']=='ok':
            if kind=='gemini':
                validator._validate_response(model,req['kind'],req['request'],raw)
            else:
                rows=raw.get('output',{}).get('results',[])
                n=len(req['request']['input']['documents']);ids=[v.get('index') for v in rows]
                if (req['endpoint']!=frozen.transport.QWEN_URL or req['request']['model']!='qwen3-rerank'
                        or len(rows)!=n or any(type(i) is not int for i in ids) or set(ids)!=set(range(n))
                        or any(not math.isfinite(float(v['relevance_score'])) for v in rows)):
                    raise RuntimeError('Invalid r4 Qwen response')
        elif kind!='gemini' or not is_qa500(req,out,raw,out['scope']):
            raise RuntimeError('Non-QA500 parent fault remains a global blocker: '+str(folder))
        entries.append(dict(kind=kind,key=key,request=req,outcome=out,raw=raw,folder=folder))
    failures=[e for e in entries if e['outcome']['status']!='ok']
    if len(failures)!=11 or not any(e['outcome']['scope']=='0a34ad58' and e['raw']['usageMetadata']['candidatesTokenCount']==498 for e in failures):
        raise RuntimeError('r4 failure snapshot changed; review before continuing')
    return entries


def validate_registration(parents,output,registration):
    if (registration['entry_sha256']!=file_sha(Path(__file__)) or registration['parents']!=list(map(str,parents))
            or registration['output']!=str(output) or registration['frozen_r4_entry_sha256']!=file_sha(Path(frozen.__file__))):
        raise RuntimeError('Recovery entry or roots changed')
    _PARENT_VALIDATE(parents[:-1],parents[-1],read(parents[-1]/'recovery_registration.json'))
    for relative,expected in registration['parent_files_sha256'].items():
        if file_sha(bound(parents[-1],parents[-1]/relative))!=expected:
            raise RuntimeError('Frozen r4 artifact changed: '+relative)
    for relative,expected in registration['copied_files_sha256'].items():
        if file_sha(bound(output,output/relative))!=expected:
            raise RuntimeError('Imported checkpoint changed: '+relative)
    for qid,expected in registration['known_missing_sha256'].items():
        if file_sha(output/'questions'/qid/'missing.json')!=expected:
            raise RuntimeError('Missing record changed')


def prepare(parents,output):
    parents=tuple(p.resolve() for p in parents);output=output.resolve()
    roots=(*parents,output)
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):
        raise RuntimeError('Five separate sibling roots required')
    output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for root in roots:stack.enter_context(FileLock(str(root/'runner.lock'),timeout=0))
        _PARENT_VALIDATE(parents[:-1],parents[-1],read(parents[-1]/'recovery_registration.json'))
        adapter=frozen.stages.make_stages(lambda *_a,**_k:(_ for _ in ()).throw(RuntimeError('Offline only')))
        contexts,config=frozen.original.frozen_prepare(SimpleNamespace(output=parents[-1],dataset='lme',workers=4),adapter)
        entries=(frozen.r2.ledger(parents[0])+frozen.r3.parent_ledger(parents[1])
                 +frozen.last_parent_ledger(parents[2])+current_parent_ledger(parents[3]))
        failures={e['outcome']['scope']:e for e in entries if e['outcome']['status']!='ok'
                  and e['outcome']['scope'] not in frozen.POLICY_IDS}
        if len(failures)!=13:
            raise RuntimeError('Expected 13 known missing questions, separate from 3 policy zeros')
        target=output/'recovery_registration.json'
        if target.exists():
            registration=read(target);validate_registration(parents,output,registration)
            return registration,entries,contexts,config
        parent=parents[-1]
        if len(list((parent/'questions').glob('*/prediction.json')))!=147:
            raise RuntimeError('Expected 147 immutable successful predictions')
        qmap={q['question_id']:q for c in contexts for q in c['questions']}
        sources=[parent/'config.json',parent/'offline_inventory.json']
        for folder in (parent/'indices').glob('*'):
            if not folder.is_dir():continue
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json')
                for name,key in (('chunks.json','chunks_sha256'),('vectors.npy','vectors_sha256')):
                    if file_sha(folder/name)!=done[key]:raise RuntimeError('Parent index checksum mismatch')
            sources.extend(p for p in folder.iterdir() if p.name in {'chunks.json','vectors.npy','complete.json'})
        for folder in (parent/'questions').glob('*'):
            if folder.name not in qmap:raise RuntimeError('Unknown parent question')
            for p in folder.iterdir():
                if p.name not in frozen.r2.CHECKPOINT_NAMES|{'missing.json'}:continue
                value=read(p)
                if p.name in {'inference.json','result.json','missing.json'} and value['question_id']!=folder.name:
                    raise RuntimeError('Question/checkpoint mismatch')
                if folder.name in failures and p.name in {'prediction.json','inference.json','result.json'}:
                    raise RuntimeError('Known missing has a valid answer; do not overwrite')
                sources.append(p)
        if shutil.disk_usage(output.anchor).free<sum(p.stat().st_size for p in sources)+15*1024**3:
            raise RuntimeError('Insufficient safe free space')
        copied={}
        for source in sources:
            relative=bound(parent,source).relative_to(parent).as_posix()
            copied[relative]=frozen.r2.checked_copy(source,bound(output,output/relative))
        missing_hashes={}
        for qid,e in failures.items():
            path=output/'questions'/qid/'missing.json'
            if not path.exists():
                save(path,dict(question_id=qid,status='missing',correct=None,judge=None,model_judge_called=False,
                    reason='Native MAX_TOKENS at unchanged request cap500; observed usage498 does not make this a different failure.',
                    original_failure=str(e['folder']),original_response_sha256=e['outcome']['response_sha256'],
                    observed_output_tokens=e['raw'].get('usageMetadata',{}).get('candidatesTokenCount')))
            value=read(path)
            if value.get('status')!='missing' or value.get('correct') is not None or value.get('judge') is not None:
                raise RuntimeError('Missing marker was converted into a score')
            missing_hashes[qid]=file_sha(path)
        preserved=set(sources)|{parent/'recovery_registration.json'}
        for sub in ('runtime','executions','execution_timing'):
            preserved.update(p for p in (parent/sub).rglob('*.json') if p.is_file())
        preserved.update((parent/'questions').glob('*/failure.json'))
        registration=dict(schema='hybrid_rag.isolated_native_maxtokens_r5.v1',entry=str(Path(__file__).resolve()),
            entry_sha256=file_sha(Path(__file__)),frozen_r4_entry=str(Path(frozen.__file__)),
            frozen_r4_entry_sha256=file_sha(Path(frozen.__file__)),parents=list(map(str,parents)),output=str(output),
            parent_files_sha256={bound(parent,p).relative_to(parent).as_posix():file_sha(p) for p in sorted(preserved)},
            copied_files_sha256=copied,known_missing_sha256=missing_hashes,inherited_predictions=147,
            policy_zero_ids=sorted(frozen.POLICY_IDS),known_missing_ids=sorted(failures),evaluation_population=500,
            runtime_layout='per_question_isolated',workers=4,qa_max_tokens=500,
            classifier_change='Request cap500 + native MAX_TOKENS; usage count absent or nonnegative integer. Above-cap usage is an accounting anomaly, not a retry/score.',
            reused_scheduler='Frozen r4.run_live; all valid inherited/new predictions enter real Judge phase.',
            in_memory_injections=['is_qa500','save_missing','validate_registration','KNOWN_MISSING'],
            source_contract='Original frozen_prepare genuinely verifies all500 and original four-source hashes.',
            policy_change=False,retry_failed_requests=False,new_missing_scored_zero=False,
            cost_roots=[str(p/'runtime') for p in roots],stop_file=str(output/'STOP'),
            created_utc=datetime.now(timezone.utc).isoformat())
        save(target,registration);validate_registration(parents,output,registration)
        return registration,entries,contexts,config


def install_registered_adapter(registration):
    """Explicit process-local adaptation; old source files/hash claims untouched."""
    frozen.is_qa500=is_qa500
    frozen.save_missing=save_missing
    frozen.validate_registration=validate_registration
    frozen.KNOWN_MISSING={qid:'registered_missing_no_retry' for qid in registration['known_missing_ids']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--allow-model-calls',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();parents=tuple(p.resolve() for p in PARENTS);output=args.output.resolve()
    registration,entries,contexts,config=prepare(parents,output)
    if not args.run:
        print(json.dumps(dict(status='offline_prepared',inherited_predictions=147,known_missing=13,
            policy_zero_count=3,runtime_layout='per_question_isolated',model_calls=0)));return
    if not args.allow_model_calls:raise RuntimeError('Live continuation requires --allow-model-calls')
    install_registered_adapter(registration)
    frozen.run_live(parents,output,registration,entries,contexts,config)


if __name__=='__main__':main()

"""Independent, one-question policy recovery; frozen RAG sources stay unchanged.

Offline: --prepare. Live (root only): --run --allow-model-calls.
Only 29f2956b_abs is policy-scored zero, without a fabricated Judge response.
All other failures still stop normally. Create OUTPUT/STOP to stop new calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from filelock import FileLock
from analysis.hybrid_rag_20261114 import run as original
from analysis.hybrid_rag_20261114 import stages, transport

POLICY_QID = '29f2956b_abs'
FAILED_CALL = '000427_qa'
PARENT = Path('D:/hybrid_rag_20261114/lme_r1')
OUTPUT = Path('D:/hybrid_rag_20261114/lme_r2')
CHECKPOINT_NAMES = {'retrieval.json', 'selected.json', 'context.json',
                    'prediction.json', 'inference.json', 'result.json'}
read, save, digest, file_sha = original.read, original.save, original.digest, original.file_sha


def bound(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise RuntimeError('File escapes its explicit experiment root')
    return path


def checked_copy(source, target):
    expected = file_sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha(target) != expected:
            raise RuntimeError('Existing recovery copy differs: ' + str(target))
    else:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if file_sha(target) != expected:
            raise RuntimeError('Recovery copy hash verification failed')
    return expected


def ledger(parent):
    """Validate native parent receipts, allowing exactly the authorized failure."""
    entries = []
    validator = object.__new__(transport.original.AuditTransport)
    failed = []
    for kind, dirname in (('gemini', 'calls'), ('qwen', 'qwen_calls')):
        for folder in sorted((parent / 'runtime' / dirname).glob('*')):
            if not folder.is_dir():
                continue
            req_path, out_path = folder / 'request.json', folder / 'outcome.json'
            if not req_path.exists() or not out_path.exists():
                raise RuntimeError('Unresolved parent physical request: ' + str(folder))
            req, out = read(req_path), read(out_path)
            model = req['model'] if kind == 'gemini' else 'qwen3-rerank'
            key = digest([model, req['request']])
            if key != out['request_key']:
                raise RuntimeError('Parent request/outcome binding mismatch')
            raw = read(folder / 'response.json')
            if digest(raw) != out['response_sha256']:
                raise RuntimeError('Parent native response hash mismatch')
            if out['status'] == 'ok':
                if kind == 'gemini':
                    validator._validate_response(model, req['kind'], req['request'], raw)
                else:
                    if req['endpoint'] != transport.QWEN_URL or req['request']['model'] != 'qwen3-rerank':
                        raise RuntimeError('Parent Qwen endpoint/model mismatch')
                    n = len(req['request']['input']['documents'])
                    results = raw.get('output', {}).get('results', [])
                    indices = [r.get('index') for r in results]
                    if (len(results) != n or any(type(i) is not int for i in indices)
                            or set(indices) != set(range(n))
                            or any(not math.isfinite(float(r['relevance_score'])) for r in results)):
                        raise RuntimeError('Parent Qwen aligned scores invalid')
            else:
                if not (kind == 'gemini' and folder.name == FAILED_CALL
                        and out['scope'] == POLICY_QID and out['stage'] == 'qa'
                        and out['model'] == 'gemini-2.5-flash'
                        and out['model_version'] == 'gemini-2.5-flash'
                        and out['finish_reason'] == 'MAX_TOKENS'
                        and req['request']['generationConfig']['maxOutputTokens'] == 500
                        and raw['candidates'][0]['finishReason'] == 'MAX_TOKENS'
                        and out['physical_calls'] == 1):
                    raise RuntimeError('Unapproved parent failure: ' + str(folder))
                failed.append(key)
            entries.append(dict(kind=kind, folder=folder, request=req, outcome=out, raw=raw, key=key))
    if len(failed) != 1:
        raise RuntimeError('Expected exactly one explicitly authorized physical failure')
    # Distinguish cancellation from another failed QA: none was sent for this ID.
    if any(e['outcome']['scope'] == '6d550036' and e['outcome']['stage'] == 'qa' for e in entries):
        raise RuntimeError('6d550036 now has a physical QA request; audit again')
    partial = [e for e in entries if e['outcome']['scope'] == 'gpt4_59c863d7/index']
    if not partial or any(e['outcome']['status'] != 'ok' for e in partial):
        raise RuntimeError('Partial index history is not exclusively successful')
    return entries


def verify_manifest(parent, output, registration):
    if registration['entry_sha256'] != file_sha(Path(__file__)):
        raise RuntimeError('Recovery entry changed after registration')
    if registration['parent'] != str(parent) or registration['output'] != str(output):
        raise RuntimeError('Recovery roots differ from registration')
    for relative, expected in registration['parent_files_sha256'].items():
        if file_sha(bound(parent, parent / relative)) != expected:
            raise RuntimeError('Frozen parent artifact changed: ' + relative)
    for relative, expected in registration['copied_files_sha256'].items():
        if file_sha(bound(output, output / relative)) != expected:
            raise RuntimeError('Imported checkpoint changed: ' + relative)
    if file_sha(output / 'questions' / POLICY_QID / 'result.json') != registration['policy_result_sha256']:
        raise RuntimeError('One-question policy result changed')


def prepare(parent, output):
    parent, output = parent.resolve(), output.resolve()
    if parent == output or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise RuntimeError('Recovery must use a separate sibling experiment root')
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(parent / 'runner.lock'), timeout=0), FileLock(str(output / 'runner.lock'), timeout=0):
        offline = stages.make_stages(lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError('Offline only')))
        contexts, config = original.frozen_prepare(SimpleNamespace(output=parent, dataset='lme', workers=4), offline)
        entries = ledger(parent)
        registration_path = output / 'recovery_registration.json'
        if registration_path.exists():
            registration = read(registration_path)
            verify_manifest(parent, output, registration)
            return registration, contexts, config, entries
        sources = [parent / 'config.json', parent / 'offline_inventory.json']
        for folder in sorted((parent / 'indices').glob('*')):
            if not folder.is_dir():
                continue
            done = folder / 'complete.json'
            if done.exists():
                receipt = read(done)
                for name, field in (('chunks.json', 'chunks_sha256'), ('vectors.npy', 'vectors_sha256')):
                    if file_sha(folder / name) != receipt[field]:
                        raise RuntimeError('Parent index checkpoint hash mismatch')
            sources.extend(p for p in folder.iterdir() if p.name in {'chunks.json', 'vectors.npy', 'complete.json'})
        qmap = {q['question_id']: q for c in contexts for q in c['questions']}
        for folder in sorted((parent / 'questions').glob('*')):
            if folder.name not in qmap:
                raise RuntimeError('Parent question outside frozen dataset')
            for p in folder.iterdir():
                if p.name not in CHECKPOINT_NAMES:
                    continue
                if folder.name == POLICY_QID and p.name in {'prediction.json', 'inference.json', 'result.json'}:
                    raise RuntimeError('Policy target unexpectedly has a successful output')
                value = read(p)
                if p.name in {'inference.json', 'result.json'} and value['question_id'] != folder.name:
                    raise RuntimeError('Parent question/checkpoint mismatch')
                sources.append(p)
        required = sum(p.stat().st_size for p in sources)
        if shutil.disk_usage(output.anchor).free < required + 15 * 1024**3:
            raise RuntimeError('Insufficient safe free space for recovery copies')
        copied = {}
        for source in sources:
            source = bound(parent, source)
            relative = source.relative_to(parent).as_posix()
            copied[relative] = checked_copy(source, bound(output, output / relative))
        preserved = set(sources)
        for sub in ('runtime', 'executions', 'execution_timing'):
            preserved.update(p for p in (parent / sub).rglob('*.json') if p.is_file())
        preserved.update((parent / 'questions').glob('*/failure.json'))
        parent_hashes = {bound(parent, p).relative_to(parent).as_posix(): file_sha(p) for p in sorted(preserved)}
        target = qmap[POLICY_QID]
        failed = next(e for e in entries if e['outcome']['status'] != 'ok')
        context = read(output / 'questions' / POLICY_QID / 'context.json')
        policy = dict(question_id=POLICY_QID, context_id=POLICY_QID,
            category=target['category'], category_name=target['category_name'],
            question=target['raw_question'], reference=target['answer'], prediction=None,
            correct=0, judge=None, status='policy_zero',
            score_origin='user_explicit_single_question_policy_2026-11-14',
            policy_reason='Original QA stopped at MAX_TOKENS=500; user requested this question count as incorrect.',
            model_judge_called=False, generation_valid=False, retry_authorized=False,
            original_failure=str(failed['folder']), original_response_sha256=failed['outcome']['response_sha256'],
            selected_ids=context['selected_ids'], context_chars=context['context_chars'],
            candidate_count=240, timing=read(parent / 'questions' / POLICY_QID / 'failure.json')['timing'],
            completed_utc=datetime.now(timezone.utc).isoformat())
        policy_path = output / 'questions' / POLICY_QID / 'result.json'
        if policy_path.exists():
            old = read(policy_path)
            if {k:v for k,v in old.items() if k != 'completed_utc'} != {k:v for k,v in policy.items() if k != 'completed_utc'}:
                raise RuntimeError('Existing policy result differs')
        else:
            save(policy_path, policy)
        registration = dict(schema='hybrid_rag.single_policy_recovery.v1', parent=str(parent), output=str(output),
            entry=str(Path(__file__).resolve()), entry_sha256=file_sha(Path(__file__)),
            original_config_sha256=file_sha(parent / 'config.json'), policy_zero_ids=[POLICY_QID],
            generation_question_count=499, evaluation_population=500,
            inherited_predictions=len(list((parent/'questions').glob('*/prediction.json'))),
            inherited_results=len(list((parent/'questions').glob('*/result.json'))),
            parent_files_sha256=parent_hashes, copied_files_sha256=copied,
            policy_result_sha256=file_sha(policy_path),
            excluded_prior_failure_markers=['29f2956b_abs', '6d550036'],
            cancellation_evidence={'6d550036':'No physical QA request exists; continue its first QA.',
                'gpt4_59c863d7':'Only successful partial embedding batch; reuse exact response.'},
            native_success_reuse='Exact request SHA; no failed response is cached as success.',
            attempt_accounting='Parent and recovery physical attempts cumulative per exact request.',
            cost_roots=[str(parent/'runtime'), str(output/'runtime')],
            stop_file=str(output/'STOP'),
            changes=['Separate recovery entry', 'Exactly one explicit external zero',
                     'Import verified successful checkpoints/responses', 'Stop-file guard'],
            unchanged='Frozen four source files, models, prompts, limits, retrieval, and all other failure policy.',
            created_utc=datetime.now(timezone.utc).isoformat())
        save(registration_path, registration)
        verify_manifest(parent, output, registration)
        return registration, contexts, config, entries


def recovery_transport(entries, output):
    base = transport.AuditTransport
    blocked_keys = {e['key'] for e in entries if e['outcome']['status'] != 'ok'}

    class RecoveryTransport(base):
        def __init__(self, runtime):
            super().__init__(runtime)
            for entry in entries:
                key, out = entry['key'], entry['outcome']
                attempts = self.attempts if entry['kind']=='gemini' else self.qwen_attempts
                cache = self.raw_success if entry['kind']=='gemini' else self.qwen_success
                attempts[key] = attempts.get(key, 0) + int(out['physical_calls'])
                if out['status']=='ok':
                    if key in cache and digest(cache[key]) != digest(entry['raw']):
                        raise transport.ExecutionStop('Parent/new response collision')
                    cache[key] = copy.deepcopy(entry['raw'])

        def guard(self):
            scope = str(getattr(self.local, 'scope', 'shared'))
            if (output/'STOP').exists() or scope.split('/')[0] == POLICY_QID:
                self.stop.set()
                raise transport.ExecutionStop('Explicit recovery stop or excluded policy question')

        def request(self, url, payload):
            self.guard()
            model, _, _ = self._endpoint(url, payload)
            if digest([model, payload]) in blocked_keys:
                self.stop.set()
                raise transport.ExecutionStop('Authorized-zero failed request must never be replayed')
            return super().request(url, payload)

        def qwen_post(self, endpoint, payload):
            self.guard()
            return super().qwen_post(endpoint, payload)
    return RecoveryTransport


def combined_summary(parent, output, base):
    """Physical events deduplicated by absolute receipt path, not request hash."""
    calls, seen = [], set()
    for root in (parent, output):
        for dirname in ('calls', 'qwen_calls'):
            for path in (root/'runtime'/dirname).glob('*/outcome.json'):
                identity = str(path.resolve())
                if identity in seen:
                    continue
                seen.add(identity)
                calls.append(read(path))
    stats = defaultdict(lambda:dict(physical_calls=0, known_input_tokens=0, known_output_tokens=0,
        thought_tokens=0,cached_input_tokens=0,usage_missing_calls=0,service_seconds=0.0))
    generation, standard, qwen = 0.0, 0.0, 0.0
    for call in calls:
        stage, usage = call['stage'], call.get('usage')
        s=stats[stage]
        s['physical_calls']+=call.get('physical_calls',0)
        s['service_seconds']+=call.get('service_seconds',0)
        if not usage:
            s['usage_missing_calls']+=call.get('physical_calls',0)
            continue
        tokens=int(usage.get('promptTokenCount',usage.get('input_tokens',usage.get('total_tokens',0))) or 0)
        out=int(usage.get('candidatesTokenCount',usage.get('output_tokens',0)) or 0)
        thought=int(usage.get('thoughtsTokenCount',0) or 0)
        cached=int(usage.get('cachedContentTokenCount',0) or 0)
        s['known_input_tokens']+=tokens
        s['known_output_tokens']+=out
        s['thought_tokens']+=thought
        s['cached_input_tokens']+=cached
        if stage!='judge' and call['model']=='gemini-2.5-flash':
            generation+=((tokens-cached)*.30+cached*.03+(out+thought)*2.50)/1e6
            standard+=(tokens*.30+(out+thought)*2.50)/1e6
        elif stage!='judge' and call['model']=='qwen3-rerank':
            qwen+=tokens*.50/1e6
    for s in stats.values():
        s['input_tokens']=None if s['usage_missing_calls'] else s['known_input_tokens']
        s['output_tokens']=None if s['usage_missing_calls'] else s['known_output_tokens']
    result={**base, 'report_scope':'Combined parent plus recovery physical events; imported results counted once.',
        'parent_output':str(parent),'policy_zero_count':1,'policy_zero_ids':[POLICY_QID],
        'physical_stage_usage':dict(stats),
        'physical_inference_cost':dict(gemini_generation_usd=generation,
            gemini_generation_standard_input_usd=standard,qwen_rerank_cny=qwen,
            embedding_usd=None,embedding_token_cost_not_estimated=True,
            truncated_failure_included=True,judge_cost_excluded=True,
            judge_usage_reported_separately=True,pricing_convention_date='2026-11-11')}
    save(output/'recovery_summaries'/(str(time.time_ns())+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--allow-model-calls',action='store_true')
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    parent,output=args.parent.resolve(),args.output.resolve()
    registration,_,_,entries=prepare(parent,output)
    if not args.run:
        print(json.dumps(dict(status='offline_prepared',output=str(output),
            inherited_predictions=registration['inherited_predictions'],policy_zero_ids=[POLICY_QID],model_calls=0)))
        return
    if not args.allow_model_calls:
        raise RuntimeError('Live recovery requires --allow-model-calls')
    if (output/'STOP').exists():
        raise RuntimeError('STOP file exists; no model calls')
    base_prepare,base_summarize=original.frozen_prepare,original.summarize
    def registered_prepare(runargs,adapter):
        contexts,config=base_prepare(runargs,adapter)  # validates all 500 and all original hashes
        verify_manifest(parent,output,registration)
        return [c for c in contexts if c['context_id']!=POLICY_QID],config
    original.frozen_prepare=registered_prepare
    original.summarize=lambda out,config,wall,phase:combined_summary(
        parent,out,base_summarize(out,config,wall,phase))
    transport.AuditTransport=recovery_transport(entries,output)
    sys.argv=[str(Path(original.__file__)),'--dataset','lme','--output',str(output),
              '--phase','full','--allow-model-calls','--workers','4']
    original.main()


if __name__=='__main__':
    main()

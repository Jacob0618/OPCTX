"""Cold-memory LME clean-B3 cost run, delegated to the frozen original pipeline.

Only observability/transport safety and exclusion of evaluation Judge differ.
Writer/Online/embedding/retrieval/Qwen30/Planner/materializer/QA stay original.
No previous experiment cache is imported. Default is offline preparation only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
DEFAULT_SELECTION = Path('D:/opctx_read_cost_20261111/lme10_r1/manifest.json')
_local = threading.local()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Immutable artifact differs: ' + str(path))
        return value
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return value


def pack(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    with path.open('xb') as handle:
        handle.write(gzip.compress(body, mtime=0))
        handle.flush()
        os.fsync(handle.fileno())


def unpack(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def disk_guard(root, minimum=15):
    if shutil.disk_usage(Path(root).anchor).free < minimum * 1024**3:
        raise RuntimeError('Disk below minimum free GiB; no new provider request')


def prepare(args):
    from creamem3 import run_lme_clean100_v1 as clean
    adapter = clean.configure('b3_scaled', freeze=False)
    core, legacy = clean.core, clean.core.legacy
    contract = adapter.assert_prompt_contract()
    data, _, _ = clean.selection.load_inputs()
    prior = read(args.selection)
    ids = [r['question_id'] for r in prior['questions']]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise RuntimeError('Expected frozen prior ten unique IDs')
    lookup = {q['question_id']: (i, q) for i, q in enumerate(data)}
    questions, jobs = [], []
    for position, qid in enumerate(ids):
        index, item = lookup[qid]
        specs, _ = legacy._ordered_session_specs(item)
        groups, audit = adapter.audit_and_group(specs)
        if not groups:
            raise RuntimeError('No nonempty history for fixed question')
        for group in groups:
            adapter.render_write_prompt(contract, adapter.render_lme_frontier(group), unit_count=len(group['members']))
        row = dict(position=position, question_id=qid, question_type=item['question_type'],
            source_index=index, record_sha256=core._canonical(item),
            physical_sessions=len(specs), nonempty_sessions=sum(bool(s['turns']) for s in specs),
            writer_batches=len(groups), expected_online_calls=len(groups)-1,
            input_audit_sha256=core._canonical(audit), frontier_plan_sha256=core._canonical(groups))
        questions.append(row)
        jobs.append((item, row))
    source_paths = [Path(__file__), Path(clean.__file__), Path(core.__file__), Path(legacy.__file__),
        ROOT / 'creamem3/lme_clean_writer_v1.py', ROOT / 'creamem3/lme_opctx_v2_b3_scaled_facts.py',
        ROOT / 'creamem3/lme_opctx_v1.py', ROOT / 'creamem3/online_memory_v4.py',
        ROOT / 'creamem3/strict_semantic_planner_v8.py', ROOT / 'creamem3/online_operation_context_v8.py',
        legacy.LME_DATASET, args.selection]
    manifest = dict(schema='opctx.lme.e2e_cost.cold_clean_b3.v1', questions=questions,
        selection='Same fixed ten question IDs as prior read-cost replay; independent of scores',
        source_hashes={str(p): sha_file(p) for p in source_paths},
        protocol=dict(model='gemini-2.5-flash', writer_batch_sessions=3,
            writer_fact_policy='soft10-70 per effective session; original prompt',
            embedding='gemini-embedding-001 / 768 L2', retrieval='20 summary +40 dense +40 BM25; cap240; Qwen3 top30',
            mathematical_selector=False, judge=False, fresh_memory=True,
            generation_safety_limit=8, qwen_safety_limit=4, writer_workers_per_question=1,
            evaluation_exclusion='Original inference code unchanged; Judge function returns None/not_run/no calls',
            transport='Original native request bodies; verified HTTP; physical intent/response/outcome per POST; no secret URL query stored'))
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / 'manifest.json', manifest)
    core.MANIFEST = args.output / 'manifest.json'
    core.MANIFEST_SHA256 = sha_file(core.MANIFEST)
    config = clean._run_config(contract)
    config['campaign'] = dict(questions=10, fixed_question_ids=ids, cost_only=True,
        source_campaign='lme_clean100_b3_scaled_d768_20260905_r1', judge_enabled=False)
    config['cost_probe'] = manifest['protocol']
    config_sha = core._canonical(config)
    save(args.output / 'run_config.json', dict(config=config, config_sha256=config_sha))
    def skip_judge(**kwargs):
        return None, 'NOT_RUN_COST_ONLY', []
    core._judge = skip_judge
    # Native materializer is deterministic; measure it separately without
    # changing its input/output or the original combined Reader wall clock.
    original_materialize = adapter.materialize_operation_context
    def measured_materialize(*a, **kw):
        started = time.perf_counter()
        result = original_materialize(*a, **kw)
        _local.materializer_seconds = time.perf_counter() - started
        return result
    adapter.materialize_operation_context = measured_materialize
    totals = {field: sum(q[field] for q in questions) for field in
        ('physical_sessions', 'nonempty_sessions', 'writer_batches', 'expected_online_calls')}
    full_counts = Counter()
    for item in data:
        specs, _ = legacy._ordered_session_specs(item)
        groups, _ = adapter.audit_and_group(specs)
        full_counts.update(questions=1, physical_sessions=len(specs),
            nonempty_sessions=sum(bool(s['turns']) for s in specs),
            writer_batches=len(groups), expected_online_calls=max(0,len(groups)-1))
    save(args.output / 'preflight.json', dict(status='PASS', api_calls=0, totals=totals,
        full500_clean_b3_counts=dict(full_counts),
        full500_note='Read-only workload estimate under one clean-B3 policy; not a claim about the mixed historical500 configurations.',
        configured_questions=len(jobs), original_contract=contract.receipt()))
    return clean, contract, jobs, config, config_sha


def usage_of(raw, kind):
    if kind == 'generation':
        u = raw.get('usageMetadata')
        if not u:
            return None
        return dict(input_tokens=int(u.get('promptTokenCount', 0) or 0),
            output_tokens=int(u.get('candidatesTokenCount', 0) or 0),
            thought_tokens=int(u.get('thoughtsTokenCount', 0) or 0),
            cached_input_tokens=int(u.get('cachedContentTokenCount', 0) or 0),
            total_tokens=int(u.get('totalTokenCount', 0) or 0))
    # Embedding and Qwen have different usage schemas. Keep them verbatim and
    # never invent zero tokens when the provider does not expose usage.
    return raw.get('usageMetadata') or raw.get('usage') or None


def audited_types(clean, release_clients, args):
    from creamem3.experiments.memeye_resume_20260921 import network
    import httpx
    core, legacy = clean.core, clean.core.legacy
    generation_base, retrieval_base = core._client_types(release_clients)
    gemini_base = release_clients.GEMINI_BASE.rstrip('/')
    qwen_url = release_clients.DASHSCOPE_RERANK_URL
    class AuditMixin:
        def generate(self, prompt, **kwargs):
            self._cost_tag = kwargs.get('tag', '')
            return super().generate(prompt, **kwargs)

        def _post(self, url, payload, *, bearer=None):
            parsed = urlsplit(url)
            safe_url = parsed._replace(query='', fragment='').geturl()
            if bearer:
                if safe_url != qwen_url:
                    raise legacy.NonRetryableProviderError('Unexpected rerank endpoint')
                kind, stage = 'rerank', 'qwen_rerank'
            elif safe_url.startswith(gemini_base + '/models/') and ':batchEmbedContents' in safe_url:
                kind = 'embedding'
                reqs = payload.get('requests', [])
                texts = [''.join(p.get('text','') for p in r.get('content',{}).get('parts',[])) for r in reqs]
                stage = 'query_embedding' if texts == [getattr(_local, 'question', None)] else 'memory_embedding'
            elif safe_url.startswith(gemini_base + '/models/') and safe_url.endswith(':generateContent'):
                kind = 'generation'
                tag = getattr(self, '_cost_tag', '')
                stage = ('writer' if tag == 'write' else 'planner' if tag == 'retrieve'
                         else 'qa' if tag == 'read' else 'online' if 'online' in tag else None)
                if stage is None:
                    raise legacy.NonRetryableProviderError('Unexpected generation stage: ' + tag)
            else:
                raise legacy.NonRetryableProviderError('Unexpected provider endpoint')
            raw_body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self._http_local.body_sha256 = hashlib.sha256(raw_body).hexdigest()
            self._http_local.url_class = 'dashscope' if bearer else 'gemini'
            # Original generations retry in the frozen upper layer (five
            # attempts). Retrieval keeps its original bounded eight attempts.
            limit = 1 if kind == 'generation' else int(self.max_retries)
            for ordinal in range(1, limit + 1):
                disk_guard(args.output)
                ident = str(time.time_ns()) + '_' + uuid.uuid4().hex[:8]
                directory = args.output / 'physical_calls' / _local.qid / stage / ident
                directory.mkdir(parents=True, exist_ok=False)
                intent = dict(question_id=_local.qid, stage=stage, kind=kind,
                    start_unix=time.time(), request_body_sha256=self._http_local.body_sha256,
                    endpoint=safe_url, model=payload.get('model') or safe_url.rsplit('/models/',1)[-1].split(':',1)[0],
                    retry_ordinal_within_transport=ordinal)
                save(directory / 'intent.json', intent)
                pack(directory / 'request.json.gz', payload)
                started = time.perf_counter()
                try:
                    headers = {'Content-Type': 'application/json'}
                    headers['Authorization' if bearer else 'x-goog-api-key'] = ('Bearer ' + bearer) if bearer else self.gemini_api_key
                    response = network.client().post(safe_url, content=raw_body, headers=headers)
                    if not 200 <= response.status_code < 300:
                        retryable = response.status_code in (408,429,500,502,503,504)
                        save(directory / 'outcome.json', dict(status='http_error', http_status=response.status_code,
                            retryable=retryable, elapsed_seconds=time.perf_counter()-started, usage=None))
                        if not retryable:
                            raise legacy.NonRetryableProviderError('Provider HTTP ' + str(response.status_code))
                        if ordinal == limit:
                            raise RuntimeError('Retryable provider HTTP ' + str(response.status_code))
                        time.sleep(min(60, 2**(ordinal-1)))
                        continue
                    raw = response.json()
                    pack(directory / 'response.json.gz', raw)
                    usage = usage_of(raw, kind)
                    candidates = raw.get('candidates', []) if kind == 'generation' else []
                    finish = [c.get('finishReason') for c in candidates]
                    blocked = bool(raw.get('promptFeedback', {}).get('blockReason')) or any(
                        r in ('SAFETY','PROHIBITED_CONTENT','BLOCKLIST','RECITATION','SPII') for r in finish)
                    served = raw.get('modelVersion')
                    mismatch = kind == 'generation' and served and (not served.startswith('gemini-2.5-flash') or 'lite' in served.lower())
                    non_stop = kind == 'generation' and (not candidates or any(r != 'STOP' for r in finish))
                    save(directory / 'outcome.json', dict(status='provider_response', http_status=response.status_code,
                        elapsed_seconds=time.perf_counter()-started, usage=usage,
                        finish_reasons=finish, served_model=served, blocked=blocked, model_mismatch=bool(mismatch), non_stop=non_stop,
                        response_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
                    if blocked or mismatch or non_stop:
                        raise legacy.NonRetryableProviderError('Blocked, truncated/non-STOP output or served-model mismatch; stop')
                    return raw
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if not (directory / 'outcome.json').exists():
                        save(directory / 'outcome.json', dict(status='pre_send_connection_failure', retryable=True,
                            elapsed_seconds=time.perf_counter()-started, usage=None))
                    if ordinal == limit:
                        raise RuntimeError('Pre-send connection failure') from None
                    time.sleep(min(60, 2**(ordinal-1)))
                except httpx.TransportError:
                    if not (directory / 'outcome.json').exists():
                        save(directory / 'outcome.json', dict(status='indeterminate_transport', retryable=False,
                            elapsed_seconds=time.perf_counter()-started, usage=None))
                    raise legacy.NonRetryableProviderError('Indeterminate transport; audit before retry') from None
                except BaseException as exc:
                    if not (directory / 'outcome.json').exists():
                        save(directory / 'outcome.json', dict(status='local_or_parse_error', retryable=False,
                            error_type=type(exc).__name__, elapsed_seconds=time.perf_counter()-started, usage=None))
                    raise
            raise RuntimeError('Bounded transport attempts exhausted')
    class Generation(AuditMixin, generation_base):
        pass
    class Retrieval(AuditMixin, retrieval_base):
        pass
    return Generation, Retrieval


def audit_existing_requests(output):
    for path in (output / 'physical_calls').glob('*/*/*/intent.json'):
        if not path.with_name('outcome.json').exists():
            raise RuntimeError('Unresolved orphan request: ' + str(path.parent))


def summary(output, elapsed, workers):
    manifest = read(output / 'manifest.json')
    qmeta = {q['question_id']: q for q in manifest['questions']}
    rows = []
    for path in (output / 'results/questions').glob('*/result.json'):
        row = read(path)
        if row.get('status') == 'complete':
            rows.append(row)
    stages, per_question = {}, defaultdict(lambda: defaultdict(Counter))
    for path in (output / 'physical_calls').glob('*/*/*/intent.json'):
        i = read(path)
        outpath = path.with_name('outcome.json')
        o = read(outpath) if outpath.exists() else {'status':'orphan','usage':None}
        stage = stages.setdefault(i['stage'], dict(physical_calls=0, provider_seconds=0.0,
            usage_missing_calls=0, input_tokens=0, output_tokens=0, thought_tokens=0,
            cached_input_tokens=0, finish_reasons=Counter(), statuses=Counter(), raw_usage=[]))
        stage['physical_calls'] += 1
        stage['provider_seconds'] += o.get('elapsed_seconds',0)
        stage['statuses'][o['status']] += 1
        stage['finish_reasons'].update(str(v) for v in o.get('finish_reasons',[]))
        u = o.get('usage')
        if u is None:
            stage['usage_missing_calls'] += 1
        elif i['kind'] == 'generation':
            for key in ('input_tokens','output_tokens','thought_tokens','cached_input_tokens'):
                stage[key] += u[key]
                per_question[i['question_id']][i['stage']][key] += u[key]
        else:
            stage['raw_usage'].append(u)
        per_question[i['question_id']][i['stage']]['physical_calls'] += 1
        per_question[i['question_id']][i['stage']]['provider_seconds'] += o.get('elapsed_seconds',0)
    for name, stage in stages.items():
        stage['finish_reasons'] = dict(stage['finish_reasons'])
        stage['statuses'] = dict(stage['statuses'])
        if name in ('memory_embedding','query_embedding','qwen_rerank'):
            for key in ('input_tokens','output_tokens','thought_tokens','cached_input_tokens'):
                stage[key] = None
    complete_ids = {r['question_id'] for r in rows}
    totals = {field:sum(qmeta[q][field] for q in complete_ids) for field in
        ('physical_sessions','nonempty_sessions','writer_batches','expected_online_calls')}
    result = dict(schema='opctx.lme.e2e_cost.summary.v1', completed=len(rows), planned=10,
        completed_ids=sorted(complete_ids), missing_ids=sorted(set(qmeta)-complete_ids),
        stages=stages, per_question={q:{s:dict(v) for s,v in d.items()} for q,d in per_question.items()},
        completed_history_totals=totals,
        result_stage_wall_seconds={r['question_id']:r['timings_seconds'] for r in rows},
        this_launch_wall_seconds=elapsed, question_workers=workers,
        judge_calls=0, mathematical_selector=False,
        note='Cold Writer+Online+embedding+Qwen30+Planner+original Reader+QA. Report unknown embedding/rerank usage as missing, not zero. No accuracy computed. Aggregate per-batch/per-session only with the matching completed histories; failed-history costs remain in stage totals.')
    save(output / ('summary_' + str(time.time_ns()) + '.json'), result)
    return result


def run_live(args, clean, contract, jobs, config, config_sha):
    core, legacy = clean.core, clean.core.legacy
    disk_guard(args.output,25)
    audit_existing_requests(args.output)
    inherited_key = os.environ.get('GEMINI_API_KEY')
    legacy._load_live_credentials()
    if inherited_key:
        os.environ['GEMINI_API_KEY'] = inherited_key
    clients, memory = legacy._load_release_runtime(contract)
    core._assert_model_stack(clients.ModelStack.for_provider('gemini'),config)
    gen_type, ret_type = audited_types(clean,clients,args)
    cache_root, output_root = args.output/'runtime_cache', args.output/'results'
    started = time.perf_counter()
    def run(job):
        item, binding = job
        _local.qid, _local.question = item['question_id'], item['question']
        _local.materializer_seconds = None
        try:
            result = core._process_question(position=binding['position'], item=item,
                manifest_binding=binding, output_root=output_root, cache_root=cache_root,
                config_sha256=config_sha, contract=contract, release_clients=clients,
                release_memory=memory, generation_client_type=gen_type,
                retrieval_client_type=ret_type, writer_workers=1)
            cost_path = args.output/'question_cost'/(_local.qid+'.json')
            cost_record = dict(question_id=_local.qid,
                status='complete', reader_materializer_seconds=_local.materializer_seconds,
                physical_sessions=binding['physical_sessions'], nonempty_sessions=binding['nonempty_sessions'],
                writer_batches=binding['writer_batches'], inference_result_path=str(output_root/'questions'/f"{binding['position']:03d}-{_local.qid}"/'result.json'))
            if not cost_path.exists():
                save(cost_path,cost_record)
            print(json.dumps({'question_complete':_local.qid,'judge_run':False},ensure_ascii=False),flush=True)
            return True
        except Exception as exc:
            save(args.output/'failures'/(_local.qid+'_'+str(time.time_ns())+'.json'),dict(
                question_id=_local.qid,error_type=type(exc).__name__,message=str(exc)[-1000:],time=time.time()))
            print(json.dumps({'question_failed':_local.qid,'error_type':type(exc).__name__},ensure_ascii=False),flush=True)
            return False
    try:
        # The first complete end-to-end history is the mandatory gate. Its
        # successful new-run cache is reused when later continuing to all ten.
        first_ok = run(jobs[0])
        if first_ok and args.phase == 'full':
            with ThreadPoolExecutor(max_workers=args.question_workers) as executor:
                list(executor.map(run,jobs[1:]))
    finally:
        result = summary(args.output,time.perf_counter()-started,args.question_workers)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    expected_ids = {jobs[0][0]['question_id']} if args.phase == 'smoke' else {j[0]['question_id'] for j in jobs}
    if not expected_ids.issubset(set(result['completed_ids'])):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('D:/opctx_e2e_cost_20261111/lme10_run_r1'))
    p.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    p.add_argument('--allow-live', action='store_true')
    p.add_argument('--phase', choices=('smoke','full'), default='smoke')
    p.add_argument('--question-workers', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.question_workers <= 8:
        p.error('question-workers must be within original safety range 1..8')
    args.output = args.output.resolve()
    clean, contract, jobs, config, config_sha = prepare(args)
    print(json.dumps({'prepared':10,'output':str(args.output),'phase':args.phase,
        'live':args.allow_live,'expected':read(args.output/'preflight.json')['totals']},ensure_ascii=False),flush=True)
    if not args.allow_live:
        return
    # Only one live coordinator may mutate the new experiment root. Lock
    # contention is an explicit stop, not a queued duplicate launch.
    with FileLock(str(args.output/'live_coordinator.lock'),timeout=0):
        run_live(args,clean,contract,jobs,config,config_sha)


if __name__ == '__main__':
    main()

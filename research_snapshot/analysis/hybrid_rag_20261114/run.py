"""Fixed, resumable raw-chunk Hybrid RAG experiments; offline by default."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv, dotenv_values
load_dotenv(ROOT / '.env', override=False)
# User explicitly updated this project's private key for this experiment.
# Do not let an older desktop-process environment shadow that setting.
_project_key = dotenv_values(ROOT / '.env').get('GEMINI_API_KEY')
if _project_key:
    os.environ['GEMINI_API_KEY'] = _project_key
del _project_key
from filelock import FileLock
import numpy as np
from analysis.hybrid_rag_20261114 import data, stages, transport

HERE = Path(__file__).resolve().parent
SETTINGS = {
    'locomo': dict(branch_k=40, union_cap=40, final_k=20, expected=1540),
    'lme': dict(branch_k=240, union_cap=240, final_k=30, expected=500),
}
save, digest = transport.save, transport.digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_id(text):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', text):
        raise ValueError('Unsafe record identifier')
    return text


def frozen_prepare(args, adapter):
    contexts = data.load_dataset(args.dataset)
    settings = SETTINGS[args.dataset]
    config = dict(dataset=args.dataset, **settings, chunk_max_words=data.MAX_WORDS,
        overlap_content_words=data.OVERLAP_WORDS, rrf_constant=60,
        embedding_model='gemini-embedding-001', embedding_dimensions=768,
        embedding_task_types=['RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'],
        query_date_appended_for_lme_retrieval=True, model_contract=adapter.contract_receipt(),
        workers=args.workers, retry_policy='physical exact request at most twice for classified temporary errors',
        question_ids=[q['question_id'] for c in contexts for q in c['questions']],
        dataset_sha256=file_sha(data.DATASET_PATHS[args.dataset]),
        source_sha256={str(HERE/name): file_sha(HERE/name)
                       for name in ('run.py','transport.py','stages.py','data.py')},
        historical_comparison_rates=dict(frozen_date='2026-11-11',
            gemini_25_flash_input_usd_per_million=.30,
            gemini_25_flash_cached_input_usd_per_million=.03,
            gemini_25_flash_output_including_thinking_usd_per_million=2.50,
            embedding_usd_per_million=.15,qwen_beijing_input_cny_per_million=.50,
            note='Fixed historical comparison convention, not a current-price quote or invoice'),
        original_transport_sha256=file_sha(Path(transport.original.__file__)))
    path = args.output / 'config.json'
    if path.exists():
        if read(path) != config:
            raise RuntimeError('Frozen experiment contract changed; use a separately registered run')
    else:
        save(path, config)
    preflight_path = args.output / 'offline_inventory.json'
    if not preflight_path.exists():
        inventory = []
        for context in contexts:
            chunks = data.build_chunks(context['sessions'])
            inventory.append(dict(context_id=context['context_id'],
                sessions=len(context['sessions']),
                nonempty_sessions=sum(bool(s['turns']) for s in context['sessions']),
                questions=len(context['questions']), chunks=len(chunks),
                words=sum(c['word_count'] for c in chunks), chars=sum(len(c['text']) for c in chunks),
                full_pool_under_cap=len(chunks) <= settings['union_cap']))
        save(preflight_path, dict(contexts=inventory,
            total_questions=sum(i['questions'] for i in inventory),
            total_chunks=sum(i['chunks'] for i in inventory),
            total_chars=sum(i['chars'] for i in inventory),
            model_calls=0, config_sha256=digest(config)))
    return contexts, config


def build_index(context, args, wire, embed):
    tick = time.perf_counter()
    directory = args.output / 'indices' / safe_id(context['context_id'])
    done = directory / 'complete.json'
    if done.exists():
        receipt = read(done)
        if file_sha(directory / 'chunks.json') != receipt['chunks_sha256'] or file_sha(directory / 'vectors.npy') != receipt['vectors_sha256']:
            raise RuntimeError('Index checkpoint hash mismatch')
        chunks = read(directory / 'chunks.json')
        vectors = np.load(directory / 'vectors.npy', allow_pickle=False)
        return chunks, vectors, receipt
    chunks = data.build_chunks(context['sessions'])
    if not chunks:
        raise RuntimeError('No nonempty raw dialogue chunks')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'chunks.json').exists():
        if read(directory / 'chunks.json') != chunks:
            raise RuntimeError('Chunk definition changed')
    else:
        save(directory / 'chunks.json', chunks)
    wire.local.scope = context['context_id'] + '/index'
    wire.local.stage = 'embedding'
    vectors = np.asarray(embed([c['text'] for c in chunks], tag='embed',
        task_type='RETRIEVAL_DOCUMENT', dim=768), dtype=np.float32)
    if vectors.shape != (len(chunks), 768) or not np.isfinite(vectors).all():
        raise RuntimeError('Index embedding shape/value mismatch')
    target = directory / 'vectors.npy'
    if target.exists():
        if not np.array_equal(np.load(target, allow_pickle=False), vectors):
            raise RuntimeError('Existing vectors differ')
    else:
        with target.open('xb') as handle:
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    receipt = dict(context_id=context['context_id'], chunks=len(chunks),
        input_chars=sum(len(c['text']) for c in chunks),
        sessions=len(context['sessions']), indexing_wall_seconds=time.perf_counter()-tick,
        chunks_sha256=file_sha(directory/'chunks.json'), vectors_sha256=file_sha(target))
    save(done, receipt)
    return chunks, vectors, receipt


def process_question(context, question, index, args, wire, embed, adapter, *, score=True):
    qid = safe_id(question['question_id'])
    directory = args.output / 'questions' / qid
    finished = directory / 'result.json'
    if finished.exists():
        return read(finished)
    inference_file = directory / 'inference.json'
    if inference_file.exists():
        inference=read(inference_file)
        return score_inference(inference,args,wire,adapter) if score else inference
    if (directory / 'failure.json').exists():
        raise RuntimeError('Prior failed question requires explicit inspection, not silent rerun')
    chunks, vectors, _ = index
    chunk_map = {c['id']: c for c in chunks}
    wire.local.scope = qid
    started = time.perf_counter()
    timing = {}
    try:
        retrieval_file = directory / 'retrieval.json'
        if retrieval_file.exists():
            retrieved = read(retrieval_file)
            candidates = [chunk_map[i] for i in retrieved['candidate_ids']]
            timing.update(retrieved['timing'])
        else:
            tick = time.perf_counter()
            qv = embed([question['question']], task_type='RETRIEVAL_QUERY', dim=768, tag='embed_query')[0]
            timing['query_embedding_seconds'] = time.perf_counter()-tick
            tick = time.perf_counter()
            dense = data.cosine_scores(vectors, qv)
            lexical = data.BM25Index(chunks).scores(question['question'])
            options = SETTINGS[args.dataset]
            candidates, trace = data.hybrid_candidates(chunks, dense, lexical,
                branch_k=options['branch_k'], union_cap=options['union_cap'], rrf_constant=60)
            timing['retrieval_seconds'] = time.perf_counter()-tick
            retrieved = dict(question_id=qid, context_id=context['context_id'],
                query=question['question'], candidate_ids=[c['id'] for c in candidates],
                trace=trace, timing=timing.copy())
            save(retrieval_file, retrieved)
        selected_file = directory / 'selected.json'
        if selected_file.exists():
            selected = read(selected_file)
            timing['rerank_seconds'] = selected['rerank_seconds']
        else:
            tick = time.perf_counter()
            selected = adapter.rerank(args.dataset, question['raw_question'], candidates)
            selected['rerank_seconds'] = time.perf_counter()-tick
            timing['rerank_seconds'] = selected['rerank_seconds']
            save(selected_file, selected)
        evidence = '\n\n'.join(chunk_map[i]['text'] for i in selected['selected_ids'])
        context_file = directory / 'context.json'
        context_record = dict(selected_ids=selected['selected_ids'], text=evidence,
                              context_chars=len(evidence), context_sha256=digest(evidence))
        if context_file.exists():
            if read(context_file) != context_record:
                raise RuntimeError('Evidence checkpoint mismatch')
        else:
            save(context_file, context_record)
        prediction_file = directory / 'prediction.json'
        if prediction_file.exists():
            answer = read(prediction_file)
            timing['qa_seconds'] = answer['qa_seconds']
        else:
            tick = time.perf_counter()
            answer = adapter.answer(args.dataset, question['raw_question'], evidence,
                                    question_date=question['question_date'])
            answer['qa_seconds'] = time.perf_counter()-tick
            timing['qa_seconds'] = answer['qa_seconds']
            save(prediction_file, answer)
        result = dict(question_id=qid, context_id=context['context_id'],
            category=question['category'], category_name=question['category_name'],
            question=question['raw_question'], reference=question['answer'],
            prediction=answer['prediction'],
            selected_ids=selected['selected_ids'], candidate_count=len(candidates),
            context_chars=len(evidence), timing=timing,
            current_execution_seconds=time.perf_counter()-started,
            inference_completed_utc=datetime.now(timezone.utc).isoformat())
        save(inference_file, result)
        return score_inference(result,args,wire,adapter) if score else result
    except BaseException as exc:
        if not (directory / 'failure.json').exists():
            save(directory / 'failure.json', dict(question_id=qid,
                error_type=type(exc).__name__,
                error=str(exc) if isinstance(exc, stages.StageFailure) else 'See preserved stage/request ledger',
                audit=getattr(exc, 'audit', None), timing=timing,
                transport_stopped=wire.stop.is_set(),
                failed_utc=datetime.now(timezone.utc).isoformat()))
        raise


def score_inference(inference,args,wire,adapter):
    qid=safe_id(inference['question_id'])
    directory=args.output/'questions'/qid
    if (directory/'result.json').exists():
        return read(directory/'result.json')
    wire.local.scope=qid
    tick=time.perf_counter()
    try:
        # Gold is exposed to a model only after prediction/inference checkpoints.
        verdict=adapter.judge(args.dataset,inference['question'],
                              inference['reference'],inference['prediction'])
        result={**inference, 'correct':verdict['correct'],'judge':verdict,
                'timing':{**inference['timing'],'judge_seconds':time.perf_counter()-tick},
                'completed_utc':datetime.now(timezone.utc).isoformat()}
        save(directory/'result.json',result)
        return result
    except BaseException as exc:
        if not (directory/'failure.json').exists():
            save(directory/'failure.json',dict(question_id=qid,stage='judge',
                error_type=type(exc).__name__,audit=getattr(exc,'audit',None),
                transport_stopped=wire.stop.is_set()))
        raise


def summarize(output, config, wall_seconds, phase):
    rows = [read(p) for p in (output / 'questions').glob('*/result.json')]
    failures = [read(p) for p in (output / 'questions').glob('*/failure.json')]
    categories = defaultdict(list)
    for row in rows:
        categories[row['category_name']].append(row)
    calls = [read(p) for p in (output / 'runtime' / 'calls').glob('*/outcome.json')]
    calls += [read(p) for p in (output / 'runtime' / 'qwen_calls').glob('*/outcome.json')]
    stats = defaultdict(lambda: dict(physical_calls=0, input_tokens=0, output_tokens=0,
        thought_tokens=0, cached_input_tokens=0, usage_missing_calls=0, service_seconds=0))
    for call in calls:
        s = stats[call['stage']]
        s['physical_calls'] += call.get('physical_calls', 0)
        s['service_seconds'] += call.get('service_seconds', 0)
        usage = call.get('usage')
        if usage is None:
            s['usage_missing_calls'] += call.get('physical_calls', 0)
        else:
            s['input_tokens'] += int(usage.get('promptTokenCount', usage.get('input_tokens', usage.get('total_tokens', 0))) or 0)
            s['output_tokens'] += int(usage.get('candidatesTokenCount', usage.get('output_tokens', 0)) or 0)
            s['thought_tokens'] += int(usage.get('thoughtsTokenCount', 0) or 0)
            s['cached_input_tokens'] += int(usage.get('cachedContentTokenCount', 0) or 0)
    counts = lambda rr: dict(n=len(rr), correct=sum(r['correct'] for r in rr),
                            accuracy_percent=100*sum(r['correct'] for r in rr)/len(rr) if rr else None)
    costs=dict(gemini_generation_usd=0.0,gemini_generation_standard_input_usd=0.0,
        qwen_rerank_cny=0.0,embedding_usd=None,judge_cost_excluded=True,
        pricing_convention_date='2026-11-11',
        unknown_usage_physical_calls=0,embedding_token_cost_not_estimated=True)
    for call in calls:
        if call['stage']=='judge':
            continue
        usage=call.get('usage')
        if not usage:
            costs['unknown_usage_physical_calls']+=call.get('physical_calls',0)
            continue
        if call['model']=='gemini-2.5-flash':
            prompt=int(usage.get('promptTokenCount',0) or 0)
            cached=int(usage.get('cachedContentTokenCount',0) or 0)
            generated=int(usage.get('candidatesTokenCount',0) or 0)+int(usage.get('thoughtsTokenCount',0) or 0)
            costs['gemini_generation_usd']+=((prompt-cached)*.30+cached*.03+generated*2.50)/1e6
            costs['gemini_generation_standard_input_usd']+=(prompt*.30+generated*2.50)/1e6
        elif call['model']=='qwen3-rerank':
            tokens=usage.get('input_tokens',usage.get('total_tokens'))
            if tokens is None:
                costs['unknown_usage_physical_calls']+=call.get('physical_calls',0)
            else:
                costs['qwen_rerank_cny']+=int(tokens)*.50/1e6
    for s in stats.values():
        s['known_input_tokens']=s['input_tokens']
        s['known_output_tokens']=s['output_tokens']
        if s['usage_missing_calls']:
            s['input_tokens']=None
            s['output_tokens']=None
    result = dict(dataset=config['dataset'], phase=phase, expected=config['expected'],
        complete=(len(rows)==config['expected'] and
                  (set(r['question_id'] for r in rows)==set(config['question_ids'])
                   if 'question_ids' in config else True)), overall=counts(rows),
        categories={k: counts(v) for k,v in categories.items()}, failures=failures,
        missing_count=config['expected']-len(rows), physical_stage_usage=dict(stats),
        physical_inference_cost=costs,
        absent_stages=['writer','online','query_projection','reader'],
        evaluation_stage_excluded_from_inference_cost='judge',
        current_execution_wall_seconds_including_evaluation=wall_seconds,
        time_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'summaries' / (str(time.time_ns()) + '.json'), result)
    print(json.dumps({k: result[k] for k in ('dataset','phase','complete','overall','missing_count')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(SETTINGS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare','preflight','smoke','full'], default='prepare')
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Each dataset has at most four workers')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / 'runner.lock'), timeout=0):
        adapter = stages.make_stages(lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError('Offline mode')))
        contexts, config = frozen_prepare(args, adapter)
        if args.phase == 'prepare':
            print(json.dumps(dict(dataset=args.dataset, status='offline_prepared',
                inventory=str(args.output/'offline_inventory.json')), ensure_ascii=False), flush=True)
            return
        if not args.allow_model_calls:
            raise RuntimeError('Live phases require explicit --allow-model-calls')
        transport.preflight(args.output, include_qwen=args.dataset=='lme')
        if args.phase == 'preflight':
            return
        (args.output / 'runtime').mkdir(exist_ok=True)
        wire = transport.AuditTransport(args.output / 'runtime')
        chat, embed = transport.make_clients(wire)
        adapter = stages.make_stages(chat, qwen_post=wire.qwen_post)
        start = time.perf_counter()
        failures = []
        selected_contexts = contexts[:1] if args.phase=='smoke' else contexts
        def context_job(context):
            if wire.stop.is_set():
                return
            questions = context['questions'][:1] if args.phase=='smoke' else context['questions']
            if all((args.output/'questions'/safe_id(q['question_id'])/'inference.json').exists()
                   for q in questions):
                return
            index = build_index(context,args,wire,embed)
            for question in questions:
                if wire.stop.is_set():
                    return
                try:
                    process_question(context,question,index,args,wire,embed,adapter,score=False)
                    print(json.dumps(dict(dataset=args.dataset, question_id=question['question_id'], status='prediction_saved')), flush=True)
                except transport.ExecutionStop:
                    raise
                except Exception as exc:
                    failures.append(dict(question_id=question['question_id'], error_type=type(exc).__name__))
        try:
            # Different histories are independent; LoCoMo has ten histories and
            # LME five hundred. Keep the same total limit during all stages.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(context_job,c): c['context_id'] for c in selected_contexts}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException as exc:
                        failures.append(dict(context_id=futures[future],error_type=type(exc).__name__))
                        if isinstance(exc, transport.ExecutionStop):
                            wire.stop.set()
            inference_wall=time.perf_counter()-start
            save(args.output/'execution_timing'/(str(time.time_ns())+'_inference.json'),
                dict(phase=args.phase,wall_seconds=inference_wall,
                     includes_indexing=True,includes_judge=False,
                     note='This execution window only; preserved checkpoints may be resumed'))
            if not wire.stop.is_set():
                judge_start=time.perf_counter()
                pending=[read(p) for p in (args.output/'questions').glob('*/inference.json')
                         if not p.with_name('result.json').exists()
                         and not p.with_name('failure.json').exists()]
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    jf={pool.submit(score_inference,row,args,wire,adapter):row['question_id']
                        for row in pending}
                    for future in as_completed(jf):
                        try:
                            future.result()
                            print(json.dumps(dict(dataset=args.dataset,question_id=jf[future],status='scored')),flush=True)
                        except BaseException as exc:
                            failures.append(dict(question_id=jf[future],stage='judge',error_type=type(exc).__name__))
                            if isinstance(exc,transport.ExecutionStop):
                                wire.stop.set()
                save(args.output/'execution_timing'/(str(time.time_ns())+'_judge.json'),
                    dict(phase=args.phase,wall_seconds=time.perf_counter()-judge_start,includes_judge=True))
        finally:
            result = summarize(args.output,config,time.perf_counter()-start,args.phase)
            save(args.output/'executions'/(str(time.time_ns())+'.json'),
                dict(phase=args.phase,failures=failures,transport_stopped=wire.stop.is_set()))
        if wire.stop.is_set() or failures or (args.phase=='full' and not result['complete']):
            raise SystemExit(2)


if __name__ == '__main__':
    main()

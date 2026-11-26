"""Explicit failure-preserving projection panel and native3072 embedding stage.

Old planner files remain immutable. The new panel does not convert a provider
failure into a successful model plan. No facts, gold, QA or Judge are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

from creamem3.experiments.iterative_memory_20260907 import query_projections as qp

ROOT = qp.ROOT.parent / 'v1_failure_fallback'
RUNTIME_SHA = 'bb782cbba5d24f08c13907f2a9fe386c8bcffa5e15f53b675b69818c4989298b'
Q0_PATH = qp.QUERY_PATH.parent.parent / 'predict/query_embeddings.json'
Q0_SHA = 'ca92abc223651ce861c77c78807e76085b5e884f3604df4a3ec888f2a09bb6cb'
Q0_CONTRACT = 'e925dbcd543b9763135771c43412ed0c0561f35f46ca878228ee185a611d5afd'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_empty_exhaustion(plan, failure, attempts):
    require(failure.get('failure_kind') == 'explicit_retryable_attempts_exhausted', 'not exhausted retryable failure')
    require(failure.get('attempts') == 5 and failure.get('provider_recall_allowed') is False, 'failure cap/recall drift')
    require(len(attempts) == 5, 'exactly five attempts required')
    for ordinal, (intent, response) in enumerate(attempts, 1):
        for key in ('request_sha256', 'resume_hash'):
            require(failure.get(key) == plan[key] == intent.get(key) == response.get(key), 'failure request binding drift')
        require(intent.get('attempt_ordinal') == response.get('attempt_ordinal') == ordinal, 'failed attempt ordinal gap')
        require(intent.get('attempt_identity_sha256') == response.get('attempt_identity_sha256'), 'failed response identity drift')
        require(response.get('outcome_kind') == 'explicit_provider_error' and response.get('retryable') is True, 'not explicit retryable error')
        require(response.get('error_type') == 'EmptyProviderResponse' and response.get('status_code') is None, 'failure is not determinate empty response')
        require(response.get('error_message') == 'Gemini returned an empty determinate response', 'empty failure message drift')
        require(response.get('usage', {}).get('output_tokens') == 0, 'empty response includes output')
        for record in (failure, intent, response):
            require(record.get('model') == 'gemini-2.5-flash-lite' and record.get('stage') == 'planner', 'failed model/stage drift')


def failure_only_projection(question):
    return {'schema': 'query_only.provider_failure_projection.v1', 'status': 'provider_failure_original_only',
            'fallback_reason': 'provider_failure_original_only', 'question': question,
            'question_sha256': qp.sha(question), 'subqueries': [question],
            'projections': [{'kind': 'original', 'construction': 'unchanged complete question', 'text': question,
                             'spans': [{'start': 0, 'end': len(question), 'quote': question}]}],
            'original_query_retained': True, 'model_plan_success': False}


def prepare_panel(root=ROOT, planner_root=qp.ROOT):
    reg = qp.read_sealed(planner_root / 'preregistration.json')
    queries = qp.load_queries()
    require(reg['questions'] == queries, 'planner population drift')
    require(qp.file_sha(Path(qp.__file__)) == reg['generator_code_sha256'], 'sealed planner source drift')
    policy = {'schema': 'query_projection.failure_fallback_policy.v1',
              'rule': 'For any exact original planner request with five bound determinate empty retryable responses and an exhausted failure seal, retain original query only; no new provider call and no model-success claim.',
              'other_provider_errors': 'stop and report; never substitute a fallback',
              'source_preregistration_sha256': qp.sha(reg), 'source_preregistration_path': str(planner_root / 'preregistration.json'),
              'code_sha256': qp.file_sha(Path(__file__)), 'old_artifacts_modified': False,
              'downstream_full_panel_is_not_full_planner_success': True}
    qp.write_once(root / 'policy.json', policy)
    rows = []
    failed_ids = set()
    for query in queries:
        name = qp.sha(query['row_id']) + '.json'
        plan = qp.read_sealed(planner_root / 'requests' / name)
        require(plan['question'] == query['question'] and plan['prompt'] == qp.build_prompt(query['question']), 'source question/prompt drift')
        if (planner_root / 'rows' / name).exists():
            saved = qp.generate_one(query, None, reg, planner_root)
            source_binding = {'kind': 'successful_planner_response', 'path': str(planner_root / 'rows' / name),
                              'file_sha256': qp.file_sha(planner_root / 'rows' / name), 'payload_sha256': qp.sha(saved)}
            projection = saved['projection']
        else:
            failure_path = planner_root / 'provider/failures/planner' / (plan['resume_hash'] + '.json')
            failure = load_json(failure_path)
            error = qp.read_sealed(planner_root / 'errors' / name)
            stop = qp.read_sealed(planner_root / 'stops' / name)
            require(error.get('row_id') == stop.get('row_id') == query['row_id'], 'error/stop row drift')
            require(error.get('error_class') == stop.get('error_class') == 'ProviderRetriesExhausted', 'unexpected stopped failure class')
            attempt_root = planner_root / 'provider/attempts/planner' / plan['resume_hash']
            intents = sorted(attempt_root.glob('attempt_*_request.json'))
            attempts = [(load_json(p), load_json(p.with_name(p.name.replace('_request.json', '_response.json')))) for p in intents]
            validate_empty_exhaustion(plan, failure, attempts)
            source_binding = {'kind': 'exhausted_empty_provider_failure', 'path': str(failure_path),
                              'file_sha256': qp.file_sha(failure_path), 'full_failure_seal': failure,
                              'error_path': str(planner_root / 'errors' / name), 'error_file_sha256': qp.file_sha(planner_root / 'errors' / name),
                              'stop_path': str(planner_root / 'stops' / name), 'stop_file_sha256': qp.file_sha(planner_root / 'stops' / name),
                              'attempts': [{'intent_path': str(p), 'intent_sha256': qp.file_sha(p),
                                            'response_path': str(p.with_name(p.name.replace('_request.json', '_response.json'))),
                                            'response_sha256': qp.file_sha(p.with_name(p.name.replace('_request.json', '_response.json')))} for p in intents]}
            failed_ids.add(query['row_id'])
            projection = failure_only_projection(query['question'])
        rows.append({**query, 'projection': projection, 'source_binding': source_binding})
    actual_failures = {qp.read_sealed(p)['row_id'] for p in (planner_root / 'errors').glob('*.json')}
    require(actual_failures == failed_ids, 'unhandled extra planner failure')
    counts = Counter(row['projection']['status'] for row in rows)
    panel = {'schema': 'query_projection.downstream_panel.v1_failure_fallback', 'policy_sha256': qp.sha(policy),
             'population_n': len(rows), 'planner_successes': sum(v for k, v in counts.items() if k != 'provider_failure_original_only'),
             'status_counts': dict(counts), 'full_downstream_panel': True, 'full_planner_success': not failed_ids,
             'new_provider_calls': 0, 'rows': rows}
    qp.write_once(root / 'panel.json', panel)
    return panel


def load_q0_vectors(queries):
    require(qp.file_sha(Q0_PATH) == Q0_SHA, 'original query embedding file drift')
    data = load_json(Q0_PATH)
    body = dict(data)
    require(body.pop('query_embedding_sha256') == Q0_CONTRACT == qp.sha(body), 'original embedding contract drift')
    query_source = load_json(qp.QUERY_PATH)
    require(data['complete'] is True and data['query_rows_sha256'] == query_source['rows_sha256'], 'original question source linkage drift')
    require(data['rows_sha256'] == qp.sha(data['rows']) and len(data['rows']) == len(queries), 'q0 rows seal drift')
    vectors = {}
    for query, row in zip(queries, data['rows'], strict=True):
        require(row['row_id'] == query['row_id'] and row['question_sha256'] == qp.sha(query['question']), 'q0 question/vector binding drift')
        vector = row['vector']
        require(row['vector_sha256'] == qp.sha(vector), 'q0 vector hash drift')
        require(len(vector) == 3072 and all(math.isfinite(v) for v in vector) and sum(v*v for v in vector) > 0, 'q0 dimension/nonfinite/zero drift')
        vectors[query['row_id']] = vector
    return vectors


def prepare_embeddings(root=ROOT):
    panel = prepare_panel(root)
    q0 = load_q0_vectors(panel['rows'])
    from creamem3 import memeye_official_embedding_batch_runtime_v19_1 as embedding
    require(qp.file_sha(Path(embedding.__file__)) == RUNTIME_SHA, 'native3072 runtime source drift')
    original = {row['question'] for row in panel['rows']}
    extras = {text for row in panel['rows'] for text in row['projection']['subqueries'][1:]}
    texts = sorted(extras - original, key=lambda text: (qp.sha(text), text))
    require(len(texts) == 365, 'approved 365 extra-text census drift')
    batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
    require(len(batches) == 4, 'approved four-batch ceiling drift')
    registration = {'schema': 'query_projection.native3072_embeddings.v1', 'panel_sha256': qp.sha(panel),
                    'code_sha256': qp.file_sha(Path(__file__)), 'runtime_source_sha256': RUNTIME_SHA,
                    'q0_cache_file': str(Q0_PATH), 'q0_cache_file_sha256': Q0_SHA, 'q0_contract_sha256': Q0_CONTRACT,
                    'q0_validated_rows': len(q0), 'q0_new_provider_calls': 0,
                    'new_unique_texts': texts, 'new_texts_sha256': qp.sha(texts), 'logical_batches': 4,
                    'batch_sizes': list(map(len, batches)), 'max_physical_attempts': 12, 'max_attempts_per_batch': 3,
                    'model': embedding.MODEL_ID, 'native_dimension': 3072, 'visible_text_rule': 'text[:8000]',
                    'omitted_provider_fields': ['task_type', 'input_type', 'output_dimensionality'],
                    'route': embedding.route_contract(), 'credentials': 'existing GEMINI_API_KEY environment only; value never persisted',
                    'provider_failure_policy': 'stop, preserve receipts; no fallback or unapproved recall', 'workers': 1}
    qp.write_once(root / 'embedding_preregistration.json', registration)
    for ordinal, texts_batch in enumerate(batches):
        metadata = {'schema': registration['schema'], 'preregistration_sha256': qp.sha(registration), 'batch_ordinal': ordinal}
        contract = embedding.request_identity(texts_batch, cache_metadata=metadata, limits=embedding.EmbeddingLimits(12, 3))
        qp.write_once(root / 'embedding_requests' / f'{ordinal:03d}.json', {'texts': texts_batch, 'metadata': metadata, 'contract': contract})
    return panel, registration, q0


def execute(root=ROOT):
    panel, reg, q0 = prepare_embeddings(root)
    if (root / 'embedding_stop.json').exists():
        raise RuntimeError('preserved embedding failure requires main-thread direction')
    from creamem3 import memeye_official_embedding_batch_runtime_v19_1 as embedding
    runtime = embedding.OfficialGoogleBatchEmbeddingRuntime(root / 'embedding_provider', containment_root=root,
               limits=embedding.EmbeddingLimits(12, 3), allow_model_calls=True, budget_binding_sha256=qp.sha(reg))
    all_vectors = {}
    for ordinal in range(4):
        plan = qp.read_sealed(root / 'embedding_requests' / f'{ordinal:03d}.json')
        try:
            result = runtime.embed_batch(plan['texts'], cache_metadata=plan['metadata'])
        except Exception as error:
            qp.write_once(root / 'embedding_stop.json', {'batch_ordinal': ordinal, 'error_class': type(error).__name__,
                          'status_code': getattr(error, 'status_code', None), 'retryable': getattr(error, 'retryable', None),
                          'message': 'embedding failure preserved; no further dispatch or fallback'})
            raise
        require(result.request_sha256 == plan['contract']['identity'] and result.resume_hash == plan['contract']['resume_hash'], 'embedding request binding drift')
        require(list(result.full_text_sha256s) == [qp.sha(t) for t in plan['texts']], 'embedding text order drift')
        record = {'batch_ordinal': ordinal, 'request_sha256': result.request_sha256, 'resume_hash': result.resume_hash,
                  'response_sha256': result.response_sha256, 'vectors_sha256': result.vectors_sha256,
                  'cache_path': result.cache_path, 'cache_file_sha256': qp.file_sha(Path(result.cache_path)),
                  'vector_rows': [{'text': t, 'text_sha256': qp.sha(t), 'vector': list(v), 'vector_sha256': h}
                                  for t, v, h in zip(plan['texts'], result.vectors, result.vector_sha256s, strict=True)]}
        qp.write_once(root / 'embedding_batches' / f'{ordinal:03d}.json', record)
        all_vectors.update({r['text']: r['vector'] for r in record['vector_rows']})
        print(f'native3072 projection batch {ordinal+1}/4 complete ({len(record["vector_rows"])} texts)', flush=True)
    q0_text = {row['question']: q0[row['row_id']] for row in panel['rows']}
    bound = []
    for row in panel['rows']:
        projections = []
        for item in row['projection']['projections']:
            text = item['text']
            vector = q0[row['row_id']] if item['kind'] == 'original' else (q0_text[text] if text in q0_text else all_vectors[text])
            projections.append({**item, 'vector': vector, 'vector_sha256': qp.sha(vector), 'origin': 'original_cached_q0' if text in q0_text else 'new_official_native3072'})
        bound.append({'row_id': row['row_id'], 'question': row['question'], 'plan_status': row['projection']['status'], 'projections': projections})
    report = {'schema': 'query_projection.embedded_panel.v1', 'complete': True, 'population_n': len(bound),
              'panel_sha256': qp.sha(panel), 'embedding_preregistration_sha256': qp.sha(reg),
              'planner_status_counts': panel['status_counts'], 'full_planner_success': panel['full_planner_success'],
              'usage': runtime.usage_receipt(), 'q0_reused_rows': 371, 'new_unique_vectors': len(all_vectors), 'rows': bound}
    qp.write_once(root / 'embedded_panel.json', report)
    return {k: v for k, v in report.items() if k != 'rows'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'live'])
    parser.add_argument('--allow-model-calls', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        panel, reg, _ = prepare_embeddings()
        out = {'status_counts': panel['status_counts'], 'batch_sizes': reg['batch_sizes'], 'new_provider_calls': 0, 'preregistration_sha256': qp.sha(reg)}
    else:
        if not args.allow_model_calls:
            parser.error('explicit main-thread authorization and --allow-model-calls required')
        out = execute()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

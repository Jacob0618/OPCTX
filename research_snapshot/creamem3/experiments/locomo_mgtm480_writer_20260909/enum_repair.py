"""Add a candidate-enum repair only after all existing rerank contracts fail."""
import json
from concurrent.futures import ThreadPoolExecutor
from creamem3.experiments.locomo_mgtm480_writer_20260909 import full

base, old, StopRun = full.base, full.old, full.StopRun
SOURCE = full.RUN


def output_schema(candidates, n):
    ids = [c['fact']['id'] for c in candidates]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids) or not 0 < n <= len(ids):
        raise StopRun('Invalid candidate inventory for enum repair')
    schema = base.output_schema(n)
    schema['properties']['ranked_fact_ids']['items']['enum'] = ids
    return schema


def validated_ids(raw, candidates, n):
    value = old.graph_api._parse_json_object(raw)
    if not isinstance(value, dict) or set(value) != {'ranked_fact_ids'}:
        return None, 'Expected exactly one ranked_fact_ids object'
    ids = value['ranked_fact_ids']; legal = {c['fact']['id'] for c in candidates}
    if not isinstance(ids, list) or len(ids) != n or any(not isinstance(i, str) for i in ids):
        return None, 'Expected exactly %d string IDs' % n
    if len(set(ids)) != n: return None, 'Duplicate IDs are not allowed'
    if not set(ids) <= legal: return None, 'IDs must come from the supplied candidate list'
    return ids, None


class Index(base.RepairedIndex):
    def _rerank(self, question, candidates, rerank_chat, *, max_attempts=2):
        try:
            return super()._rerank(question, candidates, rerank_chat, max_attempts=max_attempts)
        except StopRun as exc:
            if str(exc) != 'Structured rerank repair failed; do not fill with cosine': raise
        n = min(self.rerank_top_k, len(candidates))
        prompt = old.graph_api.FACT_RERANK_PROMPT.format(top_k=n, question=question,
            candidates_json=json.dumps([{'fact_id': c['fact']['id'], 'text': c['fact']['text']} for c in candidates],
                                       ensure_ascii=False, separators=(',', ':')))
        schema = output_schema(candidates, n); error = None
        for attempt in range(1, 3):
            actual_prompt = prompt if attempt == 1 else prompt + '\n\nOutput validation failed: ' + error + '. Return only the required unique candidate IDs.'
            raw = rerank_chat(actual_prompt, tag='creamem3_v2_rerank', response_schema=schema)
            ids, error = validated_ids(raw, candidates, n)
            if ids is not None:
                by_id = {c['fact']['id']: c for c in candidates}
                return [by_id[i] for i in ids], {'attempts': max_attempts+1+attempt, 'fallback': False,
                    'error': None, 'format_repair': True, 'candidate_enum_repair': True, 'enum_attempts': attempt,
                    'previous_validation_error': 'Structured rerank repair failed; do not fill with cosine'}
        raise StopRun('Candidate-enum rerank repair failed after two attempts: ' + str(error))


class Runtime(base.Runtime):
    def __init__(self, root, *, allow_model_calls):
        super().__init__(root, parents=(SOURCE,), allow_model_calls=allow_model_calls, concurrency=20, timeout=240)
        self.first_fatal_recorded = False

    def record_failure(self, exc):
        self.stopped.set()
        with self.mutex:
            if self.first_fatal_recorded: return
            self.first_fatal_recorded = True
        record = {'utc': base.now(), 'type': type(exc).__name__, 'message': str(exc)[:600]}
        full.seal(self.root/'first_fatal.json', record)
        self.log({'kind': 'fatal_worker_error', **record})


class FailFastPool(ThreadPoolExecutor):
    def __init__(self, runtime, *args, **kwargs):
        self.runtime = runtime
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        def guarded():
            if self.runtime.stopped.is_set(): raise StopRun('Worker queue stopped after fatal error')
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                self.runtime.record_failure(exc)
                raise
        return super().submit(guarded)

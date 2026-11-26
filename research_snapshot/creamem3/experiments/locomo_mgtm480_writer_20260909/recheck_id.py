"""One fresh, unchanged request to test the observed structured-rerank failure."""
import json
from pathlib import Path
from creamem3.experiments.locomo_mgtm480_writer_20260909 import full

SOURCE = full.RUN
RUN = SOURCE.parent/'locomo_mgtm480_rerank_id_recheck_20260910_r1'
BAD_KEY = 'c7297a1c8bd5c3828ecac3fac811356b74957100d807b42b0e2e2a180d60ea79'
read, seal, sha = full.read, full.seal, full.sha


def validate(request, response):
    text = full.base.response_text(response)
    value = full.old.graph_api._parse_json_object(text)
    ids = value.get('ranked_fact_ids') if isinstance(value, dict) else None
    facts = json.loads(request['body']['contents'][0]['parts'][0]['text'].split('Candidate facts JSON:', 1)[1])
    legal = {f['fact_id'] for f in facts}
    n = request['body']['generationConfig']['responseSchema']['properties']['ranked_fact_ids']['minItems']
    valid_list = isinstance(ids, list) and all(isinstance(i, str) for i in ids)
    unknown = sorted(set(ids)-legal) if valid_list else None
    reason = full.recover.finish_reason(response)
    return {'pass': reason == 'STOP' and isinstance(value, dict) and set(value) == {'ranked_fact_ids'}
            and valid_list and len(ids) == n and len(set(ids)) == n and not unknown,
        'finish_reason': reason, 'expected': n, 'count': len(ids) if valid_list else None,
        'unique': len(set(ids)) if valid_list else None, 'unknown_ids': unknown,
        'candidate_count': len(facts), 'selected_ids': ids}


def main():
    with full.old.Lease(RUN):
        request, _, _, prior = full.recover.completed_receipt(SOURCE/'calls/provider/rerank'/BAD_KEY)
        original_check = validate(request, prior['response'])
        if original_check['pass']: raise full.StopRun('Expected original candidate-ID failure')
        seal(RUN/'protocol.json', {'source': str(SOURCE), 'bad_request': BAD_KEY,
            'source_response_sha256': sha(prior), 'original_check': original_check,
            'script_sha256': full.file_sha(__file__), 'exact_same_request': True, 'fresh_generation_limit': 1,
            'no_new_qa_or_judge': True})
        from dotenv import load_dotenv
        load_dotenv(full.old.ROOT/'.env', override=False)
        runtime = full.original.GenerationRuntime(RUN/'calls', allow_model_calls=True, concurrency=1, timeout=240)
        previous = full.original.adapter.install(runtime.log)
        try:
            response = runtime.post(**request)
            result = {'request_sha256': sha(request), 'original_check': original_check,
                'fresh_check': validate(request, response), 'exact_same_request': True,
                'source_failure_preserved': True, 'utc': full.base.now()}
            seal(RUN/'result.json', result)
            print(json.dumps(result), flush=True)
        finally:
            full.original.adapter.restore(previous)


if __name__ == '__main__': main()

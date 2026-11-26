"""LME ledger adapter to the existing post-run embedding countTokens tool.

Only plan() is adapted. Native token counting, bounded calls, outputs and
diagnostic/pipeline separation are delegated unchanged to the shared tool.
Default/live entry points require all ten LME histories to be complete.
"""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared_embedding_token_diagnostic', HERE/'embedding_token_diagnostic.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def unpack(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def plan(source, *, require_complete=True):
    source = Path(source)
    manifests = sorted(source.glob('summary_*.json'))
    if require_complete:
        if not manifests:
            raise RuntimeError('LME has not emitted a completed summary')
        summary = shared.read(manifests[-1])
        if summary.get('completed') != 10 or summary.get('missing_ids'):
            raise RuntimeError('All ten LME histories must complete before token diagnostics')
    meta = shared.read(HERE/'embedding_metadata.json')
    model_meta = meta.get('response',meta.get('metadata',meta))
    if 'countTokens' not in model_meta.get('supportedGenerationMethods',[]):
        raise RuntimeError('Official metadata does not advertise countTokens')
    items = {}
    for path in sorted((source/'physical_calls').glob('*/*/*/intent.json')):
        intent = shared.read(path)
        if intent['kind'] != 'embedding':
            continue
        if intent['model'] != shared.MODEL:
            raise RuntimeError('Unexpected embedding model in original ledger')
        outpath = path.with_name('outcome.json')
        if not outpath.exists():
            if require_complete:
                raise RuntimeError('Orphaned embedding request must be audited first')
            continue
        outcome = shared.read(outpath)
        request = unpack(path.with_name('request.json.gz'))
        successful = outcome['status']=='provider_response' and outcome.get('http_status')==200
        for entry in request['requests']:
            if entry.get('model') != 'models/'+shared.MODEL:
                raise RuntimeError('Unexpected item model in embedding request')
            content = entry['content']
            key = hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
            item = items.setdefault(key,dict(content=content,multiplicity=0,stages=Counter(),successful_stages=Counter()))
            item['multiplicity'] += 1
            item['stages'][intent['stage']] += 1
            if successful:
                item['successful_stages'][intent['stage']] += 1
    return items,model_meta


def main():
    shared.plan = plan
    return shared.main()


if __name__ == '__main__':
    raise SystemExit(main())

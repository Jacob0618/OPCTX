"""One causal source-only request; diagnose, never automatically retry HTTP4xx."""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from creamem3.experiments.memeye_online_state_20260914 import build, state, intake
from creamem3.experiments.locomo_g35lite_20260908 import runtime as io

ROOT = Path(r'D:\creamem3_cache\online_revision_20260915_r1')


def redact(text):
    for key in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'AIza[\w-]{20,}', '[REDACTED]', text)
    return text[:16000]


def source():
    reg = build.baseline.checked(build.baseline.ROOT/'registration.json')
    task, receipt = sorted((k.split('::', 1)[1], v) for k, v in reg['source_graphs'].items()
                           if k.startswith('online::'))[0]
    path = Path(receipt['path'])
    if io.file_sha(path) != receipt['sha256']:
        raise ValueError('Source Writer hash drift')
    graph = json.loads(path.read_text(encoding='utf-8'))
    step = graph['online_fusion']['steps'][0]
    value, exposure = intake.make_input(graph, state.empty(task), step['session_id'],
        step['allowed_ids']['prior_candidate_fact_ids'])
    return task, receipt, value, exposure


def run(allow=False):
    load_dotenv(Path(__file__).resolve().parents[3]/'.env', override=False)
    task, receipt, value, exposure = source()
    body = build.body(value)
    identity = {'model': io.BACKBONE, 'method': 'generateContent', 'body': body,
                'task': task, 'source': receipt, 'exposure': exposure}
    directory = ROOT/'protocol_probe_v1'
    io.seal(directory/'request.json', identity)
    if not allow:
        print('PREFLIGHT', task, 'current_facts', len(value['current_facts']), 'no model calls', flush=True)
        return
    if (directory/'intent.json').exists():
        raise RuntimeError('Probe already dispatched; inspect receipt before a successor attempt')
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        raise RuntimeError('No configured key')
    io.seal(directory/'intent.json', {'utc': io.now(), 'request_sha256': io.sha(identity)})
    request = urllib.request.Request(f'{io.BASE}/models/{io.BACKBONE}:generateContent',
        data=io.canonical(body).encode(), headers={'Content-Type':'application/json', 'x-goog-api-key':key})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as handle:
            response = json.loads(handle.read().decode())
        io.seal(directory/'response.json', {'response':response, 'elapsed_s':time.perf_counter()-started})
        print('PROBE SUCCESS', [c.get('finishReason') for c in response.get('candidates', [])],
              response.get('usageMetadata', {}), flush=True)
    except urllib.error.HTTPError as exc:
        detail = redact(exc.read().decode('utf-8', errors='replace'))
        io.seal(directory/'error.json', {'status':exc.code, 'body':detail,
            'elapsed_s':time.perf_counter()-started, 'no_automatic_retry':True})
        print('PROBE HTTP', exc.code, detail, flush=True)
        raise RuntimeError(f'HTTP{exc.code}: inspect sealed diagnostic') from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        io.seal(directory/'error.json', {'status':None, 'type':type(exc).__name__,
            'acceptance_unknown':True, 'no_automatic_retry':True})
        raise RuntimeError('Transport failure; inspect diagnostic before retry') from None


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['preflight','call'])
    a = p.parse_args(); run(a.command == 'call')

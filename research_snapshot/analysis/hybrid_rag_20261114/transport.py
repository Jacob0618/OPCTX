"""Task-local audited transports. Never put credentials in experiment records."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from analysis.opctx_random_links_20261113 import transport as original

ExecutionStop = original.ExecutionStop
save, digest = original.save, original.digest
QWEN_URL = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'
original.STAGES['lme_judge'] = 'judge'


class AuditTransport(original.AuditTransport):
    def __init__(self, output):
        super().__init__(output)
        self.qwen_success = {}
        self.qwen_attempts = {}
        self.qwen_calls = []
        self.logical_calls = []
        for path in sorted((self.output / 'logical_calls').glob('*.json')):
            self.logical_calls.append(json.loads(path.read_text(encoding='utf8')))
        for path in sorted((self.output / 'qwen_calls').glob('*/request.json')):
            folder = path.parent
            if not (folder / 'outcome.json').exists():
                raise ExecutionStop('Unresolved Qwen request; inspect before resume')
            req = json.loads(path.read_text(encoding='utf8'))
            out = json.loads((folder / 'outcome.json').read_text(encoding='utf8'))
            key = digest(['qwen3-rerank', req['request']])
            if key != out['request_key']:
                raise ExecutionStop('Qwen request/outcome hash mismatch')
            self.qwen_calls.append(out)
            self.qwen_attempts[key] = self.qwen_attempts.get(key, 0) + out['physical_calls']
            if out['status'] == 'ok':
                raw = json.loads((folder / 'response.json').read_text(encoding='utf8'))
                if digest(raw) != out['response_sha256']:
                    raise ExecutionStop('Qwen response hash mismatch')
                self.qwen_success[key] = raw
            elif not out.get('retry_permitted'):
                raise ExecutionStop('Prior terminal Qwen failure requires inspection')

    def _logical(self, model, key, raw, reused, stage):
        value = dict(model=model, request_key=key, stage=stage,
            scope=str(getattr(self.local, 'scope', 'shared')),
            response_sha256=digest(raw), reused_exact_response=reused,
            usage=raw.get('usageMetadata') or raw.get('usage'),
            time_utc=datetime.now(timezone.utc).isoformat())
        save(self.output / 'logical_calls' / (uuid.uuid4().hex + '.json'), value)
        with self.lock:
            self.logical_calls.append(value)

    def request(self, url, payload):
        model, kind, _ = self._endpoint(url, payload)
        key = digest([model, payload])
        with self.lock:
            reused = key in self.raw_success
        raw = super().request(url, payload)
        self._logical(model, key, raw, reused, getattr(self.local, 'stage', kind))
        return raw

    def qwen_post(self, endpoint, payload):
        if endpoint != QWEN_URL or payload.get('model') != 'qwen3-rerank':
            self.stop.set()
            raise ExecutionStop('Unexpected Qwen endpoint/model')
        key = digest(['qwen3-rerank', payload])
        with self._request_lock(key):
            if self.stop.is_set():
                raise ExecutionStop('Stopped before Qwen request')
            with self.lock:
                hit = copy.deepcopy(self.qwen_success.get(key))
            if hit is not None:
                self._logical('qwen3-rerank', key, hit, True, 'qwen_rerank')
                return hit
            while self.qwen_attempts.get(key, 0) < 2:
                if self.stop.is_set():
                    raise ExecutionStop('Stopped before Qwen request')
                if shutil.disk_usage(self.output.anchor).free < 15 * 1024**3:
                    self.stop.set()
                    raise ExecutionStop('Disk free below 15 GiB')
                credential = os.environ.get('DASHSCOPE_API_KEY')
                if not credential:
                    raise ExecutionStop('Configured DASHSCOPE_API_KEY is absent')
                attempt = self.qwen_attempts.get(key, 0) + 1
                folder = self.output / 'qwen_calls' / (str(time.time_ns()) + '_' + uuid.uuid4().hex[:8])
                save(folder / 'request.json', dict(endpoint=endpoint, request=payload,
                    scope=str(getattr(self.local, 'scope', 'shared'))))
                out = dict(request_key=key, stage='qwen_rerank', kind='rerank',
                    model='qwen3-rerank', physical_calls=1, logical_attempt=attempt,
                    scope=str(getattr(self.local, 'scope', 'shared')), status='pending',
                    retry_permitted=False, usage=None, started_utc=datetime.now(timezone.utc).isoformat())
                self.qwen_attempts[key] = attempt
                start = time.perf_counter()
                retry, raw = False, None
                try:
                    response = self.network.client().post(endpoint,
                        content=json.dumps(payload, ensure_ascii=False).encode('utf8'),
                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + credential})
                    out['http_status'] = response.status_code
                    if not 200 <= response.status_code < 300:
                        try:
                            error=response.json()
                        except ValueError:
                            error={'code':'non_json_error','message':response.text[:2000]}
                        error={k:error.get(k) for k in ('code','message','request_id') if error.get(k) is not None}
                        clean=json.dumps(error,ensure_ascii=False)
                        for name in ('GEMINI_API_KEY','DASHSCOPE_API_KEY'):
                            secret=os.environ.get(name)
                            if secret:
                                clean=clean.replace(secret,'[REDACTED]')
                        save(folder/'error.json',json.loads(clean))
                        permission_or_safety=any(term in clean.lower() for term in
                            ('invalidapikey','accessdenied','permission','insufficientbalance',
                             'insufficient_quota','arrea','safety','prohibited','sensitivecontent'))
                        retry = (not permission_or_safety and
                                 response.status_code in (429, 500, 502, 503, 504) and attempt < 2)
                        out.update(status='http_error', retry_permitted=retry,
                                   possible_indeterminate_billing=True,
                                   permission_or_safety_stop=permission_or_safety)
                        if not retry:
                            raise ExecutionStop('Qwen HTTP failure; inspect preserved evidence')
                    else:
                        raw = response.json()
                        save(folder / 'response.json', raw)
                        out.update(usage=raw.get('usage'), response_sha256=digest(raw))
                        if raw.get('code') or not isinstance(raw.get('output', {}).get('results'), list):
                            raise ExecutionStop('Malformed Qwen response')
                        out['status'] = 'ok'
                        with self.lock:
                            self.qwen_success[key] = copy.deepcopy(raw)
                except (self.network.httpx.ConnectError, self.network.httpx.ConnectTimeout) as exc:
                    unsafe = any(x in str(exc).lower() for x in
                                 ('certificate', 'ssl', 'tls', 'permission', 'access denied', 'forbidden'))
                    retry = not unsafe and attempt < 2
                    out.update(status='pre_send_connection_failure', retry_permitted=retry,
                               tls_or_permission=unsafe, possible_indeterminate_billing=False)
                    if not retry:
                        self.stop.set()
                        raise ExecutionStop('Qwen connection failure; inspect before resume') from None
                except BaseException as exc:
                    if out['status'] == 'pending':
                        out.update(status='stopped', error_type=type(exc).__name__,
                                   possible_indeterminate_billing=True)
                    self.stop.set()
                    raise ExecutionStop('Qwen stopped; inspect ledger before resume') from None
                finally:
                    out['service_seconds'] = time.perf_counter() - start
                    save(folder / 'outcome.json', out)
                    with self.lock:
                        self.qwen_calls.append(out)
                if raw is not None:
                    self._logical('qwen3-rerank', key, raw, False, 'qwen_rerank')
                    return raw
                if retry and self.stop.wait(1):
                    raise ExecutionStop('Stopped before bounded Qwen retry')
            raise ExecutionStop('Qwen exact request attempt budget exhausted')


def make_clients(transport):
    chat, embed = original.make_clients(transport)

    def audited_chat(prompt, **kwargs):
        # Exact native-response reuse remains audited by request SHA. Avoid the
        # old per-text client cache bypassing our logical stage receipts.
        kwargs['use_cache'] = False
        return chat(prompt, **kwargs)

    def audited_embed(texts, **kwargs):
        kwargs['use_cache'] = False
        return embed(texts, **kwargs)

    return audited_chat, audited_embed


def preflight(output, include_qwen=False):
    from creamem3.experiments.memeye_resume_20260921 import network
    receipts = []
    if not os.environ.get('GEMINI_API_KEY'):
        raise ExecutionStop('Configured GEMINI_API_KEY is absent')
    for model in ('gemini-2.5-flash', 'gemini-3-flash-preview', 'gemini-embedding-001'):
        response = network.client().get(original.BASE + '/models/' + model,
            headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']}, timeout=30)
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get('error', {})
        value = dict(model=model, http_status=response.status_code, returned_name=body.get('name'),
            error_status=error.get('status'),
            error_reasons=[x.get('reason') for x in error.get('details', []) if x.get('reason')])
        receipts.append(value)
        if response.status_code != 200:
            save(Path(output) / ('preflight_failed_' + str(time.time_ns()) + '.json'), receipts)
            raise ExecutionStop('Configured Gemini credential/model preflight failed; no fallback credential')
    if include_qwen and not os.environ.get('DASHSCOPE_API_KEY'):
        raise ExecutionStop('Configured DASHSCOPE_API_KEY is absent')
    save(Path(output) / ('preflight_passed_' + str(time.time_ns()) + '.json'), receipts)
    return receipts

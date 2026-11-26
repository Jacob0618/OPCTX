"""Same configured Gemini proxy, bounded backoff, immutable physical receipts."""
import http.client
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from creamem3.experiments.locomo_top5_budget_20260910 import run as prior

read, seal, sha, StopRun = prior.read, prior.seal, prior.sha, prior.StopRun
HOST = 'generativelanguage.googleapis.com'
MAX_ATTEMPTS = 12
WIRE_CONCURRENCY = 8
MODELS = {'rerank':'gemini-2.5-flash','qa':'gemini-2.5-flash','judge':'gemini-3-flash-preview',
          'online':'gemini-2.5-flash','offline':'gemini-2.5-flash','offline_format_repair':'gemini-2.5-flash',
          'embedding':'gemini-embedding-001'}


def error_details(exc):
    reason = getattr(exc, 'reason', exc)
    return {'type': type(exc).__name__, 'reason_type': type(reason).__name__,
            'errno': getattr(reason, 'errno', None), 'winerror': getattr(reason, 'winerror', None),
            'ssl_reason': getattr(reason, 'reason', None) if isinstance(reason, ssl.SSLError) else None}


class SameConfiguredProxy(urllib.request.ProxyHandler):
    """The bypass decision is checked once, not DNS-resolved for every request."""
    def __init__(self, proxy):
        self.expected_proxy = proxy
        super().__init__({'https': proxy})

    def proxy_open(self, req, proxy, kind):
        if req.type != 'https' or req.host != HOST or proxy != self.expected_proxy:
            raise StopRun('Unexpected destination/proxy')
        p = urlsplit(proxy)
        if p.scheme != 'http' or p.hostname != '127.0.0.1' or p.port != 7897 or p.username or p.password:
            raise StopRun('Configured proxy contract changed')
        # Run at the normal handler phase, after Host header initialization.
        req.set_proxy(p.netloc, p.scheme)
        return None


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise StopRun('Unexpected API redirect; do not forward credentials')


class ConnectionControl:
    def __init__(self, *, live, stopped=None):
        self.stopped = stopped or threading.Event()
        self.gate = threading.BoundedSemaphore(WIRE_CONCURRENCY)
        self.lock = threading.Lock()
        self.cool_until = 0.0
        self.local = threading.local()
        self.proxy = None
        if live:
            self.proxy = urllib.request.getproxies().get('https')
            p = urlsplit(self.proxy or '')
            if (p.scheme, p.hostname, p.port, p.username, p.password) != ('http', '127.0.0.1', 7897, None, None):
                raise StopRun('Expected existing local HTTPS proxy is unavailable')
            if urllib.request.proxy_bypass(HOST):
                raise StopRun('Existing configuration bypasses proxy; refuse route change')

    def cooldown(self, attempt, key, retry_after=0):
        delay = max(min(60, 2**min(attempt, 6)) + int(key[:4], 16)%1000/1000, min(300, retry_after))
        with self.lock: self.cool_until = max(self.cool_until, time.monotonic()+delay)
        return delay

    def enter(self):
        while not self.stopped.is_set():
            with self.lock: delay = self.cool_until-time.monotonic()
            if delay > 0:
                self.stopped.wait(min(1, delay)); continue
            if self.gate.acquire(timeout=1):
                with self.lock: ready = self.cool_until <= time.monotonic()
                if ready and not self.stopped.is_set(): return
                self.gate.release()
        raise StopRun('Dispatch stopped before request')

    def open(self, request, timeout):
        if urllib.request.getproxies().get('https') != self.proxy:
            raise StopRun('Proxy configuration changed during run')
        if not hasattr(self.local, 'opener'):
            self.local.opener = urllib.request.build_opener(SameConfiguredProxy(self.proxy), RejectRedirect())
        return self.local.opener.open(request, timeout=timeout)


class Runtime(prior.Runtime):
    def __init__(self, root, *, allow_model_calls, control, forbid_rerank=False):
        super().__init__(root, allow_model_calls=allow_model_calls, stopped=control.stopped)
        self.control = control
        self.forbid_rerank = forbid_rerank

    def _result(self, stage, key, response):
        if hasattr(self.local, 'requests'):
            self.local.requests.append({'stage': stage, 'key': key, 'usage': response.get('usageMetadata', {})})
        return response

    def post(self, stage, scope, model, method, body):
        if MODELS.get(stage) != model or method != ('batchEmbedContents' if stage == 'embedding' else 'generateContent') or (self.forbid_rerank and stage == 'rerank'):
            self._halt('Forbidden stage/model/method')
        identity = dict(stage=stage, scope=scope, model=model, method=method, body=body)
        key = sha(identity); directory = self.root/'provider'/stage/key
        with self.mutex: lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            if (directory/'final.json').exists():
                _, _, _, raw = prior.full.recover.completed_receipt(directory, identity)
                self.log({'kind': 'cache_hit', 'stage': stage, 'scope': scope, 'request': key})
                return self._result(stage, key, raw['response'])
            if not self.allow: self._halt('Model calls disabled; missing '+stage+' receipt')
            if self.stopped.is_set(): raise StopRun('Dispatch already stopped')
            seal(directory/'request.json', identity)
            key_value = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not key_value: self._halt('Missing configured Gemini credential')
            for attempt in range(1, MAX_ATTEMPTS+1):
                prefix = directory/('attempt_%02d' % attempt)
                intent, response_path, error_path = (prefix.with_suffix(s) for s in ('.intent.json', '.response.json', '.error.json'))
                if intent.exists():
                    if sum(p.exists() for p in (response_path, error_path)) != 1:
                        self._halt('Orphan/duplicate physical receipt')
                    if read(intent)['request_sha256'] != key: self._halt('Intent identity drift')
                    if response_path.exists():
                        raw = read(response_path)
                        seal(directory/'final.json', {'identity': identity, 'request_sha256': key, 'attempt': attempt, 'response_sha256': sha(raw)})
                        return self._result(stage, key, raw['response'])
                    if not read(error_path)['retryable']: self._halt('Nonretryable recorded failure')
                    continue
                self.control.enter()
                retry_after = 0
                try:
                    with self.mutex:
                        self.physical_count += 1
                        if self.physical_count > 20000: self.stopped.set(); raise StopRun('Physical-call safety ceiling')
                    seal(intent, {'request_sha256': key, 'attempt': attempt, 'utc': prior.base.now()})
                    self.log({'kind': 'physical_start', 'stage': stage, 'scope': scope, 'request': key, 'attempt': attempt, 'model': model})
                    started = time.monotonic()
                    request = urllib.request.Request('https://'+HOST+'/v1beta/models/'+model+':'+method,
                        data=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode(),
                        headers={'Content-Type':'application/json','x-goog-api-key':key_value})
                    try:
                        with self.control.open(request, timeout=self.timeout) as handle:
                            response = json.loads(handle.read().decode('utf-8'))
                        raw = {'response': response, 'elapsed_s': time.monotonic()-started, 'utc': prior.base.now()}
                        seal(response_path, raw)
                        seal(directory/'final.json', {'identity': identity, 'request_sha256': key, 'attempt': attempt, 'response_sha256': sha(raw)})
                        self.log({'kind':'physical_success','stage':stage,'scope':scope,'request':key,'attempt':attempt,'model':model,
                                  'elapsed_s':raw['elapsed_s'],'usage':response.get('usageMetadata',{})})
                        return self._result(stage, key, response)
                    except urllib.error.HTTPError as exc:
                        retryable = exc.code in {429,500,502,503,504}
                        value = exc.headers.get('Retry-After','') if exc.headers else ''
                        retry_after = int(value) if value.isdigit() else 0
                        error = {'status':exc.code,'type':'HTTPError','retryable':retryable}
                    except (urllib.error.URLError, OSError, http.client.IncompleteRead, http.client.RemoteDisconnected) as exc:
                        reason = getattr(exc,'reason',exc)
                        error = {**error_details(exc),'status':None,'retryable':not isinstance(reason,ssl.SSLCertVerificationError),
                                 'acceptance_unknown':True}
                        if isinstance(exc,http.client.IncompleteRead):
                            error['partial_bytes']=len(exc.partial); error['partial_response_not_accepted']=True
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        error = {'status':None,'type':type(exc).__name__,'retryable':False,'malformed_response_not_accepted':True}
                    error.update(elapsed_s=time.monotonic()-started,utc=prior.base.now())
                    seal(error_path,error)
                    self.log({'kind':'physical_error','stage':stage,'scope':scope,'request':key,'attempt':attempt,**error})
                    if not error['retryable']: self._halt('Nonretryable transport failure '+str(error.get('status')))
                finally: self.control.gate.release()
                if attempt < MAX_ATTEMPTS:
                    delay=self.control.cooldown(attempt,key,retry_after)
                    self.log({'kind':'global_retry_cooldown','stage':stage,'request':key,'attempt':attempt,'seconds':delay})
            self._halt('Twelve physical attempts exhausted: '+stage+'/'+scope)

"""Write-once transport with sanitized error bodies; fixed official endpoint."""
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from creamem3.experiments.locomo_g35lite_20260908 import runtime as io
from .probe import redact


class Runtime(io.Runtime):
    def post(self, stage, scope, model, method, body):
        identity = dict(stage=stage, scope=scope, model=model, method=method, body=body)
        key = io.sha(identity); directory = self.root/'provider'/stage/key
        with self.mutex:
            import threading
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            io.seal(directory/'request.json', identity)
            final_path = directory/'final.json'
            if final_path.exists():
                final = io.read(final_path)
                if final['identity'] != identity or final['request_sha256'] != key:
                    self._halt('Cached request identity mismatch')
                raw = io.read(directory/f"attempt_{final['attempt']:02d}.response.json")
                if io.sha(raw) != final['response_sha256']:
                    self._halt('Cached response hash mismatch')
                return raw['response']
            if not self.allow:
                self._halt('Missing request and model calls disabled')
            secret = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not secret:
                self._halt('Missing configured Gemini credential')
            for attempt in range(1, 5):
                prefix = directory/f'attempt_{attempt:02d}'
                intent, rp, ep = [prefix.with_suffix(x) for x in
                                  ('.intent.json','.response.json','.error.json')]
                if intent.exists():
                    if io.read(intent)['request_sha256'] != key or rp.exists() == ep.exists():
                        self._halt('Orphan/ambiguous or mismatched physical receipt')
                    if rp.exists():
                        raw = io.read(rp)
                        io.seal(final_path, dict(identity=identity, request_sha256=key,
                            attempt=attempt, response_sha256=io.sha(raw)))
                        return raw['response']
                    if not io.read(ep)['retryable']:
                        self._halt('Prior nonretryable failure; no automatic retry')
                    continue
                with self.gate:
                    if self.stopped.is_set():
                        raise io.StopRun('Dispatch stopped')
                    with self.mutex:
                        self.physical_count += 1
                        if self.physical_count > 5000:
                            self._halt('Physical-call safety ceiling')
                    io.seal(intent, dict(utc=io.now(), request_sha256=key, attempt=attempt))
                    self.log(dict(kind='physical_start',stage=stage,scope=scope,request=key,attempt=attempt,model=model))
                    started = time.perf_counter()
                    request = urllib.request.Request(f'{io.BASE}/models/{model}:{method}',
                        data=io.canonical(body).encode(), headers={'Content-Type':'application/json','x-goog-api-key':secret})
                    try:
                        with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                            response = json.loads(handle.read().decode('utf-8'))
                    except urllib.error.HTTPError as exc:
                        error = dict(status=exc.code, type='HTTPError', retryable=exc.code in io.TRANSIENT,
                                     body=redact(exc.read().decode('utf-8', errors='replace')))
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        reason = getattr(exc, 'reason', exc)
                        error = dict(status=None, type=type(exc).__name__, acceptance_unknown=True,
                                     retryable=not isinstance(reason,ssl.SSLCertVerificationError))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        error = dict(status=None,type=type(exc).__name__,retryable=False,malformed_response=True)
                    else:
                        raw = dict(response=response,utc=io.now(),elapsed_s=time.perf_counter()-started)
                        io.seal(rp, raw)
                        io.seal(final_path,dict(identity=identity,request_sha256=key,attempt=attempt,response_sha256=io.sha(raw)))
                        self.log(dict(kind='physical_success',stage=stage,scope=scope,request=key,
                                      attempt=attempt,model=model,elapsed_s=raw['elapsed_s'],usage=response.get('usageMetadata',{})))
                        return response
                    error.update(utc=io.now(),elapsed_s=time.perf_counter()-started)
                    io.seal(ep,error)
                    self.log(dict(kind='physical_error',stage=stage,scope=scope,request=key,attempt=attempt,
                                  status=error['status'],retryable=error['retryable'],type=error['type']))
                    if not error['retryable']:
                        self._halt('Nonretryable '+str(error['status'])+'; inspect sanitized receipt')
                if attempt < 4 and self.stopped.wait(min(30,2**attempt)):
                    raise io.StopRun('Dispatch stopped during retry backoff')
            self._halt('Four physical attempts exhausted')

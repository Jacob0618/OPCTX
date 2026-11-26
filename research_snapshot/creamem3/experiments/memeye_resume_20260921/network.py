"""Same native Gemini request, explicit verified CA bundle, durable recovery."""
import base64
from dataclasses import asdict
import os
from pathlib import Path
import ssl
import threading
import time
import urllib.request
import certifi
import httpx
from filelock import FileLock
from ..memeye_visual_baselines_20260920 import native

b=native.b
g=native.g
_local=threading.local()
RECOVERY=None

def client():
    if not hasattr(_local,'client'):
        ctx=ssl.create_default_context(cafile=certifi.where())
        proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or urllib.request.getproxies().get('https')
        _local.client=httpx.Client(verify=ctx,proxy=proxy,trust_env=False,timeout=300)
    return _local.client

class VerifiedProvider:
    def generate_content(self,*,model,request):
        endpoint=os.environ.get('GEMINI_API_BASE',g.GEMINI_API_BASE).rstrip('/')
        if endpoint!=g.GEMINI_API_BASE: raise RuntimeError('Official endpoint changed')
        key=os.environ.get('GEMINI_API_KEY')
        if not key: raise RuntimeError('Gemini credential missing')
        try:
            response=client().post(endpoint+'/models/'+model+':generateContent',
                content=g._canonical_json(request).encode('utf8'),
                headers={'Content-Type':'application/json','x-goog-api-key':key})
        except (httpx.ConnectError,httpx.ConnectTimeout) as exc:
            raise g.ExplicitProviderError('Pre-send connection failure',retryable=True,status_code=None) from exc
        except httpx.TransportError as exc:
            raise g.IndeterminateProviderError('Post-connect transport outcome is indeterminate') from exc
        if not 200<=response.status_code<300:
            raise g.ExplicitProviderError('Gemini HTTP '+str(response.status_code),
                retryable=response.status_code in (408,429,500,502,503,504),status_code=response.status_code)
        return response.json()

def build_request(prompt,segments):
    parts=[{'text':prompt}]
    for seg in segments:
        if 'text' in seg: parts.append({'text':seg['text']}); continue
        ref=seg['image']; blob=Path(ref['path']).read_bytes()
        if b.sha(blob)!=ref['sha256']: raise RuntimeError('Source image changed')
        parts.append({'inlineData':{'mimeType':ref['mime_type'],'data':base64.b64encode(blob).decode('ascii')}})
    return dict(contents=[dict(role='user',parts=parts)],generationConfig=dict(
        temperature=0.0,maxOutputTokens=500,thinkingConfig={'thinkingBudget':0}))

def answer(root,prompt,images_or_segments,metadata,allow):
    request=build_request(prompt,images_or_segments); wire_sha=b.sha(request)
    manifest=dict(model=native.MODEL,prompt=prompt,segments=images_or_segments,metadata=metadata,
        wire_sha256=wire_sha,generation_config=request['generationConfig'])
    key=b.sha(manifest); directory=Path(root)/key[:2]/key
    directory.mkdir(parents=True,exist_ok=True)
    with FileLock(str(directory/'call.lock'),timeout=300):
        target=directory/'response.json'
        if target.exists():
            value=b.read(target)
            if value['wire_sha256']!=wire_sha or not value['text'].strip(): raise RuntimeError('Invalid frozen response')
            return value
        b.save(directory/'request.json',manifest)
        if not allow: raise RuntimeError('Model calls not authorized')
        prior=sorted(directory.glob('attempt_*_request.json'))
        for p in prior:
            outcome=p.with_name(p.name.replace('_request.json','_outcome.json'))
            if not outcome.exists(): raise RuntimeError('Orphaned request requires separate audit: '+str(p))
            receipt=b.read(outcome)
            if receipt.get('ok'): raise RuntimeError('Successful call missing response')
            if not receipt.get('retryable'):
                allowed=RECOVERY['tls_recoveries'].get(str(outcome)) if RECOVERY else None
                if not allowed or b.file_sha(outcome)!=allowed['outcome_sha256']:
                    raise RuntimeError('Unapproved prior failure: '+str(outcome))
                if b.file_sha(allowed['log'])!=allowed['log_sha256']: raise RuntimeError('TLS evidence changed')
        for attempt in range(len(prior)+1,6):
            b.save(directory/f'attempt_{attempt:03d}_request.json',dict(request_sha256=key,time=time.time(),
                recovery_registration_sha256=RECOVERY['seal_sha256']))
            try:
                raw=VerifiedProvider().generate_content(model=native.MODEL,request=request)
            except g.ExplicitProviderError as exc:
                b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=False,retryable=exc.retryable,status=exc.status_code,time=time.time()))
                if not exc.retryable or attempt==5: raise
                time.sleep(min(2**attempt,16)); continue
            except BaseException as exc:
                b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=False,retryable=False,error_type=type(exc).__name__,indeterminate=True,time=time.time()))
                raise
            text=g._normalize_text(raw).strip()
            b.save(directory/f'attempt_{attempt:03d}_provider.json',dict(response=raw))
            if not text:
                b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=False,retryable=False,error_type='EmptyResponse',time=time.time()))
                raise RuntimeError('Empty/safety-blocked QA response; no automatic retry')
            result=b.save(target,dict(text=text,usage=asdict(g._normalize_usage(raw)),response=raw,
                request_sha256=key,wire_sha256=wire_sha))
            b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=True,retryable=False,time=time.time()))
            return result
        raise RuntimeError('Five-attempt budget exhausted')

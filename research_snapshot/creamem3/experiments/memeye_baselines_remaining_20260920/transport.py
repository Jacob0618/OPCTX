"""Durable Gemini OpenAI-compatible transport for upstream internal calls.

Final QA/Judge use the pre-existing native runtimes, not this transport.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import threading
import time
import types
import openai
from filelock import FileLock
from ..memeye_baselines_20260920 import run as base

ORIGINAL_OPENAI=openai.OpenAI
MODEL='gemini-2.5-flash-lite'
ENDPOINT='https://generativelanguage.googleapis.com/v1beta/openai/'
_active=None
_scope=threading.local()

class UpstreamCallFailure(BaseException): pass

@contextmanager
def scope(name):
    previous=getattr(_scope,'value','construction')
    _scope.value=name
    try: yield
    finally: _scope.value=previous

class DurableClient:
    def __init__(self,*args,**kwargs):
        if _active is None: raise RuntimeError('No task transport configured')
        self.backend=_active
        self.chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
    def create(self,**kwargs): return self.backend.create(**kwargs)
    def close(self): pass

class Backend:
    def __init__(self,root,method,task,allow_calls=False):
        self.root=Path(root); self.method=method; self.task=task; self.allow_calls=allow_calls
        self.client=None; self.lock=threading.RLock()
    def create(self,**kwargs):
        req=copy.deepcopy(kwargs)
        if req.get('stream'): raise UpstreamCallFailure('Streaming not supported in fixed baseline adapter')
        original_model=req.get('model',MODEL); req['model']=MODEL
        req.setdefault('max_tokens',8192)
        if 'max_completion_tokens' in req: req.pop('max_tokens',None)
        payload=dict(endpoint=ENDPOINT,request=req,method=self.method,task=self.task,
            requested_model=original_model,scope=getattr(_scope,'value','construction'))
        key=base.sha(payload); directory=self.root/key[:2]/key
        directory.mkdir(parents=True,exist_ok=True)
        with FileLock(str(directory/'call.lock'),timeout=300):
            target=directory/'response.json'
            if target.exists():
                result=base.read(target)
                if result['request_sha256']!=key: raise UpstreamCallFailure('Response identity mismatch')
                return openai.types.chat.ChatCompletion.model_validate(result['response'])
            base.save(directory/'request.json',payload)
            if not self.allow_calls: raise UpstreamCallFailure('Model call requires explicit authorization')
            prior=list(directory.glob('attempt_*_request.json'))
            # Completed transient errors may be resumed; indeterminate intents may not.
            for p in prior:
                receipt=p.with_name(p.name.replace('_request.json','_outcome.json'))
                if not receipt.exists(): raise UpstreamCallFailure('Unresolved prior request intent: '+str(p))
                outcome=base.read(receipt)
                if not outcome.get('retryable',False): raise UpstreamCallFailure('Prior non-retryable failure')
            with self.lock:
                if self.client is None:
                    key_value=os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
                    if not key_value: raise UpstreamCallFailure('Configured Gemini credential missing')
                    self.client=ORIGINAL_OPENAI(api_key=key_value,base_url=ENDPOINT,timeout=120,max_retries=0)
            for attempt in range(len(prior)+1,6):
                base.save(directory/f'attempt_{attempt:03d}_request.json',dict(request_sha256=key,time=time.time()))
                try:
                    response=self.client.chat.completions.create(**req)
                    if not response.choices: raise ValueError('No choices')
                    choice=response.choices[0]
                    if not choice.message.content and not choice.message.tool_calls: raise ValueError('Empty content and tool calls')
                    result=base.save(target,dict(request_sha256=key,response=response.model_dump(mode='json')))
                    base.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=True,retryable=False,time=time.time()))
                    return response
                except Exception as exc:
                    status=getattr(exc,'status_code',None)
                    retryable=isinstance(exc,(openai.APIConnectionError,openai.APITimeoutError)) or status in (408,429,500,502,503,504)
                    base.save(directory/f'attempt_{attempt:03d}_outcome.json',
                        dict(ok=False,error_type=type(exc).__name__,status=status,retryable=retryable,time=time.time()))
                    if not retryable or attempt==5:
                        raise UpstreamCallFailure(f'{self.method}/{self.task}: {type(exc).__name__}, HTTP {status}; see durable receipt') from exc
                    time.sleep(min(2**attempt,16))
            raise UpstreamCallFailure('Five physical attempts exhausted')

def install(root,method,task,allow_calls):
    global _active
    _active=Backend(root,method,task,allow_calls)
    openai.OpenAI=DurableClient
    return _active

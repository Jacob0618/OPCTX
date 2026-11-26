"""Model-explicit, durable transports for the two approved backbone panels."""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import httpx
from filelock import FileLock
from ..memeye_resume_20260921 import network as n
from ..memeye_baselines_remaining_20260920 import transport as t
from ..memeye_visual_baselines_20260920 import native

b=n.b
MODELS={'g25flash':'gemini-2.5-flash','g31lite':'gemini-3.1-flash-lite'}
JUDGE='gemini-3.5-flash-lite'


def disk_guard(root):
    if shutil.disk_usage(Path(root).anchor).free < 15*1024**3:
        raise RuntimeError('Disk free below 15 GiB: stop new calls; preserve all artifacts')


def packed(path,value):
    """Lossless request archive; repeated image payloads compress on D, not C."""
    path=Path(path); blob=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf8')
    if path.exists():
        if gzip.decompress(path.read_bytes())!=blob: raise RuntimeError('Request archive drift')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:
            handle.write(gzip.compress(blob,mtime=0)); handle.flush(); os.fsync(handle.fileno())


def thinking(model,stage):
    if model==MODELS['g25flash']:
        return {'thinkingBudget':512 if stage in ('writer','view') else 0}
    return {'thinkingLevel':'low' if stage in ('writer','view') else 'minimal'}


def durable(root,manifest,send,validate,max_attempts=5):
    key=b.sha(manifest); directory=Path(root)/key[:2]/key
    directory.mkdir(parents=True,exist_ok=True)
    with FileLock(str(directory/'call.lock'),timeout=600):
        target=directory/'response.json'
        if target.exists():
            result=b.read(target)
            if result['request_sha256']!=key: raise RuntimeError('Response identity drift')
            validate(result['response'])
            return result,True
        packed(directory/'request.json.gz',manifest)
        prior=sorted(directory.glob('attempt_*_request.json'))
        for intent in prior:
            outcome=intent.with_name(intent.name.replace('_request.json','_outcome.json'))
            if not outcome.exists(): raise RuntimeError('Orphaned request requires audit: '+str(intent))
            if not b.read(outcome).get('retryable'): raise RuntimeError('Prior terminal call: '+str(outcome))
        for attempt in range(len(prior)+1,max_attempts+1):
            disk_guard(root)
            b.save(directory/f'attempt_{attempt:03d}_request.json',dict(time=time.time(),request_sha256=key))
            try:
                raw=send()
                b.save(directory/f'attempt_{attempt:03d}_provider.json',dict(response=raw))
                validate(raw)
            except BaseException as exc:
                retryable=isinstance(exc,n.g.ExplicitProviderError) and exc.retryable
                b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=False,time=time.time(),
                    retryable=bool(retryable),error_type=type(exc).__name__,status=getattr(exc,'status_code',None)))
                if not retryable or attempt==max_attempts: raise
                time.sleep(min(2**attempt,16)); continue
            result=b.save(target,dict(request_sha256=key,response=raw,response_sha256=b.sha(raw)))
            b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=True,time=time.time(),retryable=False))
            return result,False
        raise RuntimeError('Explicit attempt budget exhausted')


@dataclass(frozen=True)
class Reply:
    text:str
    request_sha256:str
    response_sha256:str
    model:str
    served_model:str
    usage:dict
    cache_hit:bool=False


class Runtime:
    def __init__(self,root,model,allow):
        if model not in (*MODELS.values(),JUDGE): raise ValueError('Unregistered model')
        self.root,self.model,self.allow=Path(root),model,allow

    def generate(self,stage,prompt,*,system=None,json_mode=False,max_output_tokens=500,
                 max_provider_attempts=5,cache_metadata=None,image_path=None,
                 caption_context_count=None,segments=None,response_schema=None):
        segs=list(segments or [])
        if image_path is not None: segs=[{'image':native.image_ref(image_path)}]
        request=n.build_request(prompt,segs)
        if image_path is not None:
            request['contents'][0]['parts'].reverse()  # Caption image precedes previous-two-caption text.
        gc=request['generationConfig']; gc.update(maxOutputTokens=max_output_tokens,thinkingConfig=thinking(self.model,stage))
        if stage=='caption': gc['mediaResolution']='MEDIA_RESOLUTION_HIGH'
        if json_mode: gc['responseMimeType']='application/json'
        if response_schema is not None: gc['responseJsonSchema']=response_schema
        if system: request['systemInstruction']={'parts':[{'text':system}]}
        manifest=dict(model=self.model,stage=stage,prompt=prompt,system=system,segments=segs,
            generation_config=gc,wire_sha256=b.sha(request),metadata=cache_metadata or {},caption_context_count=caption_context_count)
        def send():
            if not self.allow: raise RuntimeError('Model calls disabled')
            return n.VerifiedProvider().generate_content(model=self.model,request=request)
        def validate(raw):
            candidates=raw.get('candidates') or []
            if not candidates or candidates[0].get('finishReason')!='STOP':
                raise RuntimeError('Non-STOP generation: '+str([c.get('finishReason') for c in candidates]))
            if not n.g._normalize_text(raw).strip(): raise RuntimeError('Empty model answer')
            served=raw.get('modelVersion','')
            if served and not served.startswith(self.model): raise RuntimeError('Served model mismatch: '+served)
            if json_mode:
                obj=json.loads(n.g._normalize_text(raw))
                if response_schema is not None:
                    import jsonschema
                    jsonschema.validate(obj,response_schema)
        result,hit=durable(self.root,manifest,send,validate,max_provider_attempts)
        raw=result['response']
        return Reply(n.g._normalize_text(raw).strip(),result['request_sha256'],result['response_sha256'],
            self.model,raw.get('modelVersion',''),asdict(n.g._normalize_usage(raw)),hit)

    def judge(self,question,gold,prediction,*,cache_metadata=None):
        if self.model!=JUDGE: raise RuntimeError('Judge backbone changed')
        prompt=n.g.build_mgtm_judge_prompt(question,gold,prediction)
        out=self.generate('judge',prompt,json_mode=True,max_output_tokens=400,cache_metadata=cache_metadata)
        return out,n.g.parse_mgtm_judge_response(out.text)


class InternalBackend:
    """Exact upstream tool messages, including Gemini thought signatures."""
    MODEL=None
    def __init__(self,root,method,task,allow_calls=False):
        self.root,self.method,self.task,self.allow_calls=Path(root),method,task,allow_calls
        if self.MODEL not in MODELS.values(): raise RuntimeError('Missing scoped internal model')

    def create(self,**kwargs):
        try: return self._create(**kwargs)
        except BaseException as exc:
            if isinstance(exc,t.UpstreamCallFailure): raise
            raise t.UpstreamCallFailure(f'{self.method}/{self.task}: {type(exc).__name__}: {str(exc)[:200]}') from exc

    def _create(self,**kwargs):
        req=copy.deepcopy(kwargs)
        if req.get('stream'): raise RuntimeError('Streaming unsupported')
        req['model']=self.MODEL; req.setdefault('max_tokens',8192)
        if 'max_completion_tokens' in req: req.pop('max_tokens',None)
        # REST representation of the documented OpenAI-compatibility extension.
        req.pop('reasoning_effort',None)
        cfg=thinking(self.MODEL,'internal')
        cfg={'thinking_budget':cfg['thinkingBudget']} if 'thinkingBudget' in cfg else {'thinking_level':cfg['thinkingLevel']}
        extra=req.pop('extra_body',{})
        if 'extra_body' in extra: extra=extra['extra_body']
        extra.setdefault('google',{})['thinking_config']=cfg
        req['extra_body']=extra
        manifest=dict(endpoint=t.ENDPOINT,model=self.MODEL,request=req,method=self.method,
            task=self.task,scope=getattr(t._scope,'value','construction'))
        def send():
            if not self.allow_calls: raise RuntimeError('Internal calls disabled')
            credential=os.environ.get('GEMINI_API_KEY')
            if not credential: raise RuntimeError('Configured credential missing')
            try:
                response=n.client().post(t.ENDPOINT+'chat/completions',json=req,
                    headers={'Authorization':'Bearer '+credential})
            except (httpx.ConnectError,httpx.ConnectTimeout) as exc:
                raise n.g.ExplicitProviderError('Pre-send connection failure',retryable=True) from exc
            except httpx.TransportError as exc:
                raise n.g.IndeterminateProviderError('Indeterminate internal call') from exc
            if not 200<=response.status_code<300:
                raise n.g.ExplicitProviderError('Gemini internal HTTP '+str(response.status_code),
                    retryable=response.status_code in (408,429,500,502,503,504),status_code=response.status_code)
            return response.json()
        def validate(raw):
            choices=raw.get('choices') or []
            if not choices: raise RuntimeError('No provider choices')
            c=choices[0]; msg=c.get('message')
            if not isinstance(msg,dict): raise RuntimeError('Missing provider message: '+str(c.get('finish_reason')))
            if c.get('finish_reason') not in ('stop','tool_calls'): raise RuntimeError('Non-STOP internal response: '+str(c.get('finish_reason')))
            if not msg.get('content') and not msg.get('tool_calls') and not (req.get('tools') and c.get('finish_reason')=='stop'):
                raise RuntimeError('Empty non-tool response')
        result,_=durable(self.root,manifest,send,validate)
        raw=copy.deepcopy(result['response'])
        for c in raw['choices']: c['message']={k:v for k,v in c['message'].items() if v is not None}
        return t.openai.types.chat.ChatCompletion.model_validate(raw)


class VerifiedEmbeddingProvider:
    def __init__(self):
        from creamem3 import memeye_official_embedding_batch_runtime_v19_1 as e
        self.e=e; self.transport='direct_rest_google_ai_studio'; self.vertexai=False
        self.base_url=e.OFFICIAL_ENDPOINT; self.api_version='v1beta'; self.operation=e.OPERATION
        self.auth='x-goog-api-key_header'; self.retry_attempts=1; self.redirects_allowed=False
    def batch_embed_contents(self,*,request):
        e=self.e; e.assert_batch_embed_contents_request(request)
        body=e._canonical_json(request).encode('utf8')
        try:
            response=n.client().post(e.expected_url(),content=body,
                headers={'Content-Type':'application/json','x-goog-api-key':os.environ['GEMINI_API_KEY']})
        except (httpx.ConnectError,httpx.ConnectTimeout) as exc:
            raise e.ExplicitProviderError('Embedding pre-send failure',retryable=True) from exc
        except httpx.TransportError as exc:
            raise e.IndeterminateProviderError('Indeterminate embedding call') from exc
        if not 200<=response.status_code<300:
            raise e.ExplicitProviderError('Embedding HTTP '+str(response.status_code),
                retryable=response.status_code in (408,429,500,502,503,504),status_code=response.status_code)
        return e.ProviderEnvelope(payload=response.json(),raw_response_body=response.content,transport_receipt=dict(
            route=e.route_contract(),url=e.expected_url(),method='POST',request_body_sha256=b.sha(body),
            response_body_sha256=b.sha(response.content),http_status=response.status_code,
            physical_http_attempts=1,redirects_followed=0))

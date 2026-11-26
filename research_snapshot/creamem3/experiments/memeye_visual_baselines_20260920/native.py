"""Native multimodal QA, with durable image-referenced requests and no gold."""
from dataclasses import asdict
import base64
import mimetypes
from pathlib import Path
import time
from filelock import FileLock
from ..memeye_baselines_20260920 import run as b
from creamem3 import memeye_g35_runtime_v19 as g

MODEL='gemini-2.5-flash-lite'

def visual_prompt():
    text=b.authorities()[2]['open']
    replacements={
        'a text-plus-caption memory benchmark setting':'a multimodal memory benchmark setting',
        'using the provided conversation history and image captions only':'using the provided memory text and attached images only',
        'ground every answer in the provided memory text':'ground every answer in the provided memory text and attached images',
        'do not assume access to raw image pixels; treat memories as the only visual evidence':'use only the attached images and memory text as visual evidence',
        'not stated in the dialogue, captions, or memories':'not available in the dialogue, captions, memories, or attached images',
    }
    for old,new in replacements.items():
        if old not in text: raise RuntimeError('Text-QA template drift: '+old)
        text=text.replace(old,new)
    return text

def image_ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=b.file_sha(path),mime_type=mimetypes.guess_type(path.name)[0] or 'image/jpeg')

def data_image_ref(data_url,root):
    import os
    header,data=data_url.split(',',1)
    if not header.startswith('data:image/') or ';base64' not in header: raise RuntimeError('Unsupported visual payload')
    blob=base64.b64decode(data,validate=True); digest=b.sha(blob)
    path=Path(root)/'image_blobs'/digest
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if b.file_sha(path)!=digest: raise RuntimeError('Image blob drift')
    else:
        try:
            with path.open('xb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
        except FileExistsError:
            if b.file_sha(path)!=digest: raise
    return dict(path=str(path),sha256=digest,mime_type=header[5:].split(';')[0])

def text_segment(text): return dict(text=text)
def image_segment(ref): return dict(image=ref)

def answer(root,prompt,images_or_segments,metadata,allow):
    root=Path(root)
    parts=[{'text':prompt}]
    for seg in images_or_segments:
        if 'text' in seg: parts.append({'text':seg['text']}); continue
        ref=seg['image']; path=Path(ref['path'])
        blob=path.read_bytes()
        if b.sha(blob)!=ref['sha256']: raise RuntimeError('Image changed since selection')
        parts.append({'inlineData':{'mimeType':ref['mime_type'],'data':base64.b64encode(blob).decode('ascii')}})
    request=dict(contents=[dict(role='user',parts=parts)],generationConfig=dict(
        temperature=0.0,maxOutputTokens=500,thinkingConfig={'thinkingBudget':0}))
    wire_sha=b.sha(request)
    manifest=dict(model=MODEL,prompt=prompt,segments=images_or_segments,metadata=metadata,
        wire_sha256=wire_sha,generation_config=request['generationConfig'])
    key=b.sha(manifest); directory=root/key[:2]/key
    directory.mkdir(parents=True,exist_ok=True)
    with FileLock(str(directory/'call.lock'),timeout=300):
        target=directory/'response.json'
        if target.exists(): return b.read(target)
        b.save(directory/'request.json',manifest)
        if not allow: raise RuntimeError('Visual QA requires explicit model-call authorization')
        prior=list(directory.glob('attempt_*_request.json'))
        for p in prior:
            out=p.with_name(p.name.replace('_request.json','_outcome.json'))
            if not out.exists(): raise RuntimeError('Unresolved visual QA intent: '+str(p))
            receipt=b.read(out)
            if not receipt.get('retryable',False): raise RuntimeError('Unrecovered/nonretryable prior QA failure')
        provider=g._default_provider_factory()
        for attempt in range(len(prior)+1,6):
            b.save(directory/f'attempt_{attempt:03d}_request.json',dict(request_sha256=key,time=time.time()))
            try:
                raw=provider.generate_content(model=MODEL,request=request)
            except g.ExplicitProviderError as exc:
                b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=False,retryable=exc.retryable,status=exc.status_code,time=time.time()))
                if not exc.retryable or attempt==5: raise
                time.sleep(min(2**attempt,16)); continue
            except BaseException as exc:
                b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=False,retryable=False,error_type=type(exc).__name__,indeterminate=True,time=time.time()))
                raise
            text=g._normalize_text(raw).strip()
            result=b.save(target,dict(text=text,usage=asdict(g._normalize_usage(raw)),response=raw,request_sha256=key,wire_sha256=wire_sha))
            b.save(directory/f'attempt_{attempt:03d}_outcome.json',dict(ok=True,retryable=False,time=time.time()))
            if not text: raise RuntimeError('Empty visual QA response')
            return result
        raise RuntimeError('QA attempts exhausted')

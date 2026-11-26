"""Fixed official MemEye text-baseline reproductions with OpCtx QA/Judge."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import uuid

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
VENDOR=HERE/'vendor/MemEye'
ROOT=Path(r'D:\creamem3_cache\memeye_baselines_20260920_r1')
IMAGE_ROOT=Path(r'C:\Users\86139\Desktop\CreaMem-AutoResearch\external_memeye_data\data\image')
EMBED_ROOT=Path(r'C:\Users\86139\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf')
METHODS=('fc_text','srag_text','a_mem')
SCHEMA='memeye.official_text_baselines.20260920.v1'
EXPECTED_QA='ac3e331abe87fcc87804a7cd8fd732f7fb5e2bb6b440083be04cef90cde333cb'
EXPECTED_JUDGE='5471df5449fc3a74918ad85cd8c8ee58621a40784a2b1c97f3bff4af610bd952'
_encoder=None

def sha(value):
    if not isinstance(value,bytes):
        value=(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)).encode('utf8')
    return hashlib.sha256(value).hexdigest()

def file_sha(path): return sha(Path(path).read_bytes())

def read(path):
    value=json.loads(Path(path).read_text(encoding='utf8'))
    if isinstance(value,dict) and 'seal_sha256' in value:
        check=dict(value); observed=check.pop('seal_sha256')
        if sha(check)!=observed: raise RuntimeError('Artifact hash mismatch: '+str(path))
    return value

def save(path,value):
    value=dict(value); value['seal_sha256']=sha(value)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value: raise RuntimeError('Immutable output drift: '+str(path))
    else:
        with path.open('x',encoding='utf8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    return value

def imports():
    if str(VENDOR) not in sys.path: sys.path.insert(0,str(VENDOR))
    # The upstream __init__ eagerly imports every baseline and downloads NLTK
    # data. Load only the unchanged modules needed for these three methods.
    if 'benchmark' not in sys.modules:
        package=types.ModuleType('benchmark')
        package.__path__=[str(VENDOR/'benchmark')]
        package.__package__='benchmark'
        sys.modules['benchmark']=package
    from benchmark import dataset,methods,embeddings,a_mem
    if not Path(dataset.__file__).resolve().is_relative_to(VENDOR): raise RuntimeError('Wrong benchmark package')
    return dataset,methods,embeddings,a_mem

def authorities():
    from creamem3 import run_memeye_mec50_direct_gemini_v1 as direct
    from creamem3 import official_g25_runtime_adapter_v19 as g25
    a,receipt=direct._reader_authority()
    judge_prompt=direct.g35_runtime.load_mgtm_judge_prompt()
    if sha(a['open'])!=EXPECTED_QA or sha(judge_prompt)!=EXPECTED_JUDGE: raise RuntimeError('Prompt authority drift')
    return direct,g25,a,receipt

def hashes():
    direct,g25,_,_=authorities()
    files=[Path(__file__),HERE/'protocol.md',HERE/'test_run.py',Path(g25.__file__),
        Path(direct.g35_runtime.__file__),Path(direct.retry5.__file__)]
    files.extend(VENDOR/p for p in ('benchmark/dataset.py','benchmark/methods.py','benchmark/retrieval.py',
        'benchmark/embeddings.py','benchmark/a_mem.py','benchmark/a-mem/A-mem/memory_layer.py',
        'benchmark/evaluator.py','config/methods/full_context_text_only.yaml',
        'config/methods/semantic_rag_text_only.yaml','config/methods/a_mem_gemini.yaml'))
    return {str(p):file_sha(p) for p in files}

def register():
    audit=read(ROOT/'sources/audit.json')
    direct,g25,a,receipt=authorities()
    source=direct._canonical_sources(); gold=direct._gold_rows()
    queries=source['query_rows']; clean=[]; paths={}; caption_lengths=[]
    for item in audit['rows']:
        if not item['history_equal'] or not item['qa_order_and_answers_equal']: raise RuntimeError('Dataset version mismatch')
        p=Path(item['public_path'])
        if file_sha(p)!=item['public_sha256']: raise RuntimeError('Public snapshot changed')
        d=read(p); task=item['task']; qas=d['human-annotated QAs']
        for ordinal,q in enumerate(qas,1):
            rid=task+'::'+q.get('question_id',f'Q{ordinal}')
            if gold[rid]['question']!=q['question'].strip() or gold[rid]['gold']!=q['answer'].strip():
                raise RuntimeError('QA/gold binding drift')
        rounds=[r for s in d['multi_session_dialogues'] for r in s['dialogues']]
        if len({r['round'] for r in rounds})!=len(rounds): raise RuntimeError('Duplicate round IDs')
        for r in rounds:
            captions=r.get('image_caption',[])
            if len(captions)!=len(r.get('input_image',[])) or any(not c.strip() for c in captions):
                raise RuntimeError('Missing published captions: '+task)
            caption_lengths.extend(map(len,captions))
        history={k:d[k] for k in ('character_profile','multi_session_dialogues')}
        target=ROOT/'inputs'/f'{task}.json'; save(target,history)
        paths[task]=dict(path=str(target),sha256=file_sha(target),rounds=len(rounds),questions=len(qas),
            source_path=str(p),source_sha256=file_sha(p))
    for q in queries: clean.append({k:q[k] for k in ('row_id','task','question')})
    if len(clean)!=371 or len({r['row_id'] for r in clean})!=371: raise RuntimeError('Require 371 unique questions')
    save(ROOT/'queries.json',dict(rows=clean))
    save(ROOT/'scoring/gold.json',dict(rows=gold))
    embedding_files={str(p):file_sha(p) for p in EMBED_ROOT.rglob('*') if p.is_file()}
    if not embedding_files or not (EMBED_ROOT/'modules.json').is_file(): raise RuntimeError('Incomplete local MiniLM')
    config=dict(schema=SCHEMA,methods=list(METHODS),n=371,tasks=paths,
        source_audit_sha256=file_sha(ROOT/'sources/audit.json'),code_hashes=hashes(),embedding_files=embedding_files,
        qa_model='gemini-2.5-flash-lite',qa_prompt_sha256=EXPECTED_QA,
        judge_model='gemini-3.5-flash-lite',judge_prompt_sha256=EXPECTED_JUDGE,
        qa_max_output=500,amem_json_max_output=2600,qa_temperature=0,amem_temperature=0,
        judge_temperature=0,qa_thinking_budget=0,judge_thinking_level='minimal',
        sr_top_k=10,amem_top_k=10,qa_workers=4,amem_parallel_tasks=4,
        caption_source='public_image_caption_fields_not_verified_GPT52_dense_captions',
        caption_count=len(caption_lengths),mean_caption_chars=sum(caption_lengths)/len(caption_lengths),
        query_sha256=file_sha(ROOT/'queries.json'),gold_sha256=file_sha(ROOT/'scoring/gold.json'),
        official_github_commit=audit['github_commit'],huggingface_revision=audit['huggingface']['sha'])
    reg=save(ROOT/'registration.json',config)
    print('REGISTERED',json.dumps({k:config[k] for k in ('n','methods','caption_count','mean_caption_chars')}),flush=True)
    return reg

def validate():
    reg=read(ROOT/'registration.json')
    if reg['code_hashes']!=hashes(): raise RuntimeError('Code drift; do not mix runs')
    for info in reg['tasks'].values():
        if file_sha(info['path'])!=info['sha256']: raise RuntimeError('Input drift')
    for key,p in (('query_sha256',ROOT/'queries.json'),('gold_sha256',ROOT/'scoring/gold.json')):
        if file_sha(p)!=reg[key]: raise RuntimeError('Question/gold drift')
    return reg

class CachedEncoder:
    """Unchanged MiniLM weights/encoding; cache identical text+normalization."""
    def __init__(self):
        os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(str(EMBED_ROOT),device='cpu',local_files_only=True,trust_remote_code=False)
        self.lock=threading.Lock()
    def get_config_dict(self): return {'model_name':'all-MiniLM-L6-v2'}
    def encode(self,texts,normalize_embeddings=False,**kwargs):
        import numpy as np
        scalar=isinstance(texts,str); texts=[texts] if scalar else list(texts)
        rows=[]
        with self.lock:
            for t in texts:
                key=sha(dict(text=t,normalize=normalize_embeddings,model_revision=EMBED_ROOT.name))
                p=ROOT/'embeddings'/key[:2]/(key+'.npy')
                if p.exists(): v=np.load(p,allow_pickle=False)
                else:
                    v=self.model.encode(t,normalize_embeddings=normalize_embeddings,show_progress_bar=False,**kwargs)
                    p.parent.mkdir(parents=True,exist_ok=True)
                    tmp=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
                    with tmp.open('wb') as f: np.save(f,v,allow_pickle=False)
                    os.replace(tmp,p)
                if v.shape!=(384,) or not np.isfinite(v).all(): raise RuntimeError('Embedding shape/finite failure')
                rows.append(v)
        output=np.stack(rows) if rows else np.empty((0,384),dtype=np.float32)
        return output[0] if scalar else output

def encoder():
    global _encoder
    if _encoder is None: _encoder=CachedEncoder()
    return _encoder

def runtimes(root):
    direct,g25,_,_=authorities(); source=g25._load_source_runtime()
    limits={s:source.StageLimits(cap,20000,5) for s,cap in (('planner',512),('focused',2600),('answer',500),('structured_fused',1200))}
    facade=g25.OfficialGemini25Runtime(root/'provider/g25',allow_model_calls=True,
        provider_factory=lambda:direct.retry5._ReliableRESTProvider(source),stage_limits=limits)
    qa=direct.retry5._ConcurrentG25ReaderBackend(facade)
    judge=direct._judge_backend(root,2000)
    return qa,judge

class NoteFailure(BaseException): pass

class NoteController:
    def __init__(self,runtime,task): self.runtime,self.task,self.scope=runtime,task,'unassigned'
    def get_completion(self,prompt,response_format,temperature=0.7):
        import jsonschema
        try:
            out=self.runtime.generate('focused',prompt,json_mode=True,max_output_tokens=2600,
                cache_metadata=dict(experiment=SCHEMA,method='a_mem',task=self.task,scope=self.scope,
                    response_schema_sha256=sha(response_format),purpose='upstream_note_or_query'))
            obj=json.loads(out.text)
            jsonschema.validate(obj,response_format['json_schema']['schema'])
            return out.text
        except Exception as exc:
            # Upstream catches Exception and silently creates generic metadata.
            # Propagate a failed call instead of changing the effective method.
            raise NoteFailure(f'A-Mem output/transport failure at {self.task}/{self.scope}: {type(exc).__name__}') from exc

def amem_system(task,runtime):
    dataset,methods,embeddings,official=imports()
    sys.path.insert(0,str(VENDOR/'benchmark/a-mem/A-mem'))
    import memory_layer as ml
    if not Path(ml.__file__).resolve().is_relative_to(VENDOR): raise RuntimeError('Wrong A-Mem implementation')
    ctrl=NoteController(runtime,task)
    ml.SentenceTransformer=lambda *args,**kwargs:encoder()
    ml.LLMController=lambda *args,**kwargs:types.SimpleNamespace(llm=ctrl)
    system=ml.AgenticMemorySystem(model_name='all-MiniLM-L6-v2',llm_backend='openai',llm_model='gemini-2.5-flash-lite')
    return system,ctrl,ml,official

def load_dataset(task):
    dataset,_,_,_=imports()
    return dataset.MemoryBenchmarkDataset(ROOT/'inputs'/f'{task}.json',image_root=IMAGE_ROOT)

def frozen_context(path,payload,reg):
    value=save(path,dict(**payload,registration_sha256=reg['seal_sha256']))
    if not value['context'].strip(): raise RuntimeError('Empty context')
    return value

def prepare_contexts(method):
    reg=validate(); queries=read(ROOT/'queries.json')['rows']
    dataset,methods,embeddings,_=imports(); datasets={}; caches={}; rows=[]
    if method=='srag_text':
        shared=encoder()
        def cached_load(self): self._model=shared
        embeddings.TextEmbedder._load=cached_load
    cfg=dict(name='full_context_text_only',context_token_limit=128000,reserved_tokens=1000) if method=='fc_text' else dict(
        name='semantic_rag_text_only',retrieval_backend='dense_text',top_k=10,neighbor_window=0,
        text_embedding_model='all-MiniLM-L6-v2',text_dense_weight=1.,image_dense_weight=0.)
    worker=methods.FullContextTextMethod(cfg) if method=='fc_text' else methods.SemanticRAGTextMethod(cfg)
    for q in queries:
        task=q['task']; path=ROOT/method/'contexts'/(sha(q['row_id'])+'.json')
        if path.exists():
            v=read(path)
            if v['query']!=q or v['registration_sha256']!=reg['seal_sha256']: raise RuntimeError('Context drift')
            rows.append(v); continue
        if task not in datasets: datasets[task]=load_dataset(task)
        if method=='fc_text' and task in caches: history,debug=caches[task]
        else:
            history=worker.build_history(datasets[task],{'question':q['question']})
            debug=json.loads(json.dumps(worker.runtime_info))
            if method=='fc_text': caches[task]=(history,debug)
        if any(h.get('images') for h in history): raise RuntimeError('Images leaked into text baseline')
        context='\n\n'.join(h['role']+': '+h['text'] for h in history)
        ids=list(dict.fromkeys(h['round_id'] for h in history))
        if method=='srag_text' and len(ids)>10: raise RuntimeError('SRAG K drift')
        rows.append(frozen_context(path,dict(query=q,context=context,context_sha256=sha(context),
            selected_round_ids=ids,upstream_runtime=debug),reg))
    print('CONTEXTS',method,len(rows),flush=True)
    return rows

def checkpoint(system,task,done,reg):
    v=dict(schema=SCHEMA,registration_sha256=reg['seal_sha256'],task=task,done=done,evo_cnt=system.evo_cnt,
        memories=[m.__dict__ for m in system.memories.values()],corpus=system.retriever.corpus,
        document_ids=system.retriever.document_ids)
    v['seal_sha256']=sha(v)
    p=ROOT/'a_mem/checkpoints'/task/f'{done:04d}.json.gz'; p.parent.mkdir(parents=True,exist_ok=True)
    b=gzip.compress(json.dumps(v,ensure_ascii=False).encode('utf8'),mtime=0)
    if p.exists():
        if p.read_bytes()!=b: raise RuntimeError('Checkpoint drift')
    else:
        with p.open('xb') as f: f.write(b)

def restore(system,ml,task,reg):
    paths=sorted((ROOT/'a_mem/checkpoints'/task).glob('*.json.gz'))
    if not paths: return 0
    v=json.loads(gzip.decompress(paths[-1].read_bytes())); expected=v.pop('seal_sha256')
    if sha(v)!=expected or v['registration_sha256']!=reg['seal_sha256']: raise RuntimeError('Checkpoint identity drift')
    for attrs in v['memories']:
        m=ml.MemoryNote.__new__(ml.MemoryNote); m.__dict__.update(attrs); system.memories[m.id]=m
    system.evo_cnt=v['evo_cnt']
    system.retriever.corpus=v['corpus']; system.retriever.document_ids=v['document_ids']
    system.retriever.embeddings=encoder().encode(v['corpus'])
    return v['done']

def prepare_amem(task,runtime):
    reg=validate(); system,ctrl,ml,official=amem_system(task,runtime); d=load_dataset(task)
    done=restore(system,ml,task,reg); name=d.data.get('character_profile',{}).get('name','').strip()
    speaker='user ('+name+')' if name else 'user'; count=0
    for sid in d.session_order():
        session=d.get_session(sid); timestamp=session['date']
        for row in session['dialogues']:
            count+=1
            if count<=done: continue
            rid=row['round']; note=official.build_a_mem_note_text(d.rounds[rid],speaker,'assistant')
            if not note.strip(): raise RuntimeError('Empty upstream note')
            ctrl.scope='write/'+rid
            with redirect_stdout(io.StringIO()):
                system.add_note(note,time=timestamp,id=str(uuid.uuid5(uuid.NAMESPACE_URL,task+'/'+rid)),last_accessed=timestamp)
            if len(system.memories)!=count: raise RuntimeError('A-Mem note count mismatch')
            checkpoint(system,task,count,reg)
            if count%5==0: print('AMEM WRITE',task,count,'/',reg['tasks'][task]['rounds'],flush=True)
    if count!=reg['tasks'][task]['rounds']: raise RuntimeError('Round count mismatch')
    agent=official.AMemAgent.__new__(official.AMemAgent)
    agent.memory_system=system; agent.retrieve_k=10; agent.retriever_llm=types.SimpleNamespace(llm=ctrl)
    queries=[q for q in read(ROOT/'queries.json')['rows'] if q['task']==task]; contexts=[]
    for q in queries:
        path=ROOT/'a_mem/contexts'/(sha(q['row_id'])+'.json')
        if path.exists():
            v=read(path)
            if v['query']!=q or v['registration_sha256']!=reg['seal_sha256']: raise RuntimeError('A-Mem context drift')
            contexts.append(v); continue
        ctrl.scope='query/'+q['row_id']; keywords=agent.generate_query(q['question'])
        context=agent.retrieve_memory(keywords)
        indices=[int(i) for i in system.retriever.search(keywords,k=10)]
        contexts.append(frozen_context(path,dict(query=q,context=context,context_sha256=sha(context),
            selected_note_indices=indices,query_keywords=keywords,note_count=len(system.memories),
            rendered_note_occurrences=context.count('talk start time:')),reg))
    return contexts

def evaluate(method,contexts,qa,judge,workers=4):
    reg=validate(); direct,_,authority,_=authorities()
    # Gold is accessed only by the scoring stage, after the prediction is durable.
    def one(item):
        q=item['query']; rid=q['row_id']; key=sha(rid)
        pred_path=ROOT/method/'predictions'/(key+'.json')
        prompt=authority['open'].format(context=item['context'],question=q['question'])
        if pred_path.exists(): pred=read(pred_path)
        else:
            out=qa.generate('answer',prompt,json_mode=False,max_output_tokens=500,
                cache_metadata=dict(experiment=SCHEMA,method=method,row_id=rid,purpose='qa'))
            if not out.text.strip(): raise RuntimeError('Empty model answer')
            pred=save(pred_path,dict(query=q,prediction=out.text.strip(),prompt_sha256=sha(prompt),
                context_sha256=item['context_sha256'],registration_sha256=reg['seal_sha256'],generation=asdict(out)))
        if pred['prompt_sha256']!=sha(prompt) or pred['query']!=q or pred['registration_sha256']!=reg['seal_sha256']:
            raise RuntimeError('Prediction drift')
        path=ROOT/method/'scores'/(key+'.json')
        if path.exists(): return read(path)
        gold=read(ROOT/'scoring/gold.json')['rows'][rid]
        if q['question']!=gold['question']: raise RuntimeError('Gold question mismatch')
        result,verdict=judge.judge(q['question'],gold['gold'],pred['prediction'],
            cache_metadata=dict(experiment=SCHEMA,method=method,row_id=rid,purpose='judge'))
        return save(path,dict(row_id=rid,task=q['task'],x=gold['x'],y=gold['y'],xy=gold['x']+'-'+gold['y'],
            prediction_sha256=sha(pred['prediction']),prediction_file_sha256=file_sha(pred_path),
            gold_sha256=sha(gold['gold']),raw_score=verdict.raw_score,mapped_score=verdict.mapped_score,
            reasoning=verdict.reasoning,registration_sha256=reg['seal_sha256'],judge_generation=asdict(result)))
    outputs=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,item) for item in contexts]
        for f in as_completed(futures):
            outputs.append(f.result())
            if len(outputs)%10==0 or len(outputs)==len(futures): print('EVALUATED',method,len(outputs),'/',len(futures),flush=True)
    return outputs

def grouped(rows,key):
    groups={}
    for r in rows: groups.setdefault(r[key],[]).append(r)
    return {k:dict(n=len(v),raw_percent=100*sum(x['raw_score'] for x in v)/len(v),
        bleu1=sum(x.get('bleu1',0) for x in v)/len(v),
        mapped_percent=100*sum(x['mapped_score'] for x in v)/len(v)) for k,v in sorted(groups.items())}

def official_bleu():
    """Execute exact upstream pure metric functions, not its download side effects."""
    import ast
    import re
    import string
    from nltk.stem import PorterStemmer
    from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
    source=VENDOR/'benchmark/evaluator.py'
    parsed=ast.parse(source.read_text(encoding='utf8'))
    body=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in ('normalize','bleu_score')]
    namespace=dict(re=re,string=string,_ARTICLES=re.compile(r'\b(a|an|the)\s+(?=\w)'),
        _stemmer=PorterStemmer(),SmoothingFunction=SmoothingFunction,sentence_bleu=sentence_bleu,Tuple=tuple)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['bleu_score']

def summarize(method):
    reg=validate(); queries=read(ROOT/'queries.json')['rows']; rows=[]
    bleu=official_bleu(); gold=read(ROOT/'scoring/gold.json')['rows']
    for q in queries:
        p=ROOT/method/'scores'/(sha(q['row_id'])+'.json')
        if not p.exists(): raise RuntimeError('Incomplete '+method)
        r=read(p); pp=ROOT/method/'predictions'/(sha(q['row_id'])+'.json')
        if r['row_id']!=q['row_id'] or r['prediction_file_sha256']!=file_sha(pp): raise RuntimeError('Score identity drift')
        r=dict(r); r['bleu1']=bleu(read(pp)['prediction'],gold[q['row_id']]['gold'],weights=(1,0,0,0))
        rows.append(r)
    byxy=grouped(rows,'xy'); preds=[read(ROOT/method/'predictions'/(sha(q['row_id'])+'.json')) for q in queries]
    ctx=[read(ROOT/method/'contexts'/(sha(q['row_id'])+'.json')) for q in queries]
    result=dict(status='PASS',method=method,n=len(rows),registration_sha256=reg['seal_sha256'],
        mapped_percent=100*sum(r['mapped_score'] for r in rows)/371,raw_percent=100*sum(r['raw_score'] for r in rows)/371,
        macro12_raw_percent=sum(r['raw_percent'] for r in byxy.values())/12,
        macro12_mapped_percent=sum(r['mapped_percent'] for r in byxy.values())/12,
        bleu1=sum(r['bleu1'] for r in rows)/371,
        macro12_bleu1=sum(r['bleu1'] for r in byxy.values())/12,
        by_x=grouped(rows,'x'),by_y=grouped(rows,'y'),by_xy=byxy,by_task=grouped(rows,'task'),
        mean_context_chars=sum(len(c['context']) for c in ctx)/371,
        mean_qa_prompt_tokens=sum(p['generation']['usage']['prompt_tokens'] for p in preds)/371,
        rows_sha256=sha(rows),one_judge_per_answer=True)
    save(ROOT/method/'complete.json',result)
    print('COMPLETE',method,json.dumps({k:result[k] for k in ('n','mapped_percent','raw_percent','macro12_raw_percent')}),flush=True)
    return result

def live(method,task=None,smoke=False):
    from dotenv import load_dotenv
    load_dotenv(REPO/'.env',override=False)
    reg=validate(); runroot=ROOT/method/('tasks/'+task if task else 'runtime')
    qa,judge=runtimes(runroot)
    if method=='a_mem':
        if not task: raise RuntimeError('A-Mem requires a task')
        if smoke:
            system,ctrl,ml,official=amem_system(task,qa); d=load_dataset(task)
            first=[(s['date'],x) for s in d.sessions for x in s['dialogues']][:2]
            for stamp,row in first:
                ctrl.scope='smoke/'+row['round']
                text=official.build_a_mem_note_text(d.rounds[row['round']],'user','assistant')
                with redirect_stdout(io.StringIO()):
                    system.add_note(text,time=stamp,id=str(uuid.uuid5(uuid.NAMESPACE_URL,'smoke/'+task+'/'+row['round'])),last_accessed=stamp)
            if len(system.memories)!=2: raise RuntimeError('A-Mem smoke note count')
            save(runroot/'smoke_pass.json',dict(notes=2,registration_sha256=reg['seal_sha256'],qa=qa.usage_receipt()))
            print('AMEM SMOKE PASS',task,flush=True)
            return
        contexts=prepare_amem(task,qa)
    else: contexts=prepare_contexts(method)
    if smoke: contexts=contexts[:2]
    evaluate(method,contexts,qa,judge,workers=4)
    save(runroot/('smoke_usage.json' if smoke else 'complete_usage.json'),dict(qa=qa.usage_receipt(),judge=judge.usage_receipt()))
    if not task and not smoke: summarize(method)

def run_all():
    reg=validate()
    jobs=[('fc_text',None),('srag_text',None)]+[('a_mem',t) for t in reg['tasks']]
    amem_slots=threading.BoundedSemaphore(4)
    stop=threading.Event()
    def launch_job(job):
        if stop.is_set(): return 99
        method,task=job
        if (ROOT/method/'complete.json').exists(): return 0
        cmd=[sys.executable,'-X','utf8','-B','-m',__package__+'.run','live','--method',method,'--allow-model-calls']
        if task: cmd+=['--task',task]
        name=method+('__'+task if task else ''); log=ROOT/'logs'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('ab') as f:
            p=subprocess.Popen(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(ROOT/'launches'/f'{name}_{p.pid}_{time.time_ns()}.json',dict(pid=p.pid,method=method,task=task,command=cmd))
        print('LAUNCHED',name,p.pid,flush=True)
        code=p.wait()
        if code: stop.set()
        print('EXITED',name,code,flush=True); return code
    def launch(job):
        if job[0]=='a_mem':
            with amem_slots: return launch_job(job)
        return launch_job(job)
    # FC/SRAG consume two slots initially, four independent A-Mem histories the rest.
    with ThreadPoolExecutor(max_workers=6) as pool: codes=list(pool.map(launch,jobs))
    if any(codes): raise RuntimeError('One or more jobs failed; preserve all outputs: '+str(codes))
    result={m:summarize(m) for m in METHODS}
    save(ROOT/'complete.json',dict(status='PASS',methods=result))

def status():
    result={}
    for m in METHODS:
        result[m]=dict(predictions=len(list((ROOT/m/'predictions').glob('*.json'))),
            scores=len(list((ROOT/m/'scores').glob('*.json'))),complete=(ROOT/m/'complete.json').exists())
    result['amem_checkpoints']={p.name:len(list(p.glob('*.json.gz'))) for p in (ROOT/'a_mem/checkpoints').glob('*') if p.is_dir()}
    result['stops']=[p.name for p in (ROOT/'stops').glob('*.json')]
    print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('register','prepare','live','run','summary','status'))
    p.add_argument('--method',choices=METHODS); p.add_argument('--task'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--allow-model-calls',action='store_true'); a=p.parse_args()
    if a.allow_model_calls!=(a.phase in ('live','run')): p.error('Only live/run require --allow-model-calls')
    try:
        if a.phase=='register': register()
        elif a.phase=='prepare': prepare_contexts(a.method)
        elif a.phase=='live': live(a.method,a.task,a.smoke)
        elif a.phase=='run': run_all()
        elif a.phase=='summary': summarize(a.method)
        else: status()
    except BaseException as exc:
        save(ROOT/'stops'/f'{time.time_ns()}_{os.getpid()}.json',dict(phase=a.phase,method=a.method,task=a.task,
            error_type=type(exc).__name__,message=str(exc)[:800]))
        raise

if __name__=='__main__': main()

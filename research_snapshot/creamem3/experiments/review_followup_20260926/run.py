"""Frozen all371 reviewer controls. Prior results stay read-only."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import time
import sys
import numpy as np
from dotenv import load_dotenv
from filelock import FileLock
from creamem3.experiments.review_fast_20260926 import run as old

b,rt,sel,source=old.b,old.rt,old.sel,old.source
ROOT=Path(r'D:\opctx_review26\r2')
HERE=Path(__file__).resolve().parent
ARMS=('bm25','fusion','mmr','generic_reasoning')
GENERIC='''You are an evidence analyst assisting a final answer model. Examine the question using only the selected memory. Write evidence notes preserving the specific details needed to answer, then give a brief explanation connecting that evidence to a supported candidate answer. Distinguish directly supported information from uncertainty, and do not guess missing details. Use ordinary prose rather than a predefined question taxonomy or operation-specific tables. Aim for approximately {words} words across the evidence notes and explanation when the evidence warrants it; do not pad or invent details to reach that length.
Return JSON with three string fields: "evidence_notes", "explanation", "candidate_answer".

QUESTION:
{question}

SELECTED MEMORY:
{context}'''
FUSED='''Answer the question using only the selected memory. First write question-relevant evidence notes, then a brief explanation connecting the evidence to your final answer. Preserve decisive details, distinguish uncertainty, and do not guess missing information. Use ordinary prose rather than a predefined question taxonomy or operation-specific tables. Aim for approximately {words} words across the notes and explanation when warranted; do not pad or invent details to reach that length.
Return JSON with three string fields: "evidence_notes", "explanation", "final_answer".

QUESTION:
{question}

SELECTED MEMORY:
{context}'''

def read_input(path, hashes):
    p=Path(path);value=b.read(p);hashes[str(p)]=b.file_sha(p);return value

def embedding_bank(root,hashes):
    bank={}
    for p in (root/'embedding/responses').glob('*.json'):
        # Filter metadata via plan before opening multi-vector responses.
        pp=root/'embedding/plans'/p.name
        plan=read_input(pp,hashes)
        if plan.get('cache_metadata',{}).get('purpose')!='query_channels':continue
        v=read_input(p,hashes)
        assert v['vectors_sha256']==b.sha(v['vectors'])
        for h,vec in zip(v['full_text_sha256s'],v['vectors'],strict=True):
            assert len(vec)==3072 and np.isfinite(vec).all()
            if h in bank:assert bank[h]==vec
            bank[h]=vec
    return bank

def choose(graph,facts,ranking,budget,dense,similarity=None):
    ids=[f['id'] for f in facts];picked=[];seen=set();available=set(range(len(ids)))
    while available and len(picked)<30:
        if similarity is None:
            i=min(available,key=lambda i:(-float(ranking[i]),-float(dense[i]),ids[i]))
        else:
            values=dense if not picked else .5*dense-.5*np.max(similarity[:,picked],axis=1)
            i=min(available,key=lambda i:(-float(values[i]),-float(dense[i]),ids[i]))
        available.remove(i);f=facts[i];identity=(tuple(sorted(f['sources'])),f['text'])
        if identity in seen:continue
        trial=sel.display_ids([ids[j] for j in picked+[i]],ids,dense)
        text,_=sel.adapter.render_selection(graph,trial)
        if len(text)>budget:continue
        picked.append(i);seen.add(identity)
    result=sel.display_ids([ids[i] for i in picked],ids,dense)
    text,render=sel.adapter.render_selection(graph,result)
    assert 0<len(result)<=30 and len(text)<=budget and len(set(result))==len(result)
    assert not render['rendered_event_view_ids'] and not render['rendered_session_summary_ids']
    return dict(ids=result,context=text,context_sha256=b.sha(text),characters=len(text),render=render)

def prepare():
    if (ROOT/'registration.json').exists():return validate()
    parent=old.validate();hashes={str(old.OUT/'registration.json'):b.file_sha(old.OUT/'registration.json')};rows=[]
    for task in sorted({r['task'] for r in parent['rows']}):
        root=old.oldroot(task);memory=read_input(root/'memory_complete.json',hashes)
        graph=read_input(memory['session']['artifact_path'],hashes)
        assert b.file_sha(memory['session']['artifact_path'])==memory['session']['artifact_sha256']
        facts,groups=sel.adapter.adapt_facts(graph);ids=[f['id'] for f in facts]
        index=source.memory.GraphMemoryIndex(graph,candidate_cap=len(facts),rerank_top_k=50)
        assert ids==[f['id'] for f in index.facts]
        matrix=np.asarray(index.fact_matrix,dtype=float);matrix=matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-15)
        similarity=matrix@matrix.T;bank=embedding_bank(root,hashes)
        for ref in [r for r in parent['rows'] if r['task']==task]:
            began=time.perf_counter();item=read_input(ref['path'],hashes);q=item['query'];key=b.sha(q['row_id'])
            projection=read_input(root/'projections'/f'{key}.json',hashes)['projection']
            queries=sel.query_texts(q['question'],projection,'q0_plus_anchored_attributes')
            vectors={qq['query']:bank[b.sha(qq['query'])] for qq in queries}
            qv=read_input(root/'query_vectors'/f'{key}.json',hashes)['vector']
            assert vectors[q['question']]==qv
            channels=source.channels_for(index,facts,queries,vectors)
            utilities=sel.selector.prepare_channels(q['question'],channels,len(facts))['utilities']
            dense=np.array(channels[0]['dense_scores']);lex=np.array(channels[0]['bm25_scores'])
            selections={a:choose(graph,facts,r,item['budget'],dense,similarity if a=='mmr' else None)
                        for a,r in [('bm25',lex),('fusion',np.mean(list(utilities.values()),axis=0)),('mmr',dense)]}
            selections['generic_reasoning']=dict(ids=item['full_ids'],context=item['full_context'],context_sha256=b.sha(item['full_context']))
            historic=read_input(item['full_prediction_path'],hashes)
            target_tokens=historic['stages'][0]['generation']['usage']['output_tokens']
            data=dict(query=q,route=item['route'],caps=item['caps'],budget=item['budget'],
                selections=selections,channels=channels,candidate_count=len(facts),
                words=max(40,min(600,round(target_tokens*.7))),historical_reader_output_tokens=target_tokens,
                source_prepared=ref['path'],preparation_seconds=time.perf_counter()-began)
            path=ROOT/'prepared'/f'{key}.json';b.save(path,data)
            rows.append(dict(row_id=q['row_id'],task=task,path=str(path),sha256=b.file_sha(path)))
        print('PREPARED',task,len(rows),flush=True)
    assert len(rows)==371
    code={str(p):b.file_sha(p) for p in [HERE/'run.py',HERE/'protocol.md']}
    return b.save(ROOT/'registration.json',dict(rows=rows,arms=ARMS,inputs=hashes,code_hashes=code,
        parent_sha256=parent['seal_sha256'],model=old.MODEL,judge=old.JUDGE,workers=8,
        mmr_lambda=.5,weights='equal active utility channels',gold_blind_preparation=True,
        no_model_calls_in_preparation=True,bootstrap_seed=92602,bootstrap_n=10000,
        generic_prompt_sha=b.sha(GENERIC),generic_fused_sha=b.sha(FUSED)))

def validate():
    reg=b.read(ROOT/'registration.json');old.validate()
    for p,h in reg['inputs'].items():assert b.file_sha(p)==h,('input drift',p)
    for p,h in reg['code_hashes'].items():assert b.file_sha(p)==h,('code drift',p)
    for r in reg['rows']:assert b.file_sha(r['path'])==r['sha256']
    return reg

def score(pred,root,reg):
    key=b.sha(pred['row_id']);gold=b.read(source.SOURCE/'scoring/open.json')['rows'][pred['row_id']]
    assert gold['question']==pred['question'];judge=rt.Runtime(root/'judge_calls',old.JUDGE,True)
    began=time.perf_counter();out,v=judge.judge(pred['question'],gold['gold'],pred['prediction'],
        dict(arm=pred['arm'],row_id=pred['row_id'],prediction_sha256=pred['seal_sha256'],registration_sha256=reg['seal_sha256']))
    b.save(root/'scores'/f'{key}.json',dict(row_id=pred['row_id'],task=pred['task'],x=gold['x'],y=gold['y'],
        raw_score=v.raw_score,mapped_score=v.mapped_score,reasoning=v.reasoning,judge=old.JUDGE,
        generation=asdict(out),elapsed_s=time.perf_counter()-began,prediction_sha256=pred['seal_sha256']))

def one(arm,ref,reg):
    item=b.read(ref['path']);q=item['query'];key=b.sha(q['row_id']);root=ROOT/'runs'/arm
    pp=root/'predictions'/f'{key}.json';sp=root/'scores'/f'{key}.json';fp=root/'failures'/f'{key}.json'
    if sp.exists():assert b.read(sp)['prediction_sha256']==b.read(pp)['seal_sha256'];return 'cached'
    if fp.exists():return 'retained_failure'
    try:
        rt.prior.disk_guard(ROOT)
        if pp.exists():pred=b.read(pp)
        else:
            selection=item['selections'][arm];ctx=selection['context'];stages=[];authority=b.authorities()[2]
            gen=rt.Runtime(root/'calls',old.MODEL,True)
            def call(stage,prompt,json_mode):
                rt.prior.disk_guard(ROOT);start=time.perf_counter()
                out=gen.generate(stage,prompt,json_mode=json_mode,max_output_tokens=item['caps'][stage],
                    cache_metadata=dict(arm=arm,row_id=q['row_id'],registration_sha256=reg['seal_sha256']))
                stages.append(dict(stage=stage,generation=asdict(out),elapsed_s=time.perf_counter()-start));return out
            if item['route']=='two_stage':
                prompt=(GENERIC.format(words=item['words'],question=q['question'],context=ctx) if arm=='generic_reasoning'
                        else authority['focused'].replace('{question}',q['question']).replace('{context}',ctx))
                out=call('focused',prompt,True);parsed=sel.adapter.direct.predict._parse_json_object(out.text)
                assert isinstance(parsed,dict),'Reader JSON'
                if arm=='generic_reasoning':
                    assert all(isinstance(parsed.get(k),str) for k in ('evidence_notes','explanation','candidate_answer'))
                    ctx='AGGREGATED EVIDENCE:\nEVIDENCE: '+parsed['evidence_notes']+'\nEXPLANATION: '+parsed['explanation']+'\nCANDIDATE ANSWER: '+parsed['candidate_answer']
                else:ctx=source.display.rendered_context(authority['render'],parsed,ctx,'candidate_first','candidate_last')
                prompt=authority['open'].format(context=ctx,question=q['question']);out=call('answer',prompt,False);answer=out.text.strip()
            else:
                prompt=(FUSED.format(words=item['words'],question=q['question'],context=ctx) if arm=='generic_reasoning'
                        else authority['fused'].format(context=ctx,question=q['question']))
                out=call('structured_fused',prompt,True);parsed=sel.adapter.direct.predict._parse_json_object(out.text)
                assert isinstance(parsed,dict) and isinstance(parsed.get('final_answer'),str),'Fused JSON'
                answer=parsed['final_answer'].strip()
            assert answer
            pred=b.save(pp,dict(row_id=q['row_id'],task=q['task'],question=q['question'],arm=arm,prediction=answer,
                model=old.MODEL,route=item['route'],stages=stages,context_sha256=selection['context_sha256'],
                selected_fact_ids=selection['ids'],qa_prompt_sha256=b.sha(prompt),registration_sha256=reg['seal_sha256']))
        score(pred,root,reg);print('DONE',arm,q['row_id'],flush=True);return 'done'
    except Exception as e:
        b.save(fp,dict(row_id=q['row_id'],arm=arm,time=time.time(),error_type=type(e).__name__,message=str(e)[:1500]))
        print('FAILED',arm,q['row_id'],type(e).__name__,str(e)[:120],flush=True);return 'failed'

def execute(smoke):
    reg=validate();load_dotenv(Path.cwd()/'.env',override=False)
    if smoke:
        chosen={}
        for r in reg['rows']:chosen.setdefault(b.read(r['path'])['route'],r)
        refs=list(chosen.values())
    else:assert (ROOT/'smoke_pass.json').exists();refs=reg['rows']
    with FileLock(str(old.OUT/'runner.lock'),timeout=0),FileLock(str(ROOT/'runner.lock'),timeout=0):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=[j.result() for j in as_completed([pool.submit(one,a,r,reg) for r in refs for a in ARMS])]
        result=dict(time=time.time(),n=len(results),counts={v:results.count(v) for v in set(results)})
        p=ROOT/'smoke_pass.json' if smoke and all(v in ('done','cached') for v in results) else ROOT/'executions'/f'{time.time_ns()}.json'
        b.save(p,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':print('REGISTERED',len(prepare()['rows']))
    else:execute(sys.argv[1]=='smoke')

"""Bounded matched MemEye supplements; original experiment is read-only."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time
from dotenv import load_dotenv
from filelock import FileLock
from creamem3.experiments.memeye_lite_ablation_20260923 import run as source

b, rt, sel = source.b, source.rt, source.selection
HERE = Path(__file__).resolve().parent
OUT = Path(r'D:\opctx_review26\r1')
OLD = Path(r'D:\me_ablate23\r1')
ARMS = ('full_refresh', 'dense30', 'generic_reader')
MODEL = 'gemini-3.5-flash-lite'
JUDGE = 'gemini-3.5-flash'
GENERIC = '''Read the selected memory below and summarize the evidence relevant to the question. Preserve the specific entities, attributes, quantities, dates and relations stated in the memory. Include complementary relevant details, and identify missing or conflicting information rather than guessing. Use only the provided memory. Return a JSON object with one field, "summary", containing a concise evidence summary in ordinary prose. Do not classify the question, choose an operation-specific schema, or produce an answer proposal.

Question: {question}
Selected memory:
{context}'''
GENERIC_FUSED = '''Use only the selected memory to answer the question. First write a concise, question-relevant summary of the evidence in ordinary prose, preserving specific details and any uncertainty. Then give the answer supported by that evidence. Return a JSON object with "evidence_summary" and "final_answer". Do not classify the question or choose an operation-specific schema. If the memory does not provide the answer, say so.

Question: {question}
Selected memory:
{context}'''


def oldroot(task):
    return OLD/'g35lite/full/tasks'/task


def code_hashes():
    modules = (source, rt, rt.prior, rt.prior.n, rt.prior.n.g, sel,
               sel.adapter, sel.adapter.raw, source.display, source.memory)
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.update((HERE/'run.py', HERE/'protocol.md'))
    paths.update(Path(p) for p in b.authorities()[3].get('source_hashes', {}))
    return {str(p): b.file_sha(p) for p in sorted(paths)}


def request_manifest(task, generation):
    key = generation['request_sha256']
    path = oldroot(task)/'calls'/key[:2]/key/'request.json.gz'
    value = json.loads(gzip.decompress(path.read_bytes()))
    source.require(b.sha(value)==key, 'Historical request identity drift')
    return value, path


def prepare():
    if (OUT/'registration.json').exists():
        return validate()
    reg = b.read(OLD/'registration.json')
    authority = b.authorities()[2]
    inputs = {str(OLD/'registration.json'): b.file_sha(OLD/'registration.json')}
    rows = []
    for task in sorted({q['task'] for q in reg['questions']}):
        root = oldroot(task)
        mp = root/'memory_complete.json'; memory = b.read(mp)
        gp = Path(memory['session']['artifact_path'])
        source.require(b.file_sha(gp)==memory['session']['artifact_sha256'], 'Graph hash drift')
        graph = b.read(gp)
        inputs.update({str(mp): b.file_sha(mp), str(gp): b.file_sha(gp)})
        facts, _ = sel.adapter.adapt_facts(graph)
        ids = [f['id'] for f in facts]
        by_id = {f['id']:f for f in facts}
        index = source.memory.GraphMemoryIndex(graph,candidate_cap=len(facts),rerank_top_k=50)
        for q in [q for q in reg['questions'] if q['task']==task]:
            source.require(set(q)=={'row_id','task','question'}, 'Non-score-free query')
            key=b.sha(q['row_id']); cp=root/'contexts/enhanced_reader'/f'{key}.json'
            pp=root/'predictions/enhanced_reader'/f'{key}.json'
            vp=root/'query_vectors'/f'{key}.json'
            c,p,v=b.read(cp),b.read(pp),b.read(vp)
            for path in (cp,pp,vp):inputs[str(path)]=b.file_sha(path)
            source.require(c['query']==q and p['row_id']==q['row_id'] and v['row_id']==q['row_id'], 'Row binding')
            source.require(b.sha(c['context'])==p['context_sha256']==c['context_sha256'], 'Context binding')
            rendered,_=sel.adapter.render_selection(graph,c['selected_fact_ids'])
            source.require(rendered==c['context'], 'Full render differs')
            dense=list(map(float,index._scores(index.fact_matrix,v['vector'])))
            ranked=sel.display_ids(ids,ids,dense)
            base50,_=sel.adapter.render_selection(graph,ranked[:50])
            source.require(len(base50)==c['budget'], 'Dense50 budget differs')
            chosen=[];seen=set();skipped=0
            for fid in ranked:
                f=by_id[fid];dup=(tuple(sorted(f['sources'])),f['text'])
                if dup in seen:continue
                trial,_=sel.adapter.render_selection(graph,chosen+[fid])
                if len(trial)>c['budget']:
                    skipped+=1;continue
                chosen.append(fid);seen.add(dup)
                if len(chosen)==30:break
            dtext,render=sel.adapter.render_selection(graph,chosen)
            source.require(0<len(chosen)<=30 and len(dtext)<=c['budget'],'Dense budget')
            caps={}
            for st in p['stages']:
                m,path=request_manifest(task,st['generation'])
                inputs[str(path)]=b.file_sha(path)
                cap=m['generation_config']['maxOutputTokens']
                source.require(m['generation_config']==rt.config(MODEL,st['stage'],cap,st['stage']!='answer'), 'Decode mismatch')
                caps[st['stage']]=cap
            source.require(p['route']==reg['routes'][q['row_id']],'Route mismatch')
            item=dict(query=q,route=p['route'],caps=caps,budget=c['budget'],
                full_context=c['context'],full_ids=c['selected_fact_ids'],
                dense_context=dtext,dense_ids=chosen,dense_render=render,
                dense_skipped_for_budget=skipped,candidate_count=len(facts),
                full_prediction_path=str(pp),full_prediction_sha256=p['seal_sha256'],
                source_context_path=str(cp))
            b.save(OUT/'prepared'/f'{key}.json',item)
            rows.append(dict(row_id=q['row_id'],task=task,path=str(OUT/'prepared'/f'{key}.json'),sha256=b.file_sha(OUT/'prepared'/f'{key}.json')))
    source.require(len(rows)==371 and len({x['row_id'] for x in rows})==371,'Full population')
    source.require(all(b.file_sha(p)==h for p,h in inputs.items()),'Concurrent source mutation')
    return b.save(OUT/'registration.json',dict(schema='review26.matched.memeye.v1',
        rows=rows,arms=ARMS,model=MODEL,judge=JUDGE,code_hashes=code_hashes(),inputs=inputs,
        prompt_hashes={k:b.sha(authority[k]) for k in ('open','focused','fused')} |
        {'generic':b.sha(GENERIC),'generic_fused':b.sha(GENERIC_FUSED)},
        count=371,worker_cap=8,bootstrap_seed=92601,bootstrap_n=10000,
        generation_gold_blind=True,original_results_read_only=True,
        same_call_count_and_caps_not_exact_consumed_token_match=True))


def validate():
    reg=b.read(OUT/'registration.json')
    source.require(reg['code_hashes']==code_hashes(),'Supplement source changed')
    for p,h in reg['inputs'].items():source.require(b.file_sha(p)==h,'Input changed: '+p)
    for r in reg['rows']:source.require(b.file_sha(r['path'])==r['sha256'],'Prepared row changed')
    return reg


def one(arm, ref, reg):
    key=b.sha(ref['row_id']);item=b.read(ref['path']);q=item['query']
    rowroot=OUT/'runs'/arm; pp=rowroot/'predictions'/f'{key}.json'
    sp=rowroot/'scores'/f'{key}.json'; fail=rowroot/'failures'/f'{key}.json'
    if sp.exists():
        p=b.read(pp);s=b.read(sp)
        source.require(s['prediction_sha256']==p['seal_sha256'],'New verdict binding')
        return 'cached'
    if fail.exists():return 'retained_failure'
    try:
        gen=rt.Runtime(rowroot/'calls',MODEL,True)
        authority=b.authorities()[2]
        meta=dict(row_id=q['row_id'],arm=arm,registration_sha256=reg['seal_sha256'])
        if pp.exists():pred=b.read(pp)
        else:
            ctx=item['dense_context'] if arm=='dense30' else item['full_context']
            original_context=ctx;stages=[]
            def call(stage,prompt,*,json_mode):
                began=time.perf_counter()
                out=gen.generate(stage,prompt,json_mode=json_mode,max_output_tokens=item['caps'][stage],cache_metadata=meta)
                stages.append(dict(stage=stage,generation=asdict(out),elapsed_s=time.perf_counter()-began))
                return out
            if item['route']=='two_stage':
                template=GENERIC if arm=='generic_reader' else authority['focused']
                prompt=template.replace('{question}',q['question']).replace('{context}',ctx)
                out=call('focused',prompt,json_mode=True)
                parsed=sel.adapter.direct.predict._parse_json_object(out.text)
                if arm=='generic_reader':
                    source.require(isinstance(parsed,dict) and set(parsed)=={'summary'} and isinstance(parsed['summary'],str) and parsed['summary'].strip(),'Generic summary JSON')
                    ctx='AGGREGATED EVIDENCE:\n'+parsed['summary'].strip()
                else:
                    ctx=source.display.rendered_context(authority['render'],parsed or {'_failed':True},ctx,'candidate_first','candidate_last')
                prompt=authority['open'].format(context=ctx,question=q['question'])
                out=call('answer',prompt,json_mode=False);answer=out.text.strip()
            else:
                template=GENERIC_FUSED if arm=='generic_reader' else authority['fused']
                prompt=template.replace('{context}',ctx).replace('{question}',q['question'])
                out=call('structured_fused',prompt,json_mode=True)
                parsed=sel.adapter.direct.predict._parse_json_object(out.text)
                source.require(isinstance(parsed,dict),'Fused JSON parse')
                answer=str(parsed.get('final_answer') or 'Not found in memory.').strip()
            source.require(bool(answer),'Empty answer')
            pred=b.save(pp,dict(row_id=q['row_id'],task=q['task'],arm=arm,prediction=answer,
                model=MODEL,route=item['route'],stages=stages,context_sha256=b.sha(original_context),
                selected_fact_ids=item['dense_ids'] if arm=='dense30' else item['full_ids'],
                qa_prompt_sha256=b.sha(prompt),registration_sha256=reg['seal_sha256']))
        # Gold is loaded exclusively after the answer is durably committed.
        gold=b.read(source.SOURCE/'scoring/open.json')['rows'][q['row_id']]
        source.require(gold['question']==q['question'],'Gold question binding')
        judge=rt.Runtime(rowroot/'judge_calls',JUDGE,True);began=time.perf_counter()
        out,verdict=judge.judge(q['question'],gold['gold'],pred['prediction'],dict(meta,prediction_sha256=pred['seal_sha256']))
        b.save(sp,dict(row_id=q['row_id'],task=q['task'],x=gold['x'],y=gold['y'],
            raw_score=verdict.raw_score,mapped_score=verdict.mapped_score,reasoning=verdict.reasoning,
            judge=JUDGE,generation=asdict(out),elapsed_s=time.perf_counter()-began,
            prediction_sha256=pred['seal_sha256']))
        print('DONE',arm,q['row_id'],flush=True)
        return 'done'
    except Exception as exc:
        b.save(fail,dict(row_id=q['row_id'],arm=arm,time=time.time(),error_type=type(exc).__name__,message=str(exc)[:1500],success_not_overwritten=True))
        print('FAILED',arm,q['row_id'],type(exc).__name__,str(exc)[:240],flush=True)
        return 'failed'


def execute(smoke=False):
    reg=validate();load_dotenv(Path.cwd()/'.env',override=False)
    OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(OUT/'runner.lock'),timeout=0):
        refs=reg['rows']
        if smoke:
            chosen={}
            for ref in refs:
                route=b.read(ref['path'])['route']
                chosen.setdefault(route,ref)
            refs=list(chosen.values())
        elif not (OUT/'smoke_pass.json').exists():
            raise RuntimeError('Smoke gate not passed')
        rt.prior.disk_guard(OUT)
        outcomes=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs=[pool.submit(one,arm,ref,reg) for ref in refs for arm in ARMS]
            for job in as_completed(jobs):outcomes.append(job.result())
        result=dict(time=time.time(),counts={v:outcomes.count(v) for v in set(outcomes)},n=len(outcomes))
        b.save(OUT/('smoke_pass.json' if smoke and not any('fail' in x for x in outcomes) else 'executions'/f'{time.time_ns()}.json'),result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','run']);a=parser.parse_args()
    if a.command=='prepare':print('REGISTERED',prepare()['count'])
    else:execute(a.command=='smoke')

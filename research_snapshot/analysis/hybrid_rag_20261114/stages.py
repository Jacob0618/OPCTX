"""Original-model reranking/QA/Judge adapters for a raw-chunk Hybrid RAG.

No credentials, network, Writer, Online, or Reader/planner are owned here.
Inject ``chat(prompt, **kwargs)->str`` and ``qwen_post(url,payload)->dict``.
Those callbacks must preserve every physical request/response/usage and raise
on safety, permission, non-STOP, model identity, or unresolved transport errors.

``make_stages(chat, qwen_post=None)`` returns an independent StageAdapter:
  rerank(dataset, question, candidates) -> {selected_ids, audit}
  answer(dataset, question, context, question_date='') -> {prediction, audit}
  judge(dataset, question, reference, prediction) -> {correct, raw, audit}

dataset is 'locomo' or 'lme'; candidates contain unique {id,text} strings.
Pass the ORIGINAL question to all three stages, not a date-appended retrieval
query. LME's original QA system template receives question_date separately.
The caller determines raw-chunk rendering and retrieval, and saves outputs.

LME uses tag='lme_judge' only in the injected chat ledger, preventing the
LoCoMo client's internal Judge loop from nesting inside LME's original loop.
Map this tag to the Judge cost stage in the caller; it does not change payload.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from analysis.opctx_random_links_20261113 import run as locomo
from analysis.opctx_reviewer_20261111.e2e.locomo_e2e import load_function
from creamem3 import lme_opctx_v1 as lme
from creamem3 import run_lme_opctx100_v1 as legacy

ROOT = Path(__file__).resolve().parents[2]
QA_MODEL = 'gemini-2.5-flash'
JUDGE_MODEL = 'gemini-3-flash-preview'
QWEN_MODEL = 'qwen3-rerank'
QWEN_ENDPOINT = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'
QWEN_SEMAPHORE = threading.BoundedSemaphore(4)


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                         separators=(',',':'),allow_nan=False).encode()).hexdigest()


class StageFailure(RuntimeError):
    """A missing/invalid result, never an incorrect score or dense fallback."""
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit or {}


def _qwen_native_body():
    """Extract the unchanged payload + strict parser, excluding old cache I/O.

The first payload assignment in StrictRetrievalClient.rerank is after its cache
gate; the last extracted statement constructs the fully aligned score array.
The original local graph adapter performs the final top30 ranking separately.
"""
    path=Path(legacy.__file__)
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_client_types')
    cls=next(n for n in outer.body if isinstance(n,ast.ClassDef) and n.name=='StrictRetrievalClient')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='rerank')
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    start=next(i for i,n in enumerate(method.body) if assigns(n,'payload'))
    end=next(i for i,n in enumerate(method.body[start:],start) if assigns(n,'scores'))
    function=ast.FunctionDef(name='qwen_native',args=method.args,
        body=[*method.body[start:end+1],ast.Return(value=ast.Name(id='scores',ctx=ast.Load()))],
        decorator_list=[],returns=method.returns,type_comment=None)
    namespace=dict(Sequence=Sequence,Mapping=Mapping,math=math,QWEN_SEMAPHORE=QWEN_SEMAPHORE,
                   release_clients=SimpleNamespace(DASHSCOPE_RERANK_URL=QWEN_ENDPOINT))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['qwen_native']


class _NoCheckpoint:
    """Outer runner owns durable outputs; original Judge sees no old cache."""
    def exists(self):
        return False


class _LogicalChat:
    """Original LME application's chat interface, not a physical-call ledger.

Retries by an injected transport can create more physical calls; only its
receipts may be used for billing. No fake HTTP-body evidence is fabricated.
"""
    def __init__(self, client, contract, *, default_model, stage):
        self.client=client
        self.contract=contract
        self.default_model=default_model
        self.stage=stage
        self.attempts=[]

    def __call__(self,prompt,**kwargs):
        if len(self.attempts)>=legacy.MAX_GENERATION_ATTEMPTS:
            raise StageFailure('LME original application attempt limit reached')
        if float(kwargs.get('temperature',0))!=0 or int(kwargs.get('thinking_budget',0))!=0:
            raise StageFailure('Original LME temperature/thinking contract changed')
        model=kwargs.get('model') or self.default_model
        if model not in (QA_MODEL,JUDGE_MODEL):
            raise StageFailure('Unexpected model')
        tag=kwargs.get('tag') or self.stage
        actual_tag='lme_judge' if tag=='judge' else tag
        record=dict(application_call=len(self.attempts)+1,model=model,tag=tag,
            max_tokens=kwargs.get('max_tokens',8192),
            canonical_request_sha256=digest(dict(prompt=prompt,**kwargs)),
            filled_prompt_sha256=lme.sha256_text(prompt),
            system_sha256=lme.sha256_text(kwargs['system']) if kwargs.get('system') else None,
            measurement='logical application call; physical receipts belong to injected transport')
        self.attempts.append(record)
        tick=time.perf_counter()
        try:
            response=self.client(prompt,model=model,temperature=0.0,thinking_budget=0,
                tag=actual_tag,max_tokens=kwargs.get('max_tokens',8192),
                system=kwargs.get('system'),use_cache=False,json_mode=kwargs.get('json_mode',False))
            record.update(status='response_received',response_sha256=lme.sha256_text(str(response)))
            return str(response)
        except BaseException as error:
            record.update(status='failed',error_type=type(error).__name__)
            raise
        finally:
            record['elapsed_seconds']=time.perf_counter()-tick


def _validate_logical(attempts,*,same_canonical_body,require_nonempty=True):
    if not isinstance(attempts,list) or (require_nonempty and not attempts) or len(attempts)>5:
        raise StageFailure('LME application receipts count mismatch')
    if any(a.get('application_call')!=i or not isinstance(a.get('canonical_request_sha256'),str)
           for i,a in enumerate(attempts,1)):
        raise StageFailure('LME application receipt identity mismatch')
    if same_canonical_body and len({a['canonical_request_sha256'] for a in attempts})>1:
        raise StageFailure('LME retry changed its canonical request')
    return attempts


class StageAdapter:
    def __init__(self,chat,qwen_post=None):
        self.chat=chat
        self.qwen_post=qwen_post
        self.contract=lme.assert_prompt_contract()
        self.locomo_prompts=locomo.frozen.frozen_prompts()
        self.qwen_body=_qwen_native_body()
        ns=dict(Any=Any,Mapping=Mapping,Sequence=Sequence,chat=chat,
            FIXED_WRITER_READER_MODEL=QA_MODEL,CURRENT_SAMPLE_ID='hybrid_raw_chunks',
            CURRENT_SAMPLE_INDEX=-1,QA_PROMPT=self.locomo_prompts['QA_PROMPT'])
        self.locomo_chat=load_function(ROOT/'creamem3/run_conv26_v2.py','_model_chat',ns)
        self.locomo_answer=load_function(ROOT/'creamem3/run_conv26_v2.py','_answer_one',ns)
        jns=dict(chat=chat,os=os,re=re,JUDGE_PROMPT=self.locomo_prompts['JUDGE_PROMPT'],JUDGE_MODEL=JUDGE_MODEL)
        for name in ('_thinking_budget','parse_verdict_detailed','judge_one_detailed'):
            load_function(ROOT/'eval/judge.py',name,jns)
        self.locomo_judge=jns['judge_one_detailed']

    @staticmethod
    def _dataset(dataset):
        if dataset not in ('locomo','lme'):
            raise ValueError("dataset must be 'locomo' or 'lme'")

    def contract_receipt(self):
        sources=[Path(__file__),Path(legacy.__file__),Path(lme.__file__),lme.MGTM_CLIENT_SOURCE,
            lme.MGTM_PROMPT_SOURCE,ROOT/'creamem3/run_conv26_v2.py',ROOT/'eval/judge.py',locomo.GRAPH_PATH]
        return dict(qa_model=QA_MODEL,judge_model=JUDGE_MODEL,
            source_hashes={str(p):locomo.file_sha(p) for p in sources},
            prompt_hashes=dict(lme=self.contract.prompt_sha256,
                locomo_qa=lme.sha256_text(self.locomo_prompts['QA_PROMPT']),
                locomo_judge=lme.sha256_text(self.locomo_prompts['JUDGE_PROMPT'])),
            locomo=dict(reranker=QA_MODEL,rerank_top_k=20,rerank_max_tokens=4096,
                rerank_semantic_attempts=2,qa_max_tokens=8192,judge_initial_max_tokens=2048),
            lme=dict(reranker=QWEN_MODEL,rerank_endpoint=QWEN_ENDPOINT,rerank_top_k=30,
                qwen_return_top_n='all candidates, same as original; then local top30',
                qa_max_tokens=500,qa_system_template=True,qa_user='raw question',
                qa_application_max_attempts=5,judge_max_tokens=[200,400,800,1600,3200],
                judge_ledger_tag='lme_judge',judge_inner_client_retries=False),
            no_writer_online_planner_reader=True,rerank_failure='missing, never dense fallback',
            lme_cache_difference='only outer new-run checkpoints; no original-cache imports',
            lme_logging_adaptation='original Judge control flow with logical receipts; no fabricated physical records')

    def rerank(self,dataset,question,candidates):
        self._dataset(dataset)
        if not isinstance(question,str) or not question.strip():
            raise ValueError('Nonempty original question required')
        chunks=[dict(id=c['id'],text=c['text']) for c in candidates]
        if (not chunks or len({c['id'] for c in chunks})!=len(chunks)
                or any(not isinstance(c['id'],str) or not c['id'] or not isinstance(c['text'],str)
                       or not c['text'].strip() for c in chunks)):
            raise ValueError('Expected nonempty uniquely identified raw chunks')
        if len(chunks)>(120 if dataset=='locomo' else 240):
            raise ValueError('Candidate limit exceeds the registered original cap')
        native=[dict(fact=c) for c in chunks]
        if dataset=='locomo':
            selected,audit=locomo.gm.GraphMemoryIndex._rerank(
                SimpleNamespace(rerank_top_k=20),question,native,self.locomo_chat)
            if audit.get('fallback'):
                raise StageFailure('Original Gemini reranker exhausted validation',audit)
        else:
            if self.qwen_post is None:
                raise StageFailure('LME requires an injected audited Qwen transport')
            def post(url,payload,**kwargs):
                if url!=QWEN_ENDPOINT or payload.get('model')!=QWEN_MODEL:
                    raise StageFailure('Unexpected Qwen endpoint/model')
                return self.qwen_post(url,payload)
            qwen_client=SimpleNamespace(stack=SimpleNamespace(reranker=QWEN_MODEL),
                                        _post=post,dashscope_api_key='')
            def score(query,documents,instruction):
                return self.qwen_body(qwen_client,query,documents,instruction)
            index=SimpleNamespace(rerank_top_k=30,_score_reranker=score,_instruction=self.contract.qwen_instruction)
            selected,audit=lme.QwenGraphMemoryIndex._rerank(index,question,native,None)
        ids=[c['fact']['id'] for c in selected]
        if len(ids)!=min(20 if dataset=='locomo' else 30,len(chunks)) or len(set(ids))!=len(ids):
            raise StageFailure('Reranker output count/identity mismatch',audit)
        return dict(selected_ids=ids,audit={**audit,'selected_fact_ids':ids,
            'candidate_count':len(chunks),'input_unit':'raw_dialogue_chunk'})

    def answer(self,dataset,question,context,question_date=''):
        self._dataset(dataset)
        if not isinstance(question,str) or not question.strip() or not isinstance(context,str):
            raise ValueError('Question/context must be text')
        if dataset=='locomo':
            row=self.locomo_answer(dict(qa=dict(question=question),retrieved=context))
            if row.get('qa_error') or not row.get('response'):
                raise StageFailure('LoCoMo QA has no valid answer',dict(qa_error=row.get('qa_error')))
            return dict(prediction=row['response'],audit=dict(model=QA_MODEL,max_tokens=8192,
                context_chars=len(context),prompt_sha256=lme.sha256_text(
                    self.locomo_prompts['QA_PROMPT'].format(retrieved_texts=context,question=question))))
        system=self.contract.qa_template.format(context=context,qdate=str(question_date),question=question)
        bridge=_LogicalChat(self.chat,self.contract,default_model=QA_MODEL,stage='qa')
        # Reuse the actual five-attempt function; no alternate QA prompt.
        response=legacy._retry_same_request(bridge,question,validator=legacy._nonempty,
            model=QA_MODEL,tag='read',system=system,temperature=0.0,thinking_budget=0,
            max_tokens=500,use_cache=False,json_mode=False)
        return dict(prediction=str(response).strip(),audit=dict(model=QA_MODEL,max_tokens=500,
            context_chars=len(context),system_sha256=lme.sha256_text(system),
            application_attempts=bridge.attempts))

    def judge(self,dataset,question,reference,prediction):
        self._dataset(dataset)
        if not isinstance(prediction,str) or not prediction.strip():
            raise StageFailure('Missing predictions cannot be scored as zero')
        if dataset=='locomo':
            if os.environ.get('JUDGE_THINKING','off').lower() not in ('off','0','false'):
                raise StageFailure('Original no-thinking Judge setting required')
            verdict,raw,parsed=self.locomo_judge(question,reference,prediction)
            if not parsed:
                raise StageFailure('Invalid LoCoMo Judge verdict')
            return dict(correct=int(verdict),raw=raw,audit=dict(model=JUDGE_MODEL,parsed=True))
        captures=[]
        # The original Judge loop is reused verbatim. Only its persistence and
        # physical-transport-specific evidence validators are injected; root
        # owns durable physical receipts and final row checkpoints.
        namespace=dict(Path=Path,Mapping=Mapping,Any=Any,PromptContract=lme.PromptContract,
            sha256_text=lme.sha256_text,re=re,time=time,RUN_SCHEMA='hybrid_rag.lme.judge.v1',
            PhysicalChat=_LogicalChat,NonRetryableProviderError=legacy.NonRetryableProviderError,
            MAX_GENERATION_ATTEMPTS=legacy.MAX_GENERATION_ATTEMPTS,FIXED_BACKOFF_S=legacy.FIXED_BACKOFF_S,
            _validate_physical_attempts=_validate_logical,
            _atomic_json=lambda path,value:captures.append(value),
            _sealed_payload=legacy._sealed_payload)
        original=load_function(Path(legacy.__file__),'_judge',namespace)
        verdict,raw,attempts=original(path=_NoCheckpoint(),binding=dict(
            question_sha256=lme.sha256_text(question),reference_sha256=lme.sha256_text(reference),
            prediction_sha256=lme.sha256_text(prediction)),client=self.chat,contract=self.contract,
            model=JUDGE_MODEL,question=question,reference=reference,predicted=prediction)
        return dict(correct=int(verdict),raw=raw,audit=dict(model=JUDGE_MODEL,parsed=True,
            application_attempts=attempts,original_judge_record=captures[-1] if captures else None))


def make_stages(chat,qwen_post=None):
    """Create a no-I/O adapter; caller's callbacks own the live-call boundary."""
    return StageAdapter(chat,qwen_post)

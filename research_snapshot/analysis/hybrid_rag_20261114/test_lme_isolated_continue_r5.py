"""Offline r5 regression tests: native498 MAX and real scoring scheduling."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from analysis.hybrid_rag_20261114 import lme_isolated_continue_r5 as r
from analysis.hybrid_rag_20261114.test_lme_isolated_continue import records


class R5Tests(unittest.TestCase):
    def test_native_count_is_not_finish_condition(self):
        req,out,raw=records()
        for tokens in (0,498,500,501,800,None):
            changed=copy.deepcopy(raw)
            if tokens is None:changed['usageMetadata'].pop('candidatesTokenCount')
            else:changed['usageMetadata']['candidatesTokenCount']=tokens
            self.assertTrue(r.is_qa500(req,out,changed,'test'))
        raw['usageMetadata']=None
        self.assertTrue(r.is_qa500(req,out,raw,'test'))

    def test_invalid_count_cap_model_safety_fail_closed(self):
        req,out,raw=records()
        for tokens in (-1,True,498.0,'498'):
            changed=copy.deepcopy(raw);changed['usageMetadata']['candidatesTokenCount']=tokens
            self.assertFalse(r.is_qa500(req,out,changed,'test'))
        changed=copy.deepcopy(req);changed['request']['generationConfig']['maxOutputTokens']=1000
        self.assertFalse(r.is_qa500(changed,out,raw,'test'))
        for change in ({'modelVersion':'other'},{'error':{'status':'PERMISSION_DENIED'}},
                       {'promptFeedback':{'safetyRatings':[{'blocked':True}]}}):
            changed={**raw,**change};self.assertFalse(r.is_qa500(req,out,changed,'test'))
        changed=copy.deepcopy(raw);changed['candidates'][0]['finishReason']='SAFETY'
        self.assertFalse(r.is_qa500(req,out,changed,'test'))

    def test_above_cap_usage_is_recorded_not_scored(self):
        with tempfile.TemporaryDirectory(prefix='lme_r5_usage_') as td:
            root=Path(td)
            r.save_missing(root,'example',{'usage':{'candidatesTokenCount':501},'native':'MAX_TOKENS'})
            row=r.read(root/'questions/example/missing.json')
            self.assertIsNone(row['correct'])
            self.assertEqual(row['evidence']['usage_anomaly']['reported_output_tokens'],501)
            self.assertFalse((root/'questions/example/result.json').exists())

    def test_frozen_scheduler_scores_inherited_and_new_despite_missing(self):
        frozen=r.frozen
        with tempfile.TemporaryDirectory(prefix='lme_r5_sched_') as td, ExitStack() as stack:
            out=Path(td)/'r5';out.mkdir()
            parents=tuple(Path(td)/n for n in ('r1','r2','r3','r4'))
            ids=['old_a','old_b','new_ok','new_max']
            contexts=[{'context_id':qid,'questions':[{'question_id':qid}]} for qid in ids]
            for qid in ids[:2]:
                r.save(out/'questions'/qid/'inference.json',dict(question_id=qid,category_name='test',prediction='answer'))
            class Wire:
                def __init__(self,p):self.stop=threading.Event()
                def qwen_post(self,*a,**kw):raise AssertionError('No Qwen calls in unit test')
            def infer(context,q,index,args,wire,embed,adapter,**kw):
                if q['question_id']=='new_max':raise frozen.QuestionMaxTokens({'native':'test-MAX'})
                r.save(out/'questions'/q['question_id']/'inference.json',
                       dict(question_id=q['question_id'],category_name='test',prediction='new answer'))
            judged=[]
            def judge(row,args,wire,adapter):
                judged.append(row['question_id'])
                r.save(out/'questions'/row['question_id']/'result.json',
                       {**row,'correct':1,'judge':{'raw':'[[yes]]'}})
            mocks=[(frozen,'validate_registration',lambda *a:None),
                   (frozen.original,'frozen_prepare',lambda *a:(contexts,{'question_ids':ids})),
                   (frozen.transport,'preflight',lambda *a,**kw:None),
                   (frozen,'transport_class',lambda *a:Wire),
                   (frozen.transport,'make_clients',lambda *a:(None,None)),
                   (frozen.stages,'make_stages',lambda *a,**kw:None),
                   (frozen.original,'build_index',lambda *a:None),
                   (frozen.original,'process_question',infer),
                   (frozen.original,'score_inference',judge)]
            for obj,key,value in mocks:stack.enter_context(patch.object(obj,key,value))
            frozen.run_live(parents,out,{},[],contexts,{'question_ids':ids})
            self.assertEqual(set(judged),{'old_a','old_b','new_ok'})
            self.assertTrue((out/'questions/new_max/missing.json').exists())
            self.assertFalse((out/'questions/new_max/result.json').exists())
            self.assertFalse((out/'questions/old_a/prediction.json').exists())

    def test_actual_native_498_response(self):
        folder=r.PARENTS[-1]/'runtime/questions/0a34ad58/calls/000006_qa'
        req,out,raw=r.read(folder/'request.json'),r.read(folder/'outcome.json'),r.read(folder/'response.json')
        self.assertEqual(raw['usageMetadata']['candidatesTokenCount'],498)
        self.assertEqual(r.digest(raw),out['response_sha256'])
        self.assertTrue(r.is_qa500(req,out,raw,'0a34ad58'))


if __name__=='__main__':unittest.main()

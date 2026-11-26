"""No-network tests of per-question QA500 isolation and fail-closed behavior."""
import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from analysis.hybrid_rag_20261114 import lme_isolated_continue as r


def records(qid='test',case='qa'):
    req=dict(model='gemini-2.5-flash',kind='generation',request={'generationConfig':{'maxOutputTokens':500},'case':case})
    raw=dict(modelVersion='gemini-2.5-flash',usageMetadata={'candidatesTokenCount':500},
             candidates=[dict(finishReason='MAX_TOKENS',content={'parts':[{'text':'partial'}]})])
    out=dict(id=1,scope=qid,stage='qa',model='gemini-2.5-flash',model_version='gemini-2.5-flash',
        status='stopped',physical_calls=1,http_status=200,finish_reason='MAX_TOKENS',
        request_key=r.digest([req['model'],req['request']]),response_sha256=r.digest(raw),usage=raw['usageMetadata'])
    return req,out,raw


class FakeNative:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.attempts={};self.qwen_attempts={};self.raw_success={};self.qwen_success={}
        self.local=threading.local();self.lock=threading.RLock();self.stop=threading.Event();self.calls=[]
    def _endpoint(self,url,payload):
        return 'gemini-2.5-flash','generation',url
    def request(self,url,payload):
        if payload.get('case')=='network':
            self.stop.set()
            raise r.transport.ExecutionStop('network error')
        if payload.get('case')=='success':
            return {'valid':'answer'}
        req,out,raw=records(self.local.scope,payload.get('case','qa'))
        req['request']=payload
        out['request_key']=r.digest([req['model'],payload])
        if payload.get('case')=='blocked':
            raw['promptFeedback']={'safetyRatings':[{'blocked':True}]}
            out['response_sha256']=r.digest(raw)
        folder=self.output/'calls/000001_qa'
        r.save(folder/'request.json',req);r.save(folder/'response.json',raw);r.save(folder/'outcome.json',out)
        self.calls.append(out);self.stop.set()
        raise r.transport.ExecutionStop('native MAX_TOKENS')
    def qwen_post(self,url,payload):
        raise r.transport.ExecutionStop('qwen failure')


class IsolationTests(unittest.TestCase):
    def test_strict_predicate(self):
        req,out,raw=records()
        self.assertTrue(r.is_qa500(req,out,raw,'test'))
        for bad in ('model','tokens','safety','error'):
            changed=copy.deepcopy(raw)
            if bad=='model':changed['modelVersion']='other'
            if bad=='tokens':changed['usageMetadata']['candidatesTokenCount']=499
            if bad=='safety':changed['promptFeedback']={'safetyRatings':[{'blocked':True}]}
            if bad=='error':changed['error']={'status':'PERMISSION_DENIED'}
            self.assertFalse(r.is_qa500(req,out,changed,'test'))

    def test_local_max_does_not_stop_another_question(self):
        with tempfile.TemporaryDirectory(prefix='lme_isolation_') as td, patch.object(r.transport,'AuditTransport',FakeNative):
            shared=r.SharedNative([]);global_stop=threading.Event();cls=r.transport_class(shared,global_stop,Path(td))
            a,b=cls(Path(td)/'a'),cls(Path(td)/'b')
            a.local.scope='qa1';b.local.scope='qa2'
            with self.assertRaises(r.QuestionMaxTokens) as caught:
                a.request('unused',{'generationConfig':{'maxOutputTokens':500},'case':'qa'})
            self.assertFalse(global_stop.is_set());self.assertTrue(a.stop.is_set());self.assertFalse(b.stop.is_set())
            self.assertEqual(b.request('unused',{'case':'success'}),{'valid':'answer'})
            r.save_missing(Path(td),'qa1',caught.exception.evidence)
            self.assertIsNone(r.read(Path(td)/'questions/qa1/missing.json')['correct'])
            self.assertFalse((Path(td)/'questions/qa1/result.json').exists())

    def test_network_and_safety_stop_all_local_retry_events(self):
        for case in ('network','blocked'):
            with tempfile.TemporaryDirectory(prefix='lme_isolation_') as td, patch.object(r.transport,'AuditTransport',FakeNative):
                shared=r.SharedNative([]);global_stop=threading.Event();cls=r.transport_class(shared,global_stop,Path(td))
                a,b=cls(Path(td)/'a'),cls(Path(td)/'b')
                a.local.scope='qa1';b.local.scope='qa2'
                with self.assertRaises(r.transport.ExecutionStop):
                    a.request('unused',{'generationConfig':{'maxOutputTokens':500},'case':case})
                self.assertTrue(global_stop.is_set());self.assertTrue(a.stop.is_set());self.assertTrue(b.stop.is_set())

    def test_known_failure_key_not_replayed(self):
        req,out,raw=records('prior')
        entries=[dict(kind='gemini',key=out['request_key'],outcome=out,raw=raw)]
        with tempfile.TemporaryDirectory(prefix='lme_isolation_') as td, patch.object(r.transport,'AuditTransport',FakeNative):
            shared=r.SharedNative(entries);global_stop=threading.Event();cls=r.transport_class(shared,global_stop,Path(td))
            wire=cls(Path(td)/'a');wire.local.scope='another'
            with self.assertRaises(r.transport.ExecutionStop):wire.request('unused',req['request'])
            self.assertTrue(global_stop.is_set());self.assertEqual(wire.calls,[])

    def test_missing_not_in_score_denominator_and_nested_cost_counted(self):
        with tempfile.TemporaryDirectory(prefix='lme_isolation_') as td:
            roots=tuple(Path(td)/name for name in ('r1','r2','r3','r4'))
            for qid in r.POLICY_IDS:
                r.save(roots[-1]/'questions'/qid/'result.json',dict(question_id=qid,category_name='cat',correct=0,judge=None))
            r.save(roots[-1]/'questions/good/result.json',dict(question_id='good',category_name='cat',correct=1,judge={'raw':'yes'}))
            r.save(roots[-1]/'questions/miss/missing.json',dict(question_id='miss',correct=None))
            r.save(roots[-1]/'runtime/questions/miss/calls/1/outcome.json',dict(stage='qa',model='gemini-2.5-flash',
                physical_calls=1,usage={'promptTokenCount':100,'candidatesTokenCount':500}))
            result=r.summary(roots[:-1],roots[-1],{'question_ids':[*r.POLICY_IDS,'good','miss']},threading.Event(),1)
            self.assertEqual(result['overall']['n'],4);self.assertEqual(result['overall']['accuracy_percent'],25)
            self.assertEqual(result['missing_n'],1);self.assertEqual(result['valid_judged_n'],1)
            self.assertEqual(result['physical_stage_usage']['qa']['known_output_tokens'],500)


if __name__=='__main__':unittest.main()

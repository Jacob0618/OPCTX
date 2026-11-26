"""Offline checks only; all provider interfaces are injected fakes."""
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from analysis.hybrid_rag_20261114 import stages


class TestStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network_patch=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        cls.network_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.network_patch.stop()

    def setUp(self):
        self.calls=[]
        self.posts=[]
        self.chunks=[dict(id=f'chunk-custom:{i:03d}',text=f'Dialogue content {i}') for i in range(40)]
        def chat(prompt,**kwargs):
            self.calls.append(dict(prompt=prompt,**kwargs))
            if kwargs['tag']=='creamem3_v2_rerank':
                return json.dumps(dict(ranked_fact_ids=[c['id'] for c in self.chunks[:20]]))
            if kwargs['tag'] in ('judge','lme_judge'):
                return '[[yes]]'
            return 'Test prediction'
        def qwen(url,payload):
            self.posts.append((url,payload))
            n=len(payload['input']['documents'])
            return dict(output=dict(results=[dict(index=i,relevance_score=float(i)) for i in range(n)]))
        self.adapter=stages.make_stages(chat,qwen)

    def test_locomo_exact_original_reranker_and_qa(self):
        result=self.adapter.rerank('locomo','Original question?',self.chunks)
        self.assertEqual(result['selected_ids'],[c['id'] for c in self.chunks[:20]])
        r=self.calls[-1]
        self.assertEqual((r['model'],r['max_tokens'],r['thinking_budget'],r['json_mode']),
                         ('gemini-2.5-flash',4096,0,True))
        self.assertIn('chunk-custom:000',r['prompt'])
        answered=self.adapter.answer('locomo','Original question?','RAW CHUNKS')
        self.assertEqual(answered['prediction'],'Test prediction')
        self.assertEqual(self.calls[-1]['prompt'],self.adapter.locomo_prompts['QA_PROMPT'].format(
            retrieved_texts='RAW CHUNKS',question='Original question?'))
        self.assertEqual(self.calls[-1]['max_tokens'],8192)
        self.assertEqual(self.adapter.judge('locomo','q','gold','prediction')['correct'],1)

    def test_lme_qwen_actual_native_body_and_local30(self):
        result=self.adapter.rerank('lme','Original question?',self.chunks)
        self.assertEqual(result['selected_ids'],[c['id'] for c in self.chunks[::-1][:30]])
        endpoint,payload=self.posts[0]
        self.assertEqual(endpoint,stages.QWEN_ENDPOINT)
        self.assertEqual(payload,dict(model='qwen3-rerank',
            input=dict(query='Original question?',documents=[c['text'] for c in self.chunks]),
            parameters=dict(return_documents=False,top_n=40,instruct=self.adapter.contract.qwen_instruction)))
        self.assertFalse(self.calls)

    def test_lme_exact_system_user_500_and_judge(self):
        self.adapter.answer('lme','Original question?','RAW CHUNKS','2024/01/01')
        q=self.calls[-1]
        self.assertEqual(q['prompt'],'Original question?')
        self.assertEqual(q['system'],self.adapter.contract.qa_template.format(
            context='RAW CHUNKS',qdate='2024/01/01',question='Original question?'))
        self.assertEqual((q['max_tokens'],q['thinking_budget'],q['use_cache']),(500,0,False))
        result=self.adapter.judge('lme','Original question?','gold','prediction')
        self.assertEqual(result['correct'],1)
        j=self.calls[-1]
        self.assertEqual((j['model'],j['tag'],j['max_tokens'],j['use_cache']),
                         ('gemini-3-flash-preview','lme_judge',200,False))
        self.assertEqual(j['prompt'],self.adapter.contract.judge_template.format(
            question='Original question?',reference='gold',predicted='prediction'))
        self.assertEqual(result['audit']['application_attempts'][0]['application_call'],1)

    def test_lme_original_judge_schedule_no_nested_retry(self):
        def chat(prompt,**kwargs):
            self.calls.append(kwargs)
            return 'unclear' if kwargs['max_tokens']==200 else '[[no]]'
        adapter=stages.make_stages(chat)
        with patch('time.sleep'):
            result=adapter.judge('lme','q','reference','prediction')
        self.assertEqual(result['correct'],0)
        self.assertEqual([r['max_tokens'] for r in self.calls],[200,400])
        self.assertTrue(all(r['tag']=='lme_judge' for r in self.calls))

    def test_invalid_gemini_rerank_no_dense_fallback(self):
        adapter=stages.make_stages(lambda *args,**kwargs:'{}')
        with self.assertRaises(stages.StageFailure) as cm:
            adapter.rerank('locomo','q',self.chunks)
        self.assertTrue(cm.exception.audit['fallback'])
        self.assertEqual(cm.exception.audit['attempts'],2)

    def test_qwen_duplicate_indices_rejected(self):
        adapter=stages.make_stages(lambda *args,**kwargs:'unused',
            lambda url,payload:dict(output=dict(results=[dict(index=0,relevance_score=0.5)]*40)))
        with self.assertRaisesRegex(RuntimeError,'unique integers'):
            adapter.rerank('lme','q',self.chunks)


if __name__=='__main__':
    unittest.main()

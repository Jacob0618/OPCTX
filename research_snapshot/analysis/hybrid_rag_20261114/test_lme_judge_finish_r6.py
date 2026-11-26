"""Offline tests for exact precompiled Judge semantics and narrow local recovery."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
import tempfile
import unittest
from unittest.mock import patch
from analysis.hybrid_rag_20261114 import lme_judge_finish_r6 as r


class JudgeOnlyTests(unittest.TestCase):
    def adapter(self,chat):
        adapter=object.__new__(r.stages.StageAdapter)
        adapter.chat=chat;adapter.contract=r.stages.lme.assert_prompt_contract()
        return adapter

    def test_original_prompt_model_cap_and_namespace_parallel(self):
        observed=[];lock=threading.Lock()
        def chat(prompt,**kw):
            with lock:observed.append((prompt,kw))
            return '[[yes]]'
        adapter=self.adapter(chat);code,_=r.compile_judge()
        with patch.object(r.stages,'load_function',r.stages.load_function):
            r.install_compiled_loader(code)
            def run(i):return adapter.judge('lme','question'+str(i),'reference','prediction')
            with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(run,range(40)))
        self.assertTrue(all(v['correct']==1 and len(v['audit']['application_attempts'])==1 for v in values))
        self.assertEqual(len(observed),40)
        for prompt,kw in observed:
            self.assertEqual(kw['model'],'gemini-3-flash-preview')
            self.assertEqual(kw['max_tokens'],200);self.assertEqual(kw['tag'],'lme_judge')
            self.assertIn('Reference answer: reference',prompt)
            self.assertIn('Response: prediction',prompt)

    def test_exact_original_cap_schedule_for_invalid_tag(self):
        seen=[]
        def chat(prompt,**kw):
            seen.append(kw['max_tokens']);return 'invalid' if len(seen)==1 else '[[no]]'
        adapter=self.adapter(chat);code,_=r.compile_judge()
        with patch.object(r.stages,'load_function',r.stages.load_function),patch.object(r.stages.time,'sleep',lambda *_:None):
            r.install_compiled_loader(code)
            result=adapter.judge('lme','q','r','p')
        self.assertEqual(seen,[200,400]);self.assertEqual(result['correct'],0)

    def test_narrow_no_request_system_error_only(self):
        before={Path('existing')}
        self.assertTrue(r.local_system_error_only(SystemError('local'),before,before,False))
        for exc in (PermissionError('deny'),OSError('disk'),MemoryError('ram'),RuntimeError('unknown'),r.transport.ExecutionStop('provider')):
            self.assertFalse(r.local_system_error_only(exc,before,before,False))
        self.assertFalse(r.local_system_error_only(SystemError('local'),before,before|{Path('new')},False))
        self.assertFalse(r.local_system_error_only(SystemError('local'),before,before,True))

    def test_trace_secret_redaction(self):
        with patch.dict(os.environ,{'GEMINI_API_KEY':'unit_fake_a','DASHSCOPE_API_KEY':'unit_fake_b'}):
            value=r.redact('trace unit_fake_a and unit_fake_b')
        self.assertEqual(value,'trace [REDACTED] and [REDACTED]')

    def test_pending_parent_request_is_not_ignored(self):
        with tempfile.TemporaryDirectory(prefix='judge_pending_') as td:
            root=Path(td)
            folder=root/'runtime/questions/q/calls/000001_judge'
            r.save(folder/'request.json',{'request':'placeholder'})
            with self.assertRaises(RuntimeError):r.verify_parent_judge_requests((root,))
            r.save(folder/'outcome.json',{'status':'ok'})
            self.assertEqual(r.verify_parent_judge_requests((root,)),1)


if __name__=='__main__':unittest.main()

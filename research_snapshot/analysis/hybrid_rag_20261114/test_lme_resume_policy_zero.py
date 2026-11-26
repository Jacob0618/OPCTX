"""Offline tests for one-question recovery plumbing, with no API calls."""
import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from analysis.hybrid_rag_20261114 import lme_resume_policy_zero as recovery


class FakeTransport:
    def __init__(self, runtime):
        self.attempts={'old-ok':1}
        self.raw_success={}
        self.qwen_attempts={}
        self.qwen_success={}
        self.local=threading.local()
        self.stop=threading.Event()

    def _endpoint(self, url, payload):
        return 'gemini-2.5-flash','generation',url

    def request(self, url, payload):
        key=recovery.digest(['gemini-2.5-flash',payload])
        if key in self.raw_success:
            return copy.deepcopy(self.raw_success[key])
        raise AssertionError('No network permitted')

    def qwen_post(self, endpoint, payload):
        key=recovery.digest(['qwen3-rerank',payload])
        if key in self.qwen_success:
            return copy.deepcopy(self.qwen_success[key])
        raise AssertionError('No network permitted')


class RecoveryTests(unittest.TestCase):
    def entries(self):
        payload={'contents':['unchanged']}
        key=recovery.digest(['gemini-2.5-flash',payload])
        return payload,key,[
            dict(kind='gemini',key=key,raw={'same':'native'},outcome={'status':'ok','physical_calls':1}),
            dict(kind='gemini',key='old-ok',raw={'old':'native'},outcome={'status':'ok','physical_calls':1}),
            dict(kind='gemini',key='failure',raw={'partial':'not accepted'},outcome={'status':'stopped','physical_calls':1}),
        ]

    def test_exact_success_and_cumulative_attempts(self):
        payload,key,entries=self.entries()
        with tempfile.TemporaryDirectory(prefix='hybrid_policy_test_') as td, patch.object(recovery.transport,'AuditTransport',FakeTransport):
            client=recovery.recovery_transport(entries,Path(td))(Path(td))
            self.assertEqual(client.attempts['old-ok'],2)
            self.assertEqual(client.attempts['failure'],1)
            self.assertNotIn('failure',client.raw_success)
            self.assertEqual(client.request('unused',payload),{'same':'native'})

    def test_excluded_question_never_requested(self):
        payload,_,entries=self.entries()
        with tempfile.TemporaryDirectory(prefix='hybrid_policy_test_') as td, patch.object(recovery.transport,'AuditTransport',FakeTransport):
            client=recovery.recovery_transport(entries,Path(td))(Path(td))
            client.local.scope=recovery.POLICY_QID
            with self.assertRaises(recovery.transport.ExecutionStop):
                client.request('unused',payload)
            self.assertTrue(client.stop.is_set())

    def test_manual_stop_and_failed_exact_key(self):
        payload,key,entries=self.entries()
        entries[0]['outcome']['status']='stopped'
        with tempfile.TemporaryDirectory(prefix='hybrid_policy_test_') as td, patch.object(recovery.transport,'AuditTransport',FakeTransport):
            client=recovery.recovery_transport(entries,Path(td))(Path(td))
            with self.assertRaises(recovery.transport.ExecutionStop):
                client.request('unused',payload)
            self.assertNotIn(key,client.raw_success)
            with (Path(td)/'STOP').open('x'):
                pass
            second=recovery.recovery_transport([],Path(td))(Path(td))
            with self.assertRaises(recovery.transport.ExecutionStop):
                second.qwen_post('unused',{})

    def test_no_copy_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='hybrid_policy_test_') as td:
            root=Path(td)
            source,target=root/'source',root/'target'
            with source.open('x') as f:
                f.write('same')
            recovery.checked_copy(source,target)
            self.assertEqual(recovery.file_sha(source),recovery.file_sha(target))
            with (root/'other').open('x') as f:
                f.write('different')
            with self.assertRaises(RuntimeError):
                recovery.checked_copy(root/'other',target)


if __name__=='__main__':
    unittest.main()

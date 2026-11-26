"""Offline recovery tests; no live provider calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis.hybrid_rag_20261114 import lme_resume_two_zeros as r
from analysis.hybrid_rag_20261114.test_lme_resume_policy_zero import FakeTransport


class RecoveryTests(unittest.TestCase):
    def test_three_policy_scopes_and_unrelated_missing(self):
        with tempfile.TemporaryDirectory(prefix='lme_r3_test_') as td, patch.object(r.transport,'AuditTransport',FakeTransport):
            cls=r.recovery_transport([],Path(td))
            for qid in r.POLICY_IDS:
                client=cls(Path(td))
                client.local.scope=qid+'/index'
                with self.assertRaises(r.transport.ExecutionStop):
                    client.request('unused',{})
            client=cls(Path(td))
            client.local.scope='new_unrelated_question'
            with self.assertRaises(AssertionError):
                client.request('unused',{})
            self.assertFalse(client.stop.is_set())

    def test_cumulative_attempts_and_success_only(self):
        payload={'native':'same'}
        key=r.digest(['gemini-2.5-flash',payload])
        entries=[dict(kind='gemini',key=key,raw={'same':'response'},outcome={'status':'ok','physical_calls':1}),
                 dict(kind='gemini',key=key,raw={'same':'response'},outcome={'status':'ok','physical_calls':1}),
                 dict(kind='gemini',key='failed',raw={'partial':'not cached'},outcome={'status':'stopped','physical_calls':1})]
        with tempfile.TemporaryDirectory(prefix='lme_r3_test_') as td, patch.object(r.transport,'AuditTransport',FakeTransport):
            client=r.recovery_transport(entries,Path(td))(Path(td))
            self.assertEqual(client.attempts[key],2)
            self.assertEqual(client.attempts['failed'],1)
            self.assertNotIn('failed',client.raw_success)
            self.assertEqual(client.request('unused',payload),{'same':'response'})

    def test_conflicting_native_response_rejected(self):
        entries=[dict(kind='gemini',key='same',raw={'answer':v},outcome={'status':'ok','physical_calls':1}) for v in (1,2)]
        with tempfile.TemporaryDirectory(prefix='lme_r3_test_') as td, patch.object(r.transport,'AuditTransport',FakeTransport):
            with self.assertRaises(r.transport.ExecutionStop):
                r.recovery_transport(entries,Path(td))(Path(td))

    def test_three_roots_cost_no_request_dedup_and_failure_included(self):
        with tempfile.TemporaryDirectory(prefix='lme_r3_test_') as td:
            roots=tuple(Path(td)/name for name in ('r1','r2','r3'))
            for root in roots:
                # Same request key still represents a separate physical event.
                r.save(root/'runtime/calls/1/outcome.json',dict(stage='qa',model='gemini-2.5-flash',
                    request_key='same',status='stopped',physical_calls=1,service_seconds=1,
                    usage=dict(promptTokenCount=100,candidatesTokenCount=500)))
            result=r.combined_summary(roots,{})
            self.assertEqual(result['physical_stage_usage']['qa']['physical_calls'],3)
            self.assertEqual(result['physical_stage_usage']['qa']['known_output_tokens'],1500)
            self.assertEqual(result['policy_zero_count'],3)
            self.assertAlmostEqual(result['physical_inference_cost']['gemini_generation_usd'],3*(100*.30+500*2.5)/1e6)


if __name__=='__main__':
    unittest.main()

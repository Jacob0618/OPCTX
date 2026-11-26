"""No-network transport-observability checks for frozen LME E2E wrapper."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lme_e2e', HERE/'lme_e2e.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class E2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from creamem3 import run_lme_clean100_v1 as clean
        cls.clean = clean
        adapter = clean.configure('b3_scaled',freeze=False)
        cls.contract = adapter.assert_prompt_contract()
        cls.clients, _ = clean.core.legacy._load_release_runtime(cls.contract)

    def call_mock(self, kind, raw):
        from creamem3.experiments.memeye_resume_20260921 import network
        root_context = tempfile.TemporaryDirectory(prefix='lme_e2e_mock_')
        self.addCleanup(root_context.cleanup)
        root = Path(root_context.name)
        gen, ret = probe.audited_types(self.clean,self.clients,types.SimpleNamespace(output=root))
        obj = object.__new__(gen if kind=='generation' else ret)
        obj._http_local = threading.local()
        obj._cost_tag = 'write'
        obj.gemini_api_key = 'NOT_A_CREDENTIAL'
        obj.max_retries = 8
        probe._local.qid, probe._local.question = 'fixed-test-id', 'query'
        if kind == 'generation':
            url = self.clients.GEMINI_BASE + '/models/gemini-2.5-flash:generateContent?key=NOT_A_CREDENTIAL'
            payload = {'contents':[{'role':'user','parts':[{'text':'test history'}]}],
                'generationConfig':{'temperature':0.0,'maxOutputTokens':16384,'thinkingConfig':{'thinkingBudget':0}}}
        else:
            url = self.clients.GEMINI_BASE + '/models/gemini-embedding-001:batchEmbedContents?key=NOT_A_CREDENTIAL'
            payload = {'requests':[{'model':'models/gemini-embedding-001','content':{'parts':[{'text':'query'}]},'outputDimensionality':768}]}
        seen = []
        def post(url, *, content, headers):
            seen.append(json.loads(content))
            return types.SimpleNamespace(status_code=200,json=lambda:raw)
        with patch.object(network,'client',lambda:types.SimpleNamespace(post=post)):
            try:
                result = obj._post(url,payload)
                error = None
            except Exception as exc:
                result, error = None, exc
        self.assertEqual(seen,[payload])
        self.assertNotIn('NOT_A_CREDENTIAL',''.join(p.read_text() for p in root.rglob('*.json')))
        return root,result,error

    def test_generation_usage_and_native_body(self):
        raw = {'modelVersion':'gemini-2.5-flash','usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':20,'thoughtsTokenCount':0},
               'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'ok'}]}}]}
        root,result,error = self.call_mock('generation',raw)
        self.assertIsNone(error)
        self.assertEqual(result,raw)
        outcome = probe.read(next(root.rglob('outcome.json')))
        self.assertEqual(outcome['usage']['input_tokens'],100)
        self.assertEqual(outcome['usage']['output_tokens'],20)

    def test_embedding_unknown_usage_not_zero(self):
        root,result,error = self.call_mock('embedding',{'embeddings':[{'values':[0.1]*768}]})
        self.assertIsNone(error)
        self.assertIsNone(probe.read(next(root.rglob('outcome.json')))['usage'])
        self.assertEqual(probe.read(next(root.rglob('intent.json')))['stage'],'query_embedding')

    def test_safety_stops_and_keeps_raw(self):
        raw = {'promptFeedback':{'blockReason':'SAFETY'}}
        root,result,error = self.call_mock('generation',raw)
        self.assertIsInstance(error,self.clean.core.legacy.NonRetryableProviderError)
        self.assertEqual(len(list(root.rglob('intent.json'))),1)
        self.assertEqual(probe.unpack(next(root.rglob('response.json.gz'))),raw)

    def test_truncation_is_not_marked_complete(self):
        raw = {'modelVersion':'gemini-2.5-flash','usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':500},
               'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':'nonempty partial text'}]}}]}
        root,result,error = self.call_mock('generation',raw)
        self.assertIsInstance(error,self.clean.core.legacy.NonRetryableProviderError)
        self.assertTrue(probe.read(next(root.rglob('outcome.json')))['non_stop'])
        self.assertEqual(len(list(root.rglob('intent.json'))),1)


if __name__ == '__main__':
    unittest.main()

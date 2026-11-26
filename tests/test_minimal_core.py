"""Offline, dependency-light tests for an OpCtx minimal release directory.

Run: python test_minimal_core.py --root /path/to/opctx-code
No dataset, credential, network request, or model generation is used.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
args, unittest_args = parser.parse_known_args()
ROOT = args.root.resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import numpy as np
import config
import client
import memory
import opctx
import retrieval
import selector
import selection_utils


class StubClient:
    def __init__(self, responses=(), dim=768):
        self.responses = list(responses)
        self.calls = []
        self.embedding_calls = []
        self.dim = dim

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if not self.responses:
            raise AssertionError('Unexpected model-generation call')
        return self.responses.pop(0)

    def embed(self, texts, *, query=False):
        self.embedding_calls.append((list(texts), query))
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


def session(sid, text):
    return {'id': sid, 'timestamp': '2026-01-01',
            'turns': [{'id': 't1', 'speaker': 'user', 'text': text}]}


def writer(sid, text):
    return {'batch_summary': text, 'atomic_facts': [
        {'fact': text, 'source_dia_ids': [sid + '::t1']} ]}


def create_operation():
    return {'operations': [{'operation': 'create', 'target_view_id': '',
        'view_type': 'update_chain', 'member_fact_ids': ['F000001', 'F000002'],
        'summary_claims': [{'text': 'The cabinet changed from blue to white.',
                           'support_fact_ids': ['F000001', 'F000002']}]}]}


def fixture_memory():
    cfg = config.PROFILES['locomo']
    fake = StubClient([writer('s1', 'The cabinet was blue.'),
                       writer('s2', 'The cabinet is now white.'),
                       create_operation()], cfg.embedding_dim)
    app = opctx.OpCtx('locomo', client=fake)
    app.write([session('s1', 'Blue cabinet.'), session('s2', 'White cabinet.')])
    return app.memory


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail loudly even if future changes accidentally invoke an HTTP client.
        cls.network_guard = patch('requests.sessions.Session.request',
                                  side_effect=AssertionError('Network forbidden in offline tests'))
        cls.network_guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.network_guard.stop()

    def test_modules_import_from_release(self):
        for name in ('client', 'config', 'memory', 'numerical', 'opctx', 'prompts',
                     'retrieval', 'selection_utils', 'selector'):
            module = importlib.import_module(name)
            self.assertEqual(Path(module.__file__).resolve().parent, ROOT)

    def test_cli_help_needs_no_credential(self):
        env = dict(os.environ)
        env.pop('GEMINI_API_KEY', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--help'],
                                capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('build', result.stdout)
        self.assertIn('ask', result.stdout)
        self.assertNotIn('API key', result.stdout)

    def test_profiles(self):
        self.assertEqual(config.PROFILES['memeye'].final_k, 30)
        self.assertEqual(config.PROFILES['lme'].sessions_per_write, 3)
        self.assertEqual(config.PROFILES['locomo'].final_k, 20)

    def test_memory_config_mismatch_rejected(self):
        m = memory.empty_memory(config.PROFILES['locomo'])
        m['config']['final_k'] = 999
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            opctx.OpCtx('locomo', memory=m, client=StubClient())

    def test_writer_and_online_create_preserve_sources(self):
        m = fixture_memory()
        self.assertEqual([f['text'] for f in m['facts']],
                         ['The cabinet was blue.', 'The cabinet is now white.'])
        self.assertEqual([f['sources'] for f in m['facts']], [['s1::t1'], ['s2::t1']])
        self.assertEqual(len(m['packets']), 2)
        self.assertEqual(m['views'][0]['version'], 1)
        self.assertEqual(m['views'][0]['member_fact_ids'], ['F000001', 'F000002'])

    def test_online_extend_and_transactional_rollback(self):
        m = fixture_memory()
        before = deepcopy(m)
        operation = create_operation()['operations'][0]
        operation.update(operation='extend', target_view_id='V00001',
                         member_fact_ids=['F000001', 'F000002', 'F000003'])
        operation['summary_claims'][0]['support_fact_ids'] = operation['member_fact_ids']
        fake = StubClient([writer('s3', 'The cabinet is now red.'),
                           {'operations': [operation]}])
        app = opctx.OpCtx('locomo', memory=m, client=fake)
        app.write([session('s3', 'Red cabinet.')])
        self.assertEqual(m, before, 'The prior memory object must remain unchanged')
        self.assertEqual(app.memory['views'][0]['version'], 2)
        self.assertEqual(app.memory['views'][0]['member_fact_ids'],
                         ['F000001', 'F000002', 'F000003'])
        bad = create_operation()
        bad['operations'][0]['summary_claims'][0]['support_fact_ids'] = ['MISSING']
        app = opctx.OpCtx('locomo', memory=deepcopy(before),
                          client=StubClient([writer('s3', 'Red cabinet.'), bad]))
        with self.assertRaises(ValueError):
            app.write([session('s3', 'Red cabinet.')])
        self.assertEqual(app.memory, before)

    def test_writer_invalid_provenance_rejected(self):
        bad = writer('s1', 'Unsupported fact')
        bad['atomic_facts'][0]['source_dia_ids'] = ['unknown::turn']
        app = opctx.OpCtx('locomo', client=StubClient([bad]))
        with self.assertRaisesRegex(ValueError, 'source references'):
            app.write([session('s1', 'Text')])
        self.assertEqual(app.memory['facts'], [])

    def test_duplicate_sessions_rejected_before_call(self):
        fake = StubClient()
        app = opctx.OpCtx('locomo', client=fake)
        with self.assertRaisesRegex(ValueError, 'unique'):
            app.write([session('s1', 'One'), session('s1', 'Two')])
        self.assertEqual(fake.calls, [])

    def test_summary_branch_expands_members_then_deduplicates(self):
        m = {'facts': [{'id': f'F{i}', 'text': f'fact {i}', 'vector': [1., 0.]}
                       for i in range(60)],
             'packets': [{'id': 'P1', 'member_fact_ids': [f'F{i}' for i in range(60)],
                          'vector': [1., 0.]}], 'views': []}
        chosen = retrieval.memeye_candidates(m, 'fact', [1., 0.])
        self.assertEqual(len(chosen), 60, 'Top10 summaries is not a 50-fact cap')
        self.assertEqual(len({f['id'] for f in chosen}), 60)
        capped = retrieval.hybrid_candidates(m, 'fact', [1., 0.], summary_k=10,
                      dense_k=20, bm25_k=20, candidate_cap=25)
        self.assertEqual(len(capped), 25)

    def test_empty_memory_does_not_call_model(self):
        fake = StubClient()
        answer = opctx.OpCtx('locomo', client=fake).answer('Where is the cabinet?')
        self.assertEqual(answer['selected_fact_ids'], [])
        self.assertEqual(fake.calls, [])
        self.assertEqual(fake.embedding_calls, [])

    def test_reader_full_and_local_view_visibility(self):
        m = fixture_memory()
        full = opctx.text_reader_context(m, m['facts'], 'What cabinet?',
                                        StubClient([{'operation': 'atomic_lookup'}]))
        local = opctx.text_reader_context(m, m['facts'], 'What changed?',
                                         StubClient([{'operation': 'trace_change'}]))
        self.assertIn('Supported event V00001', full)
        self.assertNotIn('Supported event', local)
        self.assertIn('Local session summaries', local)
        self.assertIn('F000001', local)
        partial = opctx.text_reader_context(m, m['facts'][:1], 'What cabinet?',
                                        StubClient([{'operation': 'atomic_lookup'}]))
        self.assertNotIn('Supported event', partial)

    def test_memeye_qa_receives_reader_output_only(self):
        cfg = config.PROFILES['memeye']
        m = memory.empty_memory(cfg)
        m['packets'] = [{'id': 'P1', 'session_ids': ['s1'], 'timestamp': 'now',
                         'summary': 'RAW SUMMARY SENTINEL', 'vector': [1.]*cfg.embedding_dim,
                         'member_fact_ids': ['F1']}]
        m['facts'] = [{'id': 'F1', 'packet_id': 'P1', 'text': 'RAW FACT SENTINEL',
                      'sources': ['s1::t1'], 'vector': [1.]*cfg.embedding_dim}]
        fake = StubClient([{'direct_evidence': ['AGGREGATED EVIDENCE'],
                            'candidate_answer': 'AGGREGATED ANSWER'}, 'Final response'],
                          cfg.embedding_dim)
        app = opctx.OpCtx('memeye', memory=m, client=fake)
        with patch('opctx.memeye_candidates', return_value=m['facts']), \
             patch('opctx.mathematical_select', return_value=m['facts']):
            result = app.answer('What is shown?')
        self.assertEqual(result['answer'], 'Final response')
        self.assertIn('RAW FACT SENTINEL', fake.calls[0][0])
        self.assertNotIn('RAW FACT SENTINEL', fake.calls[1][0])
        self.assertNotIn('RAW SUMMARY SENTINEL', fake.calls[1][0])
        self.assertIn('AGGREGATED EVIDENCE', fake.calls[1][0])

    def test_rerank_rejects_unknown_or_duplicate_ids(self):
        facts = [{'id': 'A', 'text': 'First'}, {'id': 'B', 'text': 'Second'}]
        cfg = replace(config.PROFILES['locomo'], final_k=2)
        for ids in (['A', 'A'], ['A', 'UNKNOWN'], ['A']):
            with self.assertRaisesRegex(ValueError, 'Invalid reranker'):
                retrieval.rerank(facts, 'Question', StubClient([{'ranked_fact_ids': ids}]), cfg)
        chosen = retrieval.rerank(facts, 'Question',
                                 StubClient([{'ranked_fact_ids': ['B', 'A']}]), cfg)
        self.assertEqual([f['id'] for f in chosen], ['B', 'A'])

    def test_query_projection_requires_verbatim_spans(self):
        question = 'What color is the cabinet?'
        projected = retrieval.project_question(question, StubClient([{
            'entity': ['the cabinet'], 'relation': ['is'], 'attribute': ['color'], 'scope': []}]))
        self.assertEqual(projected[0], question)
        bad = retrieval.project_question(question, StubClient([{
            'entity': ['the desk'], 'relation': [], 'attribute': [], 'scope': []}]))
        self.assertEqual(bad, [question])

    def test_template_substitution_is_not_recursive(self):
        result = opctx.fill('Q={question}\nC={context}',
                            question='Does {context} expand?', context='Has {question}')
        self.assertEqual(result, 'Q=Does {context} expand?\nC=Has {question}')

    def test_dense50_budget_uses_all_visible_facts(self):
        m = fixture_memory()
        cfg = replace(config.PROFILES['locomo'], final_k=1, dense_budget_k=2)
        fake = StubClient([{'entity': [], 'relation': [], 'attribute': [], 'scope': []}])
        with patch('retrieval.select_multichannel', return_value={'selected_ids': ['F000001']}) as select:
            retrieval.mathematical_select(m, m['facts'][:1], 'cabinet',
                                         [1.] + [0.]*767, fake, cfg)
        self.assertEqual(select.call_args.kwargs['char_budget'],
                         len(retrieval.render_facts(m, m['facts'])))
        self.assertEqual(select.call_args.kwargs['max_units'], 1)

    def test_selector_count_cost_and_lexical_coverage(self):
        units = [{'id': 'A', 'text': 'red', 'cost': 3, 'group': 'p1'},
                 {'id': 'B', 'text': 'blue', 'cost': 4, 'group': 'p1'},
                 {'id': 'C', 'text': 'unrelated', 'cost': 9, 'group': 'p2'}]
        channels = [{'query_id': 'q0', 'query': 'red blue',
                     'dense_scores': [.8, .7, .9], 'bm25_scores': [1., 1., 0.]}]
        result = selector.select_multichannel(units, 'red blue', channels,
                          max_units=2, char_budget=9, group_costs={'p1': 2, 'p2': 2})
        self.assertEqual(set(result['selected_ids']), {'A', 'B'})
        self.assertEqual(result['cost'], 9)
        self.assertLessEqual(result['selected_count'], 2)
        self.assertTrue(result['certified_optimal'])

    def test_selector_enforces_provenance_duplicate_exclusion(self):
        units = [{'id': i, 'text': 'red', 'cost': 3, 'group': 'p1', 'sources': ['s1']}
                 for i in ['A', 'B']]
        channels = [{'query_id': 'q0', 'query': 'red',
                     'dense_scores': [.8, .7], 'bm25_scores': [1., 1.]}]
        result = selector.select_multichannel(units, 'red', channels,
                        max_units=2, char_budget=100, group_costs={'p1': 2})
        self.assertEqual(result['selected_count'], 1)
        self.assertEqual(result['cost'], 5)

    def test_selector_rejects_nonfinite_scores(self):
        units = [{'id': 'A', 'text': 'red', 'cost': 3, 'group': 'p1'}]
        channels = [{'query_id': 'q0', 'query': 'red',
                     'dense_scores': [float('nan')], 'bm25_scores': [1.]}]
        with self.assertRaisesRegex(ValueError, 'finite'):
            selector.select_multichannel(units, 'red', channels,
                         max_units=1, char_budget=100, group_costs={'p1': 2})

    def test_client_normalizes_and_validates_embeddings(self):
        cfg = replace(config.PROFILES['locomo'], embedding_dim=2)
        gemini = client.Gemini.__new__(client.Gemini)
        gemini.config = cfg
        with patch.object(gemini, '_post', return_value={'embeddings': [{'values': [3., 4.]}]}):
            self.assertTrue(np.allclose(gemini.embed(['text']), [[.6, .8]]))
        with patch.object(gemini, '_post', return_value={'embeddings': [{'values': [0., 0.]}]}):
            with self.assertRaisesRegex(ValueError, 'zero'):
                gemini.embed(['text'])

    def test_client_rejects_truncation_and_invalid_json(self):
        gemini = client.Gemini.__new__(client.Gemini)
        gemini.config = config.PROFILES['locomo']
        with patch.object(gemini, '_post', return_value={
            'candidates': [{'finishReason': 'MAX_TOKENS'}]}):
            with self.assertRaisesRegex(RuntimeError, 'MAX_TOKENS'):
                gemini.generate('test')
        with patch.object(gemini, '_post', return_value={
            'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '[]'}]}}]}):
            with self.assertRaisesRegex(ValueError, 'JSON object'):
                gemini.generate('test')


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]] + unittest_args, verbosity=2)

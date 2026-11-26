"""Offline integration checks; no provider request is allowed or needed."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import unittest

import numpy as np
from analysis.hybrid_rag_20261114 import run, data


class FakeAdapter:
    def __init__(self):
        self.events=[]

    def rerank(self, dataset, question, candidates):
        self.events.append('rerank')
        assert 'secret_gold_only' not in str(candidates)
        return dict(selected_ids=[c['id'] for c in candidates[:20]],audit={})

    def answer(self, dataset, question, context, question_date=''):
        self.events.append('answer')
        assert 'secret_gold_only' not in context
        return dict(prediction='test prediction',audit={})

    def judge(self, dataset, question, reference, prediction):
        self.events.append('judge')
        assert reference=='secret_gold_only'
        return dict(correct=True,raw='[[yes]]',audit={})


class TestRunner(unittest.TestCase):
    def test_checkpoint_gold_boundary_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            sessions=[dict(session_id='1',date='2026-01-01',
                turns=[dict(speaker='User',source_id='D1:1',text='The red bicycle is here.')])]
            context=dict(context_id='conv-test',sessions=sessions)
            question=dict(question_id='conv-test-q000',question='What bicycle?',
                raw_question='What bicycle?',question_date='',answer='secret_gold_only',
                category=4,category_name='single')
            args=SimpleNamespace(output=output,dataset='locomo')
            wire=SimpleNamespace(local=threading.local(),stop=threading.Event())
            embed_calls=[]
            def embed(texts,**kwargs):
                embed_calls.append(dict(texts=texts,kwargs=kwargs))
                return [[1.0]+[0.0]*767 for _ in texts]
            adapter=FakeAdapter()
            index=run.build_index(context,args,wire,embed)
            result=run.process_question(context,question,index,args,wire,embed,adapter)
            self.assertEqual(adapter.events,['rerank','answer','judge'])
            self.assertTrue(result['correct'])
            self.assertEqual(len(embed_calls),2)
            result2=run.process_question(context,question,index,args,wire,embed,adapter)
            self.assertEqual(result,result2)
            run.build_index(context,args,wire,embed)
            self.assertEqual(len(embed_calls),2)
            self.assertEqual(adapter.events,['rerank','answer','judge'])
            summary=run.summarize(output,dict(dataset='locomo',expected=1),1.0,'smoke')
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['overall']['accuracy_percent'],100.0)

    def test_failure_is_missing_and_not_replayed(self):
        from analysis.hybrid_rag_20261114.stages import StageFailure
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            context=dict(context_id='context',sessions=[dict(session_id='s',date='date',
                turns=[dict(speaker='User',source_id='u1',text='A small piece of evidence.')])])
            q=dict(question_id='q1',question='What?',raw_question='What?',question_date='',
                   answer='secret_gold_only',category=4,category_name='single')
            args=SimpleNamespace(output=output,dataset='locomo')
            wire=SimpleNamespace(local=threading.local(),stop=threading.Event())
            embed=lambda texts,**kwargs: [[1.0]+[0.0]*767 for _ in texts]
            index=run.build_index(context,args,wire,embed)
            adapter=FakeAdapter()
            def invalid(*a,**kw):
                raise StageFailure('Invalid rerank IDs')
            adapter.rerank=invalid
            with self.assertRaises(StageFailure):
                run.process_question(context,q,index,args,wire,embed,adapter)
            self.assertFalse((output/'questions/q1/result.json').exists())
            self.assertTrue((output/'questions/q1/failure.json').exists())
            with self.assertRaises(RuntimeError):
                run.process_question(context,q,index,args,wire,embed,adapter)


if __name__=='__main__':
    unittest.main()

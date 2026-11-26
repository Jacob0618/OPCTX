import inspect
import unittest
from unittest.mock import patch
from . import core,run


def fixture(n=8):
    facts=[dict(id=f'f{i}',text=f'event attribute{i}',session_id=1,source_dia_ids=[f's{i}']) for i in range(n)]
    candidates=[dict(fact=f,dense_score=(n-i)/n) for i,f in enumerate(facts)]
    graph=dict(summary_nodes=[dict(id='S1',session_ids=[1],node_type='session_packet',text='a summary',timestamp='date')])
    channels=[dict(query_id='q0',query='event attribute7',dense_scores=[c['dense_score'] for c in candidates],bm25_scores=[0.]*(n-1)+[1.])]
    return candidates,graph,channels


class Tests(unittest.TestCase):
    def test_dense_exact_count(self):
        c,g,ch=fixture(); s=core.choose(c,g,'event attribute7',ch,'dense50',limit=3)
        self.assertEqual(s['prefilter_fact_ids'],['f0','f1','f2'])
        self.assertEqual(s['rerank_input_count'],3)

    def test_full_pool(self):
        c,g,ch=fixture(); s=core.choose(c,g,'event attribute7',ch,'full_pool',limit=3)
        self.assertEqual(len(s['prefilter_fact_ids']),8)

    def test_math_exact_count_and_coverage(self):
        c,g,ch=fixture(); s=core.choose(c,g,'event attribute7',ch,'math50',limit=3)
        self.assertEqual(len(s['prefilter_fact_ids']),3)
        self.assertIn('f7',s['prefilter_fact_ids'])
        self.assertEqual(s['optimization']['min_units'],3)
        self.assertEqual(s['optimization']['max_units'],3)

    def test_common_pool_bound_and_order(self):
        c,g,ch=fixture(); a=core.choose(c,g,'event attribute7',ch,'dense50',limit=3); b=core.choose(c,g,'event attribute7',ch,'math50',limit=3)
        self.assertEqual(a['candidate_fact_ids'],b['candidate_fact_ids'])
        self.assertEqual(a['nonbinding_character_bound'],b['nonbinding_character_bound'])
        self.assertLessEqual(b['optimization']['cost'],b['nonbinding_character_bound'])
        ids=b['prefilter_fact_ids']; self.assertEqual(ids,sorted(ids,key=a['candidate_fact_ids'].index))

    def test_duplicate_constraint(self):
        c,g,ch=fixture(); c[1]['fact'].update(text=c[0]['fact']['text'],source_dia_ids=c[0]['fact']['source_dia_ids'])
        s=core.choose(c,g,'event attribute7',ch,'dense50',limit=3)
        self.assertEqual(s['prefilter_fact_ids'],['f0','f2','f3'])

    def test_short_pool_fails(self):
        c,g,ch=fixture(2)
        with self.assertRaises(ValueError): core.choose(c,g,'event attribute7',ch,'dense50',limit=3)

    def test_no_answer_api(self):
        for fn in (core.select,core.choose):
            self.assertFalse({'answer','category','correct','operation'}&set(inspect.signature(fn).parameters))

    def test_forbidden_stage(self):
        obj=object.__new__(run.Runtime)
        with patch.object(run.Runtime,'_halt',side_effect=run.StopRun('blocked')):
            with self.assertRaises(run.StopRun): obj.post('online','s','gemini-2.5-flash','generateContent',{})

    def test_final_rerank_validation(self):
        c,_,_=fixture()
        run.p.validate_selection(['f0','f1','f2','f3','f4'],c)
        for ids in (['f0']*5,['f0','f1','f2','f3','missing'],['f0']):
            with self.assertRaises(run.StopRun): run.p.validate_selection(ids,c)


if __name__=='__main__': unittest.main()

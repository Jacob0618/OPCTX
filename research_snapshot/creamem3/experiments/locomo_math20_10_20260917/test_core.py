import copy
import inspect
import unittest
from unittest.mock import patch
from . import core
from creamem3.experiments.locomo_prefilter50_rerank5_20260913.test_core import fixture


class Tests(unittest.TestCase):
    def test_short_pool_bypasses(self):
        for limit in core.LIMITS:
            c,g,ch=fixture(8)
            with patch.object(core.parent.old.solver,'select_multichannel',side_effect=AssertionError('Unexpected solve')):
                s=core.choose(c,g,'event attribute7',ch,limit)
            self.assertTrue(s['bypass'])
            self.assertEqual(s['prefilter_fact_ids'],[x['fact']['id'] for x in c])

    def test_exact_count_order_objective(self):
        for limit in core.LIMITS:
            c,g,ch=fixture(25); ch[0]['query']='event attribute24'
            s=core.choose(c,g,'event attribute24',ch,limit)
            self.assertEqual(len(s['prefilter_fact_ids']),limit)
            self.assertIn('f24',s['prefilter_fact_ids'])
            self.assertTrue(s['optimization']['coverage_first'])
            self.assertTrue(s['optimization']['max_min_balance'])
            self.assertEqual(s['optimization']['min_units'],limit)
            self.assertEqual(s['optimization']['max_units'],limit)
            core.verify(s,c,g,'event attribute24',ch,limit)

    def test_same_budget_shared_by_rerank_and_direct(self):
        self.assertEqual(core.MODES['math20_rerank5'][0],core.MODES['math20_direct'][0])
        self.assertEqual(core.MODES['math10_rerank5'][0],core.MODES['math10_direct'][0])
        self.assertEqual(sum(r for _,r in core.MODES.values()),2)

    def test_tamper_rejected(self):
        c,g,ch=fixture(25); ch[0]['query']='event attribute24'
        s=core.choose(c,g,'event attribute24',ch,10)
        for key,value in [('limit',20),('prefilter_fact_ids',s['prefilter_fact_ids'][::-1]),('bypass',True)]:
            bad=copy.deepcopy(s); bad[key]=value
            with self.assertRaises(ValueError): core.verify(bad,c,g,'event attribute24',ch,10)

    def test_no_gold_selection_input(self):
        self.assertFalse({'answer','category','correct','operation'} & set(inspect.signature(core.choose).parameters))

    def test_direct_never_calls_model_rerank(self):
        from types import SimpleNamespace
        from . import run
        for limit in core.LIMITS:
            candidates,_,_=fixture(limit)
            ids=[c['fact']['id'] for c in candidates]
            index=SimpleNamespace(
                _candidate_pool=lambda *args:(candidates,{'top_summaries':[]}),
                _assemble=lambda picked,summary:('context',{'ids':[c['fact']['id'] for c in picked]}),
                _rerank=lambda *args:(_ for _ in ()).throw(AssertionError('Forbidden model reranker')))
            data={'samples':{'q':'s'},'questions':{'q':{'question':'event'}},'narrow_indexes':{'s':index},
                  'vectors':{'q':[]},'graphs':{('s',run.ARM):{}},'operations':{'q':{}}}
            with patch.object(run,'selection',return_value={'prefilter_fact_ids':ids}), patch.object(run,'identity',return_value={}), patch.object(run.p,'render',return_value=('context',{})):
                result=run.retrieve(data,'math'+str(limit)+'_direct','q',None)
            self.assertEqual(result['final_fact_count'],limit)
            self.assertEqual(result['rerank_input_count'],0)
            self.assertEqual(result['trace']['rerank']['selected_fact_ids'],ids)


if __name__=='__main__': unittest.main()

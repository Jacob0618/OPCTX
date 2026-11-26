import unittest
from . import run

class Controls(unittest.TestCase):
    def test_limits_order(self):
        for n in (8,20,21,120):
            pool=[dict(fact=dict(id=str(i)),dense_score=1-i/200) for i in range(n)]
            self.assertEqual(run.choose(pool),[str(i) for i in range(min(n,20))])
    def test_reject_invalid(self):
        with self.assertRaises(run.StopRun): run.choose([])
        with self.assertRaises(run.StopRun): run.choose([dict(fact=dict(id='a'),dense_score=0),dict(fact=dict(id='b'),dense_score=1)])
    def test_cumulative_not_loo(self):
        for line in ('direct','rerank5'):
            cfg=[run.MODES[line+'_'+s] for s in ('wo_selector','wo_selector_reader','wo_selector_reader_online')]
            self.assertEqual([(x['reader'],x['online']) for x in cfg],[(True,True),(False,True),(False,False)])
            self.assertTrue(all(x['rerank']==(line=='rerank5') for x in cfg))
    def test_exact_runtime(self):
        self.assertEqual(set(run.prior.ALLOWED),{'rerank','qa','judge'})
        self.assertEqual(len(run.MODES),6)

if __name__=='__main__': unittest.main()

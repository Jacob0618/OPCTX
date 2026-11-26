import unittest
import numpy as np
from . import run as r

class Controls(unittest.TestCase):
    def test_population_and_inputs(self):
        reg=r.validate();self.assertEqual(len(reg['rows']),371)
        for ref in reg['rows']:
            item=r.b.read(ref['path']);prior=r.b.read(item['source_prepared'])
            self.assertEqual(item['caps'],prior['caps'])
            self.assertEqual(item['selections']['generic_reasoning']['context'],prior['full_context'])
            for arm,s in item['selections'].items():
                self.assertTrue(0<len(s['ids'])<=30)
                self.assertEqual(len(s['ids']),len(set(s['ids'])))
                self.assertLessEqual(len(s['context']),item['budget'])
                self.assertEqual(s['context_sha256'],r.b.sha(s['context']))
    def test_mmr_definition(self):
        dense=np.array([.9,.85,.7]);sim=np.array([[1,.95,.1],[.95,1,.1],[.1,.1,1]])
        score=.5*dense-.5*sim[:,[0]].max(axis=1)
        self.assertGreater(score[2],score[1])
    def test_fused_format_matches_original(self):
        reg=r.b.read(r.ROOT/'registration.json');auth=r.b.authorities()[2]
        for ref in reg['rows']:
            x=r.b.read(ref['path'])
            if x['route']!='structured_fused':continue
            prior=r.b.read(x['source_prepared']);hist=r.b.read(prior['full_prediction_path'])
            manifest,_=r.old.request_manifest(x['query']['task'],hist['stages'][0]['generation'])
            self.assertEqual(auth['fused'].format(context=prior['full_context'],question=x['query']['question']),manifest['prompt'])

if __name__=='__main__':unittest.main()

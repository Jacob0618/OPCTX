import unittest
from collections import Counter
from . import run as r


class SelectionTests(unittest.TestCase):
    def test_complete_balanced_partition(self):
        data, manifests, _ = r.derive()
        first, second = [set(m['ids']) for m in manifests]
        self.assertEqual((len(first), len(second), len(first & second)), (100,100,0))
        old = set()
        for root in r.OLD:
            old |= {x['question_id'] for x in r.read(root/'selection.json')['selection']['items']}
        self.assertFalse((first|second)&old)
        self.assertEqual(first|second|old,{q['question_id'] for q in data})
        self.assertEqual(manifests, r.derive()[1])
        for key in manifests[0]['type_counts']:
            self.assertLessEqual(abs(manifests[0]['type_counts'][key]-manifests[1]['type_counts'][key]),2)

    def test_inference_functions_reused(self):
        config, _ = r.bind(4)
        old = r.read(r.OLD[-1]/'summary.json')['config']
        for key in old:
            if key not in ('manifest_sha256','campaign'):
                self.assertEqual(old[key], config[key], key)
        self.assertIs(r.clean.core._process_question, r.clean._process_question)
        self.assertEqual(config['models']['writer'], 'gemini-2.5-flash')
        self.assertEqual(config['models']['judge'], 'gemini-3-flash-preview')
        self.assertEqual(config['architecture']['lme_final_k'],30)
        self.assertFalse(config['architecture']['writer_top3'])


if __name__ == '__main__':unittest.main()

"""Provider-cold parity and isolation checks; no paid calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import run as r


class ProtocolTests(unittest.TestCase):
    def test_scope_models_and_judges(self):
        self.assertEqual(len(r.ARMS),5)
        self.assertEqual(set(r.rt.MODELS.values()),{'gemini-2.5-flash-lite','gemini-3.5-flash-lite'})
        self.assertEqual(set(r.rt.JUDGES.values()),{'gemini-3.5-flash','gemini-3.5-flash-lite'})
        self.assertNotIn('gemini-2.5-flash',r.rt.MODELS.values())

    def test_stage_config(self):
        for bb,model in r.rt.MODELS.items():
            for stage in ('caption','writer','view','online_revision','online_revision_repair'):
                cfg=r.rt.config(model,stage,2400)
                expected={'thinkingBudget':512} if bb=='g25lite' else {'thinkingLevel':'low'}
                self.assertEqual(cfg['thinkingConfig'],expected)
                self.assertNotIn('temperature',cfg)
            cfg=r.rt.config(model,'focused',2600,True)
            self.assertEqual(cfg['responseMimeType'],'application/json')
        for model in r.rt.JUDGES.values():
            self.assertEqual(r.rt.config(model,'judge',400)['thinkingConfig'],{'thinkingLevel':'minimal'})

    def test_actual_historical_caption_payload(self):
        task='Cartoon_Entertainment_Companion'
        old=Path(r'D:\creamem3_cache\memeye_v19\isolated_full_arch_thinking512_low_r1')
        c=r.b.read(sorted((old/'captions/g35'/task).glob('*.json'))[0])
        prompt=r.build.matrix.build_v7_reconstruction_user_text(task,c['image'],sequence_index=1,total_images=87,previous=[])
        wire,ref=r.rt.request(r.rt.MODELS['g35lite'],'caption',prompt,r.build._load_g35_caption_system_prompt(),2400,False,r.b.IMAGE_ROOT/c['image'])
        self.assertEqual(r.b.sha(wire),'9c18cfecfbef56d656fea814ede9eb585146e7c698e75f0cbb82e5beacde4a92')
        self.assertEqual(ref['sha256'],c['input_contract']['raw_image_sha256'])
        self.assertIn('inlineData',wire['contents'][0]['parts'][0])

    def test_routes_original(self):
        reg=r.b.read(r.SOURCE/'registration.json')
        for q in r.b.read(r.SOURCE/'queries_open.json')['rows']:
            self.assertEqual(r.build.matrix.reader_route_for_task(q['task']),reg['opctx_routes'][q['row_id']])

    def test_prediction_uses_requested_backbone_and_reader_policy(self):
        class Fake:
            model='gemini-3.5-flash-lite'
            calls=[]
            def generate(self,stage,prompt,**kw):
                self.calls.append(stage)
                return r.rt.prior.Reply('fixed answer','r','s',self.model,self.model,{'prompt_tokens':1,'output_tokens':1})
        q={'row_id':'probe','task':'Cartoon_Entertainment_Companion','question':'what?'}
        item={'query':q,'context':'source','context_sha256':r.b.sha('source')}
        reg={'seal_sha256':'test','routes':{'probe':'two_stage'}}
        with tempfile.TemporaryDirectory(prefix='memeye-ablation-test-') as tmp:
            fake=Fake()
            result=r.predict(Path(tmp),'no_reader',item,fake,reg)
            self.assertEqual(fake.calls,['answer'])
            self.assertEqual(result['model'],fake.model)
            self.assertEqual(result['prediction'],'fixed answer')

    def test_two_judges_share_identical_prediction(self):
        records=[]
        class Judge:
            def __init__(self,root,model,allow):self.model=model
            def judge(self,q,gold,answer,metadata):
                records.append((self.model,answer,metadata['prediction_sha256']))
                out=r.rt.prior.Reply('judge','r','s',self.model,self.model,{})
                from types import SimpleNamespace
                return out,SimpleNamespace(raw_score=1.,mapped_score=1.,reasoning='test')
        q=r.b.read(r.SOURCE/'queries_open.json')['rows'][0]
        pred={'seal_sha256':'fixed-prediction','prediction':'same answer'}
        with tempfile.TemporaryDirectory(prefix='memeye-dual-judge-test-') as tmp:
            with patch.object(r,'predict',return_value=pred),patch.object(r.rt,'Runtime',Judge):
                from types import SimpleNamespace
                r.evaluate(Path(tmp),'no_reader',{'query':q},SimpleNamespace(model='test'),{},False)
        self.assertEqual(len(records),2)
        self.assertEqual({x[0] for x in records},set(r.rt.JUDGES.values()))
        self.assertEqual({x[1:] for x in records},{('same answer','fixed-prediction')})

    def test_projection_is_extractive_and_original_retained(self):
        q='What color is the cabinet?'
        parsed=r.selection.qp.validate_response(q,json.dumps({'entity':['cabinet'],'relation':[], 'attribute':['color'],'scope':[]}))
        self.assertEqual(parsed['status'],'valid')
        channels=r.selection.query_texts(q,parsed,'q0_plus_anchored_attributes')
        self.assertEqual(channels[0]['query'],q)
        self.assertTrue(all(word in q for word in channels[1]['query'].split()))


if __name__=='__main__':unittest.main()

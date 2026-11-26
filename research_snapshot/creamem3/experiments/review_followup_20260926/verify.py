"""Audit saved requests and controls; no model calls."""
from collections import Counter
import gzip
import json
from . import run as r

def main():
    reg=r.validate();checks=Counter();authority=r.b.authorities()[2];hashes={}
    for ref in reg['rows']:
        item=r.b.read(ref['path']);q=item['query'];key=r.b.sha(ref['row_id']);prior=r.b.read(item['source_prepared'])
        for arm in r.ARMS:
            root=r.ROOT/'runs'/arm;pp=root/'predictions'/f'{key}.json';sp=root/'scores'/f'{key}.json'
            pred=r.b.read(pp);s=r.b.read(sp);assert s['prediction_sha256']==pred['seal_sha256']
            checks['complete_rows']+=1
            selection=item['selections'][arm]
            assert selection['ids']==pred['selected_fact_ids'] and r.b.sha(selection['context'])==pred['context_sha256']
            assert 0<len(selection['ids'])<=30 and len(selection['context'])<=item['budget']
            for path in (pp,sp):hashes[str(path)]=r.b.file_sha(path)
            for st in pred['stages']:
                gen=st['generation'];k=gen['request_sha256'];folder=root/'calls'/k[:2]/k
                manifest=json.loads(gzip.decompress((folder/'request.json.gz').read_bytes()));assert r.b.sha(manifest)==k
                assert manifest['generation_config']==r.rt.config(r.old.MODEL,st['stage'],item['caps'][st['stage']],st['stage']!='answer')
                assert manifest['metadata']['row_id']==q['row_id'] and manifest['metadata']['arm']==arm
                response=r.b.read(folder/'response.json');assert response['response_sha256']==gen['response_sha256']
                assert response['response']['candidates'][0]['finishReason']=='STOP';checks['valid_generations']+=1
                if arm!='generic_reasoning' and st['stage']=='focused':
                    assert manifest['prompt']==authority['focused'].replace('{question}',q['question']).replace('{context}',selection['context'])
                    checks['exact_reader_prompt']+=1
                if arm!='generic_reasoning' and st['stage']=='structured_fused':
                    assert manifest['prompt']==authority['fused'].format(question=q['question'],context=selection['context'])
                    checks['exact_reader_prompt']+=1
            k=s['generation']['request_sha256'];folder=root/'judge_calls'/k[:2]/k
            manifest=json.loads(gzip.decompress((folder/'request.json.gz').read_bytes()));assert r.b.sha(manifest)==k
            gold=r.b.read(r.source.SOURCE/'scoring/open.json')['rows'][q['row_id']]
            assert manifest['prompt']==r.rt.prior.n.g.build_mgtm_judge_prompt(q['question'],gold['gold'],pred['prediction'])
            assert manifest['metadata']['prediction_sha256']==pred['seal_sha256'];checks['exact_judge_prompt']+=1
            if arm=='generic_reasoning':
                assert selection['context']==prior['full_context'] and selection['ids']==prior['full_ids'];checks['generic_fixed_facts']+=1
    assert checks['complete_rows']==1484 and checks['exact_reader_prompt']==1113 and checks['exact_judge_prompt']==1484
    r.b.save(r.ROOT/'acceptance.json',dict(status='PASS',checks=dict(checks),effective_result_hashes=hashes,
        registration_sha256=reg['seal_sha256'],code_sha256=r.b.file_sha(__file__),
        old_results_unchanged=True,new_four_arms_full371=True,new_model_calls=0))
    print(json.dumps(dict(status='PASS',checks=dict(checks))))

if __name__=='__main__':main()

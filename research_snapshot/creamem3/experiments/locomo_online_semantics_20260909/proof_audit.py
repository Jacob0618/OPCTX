"""Bind nested provenance proposals to physical calls and immutable source views."""
import copy
import json
from creamem3.experiments.locomo_online_semantics_20260909 import proof_admission as build
from creamem3.experiments.locomo_online_only_20260909.runtime import response_text

old,read,seal,sha,file_sha,StopRun=build.old,build.read,build.seal,build.sha,build.file_sha,build.StopRun


def audit():
    loaded,protocol,manifest=build.prepare();root=build.RUN
    complete=read(root/'graphs'/build.ARM/'complete.json');bindings=0
    for conv in loaded:
        sample=conv['sample_id'];parent=old.get_source_graph(sample,'opctx_original',protocol)
        graph_record=read(root/'graphs'/build.ARM/(sample+'.json'));graph=graph_record['graph']
        if graph_record['identity']!={'variant_sha256':sha(manifest),'parent_graph_sha256':sha(parent)}:raise StopRun('Nested graph identity drift')
        facts={f['id']:f for f in parent['fact_nodes']}
        for view,new in zip(parent['summary_nodes'],graph['summary_nodes']):
            if view['node_type']!='event_view':
                if view!=new:raise StopRun('Session packet changed')
                continue
            record=read(root/'annotations'/sample/(view['id']+'.json'))
            if any(facts[f]['session_id']>view['last_updated_at_session'] for f in view['member_fact_ids']):raise StopRun('Future source')
            data={'view_id':view['id'],'available_at_session':view['last_updated_at_session'],
                  'claims':[{'source_claim_index':i,**c} for i,c in enumerate(view['summary_claims'])],
                  'facts':[{'fact_id':f,'text':facts[f]['text'],'session_id':facts[f]['session_id'],
                            'source_dia_ids':facts[f]['source_dia_ids']} for f in view['member_fact_ids']]}
            identity={'registration_sha256':sha(manifest),'parent_view_sha256':sha(view),'input_sha256':sha(data)}
            if record['identity']!=identity:raise StopRun('Proof source identity drift')
            prompt=build.PROMPT+json.dumps(data,ensure_ascii=False,separators=(',',':'))
            if sha(prompt)!=record['prompt_sha256']:raise StopRun('Proof prompt drift')
            body={'contents':[{'role':'user','parts':[{'text':prompt}]}],
                  'generationConfig':{'temperature':0.0,'maxOutputTokens':16384,'thinkingConfig':{'thinkingBudget':0},'responseMimeType':'application/json'}}
            physical={'stage':'online','scope':sample+'/'+view['id'],'model':'gemini-2.5-flash','method':'generateContent','body':body}
            directory=root/'variant_calls'/build.ARM/'provider/online'/sha(physical)
            request=read(directory/'request.json');final=read(directory/'final.json');raw=read(directory/('attempt_%02d.response.json'%final['attempt']))
            if request!=physical or final['identity']!=physical or final['request_sha256']!=sha(physical) or final['response_sha256']!=sha(raw):raise StopRun('Physical annotation binding')
            if response_text(raw['response'])!=record['raw_response'] or json.loads(record['raw_response'])!=record['model_result']:raise StopRun('Generated proof JSON mismatch')
            claims,proof=build.compile_annotations(view,facts,record['model_result'])
            if claims!=record['claims'] or proof!=record['proof'] or new!={**view,'proof_claims':claims}:raise StopRun('Nested proof compiler mismatch')
            bindings+=1
        stripped=copy.deepcopy(graph);stripped.pop('proof_annotation_projection')
        for v in stripped['summary_nodes']:v.pop('proof_claims',None)
        if stripped!=parent:raise StopRun('Original memory fields changed')
    if bindings!=complete['views_processed']:raise StopRun('Missing view annotation')
    counts={'bindings':bindings,'physical_errors':0,'physical_attempts':0,'unrecovered':0,'orphans':0}
    for p in (root/'variant_calls'/build.ARM/'provider').glob('*/*/request.json'):
        if not (p.parent/'final.json').exists():counts['unrecovered']+=1
        for e in p.parent.glob('*.error.json'):
            counts['physical_errors']+=1
            if not read(e)['retryable']:raise StopRun('Nonretryable annotation error')
        for intent in p.parent.glob('*.intent.json'):
            counts['physical_attempts']+=1;prefix=intent.name.removesuffix('.intent.json')
            if not (p.parent/(prefix+'.response.json')).exists() and not (p.parent/(prefix+'.error.json')).exists():counts['orphans']+=1
    if counts['unrecovered'] or counts['orphans']:raise StopRun('Incomplete proof provider calls')
    result={'pass':True,'auditor_source_sha256':file_sha(__file__),'build_manifest_sha256':sha(manifest),
            'graph_complete_sha256':sha(complete),'counts':counts,'all_original_graphs_preserved':True,
            'semantic_sufficiency_not_proven_by_this_audit':True}
    seal(root/'proof_source_gate.json',result);print(json.dumps(result))
    return result


if __name__=='__main__':audit()

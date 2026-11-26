"""Score-blind longer-solve check of all original partial certificates."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import time
from . import run as r

def main():
    reg=r.validate();out=r.ROOT/'solver_sensitivity'
    audit=r.b.read(r.old.OUT/'audit_static/solver_current30.json')
    partial={x['row_id'] for x in audit['rows'] if not x['all_stages_certified']}
    refs=[x for x in reg['rows'] if x['row_id'] in partial];assert len(refs)==56
    registration=r.b.save(out/'registration.json',dict(parent_sha256=reg['seal_sha256'],rows=refs,
        code_sha256=r.b.file_sha(__file__),selection='all56 original non-fully-certified instances, regardless of score',
        time_limit_per_stage=30,original_time_limit=5,models_called=0))
    graphs={}
    for task in {x['task'] for x in refs}:
        m=r.b.read(r.old.oldroot(task)/'memory_complete.json');graphs[task]=r.b.read(m['session']['artifact_path'])
    def one(ref):
        path=out/'rows'/f"{r.b.sha(ref['row_id'])}.json"
        if path.exists():return r.b.read(path)
        item=r.b.read(ref['path']);prior=r.b.read(item['source_prepared']);g=graphs[ref['task']]
        facts,groups=r.sel.adapter.adapt_facts(g)
        began=time.perf_counter();opt=r.sel.selector.select_multichannel(facts,item['query']['question'],item['channels'],
            max_units=30,char_budget=item['budget']+1,group_costs=groups,time_limit=30.)
        ctx,_=r.sel.adapter.render_selection(g,opt['selected_ids'])
        assert len(ctx)<=item['budget'] and opt['cost']-1==len(ctx)
        value=r.b.save(path,dict(row_id=ref['row_id'],task=ref['task'],seconds=time.perf_counter()-began,
            original_context_sha256=r.b.sha(prior['full_context']),context_sha256=r.b.sha(ctx),
            same_ordered_ids=opt['selected_ids']==prior['full_ids'],same_set=set(opt['selected_ids'])==set(prior['full_ids']),
            overlap=len(set(opt['selected_ids'])&set(prior['full_ids'])),optimization=opt,
            selected_context=ctx,registration_sha256=registration['seal_sha256']))
        print('SOLVER',ref['row_id'],value['same_set'],opt['certified_optimal'],flush=True);return value
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[j.result() for j in as_completed([pool.submit(one,x) for x in refs])]
    summary=dict(n=len(rows),same_set=sum(x['same_set'] for x in rows),same_ordered=sum(x['same_ordered_ids'] for x in rows),
        complete_certificates=sum(x['optimization']['certified_optimal'] for x in rows),
        mean_overlap=sum(x['overlap'] for x in rows)/len(rows),changed_ids=[x['row_id'] for x in rows if not x['same_set']],
        model_calls=0,interpretation='Selection/optimization sensitivity only, not QA sensitivity if selected facts change.')
    r.b.save(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

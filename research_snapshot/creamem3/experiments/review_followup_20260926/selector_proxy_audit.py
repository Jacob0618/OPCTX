"""Compare literal-coverage/utility proxies; do not extend theorem outside D."""
import numpy as np
from . import run as r

def main():
    reg=r.b.read(r.ROOT/'registration.json');graphs={};rows=[]
    for ref in reg['rows']:
        item=r.b.read(ref['path']);prior=r.b.read(item['source_prepared']);task=ref['task']
        if task not in graphs:
            mem=r.b.read(r.old.oldroot(task)/'memory_complete.json');graphs[task]=r.b.read(mem['session']['artifact_path'])
        fs,_=r.sel.adapter.adapt_facts(graphs[task]);byid={f['id']:i for i,f in enumerate(fs)}
        qs=r.sel.selector.lexical.tokens(item['query']['question'])-r.sel.selector.lexical.STOPWORDS
        utils=r.sel.selector.prepare_channels(item['query']['question'],item['channels'],len(fs))['utilities']
        original=r.b.read(prior['source_context_path'])['optimization'];ideals=original['channel_ideal_values']
        assert set(ideals)==set(utils)
        def cov(ids):return len(qs & set().union(*(r.sel.selector.lexical.tokens(fs[byid[x]]['text']) for x in ids)))
        full_cov=cov(prior['full_ids']);assert full_cov==original['literal_coverage_optimum']
        for arm in ('bm25','fusion','mmr'):
            ids=item['selections'][arm]['ids'];coverage=cov(ids);ix=[byid[x] for x in ids]
            ratio=min(float(utils[k][ix].sum()/v) for k,v in ideals.items())
            rows.append(dict(row_id=ref['row_id'],arm=arm,coverage=coverage,full_coverage=full_cov,
                in_coverage_optimal_domain=coverage==full_cov,
                min_utility_ratio_against_full_domain_ideals=ratio,
                interpretation='Only domain-feasible rows qualify for same-D retention comparison; other ratios are diagnostic only.'))
    summary={}
    for arm in ('bm25','fusion','mmr'):
        rr=[x for x in rows if x['arm']==arm];valid=[x for x in rr if x['in_coverage_optimal_domain']]
        summary[arm]=dict(n=len(rr),coverage_domain_count=len(valid),
            mean_literal_coverage_loss=float(np.mean([x['full_coverage']-x['coverage'] for x in rr])),
            mean_same_domain_min_ratio=float(np.mean([x['min_utility_ratio_against_full_domain_ideals'] for x in valid])) if valid else None)
    r.b.save(r.ROOT/'selector_proxy_audit.json',dict(summary=summary,rows=rows,code_sha256=r.b.file_sha(__file__),model_calls=0))
    print(summary)

if __name__=='__main__':main()

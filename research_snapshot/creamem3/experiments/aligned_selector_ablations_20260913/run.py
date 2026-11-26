"""New isolated roots around frozen evaluation plumbing; no old-file writes."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys

from . import core
from creamem3.experiments.locomo_math_summary3_budget_20260913 import run as source

previous, p, transport, component = source.previous, source.p, source.transport, source.component
read,seal,sha,file_sha,StopRun = source.read,source.seal,source.sha,source.file_sha,source.StopRun
ROOT = Path(r'D:\creamem3_cache\aligned_selector_locomo_full1540_20260913_r1')
ARMS = previous.ARMS
CONFIGS = {m:dict(previous.CONFIGS['previous_2_5_5']) for m in core.MODES}
SOURCES = (source.ROOT/'calls', *source.SOURCES)
WORKERS = 12
Runtime = source.Runtime
runner = None


def load_private(name, path, package):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); module.__package__=package
    sys.modules[name]=module; spec.loader.exec_module(module)
    return module


def prepare():
    # Reuse the already audited cached-input loader. All of its seal operations
    # are captured IN MEMORY, preventing writes to its completed root. Its
    # normal source/hash/population checks still execute unchanged.
    input_loader = load_private(source.__package__+'._aligned_input_loader',source.__file__,source.__package__)
    emitted = {}
    def capture(path,value):
        emitted[Path(path).name]=copy.deepcopy(value)
        return value
    input_loader.seal = capture
    data = input_loader.prepare()
    source_reg = read(source.ROOT/'registration.json'); source_done = read(source.ROOT/'complete.json')
    if not source_done['audit_pass'] or source_done['registration_sha256']!=sha(source_reg):
        raise StopRun('Most recent completed source drift')
    for path,digest in source_reg['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Frozen source code drift '+path)
    frozen = emitted['source_snapshot.json']
    for name in ('complete.json','registration.json'):
        frozen[str(source.ROOT/name)] = file_sha(source.ROOT/name)
    own = [Path(__file__),Path(core.__file__),Path(__file__).with_name('protocol.md'),Path(__file__).with_name('test_core.py')]
    reg = {'schema':'aligned.locomo.selector.v1','n':1540,'total_rows':12320,'arms':list(ARMS),'configs':CONFIGS,
        'all_ids':sorted(data['questions']),'source_hashes':{**source_reg['source_hashes'],**{str(x):file_sha(x) for x in own}},
        'source_snapshot_sha256':sha(frozen),'source_registration_sha256':sha(source_reg),
        'models':source.ALLOWED,'max_final_facts':5,'budget':'unchanged ordinary globalDense5 source/session characters',
        'qa_prompt_sha256':sha(p.old.QA_PROMPT),'judge_prompt_sha256':sha(p.old.JUDGE_PROMPT),'judge_votes':1,
        'frozen_writer_views_projection_bank':True,'allowed_fresh_stages':['qa','judge'],'not_token_matched':True,
        'exact_request_sources':[str(x) for x in SOURCES],'workers':WORKERS,
        'full_math_reference':str(previous.ROOT/'rows/previous_2_5_5'),
        'memeye_counterpart_audit':str(ROOT.parent/'aligned_selector_memeye_20260913_r1/counterparts.json'),
        'no_score_based_promotion_or_tuning':True}
    mempath = ROOT.parent/'aligned_selector_memeye_20260913_r1/counterparts.json'
    counterpart = read(mempath)
    if counterpart['n']!=371 or not counterpart['pass'] or not counterpart['dense_exact_match_all_rows']:
        raise StopRun('Aligned MemEye counterpart gate not satisfied')
    frozen[str(mempath)]=file_sha(mempath); reg['source_snapshot_sha256']=sha(frozen)
    seal(ROOT/'source_snapshot.json',frozen); seal(ROOT/'registration.json',reg)
    data['registration']=reg
    data['indexes']={(c,s,a):p.source.enum_repair.Index(data['graphs'][(s,a)],**cfg)
        for c,cfg in CONFIGS.items() for s in sorted(set(data['samples'].values())) for a in ARMS}
    seal(ROOT/'questions_score_free.json',emitted['questions_score_free.json'])
    seal(ROOT/'preflight.json',{'pass':True,'n':1540,'new_model_calls':0,'registration_sha256':sha(reg),
        'memeye_counterparts_audit_sha256':sha(counterpart)})
    p.old.emit('ALIGNED PREFLIGHT PASS 4 selectors x2 memories x1540 questions')
    return data


def query_bank(question,q0,qid,runtime,registration):
    path=previous.ROOT/'query_bank'/(qid+'.json'); bank=read(path)
    if bank['question']!=question or bank['vectors'][question]!=q0: raise StopRun('Query-bank input drift')
    seal(ROOT/'query_bank_refs'/(qid+'.json'),{'registration_sha256':sha(registration),
        'source':str(path),'source_file_sha256':file_sha(path),'bank_sha256':sha(bank),'new_model_calls':0})
    return bank


def get_selection(data,c,a,qid,bank):
    path=ROOT/'selection'/c/a/(qid+'.json'); s=data['samples'][qid]
    binding={'row_identity':runner.identity(data,c,a,qid),'bank_sha256':sha(bank)}
    if path.exists():
        saved=read(path)
        if saved['identity']!=binding or saved['mode']!=c: raise StopRun('Resume selection drift')
        return saved
    saved={'identity':binding,**core.select(data['indexes'][(c,s,a)],data['indexes'][(c,s,'session_strong')],
        data['questions'][qid]['question'],bank,mode=c)}
    # Every intervention must keep EXACTLY the old full-method candidate pool,
    # q0 scores, projection bank and cap. It cannot alter initial retrieval.
    reference=read(previous.ROOT/'selection/previous_2_5_5'/a/(qid+'.json'))
    for key in ('candidate_trace','candidate_fact_ids','channels','dense5_ids','plain_budget'):
        if saved[key]!=reference[key]: raise StopRun('Component changed frozen retrieval/budget '+key)
    seal(path,saved); return saved


def verify_bank(data,qid):
    bank=previous.verify_bank(data,qid); path=previous.ROOT/'query_bank'/(qid+'.json')
    if read(ROOT/'query_bank_refs'/(qid+'.json'))!={'registration_sha256':sha(data['registration']),
        'source':str(path),'source_file_sha256':file_sha(path),'bank_sha256':sha(bank),'new_model_calls':0}:
        raise StopRun('Bank provenance mismatch')
    return bank


def finish(data,ids,smoke=False):
    banks={qid:verify_bank(data,qid) for qid in ids}
    rows={c:{a:{} for a in ARMS} for c in CONFIGS}; stats={}
    for c in CONFIGS:
        for a in ARMS:
            if not smoke and {f.stem for f in (ROOT/'rows'/c/a).glob('*.json')}!=set(ids): raise StopRun('Incomplete population')
            opts=[]
            for qid in ids:
                rows[c][a][qid]=runner.verify_one(data,c,a,qid,banks[qid])
                saved=read(ROOT/'selection'/c/a/(qid+'.json')); opt=saved['optimization']
                if saved['mode']!=c: raise StopRun('Mode mismatch')
                if c=='dense':
                    index=data['indexes'][(c,data['samples'][qid],a)]
                    candidates,_=index._candidate_pool(data['questions'][qid]['question'],data['vectors'][qid])
                    units,groups=core.adapt_candidates(candidates,data['graphs'][(data['samples'][qid],'session_strong')])
                    dense=core.dense_greedy(units,saved['channels'],max_units=5,char_budget=saved['plain_budget']+1,group_costs=groups)
                    if dense['selected_ids']!=saved['selected_fact_ids'] or dense['skipped']!=opt['skipped']: raise StopRun('Dense replay drift')
                else:
                    if opt['coverage_first']!=(c!='no_coverage') or opt['max_min_balance']!=(c!='utility_sum'): raise StopRun('Wrong objective flags')
                    if c=='q0_only' and opt['unique_queries']!=['q0']: raise StopRun('Projection ablation still used queries')
                opts.append(opt)
            stats[c+'/'+a]={'solver_certificate_applicable':c!='dense','certified_optimal':sum(o['certified_optimal'] for o in opts) if c!='dense' else None,
                'selected_counts':dict(Counter(o['selected_count'] for o in opts)), 'mean_seconds':sum(o['seconds'] for o in opts)/len(opts)}
            p.old.emit(f'ALIGNED AUDIT {c}/{a} {len(ids)}')
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path)!=digest: raise StopRun('Frozen code drift '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path)!=digest: raise StopRun('Frozen input drift '+path)
    physical=component.provider_audit(ROOT/'calls')
    if set(physical['stages'])-set(source.ALLOWED): raise StopRun('Forbidden new stage')
    reports={c:{a:component.summary(rows[c][a]) for a in ARMS} for c in CONFIGS}
    result={'registration_sha256':sha(data['registration']),'audit_pass':True,'smoke_only':smoke,'n':len(ids),
        'reports':reports,'full_reference':{a:component.summary({q:data['baselines'][a][q] for q in ids}) for a in ARMS},
        'paired_control_minus_full':{c:{a:p.paired(rows[c][a],data['baselines'][a],ids) for a in ARMS} for c in CONFIGS},
        'paired_online_vs_session':{c:p.paired(rows[c]['opctx_original'],rows[c]['session_strong'],ids) for c in CONFIGS},
        'selection_stats':stats,'physical_transport':physical,'new_model_reranker_calls':0,'not_token_matched':True}
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'),result)
    p.old.emit('ALIGNED '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))
    return result


def configure():
    global runner
    runner=load_private(__package__+'._frozen_evaluator',previous.__file__,__package__)
    for key,value in dict(ROOT=ROOT,CONFIGS=CONFIGS,ARMS=ARMS,SOURCES=SOURCES,WORKERS=WORKERS,Runtime=Runtime,
        prepare=prepare,query_bank=query_bank,get_selection=get_selection,verify_bank=verify_bank,finish=finish).items():
        setattr(runner,key,value)
    return runner


if __name__=='__main__': configure().main()

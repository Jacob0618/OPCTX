"""Audited cached-input reuse for four fixed full1540 read-side controls."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

from . import core
from creamem3.experiments.locomo_math_narrow_20260911 import run as previous

p, transport, component = previous.p, previous.transport, previous.component
read, seal, sha, file_sha, StopRun = previous.read, previous.seal, previous.sha, previous.file_sha, previous.StopRun
ROOT = Path(r'D:\creamem3_cache\locomo_math_summary3_budget_full1540_20260913_r1')
ARMS = previous.ARMS
CONFIGS = {
    'summary3_5_5_cap': dict(summary_top_k=3, dense_fact_top_k=5, bm25_fact_top_k=5, candidate_cap=120, rerank_top_k=5),
    'summary2_5_5_nocap': dict(summary_top_k=2, dense_fact_top_k=5, bm25_fact_top_k=5, candidate_cap=120, rerank_top_k=5),
}
POLICIES = {'summary3_5_5_cap': 'dense5', 'summary2_5_5_nocap': 'nonbinding'}
SOURCES = (previous.ROOT/'calls', *previous.SOURCES)
ALLOWED = {k: v for k, v in previous.ALLOWED.items() if k in {'qa', 'judge'}}
WORKERS = 12
runner = None


class Runtime(previous.Runtime):
    def post(self, stage, scope, model, method, body):
        if ALLOWED.get(stage) != model:
            self._halt('Only QA/Judge permitted in cached-bank experiment')
        return super().post(stage, scope, model, method, body)


def prepare():
    ref = read(previous.ROOT/'registration.json')
    completed = read(previous.ROOT/'complete.json')
    snapshot = read(previous.ROOT/'source_snapshot.json')
    if not completed['audit_pass'] or completed['n'] != 1540 or completed['registration_sha256'] != sha(ref) or sha(snapshot) != ref['source_snapshot_sha256']:
        raise StopRun('Prior completed mathematical lineage mismatch')
    for path, digest in ref['source_hashes'].items():
        if file_sha(path) != digest:
            raise StopRun('Frozen code drift: ' + path)
    frozen = {}
    def consume(path, *, required=True):
        path = Path(path); digest = file_sha(path)
        if required and snapshot.get(str(path)) != digest:
            raise StopRun('Consumed input drift against audited snapshot: ' + str(path))
        frozen[str(path)] = digest
    for name in ('registration.json', 'complete.json', 'source_snapshot.json'):
        consume(previous.ROOT/name, required=False)
    consume(p.old.DEFAULT_DATASET_PATH); consume(p.old.VECTOR_FILE)
    loaded = p.old.load_all_locomo()
    questions = {q['qa_id']: q for conv in loaded for q in conv['qa']}
    samples = {q['qa_id']: conv['sample_id'] for conv in loaded for q in conv['qa']}
    if len(questions) != 1540 or sorted(questions) != ref['all_ids']:
        raise StopRun('Question population drift')
    vector_record = read(p.old.VECTOR_FILE)
    vectors = dict(zip(vector_record['ids'], vector_record['vectors'], strict=True))
    if set(vectors) != set(questions):
        raise StopRun('Vector population drift')
    parent = read(p.SOURCE/'registration.json'); consume(p.SOURCE/'registration.json')
    graphs, hashes, operations, banks = {}, {}, {}, {}
    for sample in sorted(set(samples.values())):
        for arm in ARMS:
            for path in (p.original.graph_path(p.SOURCE, sample, arm), p.SOURCE/'memory_imports'/sample/(arm+'.json')):
                consume(path)
            graph = p.full.get_graph(p.SOURCE, sample, arm, parent['source_registration_sha256'])
            graphs[(sample, arm)] = graph; hashes[(sample, arm)] = sha(graph)
        sg, og = (graphs[(sample, a)] for a in ARMS)
        if sg['fact_nodes'] != og['fact_nodes'] or sg['session_packets'] != og['session_packets']:
            raise StopRun('Shared Writer equality drift')
    for qid in sorted(questions):
        path = p.old.BASE_CONTEXT/(qid+'.json'); consume(path)
        operations[qid] = p.old.old_json(path)['operation']
        p.materializer.context_policy_for_operation(operations[qid])
        core.vector_check(vectors[qid])
        path = previous.ROOT/'query_bank'/(qid+'.json'); consume(path, required=False)
        bank = read(path); question = questions[qid]['question']
        expected = {'registration_sha256': sha(ref), 'question_sha256': sha(question), 'q0_vector_sha256': sha(vectors[qid])}
        if bank['identity'] != expected or bank['question'] != question or bank['vectors'][question] != vectors[qid]:
            raise StopRun('Cached query-bank source identity drift')
        projection = core.qp.validate_response(question, bank['raw_projection'])
        if bank['projection'] != projection or bank['queries'] != core.query_texts(question, projection, 'q0_plus_anchored_attributes'):
            raise StopRun('Cached question-only projection replay drift')
        for vector in bank['vectors'].values(): core.vector_check(vector)
        banks[qid] = bank
    p.old.emit('SUMMARY3 INPUTS verified: 20 graphs, 1540 banks; no new memory/projection calls')
    baselines = {a: {} for a in ARMS}
    rerank = {a: {} for a in ARMS}
    for arm in ARMS:
        for qid, q in questions.items():
            path = previous.ROOT/'rows/previous_2_5_5'/arm/(qid+'.json'); consume(path, required=False)
            row = read(path)
            expected = {'registration_sha256': sha(ref), 'qa_sha256': sha(q),
                'graph_sha256': hashes[(samples[qid], arm)], 'condition': 'previous_2_5_5', 'arm': arm, 'vector_sha256': sha(vectors[qid])}
            if row['identity'] != expected or row['context_sha256'] != sha(row['context']) or row['bank_sha256'] != sha(banks[qid]):
                raise StopRun('Completed math reference row drift')
            baselines[arm][qid] = row
            path = previous.BASE/'conditions/narrow/rows'/arm/(qid+'.json'); consume(path)
            rerank[arm][qid] = read(path)
    if [sum(r['correct'] for r in baselines[a].values()) for a in ARMS] != [1074, 1132]:
        raise StopRun('Math reference score drift')
    if [sum(r['correct'] for r in rerank[a].values()) for a in ARMS] != [1178, 1217]:
        raise StopRun('Rerank reference score drift')
    own = [Path(__file__), Path(core.__file__), Path(__file__).with_name('protocol.md'), Path(__file__).with_name('test_core.py')]
    registration = {'schema': 'locomo.math.summary3_budget.v1', 'n': 1540, 'total_rows': 6160,
        'all_ids': sorted(questions), 'arms': list(ARMS), 'configs': CONFIGS, 'budget_policies': POLICIES,
        'source_hashes': {**ref['source_hashes'], **{str(x): file_sha(x) for x in own}},
        'source_snapshot_sha256': sha(frozen), 'prior_registration_sha256': sha(ref),
        'prior_completed_root': str(previous.ROOT), 'models': ALLOWED, 'max_final_facts': 5,
        'qa_prompt_sha256': sha(p.old.QA_PROMPT), 'judge_prompt_sha256': sha(p.old.JUDGE_PROMPT),
        'projection_bank_frozen': True, 'fresh_non_qa_judge_calls_permitted': 0,
        'historical_exact_response_reuse_roots': [str(x) for x in SOURCES], 'workers': WORKERS,
        'writer_and_event_graphs_frozen': True, 'judge_votes': 1, 'not_strict_token_matched': True,
        'reference_physical_audit': 'sealed completed lineage; not a new scan of all ancestor physical receipts'}
    seal(ROOT/'source_snapshot.json', frozen); seal(ROOT/'registration.json', registration)
    indexes = {(c,s,a): p.source.enum_repair.Index(graphs[(s,a)], **cfg)
        for c,cfg in CONFIGS.items() for s in sorted(set(samples.values())) for a in ARMS}
    data = dict(registration=registration, questions=questions, samples=samples, vectors=vectors, graphs=graphs,
        graph_hashes=hashes, indexes=indexes, operations=operations, baselines=baselines, rerank_baselines=rerank, banks=banks)
    seal(ROOT/'questions_score_free.json', {'score_free':True, 'rows':[{'qid':q,'question':questions[q]['question']} for q in sorted(questions)]})
    seal(ROOT/'preflight.json', {'pass':True, 'registration_sha256':sha(registration), 'n':1540, 'new_model_calls':0})
    # Feasibility only, not a causal attribution of correctness.
    diagnostic = {}
    for a in ARMS:
        records = []
        for qid in sorted(questions):
            idx = indexes[('summary3_5_5_cap', samples[qid], 'session_strong')]
            text,_ = core.plain_render(idx, rerank[a][qid]['selected_fact_ids'])
            limit = baselines[a][qid]['plain_budget']
            records.append({'qid':qid, 'old_rerank_ordinary_chars':len(text), 'dense5_cap':limit,
                'exceeds_cap':len(text)>limit, 'lost_vs_rerank':bool(rerank[a][qid]['correct'] and not baselines[a][qid]['correct'])})
        losses = [r for r in records if r['lost_vs_rerank']]
        diagnostic[a] = {'n':1540, 'old_set_exceeds_cap':sum(r['exceeds_cap'] for r in records),
            'losses':len(losses), 'losses_old_set_exceeds_cap':sum(r['exceeds_cap'] for r in losses), 'rows':records}
    seal(ROOT/'budget_feasibility.json', diagnostic)
    p.old.emit('SUMMARY3 PREFLIGHT passed; four fixed full1540 arms')
    return data


def query_bank(question, q0, qid, runtime, registration):
    path = previous.ROOT/'query_bank'/(qid+'.json'); bank = read(path)
    if bank['question'] != question or bank['vectors'][question] != q0:
        raise StopRun('Cached-bank question drift')
    seal(ROOT/'query_bank_refs'/(qid+'.json'), {'registration_sha256':sha(registration),
        'source':str(path), 'source_file_sha256':file_sha(path), 'bank_sha256':sha(bank), 'new_model_calls':0})
    return bank


def get_selection(data, c, a, qid, bank):
    path = ROOT/'selection'/c/a/(qid+'.json')
    binding = {'row_identity':runner.identity(data,c,a,qid), 'bank_sha256':sha(bank)}
    if path.exists():
        saved = read(path)
        if saved['identity'] != binding: raise StopRun('Selection resume drift')
        return saved
    s = data['samples'][qid]
    saved = {'identity':binding, **core.select(data['indexes'][(c,s,a)], data['indexes'][(c,s,'session_strong')],
        data['questions'][qid]['question'], bank, budget_policy=POLICIES[c])}
    seal(path,saved); return saved


def verify_bank(data, qid):
    marker = read(ROOT/'query_bank_refs'/(qid+'.json'))
    bank = previous.verify_bank(data,qid)
    path = previous.ROOT/'query_bank'/(qid+'.json')
    if marker != {'registration_sha256':sha(data['registration']), 'source':str(path),
            'source_file_sha256':file_sha(path), 'bank_sha256':sha(bank), 'new_model_calls':0}:
        raise StopRun('Bank provenance drift')
    return bank


def finish(data, ids, smoke=False):
    banks = {qid:verify_bank(data,qid) for qid in ids}
    rows = {c:{a:{} for a in ARMS} for c in CONFIGS}; stats = {}
    for c in CONFIGS:
        for a in ARMS:
            if not smoke and {f.stem for f in (ROOT/'rows'/c/a).glob('*.json')} != set(ids):
                raise StopRun('Incomplete final population')
            opts = []
            for qid in ids:
                rows[c][a][qid] = runner.verify_one(data,c,a,qid,banks[qid])
                saved = read(ROOT/'selection'/c/a/(qid+'.json'))
                if c == 'summary2_5_5_nocap':
                    index = data['indexes'][(c,data['samples'][qid],a)]
                    candidates,_ = index._candidate_pool(data['questions'][qid]['question'],data['vectors'][qid])
                    units,groups = core.adapt_candidates(candidates, data['graphs'][(data['samples'][qid],'session_strong')])
                    if saved['optimization']['char_budget'] != sum(u['cost'] for u in units)+sum(groups.values()):
                        raise StopRun('Nonbinding budget construction drift')
                opts.append(saved['optimization'])
            stats[c+'/'+a] = {'certified_optimal':sum(o['certified_optimal'] for o in opts),
                'selected_counts':dict(Counter(o['selected_count'] for o in opts)),
                'mean_seconds':sum(o['seconds'] for o in opts)/len(opts)}
            p.old.emit(f'SUMMARY3 AUDIT {c}/{a} {len(ids)}')
    for path,digest in data['registration']['source_hashes'].items():
        if file_sha(path) != digest: raise StopRun('Registered code drift '+path)
    if not smoke:
        for path,digest in read(ROOT/'source_snapshot.json').items():
            if file_sha(path) != digest: raise StopRun('Consumed source artifact drift '+path)
    physical = component.provider_audit(ROOT/'calls')
    if set(physical['stages']) - set(ALLOWED): raise StopRun('Forbidden new call stage')
    reports = {c:{a:component.summary(rows[c][a]) for a in ARMS} for c in CONFIGS}
    result = {'registration_sha256':sha(data['registration']), 'audit_pass':True, 'smoke_only':smoke, 'n':len(ids),
        'reports':reports, 'paired_online_vs_session':{c:p.paired(rows[c]['opctx_original'],rows[c]['session_strong'],ids) for c in CONFIGS},
        'paired_vs_previous_math_2_5_5_cap':{c:{a:p.paired(rows[c][a],data['baselines'][a],ids) for a in ARMS} for c in CONFIGS},
        'paired_vs_previous_gemini_rerank':{c:{a:p.paired(rows[c][a],data['rerank_baselines'][a],ids) for a in ARMS} for c in CONFIGS},
        'selection_stats':stats, 'physical_transport':physical, 'fresh_candidate_model_reranker_calls':0,
        'not_token_matched':True, 'query_bank_provider_receipts_audited_in_original_root':True}
    seal(ROOT/('smoke_complete.json' if smoke else 'complete.json'), result)
    p.old.emit('SUMMARY3 '+('SMOKE' if smoke else 'FULL')+' COMPLETE '+json.dumps(reports))
    return result


def configure():
    global runner
    name = __package__+'._frozen_runner'
    spec = importlib.util.spec_from_file_location(name, previous.__file__)
    runner = importlib.util.module_from_spec(spec); sys.modules[name] = runner
    spec.loader.exec_module(runner)
    for key,value in dict(ROOT=ROOT,CONFIGS=CONFIGS,ARMS=ARMS,SOURCES=SOURCES,WORKERS=WORKERS,
        # The legacy audit entry point registers the cached projection model.
        # New physical calls remain restricted by this module's Runtime/ALLOWED.
        ALLOWED=previous.ALLOWED,Runtime=Runtime,prepare=prepare,query_bank=query_bank,get_selection=get_selection,
        verify_bank=verify_bank,finish=finish).items():
        setattr(runner,key,value)
    return runner


if __name__ == '__main__': configure().main()

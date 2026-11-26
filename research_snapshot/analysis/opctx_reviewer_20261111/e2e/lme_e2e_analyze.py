"""Read-only postprocessor for LME E2E receipts; never sends API requests."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def unpack(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1)*p
    low, high = int(pos), min(int(pos)+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(pos-low)


def character_stats(values):
    if not values:
        return dict(n=0,min=None,mean=None,p50=None,p95=None,max=None)
    return dict(n=len(values),min=min(values),mean=statistics.mean(values),
        p50=quantile(values,.5),p95=quantile(values,.95),max=max(values))


def analyze(root, embedding_sample=None):
    manifest = read(root/'manifest.json')
    preflight = read(root/'preflight.json')
    summaries = sorted(root.glob('summary_*.json'))
    summary = read(summaries[-1]) if summaries else None
    rows = [read(p) for p in (root/'results/questions').glob('*/result.json')]
    completed = [r for r in rows if r.get('status')=='complete']
    stages = defaultdict(Counter)
    status = Counter()
    models = Counter()
    embedding = {}
    embedding_chars = defaultdict(list)
    query_rows = {}
    unknown_cost_calls = Counter()
    orphan_paths = []
    non_stop = []
    for intent_path in sorted((root/'physical_calls').glob('*/*/*/intent.json')):
        i = read(intent_path)
        path = intent_path.with_name('outcome.json')
        if not path.exists():
            orphan_paths.append(str(intent_path.parent))
            continue
        o = read(path)
        stage, kind, qid = i['stage'], i['kind'], i['question_id']
        s = stages[stage]
        s['physical_calls'] += 1
        s['provider_seconds'] += o.get('elapsed_seconds',0)
        status[o['status']] += 1
        if o.get('served_model'):
            models[o['served_model']] += 1
        if o.get('non_stop') or o.get('blocked') or o.get('model_mismatch'):
            non_stop.append(dict(question_id=qid,stage=stage,finish=o.get('finish_reasons'),path=str(path.parent)))
        u = o.get('usage')
        if kind == 'generation':
            if u is None:
                unknown_cost_calls[stage] += 1
            else:
                for key in ('input_tokens','output_tokens','thought_tokens','cached_input_tokens'):
                    s[key] += u.get(key,0)
        elif kind == 'rerank':
            if not isinstance(u,dict) or 'total_tokens' not in u:
                unknown_cost_calls[stage] += 1
            else:
                s['input_tokens_reported_total_tokens'] += u['total_tokens']
        elif kind == 'embedding':
            if u is None:
                unknown_cost_calls[stage] += 1
            req = unpack(path.with_name('request.json.gz'))
            for entry in req['requests']:
                content = entry['content']
                key = hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
                chars = sum(len(p.get('text','')) for p in content.get('parts',[]))
                record = embedding.setdefault(key,dict(characters=chars,multiplicity=0,stages=Counter(),question_ids=set()))
                record['multiplicity'] += 1
                record['stages'][stage] += 1
                record['question_ids'].add(qid)
                embedding_chars[stage].append(chars)
    generation_cost = {}
    for name in ('writer','online','planner','qa'):
        s = stages[name]
        inp,cache,out,thought = (s[k] for k in ('input_tokens','cached_input_tokens','output_tokens','thought_tokens'))
        discounted = ((inp-cache)*.30+cache*.03+(out+thought)*2.50)/1e6
        standard = (inp*.30+(out+thought)*2.50)/1e6
        generation_cost[name] = dict(**dict(s),known_usd_with_cache_discount=discounted,
            standard_usd_without_cache_discount=standard,unknown_usage_calls=unknown_cost_calls[name])
    full_complete = len(completed)==10 and not orphan_paths and not non_stop
    sample, full = preflight['totals'], preflight['full500_clean_b3_counts']
    scaling = dict(writer=full['writer_batches']/sample['writer_batches'],
        online=full['expected_online_calls']/sample['expected_online_calls'],planner=50,qa=50)
    extrapolated = {stage:dict(scale_factor=scaling[stage],
        estimated_usd_with_cache_discount=cost['known_usd_with_cache_discount']*scaling[stage],
        estimated_standard_usd_without_cache_discount=cost['standard_usd_without_cache_discount']*scaling[stage])
        for stage,cost in generation_cost.items()} if full_complete else None
    embedding_stats = dict(unique_contents=len(embedding),
        physical_content_items=sum(v['multiplicity'] for v in embedding.values()),
        unique_character_distribution=character_stats([v['characters'] for v in embedding.values()]),
        physical_character_distribution=character_stats([c for values in embedding_chars.values() for c in values]),
        by_stage={s:dict(physical_items=len(values),distinct_contents=sum(s in v['stages'] for v in embedding.values()),
                        character_distribution=character_stats(values)) for s,values in embedding_chars.items()},
        native_billed_tokens=None,native_billed_cost=None,
        note='No countTokens calls made. Character counts are not tokens or billed usage.')
    qwen_tokens = stages['qwen_rerank']['input_tokens_reported_total_tokens']
    total_discounted = sum(x['known_usd_with_cache_discount'] for x in generation_cost.values())
    total_standard = sum(x['standard_usd_without_cache_discount'] for x in generation_cost.values())
    counts = {k:sample[k] for k in ('physical_sessions','nonempty_sessions','writer_batches','expected_online_calls')}
    denominators = counts if full_complete else None
    total_generation = dict(known_usd_with_cache_discount=total_discounted,
        standard_usd_without_cache_discount=total_standard,
        unknown_usage_calls=sum(unknown_cost_calls[s] for s in ('writer','online','planner','qa')))
    if denominators:
        total_generation['amortized_per_question_usd'] = total_discounted/10
        total_generation['amortized_per_physical_session_usd'] = total_discounted/denominators['physical_sessions']
        total_generation['amortized_per_nonempty_session_usd'] = total_discounted/denominators['nonempty_sessions']
        total_generation['amortized_per_writer_batch_usd'] = total_discounted/denominators['writer_batches']
        memory_only = generation_cost['writer']['known_usd_with_cache_discount']+generation_cost['online']['known_usd_with_cache_discount']
        total_generation['memory_generation_per_writer_batch_usd'] = memory_only/denominators['writer_batches']
    wall_keys = ('writer_wall','frontier_graph_build_audit_wall','online_build_and_audit_wall',
                 'query_embed_and_rerank_wall','planner_and_opctx_wall','qa_wall')
    wall_scale = dict(writer_wall=scaling['writer'],frontier_graph_build_audit_wall=scaling['writer'],
        online_build_and_audit_wall=scaling['online'],query_embed_and_rerank_wall=50,
        planner_and_opctx_wall=50,qa_wall=50)
    wall_table = {}
    for key in wall_keys:
        total = sum(r['timings_seconds'].get(key,0) for r in completed)
        wall_table[key] = dict(sum_measured_question_seconds=total,
            mean_seconds_per_question=total/10 if full_complete else None,
            estimated_500_question_seconds=total*wall_scale[key] if full_complete else None,
            scaling_factor=wall_scale[key])
    throughput_500 = sum(v['estimated_500_question_seconds'] for v in wall_table.values())/8 if full_complete else None
    sampled_embedding_cost = None
    if embedding_sample:
        sample_result = read(embedding_sample)
        if sample_result['status'] != 'complete' or sample_result['census']['unique_contents'] != len(embedding):
            raise RuntimeError('Embedding diagnostic incomplete or population mismatch')
        estimates = sample_result['successful_input_estimate']
        memory = estimates['memory_embedding']['estimated_input_tokens']
        query = estimates['query_embedding']['estimated_input_tokens']
        embedding10 = (memory+query)*.15/1e6
        embedding500 = (memory*scaling['writer']+query*50)*.15/1e6 if full_complete else None
        low,high = estimates['ALL']['sampling_interval95']
        sampled_embedding_cost = dict(source=str(embedding_sample),
            diagnostic_requests=sample_result['diagnostics_attempted'],
            diagnostic_wall_seconds=sample_result['diagnostic_wall_seconds'],
            successful_input_token_estimates=estimates,price_usd_per_million=.15,
            estimated_10_usd=embedding10,token_sampling_interval95_cost_usd=[low*.15/1e6,high*.15/1e6],
            estimated_500_usd=embedding500,memory_scaling=scaling['writer'],query_scaling=50,
            generation_plus_embedding_estimate_10_usd=total_discounted+embedding10,
            generation_plus_embedding_estimate_500_usd=(sum(v['estimated_usd_with_cache_discount'] for v in extrapolated.values())+embedding500) if full_complete else None,
            native_billed_usage=None,
            note='Independent stratified native countTokens sample, not embedding billed usage. Memory Writer/Online embeddings not separately identified; aggregate memory embedding scaled by Writer-batch ratio as an approximation. Query items are all ten. Diagnostic calls/wall excluded from pipeline.')
    return dict(status='complete' if full_complete else 'incomplete',completed=len(completed),planned=10,
        generation_stages=generation_cost,generation_total=total_generation,
        qwen=dict(endpoint_region='Beijing',price_cny_per_million_input=.5,
            reported_input_tokens=qwen_tokens,known_cny=qwen_tokens*.5/1e6,
            estimated_500_cny=qwen_tokens*.5/1e6*50 if full_complete else None,
            usage_missing_calls=unknown_cost_calls['qwen_rerank']),
        embedding_inventory=embedding_stats,full500_workload_scaled_generation_estimate=extrapolated,
        independent_embedding_sampling_cost=sampled_embedding_cost,
        full500_estimate_note='Uniform final clean-B3 workload estimate, not historical500 rerun. Writer scaled by8533/179 and Online by8033/169; query stages by500/10. Embedding unknown, not zero.',
        workload_sample=sample,workload_full500=full,model_responses=dict(models),physical_statuses=dict(status),
        all_stage_call_counts={name:count['physical_calls'] for name,count in stages.items()},
        all_stage_provider_seconds={name:count['provider_seconds'] for name,count in stages.items()},
        all_stage_usage_missing_calls=dict(unknown_cost_calls),
        orphan_paths=orphan_paths,non_stop_or_blocked=non_stop,
        current_source_hashes_match=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['source_hashes'].items()),
        true_launch_wall_seconds=summary.get('this_launch_wall_seconds') if summary else None,
        final_run_summary_path=str(summaries[-1]) if summaries else None,
        timing_stage_summary=wall_table,
        estimated_500_idealized_parallel8_seconds=throughput_500,
        throughput_estimate_note='Sum of workload-scaled per-question stage wall times /8. Not measured; ignores queueing, startup gate and tail imbalance, API rate limits and changing contention. Ten-question timings themselves include concurrent execution conditions.',
        stage_wall_seconds_by_question={r['question_id']:r['timings_seconds'] for r in completed},
        reader_materializer_seconds={p.stem:read(p).get('reader_materializer_seconds') for p in (root/'question_cost').glob('*.json')},
        pricing=dict(date='2026-11-11',gemini_input=.30,gemini_cached_input=.03,gemini_output_including_thought=2.50,
            gemini_unit='USD/million tokens',qwen_unit='CNY/million input tokens',qwen_rate=.5),
        scope='Cold complete inference cost without evaluation Judge. Monetary known-generation subtotal excludes unknown embedding cost. Qwen reported separately in CNY, not silently converted to USD.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('D:/opctx_e2e_cost_20261111/lme10_run_r1'))
    p.add_argument('--output-json',type=Path)
    p.add_argument('--embedding-sample',type=Path)
    args=p.parse_args()
    result=analyze(args.input_root.resolve(),args.embedding_sample)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        with args.output_json.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

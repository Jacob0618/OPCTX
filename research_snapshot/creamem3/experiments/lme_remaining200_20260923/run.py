"""Bind new question manifests to the unchanged September 5 clean-B3 runner.

This module owns only selection, paths, process scheduling and result reporting.
Writer/Online/embedding/retrieval/Planner/QA/Judge are the original functions.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from filelock import FileLock
from creamem3 import run_lme_clean100_v1 as clean

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / 'creamem3/audit/lme_remaining200_20260923'
OLD = [ROOT / 'creamem3' / x for x in (
    'results_lme_opctx100_v2_b3_d768_20260830_r1',
    'results_lme_opctx100_b3_scaledfacts_d768_q6w1_20260902_r1',
    'results_lme_clean100_b3_scaled_d768_20260905_r1')]
SEED = '20260923_remaining200'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return clean.core._sha_file(Path(p))


def save(p, v):
    p = Path(p)
    if p.exists():
        if read(p) != v:
            raise RuntimeError('Refusing to replace existing artifact: ' + str(p))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x', encoding='utf8') as f:
            json.dump(v, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())


def rank(label, qid):
    return hashlib.sha256((SEED + '\0' + label + '\0' + qid).encode()).hexdigest()


def split_remaining(data, excluded):
    ids = {q['question_id'] for q in data}
    if len(data) != 500 or len(ids) != 500 or len(excluded) != 300 or not excluded <= ids:
        raise RuntimeError('Expected source500 minus distinct previous300')
    strata = defaultdict(list)
    for i, q in enumerate(data):
        if q['question_id'] not in excluded:
            strata[(q['question_type'], q['question_id'].endswith('_abs'))].append((i, q))
    batches = [[], []]
    for key, group in sorted(strata.items()):
        group.sort(key=lambda p: (rank('split', p[1]['question_id']), p[1]['question_id']))
        first = 0 if len(batches[0]) <= len(batches[1]) else 1
        for j, pair in enumerate(group):
            batches[(first + j) % 2].append(pair)
    for batch in batches:
        batch.sort(key=lambda p: (rank('order', p[1]['question_id']), p[1]['question_id']))
    if [len(b) for b in batches] != [100, 100]:
        raise RuntimeError('Partition must be two100 batches')
    return batches


def derive():
    data, _, _ = clean.selection.load_inputs()
    excluded = set()
    sources = {str(clean.selection.DEFAULT_DATASET): sha(clean.selection.DEFAULT_DATASET)}
    for root in OLD:
        p = root / 'selection.json'
        items = read(p)['selection']['items']
        ids = {r['question_id'] for r in items}
        if len(ids) != 100 or excluded & ids:
            raise RuntimeError('Previous selections overlap')
        for item in items:
            q = data[item['source_index']]
            if q['question_id'] != item['question_id'] or clean.core._canonical(q) != item['record_sha256']:
                raise RuntimeError('Previous record binding changed')
        excluded |= ids
        sources[str(p)] = sha(p)
    manifests = []
    for part, batch in enumerate(split_remaining(data, excluded), 4):
        items = [dict(position=i, question_id=q['question_id'], source_index=index,
                      record_sha256=clean.core._canonical(q)) for i, (index, q) in enumerate(batch)]
        manifests.append(dict(schema_version='lme.remaining200.two_fixed_batches.v1',
            part=part, count=100, seed=SEED, ids=[i['question_id'] for i in items], items=items,
            source_sha256=sources[str(clean.selection.DEFAULT_DATASET)],
            excluded_ids_sha256=clean.core._canonical(sorted(excluded)),
            type_counts=dict(Counter(q['question_type'] for _, q in batch)),
            abstentions=sum(q['question_id'].endswith('_abs') for _, q in batch),
            selection_uses=['question_id', 'question_type', 'question_id_abstention_suffix'],
            prior_outcomes_used=False))
    return data, manifests, sources


def bind(part):
    # Original configure and its frozen-core checks execute first, unchanged.
    clean.configure('b3_scaled', freeze=False)
    data, manifests, sources = derive()
    manifest = manifests[part - 4]
    path = AUDIT / f'part{part}_manifest.json'
    save(path, manifest)
    selected = [data[i['source_index']] for i in manifest['items']]
    def inputs(contract):
        if contract.file_sha256['lme_dataset'] != manifest['source_sha256']:
            raise RuntimeError('Dataset/prompt contract mismatch')
        return data, manifest, selected
    original_config = clean._run_config
    def config(contract):
        value = original_config(contract)
        value['manifest_sha256'] = sha(path)
        value['campaign'] = dict(seed=SEED, questions=100, part=part, manifest=str(path),
            excluded='three preserved nonoverlapping100 batches',
            interpretation='remaining200 from previously evaluated500; not an unseen dataset',
            optimization_during_run=False, inference_entry='unchanged run_lme_clean100_v1 + frozen core',
            orchestration_sha256=sha(__file__))
        return value
    clean._load_inputs = inputs
    clean._run_config = config
    clean.core.MANIFEST = clean.harness.MANIFEST = path
    clean.core.MANIFEST_SHA256 = clean.harness.MANIFEST_SHA256 = sha(path)
    clean.core._run_config = clean.harness._run_config = config
    clean.harness._load_inputs = inputs
    clean.harness.run_preflight = clean.run_preflight
    # Verify all inference knobs and implementation hashes against the actual old run.
    previous = read(OLD[-1] / 'summary.json')['config']
    current = config(clean.adapter.assert_prompt_contract())
    for k in set(previous) | set(current):
        if k not in {'manifest_sha256', 'campaign'} and previous.get(k) != current.get(k):
            raise RuntimeError('Non-selection configuration differs from clean B3: ' + k)
    return current, sources


def output(part):
    return ROOT / 'creamem3' / f'results_lme_remaining100_part{part}_clean_b3_20260923_r1'


def worker(args):
    config, sources = bind(args.part)
    preflight = AUDIT / f'part{args.part}_preflight.json'
    # All eight total question slots are shared as four per simultaneous batch.
    fixed = argparse.Namespace(allow_live=args.allow_live, resume=args.resume, limit=100,
        question_workers=4, writer_workers=1, output=output(args.part),
        cache_root=args.cache_base / f'part{args.part}', preflight_output=preflight)
    class Parser:
        def parse_args(self, argv=None):
            return fixed
    clean.harness._parser = Parser
    if args.allow_live:
        if shutil.disk_usage(args.cache_base.anchor).free < 25 * 1024**3:
            raise RuntimeError('Cache volume below25 GiB before launch')
        original_safe = clean.core._safe_process
        def disk_checked(**kwargs):
            if shutil.disk_usage(args.cache_base.anchor).free < 15 * 1024**3:
                raise RuntimeError('Cache volume below15 GiB; no new question calls')
            return original_safe(**kwargs)
        clean.core._safe_process = disk_checked
    result = clean.harness.main([])
    if not args.allow_live:
        save(AUDIT / f'part{args.part}_reuse_check.json', dict(
            all_nonselection_config_equal=True, previous_run=str(OLD[-1]),
            source_files=sources, inference_source_hashes=config['implementation_source_sha256'],
            new_code_scope='selection, paths, disk guard, scheduling, reporting only'))
    return result


def merge():
    roots = OLD + [output(4), output(5)]
    rows, records = [], []
    for root in roots:
        s = read(root / 'summary.json')
        if s['rows'] != 100 or sha(root / 'rows.jsonl') != s['rows_sha256']:
            raise RuntimeError('Incomplete or changed100-row result: ' + str(root))
        rr = [json.loads(x) for x in (root / 'rows.jsonl').read_text(encoding='utf8').splitlines()]
        rows.extend(rr)
        records.append(dict(root=str(root), correct=s['correct'], complete=s['complete'], errors=s['errors'],
                            summary_file_sha256=sha(root / 'summary.json')))
    if len(rows) != 500 or len({r['question_id'] for r in rows}) != 500:
        raise RuntimeError('Merged result is not500 unique questions')
    groups = defaultdict(list)
    for row in rows:
        groups[row['question_type']].append(row)
    result = dict(n=500, correct=sum(int(r.get('correct', 0)) for r in rows),
        complete=sum(r['status'] == 'complete' for r in rows), parts=records,
        interpretation='historical five-batch aggregate; first two writer policies differ from clean B3',
        by_type={k:dict(n=len(v), correct=sum(int(r.get('correct',0)) for r in v)) for k,v in sorted(groups.items())})
    result['accuracy'] = result['correct'] / 500
    result['errors'] = 500 - result['complete']
    save(AUDIT / 'merged500_summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def campaign(args):
    with FileLock(str(AUDIT / 'campaign.lock'), timeout=0):
        if (AUDIT / 'launch.json').exists():
            raise RuntimeError('Campaign already launched; inspect before any continuation')
        if shutil.disk_usage(args.cache_base.anchor).free < 25 * 1024**3:
            raise RuntimeError('Cache volume below25 GiB')
        jobs = []
        for part in (4,5):
            reuse = read(AUDIT / f'part{part}_reuse_check.json')
            for p,h in reuse['source_files'].items():
                if sha(p) != h:
                    raise RuntimeError('Source drift before launch')
            command = [sys.executable, '-X','utf8','-B','-u','-m',__package__+'.run','worker',
                       '--part',str(part),'--cache-base',str(args.cache_base),'--allow-live']
            stdout = AUDIT / f'part{part}.stdout.log'
            stderr = AUDIT / f'part{part}.stderr.log'
            with stdout.open('x',encoding='utf8') as out, stderr.open('x',encoding='utf8') as err:
                p = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            jobs.append((p,dict(part=part,pid=p.pid,command=command,stdout=str(stdout),stderr=str(stderr))))
        save(AUDIT / 'launch.json',dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            jobs=[r for _,r in jobs],question_workers_total=8,cache_base=str(args.cache_base)))
        codes = {r['part']:p.wait() for p,r in jobs}
        save(AUDIT / 'exit.json',dict(exit_codes=codes))
        if all((output(i)/'summary.json').exists() for i in (4,5)):
            merge()
        else:
            raise RuntimeError('A batch stopped before summary; inspect logs, do not invent missing results')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('worker','campaign','merge'))
    p.add_argument('--part',type=int,choices=(4,5))
    p.add_argument('--cache-base',type=Path,required=True)
    p.add_argument('--allow-live',action='store_true')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.phase == 'worker':
        if args.part is None:p.error('worker requires --part')
        raise SystemExit(worker(args))
    if args.phase == 'campaign':
        if not args.allow_live:p.error('campaign requires --allow-live')
        campaign(args)
    else:merge()

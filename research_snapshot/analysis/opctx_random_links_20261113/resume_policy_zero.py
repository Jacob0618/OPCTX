"""User-approved recovery: only exhausted original reranker validation counts zero.

Uses the registered runner body with explicit, recorded control-flow amendments.
No model, prompt, retrieval, or score-dependent answer selection is changed.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from filelock import FileLock
import transport as wire


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists() or not (parent/'run_summary.json').is_file():
        raise RuntimeError('Require settled parent and a new recovery output')
    spec = importlib.util.spec_from_file_location('_random_link_registered_runner', HERE/'run.py')
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    old_registration = base.read(parent/'registration.json')
    for name, expected in old_registration['source_hashes'].items():
        if base.file_sha(name) != expected:
            raise RuntimeError('Registered source changed: '+name)
    prior_rows = {}
    for path in (parent/'rows').glob('*/*.json'):
        row = base.read(path)
        if row['status'] == 'ok':
            prior_rows[(row['arm'],row['qa_id'])] = (str(path), row)
    assert len(prior_rows) == 91
    source = (HERE/'run.py').read_text(encoding='utf-8')
    edits = []

    def replace_once(old,new,label):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Cannot apply uniquely: '+label)
        source = source.replace(old,new)
        edits.append(label)

    replace_once("            if rr.get('fallback'):\n                raise RuntimeError('Original reranker exhausted validation; stop rather than silently accept fallback')",
        """            if rr.get('fallback'):
                assert rr['attempts'] == 2
                trace['rerank'] = dict(rr, selected_fact_ids=[])
                result.update(status='policy_zero', correct=0, prediction='', judge_raw=None,
                    policy_zero_reason='rerank_invalid_after_two_attempts', rerank_failure=rr,
                    retrieval_trace=trace, selected_fact_ids=[],
                    judge_skipped=True, qa_skipped=True,
                    policy_authority='User 2026-11-13: count these failures as wrong and report counts per arm')
                return result""", 'Exhausted original reranker validation -> explicit policy zero; no fallback QA/Judge')
    replace_once("        result = dict(**question, arm=arm, status='started', timings={})\n        if transport.stop.is_set():",
        """        result = dict(**question, arm=arm, status='started', timings={})
        previous = _parent_rows.get((arm,qid))
        if previous:
            source_path, prior_row = previous
            result = _copy.deepcopy(prior_row)
            result['reused_completed_from'] = source_path
            save(output/'rows'/arm/(qid+'.json'), result)
            with mutex:
                statuses.append((arm,qid,result['status']))
                print(json.dumps(dict(completed=len(statuses),arm=arm,qa_id=qid,status='reused_success')),flush=True)
            return result
        if transport.stop.is_set():""", 'Reuse all91 successful answers and judgments without generation')
    replace_once("len(statuses)==564 and all(s[2]=='ok' for s in statuses)",
        "len(statuses)==564 and all(s[2] in ('ok','policy_zero') for s in statuses)",
        'Completion includes explicitly authorized zeros, never other missing results')
    original_transport = wire.AuditTransport

    def recovery_transport(destination):
        t = original_transport(parent)
        t.output = Path(destination).resolve()
        return t

    wire.AuditTransport = recovery_transport
    namespace = dict(base.__dict__)
    namespace.update(_parent_rows=prior_rows, _copy=copy)
    exec(compile(source, str(HERE/'run.py'), 'exec'), namespace)
    raw_save = namespace['save']

    def recovery_save(path,value):
        path = Path(path)
        if path.name == 'registration.json':
            value = copy.deepcopy(value)
            value.update(parent_outputs=[str(parent)],
                parent_registration_sha256=base.file_sha(parent/'registration.json'),
                recovery_code_sha256=base.file_sha(__file__),
                effective_runner_sha256=hashlib.sha256(source.encode()).hexdigest(),
                recovery_edits=edits, reused_successful_rows=91,
                rerank_max_attempts=2, judge_format_attempts=5,
                policy_zero_reason='rerank_invalid_after_two_attempts',
                policy_zero_scope='Both arms, original reranker semantic validation exhausted; no API/safety/network failure zeros',
                user_authorization='2026-11-13: 判错，并报告每组多少这样的情况')
            raw_save(path,value)
            raw_save(path.with_name('recovery_amendment.json'),dict(
                created_utc=datetime.now(timezone.utc).isoformat(), edits=edits,
                parent=str(parent), preserve_parent=True, preserve_all_successes=True,
                bounded_semantic_attempts_unchanged=True, fallback_not_used=True,
                user_authorization=value['user_authorization']))
        else:
            raw_save(path,value)

    namespace['save'] = recovery_save
    with FileLock(str(parent/'dispatcher.lock'),timeout=0):
        namespace['run'](output,args.workers)


if __name__ == '__main__':
    main()

"""Offline explicit-user settlement of the 14 remaining LME QA truncations.

Writes only a new policy overlay. No model imports, API calls, or source edits.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = Path('D:/hybrid_rag_20261114/lme_judge_finish_r6')
SUMMARY = SOURCE / 'recovery_summaries/1794613354737159400.json'
OUTPUT = Path('D:/hybrid_rag_20261114/lme_final_20261114_r1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def metric(rows):
    n, correct = len(rows), sum(r['correct'] for r in rows)
    return dict(n=n, correct=correct, accuracy_percent=100*correct/n)


def main():
    if (OUTPUT / 'summary.json').exists():
        print(json.dumps(read(OUTPUT / 'summary.json'), indent=2))
        return
    config, previous = read(SOURCE / 'config.json'), read(SUMMARY)
    raw = read(WORKSPACE / 'data/longmemeval/longmemeval_s.json')
    qmap = {q['question_id']: q for q in raw}
    assert len(qmap) == 500 and set(qmap) == set(config['question_ids'])
    assert previous['valid_judged_n'] == 483 and previous['policy_zero_count'] == 3
    missing = set(previous['known_missing_ids'])
    assert len(missing) == 14 and not previous['unscored_remaining']
    for path, expected in config['source_sha256'].items():
        assert sha(path) == expected, f'Frozen source mismatch: {path}'
    rows, decisions = [], []
    for qid in sorted(qmap):
        folder = SOURCE / 'questions' / qid
        path = folder / ('missing.json' if qid in missing else 'result.json')
        item = read(path)
        assert item['question_id'] == qid
        row = dict(question_id=qid, category_name=qmap[qid]['question_type'],
                   source=str(path), source_sha256=sha(path))
        if qid in missing:
            assert item['correct'] is None and item['model_judge_called'] is False
            assert not (folder / 'result.json').exists()
            evidence = item.get('evidence', {})
            call = Path(evidence.get('call') or item['original_failure'])
            outcome = read(call / 'outcome.json')
            response = read(call / 'response.json')
            expected_sha = evidence.get('response_sha256') or item['original_response_sha256']
            assert outcome['response_sha256'] == expected_sha
            assert outcome['stage'] == 'qa' and outcome['http_status'] == 200
            assert outcome['finish_reason'] == 'MAX_TOKENS'
            assert response['candidates'][0]['finishReason'] == 'MAX_TOKENS'
            row.update(correct=0, prediction=None, judge=None,
                       model_judge_called=False,
                       score_origin='user_authorized_remaining_truncations_zero',
                       native_response=str(call / 'response.json'),
                       native_response_file_sha256=sha(call / 'response.json'))
            decisions.append(row)
        else:
            assert item['correct'] in (0, 1, False, True)
            policy_zero = item.get('status') == 'policy_zero'
            if policy_zero:
                assert qid in previous['policy_zero_ids']
                assert item['correct'] == 0 and item['model_judge_called'] is False
            else:
                assert item['judge'] is not None and item['prediction'] is not None
            row.update(correct=int(item['correct']), prediction=item['prediction'],
                       model_judge_called=not policy_zero,
                       score_origin='prior_authorized_zero' if policy_zero else 'actual_judge')
        rows.append(row)
    counts = Counter(r['score_origin'] for r in rows)
    assert counts == {'actual_judge': 483, 'prior_authorized_zero': 3,
                      'user_authorized_remaining_truncations_zero': 14}
    assert len(rows) == len({r['question_id'] for r in rows}) == 500
    assert sum(r['correct'] for r in rows) == 357
    groups = defaultdict(list)
    for row in rows:
        groups[row['category_name']].append(row)
    policy = dict(authorization='User: 都计0然后算一下locomo和lme的开销，以及检索量是多少',
                  scope='Exactly the 14 remaining QA-truncation missing cases in r6',
                  question_ids=sorted(missing), correct=0, judge_called=False,
                  new_api_calls=0, source_summary=str(SUMMARY),
                  source_summary_sha256=sha(SUMMARY),
                  source_config_sha256=sha(SOURCE / 'config.json'),
                  created_utc=datetime.now(timezone.utc).isoformat())
    result = dict(dataset='lme', complete=True, all_questions_accounted=True,
                  overall=metric(rows), categories={k: metric(v) for k,v in groups.items()},
                  actual_judge_count=483, policy_zero_count=17,
                  newly_authorized_zero_count=14, remaining_missing_count=0,
                  new_api_calls=0, original_artifacts_unchanged=True,
                  source_summary=str(SUMMARY), policy_file=str(OUTPUT / 'policy.json'))
    save(OUTPUT / 'policy.json', policy)
    for row in decisions:
        save(OUTPUT / 'new_decisions' / (row['question_id'] + '.json'), row)
    save(OUTPUT / 'settled_rows.json', rows)
    save(OUTPUT / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

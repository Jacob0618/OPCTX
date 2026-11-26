"""Repair four cached verdicts and issue only two previously unissued Judges."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re

from dotenv import load_dotenv
from filelock import FileLock
from ..memeye_lite_ablation_20260923 import run as base
from ..memeye_lite_ablation_limits_20260923 import run as limits

b, rt = base.b, base.rt
HERE = Path(__file__).resolve().parent
REC = base.ROOT / 'non_v8_score_repair_20260923_r1'
ORIGINAL_SAVE = b.save
ARMS = base.ARMS[:4]
# A null key means never issued, not a failed request eligible for retry.
TARGETS = (
    ('original_reader', 'g35lite', 'Social_Chat_Memory_Test::Q8',
     'e4d77ccbcb6f090104761897471def32162217fd3e3dcbb8cad50346c2a27a2f'),
    ('no_reader', 'g35flash', 'Outdoor_Navigation_Route_Memory_Assistant::Q12',
     'e69e87a2bccc93e4dfbee03b835e27663e0ea9e68364235ece492353eee7d8b6'),
    ('no_reader', 'g35lite', 'Outdoor_Navigation_Route_Memory_Assistant::Q12', None),
    ('no_reader', 'g35lite', 'Personal_Health_Dashboard_Assistant::Q28',
     '422084032f72e316ba9bb3564bfd4935d3ffebb424b3456711b9d07d0efd4919'),
    ('enhanced_reader', 'g35flash', 'Personal_Health_Dashboard_Assistant::Q34',
     '56868a4eb7bd3872c7b97f7a74e9095696f99d7d0203a65eaa42b8ded7e2556e'),
    ('enhanced_reader', 'g35lite', 'Personal_Health_Dashboard_Assistant::Q34', None),
)


def source_hashes():
    return {str(p): b.file_sha(p) for p in sorted([*HERE.glob('*.py'), HERE/'protocol.md'])}


def parse_verdict(text, allowed_text_hashes=()):
    """Whitelist-only punctuation repair; no extraction of scores from prose."""
    try:
        verdict = rt.prior.n.g.parse_mgtm_judge_response(text)
        return verdict, None
    except rt.prior.n.g.RuntimeProtocolError:
        if b.sha(text) not in allowed_text_hashes:
            raise
    match = re.fullmatch(
        r'\s*\{\s*"score"\s*:\s*(0(?:\.0|\.25|\.5|\.75)?|1(?:\.0)?)'
        r'\s*,\s*"reasoning"\s*:\s*"(.*)"\s*\}\s*', text, re.DOTALL)
    if not match:
        raise ValueError('Not the audited score-then-reasoning shape')
    reasoning = match.group(2)
    if (any(c in reasoning for c in '\\{}\n\r\t') or not reasoning.count('"')
            or reasoning.count('"') % 2 or re.search(r'"\s*:', reasoning)):
        raise ValueError('Ambiguous reasoning grammar; refuse repair')
    start, end = match.span(2)
    repaired = text[:start] + reasoning.replace('"', '\\"') + text[end:]
    value = json.loads(repaired)
    base.require(set(value) == {'score', 'reasoning'}, 'Unexpected Judge fields')
    base.require(value['reasoning'] == reasoning, 'Reasoning changed')
    base.require(value['score'] == float(match.group(1)), 'Score changed')
    verdict = rt.prior.n.g.parse_mgtm_judge_response(repaired)
    return verdict, dict(kind='escape_unescaped_inner_quotes_only',
        original_text=text, repaired_json=repaired,
        inserted_escape_count=reasoning.count('"'), score_unchanged=True,
        reasoning_unchanged=True)


def save_rejection_idempotent(path, value, *, scope=base.ROOT):
    """Narrow replay adapter, preserving immutable semantic failure evidence."""
    path = Path(path).resolve()
    scoped = path.is_relative_to(Path(scope).resolve())
    if not (scoped and path.parent.name == 'rejected'
            and path.parent.parent.name == 'v8' and path.suffix == '.json'
            and path.exists()):
        return ORIGINAL_SAVE(path, value)
    existing = b.read(path)
    before = deepcopy(existing); before.pop('seal_sha256', None)
    after = deepcopy(value)
    if before == after:
        return existing
    for row in (before, after):
        if set(row) != {'identity', 'error', 'generation'}:
            return ORIGINAL_SAVE(path, value)
        if type(row['generation'].get('cache_hit')) is not bool:
            return ORIGINAL_SAVE(path, value)
    if not before['generation']['cache_hit'] and after['generation']['cache_hit']:
        before['generation'].pop('cache_hit'); after['generation'].pop('cache_hit')
        if before == after:
            return existing
    return ORIGINAL_SAVE(path, value)  # genuine content drift still raises


@contextmanager
def rejection_replay_adapter():
    """Install only while running a future registered continuation, then restore."""
    previous = b.save
    base.require(previous is ORIGINAL_SAVE, 'Unexpected save adapter already active')
    b.save = save_rejection_idempotent
    try:
        yield
    finally:
        b.save = previous


def spec(target):
    arm, judge, rid, cached_key = target
    task = rid.split('::')[0]
    root = base.task_root('g35lite', task)
    predpath = root/'predictions'/arm/(b.sha(rid)+'.json')
    pred = b.read(predpath)
    gold = b.read(base.SOURCE/'scoring/open.json')['rows'][rid]
    q = next(q for q in base.validate()['questions'] if q['row_id'] == rid)
    base.require(q['question'] == gold['question'], 'Question mismatch')
    base.require(pred['model'] == rt.MODELS['g35lite'], 'Prediction model mismatch')
    prompt = rt.prior.n.g.build_mgtm_judge_prompt(q['question'], gold['gold'], pred['prediction'])
    metadata = dict(arm=arm, row_id=rid, prediction_sha256=pred['seal_sha256'])
    model = rt.JUDGES[judge]
    wire, ref = rt.request(model, 'judge', prompt, None, 400, True)
    manifest = dict(model=model, stage='judge', prompt=prompt, system=None, image=ref,
        generation_config=wire['generationConfig'], wire_sha256=b.sha(wire),
        metadata=metadata, caption_context_count=None)
    key = b.sha(manifest)
    call = root/'judge_calls'/judge/key[:2]/key
    if cached_key:
        base.require(key == cached_key, 'Audited request changed')
    return dict(arm=arm, judge=judge, row_id=rid, task=task, root=root, pred=pred,
        predpath=predpath, gold=gold, question=q['question'], model=model,
        prompt=prompt, metadata=metadata, manifest=manifest, key=key, call=call,
        scorepath=root/'scores'/judge/arm/(b.sha(rid)+'.json'), cached=bool(cached_key))


def cached_reply(s):
    with gzip.open(s['call']/'request.json.gz', 'rt', encoding='utf8') as f:
        manifest = json.load(f)
    base.require(manifest == s['manifest'], 'Cached Judge request drift')
    result = b.read(s['call']/'response.json'); raw = result['response']
    base.require(result['request_sha256'] == s['key'], 'Request key mismatch')
    base.require(result['response_sha256'] == b.sha(raw), 'Provider response hash mismatch')
    cs = raw.get('candidates') or []
    base.require(len(cs) == 1 and cs[0].get('finishReason') == 'STOP', 'Non-STOP')
    served = raw.get('modelVersion', '')
    base.require(not served or served.startswith(s['model']), 'Served model mismatch')
    return rt.prior.Reply(rt.prior.n.g._normalize_text(raw).strip(), s['key'],
        result['response_sha256'], s['model'], served,
        asdict(rt.prior.n.g._normalize_usage(raw)), True)


def register():
    parent = limits.validate()
    protected = {}
    for pattern in ('*/full/tasks/*/predictions/*/*.json', '*/full/tasks/*/scores/*/*/*.json',
                    '*/full/tasks/*/contexts/*/*.json', '*/full/tasks/*/**/v8/rejected/*.json'):
        for p in base.ROOT.glob(pattern):
            protected[str(p)] = b.file_sha(p)
    targets = []
    for t in TARGETS:
        s = spec(t)
        base.require(not s['scorepath'].exists(), 'Target is not missing')
        item = dict(arm=s['arm'], judge=s['judge'], row_id=s['row_id'],
                    request_sha256=s['key'], cached=s['cached'], prediction_sha256=s['pred']['seal_sha256'])
        if s['cached']:
            out = cached_reply(s)
            item.update(text_sha256=b.sha(out.text), response_sha256=out.response_sha256)
            parse_verdict(out.text, {item['text_sha256']})
            for name in ('request.json.gz', 'response.json'):
                p = s['call']/name; protected[str(p)] = b.file_sha(p)
        else:
            base.require(not s['call'].exists(), 'Unissued request has existing artifacts')
        targets.append(item)
    return b.save(REC/'registration.json', dict(parent_sha256=parent['seal_sha256'],
        code_hashes=source_hashes(), protected=protected, targets=targets,
        new_logical_judge_requests=2, cached_score_restorations=4,
        qa_requests=0, v8_model_calls=0, scope='g35lite four non-v8 arms only'))


def validate():
    parent = limits.validate(); reg = b.read(REC/'registration.json')
    base.require(reg['parent_sha256'] == parent['seal_sha256'], 'Parent registration changed')
    base.require(reg['code_hashes'] == source_hashes(), 'Repair source changed')
    for p, h in reg['protected'].items():
        base.require(b.file_sha(p) == h, 'Protected artifact changed: '+p)
    return reg


def replay_check():
    root = base.task_root('g35lite', 'Social_Chat_Memory_Test')/'limits20_v8'
    checked = []
    for attempt in range(3):
        p = root/'v8/rejected'/f'24_{attempt}.json'
        h = b.file_sha(p); saved = b.read(p); row = deepcopy(saved); row.pop('seal_sha256')
        gen = row['generation']; key = gen['request_sha256']
        response = b.read(root/'calls'/key[:2]/key/'response.json')
        base.require(response['request_sha256'] == key, 'Replay request mismatch')
        base.require(response['response_sha256'] == b.sha(response['response']) == gen['response_sha256'],
                     'Replay response mismatch')
        base.require(rt.prior.n.g._normalize_text(response['response']).strip() == gen['text'],
                     'Replay text mismatch')
        gen['cache_hit'] = True
        with rejection_replay_adapter():
            actual = b.save(p, row)
        base.require(actual == saved and b.file_sha(p) == h, 'Replay altered evidence')
        checked.append(dict(path=str(p), sha256=h, error=row['error']))
    return checked


def repair(allow_calls=False):
    with FileLock(str(base.ROOT/'dispatcher.lock'), timeout=1):
        base.require(not limits.active_workers(), 'Active workers must finish first')
        reg = validate() if (REC/'registration.json').exists() else register()
        checks = replay_check()
        b.save(REC/'rejection_replay_verification.json', dict(checks=checks, model_calls=0))
        for t, registered in zip(TARGETS, reg['targets'], strict=True):
            s = spec(t)
            base.require(s['key'] == registered['request_sha256'], 'Target drift')
            receipt = REC/'receipts'/(b.sha([s['arm'], s['judge'], s['row_id']])+'.json')
            if s['scorepath'].exists():
                old = b.read(s['scorepath']); provenance = b.read(receipt)
                base.require(old['prediction_sha256'] == s['pred']['seal_sha256']
                             and old['generation']['response_sha256'] == provenance['response_sha256'],
                             'Completed repair drift')
                continue
            if s['cached']:
                out = cached_reply(s)
                base.require(out.response_sha256 == registered['response_sha256'], 'Response changed')
                verdict, punctuation = parse_verdict(out.text, {registered['text_sha256']})
            else:
                out = rt.Runtime(s['root']/'judge_calls'/s['judge'], s['model'], allow_calls).generate(
                    'judge', s['prompt'], json_mode=True, max_output_tokens=400,
                    cache_metadata=s['metadata'])
                verdict, punctuation = parse_verdict(out.text)
            # Save provenance first so any interrupted score write can resume safely.
            b.save(receipt, dict(arm=s['arm'], judge=s['judge'], row_id=s['row_id'],
                registration_sha256=reg['seal_sha256'], request_sha256=out.request_sha256,
                response_sha256=out.response_sha256, prediction_sha256=s['pred']['seal_sha256'],
                raw_score=verdict.raw_score, mapped_score=verdict.mapped_score,
                existing_response=s['cached'], punctuation_repair=punctuation,
                qa_rerun=False, original_score_mapping=True))
            gold = s['gold']
            b.save(s['scorepath'], dict(row_id=s['row_id'], task=s['task'], arm=s['arm'],
                x=gold['x'], y=gold['y'], xy=gold['x']+'-'+gold['y'],
                raw_score=verdict.raw_score, mapped_score=verdict.mapped_score,
                reasoning=verdict.reasoning, judge=s['model'], generation=asdict(out),
                prediction_sha256=s['pred']['seal_sha256']))
            print('COMPLETED', s['arm'], s['judge'], s['row_id'], verdict.raw_score, flush=True)
        validate()
        result = base.summarize('g35lite')
        ids = {q['row_id'] for q in base.validate()['questions']}
        for arm in ARMS:
            preds = [b.read(p) for p in (base.ROOT/'g35lite/full').glob('tasks/*/predictions/'+arm+'/*.json')]
            base.require(len(preds) == 371 and {p['row_id'] for p in preds} == ids, 'Prediction coverage')
            by_id = {p['row_id']:p for p in preds}
            for name in rt.JUDGES:
                base.require(result['judges'][name][arm]['n'] == 371, 'Incomplete scores')
                for p in (base.ROOT/'g35lite/full').glob('tasks/*/scores/'+name+'/'+arm+'/*.json'):
                    score = b.read(p)
                    base.require(score['prediction_sha256'] == by_id[score['row_id']]['seal_sha256'],
                                 'Score uses different answer')
        result['judges'] = {j:{a:s for a,s in arms.items() if a in ARMS} for j,arms in result['judges'].items()}
        result.update(repair_registration_sha256=reg['seal_sha256'], v8_excluded=True,
                      preserved_artifacts=len(reg['protected']), same_answers_dual_judge=True)
        b.save(REC/'results.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-two-unissued-judges', action='store_true')
    args = parser.parse_args()
    load_dotenv(b.REPO/'.env', override=False)
    repair(args.allow_two_unissued_judges)

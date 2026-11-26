"""One fresh QA/Judge repeat plus an independent old-answer Judge for four losses."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from creamem3.experiments.locomo_joint_revision_20260909 import followup_audit as parent
from creamem3.experiments.locomo_online_only_20260909.runtime import Runtime, read, seal, sha, file_sha, StopRun, now

old = parent.old
ROOT = Path(r"D:\creamem3_cache\locomo_online_semantics_20260909_r1")
RUN = ROOT / "four_losses_repeat"


class OnlyAnswers(Runtime):
    def post(self, stage, scope, model, method, body):
        if stage not in {"qa", "judge"}:
            self._halt("Four-loss repeat cannot retrieve, rerank, or rewrite")
        return super().post(stage, scope, model, method, body)


def prepare():
    audit = read(parent.ROOT / "analysis/literal_joint_no_rerank.json")
    if audit['status'] != 'pass' or audit['losses'] != 4 or audit['correct'] != {'opctx_original':76, 'session_strong':76}:
        raise StopRun("Wrong source experiment")
    online = {r['qa_id']:r for r in audit['rows']['opctx_original']}
    session = {r['qa_id']:r for r in audit['rows']['session_strong']}
    ids = sorted(q for q in online if online[q]['correct'] < session[q]['correct'])
    rows = {}
    for qid in ids:
        entry = online[qid]
        row = read(entry['source_path'])
        if sha(row) != entry['source_sha256'] or row['context_sha256'] != entry['context_sha256'] or row['selected_fact_ids'] != entry['selection']:
            raise StopRun("Source row drift")
        parent.check_bound(parent.CACHED, row, row)
        rows[qid] = row
    registration = {'source_sha256':file_sha(__file__), 'literal_audit_sha256':sha(audit),
        'source_rows':{q:{'path':online[q]['source_path'],'sha256':sha(r)} for q,r in rows.items()},
        'qa_prompt_sha256':sha(old.QA_PROMPT), 'judge_prompt_sha256':sha(old.JUDGE_PROMPT),
        'trials':['fresh_QA_then_single_Judge','fixed_old_answer_single_fresh_Judge'],
        'fresh_cache_roots_per_trial':True, 'no_old_response_import':True, 'same_exact_retrieval_context':True,
        'no_rerank_model':True, 'no_resampling_until_success':True, 'first_valid_judge_only_no_vote':True,
        'post_selected_four_errors_not_population_estimate':True, 'never_backfill_old100_score':True}
    seal(RUN/'registration.json', registration)
    return registration, rows


def main():
    with old.Lease(RUN):
        registration, parents = prepare()
        from dotenv import load_dotenv
        load_dotenv(old.ROOT/'.env', override=False)
        runtimes = {mode:OnlyAnswers(RUN/mode/'calls', allow_model_calls=True, concurrency=4, timeout=180)
                    for mode in ['fresh_QA_then_single_Judge','fixed_old_answer_single_fresh_Judge']}

        def one(mode, qid):
            root = RUN/mode
            runtime, source = runtimes[mode], parents[qid]
            identity = {'registration_sha256':sha(registration),'source_row_sha256':sha(source),'mode':mode}
            path = root/'rows'/(qid+'.json')
            if path.exists():
                row = read(path)
                if row['identity'] != identity: raise StopRun('Repeat row identity drift')
                return mode, row
            answer = runtime.chat(qid,'qa',old.QA_PROMPT.format(question=source['question'],retrieved_texts=source['context']),max_tokens=8192) if mode.startswith('fresh_QA') else source['response']
            correct, raw = old.judge(runtime, qid, source, answer)
            row = {**source,'identity':identity,'response':answer,'correct':correct,'judge_raw':raw,
                   'old_correct':source['correct'],'old_response':source['response'],'old_judge':source['judge_raw']}
            seal(path,row)
            old.emit('FOUR REPEAT '+mode+' '+qid+' correct='+str(correct))
            return mode,row

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                completed = [f.result() for f in as_completed([pool.submit(one,mode,q) for mode in runtimes for q in parents])]
            by_mode = {mode:{} for mode in runtimes}
            for mode,row in completed:
                qid = row['qa_id']
                source, root = parents[qid], RUN/mode
                if row['context'] != source['context'] or row['selected_fact_ids'] != source['selected_fact_ids']:
                    raise StopRun('Fixed context changed')
                if mode.startswith('fresh_QA'):
                    parent.check_bound(root,source,row)
                else:
                    # The old answer intentionally has its QA binding in the immutable source root.
                    if row['response'] != source['response']: raise StopRun('Old-answer Judge probe changed answer')
                    prompt = old.JUDGE_PROMPT.format(question=source['question'],answer=source['answer'],response=source['response'])
                    found = False
                    for budget in [2048,4096,8192]:
                        identity = parent.binding.generation_identity('judge',prompt,budget)
                        if not (root/'calls/provider/judge'/sha(identity)/'final.json').exists(): continue
                        raw = parent.binding.bound_text(root,identity)
                        tags = old.re.findall(r'\[\[(yes|no)\]\]',raw.lower())
                        if len(tags)==1:
                            if raw != row['judge_raw'] or row['correct'] != int(tags[0]=='yes'): raise StopRun('Rejudge binding')
                            found = True
                            break
                    if not found: raise StopRun('Missing fresh Judge')
                by_mode[mode][qid] = row
            report = {'registration_sha256':sha(registration),'n':4,'rows':{
                q:{'question':parents[q]['question'],'reference':parents[q]['answer'],
                   'old_answer':parents[q]['response'],'old_correct':parents[q]['correct'],
                   'fresh_answer':by_mode['fresh_QA_then_single_Judge'][q]['response'],
                   'fresh_correct':by_mode['fresh_QA_then_single_Judge'][q]['correct'],
                   'old_answer_rejudged_correct':by_mode['fixed_old_answer_single_fresh_Judge'][q]['correct'],
                   'exact_answer_unchanged':parents[q]['response']==by_mode['fresh_QA_then_single_Judge'][q]['response']}
                for q in parents},'counts':{mode:sum(r['correct'] for r in rows.values()) for mode,rows in by_mode.items()},
                'all_fresh_requests_no_import':True,'original100_unchanged':True,
                'selection_bias_warning':'Four chosen errors can regress toward the mean;do not call recovered errors a new100-question accuracy or proof that all prior loss was noise.'}
            seal(RUN/'complete.json',report)
            old.emit('FOUR REPEAT COMPLETE '+json.dumps(report,ensure_ascii=False))
        except BaseException as exc:
            for runtime in runtimes.values(): runtime.stopped.set()
            seal(RUN/('stop_'+str(time.time_ns())+'.json'),{'utc':now(),'type':type(exc).__name__,'message':str(exc)[:500]})
            raise


if __name__=='__main__':main()

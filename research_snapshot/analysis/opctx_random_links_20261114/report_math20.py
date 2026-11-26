"""Read-only report for the normal/random-link mathematical top-20 experiment.

No model calls are made. Only a fresh --output directory is written. The old
reranker-failure zero policy DOES NOT apply: every scored row needs valid QA and
Judge outputs. Selection diagnostics may be available before QA finishes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "analysis/opctx_random_links_20261113/report.py"
spec = importlib.util.spec_from_file_location("_math20_report_helpers", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ARMS, EXPECTED_N = ("normal", "random"), 282
read, sha, dist = helper._read, helper._sha, helper._dist


def bank_statistics(root: Path, questions: dict) -> dict:
    statuses, fallbacks, counts, violations, seen = Counter(), Counter(), [], {}, set()
    for path in sorted((root / "query_bank").glob("*.json")):
        qid, bank = path.stem, read(path)
        if qid not in questions or bank.get("question") != questions[qid]["question"]:
            raise ValueError(f"Projection bank question mismatch: {path}")
        seen.add(qid)
        projection, queries, vectors = bank.get("projection") or {}, bank.get("queries"), bank.get("vectors")
        statuses[str(projection.get("status", "unknown"))] += 1
        if projection.get("fallback_reason"):
            fallbacks[str(projection["fallback_reason"])] += 1
        errors = []
        if not isinstance(queries, list) or not queries:
            errors.append("missing_query_list")
        else:
            counts.append(len(queries))
            texts = [q.get("query") for q in queries]
            if texts[0] != questions[qid]["question"] or len(texts) != len(set(texts)):
                errors.append("original_query_or_unique_query_contract")
            for text in texts:
                vector = vectors.get(text) if isinstance(vectors, dict) else None
                if (not isinstance(vector, list) or len(vector) != 768
                        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)
                        or not any(vector)):
                    errors.append("missing_invalid_projection_vector")
                    break
        if errors:
            violations[qid] = errors
    return dict(saved_banks=len(seen), expected_banks=len(questions), missing_ids=sorted(set(questions)-seen),
        projection_status_counts=dict(statuses), fallback_reasons=dict(fallbacks),
        query_channels_per_question=dist(counts), violations=violations,
        note="A single question-only bank is shared by both arms. Explicit original-algorithm semantic "
             "fallbacks and valid questions without distinct attribute projections are retained, not silently "
             "replaced or semantically retried. Query count counts original/projected queries, not twice the "
             "number of lexical and dense utility channels.")


def usage(root: Path) -> dict:
    """Count actual physical outcomes, including bank_runtime and qa_runtime."""
    groups, seen = defaultdict(list), set()
    roots = helper._output_roots(root)
    for current in roots:
        for path in sorted(current.rglob("outcome.json")):
            if "calls" not in path.relative_to(current).parts:
                continue
            identity = str(path.resolve()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            item = read(path)
            groups[(str(item.get("stage", "unknown")), str(item.get("model", "unknown")))].append(item)
    output = []
    fields = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount",
              "cachedContentTokenCount", "totalTokenCount")
    for (stage, model), items in sorted(groups.items()):
        native = [x.get("usage") or x.get("usageMetadata") for x in items]
        native = [x for x in native if isinstance(x, dict)]
        totals = {key: sum(x[key] for x in native if isinstance(x.get(key), (int, float)))
                  if any(isinstance(x.get(key), (int, float)) for x in native) else None for key in fields}
        seconds = [x["service_seconds"] for x in items if isinstance(x.get("service_seconds"), (int, float))]
        output.append(dict(stage=stage, model=model, outcome_records=len(items),
            physical_calls=sum(int(x.get("physical_calls", bool(x.get("attempted", True)))) for x in items),
            status_counts=dict(Counter(str(x.get("status", "unknown")) for x in items)),
            native_token_fields=totals, outcomes_with_native_usage=len(native),
            outcomes_without_native_usage=len(items)-len(native),
            summed_service_seconds=sum(seconds) if seconds else None))
    return dict(groups=output, unique_outcome_files=len(seen), lineage_roots=[str(p) for p in roots],
        note="Includes nested bank_runtime/calls and qa_runtime/calls and explicitly registered recovery parents. "
             "Distinct physical outcomes are not deduplicated by identical requests. Missing usage is unknown, "
             "not zero. Query projection and projection_embedding are separate stages. Summed service time "
             "is not parallel end-to-end wall time. Reused November 13 retrieval artifacts have no new calls here.")


def _wire_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def qa_input_tokens(root: Path, rows: dict, questions: dict) -> dict:
    """Bind logical QA inputs to native usage without charging cache hits twice.

    Exact frozen prompt, full request payload, response text, and ledger hashes
    are checked. The same successful physical request can support two logical
    inputs if their prompt and answer are identical. Physical usage is counted
    separately by its actual owner scope, including any native recorded retries.
    """
    prompt_path = ROOT / "creamem3/experiments/mec_transfer_20260906/run_locomo.py"
    prompt_spec = importlib.util.spec_from_file_location("_math20_frozen_prompts", prompt_path)
    prompt_module = importlib.util.module_from_spec(prompt_spec)
    prompt_spec.loader.exec_module(prompt_module)
    template = prompt_module.frozen_prompts(ROOT)["QA_PROMPT"]
    registration = read(root / "registration.json")
    if _wire_digest(template) != registration["qa_prompt_sha256"]:
        raise ValueError("QA template differs from the registered frozen prompt")
    model = registration["qa_model"]
    successful, outcomes, seen = defaultdict(list), [], set()
    for current in helper._output_roots(root):
        for path in sorted(current.rglob("outcome.json")):
            if "calls" not in path.relative_to(current).parts:
                continue
            identity = str(path.resolve()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            outcome = read(path)
            if outcome.get("stage") != "qa":
                continue
            outcomes.append(outcome)
            if outcome.get("status") != "ok":
                continue
            req = read(path.with_name("request.json"))
            payload = req["request"]
            key = _wire_digest([req["model"], payload])
            if (key != outcome.get("request_key") or _wire_digest(payload) != req.get("request_sha256")
                    or req["model"] != model or outcome.get("model") != model):
                raise ValueError(f"QA request identity/model mismatch: {path}")
            response = read(path.with_name("response.json"))
            if _wire_digest(response) != outcome.get("response_sha256"):
                raise ValueError(f"QA response hash mismatch: {path}")
            native = outcome.get("usage") or outcome.get("usageMetadata") or {}
            count = native.get("promptTokenCount")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                continue
            if response.get("usageMetadata", {}).get("promptTokenCount") != count:
                raise ValueError(f"QA native usage mismatch: {path}")
            parts = response["candidates"][0].get("content", {}).get("parts", [])
            answer = "".join(p.get("text", "") for p in parts)
            successful[key].append(dict(outcome_path=str(path), payload=payload, answer=answer,
                tokens=count, scope=outcome.get("scope"), started_utc=outcome.get("started_utc")))

    matched, unmatched, ambiguous = {a: {} for a in ARMS}, {a: [] for a in ARMS}, {a: [] for a in ARMS}
    logical_request_counts = Counter()
    for arm in ARMS:
        for qid, row in sorted(rows[arm].items()):
            prediction = row.get("prediction")
            if not isinstance(prediction, str) or not prediction.strip():
                continue
            prompt = template.format(retrieved_texts=row["retrieved"], question=questions[qid]["question"])
            payload = dict(contents=[dict(role="user", parts=[dict(text=prompt)])],
                generationConfig=dict(temperature=0.0, maxOutputTokens=8192,
                                      thinkingConfig=dict(thinkingBudget=0)))
            key = _wire_digest([model, payload])
            possibilities = [x for x in successful.get(key, [])
                             if x["payload"] == payload and x["answer"] == prediction]
            if not possibilities:
                unmatched[arm].append(qid)
                continue
            counts = {x["tokens"] for x in possibilities}
            if len(counts) != 1:
                ambiguous[arm].append(qid)
                continue
            owner = arm + "/" + qid
            owns_call = any(x["scope"] == owner for x in possibilities)
            matched[arm][qid] = dict(input_tokens=next(iter(counts)), request_key=key,
                native_successful_outcomes=[x["outcome_path"] for x in possibilities],
                has_own_physical_success=owns_call,
                exact_request_response_reuse_inferred=not owns_call)
            logical_request_counts[key] += 1

    summaries = {}
    for arm in ARMS:
        actual = [x for x in outcomes if str(x.get("scope", "")).startswith(arm + "/")]
        native = [x.get("usage") or x.get("usageMetadata") or {} for x in actual]
        input_values = [x["promptTokenCount"] for x in native if isinstance(x.get("promptTokenCount"), int)]
        summaries[arm] = dict(logical_inputs_with_native_tokens=len(matched[arm]),
            logical_input_tokens=dist([x["input_tokens"] for x in matched[arm].values()]),
            logical_input_token_sum=sum(x["input_tokens"] for x in matched[arm].values()),
            exact_request_response_reuse_inferred=sum(x["exact_request_response_reuse_inferred"] for x in matched[arm].values()),
            unmatched_completed_qa_ids=unmatched[arm], ambiguous_native_token_ids=ambiguous[arm],
            physical_owner_calls=sum(int(x.get("physical_calls", bool(x.get("attempted", True)))) for x in actual),
            physical_owner_input_tokens=0 if not actual else (sum(input_values) if input_values else None),
            physical_owner_outcomes_without_input_usage=len(actual)-len(input_values))
    paired_ids = sorted(set(matched["normal"]) & set(matched["random"]))
    left = [matched["normal"][q]["input_tokens"] for q in paired_ids]
    right = [matched["random"][q]["input_tokens"] for q in paired_ids]
    deltas = [a-b for a, b in zip(left, right)]
    symmetric_percent = [200*abs(a-b)/(a+b) if a+b else 0.0 for a, b in zip(left, right)]
    paired = dict(n=len(paired_ids), normal_mean=sum(left)/len(left) if left else None,
        random_mean=sum(right)/len(right) if right else None, normal_minus_random_tokens=dist(deltas),
        absolute_difference_tokens=dist([abs(x) for x in deltas]),
        symmetric_absolute_difference_percent=dist(symmetric_percent),
        exactly_equal_pairs=sum(a == b for a, b in zip(left, right)),
        within_5percent_pairs=sum(x <= 5 for x in symmetric_percent),
        within_10percent_pairs=sum(x <= 10 for x in symmetric_percent),
        symmetric_relative_difference_definition="200*abs(normal-random)/(normal+random)",
        strict_per_question_token_matched=False)
    if left and sum(left):
        paired["random_relative_to_normal_mean_percent"] = 100*(sum(right)/sum(left)-1)
    return dict(status=("not_yet_observed" if not any(matched.values()) and not outcomes else
                       "ok" if all(not unmatched[a] and not ambiguous[a] for a in ARMS) else "incomplete_native_binding"),
        model=model, registered_qa_prompt_sha256=registration["qa_prompt_sha256"], arms=summaries, paired=paired,
        logical_records=matched, unique_matched_physical_request_keys=len(logical_request_counts),
        shared_request_key_logical_multiplicity=dict(Counter(logical_request_counts.values())),
        note="Logical input tokens describe each answered question's exact full QA prompt using provider native "
             "promptTokenCount. Identical cached prompt/response reuse inherits that same logical input length "
             "but incurs no extra physical call. Physical owner totals count actual outcome files once, including "
             "recorded retries, and are not a fair per-arm cost allocation when one arm owns a shared call. "
             "Similar arm means do not establish strict per-question token matching. No countTokens API or tokenizer estimate is used.")


def _load_records(folder: Path, arm: str, questions: dict) -> dict:
    result = {}
    for path in sorted(folder.glob("*.json")):
        row = read(path)
        qid = row.get("qa_id")
        if qid not in questions or qid in result or path.stem != qid:
            raise ValueError(f"Unknown/duplicate/misnamed question: {path}")
        if row.get("arm", arm) != arm or row.get("sample_id") != questions[qid]["sample_id"]:
            raise ValueError(f"Identity mismatch: {path}")
        if row.get("question", questions[qid]["question"]) != questions[qid]["question"]:
            raise ValueError(f"Question mismatch: {path}")
        result[qid] = row
    return result


def _selection_audit(records: dict) -> dict:
    violations = {}
    counts, candidates, rendered = [], [], []
    optimization_states = Counter()
    for qid, row in records.items():
        c, s = row.get("candidate_fact_ids"), row.get("selected_fact_ids")
        errors = []
        if isinstance(c, list):
            candidates.append(len(c))
            if len(c) != len(set(c)) or len(c) > 120:
                errors.append("candidate_duplicate_or_exceeds_120")
        if isinstance(s, list):
            counts.append(len(s))
            if not isinstance(c, list):
                errors.append("selection_without_candidate_ids")
            elif (len(s) != min(20, len(c)) or len(s) != len(set(s))
                  or s != [fid for fid in c if fid in set(s)]):
                errors.append("selection_count_membership_or_order")
        selection = row.get("selection")
        if isinstance(selection, dict):
            if selection.get("character_truncation") is True:
                errors.append("unexpected_character_truncation")
            if isinstance(selection.get("prefilter_fact_ids"), list) and selection["prefilter_fact_ids"] != s:
                errors.append("selector_record_id_mismatch")
            opt = selection.get("optimization") or {}
            optimization_states[str(opt.get("termination", "unknown"))] += 1
            if ("char_budget" in opt and "nonbinding_character_bound" in selection
                    and opt["char_budget"] != selection["nonbinding_character_bound"]):
                errors.append("character_bound_mismatch")
        if isinstance(row.get("retrieved"), str):
            rendered.append(len(row["retrieved"]))
        if errors:
            violations[qid] = errors
    return dict(candidate_count=dist(candidates), selected_count=dist(counts),
        rendered_reader_context_characters=dist(rendered), optimization_termination_counts=dict(optimization_states),
        violations=violations,
        budget="Exactly min(20, candidate count) facts, preserving candidate order; candidate cap 120; "
               "the historical math20 character bound is nonbinding. No model reranker.")


def source_recall(records: dict, questions: dict, scored: dict) -> dict:
    dataset = ROOT / "data/locomo/locomo10.json"
    if not dataset.is_file():
        return dict(status="unavailable", reason="raw source annotations unavailable")
    gold = {}
    for conversation in read(dataset):
        for i, item in enumerate(conversation["qa"]):
            gold[f"{conversation['sample_id']}-q{i:03d}"] = set(re.findall(r"D\d+:\d+", str(item.get("evidence", []))))
    facts, graph_hashes = {}, {}
    for sid in sorted({q["sample_id"] for q in questions.values()}):
        path = ROOT / "creamem3/results_locomo10_v3" / sid / "graph_online.json"
        if not path.is_file():
            return dict(status="unavailable", reason=f"source graph unavailable: {sid}")
        facts[sid] = {f["id"]: set(re.findall(r"D\d+:\d+", str(f.get("source_dia_ids", []))))
                      for f in read(path)["fact_nodes"]}
        graph_hashes[sid] = sha(path)
    stage_fields = {"candidate_cap120": "candidate_fact_ids", "math20_selected": "selected_fact_ids"}
    stage_scores = {stage: {a: {} for a in ARMS} for stage in stage_fields}
    for stage, field in stage_fields.items():
        for arm in ARMS:
            for qid, row in records[arm].items():
                ids, targets = row.get(field), gold.get(qid)
                if not isinstance(ids, list) or not targets:
                    continue
                if stage == "math20_selected" and not ids:
                    continue
                source_map = facts[questions[qid]["sample_id"]]
                if any(fid not in source_map for fid in ids):
                    raise ValueError(f"Unknown fact in {arm}/{qid}/{stage}")
                recovered = set().union(*(source_map[fid] for fid in ids))
                stage_scores[stage][arm][qid] = len(recovered & targets) / len(targets)

    def summarize(values: dict) -> dict:
        return dict(n=len(values), mean_percent=100*sum(values.values())/len(values) if values else None,
                    all_gold_sources=sum(x == 1 for x in values.values()))

    def paired_at(ids: set, stage: str) -> dict:
        a, b = stage_scores[stage]["normal"], stage_scores[stage]["random"]
        ids = sorted(ids & set(a) & set(b))
        left, right = {q: a[q] for q in ids}, {q: b[q] for q in ids}
        return dict(n=len(ids), normal=summarize(left), random=summarize(right),
                    delta_pp=100*sum(left[q]-right[q] for q in ids)/len(ids) if ids else None)

    qa_pairs = set(scored["normal"]) & set(scored["random"])
    all_stage_pairs = set(questions)
    for arm in ARMS:
        for stage in stage_fields:
            all_stage_pairs &= set(stage_scores[stage][arm])
    results = {}
    for stage in stage_fields:
        a, b = stage_scores[stage]["normal"], stage_scores[stage]["random"]
        results[stage] = dict(normal=summarize(a), random=summarize(b),
            paired_available=paired_at(set(a) & set(b), stage),
            paired_identical_stage_cohort=paired_at(all_stage_pairs, stage),
            paired_valid_judge_cohort=paired_at(qa_pairs, stage))
    return dict(status="ok", results=results, dataset_sha256=sha(dataset), graph_sha256=graph_hashes,
        note="Macro-average annotated dialogue-source recall per question, not semantic answer completeness. "
             "Gold is used only for post-selection analysis. Frozen selection artifacts are included before QA "
             "completion; both stages are also compared on identical paired cohorts to avoid denominator drift.")


def analyze(root: Path | str, output: Path | str) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Report destination must be new; refusing to overwrite")
    raw = read(root / "questions.json")
    if isinstance(raw, dict):
        raw = raw.get("questions", raw.get("rows"))
    if not isinstance(raw, list):
        raise ValueError("questions.json must contain a list")
    questions = {q["qa_id"]: q for q in raw}
    if len(questions) != len(raw):
        raise ValueError("Duplicate question IDs")
    rows, diagnostics, scored, arms = {}, {}, {}, {}
    for arm in ARMS:
        rows[arm] = _load_records(root / "rows" / arm, arm, questions)
        diagnostics[arm] = _load_records(root / "selections" / arm, arm, questions)
        for qid, row in rows[arm].items():
            old = diagnostics[arm].get(qid, {})
            for field in ("candidate_fact_ids", "selected_fact_ids"):
                if field in old and field in row and old[field] != row[field]:
                    raise ValueError(f"QA/frozen selection mismatch: {arm}/{qid}/{field}")
            diagnostics[arm][qid] = {**old, **row}
        scored[arm] = {qid: row for qid, row in rows[arm].items() if helper._scored(row)}
        n, correct = len(scored[arm]), sum(int(r["correct"]) for r in scored[arm].values())
        by_conv = {}
        for sid in sorted({q["sample_id"] for q in questions.values()}):
            subset = [r for qid, r in scored[arm].items() if questions[qid]["sample_id"] == sid]
            cc = sum(int(r["correct"]) for r in subset)
            by_conv[sid] = dict(n=len(subset), correct=cc,
                expected_n=sum(q["sample_id"] == sid for q in questions.values()),
                accuracy_percent=100*cc/len(subset) if subset else None)
        arms[arm] = dict(expected_n=len(questions), saved_qa_rows=len(rows[arm]), judged_n=n,
            settled_n=n, correct=correct, accuracy_percent=100*correct/n if n else None,
            full_experiment_accuracy_percent=100*correct/n if n == len(questions) == EXPECTED_N else None,
            missing_ids=sorted(set(questions)-set(rows[arm])),
            unscored_ids=sorted(set(rows[arm])-set(scored[arm])),
            rejected_policy_zero_ids=sorted(q for q, r in rows[arm].items() if r.get("status") == "policy_zero"),
            status_counts=dict(Counter(str(r.get("status", "unknown")) for r in rows[arm].values())),
            errors={q: {k: v for k, v in r.items() if "error" in k and v}
                    for q, r in rows[arm].items() if q not in scored[arm]},
            selection_audit=_selection_audit(diagnostics[arm]), by_conversation=by_conv)
    banks = bank_statistics(root, questions)
    judgments_complete = len(questions) == EXPECTED_N and all(len(scored[a]) == EXPECTED_N for a in ARMS)
    contracts_pass = (not banks["violations"] and not banks["missing_ids"]
                      and all(not arms[a]["selection_audit"]["violations"] for a in ARMS))
    complete = judgments_complete and contracts_pass
    paired = helper._paired(scored["normal"], scored["random"], questions)
    paired["scope"] = "Only valid QA plus Judge in both arms; no policy zeros. Incomplete cohorts are provisional."
    paired["full_experiment"] = complete
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
        status="complete" if complete else "incomplete_not_final", intervention_seed=92620,
        expected_questions_per_arm=EXPECTED_N, expected_total_settled=2*EXPECTED_N,
        manifest_questions=len(questions), total_valid_judgments=sum(len(v) for v in scored.values()),
        questions_sha256=sha(root / "questions.json"), report_code_sha256=sha(Path(__file__)),
        utility_code_sha256=sha(HELPER_PATH), arms=arms, paired=paired,
        projection_banks=banks, judgments_complete=judgments_complete, available_contracts_pass=contracts_pass,
        source_recall=source_recall(diagnostics, questions, scored), calls=usage(root),
        qa_input_tokens=qa_input_tokens(root, rows, questions),
        failure_policy="No automatic failure zeros. Selector, Reader, QA, and Judge failures remain unscored. "
                       "The November 13 reranker-failure policy does not apply to this no-reranker experiment.",
        interpretation="One prespecified graph seed. Conversation-clustered intervals concern conversation/question "
                       "sampling, not graph-seed variability. Native token fields exclude unknown usage; "
                       "this report does not reconstruct unmeasured end-to-end wall time.")
    lines = ["# LoCoMo mathematical top-20, no reranker", "", f"Status: **{result['status']}**.", "",
             "| Arm | Valid Judge / expected | Correct | QA accuracy |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        a = arms[arm]
        value = f"{a['accuracy_percent']:.2f}%" if a["accuracy_percent"] is not None else "not scored"
        if not complete and a["accuracy_percent"] is not None:
            value += " (completed subset)"
        lines.append(f"| {arm} | {a['judged_n']}/{len(questions)} | {a['correct']} | {value} |")
    if paired["n"]:
        lines += ["", f"Paired normal minus random: {paired['delta_pp']:+.2f} pp on {paired['n']} questions."]
        ci = paired["cluster_bootstrap95_pp"]
        if ci:
            lines.append(f"Conversation-clustered 95% interval: [{ci[0]:+.2f}, {ci[1]:+.2f}] pp (20,000 draws).")
        lines.append(f"Normal-only / random-only correct: {paired['normal_only_correct']} / {paired['random_only_correct']}.")
        lines.append(f"Same selected fact set / order: {paired['same_selected_set']} / {paired['same_selected_order']} of {paired['selected_id_pairs_observed']} paired inputs.")
    src = result["source_recall"]
    if src["status"] == "ok":
        lines += ["", "| Source recall, same paired stage cohort | n | Normal | Random | Difference |",
                  "|---|---:|---:|---:|---:|"]
        for stage, stats in src["results"].items():
            p = stats["paired_identical_stage_cohort"]
            if p["n"]:
                lines.append(f"| {stage} | {p['n']} | {p['normal']['mean_percent']:.2f}% | {p['random']['mean_percent']:.2f}% | {p['delta_pp']:+.2f} pp |")
    lines += ["", "Failures and missing judgments are not counted as wrong. A final experiment requires 282 valid judgments in each arm.",
              "", "## Physical calls and native token usage", "",
              "| Stage | Model | Calls | Input tokens | Output tokens |", "|---|---|---:|---:|---:|"]
    for group in result["calls"]["groups"]:
        native = group["native_token_fields"]
        lines.append(f"| {group['stage']} | {group['model']} | {group['physical_calls']} | {native['promptTokenCount']} | {native['candidatesTokenCount']} |")
    tokens = result["qa_input_tokens"]
    lines += ["", "## Logical QA input length versus physical usage", "",
              "| Arm | Native-bound logical inputs | Mean input tokens/Q | Reused exact requests | Physical-owner QA input tokens |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        token_arm = tokens["arms"][arm]
        mean = token_arm["logical_input_tokens"]["mean"]
        value = f"{mean:.2f}" if mean is not None else "unknown"
        lines.append(f"| {arm} | {token_arm['logical_inputs_with_native_tokens']} | {value} | {token_arm['exact_request_response_reuse_inferred']} | {token_arm['physical_owner_input_tokens']} |")
    tp = tokens["paired"]
    if tp["n"]:
        lines += ["", f"On {tp['n']} paired inputs, mean normal-minus-random length is {tp['normal_minus_random_tokens']['mean']:+.2f} tokens; "
                  f"mean absolute paired difference is {tp['absolute_difference_tokens']['mean']:.2f} tokens. "
                  f"Exactly equal inputs: {tp['exactly_equal_pairs']}; within 5% symmetric difference: {tp['within_5percent_pairs']}."]
    lines += ["", "Logical cached inputs retain the same token length, but physical costs count the underlying API call only once. "
              "The physical-owner split depends on which arm made a shared call first; it is not a matched per-arm billing comparison. "
              "Similar average lengths do not mean strict token matching for every question.",
              "", "Native missing usage remains unknown. Summed request durations are not end-to-end wall time; shared projection calls are counted once, not once per arm.", ""]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "valid_judgments": result["total_valid_judgments"]}, ensure_ascii=False))

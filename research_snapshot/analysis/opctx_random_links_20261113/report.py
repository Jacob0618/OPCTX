"""Read-only paired analysis for the single-seed LoCoMo link-routing control.

Only a new report directory is written. Experimental records and checkpoints are
never modified. Missing or invalid Judge outputs are not scored as incorrect.
The sole exception is the separately authorized, explicitly tagged failure of
the original two reranker attempts; these are policy zeros, not Judge verdicts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("normal", "random")
EXPECTED_N = 282
BOOTSTRAP_N = 20000
BOOTSTRAP_SEED = 20261113
POLICY_ZERO_REASON = "rerank_invalid_after_two_attempts"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dist(values: list[int | float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(array.mean()), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max()),
            "histogram": {str(k): v for k, v in sorted(Counter(values).items())}}


def _scored(row: dict) -> bool:
    tags = re.findall(r"\[\[(yes|no)\]\]", str(row.get("judge_raw") or "").lower())
    return (row.get("status") == "ok" and type(row.get("correct")) in (int, bool)
            and row["correct"] in (0, 1) and isinstance(row.get("prediction"), str)
            and bool(row["prediction"].strip())
            and len(tags) == 1 and int(tags[0] == "yes") == int(row["correct"])
            and not row.get("judge_error") and not row.get("judge_parse_error"))


def _policy_zero(row: dict) -> bool:
    failure = row.get("rerank_failure")
    return (row.get("status") == "policy_zero"
            and row.get("policy_zero_reason") == POLICY_ZERO_REASON
            and type(row.get("correct")) in (int, bool) and row["correct"] == 0
            and row.get("prediction") == "" and row.get("judge_raw") is None
            and row.get("qa_skipped") is True and row.get("judge_skipped") is True
            and row.get("selected_fact_ids") == []
            and isinstance(failure, dict) and failure.get("attempts") == 2
            and failure.get("fallback") is True
            and isinstance(failure.get("error"), str) and bool(failure["error"].strip()))


def _output_roots(root: Path) -> list[Path]:
    """Visit ancestor outputs once, oldest first; reject cyclic lineage."""
    ordered, visited, visiting = [], set(), set()

    def visit(path: Path):
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Missing parent/current output directory: {path}")
        identity = str(path).casefold()
        if identity in visiting:
            raise ValueError("Cyclic parent_outputs lineage")
        if identity in visited:
            return
        visiting.add(identity)
        registration = path / "registration.json"
        value = _read(registration) if registration.is_file() else {}
        parents = value.get("parent_outputs", [])
        if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
            raise ValueError("registration.parent_outputs must be a list of paths")
        for parent in parents:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = path / parent_path
            visit(parent_path)
        visiting.remove(identity)
        visited.add(identity)
        ordered.append(path)

    visit(root)
    return ordered


def _paired(left: dict, right: dict, questions: dict) -> dict:
    ids = sorted(set(left) & set(right))
    if not ids:
        return {"n": 0, "status": "not_scored", "delta_pp": None}
    grouped = defaultdict(list)
    cross = Counter()
    for qid in ids:
        a, b = int(left[qid]["correct"]), int(right[qid]["correct"])
        grouped[questions[qid]["sample_id"]].append(a - b)
        cross[(a, b)] += 1
    cluster_ids = sorted(grouped)
    sizes = np.asarray([len(grouped[s]) for s in cluster_ids], dtype=int)
    totals = np.asarray([sum(grouped[s]) for s in cluster_ids], dtype=float)
    delta = float(totals.sum() / sizes.sum())
    ci, p = None, None
    if len(cluster_ids) >= 2:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draw = rng.integers(len(cluster_ids), size=(BOOTSTRAP_N, len(cluster_ids)))
        values = 100 * totals[draw].sum(axis=1) / sizes[draw].sum(axis=1)
        ci = [float(x) for x in np.quantile(values, [.025, .975])]
        if len(cluster_ids) <= 16:
            signs = np.asarray(list(itertools.product((-1, 1), repeat=len(cluster_ids))))
            permuted = signs @ totals / sizes.sum()
            p = float(np.mean(np.abs(permuted) >= abs(delta) - 1e-12))
    same_set = same_order = known_selected = 0
    for qid in ids:
        a, b = left[qid].get("selected_fact_ids"), right[qid].get("selected_fact_ids")
        if isinstance(a, list) and a and isinstance(b, list) and b:
            known_selected += 1
            same_set += set(a) == set(b)
            same_order += a == b
    return dict(n=len(ids), clusters=len(cluster_ids), cluster_ids=cluster_ids,
        normal_accuracy_percent=100 * sum(int(left[i]["correct"]) for i in ids) / len(ids),
        random_accuracy_percent=100 * sum(int(right[i]["correct"]) for i in ids) / len(ids),
        delta_pp=100 * delta, cluster_bootstrap95_pp=ci,
        bootstrap_replicates=BOOTSTRAP_N, bootstrap_seed=BOOTSTRAP_SEED,
        resampling="whole conversations, paired arms, question-weighted mean in each resample",
        exact_cluster_signflip_two_sided_p=p,
        signflip_note="Exploratory test; assumes exchangeable/symmetric paired conversation effects.",
        normal_only_correct=cross[(1, 0)], random_only_correct=cross[(0, 1)],
        both_correct=cross[(1, 1)], both_incorrect=cross[(0, 0)],
        selected_id_pairs_observed=known_selected, same_selected_set=same_set,
        same_selected_order=same_order,
        scope="paired settled questions; may include explicitly authorized policy zeros; incomplete subsets are not final scores")


def _usage(root: Path) -> dict:
    groups = defaultdict(list)
    roots, seen_outcomes = _output_roots(root), set()
    for current in roots:
        for path in sorted((current / "calls").glob("*/outcome.json")):
            identity = str(path.resolve()).casefold()
            if identity in seen_outcomes:
                continue
            seen_outcomes.add(identity)
            item = _read(path)
            groups[(str(item.get("stage", "unknown")), str(item.get("model", "unknown")))].append(item)
    output = []
    for (stage, model), items in sorted(groups.items()):
        usage = [x.get("usage") or x.get("usageMetadata") for x in items]
        usage = [x for x in usage if isinstance(x, dict)]
        fields = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount",
                  "cachedContentTokenCount", "totalTokenCount")
        sums = {key: sum(x[key] for x in usage if isinstance(x.get(key), (int, float)))
                if any(isinstance(x.get(key), (int, float)) for x in usage) else None for key in fields}
        seconds = [x.get("service_seconds") for x in items if isinstance(x.get("service_seconds"), (int, float))]
        output.append(dict(stage=stage, model=model, outcome_records=len(items),
            physical_calls=sum(int(x.get("physical_calls", bool(x.get("attempted", True)))) for x in items),
            status_counts=dict(Counter(str(x.get("status", "unknown")) for x in items)),
            outcomes_with_native_usage=len(usage), outcomes_without_native_usage=len(items)-len(usage),
            native_token_fields=sums, summed_service_seconds=sum(seconds) if seconds else None))
    return dict(groups=output, lineage_roots=[str(p) for p in roots],
                unique_outcome_files=len(seen_outcomes),
                note="All registration.parent_outputs are recursively included once. Distinct physical outcome files "
                "are not deduplicated by request or response contents, because repeated calls still consume resources. "
                "Native usage fields are summed independently; missing usage is unknown, not zero. "
                "Summed request service time is not parallel end-to-end wall time. No pricing estimate.")


def _source_recall(all_rows: dict, questions: dict) -> dict:
    dataset = ROOT / "data/locomo/locomo10.json"
    if not dataset.exists():
        return {"status": "unavailable", "reason": "raw source annotations unavailable"}
    gold = {}
    for conversation in _read(dataset):
        for i, q in enumerate(conversation["qa"]):
            qid = f"{conversation['sample_id']}-q{i:03d}"
            gold[qid] = set(re.findall(r"D\d+:\d+", str(q.get("evidence", []))))
    facts = {}
    for sid in sorted({q["sample_id"] for q in questions.values()}):
        path = ROOT / "creamem3/results_locomo10_v3" / sid / "graph_online.json"
        if not path.exists():
            return {"status": "unavailable", "reason": "one or more source fact graphs unavailable"}
        facts[sid] = {f["id"]: set(re.findall(r"D\d+:\d+", str(f.get("source_dia_ids", []))))
                      for f in _read(path)["fact_nodes"]}
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for field, name in (("candidate_fact_ids", "candidate"), ("selected_fact_ids", "selected")):
            scores = []
            for qid, row in all_rows[arm].items():
                ids, targets = row.get(field), gold.get(qid)
                if not isinstance(ids, list) or not targets:
                    continue
                if name == "selected" and not isinstance(row.get("reader_audit"), dict):
                    continue
                source_map = facts[questions[qid]["sample_id"]]
                if any(fid not in source_map for fid in ids):
                    raise ValueError(f"Unknown fact ID in {arm}/{qid}/{field}")
                recovered = set().union(*(source_map[f] for f in ids))
                scores.append(len(recovered & targets) / len(targets))
            output[arm][name] = dict(n=len(scores), mean_percent=100*float(np.mean(scores)) if scores else None,
                                    all_sources=sum(x == 1 for x in scores))
    return dict(status="ok", results=output,
                note="Annotated dialogue-source recall, not semantic evidence completeness. "
                     "Available candidate records are counted independently of Judge completion. "
                     "Selected-source recall only covers selections that actually reached Reader; policy-zero rerank failures are excluded, not treated as selected empty contexts.")


def analyze(root: Path | str, output: Path | str) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Report destination must be new; refusing to overwrite")
    question_path = root / "questions.json"
    raw_questions = _read(question_path)
    if isinstance(raw_questions, dict):
        raw_questions = raw_questions.get("questions", raw_questions.get("rows"))
    if not isinstance(raw_questions, list):
        raise ValueError("questions.json must contain a question list")
    questions = {q["qa_id"]: q for q in raw_questions}
    if len(questions) != len(raw_questions):
        raise ValueError("Duplicate question IDs")
    all_rows, scored, judged, summaries = {}, {}, {}, {}
    for arm in ARMS:
        rows = {}
        for path in sorted((root / "rows" / arm).glob("*.json")):
            row = _read(path)
            qid = row.get("qa_id")
            if qid not in questions or qid in rows:
                raise ValueError(f"Unknown or duplicate question ID: {arm}/{path.name}")
            if row.get("arm", arm) != arm or row.get("sample_id") != questions[qid]["sample_id"]:
                raise ValueError(f"Row identity mismatch: {arm}/{qid}")
            if row.get("question", questions[qid]["question"]) != questions[qid]["question"]:
                raise ValueError(f"Question text mismatch: {arm}/{qid}")
            rows[qid] = row
        all_rows[arm] = rows
        judged[arm] = {qid: row for qid, row in rows.items() if _scored(row)}
        policy_rows = {qid: row for qid, row in rows.items() if _policy_zero(row)}
        scored[arm] = {**judged[arm], **policy_rows}
        n = len(scored[arm])
        correct = sum(int(row["correct"]) for row in scored[arm].values())
        by_conv = {}
        for sid in sorted({q["sample_id"] for q in questions.values()}):
            rr = [r for qid, r in scored[arm].items() if questions[qid]["sample_id"] == sid]
            nn, cc = len(rr), sum(int(r["correct"]) for r in rr)
            by_conv[sid] = dict(n=nn, settled_n=nn,
                               judged_n=sum(r.get("status") == "ok" for r in rr),
                               policy_zero_n=sum(r.get("status") == "policy_zero" for r in rr),
                               expected_n=sum(q["sample_id"] == sid for q in questions.values()),
                               correct=cc, accuracy_percent=100*cc/nn if nn else None)
        summaries[arm] = dict(expected_n=len(questions), saved_records=len(rows), scored_n=n,
            settled_n=n, judged_n=len(judged[arm]), policy_zero_n=len(policy_rows),
            explicit_policy_n=len(policy_rows), policy_zero_ids=sorted(policy_rows),
            policy_zero_errors={qid: row["rerank_failure"] for qid, row in sorted(policy_rows.items())},
            rejected_policy_zero_ids=sorted(qid for qid, row in rows.items()
                                           if row.get("status") == "policy_zero" and qid not in policy_rows),
            qa_outputs_present=sum(isinstance(r.get("prediction"), str) and bool(r["prediction"].strip()) for r in rows.values()),
            correct=correct, accuracy_percent=100*correct/n if n else None,
            accuracy_denominator="valid Judge outputs plus the explicitly authorized two-attempt rerank failure zeros; all other missing cases excluded",
            full_experiment_accuracy_percent=100*correct/n if n == len(questions) == EXPECTED_N else None,
            missing_ids=sorted(set(questions)-set(rows)), unscored_ids=sorted(set(rows)-set(scored[arm])),
            status_counts=dict(Counter(str(r.get("status", "unknown")) for r in rows.values())),
            selected_count=_dist([len(r["selected_fact_ids"]) for r in rows.values()
                                  if isinstance(r.get("selected_fact_ids"), list) and isinstance(r.get("reader_audit"), dict)]),
            reader_budget_compliance=dict(
                rows_entering_reader=sum(isinstance(r.get("reader_audit"), dict) for r in rows.values()),
                non20_fact_ids=sorted(qid for qid, r in rows.items() if isinstance(r.get("reader_audit"), dict)
                                     and (not isinstance(r.get("selected_fact_ids"), list) or len(r["selected_fact_ids"]) != 20)),
                policy_zero_rows_not_entering_reader=len(policy_rows),
                note="The 20-fact constraint applies only to actual Reader inputs; skipped reranker failures are reported separately."),
            candidate_count=_dist([len(r["candidate_fact_ids"]) for r in rows.values() if isinstance(r.get("candidate_fact_ids"), list)]),
            by_conversation=by_conv)
    complete = len(questions) == EXPECTED_N and all(len(scored[a]) == EXPECTED_N for a in ARMS)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status="complete" if complete else "incomplete_not_final", root=str(root),
        intervention_seed=92620, expected_questions=EXPECTED_N, manifest_questions=len(questions),
        questions_sha256=_sha(question_path), report_code_sha256=_sha(Path(__file__)),
        arms=summaries, paired=_paired(scored["normal"], scored["random"], questions),
        paired_valid_judge_only=_paired(judged["normal"], judged["random"], questions),
        explicit_failure_policy=dict(reason=POLICY_ZERO_REASON,
            rule="Only invalid reranker output after both original semantic attempts is settled as zero; no fallback QA or fabricated Judge call.",
            authorization="User explicitly authorized this policy during the experiment; not an original pre-registration rule."),
        source_recall=_source_recall(all_rows, questions), calls=_usage(root),
        interpretation="Single preselected link-randomization seed; does not estimate variability over graph seeds. "
                       "Only the explicitly authorized two-attempt invalid-reranker failures are zero-scored; "
                       "all other incomplete or failed questions remain missing.")
    result["paired"]["full_experiment"] = complete
    result["paired_valid_judge_only"]["scope"] = "Secondary paired comparison restricted to questions with valid Judge outputs in both arms; policy zeros excluded."
    result["paired_valid_judge_only"]["full_experiment"] = False
    lines = ["# LoCoMo multi-hop randomized-routing QA", "", f"Status: **{result['status']}**.", "",
             "| Arm | Settled / expected | Valid Judge | Explicit failure zeros | Correct | QA accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        a = summaries[arm]
        value = f"{a['accuracy_percent']:.2f}%" if a['accuracy_percent'] is not None else "not scored"
        if not complete and a['accuracy_percent'] is not None:
            value += " (completed subset only)"
        lines.append(f"| {arm} | {a['settled_n']}/{len(questions)} | {a['judged_n']} | {a['policy_zero_n']} | {a['correct']} | {value} |")
    p = result["paired"]
    if p["n"]:
        ci = p["cluster_bootstrap95_pp"]
        lines += ["", f"Paired normal minus random: {p['delta_pp']:+.2f} percentage points over {p['n']} matched questions."]
        if ci is not None:
            lines.append(f"Conversation-clustered 95% bootstrap interval: [{ci[0]:+.2f}, {ci[1]:+.2f}] pp ({BOOTSTRAP_N} draws).")
        lines.append(f"Normal-only correct / random-only correct: {p['normal_only_correct']} / {p['random_only_correct']}.")
        lines.append(f"Same selected fact set / order: {p['same_selected_set']} / {p['same_selected_order']} of {p['selected_id_pairs_observed']} observed pairs.")
    secondary = result["paired_valid_judge_only"]
    if secondary["n"]:
        lines += ["", f"Secondary comparison, valid Judge in both arms only: n={secondary['n']}, normal minus random {secondary['delta_pp']:+.2f} pp."]
    lines += ["", "## Explicit reranker-failure policy", "",
              "By the user's explicit authorization, a reranker still returning invalid IDs after its original two attempts is counted as incorrect. No fallback QA or Judge call is fabricated. This policy was authorized during the run, not pre-registered."]
    for arm in ARMS:
        a = summaries[arm]
        lines.append(f"- {arm}: {a['policy_zero_n']} policy zeros; IDs: {', '.join(a['policy_zero_ids']) or 'none'}.")
        for qid, failure in a["policy_zero_errors"].items():
            error = str(failure["error"]).replace("\n", " ")
            lines.append(f"  - {qid}: {error}")
    lines += ["", "All other missing or invalid judgments remain missing. Any incomplete-subset scores are provisional, not the final 282-question result.", "",
              "The final fact limit is 20; the candidate cap is 120. Equal limits do not require equal actual deduplicated candidate counts.", "",
              "Only one fixed graph seed is evaluated; the confidence interval concerns question/conversation sampling, not graph-seed variability.", ""]
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
                      "scored": {a: result["arms"][a]["scored_n"] for a in ARMS}}, ensure_ascii=False))

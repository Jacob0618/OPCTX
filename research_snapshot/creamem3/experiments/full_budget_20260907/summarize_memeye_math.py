"""Read-only full-corpus paired audit; no selection or provider is invoked.

--status reads names/counts only. Evaluator data is inaccessible until both
score-complete markers pass a sealed full-corpus census. --write creates only
new analysis artifacts, never replaces different existing content.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from creamem3.experiments.full_budget_20260907 import run_memeye_math as harness
from creamem3.experiments.mec_transfer_20260906 import summarize_memeye as previous

direct = harness.direct
read, require, check_seal = previous.read, previous.require, previous.seal


def status_counts(scope: Path = harness.ROOT) -> dict[str, Any]:
    """No score-row bodies, predictions, gold, or model output are read."""
    return {arm: {
        "prediction_rows": len(list((scope / arm / "results/predict/rows").glob("*.json"))),
        "score_rows": len(list((scope / arm / "results/score/rows").glob("*.json"))),
        "prediction_complete": (scope / arm / "results/predict/complete.json").is_file(),
        "score_complete": (scope / arm / "results/score/complete.json").is_file(),
        "failure_or_orphan_markers": sum(1 for label in ("failures", "orphans")
            for _ in (scope / arm / "provider_cache").rglob(f"{label}/**/*.json")),
    } for arm in harness.ARMS}


def completion_barrier(scope: Path, row_ids: Sequence[str]) -> None:
    """Must precede any evaluator/gold read; merely having 371 files is not enough."""
    for arm in harness.ARMS:
        path = scope / arm / "results/score/complete.json"
        require(path.is_file(), "both arm scores must complete before final analysis")
        value = read(path)
        check_seal(value, "complete_sha256")
        require(value.get("complete") is True, "unsealed/incomplete score phase")
        require([r["row_id"] for r in value["rows"]] == list(row_ids), "completion full-corpus census drift")


def _reader_bindings(prediction: Mapping[str, Any], item: Mapping[str, Any],
                     authority: Mapping[str, Any], finals: Mapping[Any, Any],
                     branch: Path) -> set[tuple[str, str]]:
    row_id, context = item["query"]["row_id"], item["context"]
    require(prediction["question"] == item["query"]["question"], "Reader question drift")
    require(prediction["selected_fact_ids"] == item["selection"]["selected_fact_ids"], "Reader actual IDs drift")
    require(direct._sha256(prediction["selected_fact_ids"]) == prediction["selected_fact_ids_sha256"], "Reader IDs hash drift")
    require(prediction["context_stats"] == item["stats"], "Reader context stats drift")
    require(prediction["context_sha256"] == direct._sha256(context), "Reader actual context SHA drift")
    require(prediction["context_stats"]["context_chars"] == len(context), "Reader actual character length drift")
    require(prediction["assembly_sha256"] == item["assembly_sha256"], "Reader assembly drift")
    stages = prediction["reader_stages"]
    route = prediction["reader_route"]
    expected_stages = ["focused", "answer"] if route == "two_stage" else ["structured_fused"]
    require([r["stage"] for r in stages] == expected_stages, "Reader route/stages drift")
    prompt_by_stage, metadata_by_stage = {}, {}
    if route == "two_stage":
        focused_final = finals[("focused", stages[0]["request_sha256"])]
        aggregation = direct.predict._parse_json_object(focused_final["text"])
        if not aggregation:
            aggregation = {"_failed": True}
        qa_context = authority["render"](aggregation, context, "candidate_first")
        prompt_by_stage["focused"] = str(authority["focused"]).replace("{question}", item["query"]["question"]).replace("{context}", context)
        prompt_by_stage["answer"] = str(authority["open"]).format(context=qa_context, question=item["query"]["question"])
        metadata_by_stage = {"focused": {"context_sha256": direct._sha256(context)},
                             "answer": {"qa_context_sha256": direct._sha256(qa_context)}}
        expected_prediction = finals[("answer", stages[1]["request_sha256"])]["text"].strip()
    else:
        require(route == "structured_fused", "unknown Reader route")
        prompt_by_stage[route] = str(authority["fused"]).format(question=item["query"]["question"], context=context)
        metadata_by_stage[route] = {"context_sha256": direct._sha256(context)}
        parsed = direct.predict._parse_json_object(finals[(route, stages[0]["request_sha256"])]["text"])
        expected_prediction = str(parsed.get("final_answer") or "Not found in memory.").strip()
    require(prediction["prediction"] == expected_prediction, "prediction does not match physical Reader response")
    linked = set()
    for stage in stages:
        name = stage["stage"]
        final = finals[(name, stage["request_sha256"])]
        require(final["response_sha256"] == stage["response_sha256"], "Reader response binding drift")
        plan = read(branch / "plans" / name / f"{final['resume_hash']}.json")
        expected_metadata = {"cell_id": direct.CELL_ID, "row_id": row_id, **metadata_by_stage[name]}
        require(plan["cache_metadata"] == expected_metadata, "Reader context metadata drift")
        require(plan["prompt_sha256"] == direct._sha256(prompt_by_stage[name]), "actual QA prompt/context drift")
        linked.add((name, final["resume_hash"]))
    return linked


def _judge_binding(row: Mapping[str, Any], prediction: Mapping[str, Any], gold: Mapping[str, Any],
                   branch: Path, finals: Mapping[Any, Any]) -> None:
    triple = (gold["question"], gold["gold"], prediction["prediction"])
    direct.score._validate_judge_row(row, triple)
    require((row["x"], row["y"]) == (gold["x"], gold["y"]), "evaluator category drift")
    final = finals[("judge", row["request_sha256"])]
    require(final["response_sha256"] == row["response_sha256"], "score/Judge response binding drift")
    expected_path = branch / "responses/judge" / f"{row['resume_hash']}.json"
    require(Path(row["cache_path"]).resolve() == expected_path.resolve(), "Judge final cache outside exact identity")
    require(row["cache_file_sha256"] == direct._file_sha256(expected_path), "Judge final cache hash drift")
    ordinal = int(row["successful_attempt_ordinal"])
    require(final["attempt_ordinal"] == ordinal, "Judge successful ordinal drift")
    for key, linked in (("plan_file_sha256", branch / "plans/judge" / f"{row['resume_hash']}.json"),
        ("attempt_request_file_sha256", branch / "attempts/judge" / row["resume_hash"] / f"attempt_{ordinal:03d}_request.json"),
        ("attempt_response_file_sha256", branch / "attempts/judge" / row["resume_hash"] / f"attempt_{ordinal:03d}_response.json")):
        require(row[key] == direct._file_sha256(linked), f"Judge {key} drift")
    chain = []
    for previous_ordinal in range(1, ordinal):
        base = branch / "attempts/judge" / row["resume_hash"]
        request_path = base / f"attempt_{previous_ordinal:03d}_request.json"
        response_path = base / f"attempt_{previous_ordinal:03d}_response.json"
        response = read(response_path)
        require(response["outcome_kind"] == "explicit_provider_error" and response["retryable"] is True,
                "Judge retry predecessor was not a recoverable transport error")
        chain.append({"ordinal": previous_ordinal, "intent_file_sha256": direct._file_sha256(request_path),
            "response_file_sha256": direct._file_sha256(response_path), "status_code": response.get("status_code")})
    require(row["explicit_retry_chain"] == chain, "Judge explicit retry chain drift")
    verdict = direct.g35_runtime.parse_mgtm_judge_response(final["text"])
    require((row["raw_score"], row["mapped_score"], row["reasoning"]) ==
            (verdict.raw_score, verdict.mapped_score, verdict.reasoning), "Judge parser/mapping replay drift")


def _plan_census(cache: Path, reader_links: set[tuple[str, str]],
                 gold: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cover unused Judge parse attempts as well as the ultimately scored reply."""
    triple_to_row = {direct._sha256([gold[p["row_id"]]["question"], gold[p["row_id"]]["gold"], p["prediction"]]): p for p in predictions}
    counts, judge_parse_events = Counter(), []
    for phase, model in (("predict", "g25_reader"), ("score", "g35_judge")):
        branch = cache / "provider" / phase / model
        plan_keys = {(p.parent.name, p.stem) for p in (branch / "plans").glob("*/*.json")}
        attempt_keys = {(p.parent.name, p.name) for p in (branch / "attempts").glob("*/*") if p.is_dir()}
        final_keys = {(p.parent.name, p.stem) for p in (branch / "responses").glob("*/*.json")}
        require(plan_keys == attempt_keys == final_keys, "unmatched plan/attempt/final identity census")
        if phase == "predict":
            require(plan_keys == reader_links, "unused or missing Reader request identities")
        for stage, resume in sorted(plan_keys):
            plan = read(branch / "plans" / stage / f"{resume}.json")
            counts[stage] += 1
            attempts = list((branch / "attempts" / stage / resume).glob("attempt_*_request.json"))
            final = read(branch / "responses" / stage / f"{resume}.json")
            require(sorted(read(p)["attempt_ordinal"] for p in attempts) == list(range(1, int(final["attempt_ordinal"]) + 1)),
                    "missing or post-success physical attempt")
            if phase == "score":
                metadata = plan["cache_metadata"]
                triple_sha = metadata["exact_triple_sha256"]
                require(triple_sha in triple_to_row, "unrequested Judge triple")
                prediction = triple_to_row[triple_sha]
                g = gold[prediction["row_id"]]
                prompt = direct.g35_runtime.build_mgtm_judge_prompt(g["question"], g["gold"], prediction["prediction"])
                require(plan["prompt_sha256"] == direct._sha256(prompt), "Judge actual prompt/triple drift")
                require(metadata["scope"] == direct.score.JUDGE_CACHE_SCOPE, "Judge scope drift")
                allowed = {"scope", "exact_triple_sha256", "parse_retry_ordinal"}
                require(not (set(metadata) - allowed), "unexpected Judge metadata")
                try:
                    direct.g35_runtime.parse_mgtm_judge_response(final["text"])
                except Exception as exc:
                    judge_parse_events.append({"row_id": prediction["row_id"], "resume_hash": resume,
                        "parse_retry_ordinal": metadata.get("parse_retry_ordinal", 1), "exception_type": type(exc).__name__,
                        "recovered_by_final_scored_row": True})
    return {"plan_counts": dict(counts), "judge_parse_events": judge_parse_events}


def arm_audit(scope: Path, arm: str, gold: Mapping[str, Any], row_ids: Sequence[str], workers: int) -> dict[str, Any]:
    n = len(row_ids)
    root, cache = scope / arm / "results", scope / arm / "provider_cache"
    gate = harness.gate(arm, workers)
    prediction_complete, predictions = direct._load_prediction_complete(root, gate["contract"], gate["prepared"])
    score_complete = read(root / "score/complete.json")
    check_seal(score_complete, "complete_sha256")
    require(score_complete["contract_sha256"] == gate["contract"]["contract_sha256"], "score contract drift")
    require(score_complete["rows_sha256"] == direct._sha256(score_complete["rows"]), "score receipt list drift")
    require([p["row_id"] for p in predictions] == list(row_ids), "prediction full-corpus order drift")
    require(len(predictions) == len(score_complete["rows"]) == n, "completed arm census mismatch")
    require(len(list((root / "predict/rows").glob("*.json"))) == len(list((root / "score/rows").glob("*.json"))) == n,
            "extra or missing materialized result rows")
    provider = previous.provider_audit(cache)
    finals = provider.pop("_final_by_request")
    scored, reader_links = {}, set()
    for prediction, receipt, item in zip(predictions, score_complete["rows"], gate["prepared"], strict=True):
        row_id = prediction["row_id"]
        path = direct._score_path(root, prediction)
        require(receipt["row_id"] == row_id and Path(receipt["path"]).resolve() == path.resolve(), "score receipt identity drift")
        require(receipt["file_sha256"] == direct._file_sha256(path), "score receipt file SHA drift")
        row = direct._load_score_row(path, prediction, gold[row_id])
        require(receipt["row_sha256"] == row["row_sha256"], "score receipt row SHA drift")
        reader_links.update(_reader_bindings(prediction, item, gate["authority"], finals, cache / "provider/predict/g25_reader"))
        _judge_binding(row, prediction, gold[row_id], cache / "provider/score/g35_judge", finals)
        scored[row_id] = {"row_id": row_id, "task": row["task"], "x": row["x"], "y": row["y"], "xy": row["xy"],
            "question": prediction["question"], "prediction": prediction["prediction"], "reasoning": row["reasoning"],
            "mapped_score": float(row["mapped_score"]), "raw_score": float(row["raw_score"]),
            "selected_fact_ids": prediction["selected_fact_ids"], "context_sha256": prediction["context_sha256"],
            "context_chars": len(item["context"]), "reader_route": prediction["reader_route"],
            "prediction_path": str(direct._prediction_path(root, item)), "score_path": str(path)}
    provider.update(_plan_census(cache, reader_links, gold, predictions))
    mapped, raw_sum = (sum(row[key] for row in scored.values()) for key in ("mapped_score", "raw_score"))
    require(math.isclose(mapped / n, score_complete["mapped_mean"], abs_tol=1e-12), "mapped summary drift")
    require(math.isclose(raw_sum / n, score_complete["raw_mean"], abs_tol=1e-12), "raw summary drift")
    require(score_complete["qwen_provider_calls"] == 0, "Qwen provider calls found")
    for complete, key in ((prediction_complete, "reader_usage"), (score_complete, "judge_usage")):
        for stage, receipt in complete[key]["stages"].items():
            actual = provider["by_stage"].get(stage, {"physical_attempts": 0, "usage": {k: 0 for k in previous.TOKENS}})
            require(actual["physical_attempts"] == receipt["provider_calls"] and actual["usage"] == receipt["usage"], "durable usage receipt drift")
    finished = (root / "score/complete.json").stat().st_mtime
    chars = [row["context_chars"] for row in scored.values()]
    return {"output_root": str(root), "cache_root": str(cache), "rows": n, "complete": n, "pipeline_errors": 0,
        "mapped_sum": mapped, "mapped_percent": mapped / n * 100, "raw_sum": raw_sum, "raw_mean": raw_sum / n,
        "mapped_distribution": dict(Counter(str(row["mapped_score"]) for row in scored.values())),
        "context_chars_mean": statistics.fmean(chars), "context_chars_total": sum(chars), "context_chars_min": min(chars), "context_chars_max": max(chars),
        "provider": provider, "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "elapsed_from_first_request_seconds": finished - provider["first_request_epoch"],
        "score_complete_file_sha256": direct._file_sha256(root / "score/complete.json"),
        "prediction_complete_file_sha256": direct._file_sha256(root / "predict/complete.json"), "row_data": scored}


def paired_metrics(dense: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    require(set(dense) == set(new) and bool(dense), "paired question census mismatch")
    pairs = []
    for row_id in dense:
        d, m = dense[row_id], new[row_id]
        require((d["question"], d["x"], d["y"]) == (m["question"], m["x"], m["y"]), "paired input/category drift")
        delta, raw_delta = m["mapped_score"] - d["mapped_score"], m["raw_score"] - d["raw_score"]
        same = d["prediction"] == m["prediction"]
        pairs.append({"row_id": row_id, "task": d["task"], "x": d["x"], "y": d["y"], "xy": d["xy"],
            "dense_mapped": d["mapped_score"], "math_mapped": m["mapped_score"], "delta": delta,
            "dense_raw": d["raw_score"], "math_raw": m["raw_score"], "raw_delta": raw_delta,
            "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie", "same_prediction": same,
            "same_prediction_mapped_drift": same and delta != 0, "same_prediction_raw_drift": same and raw_delta != 0,
            "selected_overlap": len(set(d["selected_fact_ids"]) & set(m["selected_fact_ids"])),
            "dense_chars": d["context_chars"], "math_chars": m["context_chars"]})
    outcomes = Counter(r["outcome"] for r in pairs)
    groups = {}
    for field in ("x", "y", "xy", "task"):
        bucket = defaultdict(list)
        for row in pairs:
            bucket[str(row[field])].append(row)
        groups[field] = {name: {"n": len(items),
            "dense_mapped_sum": sum(i["dense_mapped"] for i in items), "math_mapped_sum": sum(i["math_mapped"] for i in items),
            "dense_mapped_percent": statistics.fmean(i["dense_mapped"] for i in items) * 100,
            "math_mapped_percent": statistics.fmean(i["math_mapped"] for i in items) * 100,
            "delta_pp": statistics.fmean(i["delta"] for i in items) * 100,
            "dense_raw_mean": statistics.fmean(i["dense_raw"] for i in items),
            "math_raw_mean": statistics.fmean(i["math_raw"] for i in items),
            "wins": sum(i["delta"] > 0 for i in items), "losses": sum(i["delta"] < 0 for i in items), "ties": sum(i["delta"] == 0 for i in items)}
            for name, items in sorted(bucket.items())}
    return {"n": len(pairs), "math_wins": outcomes["win"], "math_losses": outcomes["loss"], "ties": outcomes["tie"],
        "delta_mapped_points": sum(r["delta"] for r in pairs), "delta_pp": statistics.fmean(r["delta"] for r in pairs) * 100,
        "delta_raw_mean": statistics.fmean(r["raw_delta"] for r in pairs),
        "same_prediction_count": sum(r["same_prediction"] for r in pairs),
        "same_prediction_mapped_drift_count": sum(r["same_prediction_mapped_drift"] for r in pairs),
        "same_prediction_raw_drift_count": sum(r["same_prediction_raw_drift"] for r in pairs),
        "same_prediction_drift_net_mapped_points": sum(r["delta"] for r in pairs if r["same_prediction"]),
        "mean_selected_overlap": statistics.fmean(r["selected_overlap"] for r in pairs), "groups": groups, "rows": pairs}


def descriptive_uncertainty(pairs: Sequence[Mapping[str, Any]], *, seed: int = 20260907,
                            repetitions: int = 10000) -> dict[str, Any]:
    """Post-evaluation descriptive sensitivity; never used for selection/tuning.

    Row bootstrap treats questions as independent (optimistic assumption).
    Task bootstrap resamples whole task graphs, preserving all within-task
    question differences and computes a pooled, question-weighted mean. Neither
    method resamples new QA/Judge runs; eight convenience tasks are not a random
    sample of a large task population, so neither interval establishes a
    generalizable treatment effect.
    """
    require(bool(pairs) and repetitions > 0, "empty/invalid bootstrap")
    rng = np.random.default_rng(seed)
    delta = np.asarray([r["delta"] for r in pairs], dtype=float)
    row_samples = delta[rng.integers(0, len(delta), size=(repetitions, len(delta)))].mean(axis=1) * 100
    grouped = defaultdict(list)
    for row in pairs:
        grouped[str(row["task"])].append(float(row["delta"]))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names])
    sizes = np.asarray([len(grouped[name]) for name in names])
    task_indices = rng.integers(0, len(names), size=(repetitions, len(names)))
    task_samples = sums[task_indices].sum(axis=1) / sizes[task_indices].sum(axis=1) * 100
    return {"seed": seed, "repetitions": repetitions, "point_delta_pp": float(delta.mean() * 100),
        "row_naive_percentile95_delta_pp": np.quantile(row_samples, [.025, .975]).tolist(),
        "task_cluster_percentile95_delta_pp": np.quantile(task_samples, [.025, .975]).tolist(),
        "task_clusters": len(names), "task_rows": {name: len(grouped[name]) for name in names},
        "includes_zero_row": bool(np.quantile(row_samples, .025) <= 0 <= np.quantile(row_samples, .975)),
        "includes_zero_task": bool(np.quantile(task_samples, .025) <= 0 <= np.quantile(task_samples, .975)),
        "limitation": "Descriptive fixed-run sensitivity only: questions share task graphs; few nonrandom task clusters; no rerun-level QA/Judge stochasticity captured; no claim of stable/generalizable gain."}


def summarize(*, workers: int = 8) -> dict[str, Any]:
    scope = harness.ROOT
    selection, reg = harness.validate_selection()
    row_ids = [row["row_id"] for row in reg["rows"]]
    completion_barrier(scope, row_ids)
    gold = direct._gold_rows()  # The only evaluator read; deliberately after the barrier.
    require(set(gold) == set(row_ids), "full dataset/gold census drift")
    arms = {arm: arm_audit(scope, arm, gold, row_ids, workers) for arm in harness.ARMS}
    diagnostics = selection["diagnostics"]
    require(len(diagnostics) == len(row_ids), "selection diagnostic census drift")
    paired = paired_metrics(arms["dense50"]["row_data"], arms["math50"]["row_data"])
    require(all(r["math_chars"] <= r["dense_chars"] for r in paired["rows"]), "matched character budget violation")
    for diag in diagnostics:
        dense, new = (arms[a]["row_data"][diag["row_id"]] for a in harness.ARMS)
        require(diag["dense_facts"] == len(dense["selected_fact_ids"]) and diag["new_facts"] == len(new["selected_fact_ids"]), "selection/count diagnostic drift")
        require(diag["budget"] == dense["context_chars"] and diag["new_chars"] == new["context_chars"], "selection/cost diagnostic drift")
    return {"schema": "full_memeye_math_downstream_audit.v1", "status": "PASS", "scope": str(scope),
        "full_dataset_not_sample": True, "rows": len(row_ids), "analysis_only_no_model_calls": True,
        "selection_or_parameters_changed": False, "new_reranker_calls": 0,
        "summary_script_sha256": direct._file_sha256(Path(__file__)),
        "provider_audit_source_sha256": direct._file_sha256(Path(previous.__file__)),
        "selection_complete_file_sha256": direct._file_sha256(scope / "selection_complete.json"),
        "preregistration_sha256": reg["seal_sha256"], "core_sha256": harness.CORE_HASH,
        "score_meaning": "MemEye mapped graded score, not binary accuracy", "arms": arms, "paired": paired,
        "descriptive_uncertainty": descriptive_uncertainty(paired["rows"]),
        "selection": {"certified_optimal": sum(r["optimal"] is True for r in diagnostics),
            "partial": sum(r["optimal"] is not True for r in diagnostics),
            "dense_count_distribution": dict(Counter(str(r["dense_facts"]) for r in diagnostics)),
            "math_count_distribution": dict(Counter(str(r["new_facts"]) for r in diagnostics)),
            "solver_seconds_total": sum(r["solver_seconds"] for r in diagnostics)},
        "interpretation": "Full371 fixed benchmark verification. Exact optimization certifies only the frozen lexical-coverage/rank-balance proxy, not answer completeness or answer correctness. No claim of independent external generalization; the prior24 pilot overlaps this full corpus."}


def markdown(result: Mapping[str, Any]) -> str:
    n, paired = result["rows"], result["paired"]
    lines = [f"# MemEye 全 {n} 题：Dense50 与预算数学检索", "",
        "固定完整数据集，非抽样。复用原 G3.5-low Online 图；G2.5 Reader 与 G3.5 Judge 的原有路由、prompt、解析及映射不变。两臂均无新增 rerank 调用；共享原始 facts 渲染与逐题字符预算。", "",
        "| 方法 | Mapped 总分 | Mapped % | Raw 均分 | 平均正文字符 |", "|---|---:|---:|---:|---:|"]
    for arm, a in result["arms"].items():
        lines.append(f"| {arm} | {a['mapped_sum']:.1f}/{n} | {a['mapped_percent']:.2f}% | {a['raw_mean']:.4f} | {a['context_chars_mean']:.1f} |")
    lines += ["", "Mapped 是分档计分，不是二元正确率；raw 取 0/0.25/0.5/0.75/1，mapped 分别映射为 0/0.5/0.5/1/1。", "",
        f"数学方法相对 Dense：{paired['math_wins']} 胜、{paired['math_losses']} 负、{paired['ties']} 平，总差 {paired['delta_mapped_points']:+.1f} 分（{paired['delta_pp']:+.2f} pp）。",
        f"两臂预测文本完全相同 {paired['same_prediction_count']} 题；其中 mapped Judge 漂移 {paired['same_prediction_mapped_drift_count']} 题、raw 漂移 {paired['same_prediction_raw_drift_count']} 题，mapped 净贡献 {paired['same_prediction_drift_net_mapped_points']:+.1f} 分。不能把这部分漂移直接归因为检索收益。", ""]
    uncertainty = result["descriptive_uncertainty"]
    naive, clustered = (uncertainty[k] for k in ("row_naive_percentile95_delta_pp", "task_cluster_percentile95_delta_pp"))
    lines += ["## 不确定性：目前不能称为稳定提升", "",
        f"按同题差值做 {uncertainty['repetitions']:,} 次确定种子 bootstrap：把题目当独立样本的描述性 95% 区间为 [{naive[0]:+.2f}, {naive[1]:+.2f}] pp；按 {uncertainty['task_clusters']} 个 task 整组重采样、保留组内相关性的区间为 [{clustered[0]:+.2f}, {clustered[1]:+.2f}] pp。",
        "同一个 task 的问题共享记忆图，不应将 371 题全当独立事件；task 又只有少量且并非随机抽取的组。这些区间只反映当前运行对问题/任务组成的敏感性，不包含重新调用 QA/Judge 的随机波动，不是稳定泛化增益的证明。", ""]
    for field in ("x", "y", "xy", "task"):
        lines += [f"## {field.upper()} 分组", "", "| 类别 | N | Dense % | 数学 % | Δ pp | 胜/负/平 |", "|---|---:|---:|---:|---:|---:|"]
        for name, g in paired["groups"][field].items():
            lines.append(f"| {name} | {g['n']} | {g['dense_mapped_percent']:.2f} | {g['math_mapped_percent']:.2f} | {g['delta_pp']:+.2f} | {g['wins']}/{g['losses']}/{g['ties']} |")
        lines.append("")
    s = result["selection"]
    lines += ["## 选择与运行完整性", "", f"选择 {n}/{n}；数学代理目标完全最优 {s['certified_optimal']}，未证明最优/部分解 {s['partial']}；Dense 条数分布 {s['dense_count_distribution']}，数学条数分布 {s['math_count_distribution']}。每题数学正文字符不超过同题 Dense50，条数不超过 50；字符不是模型 token。", "",
        "| 方法 | 完成 | 逻辑调用 | 物理尝试 | 已恢复传输重试 | Reader 解析事件 | Judge 解析事件 | Prompt/output/thought/total tokens |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm, a in result["arms"].items():
        p, u = a["provider"], a["provider"]["usage"]
        lines.append(f"| {arm} | {a['complete']}/{n} | {p['logical_calls']} | {p['physical_attempts']} | {p['recovered_retries']} | {len(p['parse_events'])} | {len(p['judge_parse_events'])} | {' / '.join(str(u[k]) for k in previous.TOKENS)} |")
    lines += ["", "| 方法 / 阶段 | 逻辑 / 物理 | Prompt | Output | Thought | Total tokens |", "|---|---:|---:|---:|---:|---:|"]
    for arm, a in result["arms"].items():
        for stage, p in a["provider"]["by_stage"].items():
            u = p["usage"]
            lines.append(f"| {arm} / {stage} | {p['logical_calls']} / {p['physical_attempts']} | {u['prompt_tokens']} | {u['output_tokens']} | {u['thought_tokens']} | {u['total_tokens']} |")
    d, m = (result["arms"][a] for a in harness.ARMS)
    chars_saved = d["context_chars_total"] - m["context_chars_total"]
    qa_prompt = {a: sum(p["usage"]["prompt_tokens"] for stage, p in v["provider"]["by_stage"].items() if stage != "judge") for a, v in result["arms"].items()}
    lines += ["", f"数学臂正文少 {chars_saved:,} 字符（{chars_saved / d['context_chars_total'] * 100:.2f}%）；QA 实际 prompt tokens：Dense {qa_prompt['dense50']:,}、数学 {qa_prompt['math50']:,}。QA token 含模板，以及 two-stage 的中间聚合/二次回答上下文，不能将字符直接换算为 token 或费用。本报告未假定供应商单价。",
        f"两臂从最早持久化请求到最终评分完成的墙钟跨度分别为 {d['elapsed_from_first_request_seconds']:.1f} / {m['elapsed_from_first_request_seconds']:.1f} 秒（各 8 workers；不含先前建图）。", "",
        f"已核验：全量 canonical/graph/selection/manifest 合同；{2 * n} 个输入 context 的实际字符及 QA prompt 绑定；预测/评分行封印、完成清单、缓存文件哈希；每个请求计划、物理尝试、最终缓存及显式重试链；Judge 原始响应重解析与 mapped 映射；全量 token 用量回执。failure/orphan/未恢复错误为 0。物理尝试与逻辑调用之差可能同时含传输重试及 Judge 解析重试，不能全称为网络错误。", "",
        "## 解释边界", "", "这是固定全 371 题验证，不是采样试分，运行中没有按得分调参。它包含先前 pilot 题目，不能称为独立全新测试集。数学最优仅指词面覆盖与 Dense/BM25 排名均衡代理的最优，不等于答案字段完整或答案最优；源事实未写入、关系绑定错误及 QA 未使用证据仍可能发生。", "",
        "本次隔离比较的参照是同一轮、同一完整图候选池、同一 raw renderer、同字符预算的 Dense50。历史 MEC50v2 是另一套候选/选择规则及另一轮调用；其历史分数不是此数学目标的纯净对照，不能把历史分数到本轮数学分数的差额都归因于数学优化。原事实正文没有被本轮压缩或改写，字符减少来自选中的 facts 集合变化。", "",
        f"完整逐题分数、配对差异、请求/重试证据见 `{result['scope']}\\analysis\\math_full_metrics.json`。", ""]
    return "\n".join(lines)


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, "refusing to overwrite different analysis report")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    require(not (args.status and args.write), "status cannot write")
    if args.status:
        print(json.dumps(status_counts(), ensure_ascii=False, indent=2))
        return 0
    result = summarize(workers=args.workers)
    if args.write:
        target = harness.ROOT / "analysis"
        direct._write_once(target / "math_full_metrics.json", result)
        payload = markdown(result)
        _write_text_once(target / "math_full_results.md", payload)
        _write_text_once(Path(__file__).with_name("math_full_results.md"), payload)
    public = {"status": result["status"], "rows": result["rows"], "selection": result["selection"],
        "descriptive_uncertainty": result["descriptive_uncertainty"],
        "arms": {name: {k: v for k, v in arm.items() if k not in {"row_data", "provider"}} for name, arm in result["arms"].items()},
        "paired": {k: v for k, v in result["paired"].items() if k not in {"rows", "groups"}}}
    print(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

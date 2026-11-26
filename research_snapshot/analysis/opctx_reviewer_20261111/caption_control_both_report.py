"""Completed Lite-Judge comparison for both frozen public-caption arms; no API."""
from pathlib import Path
import csv
import json
import statistics
import zipfile
import numpy as np
import caption_control as c

OUT = Path(__file__).resolve().parent
ARCHIVE = Path(r"D:\memeye_baseline_archive_20260923\results_and_provenance.zip")
OLD_ENHANCED = Path(r"C:\Users\86139\Desktop\OpCtx_保留结果_20260927\MemEye_enhanced_reader.zip")
SEED, REPS = 20261111, 100000


def main():
    reg = c.validate(c.DEFAULT_ROOT)
    queries = {q["row_id"]: q for q in reg["questions"]}
    gold = c.b.read(c.SOURCE / "scoring/open.json")["rows"]
    by_text = {}
    for q in queries.values():
        by_text.setdefault((q["task"], q["question"]), []).append(q["row_id"])
    enhanced = {r["row_id"]: r for p in c.DEFAULT_ROOT.glob("tasks/*/arms/enhanced48/scores/g35lite/enhanced_reader/*.json") for r in [c.b.read(p)]}
    assert set(enhanced) == set(queries), "Enhanced Lite not yet complete; do not extrapolate"
    assert all(r["judge"] == "gemini-3.5-flash-lite" for r in enhanced.values())
    disposition_path = c.DEFAULT_ROOT / "user_dispositions/20261111_lite_only.json"
    disposition = c.b.read(disposition_path)
    assert disposition["original_registration_sha256"] == reg["seal_sha256"]
    assert disposition["flash_branch_status"] == "excluded_by_user_decision_not_pending"
    assert c.b.file_sha(c.DEFAULT_ROOT / "registration.json") == disposition["original_registration_file_sha256"]
    for path, sha in disposition["protected_rejected_artifacts"].items():
        assert c.b.file_sha(path) == sha
    refs = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for method, label in [("fc_text", "FC(T) public captions"), ("srag_text", "SRAG(T) public captions")]:
            prefix = "D/creamem3_cache/memeye_baselines_20260920_r1/" + method + "/"
            preds = [json.loads(z.read(n)) for n in z.namelist() if n.startswith(prefix + "predictions/") and n.endswith(".json")]
            mapping = {}
            for p in preds:
                q = p["query"]; candidates = by_text[(q["task"], q["question"])]
                assert q["row_id"] in candidates or len(candidates) == 1
                mapping[q["row_id"]] = q["row_id"] if q["row_id"] in candidates else candidates[0]
            assert len(mapping) == len(set(mapping.values())) == 371
            scores = [json.loads(z.read(n)) for n in z.namelist() if n.startswith(prefix + "scores/") and n.endswith(".json")]
            refs[label] = {mapping[r["row_id"]]: r for r in scores}
    with zipfile.ZipFile(OLD_ENHANCED) as z:
        rows = [json.loads(z.read(n)) for n in z.namelist() if "/scores/g35lite/enhanced_reader/" in n and n.endswith(".json")]
        refs["Enhanced historical Gemini captions (Lite Judge)"] = {r["row_id"]: r for r in rows}
    for label, rows in refs.items():
        assert set(rows) == set(queries), label
        for rid, row in rows.items():
            assert (row["task"], row["x"], row["y"]) == (enhanced[rid]["task"], enhanced[rid]["x"], enhanced[rid]["y"])
            if "gold_sha256" in row:
                assert row["gold_sha256"] == c.b.sha(gold[rid]["gold"])
    tasks = sorted({q["task"] for q in queries.values()})
    assert len(tasks) == 8
    sizes = np.array([sum(q["task"] == t for q in queries.values()) for t in tasks])
    draws = np.random.default_rng(SEED).integers(0, 8, size=(REPS, 8))
    original = json.loads((OUT / "caption_control_original_comparison.json").read_text(encoding="utf-8"))
    categories = original["categories"][:]
    comparisons = original["paired_cluster_bootstrap"][:]
    for label, rows in refs.items():
        for metric in ("mapped_score", "raw_score"):
            diffs = {rid: enhanced[rid][metric] - rows[rid][metric] for rid in queries}
            sums = np.array([sum(diffs[rid] for rid, q in queries.items() if q["task"] == t) for t in tasks])
            lo, hi = np.quantile(100 * sums[draws].sum(axis=1) / sizes[draws].sum(axis=1), [0.025, 0.975])
            comparisons.append(dict(comparison="Enhanced public captions minus " + label, metric=metric,
                n=371, clusters=8, difference_pp=100*statistics.mean(diffs.values()),
                ci95_low_pp=float(lo), ci95_high_pp=float(hi), paired_row_count=371,
                bootstrap_seed=SEED, bootstrap_repetitions=REPS))
    for label, rows in [("Enhanced public captions", enhanced), ("Enhanced historical Gemini captions (Lite Judge)", refs["Enhanced historical Gemini captions (Lite Judge)"])]:
        for axis, values in [("overall", ["Overall"]), ("x", ["X1", "X2", "X3", "X4"]), ("y", ["Y1", "Y2", "Y3"]), ("task", tasks)]:
            for value in values:
                rs = list(rows.values()) if axis == "overall" else [r for r in rows.values() if r[axis] == value]
                categories.append(dict(method=label, axis=axis, category=value, n=len(rs),
                    raw_percent=100*statistics.mean(r["raw_score"] for r in rs),
                    mapped_percent=100*statistics.mean(r["mapped_score"] for r in rs)))
    notes = ["All shown scores use 3.5 Flash-Lite Judge. Historical enhanced 48.92 uses Flash Judge; its same-Lite-Judge reference is 48.65.",
        "Original OpCtx and historical FC/SRAG QA are 2.5 Flash-Lite. Enhanced QA is 3.5 Flash-Lite; enhanced-baseline contrasts are not backbone-matched.",
        "Public image_caption is the sole prespecified input intervention within each OpCtx variant; historical counterparts were not rerun contemporaneously, so provider-time drift/generation variation are not isolated.",
        "Both retained Lite branches complete. User decided to retain only3.5 Flash-Lite scoring; the Flash branch is excluded, not awaiting recovery. Original registration and371 rejected3.6 outputs remain unchanged."]
    payload = dict(n=371, judge="gemini-3.5-flash-lite", categories=categories,
        paired_cluster_bootstrap=comparisons, notes=notes,
        rejected_flash_responses=371, requested_flash_judge_complete=False,
        flash_branch_status=disposition["flash_branch_status"], retained_scoring_plan_complete=True,
        user_disposition_path=str(disposition_path), user_disposition_sha256=disposition["seal_sha256"],
        source_registration_sha256=reg["seal_sha256"])
    (OUT / "caption_control_same_source_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for suffix, rows in [("categories", categories), ("paired", comparisons)]:
        with (OUT / ("caption_control_same_source_" + suffix + ".csv")).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    order = ["X1", "X2", "X3", "X4", "Y1", "Y2", "Y3", "Overall"]
    methods = list(dict.fromkeys(r["method"] for r in categories))
    lines = ["# MemEye 同源公开图像描述结果", "", "两版各 371/371 题，以下统一使用 3.5 Flash-Lite Judge。", "", "## 官方映射分数（%）", "",
        "| 配置 | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
    for method in methods:
        lookup = {r["category"]: r for r in categories if r["method"] == method}
        lines.append("| " + method + " | " + " | ".join(f"{lookup[x]['mapped_percent']:.2f}" for x in order) + " |")
    lines += ["", "## Raw Judge 分数（%）", "", "| 配置 | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
    for method in methods:
        lookup = {r["category"]: r for r in categories if r["method"] == method}
        lines.append("| " + method + " | " + " | ".join(f"{lookup[x]['raw_percent']:.2f}" for x in order) + " |")
    lines += ["", "## 映射分数的配对差", "", "| 对比 | 差值 pp | 8 场景聚类 95% CI |", "|---|---:|---:|"]
    for r in comparisons:
        if r["metric"] == "mapped_score":
            lines.append(f"| {r['comparison']} | {r['difference_pp']:+.2f} | [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] |")
    lines += ["", "371/371 题严格配对；8 个任务场景成组有放回抽样 100,000 次，seed=20261111；统计量保持逐题 micro 均值。", "",
        "原版与 FC/SRAG 的 QA 均为 2.5 Flash-Lite；增强版 QA 为 3.5 Flash-Lite，因此增强版对 baseline 的比较不属于同骨干比较。历史增强版 48.92 为 Flash Judge，统一 Lite Judge 后其参考分数是 48.65。", "",
        "公开描述是各 OpCtx 版本的唯一预设输入干预，但历史对照未同日重跑，不能完全排除模型服务时间或生成波动。", "",
        "按用户“先只保留3.5 Flash-Lite评分”的决定，Flash 评分分支已取消、不再列为待恢复。两个保留的 Lite Judge 分支均已完成；原注册及 371 份被拒绝的 3.6 原始输出保持不变。", ""]
    (OUT / "caption_control_same_source_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": [r for r in categories if r["axis"] == "overall"],
                      "paired": [r for r in comparisons if r["metric"] == "mapped_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Read-only paired public-caption comparison; writes only local audit reports."""
from pathlib import Path
import csv
import json
import statistics
import zipfile
import numpy as np
import caption_control as c

OUT = Path(__file__).resolve().parent
ARCHIVE = Path(r"D:\memeye_baseline_archive_20260923\results_and_provenance.zip")
HIST = Path(r"C:\Users\86139\Desktop\OpCtx_保留结果_20260927\MemEye_47.44.zip")
SEED, REPS = 20261111, 100000


def main():
    reg = c.validate(c.DEFAULT_ROOT)
    queries = {q["row_id"]: q for q in reg["questions"]}
    by_text = {}
    for q in reg["questions"]:
        by_text.setdefault((q["task"], q["question"]), []).append(q["row_id"])
    assert len(queries) == 371

    def match(q):
        # Card Q6/Q17 and Q7/Q18 repeat identical text/gold. Keep their
        # retained exact IDs; only uniquely matching renamed IDs are mapped.
        candidates = by_text[(q["task"], q["question"])]
        if q["row_id"] in candidates:
            return q["row_id"]
        assert len(candidates) == 1, (q["row_id"], candidates)
        return candidates[0]
    gold = c.b.read(c.SOURCE / "scoring/open.json")["rows"]
    new = {r["row_id"]: r for p in c.DEFAULT_ROOT.glob("tasks/*/arms/original47/scores/g35lite/original_reader/*.json")
           for r in [c.b.read(p)]}
    assert set(new) == set(queries)
    methods = {"OpCtx public captions": new}
    source_bindings = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        names = z.namelist()
        for method, label in [("fc_text", "FC(T) public captions"), ("srag_text", "SRAG(T) public captions")]:
            prefix = "D/creamem3_cache/memeye_baselines_20260920_r1/" + method + "/"
            preds = [json.loads(z.read(n)) for n in names if n.startswith(prefix + "predictions/") and n.endswith(".json")]
            id_map = {p["query"]["row_id"]: match(p["query"]) for p in preds}
            assert len(id_map) == len(set(id_map.values())) == 371
            scores = [json.loads(z.read(n)) for n in names if n.startswith(prefix + "scores/") and n.endswith(".json")]
            rows = {}
            for row in scores:
                rid = id_map[row["row_id"]]
                assert row["gold_sha256"] == c.b.sha(gold[rid]["gold"])
                assert (row["task"], row["x"], row["y"]) == (new[rid]["task"], new[rid]["x"], new[rid]["y"])
                rows[rid] = row
            assert set(rows) == set(queries)
            methods[label] = rows
            source_bindings[label] = {"archive": str(ARCHIVE), "prefix": prefix,
                                      "matching": "exact task + question text, one-to-one 371/371; gold and categories verified"}
    with zipfile.ZipFile(HIST) as z:
        rows = {r["row_id"]: r for n in z.namelist() if n.startswith("results/score/rows/") and n.endswith(".json")
                for r in [json.loads(z.read(n))]}
        assert set(rows) == set(queries)
        for rid, row in rows.items():
            assert row["question_sha256"] == c.b.sha(queries[rid]["question"])
            assert row["gold_sha256"] == c.b.sha(gold[rid]["gold"])
        methods["OpCtx historical Gemini captions"] = rows
        source_bindings["OpCtx historical Gemini captions"] = {"archive": str(HIST), "prefix": "results/score/rows/",
            "matching": "371/371 exact IDs plus question/gold hashes"}
    tasks = sorted({q["task"] for q in queries.values()})
    assert len(tasks) == 8
    sizes = np.array([sum(q["task"] == t for q in queries.values()) for t in tasks])
    draws = np.random.default_rng(SEED).integers(0, 8, size=(REPS, 8))
    paired = []
    for label, rows in methods.items():
        if label == "OpCtx public captions":
            continue
        for metric in ("mapped_score", "raw_score"):
            diffs = {rid: new[rid][metric] - rows[rid][metric] for rid in queries}
            sums = np.array([sum(diffs[rid] for rid, q in queries.items() if q["task"] == t) for t in tasks])
            boot = 100 * sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            paired.append({"comparison": "OpCtx public captions minus " + label, "metric": metric,
                           "n": 371, "clusters": 8, "difference_pp": 100 * statistics.mean(diffs.values()),
                           "ci95_low_pp": float(lo), "ci95_high_pp": float(hi),
                           "paired_row_count": len(diffs), "bootstrap_seed": SEED, "bootstrap_repetitions": REPS})
    categories = []
    for method, rows in methods.items():
        for axis, values in [("overall", ["Overall"]), ("x", ["X1", "X2", "X3", "X4"]), ("y", ["Y1", "Y2", "Y3"]), ("task", tasks)]:
            for value in values:
                chosen = list(rows.values()) if axis == "overall" else [r for r in rows.values() if r[axis] == value]
                categories.append({"method": method, "axis": axis, "category": value, "n": len(chosen),
                    "mapped_percent": 100 * statistics.mean(r["mapped_score"] for r in chosen),
                    "raw_percent": 100 * statistics.mean(r["raw_score"] for r in chosen)})
    payload = {"n": 371, "judge": "gemini-3.5-flash-lite", "source_bindings": source_bindings,
        "paired_cluster_bootstrap": paired, "categories": categories,
        "interpretation": "Public image_caption was the sole prespecified input intervention. Historical baselines/OpCtx were not rerun contemporaneously; provider-time drift and generation variance are not isolated. Eight task clusters give limited interval resolution."}
    (OUT / "caption_control_original_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for suffix, rows in [("categories", categories), ("paired", paired)]:
        with (OUT / ("caption_control_original_" + suffix + ".csv")).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    order = ["X1", "X2", "X3", "X4", "Y1", "Y2", "Y3", "Overall"]
    lines = ["# MemEye 同源公开图像描述对照（原 Reader）", "", "371 道开放题，QA 2.5 Flash-Lite，Judge 3.5 Flash-Lite；分数为官方映射后的逐题均值。", "",
             "| 配置 | " + " | ".join(order) + " |", "|---|" + "---:|" * len(order)]
    for method in methods:
        lookup = {r["category"]: r for r in categories if r["method"] == method}
        lines.append("| " + method + " | " + " | ".join(f"{lookup[x]['mapped_percent']:.2f}" for x in order) + " |")
    lines += ["", "| 配对差（公开描述 OpCtx − 对照） | 差值 pp | 场景聚类 95% CI |", "|---|---:|---:|"]
    for row in paired:
        if row["metric"] == "mapped_score":
            label = row["comparison"].split(" minus ", 1)[1]
            lines.append(f"| {label} | {row['difference_pp']:+.2f} | [{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] |")
    lines += ["", "配对通过 371/371 的题目文本、来源任务及 gold 哈希验证，不按题号后缀猜测；8 个场景整体有放回抽样 100,000 次，seed=20261111，保留 micro 加权口径。", "",
              "公开描述是预设唯一输入干预，但历史对照并非同日重跑，不能完全排除模型服务时间与生成波动影响。增强 Reader 的评分尚待完成，不据此表预判其效果。", ""]
    (OUT / "caption_control_original_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": [r for r in categories if r["axis"] == "overall"], "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

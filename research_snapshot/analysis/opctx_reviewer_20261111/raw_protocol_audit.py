"""Read retained MemEye Judge scores; no model calls, no experiment mutations."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import zipfile

OUT = Path(__file__).resolve().parent
ARCHIVE = Path(r"D:\memeye_baseline_archive_20260923\results_and_provenance.zip")
SUMMARY = Path(r"D:\memeye_baseline_archive_20260923\baseline_results.json")
PRESERVED = Path(r"C:\Users\86139\Desktop\OpCtx_保留结果_20260927")
LABELS = {"fc_text": "FC(T)", "srag_text": "SRAG(T)", "reflexion_ng": "Refl.",
          "gen_agents": "Gen.Ag.", "memoryos": "MemOS", "a_mem": "A-Mem",
          "simplemem_text": "SM(T)", "fc_v": "FC(V)", "srag_v": "SRAG(V)",
          "mma": "MMA", "m2a": "M2A", "sm_v": "SM(V)"}

def mapped(v):
    return 0.0 if v < 0.25 else 0.5 if v < 0.75 else 1.0

def stats(rows):
    return {"n": len(rows), "raw_percent": 100 * sum(r["raw_score"] for r in rows) / len(rows),
            "mapped_percent": 100 * sum(r["mapped_score"] for r in rows) / len(rows)}

def inspect_rows(panel, label, rows, source):
    assert len(rows) == 371, (label, len(rows))
    assert len({r["row_id"] for r in rows}) == 371, label
    assert all(mapped(float(r["raw_score"])) == float(r["mapped_score"]) for r in rows), label
    result = {"panel": panel, "method": label, "source": source,
              "verification": "371 unique retained per-question raw/mapped scores recomputed",
              **stats(rows), "histogram_raw": dict(sorted(Counter(float(r["raw_score"]) for r in rows).items()))}
    for axis in ("x", "y", "xy", "task"):
        result["by_" + axis] = {v: stats([r for r in rows if r[axis] == v]) for v in sorted({r[axis] for r in rows})}
    result["macro12_raw_percent"] = sum(v["raw_percent"] for v in result["by_xy"].values()) / 12
    result["macro12_mapped_percent"] = sum(v["mapped_percent"] for v in result["by_xy"].values()) / 12
    result["judge_models"] = sorted({r.get("judge", r.get("judge_generation", r.get("generation", {})).get("model", "unspecified")) for r in rows})
    return result

results = []
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
with zipfile.ZipFile(ARCHIVE) as z:
    names = z.namelist()
    for s in summary["summaries"]:
        if s["panel"] != "original_2.5_lite_QA":
            continue
        src = s["source"].replace("\\", "/").replace(":", "")
        prefix = src.rsplit("/", 1)[0] + "/scores/"
        members = [n for n in names if n.startswith(prefix) and n.endswith(".json")]
        rows = [json.loads(z.read(n)) for n in members]
        r = inspect_rows("original main table", LABELS[s["method"]], rows, str(ARCHIVE) + "!" + prefix)
        assert abs(r["raw_percent"] - s["summary"]["raw_percent"]) < 1e-9
        assert abs(r["mapped_percent"] - s["summary"]["mapped_percent"]) < 1e-9
        results.append(r)

# The sole authorized failure is not a fabricated Judge verdict. Its original
# zero-policy receipt was deleted with caches; retained contemporaneous worklog
# records the 370 judged-row sums and the 371-question denominator.
results.insert(10, {"panel": "original main table", "method": "M2A", "n": 371,
    "raw_percent": 100 * 50.75 / 371, "mapped_percent": 100 * 52.5 / 371,
    "verification": "retained worklog aggregate only; 370 judged + Social Q18 authorized policy zero; category raw not recovered",
    "source": "creamem3/experiments/memeye_resume_20260921/worklog.md:108-119",
    "judge_models": ["gemini-3.5-flash-lite"]})

with zipfile.ZipFile(PRESERVED / "MemEye_47.44.zip") as z:
    prefix = "results/score/rows/"
    rows = [json.loads(z.read(n)) for n in z.namelist() if n.startswith(prefix) and n.endswith(".json")]
    r = inspect_rows("original main table", "OpCtx (47.44)", rows, str(z.filename) + "!" + prefix)
    r["judge_models"] = [json.loads(z.read("results/contract.json"))["judge_authority"]["model"]]
    r["judge_model_provenance"] = "same ZIP results/contract.json judge_authority.model"
    results.append(r)

with zipfile.ZipFile(PRESERVED / "MemEye_enhanced_reader.zip") as z:
    for judge in ("g35flash", "g35lite"):
        marker = "/scores/" + judge + "/enhanced_reader/"
        rows = [json.loads(z.read(n)) for n in z.namelist() if marker in n and n.endswith(".json")]
        results.append(inspect_rows("all-3.5-Lite enhanced Reader, dual Judge", "OpCtx Full (" + judge + ")", rows, str(z.filename) + "!*" + marker))

payload = {"official_evaluator_commit": "0358e70d714980980dfd8c87903384db474f0b16",
           "aggregation": "100 times arithmetic mean across 371 FRQ rows; not EM and not mean across 12 coordinates",
           "summary_file_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(), "results": results}
(OUT / "raw_protocol_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
with (OUT / "raw_protocol_overall.csv").open("w", encoding="utf-8-sig", newline="") as f:
    fields = ["panel", "method", "n", "raw_percent", "mapped_percent", "mapping_gain_pp", "verification"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
        w.writerow({**{k: r[k] for k in fields if k != "mapping_gain_pp"}, "mapping_gain_pp": r["mapped_percent"]-r["raw_percent"]})
with (OUT / "raw_protocol_categories.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["panel", "method", "axis", "category", "n", "raw_percent", "mapped_percent"])
    w.writeheader()
    for r in results:
        for axis in ("x", "y", "xy", "task"):
            for label, s in r.get("by_" + axis, {}).items():
                w.writerow({"panel": r["panel"], "method": r["method"], "axis": axis, "category": label, **s})
for r in results:
    print(f"{r['method']:25} n={r['n']} raw={r['raw_percent']:.6f} mapped={r['mapped_percent']:.6f} diff={r['mapped_percent']-r['raw_percent']:.6f}")

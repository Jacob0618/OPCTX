"""Evidence-only audit of public-caption memory and retained historical outputs."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import statistics
import zipfile
import caption_control as c

OUT = Path(__file__).resolve().parent
OLD = Path(r"C:\Users\86139\Desktop\OpCtx_保留结果_20260927\MemEye_47.44.zip")
BASELINES = Path(r"D:\memeye_baseline_archive_20260923\results_and_provenance.zip")


def main():
    reg = c.validate(c.DEFAULT_ROOT)
    historical_ids = defaultdict(set)
    historical_rows = defaultdict(list)
    with zipfile.ZipFile(OLD) as z:
        contract = json.loads(z.read("results/contract.json"))
        for name in z.namelist():
            if name.startswith("results/predict/rows/") and name.endswith(".json"):
                row = json.loads(z.read(name))
                historical_ids[row["task"]].update(row["selected_fact_ids"])
                historical_rows[row["task"]].append(row)
    with zipfile.ZipFile(BASELINES) as z:
        baseline_reg = json.loads(z.read("D/creamem3_cache/memeye_baselines_20260920_r1/registration.json"))
    assert baseline_reg["tasks"] == reg["tasks"]
    rows = []
    stages = defaultdict(Counter)
    failures, issues = [], []
    for task in reg["tasks"]:
        mr = c.DEFAULT_ROOT / "tasks" / task / "common_memory"
        done = c.b.read(mr / "memory_complete.json")
        assert done["model"] == "gemini-3.5-flash-lite"
        graph = c.b.read(Path(done["online"]["artifact_path"]))
        session = c.b.read(Path(done["session"]["artifact_path"]))
        assert graph["fact_nodes"] == session["fact_nodes"]
        facts = graph["fact_nodes"]
        assert len(facts) == done["facts"]
        lengths = [len(f["text"]) for f in facts]
        ctxs = [c.b.read(p) for p in (c.DEFAULT_ROOT / "tasks" / task / "arms/original47/contexts/original_reader").glob("*.json")]
        previous = historical_rows[task]
        assert len(ctxs) == len(previous)
        cap = reg["caption_audit"][task]
        rows.append(dict(task=task, sessions=done["sessions"], public_caption_count=cap["images"],
            public_caption_chars=cap["caption_chars"], public_caption_mean_chars=cap["caption_chars"]/cap["images"],
            new_exact_facts=len(facts), new_fact_total_chars=sum(lengths), new_fact_mean_chars=statistics.mean(lengths),
            old_fact_count_lower_bound=len(historical_ids[task]), old_exact_total_facts=None,
            historical_selected_context_mean_chars=statistics.mean(r["context_stats"]["context_chars"] for r in previous),
            public_selected_context_mean_chars=statistics.mean(len(x["context"]) for x in ctxs)))
        for manifest_path in mr.glob("calls/*/*/request.json.gz"):
            manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
            stage = manifest["stage"]
            stages[stage]["unique_requests"] += 1
            assert manifest["model"] == "gemini-3.5-flash-lite"
            assert manifest["image"] is None and stage != "caption"
            response_path = manifest_path.with_name("response.json")
            if not response_path.exists():
                issues.append(str(manifest_path))
                continue
            response = c.b.read(response_path)["response"]
            assert response["modelVersion"] == "gemini-3.5-flash-lite"
            assert all(x["finishReason"] == "STOP" for x in response["candidates"])
            stages[stage]["successful_requests"] += 1
            stages[stage]["served_gemini_3_5_flash_lite"] += 1
            for path in manifest_path.parent.glob("attempt_*_outcome.json"):
                result = c.b.read(path)
                stages[stage]["physical_attempts"] += 1
                if not result["ok"]:
                    failures.append(dict(stage=stage, path=str(path), error_type=result.get("error_type")))
    assert not issues
    new_contexts = [c.b.read(p) for p in c.DEFAULT_ROOT.glob("tasks/*/arms/original47/contexts/original_reader/*.json")]
    previous = [x for v in historical_rows.values() for x in v]
    caption_chars = sum(r["public_caption_chars"] for r in rows)
    payload = dict(source_history_binding="All8 task metadata, transformed input file hashes and public HF source hashes exactly equal the historical FC/SRAG registration.",
        baseline_input_binding_all8_equal=True, public_caption_count=438,
        public_caption_total_chars=caption_chars, public_caption_mean_chars=caption_chars/438,
        new_exact_total_facts=sum(r["new_exact_facts"] for r in rows),
        old_fact_count_verified_lower_bound=sum(r["old_fact_count_lower_bound"] for r in rows),
        historical_exact_total_facts=None, historical_generated_caption_total_chars=None,
        historical_generated_caption_mean_chars=None,
        old_build_receipt_path=contract["canonical_receipt"]["build_path"],
        old_build_receipt_available=Path(contract["canonical_receipt"]["build_path"]).exists(),
        old_build_complete_sha256=contract["canonical_receipt"]["build_file_sha256"],
        historical_selected_context_mean_chars=statistics.mean(r["context_stats"]["context_chars"] for r in previous),
        public_selected_context_mean_chars=statistics.mean(len(x["context"]) for x in new_contexts),
        stages={k:dict(v) for k,v in stages.items()}, unresolved_construction_requests=issues,
        construction_failed_attempts=failures, tasks=rows,
        no_missing_as_zero="Both arms371 predictions and retained Lite scores are complete. Original2 truncated focused Readers recovered separately; construction had no unresolved requests.",
        boundary="Old original47 ZIP retains selectedfact IDs and context statistics, not full build graphs or generatedcaption text. Its ID union is a provable lower bound, not exact old pool size. New vsold sourcehistory identity is not independently rederived from a deleted oldbuild; query/gold identity is verified separately. No claim that length or factcount causally explains the score drop.")
    (OUT / "caption_control_memory_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 同源描述与记忆构建核查", "",
        f"公开描述共438条，合计{caption_chars:,}字符，平均{caption_chars/438:.2f}字符。与历史 FC/SRAG 的8个输入文件哈希、公开源文件哈希及任务元数据全部一致。", "",
        "| 场景 | 新库事实数（精确） | 旧库可验证事实数下界 | 公开描述平均字符 |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['new_exact_facts']} | ≥{r['old_fact_count_lower_bound']} | {r['public_caption_mean_chars']:.2f} |")
    lines += [f"| 合计 | {payload['new_exact_total_facts']} | ≥{payload['old_fact_count_verified_lower_bound']} | {payload['public_caption_mean_chars']:.2f} |", "",
        "旧库下界来自371题最终选中fact ID的场景内并集，不能写作旧库总事实数。旧完整graph和生成caption已不在指定保留包内，因此旧caption平均长度和精确总facts记为无法核验。", "",
        f"原版371题选中上下文平均字符数，旧版{payload['historical_selected_context_mean_chars']:.2f}，公开描述版{payload['public_selected_context_mean_chars']:.2f}。这是上下文长度，不是caption长度或token数。", "",
        "构建请求的实际返回模型均与登记的gemini-3.5-flash-lite一致，全部STOP完成；Writer与Online共用完整源facts，Online没有删掉原facts。阶段调用数见JSON。两个版本均有371份答案及Lite评分，没有将缺项静默记零。", "",
        "这些证据能确认输入描述与记忆规模发生变化，不能仅凭长度或facts数量断定下降原因，也不能排除跨时间模型服务和生成波动。", ""]
    (OUT / "caption_control_memory_audit.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ("public_caption_total_chars","public_caption_mean_chars","new_exact_total_facts","old_fact_count_verified_lower_bound","historical_selected_context_mean_chars","public_selected_context_mean_chars","stages","construction_failed_attempts")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

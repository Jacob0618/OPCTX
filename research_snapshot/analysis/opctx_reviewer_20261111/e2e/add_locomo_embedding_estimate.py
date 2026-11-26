"""Offline derived cost supplement; preserves the generation-only report."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
REPORT = HERE / "locomo_final_cost_20261111"
DIAGNOSTIC = Path("D:/opctx_e2e_cost_20261111/embedding_sample_locomo_r1")
PRICE_PER_M = 0.15


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    base = read(REPORT / "cost_summary.json")
    diag = read(DIAGNOSTIC / "summary.json")
    if diag["status"] != "complete" or diag["diagnostics_successful"] != 65:
        raise RuntimeError("Expected completed fixed sample")
    rows = [read(p) for p in (DIAGNOSTIC / "calls").glob("*/outcome.json")]
    if any(r["status"] != "ok" for r in rows):
        raise RuntimeError("Incomplete sampled counts")
    factors = {"write_embedding": 272/19, "online_embedding": 262/18, "query_embedding": 1540/152}

    def estimate(factor):
        total, variance = 0.0, 0.0
        for label, size in diag["census"]["strata"].items():
            group = [r for r in rows if r["stratum"] == label]
            assert len(group) == size["sample_n"]
            values = [r["tokens"] * sum(n * factor.get(s, 0) for s, n in r["successful_stages"].items()) for r in group]
            N, n = size["population_n"], len(group)
            total += N * statistics.mean(values)
            if n > 1:
                variance += N*N * (1-n/N) * statistics.variance(values)/n
        margin = 1.96 * math.sqrt(variance)
        return dict(estimated_tokens=total, sampling_interval95=[max(0, total-margin), total+margin],
                    estimated_usd=total*PRICE_PER_M/1e6,
                    sampling_cost_interval95=[max(0,total-margin)*PRICE_PER_M/1e6,(total+margin)*PRICE_PER_M/1e6])

    sample = estimate({s:1 for s in factors})
    full = estimate(factors)
    sample_build = estimate({"write_embedding":1,"online_embedding":1})
    full_build = estimate({k:v for k,v in factors.items() if k != "query_embedding"})
    gen_full_build = sum(base["stages"][s]["measured_usage_cache_usd"] * base["stages"][s]["full_population_factor"] for s in ("writer","online"))
    combined = dict(
        sample_generation_measured_usd=base["generation_only_measured_usage_cache_usd"],
        sample_embedding_estimated_usd=sample["estimated_usd"],
        sample_total_measured_generation_plus_estimated_embedding_usd=base["generation_only_measured_usage_cache_usd"]+sample["estimated_usd"],
        sample_total_standard_input_price_usd=base["generation_only_measured_standard_usd"]+sample["estimated_usd"],
        sample_total_per_question_usd=(base["generation_only_measured_usage_cache_usd"]+sample["estimated_usd"])/152,
        sample_memory_build_usd=base["generation_only_measured_memory_build_usd"]+sample_build["estimated_usd"],
        sample_memory_build_per_batch_usd=(base["generation_only_measured_memory_build_usd"]+sample_build["estimated_usd"])/19,
        full_generation_projected_usd=base["generation_only_projected_full_usage_cache_usd"],
        full_embedding_projected_usd=full["estimated_usd"],
        full_total_projected_usd=base["generation_only_projected_full_usage_cache_usd"]+full["estimated_usd"],
        full_total_standard_input_price_usd=base["generation_only_projected_full_standard_usd"]+full["estimated_usd"],
        full_total_per_question_usd=(base["generation_only_projected_full_usage_cache_usd"]+full["estimated_usd"])/1540,
        full_memory_build_projected_usd=gen_full_build+full_build["estimated_usd"],
        full_memory_build_per_batch_usd=(gen_full_build+full_build["estimated_usd"])/272)
    stage_data = {}
    for stage, factor in factors.items():
        stage_data[stage] = dict(sample=estimate({stage:1}), full_projection=estimate({stage:factor}), factor=factor)
    result = dict(combined=combined, embedding_sample=sample, embedding_full_projection=full,
        embedding_stages=stage_data, embedding_standard_usd_per_million_input_tokens=PRICE_PER_M,
        sampled_over_limit_contents=diag["sampled_over_limit_contents"], unsampled_over_limit_status=diag["unsampled_over_limit_status"],
        diagnostic_calls_excluded_from_pipeline=65, diagnostic_seconds_excluded_from_pipeline=diag["diagnostic_wall_seconds"],
        pipeline_execution_window_seconds_unchanged=base["active_execution_window_total_seconds"],
        notes=["Generation-only measured columns are unchanged in cost_summary.json.",
               "Embedding is estimated native input content tokens, not provider usage or invoice.",
               "Intervals quantify stratified sampling uncertainty only, not full-workload projection uncertainty.",
               "No sampled item exceeded 2048 tokens; unsampled over-limit status remains unknown.",
               "HTTP503 billing remains unknown; local pre-send failure is counted as a network attempt, not successful model use.",
               "The 65 diagnostic calls and their 9.904s are excluded from end-to-end pipeline totals."])
    with (REPORT / "cost_with_embedding_estimate.json").open("x", encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    lines = ["# LoCoMo 成本补表（加入独立 embedding 估计）", "",
        "生成费用来自实际 usage；embedding 费用来自65条固定分层样本的同模型原生 countTokens，不是实际账单 usage。原纯生成报告保留不变。", "",
        "| 范围 | 生成费用（实测/外推） | Embedding费用（估计） | 两者合计 | 每题摊销 | 每 memory batch 建库成本 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 19 batches / 152题样本 | ${combined['sample_generation_measured_usd']:.6f} | ${combined['sample_embedding_estimated_usd']:.6f} | **${combined['sample_total_measured_generation_plus_estimated_embedding_usd']:.6f}** | **${combined['sample_total_per_question_usd']:.6f}** | ${combined['sample_memory_build_per_batch_usd']:.6f} |",
        f"| 272 batches / 1540题外推 | ${combined['full_generation_projected_usd']:.6f} | ${combined['full_embedding_projected_usd']:.6f} | **${combined['full_total_projected_usd']:.6f}** | **${combined['full_total_per_question_usd']:.6f}** | ${combined['full_memory_build_per_batch_usd']:.6f} |", "",
        "| Embedding阶段 | 样本估计输入tokens | 全量缩放因子 | 全量估计输入tokens |", "|---|---:|---:|---:|"]
    for s,v in stage_data.items():
        lines.append(f"| {s} | {v['sample']['estimated_tokens']:.1f} | {v['factor']:.6f} | {v['full_projection']['estimated_tokens']:.1f} |")
    lines += ["", f"样本 embedding 估计 {sample['estimated_tokens']:.1f} tokens，近似95%抽样区间 [{sample['sampling_interval95'][0]:.1f}, {sample['sampling_interval95'][1]:.1f}]；费用 ${sample['estimated_usd']:.6f}。全量分阶段外推 {full['estimated_tokens']:.1f} tokens，费用 ${full['estimated_usd']:.6f}。",
        "", f"生成输入全部按非缓存标准价格计算时，样本合计 ${combined['sample_total_standard_input_price_usd']:.6f}，全量外推 ${combined['full_total_standard_input_price_usd']:.6f}。",
        "", "65/65诊断成功，抽样中0条超过2048-token限制；未抽中的条目仍不宣称已核实。诊断调用和9.904秒不加入pipeline，原执行窗口仍为511.647秒。",
        "", "生成包括Writer、Online、原Gemini reranker、Planner、QA。Reader为确定性组织。原81.75配置没有数学Selector或LLM query projection。HTTP503无usage的账单仍未知；上述合计为计价估计，非用户发票。",
        "", f"原生计数来源：{DIAGNOSTIC / 'summary.json'}"]
    with (REPORT / "cost_with_embedding_estimate.md").open("x",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

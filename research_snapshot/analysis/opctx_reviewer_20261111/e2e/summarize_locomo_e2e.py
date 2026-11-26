"""Offline summary of a completed, possibly resumed, cold LoCoMo profile."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    last = read(a.run / "summary.json")
    if last["status"] != "complete" or last["completed"] != 152:
        raise RuntimeError("No full-run extrapolation before all 152 queries complete")
    logical = read(a.run / "logical_run_summary.json")
    roots = [Path(x) for x in logical["source_runs"]] + [a.run]
    calls = [read(p) for root in roots for p in (root / "calls").glob("*/outcome.json")]
    questions = [read(p) for p in (a.run / "questions").glob("*.json")]
    rate = read(Path(__file__).parent.parent / "cost_probe/pricing_20261111.json")["models"]["gemini-2.5-flash"]
    stages = {}
    generation_names = {c["stage"] for c in calls if c["kind"] == "generation"}
    for stage in sorted({c["stage"] for c in calls}):
        rows = [c for c in calls if c["stage"] == stage]
        usage = [c["usage"] for c in rows if c.get("usage") is not None]
        input_n = sum(u.get("promptTokenCount", 0) for u in usage)
        output_n = sum(u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0) for u in usage)
        cache_n = sum(u.get("cachedContentTokenCount", 0) for u in usage)
        gen = stage in generation_names
        factor = 272 / 19 if stage in ("writer", "write_embedding") else 262 / 18 if stage in ("online", "online_embedding") else 1540 / 152
        stages[stage] = dict(kind=rows[0]["kind"], network_attempts=len(rows), successful_responses=sum(r["status"] == "ok" for r in rows),
            presend_failures=sum(r["status"] == "explicit_provider_error" and r.get("http_status") is None for r in rows),
            failed_http=sum(r.get("http_status") not in (None, 200) for r in rows),
            calls_without_usage=sum(r.get("usage") is None for r in rows),
            measured_input_tokens=input_n if gen else None, measured_output_tokens_including_thought=output_n if gen else None,
            measured_cached_input_tokens=cache_n if gen else None,
            measured_standard_usd=(input_n * rate["input"] + output_n * rate["output_including_thought"]) / 1e6 if gen else None,
            measured_usage_cache_usd=((input_n-cache_n) * rate["input"] + cache_n * rate["cached_input"] + output_n * rate["output_including_thought"]) / 1e6 if gen else None,
            cumulative_request_service_seconds=sum(r["service_seconds"] for r in rows),
            embedding_items=sum(r.get("input_items", 0) for r in rows),
            full_population_factor=factor)
    measured_cost = sum(stages[s]["measured_usage_cache_usd"] for s in generation_names)
    standard_cost = sum(stages[s]["measured_standard_usd"] for s in generation_names)
    projected_cost = sum(stages[s]["measured_usage_cache_usd"] * stages[s]["full_population_factor"] for s in generation_names)
    projected_standard = sum(stages[s]["measured_standard_usd"] * stages[s]["full_population_factor"] for s in generation_names)
    build_cost = sum(stages[s]["measured_usage_cache_usd"] for s in ("writer", "online"))
    query_cost = measured_cost - build_cost
    prior_windows = sum(read(root / "summary.json")["wall_clock_seconds"] for root in roots[:-1])
    phases = last["phases"]
    writer_active_windows = prior_windows + phases["writer_and_initial_embedding_wall_seconds"]
    online_wall = phases["online_with_embedding_wall_seconds"]
    query_wall = phases["all_queries_parallel_wall_seconds"] + phases["query_embedding_wall_seconds"]
    estimated_full_wall = writer_active_windows / 19 * 272 + online_wall / 18 * 262 + query_wall / 152 * 1540 + phases["retrieval_index_build_wall_seconds"] * 10
    local = {name: sum(q["local_component_timings"].get(name, 0) for q in questions) for name in ("_candidate_pool", "_rerank", "_assemble")}
    local["reader_materialization"] = sum(q["reader_local_wall_seconds"] for q in questions)
    stage_question_wall = {key: sum(q[key] for q in questions) for key in (
        "retrieval_including_rerank_wall_seconds", "planner_wall_seconds", "reader_local_wall_seconds", "qa_wall_seconds", "query_wall_seconds")}
    data = dict(sample=dict(conversation="conv-26", sessions=19, online_updates=18, questions=152), stages=stages,
        generation_only_measured_standard_usd=standard_cost, generation_only_measured_usage_cache_usd=measured_cost,
        generation_only_measured_per_query_usd=measured_cost/152,
        generation_only_measured_memory_build_usd=build_cost,
        generation_only_measured_build_per_memory_batch_usd=build_cost/19,
        generation_only_measured_query_side_usd=query_cost,
        generation_only_projected_full_standard_usd=projected_standard,
        generation_only_projected_full_usage_cache_usd=projected_cost,
        generation_only_projected_full_amortized_per_query_usd=projected_cost/1540,
        execution_windows_seconds=logical["active_execution_windows_seconds"],
        active_execution_window_total_seconds=logical["total_active_execution_window_seconds"],
        writer_and_embedding_active_execution_windows_seconds=writer_active_windows,
        online_wall_seconds=online_wall, query_phase_parallel4_wall_seconds=query_wall,
        deterministic_local_component_seconds=local,
        summed_per_question_stage_wall_seconds=stage_question_wall,
        projected_full_serial_conversations_within_conversation_parallel4_wall_seconds=estimated_full_wall,
        reranker_fallback_questions=sum(bool(q["retrieval_trace"]["rerank"].get("fallback")) for q in questions),
        planner_neutral_failure_questions=sum(q["planner"].get("source") == "neutral_failure" for q in questions),
        boundaries=["Generation cost uses real known usage and official standard list rates; not an invoice.",
                    "Embedding billed tokens and HTTP-error billing remain unknown, not zero; native countTokens is separate.",
                    "Execution-window total excludes human recovery pause; not uninterrupted wall clock.",
                    "No mathematical Selector or LLM query projection in this historical 81.75 configuration.",
                    "Full projection is one-conversation estimate, with 272 sessions / 262 Online updates / 1540 questions scaled separately."])
    a.out_dir.mkdir(parents=True, exist_ok=False)
    with (a.out_dir / "cost_summary.json").open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    lines = ["# LoCoMo 冷构建端到端成本实测", "", "固定 conv-26，19 个完整 session、18 次 Online 更新、152 题。按原 81.75% 配置重新构建，不复用历史实验缓存；本次中断后只接续自己的成功输出。", "",
        "| 阶段 | 网络尝试 | 成功响应 | 已测输入 token | 已测输出含 thinking | 累计请求服务秒 |", "|---|---:|---:|---:|---:|---:|"]
    for stage, v in stages.items():
        lines.append(f"| {stage} | {v['network_attempts']} | {v['successful_responses']} | {v['measured_input_tokens'] if v['measured_input_tokens'] is not None else 'NA'} | {v['measured_output_tokens_including_thought'] if v['measured_output_tokens_including_thought'] is not None else 'NA'} | {v['cumulative_request_service_seconds']:.3f} |")
    lines += ["", f"已知生成费用（按 usage 的缓存折扣）为 **${measured_cost:.6f}**，每题摊销 **${measured_cost/152:.6f}**；同 token 全部按标准非缓存输入价为 ${standard_cost:.6f}。",
        f"其中建库生成费用 ${build_cost:.6f}，平均每 memory batch ${build_cost/19:.6f}；查询侧生成费用 ${query_cost:.6f}。",
        f"按照全量真实 session/frontier/题数分别外推，生成费用约 **${projected_cost:.6f} / 1540 题**，每题 ${projected_cost/1540:.6f}（全标准输入价 ${projected_standard:.6f}）。",
        "", "以上费用尚不含未知的 embedding 账单 token 及无 usage 的 HTTP 错误账单，不能称为完整已核实账单。", "",
        f"实际执行窗口共 {logical['total_active_execution_window_seconds']:.3f} 秒（不含人工恢复间隔）。查询阶段并行 4 为 {query_wall:.3f} 秒；Reader 确定性组织累计 {local['reader_materialization']:.3f} 秒，无模型调用。",
        f"按同一并行策略、conversation 顺序执行粗略外推全量墙钟 {estimated_full_wall/60:.2f} 分钟；单 conversation 不足以保证全量时间。", "",
        "原配置没有数学 Selector，也没有 LLM query projection；不能虚构它们的成本。reranker 是原 Gemini 2.5 Flash，不是 Qwen。", "",
        "来源：" + ", ".join(str(x) for x in roots)]
    with (a.out_dir / "cost_summary.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

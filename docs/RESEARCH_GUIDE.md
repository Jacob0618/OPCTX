# 代码导航与后续实验

## 1. 两层代码分别做什么

根目录是之前整理的 OpCtx 精简实现，适合直接阅读、修改和开展新实验。
`research_snapshot/` 保留原工程的核心模块、关键实验入口及其本地源代码依赖；
`source_manifest.json` 记录来源相对路径和 SHA-256。

精简版使用短 Writer prompt 和统一输入接口；不能直接把它的配置或输出当成
历史论文结果。历史入口保留原来的模型、路径和版本检查，需要自行准备数据与
相应外部依赖，再用新目录进行复现。

## 2. 改某个组件从哪里开始

| 要改的部分 | 可运行精简版 | 原始研究实现 |
|---|---|---|
| Writer 与事实粒度 | `memory.py`, `prompts.py` | `creamem3/graph_memory_v2.py`, `prompts_v2.py`, LME writer 模块 |
| Online 事件关联 | `memory.py:update_views` | `creamem3/online_memory_v3.py`, `online_memory_v4_streaming.py` 与冻结版本 |
| 混合召回 | `retrieval.py:hybrid_candidates` | graph/online memory 的候选生成与对应实验入口 |
| 数学 Selector | `selector.py`, `numerical.py`, `selection_utils.py` | `creamem3/experiments/iterative_memory_20260907/` |
| 查询投影 | `retrieval.py:project_question`, `prompts.py` | `query_projections.py` |
| Reader 与 QA | `opctx.py`, `prompts.py` | `focused_answer_reader_v11.py`, `online_operation_context_v8.py`, `semantic_context_v8.py` |
| 基准输入适配 | `examples/toy_history.json` 的格式 | `memeye_adapter_v9.py`, LME runners，基准相关入口 |

右栏路径均相对于 `research_snapshot/`；完整清单以 manifest 为准。

## 3. 重要实验入口

- LoCoMo：`creamem3/experiments/locomo_math20_10_20260917/` 与
  `locomo_cumulative20_20260917/`，保留检索预算和组件对照的实现。
- MemEye：`memeye_lite_ablation_20260923/`、`review_followup_20260926/`、
  `review_main_align_20260926/`，保留同预算选择器、Reader 和主表对齐的实验逻辑。
- LME：`lme_opctx_v1.py`、`lme_opctx_v2.py`、
  `lme_opctx_v2_b3_scaled_facts.py` 以及相应 runner，区分原始与三 session
  facts 数量缩放的版本。
- 审稿补充：`analysis/opctx_reviewer_20261111/` 的同源 caption、原始评分
  及端到端成本实现；`analysis/opctx_random_links_20261113/` 与
  `opctx_random_links_20261114/` 的乱连边对照。
- 强 RAG：`analysis/hybrid_rag_20261114/` 是额外 baseline，不是 OpCtx。
  LoCoMo 是 Dense40 + BM25最多40 → RRF40 → Gemini20；LME 是
  Dense240 + BM25最多240 → RRF240 → Qwen30。不要与根目录 LME 精简配置混用。

某些实验入口通过导入复用更早实验的函数，因此快照包含必要的早期依赖；这不
表示所有被包含的版本都是当前主方法。

## 4. 需要保留的定义

- facts 是有来源的 Writer 输出；event views 是引用这些 facts 的事件组织层。
- summary/view 的 Top-k 是路由数量，展开后可能得到很多 facts，不是总候选数。
- MemEye 数学选择器的最终条数与字符预算是两个约束。精简版字符预算取全部
  可见 Writer facts 中 Dense50 的渲染长度，最终最多选30条。
- MemEye 精简 Reader 不直接接收 event-view 正文；Online 可经路由改变候选。
- 文本与多模态 Reader 的接口实现不同，以具体版本代码为准。
- 固定候选、相同所选 fact IDs、相同文本预算和相同模型是不同的控制条件，
  做消融时需要逐项写清楚。

## 5. 首次运行与再发布

先运行根目录离线测试。真实数据按原数据集许可自行下载；旧工程动态引用的
外部适配器见 `THIRD_PARTY_NOTICES.md` 和 manifest 的 unresolved dependencies。
API key 通过环境或隐藏输入提供，不放进配置文件。模型当前是否可用由你的
API 账户决定，不自动替换模型。

本次没有上传论文源稿、数据集、逐题标准答案、检索结果、运行缓存和私人对话。
它们不妨碍使用根目录核心代码；如需重现特定表格，先确认对应原实验的完整
数据/模型/提示词/检索/评分配置。

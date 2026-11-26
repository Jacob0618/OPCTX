# Research source snapshot

`research_snapshot/` preserves original research sources for method inspection
and future experiment development. Its paths mirror the research workspace;
source bytes are unchanged. `source_manifest.json` records every source path,
destination, SHA-256, size, and whether the file was explicitly selected or
included as an imported dependency.

The root-level implementation is a smaller, portable implementation. It is
not a byte-identical recreation of the historical paper experiments. The
research snapshot is likewise not a self-contained benchmark reproduction
bundle: original runners verify external prompt authorities, datasets, old
artifact manifests, and source hashes. Configure these explicitly before use.

## Navigation

| Area | Original source paths under `research_snapshot/` |
|---|---|
| Writer, provenance and hybrid retrieval | `creamem3/graph_memory_v2.py`, `creamem3/prompts_v2.py` |
| Causal event organization | `creamem3/online_memory_v3.py`, `online_memory_v4.py`, `online_memory_v4_streaming.py` |
| Query planning and operation-conditioned context | `creamem3/semantic_query_planner_v6.py`, `strict_semantic_planner_v8.py`, `semantic_context_v8.py`, `online_operation_context_v8.py` |
| MemEye adaptation and Reader | `creamem3/memeye_adapter_v9.py`, `focused_answer_reader_v11.py`, `experiments/memeye_lite_ablation_20260923/` |
| Mathematical Selector | `creamem3/experiments/iterative_memory_20260907/multichannel_selector_v2.py`, `numerical_solver_v2.py`, `set_selectors.py`, `query_projections.py` |
| LME three-session/scaled Writer and evaluation | `creamem3/lme_opctx_v1.py`, `lme_opctx_v2.py`, `lme_opctx_v2_b3_scaled_facts.py`, `run_lme_clean100_v1.py`, `experiments/lme_remaining200_20260923/` |
| LoCoMo selected-budget ablations | `creamem3/experiments/locomo_math20_10_20260917/`, `locomo_cumulative20_20260917/` |
| Selector controls, solver diagnostics and paired analysis | `creamem3/experiments/review_followup_20260926/`, `review_main_align_20260926/` |
| Same-caption comparison and cost measurement | `analysis/opctx_reviewer_20261111/`, including `e2e/` |
| Randomized event-to-fact links | `analysis/opctx_random_links_20261113/`, `opctx_random_links_20261114/` |
| Stronger Hybrid RAG comparison | `analysis/hybrid_rag_20261114/` |

Several older experiment modules are retained only because selected entry
points import them. Their inclusion does not make them the default method or
endorse all historical variants. Runtime imports do not guarantee that all
external artifact dependencies are available.

## Historical LoCoMo source identity

The historical 81.75 configuration uses the graph snapshot
`creamem3/frozen_sources/locomo81_graph_memory_v2.py`. Its Online source is
preserved here as `creamem3/frozen_sources/locomo81_online_memory_v3.py`, with
SHA-256 `3af38e30881fc6ce64d393c388c4594e1a9adbecb196136bbc306659f14f72b9`.
The latter is a relocation of an original source snapshot, not a rewritten
implementation. The manifest records its original relative path. Historical
measurement scripts still refer to that original path, so explicitly map it
when adapting those scripts; do not replace it silently with current Online.

## External requirements and attribution

- Original LME prompt/client authorities are dynamically loaded from the
  separate CreaMem-AutoResearch project under
  `release/CreaMem2/benchmarks/longmemeval/`.
- Original MemEye visual writing and Reader authorities are dynamically loaded
  from that project's `multimodal/` directory, including
  `eval_memeye.py`, `vendor_mem0_prompts_official.py`, and
  `memeye_creamem_reader.py`. The adapter identifies the reused MGTM/Mem0
  prompt semantics. These external source files are not bundled here.
- Some authority helpers require the separately obtained MemEye benchmark
  implementation. Its top-level source carries Apache-2.0, while vendored
  baseline implementations retain their own licenses. No vendor tree is
  included or relicensed by this snapshot.
- Obtain LoCoMo, LongMemEval-S, and MemEye data/images separately. No datasets,
  generated memories, cached model responses, predictions, logs, or credentials
  are included in the snapshot.

The manifest lists external import names, including standard-library names;
it is an inventory, not an automatically installable dependency specification.
The snapshot preserves original local-path defaults and intentional immutable
hash checks. Adjust experiment configuration in a new version for new runs,
rather than making an edited source pretend to satisfy an old frozen hash.

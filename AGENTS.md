# Working on OpCtx

- Read README.md and docs/RESEARCH_GUIDE.md before changing experiment behavior.
- Root modules are the editable standalone implementation. research_snapshot/
  preserves selected original source versions; create new experiment modules
  instead of silently rewriting archived configurations.
- Keep API keys in environment variables. Do not inspect or expose private
  credentials, commit .env, or log authorization headers or full requests.
- Do not upload datasets, benchmark answers, model responses, memory stores,
  private review correspondence, caches or paper drafts by default.
- Offline checks: `python -m unittest discover -s tests -v` and
  `python tools/check_release.py`. Use mock clients; no live calls are necessary.
- Live experiment changes require an explicit dataset/model/prompt/retrieval/
  scoring configuration. Preserve existing successful outputs and failure
  evidence. Never choose reruns by accuracy or silently assign missing results 0.
- Keep question/gold separation: source histories enter writing; the question
  alone enters projection; references enter evaluation only.
- Structural event validation does not establish semantic event correctness.
  Selector optimality applies to its finite objective and candidate pool only.
- Do not infer historical paper scores from the standalone defaults. Record
  exact source/config versions for new comparisons.

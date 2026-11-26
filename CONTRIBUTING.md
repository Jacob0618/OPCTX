# Continuing experiments

1. Create a branch for a specific method or experiment change.
2. Put a complete configuration alongside the new runner: models, stage token
   limits, writing batch size, retrieval branches/caps, Selector constraints,
   reranker, Reader mode, QA prompt and evaluation protocol.
3. Run offline tests before any model calls. Use synthetic histories for tests.
4. Keep datasets and paid-response caches outside Git. Record source/config
   hashes and per-stage usage in your local run directory.
5. Compare systems on the same question IDs. Separate genuine model scores,
   missing outputs and explicitly authorized failure-zero policies.
6. Record changes in a new configuration rather than relabeling historical
   results. Distinguish measured cost from estimates and inference from Judge.
7. Run `python tools/check_release.py` before committing. Review `git diff
   --cached` and never use a blanket add of external cache/data directories.

Archived scripts are reference implementations with their own runtime contracts.
They may require local dataset adapters or older source bundles not included
here. Port them deliberately; do not remove a hash or provenance check just
to make an old run appear reproducible.

# OpCtx

Source-linked memory construction and budgeted evidence reading for long-running
language and multimodal agents.

This repository has two parts:

- **Standalone core** in the repository root: a compact, editable implementation
  of Writer, Online event views, hybrid retrieval, selection, Reader and QA.
- **Research source snapshots** in `research_snapshot/`: original core modules
  and selected benchmark/ablation/reviewer-experiment entry points, with source
  hashes recorded in `docs/source_manifest.json`.

The standalone core is a starting point for new experiments. Its shorter Writer
prompt and simplified adapters are not an exact replay of every paper run.
Use the version map in [the research guide](docs/RESEARCH_GUIDE.md) when extending
an existing experiment.

## Quick start

Use Python **3.11 or 3.12**.

```bash
python -m pip install -r requirements.txt
python main.py --help
python main.py build --dataset memeye --input examples/toy_history.json --output memory.json
python main.py ask --memory memory.json --question "Where is the notebook now?"
```

The CLI prompts for a Gemini API key without displaying or saving it. You may
instead provide `GEMINI_API_KEY` through your environment. Live build/ask commands
make paid API requests. No credentials are shipped with this repository.

The example history is synthetic, not benchmark data. Use one memory file per
independent history. Existing output files are never silently overwritten.

## Core modules

| File | Purpose |
|---|---|
| `config.py` | Model names, retrieval counts, output limits and solver settings |
| `prompts.py` | Writer, Online, projection, Reader, rerank and QA prompts |
| `memory.py` | Source-linked facts and incremental create/extend event views |
| `retrieval.py` | Summary/view expansion, dense + BM25, selection and reranking |
| `selector.py` | Literal coverage and normalized multichannel max-min utility |
| `numerical.py`, `selection_utils.py` | Solver validation and selection utilities |
| `opctx.py` | Memory construction and question-conditioned evidence reading |
| `client.py` | Gemini REST transport and normalized embeddings |
| `main.py` | Build/ask command-line interface |

### Standalone profiles

These preserve the earlier compact-code profiles, rather than replacing them
with settings from a different benchmark run.

| Profile | Generation model | Summary / dense / BM25 | Selection |
|---|---|---|---|
| LoCoMo | Gemini 2.5 Flash | 10 / 20 / 20; candidate cap 120 | Gemini rerank to 20 |
| LME | Gemini 2.5 Flash | 20 / 40 / 40; candidate cap 240 | Gemini rerank to 20 |
| MemEye | Gemini 2.5 Flash-Lite | 10 / 20 / 20 | Mathematical Selector, at most 30; no model reranker |

Summary/view routes expand to their member facts: the MemEye union is **not**
limited to 50 facts. Its character budget is derived from dense top-50 over
all visible Writer facts; the final fact-count bound remains 30.

LoCoMo/LME use 768-dimensional embeddings and MemEye uses 3072-dimensional
embeddings. LME writes three sessions per batch; the other profiles write one.
Generation uses temperature 0 and thinking budget 0. Available model names and
quotas depend on your provider account; change profiles explicitly if needed.

## Input format

Supply a JSON object with a chronological `sessions` array:

```json
{
  "sessions": [{
    "id": "session_1",
    "timestamp": "2026-01-01",
    "turns": [{
      "id": "turn_1",
      "speaker": "user",
      "text": "An observation to remember."
    }]
  }]
}
```

For MemEye, a turn may additionally have `caption` or `image`. An existing
caption takes precedence; otherwise the image is described before writing.
Image paths are relative to the input JSON. Text-only profiles reject image
and caption fields. Do not place gold answers or scoring labels in histories.

## Python API

```python
from opctx import OpCtx

app = OpCtx("memeye")
app.write(sessions)
result = app.answer("Your question")
print(result["answer"])
print(result["selected_fact_ids"])
```

## Offline checks

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

Tests use synthetic inputs and mock clients: no API key or model call is needed.
SciPy/HiGHS is used for the Selector. A solver certificate concerns the finite
retrieval objective, not semantic answer correctness.

## Continuing the research

- [Research guide and implementation boundaries](docs/RESEARCH_GUIDE.md)
- [Contribution workflow](CONTRIBUTING.md)
- [Source inventory](docs/source_manifest.json)
- [Third-party/dependency notes](THIRD_PARTY_NOTICES.md)

Datasets, API responses, memory stores, experiment caches, conversation logs
and private paper drafts are intentionally excluded. This initial upload does
not assign a new license to original or third-party research code; preserve
existing attribution and check upstream licenses before redistribution.

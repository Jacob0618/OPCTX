#!/usr/bin/env python3
"""Evaluate strict semantic event budgets on all 1,540 LoCoMo questions.

The planner artifact is generated once from question text alone.  Each memory
system then deterministically materializes exactly one reader context per QA.
No answer, LoCoMo category, evidence annotation, previous response, or verdict
is visible to either materializer.  Exact historical reader/judge rows may be
reused only when the question and rendered context are byte-identical.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These must be set before importing the frozen runner and Gemini client.
os.environ.setdefault(
    "AGENT_MEMORY_CACHE_ROOT", r"D:\creamem3_cache\v8\semantic_context"
)
os.environ["BACKBONE_MODEL"] = "gemini-2.5-flash"
os.environ["JUDGE_MODEL"] = "gemini-3-flash-preview"
os.environ["BACKBONE_THINKING"] = "off"
os.environ["JUDGE_THINKING"] = "off"
os.environ["GEMINI3_THINKING_LEVEL"] = "minimal"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import creamem3.run_conv26_v2 as frozen_runner  # noqa: E402
from creamem3.graph_memory_v2 import canonical_sha256  # noqa: E402
from creamem3.locomo_all import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    file_sha256,
    load_all_locomo,
    load_locomo_conversation,
)
from creamem3.semantic_context_v8 import (  # noqa: E402
    materialize_offline_semantic_context,
    materialize_online_semantic_context,
)

EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from gemini_client import USAGE, pmap  # noqa: E402


SCHEMA_VERSION = "creamem3.semantic_context_full.v8.1"
AGGREGATE_SCHEMA_VERSION = "creamem3.semantic_context_full_aggregate.v8.1"
SYSTEMS = ("online", "offline")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results_autoresearch_v8" / "semantic_context_s01"
DEFAULT_CACHE_ROOT = Path(r"D:\creamem3_cache\v8\semantic_context_s01")
DEFAULT_PLAN_PATH = (
    SCRIPT_DIR / "results_autoresearch_v8" / "semantic_plans_s02" / "plans.json"
)
BASE_RESULTS_ROOT = SCRIPT_DIR / "results_locomo10_v3"
BASE_TAG = "full_v3_r1"
IMMUTABLE_QA_KEYS = (
    "qa_id",
    "source_index",
    "question",
    "question_occurrence",
    "answer",
    "category",
    "category_name",
    "evidence",
    "evidence_dia_ids",
    "evidence_repairs",
    "evidence_unresolved",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return value


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _usage_snapshot() -> dict[str, Any]:
    return json.loads(USAGE.snapshot())


def _numeric_delta(after: Any, before: Any) -> Any:
    if isinstance(after, Mapping) and isinstance(before, Mapping):
        return {
            key: _numeric_delta(after.get(key, 0), before.get(key, 0))
            for key in sorted(set(after) | set(before))
        }
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return after - before
    return after


def _source_snapshot() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "semantic_context": SCRIPT_DIR / "semantic_context_v8.py",
        "strict_planner": SCRIPT_DIR / "strict_semantic_planner_v8.py",
        "frozen_runner": SCRIPT_DIR / "run_conv26_v2.py",
        "qa_judge_prompts": SCRIPT_DIR / "prompts.py",
        "gemini_client": EVAL_DIR / "gemini_client.py",
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def _system_paths(sample_id: str, system: str) -> tuple[Path, Path]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    root = BASE_RESULTS_ROOT / sample_id
    return (
        (root / f"graph_{system}.json").resolve(),
        (root / f"{system}_{sample_id}_{BASE_TAG}.json").resolve(),
    )


def _baseline_path(sample_id: str, system: str) -> Path:
    return (
        BASE_RESULTS_ROOT
        / sample_id
        / f"{system}_{sample_id}_{BASE_TAG}.json"
    ).resolve()


def _child_path(output_root: Path, sample_id: str, system: str) -> Path:
    return output_root / sample_id / f"{system}_semantic_budget_v1.json"


def _historical_paths(sample_id: str, system: str) -> list[Path]:
    if system == "online":
        return [
            SCRIPT_DIR
            / "results_autoresearch_v4"
            / "v4_full_route_r1"
            / sample_id
            / "r01_route_only_m20.json",
            SCRIPT_DIR
            / "results_autoresearch_v4"
            / "v5_full_ps02_nontemporal_claimcap_r1"
            / sample_id
            / "ps02_nontemporal_claimcap_m20.json",
        ]
    return [
        SCRIPT_DIR
        / "results_autoresearch_v7_offline_full"
        / sample_id
        / "o07b_fresh_hide_event_views_exact.json"
    ]


def _validate_source_rows(
    loaded: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("source artifact lacks rows")
    by_id = {str(row.get("qa_id", "")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("source artifact has duplicate QA IDs")
    for source in loaded["qa"]:
        row = by_id.get(str(source["qa_id"]))
        if row is None:
            raise ValueError(f"source lacks {source['qa_id']}")
        if any(row.get(key) != source.get(key) for key in IMMUTABLE_QA_KEYS):
            raise ValueError(f"source QA mismatch: {source['qa_id']}")
        if row.get("correct") not in {0, 1} or not str(row.get("response", "")).strip():
            raise ValueError(f"invalid source result: {source['qa_id']}")
    return by_id


def _load_plan_index(
    plan_path: Path, loaded: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    artifact = _load_json(plan_path)
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 1540:
        raise ValueError("strict semantic plan artifact must contain 1540 rows")
    selected = [
        row for row in rows if str(row.get("sample_id")) == str(loaded["sample_id"])
    ]
    by_id = {str(row.get("qa_id", "")): row for row in selected}
    if len(by_id) != len(loaded["qa"]):
        raise ValueError(f"plan coverage mismatch: {loaded['sample_id']}")
    for source in loaded["qa"]:
        row = by_id.get(str(source["qa_id"]))
        if row is None or row.get("question_sha256") != _sha256_text(source["question"]):
            raise ValueError(f"plan question mismatch: {source['qa_id']}")
    return by_id, artifact


def _history_index(
    loaded: Mapping[str, Any], system: str
) -> tuple[dict[tuple[str, str], tuple[Mapping[str, Any], str]], list[dict[str, str]]]:
    index: dict[tuple[str, str], tuple[Mapping[str, Any], str]] = {}
    sources: list[dict[str, str]] = []
    for path in _historical_paths(str(loaded["sample_id"]), system):
        if not path.exists():
            continue
        payload = _load_json(path)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        for row in rows:
            qa_id = str(row.get("qa_id", ""))
            context = str(row.get("retrieved", ""))
            if (
                qa_id
                and context
                and row.get("correct") in {0, 1}
                and str(row.get("response", "")).strip()
                and not any(
                    row.get(key)
                    for key in (
                        "retrieval_error",
                        "qa_error",
                        "judge_parse_error",
                        "judge_error",
                        "judge_skipped",
                    )
                )
            ):
                index.setdefault((qa_id, _sha256_text(context)), (row, str(path.resolve())))
    return index, sources


def _reuse_row(
    source: Mapping[str, Any],
    historical: Mapping[str, Any],
    context: str,
    trace: Mapping[str, Any],
    reuse_source: str,
) -> dict[str, Any]:
    row = {key: copy.deepcopy(source[key]) for key in IMMUTABLE_QA_KEYS}
    row.update(
        {
            "sample_id": str(historical["sample_id"]),
            "conv_id": int(historical["conv_id"]),
            "retrieved": context,
            "retrieved_chars": len(context),
            "retrieval_trace": {"semantic_context_v8": copy.deepcopy(dict(trace))},
            "t_search": 0.0,
            "retrieval_error": None,
            "response": str(historical["response"]),
            "qa_error": None,
            "correct": int(historical["correct"]),
            "judge_raw": str(historical["judge_raw"]),
            "judge_parse_error": False,
            "judge_error": None,
            "judge_skipped": False,
            "reader_judge_reused": True,
            "reader_judge_reuse_source": reuse_source,
        }
    )
    return row


def _evaluate_system(
    *,
    loaded: Mapping[str, Any],
    system: str,
    graph: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    base_by_id = _validate_source_rows(loaded, source_payload)
    history, history_sources = _history_index(loaded, system)
    # The frozen source itself is an eligible exact-context reuse source.
    for qa_id, row in base_by_id.items():
        history.setdefault(
            (qa_id, _sha256_text(str(row["retrieved"]))),
            (row, "frozen_source"),
        )

    materializer: Callable[..., tuple[str, dict[str, Any]]]
    materializer = (
        materialize_online_semantic_context
        if system == "online"
        else materialize_offline_semantic_context
    )
    pending: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, Any]] = {}
    reuse_counts: dict[str, int] = {}
    for source in loaded["qa"]:
        qa_id = str(source["qa_id"])
        base = base_by_id[qa_id]
        sanitized_base = {
            "question": str(source["question"]),
            "retrieved": str(base["retrieved"]),
            "retrieval_trace": copy.deepcopy(base["retrieval_trace"]),
        }
        plan = copy.deepcopy(plans[qa_id]["result"])
        if system == "online":
            context, audit = materializer(
                {"question": str(source["question"])},
                sanitized_base,
                graph,
                plan,
            )
        else:
            context, audit = materializer(sanitized_base, graph, plan, strict=True)
        trace = {
            "schema_version": SCHEMA_VERSION,
            "system": system,
            "materializer_audit": audit,
            "strict_plan_row_sha256": canonical_sha256(plans[qa_id]),
            "context_sha256": _sha256_text(context),
            "final_fact_top_k_locked": 20,
            "gold_fields_visible_to_materializer": False,
            "category_visible_to_materializer": False,
            "prior_answers_or_verdicts_visible_to_materializer": False,
        }
        if audit.get("lexical_gate_used") is not False:
            raise ValueError(f"lexical-gate audit failed: {qa_id}")
        if audit.get("temporal_regex_path_called") is not False:
            raise ValueError(f"temporal-regex audit failed: {qa_id}")
        traces[qa_id] = trace
        match = history.get((qa_id, trace["context_sha256"]))
        if match is not None and str(match[0].get("retrieved", "")) == context:
            historical, reuse_source = match
            rows_by_id[qa_id] = _reuse_row(
                source, historical, context, trace, reuse_source
            )
            reuse_counts[reuse_source] = reuse_counts.get(reuse_source, 0) + 1
        else:
            pending.append(
                {
                    "qa": dict(source),
                    "retrieved": context,
                    "retrieval_trace": {"semantic_context_v8": trace},
                    "t_search": 0.0,
                    "retrieval_error": None,
                }
            )

    before = _usage_snapshot()
    started = time.time()
    answered = pmap(
        frozen_runner._answer_one,
        pending,
        workers=workers,
        desc=f"read-semantic-{system}",
    )
    judged = pmap(
        frozen_runner._judge_one_row,
        answered,
        workers=workers,
        desc=f"judge-semantic-{system}",
    )
    for row in judged:
        row["reader_judge_reused"] = False
        row["reader_judge_reuse_source"] = None
        rows_by_id[str(row["qa_id"])] = row
    after = _usage_snapshot()
    rows = [rows_by_id[str(source["qa_id"])] for source in loaded["qa"]]

    for row, source in zip(rows, loaded["qa"], strict=True):
        qa_id = str(source["qa_id"])
        if any(row.get(key) != source.get(key) for key in IMMUTABLE_QA_KEYS):
            raise ValueError(f"output QA mismatch: {qa_id}")
        if row.get("correct") not in {0, 1}:
            raise ValueError(f"non-binary verdict: {qa_id}")
        if any(
            row.get(key)
            for key in (
                "retrieval_error",
                "qa_error",
                "judge_parse_error",
                "judge_error",
                "judge_skipped",
            )
        ):
            raise ValueError(f"protocol failure: {qa_id}")
        expected = traces[qa_id]
        if row["retrieval_trace"]["semantic_context_v8"] != expected:
            raise ValueError(f"trace mismatch: {qa_id}")
        if _sha256_text(str(row["retrieved"])) != expected["context_sha256"]:
            raise ValueError(f"context hash mismatch: {qa_id}")

    summary = frozen_runner._summarize_rows(rows)
    budgets = {str(value): 0 for value in (0, 1, 2)}
    for row in rows:
        budget = str(
            row["retrieval_trace"]["semantic_context_v8"]
            ["materializer_audit"]["event_budget"]
        )
        budgets[budget] = budgets.get(budget, 0) + 1
    sizes = sorted(len(str(row["retrieved"])) for row in rows)
    summary.update(
        {
            "system": system,
            "event_budget_counts": budgets,
            "mean_retrieved_chars": sum(sizes) / len(sizes),
            "p95_retrieved_chars": sizes[math.ceil(0.95 * len(sizes)) - 1],
            "reader_judge_reused_rows": sum(bool(row["reader_judge_reused"]) for row in rows),
            "new_reader_judge_rows": len(pending),
            "reuse_counts": reuse_counts,
            "wall_s": time.time() - started,
        }
    )
    return rows, summary, _numeric_delta(after, before), history_sources


def run_child(
    *,
    dataset: Path,
    output_root: Path,
    plan_path: Path,
    sample_id: str,
    systems: Sequence[str],
    workers: int,
    force: bool,
) -> dict[str, Any]:
    loaded = load_locomo_conversation(dataset, sample_id)
    plan_by_id, plan_artifact = _load_plan_index(plan_path, loaded)
    frozen_runner.CURRENT_SAMPLE_ID = sample_id
    frozen_runner.CURRENT_SAMPLE_INDEX = int(loaded["sample_index"])
    snapshot = _source_snapshot()
    completed: dict[str, Any] = {}
    for system in systems:
        graph_path, source_path = _system_paths(sample_id, system)
        output_path = _child_path(output_root, sample_id, system)
        if output_path.exists() and not force:
            payload = _load_json(output_path)
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("sample_id") != sample_id
                or payload.get("system") != system
                or payload.get("source_snapshot") != snapshot
                or payload.get("semantic_plan_sha256") != file_sha256(plan_path)
                or len(payload.get("rows", [])) != len(loaded["qa"])
            ):
                raise ValueError(f"existing child binding mismatch: {sample_id}/{system}")
            completed[system] = payload["summary"]
            print(f"[child] {sample_id}/{system}: validated existing", flush=True)
            continue
        graph = _load_json(graph_path)
        source_payload = _load_json(source_path)
        source_hashes_before = {
            str(graph_path): file_sha256(graph_path),
            str(source_path): file_sha256(source_path),
            str(plan_path): file_sha256(plan_path),
        }
        rows, summary, usage, history_sources = _evaluate_system(
            loaded=loaded,
            system=system,
            graph=graph,
            source_payload=source_payload,
            plans=plan_by_id,
            workers=workers,
        )
        if {
            path: file_sha256(Path(path)) for path in source_hashes_before
        } != source_hashes_before:
            raise RuntimeError(f"frozen source changed: {sample_id}/{system}")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "scope": "locomo10_full_1540",
            "sample_id": sample_id,
            "sample_index": int(loaded["sample_index"]),
            "system": system,
            "qa_ids": [str(row["qa_id"]) for row in loaded["qa"]],
            "qa_selection_sha256": canonical_sha256(loaded["qa"]),
            "source_graph_path": str(graph_path),
            "source_graph_sha256": file_sha256(graph_path),
            "source_result_path": str(source_path),
            "source_result_sha256": file_sha256(source_path),
            "semantic_plan_path": str(plan_path.resolve()),
            "semantic_plan_sha256": file_sha256(plan_path),
            "semantic_plan_protocol": copy.deepcopy(plan_artifact["protocol"]),
            "source_snapshot": snapshot,
            "historical_reuse_sources": history_sources,
            "protocol": {
                "reader_model": frozen_runner.FIXED_WRITER_READER_MODEL,
                "reader_thinking_budget": 0,
                "judge_model": frozen_runner.FIXED_JUDGE_MODEL,
                "judge_thinking_level": "minimal_provider_floor",
                "final_fact_top_k": 20,
                "single_context_per_qa_system": True,
                "qa_prompt_sha256": _sha256_text(frozen_runner.QA_PROMPT),
                "judge_prompt_sha256": _sha256_text(frozen_runner.JUDGE_PROMPT),
            },
            "summary": summary,
            "usage": usage,
            "rows": rows,
        }
        _dump_json(output_path, payload)
        completed[system] = summary
        print(
            f"[child] {sample_id}/{system}: {summary['correct']}/{summary['n']} "
            f"acc={summary['acc_overall']:.4f}% new={summary['new_reader_judge_rows']}",
            flush=True,
        )
    return completed


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    keys_a = [(row["sample_id"], row["qa_id"]) for row in candidate]
    keys_b = [(row["sample_id"], row["qa_id"]) for row in baseline]
    if keys_a != keys_b:
        raise ValueError(f"paired rows not aligned: {label}")
    wins = losses = 0
    per_sample: dict[str, list[int]] = {}
    for row, base in zip(candidate, baseline, strict=True):
        delta = int(row["correct"]) - int(base["correct"])
        wins += delta > 0
        losses += delta < 0
        per_sample.setdefault(str(row["sample_id"]), []).append(delta)
    per_sample_pp = {
        sample: 100.0 * sum(values) / len(values)
        for sample, values in per_sample.items()
    }
    names = sorted(per_sample_pp)
    state = 20260807
    boot: list[float] = []
    for _ in range(10000):
        values: list[float] = []
        for _ in names:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            values.append(per_sample_pp[names[state % len(names)]])
        boot.append(mean(values))
    boot.sort()
    return {
        "contrast": label,
        "micro_delta_pp": 100.0 * (wins - losses) / len(candidate),
        "macro_conversation_delta_pp": mean(per_sample_pp.values()),
        "cluster_bootstrap_95ci_pp": [boot[249], boot[9749]],
        "wins": wins,
        "losses": losses,
        "ties": len(candidate) - wins - losses,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(wins, losses),
        "per_conversation_delta_pp": per_sample_pp,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = frozen_runner._summarize_rows(rows)
    sizes = sorted(len(str(row["retrieved"])) for row in rows)
    result["mean_retrieved_chars"] = sum(sizes) / len(sizes)
    result["p95_retrieved_chars"] = sizes[math.ceil(0.95 * len(sizes)) - 1]
    return result


def _load_pooled_baseline(
    loaded_items: Sequence[Mapping[str, Any]], system: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    for loaded in loaded_items:
        sample_id = str(loaded["sample_id"])
        path = _baseline_path(sample_id, system)
        payload = _load_json(path)
        by_id = _validate_source_rows(loaded, payload)
        rows.extend(by_id[str(source["qa_id"])] for source in loaded["qa"])
        sources[sample_id] = {"path": str(path), "sha256": file_sha256(path)}
    return rows, sources


def _category_breakdown(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in ("multi", "temporal", "open", "single"):
        cand = [row for row in candidate if row["category_name"] == category]
        base = [row for row in baseline if row["category_name"] == category]
        pair = _paired(cand, base, label=f"semantic_minus_session_{category}")
        result[category] = {
            "n": len(cand),
            "candidate_correct": sum(int(row["correct"]) for row in cand),
            "baseline_correct": sum(int(row["correct"]) for row in base),
            "delta_pp": pair["micro_delta_pp"],
            "wins": pair["wins"],
            "losses": pair["losses"],
        }
    return result


def _report(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# CreaMem3 v8 strict semantic context — full LoCoMo-10",
        "",
        "> Repeated full-corpus iterations are engineering development results, not an unseen test estimate.",
        "",
        "| System | Correct | Accuracy | Delta vs Session | New reader/judge rows |",
        "|---|---:|---:|---:|---:|",
    ]
    session = aggregate["summaries"]["session"]
    lines.append(
        f"| Session-only | {session['correct']}/{session['n']} | "
        f"{session['acc_overall']:.4f}% | — | 0 |"
    )
    for system in aggregate["systems"]:
        item = aggregate["summaries"][f"semantic_{system}"]
        pair = aggregate["paired"][f"semantic_{system}_vs_session"]
        lines.append(
            f"| Semantic {system.title()} | {item['correct']}/{item['n']} | "
            f"{item['acc_overall']:.4f}% | {pair['micro_delta_pp']:+.4f} pp | "
            f"{item['new_reader_judge_rows']} |"
        )
    lines.extend(["", "## Audit", ""])
    lines.extend(
        [
            f"- Planner rows: {aggregate['audit']['planner_rows']}; neutral failures: {aggregate['audit']['planner_neutral_failures']}.",
            "- Planner input was question text only; regex fallback was disabled.",
            "- Every QA/system had one predetermined context; no post-hoc answer selection was used.",
            f"- Lexical-gate violations: {aggregate['audit']['lexical_gate_violations']}; protocol failures: {aggregate['audit']['protocol_failures']}.",
            "- Final atomic-fact budget remained Top20 for both systems.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(
    *, dataset: Path, output_root: Path, plan_path: Path, systems: Sequence[str]
) -> dict[str, Any]:
    loaded_items = load_all_locomo(dataset)
    if sum(len(item["qa"]) for item in loaded_items) != 1540:
        raise ValueError("expected 1540 LoCoMo QA")
    plan_artifact = _load_json(plan_path)
    baselines: dict[str, list[dict[str, Any]]] = {}
    baseline_sources: dict[str, Any] = {}
    for name in ("session", "online", "offline"):
        baselines[name], baseline_sources[name] = _load_pooled_baseline(
            loaded_items, name
        )
    candidates: dict[str, list[dict[str, Any]]] = {system: [] for system in systems}
    child_sources: dict[str, Any] = {}
    per_conversation: dict[str, Any] = {}
    for loaded in loaded_items:
        sample_id = str(loaded["sample_id"])
        per_conversation[sample_id] = {}
        for system in systems:
            path = _child_path(output_root, sample_id, system)
            payload = _load_json(path)
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("sample_id") != sample_id
                or payload.get("system") != system
                or payload.get("semantic_plan_sha256") != file_sha256(plan_path)
                or len(payload.get("rows", [])) != len(loaded["qa"])
            ):
                raise ValueError(f"child artifact mismatch: {sample_id}/{system}")
            candidates[system].extend(payload["rows"])
            per_conversation[sample_id][system] = payload["summary"]
            child_sources[f"{sample_id}/{system}"] = {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
    for system in systems:
        if len(candidates[system]) != 1540:
            raise ValueError(f"expected 1540 rows for {system}")
    summaries = {name: _summary(rows) for name, rows in baselines.items()}
    paired: dict[str, Any] = {}
    categories: dict[str, Any] = {}
    for system in systems:
        summary = _summary(candidates[system])
        summary["new_reader_judge_rows"] = sum(
            int(per_conversation[item["sample_id"]][system]["new_reader_judge_rows"])
            for item in loaded_items
        )
        summary["reader_judge_reused_rows"] = 1540 - summary["new_reader_judge_rows"]
        summaries[f"semantic_{system}"] = summary
        for baseline in ("session", system):
            paired[f"semantic_{system}_vs_{baseline}"] = _paired(
                candidates[system],
                baselines[baseline],
                label=f"semantic_{system}_minus_{baseline}",
            )
        categories[system] = _category_breakdown(
            candidates[system], baselines["session"]
        )
    all_candidate_rows = [row for system in systems for row in candidates[system]]
    lexical_violations = sum(
        row["retrieval_trace"]["semantic_context_v8"]["materializer_audit"].get(
            "lexical_gate_used"
        )
        is not False
        or row["retrieval_trace"]["semantic_context_v8"]["materializer_audit"].get(
            "temporal_regex_path_called"
        )
        is not False
        for row in all_candidate_rows
    )
    protocol_failures = sum(
        bool(
            row.get("retrieval_error")
            or row.get("qa_error")
            or row.get("judge_parse_error")
            or row.get("judge_error")
            or row.get("judge_skipped")
        )
        for row in all_candidate_rows
    )
    aggregate_value = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scope": "locomo10_full_1540_engineering_development",
        "systems": list(systems),
        "sample_ids": [str(item["sample_id"]) for item in loaded_items],
        "n_per_system": 1540,
        "semantic_plan_path": str(plan_path.resolve()),
        "semantic_plan_sha256": file_sha256(plan_path),
        "semantic_plan_summary": copy.deepcopy(plan_artifact["summary"]),
        "source_snapshot": _source_snapshot(),
        "child_sources": child_sources,
        "baseline_sources": baseline_sources,
        "summaries": summaries,
        "paired": paired,
        "category_vs_session": categories,
        "per_conversation": per_conversation,
        "audit": {
            "planner_rows": len(plan_artifact["rows"]),
            "planner_neutral_failures": int(
                plan_artifact["summary"]["source_counts"].get("neutral_failure", 0)
            ),
            "lexical_gate_violations": lexical_violations,
            "protocol_failures": protocol_failures,
            "final_fact_top_k_locked": 20,
            "single_context_per_qa_system": True,
            "full_1540_reused_as_development": True,
        },
    }
    _dump_json(output_root / "aggregate.json", aggregate_value)
    report_path = output_root / "REPORT.md"
    report_path.write_text(_report(aggregate_value), encoding="utf-8", newline="\n")
    for system in systems:
        summary = summaries[f"semantic_{system}"]
        print(
            f"METRIC semantic_{system}_accuracy={summary['acc_overall']:.8f}",
            flush=True,
        )
    return aggregate_value


def _conversation_ids(dataset: Path) -> list[str]:
    return [str(item["sample_id"]) for item in load_all_locomo(dataset)]


def _launch_one(
    *,
    dataset: Path,
    output_root: Path,
    plan_path: Path,
    cache_root: Path,
    sample_id: str,
    systems: Sequence[str],
    workers: int,
    force: bool,
) -> dict[str, Any]:
    log_path = output_root / "logs" / f"{sample_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["AGENT_MEMORY_CACHE_ROOT"] = str((cache_root / sample_id).resolve())
    base_cache = Path(r"D:\creamem3_cache\locomo10_v3") / sample_id / "llm_cache.sqlite"
    if base_cache.exists():
        env["AGENT_MEMORY_BASE_CACHE"] = str(base_cache.resolve())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        str(dataset.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--plan-path",
        str(plan_path.resolve()),
        "child",
        "--conversation",
        sample_id,
        "--workers",
        str(workers),
        "--systems",
        *systems,
    ]
    if force:
        command.append("--force")
    started = time.time()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"child failed {sample_id}, rc={completed.returncode}\n{tail}")
    return {
        "sample_id": sample_id,
        "wall_s": time.time() - started,
        "log_path": str(log_path),
    }


def launch(
    *,
    dataset: Path,
    output_root: Path,
    plan_path: Path,
    cache_root: Path,
    systems: Sequence[str],
    conversation_workers: int,
    qa_workers: int,
    force: bool,
) -> dict[str, Any]:
    sample_ids = _conversation_ids(dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    print(
        f"[launch] conversations={len(sample_ids)} parallel="
        f"{min(conversation_workers, len(sample_ids))} systems={','.join(systems)}",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=min(conversation_workers, len(sample_ids))
    ) as executor:
        futures = {
            executor.submit(
                _launch_one,
                dataset=dataset,
                output_root=output_root,
                plan_path=plan_path,
                cache_root=cache_root,
                sample_id=sample_id,
                systems=systems,
                workers=qa_workers,
                force=force,
            ): sample_id
            for sample_id in sample_ids
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                result = future.result()
                print(
                    f"[launch] {sample_id} complete wall={result['wall_s']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append(sample_id)
                print(f"[launch] {sample_id} failed: {exc}", flush=True)
    if failures:
        raise RuntimeError("failed conversations: " + ", ".join(failures))
    return aggregate(
        dataset=dataset,
        output_root=output_root,
        plan_path=plan_path,
        systems=systems,
    )


def _parse_systems(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in SYSTEMS:
            raise ValueError(f"unknown system: {value}")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one system is required")
    return tuple(result)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    child = subparsers.add_parser("child")
    child.add_argument("--conversation", required=True)
    child.add_argument("--workers", type=int, default=3)
    child.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    child.add_argument("--force", action="store_true")

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    launch_parser.add_argument("--conversation-workers", type=int, default=10)
    launch_parser.add_argument("--qa-workers", type=int, default=3)
    launch_parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    launch_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    args = parser.parse_args(list(argv) if argv is not None else None)
    systems = _parse_systems(args.systems)
    if not args.plan_path.exists():
        parser.error(f"semantic plan artifact does not exist: {args.plan_path}")
    if args.command == "child":
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.conversation not in set(_conversation_ids(args.dataset)):
            parser.error(f"unknown conversation: {args.conversation}")
        run_child(
            dataset=args.dataset,
            output_root=args.output_root.resolve(),
            plan_path=args.plan_path.resolve(),
            sample_id=args.conversation,
            systems=systems,
            workers=args.workers,
            force=args.force,
        )
    elif args.command == "launch":
        if args.conversation_workers < 1 or args.qa_workers < 1:
            parser.error("worker counts must be positive")
        launch(
            dataset=args.dataset,
            output_root=args.output_root.resolve(),
            plan_path=args.plan_path.resolve(),
            cache_root=args.cache_root.resolve(),
            systems=systems,
            conversation_workers=args.conversation_workers,
            qa_workers=args.qa_workers,
            force=args.force,
        )
    else:
        aggregate(
            dataset=args.dataset,
            output_root=args.output_root.resolve(),
            plan_path=args.plan_path.resolve(),
            systems=systems,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

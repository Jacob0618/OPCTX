#!/usr/bin/env python3
"""Zero-reranker retrieval study on the frozen MemEye G3.5 Online graph.

The selector is deliberately score-free: it receives only the score-free
question, its already-cached query embedding and semantic plan, and the frozen
Online graph.  Gold clues, answers, and X/Y labels are loaded only after every
selection has been sealed in memory and are used solely by the evaluator.

This is an offline research harness.  It never calls a provider and never
modifies the canonical graph, prediction, score, or provider-cache artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from creamem3 import graph_memory_v2


SCHEMA_VERSION = "creamem3.memeye_mec50_offline.v1.0"
DEFAULT_FROZEN_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\isolated_full_arch_thinking512_low_r1"
)
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\external_memeye_data"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\mec50_offline_v1_r1"
)

METHODS = (
    "union_dense50",
    "union_dense80",
    "global_dense50",
    "global_dense80",
    "round_dense50",
    "mec50",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_ROUND_RE = re.compile(r"^(.*?):R(\d+)$")
_QUOTE_RE = re.compile(r"['\"]([^'\"]{3,})['\"]")

# Dataset-agnostic English function words.  This list is fixed and does not
# contain task names, gold-answer vocabulary, or question IDs.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for
    from had has have having he her hers him his how i if in into is it its
    may me might mine more most much my of on or our ours she should so than
    that the their theirs them then there these they this those through to too
    under up us was we were what when where which who why will with would you
    your yours image images picture pictures photo photos shown show according
    based describe described description match matches matching user assistant
    """.split()
)

_LATEST_CUES = frozenset(
    {"latest", "newest", "recent", "recently", "current", "currently", "now", "final"}
)
_EARLY_CUES = frozenset(
    {"original", "first", "initial", "initially", "earliest", "before", "previous"}
)
_TRAJECTORY_CUES = frozenset(
    {"change", "changed", "changes", "evolve", "evolved", "progress", "over", "trajectory"}
)


class OfflineStudyError(RuntimeError):
    """Raised when a frozen input or output contract is not satisfied."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic context matters
        raise OfflineStudyError(f"invalid JSON: {path}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _write_once_json(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists():
        raise OfflineStudyError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)


def _write_once_jsonl_gzip(path: Path, values: Iterable[Any]) -> None:
    path = Path(path)
    if path.exists():
        raise OfflineStudyError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")


def _validate_mec50_manifest(value: Mapping[str, Any]) -> None:
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 371:
        raise OfflineStudyError("MEC-50 manifest row census is not 371")
    if len({row.get("row_id") for row in rows if isinstance(row, dict)}) != 371:
        raise OfflineStudyError("MEC-50 manifest row IDs are missing or duplicated")
    forbidden = {"answer", "clue", "point", "x", "y", "gold"}
    for row in rows:
        if not isinstance(row, dict):
            raise OfflineStudyError("MEC-50 manifest row is not an object")
        if forbidden.intersection(str(key).lower() for key in row):
            raise OfflineStudyError("MEC-50 manifest contains evaluator-only fields")
        selected_ids = row.get("selected_fact_ids")
        trace = row.get("trace")
        if (
            not isinstance(selected_ids, list)
            or not selected_ids
            or len(selected_ids) > 50
            or len(selected_ids) != len(set(selected_ids))
            or not all(isinstance(fact_id, str) and fact_id for fact_id in selected_ids)
            or not isinstance(trace, dict)
        ):
            raise OfflineStudyError(f"invalid MEC-50 selection row: {row.get('row_id')}")
        trace_selected = trace.get("selected")
        if not isinstance(trace_selected, list):
            raise OfflineStudyError("MEC-50 trace selected list is missing")
        trace_ids = [item.get("fact_id") for item in trace_selected if isinstance(item, dict)]
        if trace_ids != selected_ids:
            raise OfflineStudyError("MEC-50 manifest IDs do not match the bound trace")
        if row.get("selected_count") != len(selected_ids):
            raise OfflineStudyError("MEC-50 selected-count drift")
        if row.get("selected_fact_ids_sha256") != _canonical_sha256(selected_ids):
            raise OfflineStudyError("MEC-50 selected-ID hash drift")
    contract = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != _canonical_sha256(contract):
        raise OfflineStudyError("MEC-50 manifest contract hash drift")


def build_mec50_selection_manifest(
    trace_rows: Sequence[Mapping[str, Any]], *, selector_digest: str
) -> dict[str, Any]:
    mec_rows = [row for row in trace_rows if row.get("method") == "mec50"]
    mec_rows.sort(key=lambda row: (str(row.get("task")), str(row.get("question_id"))))
    rows: list[dict[str, Any]] = []
    for receipt in mec_rows:
        selected = receipt.get("selected")
        if not isinstance(selected, list):
            raise OfflineStudyError("MEC-50 receipt selected list is missing")
        selected_ids = [str(item["fact_id"]) for item in selected]
        rows.append(
            {
                "row_id": receipt["row_id"],
                "task": receipt["task"],
                "question_id": receipt["question_id"],
                "question": receipt["question"],
                "semantic_plan": receipt["semantic_plan"],
                "selected_fact_ids": selected_ids,
                "selected_fact_ids_sha256": _canonical_sha256(selected_ids),
                "selected_count": len(selected_ids),
                "unique_source_rounds": receipt["unique_source_rounds"],
                "total_characters": receipt["total_characters"],
                "trace": {
                    "method": "mec50",
                    "selection_sha256": receipt["selection_sha256"],
                    "selected_source_rounds": receipt["selected_source_rounds"],
                    "roles": receipt["roles"],
                    "extra": receipt["extra"],
                    "selected": selected,
                },
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "creamem3.memeye_mec50_selection_manifest.v1.0",
        "score_free": True,
        "zero_reranker": True,
        "question_count": len(rows),
        "selector_digest": selector_digest,
        "rows": rows,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _validate_mec50_manifest(manifest)
    return manifest


def materialize_mec50_manifest_from_trace(output_root: Path) -> Path:
    output_root = Path(output_root).resolve()
    trace_path = output_root / "selection_traces.jsonl.gz"
    summary_path = output_root / "summary.json"
    if not trace_path.is_file() or not summary_path.is_file():
        raise OfflineStudyError("completed compressed trace and summary are required")
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        trace_rows = [json.loads(line) for line in handle if line.strip()]
    summary = _read_json(summary_path)
    manifest = build_mec50_selection_manifest(
        trace_rows, selector_digest=str(summary.get("selector_digest", ""))
    )
    target = output_root / "mec50_selection_manifest.json"
    _write_once_json(target, manifest)
    reread = _read_json(target)
    _validate_mec50_manifest(reread)
    return target


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _rank_percentiles(scores: Sequence[float]) -> np.ndarray:
    """Return deterministic [0, 1] percentiles with 1 assigned to rank one."""

    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    result = np.empty(n, dtype=np.float64)
    denominator = max(n - 1, 1)
    for rank, index in enumerate(order):
        result[index] = 1.0 - rank / denominator
    return result


def _corrected_log_mean_exp(values: Sequence[float], tau: float = 0.08) -> float:
    """Smooth max corrected for the number of sibling facts in a round."""

    if not values:
        return -1.0e9
    scaled = np.asarray(values, dtype=np.float64) / tau
    maximum = float(np.max(scaled))
    return tau * (
        maximum
        + math.log(float(np.exp(scaled - maximum).sum()))
        - math.log(len(values))
    )


def _source_rounds(
    fact: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    rounds: list[str] = []
    for dia_id in fact.get("source_dia_ids", []):
        source = source_map.get(str(dia_id), {})
        round_id = source.get("round")
        if isinstance(round_id, str) and round_id and round_id not in rounds:
            rounds.append(round_id)
    return tuple(rounds)


def _round_sort_key(round_id: str) -> tuple[str, int, str]:
    match = _ROUND_RE.match(round_id)
    if match is None:
        return (round_id, 0, round_id)
    return (match.group(1), int(match.group(2)), round_id)


def _primary_round(rounds: Sequence[str], fact_id: str) -> str:
    if not rounds:
        return f"NO_ROUND::{fact_id}"
    return sorted(rounds, key=_round_sort_key)[0]


@dataclass(frozen=True)
class QueryInput:
    row_id: str
    task: str
    question_id: str
    question: str
    query_vector: tuple[float, ...]
    semantic_plan: Mapping[str, Any]


@dataclass
class TaskIndex:
    task: str
    graph: Mapping[str, Any]
    index: graph_memory_v2.GraphMemoryIndex
    facts: list[Mapping[str, Any]]
    fact_by_id: dict[str, Mapping[str, Any]]
    fact_index: dict[str, int]
    fact_rounds: list[tuple[str, ...]]
    primary_rounds: list[str]
    round_to_indices: dict[str, list[int]]
    fact_tokens: list[set[str]]
    token_df: Counter[str]


@dataclass(frozen=True)
class Role:
    name: str
    kind: str
    tokens: tuple[str, ...]
    weight: float


def load_score_free_queries(frozen_root: Path) -> list[dict[str, Any]]:
    path = Path(frozen_root) / "project" / "queries_score_free.json"
    value = _read_json(path)
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 371:
        raise OfflineStudyError("score-free query census is not 371")
    forbidden = {"answer", "clue", "point", "category", "x", "y"}
    for row in rows:
        if not isinstance(row, dict) or forbidden.intersection(k.lower() for k in row):
            raise OfflineStudyError("score-free query contains forbidden evaluator fields")
    return rows


def load_cached_query_vectors(
    frozen_root: Path, expected_rows: int
) -> tuple[list[tuple[float, ...]], dict[str, Any]]:
    root = (
        Path(frozen_root)
        / "provider"
        / "predict"
        / "query_embedding_native_3072"
    )
    chunks: list[tuple[int, list[Any], Path, Path]] = []
    for plan_path in (root / "plans").glob("*.json"):
        plan = _read_json(plan_path)
        metadata = plan.get("cache_metadata", {})
        start = int(metadata.get("batch_start", -1))
        response_path = root / "responses" / plan_path.name
        response = _read_json(response_path)
        vectors = response.get("vectors")
        if start < 0 or not isinstance(vectors, list):
            raise OfflineStudyError(f"invalid query embedding batch: {plan_path}")
        if len(vectors) != int(plan.get("batch_size", -1)):
            raise OfflineStudyError(f"query embedding batch-size drift: {plan_path}")
        chunks.append((start, vectors, plan_path, response_path))
    chunks.sort(key=lambda item: item[0])
    flattened: list[tuple[float, ...]] = []
    expected_start = 0
    receipts: list[dict[str, Any]] = []
    for start, vectors, plan_path, response_path in chunks:
        if start != expected_start:
            raise OfflineStudyError("query embedding batches are not contiguous")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != 3072:
                raise OfflineStudyError("query embedding is not a 3072-vector")
            array = np.asarray(vector, dtype=np.float64)
            if not np.isfinite(array).all() or float(np.linalg.norm(array)) == 0.0:
                raise OfflineStudyError("query embedding is zero or non-finite")
            flattened.append(tuple(float(value) for value in vector))
        expected_start += len(vectors)
        receipts.append(
            {
                "batch_start": start,
                "batch_size": len(vectors),
                "plan_path": str(plan_path),
                "plan_sha256": _file_sha256(plan_path),
                "response_path": str(response_path),
                "response_sha256": _file_sha256(response_path),
            }
        )
    if len(flattened) != expected_rows:
        raise OfflineStudyError("query embedding census does not match queries")
    return flattened, {"batches": receipts, "vector_count": len(flattened)}


def load_cached_semantic_plans(
    frozen_root: Path, row_ids: Iterable[str]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    root = Path(frozen_root) / "provider" / "predict" / "g25_reader"
    plan_root = root / "plans" / "planner"
    response_root = root / "responses" / "planner"
    wanted = set(row_ids)
    plans: dict[str, Mapping[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for plan_path in plan_root.glob("*.json"):
        envelope = _read_json(plan_path)
        metadata = envelope.get("cache_metadata", {})
        row_id = metadata.get("row_id")
        if row_id not in wanted:
            continue
        response_path = response_root / plan_path.name
        response = _read_json(response_path)
        try:
            semantic_plan = json.loads(str(response.get("text", "")))
        except Exception as exc:
            raise OfflineStudyError(f"invalid cached semantic plan: {row_id}") from exc
        allowed = {
            "operation",
            "scope",
            "evidence_cardinality",
            "temporal_mode",
            "abstraction",
            "answer_shape",
            "confidence",
        }
        if not isinstance(semantic_plan, dict) or set(semantic_plan) != allowed:
            raise OfflineStudyError(f"semantic plan schema drift: {row_id}")
        if row_id in plans:
            raise OfflineStudyError(f"duplicate semantic plan: {row_id}")
        plans[str(row_id)] = semantic_plan
        receipts.append(
            {
                "row_id": row_id,
                "plan_path": str(plan_path),
                "plan_sha256": _file_sha256(plan_path),
                "response_path": str(response_path),
                "response_sha256": _file_sha256(response_path),
            }
        )
    if set(plans) != wanted:
        missing = sorted(wanted - set(plans))[:10]
        raise OfflineStudyError(f"missing semantic plans: {missing}")
    receipts.sort(key=lambda item: item["row_id"])
    return plans, {"plan_count": len(plans), "receipts_sha256": _canonical_sha256(receipts)}


def load_task_indices(frozen_root: Path, tasks: Iterable[str]) -> dict[str, TaskIndex]:
    graph_root = Path(frozen_root) / "build" / "graphs" / "g35" / "online"
    result: dict[str, TaskIndex] = {}
    for task in sorted(set(tasks)):
        graph_path = graph_root / f"{task}.json"
        graph = _read_json(graph_path)
        if graph.get("backbone") != "g35" or graph.get("graph_variant") != "online":
            raise OfflineStudyError(f"not a frozen G3.5 Online graph: {graph_path}")
        facts = list(graph.get("fact_nodes", []))
        if not facts:
            raise OfflineStudyError(f"empty graph: {graph_path}")
        index = graph_memory_v2.GraphMemoryIndex(
            graph,
            summary_top_k=10,
            dense_fact_top_k=20,
            bm25_fact_top_k=20,
            candidate_cap=len(facts),
            rerank_top_k=50,
        )
        source_map = graph.get("memeye_source_map", {})
        fact_rounds = [_source_rounds(fact, source_map) for fact in facts]
        primary_rounds = [
            _primary_round(rounds, str(fact["id"]))
            for rounds, fact in zip(fact_rounds, facts)
        ]
        round_to_indices: dict[str, list[int]] = defaultdict(list)
        for fact_index, round_id in enumerate(primary_rounds):
            round_to_indices[round_id].append(fact_index)
        fact_tokens = [set(_tokens(str(fact.get("text", "")))) for fact in facts]
        token_df: Counter[str] = Counter()
        for tokens in fact_tokens:
            token_df.update(tokens)
        result[task] = TaskIndex(
            task=task,
            graph=graph,
            index=index,
            facts=facts,
            fact_by_id={str(fact["id"]): fact for fact in facts},
            fact_index={str(fact["id"]): i for i, fact in enumerate(facts)},
            fact_rounds=fact_rounds,
            primary_rounds=primary_rounds,
            round_to_indices=dict(round_to_indices),
            fact_tokens=fact_tokens,
            token_df=token_df,
        )
    return result


def build_query_inputs(
    rows: Sequence[Mapping[str, Any]],
    vectors: Sequence[tuple[float, ...]],
    plans: Mapping[str, Mapping[str, Any]],
) -> list[QueryInput]:
    result: list[QueryInput] = []
    for row, vector in zip(rows, vectors):
        row_id = str(row["row_id"])
        result.append(
            QueryInput(
                row_id=row_id,
                task=str(row["task"]),
                question_id=str(row["question_id"]),
                question=str(row["question"]),
                query_vector=vector,
                semantic_plan=plans[row_id],
            )
        )
    return result


def _base_feature_state(
    task_index: TaskIndex, query: QueryInput
) -> dict[str, Any]:
    dense_scores = task_index.index._scores(  # type: ignore[attr-defined]
        task_index.index.fact_matrix, query.query_vector
    ).astype(np.float64)
    dense_p = _rank_percentiles(dense_scores)
    lexical = task_index.index.bm25.search(query.question, len(task_index.facts))
    bm25_score_by_id = {str(fact_id): float(score) for fact_id, score in lexical}
    bm25_scores = np.asarray(
        [bm25_score_by_id.get(str(fact["id"]), 0.0) for fact in task_index.facts],
        dtype=np.float64,
    )
    positive_indices = [i for i, value in enumerate(bm25_scores) if value > 0]
    bm25_p = np.zeros(len(task_index.facts), dtype=np.float64)
    if positive_indices:
        positive_p = _rank_percentiles([bm25_scores[i] for i in positive_indices])
        for i, value in zip(positive_indices, positive_p):
            # Keep a nonzero distinction between the last lexical hit and no hit.
            bm25_p[i] = 0.15 + 0.85 * float(value)

    query_tokens = [
        token for token in _tokens(query.question) if token not in _STOPWORDS
    ]
    unique_query_tokens = list(dict.fromkeys(query_tokens))
    n_docs = len(task_index.facts)
    token_weights = {
        token: math.log((n_docs + 1) / (task_index.token_df.get(token, 0) + 1)) + 1.0
        for token in unique_query_tokens
        if task_index.token_df.get(token, 0) > 0
    }
    weight_total = sum(token_weights.values())
    lexical_coverage = np.asarray(
        [
            (
                sum(weight for token, weight in token_weights.items() if token in tokens)
                / weight_total
                if weight_total
                else 0.0
            )
            for tokens in task_index.fact_tokens
        ],
        dtype=np.float64,
    )
    fact_base = 0.72 * dense_p + 0.18 * bm25_p + 0.10 * lexical_coverage

    round_raw: dict[str, float] = {}
    for round_id, indices in task_index.round_to_indices.items():
        round_raw[round_id] = _corrected_log_mean_exp(
            [float(fact_base[i]) for i in indices]
        )
    ordered_rounds = sorted(round_raw)
    round_percentiles = _rank_percentiles([round_raw[key] for key in ordered_rounds])
    round_prior = {
        key: float(value) for key, value in zip(ordered_rounds, round_percentiles)
    }
    return {
        "dense_scores": dense_scores,
        "dense_p": dense_p,
        "bm25_scores": bm25_scores,
        "bm25_p": bm25_p,
        "lexical_coverage": lexical_coverage,
        "fact_base": fact_base,
        "round_raw": round_raw,
        "round_prior": round_prior,
        "query_token_weights": token_weights,
    }


def _derive_roles(
    query: QueryInput, task_index: TaskIndex, state: Mapping[str, Any]
) -> list[Role]:
    token_weights = dict(state["query_token_weights"])
    roles: list[Role] = []

    # Quoted spans are treated as compound anchors when at least two of their
    # content tokens occur in the memory.  This is query-only and generic.
    used_tokens: set[str] = set()
    for match in _QUOTE_RE.finditer(query.question):
        tokens = tuple(
            token
            for token in dict.fromkeys(_tokens(match.group(1)))
            if token not in _STOPWORDS and token in token_weights
        )
        if len(tokens) < 2:
            continue
        roles.append(
            Role(
                name=f"phrase:{'_'.join(tokens[:5])}",
                kind="lexical_phrase",
                tokens=tokens[:5],
                weight=1.25,
            )
        )
        used_tokens.update(tokens)
        if len(roles) == 2:
            break

    ranked_tokens = sorted(
        token_weights,
        key=lambda token: (-float(token_weights[token]), token),
    )
    for token in ranked_tokens:
        if token in used_tokens:
            continue
        idf_weight = float(token_weights[token])
        roles.append(
            Role(
                name=f"anchor:{token}",
                kind="lexical_anchor",
                tokens=(token,),
                weight=min(1.5, max(0.75, idf_weight / 2.5)),
            )
        )
        if len([role for role in roles if role.kind.startswith("lexical")]) >= 7:
            break

    question_tokens = set(_tokens(query.question))
    plan = query.semantic_plan
    temporal_mode = str(plan.get("temporal_mode", "none"))
    operation = str(plan.get("operation", "atomic_lookup"))
    needs_early = bool(question_tokens & _EARLY_CUES)
    needs_late = bool(question_tokens & _LATEST_CUES)
    trajectory = (
        temporal_mode in {"order", "trajectory"}
        or operation in {"trace_change", "temporal_order"}
        or bool(question_tokens & _TRAJECTORY_CUES)
    )
    if trajectory:
        needs_early = True
        needs_late = True
    if needs_early:
        roles.append(Role("temporal:early", "temporal_early", (), 1.15))
    if needs_late:
        roles.append(Role("temporal:late", "temporal_late", (), 1.15))
    return roles


def _temporal_positions(task_index: TaskIndex) -> tuple[np.ndarray, np.ndarray]:
    dates = [str(fact.get("iso_date") or fact.get("timestamp") or "") for fact in task_index.facts]
    order = sorted(
        range(len(task_index.facts)),
        key=lambda i: (dates[i], _round_sort_key(task_index.primary_rounds[i]), i),
    )
    late = np.empty(len(order), dtype=np.float64)
    denominator = max(len(order) - 1, 1)
    for rank, index in enumerate(order):
        late[index] = rank / denominator
    early = 1.0 - late
    return early, late


def _role_probabilities(
    task_index: TaskIndex,
    state: Mapping[str, Any],
    roles: Sequence[Role],
) -> dict[str, np.ndarray]:
    dense_p = np.asarray(state["dense_p"], dtype=np.float64)
    early, late = _temporal_positions(task_index)
    result: dict[str, np.ndarray] = {}
    for role in roles:
        if role.kind.startswith("lexical"):
            values: list[float] = []
            for tokens in task_index.fact_tokens:
                matched = sum(1 for token in role.tokens if token in tokens)
                fraction = matched / max(len(role.tokens), 1)
                values.append(fraction)
            match = np.asarray(values, dtype=np.float64)
            # Exact lexical evidence remains visible even if global semantic
            # similarity is modest; unmatched facts retain only a tiny prior.
            probability = match * (0.58 + 0.42 * dense_p) + (1.0 - match) * 0.02 * dense_p
        elif role.kind == "temporal_early":
            probability = (0.15 + 0.85 * early) * (0.20 + 0.80 * dense_p)
        elif role.kind == "temporal_late":
            probability = (0.15 + 0.85 * late) * (0.20 + 0.80 * dense_p)
        else:  # pragma: no cover - Role construction is closed above.
            raise OfflineStudyError(f"unknown role kind: {role.kind}")
        result[role.name] = np.clip(probability, 0.0, 0.995)
    return result


def _selection_item(
    task_index: TaskIndex,
    state: Mapping[str, Any],
    index: int,
    rank: int,
    *,
    method: str,
    marginal_gain: float | None = None,
    relevance_gain: float | None = None,
    coverage_gain: float | None = None,
    role_probabilities: Mapping[str, float] | None = None,
    role_coverage_before: Mapping[str, float] | None = None,
    role_coverage_after: Mapping[str, float] | None = None,
    channels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fact = task_index.facts[index]
    round_id = task_index.primary_rounds[index]
    return {
        "rank": rank,
        "method": method,
        "fact_id": str(fact["id"]),
        "source_rounds": list(task_index.fact_rounds[index]),
        "primary_round": round_id,
        "round_sibling_facts": len(task_index.round_to_indices[round_id]),
        "characters": len(str(fact.get("text", ""))),
        "dense_score": float(state["dense_scores"][index]),
        "dense_percentile": float(state["dense_p"][index]),
        "bm25_score": float(state["bm25_scores"][index]),
        "bm25_percentile": float(state["bm25_p"][index]),
        "lexical_coverage": float(state["lexical_coverage"][index]),
        "fact_base_score": float(state["fact_base"][index]),
        "round_corrected_score": float(state["round_raw"][round_id]),
        "round_percentile": float(state["round_prior"][round_id]),
        "marginal_gain": marginal_gain,
        "relevance_gain": relevance_gain,
        "coverage_gain": coverage_gain,
        "role_probabilities": dict(role_probabilities or {}),
        "role_coverage_before": dict(role_coverage_before or {}),
        "role_coverage_after": dict(role_coverage_after or {}),
        "native_union_channels": dict(channels or {}),
    }


def _selection_receipt(
    query: QueryInput,
    method: str,
    selected: Sequence[Mapping[str, Any]],
    *,
    roles: Sequence[Role] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fact_ids = [str(item["fact_id"]) for item in selected]
    source_rounds = sorted(
        {round_id for item in selected for round_id in item["source_rounds"]},
        key=_round_sort_key,
    )
    value = {
        "row_id": query.row_id,
        "task": query.task,
        "question_id": query.question_id,
        "question": query.question,
        "semantic_plan": dict(query.semantic_plan),
        "method": method,
        "selected_count": len(selected),
        "unique_source_rounds": len(source_rounds),
        "selected_source_rounds": source_rounds,
        "total_characters": sum(int(item["characters"]) for item in selected),
        "roles": [
            {"name": role.name, "kind": role.kind, "tokens": list(role.tokens), "weight": role.weight}
            for role in roles
        ],
        "selected": list(selected),
        "extra": dict(extra or {}),
    }
    value["selected_fact_ids_sha256"] = _canonical_sha256(fact_ids)
    value["selection_sha256"] = _canonical_sha256(value)
    return value


def select_global_dense(
    query: QueryInput, task_index: TaskIndex, state: Mapping[str, Any], top_k: int
) -> dict[str, Any]:
    order = sorted(
        range(len(task_index.facts)),
        key=lambda i: (-float(state["dense_scores"][i]), i),
    )[:top_k]
    selected = [
        _selection_item(task_index, state, i, rank, method=f"global_dense{top_k}")
        for rank, i in enumerate(order, 1)
    ]
    return _selection_receipt(query, f"global_dense{top_k}", selected)


def select_union_dense(
    query: QueryInput,
    task_index: TaskIndex,
    state: Mapping[str, Any],
    top_k: int,
    *,
    full_candidates: Sequence[Mapping[str, Any]] | None = None,
    full_union_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if full_candidates is None or full_union_trace is None:
        candidates, union_trace = task_index.index._candidate_pool(  # type: ignore[attr-defined]
            query.question, query.query_vector
        )
    else:
        candidates = list(full_candidates)
        union_trace = dict(full_union_trace)
    candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item["dense_score"]),
            task_index.fact_index[str(item["fact"]["id"])],
        ),
    )[:top_k]
    selected = []
    for rank, item in enumerate(candidates, 1):
        index = task_index.fact_index[str(item["fact"]["id"])]
        selected.append(
            _selection_item(
                task_index,
                state,
                index,
                rank,
                method=f"union_dense{top_k}",
                channels=item.get("channels", {}),
            )
        )
    return _selection_receipt(
        query,
        f"union_dense{top_k}",
        selected,
        extra={
            "union_before_cap": int(union_trace["union_before_cap"]),
            "summary_top_k": 10,
            "dense_top_k": 20,
            "bm25_top_k": 20,
        },
    )


def select_round_dense50(
    query: QueryInput, task_index: TaskIndex, state: Mapping[str, Any]
) -> dict[str, Any]:
    round_order = sorted(
        task_index.round_to_indices,
        key=lambda round_id: (
            -float(state["round_raw"][round_id]),
            _round_sort_key(round_id),
        ),
    )[:50]
    selected: list[dict[str, Any]] = []
    for rank, round_id in enumerate(round_order, 1):
        index = min(
            task_index.round_to_indices[round_id],
            key=lambda i: (-float(state["dense_scores"][i]), i),
        )
        selected.append(
            _selection_item(
                task_index, state, index, rank, method="round_dense50"
            )
        )
    return _selection_receipt(
        query,
        "round_dense50",
        selected,
        extra={"round_aggregation": "tau0.08_log_mean_exp_minus_log_sibling_count"},
    )


def select_mec50(
    query: QueryInput, task_index: TaskIndex, state: Mapping[str, Any]
) -> dict[str, Any]:
    roles = _derive_roles(query, task_index, state)
    role_arrays = _role_probabilities(task_index, state, roles)
    role_weights = {role.name: role.weight for role in roles}
    coverage = {role.name: 0.0 for role in roles}
    selected_rounds: set[str] = set()
    selected: list[dict[str, Any]] = []
    epsilon = 0.15
    max_role_log_gain = math.log((epsilon + 1.0) / epsilon)
    weight_total = sum(role_weights.values())

    for rank in range(1, min(50, len(task_index.round_to_indices)) + 1):
        best: tuple[float, float, float, str, int, dict[str, float]] | None = None
        for round_id, fact_indices in task_index.round_to_indices.items():
            if round_id in selected_rounds:
                continue
            round_prior = float(state["round_prior"][round_id])
            for index in fact_indices:
                fact_relevance = float(state["fact_base"][index])
                relevance_gain = 0.55 * round_prior + 0.45 * fact_relevance
                probabilities = {
                    role.name: float(role_arrays[role.name][index]) for role in roles
                }
                raw_coverage_gain = 0.0
                for role in roles:
                    before = coverage[role.name]
                    after = before + probabilities[role.name] * (1.0 - before)
                    raw_coverage_gain += role.weight * (
                        math.log(epsilon + after) - math.log(epsilon + before)
                    )
                coverage_gain = (
                    raw_coverage_gain / (weight_total * max_role_log_gain)
                    if weight_total
                    else 0.0
                )
                # Role coverage dominates only while an uncovered query role
                # exists; its probabilistic marginal naturally vanishes later.
                marginal = 0.48 * relevance_gain + 0.52 * coverage_gain
                candidate = (
                    marginal,
                    relevance_gain,
                    coverage_gain,
                    round_id,
                    index,
                    probabilities,
                )
                if best is None or (
                    candidate[0], candidate[1], -candidate[4]
                ) > (best[0], best[1], -best[4]):
                    best = candidate
        if best is None:
            break
        marginal, relevance_gain, coverage_gain, round_id, index, probabilities = best
        before = dict(coverage)
        for role in roles:
            probability = probabilities[role.name]
            coverage[role.name] += probability * (1.0 - coverage[role.name])
        after = dict(coverage)
        selected_rounds.add(round_id)
        selected.append(
            _selection_item(
                task_index,
                state,
                index,
                rank,
                method="mec50",
                marginal_gain=float(marginal),
                relevance_gain=float(relevance_gain),
                coverage_gain=float(coverage_gain),
                role_probabilities=probabilities,
                role_coverage_before=before,
                role_coverage_after=after,
            )
        )
    return _selection_receipt(
        query,
        "mec50",
        selected,
        roles=roles,
        extra={
            "round_aggregation": "tau0.08_log_mean_exp_minus_log_sibling_count",
            "objective": "0.48*round_fact_relevance+0.52*normalized_log_role_coverage_gain",
            "epsilon": epsilon,
            "final_role_coverage": coverage,
        },
    )


def run_selector_phase(
    queries: Sequence[QueryInput], task_indices: Mapping[str, TaskIndex]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run all methods without loading any evaluator-only dataset fields."""

    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for ordinal, query in enumerate(queries, 1):
        task_index = task_indices[query.task]
        state = _base_feature_state(task_index, query)
        union_candidates, union_trace = task_index.index._candidate_pool(  # type: ignore[attr-defined]
            query.question, query.query_vector
        )
        per_method = {
            "union_dense50": select_union_dense(
                query,
                task_index,
                state,
                50,
                full_candidates=union_candidates,
                full_union_trace=union_trace,
            ),
            "union_dense80": select_union_dense(
                query,
                task_index,
                state,
                80,
                full_candidates=union_candidates,
                full_union_trace=union_trace,
            ),
            "global_dense50": select_global_dense(query, task_index, state, 50),
            "global_dense80": select_global_dense(query, task_index, state, 80),
            "round_dense50": select_round_dense50(query, task_index, state),
            "mec50": select_mec50(query, task_index, state),
        }
        if tuple(per_method) != METHODS:
            raise OfflineStudyError("method census drift")
        selections[query.row_id] = per_method
        if ordinal % 25 == 0 or ordinal == len(queries):
            print(f"selector {ordinal}/{len(queries)}", flush=True)
    return selections


# ---------------------------------------------------------------------------
# Evaluator-only section.  Nothing below is called until selector completion.
# ---------------------------------------------------------------------------


def _flatten_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_flatten_strings(item))
    return result


def load_evaluator_gold(
    data_root: Path, tasks: Iterable[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    dialog_root = Path(data_root) / "data" / "dialog"
    for task in sorted(set(tasks)):
        value = _read_json(dialog_root / f"{task}.json")
        qa_rows = value.get("human-annotated QAs") if isinstance(value, dict) else None
        if not isinstance(qa_rows, list):
            raise OfflineStudyError(f"missing evaluator QA rows: {task}")
        for row in qa_rows:
            row_id = f"{task}::{row['question_id']}"
            clues = list(dict.fromkeys(_flatten_strings(row.get("clue", []))))
            points = _flatten_strings(row.get("point", []))
            if not clues:
                raise OfflineStudyError(f"empty gold clue set: {row_id}")
            result[row_id] = {
                "clues": clues,
                "x": next((point for point in points if point.startswith("X")), "X?"),
                "y": next((point for point in points if point.startswith("Y")), "Y?"),
            }
    return result


def evaluate_selections(
    selections: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gold: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row_id, per_method in selections.items():
        if row_id not in gold:
            raise OfflineStudyError(f"missing evaluator gold: {row_id}")
        evaluator = gold[row_id]
        clues = set(str(value) for value in evaluator["clues"])
        for method, receipt in per_method.items():
            selected_rounds = set(str(value) for value in receipt["selected_source_rounds"])
            hits = clues & selected_rounds
            rows.append(
                {
                    "row_id": row_id,
                    "task": receipt["task"],
                    "question_id": receipt["question_id"],
                    "method": method,
                    "x": evaluator["x"],
                    "y": evaluator["y"],
                    "gold_clue_count": len(clues),
                    "hit_clue_count": len(hits),
                    "any": float(bool(hits)),
                    "coverage": len(hits) / len(clues),
                    "full": float(hits == clues),
                    "selected_count": int(receipt["selected_count"]),
                    "unique_source_rounds": int(receipt["unique_source_rounds"]),
                    "total_characters": int(receipt["total_characters"]),
                    "gold_clues": sorted(clues, key=_round_sort_key),
                    "hit_clues": sorted(hits, key=_round_sort_key),
                    "missing_clues": sorted(clues - hits, key=_round_sort_key),
                    "selection_sha256": receipt["selection_sha256"],
                }
            )

    def aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        chars = [int(row["total_characters"]) for row in values]
        unique = [int(row["unique_source_rounds"]) for row in values]
        selected_counts = [int(row["selected_count"]) for row in values]
        return {
            "questions": len(values),
            "any": statistics.fmean(float(row["any"]) for row in values),
            "coverage": statistics.fmean(float(row["coverage"]) for row in values),
            "full": statistics.fmean(float(row["full"]) for row in values),
            "mean_selected_count": statistics.fmean(selected_counts),
            "mean_unique_source_rounds": statistics.fmean(unique),
            "mean_characters": statistics.fmean(chars),
            "median_characters": statistics.median(chars),
            "p90_characters": float(np.percentile(chars, 90)),
        }

    summary: dict[str, Any] = {"overall": {}, "by_task": {}, "by_x": {}, "by_y": {}}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary["overall"][method] = aggregate(method_rows)
        for field, key in (("task", "by_task"), ("x", "by_x"), ("y", "by_y")):
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in method_rows:
                buckets[str(row[field])].append(row)
            summary[key][method] = {
                bucket: aggregate(bucket_rows)
                for bucket, bucket_rows in sorted(buckets.items())
            }
    return summary, rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    frozen_root = args.frozen_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise OfflineStudyError(
            f"dedicated output root already exists; choose a fresh root: {output_root}"
        )
    if output_root == frozen_root or frozen_root in output_root.parents:
        raise OfflineStudyError("output must be separate from the frozen canonical root")

    rows = load_score_free_queries(frozen_root)
    vectors, vector_receipt = load_cached_query_vectors(frozen_root, len(rows))
    plans, plan_receipt = load_cached_semantic_plans(
        frozen_root, (str(row["row_id"]) for row in rows)
    )
    task_indices = load_task_indices(frozen_root, (str(row["task"]) for row in rows))
    queries = build_query_inputs(rows, vectors, plans)

    # Gold is intentionally not loaded before this returns.
    selections = run_selector_phase(queries, task_indices)
    selector_digest = _canonical_sha256(
        {
            row_id: {
                method: receipt["selection_sha256"]
                for method, receipt in per_method.items()
            }
            for row_id, per_method in selections.items()
        }
    )

    gold = load_evaluator_gold(args.data_root.resolve(), task_indices)
    if set(gold) != set(selections):
        raise OfflineStudyError("selector/evaluator row census drift")
    metrics, evaluation_rows = evaluate_selections(selections, gold)

    trace_rows = [
        receipt
        for query in queries
        for receipt in (selections[query.row_id][method] for method in METHODS)
    ]
    input_receipt = {
        "schema_version": SCHEMA_VERSION,
        "frozen_root": str(frozen_root),
        "data_root": str(args.data_root.resolve()),
        "score_free_query_path": str(frozen_root / "project" / "queries_score_free.json"),
        "score_free_query_sha256": _file_sha256(
            frozen_root / "project" / "queries_score_free.json"
        ),
        "query_embedding_receipt": vector_receipt,
        "semantic_plan_receipt": plan_receipt,
        "graphs": {
            task: {
                "path": str(
                    frozen_root / "build" / "graphs" / "g35" / "online" / f"{task}.json"
                ),
                "content_fingerprint": index.graph.get("content_fingerprint"),
                "fact_count": len(index.facts),
                "round_count": len(index.round_to_indices),
            }
            for task, index in sorted(task_indices.items())
        },
        "selector_gold_free": True,
        "selector_digest": selector_digest,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "question_count": len(queries),
        "methods": list(METHODS),
        "zero_provider_calls": True,
        "zero_qwen_calls": True,
        "selector_gold_free": True,
        "selector_digest": selector_digest,
        "metrics": metrics,
    }

    _write_once_json(output_root / "input_receipt.json", input_receipt)
    _write_once_jsonl_gzip(output_root / "selection_traces.jsonl.gz", trace_rows)
    _write_once_json(
        output_root / "mec50_selection_manifest.json",
        build_mec50_selection_manifest(trace_rows, selector_digest=selector_digest),
    )
    _write_once_json(output_root / "evaluation_rows.json", evaluation_rows)
    _write_once_json(output_root / "summary.json", result)
    print(json.dumps(result["metrics"]["overall"], indent=2), flush=True)
    print(f"wrote {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

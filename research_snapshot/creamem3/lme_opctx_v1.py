"""LongMemEval adapter for CreaMem3 Online + operation context.

The adapter deliberately separates the *evaluation environment* from the
memory architecture:

* MGTM-7935's LongMemEval memory-write, QA, and judge prompt literals are
  loaded from their frozen source files and hash-gated before use;
* one physical LongMemEval session becomes one CreaMem session packet;
* the writer's historical string-only facts receive a compact, explicit
  session-level provenance marker (``D<ordinal>:0``) after generation;
* CreaMem's causal Online event views and operation-routed reader context are
  then applied without changing any of the three MGTM evaluation prompts;
* retrieval is 20 summaries + 40 dense facts + 40 BM25 facts, dense-pre-ranked
  to at most 240 candidates, Qwen3-reranked to 30 facts.

This module owns no credentials and performs no network calls at import time.
Every live runner must call :func:`assert_prompt_contract` before importing a
provider client.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm


SCHEMA_VERSION = "creamem3.lme_online_opctx.v1"
ROOT = Path(__file__).resolve().parents[1]
MGTM_PROJECT_ROOT = ROOT.parent / "CreaMem-AutoResearch"
MGTM_ROOT = (
    MGTM_PROJECT_ROOT
    / "release"
    / "CreaMem2"
    / "benchmarks"
    / "longmemeval"
)
MGTM_PROMPT_SOURCE = MGTM_ROOT / "prompts.py"
MGTM_MEMORY_SOURCE = MGTM_ROOT / "memory.py"
MGTM_CLIENT_SOURCE = MGTM_ROOT / "clients.py"
MGTM_RUNNER_SOURCE = MGTM_ROOT / "run.py"
MGTM_QWEN_SOURCE = MGTM_CLIENT_SOURCE
LME_DATASET = ROOT / "data" / "longmemeval" / "longmemeval_s.json"


EXPECTED_FILE_SHA256 = {
    "mgtm_prompts": "babf9b1b1987a224b77533b64a020f4f61be78fadf489f1143fe96a05f5e8879",
    "mgtm_memory": "036a14e9a81096864526cc45fed0cf3fe8937d8e613bfb7e29f98b4fd3ce5aa9",
    "mgtm_clients": "9bc76695be5d0d698e722677531efed3a2834db12346c24fc031470950b334b0",
    "mgtm_runner": "1856116b0e557a7db4781f2579caa2b4e570b13d68c25f7eca7f818882bdc43f",
    "lme_dataset": "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894",
}

EXPECTED_PROMPT_SHA256 = {
    "write_template": "7dcca5dfdf1bdf4f2ebf78b414503d5d92360483727e98724ae40d3ea19f6ce8",
    "qa_template": "35e247445e5671e98e74e615b6bceaf455343ffb0a635086ea5dc45fae071ebe",
    "judge_template": "1a71db6662773fa99845d190bb0686fec5cdc6644eca821157054006ec79a1a1",
    "qwen_instruction": "00131f7ea8fb5a59f52656f664cf6db5900f40a7a57af342762dbbcb9d5b6b67",
}

EXPECTED_PROMPT_LENGTHS = {
    "write_template": 1404,
    "qa_template": 275,
    "judge_template": 494,
    "qwen_instruction": 107,
}

WRITE_MAX_ATTEMPTS = 5
WRITE_RETRY_BACKOFF_S = 2.0
WRITE_MAX_TOKENS = 8192
SUMMARY_TOP_K = 20
DENSE_FACT_TOP_K = 40
BM25_FACT_TOP_K = 40
CANDIDATE_CAP = 240
RERANK_TOP_K = 30

ChatFn = Callable[..., str]
EmbedFn = Callable[..., Sequence[Sequence[float]]]
RerankFn = Callable[[str, Sequence[str], str | None], Sequence[float]]
DisplayTextFn = Callable[[str], tuple[str, bool]]


class PromptContractError(RuntimeError):
    """Raised before any live call if the MGTM prompt authority drifted."""


class NonRetryableProviderError(RuntimeError):
    """A provider rejected the canonical request; repeating it cannot help."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _literal_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise PromptContractError(f"{name} is not a string literal")
                return value
    raise PromptContractError(f"missing literal {name} in {path}")


@dataclass(frozen=True)
class PromptContract:
    write_template: str
    qa_template: str
    judge_template: str
    qwen_instruction: str
    file_sha256: Mapping[str, str]
    prompt_sha256: Mapping[str, str]
    contract_sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "file_sha256": dict(self.file_sha256),
            "prompt_sha256": dict(self.prompt_sha256),
            "prompt_lengths": {
                "write_template": len(self.write_template),
                "qa_template": len(self.qa_template),
                "judge_template": len(self.judge_template),
                "qwen_instruction": len(self.qwen_instruction),
            },
            "contract_sha256": self.contract_sha256,
            "write_call_contract": {
                "logical_attempts": WRITE_MAX_ATTEMPTS,
                "retry_backoff_s": [2.0, 2.0, 2.0, 2.0],
                "temperature": 0.0,
                "thinking_budget": 0,
                "max_tokens": WRITE_MAX_TOKENS,
                "use_cache": False,
                "json_mode": True,
            },
            "retrieval": {
                "summary_top_k": SUMMARY_TOP_K,
                "dense_fact_top_k": DENSE_FACT_TOP_K,
                "bm25_fact_top_k": BM25_FACT_TOP_K,
                "candidate_cap": CANDIDATE_CAP,
                "rerank_top_k": RERANK_TOP_K,
            },
        }


def assert_prompt_contract() -> PromptContract:
    """Load and fail-closed validate all frozen MGTM-LME prompt authorities."""

    paths = {
        "mgtm_prompts": MGTM_PROMPT_SOURCE,
        "mgtm_memory": MGTM_MEMORY_SOURCE,
        "mgtm_clients": MGTM_CLIENT_SOURCE,
        "mgtm_runner": MGTM_RUNNER_SOURCE,
        "lme_dataset": LME_DATASET,
    }
    actual_files: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise PromptContractError(f"missing frozen authority: {path}")
        actual = _file_sha256(path)
        expected = EXPECTED_FILE_SHA256[name]
        if actual != expected:
            raise PromptContractError(
                f"frozen authority drift for {name}: expected {expected}, got {actual}"
            )
        actual_files[name] = actual

    values = {
        "write_template": _literal_assignment(
            MGTM_PROMPT_SOURCE, "WRITE_PROMPT"
        ),
        "qa_template": _literal_assignment(MGTM_PROMPT_SOURCE, "QA_SYSTEM_PROMPT"),
        "judge_template": _literal_assignment(
            MGTM_PROMPT_SOURCE, "JUDGE_PROMPT"
        ),
        "qwen_instruction": _literal_assignment(
            MGTM_PROMPT_SOURCE, "RERANK_INSTRUCTION"
        ),
    }

    actual_prompts: dict[str, str] = {}
    for name, value in values.items():
        digest = sha256_text(value)
        if digest != EXPECTED_PROMPT_SHA256[name]:
            raise PromptContractError(
                f"prompt drift for {name}: expected {EXPECTED_PROMPT_SHA256[name]}, got {digest}"
            )
        if len(value) != EXPECTED_PROMPT_LENGTHS[name]:
            raise PromptContractError(
                f"prompt length drift for {name}: {len(value)}"
            )
        actual_prompts[name] = digest

    contract_payload = {
        "file_sha256": actual_files,
        "prompt_sha256": actual_prompts,
        "render_policy": "release_write_prompt_direct_format_no_policy_insertion",
        "qa_call": "release.longmemeval.run._answer compatible",
        "judge_call": "release.longmemeval.run._judge compatible",
    }
    return PromptContract(
        write_template=values["write_template"],
        qa_template=values["qa_template"],
        judge_template=values["judge_template"],
        qwen_instruction=values["qwen_instruction"],
        file_sha256=actual_files,
        prompt_sha256=actual_prompts,
        contract_sha256=gm.canonical_sha256(contract_payload),
    )


def assert_live_prompt_contract(contract: PromptContract) -> None:
    """Cheap per-call gate over the sole source of the three prompt literals."""

    actual_file = _file_sha256(MGTM_PROMPT_SOURCE)
    if actual_file != contract.file_sha256["mgtm_prompts"]:
        raise PromptContractError("MGTM prompt source changed after preflight")
    for contract_name, literal_name in (
        ("write_template", "WRITE_PROMPT"),
        ("qa_template", "QA_SYSTEM_PROMPT"),
        ("judge_template", "JUDGE_PROMPT"),
    ):
        actual = _literal_assignment(MGTM_PROMPT_SOURCE, literal_name)
        if sha256_text(actual) != contract.prompt_sha256[contract_name]:
            raise PromptContractError(f"live prompt drift for {contract_name}")


def render_lme_session(
    source_session_id: str,
    date: str,
    turns: Sequence[Mapping[str, Any]],
) -> str:
    """Byte-match MGTM ``_render_lme_session_batch`` for one session."""

    lines = [f"[Session {source_session_id} at {date}]"]
    for turn in turns:
        role = str(turn.get("role", "user"))
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines)


def render_write_prompt(contract: PromptContract, batch: str) -> str:
    """Render the exact MGTM-7935 LME writer prompt for one physical session."""

    return contract.write_template.format(
        unit_count=1,
        unit_label="session",
        batch=batch,
    )


def _parse_writer_json(raw: Any) -> dict[str, Any] | None:
    """Mirror MGTM's permissive first-object parser without changing prompts."""

    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _fact_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("statement", "text", "content", "fact"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


def write_one_session(
    *,
    contract: PromptContract,
    source_session_id: str,
    date: str,
    turns: Sequence[Mapping[str, Any]],
    chat_fn: ChatFn,
    write_model: str = "gemini-2.5-flash",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run MGTM's exact LME writer request contract for one session."""

    # Re-run the source gate at the live-call boundary.  A caller cannot keep a
    # stale PromptContract object while editing a source prompt underneath it.
    assert_live_prompt_contract(contract)
    current = contract
    batch = render_lme_session(source_session_id, date, turns)
    prompt = render_write_prompt(current, batch)
    last_error = "unknown writer failure"
    request_hashes: list[str] = []
    physical_attempts: list[dict[str, Any]] = []
    for attempt in range(WRITE_MAX_ATTEMPTS):
        request_hashes.append(sha256_text(prompt))
        try:
            raw = chat_fn(
                prompt,
                model=write_model,
                temperature=0.0,
                max_tokens=WRITE_MAX_TOKENS,
                thinking_budget=0,
                tag="write",
                use_cache=False,
                json_mode=True,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-300:]}"
            physical_attempts.append(
                {
                    "ordinal": attempt + 1,
                    "canonical_request_sha256": request_hashes[-1],
                    "status": "transport_error",
                    "error_type": type(exc).__name__,
                }
            )
            if isinstance(exc, NonRetryableProviderError):
                raise
        else:
            physical_attempts.append(
                {
                    "ordinal": attempt + 1,
                    "canonical_request_sha256": request_hashes[-1],
                    "status": "response_received",
                    "response_sha256": sha256_text(str(raw or "")),
                }
            )
            value = _parse_writer_json(raw)
            if value is None:
                last_error = "writer returned no valid JSON object"
            else:
                summary = str(value.get("batch_summary", "") or "").strip()
                raw_facts = value.get("atomic_facts")
                if not isinstance(raw_facts, list):
                    last_error = "writer returned atomic_facts with a non-list type"
                else:
                    facts = [_fact_text(item) for item in raw_facts]
                    facts = [fact for fact in facts if fact]
                    if summary:
                        return {
                            "source_session_id": str(source_session_id),
                            "date": str(date),
                            "batch_summary": summary,
                            "atomic_facts": facts,
                            "writer_attempts": attempt + 1,
                            "write_prompt_sha256": sha256_text(prompt),
                            "write_prompt_contract_sha256": current.contract_sha256,
                            "request_sha256s": request_hashes,
                            "physical_attempts": physical_attempts,
                            "rendered_batch_sha256": sha256_text(batch),
                        }
                    last_error = "writer returned an empty batch_summary"
        if attempt + 1 < WRITE_MAX_ATTEMPTS:
            sleep_fn(WRITE_RETRY_BACKOFF_S)
    raise gm.GraphMemoryError(
        f"MGTM-compatible write failed after {WRITE_MAX_ATTEMPTS} calls: {last_error}"
    )


def _iso_date(date_text: str) -> str:
    match = re.match(r"\s*(\d{4})[/-](\d{2})[/-](\d{2})", str(date_text))
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""


def build_session_graph_from_packets(
    packets: Sequence[Mapping[str, Any]],
    embed_fn: EmbedFn,
    *,
    contract: PromptContract,
    display_text_fn: DisplayTextFn | None = None,
) -> dict[str, Any]:
    """Build the CreaMem typed graph from exact-prompt MGTM writer outputs."""

    summaries: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    normalized_packets: list[dict[str, Any]] = []
    provenance_map: dict[str, Any] = {}
    fact_counter = 0
    for ordinal, packet in enumerate(packets, 1):
        source_session_id = str(packet["source_session_id"])
        source_position = int(packet.get("source_position", ordinal - 1))
        marker = f"D{ordinal}:0"
        summary_id = f"S{ordinal:03d}"
        date = str(packet.get("date", ""))
        turn_count = int(packet.get("turn_count", 0))
        provenance_map[marker] = {
            "kind": "physical_session",
            "source_session_id": source_session_id,
            "source_position": source_position,
            "date": date,
            "turn_count": turn_count,
            "mapping_policy": "all_writer_facts_inherit_their_single_input_session",
        }
        packet_facts: list[dict[str, Any]] = []
        for text in packet.get("atomic_facts", []):
            fact_counter += 1
            fact_id = f"F{fact_counter:04d}"
            fact_text = str(text).strip()
            if not fact_text:
                continue
            display_text, temporally_grounded = (
                display_text_fn(fact_text)
                if display_text_fn is not None
                else (fact_text, False)
            )
            packet_facts.append({"fact": fact_text, "source_dia_ids": [marker]})
            facts.append(
                {
                    "id": fact_id,
                    "node_type": "atomic_fact",
                    # The release MGTM harness embeds the raw writer fact, but
                    # indexes/reranks/renders its deterministic temporal display
                    # form.  Keeping both fields makes that asymmetry explicit.
                    "text": fact_text,
                    "raw_text": fact_text,
                    "display_text": str(display_text),
                    "temporally_grounded": bool(temporally_grounded),
                    "source_dia_ids": [marker],
                    "session_id": ordinal,
                    "source_session_id": source_session_id,
                    "source_position": source_position,
                    "timestamp": date,
                    "iso_date": _iso_date(date),
                    "embedding": [],
                }
            )
            edges.append(
                {
                    "edge_type": "SUPPORTS",
                    "source_id": summary_id,
                    "target_id": fact_id,
                }
            )
        summaries.append(
            {
                "id": summary_id,
                "node_type": "session_packet",
                "text": str(packet["batch_summary"]).strip(),
                "session_ids": [ordinal],
                "source_session_id": source_session_id,
                "source_position": source_position,
                "timestamp": date,
                "iso_date": _iso_date(date),
                "source_dia_ids": [marker],
                "embedding": [],
            }
        )
        normalized_packets.append(
            {
                "session_id": ordinal,
                "source_session_id": source_session_id,
                "source_position": source_position,
                "timestamp": date,
                "iso_date": _iso_date(date),
                "turn_count": turn_count,
                "source_dia_ids": [marker],
                "batch_summary": str(packet["batch_summary"]).strip(),
                "atomic_facts": packet_facts,
                "writer_attempts": int(packet.get("writer_attempts", 1)),
                "writer_prompt_sha256": str(packet["write_prompt_sha256"]),
                "write_prompt_contract_sha256": str(
                    packet["write_prompt_contract_sha256"]
                ),
                "rendered_batch_sha256": str(packet["rendered_batch_sha256"]),
            }
        )

    texts = [node["text"] for node in summaries] + [node["raw_text"] for node in facts]
    # MGTM's frozen Gemini Embedding 001 request supplies neither task_type nor
    # output dimensionality.  The raw fact strings (not display_text) are used.
    vectors = list(embed_fn(texts))
    if len(vectors) != len(texts):
        raise gm.GraphMemoryError("embedding count mismatch while building LME graph")
    checked: list[list[float]] = []
    dimension: int | None = None
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        dimension = dimension or len(values)
        if not values or len(values) != dimension or not all(map(math.isfinite, values)):
            raise gm.GraphMemoryError(f"invalid embedding at position {index}")
        checked.append(values)
    split = len(summaries)
    for node, vector in zip(summaries, checked[:split], strict=True):
        node["embedding"] = vector
    for node, vector in zip(facts, checked[split:], strict=True):
        node["embedding"] = vector

    graph: dict[str, Any] = {
        "schema_version": gm.GRAPH_SCHEMA_VERSION,
        "adapter_schema_version": SCHEMA_VERSION,
        "graph_variant": "session",
        "architecture": {
            "representation": "typed_many_to_many_event_graph",
            "input_scope": "dialogue_text_only",
            "fact_policy": "immutable_atomic_facts_are_source_of_truth",
            "summary_policy": "derived_routing_views_do_not_own_facts",
            "write_unit": "one_complete_physical_session",
            "write_prompt_environment": "mgtm_7935_lme_byte_exact",
            "writer_output_schema": "historical_string_facts",
            "provenance_adaptation": "single_session_marker_Dordinal_0",
        },
        "prompt_sha256": {
            "mgtm_write_template": contract.prompt_sha256["write_template"],
            "mgtm_qa_template": contract.prompt_sha256["qa_template"],
            "mgtm_judge_template": contract.prompt_sha256["judge_template"],
        },
        "prompt_contract_sha256": contract.contract_sha256,
        "session_packets": normalized_packets,
        "provenance_map": provenance_map,
        "summary_nodes": summaries,
        "fact_nodes": facts,
        "edges": edges,
        "stats": {
            "sessions": len(summaries),
            "session_summary_nodes": len(summaries),
            "event_view_nodes": 0,
            "facts": len(facts),
            "edges": len(edges),
            "writer_calls": sum(int(row["writer_attempts"]) for row in normalized_packets),
            "writer_retries": sum(
                max(0, int(row["writer_attempts"]) - 1) for row in normalized_packets
            ),
            "embedding_dim": int(dimension or 0),
        },
    }
    graph["content_fingerprint"] = gm.canonical_sha256(
        {
            "summaries": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in summaries
            ],
            "facts": [
                {key: value for key, value in node.items() if key != "embedding"}
                for node in facts
            ],
            "edges": edges,
            "provenance_map": provenance_map,
            "prompt_contract_sha256": contract.contract_sha256,
        }
    )
    return graph


class QwenGraphMemoryIndex(gm.GraphMemoryIndex):
    """CreaMem three-channel index with MGTM's Qwen score API."""

    def __init__(
        self,
        graph: Mapping[str, Any],
        *,
        reranker: RerankFn,
        instruction: str,
    ) -> None:
        # Shallow structural projection: embeddings remain shared, while only
        # fact dictionaries receive MGTM's retrieval/display text.  The Online
        # causal graph itself remains raw and is never mutated.
        retrieval_graph = dict(graph)
        retrieval_graph["fact_nodes"] = [
            {
                **dict(fact),
                "text": str(fact.get("display_text", fact.get("text", ""))),
            }
            for fact in graph.get("fact_nodes", [])
            if isinstance(fact, Mapping)
        ]
        super().__init__(
            retrieval_graph,
            summary_top_k=SUMMARY_TOP_K,
            dense_fact_top_k=DENSE_FACT_TOP_K,
            bm25_fact_top_k=BM25_FACT_TOP_K,
            candidate_cap=CANDIDATE_CAP,
            rerank_top_k=RERANK_TOP_K,
        )
        self._score_reranker = reranker
        self._instruction = instruction

    def _rerank(  # type: ignore[override]
        self,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        rerank_chat: ChatFn,
        *,
        max_attempts: int = 2,
    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
        del rerank_chat, max_attempts
        target = min(self.rerank_top_k, len(candidates))
        if target == 0:
            return [], {
                "provider": "dashscope",
                "model": "qwen3-rerank",
                "attempts": 0,
                "fallback": False,
                "error": None,
            }
        documents = [str(item["fact"]["text"]) for item in candidates]
        scores = [
            float(value)
            for value in self._score_reranker(question, documents, self._instruction)
        ]
        if len(scores) != len(candidates) or not all(map(math.isfinite, scores)):
            raise gm.GraphMemoryError("Qwen reranker returned invalid aligned scores")
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], index),
        )
        selected = [candidates[index] for index in ranked[:target]]
        return selected, {
            "provider": "dashscope",
            "model": "qwen3-rerank",
            "instruction_sha256": sha256_text(self._instruction),
            "attempts": 1,
            "fallback": False,
            "error": None,
            "candidate_scores": [
                {
                    "fact_id": str(candidates[index]["fact"]["id"]),
                    "score": scores[index],
                }
                for index in ranked
            ],
        }


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in output:
            output.append(text)
    return output


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9_]+", str(value))}


def _selected_fact_ids(
    base: Mapping[str, Any], *, final_k: int = RERANK_TOP_K
) -> list[str]:
    trace = base.get("retrieval_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("missing retrieval_trace")
    rerank = trace.get("rerank")
    if not isinstance(rerank, Mapping):
        raise ValueError("missing rerank trace")
    raw = rerank.get("selected_fact_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("selected_fact_ids must be a sequence")
    if isinstance(final_k, bool) or not isinstance(final_k, int) or final_k <= 0:
        raise ValueError("final_k must be a positive integer")
    selected = _ordered_unique(list(raw))[:final_k]
    if not selected:
        raise ValueError("empty final rerank selection")
    return selected


def _candidate_view_ids(
    base: Mapping[str, Any],
    graph: Mapping[str, Any],
    selected_ids: Sequence[str],
) -> list[str]:
    selected = set(selected_ids)
    views = {
        str(node["id"]): node
        for node in graph.get("summary_nodes", [])
        if isinstance(node, Mapping) and node.get("node_type") == "event_view"
    }
    ordered: list[str] = []
    trace = base.get("retrieval_trace", {})
    for item in trace.get("top_summaries", []) if isinstance(trace, Mapping) else []:
        if not isinstance(item, Mapping):
            continue
        view_id = str(item.get("summary_id", ""))
        view = views.get(view_id)
        if view is None:
            continue
        if (
            selected.intersection(map(str, view.get("member_fact_ids", [])))
            and view_id not in ordered
        ):
            ordered.append(view_id)
    remaining = sorted(
        (
            -len(selected.intersection(map(str, view.get("member_fact_ids", [])))),
            view_id,
        )
        for view_id, view in views.items()
        if view_id not in ordered
        and len(selected.intersection(map(str, view.get("member_fact_ids", [])))) >= 2
    )
    ordered.extend(view_id for _, view_id in remaining)
    return ordered


def _render_selected_facts(
    question: str,
    base: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    event_view_limit: int,
    final_k: int,
) -> tuple[str, dict[str, Any]]:
    # This is a parameterized copy of semantic_context_v8._render_online_context.
    # Only the frozen final_k changes (20 -> 30); all rendering caps, ordering,
    # hashes, and provenance behavior remain byte-identical.
    from creamem3.semantic_context_v8 import _grounded_claims

    selected_ids = _selected_fact_ids(base, final_k=final_k)
    rerank = base.get("retrieval_trace", {}).get("rerank", {})
    selection_source = (
        "frozen_fallback"
        if isinstance(rerank, Mapping) and bool(rerank.get("fallback"))
        else "frozen_model"
    )
    selected_set = set(selected_ids)
    fact_by_id = {str(node["id"]): node for node in graph.get("fact_nodes", [])}
    missing = [fact_id for fact_id in selected_ids if fact_id not in fact_by_id]
    if missing:
        raise ValueError(f"selected facts missing from graph: {missing}")
    session_by_id = {
        int(node["session_ids"][0]): node
        for node in graph.get("summary_nodes", [])
        if isinstance(node, Mapping)
        and node.get("node_type") == "session_packet"
        and node.get("session_ids")
    }
    view_by_id = {
        str(node["id"]): node
        for node in graph.get("summary_nodes", [])
        if isinstance(node, Mapping) and node.get("node_type") == "event_view"
    }
    lines: list[str] = []
    rendered_views: list[str] = []
    rendered_claims: list[dict[str, Any]] = []
    rendered_view_chars = 0
    if event_view_limit:
        for view_id in _candidate_view_ids(base, graph, selected_ids):
            if len(rendered_views) >= event_view_limit:
                break
            view = view_by_id[view_id]
            header = (
                f"[Selected cross-session claims | {view.get('view_type', '')} | "
                f"sessions {', '.join(str(value) for value in view.get('session_ids', []))}] "
            )
            separator_chars = 1 if rendered_views else 0
            remaining = (
                300 * event_view_limit
                - rendered_view_chars
                - separator_chars
                - len(header)
            )
            if remaining <= 0:
                continue
            eligible = _grounded_claims(
                view,
                selected_set,
                question,
                max_claim_chars=260,
            )
            chosen: list[dict[str, Any]] = []
            claim_chars = 0
            for claim in eligible:
                claim_text = str(claim["text"])
                separator = 1 if chosen else 0
                if claim_chars + separator + len(claim_text) > remaining:
                    continue
                chosen.append(claim)
                claim_chars += separator + len(claim_text)
                if len(chosen) >= 3:
                    break
            if not chosen:
                continue
            rendered_line = header + " ".join(str(claim["text"]) for claim in chosen)
            lines.append(rendered_line)
            rendered_views.append(view_id)
            rendered_view_chars += separator_chars + len(rendered_line)
            for claim in chosen:
                rendered_claims.append(
                    {
                        "view_id": view_id,
                        "claim_index": int(claim["index"]),
                        "text_sha256": sha256_text(str(claim["text"])),
                        "support_fact_ids": list(claim["support_fact_ids"]),
                    }
                )

    by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    session_order: list[int] = []
    for fact_id in selected_ids:
        fact = fact_by_id[fact_id]
        session_id = int(fact["session_id"])
        if session_id not in by_session:
            session_order.append(session_id)
        by_session[session_id].append(fact)
    rendered_summaries: list[str] = []
    for session_id in session_order:
        summary = session_by_id[session_id]
        rendered_summaries.append(str(summary["id"]))
        lines.append(
            f"[Session summary | session {session_id} | {summary.get('timestamp', '')}; "
            f"ISO date {summary.get('iso_date', '')}] {summary.get('text', '')}"
        )
        for fact in by_session[session_id]:
            sources = ", ".join(map(str, fact.get("source_dia_ids", [])))
            lines.append(
                f"  - [{sources}] "
                f"{fact.get('display_text', fact.get('text', ''))}"
            )
    text = "\n".join(lines)
    return text, {
        "selected_fact_ids": selected_ids,
        "selected_fact_count": len(selected_ids),
        "selection_source": selection_source,
        "selection_mode": f"frozen_model_top{final_k}",
        "new_reranker_called": False,
        "rendered_event_view_ids": rendered_views,
        "rendered_event_view_count": len(rendered_views),
        "rendered_event_view_claims": rendered_claims,
        "rendered_event_claim_count": len(rendered_claims),
        "rendered_event_view_chars": rendered_view_chars,
        "rendered_session_summary_ids": rendered_summaries,
        "grounded_claims_only": all(
            set(row["support_fact_ids"]).issubset(selected_set)
            for row in rendered_claims
        ),
        "context_sha256": sha256_text(text),
    }


def materialize_operation_context(
    *,
    operation: str,
    question: str,
    base_retrieved: str,
    retrieval_trace: Mapping[str, Any],
    graph: Mapping[str, Any],
    selected_top_k: int = RERANK_TOP_K,
) -> tuple[str, dict[str, Any]]:
    """Apply frozen OpCtx v3 routing while retaining the LME Top-30 facts."""

    from creamem3.online_operation_context_v8 import (
        BOUNDED_VIEWS_2,
        OPERATION_TO_CONTEXT_POLICY,
        ORIGINAL_FULL,
        ROUTE_ONLY,
        _assert_sanitized_base,
        context_policy_for_operation,
    )

    policy = context_policy_for_operation(operation)
    base = {
        "question": str(question),
        "retrieved": str(base_retrieved),
        "retrieval_trace": copy.deepcopy(dict(retrieval_trace)),
    }
    _assert_sanitized_base(base)
    if policy == ORIGINAL_FULL:
        context = str(base_retrieved)
        rerank = retrieval_trace.get("rerank", {})
        assembly = retrieval_trace.get("assembly", {})
        selected = (
            [str(value) for value in rerank.get("selected_fact_ids", [])]
            if isinstance(rerank, Mapping)
            else []
        )
        rendered_views = (
            [str(value) for value in assembly.get("rendered_event_view_ids", [])]
            if isinstance(assembly, Mapping)
            else []
        )
        rendered_summaries = (
            [str(value) for value in assembly.get("rendered_session_summary_ids", [])]
            if isinstance(assembly, Mapping)
            else []
        )
        branch = {
            "selected_fact_ids": selected,
            "selected_fact_count": len(selected),
            "selection_mode": "frozen_original_online",
            "new_reranker_called": False,
            "rendered_event_view_ids": rendered_views,
            "rendered_event_view_count": len(rendered_views),
            "rendered_session_summary_ids": rendered_summaries,
            "reused_original_context": True,
            "grounded_claims_only": None,
        }
    else:
        event_limit = 2 if policy == BOUNDED_VIEWS_2 else 0
        context, branch = _render_selected_facts(
            str(question),
            base,
            graph,
            event_view_limit=event_limit,
            final_k=selected_top_k,
        )
        branch["reused_original_context"] = False
    audit = {
        **branch,
        "schema_version": SCHEMA_VERSION,
        "policy_name": "online_semantic_operation_context_v3_lme_top30",
        "source_policy_name": "online_semantic_operation_context_v3",
        "operation": operation,
        "context_policy": policy,
        "source_operation_mapping_sha256": gm.canonical_sha256(
            OPERATION_TO_CONTEXT_POLICY
        ),
        "input_context_sha256": sha256_text(str(base_retrieved)),
        "context_sha256": sha256_text(context),
        "lme_adaptation": f"frozen_top{selected_top_k}_instead_of_locomo_top20",
        "new_reranker_called": False,
        "gold_or_category_visible": False,
        "single_context_fixed_before_answer": True,
        "bounded_event_view_limit": (
            2 if policy == BOUNDED_VIEWS_2 else (0 if policy == ROUTE_ONLY else None)
        ),
    }
    return context, audit


__all__ = [
    "BM25_FACT_TOP_K",
    "CANDIDATE_CAP",
    "DENSE_FACT_TOP_K",
    "EXPECTED_FILE_SHA256",
    "EXPECTED_PROMPT_SHA256",
    "LME_DATASET",
    "MGTM_PROJECT_ROOT",
    "MGTM_PROMPT_SOURCE",
    "MGTM_QWEN_SOURCE",
    "NonRetryableProviderError",
    "PromptContract",
    "PromptContractError",
    "QwenGraphMemoryIndex",
    "RERANK_TOP_K",
    "SCHEMA_VERSION",
    "SUMMARY_TOP_K",
    "WRITE_MAX_ATTEMPTS",
    "WRITE_RETRY_BACKOFF_S",
    "assert_prompt_contract",
    "assert_live_prompt_contract",
    "build_session_graph_from_packets",
    "materialize_operation_context",
    "render_lme_session",
    "render_write_prompt",
    "sha256_text",
    "write_one_session",
]

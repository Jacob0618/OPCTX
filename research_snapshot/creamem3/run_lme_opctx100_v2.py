"""Run LongMemEval OpCtx with 3-session frontiers and normalized 768-d vectors.

This is a fresh-root protocol.  It never reads v1/3,072-dimensional caches.
Without ``--allow-live`` it performs only the deterministic preflight.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import online_memory_v4
from creamem3 import semantic_query_planner_v6 as planner
from creamem3 import strict_semantic_planner_v8 as strict_planner
from creamem3 import lme_opctx_v2 as adapter
from creamem3 import run_lme_opctx100_v1 as legacy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = legacy.MANIFEST
MANIFEST_SHA256 = legacy.MANIFEST_SHA256
DEFAULT_OUTPUT = ROOT / "creamem3" / "results_lme_opctx100_v2_b3_d768"
DEFAULT_CACHE = Path(r"D:\creamem3_cache\lme_opctx100_v2_b3_d768")
DEFAULT_PREFLIGHT = ROOT / "creamem3" / "audit" / "lme_opctx100_v2" / "preflight.json"
RUN_SCHEMA = "creamem3.lme_opctx100.runner.v2_b3_d768"
EMBED_CACHE_SCHEMA = "creamem3.gemini_embedding_cache.v2_d768_l2"

PhysicalChat = legacy.PhysicalChat
_atomic_json = legacy._atomic_json
_canonical = legacy._canonical
_durable_retry = legacy._durable_retry
_judge = legacy._judge
_load_json = legacy._load_json
_nonempty = legacy._nonempty
_operations = legacy._operations
_require_pass = legacy._require_pass
_sealed_payload = legacy._sealed_payload
_sha_file = legacy._sha_file
_verify_sealed_payload = legacy._verify_sealed_payload


def _save_graph(
    path: Path,
    graph: Mapping[str, Any],
    config_sha256: str,
    question_binding_sha256: str,
) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".seal.json").exists():
        raise RuntimeError(f"refusing to overwrite sealed v2 graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": RUN_SCHEMA,
        "config_sha256": config_sha256,
        "question_binding_sha256": question_binding_sha256,
        "graph": dict(graph),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(envelope, handle, protocol=5)
    temporary.replace(path)
    _atomic_json(
        path.with_suffix(path.suffix + ".seal.json"),
        {
            "schema_version": RUN_SCHEMA,
            "file_sha256": _sha_file(path),
            "config_sha256": config_sha256,
            "question_binding_sha256": question_binding_sha256,
            "content_fingerprint": graph.get("content_fingerprint"),
        },
    )


def _load_graph(
    path: Path,
    config_sha256: str,
    question_binding_sha256: str,
) -> dict[str, Any] | None:
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if not path.exists() and not seal_path.exists():
        return None
    if not path.is_file() or not seal_path.is_file():
        raise RuntimeError(f"partial v2 graph checkpoint: {path}")
    seal = _load_json(seal_path)
    if (
        seal.get("schema_version") != RUN_SCHEMA
        or seal.get("config_sha256") != config_sha256
        or seal.get("question_binding_sha256") != question_binding_sha256
        or seal.get("file_sha256") != _sha_file(path)
    ):
        raise RuntimeError(f"v2 graph checkpoint seal mismatch: {path}")
    with path.open("rb") as handle:
        envelope = legacy._NoGlobalsUnpickler(handle).load()
    if (
        envelope.get("schema_version") != RUN_SCHEMA
        or envelope.get("config_sha256") != config_sha256
        or envelope.get("question_binding_sha256") != question_binding_sha256
        or not isinstance(envelope.get("graph"), dict)
    ):
        raise RuntimeError(f"v2 graph checkpoint envelope mismatch: {path}")
    graph = envelope["graph"]
    if graph.get("content_fingerprint") != seal.get("content_fingerprint"):
        raise RuntimeError(f"v2 graph fingerprint mismatch: {path}")
    return graph


def _call_estimate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    physical = sum(len(item["haystack_sessions"]) for item in selected)
    frontiers = sum(math.ceil(len(item["haystack_sessions"]) / adapter.FRONTIER_SIZE) for item in selected)
    questions = len(selected)
    online = frontiers - questions
    logical = frontiers + online + questions * 3
    return {
        "questions": questions,
        "physical_sessions": physical,
        "writer_frontiers": frontiers,
        "writer_logical_calls": frontiers,
        "online_fusion_logical_calls": online,
        "planner_logical_calls": questions,
        "qa_logical_calls": questions,
        "judge_logical_calls": questions,
        "generative_logical_success_calls": logical,
        "generative_physical_attempt_upper_bound": logical * legacy.MAX_GENERATION_ATTEMPTS,
        "qwen_rerank_logical_calls": questions,
        "embedding": {
            "model": adapter.EMBEDDING_MODEL,
            "output_dimensionality": adapter.EMBEDDING_DIM,
            "normalization": adapter.EMBEDDING_NORMALIZATION,
            "http_calls": "data/cache dependent; <=100 texts per physical batch",
        },
    }


def _run_config(contract: adapter.PromptContract) -> dict[str, Any]:
    source_paths = {
        "runner": Path(__file__).resolve(),
        "adapter": Path(adapter.__file__).resolve(),
        "online_memory": Path(online_memory_v4.__file__).resolve(),
        # v2/v4 deliberately reuse stable parser/validator/checkpoint helpers.
        # Their bytes are therefore part of the resume identity as well.
        "legacy_lme_adapter_dependency": Path(adapter.legacy.__file__).resolve(),
        "legacy_runner_dependency": Path(legacy.__file__).resolve(),
        "legacy_online_memory_dependency": Path(online_memory_v4.legacy.__file__).resolve(),
        "graph_memory": Path(gm.__file__).resolve(),
        "semantic_planner": Path(planner.__file__).resolve(),
        "strict_semantic_planner": Path(strict_planner.__file__).resolve(),
        # lme_opctx_v1 imports these lazily inside materialize_operation_context.
        "operation_context_dependency": Path(adapter.legacy.__file__).resolve().with_name(
            "online_operation_context_v8.py"
        ),
        "semantic_context_dependency": Path(adapter.legacy.__file__).resolve().with_name(
            "semantic_context_v8.py"
        ),
    }
    return {
        "schema_version": RUN_SCHEMA,
        "manifest_sha256": MANIFEST_SHA256,
        "dataset_sha256": contract.file_sha256["lme_dataset"],
        "prompt_contract": contract.receipt(),
        "models": {
            "writer": "gemini-2.5-flash",
            "answerer": "gemini-2.5-flash",
            "judge": "gemini-3-flash-preview",
            "embedding": {
                "model": adapter.EMBEDDING_MODEL,
                "output_dimensionality": adapter.EMBEDDING_DIM,
                "transport_field": "outputDimensionality_compat",
                "task_type_sent": None,
                "normalization": adapter.EMBEDDING_NORMALIZATION,
            },
            "reranker": "qwen3-rerank",
        },
        "architecture": {
            "physical_sessions_per_writer_frontier": 3,
            "final_frontier_policy": "one_or_two_without_padding",
            "causality_granularity": "writer_frontier_not_physical_session",
            "first_frontier_online_call": False,
            "online_calls": "frontiers_minus_one",
            "online": "creamem3.online_memory_v4",
            "reader": "online_semantic_operation_context_v3",
            "lme_final_k": 30,
            "write_prompt_sha256": contract.prompt_sha256["write_template"],
            "qa_prompt_sha256": contract.prompt_sha256["qa_template"],
            "judge_prompt_sha256": contract.prompt_sha256["judge_template"],
            "online_prompt_sha256": online_memory_v4.ONLINE_FRONTIER_PROMPT_SHA256,
        },
        "implementation_source_sha256": {
            name: _sha_file(path) for name, path in source_paths.items()
        },
        "cache_contract": {
            "fresh_root_required": True,
            "embedding_schema": EMBED_CACHE_SCHEMA,
            "v1_3072_cache_compatible": False,
        },
    }


def _assert_model_stack(stack: Any, config: Mapping[str, Any]) -> None:
    models = config["models"]
    expected = {
        "writer": str(models["writer"]),
        "answerer": str(models["answerer"]),
        "judge": str(models["judge"]),
        "embedding": str(models["embedding"]["model"]),
        "reranker": str(models["reranker"]),
    }
    actual = {name: str(getattr(stack, name, "")) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"live ModelStack differs from v2 run_config: {actual} != {expected}"
        )


def run_preflight(output: Path = DEFAULT_PREFLIGHT) -> dict[str, Any]:
    contract = adapter.assert_prompt_contract()
    _, manifest, selected = legacy._load_inputs(contract)
    order_rows = [legacy._ordered_session_specs(item)[1] for item in selected]
    receipt = {
        "schema_version": "creamem3.lme_opctx100.preflight.v2_b3_d768",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "network_calls": 0,
        "model_calls": 0,
        "manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA256, "count": manifest["count"]},
        "prompt_contract": contract.receipt(),
        "prompt_hashes_unchanged_from_v1": {
            "write": contract.prompt_sha256["write_template"],
            "qa": contract.prompt_sha256["qa_template"],
            "judge": contract.prompt_sha256["judge_template"],
        },
        "causal_order": {
            "policy": "stable_timestamp_then_source_position_then_consecutive_chunk_3",
            "questions_reordered": sum(bool(row["reordered"]) for row in order_rows),
        },
        "call_estimate": _call_estimate(selected),
        "run_config": _run_config(contract),
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    _atomic_json(output, receipt)
    return receipt


def _unit_normalize(values: Sequence[Any]) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != adapter.EMBEDDING_DIM or not all(math.isfinite(value) for value in vector):
        raise RuntimeError(
            f"Gemini embedding must be a finite aligned 768-d vector; received {len(vector)}"
        )
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Gemini embedding must have a positive finite norm")
    return [value / norm for value in vector]


def _client_types(release_clients: ModuleType) -> tuple[type[Any], type[Any]]:
    generation_type, retrieval_base = legacy._client_types(release_clients)

    class StrictRetrieval768(retrieval_base):  # type: ignore[misc, valid-type]
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            output: list[list[float] | None] = [None] * len(texts)
            missing: list[int] = []
            keys: list[str] = []
            cache_dir = self.embedding_cache / "d768_l2_v1"
            cache_dir.mkdir(exist_ok=True)
            for index, text in enumerate(texts):
                binding = {
                    "schema_version": EMBED_CACHE_SCHEMA,
                    "model": self.stack.embedding,
                    "output_dimensionality": adapter.EMBEDDING_DIM,
                    "task_type_sent": None,
                    "normalization": adapter.EMBEDDING_NORMALIZATION,
                    "text_sha256": adapter.sha256_text(str(text)),
                }
                key = self._cache_key([binding, str(text)])
                keys.append(key)
                path = cache_dir / f"{key}.json"
                if path.exists():
                    saved = _verify_sealed_payload(_load_json(path), path)
                    vector = saved.get("vector")
                    if saved.get("binding") != binding or saved.get("vector_sha256") != _canonical(vector):
                        raise RuntimeError("768 embedding cache binding mismatch")
                    output[index] = _unit_normalize(vector)
                else:
                    missing.append(index)
            for start in range(0, len(missing), 100):
                batch = missing[start : start + 100]
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.stack.embedding}",
                            "content": {"parts": [{"text": str(texts[index])}]},
                            # The current batch endpoint silently ignores the
                            # newer nested embedContentConfig form. Its
                            # documented compatibility field returns the
                            # requested dimensions; exact length is validated.
                            "outputDimensionality": adapter.EMBEDDING_DIM,
                        }
                        for index in batch
                    ]
                }
                response = self._post(
                    f"{release_clients.GEMINI_BASE}/models/{self.stack.embedding}:batchEmbedContents?key={self.gemini_api_key}",
                    payload,
                )
                embeddings = response.get("embeddings", [])
                if len(embeddings) != len(batch):
                    raise RuntimeError("Gemini embedding count mismatch")
                for index, embedding in zip(batch, embeddings):
                    vector = _unit_normalize(embedding.get("values", []))
                    output[index] = vector
                    binding = {
                        "schema_version": EMBED_CACHE_SCHEMA,
                        "model": self.stack.embedding,
                        "output_dimensionality": adapter.EMBEDDING_DIM,
                        "task_type_sent": None,
                        "normalization": adapter.EMBEDDING_NORMALIZATION,
                        "text_sha256": adapter.sha256_text(str(texts[index])),
                    }
                    _atomic_json(
                        cache_dir / f"{keys[index]}.json",
                        _sealed_payload(
                            {"binding": binding, "vector": vector, "vector_sha256": _canonical(vector)}
                        ),
                    )
            if any(vector is None for vector in output):
                raise RuntimeError("one or more 768 embeddings were not produced")
            return [list(vector) for vector in output if vector is not None]

    return generation_type, StrictRetrieval768


class EmbedCompat:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        unexpected = set(kwargs).difference({"task_type", "tag"})
        if unexpected:
            raise TypeError(f"unsupported embedding compatibility kwargs: {sorted(unexpected)}")
        vectors = self.client.embed(list(texts))
        if len(vectors) != len(texts) or any(len(vector) != adapter.EMBEDDING_DIM for vector in vectors):
            raise RuntimeError("embedding compatibility layer requires aligned 768-d vectors")
        self.calls.append(
            {
                "texts": len(texts),
                "input_sha256": _canonical(list(texts)),
                "requested_task_type": kwargs.get("task_type"),
                "requested_tag": kwargs.get("tag"),
                "provider_task_type_sent": None,
                "provider_output_dimensionality_sent": adapter.EMBEDDING_DIM,
                "normalization": adapter.EMBEDDING_NORMALIZATION,
            }
        )
        return vectors


def _frontier_binding(
    item: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    frontier: Mapping[str, Any],
    contract: adapter.PromptContract,
) -> dict[str, Any]:
    batch = adapter.render_lme_frontier(frontier)
    members = list(frontier["members"])
    prompt = adapter.render_write_prompt(contract, batch, unit_count=len(members))
    return {
        "schema_version": RUN_SCHEMA,
        "question_id": str(item["question_id"]),
        "manifest_sha256": MANIFEST_SHA256,
        "record_sha256": str(manifest_binding["record_sha256"]),
        "frontier_id": int(frontier["frontier_id"]),
        "frontier_size": len(members),
        "physical_sessions": [
            {
                "source_position": int(spec["source_position"]),
                "source_session_id": str(spec["source_session_id"]),
                "date": str(spec["date"]),
                "turn_count": len(spec["turns"]),
                "payload_sha256": _canonical(spec["turns"]),
            }
            for spec in members
        ],
        "rendered_batch_sha256": adapter.sha256_text(batch),
        "filled_write_prompt_sha256": adapter.sha256_text(prompt),
        "write_template_sha256": contract.prompt_sha256["write_template"],
        "prompt_contract_sha256": contract.contract_sha256,
    }


def _write_frontier_packet(
    *,
    item: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    frontier: Mapping[str, Any],
    question_cache: Path,
    contract: adapter.PromptContract,
    generation_client_type: type[Any],
    stack: Any,
) -> dict[str, Any]:
    binding = _frontier_binding(item, manifest_binding, frontier, contract)
    frontier_id = int(frontier["frontier_id"])
    path = question_cache / "writer_frontiers" / f"frontier-{frontier_id:04d}.json"
    failure_path = path.with_suffix(".failure.json")
    if path.exists():
        packet = _verify_sealed_payload(_load_json(path), path)
        if packet.get("binding") != binding:
            raise RuntimeError("writer frontier checkpoint binding mismatch")
        if not isinstance(packet.get("batch_summary"), str) or not packet["batch_summary"].strip():
            raise RuntimeError("writer frontier checkpoint has an empty summary")
        if not isinstance(packet.get("atomic_facts"), list):
            raise RuntimeError("writer frontier checkpoint facts are not a list")
        attempts = legacy._validate_physical_attempts(
            packet.get("transport_attempts"), same_canonical_body=True
        )
        if int(packet.get("writer_attempts", 0)) != len(attempts):
            raise RuntimeError("writer frontier checkpoint attempt count mismatch")
        if packet.get("write_prompt_sha256") != binding["filled_write_prompt_sha256"]:
            raise RuntimeError("writer frontier checkpoint prompt hash mismatch")
        if packet.get("rendered_batch_sha256") != binding["rendered_batch_sha256"]:
            raise RuntimeError("writer frontier checkpoint batch hash mismatch")
        return packet
    if failure_path.exists():
        failure = _verify_sealed_payload(_load_json(failure_path), failure_path)
        if failure.get("binding") != binding:
            raise RuntimeError("writer frontier failure checkpoint binding mismatch")
        attempts = legacy._validate_physical_attempts(
            failure.get("transport_attempts"), same_canonical_body=True
        )
        if failure.get("status") == "non_retryable_provider_error":
            if attempts[-1].get("status") != "non_retryable_provider_error":
                raise RuntimeError("writer frontier non-retryable evidence mismatch")
            raise adapter.NonRetryableProviderError(
                f"writer frontier non-retryable provider rejection already sealed: {failure_path}"
            )
        if len(attempts) != legacy.MAX_GENERATION_ATTEMPTS:
            raise RuntimeError("writer frontier exhausted checkpoint has an invalid attempt count")
        raise RuntimeError(f"writer frontier physical attempt cap already exhausted: {failure_path}")
    client = generation_client_type(question_cache / "generation", stack)
    chat = PhysicalChat(client, contract, default_model=stack.writer, stage=f"writer_frontier_{frontier_id:04d}")
    try:
        packet = adapter.write_one_frontier(
            contract=contract,
            frontier=frontier,
            chat_fn=chat,
            write_model=stack.writer,
        )
    except adapter.NonRetryableProviderError as exc:
        _atomic_json(
            failure_path,
            _sealed_payload(
                {
                    "schema_version": RUN_SCHEMA,
                    "binding": binding,
                    "status": "non_retryable_provider_error",
                    "transport_attempts": chat.attempts,
                    "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
                }
            ),
        )
        raise
    except Exception as exc:
        if len(chat.attempts) == legacy.MAX_GENERATION_ATTEMPTS:
            _atomic_json(
                failure_path,
                _sealed_payload(
                    {
                        "schema_version": RUN_SCHEMA,
                        "binding": binding,
                        "status": "exhausted",
                        "transport_attempts": chat.attempts,
                        "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
                    }
                ),
            )
        raise
    packet.update(
        {
            "schema_version": RUN_SCHEMA,
            "binding": binding,
            "transport_attempts": chat.attempts,
            "usage": client.usage.snapshot(),
        }
    )
    if int(packet["writer_attempts"]) != len(chat.attempts):
        raise RuntimeError("writer frontier attempt accounting mismatch")
    legacy._validate_physical_attempts(chat.attempts, same_canonical_body=True)
    _atomic_json(path, _sealed_payload(packet))
    return packet


class FrontierCheckpointChat:
    def __init__(
        self,
        *,
        question_cache: Path,
        question_binding: Mapping[str, Any],
        frontiers: Sequence[Mapping[str, Any]],
        client: Any,
        contract: adapter.PromptContract,
        default_model: str,
    ) -> None:
        self.root = question_cache / "view_frontiers"
        self.binding = dict(question_binding)
        self.frontiers = list(frontiers)
        self.client = client
        self.contract = contract
        self.default_model = default_model
        self.call_index = 0
        self.fatal = False

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        self.call_index += 1
        frontier_id = self.call_index + 1
        if self.fatal:
            raise RuntimeError("earlier Online frontier exhausted its attempt cap")
        frontier = self.frontiers[frontier_id - 1]
        binding = {
            **self.binding,
            "frontier_id": frontier_id,
            "frontier_binding_sha256": _canonical(frontier),
        }
        path = self.root / f"frontier-{frontier_id:04d}.json"
        chat = PhysicalChat(
            self.client,
            self.contract,
            default_model=self.default_model,
            stage=f"online_frontier_{frontier_id:04d}",
        )
        try:
            return str(_durable_retry(path, binding, chat, prompt, validator=_operations, **kwargs))
        except Exception:
            self.fatal = True
            raise


def _validate_frontier_graph_binding(
    graph: Mapping[str, Any],
    frontiers: Sequence[Mapping[str, Any]],
) -> None:
    packets = graph.get("session_packets", [])
    if len(packets) != len(frontiers):
        raise RuntimeError("frontier graph packet count mismatch")
    expected_positions: list[int] = []
    observed_positions: list[int] = []
    for ordinal, (packet, frontier) in enumerate(zip(packets, frontiers), 1):
        members = list(frontier["members"])
        expected_positions.extend(int(spec["source_position"]) for spec in members)
        physical = list(packet.get("physical_sessions", []))
        observed_positions.extend(int(row["source_position"]) for row in physical)
        if packet.get("session_id") != ordinal or packet.get("frontier_size") != len(members):
            raise RuntimeError(f"frontier graph identity mismatch at {ordinal}")
        marker = graph.get("provenance_map", {}).get(f"D{ordinal}:0", {})
        if marker.get("physical_sessions") != physical:
            raise RuntimeError(f"frontier provenance mismatch at {ordinal}")
    if observed_positions != expected_positions or len(set(observed_positions)) != len(observed_positions):
        raise RuntimeError("physical sessions are not covered exactly once")


def _validate_online_frontier_steps(
    online_graph: Mapping[str, Any], expected_frontiers: int
) -> None:
    steps = online_graph.get("online_fusion", {}).get("steps", [])
    if not isinstance(steps, list) or len(steps) != expected_frontiers:
        raise RuntimeError("Online did not emit one step per frontier")
    if sum(int(step.get("model_call_count", 0)) for step in steps) != max(
        0, expected_frontiers - 1
    ):
        raise RuntimeError("Online call count is not frontiers minus one")
    for ordinal, step in enumerate(steps, 1):
        if bool(step.get("model_call_made")) != (ordinal > 1):
            raise RuntimeError("Online model-call sequence differs from the frontier contract")
        if step.get("model_error") is not None:
            raise RuntimeError("Online contains a swallowed provider or embedding failure")
        if any(
            row.get("source_index") is None
            and "must contain only an operations list" in str(row.get("reason", ""))
            for row in step.get("rejected", [])
        ):
            raise RuntimeError("Online accepted a top-level response schema failure")


def _process_question(
    *,
    position: int,
    item: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    output_root: Path,
    cache_root: Path,
    config_sha256: str,
    contract: adapter.PromptContract,
    release_clients: ModuleType,
    release_memory: ModuleType,
    generation_client_type: type[Any],
    retrieval_client_type: type[Any],
    writer_workers: int,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    qid = str(item["question_id"])
    result_dir = output_root / "questions" / f"{position:03d}-{qid}"
    result_path = result_dir / "result.json"
    result_binding = {
        "position": position,
        "question_id": qid,
        "record_sha256": str(manifest_binding["record_sha256"]),
        "manifest_sha256": MANIFEST_SHA256,
        "config_sha256": config_sha256,
    }
    if result_path.exists():
        result = legacy._load_result(result_path)
        if result.get("binding") != result_binding or result.get("status") != "complete":
            raise RuntimeError("existing result binding/status mismatch")
        return result

    question_cache = cache_root / "questions" / f"{position:03d}-{qid}"
    ordered_specs, order_audit = legacy._ordered_session_specs(item)
    frontiers = adapter.group_session_specs(ordered_specs)
    frontier_plan_sha = _canonical(frontiers)
    stack = release_clients.ModelStack.for_provider("gemini")
    timings: dict[str, float] = {}

    def write(frontier: Mapping[str, Any]) -> dict[str, Any]:
        return _write_frontier_packet(
            item=item,
            manifest_binding=manifest_binding,
            frontier=frontier,
            question_cache=question_cache,
            contract=contract,
            generation_client_type=generation_client_type,
            stack=stack,
        )

    tick = time.perf_counter()
    if writer_workers == 1:
        packets = [write(frontier) for frontier in frontiers]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=writer_workers) as executor:
            packets = list(executor.map(write, frontiers))
    timings["writer_wall"] = time.perf_counter() - tick

    session_binding = _canonical(
        {
            "result_binding": result_binding,
            "frontier_plan_sha256": frontier_plan_sha,
            "writer_packet_payload_sha256": [_canonical(packet) for packet in packets],
            "embedding_contract": _run_config(contract)["models"]["embedding"],
        }
    )
    retrieval_client = retrieval_client_type(question_cache / "retrieval", stack)
    embed = EmbedCompat(retrieval_client)
    session_path = question_cache / "graphs" / "graph_frontier.pkl"
    session_graph = _load_graph(session_path, config_sha256, session_binding)
    tick = time.perf_counter()
    if session_graph is None:
        session_graph = adapter.build_frontier_graph_from_packets(
            packets,
            embed,
            contract=contract,
            display_text_fn=release_memory.temporal_display_text,
        )
        session_graph["causal_order_audit"] = {
            **order_audit,
            "frontier_size": adapter.FRONTIER_SIZE,
            "frontier_plan_sha256": frontier_plan_sha,
        }
        _validate_frontier_graph_binding(session_graph, frontiers)
        report = gm.audit_graph(
            session_graph,
            legal_dia_ids=session_graph["provenance_map"].keys(),
            expected_sessions=len(frontiers),
        )
        _require_pass(report, "frontier graph")
        _atomic_json(question_cache / "audits" / "graph_frontier.audit.json", report)
        _save_graph(session_path, session_graph, config_sha256, session_binding)
    else:
        _validate_frontier_graph_binding(session_graph, frontiers)
        resumed_frontier_report = gm.audit_graph(
            session_graph,
            legal_dia_ids=session_graph["provenance_map"].keys(),
            expected_sessions=len(frontiers),
        )
        _require_pass(resumed_frontier_report, "resumed frontier graph")
    timings["frontier_graph_build_audit_wall"] = time.perf_counter() - tick

    generation_client = generation_client_type(question_cache / "generation_main", stack)
    online_binding = _canonical(
        {
            "session_binding": session_binding,
            "session_content_fingerprint": session_graph["content_fingerprint"],
            "frontier_plan_sha256": frontier_plan_sha,
            "online_schema": online_memory_v4.ONLINE_SCHEMA_VERSION,
            "online_prompt_sha256": online_memory_v4.ONLINE_FRONTIER_PROMPT_SHA256,
        }
    )
    online_path = question_cache / "graphs" / "graph_online_frontier.pkl"
    online_graph = _load_graph(online_path, config_sha256, online_binding)
    online_graph_cache_hit = online_graph is not None
    tick = time.perf_counter()
    if online_graph is None:
        view_chat = FrontierCheckpointChat(
            question_cache=question_cache,
            question_binding={**result_binding, "frontier_plan_sha256": frontier_plan_sha},
            frontiers=frontiers,
            client=generation_client,
            contract=contract,
            default_model=stack.writer,
        )
        online_graph = online_memory_v4.add_online_event_views(session_graph, view_chat, embed)
        _validate_online_frontier_steps(online_graph, len(frontiers))
        online_report = online_memory_v4.audit_online_graph(session_graph, online_graph)
        _require_pass(online_report, "online frontier graph")
        typed_report = gm.audit_graph(
            online_graph,
            legal_dia_ids=online_graph["provenance_map"].keys(),
            expected_sessions=len(frontiers),
        )
        _require_pass(typed_report, "online typed graph")
        _atomic_json(question_cache / "audits" / "online_frontier.audit.json", online_report)
        _atomic_json(question_cache / "audits" / "graph_online.audit.json", typed_report)
        _save_graph(online_path, online_graph, config_sha256, online_binding)
    else:
        _validate_online_frontier_steps(online_graph, len(frontiers))
        online_report = online_memory_v4.audit_online_graph(session_graph, online_graph)
        _require_pass(online_report, "resumed online frontier graph")
        resumed_typed_report = gm.audit_graph(
            online_graph,
            legal_dia_ids=online_graph["provenance_map"].keys(),
            expected_sessions=len(frontiers),
        )
        _require_pass(resumed_typed_report, "resumed online typed graph")
    timings["online_build_and_audit_wall"] = time.perf_counter() - tick

    question = str(item["question"])
    tick = time.perf_counter()
    query_vector = embed([question])[0]
    index = adapter.QwenGraphMemoryIndex(
        online_graph,
        reranker=retrieval_client.rerank,
        instruction=contract.qwen_instruction,
    )
    retrieved, retrieval_trace = index.search(
        question,
        query_vector,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Gemini rerank forbidden")),
    )
    selected_ids = retrieval_trace["rerank"]["selected_fact_ids"]
    if (
        len(selected_ids) != min(30, int(retrieval_trace["candidates_after_cap"]))
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise RuntimeError("Qwen result is not exact Top30")
    if not retrieval_client.last_rerank_audit.get("strict_indices"):
        raise RuntimeError("Qwen strict index audit is missing")
    timings["query_embed_and_rerank_wall"] = time.perf_counter() - tick

    planner_path = question_cache / "reader" / "planner.json"
    planner_binding = {**result_binding, "raw_question_sha256": adapter.sha256_text(question), "labels_visible": False}

    planner_non_retryable: adapter.NonRetryableProviderError | None = None

    def planner_chat(prompt: str, **kwargs: Any) -> str:
        nonlocal planner_non_retryable
        if prompt != planner.build_planner_prompt(question):
            raise RuntimeError("planner prompt is not raw-question-only")
        chat = PhysicalChat(generation_client, contract, default_model=stack.answerer, stage="semantic_planner")
        try:
            return str(
                _durable_retry(
                    planner_path, planner_binding, chat, prompt, validator=_nonempty, **kwargs
                )
            )
        except adapter.NonRetryableProviderError as exc:
            planner_non_retryable = exc
            raise

    tick = time.perf_counter()
    planning, _ = strict_planner.classify_strict(question, model=stack.answerer, chat_fn=planner_chat)
    if planner_non_retryable is not None:
        raise planner_non_retryable
    operation = planning.plan.operation
    operation_context, opctx_audit = adapter.materialize_operation_context(
        operation=operation,
        question=question,
        base_retrieved=retrieved,
        retrieval_trace=retrieval_trace,
        graph=index.graph,
        selected_top_k=30,
    )
    if opctx_audit["selected_fact_ids"] != selected_ids:
        raise RuntimeError("OpCtx changed the strict Qwen Top30 selection")
    timings["planner_and_opctx_wall"] = time.perf_counter() - tick

    qa_system = contract.qa_template.format(
        context=operation_context,
        qdate=str(item["question_date"]),
        question=question,
    )
    qa_binding = {
        **result_binding,
        "raw_question_sha256": adapter.sha256_text(question),
        "question_date_sha256": adapter.sha256_text(str(item["question_date"])),
        "context_sha256": adapter.sha256_text(operation_context),
        "qa_template_sha256": contract.prompt_sha256["qa_template"],
    }
    tick = time.perf_counter()
    qa_chat = PhysicalChat(generation_client, contract, default_model=stack.answerer, stage="qa")
    response = str(
        _durable_retry(
            question_cache / "reader" / "qa.json",
            qa_binding,
            qa_chat,
            question,
            validator=_nonempty,
            model=stack.answerer,
            tag="read",
            system=qa_system,
            temperature=0.0,
            thinking_budget=0,
            max_tokens=500,
            use_cache=False,
            json_mode=False,
        )
    ).strip()
    timings["qa_wall"] = time.perf_counter() - tick

    reference = str(item["answer"])
    judge_binding = {
        **result_binding,
        "question_sha256": adapter.sha256_text(question),
        "reference_sha256": adapter.sha256_text(reference),
        "predicted_sha256": adapter.sha256_text(response),
        "judge_template_sha256": contract.prompt_sha256["judge_template"],
    }
    tick = time.perf_counter()
    correct, judge_output, judge_attempts = _judge(
        path=question_cache / "reader" / "judge.json",
        binding=judge_binding,
        client=generation_client,
        contract=contract,
        model=stack.judge,
        question=question,
        reference=reference,
        predicted=response,
    )
    timings["judge_wall"] = time.perf_counter() - tick
    timings["total_wall"] = time.perf_counter() - started_total

    historical_online_provider_seconds = sum(
        float(attempt.get("elapsed_seconds", 0.0))
        for checkpoint in sorted((question_cache / "view_frontiers").glob("frontier-*.json"))
        for attempt in _verify_sealed_payload(_load_json(checkpoint), checkpoint).get("attempts", [])
    )
    online_provider_seconds = (
        0.0 if online_graph_cache_hit else historical_online_provider_seconds
    )
    timings["online_provider_seconds"] = online_provider_seconds
    timings["online_provider_historical_seconds"] = historical_online_provider_seconds
    timings["online_local_estimate_seconds"] = max(
        0.0, timings["online_build_and_audit_wall"] - online_provider_seconds
    )

    writer_attempt_rows = [
        attempt
        for packet in packets
        for attempt in packet.get("transport_attempts", [])
    ]
    online_attempt_rows = [
        attempt
        for checkpoint in sorted((question_cache / "view_frontiers").glob("frontier-*.json"))
        for attempt in legacy._checkpoint_attempts(
            checkpoint, same_canonical_body=True
        )
    ]
    planner_attempt_rows = legacy._checkpoint_attempts(
        planner_path, same_canonical_body=True
    )
    qa_attempt_rows = legacy._checkpoint_attempts(
        question_cache / "reader" / "qa.json", same_canonical_body=True
    )
    all_generation_attempts = [
        *writer_attempt_rows,
        *online_attempt_rows,
        *planner_attempt_rows,
        *qa_attempt_rows,
        *judge_attempts,
    ]

    result = {
        "schema_version": RUN_SCHEMA,
        "binding": result_binding,
        "status": "complete",
        "question_id": qid,
        "question": question,
        "question_date": str(item["question_date"]),
        "question_type": str(item["question_type"]),
        "is_abstention": qid.endswith("_abs"),
        "reference": reference,
        "response": response,
        "correct": correct,
        "judge_output": judge_output,
        "operation": operation,
        "semantic_plan": planning.to_dict(),
        "context_policy": opctx_audit["context_policy"],
        "retrieved_context": operation_context,
        "retrieved_characters": len(operation_context),
        "retrieval_trace": retrieval_trace,
        "operation_context_audit": opctx_audit,
        "qwen_strict_audit": retrieval_client.last_rerank_audit,
        "online_fusion_diagnostics": {
            "accepted_operations": int(
                online_graph.get("stats", {}).get("online_accepted_operations", 0)
            ),
            "rejected_operations": int(
                online_graph.get("stats", {}).get("online_rejected_operations", 0)
            ),
            "event_view_nodes": int(
                online_graph.get("stats", {}).get("event_view_nodes", 0)
            ),
            "logical_calls": int(
                online_graph.get("stats", {}).get("online_fusion_calls", 0)
            ),
        },
        "frontier_diagnostics": {
            "physical_sessions": len(ordered_specs),
            "frontiers": len(frontiers),
            "frontier_sizes": [len(frontier["members"]) for frontier in frontiers],
            "online_calls": int(online_graph["stats"]["online_fusion_calls"]),
            "frontier_plan_sha256": frontier_plan_sha,
            "causality_granularity": "frontier_not_physical_session",
        },
        "prompt_evidence": {
            "write_template_sha256": contract.prompt_sha256["write_template"],
            "qa_template_sha256": contract.prompt_sha256["qa_template"],
            "judge_template_sha256": contract.prompt_sha256["judge_template"],
            "filled_qa_system_sha256": adapter.sha256_text(qa_system),
        },
        "embedding_contract": _run_config(contract)["models"]["embedding"],
        "embedding_compat_calls": embed.calls,
        "physical_attempts": {
            "writer": len(writer_attempt_rows),
            "online_fusion": len(online_attempt_rows),
            "strict_planner": len(planner_attempt_rows),
            "qa": len(qa_attempt_rows),
            "judge": len(judge_attempts),
            "total_generation": len(all_generation_attempts),
            "total_generation_elapsed_seconds": sum(
                float(row.get("elapsed_seconds", 0.0))
                for row in all_generation_attempts
            ),
        },
        "timings_seconds": timings,
        "generation_usage": generation_client.usage.snapshot(),
        "error": "",
    }
    legacy._save_result(result_path, result)
    return result


def _safe_process(**kwargs: Any) -> dict[str, Any]:
    try:
        return _process_question(**kwargs)
    except Exception as exc:
        item = kwargs["item"]
        result = {
            "schema_version": RUN_SCHEMA,
            "binding": {
                "position": int(kwargs["position"]),
                "question_id": str(item["question_id"]),
                "record_sha256": str(kwargs["manifest_binding"]["record_sha256"]),
                "manifest_sha256": MANIFEST_SHA256,
                "config_sha256": str(kwargs["config_sha256"]),
            },
            "status": "error",
            "question_id": str(item["question_id"]),
            "question_type": str(item["question_type"]),
            "is_abstention": str(item["question_id"]).endswith("_abs"),
            "correct": 0,
            "error": f"{type(exc).__name__}: {str(exc)[-1000:]}",
        }
        error_path = (
            kwargs["output_root"]
            / "questions"
            / f"{int(kwargs['position']):03d}-{item['question_id']}"
            / "errors"
            / f"attempt-{time.time_ns()}.json"
        )
        _atomic_json(error_path, _sealed_payload(result))
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--question-workers", type=int, default=4)
    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be 1..100")
    if not 1 <= args.question_workers <= 8 or not 1 <= args.writer_workers <= 8:
        raise SystemExit("worker counts must be 1..8")
    if args.question_workers * args.writer_workers > 8:
        raise SystemExit("question-workers x writer-workers must not exceed 8")
    preflight = run_preflight(args.preflight_output.resolve())
    if not args.allow_live:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    contract = adapter.assert_prompt_contract()
    _, manifest, selected_all = legacy._load_inputs(contract)
    selected = selected_all[: args.limit]
    bindings = manifest["items"][: args.limit]
    legacy._load_live_credentials()
    release_clients, release_memory = legacy._load_release_runtime(contract)
    generation_type, retrieval_type = _client_types(release_clients)
    config = _run_config(contract)
    _assert_model_stack(release_clients.ModelStack.for_provider("gemini"), config)
    config_sha = _canonical(config)
    output_root = args.output.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    config_path = output_root / "run_config.json"
    selection_path = output_root / "selection.json"
    existing = (output_root.exists() and any(output_root.iterdir())) or (cache_root.exists() and any(cache_root.iterdir()))
    if existing and not args.resume:
        raise SystemExit("v2 requires fresh output/cache roots; use --resume only for this exact v2 run")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    envelope = {"config": config, "config_sha256": config_sha}
    if config_path.exists() and _load_json(config_path) != envelope:
        raise SystemExit("existing run_config differs from v2 contract")
    if not config_path.exists():
        _atomic_json(config_path, envelope)

    selection = {
        "schema_version": f"{RUN_SCHEMA}.selection",
        "limit": args.limit,
        "manifest_sha256": MANIFEST_SHA256,
        "items": [
            {
                "position": position,
                "question_id": str(item["question_id"]),
                "record_sha256": str(binding["record_sha256"]),
                "source_index": int(binding["source_index"]),
            }
            for position, (item, binding) in enumerate(zip(selected, bindings))
        ],
    }
    selection_envelope = {
        "selection": selection,
        "selection_sha256": _canonical(selection),
    }
    if selection_path.exists():
        if _load_json(selection_path) != selection_envelope:
            raise SystemExit(
                "resume selection differs; use a fresh root instead of changing --limit"
            )
    elif existing and args.resume:
        raise SystemExit("existing v2 run has no sealed selection receipt")
    else:
        _atomic_json(selection_path, selection_envelope)

    jobs = [
        {
            "position": position,
            "item": item,
            "manifest_binding": binding,
            "output_root": output_root,
            "cache_root": cache_root,
            "config_sha256": config_sha,
            "contract": contract,
            "release_clients": release_clients,
            "release_memory": release_memory,
            "generation_client_type": generation_type,
            "retrieval_client_type": retrieval_type,
            "writer_workers": args.writer_workers,
        }
        for position, (item, binding) in enumerate(zip(selected, bindings))
    ]
    rows: list[dict[str, Any]] = []
    if args.question_workers == 1:
        rows = [_safe_process(**job) for job in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.question_workers) as executor:
            rows = list(executor.map(lambda job: _safe_process(**job), jobs))
    rows_path = output_root / "rows.jsonl"
    rows_temporary = rows_path.with_suffix(".jsonl.tmp")
    rows_temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    rows_temporary.replace(rows_path)
    complete = [row for row in rows if row.get("status") == "complete"]
    timing_keys = sorted(
        {
            key
            for row in complete
            for key in row.get("timings_seconds", {})
        }
    )
    summary = {
        "schema_version": RUN_SCHEMA,
        "rows": len(rows),
        "complete": len(complete),
        "errors": len(rows) - len(complete),
        "correct": sum(int(row.get("correct", 0)) for row in rows),
        "accuracy_errors_as_incorrect": sum(int(row.get("correct", 0)) for row in rows) / len(rows),
        "selection_sha256": selection_envelope["selection_sha256"],
        "selected_count": args.limit,
        "frontier_totals": {
            "physical_sessions": sum(
                int(row.get("frontier_diagnostics", {}).get("physical_sessions", 0))
                for row in complete
            ),
            "frontiers": sum(
                int(row.get("frontier_diagnostics", {}).get("frontiers", 0))
                for row in complete
            ),
            "online_calls": sum(
                int(row.get("frontier_diagnostics", {}).get("online_calls", 0))
                for row in complete
            ),
        },
        "mean_timings_seconds": {
            key: sum(float(row["timings_seconds"].get(key, 0.0)) for row in complete)
            / len(complete)
            for key in timing_keys
        }
        if complete
        else {},
        "config": config,
        "rows_sha256": _sha_file(rows_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["summary_sha256"] = _canonical(summary)
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

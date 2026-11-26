"""Resumable MemEye v19 build executor with provider-delayed transports.

The public entrypoint is intentionally live-blocked until three independent
availability-smoke receipts (2.5 generation, 3.5 generation, and native
3,072-d embedding) have been sealed.  All planning, caption-bank validation,
native 10/10 census, fake-provider integration, hashing, and resume audits are
provider-cold.

This executor builds, per backbone and per task:

* one caption bank (frozen V7 validation for 2.5; fresh sequential prev-2
  reconstruction for 3.5),
* one CreaMem physical-session graph shared by all four CreaMem arms,
* one strict-causal Online graph shared by Online-base and operation-context,
* one full-history Offline graph, and
* one native MGTM greedy-10-image/10-round tree.

It does not implement retrieval, prediction, judging, or scoring.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import mimetypes
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import memeye_adapter_v9 as adapter
from creamem3 import memeye_matched_ingest_v19 as ingest
from creamem3 import online_memory_v3 as online
from creamem3 import run_memeye_full_arch_matrix_v19 as matrix


BUILD_EXECUTOR_SCHEMA = "creamem3.memeye_full_arch_build_executor.v19.0"
BUILD_PLAN_SCHEMA = "creamem3.memeye_full_arch_build_plan.v19.0"
BUILD_ARTIFACT_RECEIPT_SCHEMA = (
    "creamem3.memeye_full_arch_build_artifact_receipt.v19.0"
)
BUILD_COMPLETE_SCHEMA = "creamem3.memeye_full_arch_build_complete.v19.0"
OFFICIAL_SMOKE_BRIDGE_SCHEMA = (
    "creamem3.memeye.official_provider_capability_bridge.v19.1"
)
ZERO_CALL_RELEASE_SCHEMA = "creamem3.memeye_build_zero_call_release.v19.0"
WRITER_COMPLETE_SCHEMA = "creamem3.memeye_build_writer_complete.v19.0"
WRITER_GENERATION_EVIDENCE_SCHEMA = (
    "creamem3.memeye_writer_generation_evidence.v19.0"
)
GENERATION_USAGE_SCHEMA = "creamem3.memeye_durable_generation_usage.v19.0"
FINAL_GENERATION_EVIDENCE_SCHEMA = (
    "creamem3.memeye_final_generation_evidence.v19.0"
)
EMBEDDING_BUDGET_SCHEMA = "creamem3.memeye_dynamic_embedding_budget.v19.0"
DYNAMIC_EMBEDDING_GATE_SCHEMA = (
    "creamem3.memeye_dynamic_embedding_budget_gate.v19.0"
)

EXPECTED_IMAGES = 438
EXPECTED_TASKS = 8
EXPECTED_SESSIONS = 221
EXPECTED_NATIVE_BATCHES = 87
EXPECTED_ONLINE_LOGICAL_CALLS = 213
EMBEDDING_DIMENSION = 3072
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_VISIBLE_CHARACTER_LIMIT = 8000
EMBEDDING_MAX_PROVIDER_ATTEMPTS = 3
OFFICIAL_EMBEDDING_RUNTIME_SHA256 = (
    "bb782cbba5d24f08c13907f2a9fe386c8bcffa5e15f53b675b69818c4989298b"
)
OFFICIAL_SMOKE_RUNNER_SHA256 = (
    "63f48f3d5d26400195c23e9831c24ad852958cca54dde64d305d36c721d400de"
)
OFFICIAL_SMOKE_GATE_SHA256 = {
    "g25_summary_writer": (
        "dd8b304c2f92ee7d549cb63c8f65b8ffe1ed9d467598625def7d24b5b10ed53e"
    ),
    "g35_caption": (
        "3df1c983c1ea6a4d7b4298178c259e824a0a65537acd2c8192d1a483ddf2ccf4"
    ),
    "embedding_batch": (
        "b853ef17ce71b3b7cd1e8fc38c9657d04cf7b3b10b4616ad8fe7aae721628f02"
    ),
}
OFFICIAL_SMOKE_RECEIPT_SHA256 = {
    "g25_summary_writer": (
        "ef4944a43b579e3ebfc0699cb2c1d5d4136bf1cc0a8a61468c442e587e87b997"
    ),
    "g35_caption": (
        "ccfcbb7c6b927b8e4e0e6ee38bfec05e148468919aaa40da442a90a85b8f36ab"
    ),
    "embedding_batch": (
        "51d4da9d985cb00665a8a58a5225a0c89e66e6f6c35a19f69e6cfc196bd39136"
    ),
}

# Exact physical generation-attempt hard ceilings. Embedding cannot be
# bounded from the static dataset census because writer output determines the
# fact count. It is sealed from writer packets before any embedding call.
GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS: dict[str, int] = {
    "g35_caption": 1314,
    "g25_writer": 1848,
    "g35_writer": 1848,
    "g25_view": 735,
    "g35_view": 735,
}
BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING = sum(
    GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS.values()
)

G35_LOGICAL_LIMITS = {
    "caption": 438,
    "writer": 616,  # 2 * (221 physical sessions + 87 native batches)
    "view": 245,  # 213 Online + at most 8 * 4 Offline schema requests
}
G25_LOGICAL_LIMITS = {
    "writer": 616,
    "view": 245,
}

ONLINE_PROMPT_SHA256 = adapter.MEMEYE_ONLINE_EVENT_PROMPT_SHA256
OFFLINE_STANDARD_PROMPT_SHA256 = adapter.MEMEYE_OFFLINE_EVENT_PROMPT_SHA256
OFFLINE_LARGE_PROMPT_SHA256 = adapter.MEMEYE_OFFLINE_LARGE_EVENT_PROMPT_SHA256


class BuildExecutorError(RuntimeError):
    """A build request, budget, artifact, provider, or resume invariant failed."""


@dataclass(frozen=True)
class GenerationReply:
    text: str
    request_sha256: str
    response_sha256: str
    cache_hit: bool = False


class GenerationBackend(Protocol):
    backbone: str

    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_output_tokens: int,
        max_provider_attempts: int,
        cache_metadata: Mapping[str, Any] | None = None,
        image_path: Path | None = None,
        caption_context_count: int | None = None,
    ) -> GenerationReply: ...

    def usage_receipt(self) -> Mapping[str, Any]: ...


class EmbeddingBackend(Protocol):
    def embed(
        self,
        texts: Sequence[str],
        *,
        tag: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> list[list[float]]: ...

    def usage_receipt(self) -> Mapping[str, Any]: ...


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise BuildExecutorError(f"missing {label}: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise BuildExecutorError(f"{label} must be a JSON object")
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        if _read_mapping(path, path.name) != dict(value):
            raise BuildExecutorError(f"write-once artifact drift: {path}")
        return
    _write_json_atomic(path, value)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _image_occurrences(memory: Mapping[str, Any]) -> list[str]:
    output: list[str] = []
    for session in memory.get("multi_session_dialogues", []):
        for row in session.get("dialogues", []):
            images = row.get("input_image") or []
            images = images if isinstance(images, list) else [images]
            output.extend(str(value) for value in images if str(value).strip())
    return output


def _chronological_unique_images(
    task: str, memory: Mapping[str, Any]
) -> list[str]:
    """Recover the exact V7 caption order from the pinned native collector."""

    authority = ingest.load_native_pure_authority()
    rounds = authority["collect_rounds"](
        task,
        {
            "multi_session_dialogues": copy.deepcopy(
                list(memory.get("multi_session_dialogues", []))
            )
        },
        Path("."),
    )
    output: list[str] = []
    seen: set[str] = set()
    for row in rounds:
        for raw in row.get("images", []):
            image = str(raw)
            if image and image not in seen:
                seen.add(image)
                output.append(image)
    occurrences = _image_occurrences(memory)
    if len(output) != len(set(occurrences)):
        raise BuildExecutorError(f"{task} chronological image projection drift")
    return output


def _resolve_image(image_root: Path, relative: str) -> Path:
    direct = Path(image_root) / relative
    fallback = Path(image_root) / Path(relative).name
    path = direct if direct.is_file() else fallback
    if not path.is_file():
        raise BuildExecutorError(f"missing raw image: {relative}")
    return path.resolve()


def _artifact_pair(
    root: Path, category: str, backbone: str, task: str
) -> tuple[Path, Path]:
    artifact = root / "build" / "graphs" / backbone / category / f"{task}.json"
    receipt = (
        root
        / "build"
        / "receipts"
        / backbone
        / category
        / f"{task}.json"
    )
    return artifact, receipt


def _inspect_artifact_pair(
    artifact_path: Path,
    receipt_path: Path,
    *,
    expected_input_sha256: str,
    expected_category: str,
    expected_backbone: str,
    expected_task: str,
) -> dict[str, Any] | None:
    if artifact_path.exists() != receipt_path.exists():
        raise BuildExecutorError(
            f"partial artifact/receipt pair: {artifact_path} / {receipt_path}"
        )
    if not artifact_path.exists():
        return None
    receipt = _read_mapping(receipt_path, "build artifact receipt")
    receipt_keys = {
        "schema_version",
        "category",
        "backbone",
        "task",
        "input_sha256",
        "artifact_path",
        "artifact_sha256",
        "content_fingerprint",
        "receipt_sha256",
    }
    receipt_contract = dict(receipt)
    observed_receipt_sha = receipt_contract.pop("receipt_sha256", None)
    artifact = _read_mapping(artifact_path, "resumed build artifact")
    if (
        set(receipt) != receipt_keys
        or observed_receipt_sha != _sha256(receipt_contract)
        or receipt.get("schema_version") != BUILD_ARTIFACT_RECEIPT_SCHEMA
        or receipt.get("category") != expected_category
        or receipt.get("backbone") != expected_backbone
        or receipt.get("task") != expected_task
        or receipt.get("input_sha256") != expected_input_sha256
        or Path(str(receipt.get("artifact_path", ""))).resolve()
        != artifact_path.resolve()
        or receipt.get("artifact_sha256") != _file_sha256(artifact_path)
        or receipt.get("content_fingerprint")
        != artifact.get("content_fingerprint")
    ):
        raise BuildExecutorError(f"stale or corrupt artifact receipt: {receipt_path}")
    return artifact


def _seal_artifact_pair(
    artifact_path: Path,
    receipt_path: Path,
    artifact: Mapping[str, Any],
    *,
    category: str,
    backbone: str,
    task: str,
    input_sha256: str,
) -> dict[str, Any]:
    _write_once_json(artifact_path, artifact)
    receipt = {
        "schema_version": BUILD_ARTIFACT_RECEIPT_SCHEMA,
        "category": category,
        "backbone": backbone,
        "task": task,
        "input_sha256": input_sha256,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _file_sha256(artifact_path),
        "content_fingerprint": artifact.get("content_fingerprint"),
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    _write_once_json(receipt_path, receipt)
    return receipt


def _project_tasks(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt in project.get("task_receipts", []):
        task = str(receipt.get("task", ""))
        path = Path(str(receipt.get("memory_path", "")))
        if not task or not path.is_file() or _file_sha256(path) != receipt.get(
            "memory_sha256"
        ):
            raise BuildExecutorError("projected memory receipt drift")
        memory = _read_mapping(path, f"projected memory {task}")
        if "human-annotated QAs" in memory:
            raise BuildExecutorError("gold QA entered build boundary")
        rows.append(
            {
                "task": task,
                "memory_path": path,
                "memory_sha256": receipt["memory_sha256"],
                "memory": memory,
                "images": _chronological_unique_images(task, memory),
                "sessions": len(memory.get("multi_session_dialogues", [])),
            }
        )
    return sorted(rows, key=lambda row: row["task"])


def load_g25_caption_bank_for_task(
    task_row: Mapping[str, Any], caption_root: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    task = str(task_row["task"])
    images = list(task_row["images"])
    captions: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        path = Path(caption_root) / task / f"{_safe_name(image)}.json"
        value = _read_mapping(path, f"V7 caption {image}")
        expected_previous = images[max(0, index - 3) : index - 1]
        if (
            value.get("task") != task
            or value.get("image") != image
            or value.get("model") != "gemini-2.5-flash-lite"
            or value.get("prompt_version") != "memeye_v7_prev2_context_b10"
            or value.get("sequence_index") != index
            or value.get("total_images") != len(images)
            or value.get("previous_images") != expected_previous
            or value.get("previous_summary_ids") not in ([], None)
            or value.get("error") not in ("", None)
            or not str(value.get("caption", "")).strip()
        ):
            raise BuildExecutorError(f"frozen V7 caption metadata drift: {path}")
        caption = str(value["caption"]).strip()
        captions[image] = caption
        rows.append(
            {
                "image": image,
                "sequence_index": index,
                "previous_images": expected_previous,
                "caption_sha256": _sha256(caption),
                "file_sha256": _file_sha256(path),
            }
        )
    return captions, {
        "task": task,
        "images": len(images),
        "model": "gemini-2.5-flash-lite",
        "prompt_version": "memeye_v7_prev2_context_b10",
        "rows_sha256": _sha256(rows),
    }


def _load_g35_caption_system_prompt() -> str:
    """Load the exact image-reader instruction only after its source is pinned."""

    protocol = matrix.load_protocol()
    authority = protocol["authority_files"]["g35_runtime"]
    runtime_path = (Path(__file__).resolve().parent.parent / authority["path"]).resolve()
    if _file_sha256(runtime_path) != str(authority["sha256"]):
        raise BuildExecutorError("G35 runtime authority drift before caption import")
    runtime = importlib.import_module("creamem3.memeye_g35_runtime_v19")
    if Path(runtime.__file__).resolve() != runtime_path:
        raise BuildExecutorError("G35 runtime import path drift")
    prompt = str(runtime.load_current_image_read_prompt())
    expected_sha = str(protocol["caption_protocol"]["system_prompt_sha256"])
    if _sha256(prompt) != expected_sha:
        raise BuildExecutorError("G35 caption IMAGE_READ_PROMPT digest drift")
    if len(prompt.splitlines()) != 31:
        raise BuildExecutorError("G35 caption IMAGE_READ_PROMPT line-count drift")
    return prompt


def build_g35_caption_bank_for_task(
    task_row: Mapping[str, Any],
    *,
    backend: GenerationBackend,
    image_root: Path,
    output_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    task = str(task_row["task"])
    images = list(task_row["images"])
    captions: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    history: list[dict[str, str]] = []
    system_prompt = _load_g35_caption_system_prompt()
    for index, image in enumerate(images, start=1):
        previous = history[-2:]
        prompt = matrix.build_v7_reconstruction_user_text(
            task,
            image,
            sequence_index=index,
            total_images=len(images),
            previous=previous,
        )
        image_path = _resolve_image(image_root, image)
        input_contract = {
            "schema_version": BUILD_EXECUTOR_SCHEMA,
            "backbone": "g35",
            "task": task,
            "image": image,
            "sequence_index": index,
            "total_images": len(images),
            "previous": previous,
            "prompt_sha256": _sha256(prompt),
            "raw_image_sha256": _file_sha256(image_path),
            "system_prompt_sha256": _sha256(system_prompt),
            "system_prompt_line_count": len(system_prompt.splitlines()),
            "user_part_order": ["raw_image", "previous_two_caption_user_text"],
            "media_resolution": "MEDIA_RESOLUTION_HIGH",
            "max_output_tokens": 2400,
            "max_provider_attempts": 3,
        }
        input_sha = _sha256(input_contract)
        path = (
            output_root
            / "captions"
            / "g35"
            / task
            / f"{index:04d}_{_safe_name(image)}.json"
        )
        if path.is_file():
            record = _read_mapping(path, "G35 caption record")
            if record.get("input_contract_sha256") != input_sha:
                raise BuildExecutorError(f"G35 caption resume drift: {path}")
            caption = str(record.get("caption", "")).strip()
            if not caption:
                raise BuildExecutorError(f"empty resumed G35 caption: {path}")
        else:
            reply = backend.generate(
                "caption",
                prompt,
                system=system_prompt,
                max_output_tokens=2400,
                max_provider_attempts=3,
                cache_metadata={
                    "input_contract_sha256": input_sha,
                    "task": task,
                    "image": image,
                    "sequence_index": index,
                },
                image_path=image_path,
                caption_context_count=len(previous),
            )
            caption = str(reply.text).strip()
            if not caption:
                raise BuildExecutorError(f"empty G35 caption: {image}")
            record = {
                "schema_version": "creamem3.memeye_g35_caption.v19.0",
                "task": task,
                "image": image,
                "caption": caption,
                "model": "gemini-3.5-flash-lite",
                "lineage": "best_available_v7_reconstruction_not_byte_exact",
                "sequence_index": index,
                "total_images": len(images),
                "previous_images": [row["image"] for row in previous],
                "previous_caption_sha256": [
                    _sha256(row["caption"]) for row in previous
                ],
                "summary_contexts": 0,
                "input_contract": input_contract,
                "input_contract_sha256": input_sha,
                "provider_request_sha256": reply.request_sha256,
                "provider_response_sha256": reply.response_sha256,
            }
            _write_once_json(path, record)
        captions[image] = caption
        history.append({"image": image, "caption": caption})
        rows.append(
            {
                "image": image,
                "sequence_index": index,
                "caption_sha256": _sha256(caption),
                "input_contract_sha256": input_sha,
                "file_sha256": _file_sha256(path),
            }
        )
    return captions, {
        "task": task,
        "images": len(images),
        "model": "gemini-3.5-flash-lite",
        "lineage": "best_available_v7_reconstruction_not_byte_exact",
        "rows_sha256": _sha256(rows),
    }


class StrictEmbeddingFacade:
    """Hide caller hints and enforce the native omitted-field 3072 contract."""

    def __init__(self, backend: EmbeddingBackend, *, backbone: str, task: str) -> None:
        self.backend = backend
        self.backbone = backbone
        self.task = task
        self.invocations: list[dict[str, Any]] = []

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        materialized = [str(value) for value in texts]
        tag = str(kwargs.get("tag", "memeye_v19_embed"))
        # Some frozen graph helpers pass a semantic task_type hint.  It is
        # recorded as a caller hint but never forwarded to the provider.
        caller_hint = kwargs.get("task_type") or kwargs.get("input_type")
        cache_metadata = {
            "backbone": self.backbone,
            "task": self.task,
            "texts_sha256": _sha256(materialized),
            "provider_visible_texts_sha256": _sha256(
                [
                    value[:EMBEDDING_VISIBLE_CHARACTER_LIMIT]
                    for value in materialized
                ]
            ),
            "batch_size": EMBEDDING_BATCH_SIZE,
            "provider_visible_character_limit": (
                EMBEDDING_VISIBLE_CHARACTER_LIMIT
            ),
            "cache_identity_uses_complete_untruncated_text": True,
            "task_type_omitted": True,
            "input_type_omitted": True,
            "output_dimensionality_omitted": True,
        }
        vectors = self.backend.embed(
            materialized,
            tag=tag,
            cache_metadata=cache_metadata,
        )
        if len(vectors) != len(materialized):
            raise BuildExecutorError("embedding vector count mismatch")
        normalized: list[list[float]] = []
        for vector in vectors:
            values = [float(value) for value in vector]
            if len(values) != EMBEDDING_DIMENSION:
                raise BuildExecutorError(
                    f"embedding dimension drift: {len(values)} != 3072"
                )
            if not all(math.isfinite(value) for value in values):
                raise BuildExecutorError("embedding contains a non-finite value")
            normalized.append(values)
        self.invocations.append(
            {
                "tag": tag,
                "texts": len(materialized),
                "materialized_texts": materialized,
                "text_sha256s": [_sha256(value) for value in materialized],
                "visible_text_sha256s": [
                    _sha256(value[:EMBEDDING_VISIBLE_CHARACTER_LIMIT])
                    for value in materialized
                ],
                "texts_sha256": _sha256(materialized),
                "provider_visible_texts_sha256": _sha256(
                    [
                        value[:EMBEDDING_VISIBLE_CHARACTER_LIMIT]
                        for value in materialized
                    ]
                ),
                "physical_batches": _embedding_batches(len(materialized)),
                "cache_metadata": cache_metadata,
                "vector_sha256s": [_sha256(vector) for vector in normalized],
                "vectors_sha256": _sha256(normalized),
                "caller_task_type_hint_ignored": caller_hint,
                "provider_task_type": None,
                "provider_input_type": None,
                "provider_output_dimensionality": None,
                "dimension": EMBEDDING_DIMENSION,
                "batch_size": EMBEDDING_BATCH_SIZE,
                "provider_visible_character_limit": (
                    EMBEDDING_VISIBLE_CHARACTER_LIMIT
                ),
                "cache_identity_uses_complete_untruncated_text": True,
                "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
            }
        )
        return normalized


def _reply_adapter(backend: GenerationBackend) -> Callable[..., Any]:
    def generate(
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_output_tokens: int,
        max_provider_attempts: int,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> GenerationReply:
        return backend.generate(
            stage,
            prompt,
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            max_provider_attempts=max_provider_attempts,
            cache_metadata=cache_metadata,
        )

    return generate


def _view_chat(backend: GenerationBackend, *, backbone: str, task: str) -> Callable[..., str]:
    def chat(prompt: str, **kwargs: Any) -> str:
        max_tokens = int(kwargs.get("max_tokens", 16384))
        if max_tokens not in {4096, 16384}:
            raise BuildExecutorError(f"unexpected view output cap: {max_tokens}")
        reply = backend.generate(
            "view",
            str(prompt),
            system=kwargs.get("system"),
            json_mode=bool(kwargs.get("json_mode", True)),
            max_output_tokens=max_tokens,
            max_provider_attempts=3,
            cache_metadata={
                "backbone": backbone,
                "task": task,
                "tag": str(kwargs.get("tag", "")),
                "prompt_sha256": _sha256(str(prompt)),
            },
        )
        return reply.text

    return chat


def _legal_dia_ids(graph: Mapping[str, Any]) -> list[str]:
    source_map = graph.get("memeye_source_map") or {}
    if isinstance(source_map, Mapping) and source_map:
        return [str(value) for value in source_map]
    return list(
        dict.fromkeys(
            str(source)
            for fact in graph.get("fact_nodes", [])
            for source in fact.get("source_dia_ids", [])
        )
    )


def _assert_session_graph(
    graph: Mapping[str, Any], *, expected_sessions: int
) -> None:
    audit = adapter.audit_memeye_graph(
        graph,
        legal_dia_ids=_legal_dia_ids(graph),
        expected_sessions=expected_sessions,
    )
    if audit.get("status") != "pass":
        failures = [
            row.get("name")
            for row in audit.get("checks", [])
            if not row.get("passed")
        ]
        raise BuildExecutorError(f"session graph audit failed: {failures}")
    for node in [*graph.get("summary_nodes", []), *graph.get("fact_nodes", [])]:
        if len(node.get("embedding", [])) != EMBEDDING_DIMENSION:
            raise BuildExecutorError("session graph is not native 3072-dimensional")


def _assert_exact_session_materialization(
    graph: Mapping[str, Any],
    *,
    packets: Sequence[Mapping[str, Any]],
    source_map: Mapping[str, Mapping[str, Any]],
    backbone: str,
    task: str,
) -> None:
    """Re-run the provider-free projector and compare every semantic field."""

    def zero_embed(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSION for _text in texts]

    expected = ingest.materialize_creamem_session_graph(
        packets,
        source_map,
        zero_embed,
        backbone=backbone,
        task=task,
    )
    actual = copy.deepcopy(dict(graph))
    embedding_contract = actual.pop("embedding_contract", None)
    for node in [*actual.get("summary_nodes", []), *actual.get("fact_nodes", [])]:
        node["embedding"] = [0.0] * EMBEDDING_DIMENSION
    if actual != expected:
        raise BuildExecutorError("session graph deterministic materialization drift")
    if not isinstance(embedding_contract, Mapping) or set(embedding_contract) != {
        "model",
        "dimension",
        "batch_size",
        "provider_visible_character_limit",
        "cache_identity_uses_complete_untruncated_text",
        "failure_policy",
        "task_type",
        "input_type",
        "output_dimensionality",
        "invocations",
    }:
        raise BuildExecutorError("session graph embedding contract shape drift")


def _assert_offline_graph(
    session_graph: Mapping[str, Any], graph: Mapping[str, Any]
) -> None:
    """Validate that Offline only adds auditable routing views to the base graph."""

    large = len(session_graph.get("fact_nodes", [])) > 1000
    prompt_sha = (
        OFFLINE_LARGE_PROMPT_SHA256 if large else OFFLINE_STANDARD_PROMPT_SHA256
    )
    max_schema_requests = 4 if large else 3
    if (
        graph.get("schema_version") != session_graph.get("schema_version")
        or graph.get("graph_variant") != "offline"
        or graph.get("backbone") != session_graph.get("backbone")
        or graph.get("task") != session_graph.get("task")
        or graph.get("fact_nodes") != session_graph.get("fact_nodes")
        or graph.get("session_packets") != session_graph.get("session_packets")
        or graph.get("memeye_source_map") != session_graph.get("memeye_source_map")
    ):
        raise BuildExecutorError("Offline mutated the sealed session graph")
    base_summaries = list(session_graph.get("summary_nodes", []))
    summaries = list(graph.get("summary_nodes", []))
    if summaries[: len(base_summaries)] != base_summaries:
        raise BuildExecutorError("Offline mutated base summary nodes")
    views = summaries[len(base_summaries) :]
    facts = {str(row.get("id")): row for row in graph.get("fact_nodes", [])}
    seen_view_ids: set[str] = set()
    expected_view_edges: list[dict[str, str]] = []
    for view in views:
        view_id = str(view.get("id", ""))
        members = view.get("member_fact_ids")
        sessions = view.get("session_ids")
        vector = [float(value) for value in view.get("embedding", [])]
        if (
            not view_id
            or view_id in seen_view_ids
            or view.get("node_type") != "event_view"
            or not str(view.get("text", "")).strip()
            or not isinstance(members, list)
            or len(set(str(value) for value in members)) != len(members)
            or any(str(value) not in facts for value in members)
            or not isinstance(sessions, list)
            or len(set(int(value) for value in sessions)) < 2
            or len(vector) != EMBEDDING_DIMENSION
            or not all(math.isfinite(value) for value in vector)
        ):
            raise BuildExecutorError("Offline event-view invariant drift")
        expected_sessions = sorted(
            {int(facts[str(fact_id)]["session_id"]) for fact_id in members}
        )
        if sorted(int(value) for value in sessions) != expected_sessions:
            raise BuildExecutorError("Offline view membership/session drift")
        seen_view_ids.add(view_id)
        expected_view_edges.extend(
            {
                "edge_type": "VIEW_SUPPORTS",
                "source_id": view_id,
                "target_id": str(fact_id),
            }
            for fact_id in members
        )
    base_edges = list(session_graph.get("edges", []))
    edges = list(graph.get("edges", []))
    if edges[: len(base_edges)] != base_edges or edges[len(base_edges) :] != expected_view_edges:
        raise BuildExecutorError("Offline view-edge projection drift")
    fusion = graph.get("offline_fusion")
    if not isinstance(fusion, Mapping):
        raise BuildExecutorError("Offline fusion receipt missing")
    attempts_key = "grouping_attempts" if large else "attempts"
    attempts = int(fusion.get(attempts_key, 0))
    if (
        fusion.get("prompt_sha256") != prompt_sha
        or not 1 <= attempts <= max_schema_requests
        or fusion.get("view_ids") != [str(row["id"]) for row in views]
    ):
        raise BuildExecutorError("Offline fusion authority/census drift")
    if large:
        projection = fusion.get("fact_projection")
        if (
            fusion.get("policy")
            != "global_full_context_session_group_then_balanced_fact_projection"
            or not isinstance(projection, Mapping)
            or projection.get("policy")
            != "embedding_top_with_one_fact_per_member_session"
            or projection.get("member_fact_cap") != 12
            or not isinstance(projection.get("rows"), list)
            or len(views) > 16
        ):
            raise BuildExecutorError("large Offline projection contract drift")
        projected_members = [row.get("member_fact_ids") for row in projection["rows"]]
        if projected_members != [row.get("member_fact_ids") for row in views]:
            raise BuildExecutorError("large Offline projection membership drift")
    else:
        if fusion.get("policy") != "non_destructive_global_cross_session_views":
            raise BuildExecutorError("standard Offline policy drift")
    stats = graph.get("stats")
    if (
        not isinstance(stats, Mapping)
        or stats.get("event_view_nodes") != len(views)
        or stats.get("summary_nodes") != len(summaries)
        or stats.get("edges") != len(edges)
        or stats.get("cross_session_view_edges") != len(expected_view_edges)
    ):
        raise BuildExecutorError("Offline stats census drift")
    expected_fingerprint = gm.canonical_sha256(
        {
            "base": session_graph.get("content_fingerprint"),
            "views": [
                {key: value for key, value in view.items() if key != "embedding"}
                for view in views
            ],
        }
    )
    if graph.get("content_fingerprint") != expected_fingerprint:
        raise BuildExecutorError("Offline content fingerprint drift")


def _assert_native_tree(
    tree: Mapping[str, Any],
    *,
    task: str,
    backbone: str,
    rounds: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
) -> None:
    safe_rounds = [
        {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in {"_sort", "image_root"}
        }
        for raw in rounds
    ]
    expected_summaries: list[dict[str, Any]] = []
    expected_facts: list[dict[str, Any]] = []
    for packet in sorted(packets, key=lambda row: int(row["batch_index"])):
        summary_id = len(expected_summaries)
        expected_summaries.append(
            {
                "id": summary_id,
                "text": str(packet["summary"]),
                "t_start": str(packet["t_start"]),
                "t_end": str(packet["t_end"]),
                "apps": [task],
                "rounds": int(packet["rounds"]),
                "images": int(packet["images"]),
            }
        )
        for text_value in packet["facts"]:
            expected_facts.append(
                {
                    "id": len(expected_facts),
                    "parent_summary_id": summary_id,
                    "text": str(text_value),
                }
            )
    summaries = tree.get("SUMMARIES")
    facts = tree.get("FACTS")
    if not isinstance(summaries, list) or not isinstance(facts, list):
        raise BuildExecutorError("native tree node lists missing")
    projected_summaries = [
        {key: value for key, value in row.items() if key != "emb"}
        for row in summaries
    ]
    projected_facts = [
        {key: value for key, value in row.items() if key != "emb"} for row in facts
    ]
    if (
        tree.get("schema_version") != ingest.NATIVE_TREE_SCHEMA
        or tree.get("backbone") != backbone
        or tree.get("work_model") != ingest.WORK_MODELS[backbone]
        or tree.get("task") != task
        or tree.get("items") != safe_rounds
        or tree.get("batch_packets") != list(packets)
        or projected_summaries != expected_summaries
        or projected_facts != expected_facts
    ):
        raise BuildExecutorError("native MGTM tree structural drift")
    for row in [*summaries, *facts]:
        vector = [float(value) for value in row.get("emb", [])]
        if len(vector) != EMBEDDING_DIMENSION or not all(
            math.isfinite(value) for value in vector
        ):
            raise BuildExecutorError("native MGTM embedding drift")
    expected_stats = {
        "rounds": len(safe_rounds),
        "write_batches": len(packets),
        "summaries": len(expected_summaries),
        "facts": len(expected_facts),
        "summary_logical_calls": len(packets),
        "fact_logical_calls": len(packets),
    }
    if tree.get("stats") != expected_stats:
        raise BuildExecutorError("native MGTM stats drift")
    expected_fingerprint = _sha256(
        {"summaries": expected_summaries, "facts": expected_facts}
    )
    if tree.get("content_fingerprint") != expected_fingerprint:
        raise BuildExecutorError("native MGTM content fingerprint drift")


def _ensure_session_packets(
    task_row: Mapping[str, Any],
    captions: Mapping[str, str],
    *,
    backbone: str,
    generation: GenerationBackend,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Extract or resume every physical-session packet without embedding."""

    task = str(task_row["task"])
    records, source_map = ingest.build_records_from_captions(
        task, task_row["memory"], captions
    )
    prompt_bundle = ingest.load_prompt_bundle()
    packet_root = output_root / "build" / "packets" / backbone / "session" / task
    packets: list[dict[str, Any]] = []
    generate_fn = _reply_adapter(generation)
    for record in records:
        session_id = int(record["session_id"])
        rendered = adapter._session_batch(record)  # type: ignore[attr-defined]
        input_sha = _sha256(
            {
                "schema_version": ingest.SCHEMA_VERSION,
                "backbone": backbone,
                "task": task,
                "session_id": session_id,
                "rendered_sha256": _sha256(rendered),
                "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
                "summary_tokens": 1200,
                "fact_tokens": 16000,
                "attempts": 3,
                "current_date": "2026-07-05",
            }
        )
        path = packet_root / f"S{session_id:04d}.json"
        if path.is_file():
            packet = _read_mapping(path, "session packet")
            if (
                packet.get("schema_version") != ingest.CREAMEM_PACKET_SCHEMA
                or packet.get("backbone") != backbone
                or packet.get("task") != task
                or packet.get("input_sha256") != _sha256(rendered)
            ):
                raise BuildExecutorError(f"session packet resume drift: {path}")
        else:
            packet = ingest.extract_creamem_session_packet(
                record,
                generate_fn,
                prompt_bundle,
                backbone=backbone,
                task=task,
            )
            packet["build_input_sha256"] = input_sha
            _write_once_json(path, packet)
        if packet.get("build_input_sha256") != input_sha:
            raise BuildExecutorError(f"session packet input binding drift: {path}")
        packets.append(packet)
    return records, source_map, packets


def build_session_graph_for_task(
    task_row: Mapping[str, Any],
    captions: Mapping[str, str],
    *,
    backbone: str,
    generation: GenerationBackend,
    embedding: EmbeddingBackend,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(task_row["task"])
    records, source_map, packets = _ensure_session_packets(
        task_row,
        captions,
        backbone=backbone,
        generation=generation,
        output_root=output_root,
    )

    graph_input_sha = _sha256(
        {
            "backbone": backbone,
            "task": task,
            "memory_sha256": task_row["memory_sha256"],
            "caption_map_sha256": _sha256(dict(captions)),
            "packet_sha256": [_sha256(packet) for packet in packets],
            "embedding": "gemini-embedding-001_native_3072_omitted_fields",
        }
    )
    artifact_path, receipt_path = _artifact_pair(
        output_root, "session", backbone, task
    )
    resumed = _inspect_artifact_pair(
        artifact_path,
        receipt_path,
        expected_input_sha256=graph_input_sha,
        expected_category="session",
        expected_backbone=backbone,
        expected_task=task,
    )
    if resumed is not None:
        _assert_session_graph(resumed, expected_sessions=len(records))
        _assert_exact_session_materialization(
            resumed,
            packets=packets,
            source_map=source_map,
            backbone=backbone,
            task=task,
        )
        return resumed, _read_mapping(receipt_path, "session graph receipt")

    embed = StrictEmbeddingFacade(embedding, backbone=backbone, task=task)
    graph = ingest.materialize_creamem_session_graph(
        packets,
        source_map,
        embed,
        backbone=backbone,
        task=task,
    )
    graph["embedding_contract"] = {
        "model": "gemini-embedding-001",
        "dimension": 3072,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "task_type": None,
        "input_type": None,
        "output_dimensionality": None,
        "invocations": copy.deepcopy(embed.invocations),
    }
    _assert_session_graph(graph, expected_sessions=len(records))
    _assert_exact_session_materialization(
        graph,
        packets=packets,
        source_map=source_map,
        backbone=backbone,
        task=task,
    )
    receipt = _seal_artifact_pair(
        artifact_path,
        receipt_path,
        graph,
        category="session",
        backbone=backbone,
        task=task,
        input_sha256=graph_input_sha,
    )
    return graph, receipt


def build_online_graph_for_task(
    task_row: Mapping[str, Any],
    session_graph: Mapping[str, Any],
    *,
    backbone: str,
    generation: GenerationBackend,
    embedding: EmbeddingBackend,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(task_row["task"])
    graph_input_sha = _sha256(
        {
            "base_content_fingerprint": session_graph.get("content_fingerprint"),
            "prompt_sha256": ONLINE_PROMPT_SHA256,
            "current_fact_cap": 16,
            "prior_fact_cap": 24,
            "prune_unclaimed_create_members": True,
            "embedding": "gemini-embedding-001_native_3072_omitted_fields",
        }
    )
    artifact_path, receipt_path = _artifact_pair(
        output_root, "online", backbone, task
    )
    resumed = _inspect_artifact_pair(
        artifact_path,
        receipt_path,
        expected_input_sha256=graph_input_sha,
        expected_category="online",
        expected_backbone=backbone,
        expected_task=task,
    )
    expected_sessions = int(task_row["sessions"])
    if resumed is not None:
        audit = online.audit_online_graph(
            session_graph,
            resumed,
            expected_prompt_sha256=ONLINE_PROMPT_SHA256,
            expected_current_fact_cap=16,
            expected_prior_fact_cap=24,
            expected_prune_unclaimed_create_members=True,
        )
        if audit.get("status") != "pass":
            raise BuildExecutorError("resumed Online graph failed causal audit")
        return resumed, _read_mapping(receipt_path, "Online graph receipt")

    embed = StrictEmbeddingFacade(embedding, backbone=backbone, task=task)
    graph = online.add_online_event_views(
        session_graph,
        _view_chat(generation, backbone=backbone, task=task),
        embed,
        prompt_template=adapter.MEMEYE_ONLINE_EVENT_UPDATE_PROMPT,
        prompt_sha256=ONLINE_PROMPT_SHA256,
        current_fact_cap=16,
        prior_fact_cap=24,
        prune_unclaimed_create_members=True,
    )
    audit = online.audit_online_graph(
        session_graph,
        graph,
        expected_prompt_sha256=ONLINE_PROMPT_SHA256,
        expected_current_fact_cap=16,
        expected_prior_fact_cap=24,
        expected_prune_unclaimed_create_members=True,
    )
    if audit.get("status") != "pass":
        raise BuildExecutorError("Online graph failed strict causal audit")
    if graph.get("stats", {}).get("online_fusion_calls") != max(
        0, expected_sessions - 1
    ):
        raise BuildExecutorError("Online logical call census drift")
    graph["embedding_contract"] = {
        "model": "gemini-embedding-001",
        "dimension": 3072,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "invocations": copy.deepcopy(embed.invocations),
    }
    receipt = _seal_artifact_pair(
        artifact_path,
        receipt_path,
        graph,
        category="online",
        backbone=backbone,
        task=task,
        input_sha256=graph_input_sha,
    )
    return graph, receipt


def build_offline_graph_for_task(
    task_row: Mapping[str, Any],
    session_graph: Mapping[str, Any],
    *,
    backbone: str,
    generation: GenerationBackend,
    embedding: EmbeddingBackend,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(task_row["task"])
    large = len(session_graph.get("fact_nodes", [])) > 1000
    prompt_sha = OFFLINE_LARGE_PROMPT_SHA256 if large else OFFLINE_STANDARD_PROMPT_SHA256
    max_schema_requests = 4 if large else 3
    graph_input_sha = _sha256(
        {
            "base_content_fingerprint": session_graph.get("content_fingerprint"),
            "branch": "large" if large else "standard",
            "prompt_sha256": prompt_sha,
            "max_schema_requests": max_schema_requests,
            "member_fact_cap": 12 if large else None,
            "embedding": "gemini-embedding-001_native_3072_omitted_fields",
        }
    )
    artifact_path, receipt_path = _artifact_pair(
        output_root, "offline", backbone, task
    )
    resumed = _inspect_artifact_pair(
        artifact_path,
        receipt_path,
        expected_input_sha256=graph_input_sha,
        expected_category="offline",
        expected_backbone=backbone,
        expected_task=task,
    )
    if resumed is not None:
        _assert_offline_graph(session_graph, resumed)
        return resumed, _read_mapping(receipt_path, "Offline graph receipt")
    embed = StrictEmbeddingFacade(embedding, backbone=backbone, task=task)
    chat = _view_chat(generation, backbone=backbone, task=task)
    if large:
        graph = adapter.add_large_memeye_offline_event_views(
            session_graph,
            chat,
            embed,
            max_attempts=4,
            member_fact_cap=12,
        )
    else:
        graph = gm.add_offline_event_views(
            session_graph,
            chat,
            embed,
            max_attempts=3,
            prompt_template=adapter.MEMEYE_OFFLINE_EVENT_VIEW_PROMPT,
            prompt_sha256=OFFLINE_STANDARD_PROMPT_SHA256,
        )
    fusion = graph.get("offline_fusion", {})
    if fusion.get("prompt_sha256") != prompt_sha:
        raise BuildExecutorError("Offline prompt authority drift")
    attempts = int(fusion.get("grouping_attempts" if large else "attempts", 0))
    if not 1 <= attempts <= max_schema_requests:
        raise BuildExecutorError("Offline schema-request ceiling drift")
    graph["embedding_contract"] = {
        "model": "gemini-embedding-001",
        "dimension": 3072,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "invocations": copy.deepcopy(embed.invocations),
    }
    _assert_offline_graph(session_graph, graph)
    receipt = _seal_artifact_pair(
        artifact_path,
        receipt_path,
        graph,
        category="offline",
        backbone=backbone,
        task=task,
        input_sha256=graph_input_sha,
    )
    return graph, receipt


def _ensure_native_packets(
    task_row: Mapping[str, Any],
    captions: Mapping[str, str],
    *,
    backbone: str,
    generation: GenerationBackend,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract or resume native greedy-10/10 packets without embedding."""

    task = str(task_row["task"])
    rounds, batches, rendered = ingest.native_batch_plan(
        task, task_row["memory"], captions
    )
    prompt_bundle = ingest.load_prompt_bundle()
    packet_root = output_root / "build" / "packets" / backbone / "mgtm" / task
    packets: list[dict[str, Any]] = []
    generate_fn = _reply_adapter(generation)
    for index, (batch, text) in enumerate(zip(batches, rendered), start=1):
        input_sha = _sha256(
            {
                "backbone": backbone,
                "task": task,
                "batch_index": index,
                "rendered_sha256": _sha256(text),
                "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
                "summary_tokens": 1200,
                "fact_tokens": 16000,
                "attempts": 3,
                "current_date": "2026-07-05",
                "max_images": 10,
                "max_rounds": 10,
            }
        )
        path = packet_root / f"NB{index:04d}.json"
        if path.is_file():
            packet = _read_mapping(path, "native MGTM packet")
            if (
                packet.get("schema_version") != ingest.NATIVE_PACKET_SCHEMA
                or packet.get("backbone") != backbone
                or packet.get("task") != task
                or packet.get("rendered_input_sha256") != _sha256(text)
            ):
                raise BuildExecutorError(f"native packet resume drift: {path}")
        else:
            packet = ingest.extract_native_batch_packet(
                task=task,
                batch_index=index,
                batch=batch,
                rendered=text,
                generate_fn=generate_fn,
                prompt_bundle=prompt_bundle,
                backbone=backbone,
            )
            packet["build_input_sha256"] = input_sha
            _write_once_json(path, packet)
        if packet.get("build_input_sha256") != input_sha:
            raise BuildExecutorError(f"native packet input binding drift: {path}")
        packets.append(packet)
    return rounds, packets


def build_native_tree_for_task(
    task_row: Mapping[str, Any],
    captions: Mapping[str, str],
    *,
    backbone: str,
    generation: GenerationBackend,
    embedding: EmbeddingBackend,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(task_row["task"])
    rounds, packets = _ensure_native_packets(
        task_row,
        captions,
        backbone=backbone,
        generation=generation,
        output_root=output_root,
    )
    graph_input_sha = _sha256(
        {
            "backbone": backbone,
            "task": task,
            "memory_sha256": task_row["memory_sha256"],
            "caption_map_sha256": _sha256(dict(captions)),
            "packet_sha256": [_sha256(packet) for packet in packets],
            "embedding": "gemini-embedding-001_native_3072_omitted_fields",
        }
    )
    artifact_path, receipt_path = _artifact_pair(
        output_root, "mgtm", backbone, task
    )
    resumed = _inspect_artifact_pair(
        artifact_path,
        receipt_path,
        expected_input_sha256=graph_input_sha,
        expected_category="mgtm",
        expected_backbone=backbone,
        expected_task=task,
    )
    if resumed is not None:
        _assert_native_tree(
            resumed,
            task=task,
            backbone=backbone,
            rounds=rounds,
            packets=packets,
        )
        return resumed, _read_mapping(receipt_path, "MGTM tree receipt")
    embed = StrictEmbeddingFacade(embedding, backbone=backbone, task=task)
    tree = ingest.materialize_native_tree(
        task=task,
        backbone=backbone,
        rounds=rounds,
        packets=packets,
        embed_fn=embed,
    )
    tree["embedding_contract"] = {
        "model": "gemini-embedding-001",
        "dimension": 3072,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "invocations": copy.deepcopy(embed.invocations),
    }
    for row in [*tree["SUMMARIES"], *tree["FACTS"]]:
        if len(row["emb"]) != 3072:
            raise BuildExecutorError("native MGTM tree embedding dimension drift")
    _assert_native_tree(
        tree,
        task=task,
        backbone=backbone,
        rounds=rounds,
        packets=packets,
    )
    receipt = _seal_artifact_pair(
        artifact_path,
        receipt_path,
        tree,
        category="mgtm",
        backbone=backbone,
        task=task,
        input_sha256=graph_input_sha,
    )
    return tree, receipt


def plan_build(
    *,
    project: Mapping[str, Any],
    output_root: Path,
    caption_root: Path = matrix.DEFAULT_V7_CAPTION_ROOT,
    enforce_full_scope: bool = True,
) -> dict[str, Any]:
    """Create the exact zero-provider build plan and native batch census."""

    protocol = matrix.load_protocol()
    tasks = _project_tasks(project)
    task_rows: list[dict[str, Any]] = []
    total_images = 0
    total_sessions = 0
    total_native_batches = 0
    for task_row in tasks:
        captions, caption_receipt = load_g25_caption_bank_for_task(
            task_row, caption_root
        )
        rounds, batches, rendered = ingest.native_batch_plan(
            str(task_row["task"]), task_row["memory"], captions
        )
        image_count = len(task_row["images"])
        session_count = int(task_row["sessions"])
        total_images += image_count
        total_sessions += session_count
        total_native_batches += len(batches)
        task_rows.append(
            {
                "task": task_row["task"],
                "memory_path": str(Path(task_row["memory_path"]).resolve()),
                "memory_sha256": task_row["memory_sha256"],
                "images": image_count,
                "sessions": session_count,
                "rounds": len(rounds),
                "native_batches": len(batches),
                "online_logical_calls_per_backbone": max(0, session_count - 1),
                "offline_first_schema_requests_per_backbone": 1,
                "offline_schema_request_hard_per_backbone": 4,
                "g25_caption_receipt": caption_receipt,
                "native_rendered_input_sha256": _sha256(rendered),
            }
        )
    if enforce_full_scope:
        if (
            len(tasks) != EXPECTED_TASKS
            or total_images != EXPECTED_IMAGES
            or total_sessions != EXPECTED_SESSIONS
            or total_native_batches != EXPECTED_NATIVE_BATCHES
            or sum(row["online_logical_calls_per_backbone"] for row in task_rows)
            != EXPECTED_ONLINE_LOGICAL_CALLS
        ):
            raise BuildExecutorError("full MemEye build census drift")
    plan = {
        "schema_version": BUILD_PLAN_SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": _sha256(protocol),
        "output_root": str(Path(output_root).resolve()),
        "caption_lineage": {
            "g25": "frozen_v7_438_validated",
            "g35": "fresh_best_available_v7_reconstruction_not_byte_exact",
        },
        "graph_sharing": {
            "one_session_graph_per_task_per_backbone_shared_by_creamem_arms": True,
            "one_online_graph_per_task_per_backbone_shared_by_online_and_operation_context": True,
            "offline_graph_private_per_task_per_backbone": True,
            "mgtm_native_tree_private_per_task_per_backbone": True,
        },
        "ingest": {
            "summary_max_output_tokens": 1200,
            "fact_max_output_tokens": 16000,
            "max_provider_attempts": 3,
            "frozen_current_date": "2026-07-05",
            "native_batch_max_images": 10,
            "native_batch_max_rounds": 10,
        },
        "embedding": {
            "model": "gemini-embedding-001",
            "dimension": 3072,
            "batch_size": EMBEDDING_BATCH_SIZE,
            "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
            "cache_identity_scope": "complete_untruncated_text",
            "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
            "max_provider_attempts_per_physical_batch": (
                EMBEDDING_MAX_PROVIDER_ATTEMPTS
            ),
            "task_type": None,
            "input_type": None,
            "output_dimensionality": None,
            "uniform_across_all_cells": True,
        },
        "counts": {
            "tasks": len(tasks),
            "images": total_images,
            "physical_sessions": total_sessions,
            "native_batches_per_backbone": total_native_batches,
            "online_logical_calls_per_backbone": sum(
                row["online_logical_calls_per_backbone"] for row in task_rows
            ),
        },
        "task_rows": task_rows,
        "generation_stage_provider_attempt_hard_ceilings": dict(
            GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
        ),
        "generation_provider_attempt_hard_ceiling": (
            BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        ),
        "embedding_provider_attempt_hard_ceiling": (
            "pending_post_writer_100_item_batch_census"
        ),
        "overall_provider_attempt_hard_ceiling": (
            "pending_post_writer_100_item_batch_census"
        ),
        "model_calls": 0,
        "provider_calls": 0,
    }
    plan["plan_sha256"] = _sha256(plan)
    _write_once_json(Path(output_root) / "build" / "plan.json", plan)
    return plan


def validate_build_plan(
    *, output_root: Path, project: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(output_root) / "build" / "plan.json"
    plan = _read_mapping(path, "build plan")
    contract = dict(plan)
    observed = contract.pop("plan_sha256", None)
    if observed != _sha256(contract):
        raise BuildExecutorError("build plan digest drift")
    if (
        plan.get("schema_version") != BUILD_PLAN_SCHEMA
        or plan.get("protocol_sha256") != _sha256(matrix.load_protocol())
        or plan.get("generation_provider_attempt_hard_ceiling")
        != BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        or plan.get("generation_stage_provider_attempt_hard_ceilings")
        != GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
        or plan.get("embedding_provider_attempt_hard_ceiling")
        != "pending_post_writer_100_item_batch_census"
        or plan.get("overall_provider_attempt_hard_ceiling")
        != "pending_post_writer_100_item_batch_census"
        or plan.get("model_calls") != 0
        or plan.get("provider_calls") != 0
    ):
        raise BuildExecutorError("build plan contract drift")
    if len(_project_tasks(project)) != int(plan["counts"]["tasks"]):
        raise BuildExecutorError("build plan/project task drift")
    return plan


def validate_provider_smoke_seals(
    *, output_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Strictly consume audited official smokes through a post-live bridge."""

    output_root = Path(output_root).resolve()
    bridge_path = output_root / "releases" / "official_provider_capability_bridge.json"
    if not bridge_path.is_file():
        raise BuildExecutorError("missing official provider capability bridge")
    bridge = _read_mapping(bridge_path, "official provider capability bridge")
    contract = dict(bridge)
    observed_bridge_sha = contract.pop("bridge_sha256", None)
    if observed_bridge_sha != _sha256(contract):
        raise BuildExecutorError("official provider bridge digest drift")
    required_bridge = {
        "schema_version",
        "status",
        "authorized",
        "protocol_sha256",
        "runner_path",
        "runner_file_sha256",
        "smokes",
        "independent_audit_status",
        "formal_cache_reuse_allowed",
        "bridge_sha256",
    }
    if (
        set(bridge) != required_bridge
        or bridge.get("schema_version") != OFFICIAL_SMOKE_BRIDGE_SCHEMA
        or bridge.get("status") != "ready"
        or bridge.get("authorized") is not True
        or bridge.get("protocol_sha256") != _sha256(protocol)
        or bridge.get("runner_file_sha256") != OFFICIAL_SMOKE_RUNNER_SHA256
        or bridge.get("independent_audit_status") != "pass"
        or bridge.get("formal_cache_reuse_allowed") is not False
    ):
        raise BuildExecutorError("official provider bridge contract drift")
    runner_path = Path(__file__).with_name(
        "run_official_provider_availability_smokes_v19_1.py"
    ).resolve()
    if (
        Path(str(bridge.get("runner_path", ""))).resolve() != runner_path
        or _file_sha256(runner_path) != OFFICIAL_SMOKE_RUNNER_SHA256
    ):
        raise BuildExecutorError("official smoke runner drift before import")
    smoke_rows = bridge.get("smokes")
    expected_ids = {"g25_summary_writer", "g35_caption", "embedding_batch"}
    if not isinstance(smoke_rows, Mapping) or set(smoke_rows) != expected_ids:
        raise BuildExecutorError("official provider bridge smoke census drift")
    fixed = {
        "g25_summary_writer": (
            output_root
            / "provider_smokes"
            / "g25_summary_1200_r1"
            / "complete.json",
            Path(__file__).with_name("research_memeye_full_arch_g35_v19")
            / "official_g25_summary_writer_smoke_gate_v19_1.json",
            1,
        ),
        "g35_caption": (
            output_root
            / "provider_smokes"
            / "g35_caption_writer_view_r1"
            / "complete.json",
            Path(__file__).with_name("research_memeye_full_arch_g35_v19")
            / "official_g35_caption_writer_view_smoke_gate_v19_1.json",
            3,
        ),
        "embedding_batch": (
            output_root
            / "provider_smokes"
            / "gemini_embedding_001_r1"
            / "complete.json",
            Path(__file__).with_name("research_memeye_full_arch_g35_v19")
            / "official_embedding_batch_shape_smoke_gate_v19_1.json",
            1,
        ),
    }
    # Authority-before-import: every validator and gate byte is checked first.
    for smoke_id, (_receipt_path, gate_path, _calls) in fixed.items():
        if _file_sha256(gate_path) != OFFICIAL_SMOKE_GATE_SHA256[smoke_id]:
            raise BuildExecutorError(f"official {smoke_id} gate authority drift")
    runner = importlib.import_module(
        "creamem3.run_official_provider_availability_smokes_v19_1"
    )
    if (
        Path(runner.__file__).resolve() != runner_path
        or _file_sha256(Path(runner.__file__)) != OFFICIAL_SMOKE_RUNNER_SHA256
    ):
        raise BuildExecutorError("official smoke runner import drift")
    output: dict[str, Any] = {}
    row_keys = {
        "receipt_path",
        "receipt_file_sha256",
        "gate_path",
        "gate_file_sha256",
        "provider_calls",
        "formal_cache_reuse_allowed",
    }
    for smoke_id, (receipt_path, gate_path, provider_calls) in fixed.items():
        raw_row = smoke_rows[smoke_id]
        if not isinstance(raw_row, Mapping) or set(raw_row) != row_keys:
            raise BuildExecutorError(f"official {smoke_id} bridge row drift")
        row = dict(raw_row)
        if (
            Path(str(row["receipt_path"])).resolve() != receipt_path.resolve()
            or Path(str(row["gate_path"])).resolve() != gate_path.resolve()
            or row["gate_file_sha256"] != OFFICIAL_SMOKE_GATE_SHA256[smoke_id]
            or row["receipt_file_sha256"]
            != OFFICIAL_SMOKE_RECEIPT_SHA256[smoke_id]
            or row["provider_calls"] != provider_calls
            or row["formal_cache_reuse_allowed"] is not False
        ):
            raise BuildExecutorError(f"official {smoke_id} bridge binding drift")
        validated = runner.validate_official_smoke_receipt(
            smoke_id,
            receipt_path,
            expected_file_sha256=str(row["receipt_file_sha256"]),
            gate_path=gate_path,
        )
        if validated.get("formal_cache_reuse_allowed") is not False:
            raise BuildExecutorError(f"official {smoke_id} smoke reuse drift")
        output[smoke_id] = {
            "path": str(receipt_path.resolve()),
            "sha256": str(row["receipt_file_sha256"]),
            "gate_path": str(gate_path.resolve()),
            "gate_sha256": OFFICIAL_SMOKE_GATE_SHA256[smoke_id],
            "provider_calls": provider_calls,
            "formal_cache_reuse_allowed": False,
        }
    output["bridge"] = {
        "path": str(bridge_path.resolve()),
        "sha256": _file_sha256(bridge_path),
        "bridge_sha256": observed_bridge_sha,
    }
    return output


def _embedding_batches(items: int) -> int:
    if items < 0:
        raise BuildExecutorError("negative embedding item census")
    return math.ceil(items / EMBEDDING_BATCH_SIZE) if items else 0


def _strict_writer_task_rows(
    *, output_root: Path, writer: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Re-open every writer input from its sealed path and project task rows."""

    raw_rows = writer.get("task_inputs")
    required = {
        "task",
        "memory_path",
        "memory_sha256",
        "images",
        "physical_sessions",
    }
    if not isinstance(raw_rows, list) or not raw_rows:
        raise BuildExecutorError("writer task inputs missing")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise BuildExecutorError("writer task input shape drift")
        task = str(raw["task"])
        path = Path(str(raw["memory_path"])).resolve()
        if task in seen or not task or _file_sha256(path) != raw["memory_sha256"]:
            raise BuildExecutorError("writer task memory binding drift")
        seen.add(task)
        memory = _read_mapping(path, f"writer projected memory {task}")
        if memory.get("task") != task or "human-annotated QAs" in memory:
            raise BuildExecutorError("writer task memory boundary drift")
        images = _chronological_unique_images(task, memory)
        sessions = len(memory.get("multi_session_dialogues", []))
        if len(images) != raw["images"] or sessions != raw["physical_sessions"]:
            raise BuildExecutorError("writer task census drift")
        tasks.append(
            {
                "task": task,
                "memory_path": path,
                "memory_sha256": str(raw["memory_sha256"]),
                "memory": memory,
                "images": images,
                "sessions": sessions,
            }
        )
    tasks.sort(key=lambda row: row["task"])
    if [str(row["task"]) for row in raw_rows] != [row["task"] for row in tasks]:
        raise BuildExecutorError("writer task order drift")

    manifest_path = Path(output_root) / "project" / "manifest.json"
    if manifest_path.is_file():
        project = matrix.validate_project(output_root=Path(output_root))
        if writer.get("project_contract_sha256") != project.get(
            "project_contract_sha256"
        ):
            raise BuildExecutorError("writer/project contract drift")
        projected = {
            (
                str(row["task"]),
                str(Path(str(row["memory_path"])).resolve()),
                str(row["memory_sha256"]),
            )
            for row in project["task_receipts"]
        }
        observed = {
            (row["task"], str(row["memory_path"]), row["memory_sha256"])
            for row in tasks
        }
        if observed != projected:
            raise BuildExecutorError("writer/project task identity drift")
    return tasks


def _load_g35_caption_bank_readonly(
    task_row: Mapping[str, Any], *, output_root: Path, image_root: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    """Pure-read reconstruction of the sequential G35 caption bank."""

    task = str(task_row["task"])
    images = list(task_row["images"])
    system_prompt = _load_g35_caption_system_prompt()
    captions: dict[str, str] = {}
    history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    allowed_record_keys = {
        "schema_version",
        "task",
        "image",
        "caption",
        "model",
        "lineage",
        "sequence_index",
        "total_images",
        "previous_images",
        "previous_caption_sha256",
        "summary_contexts",
        "input_contract",
        "input_contract_sha256",
        "provider_request_sha256",
        "provider_response_sha256",
    }
    for index, image in enumerate(images, 1):
        previous = history[-2:]
        prompt = matrix.build_v7_reconstruction_user_text(
            task,
            image,
            sequence_index=index,
            total_images=len(images),
            previous=previous,
        )
        image_path = _resolve_image(image_root, image)
        input_contract = {
            "schema_version": BUILD_EXECUTOR_SCHEMA,
            "backbone": "g35",
            "task": task,
            "image": image,
            "sequence_index": index,
            "total_images": len(images),
            "previous": previous,
            "prompt_sha256": _sha256(prompt),
            "raw_image_sha256": _file_sha256(image_path),
            "system_prompt_sha256": _sha256(system_prompt),
            "system_prompt_line_count": 31,
            "user_part_order": ["raw_image", "previous_two_caption_user_text"],
            "media_resolution": "MEDIA_RESOLUTION_HIGH",
            "max_output_tokens": 2400,
            "max_provider_attempts": 3,
        }
        input_sha = _sha256(input_contract)
        path = (
            Path(output_root)
            / "captions"
            / "g35"
            / task
            / f"{index:04d}_{_safe_name(image)}.json"
        ).resolve()
        record = _read_mapping(path, f"sealed G35 caption {image}")
        caption = str(record.get("caption", "")).strip()
        if (
            set(record) != allowed_record_keys
            or record.get("schema_version") != "creamem3.memeye_g35_caption.v19.0"
            or record.get("task") != task
            or record.get("image") != image
            or record.get("model") != "gemini-3.5-flash-lite"
            or record.get("lineage")
            != "best_available_v7_reconstruction_not_byte_exact"
            or record.get("sequence_index") != index
            or record.get("total_images") != len(images)
            or record.get("previous_images") != [row["image"] for row in previous]
            or record.get("previous_caption_sha256")
            != [_sha256(row["caption"]) for row in previous]
            or record.get("summary_contexts") != 0
            or record.get("input_contract") != input_contract
            or record.get("input_contract_sha256") != input_sha
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("provider_request_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("provider_response_sha256", "")))
            or not caption
        ):
            raise BuildExecutorError(f"sealed G35 caption drift: {path}")
        captions[image] = caption
        history.append({"image": image, "caption": caption})
        rows.append(
            {
                "image": image,
                "sequence_index": index,
                "caption_sha256": _sha256(caption),
                "input_contract_sha256": input_sha,
                "file_sha256": _file_sha256(path),
            }
        )
    return captions, {
        "task": task,
        "images": len(images),
        "model": "gemini-3.5-flash-lite",
        "lineage": "best_available_v7_reconstruction_not_byte_exact",
        "rows_sha256": _sha256(rows),
    }


def _strict_writer_caption_banks(
    *, output_root: Path, writer: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, str]]]:
    sources = writer.get("caption_sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "g25_caption_root",
        "g35_caption_root",
        "image_root",
    }:
        raise BuildExecutorError("writer caption source contract drift")
    g25_root = Path(str(sources["g25_caption_root"])).resolve()
    g35_root = Path(str(sources["g35_caption_root"])).resolve()
    image_root = Path(str(sources["image_root"])).resolve()
    if g35_root != (Path(output_root) / "captions" / "g35").resolve():
        raise BuildExecutorError("G35 caption root escaped build root")
    if (Path(output_root) / "project" / "manifest.json").is_file():
        if (
            g25_root != matrix.DEFAULT_V7_CAPTION_ROOT.resolve()
            or image_root
            != (matrix.DEFAULT_DATA_ROOT / "data" / "image").resolve()
        ):
            raise BuildExecutorError("formal caption/image source drift")
    maps: dict[str, dict[str, dict[str, str]]] = {"g25": {}, "g35": {}}
    expected_receipts: list[dict[str, Any]] = []
    expected_g35_files: set[Path] = set()
    for task_row in tasks:
        task = str(task_row["task"])
        g25, receipt25 = load_g25_caption_bank_for_task(task_row, g25_root)
        g35, receipt35 = _load_g35_caption_bank_readonly(
            task_row, output_root=Path(output_root), image_root=image_root
        )
        maps["g25"][task] = g25
        maps["g35"][task] = g35
        expected_receipts.extend(
            [{"backbone": "g25", **receipt25}, {"backbone": "g35", **receipt35}]
        )
        for index, image in enumerate(task_row["images"], 1):
            expected_g35_files.add(
                (
                    g35_root
                    / task
                    / f"{index:04d}_{_safe_name(str(image))}.json"
                ).resolve()
            )
    if writer.get("caption_receipts") != expected_receipts:
        raise BuildExecutorError("writer caption receipt census drift")
    observed_g35_files = {
        path.resolve() for path in g35_root.rglob("*.json") if path.is_file()
    }
    if observed_g35_files != expected_g35_files:
        raise BuildExecutorError("writer G35 caption file census drift")
    return maps


def _session_packet_input_sha(
    *, backbone: str, task: str, session_id: int, rendered: str, prompt_sha: str
) -> str:
    return _sha256(
        {
            "schema_version": ingest.SCHEMA_VERSION,
            "backbone": backbone,
            "task": task,
            "session_id": session_id,
            "rendered_sha256": _sha256(rendered),
            "prompt_bundle_sha256": prompt_sha,
            "summary_tokens": 1200,
            "fact_tokens": 16000,
            "attempts": 3,
            "current_date": "2026-07-05",
        }
    )


def _native_packet_input_sha(
    *, backbone: str, task: str, batch_index: int, rendered: str, prompt_sha: str
) -> str:
    return _sha256(
        {
            "backbone": backbone,
            "task": task,
            "batch_index": batch_index,
            "rendered_sha256": _sha256(rendered),
            "prompt_bundle_sha256": prompt_sha,
            "summary_tokens": 1200,
            "fact_tokens": 16000,
            "attempts": 3,
            "current_date": "2026-07-05",
            "max_images": 10,
            "max_rounds": 10,
        }
    )


def _strict_writer_packet_rows(
    *,
    output_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    caption_maps: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> list[dict[str, Any]]:
    prompt_bundle = ingest.load_prompt_bundle()
    prompt_sha = str(prompt_bundle["prompt_bundle_sha256"])
    expected_rows: list[dict[str, Any]] = []
    expected_files: set[Path] = set()
    for backbone in ("g25", "g35"):
        for task_row in tasks:
            task = str(task_row["task"])
            captions = caption_maps[backbone][task]
            records, source_map = ingest.build_records_from_captions(
                task, task_row["memory"], captions
            )
            rounds, batches, rendered_batches = ingest.native_batch_plan(
                task, task_row["memory"], captions
            )
            families: list[tuple[str, list[tuple[Path, dict[str, Any]]]]] = []
            session_items: list[tuple[Path, dict[str, Any]]] = []
            for record in records:
                session_id = int(record["session_id"])
                rendered = adapter._session_batch(record)  # type: ignore[attr-defined]
                path = (
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "session"
                    / task
                    / f"S{session_id:04d}.json"
                ).resolve()
                packet = _read_mapping(path, "sealed session packet")
                session_packet_keys = {
                    "schema_version",
                    "backbone",
                    "model",
                    "task",
                    "session_id",
                    "source_session_id",
                    "timestamp",
                    "iso_date",
                    "turn_count",
                    "source_dia_ids",
                    "batch_summary",
                    "atomic_facts",
                    "summary_request_contract_sha256",
                    "fact_request_contract_sha256",
                    "summary_response_sha256",
                    "fact_response_sha256",
                    "summary_logical_calls",
                    "fact_logical_calls",
                    "max_provider_attempts_per_call",
                    "input_sha256",
                    "prompt_bundle_sha256",
                    "build_input_sha256",
                }
                summary_request = ingest.build_summary_request(
                    rendered,
                    prompt_bundle,
                    backbone=backbone,
                    unit_id=f"{task}::session::{session_id:04d}",
                )
                fact_request = ingest.build_creamem_fact_request(
                    record,
                    str(packet.get("batch_summary", "")),
                    prompt_bundle,
                    backbone=backbone,
                    unit_id=f"{task}::session::{session_id:04d}",
                )
                legal = [str(turn["dia_id"]) for turn in record.get("turns", [])]
                expected_timestamp = str(record.get("timestamp", "") or "")
                try:
                    normalized_summary, normalized_facts = gm._validate_session_packet(  # type: ignore[attr-defined]
                        {
                            "batch_summary": packet.get("batch_summary"),
                            "atomic_facts": packet.get("atomic_facts"),
                        },
                        legal,
                    )
                except Exception as exc:
                    raise BuildExecutorError("sealed session packet validation failed") from exc
                if (
                    set(packet) != session_packet_keys
                    or packet.get("schema_version") != ingest.CREAMEM_PACKET_SCHEMA
                    or packet.get("backbone") != backbone
                    or packet.get("model") != ingest.WORK_MODELS[backbone]
                    or packet.get("task") != task
                    or packet.get("session_id") != session_id
                    or packet.get("source_session_id")
                    != str(record.get("source_session_id", ""))
                    or packet.get("timestamp") != expected_timestamp
                    or packet.get("iso_date")
                    != adapter._iso_date(expected_timestamp)  # type: ignore[attr-defined]
                    or packet.get("source_dia_ids") != legal
                    or packet.get("turn_count") != len(legal)
                    or packet.get("batch_summary") != normalized_summary
                    or packet.get("atomic_facts") != normalized_facts
                    or packet.get("summary_request_contract_sha256")
                    != summary_request["request_contract_sha256"]
                    or packet.get("fact_request_contract_sha256")
                    != fact_request["request_contract_sha256"]
                    or packet.get("summary_logical_calls") != 1
                    or packet.get("fact_logical_calls") != 1
                    or packet.get("max_provider_attempts_per_call") != 3
                    or packet.get("input_sha256") != _sha256(rendered)
                    or packet.get("prompt_bundle_sha256") != prompt_sha
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(packet.get("summary_response_sha256", ""))
                    )
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(packet.get("fact_response_sha256", ""))
                    )
                    or packet.get("build_input_sha256")
                    != _session_packet_input_sha(
                        backbone=backbone,
                        task=task,
                        session_id=session_id,
                        rendered=rendered,
                        prompt_sha=prompt_sha,
                    )
                ):
                    raise BuildExecutorError(f"sealed session packet drift: {path}")
                session_items.append((path, packet))
            families.append(("session", session_items))

            native_items: list[tuple[Path, dict[str, Any]]] = []
            authority = ingest.load_native_pure_authority()
            for index, (batch, rendered) in enumerate(
                zip(batches, rendered_batches), start=1
            ):
                path = (
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "mgtm"
                    / task
                    / f"NB{index:04d}.json"
                ).resolve()
                packet = _read_mapping(path, "sealed native packet")
                native_packet_keys = {
                    "schema_version",
                    "backbone",
                    "model",
                    "task",
                    "batch_index",
                    "batch_id",
                    "summary",
                    "facts",
                    "t_start",
                    "t_end",
                    "rounds",
                    "images",
                    "summary_request_contract_sha256",
                    "fact_request_contract_sha256",
                    "summary_response_sha256",
                    "fact_response_sha256",
                    "summary_logical_calls",
                    "fact_logical_calls",
                    "max_provider_attempts_per_call",
                    "rendered_input_sha256",
                    "build_input_sha256",
                }
                summary_request = ingest.build_summary_request(
                    rendered,
                    prompt_bundle,
                    backbone=backbone,
                    unit_id=f"{task}::native::{index:04d}",
                )
                observation_date = str(batch[0].get("date", "") or "2026-07-05")
                native_fact_request = {
                    "stage": "writer",
                    "substage": "native_fact",
                    "backbone": backbone,
                    "model": ingest.WORK_MODELS[backbone],
                    "unit_id": f"{task}::native::{index:04d}",
                    "prompt": authority["build_mem0_user_prompt"](
                        rendered, observation_date, "2026-07-05"
                    ),
                    "system": "\n\n".join(
                        [
                            str(prompt_bundle["mem0_additive_prompt"]),
                            str(prompt_bundle["visual_system_override"]),
                        ]
                    ),
                    "json_mode": True,
                    "max_output_tokens": 16000,
                    "max_provider_attempts": 3,
                    "current_date": "2026-07-05",
                    "rendered_input_sha256": _sha256(rendered),
                    "prompt_bundle_sha256": prompt_sha,
                }
                native_fact_request["request_contract_sha256"] = _sha256(
                    native_fact_request
                )
                if (
                    set(packet) != native_packet_keys
                    or packet.get("schema_version") != ingest.NATIVE_PACKET_SCHEMA
                    or packet.get("backbone") != backbone
                    or packet.get("model") != ingest.WORK_MODELS[backbone]
                    or packet.get("task") != task
                    or packet.get("batch_index") != index
                    or packet.get("batch_id") != f"NB{index:04d}"
                    or not str(packet.get("summary", "")).strip()
                    or not isinstance(packet.get("facts"), list)
                    or any(not str(value).strip() for value in packet.get("facts", []))
                    or packet.get("t_start")
                    != f"{batch[0]['date']} {batch[0]['round']}"
                    or packet.get("t_end")
                    != f"{batch[-1]['date']} {batch[-1]['round']}"
                    or packet.get("rounds") != len(batch)
                    or packet.get("images")
                    != sum(len(row.get("images", [])) for row in batch)
                    or packet.get("summary_request_contract_sha256")
                    != summary_request["request_contract_sha256"]
                    or packet.get("fact_request_contract_sha256")
                    != native_fact_request["request_contract_sha256"]
                    or packet.get("summary_logical_calls") != 1
                    or packet.get("fact_logical_calls") != 1
                    or packet.get("max_provider_attempts_per_call") != 3
                    or packet.get("rendered_input_sha256") != _sha256(rendered)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(packet.get("summary_response_sha256", ""))
                    )
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(packet.get("fact_response_sha256", ""))
                    )
                    or packet.get("build_input_sha256")
                    != _native_packet_input_sha(
                        backbone=backbone,
                        task=task,
                        batch_index=index,
                        rendered=rendered,
                        prompt_sha=prompt_sha,
                    )
                ):
                    raise BuildExecutorError(f"sealed native packet drift: {path}")
                native_items.append((path, packet))
            families.append(("mgtm", native_items))

            for family, items in families:
                expected_files.update(path for path, _packet in items)
                fact_key = "atomic_facts" if family == "session" else "facts"
                expected_rows.append(
                    {
                        "backbone": backbone,
                        "task": task,
                        "family": family,
                        "packets": len(items),
                        "summaries": len(items),
                        "facts": sum(len(packet[fact_key]) for _path, packet in items),
                        "physical_sessions": len(records),
                        "packet_paths": [str(path) for path, _packet in items],
                        "packet_file_sha256": [
                            _file_sha256(path) for path, _packet in items
                        ],
                    }
                )
    observed_files = {
        path.resolve()
        for path in (Path(output_root) / "build" / "packets").rglob("*.json")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise BuildExecutorError("writer packet filesystem census drift")
    return sorted(
        expected_rows, key=lambda row: (row["backbone"], row["task"], row["family"])
    )


def _validate_writer_usage(
    writer: Mapping[str, Any],
    *,
    output_root: Path,
    formal_project: bool,
    packet_rows: Sequence[Mapping[str, Any]],
) -> None:
    usage = writer.get("provider_usage")
    if not isinstance(usage, Mapping) or set(usage) != {"g25", "g35"}:
        raise BuildExecutorError("writer provider usage shape drift")
    calls = sum(_usage_provider_calls(dict(usage[name])) for name in ("g25", "g35"))
    if (
        writer.get("provider_calls") != calls
        or not isinstance(calls, int)
        or calls > BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        or writer.get("generation_provider_attempt_hard_ceiling")
        != BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
    ):
        raise BuildExecutorError("writer provider usage total drift")
    logical_by_backbone = {
        backbone: 2
        * sum(
            int(row["packets"])
            for row in packet_rows
            if row["backbone"] == backbone
        )
        + (
            sum(
                int(row["images"])
                for row in writer["caption_receipts"]
                if row["backbone"] == "g35"
            )
            if backbone == "g35"
            else 0
        )
        for backbone in ("g25", "g35")
    }
    for backbone in ("g25", "g35"):
        observed = _usage_provider_calls(dict(usage[backbone]))
        if observed < logical_by_backbone[backbone]:
            raise BuildExecutorError("writer durable provider census below logical floor")
    if formal_project:
        expected_usage = {
            "g25": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g25",
                included_stages={"writer"},
            ),
            "g35": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g35",
                included_stages={"caption", "writer"},
            ),
        }
        if usage != expected_usage:
            raise BuildExecutorError("writer canonical durable usage drift")
        expected_stage_sets = {"g25": {"writer", "view"}, "g35": {"caption", "writer", "view", "judge"}}
        for backbone, expected_stages in expected_stage_sets.items():
            row = usage[backbone]
            stages = row.get("stages") if isinstance(row, Mapping) else None
            if not isinstance(stages, Mapping) or set(stages) != expected_stages:
                raise BuildExecutorError("writer durable stage census drift")
            if int(stages["view"].get("provider_calls", -1)) != 0:
                raise BuildExecutorError("view call entered writer-only seal")
            if backbone == "g35" and int(stages["judge"].get("provider_calls", -1)) != 0:
                raise BuildExecutorError("judge call entered writer-only seal")
            stage_total = sum(int(stage["provider_calls"]) for stage in stages.values())
            if stage_total != _usage_provider_calls(dict(row)):
                raise BuildExecutorError("writer durable stage/total mismatch")


def _validate_usage_value(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "prompt_tokens",
        "output_tokens",
        "thought_tokens",
        "total_tokens",
    }:
        raise BuildExecutorError(f"{label} usage keyset drift")
    normalized: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "output_tokens",
        "thought_tokens",
        "total_tokens",
    ):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BuildExecutorError(f"{label} usage value drift")
        normalized[key] = int(item)
    if normalized["total_tokens"] != (
        normalized["prompt_tokens"]
        + normalized["output_tokens"]
        + normalized["thought_tokens"]
    ):
        raise BuildExecutorError(f"{label} usage total drift")
    return normalized


def _generation_profile(backbone: str) -> tuple[str, str, dict[str, dict[str, int]]]:
    if backbone == "g25":
        return (
            "gemini-2.5-flash-lite",
            "build_writer_view",
            {
                "writer": {
                    "max_output_tokens": 16000,
                    "max_live_calls": 1848,
                    "max_attempts": 3,
                },
                "view": {
                    "max_output_tokens": 16384,
                    "max_live_calls": 735,
                    "max_attempts": 3,
                },
            },
        )
    if backbone == "g35":
        return (
            "gemini-3.5-flash-lite",
            "build_caption_writer_view",
            {
                "caption": {
                    "max_output_tokens": 2400,
                    "max_live_calls": 1314,
                    "max_attempts": 3,
                },
                "writer": {
                    "max_output_tokens": 16000,
                    "max_live_calls": 1848,
                    "max_attempts": 3,
                },
                "view": {
                    "max_output_tokens": 16384,
                    "max_live_calls": 735,
                    "max_attempts": 3,
                },
                "judge": {
                    "max_output_tokens": 400,
                    "max_live_calls": 0,
                    "max_attempts": 3,
                },
            },
        )
    raise BuildExecutorError(f"unknown generation backbone: {backbone}")


def _canonical_generation_usage(
    *,
    output_root: Path,
    backbone: str,
    included_stages: set[str],
) -> dict[str, Any]:
    """Rebuild deterministic usage only from durable attempt records.

    This deliberately excludes process-local cache-hit/provider counters.  A
    resumed process therefore seals the same receipt as the process that made
    the calls.  The caller controls the phase scope: writer-only seals count
    caption/writer while the final build seal also counts view.
    """

    model, profile, limits = _generation_profile(backbone)
    stages = set(limits)
    if not included_stages.issubset(stages):
        raise BuildExecutorError("generation usage stage scope drift")
    cache_root = (Path(output_root) / "provider" / backbone).resolve()
    stage_rows: dict[str, dict[str, Any]] = {}
    for stage in sorted(stages):
        request_root = cache_root / "attempts" / stage
        request_paths = (
            sorted(request_root.rglob("attempt_*_request.json"))
            if request_root.is_dir()
            else []
        )
        response_paths = (
            sorted(request_root.rglob("attempt_*_response.json"))
            if request_root.is_dir()
            else []
        )
        final_root = cache_root / "responses" / stage
        final_paths = (
            sorted(final_root.glob("*.json")) if final_root.is_dir() else []
        )
        if stage not in included_stages:
            request_paths = []
            response_paths = []
            final_paths = []
        elif len(request_paths) != len(response_paths):
            raise BuildExecutorError(
                f"{backbone} {stage} durable attempt response census drift"
            )
        usage = {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }
        for response_path in response_paths:
            response = _read_mapping(
                response_path, f"{backbone} {stage} durable attempt response"
            )
            if "usage" not in response:
                if response.get("outcome_kind") != "explicit_provider_error":
                    raise BuildExecutorError(
                        f"{backbone} {stage} attempt lacks usage unexpectedly"
                    )
                continue
            observed = _validate_usage_value(
                response["usage"], label=f"{backbone} {stage} attempt"
            )
            for key, item in observed.items():
                usage[key] += item
        stage_rows[stage] = {
            "provider_calls": len(request_paths),
            "durable_provider_attempt_intents": len(request_paths),
            "completed_identities": len(final_paths),
            "usage": usage,
            "limits": limits[stage],
        }
    return {
        "schema_version": GENERATION_USAGE_SCHEMA,
        "backbone": backbone,
        "model": model,
        "profile": profile,
        "cache_root": str(cache_root),
        "included_stages": sorted(included_stages),
        "stages": stage_rows,
    }


def validate_writer_complete(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the pre-embedding writer packet seal and every packet hash."""

    protocol = dict(protocol or matrix.load_protocol())
    path = Path(output_root) / "build" / "writer_complete.json"
    value = _read_mapping(path, "writer packet completion seal")
    contract = dict(value)
    observed = contract.pop("writer_complete_sha256", None)
    if observed != _sha256(contract):
        raise BuildExecutorError("writer completion digest drift")
    required_keys = {
        "schema_version",
        "complete",
        "experiment_id",
        "protocol_sha256",
        "project_contract_sha256",
        "task_inputs",
        "caption_sources",
        "caption_receipts",
        "packet_rows",
        "packet_rows_sha256",
        "provider_usage",
        "provider_calls",
        "generation_provider_attempt_hard_ceiling",
        "embedding_calls",
        "embedding_budget_status",
        "writer_complete_sha256",
    }
    if (
        set(value) != required_keys
        or value.get("schema_version") != WRITER_COMPLETE_SCHEMA
        or value.get("complete") is not True
        or value.get("experiment_id") != protocol.get("experiment_id")
        or value.get("protocol_sha256") != _sha256(protocol)
        or value.get("embedding_calls") != 0
        or value.get("embedding_budget_status")
        != "pending_zero_call_census_and_independent_gate"
    ):
        raise BuildExecutorError("writer completion contract drift")
    tasks = _strict_writer_task_rows(output_root=Path(output_root), writer=value)
    caption_maps = _strict_writer_caption_banks(
        output_root=Path(output_root), writer=value, tasks=tasks
    )
    rows = value.get("packet_rows")
    if not isinstance(rows, list):
        raise BuildExecutorError("writer packet rows missing")
    expected_rows = _strict_writer_packet_rows(
        output_root=Path(output_root), tasks=tasks, caption_maps=caption_maps
    )
    if rows != expected_rows or _sha256(expected_rows) != value.get(
        "packet_rows_sha256"
    ):
        raise BuildExecutorError("writer packet row census drift")
    formal_project = (Path(output_root) / "project" / "manifest.json").is_file()
    if formal_project:
        if (
            len(tasks) != EXPECTED_TASKS
            or len(rows) != EXPECTED_TASKS * 2 * 2
            or len(value["caption_receipts"]) != EXPECTED_TASKS * 2
            or sum(int(row["images"]) for row in value["caption_receipts"])
            != EXPECTED_IMAGES * 2
            or sum(
                int(row["packets"])
                for row in rows
                if row["family"] == "session"
            )
            != EXPECTED_SESSIONS * 2
            or sum(
                int(row["packets"])
                for row in rows
                if row["family"] == "mgtm"
            )
            != EXPECTED_NATIVE_BATCHES * 2
        ):
            raise BuildExecutorError("formal writer exact census drift")
    _validate_writer_usage(
        value,
        output_root=Path(output_root),
        formal_project=formal_project,
        packet_rows=rows,
    )
    return value


def _iter_writer_generation_specs(
    *,
    output_root: Path,
    writer: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    caption_maps: Mapping[str, Mapping[str, Mapping[str, str]]],
):
    """Yield every immutable caption/summary/fact request and output digest."""

    prompt_bundle = ingest.load_prompt_bundle()
    image_root = Path(str(writer["caption_sources"]["image_root"])).resolve()
    caption_system = _load_g35_caption_system_prompt()
    for task_row in tasks:
        task = str(task_row["task"])
        images = list(task_row["images"])
        history: list[dict[str, str]] = []
        for index, image in enumerate(images, 1):
            previous = history[-2:]
            prompt = matrix.build_v7_reconstruction_user_text(
                task,
                image,
                sequence_index=index,
                total_images=len(images),
                previous=previous,
            )
            record_path = (
                Path(output_root)
                / "captions"
                / "g35"
                / task
                / f"{index:04d}_{_safe_name(image)}.json"
            )
            record = _read_mapping(record_path, "G35 caption generation evidence")
            caption = str(record["caption"])
            metadata = {
                "input_contract_sha256": str(record["input_contract_sha256"]),
                "task": task,
                "image": image,
                "sequence_index": index,
                "matched_substage": None,
                "max_provider_attempts": 3,
            }
            yield {
                "backbone": "g35",
                "stage": "caption",
                "unit_id": f"{task}::caption::{index:04d}",
                "prompt": prompt,
                "system": caption_system,
                "json_mode": False,
                "max_output_tokens": 2400,
                "cache_metadata": metadata,
                "image_path": _resolve_image(image_root, image),
                "caption_context_count": len(previous),
                "expected_response_sha256": str(
                    record["provider_response_sha256"]
                ),
                "output_kind": "caption",
                "expected_output": caption,
                "legal_source_dia_ids": None,
            }
            history.append({"image": image, "caption": caption})

    for backbone in ("g25", "g35"):
        for task_row in tasks:
            task = str(task_row["task"])
            captions = caption_maps[backbone][task]
            records, source_map = ingest.build_records_from_captions(
                task, task_row["memory"], captions
            )
            for record in records:
                session_id = int(record["session_id"])
                unit_id = f"{task}::session::{session_id:04d}"
                packet_path = (
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "session"
                    / task
                    / f"S{session_id:04d}.json"
                )
                packet = _read_mapping(packet_path, "session generation evidence")
                rendered = adapter._session_batch(record)  # type: ignore[attr-defined]
                requests = [
                    (
                        ingest.build_summary_request(
                            rendered,
                            prompt_bundle,
                            backbone=backbone,
                            unit_id=unit_id,
                        ),
                        str(packet["summary_response_sha256"]),
                    ),
                    (
                        ingest.build_creamem_fact_request(
                            record,
                            str(packet["batch_summary"]),
                            prompt_bundle,
                            backbone=backbone,
                            unit_id=unit_id,
                        ),
                        str(packet["fact_response_sha256"]),
                    ),
                ]
                for request, response_sha in requests:
                    metadata = {
                        "schema_version": ingest.SCHEMA_VERSION,
                        "substage": request["substage"],
                        "backbone": backbone,
                        "unit_id": unit_id,
                        "request_contract_sha256": request[
                            "request_contract_sha256"
                        ],
                        "matched_substage": request["substage"],
                        "max_provider_attempts": 3,
                    }
                    yield {
                        "backbone": backbone,
                        "stage": "writer",
                        "unit_id": f"{unit_id}::{request['substage']}",
                        "prompt": str(request["prompt"]),
                        "system": request.get("system"),
                        "json_mode": True,
                        "max_output_tokens": int(request["max_output_tokens"]),
                        "cache_metadata": metadata,
                        "image_path": None,
                        "caption_context_count": None,
                        "expected_response_sha256": response_sha,
                        "output_kind": (
                            "session_summary"
                            if request["substage"] == "summary"
                            else "session_facts"
                        ),
                        "expected_output": (
                            str(packet["batch_summary"])
                            if request["substage"] == "summary"
                            else copy.deepcopy(packet["atomic_facts"])
                        ),
                        "legal_source_dia_ids": [
                            str(turn["dia_id"]) for turn in record.get("turns", [])
                        ],
                    }
            rounds, batches, rendered_batches = ingest.native_batch_plan(
                task, task_row["memory"], captions
            )
            authority = ingest.load_native_pure_authority()
            for index, (batch, rendered) in enumerate(
                zip(batches, rendered_batches), 1
            ):
                unit_id = f"{task}::native::{index:04d}"
                packet_path = (
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "mgtm"
                    / task
                    / f"NB{index:04d}.json"
                )
                packet = _read_mapping(packet_path, "native generation evidence")
                summary_request = ingest.build_summary_request(
                    rendered,
                    prompt_bundle,
                    backbone=backbone,
                    unit_id=unit_id,
                )
                observation_date = str(batch[0].get("date", "") or "2026-07-05")
                fact_request = {
                    "stage": "writer",
                    "substage": "native_fact",
                    "backbone": backbone,
                    "model": ingest.WORK_MODELS[backbone],
                    "unit_id": unit_id,
                    "prompt": authority["build_mem0_user_prompt"](
                        rendered, observation_date, "2026-07-05"
                    ),
                    "system": "\n\n".join(
                        [
                            str(prompt_bundle["mem0_additive_prompt"]),
                            str(prompt_bundle["visual_system_override"]),
                        ]
                    ),
                    "json_mode": True,
                    "max_output_tokens": 16000,
                    "max_provider_attempts": 3,
                    "current_date": "2026-07-05",
                    "rendered_input_sha256": _sha256(rendered),
                    "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
                }
                fact_request["request_contract_sha256"] = _sha256(fact_request)
                for request, response_sha in (
                    (summary_request, str(packet["summary_response_sha256"])),
                    (fact_request, str(packet["fact_response_sha256"])),
                ):
                    metadata = {
                        "schema_version": ingest.SCHEMA_VERSION,
                        "substage": request["substage"],
                        "backbone": backbone,
                        "unit_id": unit_id,
                        "request_contract_sha256": request[
                            "request_contract_sha256"
                        ],
                        "matched_substage": request["substage"],
                        "max_provider_attempts": 3,
                    }
                    yield {
                        "backbone": backbone,
                        "stage": "writer",
                        "unit_id": f"{unit_id}::{request['substage']}",
                        "prompt": str(request["prompt"]),
                        "system": request.get("system"),
                        "json_mode": True,
                        "max_output_tokens": int(request["max_output_tokens"]),
                        "cache_metadata": metadata,
                        "image_path": None,
                        "caption_context_count": None,
                        "expected_response_sha256": response_sha,
                        "output_kind": (
                            "native_summary"
                            if request["substage"] == "summary"
                            else "native_facts"
                        ),
                        "expected_output": (
                            str(packet["summary"])
                            if request["substage"] == "summary"
                            else copy.deepcopy(packet["facts"])
                        ),
                        "native_summary_fallback": (
                            f"{task} rounds {batch[0]['round']}..{batch[-1]['round']}"
                            if request["substage"] == "summary"
                            else None
                        ),
                        "legal_source_dia_ids": None,
                    }


def _generation_cache_identity_sets(cache_root: Path, stages: set[str]) -> dict[str, Any]:
    def stems(root: Path, suffix: str) -> set[str]:
        return {path.stem for path in root.glob(f"*{suffix}") if path.is_file()}

    return {
        "responses": {
            stage: stems(cache_root / "responses" / stage, ".json")
            for stage in stages
        },
        "plans": {
            stage: stems(cache_root / "plans" / stage, ".json") for stage in stages
        },
        "attempts": {
            stage: {
                path.name
                for path in (cache_root / "attempts" / stage).iterdir()
                if path.is_dir()
            }
            if (cache_root / "attempts" / stage).is_dir()
            else set()
            for stage in stages
        },
        "lifecycle": {
            stage: stems(cache_root / "lifecycle_locks" / stage, ".lock")
            for stage in stages
        },
    }


def _assert_writer_generation_body_output(
    spec: Mapping[str, Any], text: str
) -> None:
    kind = str(spec["output_kind"])
    expected = spec["expected_output"]
    if kind == "caption":
        observed: Any = str(text).strip()
    elif kind in {"session_summary", "native_summary"}:
        parsed = adapter._parse_json(str(text))  # type: ignore[attr-defined]
        raw_summary = str(parsed.get("batch_summary", "")).strip()
        if kind == "native_summary" and not raw_summary:
            raw_summary = str(spec.get("native_summary_fallback", ""))
        observed = (
            gm._normalize_text(raw_summary)  # type: ignore[attr-defined]
            if kind == "session_summary"
            else raw_summary
        )
    elif kind == "session_facts":
        parsed = adapter._parse_json(str(text))  # type: ignore[attr-defined]
        if set(parsed) != {"atomic_facts"}:
            raise BuildExecutorError("session fact raw body schema drift")
        _summary, observed = gm._validate_session_packet(  # type: ignore[attr-defined]
            {
                "batch_summary": "generation evidence placeholder",
                "atomic_facts": parsed["atomic_facts"],
            },
            list(spec["legal_source_dia_ids"]),
        )
    elif kind == "native_facts":
        observed = ingest.parse_native_facts(str(text))
    else:
        raise BuildExecutorError("unknown writer generation output kind")
    if observed != expected:
        raise BuildExecutorError(f"writer generation body/output drift: {kind}")


def _writer_generation_evidence_contract(
    *,
    output_root: Path,
    protocol: Mapping[str, Any],
    writer: Mapping[str, Any],
    allow_downstream_view: bool = False,
) -> dict[str, Any]:
    """Pure-read replay of every formal writer/caption cache identity."""

    if not (Path(output_root) / "project" / "manifest.json").is_file():
        return {
            "schema_version": WRITER_GENERATION_EVIDENCE_SCHEMA,
            "status": "test_injected_generation_backends_not_formal",
            "protocol_sha256": _sha256(protocol),
            "writer_complete_file_sha256": _file_sha256(
                Path(output_root) / "build" / "writer_complete.json"
            ),
            "provider_calls_during_validation": 0,
        }

    tasks = _strict_writer_task_rows(output_root=Path(output_root), writer=writer)
    caption_maps = _strict_writer_caption_banks(
        output_root=Path(output_root), writer=writer, tasks=tasks
    )
    g25_adapter_path = Path(__file__).with_name(
        "official_g25_build_runtime_adapter_v19.py"
    )
    g25_adapter_sha = (
        "c04d8d16084e623faf1ee771ae91fe41e2518b57fc48cee22056c77819908a1f"
    )
    if _file_sha256(g25_adapter_path) != g25_adapter_sha:
        raise BuildExecutorError("G25 writer evidence authority drift before import")
    g35_authority = protocol["authority_files"]["g35_runtime"]
    g35_path = matrix._resolve_authority_path(str(g35_authority["path"]))
    if _file_sha256(g35_path) != str(g35_authority["sha256"]):
        raise BuildExecutorError("G35 writer evidence authority drift before import")
    g25_module = importlib.import_module(
        "creamem3.official_g25_build_runtime_adapter_v19"
    )
    g35_module = importlib.import_module("creamem3.memeye_g35_runtime_v19")
    if (
        Path(g25_module.__file__).resolve() != g25_adapter_path.resolve()
        or Path(g35_module.__file__).resolve() != g35_path.resolve()
    ):
        raise BuildExecutorError("writer evidence runtime import path drift")
    g25_cache = (Path(output_root) / "provider" / "g25").resolve()
    g35_cache = (Path(output_root) / "provider" / "g35").resolve()
    g25_facade = g25_module.OfficialGemini25BuildRuntime(
        g25_cache, allow_model_calls=False
    )
    g25_core = g25_facade._runtime
    g25_source = g25_facade._module
    g35_limits = dict(g35_module.DEFAULT_STAGE_LIMITS)
    g35_limits["caption"] = g35_module.StageLimits(2400, 1314, 3)
    g35_limits["writer"] = g35_module.StageLimits(16000, 1848, 3)
    g35_limits["view"] = g35_module.StageLimits(16384, 735, 3)
    g35_limits["judge"] = g35_module.StageLimits(400, 0, 3)
    g35_core = g35_module.Gemini35Runtime(
        g35_cache, allow_model_calls=False, stage_limits=g35_limits
    )

    expected: dict[str, dict[str, dict[str, Any]]] = {
        "g25": {"writer": {}, "view": {}},
        "g35": {"caption": {}, "writer": {}, "view": {}, "judge": {}},
    }
    identity_rows: list[dict[str, Any]] = []
    for spec in _iter_writer_generation_specs(
        output_root=Path(output_root),
        writer=writer,
        tasks=tasks,
        caption_maps=caption_maps,
    ):
        backbone = str(spec["backbone"])
        stage = str(spec["stage"])
        parts: list[Mapping[str, Any]] = []
        if spec["image_path"] is not None:
            image_path = Path(spec["image_path"])
            mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
            parts = [g35_module.inline_image_part(image_path.read_bytes(), mime)]
        if backbone == "g25":
            source = g25_source
            limits = g25_facade.stage_limits
        else:
            source = g35_module
            limits = g35_limits
        request = source.build_generate_content_request(
            stage,
            str(spec["prompt"]),
            system=spec["system"],
            parts=parts,
            json_mode=bool(spec["json_mode"]),
            max_output_tokens=int(spec["max_output_tokens"]),
            stage_limits=limits,
            caption_context_count=spec["caption_context_count"],
        )
        request_sha = source.request_identity(
            stage=stage,
            request=request,
            stage_limits=limits,
            cache_metadata=spec["cache_metadata"],
        )
        resumable = source.resume_hash(
            stage=stage,
            request_sha256=request_sha,
            stage_limits=limits,
        )
        if resumable in expected[backbone][stage]:
            raise BuildExecutorError("duplicate writer generation identity")
        expected[backbone][stage][resumable] = {
            "request_sha256": request_sha,
            "response_sha256": str(spec["expected_response_sha256"]),
            "unit_id": str(spec["unit_id"]),
            "request": request,
            "prompt": str(spec["prompt"]),
            "cache_metadata": dict(spec["cache_metadata"]),
            "caption_context_count": spec["caption_context_count"],
        }
        identity_rows.append(
            {
                "backbone": backbone,
                "stage": stage,
                "unit_id": str(spec["unit_id"]),
                "resume_hash": resumable,
                "request_sha256": request_sha,
                "response_sha256": str(spec["expected_response_sha256"]),
            }
        )
    if (
        len(expected["g25"]["writer"]) != 616
        or len(expected["g35"]["caption"]) != 438
        or len(expected["g35"]["writer"]) != 616
    ):
        raise BuildExecutorError("formal writer generation identity census drift")

    stage_attempts: dict[str, dict[str, int]] = {"g25": {}, "g35": {}}
    for backbone, (cache_root, runtime, source) in {
        "g25": (g25_cache, g25_core, g25_source),
        "g35": (g35_cache, g35_core, g35_module),
    }.items():
        expected_cache_files: set[Path] = set()
        stages = set(expected[backbone])
        for category in ("responses", "plans", "attempts", "lifecycle_locks"):
            category_root = cache_root / category
            observed_stage_dirs = (
                {path.name for path in category_root.iterdir() if path.is_dir()}
                if category_root.is_dir()
                else set()
            )
            if not observed_stage_dirs.issubset(stages):
                raise BuildExecutorError(
                    f"{backbone} unexpected generation stage cache"
                )
        for json_path in cache_root.rglob("*.json") if cache_root.is_dir() else ():
            relative = json_path.relative_to(cache_root)
            if not relative.parts or relative.parts[0] not in {
                "responses",
                "plans",
                "attempts",
                "orphans",
                "failures",
            }:
                raise BuildExecutorError(
                    f"{backbone} unexpected generation cache JSON"
                )
        census = _generation_cache_identity_sets(cache_root, stages)
        expected_sets = {
            stage: set(expected[backbone][stage]) for stage in stages
        }
        for kind in ("responses", "plans", "attempts", "lifecycle"):
            for stage in stages:
                if allow_downstream_view and stage == "view":
                    continue
                if census[kind][stage] != expected_sets[stage]:
                    raise BuildExecutorError(
                        f"{backbone} writer generation {kind} exact census drift"
                    )
        for forbidden in ("orphans", "failures"):
            root = cache_root / forbidden
            if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
                raise BuildExecutorError(
                    f"{backbone} writer generation {forbidden} is nonempty"
                )
        outer_lease = cache_root / "formal_global_generation_budget.lock"
        if outer_lease.exists():
            raise BuildExecutorError(f"{backbone} writer generation lease is stale")
        for stage, identities in expected[backbone].items():
            attempts = 0
            for resumable, binding in identities.items():
                final_path = cache_root / "responses" / stage / f"{resumable}.json"
                expected_cache_files.add(final_path.resolve())
                final = _read_mapping(final_path, "writer generation final")
                ordinal = int(final.get("attempt_ordinal", 0))
                if ordinal < 1 or ordinal > 3:
                    raise BuildExecutorError("writer generation success ordinal drift")
                attempt_dir = cache_root / "attempts" / stage / resumable
                expected_attempt_files = {
                    attempt_dir / f"attempt_{number:03d}_{suffix}.json"
                    for number in range(1, ordinal + 1)
                    for suffix in ("request", "response")
                }
                expected_cache_files.update(
                    path.resolve() for path in expected_attempt_files
                )
                observed_attempt_files = {
                    path for path in attempt_dir.iterdir() if path.is_file()
                }
                if (
                    observed_attempt_files != expected_attempt_files
                    or any(path.is_dir() for path in attempt_dir.iterdir())
                ):
                    raise BuildExecutorError("writer generation attempt chain drift")
                plan = runtime._plan_record(
                    stage=stage,
                    request=binding["request"],
                    prompt=binding["prompt"],
                    request_sha=binding["request_sha256"],
                    resumable=resumable,
                    caption_context_count=binding["caption_context_count"],
                    cache_metadata=binding["cache_metadata"],
                )
                plan_path = cache_root / "plans" / stage / f"{resumable}.json"
                expected_cache_files.add(plan_path.resolve())
                if _read_mapping(plan_path, "writer generation plan") != plan:
                    raise BuildExecutorError("writer generation plan drift")
                success_response: dict[str, Any] | None = None
                for number in range(1, ordinal + 1):
                    expected_request = runtime._attempt_request_record(
                        stage=stage,
                        request_sha=binding["request_sha256"],
                        resumable=resumable,
                        ordinal=number,
                        plan=plan,
                    )
                    request_path = (
                        attempt_dir / f"attempt_{number:03d}_request.json"
                    )
                    response_path = (
                        attempt_dir / f"attempt_{number:03d}_response.json"
                    )
                    if _read_mapping(
                        request_path, "writer generation attempt request"
                    ) != expected_request:
                        raise BuildExecutorError(
                            "writer generation attempt request drift"
                        )
                    response = _read_mapping(
                        response_path, "writer generation attempt response"
                    )
                    common = {
                        "schema_version": expected_request["schema_version"],
                        "protocol_sha256": expected_request["protocol_sha256"],
                        "stage": stage,
                        "model": expected_request["model"],
                        "request_sha256": binding["request_sha256"],
                        "resume_hash": resumable,
                        "attempt_ordinal": number,
                        "attempt_identity_sha256": expected_request[
                            "attempt_identity_sha256"
                        ],
                    }
                    if any(response.get(key) != item for key, item in common.items()):
                        raise BuildExecutorError(
                            "writer generation attempt response binding drift"
                        )
                    outcome = response.get("outcome_kind")
                    if number < ordinal:
                        required_error_keys = set(common) | {
                            "outcome_kind",
                            "retryable",
                            "status_code",
                            "error_type",
                            "error_message",
                            "error_sha256",
                        }
                        if (
                            outcome != "explicit_provider_error"
                            or response.get("retryable") is not True
                            or frozenset(response)
                            not in {frozenset(required_error_keys), frozenset(required_error_keys | {"usage"})}
                            or not str(response.get("error_type", ""))
                            or response.get("error_sha256")
                            != _sha256(str(response.get("error_message", "")))
                        ):
                            raise BuildExecutorError(
                                "writer generation explicit retry chain drift"
                            )
                    else:
                        raw_keys = set(common) | {
                            "outcome_kind",
                            "response_sha256",
                            "text",
                            "usage",
                        }
                        if (
                            set(response) != raw_keys
                            or outcome != "raw_response"
                            or response.get("response_sha256")
                            != _sha256(str(response.get("text", "")))
                        ):
                            raise BuildExecutorError(
                                "writer generation successful response drift"
                            )
                        success_response = response
                if success_response is None:
                    raise BuildExecutorError("writer generation success response missing")
                expected_final = {
                    **{
                        key: success_response[key]
                        for key in (
                            "schema_version",
                            "protocol_sha256",
                            "stage",
                            "model",
                            "request_sha256",
                            "resume_hash",
                            "attempt_ordinal",
                            "response_sha256",
                            "text",
                            "usage",
                        )
                    },
                    "cache_path": str(final_path),
                }
                if final != expected_final:
                    raise BuildExecutorError("writer generation final record drift")
                lock_path = cache_root / "lifecycle_locks" / stage / f"{resumable}.lock"
                expected_cache_files.add(lock_path.resolve())
                if lock_path.read_bytes() != b"\0":
                    raise BuildExecutorError("writer generation lifecycle bytes drift")
                with source.acquire_request_lifecycle_lock(
                    lock_path, timeout_seconds=0.1, poll_seconds=0.01
                ):
                    pass
                attempts += ordinal
                if (
                    final.get("request_sha256") != binding["request_sha256"]
                    or final.get("response_sha256") != binding["response_sha256"]
                    or _sha256(str(final.get("text", "")))
                    != binding["response_sha256"]
                ):
                    raise BuildExecutorError("writer generation final/output drift")
            stage_attempts[backbone][stage] = attempts
        observed_cache_files: set[Path] = set()
        for path in cache_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(cache_root)
            # A post-view audit owns the view namespace in the final evidence
            # seal; all other namespaces remain phase-1 exact here.
            if (
                allow_downstream_view
                and len(relative.parts) >= 2
                and relative.parts[0]
                in {"responses", "plans", "attempts", "lifecycle_locks"}
                and relative.parts[1] == "view"
            ):
                continue
            observed_cache_files.add(path.resolve())
        if observed_cache_files != expected_cache_files:
            raise BuildExecutorError(
                f"{backbone} writer generation exact filesystem census drift"
            )

    # Only after every expected cache file exists and every intent is closed do
    # we ask the frozen runtimes to replay their full plan/attempt/final checks.
    for spec in _iter_writer_generation_specs(
        output_root=Path(output_root),
        writer=writer,
        tasks=tasks,
        caption_maps=caption_maps,
    ):
        backbone = str(spec["backbone"])
        parts: list[Mapping[str, Any]] = []
        if spec["image_path"] is not None:
            image_path = Path(spec["image_path"])
            mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
            parts = [g35_module.inline_image_part(image_path.read_bytes(), mime)]
        runtime = g25_core if backbone == "g25" else g35_core
        result = runtime.generate(
            str(spec["stage"]),
            str(spec["prompt"]),
            system=spec["system"],
            parts=parts,
            json_mode=bool(spec["json_mode"]),
            max_output_tokens=int(spec["max_output_tokens"]),
            caption_context_count=spec["caption_context_count"],
            cache_metadata=spec["cache_metadata"],
        )
        if (
            result.cache_hit is not True
            or result.response_sha256 != spec["expected_response_sha256"]
            or _sha256(result.text) != spec["expected_response_sha256"]
        ):
            raise BuildExecutorError("writer generation pure-read replay drift")
        _assert_writer_generation_body_output(spec, result.text)
    g25_usage = g25_core.usage_receipt()
    g35_usage = g35_core.usage_receipt()
    if sum(int(row["provider_calls"]) for row in g25_usage["stages"].values()) != 0:
        raise BuildExecutorError("G25 writer evidence made a provider call")
    if sum(int(row["provider_calls"]) for row in g35_usage["stages"].values()) != 0:
        raise BuildExecutorError("G35 writer evidence made a provider call")
    sealed_usage = writer["provider_usage"]
    canonical_usage = {
        "g25": _canonical_generation_usage(
            output_root=Path(output_root),
            backbone="g25",
            included_stages={"writer"},
        ),
        "g35": _canonical_generation_usage(
            output_root=Path(output_root),
            backbone="g35",
            included_stages={"caption", "writer"},
        ),
    }
    if sealed_usage != canonical_usage:
        raise BuildExecutorError("writer generation canonical usage drift")
    for backbone in ("g25", "g35"):
        for stage, attempts in stage_attempts[backbone].items():
            if int(sealed_usage[backbone]["stages"][stage]["provider_calls"]) != attempts:
                raise BuildExecutorError("writer generation durable usage drift")
    identity_rows.sort(
        key=lambda row: (row["backbone"], row["stage"], row["unit_id"])
    )
    return {
        "schema_version": WRITER_GENERATION_EVIDENCE_SCHEMA,
        "status": "pass",
        "protocol_sha256": _sha256(protocol),
        "writer_complete_file_sha256": _file_sha256(
            Path(output_root) / "build" / "writer_complete.json"
        ),
        "g25_runtime_adapter_sha256": g25_adapter_sha,
        "g35_runtime_sha256": str(g35_authority["sha256"]),
        "cache_roots": {"g25": str(g25_cache), "g35": str(g35_cache)},
        "logical_identities": {
            "g25_writer": 616,
            "g35_caption": 438,
            "g35_writer": 616,
            "total": 1670,
        },
        "durable_provider_attempts": stage_attempts,
        "durable_usage_sha256": _sha256(canonical_usage),
        "identity_rows_sha256": _sha256(identity_rows),
        "exact_cache_identity_census": True,
        "attempt_chains_validated": True,
        "orphans_and_failures_empty": True,
        "provider_calls_during_validation": 0,
        "downstream_view_namespace_allowed": allow_downstream_view,
    }


def seal_writer_generation_evidence(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    protocol = dict(protocol or matrix.load_protocol())
    writer = validate_writer_complete(output_root=output_root, protocol=protocol)
    value = _writer_generation_evidence_contract(
        output_root=Path(output_root), protocol=protocol, writer=writer
    )
    value["generation_evidence_sha256"] = _sha256(value)
    path = Path(output_root) / "build" / "writer_generation_evidence.json"
    _write_once_json(path, value)
    if _read_mapping(path, "sealed writer generation evidence") != value:
        raise BuildExecutorError("writer generation evidence write-once drift")
    return value


def validate_writer_generation_evidence(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    protocol = dict(protocol or matrix.load_protocol())
    writer = validate_writer_complete(output_root=output_root, protocol=protocol)
    path = Path(output_root) / "build" / "writer_generation_evidence.json"
    value = _read_mapping(path, "writer generation evidence")
    contract = dict(value)
    observed = contract.pop("generation_evidence_sha256", None)
    if observed != _sha256(contract):
        raise BuildExecutorError("writer generation evidence digest drift")
    expected = _writer_generation_evidence_contract(
        output_root=Path(output_root), protocol=protocol, writer=writer
    )
    expected["generation_evidence_sha256"] = _sha256(expected)
    if value != expected:
        raise BuildExecutorError("writer generation evidence contract drift")
    return value


def _validate_sealed_writer_generation_evidence_binding(
    *, output_root: Path, protocol: Mapping[str, Any], writer: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the immutable phase-1 seal without reasserting view emptiness.

    The full validator above is intentionally pre-view and proves that view and
    judge namespaces were empty when the dynamic budget was signed.  Once
    phase 2 legitimately creates view cache entries, the phase-1 file remains
    immutable and is consumed through this hash/self-hash binding; a separate
    final generation evidence seal owns the enlarged cache census.
    """

    path = Path(output_root) / "build" / "writer_generation_evidence.json"
    value = _read_mapping(path, "sealed writer generation evidence binding")
    contract = dict(value)
    observed = contract.pop("generation_evidence_sha256", None)
    if (
        observed != _sha256(contract)
        or value.get("schema_version") != WRITER_GENERATION_EVIDENCE_SCHEMA
        or value.get("status")
        not in {"pass", "test_injected_generation_backends_not_formal"}
        or value.get("protocol_sha256") != _sha256(protocol)
        or value.get("writer_complete_file_sha256")
        != _file_sha256(Path(output_root) / "build" / "writer_complete.json")
    ):
        raise BuildExecutorError("sealed writer generation evidence binding drift")
    if (
        (Path(output_root) / "project" / "manifest.json").is_file()
        and (
            value.get("status") != "pass"
            or value.get("provider_calls_during_validation") != 0
            or value.get("logical_identities")
            != {
                "g25_writer": 616,
                "g35_caption": 438,
                "g35_writer": 616,
                "total": 1670,
            }
            or not isinstance(value.get("durable_usage_sha256"), str)
        )
    ):
        raise BuildExecutorError("formal writer generation evidence binding drift")
    if (Path(output_root) / "project" / "manifest.json").is_file():
        deep = _writer_generation_evidence_contract(
            output_root=Path(output_root),
            protocol=protocol,
            writer=writer,
            allow_downstream_view=True,
        )
        # The phase-1 seal records that downstream namespaces were empty at
        # signing time; the post-view replay audits the immutable writer subset
        # while delegating the now-populated view namespace to the final seal.
        deep["downstream_view_namespace_allowed"] = False
        deep["generation_evidence_sha256"] = _sha256(deep)
        if value != deep:
            raise BuildExecutorError(
                "post-view writer generation subset replay drift"
            )
    return value


class _ReadOnlyViewGenerationReplay:
    """Reconstruct exact view requests and consume only already-sealed finals."""

    def __init__(
        self,
        *,
        output_root: Path,
        backbone: str,
        protocol: Mapping[str, Any],
    ) -> None:
        self.backbone = backbone
        self.cache_root = (Path(output_root) / "provider" / backbone).resolve()
        self.records: list[dict[str, Any]] = []
        if backbone == "g25":
            authority_path = Path(__file__).with_name(
                "official_g25_build_runtime_adapter_v19.py"
            )
            authority_sha = (
                "c04d8d16084e623faf1ee771ae91fe41e2518b57fc48cee22056c77819908a1f"
            )
            if _file_sha256(authority_path) != authority_sha:
                raise BuildExecutorError(
                    "G25 view replay authority drift before import"
                )
            module = importlib.import_module(
                "creamem3.official_g25_build_runtime_adapter_v19"
            )
            if Path(module.__file__).resolve() != authority_path.resolve():
                raise BuildExecutorError("G25 view replay import path drift")
            facade = module.OfficialGemini25BuildRuntime(
                self.cache_root, allow_model_calls=False
            )
            self.source = facade._module
            self.runtime = facade._runtime
            self.stage_limits = facade.stage_limits
        elif backbone == "g35":
            authority = protocol["authority_files"]["g35_runtime"]
            authority_path = matrix._resolve_authority_path(str(authority["path"]))
            if _file_sha256(authority_path) != str(authority["sha256"]):
                raise BuildExecutorError(
                    "G35 view replay authority drift before import"
                )
            module = importlib.import_module("creamem3.memeye_g35_runtime_v19")
            if Path(module.__file__).resolve() != authority_path.resolve():
                raise BuildExecutorError("G35 view replay import path drift")
            limits = dict(module.DEFAULT_STAGE_LIMITS)
            limits["caption"] = module.StageLimits(2400, 1314, 3)
            limits["writer"] = module.StageLimits(16000, 1848, 3)
            limits["view"] = module.StageLimits(16384, 735, 3)
            limits["judge"] = module.StageLimits(400, 0, 3)
            self.source = module
            self.stage_limits = limits
            self.runtime = module.Gemini35Runtime(
                self.cache_root,
                allow_model_calls=False,
                stage_limits=limits,
            )
        else:
            raise BuildExecutorError("view replay backbone drift")

    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_output_tokens: int,
        max_provider_attempts: int,
        cache_metadata: Mapping[str, Any] | None = None,
        image_path: Path | None = None,
        caption_context_count: int | None = None,
    ) -> GenerationReply:
        if (
            stage != "view"
            or max_provider_attempts != 3
            or image_path is not None
            or caption_context_count is not None
        ):
            raise BuildExecutorError("view replay request shape drift")
        metadata = dict(cache_metadata or {})
        metadata["matched_substage"] = metadata.get("substage")
        metadata["max_provider_attempts"] = 3
        request = self.source.build_generate_content_request(
            stage,
            str(prompt),
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            stage_limits=self.stage_limits,
        )
        request_sha = self.source.request_identity(
            stage=stage,
            request=request,
            stage_limits=self.stage_limits,
            cache_metadata=metadata,
        )
        resumable = self.source.resume_hash(
            stage=stage,
            request_sha256=request_sha,
            stage_limits=self.stage_limits,
        )
        final_path = self.cache_root / "responses" / stage / f"{resumable}.json"
        plan_path = self.cache_root / "plans" / stage / f"{resumable}.json"
        attempt_root = self.cache_root / "attempts" / stage / resumable
        lock_path = self.cache_root / "lifecycle_locks" / stage / f"{resumable}.lock"
        if not (
            final_path.is_file()
            and plan_path.is_file()
            and attempt_root.is_dir()
            and lock_path.is_file()
        ):
            raise BuildExecutorError("view replay cache identity is incomplete")
        result = self.runtime.generate(
            stage,
            str(prompt),
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            cache_metadata=metadata,
        )
        if (
            result.cache_hit is not True
            or result.request_sha256 != request_sha
            or result.resume_hash != resumable
            or result.response_sha256 != _sha256(result.text)
        ):
            raise BuildExecutorError("view replay cache result drift")
        self.records.append(
            {
                "backbone": self.backbone,
                "task": str(metadata.get("task", "")),
                "tag": str(metadata.get("tag", "")),
                "prompt_sha256": _sha256(str(prompt)),
                "request_sha256": request_sha,
                "resume_hash": resumable,
                "response_sha256": result.response_sha256,
                "attempt_ordinal": int(
                    _read_mapping(final_path, "view replay final")[
                        "attempt_ordinal"
                    ]
                ),
            }
        )
        return GenerationReply(
            text=result.text,
            request_sha256=request_sha,
            response_sha256=result.response_sha256,
            cache_hit=True,
        )

    def provider_calls_during_replay(self) -> int:
        receipt = self.runtime.usage_receipt()
        return sum(
            int(row.get("provider_calls", 0))
            for row in receipt["stages"].values()
        )


def _semantic_view_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(graph))
    value.pop("embedding_contract", None)
    for node in value.get("summary_nodes", []):
        if node.get("node_type") == "event_view":
            node["embedding"] = [0.0] * EMBEDDING_DIMENSION
    return value


def _replay_all_view_graphs(
    *, output_root: Path, protocol: Mapping[str, Any], writer: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tasks = _strict_writer_task_rows(output_root=Path(output_root), writer=writer)
    replays = {
        backbone: _ReadOnlyViewGenerationReplay(
            output_root=Path(output_root), backbone=backbone, protocol=protocol
        )
        for backbone in ("g25", "g35")
    }

    def zero_embed(texts: Sequence[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSION for _text in texts]

    for backbone in ("g25", "g35"):
        replay = replays[backbone]
        for task_row in tasks:
            task = str(task_row["task"])
            session_path, _session_receipt = _artifact_pair(
                Path(output_root), "session", backbone, task
            )
            online_path, _online_receipt = _artifact_pair(
                Path(output_root), "online", backbone, task
            )
            offline_path, _offline_receipt = _artifact_pair(
                Path(output_root), "offline", backbone, task
            )
            session_graph = _read_mapping(session_path, "view replay session graph")
            online_graph = _read_mapping(online_path, "view replay Online graph")
            offline_graph = _read_mapping(offline_path, "view replay Offline graph")

            replayed_online = online.add_online_event_views(
                session_graph,
                _view_chat(replay, backbone=backbone, task=task),
                zero_embed,
                prompt_template=adapter.MEMEYE_ONLINE_EVENT_UPDATE_PROMPT,
                prompt_sha256=ONLINE_PROMPT_SHA256,
                current_fact_cap=16,
                prior_fact_cap=24,
                prune_unclaimed_create_members=True,
            )
            if _semantic_view_graph(replayed_online) != _semantic_view_graph(
                online_graph
            ):
                raise BuildExecutorError(
                    f"{backbone}/{task} Online body-to-artifact replay drift"
                )

            large = len(session_graph.get("fact_nodes", [])) > 1000
            if large:
                vector_by_text = {
                    str(node.get("text", "")): [
                        float(item) for item in node.get("embedding", [])
                    ]
                    for node in offline_graph.get("summary_nodes", [])
                    if node.get("node_type") == "event_view"
                }

                def offline_embed(
                    texts: Sequence[str], *, tag: str, **_kwargs: Any
                ) -> list[list[float]]:
                    if tag == "creamem3_memeye_large_offline_group_embed":
                        values: list[list[float]] = []
                        for text_value in texts:
                            vector = vector_by_text.get(str(text_value))
                            if (
                                vector is None
                                or len(vector) != EMBEDDING_DIMENSION
                                or not all(math.isfinite(item) for item in vector)
                            ):
                                raise BuildExecutorError(
                                    "large Offline query vector replay drift"
                                )
                            values.append(vector)
                        return values
                    return zero_embed(texts)

                replayed_offline = adapter.add_large_memeye_offline_event_views(
                    session_graph,
                    _view_chat(replay, backbone=backbone, task=task),
                    offline_embed,
                    max_attempts=4,
                    member_fact_cap=12,
                )
            else:
                replayed_offline = gm.add_offline_event_views(
                    session_graph,
                    _view_chat(replay, backbone=backbone, task=task),
                    zero_embed,
                    max_attempts=3,
                    prompt_template=adapter.MEMEYE_OFFLINE_EVENT_VIEW_PROMPT,
                    prompt_sha256=OFFLINE_STANDARD_PROMPT_SHA256,
                )
            if _semantic_view_graph(replayed_offline) != _semantic_view_graph(
                offline_graph
            ):
                raise BuildExecutorError(
                    f"{backbone}/{task} Offline body-to-artifact replay drift"
                )
    records = sorted(
        [record for replay in replays.values() for record in replay.records],
        key=lambda row: (
            row["backbone"],
            row["task"],
            row["tag"],
            row["resume_hash"],
        ),
    )
    calls = {
        backbone: replay.provider_calls_during_replay()
        for backbone, replay in replays.items()
    }
    if calls != {"g25": 0, "g35": 0}:
        raise BuildExecutorError("view graph replay made a provider call")
    return records, calls


def _final_generation_evidence_contract(
    *, output_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if not (Path(output_root) / "project" / "manifest.json").is_file():
        return {
            "schema_version": FINAL_GENERATION_EVIDENCE_SCHEMA,
            "status": "test_injected_generation_backends_not_formal",
            "protocol_sha256": _sha256(protocol),
            "provider_calls_during_validation": 0,
        }
    writer = validate_writer_complete(output_root=output_root, protocol=protocol)
    writer_subset = _writer_generation_evidence_contract(
        output_root=Path(output_root),
        protocol=protocol,
        writer=writer,
        allow_downstream_view=True,
    )
    view_rows, replay_calls = _replay_all_view_graphs(
        output_root=Path(output_root), protocol=protocol, writer=writer
    )
    expected_view: dict[str, set[str]] = {"g25": set(), "g35": set()}
    for row in view_rows:
        backbone = str(row["backbone"])
        resumable = str(row["resume_hash"])
        if resumable in expected_view[backbone]:
            raise BuildExecutorError("duplicate final view generation identity")
        expected_view[backbone].add(resumable)

    whole_cache_rows: list[dict[str, Any]] = []
    for backbone in ("g25", "g35"):
        cache_root = (Path(output_root) / "provider" / backbone).resolve()
        _model, _profile, limits = _generation_profile(backbone)
        census = _generation_cache_identity_sets(cache_root, set(limits))
        expected_by_stage: dict[str, set[str]] = {}
        for stage in limits:
            if stage == "view":
                expected_by_stage[stage] = expected_view[backbone]
            elif stage == "judge":
                expected_by_stage[stage] = set()
            else:
                expected_by_stage[stage] = set(census["responses"][stage])
        for kind in ("responses", "plans", "attempts", "lifecycle"):
            if census[kind] != expected_by_stage:
                raise BuildExecutorError(
                    f"{backbone} final generation {kind} census drift"
                )
        expected_files: set[Path] = set()
        for stage, identities in expected_by_stage.items():
            for resumable in identities:
                final_path = cache_root / "responses" / stage / f"{resumable}.json"
                plan_path = cache_root / "plans" / stage / f"{resumable}.json"
                lock_path = (
                    cache_root / "lifecycle_locks" / stage / f"{resumable}.lock"
                )
                final = _read_mapping(final_path, "final generation response")
                ordinal = int(final.get("attempt_ordinal", 0))
                if not 1 <= ordinal <= 3:
                    raise BuildExecutorError("final generation ordinal drift")
                identity_files = {final_path, plan_path, lock_path}
                identity_files.update(
                    cache_root
                    / "attempts"
                    / stage
                    / resumable
                    / f"attempt_{number:03d}_{suffix}.json"
                    for number in range(1, ordinal + 1)
                    for suffix in ("request", "response")
                )
                if not all(path.is_file() for path in identity_files):
                    raise BuildExecutorError(
                        "final generation identity file chain is incomplete"
                    )
                expected_files.update(path.resolve() for path in identity_files)
                whole_cache_rows.append(
                    {
                        "backbone": backbone,
                        "stage": stage,
                        "resume_hash": resumable,
                        "request_sha256": str(final.get("request_sha256", "")),
                        "response_sha256": str(final.get("response_sha256", "")),
                        "attempt_ordinal": ordinal,
                    }
                )
        observed_files = {
            path.resolve() for path in cache_root.rglob("*") if path.is_file()
        }
        if observed_files != expected_files:
            raise BuildExecutorError(
                f"{backbone} final generation exact filesystem census drift"
            )
        for forbidden in ("orphans", "failures"):
            root = cache_root / forbidden
            if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
                raise BuildExecutorError(
                    f"{backbone} final generation {forbidden} is nonempty"
                )
        if (cache_root / "formal_global_generation_budget.lock").exists():
            raise BuildExecutorError(
                f"{backbone} final generation global lease is stale"
            )
    canonical_usage = {
        "g25": _canonical_generation_usage(
            output_root=Path(output_root),
            backbone="g25",
            included_stages={"writer", "view"},
        ),
        "g35": _canonical_generation_usage(
            output_root=Path(output_root),
            backbone="g35",
            included_stages={"caption", "writer", "view"},
        ),
    }
    whole_cache_rows.sort(
        key=lambda row: (row["backbone"], row["stage"], row["resume_hash"])
    )
    return {
        "schema_version": FINAL_GENERATION_EVIDENCE_SCHEMA,
        "status": "pass",
        "protocol_sha256": _sha256(protocol),
        "writer_complete_file_sha256": _file_sha256(
            Path(output_root) / "build" / "writer_complete.json"
        ),
        "writer_generation_evidence_file_sha256": _file_sha256(
            Path(output_root) / "build" / "writer_generation_evidence.json"
        ),
        "writer_subset_evidence_sha256": _sha256(writer_subset),
        "view_logical_identities": {
            "g25": len(expected_view["g25"]),
            "g35": len(expected_view["g35"]),
            "total": len(view_rows),
        },
        "view_identity_rows_sha256": _sha256(view_rows),
        "all_identity_rows_sha256": _sha256(whole_cache_rows),
        "durable_usage_sha256": _sha256(canonical_usage),
        "exact_cache_identity_census": True,
        "body_to_artifact_replay": True,
        "orphans_and_failures_empty": True,
        "provider_calls_during_validation": sum(replay_calls.values()),
    }


def seal_final_generation_evidence(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    protocol = dict(protocol or matrix.load_protocol())
    value = _final_generation_evidence_contract(
        output_root=Path(output_root), protocol=protocol
    )
    value["final_generation_evidence_sha256"] = _sha256(value)
    path = Path(output_root) / "build" / "final_generation_evidence.json"
    _write_once_json(path, value)
    return validate_final_generation_evidence(
        output_root=Path(output_root), protocol=protocol
    )


def validate_final_generation_evidence(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    protocol = dict(protocol or matrix.load_protocol())
    path = Path(output_root) / "build" / "final_generation_evidence.json"
    value = _read_mapping(path, "final generation evidence")
    contract = dict(value)
    observed = contract.pop("final_generation_evidence_sha256", None)
    if observed != _sha256(contract):
        raise BuildExecutorError("final generation evidence digest drift")
    expected = _final_generation_evidence_contract(
        output_root=Path(output_root), protocol=protocol
    )
    expected["final_generation_evidence_sha256"] = _sha256(expected)
    if value != expected:
        raise BuildExecutorError("final generation evidence contract drift")
    return value


def prepare_writer_packets(
    *,
    project: Mapping[str, Any],
    output_root: Path,
    generation_backends: Mapping[str, GenerationBackend],
    caption_root: Path = matrix.DEFAULT_V7_CAPTION_ROOT,
    image_root: Path | None = None,
    enforce_full_scope: bool = True,
) -> dict[str, Any]:
    """Generate/resume captions and both writer packet families, then seal.

    This subphase has no embedding backend argument by design.  The returned
    fact census is therefore available for independent budget review before
    materialization can reach an embedding provider.
    """

    protocol = matrix.load_protocol()
    complete_path = Path(output_root) / "build" / "writer_complete.json"
    if complete_path.is_file():
        resumed_writer = validate_writer_complete(
            output_root=output_root, protocol=protocol
        )
        if not (Path(output_root) / "project" / "manifest.json").is_file():
            evidence_path = (
                Path(output_root) / "build" / "writer_generation_evidence.json"
            )
            if not evidence_path.is_file():
                seal_writer_generation_evidence(
                    output_root=output_root, protocol=protocol
                )
        return resumed_writer
    if set(generation_backends) != {"g25", "g35"}:
        raise BuildExecutorError("both generation backbones are required")
    tasks = _project_tasks(project)
    if image_root is None:
        image_root = matrix.DEFAULT_DATA_ROOT / "data" / "image"
    caption_maps: dict[str, dict[str, dict[str, str]]] = {"g25": {}, "g35": {}}
    caption_receipts: list[dict[str, Any]] = []
    for task_row in tasks:
        task = str(task_row["task"])
        g25, receipt25 = load_g25_caption_bank_for_task(task_row, caption_root)
        g35, receipt35 = build_g35_caption_bank_for_task(
            task_row,
            backend=generation_backends["g35"],
            image_root=image_root,
            output_root=output_root,
        )
        caption_maps["g25"][task] = g25
        caption_maps["g35"][task] = g35
        caption_receipts.extend(
            [
                {"backbone": "g25", **receipt25},
                {"backbone": "g35", **receipt35},
            ]
        )
    if enforce_full_scope and sum(row["images"] for row in caption_receipts) != 876:
        raise BuildExecutorError("two-backbone caption census drift")

    packet_rows: list[dict[str, Any]] = []
    for backbone in ("g25", "g35"):
        generation = generation_backends[backbone]
        for task_row in tasks:
            task = str(task_row["task"])
            captions = caption_maps[backbone][task]
            records, _source_map, session_packets = _ensure_session_packets(
                task_row,
                captions,
                backbone=backbone,
                generation=generation,
                output_root=output_root,
            )
            _rounds, native_packets = _ensure_native_packets(
                task_row,
                captions,
                backbone=backbone,
                generation=generation,
                output_root=output_root,
            )
            for family, packets, fact_key, packet_root in (
                (
                    "session",
                    session_packets,
                    "atomic_facts",
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "session"
                    / task,
                ),
                (
                    "mgtm",
                    native_packets,
                    "facts",
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "mgtm"
                    / task,
                ),
            ):
                paths = sorted(packet_root.glob("*.json"))
                if len(paths) != len(packets):
                    raise BuildExecutorError("writer packet directory census drift")
                facts = sum(len(packet.get(fact_key, [])) for packet in packets)
                if any(not str(packet.get("batch_summary" if family == "session" else "summary", "")).strip() for packet in packets):
                    raise BuildExecutorError("writer packet contains an empty summary")
                if any(
                    not str(item.get("fact", "")).strip()
                    for packet in packets
                    for item in packet.get("atomic_facts", [])
                    if family == "session"
                ) or any(
                    not str(item).strip()
                    for packet in packets
                    for item in packet.get("facts", [])
                    if family == "mgtm"
                ):
                    raise BuildExecutorError("writer packet contains an empty fact")
                packet_rows.append(
                    {
                        "backbone": backbone,
                        "task": task,
                        "family": family,
                        "packets": len(packets),
                        "summaries": len(packets),
                        "facts": facts,
                        "physical_sessions": len(records),
                        "packet_paths": [str(path.resolve()) for path in paths],
                        "packet_file_sha256": [_file_sha256(path) for path in paths],
                    }
                )
    packet_rows.sort(key=lambda row: (row["backbone"], row["task"], row["family"]))
    if enforce_full_scope:
        session_rows = [row for row in packet_rows if row["family"] == "session"]
        native_rows = [row for row in packet_rows if row["family"] == "mgtm"]
        if (
            len(packet_rows) != EXPECTED_TASKS * 2 * 2
            or sum(row["packets"] for row in session_rows) != EXPECTED_SESSIONS * 2
            or sum(row["packets"] for row in native_rows) != EXPECTED_NATIVE_BATCHES * 2
        ):
            raise BuildExecutorError("full writer packet census drift")
    formal_project = (Path(output_root) / "project" / "manifest.json").is_file()
    if formal_project:
        usage = {
            "g25": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g25",
                included_stages={"writer"},
            ),
            "g35": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g35",
                included_stages={"caption", "writer"},
            ),
        }
    else:
        usage = {
            backbone: dict(generation_backends[backbone].usage_receipt())
            for backbone in ("g25", "g35")
        }
    provider_calls = sum(_usage_provider_calls(row) for row in usage.values())
    if provider_calls > BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING:
        raise BuildExecutorError("writer subphase generation ceiling exceeded")
    value = {
        "schema_version": WRITER_COMPLETE_SCHEMA,
        "complete": True,
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": _sha256(protocol),
        "project_contract_sha256": project["project_contract_sha256"],
        "task_inputs": [
            {
                "task": str(row["task"]),
                "memory_path": str(Path(row["memory_path"]).resolve()),
                "memory_sha256": str(row["memory_sha256"]),
                "images": len(row["images"]),
                "physical_sessions": int(row["sessions"]),
            }
            for row in tasks
        ],
        "caption_sources": {
            "g25_caption_root": str(Path(caption_root).resolve()),
            "g35_caption_root": str(
                (Path(output_root) / "captions" / "g35").resolve()
            ),
            "image_root": str(Path(image_root).resolve()),
        },
        "caption_receipts": caption_receipts,
        "packet_rows": packet_rows,
        "packet_rows_sha256": _sha256(packet_rows),
        "provider_usage": usage,
        "provider_calls": provider_calls,
        "generation_provider_attempt_hard_ceiling": (
            BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        ),
        "embedding_calls": 0,
        "embedding_budget_status": "pending_zero_call_census_and_independent_gate",
    }
    value["writer_complete_sha256"] = _sha256(value)
    _write_once_json(complete_path, value)
    validated = validate_writer_complete(output_root=output_root, protocol=protocol)
    if not (Path(output_root) / "project" / "manifest.json").is_file():
        seal_writer_generation_evidence(output_root=output_root, protocol=protocol)
    return validated


def compute_dynamic_embedding_budget(
    *,
    output_root: Path,
    protocol: Mapping[str, Any] | None = None,
    allow_downstream_view: bool = False,
) -> dict[str, Any]:
    """Seal the post-writer 100-item physical-batch hard ceiling, zero-call."""

    protocol = dict(protocol or matrix.load_protocol())
    writer = validate_writer_complete(output_root=output_root, protocol=protocol)
    generation_evidence_path = (
        Path(output_root) / "build" / "writer_generation_evidence.json"
    )
    budget_path = Path(output_root) / "build" / "embedding_budget.json"
    if allow_downstream_view:
        gate_path = Path(output_root) / "build" / "dynamic_embedding_budget_gate.json"
        if not budget_path.is_file() or not gate_path.is_file():
            raise BuildExecutorError(
                "post-view budget mode requires an already sealed independent gate"
            )
        gate_binding = _read_mapping(
            gate_path, "post-view dynamic embedding gate binding"
        )
        if (
            gate_binding.get("schema_version") != DYNAMIC_EMBEDDING_GATE_SCHEMA
            or gate_binding.get("decision") != "approve"
            or gate_binding.get("protocol_sha256") != _sha256(protocol)
            or gate_binding.get("embedding_budget_file_sha256")
            != _file_sha256(budget_path)
            or gate_binding.get("writer_complete_file_sha256")
            != _file_sha256(Path(output_root) / "build" / "writer_complete.json")
            or gate_binding.get("writer_generation_evidence_file_sha256")
            != _file_sha256(generation_evidence_path)
        ):
            raise BuildExecutorError("post-view dynamic gate binding drift")
    generation_evidence = (
        _validate_sealed_writer_generation_evidence_binding(
            output_root=Path(output_root), protocol=protocol, writer=writer
        )
        if allow_downstream_view
        else validate_writer_generation_evidence(
            output_root=output_root, protocol=protocol
        )
    )
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in writer["packet_rows"]:
        grouped.setdefault((row["backbone"], row["task"]), {})[
            row["family"]
        ] = row
    rows: list[dict[str, Any]] = []
    total_physical_batches = 0
    for (backbone, task), families in sorted(grouped.items()):
        if set(families) != {"session", "mgtm"}:
            raise BuildExecutorError("dynamic census lacks a packet family")
        session = families["session"]
        native = families["mgtm"]
        session_base_items = int(session["summaries"]) + int(session["facts"])
        session_base_batches = _embedding_batches(session_base_items)
        native_summary_batches = _embedding_batches(int(native["summaries"]))
        native_fact_batches = _embedding_batches(int(native["facts"]))
        online_steps = max(0, int(session["physical_sessions"]) - 1)
        # MAX_OPERATIONS=12, so every causal step can emit at most one <=100 batch.
        if online.MAX_OPERATIONS > EMBEDDING_BATCH_SIZE:
            raise BuildExecutorError("Online operation cap exceeds embedding batch")
        online_view_batch_hard = online_steps
        large_offline = int(session["facts"]) > 1000
        # Large mode embeds at most 16 group queries and 16 final views.  The
        # standard 16,384-token JSON response has no explicit list cap, so one
        # output-token per possible view is used as a deliberately loose hard
        # item ceiling (164 physical batches).
        offline_view_batch_hard = (
            2 if large_offline else _embedding_batches(16384)
        )
        physical_batches = (
            session_base_batches
            + native_summary_batches
            + native_fact_batches
            + online_view_batch_hard
            + offline_view_batch_hard
        )
        total_physical_batches += physical_batches
        rows.append(
            {
                "backbone": backbone,
                "task": task,
                "session_summaries": int(session["summaries"]),
                "session_facts": int(session["facts"]),
                "session_base_embedding_items_exact": session_base_items,
                "session_base_physical_batches_exact": session_base_batches,
                "native_summaries": int(native["summaries"]),
                "native_facts": int(native["facts"]),
                "native_summary_physical_batches_exact": native_summary_batches,
                "native_fact_physical_batches_exact": native_fact_batches,
                "online_steps": online_steps,
                "online_max_operations_per_step": online.MAX_OPERATIONS,
                "online_view_physical_batch_hard": online_view_batch_hard,
                "offline_branch": "large" if large_offline else "standard",
                "offline_view_physical_batch_hard": offline_view_batch_hard,
                "physical_embedding_batch_hard": physical_batches,
            }
        )
    embedding_attempt_hard = (
        total_physical_batches * EMBEDDING_MAX_PROVIDER_ATTEMPTS
    )
    overall_attempt_hard = (
        BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING + embedding_attempt_hard
    )
    value = {
        "schema_version": EMBEDDING_BUDGET_SCHEMA,
        "status": "census_complete_waiting_independent_gate",
        "protocol_sha256": _sha256(protocol),
        "writer_complete_path": str(
            (Path(output_root) / "build" / "writer_complete.json").resolve()
        ),
        "writer_complete_file_sha256": _file_sha256(
            Path(output_root) / "build" / "writer_complete.json"
        ),
        "writer_complete_sha256": writer["writer_complete_sha256"],
        "packet_rows_sha256": writer["packet_rows_sha256"],
        "writer_generation_evidence_path": str(
            generation_evidence_path.resolve()
        ),
        "writer_generation_evidence_file_sha256": _file_sha256(
            generation_evidence_path
        ),
        "writer_generation_evidence_sha256": generation_evidence[
            "generation_evidence_sha256"
        ],
        "model": "gemini-embedding-001",
        "dimension": EMBEDDING_DIMENSION,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "max_provider_attempts_per_physical_batch": (
            EMBEDDING_MAX_PROVIDER_ATTEMPTS
        ),
        "rows": rows,
        "rows_sha256": _sha256(rows),
        "physical_embedding_batch_hard": total_physical_batches,
        "embedding_provider_attempt_hard_ceiling": embedding_attempt_hard,
        "generation_provider_attempt_hard_ceiling": (
            BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        ),
        "build_overall_provider_attempt_hard_ceiling": overall_attempt_hard,
        "model_calls": 0,
        "provider_calls": 0,
    }
    value["embedding_budget_sha256"] = _sha256(value)
    if budget_path.is_file():
        observed_value = _read_mapping(budget_path, "dynamic embedding budget")
        if observed_value != value:
            raise BuildExecutorError(
                "dynamic embedding budget does not exactly match recomputed census"
            )
        return observed_value
    _write_once_json(budget_path, value)
    return value


def validate_dynamic_embedding_budget_gate(
    *,
    output_root: Path,
    protocol: Mapping[str, Any] | None = None,
    allow_downstream_view: bool = False,
) -> dict[str, Any]:
    """Require an independent, exact-budget approval before embedding."""

    protocol = dict(protocol or matrix.load_protocol())
    budget_path = Path(output_root) / "build" / "embedding_budget.json"
    if allow_downstream_view:
        # The gate must already exist and bind the immutable budget before the
        # weaker namespace allowance is used.  Its exact fields are checked
        # below, then the budget is deeply recomputed with writer-subset replay.
        budget = _read_mapping(budget_path, "sealed dynamic embedding budget")
        budget_contract = dict(budget)
        observed_budget_sha = budget_contract.pop("embedding_budget_sha256", None)
        if (
            observed_budget_sha != _sha256(budget_contract)
            or budget.get("protocol_sha256") != _sha256(protocol)
        ):
            raise BuildExecutorError("sealed dynamic embedding budget drift")
    else:
        budget = compute_dynamic_embedding_budget(
            output_root=output_root,
            protocol=protocol,
            allow_downstream_view=False,
        )
    gate_path = Path(output_root) / "build" / "dynamic_embedding_budget_gate.json"
    gate = _read_mapping(gate_path, "dynamic embedding budget gate")
    required = {
        "schema_version",
        "decision",
        "reviewer_role",
        "reviewer",
        "protocol_sha256",
        "writer_complete_file_sha256",
        "writer_generation_evidence_file_sha256",
        "embedding_budget_file_sha256",
        "approved_embedding_provider_attempt_hard_ceiling",
        "approved_build_overall_provider_attempt_hard_ceiling",
        "model",
        "dimension",
        "batch_size",
        "provider_visible_character_limit",
        "cache_identity_uses_complete_untruncated_text",
        "provider_request_fields_omitted",
        "failure_policy",
        "max_provider_attempts_per_physical_batch",
        "legacy_cache_readthrough",
    }
    if set(gate) != required:
        raise BuildExecutorError("dynamic embedding budget gate keyset drift")
    expected = {
        "schema_version": DYNAMIC_EMBEDDING_GATE_SCHEMA,
        "decision": "approve",
        "reviewer_role": "independent_protocol_auditor",
        "protocol_sha256": _sha256(protocol),
        "writer_complete_file_sha256": budget["writer_complete_file_sha256"],
        "writer_generation_evidence_file_sha256": budget[
            "writer_generation_evidence_file_sha256"
        ],
        "embedding_budget_file_sha256": _file_sha256(budget_path),
        "approved_embedding_provider_attempt_hard_ceiling": budget[
            "embedding_provider_attempt_hard_ceiling"
        ],
        "approved_build_overall_provider_attempt_hard_ceiling": budget[
            "build_overall_provider_attempt_hard_ceiling"
        ],
        "model": "gemini-embedding-001",
        "dimension": EMBEDDING_DIMENSION,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "max_provider_attempts_per_physical_batch": (
            EMBEDDING_MAX_PROVIDER_ATTEMPTS
        ),
        "legacy_cache_readthrough": False,
    }
    for key, expected_value in expected.items():
        if gate.get(key) != expected_value:
            raise BuildExecutorError(f"dynamic embedding gate drift: {key}")
    if not str(gate.get("reviewer", "")).strip():
        raise BuildExecutorError("dynamic embedding gate lacks reviewer identity")
    if allow_downstream_view:
        recomputed = compute_dynamic_embedding_budget(
            output_root=output_root,
            protocol=protocol,
            allow_downstream_view=True,
        )
        if recomputed != budget:
            raise BuildExecutorError("post-view dynamic budget replay drift")
    return {
        "path": str(gate_path.resolve()),
        "file_sha256": _file_sha256(gate_path),
        "budget": budget,
    }


def capability_preflight(
    *,
    output_root: Path = matrix.DEFAULT_ROOT,
    caption_root: Path = matrix.DEFAULT_V7_CAPTION_ROOT,
) -> dict[str, Any]:
    """Full-scope zero-call executor preflight; never imports a provider."""

    before = matrix._loaded_provider_modules()
    if before:
        raise BuildExecutorError(f"provider already imported: {before}")
    protocol = matrix.load_protocol()
    project = matrix.validate_project(output_root=output_root)
    plan_path = Path(output_root) / "build" / "plan.json"
    plan = (
        validate_build_plan(output_root=output_root, project=project)
        if plan_path.is_file()
        else plan_build(
            project=project,
            output_root=output_root,
            caption_root=caption_root,
            enforce_full_scope=True,
        )
    )
    ingest_protocol = ingest.protocol_manifest()
    g35_spec = importlib.util.find_spec("creamem3.memeye_g35_runtime_v19")
    g35_path = Path(g35_spec.origin).resolve() if g35_spec and g35_spec.origin else None
    smoke_paths = {
        "g25_summary_writer": Path(output_root)
        / "provider_smokes"
        / "g25_summary_1200_r1"
        / "complete.json",
        "g35_caption": Path(output_root)
        / "provider_smokes"
        / "g35_caption_writer_view_r1"
        / "complete.json",
        "embedding_batch": Path(output_root)
        / "provider_smokes"
        / "gemini_embedding_001_r1"
        / "complete.json",
        "formal_bridge": Path(output_root)
        / "releases"
        / "official_provider_capability_bridge.json",
    }
    smoke_status: dict[str, Any] = {}
    for name, path in smoke_paths.items():
        smoke_status[name] = {
            "path": str(path.resolve()),
            "present": path.is_file(),
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
    report = {
        "schema_version": ZERO_CALL_RELEASE_SCHEMA,
        "status": "zero_call_pass_live_waiting",
        "protocol_sha256": _sha256(protocol),
        "project_contract_sha256": project["project_contract_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "executor_path": str(Path(__file__).resolve()),
        "executor_sha256": _file_sha256(Path(__file__).resolve()),
        "matched_ingest_path": str(Path(ingest.__file__).resolve()),
        "matched_ingest_sha256": _file_sha256(Path(ingest.__file__).resolve()),
        "matched_ingest_protocol_sha256": ingest_protocol["protocol_sha256"],
        "g35_runtime_path": str(g35_path) if g35_path else None,
        "g35_runtime_sha256": _file_sha256(g35_path) if g35_path else None,
        "g35_runtime_expected_sha256": protocol["authority_files"]["g35_runtime"][
            "sha256"
        ],
        "provider_smoke_seals": smoke_status,
        "live_blockers": [
            name for name, row in smoke_status.items() if not row["present"]
        ],
        "official_g35_region_status": protocol[
            "provider_availability_protocol"
        ]["g35_prebuild_smoke"]["official_google_status"],
        "alternate_transport_authorized": False,
        "generation_provider_attempt_hard_ceiling": (
            BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        ),
        "generation_stage_provider_attempt_hard_ceilings": dict(
            GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
        ),
        "embedding_provider_attempt_hard_ceiling": (
            "pending_post_writer_100_item_batch_census"
        ),
        "overall_provider_attempt_hard_ceiling": (
            "pending_post_writer_100_item_batch_census"
        ),
        "dynamic_embedding_budget_gate": "not_sealed",
        "model_calls": 0,
        "provider_calls": 0,
        "provider_modules_imported": matrix._loaded_provider_modules(),
    }
    report["report_sha256"] = _sha256(report)
    _write_json_atomic(
        Path(output_root) / "releases" / "build_executor_zero_call.json",
        report,
    )
    if matrix._loaded_provider_modules() != before:
        raise BuildExecutorError("provider imported during executor preflight")
    return report


class G35RuntimeBackend:
    """Final official runtime adapter; constructed only after all smoke seals."""

    backbone = "g35"

    def __init__(self, cache_root: Path) -> None:
        protocol = matrix.load_protocol()
        authority = protocol["authority_files"]["g35_runtime"]
        runtime_path = (
            Path(__file__).resolve().parent.parent / authority["path"]
        ).resolve()
        if _file_sha256(runtime_path) != str(authority["sha256"]):
            raise BuildExecutorError("G35 runtime authority drift before import")
        runtime = importlib.import_module("creamem3.memeye_g35_runtime_v19")
        if Path(runtime.__file__).resolve() != runtime_path:
            raise BuildExecutorError("G35 runtime import path drift")

        limits = dict(runtime.DEFAULT_STAGE_LIMITS)
        limits["caption"] = runtime.StageLimits(2400, 1314, 3)
        limits["writer"] = runtime.StageLimits(16000, 1848, 3)
        limits["view"] = runtime.StageLimits(16384, 735, 3)
        limits["judge"] = runtime.StageLimits(400, 0, 3)
        self._module = runtime
        self.cache_root = Path(cache_root).resolve()
        self.stage_limits = limits
        self._runtime = runtime.Gemini35Runtime(
            self.cache_root,
            allow_model_calls=True,
            stage_limits=limits,
        )

    def _durable_attempts(self, stage: str) -> int:
        root = self.cache_root / "attempts" / stage
        return len(list(root.rglob("attempt_*_request.json"))) if root.is_dir() else 0

    def _acquire_global_lease(self) -> Path:
        path = self.cache_root / "formal_global_generation_budget.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise BuildExecutorError(
                "stale G35 build global lease requires manual audit"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_output_tokens: int,
        max_provider_attempts: int,
        cache_metadata: Mapping[str, Any] | None = None,
        image_path: Path | None = None,
        caption_context_count: int | None = None,
    ) -> GenerationReply:
        if max_provider_attempts != 3:
            raise BuildExecutorError("G35 matrix requests require exactly three attempts")
        if stage not in {"caption", "writer", "view"}:
            raise BuildExecutorError(f"G35 build stage drift: {stage}")
        parts: list[Mapping[str, Any]] = []
        if stage == "caption":
            if image_path is None or caption_context_count is None:
                raise BuildExecutorError("G35 caption lacks image/context count")
            mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
            parts.append(
                self._module.inline_image_part(Path(image_path).read_bytes(), mime)
            )
        elif image_path is not None or caption_context_count is not None:
            raise BuildExecutorError("raw image escaped caption stage")
        metadata = dict(cache_metadata or {})
        metadata["matched_substage"] = metadata.get("substage")
        metadata["max_provider_attempts"] = 3
        request = self._module.build_generate_content_request(
            stage,
            prompt,
            system=system,
            parts=parts,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            stage_limits=self.stage_limits,
            caption_context_count=caption_context_count,
        )
        request_sha = self._module.request_identity(
            stage=stage,
            request=request,
            stage_limits=self.stage_limits,
            cache_metadata=metadata,
        )
        resumable = self._module.resume_hash(
            stage=stage,
            request_sha256=request_sha,
            stage_limits=self.stage_limits,
        )
        response_path = self.cache_root / "responses" / stage / f"{resumable}.json"
        lease = self._acquire_global_lease()
        try:
            if (
                not response_path.is_file()
                and self._durable_attempts(stage)
                >= self.stage_limits[stage].max_live_calls
            ):
                raise BuildExecutorError(
                    f"durable G35 {stage} provider-attempt ceiling exhausted"
                )
            result = self._runtime.generate(
                stage,
                prompt,
                system=system,
                parts=parts,
                json_mode=json_mode,
                max_output_tokens=max_output_tokens,
                caption_context_count=caption_context_count,
                cache_metadata=metadata,
            )
        finally:
            lease.unlink(missing_ok=True)
        return GenerationReply(
            text=result.text,
            request_sha256=result.request_sha256,
            response_sha256=result.response_sha256,
            cache_hit=result.cache_hit,
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        value = self._runtime.usage_receipt()
        for stage, row in value["stages"].items():
            in_process = int(row["provider_calls"])
            durable = self._durable_attempts(stage)
            row["in_process_provider_calls"] = in_process
            row["durable_provider_attempt_intents"] = durable
            row["provider_calls"] = durable
            row["remaining_live_calls"] = self.stage_limits[
                stage
            ].max_live_calls - durable
        value["profile"] = "build_caption_writer_view"
        return value


class G25AuthorityBackend:
    """Durable direct-REST 2.5 build adapter; no legacy SDK/cache path."""

    backbone = "g25"

    def __init__(self, cache_root: Path) -> None:
        adapter_path = Path(__file__).with_name(
            "official_g25_build_runtime_adapter_v19.py"
        )
        if _file_sha256(adapter_path) != (
            "c04d8d16084e623faf1ee771ae91fe41e2518b57fc48cee22056c77819908a1f"
        ):
            raise BuildExecutorError("G25 durable build adapter authority drift")
        module = importlib.import_module(
            "creamem3.official_g25_build_runtime_adapter_v19"
        )
        if Path(module.__file__).resolve() != adapter_path.resolve():
            raise BuildExecutorError("G25 durable build adapter import drift")
        self._runtime = module.OfficialGemini25BuildRuntime(
            Path(cache_root), allow_model_calls=True
        )
        self._logical_calls = 0

    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_output_tokens: int,
        max_provider_attempts: int,
        cache_metadata: Mapping[str, Any] | None = None,
        image_path: Path | None = None,
        caption_context_count: int | None = None,
    ) -> GenerationReply:
        if stage not in {"writer", "view"} or image_path is not None:
            raise BuildExecutorError("G25 build request stage/image drift")
        if max_provider_attempts != 3:
            raise BuildExecutorError("G25 matrix requests require exactly three attempts")
        self._logical_calls += 1
        limit = G25_LOGICAL_LIMITS[stage]
        stage_calls = getattr(self, f"_{stage}_calls", 0) + 1
        setattr(self, f"_{stage}_calls", stage_calls)
        if stage_calls > limit:
            raise BuildExecutorError(f"G25 {stage} logical call ceiling exceeded")
        metadata = dict(cache_metadata or {})
        metadata["matched_substage"] = metadata.get("substage")
        metadata["max_provider_attempts"] = 3
        result = self._runtime.generate(
            stage,
            prompt,
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            cache_metadata=metadata,
        )
        return GenerationReply(
            str(result.text),
            str(result.request_sha256),
            str(result.response_sha256),
            bool(result.cache_hit),
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        value = dict(self._runtime.usage_receipt())
        value["logical_calls"] = self._logical_calls
        value["writer_logical_calls"] = getattr(self, "_writer_calls", 0)
        value["view_logical_calls"] = getattr(self, "_view_calls", 0)
        return value


class OfficialNative3072EmbeddingBackend:
    """Formal direct-REST batchEmbedContents adapter shared with its smoke."""

    def __init__(
        self,
        cache_root: Path,
        *,
        containment_root: Path,
        approved_provider_attempt_hard_ceiling: int,
        budget_binding_sha256: str,
        allow_model_calls: bool = True,
        provider_factory: Any | None = None,
    ) -> None:
        runtime_path = Path(__file__).with_name(
            "memeye_official_embedding_batch_runtime_v19_1.py"
        )
        # Candidate pin is refreshed only after the independent smoke runtime
        # audit signs the shared source.  Drift is fail-closed before import.
        expected_sha = OFFICIAL_EMBEDDING_RUNTIME_SHA256
        actual_sha = _file_sha256(runtime_path)
        if actual_sha != expected_sha:
            raise BuildExecutorError(
                "official embedding shared runtime candidate authority drift"
            )
        module = importlib.import_module(
            "creamem3.memeye_official_embedding_batch_runtime_v19_1"
        )
        if Path(module.__file__).resolve() != runtime_path.resolve():
            raise BuildExecutorError("official embedding runtime import drift")
        self.cache_root = Path(cache_root).resolve()
        self.containment_root = Path(containment_root).resolve()
        self.approved_provider_attempt_hard_ceiling = int(
            approved_provider_attempt_hard_ceiling
        )
        self.budget_binding_sha256 = str(budget_binding_sha256)
        self._module = module
        self._runtime = module.OfficialGoogleBatchEmbeddingRuntime(
            self.cache_root,
            containment_root=self.containment_root,
            limits=module.EmbeddingLimits(
                max_live_calls=self.approved_provider_attempt_hard_ceiling,
                max_attempts=EMBEDDING_MAX_PROVIDER_ATTEMPTS,
            ),
            allow_model_calls=allow_model_calls,
            budget_binding_sha256=self.budget_binding_sha256,
            provider_factory=provider_factory,
        )
        self._batch_receipts: list[dict[str, Any]] = []

    def embed(
        self,
        texts: Sequence[str],
        *,
        tag: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> list[list[float]]:
        materialized = [str(value) for value in texts]
        if not materialized or any(not value.strip() for value in materialized):
            raise BuildExecutorError("formal embedding input is empty")
        output: list[list[float]] = []
        for start in range(0, len(materialized), EMBEDDING_BATCH_SIZE):
            batch = materialized[start : start + EMBEDDING_BATCH_SIZE]
            metadata = {
                **dict(cache_metadata or {}),
                "tag": str(tag),
                "batch_start": start,
                "batch_size": len(batch),
            }
            result = self._runtime.embed_batch(batch, cache_metadata=metadata)
            vectors = [list(map(float, vector)) for vector in result.vectors]
            if len(vectors) != len(batch) or any(
                len(vector) != EMBEDDING_DIMENSION
                or not all(math.isfinite(value) for value in vector)
                for vector in vectors
            ):
                raise BuildExecutorError("official embedding result shape drift")
            self._batch_receipts.append(
                {
                    "tag": str(tag),
                    "batch_start": start,
                    "batch_size": len(batch),
                    "cache_metadata": metadata,
                    "request_sha256": result.request_sha256,
                    "resume_hash": result.resume_hash,
                    "response_sha256": result.response_sha256,
                    "vector_sha256s": list(result.vector_sha256s),
                    "vectors_sha256": result.vectors_sha256,
                    "full_text_sha256s": list(result.full_text_sha256s),
                    "visible_text_sha256s": list(result.visible_text_sha256s),
                    "cache_path": result.cache_path,
                    "cache_file_sha256": _file_sha256(Path(result.cache_path)),
                    "successful_attempt_ordinal": (
                        result.successful_attempt_ordinal
                    ),
                }
            )
            output.extend(vectors)
        return output

    def batch_receipts(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._batch_receipts)

    def usage_receipt(self) -> Mapping[str, Any]:
        value = dict(self._runtime.usage_receipt())
        in_process = int(value.get("provider_calls", 0))
        durable = int(value.get("durable_provider_attempts", 0))
        value.update(
            {
                "provider_calls": durable,
                "provider_calls_this_process": in_process,
                "dimension": EMBEDDING_DIMENSION,
                "batch_size": EMBEDDING_BATCH_SIZE,
                "provider_visible_character_limit": (
                    EMBEDDING_VISIBLE_CHARACTER_LIMIT
                ),
                "cache_identity_uses_complete_untruncated_text": True,
                "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
                "max_provider_attempts_per_physical_batch": (
                    EMBEDDING_MAX_PROVIDER_ATTEMPTS
                ),
                "task_type": None,
                "input_type": None,
                "output_dimensionality": None,
                "cache_root": str(self.cache_root),
                "batch_receipts": self.batch_receipts(),
                "batch_receipts_sha256": _sha256(self.batch_receipts()),
            }
        )
        return value


def _embedding_census(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_nodes = 0
    invalid_nodes: list[str] = []
    per_artifact: list[dict[str, Any]] = []
    for row in artifacts:
        category = str(row["category"])
        path = Path(str(row["artifact_path"]))
        value = _read_mapping(path, f"{category} artifact")
        nodes: list[tuple[str, Sequence[Any]]] = []
        if category == "mgtm":
            nodes.extend(
                (f"summary:{item.get('id')}", item.get("emb", []))
                for item in value.get("SUMMARIES", [])
            )
            nodes.extend(
                (f"fact:{item.get('id')}", item.get("emb", []))
                for item in value.get("FACTS", [])
            )
        else:
            nodes.extend(
                (f"summary:{item.get('id')}", item.get("embedding", []))
                for item in value.get("summary_nodes", [])
            )
            nodes.extend(
                (f"fact:{item.get('id')}", item.get("embedding", []))
                for item in value.get("fact_nodes", [])
            )
            nodes.extend(
                (f"claim:{item.get('id')}", item.get("embedding", []))
                for item in value.get("derived_claim_nodes", [])
            )
        artifact_invalid: list[str] = []
        for node_id, raw_vector in nodes:
            vector = [float(number) for number in raw_vector]
            if len(vector) != 3072 or not all(math.isfinite(number) for number in vector):
                artifact_invalid.append(node_id)
        total_nodes += len(nodes)
        if artifact_invalid:
            invalid_nodes.extend(
                f"{row['backbone']}/{row['task']}/{category}/{node_id}"
                for node_id in artifact_invalid
            )
        contract = value.get("embedding_contract", {})
        omitted = contract.get("provider_request_fields_omitted")
        if category == "session":
            omitted_ok = (
                contract.get("task_type") is None
                and contract.get("input_type") is None
                and contract.get("output_dimensionality") is None
            )
        else:
            omitted_ok = omitted == [
                "task_type",
                "input_type",
                "output_dimensionality",
            ]
        transport_ok = (
            contract.get("model") == "gemini-embedding-001"
            and contract.get("dimension") == EMBEDDING_DIMENSION
            and contract.get("batch_size") == EMBEDDING_BATCH_SIZE
            and contract.get("provider_visible_character_limit")
            == EMBEDDING_VISIBLE_CHARACTER_LIMIT
            and contract.get("cache_identity_uses_complete_untruncated_text")
            is True
            and contract.get("failure_policy")
            == "fail_closed_no_per_item_or_zero_fallback"
        )
        per_artifact.append(
            {
                "backbone": row["backbone"],
                "task": row["task"],
                "category": category,
                "nodes": len(nodes),
                "invalid_nodes": len(artifact_invalid),
                "provider_fields_omitted": omitted_ok,
                "transport_contract_passed": transport_ok,
            }
        )
        if not omitted_ok or not transport_ok:
            invalid_nodes.append(
                f"{row['backbone']}/{row['task']}/{category}/embedding_contract"
            )
    if invalid_nodes:
        raise BuildExecutorError(
            "postbuild embedding census failed: " + ", ".join(invalid_nodes[:20])
        )
    return {
        "status": "pass",
        "model": "gemini-embedding-001",
        "dimension": 3072,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "task_type": None,
        "input_type": None,
        "output_dimensionality": None,
        "artifacts": len(artifacts),
        "nodes": total_nodes,
        "rows": per_artifact,
        "rows_sha256": _sha256(per_artifact),
    }


def _strict_official_embedding_evidence(
    *,
    output_root: Path,
    artifacts: Sequence[Mapping[str, Any]],
    approved_attempt_ceiling: int,
    budget_binding_sha256: str,
) -> dict[str, Any]:
    """Pure-read replay of every formal embedding batch and graph vector."""

    runtime_path = Path(__file__).with_name(
        "memeye_official_embedding_batch_runtime_v19_1.py"
    )
    if _file_sha256(runtime_path) != OFFICIAL_EMBEDDING_RUNTIME_SHA256:
        raise BuildExecutorError("official embedding runtime drift before audit import")
    runtime = importlib.import_module(
        "creamem3.memeye_official_embedding_batch_runtime_v19_1"
    )
    if Path(runtime.__file__).resolve() != runtime_path.resolve():
        raise BuildExecutorError("official embedding audit import path drift")
    cache_root = Path(output_root) / "provider" / "embedding_native_3072"
    audit_runtime = runtime.OfficialGoogleBatchEmbeddingRuntime(
        cache_root,
        containment_root=Path(output_root),
        limits=runtime.EmbeddingLimits(
            max_live_calls=int(approved_attempt_ceiling),
            max_attempts=EMBEDDING_MAX_PROVIDER_ATTEMPTS,
        ),
        allow_model_calls=False,
        budget_binding_sha256=str(budget_binding_sha256),
    )
    expected_resume_hashes: set[str] = set()
    validated_text_vector_pairs: set[tuple[str, str]] = set()
    invocation_rows: list[dict[str, Any]] = []
    graph_pairs: list[tuple[str, str, str]] = []
    category_order = {"session": 0, "online": 1, "offline": 2, "mgtm": 3}
    for artifact_row in sorted(
        artifacts,
        key=lambda row: (
            str(row["backbone"]),
            str(row["task"]),
            category_order.get(str(row["category"]), 99),
        ),
    ):
        backbone = str(artifact_row["backbone"])
        task = str(artifact_row["task"])
        category = str(artifact_row["category"])
        artifact = _read_mapping(
            Path(str(artifact_row["artifact_path"])),
            f"{category} embedding artifact",
        )
        contract = artifact.get("embedding_contract")
        if not isinstance(contract, Mapping):
            raise BuildExecutorError("artifact lacks embedding_contract")
        invocations = contract.get("invocations")
        if not isinstance(invocations, list):
            raise BuildExecutorError("artifact embedding invocations are invalid")
        for invocation_index, raw in enumerate(invocations):
            if not isinstance(raw, Mapping):
                raise BuildExecutorError("embedding invocation row is invalid")
            invocation = dict(raw)
            texts = invocation.get("materialized_texts")
            vector_sha256s = invocation.get("vector_sha256s")
            cache_metadata = invocation.get("cache_metadata")
            if (
                not isinstance(texts, list)
                or not texts
                or not all(isinstance(text, str) and text.strip() for text in texts)
                or not isinstance(vector_sha256s, list)
                or len(vector_sha256s) != len(texts)
                or not isinstance(cache_metadata, Mapping)
            ):
                raise BuildExecutorError("embedding invocation evidence shape drift")
            materialized = [str(text) for text in texts]
            visible = [
                text[:EMBEDDING_VISIBLE_CHARACTER_LIMIT] for text in materialized
            ]
            expected_metadata = {
                "backbone": backbone,
                "task": task,
                "texts_sha256": _sha256(materialized),
                "provider_visible_texts_sha256": _sha256(visible),
                "batch_size": EMBEDDING_BATCH_SIZE,
                "provider_visible_character_limit": (
                    EMBEDDING_VISIBLE_CHARACTER_LIMIT
                ),
                "cache_identity_uses_complete_untruncated_text": True,
                "task_type_omitted": True,
                "input_type_omitted": True,
                "output_dimensionality_omitted": True,
            }
            if (
                dict(cache_metadata) != expected_metadata
                or invocation.get("texts") != len(materialized)
                or invocation.get("texts_sha256") != _sha256(materialized)
                or invocation.get("text_sha256s")
                != [_sha256(text) for text in materialized]
                or invocation.get("visible_text_sha256s")
                != [_sha256(text) for text in visible]
                or invocation.get("provider_visible_texts_sha256")
                != _sha256(visible)
                or invocation.get("physical_batches")
                != _embedding_batches(len(materialized))
            ):
                raise BuildExecutorError("embedding invocation contract drift")
            replayed_vectors: list[list[float]] = []
            replayed_vector_hashes: list[str] = []
            batch_rows: list[dict[str, Any]] = []
            for start in range(0, len(materialized), EMBEDDING_BATCH_SIZE):
                batch = materialized[start : start + EMBEDDING_BATCH_SIZE]
                metadata = {
                    **expected_metadata,
                    "tag": str(invocation["tag"]),
                    "batch_start": start,
                    "batch_size": len(batch),
                }
                result = audit_runtime.embed_batch(batch, cache_metadata=metadata)
                expected_resume_hashes.add(str(result.resume_hash))
                vectors = [list(map(float, vector)) for vector in result.vectors]
                replayed_vectors.extend(vectors)
                replayed_vector_hashes.extend(str(value) for value in result.vector_sha256s)
                batch_rows.append(
                    {
                        "resume_hash": str(result.resume_hash),
                        "request_sha256": str(result.request_sha256),
                        "response_sha256": str(result.response_sha256),
                        "vectors_sha256": str(result.vectors_sha256),
                        "cache_path": str(result.cache_path),
                        "cache_file_sha256": _file_sha256(Path(result.cache_path)),
                        "successful_attempt_ordinal": int(
                            result.successful_attempt_ordinal
                        ),
                    }
                )
            if (
                replayed_vector_hashes != [str(value) for value in vector_sha256s]
                or _sha256(replayed_vectors) != invocation.get("vectors_sha256")
            ):
                raise BuildExecutorError("embedding invocation/cache vector drift")
            validated_text_vector_pairs.update(
                (_sha256(text), _sha256(vector))
                for text, vector in zip(materialized, replayed_vectors)
            )
            invocation_rows.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "category": category,
                    "invocation_index": invocation_index,
                    "tag": str(invocation["tag"]),
                    "texts": len(materialized),
                    "batches": batch_rows,
                    "batches_sha256": _sha256(batch_rows),
                }
            )
        if category == "mgtm":
            nodes = [*artifact.get("SUMMARIES", []), *artifact.get("FACTS", [])]
            vector_field = "emb"
        else:
            nodes = [
                *artifact.get("summary_nodes", []),
                *artifact.get("fact_nodes", []),
                *artifact.get("derived_claim_nodes", []),
            ]
            vector_field = "embedding"
        for node in nodes:
            text_value = str(node.get("text", ""))
            vector = [float(value) for value in node.get(vector_field, [])]
            pair = (_sha256(text_value), _sha256(vector))
            if pair not in validated_text_vector_pairs:
                graph_pairs.append(
                    (backbone, task, f"{category}:{node.get('id')}")
                )
    if graph_pairs:
        raise BuildExecutorError(
            "graph vectors lack strict provider evidence: "
            + ", ".join("/".join(row) for row in graph_pairs[:20])
        )
    response_root = cache_root / "responses"
    plan_root = cache_root / "plans"
    attempt_root = cache_root / "attempts"
    observed_responses = {
        path.stem for path in response_root.glob("*.json") if path.is_file()
    } if response_root.is_dir() else set()
    observed_plans = {
        path.stem for path in plan_root.glob("*.json") if path.is_file()
    } if plan_root.is_dir() else set()
    observed_attempt_roots = {
        path.name for path in attempt_root.iterdir() if path.is_dir()
    } if attempt_root.is_dir() else set()
    if not (
        observed_responses
        == observed_plans
        == observed_attempt_roots
        == expected_resume_hashes
    ):
        raise BuildExecutorError("official embedding cache identity census drift")
    for forbidden in ("orphans", "failures"):
        root = cache_root / forbidden
        if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
            raise BuildExecutorError(f"official embedding {forbidden} census is nonempty")
    lifecycle_root = cache_root / "lifecycle_locks"
    observed_lifecycle = {
        path.stem for path in lifecycle_root.glob("*.lock") if path.is_file()
    } if lifecycle_root.is_dir() else set()
    if observed_lifecycle != expected_resume_hashes:
        raise BuildExecutorError("official embedding lifecycle-lock census drift")
    for resume_hash_value in sorted(expected_resume_hashes):
        lock_path = lifecycle_root / f"{resume_hash_value}.lock"
        if lock_path.read_bytes() != b"\0":
            raise BuildExecutorError("official embedding lifecycle-lock bytes drift")
        with runtime.acquire_request_lifecycle_lock(
            lock_path, timeout_seconds=0.1, poll_seconds=0.01
        ):
            pass
    budget_lock = cache_root / "budget_lifecycle" / "global.lock"
    if budget_lock.read_bytes() != b"\0":
        raise BuildExecutorError("official embedding global budget-lock bytes drift")
    with runtime.acquire_request_lifecycle_lock(
        budget_lock, timeout_seconds=0.1, poll_seconds=0.01
    ):
        pass
    usage = audit_runtime.usage_receipt()
    if (
        int(usage.get("provider_calls", -1)) != 0
        or int(usage.get("durable_provider_attempts", -1))
        < len(expected_resume_hashes)
        or int(usage.get("durable_provider_attempts", -1))
        > int(approved_attempt_ceiling)
    ):
        raise BuildExecutorError("official embedding pure-read usage census drift")
    invocation_rows.sort(
        key=lambda row: (
            row["backbone"], row["task"], row["category"], row["invocation_index"]
        )
    )
    return {
        "status": "pass",
        "runtime_sha256": OFFICIAL_EMBEDDING_RUNTIME_SHA256,
        "cache_root": str(cache_root.resolve()),
        "budget_binding_sha256": str(budget_binding_sha256),
        "identities": len(expected_resume_hashes),
        "durable_provider_attempts": int(usage["durable_provider_attempts"]),
        "provider_calls_during_validation": 0,
        "invocations": len(invocation_rows),
        "invocation_rows_sha256": _sha256(invocation_rows),
    }


def _usage_provider_calls(receipt: Mapping[str, Any]) -> int:
    if isinstance(receipt.get("provider_calls"), int):
        return int(receipt["provider_calls"])
    stages = receipt.get("stages")
    if isinstance(stages, Mapping):
        return sum(
            int(row.get("provider_calls", 0))
            for row in stages.values()
            if isinstance(row, Mapping)
        )
    usage = receipt.get("usage")
    if isinstance(usage, Mapping):
        return sum(
            int(row.get("calls", 0))
            for row in usage.values()
            if isinstance(row, Mapping)
        )
    return int(receipt.get("logical_calls", 0))


def _strict_build_artifact_rows(
    *, output_root: Path, writer: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct all fixed artifact/receipt identities from sealed writer input."""

    tasks = _strict_writer_task_rows(output_root=Path(output_root), writer=writer)
    caption_maps = _strict_writer_caption_banks(
        output_root=Path(output_root), writer=writer, tasks=tasks
    )
    expected_artifacts: list[dict[str, Any]] = []
    architecture: list[dict[str, Any]] = []
    expected_artifact_files: set[Path] = set()
    expected_receipt_files: set[Path] = set()
    for backbone in ("g25", "g35"):
        for task_row in tasks:
            task = str(task_row["task"])
            captions = caption_maps[backbone][task]
            records, source_map = ingest.build_records_from_captions(
                task, task_row["memory"], captions
            )
            _rounds_native, _batches, _rendered = ingest.native_batch_plan(
                task, task_row["memory"], captions
            )
            session_packet_paths = [
                (
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "session"
                    / task
                    / f"S{int(record['session_id']):04d}.json"
                ).resolve()
                for record in records
            ]
            session_packets = [
                _read_mapping(path, "strict session packet")
                for path in session_packet_paths
            ]
            session_input_sha = _sha256(
                {
                    "backbone": backbone,
                    "task": task,
                    "memory_sha256": task_row["memory_sha256"],
                    "caption_map_sha256": _sha256(dict(captions)),
                    "packet_sha256": [_sha256(packet) for packet in session_packets],
                    "embedding": "gemini-embedding-001_native_3072_omitted_fields",
                }
            )
            session_path, session_receipt_path = _artifact_pair(
                Path(output_root), "session", backbone, task
            )
            session = _inspect_artifact_pair(
                session_path,
                session_receipt_path,
                expected_input_sha256=session_input_sha,
                expected_category="session",
                expected_backbone=backbone,
                expected_task=task,
            )
            if session is None or session.get("session_packets") != session_packets:
                raise BuildExecutorError("sealed session artifact/packet drift")
            _assert_session_graph(session, expected_sessions=len(records))
            _assert_exact_session_materialization(
                session,
                packets=session_packets,
                source_map=source_map,
                backbone=backbone,
                task=task,
            )

            online_input_sha = _sha256(
                {
                    "base_content_fingerprint": session.get("content_fingerprint"),
                    "prompt_sha256": ONLINE_PROMPT_SHA256,
                    "current_fact_cap": 16,
                    "prior_fact_cap": 24,
                    "prune_unclaimed_create_members": True,
                    "embedding": "gemini-embedding-001_native_3072_omitted_fields",
                }
            )
            online_path, online_receipt_path = _artifact_pair(
                Path(output_root), "online", backbone, task
            )
            online_graph = _inspect_artifact_pair(
                online_path,
                online_receipt_path,
                expected_input_sha256=online_input_sha,
                expected_category="online",
                expected_backbone=backbone,
                expected_task=task,
            )
            if online_graph is None:
                raise BuildExecutorError("sealed Online artifact missing")
            online_audit = online.audit_online_graph(
                session,
                online_graph,
                expected_prompt_sha256=ONLINE_PROMPT_SHA256,
                expected_current_fact_cap=16,
                expected_prior_fact_cap=24,
                expected_prune_unclaimed_create_members=True,
            )
            if (
                online_audit.get("status") != "pass"
                or online_graph.get("stats", {}).get("online_fusion_calls")
                != max(0, len(records) - 1)
            ):
                raise BuildExecutorError("sealed Online causal audit drift")

            large = len(session.get("fact_nodes", [])) > 1000
            offline_input_sha = _sha256(
                {
                    "base_content_fingerprint": session.get("content_fingerprint"),
                    "branch": "large" if large else "standard",
                    "prompt_sha256": (
                        OFFLINE_LARGE_PROMPT_SHA256
                        if large
                        else OFFLINE_STANDARD_PROMPT_SHA256
                    ),
                    "max_schema_requests": 4 if large else 3,
                    "member_fact_cap": 12 if large else None,
                    "embedding": "gemini-embedding-001_native_3072_omitted_fields",
                }
            )
            offline_path, offline_receipt_path = _artifact_pair(
                Path(output_root), "offline", backbone, task
            )
            offline_graph = _inspect_artifact_pair(
                offline_path,
                offline_receipt_path,
                expected_input_sha256=offline_input_sha,
                expected_category="offline",
                expected_backbone=backbone,
                expected_task=task,
            )
            if offline_graph is None:
                raise BuildExecutorError("sealed Offline artifact missing")
            _assert_offline_graph(session, offline_graph)

            rounds, batches, rendered = ingest.native_batch_plan(
                task, task_row["memory"], captions
            )
            native_packet_paths = [
                (
                    Path(output_root)
                    / "build"
                    / "packets"
                    / backbone
                    / "mgtm"
                    / task
                    / f"NB{index:04d}.json"
                ).resolve()
                for index in range(1, len(batches) + 1)
            ]
            native_packets = [
                _read_mapping(path, "strict native packet")
                for path in native_packet_paths
            ]
            native_input_sha = _sha256(
                {
                    "backbone": backbone,
                    "task": task,
                    "memory_sha256": task_row["memory_sha256"],
                    "caption_map_sha256": _sha256(dict(captions)),
                    "packet_sha256": [_sha256(packet) for packet in native_packets],
                    "embedding": "gemini-embedding-001_native_3072_omitted_fields",
                }
            )
            native_path, native_receipt_path = _artifact_pair(
                Path(output_root), "mgtm", backbone, task
            )
            native_tree = _inspect_artifact_pair(
                native_path,
                native_receipt_path,
                expected_input_sha256=native_input_sha,
                expected_category="mgtm",
                expected_backbone=backbone,
                expected_task=task,
            )
            if native_tree is None:
                raise BuildExecutorError("sealed native MGTM artifact missing")
            _assert_native_tree(
                native_tree,
                task=task,
                backbone=backbone,
                rounds=rounds,
                packets=native_packets,
            )

            category_values = {
                "session": (session_path, session_receipt_path),
                "online": (online_path, online_receipt_path),
                "offline": (offline_path, offline_receipt_path),
                "mgtm": (native_path, native_receipt_path),
            }
            receipt_values: dict[str, dict[str, Any]] = {}
            for category, (artifact_path, receipt_path) in category_values.items():
                receipt = _read_mapping(receipt_path, f"strict {category} receipt")
                receipt_values[category] = receipt
                expected_artifact_files.add(artifact_path.resolve())
                expected_receipt_files.add(receipt_path.resolve())
                expected_artifacts.append(
                    {
                        "backbone": backbone,
                        "task": task,
                        "category": category,
                        "artifact_path": str(artifact_path.resolve()),
                        "artifact_sha256": str(receipt["artifact_sha256"]),
                        "content_fingerprint": receipt.get("content_fingerprint"),
                        "receipt_sha256": str(receipt["receipt_sha256"]),
                    }
                )
            architecture.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "mgtm_matched_path": str(native_path.resolve()),
                    "mgtm_matched_sha256": receipt_values["mgtm"]["artifact_sha256"],
                    "session_only_path": str(session_path.resolve()),
                    "session_only_sha256": receipt_values["session"]["artifact_sha256"],
                    "online_base_path": str(online_path.resolve()),
                    "online_base_sha256": receipt_values["online"]["artifact_sha256"],
                    "online_operation_context_v3_path": str(online_path.resolve()),
                    "online_operation_context_v3_sha256": receipt_values["online"]["artifact_sha256"],
                    "offline_base_path": str(offline_path.resolve()),
                    "offline_base_sha256": receipt_values["offline"]["artifact_sha256"],
                }
            )
    graph_root = Path(output_root) / "build" / "graphs"
    receipt_root = Path(output_root) / "build" / "receipts"
    observed_artifacts = {
        path.resolve() for path in graph_root.rglob("*.json") if path.is_file()
    }
    observed_receipts = {
        path.resolve() for path in receipt_root.rglob("*.json") if path.is_file()
    }
    if observed_artifacts != expected_artifact_files:
        raise BuildExecutorError("build graph filesystem census drift")
    if observed_receipts != expected_receipt_files:
        raise BuildExecutorError("build receipt filesystem census drift")
    return (
        sorted(
            expected_artifacts,
            key=lambda row: (row["backbone"], row["task"], row["category"]),
        ),
        sorted(architecture, key=lambda row: (row["backbone"], row["task"])),
    )


def validate_build_complete(
    *, output_root: Path, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    protocol = dict(protocol or matrix.load_protocol())
    path = Path(output_root) / "build" / "complete.json"
    value = _read_mapping(path, "BUILD_COMPLETE seal")
    contract = dict(value)
    observed = contract.pop("complete_sha256", None)
    if observed != _sha256(contract):
        raise BuildExecutorError("BUILD_COMPLETE digest drift")
    required_keys = {
        "schema_version",
        "complete",
        "experiment_id",
        "protocol_sha256",
        "project_contract_sha256",
        "plan_sha256",
        "executor_sha256",
        "matched_ingest_sha256",
        "caption_receipts",
        "artifacts",
        "architecture_bindings",
        "embedding_census",
        "embedding_evidence_census",
        "provider_smoke_seals",
        "writer_complete_file_sha256",
        "writer_generation_evidence_file_sha256",
        "embedding_budget_file_sha256",
        "dynamic_embedding_gate_file_sha256",
        "provider_usage",
        "writer_subphase_provider_usage",
        "writer_subphase_provider_calls",
        "materialize_view_subphase_provider_calls",
        "provider_calls",
        "generation_provider_attempt_hard_ceiling",
        "embedding_provider_attempt_hard_ceiling",
        "provider_attempt_hard_ceiling",
        "generation_stage_provider_attempt_hard_ceilings",
        "legacy_cache_readthrough",
        "predict_or_score_implemented",
        "complete_sha256",
    }
    if set(value) != required_keys:
        raise BuildExecutorError("BUILD_COMPLETE keyset drift")
    writer = validate_writer_complete(output_root=output_root, protocol=protocol)
    gate = validate_dynamic_embedding_budget_gate(
        output_root=output_root,
        protocol=protocol,
        allow_downstream_view=True,
    )
    budget = gate["budget"]
    plan_path = Path(output_root) / "build" / "plan.json"
    plan = _read_mapping(plan_path, "sealed build plan")
    plan_contract = dict(plan)
    observed_plan_sha = plan_contract.pop("plan_sha256", None)
    if (
        observed_plan_sha != _sha256(plan_contract)
        or plan.get("protocol_sha256") != _sha256(protocol)
        or Path(str(plan.get("output_root", ""))).resolve()
        != Path(output_root).resolve()
    ):
        raise BuildExecutorError("sealed build plan drift")
    if (
        value.get("schema_version") != BUILD_COMPLETE_SCHEMA
        or value.get("complete") is not True
        or value.get("experiment_id") != protocol.get("experiment_id")
        or value.get("protocol_sha256") != _sha256(protocol)
        or value.get("project_contract_sha256")
        != writer.get("project_contract_sha256")
        or value.get("plan_sha256") != plan.get("plan_sha256")
        or value.get("executor_sha256")
        != _file_sha256(Path(__file__).resolve())
        or value.get("matched_ingest_sha256")
        != _file_sha256(Path(ingest.__file__).resolve())
        or value.get("caption_receipts") != writer.get("caption_receipts")
        or value.get("provider_attempt_hard_ceiling")
        != budget["build_overall_provider_attempt_hard_ceiling"]
        or value.get("embedding_provider_attempt_hard_ceiling")
        != budget["embedding_provider_attempt_hard_ceiling"]
        or value.get("generation_provider_attempt_hard_ceiling")
        != BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        or value.get("writer_complete_file_sha256")
        != budget["writer_complete_file_sha256"]
        or value.get("writer_generation_evidence_file_sha256")
        != budget["writer_generation_evidence_file_sha256"]
        or value.get("embedding_budget_file_sha256")
        != _file_sha256(Path(output_root) / "build" / "embedding_budget.json")
        or value.get("dynamic_embedding_gate_file_sha256")
        != gate["file_sha256"]
        or value.get("generation_stage_provider_attempt_hard_ceilings")
        != GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
        or value.get("legacy_cache_readthrough") is not False
        or value.get("predict_or_score_implemented") is not False
    ):
        raise BuildExecutorError("BUILD_COMPLETE contract drift")
    artifacts = value.get("artifacts")
    architecture = value.get("architecture_bindings")
    if not isinstance(artifacts, list) or not isinstance(architecture, list):
        raise BuildExecutorError("BUILD_COMPLETE artifacts are invalid")
    expected_artifacts, expected_architecture = _strict_build_artifact_rows(
        output_root=Path(output_root), writer=writer
    )
    if artifacts != expected_artifacts:
        raise BuildExecutorError("BUILD_COMPLETE exact artifact census drift")
    if architecture != expected_architecture:
        raise BuildExecutorError("BUILD_COMPLETE architecture binding census drift")
    formal_project = (Path(output_root) / "project" / "manifest.json").is_file()
    if formal_project and (
        len(artifacts) != EXPECTED_TASKS * 2 * 4
        or len(architecture) != EXPECTED_TASKS * 2
    ):
        raise BuildExecutorError("formal BUILD_COMPLETE identity cardinality drift")
    expected_embedding_census = _embedding_census(expected_artifacts)
    if value.get("embedding_census") != expected_embedding_census:
        raise BuildExecutorError("BUILD_COMPLETE embedding census drift")
    provider_usage = value.get("provider_usage")
    if not isinstance(provider_usage, Mapping) or set(provider_usage) != {
        "g25",
        "g35",
        "embedding",
    }:
        raise BuildExecutorError("BUILD_COMPLETE provider usage shape drift")
    if formal_project:
        expected_generation_usage = {
            "g25": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g25",
                included_stages={"writer", "view"},
            ),
            "g35": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g35",
                included_stages={"caption", "writer", "view"},
            ),
        }
        if {
            "g25": provider_usage["g25"],
            "g35": provider_usage["g35"],
        } != expected_generation_usage:
            raise BuildExecutorError(
                "BUILD_COMPLETE canonical generation usage drift"
            )
    generation_calls = sum(
        _usage_provider_calls(dict(provider_usage[name])) for name in ("g25", "g35")
    )
    embedding_calls = _usage_provider_calls(dict(provider_usage["embedding"]))
    if (
        value.get("writer_subphase_provider_usage") != writer.get("provider_usage")
        or value.get("writer_subphase_provider_calls") != writer.get("provider_calls")
        or value.get("materialize_view_subphase_provider_calls")
        != generation_calls - int(writer["provider_calls"]) + embedding_calls
        or value.get("provider_calls") != generation_calls + embedding_calls
        or generation_calls < int(writer["provider_calls"])
        or generation_calls > BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        or embedding_calls > int(budget["embedding_provider_attempt_hard_ceiling"])
    ):
        raise BuildExecutorError("BUILD_COMPLETE provider usage census drift")
    if formal_project:
        expected_smokes = validate_provider_smoke_seals(
            output_root=Path(output_root), protocol=protocol
        )
        if value.get("provider_smoke_seals") != expected_smokes:
            raise BuildExecutorError("BUILD_COMPLETE provider smoke binding drift")
    elif value.get("provider_smoke_seals") not in ({}, None):
        raise BuildExecutorError("nonformal BUILD_COMPLETE smoke marker drift")
    if Path(output_root).resolve() == matrix.DEFAULT_ROOT.resolve():
        expected_evidence = _strict_official_embedding_evidence(
            output_root=Path(output_root),
            artifacts=expected_artifacts,
            approved_attempt_ceiling=int(
                budget["embedding_provider_attempt_hard_ceiling"]
            ),
            budget_binding_sha256=str(
                value["dynamic_embedding_gate_file_sha256"]
            ),
        )
        if value.get("embedding_evidence_census") != expected_evidence:
            raise BuildExecutorError("BUILD_COMPLETE embedding evidence drift")
    elif value.get("embedding_evidence_census") != {
        "status": "test_injected_backend_not_formal"
    }:
        raise BuildExecutorError("nonformal embedding evidence marker drift")
    return value


def execute_build_with_backends(
    *,
    project: Mapping[str, Any],
    output_root: Path,
    generation_backends: Mapping[str, GenerationBackend],
    embedding_backend: EmbeddingBackend,
    caption_root: Path = matrix.DEFAULT_V7_CAPTION_ROOT,
    image_root: Path | None = None,
    enforce_full_scope: bool = True,
    smoke_seals: Mapping[str, Any] | None = None,
    dynamic_embedding_gate_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute/resume the build using injected live or fake providers."""

    protocol = matrix.load_protocol()
    complete_path = Path(output_root) / "build" / "complete.json"
    if complete_path.is_file():
        resumed = validate_build_complete(output_root=output_root, protocol=protocol)
        return {
            "status": "resumed_complete",
            "provider_calls": 0,
            "complete_path": str(complete_path.resolve()),
            "complete_sha256": resumed["complete_sha256"],
        }
    if set(generation_backends) != {"g25", "g35"}:
        raise BuildExecutorError("both generation backbones are required")
    for backbone, backend in generation_backends.items():
        if backend.backbone != backbone:
            raise BuildExecutorError("generation backend identity drift")
    # No graph helper can reach ``embedding_backend`` before this binding is
    # present.  Formal execution obtains it only from the independently sealed
    # dynamic gate; fake tests must create the same gate-shaped receipt.
    if not isinstance(dynamic_embedding_gate_receipt, Mapping):
        raise BuildExecutorError(
            "dynamic embedding budget gate is required before materialization"
        )
    dynamic_budget = dynamic_embedding_gate_receipt.get("budget")
    if (
        not isinstance(dynamic_budget, Mapping)
        or dynamic_budget.get("schema_version") != EMBEDDING_BUDGET_SCHEMA
        or dynamic_budget.get("protocol_sha256") != _sha256(protocol)
        or dynamic_budget.get("writer_complete_file_sha256")
        != _file_sha256(Path(output_root) / "build" / "writer_complete.json")
    ):
        raise BuildExecutorError("dynamic embedding authorization binding drift")
    overall_provider_attempt_hard_ceiling = int(
        dynamic_budget["build_overall_provider_attempt_hard_ceiling"]
    )
    plan_path = Path(output_root) / "build" / "plan.json"
    plan = (
        validate_build_plan(output_root=output_root, project=project)
        if plan_path.is_file()
        else plan_build(
            project=project,
            output_root=output_root,
            caption_root=caption_root,
            enforce_full_scope=enforce_full_scope,
        )
    )
    tasks = _project_tasks(project)
    if image_root is None:
        image_root = matrix.DEFAULT_DATA_ROOT / "data" / "image"

    caption_maps: dict[str, dict[str, dict[str, str]]] = {"g25": {}, "g35": {}}
    caption_receipts: list[dict[str, Any]] = []
    for task_row in tasks:
        task = str(task_row["task"])
        g25, receipt25 = load_g25_caption_bank_for_task(task_row, caption_root)
        g35, receipt35 = build_g35_caption_bank_for_task(
            task_row,
            backend=generation_backends["g35"],
            image_root=image_root,
            output_root=output_root,
        )
        caption_maps["g25"][task] = g25
        caption_maps["g35"][task] = g35
        caption_receipts.extend(
            [
                {"backbone": "g25", **receipt25},
                {"backbone": "g35", **receipt35},
            ]
        )
    if enforce_full_scope and sum(row["images"] for row in caption_receipts) != 876:
        raise BuildExecutorError("two-backbone caption census drift")

    artifact_rows: list[dict[str, Any]] = []
    architecture_bindings: list[dict[str, Any]] = []
    for backbone in ("g25", "g35"):
        generation = generation_backends[backbone]
        for task_row in tasks:
            task = str(task_row["task"])
            captions = caption_maps[backbone][task]
            session, session_receipt = build_session_graph_for_task(
                task_row,
                captions,
                backbone=backbone,
                generation=generation,
                embedding=embedding_backend,
                output_root=output_root,
            )
            online_graph, online_receipt = build_online_graph_for_task(
                task_row,
                session,
                backbone=backbone,
                generation=generation,
                embedding=embedding_backend,
                output_root=output_root,
            )
            offline_graph, offline_receipt = build_offline_graph_for_task(
                task_row,
                session,
                backbone=backbone,
                generation=generation,
                embedding=embedding_backend,
                output_root=output_root,
            )
            native, native_receipt = build_native_tree_for_task(
                task_row,
                captions,
                backbone=backbone,
                generation=generation,
                embedding=embedding_backend,
                output_root=output_root,
            )
            receipts = {
                "session": session_receipt,
                "online": online_receipt,
                "offline": offline_receipt,
                "mgtm": native_receipt,
            }
            for category, receipt in receipts.items():
                artifact_rows.append(
                    {
                        "backbone": backbone,
                        "task": task,
                        "category": category,
                        "artifact_path": receipt["artifact_path"],
                        "artifact_sha256": receipt["artifact_sha256"],
                        "content_fingerprint": receipt.get("content_fingerprint"),
                        "receipt_sha256": receipt["receipt_sha256"],
                    }
                )
            architecture_bindings.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "mgtm_matched_path": native_receipt["artifact_path"],
                    "mgtm_matched_sha256": native_receipt["artifact_sha256"],
                    "session_only_path": session_receipt["artifact_path"],
                    "session_only_sha256": session_receipt["artifact_sha256"],
                    "online_base_path": online_receipt["artifact_path"],
                    "online_base_sha256": online_receipt["artifact_sha256"],
                    "online_operation_context_v3_path": online_receipt[
                        "artifact_path"
                    ],
                    "online_operation_context_v3_sha256": online_receipt[
                        "artifact_sha256"
                    ],
                    "offline_base_path": offline_receipt["artifact_path"],
                    "offline_base_sha256": offline_receipt["artifact_sha256"],
                }
            )

    expected_artifacts = len(tasks) * 2 * 4
    if len(artifact_rows) != expected_artifacts:
        raise BuildExecutorError("build artifact census drift")
    sealed_artifact_rows = sorted(
        artifact_rows,
        key=lambda row: (row["backbone"], row["task"], row["category"]),
    )
    embedding_census = _embedding_census(sealed_artifact_rows)
    formal_project = (Path(output_root) / "project" / "manifest.json").is_file()
    if formal_project:
        usage = {
            "g25": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g25",
                included_stages={"writer", "view"},
            ),
            "g35": _canonical_generation_usage(
                output_root=Path(output_root),
                backbone="g35",
                included_stages={"caption", "writer", "view"},
            ),
        }
    else:
        usage = {
            backbone: dict(generation_backends[backbone].usage_receipt())
            for backbone in ("g25", "g35")
        }
    usage["embedding"] = dict(embedding_backend.usage_receipt())
    writer_complete = validate_writer_complete(
        output_root=output_root, protocol=protocol
    )
    prior_writer_provider_calls = int(writer_complete["provider_calls"])
    generation_provider_calls = sum(
        _usage_provider_calls(usage[backbone]) for backbone in ("g25", "g35")
    )
    embedding_provider_calls = _usage_provider_calls(usage["embedding"])
    if generation_provider_calls < prior_writer_provider_calls:
        raise BuildExecutorError(
            "durable generation census fell below sealed writer subphase"
        )
    # Generation receipts are cumulative across the writer and view phases;
    # adding writer_complete again would double-count every resumed attempt.
    materialize_view_provider_calls = (
        generation_provider_calls
        - prior_writer_provider_calls
        + embedding_provider_calls
    )
    provider_calls = generation_provider_calls + embedding_provider_calls
    if provider_calls > overall_provider_attempt_hard_ceiling:
        raise BuildExecutorError("build provider-attempt hard ceiling exceeded")
    if Path(output_root).resolve() == matrix.DEFAULT_ROOT.resolve():
        embedding_evidence_census = _strict_official_embedding_evidence(
            output_root=Path(output_root),
            artifacts=artifact_rows,
            approved_attempt_ceiling=int(
                dynamic_budget["embedding_provider_attempt_hard_ceiling"]
            ),
            budget_binding_sha256=str(
                dynamic_embedding_gate_receipt["file_sha256"]
            ),
        )
    else:
        embedding_evidence_census = {
            "status": "test_injected_backend_not_formal"
        }
    complete = {
        "schema_version": BUILD_COMPLETE_SCHEMA,
        "complete": True,
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": _sha256(protocol),
        "project_contract_sha256": project["project_contract_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "executor_sha256": _file_sha256(Path(__file__).resolve()),
        "matched_ingest_sha256": _file_sha256(Path(ingest.__file__).resolve()),
        "caption_receipts": caption_receipts,
        "artifacts": sealed_artifact_rows,
        "architecture_bindings": sorted(
            architecture_bindings,
            key=lambda row: (row["backbone"], row["task"]),
        ),
        "embedding_census": embedding_census,
        "embedding_evidence_census": embedding_evidence_census,
        "provider_smoke_seals": copy.deepcopy(dict(smoke_seals or {})),
        "writer_complete_file_sha256": dynamic_budget[
            "writer_complete_file_sha256"
        ],
        "writer_generation_evidence_file_sha256": dynamic_budget[
            "writer_generation_evidence_file_sha256"
        ],
        "embedding_budget_file_sha256": _file_sha256(
            Path(output_root) / "build" / "embedding_budget.json"
        ),
        "dynamic_embedding_gate_file_sha256": dynamic_embedding_gate_receipt.get(
            "file_sha256"
        ),
        "provider_usage": usage,
        "writer_subphase_provider_usage": writer_complete["provider_usage"],
        "writer_subphase_provider_calls": prior_writer_provider_calls,
        "materialize_view_subphase_provider_calls": (
            materialize_view_provider_calls
        ),
        "provider_calls": provider_calls,
        "generation_provider_attempt_hard_ceiling": (
            BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        ),
        "embedding_provider_attempt_hard_ceiling": dynamic_budget[
            "embedding_provider_attempt_hard_ceiling"
        ],
        "provider_attempt_hard_ceiling": overall_provider_attempt_hard_ceiling,
        "generation_stage_provider_attempt_hard_ceilings": dict(
            GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
        ),
        "legacy_cache_readthrough": False,
        "predict_or_score_implemented": False,
    }
    complete["complete_sha256"] = _sha256(complete)
    _write_once_json(complete_path, complete)
    validate_build_complete(output_root=output_root, protocol=protocol)
    return {
        "status": "complete",
        "provider_calls": provider_calls,
        "complete_path": str(complete_path.resolve()),
        "complete_sha256": complete["complete_sha256"],
        "artifacts": len(artifact_rows),
        "embedding_nodes": embedding_census["nodes"],
    }


def run_matrix_phase(request: Mapping[str, Any]) -> dict[str, Any]:
    """Formal gated entrypoint used by the main matrix runner."""

    if request.get("phase") != "build":
        raise BuildExecutorError("build executor accepts only phase=build")
    protocol = matrix.load_protocol(Path(str(request["protocol_path"])))
    if (
        request.get("protocol_sha256") != _sha256(protocol)
        or request.get("experiment_id") != protocol["experiment_id"]
        or request.get("legacy_cache_readthrough") is not False
        or not isinstance(request.get("logical_call_hard_ceiling"), int)
        or int(request["logical_call_hard_ceiling"])
        < BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
    ):
        raise BuildExecutorError("formal build request contract drift")
    output_root = Path(str(request["output_root"])).resolve()
    if output_root != matrix.DEFAULT_ROOT.resolve():
        raise BuildExecutorError("formal build escaped frozen r2 root")
    dynamic_gate_path = output_root / "build" / "dynamic_embedding_budget_gate.json"
    if not dynamic_gate_path.is_file() and (
        request.get("logical_call_hard_ceiling")
        != BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        or request.get("stage_call_hard_ceilings")
        != GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
    ):
        raise BuildExecutorError(
            "writer subphase release is not bound to exact generation ceilings"
        )
    project = matrix.validate_project(output_root=output_root)
    validate_build_plan(output_root=output_root, project=project)
    # This check deliberately precedes every provider-bearing import.
    smoke_seals = validate_provider_smoke_seals(
        output_root=output_root, protocol=protocol
    )
    complete_path = output_root / "build" / "complete.json"
    if complete_path.is_file():
        resumed = validate_build_complete(output_root=output_root, protocol=protocol)
        return {
            "status": "resumed_complete",
            "provider_calls": 0,
            "complete_path": str(complete_path.resolve()),
            "complete_sha256": resumed["complete_sha256"],
        }
    generation = {
        "g25": G25AuthorityBackend(output_root / "provider" / "g25"),
        "g35": G35RuntimeBackend(output_root / "provider" / "g35"),
    }
    writer = prepare_writer_packets(
        project=project,
        output_root=output_root,
        generation_backends=generation,
        caption_root=matrix.DEFAULT_V7_CAPTION_ROOT,
        image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
        enforce_full_scope=True,
    )
    if not dynamic_gate_path.is_file():
        generation_evidence = seal_writer_generation_evidence(
            output_root=output_root, protocol=protocol
        )
        budget = compute_dynamic_embedding_budget(
            output_root=output_root,
            protocol=protocol,
            allow_downstream_view=False,
        )
        return {
            "status": "waiting_dynamic_embedding_budget_gate",
            "provider_calls": int(writer["provider_calls"]),
            "writer_complete_path": str(
                (output_root / "build" / "writer_complete.json").resolve()
            ),
            "writer_complete_sha256": writer["writer_complete_sha256"],
            "writer_generation_evidence_path": str(
                (
                    output_root / "build" / "writer_generation_evidence.json"
                ).resolve()
            ),
            "writer_generation_evidence_sha256": generation_evidence[
                "generation_evidence_sha256"
            ],
            "embedding_budget_path": str(
                (output_root / "build" / "embedding_budget.json").resolve()
            ),
            "embedding_budget_sha256": budget["embedding_budget_sha256"],
            "generation_provider_attempt_hard_ceiling": (
                BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
            ),
            "embedding_provider_attempt_hard_ceiling": budget[
                "embedding_provider_attempt_hard_ceiling"
            ],
            "build_overall_provider_attempt_hard_ceiling": budget[
                "build_overall_provider_attempt_hard_ceiling"
            ],
            "embedding_calls": 0,
        }
    generation_evidence = _validate_sealed_writer_generation_evidence_binding(
        output_root=output_root,
        protocol=protocol,
        writer=writer,
    )
    dynamic_gate = validate_dynamic_embedding_budget_gate(
        output_root=output_root,
        protocol=protocol,
        allow_downstream_view=True,
    )
    budget = dynamic_gate["budget"]
    expected_stage_ceilings = {
        **GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS,
        "embedding": budget["embedding_provider_attempt_hard_ceiling"],
    }
    if (
        request.get("logical_call_hard_ceiling")
        != budget["build_overall_provider_attempt_hard_ceiling"]
        or request.get("stage_call_hard_ceilings") != expected_stage_ceilings
    ):
        raise BuildExecutorError(
            "formal build release was not rebound to the dynamic embedding budget"
        )
    # Provider-bearing embedding construction is deliberately below the
    # independently sealed dynamic gate.
    embedding = OfficialNative3072EmbeddingBackend(
        output_root / "provider" / "embedding_native_3072",
        containment_root=output_root,
        approved_provider_attempt_hard_ceiling=budget[
            "embedding_provider_attempt_hard_ceiling"
        ],
        budget_binding_sha256=str(dynamic_gate["file_sha256"]),
    )
    return execute_build_with_backends(
        project=project,
        output_root=output_root,
        generation_backends=generation,
        embedding_backend=embedding,
        caption_root=matrix.DEFAULT_V7_CAPTION_ROOT,
        image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
        enforce_full_scope=True,
        smoke_seals=smoke_seals,
        dynamic_embedding_gate_receipt=dynamic_gate,
    )


def blocked_formal_release() -> dict[str, Any]:
    """Formal-shaped WAITING release; it is intentionally not live-ready."""

    protocol = matrix.load_protocol()
    return {
        "schema_version": matrix.RELEASE_SCHEMA,
        "phase": "build",
        "status": "blocked_provider_smokes_and_g35_transport_decision",
        "protocol_sha256": _sha256(protocol),
        "module": "creamem3.build_executor_v19",
        "module_file_sha256": _file_sha256(Path(__file__).resolve()),
        "entrypoint": "run_matrix_phase",
        "logical_call_hard_ceiling": (
            BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING
        ),
        "stage_call_hard_ceilings": dict(
            GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS
        ),
        "dynamic_embedding_budget_status": (
            "pending_post_writer_100_item_batch_census_and_independent_gate"
        ),
        "provider_cache_root": str((matrix.DEFAULT_ROOT / "provider").resolve()),
        "legacy_cache_readthrough": False,
    }


def write_blocked_formal_release(
    output_root: Path = matrix.DEFAULT_ROOT,
) -> dict[str, Any]:
    value = blocked_formal_release()
    _write_json_atomic(Path(output_root) / "releases" / "build_executor.json", value)
    return value


__all__ = [
    "BUILD_COMPLETE_SCHEMA",
    "BUILD_EXECUTOR_SCHEMA",
    "BUILD_GENERATION_PROVIDER_ATTEMPT_HARD_CEILING",
    "BUILD_PLAN_SCHEMA",
    "DYNAMIC_EMBEDDING_GATE_SCHEMA",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_BUDGET_SCHEMA",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_VISIBLE_CHARACTER_LIMIT",
    "GENERATION_STAGE_PROVIDER_ATTEMPT_HARD_CEILINGS",
    "GenerationReply",
    "WRITER_COMPLETE_SCHEMA",
    "BuildExecutorError",
    "StrictEmbeddingFacade",
    "blocked_formal_release",
    "build_g35_caption_bank_for_task",
    "capability_preflight",
    "execute_build_with_backends",
    "compute_dynamic_embedding_budget",
    "load_g25_caption_bank_for_task",
    "plan_build",
    "prepare_writer_packets",
    "run_matrix_phase",
    "validate_build_complete",
    "validate_build_plan",
    "validate_dynamic_embedding_budget_gate",
    "validate_provider_smoke_seals",
    "validate_writer_complete",
    "write_blocked_formal_release",
]

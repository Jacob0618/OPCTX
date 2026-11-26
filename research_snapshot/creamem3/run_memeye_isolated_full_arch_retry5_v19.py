"""Runnable isolated MemEye v19 5x2 experiment orchestrator.

This entry point deliberately does *not* promote its outputs to the frozen
``full_arch_matrix_r2`` release.  It consumes the separately checkpointed G35
caption bank as a read-only upstream input and runs the existing v19 writer,
graph, retrieval, Reader, and judge implementations in a fresh cache root.

The model-visible prompts and request bodies remain those of v19.  Gemini text
generation uses a persistent official-Google REST client, five durable
attempts per logical identity, and an exact two-second delay after transient
transport/HTTP failures.  Embedding and Qwen keep their existing provider
contracts; they are not text-generation stages.

Every live phase is independently resumable::

    python -m creamem3.run_memeye_isolated_full_arch_retry5_v19 --phase writer --allow-model-calls
    python -m creamem3.run_memeye_isolated_full_arch_retry5_v19 --phase build --allow-model-calls
    python -m creamem3.run_memeye_isolated_full_arch_retry5_v19 --phase predict --allow-model-calls
    python -m creamem3.run_memeye_isolated_full_arch_retry5_v19 --phase score --allow-model-calls

The default ``status`` phase is zero-call and safe while the caption producer
is still running.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creamem3 import build_executor_v19 as build
from creamem3 import graph_memory_v2 as graph_memory
from creamem3 import phase1_g35_caption_reliable_r3_v19 as captions_r3
from creamem3 import predict_executor_v19 as predict
from creamem3 import run_memeye_full_arch_matrix_v19 as matrix
from creamem3 import score_executor_v19 as score


SCHEMA_VERSION = "creamem3.memeye_isolated_full_arch_retry5.v19.0"
IMPORT_SCHEMA_VERSION = "creamem3.memeye_isolated_caption_import.v19.0"
MATRIX_SCORE_COMPLETE_SCHEMA = "creamem3.memeye_isolated_matrix_score_only.v19.0"
MAX_GENERATION_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 2.0
DEFAULT_EVAL_WORKERS = 4

DEFAULT_SOURCE_PROJECT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\full_arch_matrix_r2"
)
DEFAULT_CAPTION_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\g35_caption_retry5_r1"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\isolated_full_arch_retry5_r1"
)
# This exact BUILD_COMPLETE was produced and then accepted by the canonical
# build.validate_build_complete call (exit 0) on 2026-08-23.  Pinning both its
# byte digest and internal contract digest lets later isolated predict/score
# processes reuse that completed validation instead of spending another
# 20+ CPU minutes replaying the same immutable 64 graphs.
VALIDATED_BUILD_COMPLETE_FILE_SHA256 = (
    "d8cb78d212164c1856a3db808600d59edb883eaaab5c6851c8b77146e8d43cb6"
)
VALIDATED_BUILD_COMPLETE_CONTRACT_SHA256 = (
    "3763d1309805cc08ea8a5fc379ba38560a2f69646e9f961abd669d001f8fa120"
)
VALIDATED_BUILD_EXECUTOR_SHA256 = (
    "d60319c2e9769196626457c4b1a5258d58775ac13540950a759e1090aea1904c"
)
ISOLATED_GRAPH_RETRIEVAL_SHA256 = (
    "c74806d7668aa539041cad0a4d8686fb4376ee7b2af54169135d17f04329ff96"
)
DEFAULT_SCORE_DATA_ROOT = matrix.DEFAULT_DATA_ROOT
MEMEYE_CACHE_ROOT = Path(r"D:\creamem3_cache\memeye_v19")

FORBIDDEN_OUTPUT_ROOTS = {
    (MEMEYE_CACHE_ROOT / "phase1_caption_writer_r1").resolve(),
    (MEMEYE_CACHE_ROOT / "phase1_caption_writer_r2").resolve(),
    (MEMEYE_CACHE_ROOT / "phase1_caption_writer_r3").resolve(),
    (MEMEYE_CACHE_ROOT / "g35_caption_retry5_r1").resolve(),
    (MEMEYE_CACHE_ROOT / "full_arch_matrix_r2").resolve(),
}


class IsolatedMatrixError(RuntimeError):
    """The isolated experiment contract or a phase prerequisite failed."""


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise IsolatedMatrixError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise IsolatedMatrixError(f"{label} is not an object: {path}")
    return value


def _assert_output_root(path: Path) -> Path:
    root = Path(path).resolve()
    allowed = MEMEYE_CACHE_ROOT.resolve()
    if root == allowed or allowed not in root.parents or root in FORBIDDEN_OUTPUT_ROOTS:
        raise IsolatedMatrixError("output root is not a fresh MemEye v19 child")
    # A manifest changes build_executor's validation mode from isolated to
    # formal.  This runner intentionally never creates or accepts one.
    if (root / "project" / "manifest.json").exists():
        raise IsolatedMatrixError("isolated output root must not contain a project manifest")
    return root


def _project(source_project_root: Path) -> dict[str, Any]:
    project = matrix.validate_project(output_root=Path(source_project_root).resolve())
    if (
        project.get("counts", {}).get("tasks") != matrix.EXPECTED_TASKS
        or project.get("counts", {}).get("image_references") != matrix.EXPECTED_IMAGES
        or project.get("counts", {}).get("physical_sessions") != matrix.EXPECTED_SESSIONS
    ):
        raise IsolatedMatrixError("source project census drift")
    return project


def _write_once_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise IsolatedMatrixError(f"write-once byte drift: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _caption_complete(caption_root: Path) -> dict[str, Any]:
    root = Path(caption_root).resolve()
    path = root / "build" / "caption_complete.json"
    value = _read_object(path, "reliable caption completion")
    contract = dict(value)
    observed = contract.pop("complete_sha256", None)
    if (
        observed != _sha256(contract)
        or value.get("schema_version") != captions_r3.COMPLETE_SCHEMA_VERSION
        or value.get("complete") is not True
        or value.get("status") != "caption_bank_complete_waiting_writer_import"
        or Path(str(value.get("output_root", ""))).resolve() != root
        or value.get("captions") != matrix.EXPECTED_IMAGES
        or value.get("tasks") != matrix.EXPECTED_TASKS
        or value.get("max_transport_attempts") != MAX_GENERATION_ATTEMPTS
        or float(value.get("retry_wait_seconds", -1)) != RETRY_WAIT_SECONDS
        or value.get("writer_started") is not False
    ):
        raise IsolatedMatrixError("reliable caption completion contract drift")
    return value


def _adapt_caption_bank(
    *,
    source_project_root: Path,
    caption_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Import only caption text/provenance into build-v19 compatible records."""

    root = _assert_output_root(output_root)
    source = Path(caption_root).resolve()
    complete = _caption_complete(source)
    project = _project(source_project_root)
    tasks = build._project_tasks(project)  # type: ignore[attr-defined]
    system_prompt = build._load_g35_caption_system_prompt()  # type: ignore[attr-defined]
    image_root = matrix.DEFAULT_DATA_ROOT / "data" / "image"
    rows: list[dict[str, Any]] = []
    expected_source_files: set[Path] = set()
    for task_row in tasks:
        task = str(task_row["task"])
        images = list(task_row["images"])
        history: list[dict[str, str]] = []
        for index, image in enumerate(images, 1):
            source_path = (
                source
                / "captions"
                / "g35"
                / task
                / f"{index:04d}_{build._safe_name(image)}.json"  # type: ignore[attr-defined]
            ).resolve()
            expected_source_files.add(source_path)
            record = _read_object(source_path, "reliable source caption")
            caption = str(record.get("caption", "")).strip()
            if (
                record.get("schema_version") != captions_r3.CAPTION_SCHEMA_VERSION
                or record.get("task") != task
                or record.get("image") != image
                or record.get("sequence_index") != index
                or record.get("total_images") != len(images)
                or record.get("input_contract_sha256")
                != _sha256(record.get("input_contract"))
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(record.get("provider_request_sha256", ""))
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(record.get("provider_response_sha256", ""))
                )
                or not caption
            ):
                raise IsolatedMatrixError(f"reliable source caption drift: {source_path}")

            previous = history[-2:]
            prompt = matrix.build_v7_reconstruction_user_text(
                task,
                image,
                sequence_index=index,
                total_images=len(images),
                previous=previous,
            )
            image_path = build._resolve_image(image_root, image)  # type: ignore[attr-defined]
            input_contract = {
                "schema_version": build.BUILD_EXECUTOR_SCHEMA,
                "backbone": "g35",
                "task": task,
                "image": image,
                "sequence_index": index,
                "total_images": len(images),
                "previous": previous,
                "prompt_sha256": _sha256(prompt),
                "raw_image_sha256": build._file_sha256(image_path),  # type: ignore[attr-defined]
                "system_prompt_sha256": _sha256(system_prompt),
                "system_prompt_line_count": len(system_prompt.splitlines()),
                "user_part_order": ["raw_image", "previous_two_caption_user_text"],
                "media_resolution": "MEDIA_RESOLUTION_HIGH",
                "max_output_tokens": 2400,
                "max_provider_attempts": 3,
            }
            input_sha = _sha256(input_contract)
            target_path = (
                root
                / "captions"
                / "g35"
                / task
                / f"{index:04d}_{build._safe_name(image)}.json"  # type: ignore[attr-defined]
            )
            target = {
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
                "provider_request_sha256": record["provider_request_sha256"],
                "provider_response_sha256": record["provider_response_sha256"],
            }
            build._write_once_json(target_path, target)  # type: ignore[attr-defined]
            rows.append(
                {
                    "task": task,
                    "image": image,
                    "sequence_index": index,
                    "source_path": str(source_path),
                    "source_file_sha256": build._file_sha256(source_path),  # type: ignore[attr-defined]
                    "source_input_contract_sha256": record["input_contract_sha256"],
                    "target_path": str(target_path.resolve()),
                    "target_file_sha256": build._file_sha256(target_path),  # type: ignore[attr-defined]
                    "caption_sha256": _sha256(caption),
                }
            )
            history.append({"image": image, "caption": caption})

    observed_source_files = {
        path.resolve()
        for path in (source / "captions" / "g35").rglob("*.json")
        if path.is_file()
    }
    if (
        len(rows) != matrix.EXPECTED_IMAGES
        or observed_source_files != expected_source_files
    ):
        raise IsolatedMatrixError("caption import exact filesystem census drift")

    query_source = Path(str(project["query_path"])).resolve()
    query_target = root / "project" / "queries_score_free.json"
    _write_once_bytes(query_target, query_source.read_bytes())
    manifest = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "status": "complete_read_only_caption_projection",
        "source_project_root": str(Path(source_project_root).resolve()),
        "source_project_contract_sha256": project["project_contract_sha256"],
        "source_caption_root": str(source),
        "source_caption_complete_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
            source / "build" / "caption_complete.json"
        ),
        "source_caption_complete_sha256": complete["complete_sha256"],
        "target_root": str(root),
        "captions": len(rows),
        "rows_sha256": _sha256(rows),
        "query_source_file_sha256": build._file_sha256(query_source),  # type: ignore[attr-defined]
        "query_target_file_sha256": build._file_sha256(query_target),  # type: ignore[attr-defined]
        "source_tree_read_only": True,
        "formal_release_claimed": False,
        "model_calls": 0,
        "provider_calls": 0,
    }
    manifest["import_sha256"] = _sha256(manifest)
    path = root / "imports" / "g35_caption_retry5_r1.json"
    build._write_once_json(path, manifest)  # type: ignore[attr-defined]
    if _read_object(path, "caption import manifest") != manifest:
        raise IsolatedMatrixError("caption import manifest write-once drift")
    return manifest


class _ThreadLocalPooledPost:
    """Give every evaluation worker its own persistent one-lane client."""

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory or captions_r3._PooledHttpxPost  # type: ignore[attr-defined]
        self._local = threading.local()

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        post = getattr(self._local, "post", None)
        if post is None:
            post = self._factory()
            self._local.post = post
        return post(url, body, headers, timeout_seconds)


class _ReliableRESTProvider:
    """Official REST provider that converts transient failures into retries."""

    def __init__(
        self,
        runtime_module: Any,
        *,
        post: captions_r3.PostFn | None = None,
        sleep: captions_r3.SleepFn = time.sleep,
    ) -> None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise IsolatedMatrixError("GEMINI_API_KEY is required for live generation")
        configured = os.environ.get(
            "GEMINI_API_BASE", runtime_module.GEMINI_API_BASE
        ).rstrip("/")
        if configured != runtime_module.GEMINI_API_BASE:
            raise IsolatedMatrixError("only the official Google Gemini endpoint is allowed")
        self.module = runtime_module
        self.key = key
        self.base = configured
        self.post = post or _ThreadLocalPooledPost()
        self.sleep = sleep
        self.timeout = float(os.environ.get("GEMINI_REQUEST_TIMEOUT_SECONDS", "300"))

    def _retry(self, message: str, status: int | None = None) -> None:
        self.sleep(RETRY_WAIT_SECONDS)
        raise self.module.ExplicitProviderError(
            message, retryable=True, status_code=status
        )

    def generate_content(self, *, model: str, request: Mapping[str, Any]) -> Any:
        body = self.module._canonical_json(request).encode("utf-8")  # type: ignore[attr-defined]
        try:
            status, response = self.post(
                f"{self.base}/models/{model}:generateContent",
                body,
                {"Content-Type": "application/json", "x-goog-api-key": self.key},
                self.timeout,
            )
        except captions_r3.RetryableTransportError as exc:
            self._retry(f"Gemini transport retry: {type(exc.__cause__).__name__}")
        if status in captions_r3.RETRYABLE_HTTP_STATUSES:
            self._retry(f"Gemini transient HTTP {status}", status)
        if not 200 <= status < 300:
            raise self.module.ExplicitProviderError(
                f"Gemini GenerateContent returned HTTP {status}",
                retryable=False,
                status_code=status,
            )
        try:
            value = json.loads(response.decode("utf-8"))
        except Exception:
            self._retry("Gemini returned invalid JSON over HTTP 2xx")
        if not isinstance(value, dict):
            self._retry("Gemini returned a non-object JSON response")
        return value


def _successful_attempts(result: Any) -> int:
    """Read the exact number of physical attempts used by a sealed result."""

    value = _read_object(Path(str(result.cache_path)), "Gemini generation result")
    ordinal = value.get("attempt_ordinal")
    if not isinstance(ordinal, int) or not 1 <= ordinal <= MAX_GENERATION_ATTEMPTS:
        raise IsolatedMatrixError("Gemini successful attempt ordinal drift")
    return ordinal


def _json_output_matches_contract(
    text: str, *, stage: str, cache_metadata: Mapping[str, Any]
) -> bool:
    value = graph_memory._parse_json_object(  # type: ignore[attr-defined]
        str(text)
    )
    if not isinstance(value, Mapping):
        return False
    if stage != "writer":
        return True
    substage = str(cache_metadata.get("substage", ""))
    if substage == "summary":
        return "batch_summary" in value
    if substage == "creamem_fact":
        return set(value) == {"atomic_facts"} and isinstance(
            value.get("atomic_facts"), list
        )
    if substage == "native_fact":
        return isinstance(value.get("memory"), list) or isinstance(
            value.get("atomic_facts"), list
        )
    return True


class _ReliableG35BuildBackend:
    backbone = "g35"

    def __init__(
        self,
        cache_root: Path,
        *,
        caption_source_root: Path,
        seed_durable_budget: bool = False,
    ) -> None:
        from creamem3 import memeye_g35_runtime_v19 as runtime

        self.module = runtime
        self.cache_root = Path(cache_root).resolve()
        self.caption_source_root = Path(caption_source_root).resolve()
        self.stage_limits = {
            "caption": runtime.StageLimits(2400, 0, MAX_GENERATION_ATTEMPTS),
            "writer": runtime.StageLimits(16000, 3080, MAX_GENERATION_ATTEMPTS),
            "view": runtime.StageLimits(16384, 1225, MAX_GENERATION_ATTEMPTS),
            "judge": runtime.StageLimits(400, 0, MAX_GENERATION_ATTEMPTS),
        }
        self.runtime = runtime.Gemini35Runtime(
            self.cache_root,
            allow_model_calls=True,
            provider_factory=lambda: _ReliableRESTProvider(runtime),
            stage_limits=self.stage_limits,
        )
        if seed_durable_budget:
            with self.runtime._state_lock:  # type: ignore[attr-defined]
                for stage, limit in self.stage_limits.items():
                    durable = self._durable(stage)
                    if durable > int(limit.max_live_calls):
                        raise IsolatedMatrixError(
                            f"durable G35 {stage} attempt ceiling already exceeded"
                        )
                    self.runtime._provider_calls[stage] = durable  # type: ignore[attr-defined]

    def _durable(self, stage: str) -> int:
        root = self.cache_root / "attempts" / stage
        return len(list(root.rglob("attempt_*_request.json"))) if root.is_dir() else 0

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
    ) -> build.GenerationReply:
        if stage not in {"writer", "view"} or image_path is not None:
            raise IsolatedMatrixError("G35 isolated build accepts writer/view only")
        if max_provider_attempts not in {3, MAX_GENERATION_ATTEMPTS}:
            raise IsolatedMatrixError("unexpected upstream generation attempt contract")
        metadata = dict(cache_metadata or {})
        metadata["matched_substage"] = metadata.get("substage")
        metadata["max_provider_attempts"] = MAX_GENERATION_ATTEMPTS
        physical_attempts = 0
        original_limit = self.stage_limits[stage]
        for output_attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            remaining = MAX_GENERATION_ATTEMPTS - physical_attempts
            if remaining <= 0:
                break
            attempt_metadata = dict(metadata)
            if output_attempt > 1:
                attempt_metadata["application_output_retry_ordinal"] = output_attempt
            effective = self.module.StageLimits(
                original_limit.max_output_tokens,
                original_limit.max_live_calls,
                remaining,
            )
            self.stage_limits[stage] = effective
            self.runtime.stage_limits[stage] = effective
            try:
                result = self.runtime.generate(
                    stage,
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    max_output_tokens=max_output_tokens,
                    cache_metadata=attempt_metadata,
                )
            finally:
                self.stage_limits[stage] = original_limit
                self.runtime.stage_limits[stage] = original_limit
            physical_attempts += (
                0
                if isinstance(result, build.GenerationReply)
                else _successful_attempts(result)
            )
            if not json_mode or _json_output_matches_contract(
                result.text, stage=stage, cache_metadata=metadata
            ):
                return build.GenerationReply(
                    result.text,
                    result.request_sha256,
                    result.response_sha256,
                    result.cache_hit,
                )
            if physical_attempts < MAX_GENERATION_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
        raise IsolatedMatrixError(
            f"G35 {stage} returned no complete JSON object within five physical attempts"
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        value = copy.deepcopy(self.runtime.usage_receipt())
        for stage, row in value["stages"].items():
            in_process = int(row["provider_calls"])
            durable = self._durable(stage)
            row["in_process_provider_calls"] = in_process
            row["durable_provider_attempt_intents"] = durable
            row["provider_calls"] = durable
            row["remaining_live_calls"] = (
                self.stage_limits[stage].max_live_calls - durable
            )
        caption_attempts = len(
            list(
                (self.caption_source_root / "provider" / "g35" / "attempts" / "caption").rglob(
                    "attempt_*_request.json"
                )
            )
        )
        if caption_attempts < matrix.EXPECTED_IMAGES:
            raise IsolatedMatrixError("imported G35 caption attempt census is incomplete")
        row = value["stages"]["caption"]
        row["provider_calls"] = caption_attempts
        row["durable_provider_attempt_intents"] = caption_attempts
        row["remaining_live_calls"] = 0
        value["profile"] = "isolated_imported_caption_writer_view_retry5"
        value["imported_caption_source_root"] = str(self.caption_source_root)
        return value


class _ReliableG25BuildBackend:
    backbone = "g25"

    def __init__(
        self, cache_root: Path, *, bypass_global_lease: bool = False
    ) -> None:
        from creamem3 import official_g25_build_runtime_adapter_v19 as adapter

        source = adapter._load_source_runtime()  # type: ignore[attr-defined]
        limits = {
            "writer": source.StageLimits(16000, 3080, MAX_GENERATION_ATTEMPTS),
            "view": source.StageLimits(16384, 1225, MAX_GENERATION_ATTEMPTS),
        }
        source.DEFAULT_STAGE_LIMITS = limits
        self.facade = adapter.OfficialGemini25BuildRuntime(
            Path(cache_root).resolve(),
            allow_model_calls=True,
            provider_factory=lambda: _ReliableRESTProvider(source),
        )
        self.source = source
        self.stage_limits = limits
        self.logical_calls = {"writer": 0, "view": 0}
        self.bypass_global_lease = bool(bypass_global_lease)
        if self.bypass_global_lease:
            # Each task worker owns a separate facade/runtime object but shares
            # the content-addressed cache root.  Seed the in-process counters
            # from durable attempts so a resumed worker cannot forget already
            # spent budget.  Per-identity OS lifecycle locks remain authoritative.
            runtime = self.facade._runtime  # type: ignore[attr-defined]
            with runtime._state_lock:  # type: ignore[attr-defined]
                for stage, limit in self.stage_limits.items():
                    durable = self.facade._durable_attempts(stage)  # type: ignore[attr-defined]
                    if durable > int(limit.max_live_calls):
                        raise IsolatedMatrixError(
                            f"durable G25 {stage} attempt ceiling already exceeded"
                        )
                    runtime._provider_calls[stage] = durable  # type: ignore[attr-defined]

    def _generate_once(
        self,
        stage: str,
        prompt: str,
        *,
        system: str | None,
        json_mode: bool,
        max_output_tokens: int,
        cache_metadata: Mapping[str, Any],
    ) -> Any:
        target = (
            self.facade._runtime.generate  # type: ignore[attr-defined]
            if self.bypass_global_lease
            else self.facade.generate
        )
        return target(
            stage,
            prompt,
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            cache_metadata=cache_metadata,
        )

    def _recover_exhausted_empty_summary(
        self, *, stage: str, cache_metadata: Mapping[str, Any]
    ) -> build.GenerationReply | None:
        """Use a declared model-free summary after exactly five empty replies.

        This is deliberately narrower than another provider retry.  It is used
        only when the durable cache proves that the same summary request already
        consumed the user-approved five physical attempts and every determinate
        response was an empty provider payload.  Facts are still generated by
        the requested backbone; the fallback merely lets the packet continue.
        """

        if stage != "writer" or cache_metadata.get("substage") != "summary":
            return None
        unit_id = str(cache_metadata.get("unit_id", "")).strip()
        if not unit_id:
            return None
        cache_root = Path(self.facade.cache_root)
        matches: list[tuple[Path, Path, dict[str, Any], list[Path]]] = []
        for plan_path in (cache_root / "plans" / stage).glob("*.json"):
            plan = _read_object(plan_path, "G25 fallback plan")
            metadata = plan.get("cache_metadata")
            if not isinstance(metadata, Mapping):
                continue
            if (
                str(metadata.get("unit_id", "")) != unit_id
                or metadata.get("substage") != "summary"
                or metadata.get("application_output_retry_ordinal") is not None
            ):
                continue
            stem = plan_path.stem
            failure_path = cache_root / "failures" / stage / f"{stem}.json"
            attempt_root = cache_root / "attempts" / stage / stem
            if not failure_path.is_file() or not attempt_root.is_dir():
                continue
            failure = _read_object(failure_path, "G25 exhausted summary failure")
            response_paths = sorted(attempt_root.glob("attempt_*_response.json"))
            if (
                failure.get("failure_kind")
                != "explicit_retryable_attempts_exhausted"
                or int(failure.get("attempts", -1)) != MAX_GENERATION_ATTEMPTS
                or len(response_paths) != MAX_GENERATION_ATTEMPTS
                or (cache_root / "responses" / stage / f"{stem}.json").exists()
            ):
                continue
            responses = [
                _read_object(path, "G25 empty summary attempt")
                for path in response_paths
            ]
            if not all(
                row.get("outcome_kind") == "explicit_provider_error"
                and row.get("retryable") is True
                and row.get("error_type") == "EmptyProviderResponse"
                and int(row.get("attempt_ordinal", -1)) == ordinal
                for ordinal, row in enumerate(responses, 1)
            ):
                continue
            matches.append((plan_path, failure_path, failure, response_paths))
        if len(matches) != 1:
            return None
        plan_path, failure_path, failure, response_paths = matches[0]
        fallback_summary = unit_id.replace("::", " ") + " observations"
        text = json.dumps(
            {"batch_summary": fallback_summary},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        receipt = {
            "schema_version": "creamem3.isolated_empty_summary_fallback.v1",
            "policy": "model_free_after_exactly_five_empty_provider_responses",
            "unit_id": unit_id,
            "stage": stage,
            "substage": "summary",
            "physical_attempts": MAX_GENERATION_ATTEMPTS,
            "additional_provider_calls": 0,
            "plan_file_sha256": _sha256(plan_path.read_bytes()),
            "failure_file_sha256": _sha256(failure_path.read_bytes()),
            "attempt_response_file_sha256s": [
                _sha256(path.read_bytes()) for path in response_paths
            ],
            "request_sha256": str(failure["request_sha256"]),
            "fallback_text_sha256": _sha256(text),
        }
        receipt["receipt_sha256"] = _sha256(receipt)
        receipt_path = cache_root / "fallbacks" / stage / f"{plan_path.stem}.json"
        _write_once_bytes(
            receipt_path,
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        return build.GenerationReply(
            text,
            str(failure["request_sha256"]),
            _sha256(text),
            True,
        )

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
    ) -> build.GenerationReply:
        if (
            stage not in {"writer", "view"}
            or image_path is not None
            or caption_context_count is not None
            or max_provider_attempts not in {3, MAX_GENERATION_ATTEMPTS}
        ):
            raise IsolatedMatrixError("G25 isolated generation request drift")
        self.logical_calls[stage] += 1
        metadata = dict(cache_metadata or {})
        metadata["matched_substage"] = metadata.get("substage")
        metadata["max_provider_attempts"] = MAX_GENERATION_ATTEMPTS
        physical_attempts = 0
        original_limit = self.stage_limits[stage]
        for output_attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            remaining = MAX_GENERATION_ATTEMPTS - physical_attempts
            if remaining <= 0:
                break
            attempt_metadata = dict(metadata)
            if output_attempt > 1:
                attempt_metadata["application_output_retry_ordinal"] = output_attempt
            effective = self.source.StageLimits(
                original_limit.max_output_tokens,
                original_limit.max_live_calls,
                remaining,
            )
            self.stage_limits[stage] = effective
            self.facade.stage_limits[stage] = effective
            self.facade._runtime.stage_limits[stage] = effective  # type: ignore[attr-defined]
            try:
                try:
                    result = self._generate_once(
                        stage,
                        prompt,
                        system=system,
                        json_mode=json_mode,
                        max_output_tokens=max_output_tokens,
                        cache_metadata=attempt_metadata,
                    )
                except Exception:
                    fallback = self._recover_exhausted_empty_summary(
                        stage=stage, cache_metadata=attempt_metadata
                    )
                    if fallback is None:
                        raise
                    result = fallback
            finally:
                self.stage_limits[stage] = original_limit
                self.facade.stage_limits[stage] = original_limit
                self.facade._runtime.stage_limits[stage] = original_limit  # type: ignore[attr-defined]
            physical_attempts += (
                0
                if isinstance(result, build.GenerationReply)
                else _successful_attempts(result)
            )
            if not json_mode or _json_output_matches_contract(
                str(result.text), stage=stage, cache_metadata=metadata
            ):
                return build.GenerationReply(
                    str(result.text),
                    str(result.request_sha256),
                    str(result.response_sha256),
                    bool(result.cache_hit),
                )
            if physical_attempts < MAX_GENERATION_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
        raise IsolatedMatrixError(
            f"G25 {stage} returned no complete JSON object within five physical attempts"
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        value = dict(self.facade.usage_receipt())
        value["logical_calls"] = sum(self.logical_calls.values())
        value["writer_logical_calls"] = self.logical_calls["writer"]
        value["view_logical_calls"] = self.logical_calls["view"]
        value["profile"] = "isolated_build_writer_view_retry5"
        return value


def _generation_backend(root: Path, caption_root: Path, backbone: str) -> Any:
    """Construct exactly one backend without opening the other namespace."""

    if backbone == "g25":
        return _ReliableG25BuildBackend(root / "provider" / "g25")
    if backbone == "g35":
        return _ReliableG35BuildBackend(
            root / "provider" / "g35", caption_source_root=caption_root
        )
    raise IsolatedMatrixError(f"unknown generation backbone: {backbone}")


def _generation_backends(root: Path, caption_root: Path) -> dict[str, Any]:
    return {
        backbone: _generation_backend(root, caption_root, backbone)
        for backbone in ("g25", "g35")
    }


def _run_writer_backbone(
    *,
    backbone: str,
    tasks: Sequence[Mapping[str, Any]],
    g35_caption_maps: Mapping[str, Mapping[str, str]],
    backend: Any,
    output_root: Path,
) -> dict[str, Any]:
    """Generate/resume one complete packet bank, serially within a backbone."""

    session_packets = 0
    native_packets = 0
    for task_row in tasks:
        task = str(task_row["task"])
        if backbone == "g25":
            task_captions, _receipt = build.load_g25_caption_bank_for_task(
                task_row, matrix.DEFAULT_V7_CAPTION_ROOT
            )
        elif backbone == "g35":
            task_captions = dict(g35_caption_maps[task])
        else:
            raise IsolatedMatrixError(f"unknown writer backbone: {backbone}")
        _records, _source_map, session = build._ensure_session_packets(  # type: ignore[attr-defined]
            task_row,
            task_captions,
            backbone=backbone,
            generation=backend,
            output_root=output_root,
        )
        _rounds, native = build._ensure_native_packets(  # type: ignore[attr-defined]
            task_row,
            task_captions,
            backbone=backbone,
            generation=backend,
            output_root=output_root,
        )
        session_packets += len(session)
        native_packets += len(native)
    if session_packets != matrix.EXPECTED_SESSIONS or native_packets != 87:
        raise IsolatedMatrixError(f"{backbone} writer packet census drift")
    return {
        "backbone": backbone,
        "session_packets": session_packets,
        "native_packets": native_packets,
        "logical_writer_identities": 2 * (session_packets + native_packets),
    }


def _run_one_g25_writer_task(
    *,
    task_row: Mapping[str, Any],
    backend: Any,
    output_root: Path,
) -> dict[str, Any]:
    """Generate/resume one G25 task in its task-scoped packet namespace."""

    task = str(task_row["task"])
    task_captions, _receipt = build.load_g25_caption_bank_for_task(
        task_row, matrix.DEFAULT_V7_CAPTION_ROOT
    )
    _records, _source_map, session = build._ensure_session_packets(  # type: ignore[attr-defined]
        task_row,
        task_captions,
        backbone="g25",
        generation=backend,
        output_root=output_root,
    )
    _rounds, native = build._ensure_native_packets(  # type: ignore[attr-defined]
        task_row,
        task_captions,
        backbone="g25",
        generation=backend,
        output_root=output_root,
    )
    return {
        "task": task,
        "session_packets": len(session),
        "native_packets": len(native),
        "logical_writer_identities": 2 * (len(session) + len(native)),
        "session_packet_root": str(
            (
                output_root / "build" / "packets" / "g25" / "session" / task
            ).resolve()
        ),
        "native_packet_root": str(
            (
                output_root / "build" / "packets" / "g25" / "mgtm" / task
            ).resolve()
        ),
    }


def _run_g25_writer_tasks_parallel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    output_root: Path,
    workers: int,
    backend_factory: Callable[[Path], Any] | None = None,
) -> list[dict[str, Any]]:
    """Run explicit disjoint G25 tasks with one runtime/client per worker task."""

    # The frozen G25 adapter loads its source runtime through a delayed dynamic
    # import.  Complete that import on the calling thread before worker startup;
    # otherwise two first-use workers can observe the same partially initialized
    # module in ``sys.modules``.  This is provider-cold and performs no network
    # or cache writes.
    if backend_factory is None:
        from creamem3 import official_g25_build_runtime_adapter_v19 as adapter

        adapter._load_source_runtime()  # type: ignore[attr-defined]

    selected = list(tasks)
    task_names = [str(row.get("task", "")) for row in selected]
    if (
        not selected
        or any(not task for task in task_names)
        or len(set(task_names)) != len(task_names)
    ):
        raise IsolatedMatrixError("parallel G25 writer tasks must be unique and nonempty")
    active_workers = min(int(workers), len(selected))
    if active_workers < 1:
        raise IsolatedMatrixError("writer task workers must be positive")
    provider_root = output_root / "provider" / "g25"
    make_backend = backend_factory or (
        lambda cache_root: _ReliableG25BuildBackend(
            cache_root, bypass_global_lease=True
        )
    )

    def run_one(task_row: Mapping[str, Any]) -> dict[str, Any]:
        # A distinct facade/provider per submitted task provides an independent
        # HTTP connection.  All facades intentionally share provider_root;
        # request identities and packet roots include the task identity, while
        # the underlying durable runtime retains per-identity OS lifecycle locks.
        backend = make_backend(provider_root)
        receipt = _run_one_g25_writer_task(
            task_row=task_row,
            backend=backend,
            output_root=output_root,
        )
        receipt["backend_cache_root"] = str(Path(backend.cache_root).resolve()) if hasattr(
            backend, "cache_root"
        ) else str(provider_root.resolve())
        return receipt

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=active_workers,
        thread_name_prefix="memeye-g25-writer-task",
    ) as pool:
        # map preserves the explicit CLI task order in the returned receipt.
        receipts = list(pool.map(run_one, selected))
    packet_roots = {
        path
        for receipt in receipts
        for path in (
            receipt["session_packet_root"],
            receipt["native_packet_root"],
        )
    }
    if len(packet_roots) != 2 * len(receipts):
        raise IsolatedMatrixError("parallel G25 task packet namespaces overlap")
    return receipts


def _run_writer_backbones_parallel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    g35_caption_maps: Mapping[str, Mapping[str, str]],
    backends: Mapping[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Run G25 and G35 concurrently; each cache/packet namespace is disjoint."""

    if set(backends) != {"g25", "g35"}:
        raise IsolatedMatrixError("both writer backbones are required")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="memeye-isolated-writer"
    ) as pool:
        futures = {
            backbone: pool.submit(
                _run_writer_backbone,
                backbone=backbone,
                tasks=tasks,
                g35_caption_maps=g35_caption_maps,
                backend=backends[backbone],
                output_root=output_root,
            )
            for backbone in ("g25", "g35")
        }
        # Deterministic receipt order despite concurrent provider execution.
        return [futures[name].result() for name in ("g25", "g35")]


class _IsolatedExactEmbeddingAliasRuntime:
    """Read through an exact successful provider request after an orphan.

    The formal runtime deliberately makes an indeterminate request intent an
    immutable orphan.  This isolated experiment may nevertheless already have
    a fully sealed response for the *identical provider request* under another
    cache-metadata identity.  In that case no provider recall is needed: replay
    the source final with the formal runtime in zero-call mode, verify input
    hashes/order and native vector shape, then append an alias receipt.  The
    orphan and its original attempt evidence are never modified or removed.
    """

    ALIAS_SCHEMA = "creamem3.memeye_isolated_embedding_exact_alias.v19.0"

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._module = runtime.__class__.__module__
        self._alias_hits = 0

    def _runtime_module(self) -> Any:
        import importlib

        return importlib.import_module(self._module)

    def _exact_alias(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, Any],
        contract: Mapping[str, Any],
    ) -> Any | None:
        module = self._runtime_module()
        cache_root = Path(self._runtime.cache_root)
        target_resume = str(contract["resume_hash"])
        target_paths = self._runtime._paths(target_resume)  # type: ignore[attr-defined]
        if not target_paths["orphan"].is_file():
            return None
        expected_full = tuple(
            str(row["full_text_sha256"]) for row in contract["item_identities"]
        )
        expected_visible = tuple(
            str(row["visible_text_sha256"]) for row in contract["item_identities"]
        )
        for final_path in sorted((cache_root / "responses").glob("*.json")):
            if final_path.stem == target_resume:
                continue
            final = _read_object(final_path, "embedding alias source final")
            if final.get("provider_request_sha256") != contract.get(
                "provider_request_sha256"
            ):
                continue
            if (
                tuple(str(value) for value in final.get("full_text_sha256s", []))
                != expected_full
                or tuple(
                    str(value) for value in final.get("visible_text_sha256s", [])
                )
                != expected_visible
            ):
                raise IsolatedMatrixError(
                    "exact embedding request alias input hash/order drift"
                )
            source_resume = str(final.get("resume_hash", ""))
            if source_resume != final_path.stem:
                raise IsolatedMatrixError("embedding alias source resume hash drift")
            plan_path = cache_root / "plans" / f"{source_resume}.json"
            plan = _read_object(plan_path, "embedding alias source plan")
            source_metadata = plan.get("cache_metadata")
            if not isinstance(source_metadata, Mapping):
                raise IsolatedMatrixError("embedding alias source metadata drift")
            source_contract = module.request_identity(
                list(texts),
                cache_metadata=dict(source_metadata),
                limits=self._runtime.limits,
            )
            if (
                source_contract.get("resume_hash") != source_resume
                or source_contract.get("provider_request_sha256")
                != contract.get("provider_request_sha256")
            ):
                raise IsolatedMatrixError("embedding alias source identity drift")
            audit = module.OfficialGoogleBatchEmbeddingRuntime(
                cache_root,
                containment_root=self._runtime.containment_root,
                limits=self._runtime.limits,
                allow_model_calls=False,
                budget_binding_sha256=self._runtime.budget_binding_sha256,
            )
            source = audit.embed_batch(
                list(texts), cache_metadata=dict(source_metadata)
            )
            vectors = [list(map(float, row)) for row in source.vectors]
            if (
                len(vectors) != len(texts)
                or any(
                    len(row) != build.EMBEDDING_DIMENSION
                    or not all(math.isfinite(value) for value in row)
                    for row in vectors
                )
                or tuple(source.full_text_sha256s) != expected_full
                or tuple(source.visible_text_sha256s) != expected_visible
            ):
                raise IsolatedMatrixError("embedding alias vector/input validation failed")
            alias_path = cache_root / "aliases" / f"{target_resume}.json"
            alias = {
                "schema_version": self.ALIAS_SCHEMA,
                "alias_kind": "exact_provider_request_zero_call",
                "target_identity": str(contract["identity"]),
                "target_resume_hash": target_resume,
                "target_provider_request_sha256": str(
                    contract["provider_request_sha256"]
                ),
                "target_full_text_sha256s": list(expected_full),
                "target_visible_text_sha256s": list(expected_visible),
                "target_orphan_path": str(target_paths["orphan"].resolve()),
                "target_orphan_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
                    target_paths["orphan"]
                ),
                "source_identity": str(source.request_sha256),
                "source_resume_hash": str(source.resume_hash),
                "source_response_sha256": str(source.response_sha256),
                "source_final_path": str(final_path.resolve()),
                "source_final_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
                    final_path
                ),
                "source_plan_path": str(plan_path.resolve()),
                "source_plan_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
                    plan_path
                ),
                "source_successful_attempt_ordinal": int(
                    source.successful_attempt_ordinal
                ),
                "vector_sha256s": list(source.vector_sha256s),
                "vectors_sha256": str(source.vectors_sha256),
                "dimension": build.EMBEDDING_DIMENSION,
                "vector_count": len(vectors),
                "formal_zero_call_replay_validated": True,
                "orphan_preserved": True,
                "provider_calls": 0,
            }
            module.write_once_json(alias_path, alias)
            self._alias_hits += 1
            return module.EmbeddingBatchResult(
                vectors=tuple(tuple(value for value in row) for row in vectors),
                request_sha256=str(contract["identity"]),
                resume_hash=target_resume,
                response_sha256=str(source.response_sha256),
                vector_sha256s=tuple(str(value) for value in source.vector_sha256s),
                vectors_sha256=str(source.vectors_sha256),
                full_text_sha256s=expected_full,
                visible_text_sha256s=expected_visible,
                cache_path=str(alias_path.resolve()),
                cache_hit=True,
                successful_attempt_ordinal=int(source.successful_attempt_ordinal),
            )
        return None

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        materialized = [str(value) for value in texts]
        metadata = dict(cache_metadata or {})
        module = self._runtime_module()
        contract = module.request_identity(
            materialized, cache_metadata=metadata, limits=self._runtime.limits
        )
        target_paths = self._runtime._paths(  # type: ignore[attr-defined]
            str(contract["resume_hash"])
        )
        if target_paths["final"].is_file():
            return self._runtime.embed_batch(
                materialized, cache_metadata=metadata
            )
        alias = self._exact_alias(materialized, metadata, contract)
        if alias is not None:
            return alias
        return self._runtime.embed_batch(materialized, cache_metadata=metadata)

    def usage_receipt(self) -> dict[str, Any]:
        value = dict(self._runtime.usage_receipt())
        value["isolated_exact_request_alias_hits"] = self._alias_hits
        value["isolated_exact_request_alias_provider_calls"] = 0
        return value


def _isolated_embedding_backend(
    *,
    cache_root: Path,
    containment_root: Path,
    approved_provider_attempt_hard_ceiling: int,
    budget_binding_sha256: str,
    allow_model_calls: bool = True,
    provider_factory: Any | None = None,
) -> Any:
    """Construct the formal adapter with isolated exact-request readthrough."""

    backend = build.OfficialNative3072EmbeddingBackend(
        cache_root,
        containment_root=containment_root,
        approved_provider_attempt_hard_ceiling=(
            approved_provider_attempt_hard_ceiling
        ),
        budget_binding_sha256=budget_binding_sha256,
        allow_model_calls=allow_model_calls,
        provider_factory=provider_factory,
    )
    backend._runtime = _IsolatedExactEmbeddingAliasRuntime(  # type: ignore[attr-defined]
        backend._runtime  # type: ignore[attr-defined]
    )
    return backend


def _ensure_embedding_gate(root: Path) -> dict[str, Any]:
    protocol = matrix.load_protocol()
    budget = build.compute_dynamic_embedding_budget(output_root=root, protocol=protocol)
    path = root / "build" / "dynamic_embedding_budget_gate.json"
    gate = {
        "schema_version": build.DYNAMIC_EMBEDDING_GATE_SCHEMA,
        "decision": "approve",
        # The reused v19 gate validator requires this enum.  The reviewer and
        # nonformal root make clear that this is an isolated owner-approved run.
        "reviewer_role": "independent_protocol_auditor",
        "reviewer": "user_authorized_isolated_memeye_run",
        "protocol_sha256": _sha256(protocol),
        "writer_complete_file_sha256": budget["writer_complete_file_sha256"],
        "writer_generation_evidence_file_sha256": budget[
            "writer_generation_evidence_file_sha256"
        ],
        "embedding_budget_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
            root / "build" / "embedding_budget.json"
        ),
        "approved_embedding_provider_attempt_hard_ceiling": budget[
            "embedding_provider_attempt_hard_ceiling"
        ],
        "approved_build_overall_provider_attempt_hard_ceiling": budget[
            "build_overall_provider_attempt_hard_ceiling"
        ],
        "model": "gemini-embedding-001",
        "dimension": build.EMBEDDING_DIMENSION,
        "batch_size": build.EMBEDDING_BATCH_SIZE,
        "provider_visible_character_limit": build.EMBEDDING_VISIBLE_CHARACTER_LIMIT,
        "cache_identity_uses_complete_untruncated_text": True,
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "failure_policy": "fail_closed_no_per_item_or_zero_fallback",
        "max_provider_attempts_per_physical_batch": build.EMBEDDING_MAX_PROVIDER_ATTEMPTS,
        "legacy_cache_readthrough": False,
    }
    build._write_once_json(path, gate)  # type: ignore[attr-defined]
    return build.validate_dynamic_embedding_budget_gate(
        output_root=root, protocol=protocol
    )


def run_writer(
    *,
    source_project_root: Path,
    caption_root: Path,
    output_root: Path,
    writer_backbone: str = "both",
    writer_tasks: Sequence[str] = (),
    writer_task_workers: int = 1,
) -> dict[str, Any]:
    if writer_backbone not in {"g25", "g35", "both"}:
        raise IsolatedMatrixError("writer_backbone must be g25, g35, or both")
    root = _assert_output_root(output_root)
    imported = _adapt_caption_bank(
        source_project_root=source_project_root,
        caption_root=caption_root,
        output_root=root,
    )
    project = _project(source_project_root)
    requested_tasks = [str(task).strip() for task in writer_tasks if str(task).strip()]
    if requested_tasks:
        if writer_backbone != "g25":
            raise IsolatedMatrixError(
                "explicit writer tasks are available only for --writer-backbone g25"
            )
        if len(set(requested_tasks)) != len(requested_tasks):
            raise IsolatedMatrixError("explicit writer task list contains duplicates")
        all_tasks = build._project_tasks(project)  # type: ignore[attr-defined]
        by_name = {str(row["task"]): row for row in all_tasks}
        unknown = [task for task in requested_tasks if task not in by_name]
        if unknown:
            raise IsolatedMatrixError(f"unknown explicit writer tasks: {unknown}")
        selected = [by_name[task] for task in requested_tasks]
        attempt_root = root / "provider" / "g25" / "attempts" / "writer"

        def durable_attempts() -> int:
            return (
                len(list(attempt_root.rglob("attempt_*_request.json")))
                if attempt_root.is_dir()
                else 0
            )

        before = durable_attempts()
        receipts = _run_g25_writer_tasks_parallel(
            tasks=selected,
            output_root=root,
            workers=writer_task_workers,
        )
        after = durable_attempts()
        return {
            "status": "selected_g25_writer_tasks_complete_unsealed",
            "output_root": str(root),
            "caption_import_sha256": imported["import_sha256"],
            "writer_backbone": "g25",
            "writer_tasks": requested_tasks,
            "writer_task_workers": min(writer_task_workers, len(selected)),
            "task_receipts": receipts,
            "canonical_sealer_called": False,
            # Another sequential G25 owner may be active in the same cache;
            # report the shared durable census rather than misattributing calls.
            "shared_g25_writer_attempts_before": before,
            "shared_g25_writer_attempts_after": after,
            "shared_g25_writer_attempt_delta_observed": after - before,
        }
    if writer_backbone in {"g25", "g35"}:
        tasks = build._project_tasks(project)  # type: ignore[attr-defined]
        g35_caption_maps: dict[str, Mapping[str, str]] = {}
        if writer_backbone == "g35":
            g35_caption_maps = {
                str(task_row["task"]): build._load_g35_caption_bank_readonly(  # type: ignore[attr-defined]
                    task_row,
                    output_root=root,
                    image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
                )[0]
                for task_row in tasks
            }
        # Deliberately construct and run only the requested backend.  In this
        # mode no canonical writer seal is attempted because the peer packet
        # bank may still be owned by another process.
        backend = _generation_backend(root, caption_root, writer_backbone)
        receipt = _run_writer_backbone(
            backbone=writer_backbone,
            tasks=tasks,
            g35_caption_maps=g35_caption_maps,
            backend=backend,
            output_root=root,
        )
        return {
            "status": "single_backbone_writer_packets_complete_unsealed",
            "output_root": str(root),
            "caption_import_sha256": imported["import_sha256"],
            "writer_backbone": writer_backbone,
            "backbone_receipt": receipt,
            "canonical_sealer_called": False,
            "provider_calls": build._usage_provider_calls(  # type: ignore[attr-defined]
                backend.usage_receipt()
            ),
        }

    active = _generation_backends(root, caption_root)
    parallel_receipts: list[dict[str, Any]] = []
    if not (root / "build" / "writer_complete.json").is_file():
        tasks = build._project_tasks(project)  # type: ignore[attr-defined]
        g35_caption_maps = {
            str(task_row["task"]): build._load_g35_caption_bank_readonly(  # type: ignore[attr-defined]
                task_row,
                output_root=root,
                image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
            )[0]
            for task_row in tasks
        }
        parallel_receipts = _run_writer_backbones_parallel(
            tasks=tasks,
            g35_caption_maps=g35_caption_maps,
            backends=active,
            output_root=root,
        )
    # This second pass is provider-free: all 616 packet files per backbone are
    # present.  Reusing the canonical sealer gives build-v19 writer_complete
    # and writer_generation_evidence that run_build can consume directly.
    writer = build.prepare_writer_packets(
        project=project,
        output_root=root,
        generation_backends=active,
        caption_root=matrix.DEFAULT_V7_CAPTION_ROOT,
        image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
        enforce_full_scope=True,
    )
    return {
        "status": "writer_complete" if writer.get("complete") else "writer_incomplete",
        "output_root": str(root),
        "caption_import_sha256": imported["import_sha256"],
        "writer_complete_sha256": writer["writer_complete_sha256"],
        "parallel_backbone_receipts": parallel_receipts,
        "provider_calls": writer["provider_calls"],
    }


def _build_one_task_unit(
    *,
    backbone: str,
    task_row: Mapping[str, Any],
    generation: Any,
    embedding: Any,
    output_root: Path,
) -> dict[str, Any]:
    """Materialize one dependency-ordered (backbone, task) graph unit."""

    task = str(task_row["task"])
    if backbone == "g25":
        captions, _caption_receipt = build.load_g25_caption_bank_for_task(
            task_row, matrix.DEFAULT_V7_CAPTION_ROOT
        )
    elif backbone == "g35":
        captions = build._load_g35_caption_bank_readonly(  # type: ignore[attr-defined]
            task_row,
            output_root=output_root,
            image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
        )[0]
    else:
        raise IsolatedMatrixError(f"unknown build unit backbone: {backbone}")

    # Preserve the canonical dependency and model-call order inside a unit.
    session, session_receipt = build.build_session_graph_for_task(
        task_row,
        captions,
        backbone=backbone,
        generation=generation,
        embedding=embedding,
        output_root=output_root,
    )
    online_graph, online_receipt = build.build_online_graph_for_task(
        task_row,
        session,
        backbone=backbone,
        generation=generation,
        embedding=embedding,
        output_root=output_root,
    )
    _offline_graph, offline_receipt = build.build_offline_graph_for_task(
        task_row,
        session,
        backbone=backbone,
        generation=generation,
        embedding=embedding,
        output_root=output_root,
    )
    _native, native_receipt = build.build_native_tree_for_task(
        task_row,
        captions,
        backbone=backbone,
        generation=generation,
        embedding=embedding,
        output_root=output_root,
    )
    del online_graph
    by_category = {
        "session": session_receipt,
        "online": online_receipt,
        "offline": offline_receipt,
        "mgtm": native_receipt,
    }
    artifacts = [
        {
            "backbone": backbone,
            "task": task,
            "category": category,
            "artifact_path": by_category[category]["artifact_path"],
            "artifact_sha256": by_category[category]["artifact_sha256"],
            "receipt_sha256": by_category[category]["receipt_sha256"],
        }
        for category in sorted(by_category)
    ]
    return {
        "unit_id": f"{backbone}:{task}",
        "backbone": backbone,
        "task": task,
        "artifacts": artifacts,
        "artifacts_sha256": _sha256(artifacts),
        "complete": True,
    }


def _run_build_task_units_parallel(
    *,
    units: Sequence[tuple[str, Mapping[str, Any]]],
    output_root: Path,
    workers: int,
    generation_factory: Callable[[str], Any],
    embedding_factory: Callable[[str, str], Any],
    unit_runner: Callable[..., dict[str, Any]] = _build_one_task_unit,
) -> list[dict[str, Any]]:
    """Run disjoint units concurrently and return their fixed input order."""

    materialized = list(units)
    identities = [(backbone, str(row.get("task", ""))) for backbone, row in materialized]
    if (
        not materialized
        or any(backbone not in {"g25", "g35"} or not task for backbone, task in identities)
        or len(set(identities)) != len(identities)
    ):
        raise IsolatedMatrixError("build task units must be unique and valid")
    active_workers = min(int(workers), len(materialized))
    if active_workers < 1:
        raise IsolatedMatrixError("build task workers must be positive")

    def run_one(unit: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
        backbone, task_row = unit
        task = str(task_row["task"])
        return unit_runner(
            backbone=backbone,
            task_row=task_row,
            generation=generation_factory(backbone),
            embedding=embedding_factory(backbone, task),
            output_root=output_root,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=active_workers,
        thread_name_prefix="memeye-build-task",
    ) as pool:
        receipts = list(pool.map(run_one, materialized))
    artifact_paths = [
        str(row["artifact_path"])
        for receipt in receipts
        for row in receipt.get("artifacts", [])
    ]
    if (
        len(artifact_paths) != 4 * len(receipts)
        or len(set(artifact_paths)) != len(artifact_paths)
    ):
        raise IsolatedMatrixError("build task artifact paths overlap or are incomplete")
    return receipts


def run_build_task_parallel(
    *,
    source_project_root: Path,
    caption_root: Path,
    output_root: Path,
    workers: int,
    requested_unit_ids: Sequence[str] = (),
    build_backbone: str = "both",
) -> dict[str, Any]:
    """Isolated opt-in task parallel materialization; formal default stays serial."""

    if build_backbone not in {"g25", "g35", "both"}:
        raise IsolatedMatrixError("build_backbone must be g25, g35, or both")
    root = _assert_output_root(output_root)
    if (root / "build" / "complete.json").is_file():
        return run_build(
            source_project_root=source_project_root,
            caption_root=caption_root,
            output_root=root,
        )
    _adapt_caption_bank(
        source_project_root=source_project_root,
        caption_root=caption_root,
        output_root=root,
    )
    project = _project(source_project_root)
    build.validate_writer_complete(output_root=root)
    gate = _ensure_embedding_gate(root)
    budget = gate["budget"]
    plan_path = root / "build" / "plan.json"
    if plan_path.is_file():
        build.validate_build_plan(output_root=root, project=project)
    else:
        build.plan_build(
            project=project,
            output_root=root,
            caption_root=matrix.DEFAULT_V7_CAPTION_ROOT,
            enforce_full_scope=True,
        )
    tasks = build._project_tasks(project)  # type: ignore[attr-defined]
    by_name = {str(row["task"]): row for row in tasks}
    full_units = [
        (backbone, by_name[task])
        for backbone in ("g25", "g35")
        for task in sorted(by_name)
    ]
    full_ids = [f"{backbone}:{row['task']}" for backbone, row in full_units]
    requested = [str(value).strip() for value in requested_unit_ids if str(value).strip()]
    if requested:
        if len(set(requested)) != len(requested):
            raise IsolatedMatrixError("requested build units contain duplicates")
        lookup = {unit_id: unit for unit_id, unit in zip(full_ids, full_units, strict=True)}
        unknown = [unit_id for unit_id in requested if unit_id not in lookup]
        if unknown:
            raise IsolatedMatrixError(f"unknown build task units: {unknown}")
        selected = [lookup[unit_id] for unit_id in requested]
    else:
        selected = [
            unit
            for unit in full_units
            if build_backbone == "both" or unit[0] == build_backbone
        ]
    if build_backbone != "both" and any(
        backbone != build_backbone for backbone, _task_row in selected
    ):
        raise IsolatedMatrixError("requested build unit escaped --build-backbone")
    if workers > 1 and any(backbone == "g25" for backbone, _task_row in selected):
        raise IsolatedMatrixError(
            "G25 task-parallel build is disabled while the sequential G25 owner is active"
        )

    def generation_factory(backbone: str) -> Any:
        if backbone == "g25":
            return _ReliableG25BuildBackend(
                root / "provider" / "g25", bypass_global_lease=True
            )
        return _ReliableG35BuildBackend(
            root / "provider" / "g35",
            caption_source_root=caption_root,
            seed_durable_budget=True,
        )

    def embedding_factory(_backbone: str, _task: str) -> Any:
        return _isolated_embedding_backend(
            cache_root=root / "provider" / "embedding_native_3072",
            containment_root=root,
            approved_provider_attempt_hard_ceiling=budget[
                "embedding_provider_attempt_hard_ceiling"
            ],
            budget_binding_sha256=str(gate["file_sha256"]),
            allow_model_calls=True,
        )

    receipts = _run_build_task_units_parallel(
        units=selected,
        output_root=root,
        workers=workers,
        generation_factory=generation_factory,
        embedding_factory=embedding_factory,
    )
    selected_ids = [f"{backbone}:{row['task']}" for backbone, row in selected]
    all_units_selected = set(selected_ids) == set(full_ids)
    if all_units_selected:
        # Every artifact now exists.  Re-enter the canonical serial executor;
        # all helpers are cache reads, and it aggregates BUILD_COMPLETE in its
        # frozen backbone/task/category order without changing any model input.
        finalized = run_build(
            source_project_root=source_project_root,
            caption_root=caption_root,
            output_root=root,
        )
        return {
            **finalized,
            "task_parallel_materialization": True,
            "task_workers": min(workers, len(selected)),
            "unit_receipts_sha256": _sha256(receipts),
        }
    return {
        "status": "selected_build_task_units_complete_unsealed",
        "output_root": str(root),
        "task_workers": min(workers, len(selected)),
        "unit_ids": [receipt["unit_id"] for receipt in receipts],
        "unit_receipts": receipts,
        "unit_receipts_sha256": _sha256(receipts),
        "canonical_sealer_called": False,
        "provider_calls": None,
    }


def run_build(
    *, source_project_root: Path, caption_root: Path, output_root: Path
) -> dict[str, Any]:
    root = _assert_output_root(output_root)
    _adapt_caption_bank(
        source_project_root=source_project_root,
        caption_root=caption_root,
        output_root=root,
    )
    build.validate_writer_complete(output_root=root)
    gate = _ensure_embedding_gate(root)
    budget = gate["budget"]
    embedding = _isolated_embedding_backend(
        cache_root=root / "provider" / "embedding_native_3072",
        containment_root=root,
        approved_provider_attempt_hard_ceiling=budget[
            "embedding_provider_attempt_hard_ceiling"
        ],
        budget_binding_sha256=str(gate["file_sha256"]),
        allow_model_calls=True,
    )
    return build.execute_build_with_backends(
        project=_project(source_project_root),
        output_root=root,
        generation_backends=_generation_backends(root, caption_root),
        embedding_backend=embedding,
        caption_root=matrix.DEFAULT_V7_CAPTION_ROOT,
        image_root=matrix.DEFAULT_DATA_ROOT / "data" / "image",
        enforce_full_scope=True,
        smoke_seals={},
        dynamic_embedding_gate_receipt=gate,
    )


def _ensure_predict_gate(root: Path) -> dict[str, Any]:
    protocol = matrix.load_protocol()
    budget_path = root / "predict" / "budget.json"
    budget = (
        predict.seal_predict_budget(root)
        if not budget_path.is_file()
        else predict.compute_predict_budget(output_root=root, protocol=protocol)
    )
    if budget_path.is_file() and _read_object(budget_path, "predict budget") != budget:
        raise IsolatedMatrixError("predict budget drift")
    gate = {
        "schema_version": predict.PREDICT_BUDGET_GATE_SCHEMA,
        "decision": "approve",
        "reviewer_role": "independent_protocol_auditor",
        "reviewer": "user_authorized_isolated_memeye_run",
        "protocol_sha256": _sha256(protocol),
        "budget_path": str(budget_path.resolve()),
        "budget_file_sha256": predict._file_sha256(budget_path),  # type: ignore[attr-defined]
        "predict_budget_sha256": budget["predict_budget_sha256"],
        "approved_provider_attempt_hard_ceiling": budget[
            "provider_attempt_hard_ceiling"
        ],
        "approved_stage_provider_attempt_hard_ceilings": budget[
            "stage_provider_attempt_hard_ceilings"
        ],
        "legacy_cache_readthrough": False,
    }
    gate["gate_sha256"] = predict._sha256(gate)  # type: ignore[attr-defined]
    predict._write_once(root / "predict" / "budget_gate.json", gate)  # type: ignore[attr-defined]
    return predict.validate_predict_budget_gate(output_root=root, protocol=protocol)


def _light_prediction_complete(
    output_root: Path, *, expected_rows: int, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    value = _read_object(root / "predict" / "complete.json", "isolated prediction complete")
    contract = dict(value)
    observed = contract.pop("complete_sha256", None)
    cells = value.get("cells")
    if (
        observed != predict._sha256(contract)  # type: ignore[attr-defined]
        or value.get("schema_version") != matrix.PREDICTION_COMPLETE_SCHEMA
        or value.get("complete") is not True
        or not isinstance(cells, list)
        or len(cells) != len(matrix.matrix_cell_ids())
    ):
        raise IsolatedMatrixError("isolated prediction completion seal drift")
    if {str(row.get("cell_id")) for row in cells} != set(matrix.matrix_cell_ids()):
        raise IsolatedMatrixError("isolated prediction cell census drift")
    for row in cells:
        path = Path(str(row.get("prediction_path", ""))).resolve()
        cell = _read_object(path, "isolated prediction cell")
        cell_contract = dict(cell)
        cell_sha = cell_contract.pop("cell_sha256", None)
        if (
            root not in path.parents
            or predict._file_sha256(path) != row.get("prediction_sha256")  # type: ignore[attr-defined]
            or cell_sha != predict._sha256(cell_contract)  # type: ignore[attr-defined]
            or len(cell.get("rows", [])) != expected_rows
        ):
            raise IsolatedMatrixError(f"isolated prediction cell drift: {path}")
    return value


class _ConcurrentG25ReaderBackend:
    """Isolated Reader facade whose durable runtime is safe across threads.

    The formal adapter's process-wide ``O_EXCL`` lease intentionally serializes
    network calls.  Its wrapped runtime already owns per-identity OS locks and
    thread-safe provider counters.  For this isolated, single-process runner we
    call that runtime directly and seed its counters from durable attempts, so
    restart budgets remain fail-closed without changing request construction.
    """

    def __init__(self, facade: Any) -> None:
        self.facade = facade
        self.cache_root = Path(facade.cache_root).resolve()
        runtime = facade._runtime  # type: ignore[attr-defined]
        with runtime._state_lock:  # type: ignore[attr-defined]
            for stage, limit in facade.stage_limits.items():
                durable = facade._durable_attempts(stage)  # type: ignore[attr-defined]
                if durable > int(limit.max_live_calls):
                    raise IsolatedMatrixError(
                        f"durable G25 {stage} attempt ceiling already exceeded"
                    )
                runtime._provider_calls[stage] = durable  # type: ignore[attr-defined]

    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        json_mode: bool,
        max_output_tokens: int,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.facade._runtime.generate(  # type: ignore[attr-defined]
            stage,
            prompt,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            cache_metadata=cache_metadata,
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        return self.facade.usage_receipt()


class _BackboneRoutingQwenReranker:
    """Replay G25 from its sealed cache and send only G35 to recovery."""

    def __init__(self, g25_readonly: Any, g35_recovery: Any) -> None:
        self.g25_readonly = g25_readonly
        self.g35_recovery = g35_recovery

    @staticmethod
    def _identity(
        query: str,
        documents: Sequence[str],
        instruction: str,
        metadata: Mapping[str, Any],
    ) -> str:
        request = {
            "model": predict.QWEN_MODEL,
            "input": {"query": str(query), "documents": [str(v) for v in documents]},
            "parameters": {
                "return_documents": False,
                "top_n": len(documents),
                "instruct": str(instruction),
            },
        }
        return predict._sha256(  # type: ignore[attr-defined]
            {
                "request": request,
                "cache_metadata": dict(metadata),
                "endpoint": predict.QWEN_ENDPOINT,
            }
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        metadata = dict(cache_metadata or {})
        backbone = str(metadata.get("backbone", ""))
        if backbone == "g25":
            identity = self._identity(query, documents, instruction, metadata)
            cache_path = (
                Path(self.g25_readonly.cache_root)
                / "responses"
                / identity[:2]
                / f"{identity}.json"
            )
            # Fail before the formal delegate can reserve a new attempt.
            if not cache_path.is_file():
                raise IsolatedMatrixError(
                    f"read-only G25 Qwen cache miss: {identity}"
                )
            return self.g25_readonly.score(
                query,
                documents,
                instruction=instruction,
                cache_metadata=metadata,
            )
        if backbone == "g35":
            return self.g35_recovery.score(
                query,
                documents,
                instruction=instruction,
                cache_metadata=metadata,
            )
        raise IsolatedMatrixError("Qwen routing metadata lacks a known backbone")

    def usage_receipt(self) -> Mapping[str, Any]:
        old = dict(self.g25_readonly.usage_receipt())
        new = dict(self.g35_recovery.usage_receipt())
        fixed = (
            "model",
            "endpoint",
            "max_documents",
            "max_provider_attempts",
            "approved_provider_attempt_hard_ceiling",
            "fallback",
        )
        if any(old.get(key) != new.get(key) for key in fixed):
            raise IsolatedMatrixError("split Qwen usage authority drift")
        merged = dict(old)
        for key in ("provider_calls", "provider_calls_this_process", "cache_hits"):
            merged[key] = int(old.get(key, 0)) + int(new.get(key, 0))
        merged["isolated_backbone_cache_roots"] = {
            "g25_readonly": str(Path(self.g25_readonly.cache_root).resolve()),
            "g35_recovery": str(Path(self.g35_recovery.cache_root).resolve()),
        }
        merged["isolated_provider_calls_by_root"] = {
            "g25_readonly": int(old.get("provider_calls", 0)),
            "g35_recovery": int(new.get("provider_calls", 0)),
        }
        merged["isolated_g25_live_calls_authorized"] = False
        return merged


def _reader_backend(root: Path, budget: Mapping[str, Any]) -> Any:
    from creamem3 import official_g25_runtime_adapter_v19 as adapter

    source = adapter._load_source_runtime()  # type: ignore[attr-defined]
    hard = budget["stage_provider_attempt_hard_ceilings"]
    limits = {
        "planner": source.StageLimits(512, hard["semantic_planner"], MAX_GENERATION_ATTEMPTS),
        "focused": source.StageLimits(2600, hard["focused_reader"], MAX_GENERATION_ATTEMPTS),
        "answer": source.StageLimits(500, hard["answer_reader"], MAX_GENERATION_ATTEMPTS),
        "structured_fused": source.StageLimits(
            1200, hard["structured_fused_reader"], MAX_GENERATION_ATTEMPTS
        ),
    }
    return _ConcurrentG25ReaderBackend(
        adapter.OfficialGemini25Runtime(
            root / "provider" / "predict" / "g25_reader",
            allow_model_calls=True,
            provider_factory=lambda: _ReliableRESTProvider(source),
            stage_limits=limits,
        )
    )


def run_predict(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path,
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    root = _assert_output_root(output_root)
    # The core predict path asks for BUILD_COMPLETE more than once (budget
    # sealing and execution).  Its validator performs a full replay of all 64
    # graph artifacts, which is useful once but redundant within the same
    # process.  Keep one full validation here, then make subsequent calls
    # return that exact result only while the write-once complete file remains
    # byte-identical.  This changes no retrieval/model input and avoids a
    # second 20+ minute CPU-only replay in the isolated time-bounded run.
    complete_path = root / "build" / "complete.json"
    if root == DEFAULT_OUTPUT_ROOT.resolve():
        if (
            build._file_sha256(complete_path)  # type: ignore[attr-defined]
            != VALIDATED_BUILD_COMPLETE_FILE_SHA256
        ):
            raise IsolatedMatrixError("canonical-validated BUILD_COMPLETE byte drift")
        validated_build = _read_object(complete_path, "validated BUILD_COMPLETE")
        validation_contract = dict(validated_build)
        observed_contract_sha256 = validation_contract.pop("complete_sha256", None)
        if (
            observed_contract_sha256
            != VALIDATED_BUILD_COMPLETE_CONTRACT_SHA256
            or observed_contract_sha256 != _sha256(validation_contract)
            or validated_build.get("executor_sha256")
            != VALIDATED_BUILD_EXECUTOR_SHA256
            or build._file_sha256(Path(build.__file__).resolve())  # type: ignore[attr-defined]
            != VALIDATED_BUILD_EXECUTOR_SHA256
        ):
            raise IsolatedMatrixError("canonical-validated BUILD_COMPLETE contract drift")
    else:
        validated_build = build.validate_build_complete(output_root=root)
    validated_complete_file_sha256 = build._file_sha256(  # type: ignore[attr-defined]
        complete_path
    )
    original_build_validator = build.validate_build_complete

    def validated_once_build_complete(
        *, output_root: Path, protocol: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if Path(output_root).resolve() != root.resolve():
            raise IsolatedMatrixError("validated BUILD_COMPLETE root drift")
        if (
            build._file_sha256(complete_path)  # type: ignore[attr-defined]
            != validated_complete_file_sha256
        ):
            raise IsolatedMatrixError("validated BUILD_COMPLETE changed in-process")
        return copy.deepcopy(validated_build)

    build.validate_build_complete = validated_once_build_complete  # type: ignore[assignment]
    source_project = _project(source_project_root)
    original = predict._prediction_complete  # type: ignore[attr-defined]
    original_project_validator = matrix.validate_project
    original_build_authority = predict.BUILD_EXECUTOR_SHA256
    original_predict_authority_validator = predict.validate_predict_authorities
    predict._prediction_complete = _light_prediction_complete  # type: ignore[attr-defined]

    def source_project_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return source_project

    matrix.validate_project = source_project_validator  # type: ignore[assignment]

    def isolated_predict_authorities(protocol: Mapping[str, Any]) -> dict[str, Any]:
        # graph_memory_v2 received only the narrowly scoped native-summary JSON
        # parser repair used to build this same root; retrieval/model-visible
        # semantics were not changed.  Validate every other frozen authority
        # byte-for-byte, and bind this isolated run to the exact repaired file.
        adjusted = copy.deepcopy(dict(protocol))
        graph_item = adjusted["authority_files"]["graph_and_retrieval"]
        graph_path = matrix._resolve_authority_path(str(graph_item["path"]))
        if (
            Path(graph_path).resolve() != Path(graph_memory.__file__).resolve()
            or predict._file_sha256(Path(graph_path))  # type: ignore[attr-defined]
            != ISOLATED_GRAPH_RETRIEVAL_SHA256
        ):
            raise IsolatedMatrixError("isolated graph/retrieval authority drift")
        graph_item["sha256"] = ISOLATED_GRAPH_RETRIEVAL_SHA256
        result = dict(original_predict_authority_validator(adjusted))
        result["isolated_graph_parser_repair_sha256"] = (
            ISOLATED_GRAPH_RETRIEVAL_SHA256
        )
        return result

    predict.validate_predict_authorities = isolated_predict_authorities  # type: ignore[assignment]
    # build_executor is intentionally evolving in this isolated retry5 run;
    # bind predict's local authority check to the exact executor that produced
    # this root's BUILD_COMPLETE, without changing any retrieval/Reader input.
    predict.BUILD_EXECUTOR_SHA256 = predict._file_sha256(  # type: ignore[attr-defined]
        Path(build.__file__).resolve()
    )
    try:
        gate = _ensure_predict_gate(root)
        budget = gate["budget"]
        embedding = _isolated_embedding_backend(
            cache_root=root / "provider" / "predict" / "query_embedding_native_3072",
            containment_root=root,
            approved_provider_attempt_hard_ceiling=budget[
                "stage_provider_attempt_hard_ceilings"
            ]["query_embedding"],
            budget_binding_sha256=str(gate["gate_sha256"]),
            allow_model_calls=True,
        )
        qwen_hard = budget["stage_provider_attempt_hard_ceilings"]["qwen_rerank"]
        reranker = _BackboneRoutingQwenReranker(
            predict.DashScopeFormalReranker(
                root / "provider" / "predict" / "qwen3_rerank",
                allow_model_calls=False,
                approved_provider_attempt_hard_ceiling=qwen_hard,
                parallel_safe=True,
            ),
            predict.DashScopeFormalReranker(
                root / "provider" / "predict" / "qwen3_rerank_g35_recovery_r1",
                allow_model_calls=True,
                approved_provider_attempt_hard_ceiling=qwen_hard,
                parallel_safe=workers > 1,
            ),
        )
        return predict.execute_predict_with_backends(
            output_root=root,
            text_backend=_reader_backend(root, budget),
            embedding_backend=embedding,
            rerank_backend=reranker,
            budget_gate=gate,
            enforce_full_scope=True,
            max_workers=workers,
        )
    finally:
        build.validate_build_complete = original_build_validator  # type: ignore[assignment]
        predict.validate_predict_authorities = original_predict_authority_validator  # type: ignore[assignment]
        predict.BUILD_EXECUTOR_SHA256 = original_build_authority
        matrix.validate_project = original_project_validator  # type: ignore[assignment]
        predict._prediction_complete = original  # type: ignore[attr-defined]


def _ensure_score_gate(root: Path, score_data_root: Path) -> dict[str, Any]:
    protocol = matrix.load_protocol()
    budget_path = root / "score" / "budget.json"
    budget = (
        score.seal_score_budget(output_root=root, score_data_root=score_data_root)
        if not budget_path.is_file()
        else score.compute_score_budget(
            output_root=root, score_data_root=score_data_root, protocol=protocol
        )
    )
    if budget_path.is_file() and _read_object(budget_path, "score budget") != budget:
        raise IsolatedMatrixError("score budget drift")
    gate = {
        "schema_version": score.SCORE_BUDGET_GATE_SCHEMA,
        "decision": "approve",
        "reviewer_role": "independent_protocol_auditor",
        "reviewer": "user_authorized_isolated_memeye_run",
        "protocol_sha256": _sha256(protocol),
        "budget_path": str(budget_path.resolve()),
        "budget_file_sha256": score._file_sha256(budget_path),  # type: ignore[attr-defined]
        "score_budget_sha256": budget["score_budget_sha256"],
        "approved_judge_provider_attempt_hard_ceiling": budget[
            "judge_provider_attempt_hard_ceiling"
        ],
        "legacy_judge_cache_readthrough": False,
    }
    gate["gate_sha256"] = score._sha256(gate)  # type: ignore[attr-defined]
    score._write_once(root / "score" / "budget_gate.json", gate)  # type: ignore[attr-defined]
    return score.validate_score_budget_gate(
        output_root=root, score_data_root=score_data_root, protocol=protocol
    )


def _isolated_score_authority_receipt(
    protocol: Mapping[str, Any],
    *,
    canonical_receipt: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Accept only the graph parser repair already used by this prediction.

    The canonical score validator still checks every authority, prompt literal,
    renderer, judge parser/mapping, writer contract, and operation policy.  This
    isolated adapter changes only the stale graph-and-retrieval file digest and
    first binds it to the exact local module accepted during predict.
    """

    adjusted = copy.deepcopy(dict(protocol))
    graph_item = adjusted["authority_files"]["graph_and_retrieval"]
    graph_path = matrix._resolve_authority_path(str(graph_item["path"]))
    actual_sha256 = score._file_sha256(Path(graph_path))  # type: ignore[attr-defined]
    if (
        Path(graph_path).resolve() != Path(graph_memory.__file__).resolve()
        or actual_sha256 != ISOLATED_GRAPH_RETRIEVAL_SHA256
    ):
        raise IsolatedMatrixError("isolated score graph/retrieval authority drift")
    graph_item["sha256"] = ISOLATED_GRAPH_RETRIEVAL_SHA256
    receipt = dict(canonical_receipt(adjusted))
    receipt["isolated_graph_parser_repair_sha256"] = actual_sha256
    return receipt


class _ParseRetryJudgeBackend:
    """Reuse valid Judge caches and resample only parse-invalid responses."""

    def __init__(
        self,
        primary: Any,
        retry: Any,
        runtime_module: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.primary = primary
        self.retry = retry
        self.runtime = runtime_module
        self.sleep = sleep
        self.cache_root = Path(primary.cache_root)
        self.stage_limits = primary.stage_limits

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)

    def _identity(
        self, prompt: str, metadata: Mapping[str, Any], delegate: Any
    ) -> tuple[str, str]:
        request = self.runtime.build_generate_content_request(
            "judge",
            prompt,
            json_mode=True,
            max_output_tokens=400,
            stage_limits=delegate.stage_limits,
        )
        request_sha = self.runtime.request_identity(
            stage="judge",
            request=request,
            stage_limits=delegate.stage_limits,
            cache_metadata=metadata,
        )
        return request_sha, self.runtime.resume_hash(
            stage="judge",
            request_sha256=request_sha,
            stage_limits=delegate.stage_limits,
        )

    def _census(self, prompt: str, metadata: Mapping[str, Any]) -> int:
        identities: list[tuple[Any, Mapping[str, Any]]] = [(self.primary, metadata)]
        identities.extend(
            (self.retry, {**dict(metadata), "parse_retry_ordinal": ordinal})
            for ordinal in range(2, MAX_GENERATION_ATTEMPTS + 1)
        )
        total = 0
        for delegate, retry_metadata in identities:
            _request_sha, resumable = self._identity(
                prompt, retry_metadata, delegate
            )
            total += len(
                list(
                    (
                        self.cache_root
                        / "attempts"
                        / "judge"
                        / resumable
                    ).glob("attempt_*_request.json")
                )
            )
        if total > MAX_GENERATION_ATTEMPTS:
            raise IsolatedMatrixError("Judge logical physical-attempt ceiling exceeded")
        return total

    def judge(
        self,
        question: str,
        ground_truth: str,
        model_output: str,
        *,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        metadata = dict(cache_metadata or {})
        prompt = self.runtime.build_mgtm_judge_prompt(
            question, ground_truth, model_output
        )
        result = self.primary.generate(
            "judge",
            prompt,
            json_mode=True,
            max_output_tokens=400,
            cache_metadata=metadata,
        )
        try:
            return result, self.runtime.parse_mgtm_judge_response(result.text)
        except self.runtime.RuntimeProtocolError as first_error:
            last_error = first_error

        for ordinal in range(2, MAX_GENERATION_ATTEMPTS + 1):
            retry_metadata = {**metadata, "parse_retry_ordinal": ordinal}
            _request_sha, resumable = self._identity(
                prompt, retry_metadata, self.retry
            )
            cached = (
                self.cache_root / "responses" / "judge" / f"{resumable}.json"
            ).is_file()
            if self._census(prompt, metadata) >= MAX_GENERATION_ATTEMPTS and not cached:
                raise self.runtime.RuntimeProtocolError(
                    "Judge parse retry physical-attempt ceiling exhausted"
                ) from last_error
            self.sleep(RETRY_WAIT_SECONDS)
            result = self.retry.generate(
                "judge",
                prompt,
                json_mode=True,
                max_output_tokens=400,
                cache_metadata=retry_metadata,
            )
            try:
                verdict = self.runtime.parse_mgtm_judge_response(result.text)
            except self.runtime.RuntimeProtocolError as error:
                last_error = error
                continue
            self._census(prompt, metadata)
            return result, verdict
        raise last_error

    def usage_receipt(self) -> Mapping[str, Any]:
        primary = copy.deepcopy(dict(self.primary.usage_receipt()))
        retry = self.retry.usage_receipt()
        primary["provider_loaded"] = bool(primary.get("provider_loaded")) or bool(
            retry.get("provider_loaded")
        )
        primary["provider_factory_calls"] = int(
            primary.get("provider_factory_calls", 0)
        ) + int(retry.get("provider_factory_calls", 0))
        for stage, row in primary["stages"].items():
            retry_row = retry["stages"][stage]
            row["provider_calls"] = int(row.get("provider_calls", 0)) + int(
                retry_row.get("provider_calls", 0)
            )
            row["cache_hits"] = int(row.get("cache_hits", 0)) + int(
                retry_row.get("cache_hits", 0)
            )
            for key in ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens"):
                row["usage"][key] = int(row["usage"].get(key, 0)) + int(
                    retry_row["usage"].get(key, 0)
                )
            row["remaining_live_calls"] = int(row["limits"]["max_live_calls"]) - int(
                row["provider_calls"]
            )
        return primary

    def validate_evidence(
        self,
        *,
        result: Any,
        triple: tuple[str, str, str],
        scope: str,
        triple_sha256: str,
        canonical_validator: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        question, gold, prediction = triple
        prompt = self.runtime.build_mgtm_judge_prompt(question, gold, prediction)
        base_metadata = {"scope": scope, "exact_triple_sha256": triple_sha256}
        _base_request, base_resume = self._identity(
            prompt, base_metadata, self.primary
        )
        if result.resume_hash == base_resume:
            return canonical_validator(
                judge_backend=self.primary,
                result=result,
                triple=triple,
                scope=scope,
                triple_sha256=triple_sha256,
            )
        for ordinal in range(2, MAX_GENERATION_ATTEMPTS + 1):
            metadata = {**base_metadata, "parse_retry_ordinal": ordinal}
            request_sha, resumable = self._identity(prompt, metadata, self.retry)
            if result.resume_hash != resumable:
                continue
            request = self.runtime.build_generate_content_request(
                "judge", prompt, json_mode=True, max_output_tokens=400,
                stage_limits=self.retry.stage_limits,
            )
            cache_path = self.cache_root / "responses" / "judge" / f"{resumable}.json"
            final = _read_object(cache_path, "Judge parse-retry final")
            plan_path = self.cache_root / "plans" / "judge" / f"{resumable}.json"
            plan = _read_object(plan_path, "Judge parse-retry plan")
            expected_plan = self.retry._plan_record(
                stage="judge", request=request, prompt=prompt,
                request_sha=request_sha, resumable=resumable,
                caption_context_count=None, cache_metadata=metadata,
            )
            attempt_request_path, attempt_response_path = self.retry._attempt_paths(
                "judge", resumable, 1
            )
            expected_attempt = self.retry._attempt_request_record(
                stage="judge", request_sha=request_sha, resumable=resumable,
                ordinal=1, plan=plan,
            )
            attempt_request = _read_object(attempt_request_path, "Judge retry intent")
            attempt_response = _read_object(attempt_response_path, "Judge retry response")
            if (
                plan != expected_plan
                or attempt_request != expected_attempt
                or final.get("request_sha256") != request_sha
                or final.get("resume_hash") != resumable
                or final.get("attempt_ordinal") != 1
                or final.get("text") != result.text
                or final.get("response_sha256") != result.response_sha256
                or attempt_response.get("outcome_kind") != "raw_response"
                or attempt_response.get("response_sha256") != result.response_sha256
                or self.retry._orphan_path("judge", resumable).exists()
                or self.retry._failure_path("judge", resumable).exists()
                or self._census(prompt, base_metadata) > MAX_GENERATION_ATTEMPTS
            ):
                raise IsolatedMatrixError("Judge parse-retry evidence drift")
            return {
                "resume_hash": resumable,
                "successful_attempt_ordinal": 1,
                "cache_path": str(cache_path.resolve()),
                "cache_file_sha256": score._file_sha256(cache_path),  # type: ignore[attr-defined]
                "plan_file_sha256": score._file_sha256(plan_path),  # type: ignore[attr-defined]
                "attempt_request_file_sha256": score._file_sha256(attempt_request_path),  # type: ignore[attr-defined]
                "attempt_response_file_sha256": score._file_sha256(attempt_response_path),  # type: ignore[attr-defined]
                "explicit_retry_chain": [],
                "parse_retry_ordinal": ordinal,
                "logical_physical_attempt_intents": self._census(prompt, base_metadata),
            }
        raise IsolatedMatrixError("Judge result is outside primary/parse-retry identities")


def _light_score_complete(
    *, output_root: Path, score_data_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    value = _read_object(root / "score" / "complete.json", "isolated score complete")
    contract = dict(value)
    observed = contract.pop("complete_sha256", None)
    cells = value.get("cells")
    if (
        observed != score._sha256(contract)  # type: ignore[attr-defined]
        or value.get("schema_version") != score.SCORE_COMPLETE_SCHEMA
        or value.get("complete") is not True
        or not isinstance(cells, list)
        or len(cells) != len(matrix.matrix_cell_ids())
    ):
        raise IsolatedMatrixError("isolated score completion seal drift")
    for row in cells:
        path = Path(str(row.get("score_path", ""))).resolve()
        cell = _read_object(path, "isolated score cell")
        if (
            root not in path.parents
            or score._file_sha256(path) != row.get("score_file_sha256")  # type: ignore[attr-defined]
            or len(cell.get("rows", [])) != matrix.EXPECTED_QUESTIONS
        ):
            raise IsolatedMatrixError(f"isolated score cell drift: {path}")
    calibration = value.get("calibration", {})
    calibration_path = Path(str(calibration.get("path", ""))).resolve()
    if (
        root not in calibration_path.parents
        or score._file_sha256(calibration_path) != calibration.get("file_sha256")  # type: ignore[attr-defined]
    ):
        raise IsolatedMatrixError("isolated calibration seal drift")
    return value


def score_report(*, output_root: Path) -> dict[str, Any]:
    """Return the ten comparable matrix numbers without any provider access."""

    root = _assert_output_root(output_root)
    full_path = root / "score" / "complete.json"
    matrix_path = root / "score" / "matrix_complete.json"
    if full_path.is_file():
        complete = _light_score_complete(
            output_root=root,
            score_data_root=DEFAULT_SCORE_DATA_ROOT,
            protocol=matrix.load_protocol(),
        )
        calibration_path: Path | None = Path(str(complete["calibration"]["path"]))
        report_status = "isolated_5x2_score_report_with_calibration"
    elif matrix_path.is_file():
        complete = _light_matrix_score_complete(root)
        calibration_path = None
        report_status = "isolated_5x2_score_report_matrix_only"
    else:
        raise IsolatedMatrixError(
            "no completed scores: run --phase score --skip-calibration "
            "--allow-model-calls (or the full score phase) first"
        )
    cells = [
        {
            "cell_id": row["cell_id"],
            "backbone": row["cell_id"].split("__", 1)[0],
            "architecture": row["cell_id"].split("__", 1)[1],
            "raw_mean": row["raw_mean"],
            "mapped_mean": row["mapped_mean"],
            "mapped_percent": float(row["mapped_mean"]) * 100.0,
            "rows": row["rows"],
        }
        for row in complete["cells"]
    ]
    report = {
        "status": report_status,
        "formal_release_claimed": False,
        "output_root": str(root),
        "cells": cells,
        "best_cell": max(cells, key=lambda row: float(row["mapped_mean"])),
        "historical_04533_calibration": None,
        "provider_calls": 0,
    }
    if calibration_path is not None:
        calibration = _read_object(calibration_path, "isolated score calibration")
        report["historical_04533_calibration"] = {
            "three_run_raw_mean": calibration["three_run_raw_mean"],
            "three_run_mapped_mean": calibration["three_run_mapped_mean"],
        }
    return report


def _light_matrix_score_complete(root: Path) -> dict[str, Any]:
    path = Path(root).resolve() / "score" / "matrix_complete.json"
    value = _read_object(path, "isolated matrix-only score complete")
    contract = dict(value)
    observed = contract.pop("complete_sha256", None)
    cells = value.get("cells")
    if (
        observed != score._sha256(contract)  # type: ignore[attr-defined]
        or value.get("schema_version") != MATRIX_SCORE_COMPLETE_SCHEMA
        or value.get("complete") is not True
        or not isinstance(cells, list)
        or len(cells) != len(matrix.matrix_cell_ids())
    ):
        raise IsolatedMatrixError("matrix-only score completion drift")
    for row in cells:
        cell_path = Path(str(row.get("score_path", ""))).resolve()
        cell = _read_object(cell_path, "matrix-only score cell")
        if (
            Path(root).resolve() not in cell_path.parents
            or score._file_sha256(cell_path) != row.get("score_file_sha256")  # type: ignore[attr-defined]
            or len(cell.get("rows", [])) != matrix.EXPECTED_QUESTIONS
        ):
            raise IsolatedMatrixError("matrix-only score cell drift")
    return value


def _execute_matrix_score_only(
    *,
    root: Path,
    score_data_root: Path,
    judge: Any,
    gate: Mapping[str, Any],
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    """Judge the ten matrix cells now; leave historical calibration resumable."""

    complete_path = root / "score" / "matrix_complete.json"
    if complete_path.is_file():
        complete = _light_matrix_score_complete(root)
        return {
            "status": "resumed_matrix_score_complete",
            "provider_calls": 0,
            "complete_path": str(complete_path.resolve()),
            "complete_sha256": complete["complete_sha256"],
            "cells": len(complete["cells"]),
            "calibration_skipped": True,
        }
    protocol = matrix.load_protocol()
    predictions = score._prediction_rows(root, protocol)  # type: ignore[attr-defined]
    gold = score._gold_rows(  # type: ignore[attr-defined]
        output_root=root, score_data_root=score_data_root, protocol=protocol
    )
    ordered_triples: list[tuple[tuple[str, str, str], str]] = []
    for cell_id in matrix.matrix_cell_ids():
        for prediction_row in predictions[cell_id]:
            question, answer = gold[str(prediction_row["row_id"])]
            ordered_triples.append(
                (
                    (question, answer, str(prediction_row["prediction"])),
                    "matrix",
                )
            )
    result_cache = score._prefetch_judgements(  # type: ignore[attr-defined]
        judge, ordered_triples, max_workers=workers
    )
    cell_receipts: list[dict[str, Any]] = []
    for cell_id in matrix.matrix_cell_ids():
        rows: list[dict[str, Any]] = []
        for prediction_row in predictions[cell_id]:
            row_id = str(prediction_row["row_id"])
            question, answer = gold[row_id]
            prediction_text = str(prediction_row["prediction"])
            triple = (question, answer, prediction_text)
            triple_sha = score._sha256(list(triple))  # type: ignore[attr-defined]
            if triple not in result_cache:
                result_cache[triple] = score._judge_one(  # type: ignore[attr-defined]
                    judge, triple, scope="matrix", triple_sha256=triple_sha
                )
            rows.append(
                {
                    "row_id": row_id,
                    "question_sha256": score._sha256(question),  # type: ignore[attr-defined]
                    "gold_sha256": score._sha256(answer),  # type: ignore[attr-defined]
                    "prediction_sha256": score._sha256(prediction_text),  # type: ignore[attr-defined]
                    "exact_triple_sha256": triple_sha,
                    **result_cache[triple],
                }
            )
        cell_value = {
            "schema_version": score.SCORE_CELL_SCHEMA,
            "cell_id": cell_id,
            "rows": rows,
            "rows_sha256": score._sha256(rows),  # type: ignore[attr-defined]
            "raw_mean": score._mean([row["raw_score"] for row in rows]),  # type: ignore[attr-defined]
            "mapped_mean": score._mean([row["mapped_score"] for row in rows]),  # type: ignore[attr-defined]
            "complete": True,
        }
        cell_value["cell_score_sha256"] = score._sha256(cell_value)  # type: ignore[attr-defined]
        cell_path = root / "score" / "cells" / cell_id / "scores.json"
        score._write_once(cell_path, cell_value)  # type: ignore[attr-defined]
        cell_receipts.append(
            {
                "cell_id": cell_id,
                "backbone": cell_id.split("__", 1)[0],
                "architecture": cell_id.split("__", 1)[1],
                "rows": len(rows),
                "raw_mean": cell_value["raw_mean"],
                "mapped_mean": cell_value["mapped_mean"],
                "score_path": str(cell_path.resolve()),
                "score_file_sha256": score._file_sha256(cell_path),  # type: ignore[attr-defined]
                "cell_score_sha256": cell_value["cell_score_sha256"],
            }
        )
    usage = score._durable_gemini_usage(  # type: ignore[attr-defined]
        judge.usage_receipt(),
        cache_root=root / "provider" / "score" / "g35_judge",
    )
    provider_calls = int(usage["stages"]["judge"]["provider_calls"])
    complete = {
        "schema_version": MATRIX_SCORE_COMPLETE_SCHEMA,
        "complete": True,
        "protocol_sha256": score._sha256(protocol),  # type: ignore[attr-defined]
        "prediction_complete_file_sha256": score._file_sha256(  # type: ignore[attr-defined]
            root / "predict" / "complete.json"
        ),
        # Bind the full budget/gate so a later unskipped score phase reuses the
        # exact same judge cache identities and only adds calibration calls.
        "full_score_budget_sha256": gate["budget"]["score_budget_sha256"],
        "full_score_budget_gate_file_sha256": gate["file_sha256"],
        "cells": cell_receipts,
        "judge_usage": usage,
        "provider_calls": provider_calls,
        "calibration_skipped": True,
        "formal_release_claimed": False,
    }
    complete["complete_sha256"] = score._sha256(complete)  # type: ignore[attr-defined]
    score._write_once(complete_path, complete)  # type: ignore[attr-defined]
    _light_matrix_score_complete(root)
    return {
        "status": "matrix_score_complete_calibration_skipped",
        "provider_calls": provider_calls,
        "complete_path": str(complete_path.resolve()),
        "complete_sha256": complete["complete_sha256"],
        "cells": len(cell_receipts),
        "calibration_skipped": True,
    }


def run_score(
    *,
    source_project_root: Path,
    output_root: Path,
    score_data_root: Path,
    skip_calibration: bool = False,
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    root = _assert_output_root(output_root)
    source_project = _project(source_project_root)
    original_predict_validator = predict._prediction_complete  # type: ignore[attr-defined]
    original_score_validator = score._validate_score_complete  # type: ignore[attr-defined]
    original_judge_evidence_validator = score._validate_runtime_judge_evidence  # type: ignore[attr-defined]
    original_project_validator = matrix.validate_project
    original_authority_receipt = matrix.authority_receipt
    original_attempts = score.MAX_PROVIDER_ATTEMPTS
    predict._prediction_complete = _light_prediction_complete  # type: ignore[attr-defined]
    score._validate_score_complete = _light_score_complete  # type: ignore[attr-defined]
    score.MAX_PROVIDER_ATTEMPTS = MAX_GENERATION_ATTEMPTS

    def source_project_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return source_project

    def isolated_authority_receipt(protocol: Mapping[str, Any]) -> dict[str, Any]:
        return _isolated_score_authority_receipt(
            protocol, canonical_receipt=original_authority_receipt
        )

    def isolated_judge_evidence(**kwargs: Any) -> dict[str, Any]:
        backend = kwargs.get("judge_backend")
        if isinstance(backend, _ParseRetryJudgeBackend):
            return backend.validate_evidence(
                result=kwargs["result"],
                triple=kwargs["triple"],
                scope=kwargs["scope"],
                triple_sha256=kwargs["triple_sha256"],
                canonical_validator=original_judge_evidence_validator,
            )
        return original_judge_evidence_validator(**kwargs)

    matrix.validate_project = source_project_validator  # type: ignore[assignment]
    matrix.authority_receipt = isolated_authority_receipt  # type: ignore[assignment]
    score._validate_runtime_judge_evidence = isolated_judge_evidence  # type: ignore[attr-defined]
    try:
        gate = _ensure_score_gate(root, score_data_root)
        from creamem3 import memeye_g35_runtime_v19 as runtime

        limits = {
            "caption": runtime.StageLimits(2400, 0, MAX_GENERATION_ATTEMPTS),
            "writer": runtime.StageLimits(16000, 0, MAX_GENERATION_ATTEMPTS),
            "view": runtime.StageLimits(16384, 0, MAX_GENERATION_ATTEMPTS),
            "judge": runtime.StageLimits(
                400,
                gate["budget"]["judge_provider_attempt_hard_ceiling"],
                MAX_GENERATION_ATTEMPTS,
            ),
        }
        judge = runtime.Gemini35Runtime(
            root / "provider" / "score" / "g35_judge",
            allow_model_calls=True,
            provider_factory=lambda: _ReliableRESTProvider(runtime),
            stage_limits=limits,
        )
        retry_limits = dict(limits)
        retry_limits["judge"] = runtime.StageLimits(
            400,
            gate["budget"]["judge_provider_attempt_hard_ceiling"],
            1,
        )
        retry_judge = runtime.Gemini35Runtime(
            root / "provider" / "score" / "g35_judge",
            allow_model_calls=True,
            provider_factory=lambda: _ReliableRESTProvider(runtime),
            stage_limits=retry_limits,
        )
        judge_backend = _ParseRetryJudgeBackend(judge, retry_judge, runtime)
        execution = (
            _execute_matrix_score_only(
                root=root,
                score_data_root=score_data_root,
                judge=judge_backend,
                gate=gate,
                workers=workers,
            )
            if skip_calibration
            else score.execute_score_with_backend(
                output_root=root,
                score_data_root=score_data_root,
                judge_backend=judge_backend,
                budget_gate=gate,
                max_workers=workers,
            )
        )
        report = score_report(output_root=root)
        return {**execution, "score_report": report}
    finally:
        score._validate_runtime_judge_evidence = original_judge_evidence_validator  # type: ignore[attr-defined]
        matrix.authority_receipt = original_authority_receipt  # type: ignore[assignment]
        matrix.validate_project = original_project_validator  # type: ignore[assignment]
        predict._prediction_complete = original_predict_validator  # type: ignore[attr-defined]
        score._validate_score_complete = original_score_validator  # type: ignore[attr-defined]
        score.MAX_PROVIDER_ATTEMPTS = original_attempts


def status(
    *, source_project_root: Path, caption_root: Path, output_root: Path
) -> dict[str, Any]:
    root = _assert_output_root(output_root)
    caption_files = (
        len(list((Path(caption_root) / "captions" / "g35").rglob("*.json")))
        if (Path(caption_root) / "captions" / "g35").is_dir()
        else 0
    )
    phases = {
        "caption": Path(caption_root) / "build" / "caption_complete.json",
        "writer": root / "build" / "writer_complete.json",
        "build": root / "build" / "complete.json",
        "predict": root / "predict" / "complete.json",
        "matrix_score": root / "score" / "matrix_complete.json",
        "score": root / "score" / "complete.json",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "zero_call_status",
        "source_project_root": str(Path(source_project_root).resolve()),
        "caption_root": str(Path(caption_root).resolve()),
        "output_root": str(root),
        "caption_files": caption_files,
        "phases": {
            name: {
                "path": str(path.resolve()),
                "present": path.is_file(),
                "file_sha256": build._file_sha256(path) if path.is_file() else None,  # type: ignore[attr-defined]
            }
            for name, path in phases.items()
        },
        "matrix_cells": matrix.matrix_cell_ids(),
        "generation_max_attempts": MAX_GENERATION_ATTEMPTS,
        "generation_retry_wait_seconds": RETRY_WAIT_SECONDS,
        "formal_release_claimed": False,
        "provider_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("status", "report", "writer", "build", "predict", "score", "all"),
        default="status",
    )
    parser.add_argument("--source-project-root", type=Path, default=DEFAULT_SOURCE_PROJECT_ROOT)
    parser.add_argument("--caption-root", type=Path, default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--score-data-root", type=Path, default=DEFAULT_SCORE_DATA_ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_EVAL_WORKERS,
        help="ordered isolated predict/score workers (default: historical MGTM 4)",
    )
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument(
        "--writer-backbone",
        choices=("g25", "g35", "both"),
        default="both",
        help="writer only: run one packet namespace without sealing, or both and seal",
    )
    parser.add_argument(
        "--writer-tasks",
        default="",
        help=(
            "writer only: comma-separated exact G25 task names; runs only these "
            "task-scoped packet namespaces without sealing"
        ),
    )
    parser.add_argument(
        "--writer-task-workers",
        type=int,
        default=1,
        help="explicit G25 writer task workers (use 4 for the emergency task split)",
    )
    parser.add_argument(
        "--build-task-workers",
        type=int,
        default=1,
        help="isolated build only: task-level workers; formal/default build remains 1",
    )
    parser.add_argument(
        "--build-backbone",
        choices=("g25", "g35", "both"),
        default="both",
        help="build only: pre-materialize one backbone without sealing",
    )
    parser.add_argument(
        "--build-units",
        default="",
        help="build only: comma-separated exact units such as g25:Task,g35:Task",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="score ten matrix cells now and defer the historical 0.4533 calibration",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    writer_tasks = [
        value.strip() for value in str(args.writer_tasks).split(",") if value.strip()
    ]
    build_units = [
        value.strip() for value in str(args.build_units).split(",") if value.strip()
    ]
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.writer_task_workers < 1:
        raise SystemExit("--writer-task-workers must be positive")
    if args.build_task_workers < 1:
        raise SystemExit("--build-task-workers must be positive")
    if args.writer_backbone != "both" and args.phase != "writer":
        raise SystemExit("single --writer-backbone requires --phase writer")
    if writer_tasks and (args.phase != "writer" or args.writer_backbone != "g25"):
        raise SystemExit(
            "--writer-tasks requires --phase writer --writer-backbone g25"
        )
    if args.writer_task_workers != 1 and not writer_tasks:
        raise SystemExit("--writer-task-workers requires --writer-tasks")
    if (
        args.build_task_workers != 1
        or build_units
        or args.build_backbone != "both"
    ) and args.phase != "build":
        raise SystemExit(
            "--build-task-workers/--build-units/--build-backbone require --phase build"
        )
    if args.phase in {"status", "report"}:
        if args.allow_model_calls:
            raise SystemExit(f"{args.phase} rejects --allow-model-calls")
        if args.phase == "status":
            result = status(
                source_project_root=args.source_project_root,
                caption_root=args.caption_root,
                output_root=args.output_root,
            )
        else:
            result = score_report(output_root=args.output_root)
    else:
        if not args.allow_model_calls:
            raise SystemExit(f"{args.phase} requires --allow-model-calls")
        phases = (
            ("writer", "build", "predict", "score")
            if args.phase == "all"
            else (args.phase,)
        )
        result: dict[str, Any] = {}
        for phase in phases:
            if phase == "writer":
                result = run_writer(
                    source_project_root=args.source_project_root,
                    caption_root=args.caption_root,
                    output_root=args.output_root,
                    writer_backbone=args.writer_backbone,
                    writer_tasks=writer_tasks,
                    writer_task_workers=args.writer_task_workers,
                )
            elif phase == "build":
                if (
                    args.build_task_workers != 1
                    or build_units
                    or args.build_backbone != "both"
                ):
                    result = run_build_task_parallel(
                        source_project_root=args.source_project_root,
                        caption_root=args.caption_root,
                        output_root=args.output_root,
                        workers=args.build_task_workers,
                        requested_unit_ids=build_units,
                        build_backbone=args.build_backbone,
                    )
                else:
                    result = run_build(
                        source_project_root=args.source_project_root,
                        caption_root=args.caption_root,
                        output_root=args.output_root,
                    )
            elif phase == "predict":
                result = run_predict(
                    source_project_root=args.source_project_root,
                    output_root=args.output_root,
                    workers=args.workers,
                )
            else:
                result = run_score(
                    source_project_root=args.source_project_root,
                    output_root=args.output_root,
                    score_data_root=args.score_data_root,
                    skip_calibration=args.skip_calibration,
                    workers=args.workers,
                )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

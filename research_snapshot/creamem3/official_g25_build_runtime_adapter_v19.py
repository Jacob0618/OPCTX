"""Durable official-Google Gemini 2.5 build runtime for MemEye v19.

This is an immutable build profile over the frozen direct-REST v19 runtime.
It uses a distinct isolated module namespace from the Reader profile, so
writer/view limits cannot contaminate planner/Reader limits in one process.
The global OS lease and durable attempt census make stage ceilings survive
process restarts; an unresolved lease is fail-closed for manual audit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_ID = "gemini-2.5-flash-lite"
MODEL_API = "official_google_v1beta_generateContent_REST"
STAGES = ("writer", "view")
SOURCE_RUNTIME = Path(__file__).with_name("memeye_g35_runtime_v19.py")
SOURCE_RUNTIME_SHA256 = (
    "6f2155d52c69f465a29425b2eec74f2e0229b4629610894f6be310620539a7f9"
)
PROTOCOL_VERSION = "creamem3.memeye_official_g25_build_runtime.v19.0"
DEFAULT_CACHE_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\full_arch_matrix_r2\provider\build\g25"
)


class G25BuildRuntimeError(RuntimeError):
    """Frozen source, request, cache, or durable-budget drift."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def _patch_build_profile(module: types.ModuleType) -> None:
    StageLimits = module.StageLimits
    limits = {
        "writer": StageLimits(16000, 1848, 3),
        "view": StageLimits(16384, 735, 3),
    }
    module.MODEL_ID = MODEL_ID
    module.MODEL_API = MODEL_API
    module.STAGES = STAGES
    module.DEFAULT_STAGE_LIMITS = limits
    module.DEFAULT_CACHE_ROOT = DEFAULT_CACHE_ROOT

    def assert_request_contract(
        request: Mapping[str, Any],
        *,
        stage: str,
        caption_context_count: int | None = None,
        token_limit: int | None = None,
    ) -> None:
        if stage not in STAGES or caption_context_count is not None:
            raise module.RequestContractError("G25 build stage/context drift")
        generation = request.get("generationConfig")
        if not isinstance(generation, Mapping) or set(generation) - {
            "temperature",
            "maxOutputTokens",
            "thinkingConfig",
            "responseMimeType",
        }:
            raise module.RequestContractError("G25 build generation controls drift")
        if generation.get("temperature") != 0.0 or generation.get(
            "thinkingConfig"
        ) != {"thinkingBudget": 0}:
            raise module.RequestContractError("G25 build sampling/thinking drift")
        output_tokens = generation.get("maxOutputTokens")
        if (
            not isinstance(output_tokens, int)
            or output_tokens <= 0
            or (token_limit is not None and output_tokens > token_limit)
        ):
            raise module.RequestContractError("G25 build token cap drift")
        contents = request.get("contents")
        if not isinstance(contents, list) or len(contents) != 1:
            raise module.RequestContractError("G25 build content census drift")
        parts = contents[0].get("parts") if isinstance(contents[0], Mapping) else None
        if (
            not isinstance(parts, list)
            or len(parts) != 1
            or not isinstance(parts[0], Mapping)
            or set(parts[0]) != {"text"}
            or not str(parts[0].get("text", "")).strip()
        ):
            raise module.RequestContractError("G25 build request must be text-only")

    def build_generate_content_request(
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        parts: Sequence[Mapping[str, Any]] = (),
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        stage_limits: Mapping[str, Any] | None = None,
        caption_context_count: int | None = None,
    ) -> dict[str, Any]:
        active = dict(stage_limits or module.DEFAULT_STAGE_LIMITS)
        if stage not in STAGES or set(active) != set(STAGES):
            raise module.RequestContractError("G25 build stage-limit map drift")
        if not str(prompt).strip() or parts or caption_context_count is not None:
            raise module.RequestContractError("G25 build prompt/parts drift")
        hard = int(active[stage].max_output_tokens)
        requested = hard if max_output_tokens is None else int(max_output_tokens)
        if requested <= 0 or requested > hard:
            raise module.RequestContractError("G25 build output cap drift")
        generation: dict[str, Any] = {
            "temperature": 0.0,
            "maxOutputTokens": requested,
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if json_mode:
            generation["responseMimeType"] = "application/json"
        request: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [module.text_part(str(prompt))]}
            ],
            "generationConfig": generation,
        }
        if system:
            request["systemInstruction"] = {
                "parts": [module.text_part(str(system))]
            }
        assert_request_contract(request, stage=stage, token_limit=hard)
        return request

    def protocol_manifest(
        stage_limits: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = dict(stage_limits or module.DEFAULT_STAGE_LIMITS)
        if set(active) != set(STAGES):
            raise module.RequestContractError("G25 build protocol stage map drift")
        return {
            "schema_version": PROTOCOL_VERSION,
            "profile": "build_writer_view",
            "model": MODEL_ID,
            "api": MODEL_API,
            "generation_controls": {"temperature": 0.0, "thinking_budget": 0},
            "stages": {name: asdict(active[name]) for name in STAGES},
            "source_runtime_path": str(SOURCE_RUNTIME.resolve()),
            "source_runtime_sha256": SOURCE_RUNTIME_SHA256,
            "cache_policy": {
                "content_addressed": True,
                "write_once_attempt_receipts": True,
                "global_os_budget_lease": True,
                "durable_stage_attempt_census": True,
                "indeterminate_transport": "orphan_fail_closed_no_recall",
                "explicit_retryable_http_only": True,
                "legacy_cache_readthrough": False,
            },
        }

    def protocol_sha256(stage_limits: Mapping[str, Any] | None = None) -> str:
        return _sha256(protocol_manifest(stage_limits))

    module.assert_request_contract = assert_request_contract
    module.build_generate_content_request = build_generate_content_request
    module.protocol_manifest = protocol_manifest
    module.protocol_sha256 = protocol_sha256


def _load_source_runtime() -> types.ModuleType:
    source_path = SOURCE_RUNTIME.resolve()
    if _file_sha256(source_path) != SOURCE_RUNTIME_SHA256:
        raise G25BuildRuntimeError("frozen durable runtime source drift")
    module_name = "creamem3._official_g25_build_runtime_v19_isolated"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise G25BuildRuntimeError("cannot load frozen durable runtime source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        _patch_build_profile(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class OfficialGemini25BuildRuntime:
    """Restart-safe writer/view facade; provider remains delayed on cache hits."""

    def __init__(
        self,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        *,
        allow_model_calls: bool = False,
        provider_factory: Any | None = None,
    ) -> None:
        self._module = _load_source_runtime()
        self.cache_root = Path(cache_root).resolve()
        self.stage_limits = dict(self._module.DEFAULT_STAGE_LIMITS)
        self._runtime = self._module.Gemini35Runtime(
            self.cache_root,
            allow_model_calls=allow_model_calls,
            provider_factory=provider_factory,
            stage_limits=self.stage_limits,
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
            raise G25BuildRuntimeError(
                "stale G25 build global lease requires manual audit"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_sha256({name: asdict(limit) for name, limit in self.stage_limits.items()}))
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
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        if stage not in STAGES:
            raise G25BuildRuntimeError("G25 build stage drift")
        request = self._module.build_generate_content_request(
            stage,
            prompt,
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            stage_limits=self.stage_limits,
        )
        request_sha = self._module.request_identity(
            stage=stage,
            request=request,
            stage_limits=self.stage_limits,
            cache_metadata=cache_metadata,
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
                raise G25BuildRuntimeError(
                    f"durable G25 {stage} provider-attempt ceiling exhausted"
                )
            return self._runtime.generate(
                stage,
                prompt,
                system=system,
                json_mode=json_mode,
                max_output_tokens=max_output_tokens,
                cache_metadata=cache_metadata,
            )
        finally:
            lease.unlink(missing_ok=True)

    def usage_receipt(self) -> Mapping[str, Any]:
        value = self._runtime.usage_receipt()
        for stage, row in value["stages"].items():
            in_process = int(row["provider_calls"])
            durable = self._durable_attempts(stage)
            row["in_process_provider_calls"] = in_process
            row["durable_provider_attempt_intents"] = durable
            row["provider_calls"] = durable
            row["remaining_live_calls"] = (
                self.stage_limits[stage].max_live_calls - durable
            )
        value["profile"] = "build_writer_view"
        return value


def protocol_manifest() -> dict[str, Any]:
    return dict(_load_source_runtime().protocol_manifest())


def zero_call_preflight() -> dict[str, Any]:
    module = _load_source_runtime()
    requests = {
        "writer_summary": module.build_generate_content_request(
            "writer", "summary", json_mode=True, max_output_tokens=1200
        ),
        "writer_fact": module.build_generate_content_request(
            "writer", "facts", json_mode=True, max_output_tokens=16000
        ),
        "view": module.build_generate_content_request(
            "view", "view", json_mode=True, max_output_tokens=16384
        ),
    }
    return {
        "status": "pass",
        "protocol": protocol_manifest(),
        "request_sha256": {name: _sha256(value) for name, value in requests.items()},
        "provider_calls": 0,
        "cache_files_written": 0,
    }


__all__ = [
    "G25BuildRuntimeError",
    "MODEL_ID",
    "OfficialGemini25BuildRuntime",
    "PROTOCOL_VERSION",
    "SOURCE_RUNTIME_SHA256",
    "STAGES",
    "protocol_manifest",
    "zero_call_preflight",
]

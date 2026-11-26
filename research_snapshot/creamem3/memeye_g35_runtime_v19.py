"""Isolated Gemini 3.5 Flash-Lite runtime contract for MemEye v19.

This module is deliberately provider-cold at import time.  A provider is
constructed only after all of the following are true:

* the request is not already present in the content-addressed cache;
* ``allow_model_calls`` was explicitly enabled by the caller;
* the independent live-call cap for the requested stage is not exhausted.

The adapter owns only the Gemini 3.5 request/caching contract.  It does not
select memories, render CreaMem graphs, or change the frozen MGTM prompts.
The full-architecture runner supplies those inputs through :meth:`generate`.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence


MODEL_ID = "gemini-3.5-flash-lite"
PROTOCOL_VERSION = "creamem3.memeye_g35_runtime.v19.0"
CACHE_SCHEMA_VERSION = "creamem3.memeye_g35_cache.v19.0"
MODEL_API = "GenerateContent"
THINKING_LEVEL = "minimal"
MEDIA_RESOLUTION_HIGH = "MEDIA_RESOLUTION_HIGH"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "CREAMEM3_MEMEYE_G35_CACHE_ROOT",
        r"D:\creamem3_cache\memeye_v19\full_arch_g35_matched_r1",
    )
)

STAGES = ("caption", "writer", "view", "judge")
FORBIDDEN_GENERATION_FIELDS = frozenset(
    {
        "temperature",
        "topP",
        "top_p",
        "topK",
        "top_k",
        "candidateCount",
        "candidate_count",
        "thinkingBudget",
        "thinking_budget",
    }
)

# The old V7 source bundle is not recoverable byte-for-byte.  The live 3.5
# branch therefore uses the current upstream IMAGE_READ_PROMPT as the best
# available reconstruction while preserving the recoverable granularity.
CAPTION_LINEAGE: dict[str, Any] = {
    "lineage_status": "best_available_v7_reconstruction_not_byte_exact",
    "prompt_authority": "current_eval_memeye.py_IMAGE_READ_PROMPT",
    "historical_v7_complete_prompt_source_recoverable": False,
    "byte_exact_v7_reproduction_claim_allowed": False,
    "raw_images_per_call": 1,
    "previous_caption_text_max": 2,
    "media_resolution": MEDIA_RESOLUTION_HIGH,
    "current_image_read_prompt_sha256": (
        "efeaedca8552c9b59e3a014259f296435755fdcdfecbf5ba895990abd6a7e8af"
    ),
}

DEFAULT_IMAGE_PROMPT_SOURCE = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\multimodal\eval_memeye.py"
)
EXPECTED_CURRENT_IMAGE_READ_PROMPT_SHA256 = (
    "efeaedca8552c9b59e3a014259f296435755fdcdfecbf5ba895990abd6a7e8af"
)
DEFAULT_MGTM_PROMPT_SOURCE = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\multimodal\eval_memeye.py"
)
EXPECTED_MGTM_JUDGE_PROMPT_SHA256 = (
    "5471df5449fc3a74918ad85cd8c8ee58621a40784a2b1c97f3bff4af610bd952"
)
ALLOWED_JUDGE_SCORES = frozenset({0.0, 0.25, 0.5, 0.75, 1.0})


class RuntimeProtocolError(RuntimeError):
    """Base class for a fail-closed v19 runtime contract violation."""


class ProviderCallBlocked(RuntimeProtocolError):
    """Raised on an uncached request when live calls were not authorized."""


class StageBudgetExceeded(RuntimeProtocolError):
    """Raised before a provider call would exceed a stage-specific cap."""


class CacheIntegrityError(RuntimeProtocolError):
    """Raised when a write-once or resumable cache artifact has drifted."""


class RequestContractError(RuntimeProtocolError):
    """Raised when a request violates the frozen Gemini 3.5 contract."""


class LifecycleBusyError(RuntimeProtocolError):
    """Raised when another process still owns the request lifecycle lock."""


class OrphanedCallError(RuntimeProtocolError):
    """Raised for an intent whose provider outcome cannot be reconstructed."""


class ExplicitProviderError(RuntimeProtocolError):
    """A provider returned a determinate error response."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code


class IndeterminateProviderError(RuntimeProtocolError):
    """Transport state does not prove whether the provider accepted the call."""


class ProviderRetriesExhausted(RuntimeProtocolError):
    """All registered, explicitly retryable attempt ordinals were consumed."""


@dataclass(frozen=True)
class StageLimits:
    """Independent output-token and live-call ceilings for one stage."""

    max_output_tokens: int
    max_live_calls: int
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_live_calls < 0:
            raise ValueError("max_live_calls must be non-negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


DEFAULT_STAGE_LIMITS: dict[str, StageLimits] = {
    # One fresh caption per MemEye image.  Retries require an explicitly raised
    # cap rather than silently expanding the preregistered 438-call budget.
    "caption": StageLimits(max_output_tokens=2400, max_live_calls=1314, max_attempts=3),
    # These are hard safety ceilings, not expected call counts.  The matrix
    # runner records the tighter architecture-derived request plan separately.
    "writer": StageLimits(max_output_tokens=16000, max_live_calls=4096, max_attempts=6),
    "view": StageLimits(max_output_tokens=16384, max_live_calls=4096, max_attempts=3),
    # Ten full-371 cells need at most 3,710 first-pass judge requests before
    # exact-triple cache reuse; 4,096 leaves no room for an accidental rerun.
    "judge": StageLimits(max_output_tokens=400, max_live_calls=11130, max_attempts=3),
}


@dataclass(frozen=True)
class UsageRecord:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    stage: str
    model: str
    text: str
    usage: UsageRecord
    request_sha256: str
    resume_hash: str
    response_sha256: str
    cache_path: str
    cache_hit: bool


@dataclass(frozen=True)
class JudgeVerdict:
    raw_score: float
    mapped_score: float
    reasoning: str


class GenerateContentProvider(Protocol):
    """Minimal injectable provider used by the isolated runtime."""

    def generate_content(
        self, *, model: str, request: Mapping[str, Any]
    ) -> Any: ...


ProviderFactory = Callable[[], GenerateContentProvider]


@dataclass
class RequestLifecycleLease:
    """One OS lock held from the first state read through final fsync."""

    path: Path
    handle: BinaryIO | None
    acquired: bool

    def __enter__(self) -> "RequestLifecycleLease":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.handle is None:
            return
        try:
            if self.acquired:
                _release_os_file_lock(self.handle)
        finally:
            self.handle.close()
            self.handle = None
            self.acquired = False


def _try_os_file_lock(handle: BinaryIO) -> bool:
    """Try one exclusive byte-range lock; process exit releases it."""

    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {13, 11, 36} or getattr(error, "winerror", None) in {
                33,
                36,
            }:
                return False
            raise
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_os_file_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def acquire_request_lifecycle_lock(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> RequestLifecycleLease:
    """Wait for one lifecycle owner without reading or writing call state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        handle = os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            acquired = _try_os_file_lock(handle)
        except BaseException:
            handle.close()
            raise
        if acquired:
            return RequestLifecycleLease(path=path, handle=handle, acquired=True)
        if time.monotonic() >= deadline:
            handle.close()
            raise LifecycleBusyError(
                f"request lifecycle owner remained active for {timeout_seconds:g}s"
            )
        time.sleep(poll_seconds)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically create canonical JSON once; replay never overwrites."""

    path = Path(path)
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    payload_sha = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or file_sha256(path) != payload_sha:
            raise CacheIntegrityError(f"write-once artifact drift: {path}")
        return payload_sha
    temporary = path.parent / f".tmp-{os.getpid():x}-{uuid.uuid4().hex[:16]}"
    try:
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link creation is atomic create-if-absent and cannot replace
            # a concurrently committed immutable destination.
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or file_sha256(path) != payload_sha:
                raise CacheIntegrityError(f"write-once artifact drift: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return payload_sha


def text_part(text: str) -> dict[str, str]:
    return {"text": str(text)}


def inline_image_part(data: bytes | str, mime_type: str) -> dict[str, Any]:
    """Build one high-resolution inline image part for GenerateContent."""

    if not str(mime_type).startswith("image/"):
        raise RequestContractError("inline image mime_type must start with image/")
    encoded = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
    try:
        base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise RequestContractError("inline image data is not valid base64") from exc
    return {
        "inlineData": {"mimeType": str(mime_type), "data": encoded},
        "mediaResolution": {"level": MEDIA_RESOLUTION_HIGH},
    }


def _is_image_part(part: Mapping[str, Any]) -> bool:
    inline = part.get("inlineData")
    file_data = part.get("fileData")
    mime = ""
    if isinstance(inline, Mapping):
        mime = str(inline.get("mimeType", ""))
    elif isinstance(file_data, Mapping):
        mime = str(file_data.get("mimeType", ""))
    return mime.startswith("image/")


def _validate_part(part: Mapping[str, Any]) -> None:
    payload_fields = [
        key for key in ("text", "inlineData", "fileData") if key in part
    ]
    if len(payload_fields) != 1:
        raise RequestContractError(
            "each part must contain exactly one of text/inlineData/fileData"
        )
    if _is_image_part(part):
        resolution = part.get("mediaResolution")
        if resolution != {"level": MEDIA_RESOLUTION_HIGH}:
            raise RequestContractError(
                "every visual input must use media_resolution=high"
            )


def build_generate_content_request(
    stage: str,
    prompt: str,
    *,
    system: str | None = None,
    parts: Sequence[Mapping[str, Any]] = (),
    json_mode: bool = False,
    max_output_tokens: int | None = None,
    stage_limits: Mapping[str, StageLimits] | None = None,
    caption_context_count: int | None = None,
) -> dict[str, Any]:
    """Build and validate the frozen REST-shaped GenerateContent payload."""

    limits = dict(stage_limits or DEFAULT_STAGE_LIMITS)
    if stage not in STAGES or stage not in limits:
        raise RequestContractError(f"unknown stage: {stage}")
    if not str(prompt).strip():
        raise RequestContractError("prompt must not be empty")
    token_limit = limits[stage].max_output_tokens
    requested_tokens = token_limit if max_output_tokens is None else int(max_output_tokens)
    if requested_tokens <= 0 or requested_tokens > token_limit:
        raise RequestContractError(
            f"{stage} max_output_tokens must be in [1, {token_limit}]"
        )

    normalized_parts = [dict(part) for part in parts]
    for part in normalized_parts:
        _validate_part(part)
    image_count = sum(_is_image_part(part) for part in normalized_parts)
    if stage == "caption":
        if caption_context_count is None or not 0 <= caption_context_count <= 2:
            raise RequestContractError(
                "caption stage requires caption_context_count in [0, 2]"
            )
        if image_count != 1:
            raise RequestContractError(
                "caption stage requires exactly one raw image per call"
            )
    elif image_count:
        raise RequestContractError(
            "raw visual inputs are confined to the caption stage"
        )

    generation_config: dict[str, Any] = {
        "maxOutputTokens": requested_tokens,
        "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    user_parts = (
        [*normalized_parts, text_part(prompt)]
        if stage == "caption"
        else [text_part(prompt), *normalized_parts]
    )
    request: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": user_parts,
            }
        ],
        "generationConfig": generation_config,
    }
    if stage == "caption":
        caption_system = system or load_current_image_read_prompt()
        if _sha256(caption_system) != EXPECTED_CURRENT_IMAGE_READ_PROMPT_SHA256:
            raise RequestContractError("caption system IMAGE_READ_PROMPT drift")
        request["systemInstruction"] = {"parts": [text_part(caption_system)]}
    elif system:
        request["systemInstruction"] = {"parts": [text_part(system)]}
    assert_request_contract(
        request,
        stage=stage,
        caption_context_count=caption_context_count,
        token_limit=token_limit,
    )
    return request


def assert_request_contract(
    request: Mapping[str, Any],
    *,
    stage: str,
    caption_context_count: int | None = None,
    token_limit: int | None = None,
) -> None:
    """Fail closed if a payload can alter the registered 3.5 protocol."""

    if stage not in STAGES:
        raise RequestContractError(f"unknown stage: {stage}")
    generation = request.get("generationConfig")
    if not isinstance(generation, Mapping):
        raise RequestContractError("missing generationConfig")
    forbidden = FORBIDDEN_GENERATION_FIELDS.intersection(generation)
    if forbidden:
        raise RequestContractError(
            "forbidden Gemini 3.5 generation fields: " + ", ".join(sorted(forbidden))
        )
    if generation.get("thinkingConfig") != {"thinkingLevel": THINKING_LEVEL}:
        raise RequestContractError("thinkingLevel must be minimal")
    output_tokens = generation.get("maxOutputTokens")
    if not isinstance(output_tokens, int) or output_tokens <= 0:
        raise RequestContractError("maxOutputTokens must be a positive integer")
    if token_limit is not None and output_tokens > token_limit:
        raise RequestContractError("stage output-token cap exceeded")
    contents = request.get("contents")
    if not isinstance(contents, list) or len(contents) != 1:
        raise RequestContractError("exactly one user content is required")
    content = contents[0]
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list) or not parts:
        raise RequestContractError("user content requires non-empty parts")
    for part in parts:
        if not isinstance(part, Mapping):
            raise RequestContractError("parts must be mappings")
        _validate_part(part)
    image_count = sum(_is_image_part(part) for part in parts)
    if stage == "caption":
        if caption_context_count is None or not 0 <= caption_context_count <= 2:
            raise RequestContractError("invalid previous-caption context count")
        if image_count != 1:
            raise RequestContractError("caption request image cardinality drift")
        if not _is_image_part(parts[0]) or "text" not in parts[-1]:
            raise RequestContractError(
                "caption user parts must be ordered raw-image first, user-text second"
            )
        if len(parts) != 2:
            raise RequestContractError("caption request must have exactly two user parts")
        instruction = request.get("systemInstruction")
        instruction_parts = (
            instruction.get("parts") if isinstance(instruction, Mapping) else None
        )
        if (
            not isinstance(instruction_parts, list)
            or len(instruction_parts) != 1
            or not isinstance(instruction_parts[0], Mapping)
            or _sha256(str(instruction_parts[0].get("text", "")))
            != EXPECTED_CURRENT_IMAGE_READ_PROMPT_SHA256
        ):
            raise RequestContractError("caption system IMAGE_READ_PROMPT drift")
    elif image_count:
        raise RequestContractError("non-caption request contains raw image")


def protocol_manifest(
    stage_limits: Mapping[str, StageLimits] | None = None,
) -> dict[str, Any]:
    limits = dict(stage_limits or DEFAULT_STAGE_LIMITS)
    if set(limits) != set(STAGES):
        raise RequestContractError("stage limit map must contain exactly four stages")
    return {
        "schema_version": PROTOCOL_VERSION,
        "model": MODEL_ID,
        "api": MODEL_API,
        "thinking_level": THINKING_LEVEL,
        "omitted_generation_fields": sorted(FORBIDDEN_GENERATION_FIELDS),
        "stage_limits": {stage: asdict(limits[stage]) for stage in STAGES},
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "default_cache_root": str(DEFAULT_CACHE_ROOT),
        "cache_policy": {
            "namespace_independent_from_2_5": True,
            "content_addressed_resume": True,
            "write_once": True,
            "provider_on_cache_hit": False,
        },
        "provider_policy": {
            "import": "delayed_until_authorized_cache_miss",
            "explicit_calls_gate": True,
            "transport": "direct_official_v1beta_generateContent_REST",
            "endpoint": GEMINI_API_BASE,
            "authentication": "x-goog-api-key_header_not_url_or_artifact",
            "endpoint_override": "fail_closed",
            "google_genai_sdk_dependency": False,
            "retry": "ordinal_receipts_only_after_explicit_retryable_HTTP_error",
            "indeterminate_transport": "fail_closed_without_recall",
        },
        "caption_lineage": dict(CAPTION_LINEAGE),
        "judge": {
            "model": MODEL_ID,
            "prompt": "frozen_original_mgtm_memeye_judge",
            "prompt_sha256": EXPECTED_MGTM_JUDGE_PROMPT_SHA256,
            "raw_scores": sorted(ALLOWED_JUDGE_SCORES),
            "official_mapping": {
                "0": 0.0,
                "0.25": 0.5,
                "0.5": 0.5,
                "0.75": 1.0,
                "1": 1.0,
            },
            "gpt_5_2_calls": 0,
        },
    }


def protocol_sha256(
    stage_limits: Mapping[str, StageLimits] | None = None,
) -> str:
    return _sha256(protocol_manifest(stage_limits))


def request_identity(
    *,
    stage: str,
    request: Mapping[str, Any],
    stage_limits: Mapping[str, StageLimits] | None = None,
    cache_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash every model-visible field and the frozen protocol contract."""

    return _sha256(
        {
            "schema_version": PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256(stage_limits),
            "stage": stage,
            "model": MODEL_ID,
            "api": MODEL_API,
            "request": request,
            "cache_metadata": dict(cache_metadata or {}),
        }
    )


def resume_hash(
    *,
    stage: str,
    request_sha256: str,
    stage_limits: Mapping[str, StageLimits] | None = None,
) -> str:
    """Bind a resumable cache location to the request and protocol hashes."""

    return _sha256(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(stage_limits),
            "stage": stage,
            "model": MODEL_ID,
            "request_sha256": request_sha256,
        }
    )


def _literal_string(path: Path, name: str) -> str:
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
            raise TypeError(f"{name} is not a string")
    raise KeyError(f"missing {name} in {path}")


def load_current_image_read_prompt(
    source: Path = DEFAULT_IMAGE_PROMPT_SOURCE,
) -> str:
    """Load the pinned best-available caption system prompt by AST."""

    value = _literal_string(Path(source), "IMAGE_READ_PROMPT")
    actual = _sha256(value)
    if actual != EXPECTED_CURRENT_IMAGE_READ_PROMPT_SHA256:
        raise RuntimeProtocolError(
            "current IMAGE_READ_PROMPT drift: "
            f"expected {EXPECTED_CURRENT_IMAGE_READ_PROMPT_SHA256}, got {actual}"
        )
    return value


def load_mgtm_judge_prompt(
    source: Path = DEFAULT_MGTM_PROMPT_SOURCE,
) -> str:
    """Load the original MGTM judge literal without importing its runner."""

    value = _literal_string(Path(source), "MEMEYE_JUDGE_PROMPT")
    actual = _sha256(value)
    if actual != EXPECTED_MGTM_JUDGE_PROMPT_SHA256:
        raise RuntimeProtocolError(
            "MGTM judge prompt drift: "
            f"expected {EXPECTED_MGTM_JUDGE_PROMPT_SHA256}, got {actual}"
        )
    return value


def build_mgtm_judge_prompt(
    question: str,
    ground_truth: str,
    model_output: str,
    *,
    source: Path = DEFAULT_MGTM_PROMPT_SOURCE,
) -> str:
    return load_mgtm_judge_prompt(source).format(
        question=question,
        ground_truth=ground_truth,
        model_output=model_output,
    )


def _parse_json_object(text: Any) -> dict[str, Any]:
    rendered = str(text or "").strip()
    if rendered.startswith("```"):
        lines = rendered.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        rendered = "\n".join(lines).strip()
    try:
        value = json.loads(rendered)
        return value if isinstance(value, dict) else {}
    except Exception:
        start, end = rendered.find("{"), rendered.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(rendered[start : end + 1])
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}


def parse_mgtm_judge_response(text: Any) -> JudgeVerdict:
    """Parse the historical five-level rubric and official coerced mapping."""

    value = _parse_json_object(text)
    try:
        raw = float(value["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeProtocolError("invalid MGTM judge JSON score") from exc
    if raw not in ALLOWED_JUDGE_SCORES:
        raise RuntimeProtocolError(f"judge score outside frozen five levels: {raw}")
    mapped = 0.0 if raw < 0.25 else (0.5 if raw < 0.75 else 1.0)
    return JudgeVerdict(
        raw_score=raw,
        mapped_score=mapped,
        reasoning=str(value.get("reasoning", "")),
    )


def _normalize_usage(response: Any) -> UsageRecord:
    if isinstance(response, Mapping):
        source = response.get("usageMetadata", response.get("usage", {}))
        if not isinstance(source, Mapping):
            source = {}
        prompt = source.get("promptTokenCount", source.get("prompt_tokens", 0))
        output = source.get(
            "candidatesTokenCount", source.get("output_tokens", source.get("completion_tokens", 0))
        )
        thought = source.get("thoughtsTokenCount", source.get("thought_tokens", 0))
        total = source.get("totalTokenCount", source.get("total_tokens", 0))
    else:
        source = getattr(response, "usage_metadata", None)
        prompt = getattr(source, "prompt_token_count", 0) if source else 0
        output = getattr(source, "candidates_token_count", 0) if source else 0
        thought = getattr(source, "thoughts_token_count", 0) if source else 0
        total = getattr(source, "total_token_count", 0) if source else 0
    prompt_i, output_i, thought_i = int(prompt or 0), int(output or 0), int(thought or 0)
    total_i = int(total or (prompt_i + output_i + thought_i))
    return UsageRecord(prompt_i, output_i, thought_i, total_i)


def _normalize_text(response: Any) -> str:
    if isinstance(response, Mapping):
        if response.get("text") is not None:
            return str(response.get("text") or "")
        try:
            parts = response["candidates"][0]["content"]["parts"]
            return "".join(str(part.get("text", "")) for part in parts)
        except (KeyError, IndexError, TypeError):
            return ""
    return str(getattr(response, "text", "") or "")


def _usage_to_dict(usage: UsageRecord) -> dict[str, int]:
    return asdict(usage)


def _record_to_result(record: Mapping[str, Any], *, cache_hit: bool) -> GenerationResult:
    usage_value = record.get("usage", {})
    usage = UsageRecord(
        prompt_tokens=int(usage_value.get("prompt_tokens", 0)),
        output_tokens=int(usage_value.get("output_tokens", 0)),
        thought_tokens=int(usage_value.get("thought_tokens", 0)),
        total_tokens=int(usage_value.get("total_tokens", 0)),
    )
    return GenerationResult(
        stage=str(record["stage"]),
        model=str(record["model"]),
        text=str(record["text"]),
        usage=usage,
        request_sha256=str(record["request_sha256"]),
        resume_hash=str(record["resume_hash"]),
        response_sha256=str(record["response_sha256"]),
        cache_path=str(record["cache_path"]),
        cache_hit=cache_hit,
    )


class _GeminiRESTProvider:
    """Direct official REST provider; no google-genai SDK dependency."""

    def __init__(self) -> None:
        # Even standard-library HTTP modules are resolved only after the live
        # call gate, making import-time and cache-hit audits provider-cold.
        self._request_module = importlib.import_module("urllib.request")
        self._error_module = importlib.import_module("urllib.error")
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeProtocolError("GEMINI_API_KEY is required for live calls")
        configured_base = os.environ.get("GEMINI_API_BASE", GEMINI_API_BASE).rstrip("/")
        if configured_base != GEMINI_API_BASE:
            raise RuntimeProtocolError(
                "GEMINI_API_BASE must remain the frozen official Google endpoint"
            )
        self._base = GEMINI_API_BASE
        self._timeout = float(os.environ.get("GEMINI_REQUEST_TIMEOUT_SECONDS", "300"))

    def generate_content(self, *, model: str, request: Mapping[str, Any]) -> Any:
        url = f"{self._base}/models/{model}:generateContent"
        body = _canonical_json(request).encode("utf-8")
        http_request = self._request_module.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with self._request_module.urlopen(http_request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except self._error_module.HTTPError as error:
            status = int(error.code)
            # An HTTP response proves that this attempt completed at the
            # provider boundary.  Only explicitly transient statuses retry.
            retryable = status in {429, 500, 502, 503, 504}
            raise ExplicitProviderError(
                f"Gemini GenerateContent returned HTTP {status}",
                retryable=retryable,
                status_code=status,
            ) from error
        except (TimeoutError, self._error_module.URLError) as error:
            # A transport failure cannot prove whether the request reached the
            # provider.  Retrying it would violate the exact attempt budget.
            raise IndeterminateProviderError(
                "Gemini transport outcome is indeterminate"
            ) from error


def _default_provider_factory() -> GenerateContentProvider:
    return _GeminiRESTProvider()


class Gemini35Runtime:
    """Content-addressed Gemini 3.5 runtime with explicit live-call gating."""

    def __init__(
        self,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        *,
        allow_model_calls: bool = False,
        provider_factory: ProviderFactory | None = None,
        stage_limits: Mapping[str, StageLimits] | None = None,
        lock_timeout_seconds: float = 600.0,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.allow_model_calls = bool(allow_model_calls)
        self.provider_factory = provider_factory or _default_provider_factory
        self.stage_limits = dict(stage_limits or DEFAULT_STAGE_LIMITS)
        if set(self.stage_limits) != set(STAGES):
            raise RequestContractError(
                "stage_limits must contain caption/writer/view/judge exactly"
            )
        if lock_timeout_seconds <= 0 or lock_poll_seconds <= 0:
            raise ValueError("lifecycle lock timing must be positive")
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_poll_seconds = float(lock_poll_seconds)
        self._provider: GenerateContentProvider | None = None
        self._provider_factory_calls = 0
        self._provider_calls = {stage: 0 for stage in STAGES}
        self._cache_hits = {stage: 0 for stage in STAGES}
        self._usage = {stage: UsageRecord() for stage in STAGES}
        self._state_lock = threading.Lock()
        self._identity_lock_guard = threading.Lock()
        self._identity_locks: dict[str, threading.Lock] = {}

    def _identity_lock(self, identity: str) -> threading.Lock:
        with self._identity_lock_guard:
            return self._identity_locks.setdefault(identity, threading.Lock())

    def _provider_after_gate(self, stage: str) -> GenerateContentProvider:
        with self._state_lock:
            if not self.allow_model_calls:
                raise ProviderCallBlocked(
                    "uncached Gemini request requires allow_model_calls=True"
                )
            if self._provider_calls[stage] >= self.stage_limits[stage].max_live_calls:
                raise StageBudgetExceeded(
                    f"{stage} live-call cap exhausted: "
                    f"{self.stage_limits[stage].max_live_calls}"
                )
            if self._provider is None:
                self._provider_factory_calls += 1
                self._provider = self.provider_factory()
            # Count a reserved network attempt, including determinate and
            # indeterminate provider failures, only after provider setup works.
            self._provider_calls[stage] += 1
            return self._provider

    def _response_path(self, stage: str, identity: str) -> Path:
        return self.cache_root / "responses" / stage / f"{identity}.json"

    def _plan_path(self, stage: str, identity: str) -> Path:
        return self.cache_root / "plans" / stage / f"{identity}.json"

    def _lifecycle_lock_path(self, stage: str, identity: str) -> Path:
        return self.cache_root / "lifecycle_locks" / stage / f"{identity}.lock"

    def _attempt_paths(self, stage: str, identity: str, ordinal: int) -> tuple[Path, Path]:
        root = self.cache_root / "attempts" / stage / identity
        return (
            root / f"attempt_{ordinal:03d}_request.json",
            root / f"attempt_{ordinal:03d}_response.json",
        )

    def _orphan_path(self, stage: str, identity: str) -> Path:
        return self.cache_root / "orphans" / stage / f"{identity}.json"

    def _failure_path(self, stage: str, identity: str) -> Path:
        return self.cache_root / "failures" / stage / f"{identity}.json"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CacheIntegrityError(f"invalid cache JSON: {path}") from exc
        if not isinstance(value, dict):
            raise CacheIntegrityError(f"cache record is not an object: {path}")
        return value

    @staticmethod
    def _require_record(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
        record = Gemini35Runtime._load_json(path)
        if record != dict(expected):
            raise CacheIntegrityError(f"immutable cache record drift: {path}")
        return record

    def _plan_record(
        self,
        *,
        stage: str,
        request: Mapping[str, Any],
        prompt: str,
        request_sha: str,
        resumable: str,
        caption_context_count: int | None,
        cache_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        system = request.get("systemInstruction", {})
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.stage_limits),
            "stage": stage,
            "model": MODEL_ID,
            "api": MODEL_API,
            "request_sha256": request_sha,
            "resume_hash": resumable,
            "payload_sha256": _sha256(request),
            "prompt_sha256": _sha256(prompt),
            "system_instruction_sha256": _sha256(system),
            "cache_metadata": dict(cache_metadata or {}),
            "caption_context_count": caption_context_count,
            "image_parts": sum(
                _is_image_part(part)
                for content in request["contents"]
                for part in content["parts"]
            ),
            "max_attempts": self.stage_limits[stage].max_attempts,
        }

    def _attempt_request_record(
        self,
        *,
        stage: str,
        request_sha: str,
        resumable: str,
        ordinal: int,
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.stage_limits),
            "stage": stage,
            "model": MODEL_ID,
            "request_sha256": request_sha,
            "resume_hash": resumable,
            "attempt_ordinal": ordinal,
            "attempt_identity_sha256": _sha256(
                {
                    "resume_hash": resumable,
                    "attempt_ordinal": ordinal,
                    "plan_sha256": _sha256(plan),
                }
            ),
            "plan_sha256": _sha256(plan),
        }

    def _read_cache(
        self,
        path: Path,
        *,
        stage: str,
        request_sha256_value: str,
        resume_hash_value: str,
        expected_plan: Mapping[str, Any],
    ) -> GenerationResult | None:
        if not path.is_file():
            return None
        record = self._load_json(path)
        plan_path = self._plan_path(stage, resume_hash_value)
        if not plan_path.is_file():
            raise CacheIntegrityError(f"completed response lacks frozen plan: {path}")
        self._require_record(plan_path, expected_plan)
        if self._orphan_path(stage, resume_hash_value).exists():
            raise CacheIntegrityError("completed response conflicts with orphan receipt")
        if self._failure_path(stage, resume_hash_value).exists():
            raise CacheIntegrityError("completed response conflicts with failure receipt")
        required = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.stage_limits),
            "stage": stage,
            "model": MODEL_ID,
            "request_sha256": request_sha256_value,
            "resume_hash": resume_hash_value,
            "cache_path": str(path),
        }
        for key, expected in required.items():
            if record.get(key) != expected:
                raise CacheIntegrityError(f"response cache binding drift for {key}: {path}")
        if record.get("response_sha256") != _sha256(record.get("text", "")):
            raise CacheIntegrityError(f"response text digest drift: {path}")
        ordinal = record.get("attempt_ordinal")
        if not isinstance(ordinal, int) or not 1 <= ordinal <= self.stage_limits[stage].max_attempts:
            raise CacheIntegrityError(f"completed response attempt ordinal drift: {path}")
        request_path, attempt_response_path = self._attempt_paths(
            stage, resume_hash_value, ordinal
        )
        if not request_path.is_file() or not attempt_response_path.is_file():
            raise CacheIntegrityError(f"completed response lacks attempt receipts: {path}")
        with self._state_lock:
            self._cache_hits[stage] += 1
        return _record_to_result(record, cache_hit=True)

    def generate(
        self,
        stage: str,
        prompt: str,
        *,
        system: str | None = None,
        parts: Sequence[Mapping[str, Any]] = (),
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        caption_context_count: int | None = None,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        request = build_generate_content_request(
            stage,
            prompt,
            system=system,
            parts=parts,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            stage_limits=self.stage_limits,
            caption_context_count=caption_context_count,
        )
        request_sha = request_identity(
            stage=stage,
            request=request,
            stage_limits=self.stage_limits,
            cache_metadata=cache_metadata,
        )
        resumable = resume_hash(
            stage=stage,
            request_sha256=request_sha,
            stage_limits=self.stage_limits,
        )
        response_path = self._response_path(stage, resumable)
        plan = self._plan_record(
            stage=stage,
            request=request,
            prompt=prompt,
            request_sha=request_sha,
            resumable=resumable,
            caption_context_count=caption_context_count,
            cache_metadata=cache_metadata,
        )

        # The in-process lock avoids needless local OS-lock contention.  The OS
        # lease is the authoritative cross-process lifecycle owner and is held
        # before any request/response/failure state read through final fsync.
        with self._identity_lock(resumable):
            lock_path = self._lifecycle_lock_path(stage, resumable)
            with acquire_request_lifecycle_lock(
                lock_path,
                timeout_seconds=self.lock_timeout_seconds,
                poll_seconds=self.lock_poll_seconds,
            ):
                hit = self._read_cache(
                    response_path,
                    stage=stage,
                    request_sha256_value=request_sha,
                    resume_hash_value=resumable,
                    expected_plan=plan,
                )
                if hit is not None:
                    return hit

                plan_path = self._plan_path(stage, resumable)
                if plan_path.exists():
                    self._require_record(plan_path, plan)
                elif self._failure_path(stage, resumable).exists() or self._orphan_path(
                    stage, resumable
                ).exists():
                    raise CacheIntegrityError("terminal receipt exists without request plan")

                retry_errors: list[dict[str, Any]] = []
                next_ordinal: int | None = None
                for ordinal in range(1, self.stage_limits[stage].max_attempts + 1):
                    attempt_request_path, attempt_response_path = self._attempt_paths(
                        stage, resumable, ordinal
                    )
                    expected_request = self._attempt_request_record(
                        stage=stage,
                        request_sha=request_sha,
                        resumable=resumable,
                        ordinal=ordinal,
                        plan=plan,
                    )
                    if not attempt_request_path.exists():
                        if attempt_response_path.exists():
                            raise CacheIntegrityError(
                                "attempt response exists without request intent"
                            )
                        next_ordinal = ordinal
                        break
                    self._require_record(attempt_request_path, expected_request)
                    if not attempt_response_path.exists():
                        orphan = {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "protocol_sha256": protocol_sha256(self.stage_limits),
                            "stage": stage,
                            "model": MODEL_ID,
                            "request_sha256": request_sha,
                            "resume_hash": resumable,
                            "attempt_ordinal": ordinal,
                            "failure_kind": "orphaned_indeterminate_request_intent",
                            "provider_recall_allowed": False,
                        }
                        write_once_json(self._orphan_path(stage, resumable), orphan)
                        raise OrphanedCallError(
                            "orphaned indeterminate request intent; provider recall prohibited"
                        )
                    attempt_response = self._load_json(attempt_response_path)
                    binding = {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "protocol_sha256": protocol_sha256(self.stage_limits),
                        "stage": stage,
                        "model": MODEL_ID,
                        "request_sha256": request_sha,
                        "resume_hash": resumable,
                        "attempt_ordinal": ordinal,
                        "attempt_identity_sha256": expected_request[
                            "attempt_identity_sha256"
                        ],
                    }
                    for key, expected in binding.items():
                        if attempt_response.get(key) != expected:
                            raise CacheIntegrityError(
                                f"attempt response binding drift for {key}"
                            )
                    outcome = attempt_response.get("outcome_kind")
                    if outcome == "raw_response":
                        if attempt_response.get("response_sha256") != _sha256(
                            attempt_response.get("text", "")
                        ):
                            raise CacheIntegrityError("attempt response text digest drift")
                        final_record = {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "protocol_sha256": protocol_sha256(self.stage_limits),
                            "stage": stage,
                            "model": MODEL_ID,
                            "request_sha256": request_sha,
                            "resume_hash": resumable,
                            "attempt_ordinal": ordinal,
                            "response_sha256": attempt_response["response_sha256"],
                            "text": attempt_response["text"],
                            "usage": attempt_response["usage"],
                            "cache_path": str(response_path),
                        }
                        write_once_json(response_path, final_record)
                        with self._state_lock:
                            self._cache_hits[stage] += 1
                        return _record_to_result(final_record, cache_hit=True)
                    if outcome != "explicit_provider_error":
                        raise CacheIntegrityError("unknown attempt response outcome")
                    retryable = attempt_response.get("retryable") is True
                    retry_errors.append(attempt_response)
                    if not retryable:
                        raise ExplicitProviderError(
                            str(attempt_response.get("error_message", "provider error")),
                            retryable=False,
                            status_code=attempt_response.get("status_code"),
                        )

                if next_ordinal is None:
                    failure = {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "protocol_sha256": protocol_sha256(self.stage_limits),
                        "stage": stage,
                        "model": MODEL_ID,
                        "request_sha256": request_sha,
                        "resume_hash": resumable,
                        "failure_kind": "explicit_retryable_attempts_exhausted",
                        "attempts": len(retry_errors),
                        "provider_recall_allowed": False,
                        "attempt_response_sha256": [
                            _sha256(value) for value in retry_errors
                        ],
                    }
                    write_once_json(self._failure_path(stage, resumable), failure)
                    raise ProviderRetriesExhausted(
                        f"{stage} exhausted {len(retry_errors)} explicit retry attempts"
                    )

                provider = self._provider_after_gate(stage)
                if not plan_path.exists():
                    write_once_json(plan_path, plan)

                # Continue under the same lifecycle lease for controlled retry
                # ordinals.  Each explicit provider error closes one ordinal.
                ordinal = next_ordinal
                while ordinal <= self.stage_limits[stage].max_attempts:
                    attempt_request_path, attempt_response_path = self._attempt_paths(
                        stage, resumable, ordinal
                    )
                    attempt_request = self._attempt_request_record(
                        stage=stage,
                        request_sha=request_sha,
                        resumable=resumable,
                        ordinal=ordinal,
                        plan=plan,
                    )
                    write_once_json(attempt_request_path, attempt_request)
                    try:
                        response = provider.generate_content(model=MODEL_ID, request=request)
                    except ExplicitProviderError as error:
                        error_record = {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "protocol_sha256": protocol_sha256(self.stage_limits),
                            "stage": stage,
                            "model": MODEL_ID,
                            "request_sha256": request_sha,
                            "resume_hash": resumable,
                            "attempt_ordinal": ordinal,
                            "attempt_identity_sha256": attempt_request[
                                "attempt_identity_sha256"
                            ],
                            "outcome_kind": "explicit_provider_error",
                            "retryable": error.retryable,
                            "status_code": error.status_code,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "error_sha256": _sha256(str(error)),
                        }
                        write_once_json(attempt_response_path, error_record)
                        if not error.retryable:
                            raise
                        ordinal += 1
                        if ordinal > self.stage_limits[stage].max_attempts:
                            break
                        provider = self._provider_after_gate(stage)
                        continue
                    except Exception as error:
                        orphan = {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "protocol_sha256": protocol_sha256(self.stage_limits),
                            "stage": stage,
                            "model": MODEL_ID,
                            "request_sha256": request_sha,
                            "resume_hash": resumable,
                            "attempt_ordinal": ordinal,
                            "failure_kind": "orphaned_indeterminate_request_intent",
                            "provider_recall_allowed": False,
                        }
                        write_once_json(self._orphan_path(stage, resumable), orphan)
                        raise OrphanedCallError(
                            "provider outcome indeterminate; provider recall prohibited"
                        ) from error

                    text = _normalize_text(response)
                    usage = _normalize_usage(response)
                    with self._state_lock:
                        old = self._usage[stage]
                        self._usage[stage] = UsageRecord(
                            old.prompt_tokens + usage.prompt_tokens,
                            old.output_tokens + usage.output_tokens,
                            old.thought_tokens + usage.thought_tokens,
                            old.total_tokens + usage.total_tokens,
                        )
                    if not text.strip():
                        empty_error = ExplicitProviderError(
                            "Gemini returned an empty determinate response",
                            retryable=True,
                            status_code=None,
                        )
                        error_record = {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "protocol_sha256": protocol_sha256(self.stage_limits),
                            "stage": stage,
                            "model": MODEL_ID,
                            "request_sha256": request_sha,
                            "resume_hash": resumable,
                            "attempt_ordinal": ordinal,
                            "attempt_identity_sha256": attempt_request[
                                "attempt_identity_sha256"
                            ],
                            "outcome_kind": "explicit_provider_error",
                            "retryable": True,
                            "status_code": None,
                            "error_type": "EmptyProviderResponse",
                            "error_message": str(empty_error),
                            "error_sha256": _sha256(str(empty_error)),
                            "usage": _usage_to_dict(usage),
                        }
                        write_once_json(attempt_response_path, error_record)
                        ordinal += 1
                        if ordinal > self.stage_limits[stage].max_attempts:
                            break
                        provider = self._provider_after_gate(stage)
                        continue

                    attempt_response = {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "protocol_sha256": protocol_sha256(self.stage_limits),
                        "stage": stage,
                        "model": MODEL_ID,
                        "request_sha256": request_sha,
                        "resume_hash": resumable,
                        "attempt_ordinal": ordinal,
                        "attempt_identity_sha256": attempt_request[
                            "attempt_identity_sha256"
                        ],
                        "outcome_kind": "raw_response",
                        "response_sha256": _sha256(text),
                        "text": text,
                        "usage": _usage_to_dict(usage),
                    }
                    write_once_json(attempt_response_path, attempt_response)
                    final_record = {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "protocol_sha256": protocol_sha256(self.stage_limits),
                        "stage": stage,
                        "model": MODEL_ID,
                        "request_sha256": request_sha,
                        "resume_hash": resumable,
                        "attempt_ordinal": ordinal,
                        "response_sha256": attempt_response["response_sha256"],
                        "text": text,
                        "usage": _usage_to_dict(usage),
                        "cache_path": str(response_path),
                    }
                    # Final completion is the last write under the OS lease.
                    write_once_json(response_path, final_record)
                    return _record_to_result(final_record, cache_hit=False)

                failure = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "protocol_sha256": protocol_sha256(self.stage_limits),
                    "stage": stage,
                    "model": MODEL_ID,
                    "request_sha256": request_sha,
                    "resume_hash": resumable,
                    "failure_kind": "explicit_retryable_attempts_exhausted",
                    "attempts": self.stage_limits[stage].max_attempts,
                    "provider_recall_allowed": False,
                }
                write_once_json(self._failure_path(stage, resumable), failure)
                raise ProviderRetriesExhausted(
                    f"{stage} exhausted {self.stage_limits[stage].max_attempts} attempts"
                )

    def judge(
        self,
        question: str,
        ground_truth: str,
        model_output: str,
        *,
        prompt_source: Path = DEFAULT_MGTM_PROMPT_SOURCE,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[GenerationResult, JudgeVerdict]:
        prompt = build_mgtm_judge_prompt(
            question,
            ground_truth,
            model_output,
            source=prompt_source,
        )
        result = self.generate(
            "judge",
            prompt,
            json_mode=True,
            max_output_tokens=400,
            cache_metadata=cache_metadata,
        )
        return result, parse_mgtm_judge_response(result.text)

    def usage_receipt(self) -> dict[str, Any]:
        """Return a key-free, deterministic accounting snapshot."""

        with self._state_lock:
            stages = {
                stage: {
                    "provider_calls": self._provider_calls[stage],
                    "cache_hits": self._cache_hits[stage],
                    "usage": _usage_to_dict(self._usage[stage]),
                    "limits": asdict(self.stage_limits[stage]),
                    "remaining_live_calls": (
                        self.stage_limits[stage].max_live_calls
                        - self._provider_calls[stage]
                    ),
                }
                for stage in STAGES
            }
            return {
                "schema_version": PROTOCOL_VERSION,
                "model": MODEL_ID,
                "cache_root": str(self.cache_root),
                "allow_model_calls": self.allow_model_calls,
                "provider_loaded": self._provider is not None,
                "provider_factory_calls": self._provider_factory_calls,
                "stages": stages,
            }


def zero_call_preflight(
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    judge_prompt_source: Path = DEFAULT_MGTM_PROMPT_SOURCE,
) -> dict[str, Any]:
    """Verify all frozen contracts without touching a provider or cache file."""

    runtime = Gemini35Runtime(cache_root, allow_model_calls=False)
    image = inline_image_part(b"not-a-real-image", "image/png")
    requests = {
        "caption": build_generate_content_request(
            "caption",
            "caption prompt",
            parts=[image],
            caption_context_count=2,
        ),
        "writer": build_generate_content_request("writer", "writer prompt", json_mode=True),
        "view": build_generate_content_request("view", "view prompt", json_mode=True),
        "judge": build_generate_content_request("judge", "judge prompt", json_mode=True),
    }
    for stage, request in requests.items():
        assert_request_contract(
            request,
            stage=stage,
            caption_context_count=2 if stage == "caption" else None,
            token_limit=DEFAULT_STAGE_LIMITS[stage].max_output_tokens,
        )
    judge_prompt = load_mgtm_judge_prompt(judge_prompt_source)
    image_read_prompt = load_current_image_read_prompt()
    receipt = runtime.usage_receipt()
    return {
        "status": "zero_call_preflight_pass",
        "protocol": protocol_manifest(),
        "protocol_sha256": protocol_sha256(),
        "request_sha256": {
            stage: request_identity(stage=stage, request=request)
            for stage, request in requests.items()
        },
        "judge_prompt_sha256": _sha256(judge_prompt),
        "image_read_prompt_sha256": _sha256(image_read_prompt),
        "provider_loaded": receipt["provider_loaded"],
        "provider_factory_calls": receipt["provider_factory_calls"],
        "provider_calls": sum(
            values["provider_calls"] for values in receipt["stages"].values()
        ),
        "network_calls": 0,
        "google_genai_sdk_imported_by_runtime": False,
        "cache_files_written": 0,
    }


__all__ = [
    "ALLOWED_JUDGE_SCORES",
    "CACHE_SCHEMA_VERSION",
    "CAPTION_LINEAGE",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_IMAGE_PROMPT_SOURCE",
    "DEFAULT_MGTM_PROMPT_SOURCE",
    "DEFAULT_STAGE_LIMITS",
    "EXPECTED_MGTM_JUDGE_PROMPT_SHA256",
    "EXPECTED_CURRENT_IMAGE_READ_PROMPT_SHA256",
    "ExplicitProviderError",
    "FORBIDDEN_GENERATION_FIELDS",
    "GenerationResult",
    "GEMINI_API_BASE",
    "Gemini35Runtime",
    "JudgeVerdict",
    "IndeterminateProviderError",
    "LifecycleBusyError",
    "MEDIA_RESOLUTION_HIGH",
    "MODEL_API",
    "MODEL_ID",
    "PROTOCOL_VERSION",
    "ProviderCallBlocked",
    "ProviderRetriesExhausted",
    "OrphanedCallError",
    "RequestContractError",
    "RuntimeProtocolError",
    "STAGES",
    "StageBudgetExceeded",
    "StageLimits",
    "THINKING_LEVEL",
    "UsageRecord",
    "assert_request_contract",
    "build_generate_content_request",
    "build_mgtm_judge_prompt",
    "acquire_request_lifecycle_lock",
    "inline_image_part",
    "load_mgtm_judge_prompt",
    "load_current_image_read_prompt",
    "parse_mgtm_judge_response",
    "protocol_manifest",
    "protocol_sha256",
    "request_identity",
    "resume_hash",
    "text_part",
    "write_once_json",
    "zero_call_preflight",
]

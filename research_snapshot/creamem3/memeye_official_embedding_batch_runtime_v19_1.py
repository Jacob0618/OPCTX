"""Durable official-Google BatchEmbedContents runtime for MemEye v19.1.

The runtime sends one physical AI Studio ``batchEmbedContents`` request for
each logical batch of 1..100 texts.  Provider-visible text is exactly
``text[:8000]`` while the cache identity binds both complete and visible
forms.  Optional task/input/output-dimension fields are never accepted or
sent.  The direct REST transport has no internal retry or redirect path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from creamem3.memeye_g35_runtime_v19 import (
    CacheIntegrityError,
    ExplicitProviderError,
    IndeterminateProviderError,
    OrphanedCallError,
    ProviderCallBlocked,
    ProviderRetriesExhausted,
    RequestContractError,
    StageBudgetExceeded,
    acquire_request_lifecycle_lock,
    write_once_json,
)


MODEL_ID = "gemini-embedding-001"
OFFICIAL_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
OPERATION = "batchEmbedContents"
MAX_BATCH_SIZE = 100
VISIBLE_TEXT_CHARACTER_LIMIT = 8000
NATIVE_DIMENSIONS = 3072
PROTOCOL_VERSION = (
    "creamem3.memeye_official_google_embedding_batch_runtime.v19.1"
)
CACHE_SCHEMA_VERSION = (
    "creamem3.memeye_official_google_embedding_batch_cache.v19.1"
)
RUNTIME_PATH = Path(__file__).resolve()
ZERO_SHA256 = "0" * 64


class OfficialEmbeddingRuntimeError(RuntimeError):
    """An official embedding route, cache, or lifecycle invariant drifted."""


@dataclass(frozen=True)
class EmbeddingLimits:
    max_live_calls: int
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_live_calls < 0:
            raise ValueError("max_live_calls must be non-negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


@dataclass(frozen=True)
class ProviderEnvelope:
    payload: Mapping[str, Any]
    transport_receipt: Mapping[str, Any]
    raw_response_body: bytes


@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: tuple[tuple[float, ...], ...]
    request_sha256: str
    resume_hash: str
    response_sha256: str
    vector_sha256s: tuple[str, ...]
    vectors_sha256: str
    full_text_sha256s: tuple[str, ...]
    visible_text_sha256s: tuple[str, ...]
    cache_path: str
    cache_hit: bool
    successful_attempt_ordinal: int


class BatchEmbedContentsProvider(Protocol):
    transport: str
    vertexai: bool
    base_url: str
    api_version: str
    operation: str
    auth: str
    retry_attempts: int
    redirects_allowed: bool

    def batch_embed_contents(
        self, *, request: Mapping[str, Any]
    ) -> ProviderEnvelope: ...


ProviderFactory = Callable[[], BatchEmbedContentsProvider]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def route_contract() -> dict[str, Any]:
    return {
        "transport": "direct_rest_google_ai_studio",
        "vertexai": False,
        "base_url": OFFICIAL_ENDPOINT,
        "api_version": "v1beta",
        "operation": OPERATION,
        "model": MODEL_ID,
        "auth": "x-goog-api-key_header",
        "retry_attempts": 1,
        "redirects_allowed": False,
    }


def expected_url() -> str:
    return f"{OFFICIAL_ENDPOINT}/models/{MODEL_ID}:{OPERATION}"


def prepare_batch_inputs(
    texts: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise RequestContractError("embedding batch must be a text sequence")
    if not 1 <= len(texts) <= MAX_BATCH_SIZE:
        raise RequestContractError("embedding batch size must be in [1, 100]")
    visible: list[str] = []
    identities: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise RequestContractError(
                f"embedding batch item {index} must be non-empty text"
            )
        truncated = text[:VISIBLE_TEXT_CHARACTER_LIMIT]
        if not truncated.strip():
            raise RequestContractError(
                f"embedding batch item {index} is empty after truncation"
            )
        visible.append(truncated)
        identities.append(
            {
                "index": index,
                "full_text_sha256": _sha256(text),
                "visible_text_sha256": _sha256(truncated),
                "full_characters": len(text),
                "visible_characters": len(truncated),
                "truncated": len(text) > VISIBLE_TEXT_CHARACTER_LIMIT,
            }
        )
    return visible, identities


def build_batch_embed_contents_request(
    texts: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible, identities = prepare_batch_inputs(texts)
    request = {
        "requests": [
            {
                "model": f"models/{MODEL_ID}",
                "content": {"parts": [{"text": text}]},
            }
            for text in visible
        ]
    }
    assert_batch_embed_contents_request(request)
    return request, identities


def assert_batch_embed_contents_request(request: Mapping[str, Any]) -> None:
    if set(request) != {"requests"}:
        raise RequestContractError("BatchEmbedContents top-level fields drift")
    requests = request.get("requests")
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_BATCH_SIZE:
        raise RequestContractError("BatchEmbedContents requires 1..100 requests")
    for index, item in enumerate(requests):
        if not isinstance(item, Mapping) or set(item) != {"model", "content"}:
            raise RequestContractError(f"embedding request item {index} fields drift")
        if item.get("model") != f"models/{MODEL_ID}":
            raise RequestContractError(f"embedding request item {index} model drift")
        content = item.get("content")
        if not isinstance(content, Mapping) or set(content) != {"parts"}:
            raise RequestContractError(f"embedding request item {index} content drift")
        parts = content.get("parts")
        if (
            not isinstance(parts, list)
            or len(parts) != 1
            or not isinstance(parts[0], Mapping)
            or set(parts[0]) != {"text"}
            or not isinstance(parts[0].get("text"), str)
            or not parts[0]["text"].strip()
            or len(parts[0]["text"]) > VISIBLE_TEXT_CHARACTER_LIMIT
        ):
            raise RequestContractError(f"embedding request item {index} text drift")


def protocol_manifest(limits: EmbeddingLimits) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION,
        "model": MODEL_ID,
        "api": OPERATION,
        "route": route_contract(),
        "batch_size": MAX_BATCH_SIZE,
        "provider_visible_character_limit": VISIBLE_TEXT_CHARACTER_LIMIT,
        "native_dimensions": NATIVE_DIMENSIONS,
        "provider_request_fields_omitted": [
            "task_type",
            "input_type",
            "output_dimensionality",
        ],
        "max_live_calls": limits.max_live_calls,
        "max_attempts_per_batch": limits.max_attempts,
        "durable_global_budget_census": True,
        "indeterminate_transport_policy": "orphan_no_automatic_recall",
        "formal_cache_reuse_from_smoke": False,
    }


def protocol_sha256(limits: EmbeddingLimits) -> str:
    return _sha256(protocol_manifest(limits))


def request_identity(
    texts: Sequence[str],
    *,
    cache_metadata: Mapping[str, Any],
    limits: EmbeddingLimits,
) -> dict[str, Any]:
    request, item_identities = build_batch_embed_contents_request(texts)
    full = list(texts)
    visible = [
        item["content"]["parts"][0]["text"] for item in request["requests"]
    ]
    identity = _sha256(
        {
            "schema_version": PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256(limits),
            "model": MODEL_ID,
            "route": route_contract(),
            "provider_request_sha256": _sha256(request),
            "full_contents_sha256": _sha256(full),
            "visible_contents_sha256": _sha256(visible),
            "item_identities": item_identities,
            "cache_metadata": dict(cache_metadata),
        }
    )
    resumable = _sha256(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity": identity,
            "protocol_sha256": protocol_sha256(limits),
        }
    )
    return {
        "identity": identity,
        "resume_hash": resumable,
        "request": request,
        "provider_request_sha256": _sha256(request),
        "item_identities": item_identities,
        "full_contents_sha256": _sha256(full),
        "visible_contents_sha256": _sha256(visible),
        "cache_metadata": dict(cache_metadata),
    }


def _actual_route(provider: Any) -> dict[str, Any]:
    return {
        "transport": getattr(provider, "transport", None),
        "vertexai": getattr(provider, "vertexai", None),
        "base_url": getattr(provider, "base_url", None),
        "api_version": getattr(provider, "api_version", None),
        "operation": getattr(provider, "operation", None),
        "model": MODEL_ID,
        "auth": getattr(provider, "auth", None),
        "retry_attempts": getattr(provider, "retry_attempts", None),
        "redirects_allowed": getattr(provider, "redirects_allowed", None),
    }


class OfficialGoogleBatchEmbedContentsProvider:
    """One HTTP attempt, no redirects, no SDK routing or retry ambiguity."""

    def __init__(self) -> None:
        request_module = importlib.import_module("urllib.request")
        self._error_module = importlib.import_module("urllib.error")

        class _RejectRedirect(request_module.HTTPRedirectHandler):
            def redirect_request(self, *args: Any, **kwargs: Any) -> None:
                return None

        self._request_module = request_module
        self._opener = request_module.build_opener(_RejectRedirect())
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ProviderCallBlocked("GEMINI_API_KEY is required")
        configured_base = os.environ.get("GEMINI_API_BASE", OFFICIAL_ENDPOINT).rstrip(
            "/"
        )
        if configured_base != OFFICIAL_ENDPOINT:
            raise ProviderCallBlocked("official embedding endpoint override is forbidden")
        self.transport = "direct_rest_google_ai_studio"
        self.vertexai = False
        self.base_url = OFFICIAL_ENDPOINT
        self.api_version = "v1beta"
        self.operation = OPERATION
        self.auth = "x-goog-api-key_header"
        self.retry_attempts = 1
        self.redirects_allowed = False

    def batch_embed_contents(
        self, *, request: Mapping[str, Any]
    ) -> ProviderEnvelope:
        assert_batch_embed_contents_request(request)
        body = _canonical_json(request).encode("utf-8")
        http_request = self._request_module.Request(
            expected_url(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with self._opener.open(http_request, timeout=300) as response:
                raw_body = response.read()
                status = int(getattr(response, "status", 200))
        except self._error_module.HTTPError as error:
            status = int(error.code)
            raise ExplicitProviderError(
                f"BatchEmbedContents returned HTTP {status}",
                retryable=status in {429, 500, 502, 503, 504},
                status_code=status,
            ) from error
        except (TimeoutError, self._error_module.URLError) as error:
            raise IndeterminateProviderError(
                "BatchEmbedContents transport outcome is indeterminate"
            ) from error
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as error:
            raise ExplicitProviderError(
                "BatchEmbedContents returned invalid JSON",
                retryable=False,
                status_code=status,
            ) from error
        if not isinstance(payload, Mapping):
            raise ExplicitProviderError(
                "BatchEmbedContents returned a non-object envelope",
                retryable=False,
                status_code=status,
            )
        return ProviderEnvelope(
            payload=dict(payload),
            transport_receipt={
                "route": route_contract(),
                "url": expected_url(),
                "method": "POST",
                "request_body_sha256": hashlib.sha256(body).hexdigest(),
                "response_body_sha256": hashlib.sha256(raw_body).hexdigest(),
                "http_status": status,
                "physical_http_attempts": 1,
                "redirects_followed": 0,
            },
            raw_response_body=raw_body,
        )


def _parse_provider_envelope(
    envelope: ProviderEnvelope,
    *,
    request: Mapping[str, Any],
) -> tuple[list[list[float]], dict[str, Any], dict[str, Any]]:
    if not isinstance(envelope, ProviderEnvelope):
        raise ExplicitProviderError(
            "provider did not return a bound transport envelope",
            retryable=False,
            status_code=200,
        )
    payload = dict(envelope.payload)
    transport = dict(envelope.transport_receipt)
    if not isinstance(envelope.raw_response_body, bytes):
        raise ExplicitProviderError(
            "embedding raw response body is not bytes",
            retryable=False,
            status_code=200,
        )
    try:
        reparsed_payload = json.loads(envelope.raw_response_body.decode("utf-8"))
    except Exception as error:
        raise ExplicitProviderError(
            "embedding raw response body is not valid UTF-8 JSON",
            retryable=False,
            status_code=200,
        ) from error
    if reparsed_payload != payload:
        raise ExplicitProviderError(
            "embedding parsed payload/raw response body drift",
            retryable=False,
            status_code=200,
        )
    required_transport = {
        "route": route_contract(),
        "url": expected_url(),
        "method": "POST",
        "request_body_sha256": _sha256(request),
        "http_status": 200,
        "physical_http_attempts": 1,
        "redirects_followed": 0,
    }
    for key, expected in required_transport.items():
        if transport.get(key) != expected:
            raise ExplicitProviderError(
                f"embedding transport receipt drift: {key}",
                retryable=False,
                status_code=200,
            )
    if set(transport) != set(required_transport) | {"response_body_sha256"}:
        raise ExplicitProviderError(
            "embedding transport receipt field-set drift",
            retryable=False,
            status_code=200,
        )
    if not _is_sha256(transport.get("response_body_sha256")):
        raise ExplicitProviderError(
            "embedding response body digest drift",
            retryable=False,
            status_code=200,
        )
    if transport["response_body_sha256"] != hashlib.sha256(
        envelope.raw_response_body
    ).hexdigest():
        raise ExplicitProviderError(
            "embedding raw response body digest mismatch",
            retryable=False,
            status_code=200,
        )
    embeddings = payload.get("embeddings")
    expected_count = len(request["requests"])
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        raise ExplicitProviderError(
            "embedding vector count/order envelope drift",
            retryable=False,
            status_code=200,
        )
    vectors: list[list[float]] = []
    for index, item in enumerate(embeddings):
        if not isinstance(item, Mapping) or not isinstance(item.get("values"), list):
            raise ExplicitProviderError(
                f"embedding vector {index} envelope drift",
                retryable=False,
                status_code=200,
            )
        try:
            vector = [float(value) for value in item["values"]]
        except Exception as error:
            raise ExplicitProviderError(
                f"embedding vector {index} value drift",
                retryable=False,
                status_code=200,
            ) from error
        if len(vector) != NATIVE_DIMENSIONS or not all(
            math.isfinite(value) for value in vector
        ):
            raise ExplicitProviderError(
                f"embedding vector {index} dimension/nonfinite drift",
                retryable=False,
                status_code=200,
            )
        vectors.append(vector)
    return vectors, payload, transport


def synthetic_provider_envelope(
    request: Mapping[str, Any],
    vectors: Sequence[Sequence[float]],
) -> ProviderEnvelope:
    """Build a deterministic fake envelope for zero-network tests only."""

    payload = {"embeddings": [{"values": list(vector)} for vector in vectors]}
    response_body = _canonical_json(payload).encode("utf-8")
    return ProviderEnvelope(
        payload=payload,
        transport_receipt={
            "route": route_contract(),
            "url": expected_url(),
            "method": "POST",
            "request_body_sha256": _sha256(request),
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "http_status": 200,
            "physical_http_attempts": 1,
            "redirects_followed": 0,
        },
        raw_response_body=response_body,
    )


class OfficialGoogleBatchEmbeddingRuntime:
    """Content-addressed, cross-process bounded official embedding runtime."""

    def __init__(
        self,
        cache_root: Path,
        *,
        containment_root: Path,
        limits: EmbeddingLimits,
        allow_model_calls: bool,
        budget_binding_sha256: str,
        provider_factory: ProviderFactory | None = None,
        lock_timeout_seconds: float = 600.0,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.containment_root = Path(containment_root)
        try:
            self.cache_root.resolve(strict=False).relative_to(
                self.containment_root.resolve(strict=False)
            )
        except ValueError as error:
            raise RequestContractError("embedding cache escaped containment root") from error
        if self.cache_root.resolve(strict=False) == self.containment_root.resolve(
            strict=False
        ):
            raise RequestContractError("embedding cache must be a strict descendant")
        if not _is_sha256(budget_binding_sha256):
            raise RequestContractError("embedding budget binding must be a SHA-256")
        if allow_model_calls and budget_binding_sha256 == ZERO_SHA256:
            raise RequestContractError("live embedding requires a nonzero budget binding")
        if lock_timeout_seconds <= 0 or lock_poll_seconds <= 0:
            raise ValueError("lock timing must be positive")
        self.limits = limits
        self.allow_model_calls = bool(allow_model_calls)
        self.budget_binding_sha256 = budget_binding_sha256
        self.provider_factory = provider_factory or OfficialGoogleBatchEmbedContentsProvider
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_poll_seconds = float(lock_poll_seconds)
        self._provider: BatchEmbedContentsProvider | None = None
        self.provider_calls = 0
        self.cache_hits = 0

    def _paths(self, resume_hash_value: str) -> dict[str, Path]:
        return {
            "plan": self.cache_root / "plans" / f"{resume_hash_value}.json",
            "final": self.cache_root / "responses" / f"{resume_hash_value}.json",
            "orphan": self.cache_root / "orphans" / f"{resume_hash_value}.json",
            "failure": self.cache_root / "failures" / f"{resume_hash_value}.json",
            "identity_lock": (
                self.cache_root / "lifecycle_locks" / f"{resume_hash_value}.lock"
            ),
            "budget_lock": self.cache_root / "budget_lifecycle" / "global.lock",
        }

    def _attempt_paths(self, resume_hash_value: str, ordinal: int) -> tuple[Path, Path]:
        root = self.cache_root / "attempts" / resume_hash_value
        return (
            root / f"attempt_{ordinal:03d}_request.json",
            root / f"attempt_{ordinal:03d}_response.json",
        )

    @staticmethod
    def _load(path: Path, label: str) -> dict[str, Any]:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as error:
            raise CacheIntegrityError(f"invalid {label}: {path}") from error
        if not isinstance(value, dict):
            raise CacheIntegrityError(f"{label} is not a JSON object")
        return value

    def _expected_plan(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.limits),
            "model": MODEL_ID,
            "route": route_contract(),
            "identity": contract["identity"],
            "resume_hash": contract["resume_hash"],
            "provider_request_sha256": contract["provider_request_sha256"],
            "full_contents_sha256": contract["full_contents_sha256"],
            "visible_contents_sha256": contract["visible_contents_sha256"],
            "item_identities": contract["item_identities"],
            "cache_metadata": contract["cache_metadata"],
            "batch_size": len(contract["item_identities"]),
            "native_dimensions": NATIVE_DIMENSIONS,
            "budget_binding_sha256": self.budget_binding_sha256,
            "approved_provider_attempt_hard_ceiling": self.limits.max_live_calls,
            "max_attempts_per_batch": self.limits.max_attempts,
        }

    def _expected_intent(
        self,
        contract: Mapping[str, Any],
        plan: Mapping[str, Any],
        ordinal: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.limits),
            "model": MODEL_ID,
            "route": route_contract(),
            "identity": contract["identity"],
            "resume_hash": contract["resume_hash"],
            "attempt_ordinal": ordinal,
            "attempt_identity_sha256": _sha256(
                {
                    "resume_hash": contract["resume_hash"],
                    "attempt_ordinal": ordinal,
                    "plan_sha256": _sha256(plan),
                }
            ),
            "plan_sha256": _sha256(plan),
            "provider_request_sha256": contract["provider_request_sha256"],
            "provider_request": contract["request"],
            "budget_binding_sha256": self.budget_binding_sha256,
            "approved_provider_attempt_hard_ceiling": self.limits.max_live_calls,
        }

    def _durable_attempt_count(self) -> int:
        attempts = self.cache_root / "attempts"
        if not attempts.is_dir():
            return 0
        return len(list(attempts.rglob("attempt_*_request.json")))

    def _reserve_intent(
        self,
        path: Path,
        intent: Mapping[str, Any],
    ) -> None:
        paths = self._paths(str(intent["resume_hash"]))
        with acquire_request_lifecycle_lock(
            paths["budget_lock"],
            timeout_seconds=self.lock_timeout_seconds,
            poll_seconds=self.lock_poll_seconds,
        ):
            count = self._durable_attempt_count()
            if count >= self.limits.max_live_calls:
                raise StageBudgetExceeded(
                    "official embedding durable provider-attempt ceiling exhausted"
                )
            write_once_json(path, intent)

    def _provider_after_intent(self) -> BatchEmbedContentsProvider:
        if not self.allow_model_calls:
            raise ProviderCallBlocked("official embedding provider calls are disabled")
        if self._provider is None:
            self._provider = self.provider_factory()
        if _actual_route(self._provider) != route_contract():
            raise ProviderCallBlocked("official embedding actual route drift")
        self.provider_calls += 1
        return self._provider

    def _success_raw(
        self,
        *,
        intent: Mapping[str, Any],
        envelope: ProviderEnvelope,
    ) -> tuple[dict[str, Any], list[list[float]]]:
        vectors, payload, transport = _parse_provider_envelope(
            envelope, request=intent["provider_request"]
        )
        vector_sha256s = [_sha256(vector) for vector in vectors]
        raw = {
            **intent,
            "outcome_kind": "raw_response",
            "actual_route": route_contract(),
            "transport_receipt": transport,
            "provider_response": payload,
            "provider_response_body_base64": base64.b64encode(
                envelope.raw_response_body
            ).decode("ascii"),
            "response_sha256": _sha256(payload),
            "vector_sha256s": vector_sha256s,
            "vectors_sha256": _sha256(vectors),
            "vector_count": len(vectors),
            "vector_dimension": NATIVE_DIMENSIONS,
        }
        return raw, vectors

    @staticmethod
    def _validate_explicit_error_response(
        intent: Mapping[str, Any], response: Mapping[str, Any]
    ) -> None:
        if (
            set(response)
            != set(intent)
            | {
                "outcome_kind",
                "retryable",
                "status_code",
                "error_type",
                "error_sha256",
            }
            or {key: response.get(key) for key in intent} != dict(intent)
            or response.get("outcome_kind") != "explicit_provider_error"
            or not isinstance(response.get("retryable"), bool)
            or not isinstance(response.get("error_type"), str)
            or not _is_sha256(response.get("error_sha256"))
        ):
            raise CacheIntegrityError("embedding explicit-error receipt drift")

    def _final_from_raw(
        self,
        *,
        contract: Mapping[str, Any],
        plan: Mapping[str, Any],
        intent: Mapping[str, Any],
        raw: Mapping[str, Any],
        raw_path: Path,
        final_path: Path,
    ) -> dict[str, Any]:
        try:
            raw_response_body = base64.b64decode(
                str(raw.get("provider_response_body_base64", "")),
                validate=True,
            )
        except Exception as error:
            raise CacheIntegrityError(
                "embedding cached raw response body base64 drift"
            ) from error
        envelope = ProviderEnvelope(
            payload=dict(raw.get("provider_response", {})),
            transport_receipt=dict(raw.get("transport_receipt", {})),
            raw_response_body=raw_response_body,
        )
        try:
            vectors, payload, transport = _parse_provider_envelope(
                envelope, request=contract["request"]
            )
            expected_raw, expected_vectors = self._success_raw(
                intent=intent, envelope=envelope
            )
        except ExplicitProviderError as error:
            raise CacheIntegrityError(
                "embedding cached provider response contract drift"
            ) from error
        if dict(raw) != expected_raw or vectors != expected_vectors:
            raise CacheIntegrityError("embedding raw response evidence drift")
        intent_path, _ = self._attempt_paths(
            str(contract["resume_hash"]), int(intent["attempt_ordinal"])
        )
        final = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.limits),
            "model": MODEL_ID,
            "route": route_contract(),
            "identity": contract["identity"],
            "resume_hash": contract["resume_hash"],
            "successful_attempt_ordinal": intent["attempt_ordinal"],
            "attempt_identity_sha256": intent["attempt_identity_sha256"],
            "plan_sha256": _sha256(plan),
            "attempt_request_file_sha256": file_sha256(intent_path),
            "attempt_response_file_sha256": file_sha256(raw_path),
            "provider_request_sha256": contract["provider_request_sha256"],
            "actual_route": route_contract(),
            "transport_receipt": transport,
            "response_sha256": _sha256(payload),
            "vector_sha256s": [_sha256(vector) for vector in vectors],
            "vectors_sha256": _sha256(vectors),
            "full_text_sha256s": [
                item["full_text_sha256"] for item in contract["item_identities"]
            ],
            "visible_text_sha256s": [
                item["visible_text_sha256"] for item in contract["item_identities"]
            ],
            "vectors": vectors,
            "cache_path": str(final_path),
            "budget_binding_sha256": self.budget_binding_sha256,
        }
        return final

    def _validate_attempt_census(
        self,
        resume_hash_value: str,
        successful_ordinal: int,
    ) -> None:
        root = self.cache_root / "attempts" / resume_hash_value
        expected = {
            self._attempt_paths(resume_hash_value, ordinal)[index].resolve(
                strict=False
            )
            for ordinal in range(1, successful_ordinal + 1)
            for index in (0, 1)
        }
        observed = {
            path.resolve(strict=False)
            for path in root.glob("attempt_*.json")
            if path.is_file()
        }
        if observed != expected:
            raise CacheIntegrityError("embedding attempt file census drift")

    def _record_to_result(
        self, final: Mapping[str, Any], *, cache_hit: bool
    ) -> EmbeddingBatchResult:
        return EmbeddingBatchResult(
            vectors=tuple(tuple(float(value) for value in row) for row in final["vectors"]),
            request_sha256=str(final["identity"]),
            resume_hash=str(final["resume_hash"]),
            response_sha256=str(final["response_sha256"]),
            vector_sha256s=tuple(str(value) for value in final["vector_sha256s"]),
            vectors_sha256=str(final["vectors_sha256"]),
            full_text_sha256s=tuple(str(value) for value in final["full_text_sha256s"]),
            visible_text_sha256s=tuple(
                str(value) for value in final["visible_text_sha256s"]
            ),
            cache_path=str(final["cache_path"]),
            cache_hit=cache_hit,
            successful_attempt_ordinal=int(final["successful_attempt_ordinal"]),
        )

    def _validate_final(
        self,
        contract: Mapping[str, Any],
        plan: Mapping[str, Any],
        final_path: Path,
    ) -> EmbeddingBatchResult:
        final = self._load(final_path, "official embedding final")
        paths = self._paths(str(contract["resume_hash"]))
        if paths["orphan"].exists() or paths["failure"].exists():
            raise CacheIntegrityError("embedding final conflicts with orphan/failure")
        ordinal = final.get("successful_attempt_ordinal")
        if not isinstance(ordinal, int) or not 1 <= ordinal <= self.limits.max_attempts:
            raise CacheIntegrityError("embedding final attempt ordinal drift")
        for earlier in range(1, ordinal):
            earlier_intent_path, earlier_response_path = self._attempt_paths(
                str(contract["resume_hash"]), earlier
            )
            if not earlier_intent_path.is_file() or not earlier_response_path.is_file():
                raise CacheIntegrityError("embedding retry chain lacks earlier evidence")
            earlier_intent = self._load(
                earlier_intent_path, "official embedding earlier intent"
            )
            expected_earlier_intent = self._expected_intent(
                contract, plan, earlier
            )
            earlier_response = self._load(
                earlier_response_path, "official embedding earlier response"
            )
            if earlier_intent != expected_earlier_intent:
                raise CacheIntegrityError("embedding earlier retry evidence drift")
            self._validate_explicit_error_response(
                expected_earlier_intent, earlier_response
            )
            if earlier_response.get("retryable") is not True:
                raise CacheIntegrityError("embedding earlier retry was not retryable")
        intent_path, raw_path = self._attempt_paths(str(contract["resume_hash"]), ordinal)
        if not intent_path.is_file() or not raw_path.is_file():
            raise CacheIntegrityError("embedding final lacks attempt evidence")
        intent = self._load(intent_path, "official embedding intent")
        expected_intent = self._expected_intent(contract, plan, ordinal)
        if intent != expected_intent:
            raise CacheIntegrityError("embedding final intent binding drift")
        raw = self._load(raw_path, "official embedding raw response")
        expected_final = self._final_from_raw(
            contract=contract,
            plan=plan,
            intent=intent,
            raw=raw,
            raw_path=raw_path,
            final_path=final_path,
        )
        if final != expected_final:
            raise CacheIntegrityError("embedding final/raw cross-binding drift")
        self._validate_attempt_census(str(contract["resume_hash"]), ordinal)
        return self._record_to_result(final, cache_hit=True)

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> EmbeddingBatchResult:
        materialized = list(texts)
        metadata = dict(cache_metadata or {})
        contract = request_identity(
            materialized, cache_metadata=metadata, limits=self.limits
        )
        resume_hash_value = str(contract["resume_hash"])
        paths = self._paths(resume_hash_value)
        plan = self._expected_plan(contract)
        if paths["final"].is_file():
            observed_plan = self._load(paths["plan"], "official embedding plan")
            if observed_plan != plan:
                raise CacheIntegrityError("embedding cached plan drift")
            self.cache_hits += 1
            return self._validate_final(contract, plan, paths["final"])
        if not self.allow_model_calls:
            raise ProviderCallBlocked(
                "cold official embedding request is provider-blocked without writes"
            )
        with acquire_request_lifecycle_lock(
            paths["identity_lock"],
            timeout_seconds=self.lock_timeout_seconds,
            poll_seconds=self.lock_poll_seconds,
        ):
            if paths["final"].is_file():
                observed_plan = self._load(paths["plan"], "official embedding plan")
                if observed_plan != plan:
                    raise CacheIntegrityError("embedding concurrent plan drift")
                self.cache_hits += 1
                return self._validate_final(contract, plan, paths["final"])
            if paths["orphan"].exists():
                raise OrphanedCallError("embedding orphan prohibits provider recall")
            if paths["failure"].exists():
                raise ProviderRetriesExhausted("embedding failure prohibits provider recall")
            if paths["plan"].exists():
                if self._load(paths["plan"], "official embedding plan") != plan:
                    raise CacheIntegrityError("embedding plan drift")
            else:
                write_once_json(paths["plan"], plan)
            for ordinal in range(1, self.limits.max_attempts + 1):
                intent_path, response_path = self._attempt_paths(
                    resume_hash_value, ordinal
                )
                expected_intent = self._expected_intent(
                    contract, plan, ordinal
                )
                if intent_path.exists():
                    if self._load(intent_path, "official embedding intent") != expected_intent:
                        raise CacheIntegrityError("embedding existing intent drift")
                    if not response_path.exists():
                        orphan = {
                            **expected_intent,
                            "failure_kind": "orphaned_indeterminate_request_intent",
                            "provider_recall_allowed": False,
                        }
                        write_once_json(paths["orphan"], orphan)
                        raise OrphanedCallError(
                            "embedding request intent has no response; recall prohibited"
                        )
                    observed = self._load(
                        response_path, "official embedding attempt response"
                    )
                    if observed.get("outcome_kind") == "raw_response":
                        final = self._final_from_raw(
                            contract=contract,
                            plan=plan,
                            intent=expected_intent,
                            raw=observed,
                            raw_path=response_path,
                            final_path=paths["final"],
                        )
                        write_once_json(paths["final"], final)
                        return self._validate_final(contract, plan, paths["final"])
                    self._validate_explicit_error_response(
                        expected_intent, observed
                    )
                    if observed.get("retryable") is not True:
                        raise CacheIntegrityError(
                            "embedding nonretryable response lacks failure seal"
                        )
                    if ordinal == self.limits.max_attempts:
                        break
                    continue
                if response_path.exists():
                    raise CacheIntegrityError("embedding response exists without intent")
                self._reserve_intent(intent_path, expected_intent)
                try:
                    provider = self._provider_after_intent()
                    envelope = provider.batch_embed_contents(
                        request=contract["request"]
                    )
                    raw, _vectors = self._success_raw(
                        intent=expected_intent, envelope=envelope
                    )
                except ExplicitProviderError as error:
                    error_record = {
                        **expected_intent,
                        "outcome_kind": "explicit_provider_error",
                        "retryable": bool(error.retryable),
                        "status_code": error.status_code,
                        "error_type": type(error).__name__,
                        "error_sha256": _sha256(str(error)),
                    }
                    write_once_json(response_path, error_record)
                    if not error.retryable:
                        write_once_json(
                            paths["failure"],
                            {
                                **error_record,
                                "failure_kind": "determinate_provider_error_no_retry",
                                "provider_recall_allowed": False,
                            },
                        )
                        raise
                    if ordinal == self.limits.max_attempts:
                        break
                    continue
                except Exception as error:
                    orphan = {
                        **expected_intent,
                        "failure_kind": "indeterminate_or_provider_setup_failure",
                        "error_type": type(error).__name__,
                        "error_sha256": _sha256(str(error)),
                        "provider_recall_allowed": False,
                    }
                    write_once_json(paths["orphan"], orphan)
                    raise OrphanedCallError(
                        "official embedding outcome indeterminate; recall prohibited"
                    ) from error
                write_once_json(response_path, raw)
                final = self._final_from_raw(
                    contract=contract,
                    plan=plan,
                    intent=expected_intent,
                    raw=raw,
                    raw_path=response_path,
                    final_path=paths["final"],
                )
                write_once_json(paths["final"], final)
                self._validate_final(contract, plan, paths["final"])
                return self._record_to_result(final, cache_hit=False)
            failure = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "protocol_sha256": protocol_sha256(self.limits),
                "identity": contract["identity"],
                "resume_hash": resume_hash_value,
                "failure_kind": "explicit_retryable_attempts_exhausted",
                "attempts": self.limits.max_attempts,
                "provider_recall_allowed": False,
            }
            write_once_json(paths["failure"], failure)
            raise ProviderRetriesExhausted(
                "official embedding explicit retries exhausted"
            )

    def usage_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256(self.limits),
            "model": MODEL_ID,
            "route": route_contract(),
            "provider_calls": self.provider_calls,
            "cache_hits": self.cache_hits,
            "durable_provider_attempts": self._durable_attempt_count(),
            "approved_provider_attempt_hard_ceiling": self.limits.max_live_calls,
            "max_attempts_per_batch": self.limits.max_attempts,
            "budget_binding_sha256": self.budget_binding_sha256,
        }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "EmbeddingBatchResult",
    "EmbeddingLimits",
    "MAX_BATCH_SIZE",
    "MODEL_ID",
    "NATIVE_DIMENSIONS",
    "OFFICIAL_ENDPOINT",
    "OPERATION",
    "OfficialEmbeddingRuntimeError",
    "OfficialGoogleBatchEmbedContentsProvider",
    "OfficialGoogleBatchEmbeddingRuntime",
    "PROTOCOL_VERSION",
    "ProviderEnvelope",
    "RUNTIME_PATH",
    "VISIBLE_TEXT_CHARACTER_LIMIT",
    "ZERO_SHA256",
    "assert_batch_embed_contents_request",
    "build_batch_embed_contents_request",
    "expected_url",
    "file_sha256",
    "prepare_batch_inputs",
    "protocol_manifest",
    "protocol_sha256",
    "request_identity",
    "route_contract",
    "synthetic_provider_envelope",
]

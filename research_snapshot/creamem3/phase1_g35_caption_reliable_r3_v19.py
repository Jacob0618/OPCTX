"""Reliable, checkpointed Gemini 3.5 caption phase for MemEye v19.

This is a deliberately narrow recovery entry point.  It keeps the frozen
caption prompt and GenerateContent JSON body, but changes transport handling:

* retry the *same canonical request bytes* after transient HTTP/transport
  failures;
* wait exactly two seconds between attempts;
* allow at most five physical attempts for one logical caption;
* write every successful caption immediately, so a normal restart resumes at
  the first missing image instead of restarting all 438 images.

Nothing in this module performs a network call unless both ``--execute`` and
``--allow-model-calls`` are supplied.  The default command is a zero-call
preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creamem3 import build_executor_v19 as build
from creamem3 import memeye_g35_runtime_v19 as runtime
from creamem3 import run_memeye_full_arch_matrix_v19 as matrix


SCHEMA_VERSION = "creamem3.memeye_g35_caption_reliable_r3.v19.0"
CAPTION_SCHEMA_VERSION = "creamem3.memeye_g35_caption.v19.1_transport_retry"
CHECKPOINT_SCHEMA_VERSION = "creamem3.memeye_g35_caption_checkpoint_r3.v19.0"
COMPLETE_SCHEMA_VERSION = "creamem3.memeye_g35_caption_complete_r3.v19.0"

DEFAULT_SOURCE_PROJECT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\full_arch_matrix_r2"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\phase1_caption_writer_r3"
)
DEFAULT_IMAGE_ROOT = matrix.DEFAULT_DATA_ROOT / "data" / "image"

MAX_TRANSPORT_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 2.0
EXPECTED_TASKS = 8
EXPECTED_IMAGES = 438
CAPTION_PROVIDER_ATTEMPT_HARD_CEILING = EXPECTED_IMAGES * MAX_TRANSPORT_ATTEMPTS
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class ReliableCaptionError(RuntimeError):
    """The reliable caption recovery contract was violated."""


class RetryableTransportError(RuntimeError):
    """A request had no usable HTTP response and may be repeated."""


PostFn = Callable[[str, bytes, Mapping[str, str], float], tuple[int, bytes]]
SleepFn = Callable[[float], None]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    return _sha256_bytes(payload)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReliableCaptionError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ReliableCaptionError(f"{label} is not an object: {path}")
    return value


def _assert_output_root(root: Path) -> Path:
    resolved = Path(root).resolve()
    allowed = Path(r"D:\creamem3_cache\memeye_v19").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ReliableCaptionError("output root must be a child of the MemEye v19 cache")
    forbidden = {
        (allowed / "phase1_caption_writer_r1").resolve(),
        (allowed / "phase1_caption_writer_r2").resolve(),
        (allowed / "full_arch_matrix_r2").resolve(),
    }
    if resolved in forbidden:
        raise ReliableCaptionError("reliable recovery requires a fresh output root")
    return resolved


def _windows_https_proxy() -> str | None:
    """Return the active WinINET HTTPS proxy without persisting its address."""

    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = int(winreg.QueryValueEx(key, "ProxyEnable")[0])
            raw = str(winreg.QueryValueEx(key, "ProxyServer")[0]).strip()
    except (OSError, ValueError):
        return None
    if not enabled or not raw:
        return None
    selected = raw
    if ";" in raw or "=" in raw:
        routes: dict[str, str] = {}
        for item in raw.split(";"):
            name, separator, value = item.partition("=")
            if separator and name.strip() and value.strip():
                routes[name.strip().lower()] = value.strip()
        selected = routes.get("https") or routes.get("http") or ""
    if not selected:
        return None
    return selected if "://" in selected else "http://" + selected


class _PooledHttpxPost:
    """Persistent one-lane HTTP/2 transport, matching MGTM's client reuse."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def _new_client(self, timeout_seconds: float) -> Any:
        import httpx

        proxy = _windows_https_proxy()
        transport = httpx.HTTPTransport(
            http1=True,
            http2=True,
            limits=httpx.Limits(
                max_connections=1,
                max_keepalive_connections=1,
                keepalive_expiry=120.0,
            ),
            proxy=proxy,
            retries=0,
            trust_env=proxy is None,
        )
        return httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )

    def _discard_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        """Send exactly one physical attempt with caller-supplied bytes."""

        import httpx

        if self._client is None:
            self._client = self._new_client(timeout_seconds)
        try:
            response = self._client.post(url, content=body, headers=dict(headers))
            return int(response.status_code), bytes(response.content)
        except (httpx.TransportError, OSError) as exc:
            # The next durable attempt gets a fresh pool after a broken VPN,
            # proxy CONNECT tunnel, or upstream connection.
            self._discard_client()
            raise RetryableTransportError(type(exc).__name__) from exc


class ReliableGeminiRESTProvider:
    """One-attempt provider used by the durable runtime retry loop."""

    def __init__(
        self,
        *,
        post: PostFn | None = None,
        sleep: SleepFn = time.sleep,
        timeout_seconds: float | None = None,
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise runtime.RuntimeProtocolError(
                "GEMINI_API_KEY is required only for live execution"
            )
        configured_base = os.environ.get(
            "GEMINI_API_BASE", runtime.GEMINI_API_BASE
        ).rstrip("/")
        if configured_base != runtime.GEMINI_API_BASE:
            raise runtime.RuntimeProtocolError(
                "GEMINI_API_BASE must remain the official Google endpoint"
            )
        self._api_key = api_key
        self._base = runtime.GEMINI_API_BASE
        self._post = post or _PooledHttpxPost()
        self._sleep = sleep
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get("GEMINI_REQUEST_TIMEOUT_SECONDS", "300")
        )

    def _retryable(self, message: str, status_code: int | None = None) -> None:
        # The two-second delay belongs to the failed physical attempt.  The
        # durable runtime then advances the ordinal and repeats the same body.
        self._sleep(RETRY_WAIT_SECONDS)
        raise runtime.ExplicitProviderError(
            message,
            retryable=True,
            status_code=status_code,
        )

    def generate_content(self, *, model: str, request: Mapping[str, Any]) -> Any:
        url = f"{self._base}/models/{model}:generateContent"
        # This is byte-for-byte the serializer used by the frozen d603 runtime.
        body = runtime._canonical_json(request).encode("utf-8")  # type: ignore[attr-defined]
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        try:
            status, response_body = self._post(
                url, body, headers, self._timeout
            )
        except RetryableTransportError as exc:
            self._retryable(f"Gemini transport retry: {type(exc.__cause__).__name__}")

        if status in RETRYABLE_HTTP_STATUSES:
            self._retryable(
                f"Gemini GenerateContent returned transient HTTP {status}",
                status_code=status,
            )
        if not 200 <= status < 300:
            raise runtime.ExplicitProviderError(
                f"Gemini GenerateContent returned HTTP {status}",
                retryable=False,
                status_code=status,
            )
        try:
            value = json.loads(response_body.decode("utf-8"))
        except Exception:
            self._retryable("Gemini returned invalid JSON over HTTP 2xx")
        if not isinstance(value, dict):
            self._retryable("Gemini returned a non-object JSON response")
        return value


class ReliableG35CaptionBackend:
    """G35 caption-only backend with five durable physical attempts."""

    backbone = "g35"

    def __init__(
        self,
        cache_root: Path,
        *,
        post: PostFn | None = None,
        sleep: SleepFn = time.sleep,
    ) -> None:
        limits = dict(runtime.DEFAULT_STAGE_LIMITS)
        limits["caption"] = runtime.StageLimits(
            max_output_tokens=2400,
            max_live_calls=CAPTION_PROVIDER_ATTEMPT_HARD_CEILING,
            max_attempts=MAX_TRANSPORT_ATTEMPTS,
        )
        # Every other stage is unreachable from this entry point.
        limits["writer"] = runtime.StageLimits(16000, 0, 1)
        limits["view"] = runtime.StageLimits(16384, 0, 1)
        limits["judge"] = runtime.StageLimits(400, 0, 1)
        self.cache_root = Path(cache_root).resolve()
        self.stage_limits = limits
        self._runtime = runtime.Gemini35Runtime(
            self.cache_root,
            allow_model_calls=True,
            stage_limits=limits,
            provider_factory=lambda: ReliableGeminiRESTProvider(
                post=post,
                sleep=sleep,
            ),
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
        if stage != "caption":
            raise ReliableCaptionError("the reliable R3 backend is caption-only")
        if max_provider_attempts != MAX_TRANSPORT_ATTEMPTS:
            raise ReliableCaptionError("caption max attempts must be exactly five")
        if image_path is None or caption_context_count is None:
            raise ReliableCaptionError("caption image/context is missing")
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        parts = [runtime.inline_image_part(Path(image_path).read_bytes(), mime)]
        metadata = dict(cache_metadata or {})
        metadata["matched_substage"] = metadata.get("substage")
        metadata["max_provider_attempts"] = MAX_TRANSPORT_ATTEMPTS
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
        return build.GenerationReply(
            text=result.text,
            request_sha256=result.request_sha256,
            response_sha256=result.response_sha256,
            cache_hit=result.cache_hit,
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        return self._runtime.usage_receipt()


def _caption_input_contract(
    *,
    task: str,
    image: str,
    index: int,
    total_images: int,
    previous: Sequence[Mapping[str, str]],
    prompt: str,
    image_path: Path,
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "schema_version": build.BUILD_EXECUTOR_SCHEMA,
        "backbone": "g35",
        "task": task,
        "image": image,
        "sequence_index": index,
        "total_images": total_images,
        "previous": [dict(row) for row in previous],
        "prompt_sha256": _sha256(prompt),
        "raw_image_sha256": build._file_sha256(image_path),
        "system_prompt_sha256": _sha256(system_prompt),
        "system_prompt_line_count": len(system_prompt.splitlines()),
        "user_part_order": ["raw_image", "previous_two_caption_user_text"],
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "max_output_tokens": 2400,
        "max_provider_attempts": MAX_TRANSPORT_ATTEMPTS,
        "transport_retry_wait_seconds": RETRY_WAIT_SECONDS,
        "retry_request_body_policy": "same_canonical_bytes",
    }


def _caption_path(root: Path, task: str, index: int, image: str) -> Path:
    return (
        root
        / "captions"
        / "g35"
        / task
        / f"{index:04d}_{build._safe_name(image)}.json"
    )


def _load_or_generate_task(
    task_row: Mapping[str, Any],
    *,
    backend: ReliableG35CaptionBackend,
    image_root: Path,
    output_root: Path,
    system_prompt: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    task = str(task_row["task"])
    images = list(task_row["images"])
    captions: dict[str, str] = {}
    history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        previous = history[-2:]
        prompt = matrix.build_v7_reconstruction_user_text(
            task,
            image,
            sequence_index=index,
            total_images=len(images),
            previous=previous,
        )
        image_path = build._resolve_image(image_root, image)
        input_contract = _caption_input_contract(
            task=task,
            image=image,
            index=index,
            total_images=len(images),
            previous=previous,
            prompt=prompt,
            image_path=image_path,
            system_prompt=system_prompt,
        )
        input_sha = _sha256(input_contract)
        path = _caption_path(output_root, task, index, image)
        if path.is_file():
            record = _read_object(path, "R3 caption checkpoint")
            if (
                record.get("schema_version") != CAPTION_SCHEMA_VERSION
                or record.get("input_contract_sha256") != input_sha
                or record.get("task") != task
                or record.get("image") != image
                or record.get("sequence_index") != index
                or record.get("total_images") != len(images)
                or not str(record.get("caption", "")).strip()
            ):
                raise ReliableCaptionError(f"caption checkpoint drift: {path}")
            caption = str(record["caption"]).strip()
        else:
            reply = backend.generate(
                "caption",
                prompt,
                system=system_prompt,
                max_output_tokens=2400,
                max_provider_attempts=MAX_TRANSPORT_ATTEMPTS,
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
                raise ReliableCaptionError(f"empty G35 caption: {image}")
            record = {
                "schema_version": CAPTION_SCHEMA_VERSION,
                "task": task,
                "image": image,
                "caption": caption,
                "model": runtime.MODEL_ID,
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
            build._write_once_json(path, record)
        captions[image] = caption
        history.append({"image": image, "caption": caption})
        rows.append(
            {
                "task": task,
                "image": image,
                "sequence_index": index,
                "caption_sha256": _sha256(caption),
                "caption_file": str(path.resolve()),
                "caption_file_sha256": build._file_sha256(path),
            }
        )
        _atomic_json(
            output_root / "build" / "caption_checkpoint.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "status": "in_progress",
                "completed_caption_files": len(
                    list((output_root / "captions" / "g35").rglob("*.json"))
                ),
                "last_completed": rows[-1],
                "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
                "retry_wait_seconds": RETRY_WAIT_SECONDS,
            },
        )
    return captions, rows


def _project_tasks(source_project_root: Path) -> list[dict[str, Any]]:
    project = matrix.validate_project(output_root=source_project_root)
    tasks = build._project_tasks(project)
    if len(tasks) != EXPECTED_TASKS:
        raise ReliableCaptionError("project task census drift")
    if sum(len(row["images"]) for row in tasks) != EXPECTED_IMAGES:
        raise ReliableCaptionError("project image census drift")
    return tasks


def run_caption_phase(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    backend: ReliableG35CaptionBackend | None = None,
) -> dict[str, Any]:
    root = _assert_output_root(output_root)
    tasks = _project_tasks(Path(source_project_root).resolve())
    system_prompt = build._load_g35_caption_system_prompt()
    active_backend = backend or ReliableG35CaptionBackend(
        root / "provider" / "g35"
    )
    all_rows: list[dict[str, Any]] = []
    task_receipts: list[dict[str, Any]] = []
    for task_row in tasks:
        captions, rows = _load_or_generate_task(
            task_row,
            backend=active_backend,
            image_root=Path(image_root).resolve(),
            output_root=root,
            system_prompt=system_prompt,
        )
        all_rows.extend(rows)
        task_receipts.append(
            {
                "task": str(task_row["task"]),
                "images": len(captions),
                "rows_sha256": _sha256(rows),
            }
        )
    if len(all_rows) != EXPECTED_IMAGES:
        raise ReliableCaptionError("completed caption census drift")
    complete = {
        "schema_version": COMPLETE_SCHEMA_VERSION,
        "complete": True,
        "status": "caption_bank_complete_waiting_writer_import",
        "source_project_root": str(Path(source_project_root).resolve()),
        "output_root": str(root),
        "model": runtime.MODEL_ID,
        "captions": len(all_rows),
        "tasks": len(task_receipts),
        "task_receipts": task_receipts,
        "caption_rows_sha256": _sha256(all_rows),
        "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "request_body_policy": "same_canonical_bytes_per_retry",
        "checkpoint_policy": "one_write_once_caption_file_per_success",
        "usage": dict(active_backend.usage_receipt()),
        "writer_started": False,
        "embedding_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
    }
    complete["complete_sha256"] = _sha256(complete)
    build._write_once_json(root / "build" / "caption_complete.json", complete)
    _atomic_json(
        root / "build" / "caption_checkpoint.json",
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "complete",
            "completed_caption_files": EXPECTED_IMAGES,
            "complete_sha256": complete["complete_sha256"],
            "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
            "retry_wait_seconds": RETRY_WAIT_SECONDS,
        },
    )
    return complete


def zero_call_preflight(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = _assert_output_root(output_root)
    tasks = _project_tasks(Path(source_project_root).resolve())
    prompt = build._load_g35_caption_system_prompt()
    existing = (
        len(list((root / "captions" / "g35").rglob("*.json")))
        if (root / "captions" / "g35").is_dir()
        else 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "zero_call_ready",
        "authorized": False,
        "source_project_root": str(Path(source_project_root).resolve()),
        "output_root": str(root),
        "tasks": len(tasks),
        "images": sum(len(row["images"]) for row in tasks),
        "existing_caption_checkpoints": existing,
        "system_prompt_sha256": _sha256(prompt),
        "system_prompt_lines": len(prompt.splitlines()),
        "model": runtime.MODEL_ID,
        "endpoint": runtime.GEMINI_API_BASE,
        "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "request_body_policy": "same_canonical_bytes_per_retry",
        "transport": "httpx_0.28_pooled_http2_single_lane_rebuild_on_error",
        "model_calls": 0,
        "provider_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--source-project-root", type=Path, default=DEFAULT_SOURCE_PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.execute:
        if not args.allow_model_calls:
            raise ReliableCaptionError(
                "live execution requires --execute --allow-model-calls"
            )
        result = run_caption_phase(
            source_project_root=args.source_project_root,
            output_root=args.output_root,
            image_root=args.image_root,
        )
    else:
        if args.allow_model_calls:
            raise ReliableCaptionError(
                "--allow-model-calls is valid only together with --execute"
            )
        result = zero_call_preflight(
            source_project_root=args.source_project_root,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent MemEye v19 thinking-round experiment.

This runner leaves ``isolated_full_arch_retry5_r1`` untouched and changes only
the registered Gemini thinking controls:

* Gemini 2.5 Flash-Lite writer/view: ``thinkingBudget=512``;
* Gemini 2.5 Flash-Lite Reader: ``thinkingBudget=0`` (fixed evaluator);
* Gemini 3.5 Flash-Lite caption/writer/view: ``thinkingLevel=low``;
* Gemini 3.5 Flash-Lite judge: ``thinkingLevel=minimal`` (fixed evaluator).

All generation stages retain at most five physical attempts and the reliable
runner's exact two-second transient retry delay.  The live phases are ordered
caption -> writer -> build -> predict -> score and are independently resumable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from creamem3 import build_executor_v19 as build
from creamem3 import graph_memory_v2 as gm
from creamem3 import memeye_matched_ingest_v19 as ingest
from creamem3 import memeye_g35_runtime_v19 as g35_runtime
from creamem3 import phase1_g35_caption_reliable_r3_v19 as captions
from creamem3 import run_memeye_isolated_full_arch_retry5_v19 as base
from creamem3 import run_memeye_full_arch_matrix_v19 as matrix


SCHEMA_VERSION = "creamem3.memeye_thinking512_low_round.v19.0"
G25_THINKING_BUDGET = 512
G35_WORK_THINKING_LEVEL = "low"
G35_JUDGE_THINKING_LEVEL = "minimal"
MAX_GENERATION_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 2.0
DEFAULT_EVAL_WORKERS = 32

DEFAULT_SOURCE_PROJECT_ROOT = base.DEFAULT_SOURCE_PROJECT_ROOT
DEFAULT_SCORE_DATA_ROOT = base.DEFAULT_SCORE_DATA_ROOT
DEFAULT_CAPTION_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\g35_caption_thinking512_low_r1"
)
DEFAULT_FALLBACK_SOURCE_CAPTION_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\g35_caption_retry5_r1"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\creamem3_cache\memeye_v19\isolated_full_arch_thinking512_low_r1"
)


class ThinkingRoundError(RuntimeError):
    """The independent thinking-round contract was violated."""


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


def thinking_contract() -> dict[str, Any]:
    """Return the model-visible delta and fixed execution controls."""

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": "independent_thinking_round_not_formal_release",
        "gemini_25_flash_lite": {
            "thinking_config": {"thinkingBudget": G25_THINKING_BUDGET},
            "affected_stages": ["writer", "view"],
            "reader_thinking_config": {"thinkingBudget": 0},
            "reader_stages": ["planner", "focused", "answer", "structured_fused"],
            "caption_source": "frozen_v7_read_only_unchanged",
        },
        "gemini_35_flash_lite": {
            "work_thinking_config": {"thinkingLevel": G35_WORK_THINKING_LEVEL},
            "work_stages": ["caption", "writer", "view"],
            "judge_thinking_config": {"thinkingLevel": G35_JUDGE_THINKING_LEVEL},
            "judge_stages": ["judge"],
        },
        "generation_max_physical_attempts": MAX_GENERATION_ATTEMPTS,
        "transient_retry_wait_seconds": RETRY_WAIT_SECONDS,
        "openrouter_allowed": False,
        "gemini_endpoint": g35_runtime.GEMINI_API_BASE,
        "judge_model": g35_runtime.MODEL_ID,
    }


def _write_contract(root: Path) -> Path:
    path = Path(root).resolve() / "build" / "thinking_round_contract.json"
    value = thinking_contract()
    value["root"] = str(Path(root).resolve())
    value["contract_sha256"] = _sha256(value)
    build._write_once_json(path, value)  # type: ignore[attr-defined]
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != value:
        raise ThinkingRoundError(f"thinking contract write-once drift: {path}")
    return path


@contextmanager
def _g35_level(level: str) -> Iterator[None]:
    if level not in {G35_WORK_THINKING_LEVEL, G35_JUDGE_THINKING_LEVEL}:
        raise ThinkingRoundError(f"unregistered Gemini 3.5 thinking level: {level}")
    previous = g35_runtime.THINKING_LEVEL
    g35_runtime.THINKING_LEVEL = level
    try:
        yield
    finally:
        g35_runtime.THINKING_LEVEL = previous


def _patch_g25_module(module: Any, *, profile: str) -> None:
    """Patch one isolated official-G25 module without editing its source file."""

    marker = getattr(module, "_memeye_thinking_round_patch", None)
    expected = {"budget": G25_THINKING_BUDGET, "profile": profile}
    if marker is not None:
        if marker != expected:
            raise ThinkingRoundError(f"G25 {profile} thinking patch drift")
        return

    original_build = module.build_generate_content_request
    original_assert = module.assert_request_contract
    original_manifest = module.protocol_manifest

    def assert_request_contract(
        request: Mapping[str, Any],
        *,
        stage: str,
        caption_context_count: int | None = None,
        token_limit: int | None = None,
    ) -> None:
        generation = request.get("generationConfig")
        if not isinstance(generation, Mapping) or generation.get("thinkingConfig") != {
            "thinkingBudget": G25_THINKING_BUDGET
        }:
            raise module.RequestContractError(
                f"G25 {profile} thinkingBudget must be {G25_THINKING_BUDGET}"
            )
        canonical = copy.deepcopy(dict(request))
        canonical["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        original_assert(
            canonical,
            stage=stage,
            caption_context_count=caption_context_count,
            token_limit=token_limit,
        )

    def build_generate_content_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        request = original_build(*args, **kwargs)
        generation = request.get("generationConfig")
        if not isinstance(generation, dict) or generation.get("thinkingConfig") != {
            "thinkingBudget": 0
        }:
            raise ThinkingRoundError(f"G25 {profile} zero-budget source contract drift")
        generation["thinkingConfig"] = {"thinkingBudget": G25_THINKING_BUDGET}
        return request

    def protocol_manifest(stage_limits: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value = copy.deepcopy(original_manifest(stage_limits))
        value["schema_version"] = f"{SCHEMA_VERSION}.g25_{profile}"
        controls = value.setdefault("generation_controls", {})
        controls["thinking_budget"] = G25_THINKING_BUDGET
        value["thinking_round_contract_sha256"] = _sha256(thinking_contract())
        return value

    def protocol_sha256(stage_limits: Mapping[str, Any] | None = None) -> str:
        return _sha256(protocol_manifest(stage_limits))

    module.assert_request_contract = assert_request_contract
    module.build_generate_content_request = build_generate_content_request
    module.protocol_manifest = protocol_manifest
    module.protocol_sha256 = protocol_sha256
    module._memeye_thinking_round_patch = expected


def install_g25_thinking_contracts() -> None:
    """Install 512 only in G25 writer/view; Reader remains the fixed zero arm."""

    from creamem3 import official_g25_build_runtime_adapter_v19 as build_adapter

    _patch_g25_module(
        build_adapter._load_source_runtime(),  # type: ignore[attr-defined]
        profile="build",
    )


def caption_preflight(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
) -> dict[str, Any]:
    """Provider-cold preflight; deliberately writes nothing."""

    with _g35_level(G35_WORK_THINKING_LEVEL):
        result = captions.zero_call_preflight(
            source_project_root=source_project_root,
            output_root=caption_root,
        )
    return {
        **result,
        "thinking_round": thinking_contract(),
        "caption_root": str(Path(caption_root).resolve()),
        "provider_calls": 0,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ThinkingRoundError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ThinkingRoundError(f"{label} is not an object: {path}")
    return value


def _caption_protocol_sha256(level: str) -> str:
    limits = dict(g35_runtime.DEFAULT_STAGE_LIMITS)
    limits["caption"] = g35_runtime.StageLimits(
        2400,
        captions.CAPTION_PROVIDER_ATTEMPT_HARD_CEILING,
        MAX_GENERATION_ATTEMPTS,
    )
    limits["writer"] = g35_runtime.StageLimits(16000, 0, 1)
    limits["view"] = g35_runtime.StageLimits(16384, 0, 1)
    limits["judge"] = g35_runtime.StageLimits(400, 0, 1)
    with _g35_level(level):
        return g35_runtime.protocol_sha256(limits)


def _caption_fallback_receipts(root: Path) -> list[Path]:
    path = Path(root).resolve() / "fallbacks" / "caption"
    return sorted(path.glob("*.json")) if path.is_dir() else []


def _validate_mixed_fallback_receipt(
    root: Path, receipt_path: Path
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, "caption fallback receipt")
    body = dict(receipt)
    digest = body.pop("fallback_sha256", None)
    if (
        digest != _sha256(body)
        or receipt.get("schema_version") != f"{SCHEMA_VERSION}.caption_fallback"
        or receipt.get("status") != "zero_call_prior_minimal_caption_fallback"
        or receipt.get("additional_provider_calls") != 0
        or receipt.get("failed_low_physical_attempts") != MAX_GENERATION_ATTEMPTS
        or receipt.get("failed_low_error_type") != "EmptyProviderResponse"
        or receipt.get("provider_final_synthesized") is not False
        or receipt.get("failed_evidence_preserved") is not True
        or Path(str(receipt.get("current_caption_root", ""))).resolve()
        != Path(root).resolve()
    ):
        raise ThinkingRoundError(f"caption fallback receipt drift: {receipt_path}")
    failure_path = Path(str(receipt.get("current_failure_file", ""))).resolve()
    target_path = Path(str(receipt.get("target_caption_file", ""))).resolve()
    source_path = Path(str(receipt.get("source_caption_file", ""))).resolve()
    identity = str(receipt.get("current_failed_resume_hash", ""))
    if (
        receipt_path.resolve()
        != (Path(root).resolve() / "fallbacks" / "caption" / f"{identity}.json")
        or not failure_path.is_file()
        or failure_path.resolve()
        != (
            Path(root).resolve()
            / "provider"
            / "g35"
            / "failures"
            / "caption"
            / f"{identity}.json"
        )
        or build._file_sha256(failure_path)  # type: ignore[attr-defined]
        != receipt.get("current_failure_file_sha256")
        or not target_path.is_file()
        or build._file_sha256(target_path)  # type: ignore[attr-defined]
        != receipt.get("target_caption_file_sha256")
        or not source_path.is_file()
        or build._file_sha256(source_path)  # type: ignore[attr-defined]
        != receipt.get("source_caption_file_sha256")
    ):
        raise ThinkingRoundError("caption fallback bound evidence changed")
    failed_attempt_rows = receipt.get("failed_low_attempt_rows")
    if (
        not isinstance(failed_attempt_rows, list)
        or len(failed_attempt_rows) != MAX_GENERATION_ATTEMPTS
    ):
        raise ThinkingRoundError("caption fallback attempt receipt census drift")
    expected_attempt_root = (
        Path(root).resolve()
        / "provider"
        / "g35"
        / "attempts"
        / "caption"
        / identity
    )
    for ordinal, row in enumerate(failed_attempt_rows, start=1):
        if not isinstance(row, Mapping) or row.get("ordinal") != ordinal:
            raise ThinkingRoundError("caption fallback attempt receipt order drift")
        request_path = Path(str(row.get("request_file", ""))).resolve()
        response_path = Path(str(row.get("response_file", ""))).resolve()
        if (
            request_path != expected_attempt_root / f"attempt_{ordinal:03d}_request.json"
            or response_path
            != expected_attempt_root / f"attempt_{ordinal:03d}_response.json"
            or not request_path.is_file()
            or not response_path.is_file()
            or build._file_sha256(request_path)  # type: ignore[attr-defined]
            != row.get("request_file_sha256")
            or build._file_sha256(response_path)  # type: ignore[attr-defined]
            != row.get("response_file_sha256")
        ):
            raise ThinkingRoundError("caption fallback attempt evidence changed")

    source_root = Path(str(receipt.get("source_caption_root", ""))).resolve()
    source_identity = str(receipt.get("source_provider_resume_hash", ""))
    source_complete_path = source_root / "build" / "caption_complete.json"
    source_plan_path = (
        source_root / "provider" / "g35" / "plans" / "caption" / f"{source_identity}.json"
    )
    source_final_path = (
        source_root
        / "provider"
        / "g35"
        / "responses"
        / "caption"
        / f"{source_identity}.json"
    )
    if (
        not source_complete_path.is_file()
        or build._file_sha256(source_complete_path)  # type: ignore[attr-defined]
        != receipt.get("source_caption_complete_file_sha256")
        or not source_plan_path.is_file()
        or build._file_sha256(source_plan_path)  # type: ignore[attr-defined]
        != receipt.get("source_provider_plan_file_sha256")
        or not source_final_path.is_file()
        or build._file_sha256(source_final_path)  # type: ignore[attr-defined]
        != receipt.get("source_provider_final_file_sha256")
    ):
        raise ThinkingRoundError("caption fallback source provider evidence changed")
    source_attempt_rows = receipt.get("source_provider_attempt_rows")
    if not isinstance(source_attempt_rows, list) or not source_attempt_rows:
        raise ThinkingRoundError("caption fallback source attempt receipt drift")
    for ordinal, row in enumerate(source_attempt_rows, start=1):
        request_path = (
            source_root
            / "provider"
            / "g35"
            / "attempts"
            / "caption"
            / source_identity
            / f"attempt_{ordinal:03d}_request.json"
        )
        response_path = request_path.with_name(f"attempt_{ordinal:03d}_response.json")
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != ordinal
            or not request_path.is_file()
            or not response_path.is_file()
            or build._file_sha256(request_path)  # type: ignore[attr-defined]
            != row.get("request_file_sha256")
            or build._file_sha256(response_path)  # type: ignore[attr-defined]
            != row.get("response_file_sha256")
        ):
            raise ThinkingRoundError("caption fallback source attempt evidence changed")
    return receipt


def _install_one_caption_empty_fallback(
    *,
    failure_path: Path,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    source_caption_root: Path = DEFAULT_FALLBACK_SOURCE_CAPTION_ROOT,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    image_root: Path = captions.DEFAULT_IMAGE_ROOT,
) -> dict[str, Any]:
    """Append one zero-call fallback after an exact five-empty terminal chain."""

    root = Path(caption_root).resolve()
    source_root = Path(source_caption_root).resolve()
    failure_path = Path(failure_path).resolve()
    expected_failure_root = (
        root / "provider" / "g35" / "failures" / "caption"
    ).resolve()
    if failure_path.parent != expected_failure_root or not failure_path.is_file():
        raise ThinkingRoundError("fallback failure escaped the current caption root")
    failure = _read_json(failure_path, "failed low caption terminal")
    identity = failure_path.stem
    receipt_path = root / "fallbacks" / "caption" / f"{identity}.json"
    if receipt_path.is_file():
        receipt = _validate_mixed_fallback_receipt(root, receipt_path)
        return {**receipt, "cache_hit": True, "provider_calls": 0}
    identity_orphan = (
        root / "provider" / "g35" / "orphans" / "caption" / f"{identity}.json"
    )
    if identity_orphan.exists():
        raise ThinkingRoundError("fallback is forbidden for an orphaned identity")
    if (
        failure.get("failure_kind") != "explicit_retryable_attempts_exhausted"
        or failure.get("provider_recall_allowed") is not False
        or failure.get("attempts") != MAX_GENERATION_ATTEMPTS
        or failure.get("stage") != "caption"
        or failure.get("model") != g35_runtime.MODEL_ID
        or failure.get("protocol_sha256") != _caption_protocol_sha256(
            G35_WORK_THINKING_LEVEL
        )
        or failure.get("resume_hash") != identity
    ):
        raise ThinkingRoundError("failed low caption terminal is not eligible")
    current_final_path = root / "provider" / "g35" / "responses" / "caption" / f"{identity}.json"
    if current_final_path.exists():
        raise ThinkingRoundError("fallback cannot shadow a current provider final")
    plan_path = root / "provider" / "g35" / "plans" / "caption" / f"{identity}.json"
    plan = _read_json(plan_path, "failed low caption plan")
    if (
        plan.get("resume_hash") != identity
        or plan.get("request_sha256") != failure.get("request_sha256")
        or plan.get("protocol_sha256") != failure.get("protocol_sha256")
        or plan.get("max_attempts") != MAX_GENERATION_ATTEMPTS
    ):
        raise ThinkingRoundError("failed low caption plan/terminal drift")

    attempt_root = root / "provider" / "g35" / "attempts" / "caption" / identity
    attempt_rows: list[dict[str, Any]] = []
    for ordinal in range(1, MAX_GENERATION_ATTEMPTS + 1):
        request_path = attempt_root / f"attempt_{ordinal:03d}_request.json"
        response_path = attempt_root / f"attempt_{ordinal:03d}_response.json"
        request = _read_json(request_path, "failed low attempt intent")
        response = _read_json(response_path, "failed low empty response")
        usage = response.get("usage")
        if (
            request.get("attempt_ordinal") != ordinal
            or response.get("attempt_ordinal") != ordinal
            or request.get("resume_hash") != identity
            or response.get("resume_hash") != identity
            or response.get("outcome_kind") != "explicit_provider_error"
            or response.get("retryable") is not True
            or response.get("error_type") != "EmptyProviderResponse"
            or not isinstance(usage, Mapping)
            or int(usage.get("output_tokens", -1)) != 0
            or int(usage.get("thought_tokens", -1)) != 0
        ):
            raise ThinkingRoundError(f"attempt {ordinal} is not an exact empty response")
        attempt_rows.append(
            {
                "ordinal": ordinal,
                "request_file": str(request_path.resolve()),
                "request_file_sha256": build._file_sha256(request_path),  # type: ignore[attr-defined]
                "response_file": str(response_path.resolve()),
                "response_file_sha256": build._file_sha256(response_path),  # type: ignore[attr-defined]
                "error_sha256": response.get("error_sha256"),
                "usage": dict(usage),
            }
        )

    metadata = plan.get("cache_metadata")
    if not isinstance(metadata, Mapping):
        raise ThinkingRoundError("failed plan lacks caption metadata")
    task = str(metadata.get("task", ""))
    image = str(metadata.get("image", ""))
    index = int(metadata.get("sequence_index", 0))
    tasks = captions._project_tasks(Path(source_project_root).resolve())  # type: ignore[attr-defined]
    task_rows = [row for row in tasks if str(row["task"]) == task]
    if len(task_rows) != 1:
        raise ThinkingRoundError("failed caption task is not in the source project")
    task_row = task_rows[0]
    images = list(task_row["images"])
    if not 1 <= index <= len(images) or images[index - 1] != image:
        raise ThinkingRoundError("failed caption task/image/index drift")
    history: list[dict[str, str]] = []
    for previous_index, previous_image in enumerate(images[: index - 1], start=1):
        previous_path = captions._caption_path(  # type: ignore[attr-defined]
            root, task, previous_index, previous_image
        )
        previous_record = _read_json(previous_path, "preceding low caption")
        if (
            previous_record.get("task") != task
            or previous_record.get("image") != previous_image
            or previous_record.get("sequence_index") != previous_index
            or not str(previous_record.get("caption", "")).strip()
        ):
            raise ThinkingRoundError("preceding low caption checkpoint drift")
        history.append(
            {"image": previous_image, "caption": str(previous_record["caption"])}
        )
    previous = history[-2:]
    prompt = matrix.build_v7_reconstruction_user_text(
        task,
        image,
        sequence_index=index,
        total_images=len(images),
        previous=previous,
    )
    resolved_image = build._resolve_image(Path(image_root).resolve(), image)  # type: ignore[attr-defined]
    system_prompt = build._load_g35_caption_system_prompt()  # type: ignore[attr-defined]
    input_contract = captions._caption_input_contract(  # type: ignore[attr-defined]
        task=task,
        image=image,
        index=index,
        total_images=len(images),
        previous=previous,
        prompt=prompt,
        image_path=resolved_image,
        system_prompt=system_prompt,
    )
    input_sha = _sha256(input_contract)
    if input_sha != metadata.get("input_contract_sha256"):
        raise ThinkingRoundError("failed plan is not bound to reconstructed current input")

    source_complete = base._caption_complete(source_root)  # type: ignore[attr-defined]
    source_path = captions._caption_path(source_root, task, index, image)  # type: ignore[attr-defined]
    source_record = _read_json(source_path, "prior-minimal source caption")
    source_caption = str(source_record.get("caption", "")).strip()
    if (
        source_record.get("schema_version") != captions.CAPTION_SCHEMA_VERSION
        or source_record.get("task") != task
        or source_record.get("image") != image
        or source_record.get("sequence_index") != index
        or source_record.get("total_images") != len(images)
        or source_record.get("model") != g35_runtime.MODEL_ID
        or not source_caption
    ):
        raise ThinkingRoundError("prior-minimal source caption contract drift")
    source_request_sha = str(source_record.get("provider_request_sha256", ""))
    source_response_sha = str(source_record.get("provider_response_sha256", ""))
    source_finals = []
    for path in (source_root / "provider" / "g35" / "responses" / "caption").glob(
        "*.json"
    ):
        row = _read_json(path, "source provider final")
        if row.get("request_sha256") == source_request_sha:
            source_finals.append((path, row))
    if len(source_finals) != 1:
        raise ThinkingRoundError("prior-minimal source provider final is not unique")
    source_final_path, source_final = source_finals[0]
    source_identity = source_final_path.stem
    source_plan_path = (
        source_root / "provider" / "g35" / "plans" / "caption" / f"{source_identity}.json"
    )
    source_plan = _read_json(source_plan_path, "source provider plan")
    source_ordinal = int(source_final.get("attempt_ordinal", 0))
    if (
        source_plan.get("protocol_sha256")
        != _caption_protocol_sha256(G35_JUDGE_THINKING_LEVEL)
        or source_final.get("response_sha256") != source_response_sha
        or source_final.get("request_sha256") != source_request_sha
        or not 1 <= source_ordinal <= MAX_GENERATION_ATTEMPTS
        or (
            source_root
            / "provider"
            / "g35"
            / "failures"
            / "caption"
            / f"{source_identity}.json"
        ).exists()
        or (
            source_root
            / "provider"
            / "g35"
            / "orphans"
            / "caption"
            / f"{source_identity}.json"
        ).exists()
    ):
        raise ThinkingRoundError("prior-minimal source provider chain drift")
    source_attempt_rows: list[dict[str, Any]] = []
    for ordinal in range(1, source_ordinal + 1):
        source_attempt_request = (
            source_root
            / "provider"
            / "g35"
            / "attempts"
            / "caption"
            / source_identity
            / f"attempt_{ordinal:03d}_request.json"
        )
        source_attempt_response = source_attempt_request.with_name(
            f"attempt_{ordinal:03d}_response.json"
        )
        if not source_attempt_request.is_file() or not source_attempt_response.is_file():
            raise ThinkingRoundError("source provider attempt chain is incomplete")
        source_attempt_rows.append(
            {
                "ordinal": ordinal,
                "request_file_sha256": build._file_sha256(source_attempt_request),  # type: ignore[attr-defined]
                "response_file_sha256": build._file_sha256(source_attempt_response),  # type: ignore[attr-defined]
            }
        )

    target_path = captions._caption_path(root, task, index, image)  # type: ignore[attr-defined]
    target_record = {
        "schema_version": captions.CAPTION_SCHEMA_VERSION,
        "task": task,
        "image": image,
        "caption": source_caption,
        "model": g35_runtime.MODEL_ID,
        "lineage": "transparent_prior_minimal_fallback_after_exact5_low_empty",
        "sequence_index": index,
        "total_images": len(images),
        "previous_images": [row["image"] for row in previous],
        "previous_caption_sha256": [_sha256(row["caption"]) for row in previous],
        "summary_contexts": 0,
        "input_contract": input_contract,
        "input_contract_sha256": input_sha,
        # These are authentic hashes from the sealed prior-minimal provider
        # chain, never a synthesized current-root provider final.
        "provider_request_sha256": source_request_sha,
        "provider_response_sha256": source_response_sha,
        "thinking_round_fallback": {
            "kind": "zero_call_prior_minimal_caption",
            "failed_low_resume_hash": identity,
            "failed_low_request_sha256": failure["request_sha256"],
            "failed_low_physical_attempts": MAX_GENERATION_ATTEMPTS,
            "source_caption_root": str(source_root),
            "source_caption_file": str(source_path.resolve()),
            "source_caption_file_sha256": build._file_sha256(source_path),  # type: ignore[attr-defined]
            "source_provider_resume_hash": source_identity,
            "source_thinking_level": G35_JUDGE_THINKING_LEVEL,
            "additional_provider_calls": 0,
            "provider_final_synthesized": False,
            "fallback_receipt": str(receipt_path.resolve()),
        },
    }
    if target_path.is_file() and _read_json(
        target_path, "partially installed fallback caption"
    ) != target_record:
        raise ThinkingRoundError("fallback target caption already exists with drift")
    build._write_once_json(target_path, target_record)  # type: ignore[attr-defined]
    receipt = {
        "schema_version": f"{SCHEMA_VERSION}.caption_fallback",
        "status": "zero_call_prior_minimal_caption_fallback",
        "current_caption_root": str(root),
        "current_failed_resume_hash": identity,
        "current_failed_request_sha256": failure["request_sha256"],
        "current_failure_file": str(failure_path.resolve()),
        "current_failure_file_sha256": build._file_sha256(failure_path),  # type: ignore[attr-defined]
        "failed_low_physical_attempts": MAX_GENERATION_ATTEMPTS,
        "failed_low_error_type": "EmptyProviderResponse",
        "failed_low_attempt_rows": attempt_rows,
        "failed_evidence_preserved": True,
        "provider_final_synthesized": False,
        "source_caption_root": str(source_root),
        "source_caption_complete_sha256": source_complete["complete_sha256"],
        "source_caption_complete_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
            source_root / "build" / "caption_complete.json"
        ),
        "source_caption_file": str(source_path.resolve()),
        "source_caption_file_sha256": build._file_sha256(source_path),  # type: ignore[attr-defined]
        "source_caption_sha256": _sha256(source_caption),
        "source_provider_resume_hash": source_identity,
        "source_provider_request_sha256": source_request_sha,
        "source_provider_response_sha256": source_response_sha,
        "source_provider_plan_file_sha256": build._file_sha256(source_plan_path),  # type: ignore[attr-defined]
        "source_provider_final_file_sha256": build._file_sha256(source_final_path),  # type: ignore[attr-defined]
        "source_provider_attempt_rows": source_attempt_rows,
        "source_thinking_level": G35_JUDGE_THINKING_LEVEL,
        "task": task,
        "image": image,
        "sequence_index": index,
        "target_caption_file": str(target_path.resolve()),
        "target_caption_file_sha256": build._file_sha256(target_path),  # type: ignore[attr-defined]
        "prev2_possible_affected_sequence_indices": [
            value for value in (index + 1, index + 2) if value <= len(images)
        ],
        "additional_provider_calls": 0,
    }
    receipt["fallback_sha256"] = _sha256(receipt)
    build._write_once_json(receipt_path, receipt)  # type: ignore[attr-defined]
    _validate_mixed_fallback_receipt(root, receipt_path)
    return {**receipt, "cache_hit": False, "provider_calls": 0}


def install_caption_empty_fallbacks(
    *,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    source_caption_root: Path = DEFAULT_FALLBACK_SOURCE_CAPTION_ROOT,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    image_root: Path = captions.DEFAULT_IMAGE_ROOT,
) -> dict[str, Any]:
    """Install every eligible exact-five-empty fallback; reject every other failure."""

    root = Path(caption_root).resolve()
    existing_paths = _caption_fallback_receipts(root)
    existing = [_validate_mixed_fallback_receipt(root, path) for path in existing_paths]
    failure_files = sorted(
        (root / "provider" / "g35" / "failures" / "caption").glob("*.json")
    )
    handled = {str(row["current_failed_resume_hash"]) for row in existing}
    unhandled = [path for path in failure_files if path.stem not in handled]
    if (root / "build" / "caption_complete.json").exists() and unhandled:
        raise ThinkingRoundError("new fallback installation is forbidden after caption complete")
    installed: list[dict[str, Any]] = []
    for failure_path in unhandled:
        # `_install_one` is deliberately strict.  A non-empty, malformed,
        # final-bearing, orphaned, or non-five-attempt failure aborts here.
        installed.append(
            _install_one_caption_empty_fallback(
                failure_path=failure_path,
                caption_root=root,
                source_caption_root=source_caption_root,
                source_project_root=source_project_root,
                image_root=image_root,
            )
        )
    all_paths = _caption_fallback_receipts(root)
    all_rows = [_validate_mixed_fallback_receipt(root, path) for path in all_paths]
    if {str(row["current_failed_resume_hash"]) for row in all_rows} != {
        path.stem for path in failure_files
    }:
        raise ThinkingRoundError("a caption failure remains outside the fallback registry")
    return {
        "schema_version": f"{SCHEMA_VERSION}.caption_fallback_batch",
        "status": "eligible_empty_caption_fallbacks_registered",
        "caption_root": str(root),
        "fallbacks": len(all_rows),
        "new_fallbacks": len(installed),
        "fallback_receipt_sha256s": [row["fallback_sha256"] for row in all_rows],
        "additional_provider_calls": 0,
        "provider_calls": 0,
    }


def run_caption_fallback(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    fallback_source_caption_root: Path = DEFAULT_FALLBACK_SOURCE_CAPTION_ROOT,
    image_root: Path = captions.DEFAULT_IMAGE_ROOT,
) -> dict[str, Any]:
    """Provider-cold registration for all and only exact-five-empty failures."""

    contract_path = _write_contract(caption_root)
    result = install_caption_empty_fallbacks(
        caption_root=caption_root,
        source_caption_root=fallback_source_caption_root,
        source_project_root=source_project_root,
        image_root=image_root,
    )
    return {
        **result,
        "thinking_round_contract_file": str(contract_path.resolve()),
        "thinking_round_contract_file_sha256": build._file_sha256(contract_path),  # type: ignore[attr-defined]
        "provider_calls": 0,
    }


@contextmanager
def _mixed_caption_complete_patch(root: Path) -> Iterator[None]:
    """Add explicit mixed provenance before the write-once complete is sealed."""

    active_root = Path(root).resolve()
    receipt_paths = _caption_fallback_receipts(active_root)
    if not receipt_paths:
        yield
        return
    receipts = [
        _validate_mixed_fallback_receipt(active_root, path) for path in receipt_paths
    ]
    target = active_root / "build" / "caption_complete.json"
    original = build._write_once_json  # type: ignore[attr-defined]

    def write_once(path: Path, value: Mapping[str, Any]) -> None:
        if Path(path).resolve() == target:
            if not isinstance(value, dict):
                raise ThinkingRoundError("caption complete must be mutable mapping")
            value.pop("complete_sha256", None)
            value["thinking_round_mixed_provenance"] = {
                "low_provider_final_captions": captions.EXPECTED_IMAGES
                - len(receipts),
                "prior_minimal_fallback_captions": len(receipts),
                "fallback_receipts": [
                    {
                        "resume_hash": receipt["current_failed_resume_hash"],
                        "task": receipt["task"],
                        "image": receipt["image"],
                        "sequence_index": receipt["sequence_index"],
                        "receipt": str(path.resolve()),
                        "receipt_sha256": receipt["fallback_sha256"],
                        "prev2_possible_affected_sequence_indices": receipt[
                            "prev2_possible_affected_sequence_indices"
                        ],
                    }
                    for path, receipt in zip(receipt_paths, receipts, strict=True)
                ],
                "failed_low_physical_attempts_per_fallback": MAX_GENERATION_ATTEMPTS,
                "additional_fallback_provider_calls": 0,
                "provider_final_synthesized": False,
            }
            value["complete_sha256"] = _sha256(value)
        original(path, value)

    build._write_once_json = write_once  # type: ignore[attr-defined]
    try:
        yield
    finally:
        build._write_once_json = original  # type: ignore[attr-defined]


def run_caption(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    fallback_source_caption_root: Path = DEFAULT_FALLBACK_SOURCE_CAPTION_ROOT,
    image_root: Path = captions.DEFAULT_IMAGE_ROOT,
) -> dict[str, Any]:
    """Resume all captions, transparently importing only exact-five-empty failures."""

    _write_contract(caption_root)
    recovery = install_caption_empty_fallbacks(
        caption_root=caption_root,
        source_caption_root=fallback_source_caption_root,
        source_project_root=source_project_root,
        image_root=image_root,
    )
    while True:
        try:
            with _g35_level(G35_WORK_THINKING_LEVEL), _mixed_caption_complete_patch(
                caption_root
            ):
                captions.run_caption_phase(
                    source_project_root=source_project_root,
                    output_root=caption_root,
                    image_root=image_root,
                )
            break
        except Exception:
            next_recovery = install_caption_empty_fallbacks(
                caption_root=caption_root,
                source_caption_root=fallback_source_caption_root,
                source_project_root=source_project_root,
                image_root=image_root,
            )
            if int(next_recovery["new_fallbacks"]) < 1:
                raise
            recovery = next_recovery
    result = _read_json(
        Path(caption_root).resolve() / "build" / "caption_complete.json",
        "mixed caption complete",
    )
    mixed = result.get("thinking_round_mixed_provenance", {})
    if _caption_fallback_receipts(Path(caption_root)) and not isinstance(mixed, Mapping):
        raise ThinkingRoundError("mixed caption complete lacks explicit provenance")
    seal = seal_caption(caption_root=caption_root)
    return {
        **result,
        "thinking_round_contract_file": str(
            (Path(caption_root).resolve() / "build" / "thinking_round_contract.json")
        ),
        "thinking_level": G35_WORK_THINKING_LEVEL,
        "caption_thinking_seal_sha256": seal["seal_sha256"],
        "fallbacks": seal["prior_minimal_fallback_captions"],
        "last_recovery": recovery,
    }


def _caption_low_protocol_sha256() -> str:
    return _caption_protocol_sha256(G35_WORK_THINKING_LEVEL)


def seal_caption(
    *, caption_root: Path = DEFAULT_CAPTION_ROOT,
) -> dict[str, Any]:
    """Provider-cold audit and seal for a completed externally-started caption run."""

    root = Path(caption_root).resolve()
    complete = base._caption_complete(root)  # type: ignore[attr-defined]
    expected_protocol = _caption_low_protocol_sha256()
    plan_root = root / "provider" / "g35" / "plans" / "caption"
    response_root = root / "provider" / "g35" / "responses" / "caption"
    attempt_root = root / "provider" / "g35" / "attempts" / "caption"
    caption_files = sorted((root / "captions" / "g35").rglob("*.json"))
    plan_files = sorted(plan_root.glob("*.json"))
    response_files = sorted(response_root.glob("*.json"))
    receipt_paths = _caption_fallback_receipts(root)
    receipts = [_validate_mixed_fallback_receipt(root, path) for path in receipt_paths]
    fallback_count = len(receipts)
    if (
        len(caption_files) != captions.EXPECTED_IMAGES
        or len(plan_files) != captions.EXPECTED_IMAGES
        or len(response_files) != captions.EXPECTED_IMAGES - fallback_count
    ):
        raise ThinkingRoundError("mixed caption/plan/provider-final census drift")

    plan_request_hashes: set[str] = set()
    plan_rows: list[dict[str, Any]] = []
    attempt_intents = 0
    for plan_path in plan_files:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            not isinstance(plan, dict)
            or plan.get("stage") != "caption"
            or plan.get("model") != g35_runtime.MODEL_ID
            or plan.get("protocol_sha256") != expected_protocol
            or plan.get("max_attempts") != MAX_GENERATION_ATTEMPTS
            or plan.get("resume_hash") != plan_path.stem
        ):
            raise ThinkingRoundError(f"G35-low caption plan drift: {plan_path}")
        request_sha = str(plan.get("request_sha256", ""))
        if len(request_sha) != 64:
            raise ThinkingRoundError(f"caption plan request hash drift: {plan_path}")
        plan_request_hashes.add(request_sha)
        identity_attempt_root = attempt_root / plan_path.stem
        intents = sorted(identity_attempt_root.glob("attempt_*_request.json"))
        replies = sorted(identity_attempt_root.glob("attempt_*_response.json"))
        if not 1 <= len(intents) <= MAX_GENERATION_ATTEMPTS or len(replies) != len(intents):
            raise ThinkingRoundError(f"caption physical-attempt chain drift: {plan_path.stem}")
        attempt_intents += len(intents)
        plan_rows.append(
            {
                "resume_hash": plan_path.stem,
                "request_sha256": request_sha,
                "attempts": len(intents),
                "plan_file_sha256": build._file_sha256(plan_path),  # type: ignore[attr-defined]
            }
        )
    if len(plan_request_hashes) != captions.EXPECTED_IMAGES:
        raise ThinkingRoundError("caption logical request identities are not unique")

    final_request_hashes: set[str] = set()
    final_identities: set[str] = set()
    for response_path in response_files:
        final = json.loads(response_path.read_text(encoding="utf-8"))
        ordinal = final.get("attempt_ordinal") if isinstance(final, dict) else None
        if (
            not isinstance(final, dict)
            or final.get("model") != g35_runtime.MODEL_ID
            or final.get("stage") != "caption"
            or not isinstance(ordinal, int)
            or not 1 <= ordinal <= MAX_GENERATION_ATTEMPTS
            or final.get("resume_hash") != response_path.stem
        ):
            raise ThinkingRoundError(f"caption final response drift: {response_path}")
        final_request_hashes.add(str(final.get("request_sha256", "")))
        final_identities.add(response_path.stem)

    failure_files = sorted(
        (root / "provider" / "g35" / "failures" / "caption").glob("*.json")
    )
    failure_identities = {path.stem for path in failure_files}
    receipt_identities = {
        str(receipt["current_failed_resume_hash"]) for receipt in receipts
    }
    plan_identities = {path.stem for path in plan_files}
    if (
        failure_identities != receipt_identities
        or plan_identities - final_identities != receipt_identities
        or final_identities & receipt_identities
        or final_request_hashes
        != {
            str(_read_json(path, "low caption plan")["request_sha256"])
            for path in plan_files
            if path.stem in final_identities
        }
    ):
        raise ThinkingRoundError("provider finals/failures/fallback receipts disagree")

    normal_record_request_hashes: set[str] = set()
    record_fallback_identities: set[str] = set()
    for caption_path in caption_files:
        record = _read_json(caption_path, "caption checkpoint")
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != captions.CAPTION_SCHEMA_VERSION
            or record.get("model") != g35_runtime.MODEL_ID
            or not str(record.get("caption", "")).strip()
        ):
            raise ThinkingRoundError(f"caption checkpoint drift: {caption_path}")
        fallback = record.get("thinking_round_fallback")
        if isinstance(fallback, Mapping):
            failed_identity = str(fallback.get("failed_low_resume_hash", ""))
            matches = [
                receipt
                for receipt in receipts
                if receipt["current_failed_resume_hash"] == failed_identity
            ]
            if (
                len(matches) != 1
                or fallback.get("kind") != "zero_call_prior_minimal_caption"
                or fallback.get("additional_provider_calls") != 0
                or fallback.get("provider_final_synthesized") is not False
                or record.get("provider_request_sha256")
                != matches[0]["source_provider_request_sha256"]
                or record.get("provider_response_sha256")
                != matches[0]["source_provider_response_sha256"]
            ):
                raise ThinkingRoundError("fallback caption/source receipt drift")
            record_fallback_identities.add(failed_identity)
        else:
            normal_record_request_hashes.add(
                str(record.get("provider_request_sha256", ""))
            )
    if (
        normal_record_request_hashes != final_request_hashes
        or record_fallback_identities != receipt_identities
    ):
        raise ThinkingRoundError("caption checkpoints are not bound to mixed evidence")

    orphan_files = list((root / "provider" / "g35" / "orphans" / "caption").glob("*.json"))
    if orphan_files:
        raise ThinkingRoundError("completed caption bank contains an orphan terminal")
    usage_calls = int(complete["usage"]["stages"]["caption"]["provider_calls"])
    if not 0 <= usage_calls <= attempt_intents:
        raise ThinkingRoundError("caption usage/attempt census drift")
    mixed = complete.get("thinking_round_mixed_provenance")
    if fallback_count:
        if (
            not isinstance(mixed, Mapping)
            or mixed.get("low_provider_final_captions")
            != captions.EXPECTED_IMAGES - fallback_count
            or mixed.get("prior_minimal_fallback_captions") != fallback_count
            or mixed.get("additional_fallback_provider_calls") != 0
            or mixed.get("provider_final_synthesized") is not False
        ):
            raise ThinkingRoundError("caption complete lacks exact mixed provenance")

    contract_path = _write_contract(root)
    seal = {
        "schema_version": f"{SCHEMA_VERSION}.caption_seal",
        "complete": True,
        "status": (
            "caption_mixed_low_prior_minimal_complete_provider_cold_seal"
            if fallback_count
            else "caption_low_complete_provider_cold_seal"
        ),
        "caption_root": str(root),
        "captions": captions.EXPECTED_IMAGES,
        "logical_requests": len(plan_files),
        "physical_attempt_intents": attempt_intents,
        "complete_process_provider_calls": usage_calls,
        "max_physical_attempts_per_identity": MAX_GENERATION_ATTEMPTS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "thinking_level": G35_WORK_THINKING_LEVEL,
        "g35_low_protocol_sha256": expected_protocol,
        "caption_complete_sha256": complete["complete_sha256"],
        "caption_complete_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
            root / "build" / "caption_complete.json"
        ),
        "thinking_contract_file_sha256": build._file_sha256(contract_path),  # type: ignore[attr-defined]
        "plan_rows_sha256": _sha256(plan_rows),
        "low_provider_final_captions": len(response_files),
        "prior_minimal_fallback_captions": fallback_count,
        "fallback_receipts": [
            {
                "resume_hash": receipt["current_failed_resume_hash"],
                "task": receipt["task"],
                "image": receipt["image"],
                "sequence_index": receipt["sequence_index"],
                "receipt_path": str(path.resolve()),
                "receipt_sha256": receipt["fallback_sha256"],
                "prev2_possible_affected_sequence_indices": receipt[
                    "prev2_possible_affected_sequence_indices"
                ],
            }
            for path, receipt in zip(receipt_paths, receipts, strict=True)
        ],
        "provider_final_synthesized": False,
        "orphans": 0,
        "failures_preserved": len(failure_files),
        "unregistered_failures": 0,
        "provider_calls": 0,
    }
    seal["seal_sha256"] = _sha256(seal)
    path = root / "build" / "caption_thinking_seal.json"
    build._write_once_json(path, seal)  # type: ignore[attr-defined]
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != seal:
        raise ThinkingRoundError("caption thinking seal write-once drift")
    return seal


def _require_caption_seal(caption_root: Path) -> dict[str, Any]:
    root = Path(caption_root).resolve()
    path = root / "build" / "caption_thinking_seal.json"
    if not path.is_file():
        raise ThinkingRoundError("caption thinking seal is required before writer")
    seal = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(seal, dict):
        raise ThinkingRoundError("caption thinking seal is not an object")
    low_count = seal.get("low_provider_final_captions")
    fallback_count = seal.get("prior_minimal_fallback_captions")
    if (
        seal.get("schema_version") != f"{SCHEMA_VERSION}.caption_seal"
        or seal.get("complete") is not True
        or Path(str(seal.get("caption_root", ""))).resolve() != root
        or seal.get("captions") != captions.EXPECTED_IMAGES
        or seal.get("thinking_level") != G35_WORK_THINKING_LEVEL
        or seal.get("g35_low_protocol_sha256") != _caption_low_protocol_sha256()
        or seal.get("max_physical_attempts_per_identity") != MAX_GENERATION_ATTEMPTS
        or float(seal.get("retry_wait_seconds", -1)) != RETRY_WAIT_SECONDS
        or seal.get("provider_calls") != 0
        or seal.get("orphans") != 0
        or not isinstance(low_count, int)
        or not isinstance(fallback_count, int)
        or low_count + fallback_count != captions.EXPECTED_IMAGES
        or seal.get("failures_preserved") != fallback_count
        or seal.get("unregistered_failures") != 0
        or seal.get("provider_final_synthesized") is not False
    ):
        raise ThinkingRoundError("caption thinking seal contract drift")
    body = dict(seal)
    digest = body.pop("seal_sha256", None)
    if digest != _sha256(body):
        raise ThinkingRoundError("caption thinking seal digest drift")
    complete_path = root / "build" / "caption_complete.json"
    if seal.get("caption_complete_file_sha256") != build._file_sha256(complete_path):  # type: ignore[attr-defined]
        raise ThinkingRoundError("sealed caption complete file changed")
    receipt_paths = _caption_fallback_receipts(root)
    receipts = [_validate_mixed_fallback_receipt(root, path) for path in receipt_paths]
    sealed_receipts = seal.get("fallback_receipts")
    if (
        len(receipts) != fallback_count
        or not isinstance(sealed_receipts, list)
        or [
            (row["current_failed_resume_hash"], row["fallback_sha256"])
            for row in receipts
        ]
        != [
            (str(row.get("resume_hash", "")), str(row.get("receipt_sha256", "")))
            for row in sealed_receipts
            if isinstance(row, Mapping)
        ]
    ):
        raise ThinkingRoundError("sealed caption fallback registry changed")
    return seal


WRITER_PROJECTION_SCHEMA = f"{SCHEMA_VERSION}.writer_declared_projection"


def _session_legal_source_ids(
    source_project_root: Path,
    unit_id: str,
    cache: dict[str, list[str]],
) -> list[str]:
    """Rebuild the actual session authority; model output cannot define legality."""

    if unit_id in cache:
        return list(cache[unit_id])
    marker = "::session::"
    if marker not in unit_id:
        raise ThinkingRoundError("creamem projection unit is not a session")
    task, rendered_session = unit_id.rsplit(marker, 1)
    try:
        session_id = int(rendered_session)
    except ValueError as exc:
        raise ThinkingRoundError("creamem projection session id drift") from exc
    project = base._project(Path(source_project_root).resolve())  # type: ignore[attr-defined]
    task_rows = [
        row
        for row in build._project_tasks(project)  # type: ignore[attr-defined]
        if str(row.get("task", "")) == task
    ]
    if len(task_rows) != 1:
        raise ThinkingRoundError("creamem projection task authority drift")
    task_row = task_rows[0]
    caption_map, _receipt = build.load_g25_caption_bank_for_task(
        task_row, matrix.DEFAULT_V7_CAPTION_ROOT
    )
    records, _source_map = ingest.build_records_from_captions(
        task, task_row["memory"], caption_map
    )
    matches = [row for row in records if int(row["session_id"]) == session_id]
    if len(matches) != 1:
        raise ThinkingRoundError("creamem projection session authority drift")
    legal = [str(turn["dia_id"]) for turn in matches[0].get("turns", [])]
    if not legal or len(set(legal)) != len(legal):
        raise ThinkingRoundError("creamem projection legal source IDs drift")
    cache[unit_id] = legal
    return list(legal)


def _decode_native_truncated_memory_prefix(
    raw_text: str,
) -> tuple[list[dict[str, Any]], str]:
    """Decode only complete array items; never infer the truncated next item."""

    text = str(raw_text)
    decoder = json.JSONDecoder()
    cursor = 0

    def whitespace(position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    cursor = whitespace(cursor)
    if cursor >= len(text) or text[cursor] != "{":
        raise ThinkingRoundError("native projection lacks outer object")
    cursor = whitespace(cursor + 1)
    try:
        key, consumed = decoder.raw_decode(text[cursor:])
    except json.JSONDecodeError as exc:
        raise ThinkingRoundError("native projection key is incomplete") from exc
    cursor = whitespace(cursor + consumed)
    if key != "memory" or cursor >= len(text) or text[cursor] != ":":
        raise ThinkingRoundError("native projection is not the memory schema")
    cursor = whitespace(cursor + 1)
    if cursor >= len(text) or text[cursor] != "[":
        raise ThinkingRoundError("native projection memory is not an array")
    cursor += 1
    items: list[dict[str, Any]] = []
    while True:
        cursor = whitespace(cursor)
        if cursor >= len(text):
            raise ThinkingRoundError("native projection ended before a next item")
        if items:
            if text[cursor] != ",":
                raise ThinkingRoundError("native projection item separator drift")
            cursor = whitespace(cursor + 1)
        if cursor < len(text) and text[cursor] == "]":
            raise ThinkingRoundError("native projection was complete, not truncated")
        try:
            value, consumed = decoder.raw_decode(text[cursor:])
        except json.JSONDecodeError:
            fragment = text[cursor:]
            break
        if not isinstance(value, dict):
            raise ThinkingRoundError("native projection contains a non-object item")
        expected_keys = {"id", "text", "attributed_to"}
        if (
            set(value) != expected_keys
            or value.get("id") != str(len(items))
            or not isinstance(value.get("text"), str)
            or not str(value["text"]).strip()
            or value.get("attributed_to") not in {"user", "assistant"}
        ):
            raise ThinkingRoundError("native projection complete item schema drift")
        items.append(dict(value))
        cursor += consumed
    expected_fragment_prefix = f'{{"id": "{len(items)}", "'
    if (
        not items
        or not 1 <= len(fragment) <= 63
        or not fragment.startswith(expected_fragment_prefix)
        or "}" in fragment
        or "]" in fragment
    ):
        raise ThinkingRoundError("native projection incomplete tail is not narrow")
    return items, fragment


def _decode_native_python_apostrophe_escape(
    raw_text: str,
) -> tuple[list[dict[str, Any]], list[int], str]:
    """Remove exactly two Python-style ``\'`` escapes, preserving all text."""

    text = str(raw_text)
    if not text.rstrip().endswith("]}"):
        raise ThinkingRoundError("apostrophe repair source is not a closed outer object")
    positions = [index for index, value in enumerate(text) if value == "\\"]
    if len(positions) != 2 or any(
        index + 1 >= len(text) or text[index + 1] != "'" for index in positions
    ):
        raise ThinkingRoundError("apostrophe repair is not exactly two invalid escapes")
    # Remove the two backslashes by position; the apostrophes themselves and
    # every other provider byte remain represented in the rebuilt JSON value.
    repaired_text = "".join(
        value for index, value in enumerate(text) if index not in set(positions)
    )
    try:
        value = json.loads(repaired_text)
    except json.JSONDecodeError as exc:
        raise ThinkingRoundError("apostrophe repair did not yield complete JSON") from exc
    if not isinstance(value, dict) or set(value) != {"memory"}:
        raise ThinkingRoundError("apostrophe repair outer memory schema drift")
    items = value["memory"]
    if not isinstance(items, list) or not items:
        raise ThinkingRoundError("apostrophe repair memory list drift")
    for index, item in enumerate(items):
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "text", "attributed_to"}
            or item.get("id") != str(index)
            or not isinstance(item.get("text"), str)
            or not str(item["text"]).strip()
            or item.get("attributed_to") not in {"user", "assistant"}
        ):
            raise ThinkingRoundError("apostrophe repair complete item schema drift")
    return [dict(item) for item in items], positions, repaired_text


def _writer_projection_evidence_rows(
    backend: Any,
    *,
    stage: str,
    prompt: str,
    system: str | None,
    json_mode: bool,
    max_output_tokens: int,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Validate five successful application outputs without touching provider state."""

    if stage != "writer" or not json_mode:
        return None
    substage = str(metadata.get("substage", ""))
    if substage not in {"creamem_fact", "native_fact"}:
        return None
    cache_root = Path(backend.facade.cache_root).resolve()
    source = backend.source
    original = backend.stage_limits[stage]
    normalized_metadata = dict(metadata)
    normalized_metadata["matched_substage"] = substage
    normalized_metadata["max_provider_attempts"] = MAX_GENERATION_ATTEMPTS
    expected: list[dict[str, Any]] = []
    for output_ordinal in range(1, MAX_GENERATION_ATTEMPTS + 1):
        attempt_metadata = dict(normalized_metadata)
        if output_ordinal > 1:
            attempt_metadata["application_output_retry_ordinal"] = output_ordinal
        remaining = MAX_GENERATION_ATTEMPTS - output_ordinal + 1
        limits = dict(backend.stage_limits)
        limits[stage] = source.StageLimits(
            original.max_output_tokens,
            original.max_live_calls,
            remaining,
        )
        request = source.build_generate_content_request(
            stage,
            prompt,
            system=system,
            json_mode=json_mode,
            max_output_tokens=max_output_tokens,
            stage_limits=limits,
        )
        request_sha = source.request_identity(
            stage=stage,
            request=request,
            stage_limits=limits,
            cache_metadata=attempt_metadata,
        )
        resume_hash = source.resume_hash(
            stage=stage,
            request_sha256=request_sha,
            stage_limits=limits,
        )
        protocol_sha = source.protocol_sha256(limits)
        plan = {
            "schema_version": source.CACHE_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha,
            "stage": stage,
            "model": source.MODEL_ID,
            "api": source.MODEL_API,
            "request_sha256": request_sha,
            "resume_hash": resume_hash,
            "payload_sha256": _sha256(request),
            "prompt_sha256": _sha256(prompt),
            "system_instruction_sha256": _sha256(
                request.get("systemInstruction", {})
            ),
            "cache_metadata": attempt_metadata,
            "caption_context_count": None,
            "image_parts": 0,
            "max_attempts": remaining,
        }
        expected.append(
            {
                "application_output_ordinal": output_ordinal,
                "resume_hash": resume_hash,
                "request_sha256": request_sha,
                "protocol_sha256": protocol_sha,
                "plan": plan,
                "plan_path": cache_root / "plans" / stage / f"{resume_hash}.json",
                "final_path": cache_root
                / "responses"
                / stage
                / f"{resume_hash}.json",
                "attempt_root": cache_root / "attempts" / stage / resume_hash,
                "lock_path": cache_root
                / "lifecycle_locks"
                / stage
                / f"{resume_hash}.lock",
            }
        )
    # Ordinal five is the durable proof that the application exhausted its
    # approved five outputs.  Before that point the normal backend owns retry.
    if not expected[-1]["final_path"].is_file():
        return None

    expected_ids = {str(row["resume_hash"]) for row in expected}
    unit_id = str(normalized_metadata.get("unit_id", ""))
    observed_ids: set[str] = set()
    for plan_path in (cache_root / "plans" / stage).glob("*.json"):
        plan = _read_json(plan_path, "writer projection candidate plan")
        candidate = plan.get("cache_metadata")
        if (
            isinstance(candidate, Mapping)
            and str(candidate.get("unit_id", "")) == unit_id
            and candidate.get("substage") == substage
        ):
            observed_ids.add(plan_path.stem)
    if observed_ids != expected_ids:
        raise ThinkingRoundError("writer projection application output census drift")

    evidence_rows: list[dict[str, Any]] = []
    raw_texts: list[str] = []
    for row in expected:
        identity = str(row["resume_hash"])
        plan_path = Path(row["plan_path"])
        final_path = Path(row["final_path"])
        attempt_root = Path(row["attempt_root"])
        lock_path = Path(row["lock_path"])
        ordinal = int(row["application_output_ordinal"])
        attempt_request_path = attempt_root / "attempt_001_request.json"
        attempt_response_path = attempt_root / "attempt_001_response.json"
        if (
            _read_json(plan_path, "writer projection plan") != row["plan"]
            or {path.name for path in attempt_root.iterdir() if path.is_file()}
            != {attempt_request_path.name, attempt_response_path.name}
            or any(path.is_dir() for path in attempt_root.iterdir())
            or not final_path.is_file()
            or not lock_path.is_file()
            or lock_path.read_bytes() != b"\0"
            or (cache_root / "failures" / stage / f"{identity}.json").exists()
            or (cache_root / "orphans" / stage / f"{identity}.json").exists()
        ):
            raise ThinkingRoundError("writer projection immutable chain drift")
        plan_sha = _sha256(row["plan"])
        expected_attempt_request = {
            "schema_version": backend.source.CACHE_SCHEMA_VERSION,
            "protocol_sha256": row["protocol_sha256"],
            "stage": stage,
            "model": backend.source.MODEL_ID,
            "request_sha256": row["request_sha256"],
            "resume_hash": identity,
            "attempt_ordinal": 1,
            "attempt_identity_sha256": _sha256(
                {
                    "resume_hash": identity,
                    "attempt_ordinal": 1,
                    "plan_sha256": plan_sha,
                }
            ),
            "plan_sha256": plan_sha,
        }
        attempt_request = _read_json(
            attempt_request_path, "writer projection attempt request"
        )
        attempt_response = _read_json(
            attempt_response_path, "writer projection attempt response"
        )
        final = _read_json(final_path, "writer projection provider final")
        common = {
            key: expected_attempt_request[key]
            for key in (
                "schema_version",
                "protocol_sha256",
                "stage",
                "model",
                "request_sha256",
                "resume_hash",
                "attempt_ordinal",
            )
        }
        raw_text = str(attempt_response.get("text", ""))
        usage = attempt_response.get("usage")
        if (
            attempt_request != expected_attempt_request
            or set(attempt_response)
            != set(common) | {"attempt_identity_sha256", "outcome_kind", "response_sha256", "text", "usage"}
            or any(attempt_response.get(key) != value for key, value in common.items())
            or attempt_response.get("attempt_identity_sha256")
            != expected_attempt_request["attempt_identity_sha256"]
            or attempt_response.get("outcome_kind") != "raw_response"
            or attempt_response.get("response_sha256") != _sha256(raw_text)
            or not isinstance(usage, Mapping)
        ):
            raise ThinkingRoundError("writer projection successful attempt drift")
        build._validate_usage_value(  # type: ignore[attr-defined]
            usage, label="writer projection source attempt"
        )
        expected_final = {
            **{
                key: attempt_response[key]
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
            raise ThinkingRoundError("writer projection final/attempt drift")
        # Prove that no worker still owns this immutable request before using it.
        with backend.source.acquire_request_lifecycle_lock(
            lock_path, timeout_seconds=0.1, poll_seconds=0.01
        ):
            pass
        raw_texts.append(raw_text)
        evidence_rows.append(
            {
                "application_output_ordinal": ordinal,
                "resume_hash": identity,
                "request_sha256": row["request_sha256"],
                "response_sha256": attempt_response["response_sha256"],
                "plan_path": str(plan_path),
                "plan_file_sha256": build._file_sha256(plan_path),  # type: ignore[attr-defined]
                "attempt_request_path": str(attempt_request_path),
                "attempt_request_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
                    attempt_request_path
                ),
                "attempt_response_path": str(attempt_response_path),
                "attempt_response_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
                    attempt_response_path
                ),
                "final_path": str(final_path),
                "final_file_sha256": build._file_sha256(final_path),  # type: ignore[attr-defined]
                "usage": dict(usage),
            }
        )
    if len(set(raw_texts)) != 1:
        raise ThinkingRoundError("five writer application outputs are not identical")
    return evidence_rows


def _recover_declared_writer_projection(
    backend: Any,
    *,
    source_project_root: Path,
    output_root: Path,
    legal_cache: dict[str, list[str]],
    stage: str,
    prompt: str,
    system: str | None,
    json_mode: bool,
    max_output_tokens: int,
    cache_metadata: Mapping[str, Any],
    write_receipt: bool = True,
) -> build.GenerationReply | None:
    """Project only two declared, fully evidenced retry-exhausted writer shapes."""

    evidence = _writer_projection_evidence_rows(
        backend,
        stage=stage,
        prompt=prompt,
        system=system,
        json_mode=json_mode,
        max_output_tokens=max_output_tokens,
        metadata=cache_metadata,
    )
    if evidence is None:
        return None
    unit_id = str(cache_metadata.get("unit_id", ""))
    substage = str(cache_metadata.get("substage", ""))
    primary_final = _read_json(
        Path(evidence[0]["final_path"]), "selected ordinal-one projection final"
    )
    raw_text = str(primary_final["text"])
    item_audit: list[dict[str, Any]] = []
    projection_details: dict[str, Any]
    if substage == "creamem_fact":
        try:
            raw_value = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ThinkingRoundError("creamem projection raw output is incomplete") from exc
        if not isinstance(raw_value, dict) or set(raw_value) != {"memory"}:
            raise ThinkingRoundError("creamem projection raw top-level schema drift")
        raw_items = raw_value["memory"]
        if not isinstance(raw_items, list) or not raw_items:
            raise ThinkingRoundError("creamem projection memory list drift")
        direct: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if (
                not isinstance(item, dict)
                or set(item)
                != {"id", "text", "source_dia_ids", "attributed_to"}
                or not isinstance(item.get("id"), str)
                or not isinstance(item.get("text"), str)
                or not str(item["text"]).strip()
                or not isinstance(item.get("source_dia_ids"), list)
                or not item["source_dia_ids"]
                or any(not isinstance(value, str) for value in item["source_dia_ids"])
                or item.get("attributed_to") not in {"user", "assistant"}
            ):
                raise ThinkingRoundError("creamem projection memory item schema drift")
            projected_item = {
                "fact": item["text"],
                "source_dia_ids": list(item["source_dia_ids"]),
            }
            direct.append(projected_item)
            item_audit.append(
                {
                    "index": index,
                    "raw_item_sha256": _sha256(item),
                    "discarded_id_sha256": _sha256(item["id"]),
                    "discarded_attributed_to_sha256": _sha256(
                        item["attributed_to"]
                    ),
                    "projected_item_sha256": _sha256(projected_item),
                }
            )
        legal = _session_legal_source_ids(
            source_project_root, unit_id, legal_cache
        )
        _summary, validated = gm._validate_session_packet(  # type: ignore[attr-defined]
            {
                "batch_summary": "declared projection validation",
                "atomic_facts": direct,
            },
            legal,
        )
        if validated != direct:
            raise ThinkingRoundError("creamem projection changed under packet validation")
        projected_value = {"atomic_facts": direct}
        projection_details = {
            "kind": "memory_text_source_ids_to_atomic_facts",
            "raw_top_level_keys": ["memory"],
            "raw_item_exact_keys": [
                "attributed_to",
                "id",
                "source_dia_ids",
                "text",
            ],
            "projected_item_exact_keys": ["fact", "source_dia_ids"],
            "items": len(direct),
            "legal_source_dia_ids": legal,
            "legal_source_dia_ids_sha256": _sha256(legal),
            "gm_validate_session_packet_passed": True,
        }
    elif substage == "native_fact":
        dropped_fragment: str | None = None
        apostrophe_positions: list[int] | None = None
        repaired_source_text: str | None = None
        try:
            complete_items, dropped_fragment = (
                _decode_native_truncated_memory_prefix(raw_text)
            )
            native_projection_kind = "truncated_memory_complete_prefix_rebuild"
        except ThinkingRoundError:
            try:
                (
                    complete_items,
                    apostrophe_positions,
                    repaired_source_text,
                ) = _decode_native_python_apostrophe_escape(raw_text)
            except ThinkingRoundError as apostrophe_error:
                raise ThinkingRoundError(
                    "native output matches neither registered projection policy"
                ) from apostrophe_error
            native_projection_kind = "remove_python_style_apostrophe_escape"
        rebuilt_value = {"memory": complete_items}
        rebuilt_text = json.dumps(
            rebuilt_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        facts = ingest.parse_native_facts(rebuilt_text)
        expected_facts = [" ".join(str(item["text"]).split()) for item in complete_items]
        if facts != expected_facts or not facts:
            raise ThinkingRoundError("native projection failed existing fact parser")
        for index, item in enumerate(complete_items):
            item_audit.append(
                {
                    "index": index,
                    "raw_item_sha256": _sha256(item),
                    "retained_item_sha256": _sha256(item),
                    "packet_fact_sha256": _sha256(facts[index]),
                }
            )
        if native_projection_kind == "truncated_memory_complete_prefix_rebuild":
            for row in evidence:
                usage = row["usage"]
                generated = int(usage["output_tokens"]) + int(
                    usage["thought_tokens"]
                )
                if not int(max_output_tokens * 0.95) <= generated <= max_output_tokens:
                    raise ThinkingRoundError(
                        "native truncated output was not near token cap"
                    )
        projected_value = rebuilt_value
        projection_details = {
            "kind": native_projection_kind,
            "complete_items": len(complete_items),
            "complete_items_sha256": _sha256(complete_items),
            "rebuilt_memory_sha256": _sha256(rebuilt_value),
            "existing_parse_native_facts_passed": True,
            "packet_facts": len(facts),
            "packet_facts_sha256": _sha256(facts),
        }
        if dropped_fragment is not None:
            projection_details.update(
                {
                    "dropped_incomplete_fragment_length": len(dropped_fragment),
                    "dropped_incomplete_fragment_sha256": _sha256(
                        dropped_fragment
                    ),
                    "dropped_incomplete_item_inferred": False,
                    "near_output_token_cap_all_five": True,
                }
            )
        else:
            if apostrophe_positions is None or repaired_source_text is None:
                raise ThinkingRoundError("apostrophe projection state drift")
            projection_details.update(
                {
                    "removed_invalid_escape": "backslash_before_apostrophe",
                    "removed_backslash_count": len(apostrophe_positions),
                    "removed_backslash_positions": apostrophe_positions,
                    "removed_backslash_positions_sha256": _sha256(
                        apostrophe_positions
                    ),
                    "repaired_source_text_sha256": _sha256(
                        repaired_source_text
                    ),
                    "complete_outer_object_after_repair": True,
                    "items_dropped": 0,
                }
            )
    else:
        raise ThinkingRoundError("unregistered writer projection substage")

    projected_text = json.dumps(
        projected_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    root = Path(output_root).resolve()
    cache_root = Path(backend.facade.cache_root).resolve()
    if cache_root != root / "provider" / "g25":
        raise ThinkingRoundError("writer projection escaped isolated G25 root")
    receipt = {
        "schema_version": WRITER_PROJECTION_SCHEMA,
        "status": "zero_call_declared_schema_projection",
        "output_root": str(root),
        "backbone": "g25",
        "stage": "writer",
        "substage": substage,
        "unit_id": unit_id,
        "application_outputs": MAX_GENERATION_ATTEMPTS,
        "physical_provider_attempts": MAX_GENERATION_ATTEMPTS,
        "all_application_outputs_identical": True,
        "selected_application_output_ordinal": 1,
        "selection_policy": "fixed_ordinal_one_never_quality_selected",
        "source_raw_text_sha256": _sha256(raw_text),
        "source_raw_text_length": len(raw_text),
        "projected_text_sha256": _sha256(projected_text),
        "projection_details": projection_details,
        "item_audit": item_audit,
        "source_application_output_rows": evidence,
        "raw_provider_finals_preserved": True,
        "provider_final_synthesized": False,
        "provider_namespace_modified": False,
        "additional_provider_calls": 0,
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    receipt_path = (
        root
        / "fallbacks"
        / "writer_schema_projection"
        / f"{evidence[0]['resume_hash']}.json"
    )
    if write_receipt:
        build._write_once_json(receipt_path, receipt)  # type: ignore[attr-defined]
        if _read_json(receipt_path, "writer schema projection receipt") != receipt:
            raise ThinkingRoundError("writer schema projection receipt write-once drift")
    return build.GenerationReply(
        projected_text,
        str(evidence[0]["request_sha256"]),
        _sha256(projected_text),
        True,
    )


@contextmanager
def _g25_writer_projection_scope(
    *, source_project_root: Path, output_root: Path
) -> Iterator[None]:
    """Install the isolated-only projection backend for this runner call."""

    original = base._ReliableG25BuildBackend  # type: ignore[attr-defined]
    legal_cache: dict[str, list[str]] = {}
    source_root = Path(source_project_root).resolve()
    active_root = Path(output_root).resolve()

    class ProjectionBackend(original):
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
            metadata = dict(cache_metadata or {})
            if image_path is None and caption_context_count is None:
                projected = _recover_declared_writer_projection(
                    self,
                    source_project_root=source_root,
                    output_root=active_root,
                    legal_cache=legal_cache,
                    stage=stage,
                    prompt=prompt,
                    system=system,
                    json_mode=json_mode,
                    max_output_tokens=max_output_tokens,
                    cache_metadata=metadata,
                )
                if projected is not None:
                    self.logical_calls[stage] += 1
                    return projected
            try:
                return super().generate(
                    stage,
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    max_output_tokens=max_output_tokens,
                    max_provider_attempts=max_provider_attempts,
                    cache_metadata=metadata,
                    image_path=image_path,
                    caption_context_count=caption_context_count,
                )
            except Exception:
                # The fifth successful but contract-invalid application output
                # is created inside ``super().generate``.  Re-evaluate only
                # after that exact exhaustion so a newly observed registered
                # shape can continue in the same process with no sixth call.
                if image_path is None and caption_context_count is None:
                    projected = _recover_declared_writer_projection(
                        self,
                        source_project_root=source_root,
                        output_root=active_root,
                        legal_cache=legal_cache,
                        stage=stage,
                        prompt=prompt,
                        system=system,
                        json_mode=json_mode,
                        max_output_tokens=max_output_tokens,
                        cache_metadata=metadata,
                    )
                    if projected is not None:
                        return projected
                raise

    ProjectionBackend.__name__ = "_ThinkingRoundG25DeclaredProjectionBackend"
    base._ReliableG25BuildBackend = ProjectionBackend  # type: ignore[attr-defined]
    try:
        yield
    finally:
        base._ReliableG25BuildBackend = original  # type: ignore[attr-defined]


def _writer_projection_receipt_paths(output_root: Path) -> list[Path]:
    root = Path(output_root).resolve() / "fallbacks" / "writer_schema_projection"
    return sorted(root.glob("*.json")) if root.is_dir() else []


def _validate_writer_projection_receipts(
    *, source_project_root: Path, output_root: Path
) -> list[dict[str, Any]]:
    """Replay raw-to-projection-to-packet bindings without a provider call."""

    root = Path(output_root).resolve()
    provider_root = root / "provider" / "g25"
    legal_cache: dict[str, list[str]] = {}
    summaries: list[dict[str, Any]] = []
    for receipt_path in _writer_projection_receipt_paths(root):
        receipt = _read_json(receipt_path, "writer projection receipt")
        body = dict(receipt)
        digest = body.pop("receipt_sha256", None)
        if (
            digest != _sha256(body)
            or receipt.get("schema_version") != WRITER_PROJECTION_SCHEMA
            or receipt.get("status") != "zero_call_declared_schema_projection"
            or Path(str(receipt.get("output_root", ""))).resolve() != root
            or receipt.get("backbone") != "g25"
            or receipt.get("stage") != "writer"
            or receipt.get("application_outputs") != MAX_GENERATION_ATTEMPTS
            or receipt.get("physical_provider_attempts")
            != MAX_GENERATION_ATTEMPTS
            or receipt.get("selected_application_output_ordinal") != 1
            or receipt.get("selection_policy")
            != "fixed_ordinal_one_never_quality_selected"
            or receipt.get("all_application_outputs_identical") is not True
            or receipt.get("raw_provider_finals_preserved") is not True
            or receipt.get("provider_final_synthesized") is not False
            or receipt.get("provider_namespace_modified") is not False
            or receipt.get("additional_provider_calls") != 0
        ):
            raise ThinkingRoundError("writer projection receipt contract drift")
        evidence = receipt.get("source_application_output_rows")
        if not isinstance(evidence, list) or len(evidence) != MAX_GENERATION_ATTEMPTS:
            raise ThinkingRoundError("writer projection evidence census drift")
        raw_texts: list[str] = []
        identities: list[str] = []
        for ordinal, row in enumerate(evidence, start=1):
            if (
                not isinstance(row, Mapping)
                or row.get("application_output_ordinal") != ordinal
            ):
                raise ThinkingRoundError("writer projection evidence order drift")
            identity = str(row.get("resume_hash", ""))
            plan_path = Path(str(row.get("plan_path", ""))).resolve()
            attempt_request_path = Path(
                str(row.get("attempt_request_path", ""))
            ).resolve()
            attempt_response_path = Path(
                str(row.get("attempt_response_path", ""))
            ).resolve()
            final_path = Path(str(row.get("final_path", ""))).resolve()
            expected_plan = provider_root / "plans" / "writer" / f"{identity}.json"
            expected_attempt_root = (
                provider_root / "attempts" / "writer" / identity
            )
            if (
                plan_path != expected_plan
                or attempt_request_path
                != expected_attempt_root / "attempt_001_request.json"
                or attempt_response_path
                != expected_attempt_root / "attempt_001_response.json"
                or final_path
                != provider_root / "responses" / "writer" / f"{identity}.json"
            ):
                raise ThinkingRoundError("writer projection evidence escaped root")
            for path, sha_key in (
                (plan_path, "plan_file_sha256"),
                (attempt_request_path, "attempt_request_file_sha256"),
                (attempt_response_path, "attempt_response_file_sha256"),
                (final_path, "final_file_sha256"),
            ):
                if (
                    not path.is_file()
                    or build._file_sha256(path) != row.get(sha_key)  # type: ignore[attr-defined]
                ):
                    raise ThinkingRoundError("writer projection source file changed")
            if (
                (provider_root / "failures" / "writer" / f"{identity}.json").exists()
                or (
                    provider_root / "orphans" / "writer" / f"{identity}.json"
                ).exists()
            ):
                raise ThinkingRoundError("writer projection source became terminal")
            plan = _read_json(plan_path, "writer projection source plan")
            attempt_response = _read_json(
                attempt_response_path, "writer projection source attempt"
            )
            final = _read_json(final_path, "writer projection source final")
            raw_text = str(final.get("text", ""))
            if (
                plan.get("resume_hash") != identity
                or plan.get("stage") != "writer"
                or plan.get("cache_metadata", {}).get("unit_id")
                != receipt.get("unit_id")
                or plan.get("cache_metadata", {}).get("substage")
                != receipt.get("substage")
                or plan.get("max_attempts")
                != MAX_GENERATION_ATTEMPTS - ordinal + 1
                or final.get("resume_hash") != identity
                or final.get("attempt_ordinal") != 1
                or final.get("request_sha256") != row.get("request_sha256")
                or final.get("response_sha256") != row.get("response_sha256")
                or final.get("response_sha256") != _sha256(raw_text)
                or attempt_response.get("text") != raw_text
                or attempt_response.get("response_sha256")
                != final.get("response_sha256")
                or attempt_response.get("outcome_kind") != "raw_response"
                or row.get("usage") != final.get("usage")
            ):
                raise ThinkingRoundError("writer projection source chain drift")
            raw_texts.append(raw_text)
            identities.append(identity)
        if len(set(raw_texts)) != 1 or len(set(identities)) != len(identities):
            raise ThinkingRoundError("writer projection five-output binding drift")
        raw_text = raw_texts[0]
        if (
            receipt_path.stem != identities[0]
            or receipt.get("source_raw_text_sha256") != _sha256(raw_text)
            or receipt.get("source_raw_text_length") != len(raw_text)
        ):
            raise ThinkingRoundError("writer projection selected raw binding drift")

        substage = str(receipt.get("substage", ""))
        unit_id = str(receipt.get("unit_id", ""))
        expected_item_audit: list[dict[str, Any]] = []
        if substage == "creamem_fact":
            raw_value = json.loads(raw_text)
            if not isinstance(raw_value, dict) or set(raw_value) != {"memory"}:
                raise ThinkingRoundError("sealed creamem raw schema drift")
            direct: list[dict[str, Any]] = []
            for index, item in enumerate(raw_value["memory"]):
                if not isinstance(item, dict):
                    raise ThinkingRoundError("sealed creamem item drift")
                projected = {
                    "fact": item["text"],
                    "source_dia_ids": list(item["source_dia_ids"]),
                }
                direct.append(projected)
                expected_item_audit.append(
                    {
                        "index": index,
                        "raw_item_sha256": _sha256(item),
                        "discarded_id_sha256": _sha256(item["id"]),
                        "discarded_attributed_to_sha256": _sha256(
                            item["attributed_to"]
                        ),
                        "projected_item_sha256": _sha256(projected),
                    }
                )
            legal = _session_legal_source_ids(
                source_project_root, unit_id, legal_cache
            )
            _summary, normalized = gm._validate_session_packet(  # type: ignore[attr-defined]
                {
                    "batch_summary": "declared projection validation",
                    "atomic_facts": direct,
                },
                legal,
            )
            if normalized != direct:
                raise ThinkingRoundError("sealed creamem projection validation drift")
            projected_value: dict[str, Any] = {"atomic_facts": direct}
            task, rendered_index = unit_id.rsplit("::session::", 1)
            packet_path = (
                root
                / "build"
                / "packets"
                / "g25"
                / "session"
                / task
                / f"S{int(rendered_index):04d}.json"
            )
            packet_field = "atomic_facts"
            expected_packet_value: Any = direct
        elif substage == "native_fact":
            details = receipt.get("projection_details")
            if not isinstance(details, Mapping):
                raise ThinkingRoundError("sealed native projection details drift")
            kind = details.get("kind")
            if kind == "truncated_memory_complete_prefix_rebuild":
                complete_items, fragment = _decode_native_truncated_memory_prefix(
                    raw_text
                )
                if (
                    details.get("dropped_incomplete_fragment_length")
                    != len(fragment)
                    or details.get("dropped_incomplete_fragment_sha256")
                    != _sha256(fragment)
                    or details.get("near_output_token_cap_all_five") is not True
                ):
                    raise ThinkingRoundError(
                        "sealed native dropped-tail receipt drift"
                    )
            elif kind == "remove_python_style_apostrophe_escape":
                (
                    complete_items,
                    positions,
                    repaired_source_text,
                ) = _decode_native_python_apostrophe_escape(raw_text)
                if (
                    details.get("removed_backslash_count") != len(positions)
                    or details.get("removed_backslash_positions") != positions
                    or details.get("removed_backslash_positions_sha256")
                    != _sha256(positions)
                    or details.get("repaired_source_text_sha256")
                    != _sha256(repaired_source_text)
                    or details.get("items_dropped") != 0
                ):
                    raise ThinkingRoundError(
                        "sealed native apostrophe repair receipt drift"
                    )
            else:
                raise ThinkingRoundError("sealed native projection kind drift")
            projected_value = {"memory": complete_items}
            rebuilt = json.dumps(
                projected_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            facts = ingest.parse_native_facts(rebuilt)
            expected = [" ".join(str(item["text"]).split()) for item in complete_items]
            if facts != expected:
                raise ThinkingRoundError("sealed native projection parser drift")
            expected_item_audit = [
                {
                    "index": index,
                    "raw_item_sha256": _sha256(item),
                    "retained_item_sha256": _sha256(item),
                    "packet_fact_sha256": _sha256(facts[index]),
                }
                for index, item in enumerate(complete_items)
            ]
            task, rendered_index = unit_id.rsplit("::native::", 1)
            packet_path = (
                root
                / "build"
                / "packets"
                / "g25"
                / "mgtm"
                / task
                / f"NB{int(rendered_index):04d}.json"
            )
            packet_field = "facts"
            expected_packet_value = facts
        else:
            raise ThinkingRoundError("sealed writer projection substage drift")
        projected_text = json.dumps(
            projected_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        packet = _read_json(packet_path, "writer projection target packet")
        if (
            receipt.get("projected_text_sha256") != _sha256(projected_text)
            or receipt.get("item_audit") != expected_item_audit
            or packet.get("backbone") != "g25"
            or packet.get(packet_field) != expected_packet_value
            or packet.get("fact_response_sha256") != _sha256(projected_text)
        ):
            raise ThinkingRoundError("writer projection raw-to-packet drift")
        summaries.append(
            {
                "unit_id": unit_id,
                "substage": substage,
                "selected_application_output_ordinal": 1,
                "source_raw_text_sha256": receipt["source_raw_text_sha256"],
                "projected_text_sha256": receipt["projected_text_sha256"],
                "receipt_path": str(receipt_path.resolve()),
                "receipt_file_sha256": build._file_sha256(receipt_path),  # type: ignore[attr-defined]
                "receipt_sha256": receipt["receipt_sha256"],
                "packet_path": str(packet_path.resolve()),
                "packet_file_sha256": build._file_sha256(packet_path),  # type: ignore[attr-defined]
                "additional_provider_calls": 0,
            }
        )
    return summaries


@contextmanager
def _writer_projection_evidence_scope(
    *, source_project_root: Path, output_root: Path
) -> Iterator[None]:
    """Make the isolated generation-evidence seal disclose every projection."""

    original = build._writer_generation_evidence_contract  # type: ignore[attr-defined]

    def contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original(*args, **kwargs)
        receipts = _validate_writer_projection_receipts(
            source_project_root=source_project_root,
            output_root=output_root,
        )
        if receipts:
            value["declared_writer_schema_projections"] = {
                "count": len(receipts),
                "rows": receipts,
                "rows_sha256": _sha256(receipts),
                "raw_to_projection_to_packet_validated": True,
                "fixed_ordinal_one": True,
                "additional_provider_calls": 0,
                "provider_finals_synthesized": 0,
            }
        return value

    build._writer_generation_evidence_contract = contract  # type: ignore[attr-defined]
    try:
        yield
    finally:
        build._writer_generation_evidence_contract = original  # type: ignore[attr-defined]


def _seal_writer_projection_complete(
    *, source_project_root: Path, output_root: Path
) -> dict[str, Any] | None:
    root = Path(output_root).resolve()
    writer_path = root / "build" / "writer_complete.json"
    if not writer_path.is_file():
        return None
    receipts = _validate_writer_projection_receipts(
        source_project_root=source_project_root,
        output_root=root,
    )
    evidence_path = root / "build" / "writer_generation_evidence.json"
    if not evidence_path.is_file():
        raise ThinkingRoundError("projection-aware writer generation evidence missing")
    evidence = _read_json(evidence_path, "projection-aware generation evidence")
    declared = evidence.get("declared_writer_schema_projections")
    if (
        not isinstance(declared, Mapping)
        or declared.get("count") != len(receipts)
        or declared.get("rows") != receipts
        or declared.get("additional_provider_calls") != 0
    ):
        raise ThinkingRoundError("writer generation evidence projection drift")
    value = {
        "schema_version": f"{SCHEMA_VERSION}.writer_projection_complete",
        "complete": True,
        "status": "writer_complete_with_declared_schema_projections",
        "output_root": str(root),
        "declared_schema_projections": len(receipts),
        "projection_rows": receipts,
        "projection_rows_sha256": _sha256(receipts),
        "writer_complete_file_sha256": build._file_sha256(writer_path),  # type: ignore[attr-defined]
        "writer_generation_evidence_file_sha256": build._file_sha256(  # type: ignore[attr-defined]
            evidence_path
        ),
        "raw_to_projection_to_packet_validated": True,
        "provider_finals_synthesized": 0,
        "additional_provider_calls": 0,
        "provider_calls": 0,
    }
    value["complete_sha256"] = _sha256(value)
    path = root / "build" / "writer_projection_complete.json"
    build._write_once_json(path, value)  # type: ignore[attr-defined]
    if _read_json(path, "writer projection completion") != value:
        raise ThinkingRoundError("writer projection completion write-once drift")
    return value


class _FreshAllBackbonesQwenReranker(base._BackboneRoutingQwenReranker):  # type: ignore[attr-defined]
    """Use the fresh live namespace for both new thinking-round backbones."""

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        cache_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        metadata = dict(cache_metadata or {})
        if str(metadata.get("backbone", "")) not in {"g25", "g35"}:
            raise ThinkingRoundError("Qwen metadata lacks a thinking-round backbone")
        return self.g35_recovery.score(
            query,
            documents,
            instruction=instruction,
            cache_metadata=metadata,
        )

    def usage_receipt(self) -> Mapping[str, Any]:
        value = dict(super().usage_receipt())
        value["profile"] = "thinking_round_fresh_all_backbones_live"
        value["isolated_g25_live_calls_authorized"] = True
        return value


@contextmanager
def _fresh_qwen_routing() -> Iterator[None]:
    original = base._BackboneRoutingQwenReranker  # type: ignore[attr-defined]
    base._BackboneRoutingQwenReranker = _FreshAllBackbonesQwenReranker  # type: ignore[attr-defined]
    try:
        yield
    finally:
        base._BackboneRoutingQwenReranker = original  # type: ignore[attr-defined]


def run_writer(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    writer_backbone: str = "both",
    writer_tasks: Sequence[str] = (),
    writer_task_workers: int = 1,
) -> dict[str, Any]:
    seal = _require_caption_seal(caption_root)
    _write_contract(output_root)
    install_g25_thinking_contracts()
    with _g35_level(G35_WORK_THINKING_LEVEL), _g25_writer_projection_scope(
        source_project_root=source_project_root,
        output_root=output_root,
    ), _writer_projection_evidence_scope(
        source_project_root=source_project_root,
        output_root=output_root,
    ):
        result = base.run_writer(
            source_project_root=source_project_root,
            caption_root=caption_root,
            output_root=output_root,
            writer_backbone=writer_backbone,
            writer_tasks=writer_tasks,
            writer_task_workers=writer_task_workers,
        )
        projection_complete = _seal_writer_projection_complete(
            source_project_root=source_project_root,
            output_root=output_root,
        )
    return {
        **result,
        "caption_thinking_seal_sha256": seal["seal_sha256"],
        "declared_writer_schema_projections": len(
            _writer_projection_receipt_paths(output_root)
        ),
        "writer_projection_complete_sha256": (
            projection_complete["complete_sha256"]
            if projection_complete is not None
            else None
        ),
    }


def run_build(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = 1,
    requested_unit_ids: Sequence[str] = (),
    build_backbone: str = "both",
) -> dict[str, Any]:
    _require_caption_seal(caption_root)
    _write_contract(output_root)
    install_g25_thinking_contracts()
    with _g35_level(G35_WORK_THINKING_LEVEL), _g25_writer_projection_scope(
        source_project_root=source_project_root,
        output_root=output_root,
    ), _writer_projection_evidence_scope(
        source_project_root=source_project_root,
        output_root=output_root,
    ):
        if _writer_projection_receipt_paths(output_root):
            _seal_writer_projection_complete(
                source_project_root=source_project_root,
                output_root=output_root,
            )
        if workers != 1 or requested_unit_ids or build_backbone != "both":
            return base.run_build_task_parallel(
                source_project_root=source_project_root,
                caption_root=caption_root,
                output_root=output_root,
                workers=workers,
                requested_unit_ids=requested_unit_ids,
                build_backbone=build_backbone,
            )
        return base.run_build(
            source_project_root=source_project_root,
            caption_root=caption_root,
            output_root=output_root,
        )


def run_predict(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    _write_contract(output_root)
    # Build validation must reconstruct the G25=512/G35=low evidence.  The
    # Reader adapter is intentionally not patched and therefore remains zero.
    install_g25_thinking_contracts()
    with _g35_level(G35_WORK_THINKING_LEVEL), _fresh_qwen_routing(), _writer_projection_evidence_scope(
        source_project_root=source_project_root,
        output_root=output_root,
    ):
        return base.run_predict(
            source_project_root=source_project_root,
            output_root=output_root,
            workers=workers,
        )


def run_score(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    score_data_root: Path = DEFAULT_SCORE_DATA_ROOT,
    skip_calibration: bool = True,
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    _write_contract(output_root)
    with _g35_level(G35_JUDGE_THINKING_LEVEL):
        return base.run_score(
            source_project_root=source_project_root,
            output_root=output_root,
            score_data_root=score_data_root,
            skip_calibration=skip_calibration,
            workers=workers,
        )


def status(
    *,
    source_project_root: Path = DEFAULT_SOURCE_PROJECT_ROOT,
    caption_root: Path = DEFAULT_CAPTION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    value = base.status(
        source_project_root=source_project_root,
        caption_root=caption_root,
        output_root=output_root,
    )
    seal_path = Path(caption_root).resolve() / "build" / "caption_thinking_seal.json"
    return {
        **value,
        "schema_version": SCHEMA_VERSION,
        "thinking_round": thinking_contract(),
        "caption_thinking_seal_present": seal_path.is_file(),
        "provider_calls": 0,
    }


def report(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    value = base.score_report(output_root=output_root)
    return {
        **value,
        "schema_version": SCHEMA_VERSION,
        "thinking_round": thinking_contract(),
        "provider_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "status",
            "report",
            "caption-preflight",
            "caption-fallback",
            "caption",
            "caption-resume",
            "caption-seal",
            "writer",
            "build",
            "predict",
            "score",
            "all",
        ),
        default="status",
    )
    parser.add_argument("--source-project-root", type=Path, default=DEFAULT_SOURCE_PROJECT_ROOT)
    parser.add_argument("--caption-root", type=Path, default=DEFAULT_CAPTION_ROOT)
    parser.add_argument(
        "--fallback-source-caption-root",
        type=Path,
        default=DEFAULT_FALLBACK_SOURCE_CAPTION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--score-data-root", type=Path, default=DEFAULT_SCORE_DATA_ROOT)
    parser.add_argument("--image-root", type=Path, default=captions.DEFAULT_IMAGE_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_EVAL_WORKERS)
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument(
        "--writer-backbone", choices=("g25", "g35", "both"), default="both"
    )
    parser.add_argument("--writer-tasks", default="")
    parser.add_argument("--writer-task-workers", type=int, default=1)
    parser.add_argument("--build-task-workers", type=int, default=1)
    parser.add_argument(
        "--build-backbone", choices=("g25", "g35", "both"), default="both"
    )
    parser.add_argument("--build-units", default="")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="score the ten cells and defer the historical calibration",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    writer_tasks = [value.strip() for value in args.writer_tasks.split(",") if value.strip()]
    build_units = [value.strip() for value in args.build_units.split(",") if value.strip()]
    if args.workers < 1 or args.writer_task_workers < 1 or args.build_task_workers < 1:
        raise SystemExit("all worker counts must be positive")
    if args.writer_backbone != "both" and args.phase != "writer":
        raise SystemExit("single --writer-backbone requires --phase writer")
    if writer_tasks and (args.phase != "writer" or args.writer_backbone != "g25"):
        raise SystemExit("--writer-tasks requires --phase writer --writer-backbone g25")
    if args.writer_task_workers != 1 and not writer_tasks:
        raise SystemExit("--writer-task-workers requires --writer-tasks")
    if (
        args.build_task_workers != 1 or build_units or args.build_backbone != "both"
    ) and args.phase != "build":
        raise SystemExit(
            "--build-task-workers/--build-units/--build-backbone require --phase build"
        )

    zero_call_phases = {
        "status",
        "report",
        "caption-preflight",
        "caption-fallback",
        "caption-seal",
    }
    if args.phase in zero_call_phases and args.allow_model_calls:
        raise SystemExit(f"{args.phase} rejects --allow-model-calls")
    if args.phase not in zero_call_phases and not args.allow_model_calls:
        raise SystemExit(f"{args.phase} requires --allow-model-calls")

    if args.phase == "status":
        result = status(
            source_project_root=args.source_project_root,
            caption_root=args.caption_root,
            output_root=args.output_root,
        )
    elif args.phase == "report":
        result = report(output_root=args.output_root)
    elif args.phase == "caption-preflight":
        result = caption_preflight(
            source_project_root=args.source_project_root,
            caption_root=args.caption_root,
        )
    elif args.phase == "caption-fallback":
        result = run_caption_fallback(
            source_project_root=args.source_project_root,
            caption_root=args.caption_root,
            fallback_source_caption_root=args.fallback_source_caption_root,
            image_root=args.image_root,
        )
    elif args.phase in {"caption", "caption-resume"}:
        result = run_caption(
            source_project_root=args.source_project_root,
            caption_root=args.caption_root,
            fallback_source_caption_root=args.fallback_source_caption_root,
            image_root=args.image_root,
        )
    elif args.phase == "caption-seal":
        result = seal_caption(caption_root=args.caption_root)
    elif args.phase == "writer":
        result = run_writer(
            source_project_root=args.source_project_root,
            caption_root=args.caption_root,
            output_root=args.output_root,
            writer_backbone=args.writer_backbone,
            writer_tasks=writer_tasks,
            writer_task_workers=args.writer_task_workers,
        )
    elif args.phase == "build":
        result = run_build(
            source_project_root=args.source_project_root,
            caption_root=args.caption_root,
            output_root=args.output_root,
            workers=args.build_task_workers,
            requested_unit_ids=build_units,
            build_backbone=args.build_backbone,
        )
    elif args.phase == "predict":
        result = run_predict(
            source_project_root=args.source_project_root,
            output_root=args.output_root,
            workers=args.workers,
        )
    elif args.phase == "score":
        result = run_score(
            source_project_root=args.source_project_root,
            output_root=args.output_root,
            score_data_root=args.score_data_root,
            skip_calibration=args.skip_calibration,
            workers=args.workers,
        )
    else:
        phase_results: list[dict[str, Any]] = []
        phase_results.append(
            run_caption(
                source_project_root=args.source_project_root,
                caption_root=args.caption_root,
                fallback_source_caption_root=args.fallback_source_caption_root,
                image_root=args.image_root,
            )
        )
        phase_results.append(
            run_writer(
                source_project_root=args.source_project_root,
                caption_root=args.caption_root,
                output_root=args.output_root,
            )
        )
        phase_results.append(
            run_build(
                source_project_root=args.source_project_root,
                caption_root=args.caption_root,
                output_root=args.output_root,
            )
        )
        phase_results.append(
            run_predict(
                source_project_root=args.source_project_root,
                output_root=args.output_root,
                workers=args.workers,
            )
        )
        phase_results.append(
            run_score(
                source_project_root=args.source_project_root,
                output_root=args.output_root,
                score_data_root=args.score_data_root,
                skip_calibration=args.skip_calibration,
                workers=args.workers,
            )
        )
        result = {
            "status": "thinking_round_all_live_phases_complete",
            "phase_results": phase_results,
            "provider_calls": sum(
                int(row.get("provider_calls", 0)) for row in phase_results
            ),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

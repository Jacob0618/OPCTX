#!/usr/bin/env python3
"""Fail-closed orchestration for the one-pass MemEye v19 architecture matrix.

``project`` and ``preflight`` are deliberately zero-provider phases.  They
project score-free memory/query artifacts, inventory the frozen V7 captions,
bind prompt/code authorities, and expose every unresolved live capability.

``build``, ``predict``, and ``score`` are provider-delayed dispatch phases.
They cannot import a live executor until an independently authored gate binds
the exact protocol, project, preflight, runner, executor release, and request
ceiling.  Score is additionally air-gapped until all ten 371-row prediction
artifacts have been sealed.  No historical provider cache can be read through.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL_PATH = (
    SCRIPT_DIR
    / "research_memeye_full_arch_g35_v19"
    / "matrix_manifest_v19.json"
)
DEFAULT_ROOT = Path(r"D:\creamem3_cache\memeye_v19\full_arch_matrix_r2")
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\external_memeye_data"
)
DEFAULT_V7_CAPTION_ROOT = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\multimodal\cache"
    r"\memeye_captions_memeye_v7_prev2_context_b10"
)

PROTOCOL_SCHEMA = "creamem3.memeye_full_arch_matrix_protocol.v19.0"
PROJECT_SCHEMA = "creamem3.memeye_full_arch_project.v19.0"
QUERY_SCHEMA = "creamem3.memeye_full_arch_queries.v19.0"
MEMORY_SCHEMA = "creamem3.memeye_full_arch_memory.v19.0"
PREFLIGHT_SCHEMA = "creamem3.memeye_full_arch_preflight.v19.0"
RELEASE_SCHEMA = "creamem3.memeye_full_arch_executor_release.v19.0"
GATE_SCHEMA = "creamem3.memeye_full_arch_independent_gate.v19.0"
PHASE_RECEIPT_SCHEMA = "creamem3.memeye_full_arch_phase_receipt.v19.0"
PREDICTION_COMPLETE_SCHEMA = (
    "creamem3.memeye_full_arch_prediction_complete.v19.0"
)
PROVIDER_SMOKE_SCHEMA = "creamem3.memeye_provider_smoke.v19.0"
OFFICIAL_SMOKE_BRIDGE_SCHEMA = (
    "creamem3.memeye.official_provider_capability_bridge.v19.1"
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

ARCHITECTURE_IDS = (
    "mgtm_matched",
    "session_only",
    "online_base",
    "offline_base",
    "online_operation_context_v3",
)
BACKBONE_IDS = ("g25", "g35")
LIVE_PHASES = frozenset({"build", "predict", "score"})
PHASE_ORDER = ("project", "preflight", "build", "predict", "score")

TASK_QUESTION_COUNTS = {
    "Brand_Memory_Test": 29,
    "Card_Playlog_Test": 48,
    "Cartoon_Entertainment_Companion": 76,
    "Home_Renovation_Interior_Design": 52,
    "Multi-Scene_Visual_Case_Archive_Assistant": 50,
    "Outdoor_Navigation_Route_Memory_Assistant": 28,
    "Personal_Health_Dashboard_Assistant": 51,
    "Social_Chat_Memory_Test": 37,
}
TASK_SESSION_COUNTS = {
    "Brand_Memory_Test": 42,
    "Card_Playlog_Test": 4,
    "Cartoon_Entertainment_Companion": 86,
    "Home_Renovation_Interior_Design": 13,
    "Multi-Scene_Visual_Case_Archive_Assistant": 15,
    "Outdoor_Navigation_Route_Memory_Assistant": 10,
    "Personal_Health_Dashboard_Assistant": 12,
    "Social_Chat_Memory_Test": 39,
}
STRUCTURED_FUSED_TASKS = frozenset(
    {
        "Social_Chat_Memory_Test",
        "Outdoor_Navigation_Route_Memory_Assistant",
    }
)
TASK_TOP_K = {
    "Brand_Memory_Test": 30,
    "Social_Chat_Memory_Test": 30,
    "Multi-Scene_Visual_Case_Archive_Assistant": 35,
    "Outdoor_Navigation_Route_Memory_Assistant": 35,
    "Personal_Health_Dashboard_Assistant": 40,
    "Card_Playlog_Test": 100,
    "Cartoon_Entertainment_Companion": 100,
    "Home_Renovation_Interior_Design": 100,
}

EXPECTED_TASKS = 8
EXPECTED_QUESTIONS = 371
EXPECTED_IMAGES = 438
EXPECTED_SESSIONS = 221
EXPECTED_TWO_STAGE_ROWS = 306
EXPECTED_STRUCTURED_ROWS = 65
EXPECTED_MATRIX_CELLS = 10

FORBIDDEN_QUERY_KEYS = frozenset(
    {
        "answer",
        "answers",
        "category",
        "clue",
        "correct",
        "evidence",
        "gold",
        "ground_truth",
        "judge",
        "llm_judge",
        "point",
        "prediction",
        "score",
        "session_id",
        "verdict",
        "x",
        "y",
    }
)
FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "category",
        "clue",
        "correct",
        "evidence",
        "gold",
        "ground_truth",
        "judge",
        "llm_judge",
        "point",
        "reference_answer",
        "score",
        "session_id",
        "verdict",
        "x",
        "y",
    }
)
PROVIDER_MODULE_NAMES = frozenset(
    {
        "gemini_client",
        "eval.gemini_client",
        "google.genai",
        "google.generativeai",
        "openai",
    }
)


class MatrixProtocolError(ValueError):
    """A frozen matrix or score-air-gap invariant failed."""


class MissingCapabilityError(RuntimeError):
    """A live phase was requested before its executor release was sealed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MatrixProtocolError(f"missing {label}: {path}")
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise MatrixProtocolError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_mapping(path, path.name)
        if existing != dict(value):
            raise MatrixProtocolError(
                f"write-once artifact conflicts with existing bytes: {path}"
            )
        return
    _write_json_atomic(path, value)


def _loaded_provider_modules() -> list[str]:
    loaded: list[str] = []
    for name in sys.modules:
        if name in PROVIDER_MODULE_NAMES or any(
            name.startswith(prefix + ".")
            for prefix in PROVIDER_MODULE_NAMES
        ):
            loaded.append(name)
    return sorted(set(loaded))


def _require_no_provider_import(label: str) -> None:
    loaded = _loaded_provider_modules()
    if loaded:
        raise MatrixProtocolError(
            f"provider imported during zero-call {label}: {', '.join(loaded)}"
        )


def _resolve_authority_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _literal_node(source: str, name: str) -> tuple[Any, ast.AST]:
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value), node
    raise MatrixProtocolError(f"missing literal authority {name}")


def _function_source(source: str, name: str) -> tuple[str, ast.AST]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            if not segment:
                raise MatrixProtocolError(f"could not recover source for {name}")
            return segment, node
    raise MatrixProtocolError(f"missing function authority {name}")


def _literal_receipt(path: Path, name: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    value, node = _literal_node(source, name)
    if not isinstance(value, str):
        raise MatrixProtocolError(f"{path}::{name} must be a string literal")
    return {
        "path": str(path),
        "name": name,
        "value_sha256": _sha256(value),
        "runtime_splitlines": len(value.splitlines()),
        "chars": len(value),
        "source_start_line": int(getattr(node, "lineno", 0)),
        "source_end_line": int(getattr(node, "end_lineno", 0)),
    }


def _function_receipt(path: Path, name: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    segment, node = _function_source(source, name)
    return {
        "path": str(path),
        "name": name,
        "source_sha256": _sha256(segment),
        "source_start_line": int(getattr(node, "lineno", 0)),
        "source_end_line": int(getattr(node, "end_lineno", 0)),
    }


def matrix_cell_ids() -> list[str]:
    return [
        f"{backbone}__{architecture}"
        for backbone in BACKBONE_IDS
        for architecture in ARCHITECTURE_IDS
    ]


def reader_route_for_task(task: str) -> str:
    if task not in TASK_QUESTION_COUNTS:
        raise MatrixProtocolError(f"unknown MemEye task: {task}")
    return "structured_fused" if task in STRUCTURED_FUSED_TASKS else "two_stage"


def top_k_for_task(task: str) -> int:
    try:
        return TASK_TOP_K[task]
    except KeyError as exc:
        raise MatrixProtocolError(f"unknown MemEye task: {task}") from exc


def build_v7_reconstruction_user_text(
    task: str,
    image: str,
    *,
    sequence_index: int,
    total_images: int,
    previous: Sequence[Mapping[str, str]],
) -> str:
    """Reconstruct the best available V7 one-image/previous-two envelope.

    This helper is intentionally not called an exact V7 prompt.  The historical
    source was not retained; only the cache metadata establishes the context
    cardinality and ordering.  The system literal is bound separately.
    """

    if len(previous) > 2:
        raise MatrixProtocolError("V7 reconstruction accepts at most two captions")
    if not 1 <= sequence_index <= total_images:
        raise MatrixProtocolError("invalid chronological image index")
    context_lines: list[str] = []
    if previous:
        context_lines.append(
            f"Previous {len(previous)} chronological image caption(s). "
            "Use them for continuity and change detection only; do not replace "
            "the current image evidence."
        )
        for item in previous:
            context_lines.append(
                f"- Previous image {item['image']}: {item['caption']}"
            )
    else:
        context_lines.append(
            "No previous image caption context is available for this image."
        )
    context_lines.append(
        "\nNo earlier same-category caption summaries are available."
    )
    return (
        f"Task: {task}\n"
        f"Image file: {image}\n"
        f"Chronological image index in this task: "
        f"{sequence_index}/{total_images}\n\n"
        + "\n".join(context_lines)
        + "\n\nDescribe the current image for long-term memory."
    )


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_mapping(path, "matrix protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise MatrixProtocolError("matrix protocol schema drift")
    if tuple(protocol.get("phase_order", [])) != PHASE_ORDER:
        raise MatrixProtocolError("phase order drift")
    architectures = tuple(
        str(item.get("id", ""))
        for item in protocol.get("architectures", [])
        if isinstance(item, Mapping)
    )
    backbones = tuple(
        str(item.get("id", ""))
        for item in protocol.get("backbones", [])
        if isinstance(item, Mapping)
    )
    if architectures != ARCHITECTURE_IDS:
        raise MatrixProtocolError("five-architecture matrix drift")
    if backbones != BACKBONE_IDS:
        raise MatrixProtocolError("two-backbone matrix drift")
    scope = protocol.get("scope", {})
    expected_scope = {
        "tasks": EXPECTED_TASKS,
        "open_questions": EXPECTED_QUESTIONS,
        "images": EXPECTED_IMAGES,
        "physical_sessions": EXPECTED_SESSIONS,
        "matrix_cells": EXPECTED_MATRIX_CELLS,
        "repetitions": 1,
    }
    if scope != expected_scope:
        raise MatrixProtocolError("matrix scope drift")
    if protocol.get("task_top_k") != TASK_TOP_K:
        raise MatrixProtocolError("task-specific retrieval K drift")
    reader = protocol.get("reader_protocol", {})
    if (
        reader.get("model") != "gemini-2.5-flash-lite"
        or reader.get("two_stage_rows") != EXPECTED_TWO_STAGE_ROWS
        or reader.get("structured_fused_rows") != EXPECTED_STRUCTURED_ROWS
        or reader.get("evidence_max_output_tokens") != 2600
        or reader.get("final_max_output_tokens") != 500
        or reader.get("structured_fused_max_output_tokens") != 1200
        or set(reader.get("structured_fused_tasks", []))
        != STRUCTURED_FUSED_TASKS
    ):
        raise MatrixProtocolError("historical 306/65 Reader route drift")
    ingest = protocol.get("ingest_protocol", {})
    if (
        ingest.get("authority") != "creamem3.memeye_matched_ingest_v19"
        or ingest.get("summary_max_output_tokens") != 1200
        or ingest.get("fact_max_output_tokens") != 16000
        or ingest.get("max_attempts_per_stage") != 3
        or ingest.get("frozen_current_date") != "2026-07-05"
    ):
        raise MatrixProtocolError("historical 0.4533 writer token-cap drift")
    operation = protocol.get("operation_context_protocol", {})
    expected_operation_mapping = {
        "atomic_lookup": "original_full",
        "count": "original_full",
        "explain_cause": "original_full",
        "summarize_state": "bounded_grounded_views_2",
        "enumerate": "route_only",
        "compare": "route_only",
        "temporal_point": "route_only",
        "temporal_duration": "route_only",
        "temporal_order": "route_only",
        "trace_change": "route_only",
        "hypothetical": "route_only",
    }
    if (
        operation.get("policy_name")
        != "online_semantic_operation_context_v3"
        or operation.get("runtime_mapping") != expected_operation_mapping
        or operation.get("runtime_mapping_sha256") != _sha256(
            expected_operation_mapping
        )
        or operation.get("locomo_development_result", {}).get("correct")
        != 1259
        or operation.get("locomo_development_result", {}).get("rows") != 1540
    ):
        raise MatrixProtocolError("LoCoMo 81.7532 operation policy drift")
    judge = protocol.get("judge_protocol", {})
    if (
        judge.get("model") != "gemini-3.5-flash-lite"
        or judge.get("gpt52_calls_allowed") is not False
        or judge.get("legacy_judge_cache_readthrough") is not False
    ):
        raise MatrixProtocolError("Gemini 3.5 judge-only contract drift")
    calibration = protocol.get(
        "historical_04533_g35_rejudge_calibration", {}
    )
    if (
        calibration.get("matrix_cell") is not False
        or calibration.get("score_phase_only") is not True
        or calibration.get(
            "opens_after_all_matrix_predictions_complete_and_sealed"
        )
        is not True
        or calibration.get("union_unique_exact_triples") != 566
        or calibration.get("judge_first_attempt_ceiling") != 566
        or calibration.get("judge_hard_ceiling_at_3") != 1698
        or calibration.get("model_visible_fields")
        != ["question", "gold", "prediction"]
        or calibration.get("old_judge_fields_model_visible") is not False
        or calibration.get("legacy_judge_scores_reused") is not False
    ):
        raise MatrixProtocolError("historical 0.4533 calibration drift")
    caption = protocol.get("caption_protocol", {})
    if (
        caption.get("historical_v7_source_recoverable") is not False
        or caption.get("fidelity_label")
        != "best_available_v7_reconstruction_not_byte_exact"
        or caption.get("raw_images_per_call") != 1
        or caption.get("previous_caption_texts") != 2
        or caption.get("summary_contexts") != 0
    ):
        raise MatrixProtocolError("caption reconstruction disclosure drift")
    cache = protocol.get("cache_policy", {})
    if (
        cache.get("root") != str(DEFAULT_ROOT)
        or cache.get("legacy_provider_cache_readthrough") is not False
        or cache.get("provider_import_after_gate_only") is not True
    ):
        raise MatrixProtocolError("isolated D-drive cache contract drift")
    embedding = protocol.get("embedding_protocol", {})
    if (
        embedding.get("model") != "gemini-embedding-001"
        or embedding.get("dimension") != 3072
        or embedding.get("batch_size") != 100
        or embedding.get("provider_visible_character_limit") != 8000
        or embedding.get("cache_identity_scope")
        != "complete_untruncated_text"
        or embedding.get("failure_policy")
        != "fail_closed_no_per_item_or_zero_fallback"
        or embedding.get("max_provider_attempts_per_physical_batch") != 3
        or embedding.get("durable_attempt_accounting")
        != "write_once_intent_and_result_per_physical_batch_ordinal"
        or embedding.get("concurrency_control")
        != "exclusive_os_lifecycle_lock_fail_closed_on_stale_lock"
        or embedding.get("indeterminate_transport_policy")
        != "orphan_intent_fail_closed_zero_automatic_recall"
        or embedding.get("document_task_type") != "omitted"
        or embedding.get("query_task_type") != "omitted"
        or embedding.get("output_dimensionality")
        != "omitted_native_3072"
        or embedding.get("uniform_across_all_matrix_cells") is not True
        or embedding.get("authority_sha256")
        != "b74e214b77b4017c234a9a38b4c484535d88f3781275a7cbd384e4577dae7da2"
        or embedding.get("adaptation_scope")
        != "all_architectures_in_historical_mgtm_retrieval_environment"
        or embedding.get("legacy_cache_readthrough") is not False
    ):
        raise MatrixProtocolError("embedding contract drift")
    availability = protocol.get("provider_availability_protocol", {})
    if (
        availability.get("live_calls_currently_authorized") is not False
        or availability.get("g25_prebuild_smoke", {}).get(
            "max_provider_attempts"
        )
        != 1
        or availability.get("g25_prebuild_smoke", {}).get(
            "cache_or_output_may_flow_into_build"
        )
        is not False
        or availability.get("g35_prebuild_smoke", {}).get(
            "alternate_transport_allowed_without_new_protocol_and_gate"
        )
        is not False
        or availability.get("embedding_prebuild_smoke", {}).get(
            "expected_dimension"
        )
        != 3072
        or availability.get("dynamic_embedding_budget_gate", {}).get(
            "required_before_first_embedding"
        )
        is not True
        or availability.get("dynamic_embedding_budget_gate", {}).get(
            "writer_packet_census_required"
        )
        is not True
        or availability.get("dynamic_embedding_budget_gate", {}).get(
            "batch_size"
        )
        != 100
        or availability.get("dynamic_embedding_budget_gate", {}).get(
            "provider_visible_character_limit"
        )
        != 8000
        or availability.get("dynamic_embedding_budget_gate", {}).get(
            "status"
        )
        != "not_sealed"
    ):
        raise MatrixProtocolError("provider availability smoke contract drift")
    reranker_smoke = availability.get("reranker_prepredict_smoke", {})
    if (
        reranker_smoke.get("required_before_predict_not_build") is not True
        or reranker_smoke.get("authorized") is not False
        or reranker_smoke.get("namespace")
        != "provider_smokes/qwen3_rerank_r1"
        or reranker_smoke.get("provider") != "dashscope"
        or reranker_smoke.get("endpoint")
        != "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        or reranker_smoke.get("model") != "qwen3-rerank"
        or reranker_smoke.get("historical_provider_authority_sha256")
        != "fae368056a63543dc5a92d9aab2795d05b190db8e0deaaff9e23cc01da674c1a"
        or reranker_smoke.get("historical_instruction_sha256")
        != "cce4cd74ba087d68416210dec7b9849846e5f02e6735eafa1e8b08a3506f3a78"
        or reranker_smoke.get("synthetic_query_count") != 1
        or reranker_smoke.get("synthetic_document_count") != 3
        or reranker_smoke.get("return_documents") is not False
        or reranker_smoke.get("top_n") != 3
        or reranker_smoke.get("expected_response")
        != "complete_index_permutation_and_finite_scores"
        or reranker_smoke.get("provider_call_hard_ceiling") != 1
        or reranker_smoke.get("max_provider_attempts") != 1
        or reranker_smoke.get("independent_gate_required") is not True
        or reranker_smoke.get("key_source")
        != "environment_only:DASHSCOPE_API_KEY"
        or reranker_smoke.get("key_or_fingerprint_may_be_persisted") is not False
        or reranker_smoke.get("formal_ranking_cache_namespace")
        != "provider/qwen3_rerank"
        or reranker_smoke.get("cache_or_output_may_flow_into_predict") is not False
        or reranker_smoke.get("os_lifecycle_lock_required") is not True
        or reranker_smoke.get("indeterminate_transport_policy")
        != "orphan_intent_and_stale_lock_zero_automatic_recall"
        or reranker_smoke.get("status") != "not_run"
    ):
        raise MatrixProtocolError("Qwen reranker smoke contract drift")
    budget = protocol.get("logical_call_ceiling", {})
    if (
        budget.get("build_generation_provider_attempt_hard_total") != 6480
        or budget.get("embedding_provider_attempt_hard_full_build")
        != "pending_post_writer_100_item_batch_census"
        or budget.get("build_overall_provider_attempt_hard_total")
        != "pending_post_writer_100_item_batch_census"
    ):
        raise MatrixProtocolError("dynamic build budget contract drift")
    return protocol


def _task_files(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (data_root / "data" / "dialog").glob("*_Open.json")
        if not path.name.startswith("concat_")
    )


def _images_in_sessions(sessions: Sequence[Mapping[str, Any]]) -> list[str]:
    images: list[str] = []
    for session in sessions:
        dialogues = session.get("dialogues", [])
        if not isinstance(dialogues, list):
            raise MatrixProtocolError("session dialogues must be a list")
        for row in dialogues:
            if not isinstance(row, Mapping):
                raise MatrixProtocolError("dialogue row must be an object")
            values = row.get("input_image") or []
            values = values if isinstance(values, list) else [values]
            images.extend(str(value) for value in values if str(value).strip())
    return images


def _assert_query_rows_score_free(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    allowed = {
        "row_id",
        "task",
        "question_id",
        "question",
        "reader_route",
        "top_k",
        "source_ordinal",
    }
    for row in rows:
        keys = {str(key) for key in row}
        forbidden = {key.casefold() for key in keys} & FORBIDDEN_QUERY_KEYS
        if forbidden:
            raise MatrixProtocolError(
                "score-bearing keys entered query projection: "
                + ", ".join(sorted(forbidden))
            )
        unexpected = keys - allowed
        if unexpected:
            raise MatrixProtocolError(
                "unexpected query projection keys: "
                + ", ".join(sorted(unexpected))
            )
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in seen:
            raise MatrixProtocolError("query row IDs must be non-empty and unique")
        seen.add(row_id)
        task = str(row.get("task", ""))
        if row.get("reader_route") != reader_route_for_task(task):
            raise MatrixProtocolError("query Reader route drift")
        if row.get("top_k") != top_k_for_task(task):
            raise MatrixProtocolError("query retrieval K drift")
        if not str(row.get("question", "")).strip():
            raise MatrixProtocolError("query text is empty")


def project_dataset(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Write deterministic memory-only and score-free query artifacts."""

    _require_no_provider_import("project/start")
    protocol = load_protocol()
    files = _task_files(data_root)
    if len(files) != EXPECTED_TASKS:
        raise MatrixProtocolError(
            f"expected {EXPECTED_TASKS} Open tasks, found {len(files)}"
        )
    project_root = output_root / "project"
    memory_root = project_root / "memory"
    query_rows: list[dict[str, Any]] = []
    task_receipts: list[dict[str, Any]] = []
    all_images: list[str] = []
    total_sessions = 0
    for task_file in files:
        task = task_file.stem.removesuffix("_Open")
        if task not in TASK_QUESTION_COUNTS:
            raise MatrixProtocolError(f"unexpected Open task: {task}")
        value = _read_mapping(task_file, f"task {task}")
        raw_sessions = value.get("multi_session_dialogues")
        raw_qas = value.get("human-annotated QAs")
        if not isinstance(raw_sessions, list) or not all(
            isinstance(item, Mapping) for item in raw_sessions
        ):
            raise MatrixProtocolError(f"{task} has invalid physical sessions")
        if not isinstance(raw_qas, list) or not all(
            isinstance(item, Mapping) for item in raw_qas
        ):
            raise MatrixProtocolError(f"{task} has invalid Open QA list")
        if len(raw_sessions) != TASK_SESSION_COUNTS[task]:
            raise MatrixProtocolError(f"{task} physical-session census drift")
        if len(raw_qas) != TASK_QUESTION_COUNTS[task]:
            raise MatrixProtocolError(f"{task} question census drift")

        sessions = [copy.deepcopy(dict(item)) for item in raw_sessions]
        images = _images_in_sessions(sessions)
        all_images.extend(images)
        total_sessions += len(sessions)
        memory_payload = {
            "schema_version": MEMORY_SCHEMA,
            "task": task,
            "source_file_name": task_file.name,
            "source_file_sha256": _file_sha256(task_file),
            "dataset_image_hints_retained": True,
            "multi_session_dialogues": sessions,
        }
        memory_path = memory_root / f"{task}.json"
        _write_once_json(memory_path, memory_payload)
        task_start = len(query_rows)
        for ordinal, qa in enumerate(raw_qas, start=1):
            question_id = str(qa.get("question_id", f"Q{ordinal}"))
            query_rows.append(
                {
                    "row_id": f"{task}::{question_id}",
                    "task": task,
                    "question_id": question_id,
                    "question": str(qa.get("question", "")).strip(),
                    "reader_route": reader_route_for_task(task),
                    "top_k": top_k_for_task(task),
                    "source_ordinal": ordinal,
                }
            )
        task_receipts.append(
            {
                "task": task,
                "source_file_name": task_file.name,
                "source_file_sha256": _file_sha256(task_file),
                "memory_path": str(memory_path.resolve()),
                "memory_sha256": _file_sha256(memory_path),
                "physical_sessions": len(sessions),
                "image_references": len(images),
                "questions": len(query_rows) - task_start,
                "reader_route": reader_route_for_task(task),
                "top_k": top_k_for_task(task),
            }
        )

    _assert_query_rows_score_free(query_rows)
    if len(query_rows) != EXPECTED_QUESTIONS:
        raise MatrixProtocolError("full Open query census drift")
    if total_sessions != EXPECTED_SESSIONS:
        raise MatrixProtocolError("full physical-session census drift")
    if len(all_images) != EXPECTED_IMAGES or len(set(all_images)) != EXPECTED_IMAGES:
        raise MatrixProtocolError("full unique-image census drift")
    route_counts = {
        route: sum(row["reader_route"] == route for row in query_rows)
        for route in ("two_stage", "structured_fused")
    }
    if route_counts != {
        "two_stage": EXPECTED_TWO_STAGE_ROWS,
        "structured_fused": EXPECTED_STRUCTURED_ROWS,
    }:
        raise MatrixProtocolError("306/65 Reader route census drift")

    query_payload = {
        "schema_version": QUERY_SCHEMA,
        "score_free": True,
        "forbidden_fields_removed": sorted(FORBIDDEN_QUERY_KEYS),
        "rows": query_rows,
        "rows_sha256": _sha256(query_rows),
    }
    query_path = project_root / "queries_score_free.json"
    _write_once_json(query_path, query_payload)
    protocol_sha256 = _sha256(protocol)
    manifest = {
        "schema_version": PROJECT_SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": protocol_sha256,
        "score_free": True,
        "raw_data_paths_exposed_to_build_or_predict": False,
        "query_path": str(query_path.resolve()),
        "query_file_sha256": _file_sha256(query_path),
        "query_rows_sha256": query_payload["rows_sha256"],
        "memory_root": str(memory_root.resolve()),
        "counts": {
            "tasks": len(task_receipts),
            "questions": len(query_rows),
            "physical_sessions": total_sessions,
            "image_references": len(all_images),
            "unique_images": len(set(all_images)),
            "two_stage_rows": route_counts["two_stage"],
            "structured_fused_rows": route_counts["structured_fused"],
        },
        "task_receipts": task_receipts,
        "matrix_cell_ids": matrix_cell_ids(),
        "model_calls": 0,
        "provider_calls": 0,
    }
    manifest["project_contract_sha256"] = _sha256(manifest)
    _write_once_json(project_root / "manifest.json", manifest)
    _require_no_provider_import("project/end")
    return manifest


def validate_project(
    *, output_root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    manifest_path = output_root / "project" / "manifest.json"
    manifest = _read_mapping(manifest_path, "project manifest")
    if manifest.get("schema_version") != PROJECT_SCHEMA:
        raise MatrixProtocolError("project manifest schema drift")
    contract = dict(manifest)
    observed = contract.pop("project_contract_sha256", None)
    if observed != _sha256(contract):
        raise MatrixProtocolError("project contract digest mismatch")
    if manifest.get("matrix_cell_ids") != matrix_cell_ids():
        raise MatrixProtocolError("project matrix cells drift")
    if manifest.get("score_free") is not True:
        raise MatrixProtocolError("project manifest is not score-free")
    query_path = Path(str(manifest.get("query_path", "")))
    if _file_sha256(query_path) != manifest.get("query_file_sha256"):
        raise MatrixProtocolError("project query file digest mismatch")
    query_payload = _read_mapping(query_path, "score-free query manifest")
    if query_payload.get("schema_version") != QUERY_SCHEMA:
        raise MatrixProtocolError("query schema drift")
    rows = query_payload.get("rows")
    if not isinstance(rows, list) or not all(
        isinstance(row, Mapping) for row in rows
    ):
        raise MatrixProtocolError("query rows must be objects")
    _assert_query_rows_score_free(rows)
    if len(rows) != EXPECTED_QUESTIONS or _sha256(rows) != query_payload.get(
        "rows_sha256"
    ):
        raise MatrixProtocolError("score-free query rows drift")
    receipts = manifest.get("task_receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_TASKS:
        raise MatrixProtocolError("project task receipts drift")
    for receipt in receipts:
        memory_path = Path(str(receipt.get("memory_path", "")))
        if _file_sha256(memory_path) != receipt.get("memory_sha256"):
            raise MatrixProtocolError("project memory artifact digest mismatch")
        memory = _read_mapping(memory_path, "projected memory")
        if memory.get("schema_version") != MEMORY_SCHEMA:
            raise MatrixProtocolError("projected memory schema drift")
        if "human-annotated QAs" in memory:
            raise MatrixProtocolError("gold QA entered projected memory")
    return manifest


def caption_inventory(
    caption_root: Path = DEFAULT_V7_CAPTION_ROOT,
) -> dict[str, Any]:
    files = sorted(caption_root.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    task_counts: dict[str, int] = {}
    for path in files:
        value = _read_mapping(path, "V7 caption")
        image = str(value.get("image", ""))
        task = str(value.get("task", ""))
        previous = value.get("previous_images", [])
        if (
            not image
            or image in seen_images
            or task not in TASK_QUESTION_COUNTS
            or not str(value.get("caption", "")).strip()
            or value.get("error") not in {"", None}
            or value.get("model") != "gemini-2.5-flash-lite"
            or value.get("prompt_version") != "memeye_v7_prev2_context_b10"
            or not isinstance(previous, list)
            or len(previous) > 2
        ):
            raise MatrixProtocolError(f"invalid frozen V7 caption: {path}")
        seen_images.add(image)
        task_counts[task] = task_counts.get(task, 0) + 1
        rows.append(
            {
                "task": task,
                "image": image,
                "caption_sha256": _sha256(str(value["caption"])),
                "sequence_index": int(value.get("sequence_index", 0)),
                "total_images": int(value.get("total_images", 0)),
                "previous_images": [str(item) for item in previous],
                "file_sha256": _file_sha256(path),
            }
        )
    if len(rows) != EXPECTED_IMAGES or len(seen_images) != EXPECTED_IMAGES:
        raise MatrixProtocolError("frozen V7 caption census drift")
    if set(task_counts) != set(TASK_QUESTION_COUNTS):
        raise MatrixProtocolError("frozen V7 caption task coverage drift")
    return {
        "status": "pass",
        "caption_root": str(caption_root.resolve()),
        "captions": len(rows),
        "unique_images": len(seen_images),
        "model": "gemini-2.5-flash-lite",
        "prompt_version": "memeye_v7_prev2_context_b10",
        "max_previous_caption_texts": max(
            (len(row["previous_images"]) for row in rows), default=0
        ),
        "task_counts": dict(sorted(task_counts.items())),
        "manifest_sha256": _sha256(rows),
    }


def authority_receipt(protocol: Mapping[str, Any]) -> dict[str, Any]:
    authorities = protocol.get("authority_files")
    if not isinstance(authorities, Mapping):
        raise MatrixProtocolError("authority_files missing from protocol")
    file_rows: dict[str, dict[str, Any]] = {}
    for name, raw in authorities.items():
        if not isinstance(raw, Mapping):
            raise MatrixProtocolError(f"invalid authority record: {name}")
        path = _resolve_authority_path(str(raw.get("path", "")))
        actual = _file_sha256(path)
        expected = str(raw.get("sha256", ""))
        file_rows[str(name)] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }

    focused_path = _resolve_authority_path(
        str(authorities["focused_reader"]["path"])
    )
    structured_path = _resolve_authority_path(
        str(authorities["structured_fused_reader"]["path"])
    )
    eval_path = _resolve_authority_path(
        str(authorities["historical_eval_memeye"]["path"])
    )
    focused_prompt = _literal_receipt(focused_path, "FOCUSED_ANSWER_PROMPT")
    focused_renderer = _function_receipt(focused_path, "render_aggregation")
    structured_prompt = _literal_receipt(structured_path, "PROMPT")
    open_qa = _literal_receipt(eval_path, "MEMEYE_OPEN_QA_PROMPT")
    judge_prompt = _literal_receipt(eval_path, "MEMEYE_JUDGE_PROMPT")
    image_prompt = _literal_receipt(eval_path, "IMAGE_READ_PROMPT")
    judge_parser = _function_receipt(eval_path, "parse_judge_response")
    judge_mapping = _function_receipt(
        eval_path, "_coerce_official_judge_score"
    )
    historical_summary_call = _function_receipt(eval_path, "summarize_batch")
    historical_fact_call = _function_receipt(eval_path, "ingest_batch")
    historical_batcher = _function_receipt(eval_path, "make_batches")
    historical_batch_renderer = _function_receipt(eval_path, "render_batch")

    reader = protocol["reader_protocol"]
    judge = protocol["judge_protocol"]
    caption = protocol["caption_protocol"]
    literal_checks = [
        focused_prompt["value_sha256"] == reader["evidence_prompt_sha256"],
        focused_renderer["source_sha256"]
        == reader["evidence_renderer_sha256"],
        structured_prompt["value_sha256"]
        == reader["structured_fused_prompt_sha256"],
        open_qa["value_sha256"] == reader["final_prompt_sha256"],
        judge_prompt["value_sha256"] == judge["prompt_sha256"],
        image_prompt["value_sha256"] == caption["system_prompt_sha256"],
    ]

    # Both adapters are provider-free.  The legacy adapter remains the pure
    # prompt loader, while v19 is the sole call-level cap/retry/date authority.
    from creamem3 import memeye_adapter_v9 as adapter
    from creamem3 import memeye_matched_ingest_v19 as matched
    from creamem3 import online_operation_context_v8 as operation_context

    bundle = matched.load_prompt_bundle()
    matched_protocol = matched.protocol_manifest()
    mem0_source = adapter.SOURCE_MEM0_FILE.read_text(encoding="utf-8")
    _, mem0_node = _literal_node(mem0_source, "ADDITIVE_EXTRACTION_PROMPT")
    source_span = int(getattr(mem0_node, "end_lineno", 0)) - int(
        getattr(mem0_node, "lineno", 0)
    ) + 1
    ingest = protocol["ingest_protocol"]
    writer_checks = [
        _sha256(bundle["mem0_additive_prompt"])
        == ingest["additive_prompt_sha256"],
        bundle["prompt_bundle_sha256"]
        == ingest["historical_bundle_sha256"],
        source_span == ingest["official_mem0_source_span_lines"],
        len(str(bundle["mem0_additive_prompt"]).splitlines())
        == ingest["runtime_additive_prompt_splitlines"],
        matched_protocol["summary_max_output_tokens"] == 1200,
        matched_protocol["fact_max_output_tokens"] == 16000,
        matched_protocol["max_provider_attempts"] == 3,
        matched_protocol["frozen_current_date"] == "2026-07-05",
    ]
    operation_protocol = protocol["operation_context_protocol"]
    operation_checks = [
        operation_context.POLICY_NAME == operation_protocol["policy_name"],
        operation_context.OPERATION_TO_CONTEXT_POLICY
        == operation_protocol["runtime_mapping"],
        _sha256(operation_context.OPERATION_TO_CONTEXT_POLICY)
        == operation_protocol["runtime_mapping_sha256"],
    ]
    passed = (
        all(row["passed"] for row in file_rows.values())
        and all(literal_checks)
        and all(writer_checks)
        and all(operation_checks)
    )
    return {
        "status": "pass" if passed else "fail",
        "files": file_rows,
        "literals": {
            "focused_answer": focused_prompt,
            "structured_fused": structured_prompt,
            "open_qa": open_qa,
            "judge": judge_prompt,
            "image_read": image_prompt,
        },
        "functions": {
            "focused_candidate_first_renderer": focused_renderer,
            "judge_parser": judge_parser,
            "judge_mapping": judge_mapping,
            "historical_summary_call": historical_summary_call,
            "historical_fact_call": historical_fact_call,
            "historical_native_batcher": historical_batcher,
            "historical_native_batch_renderer": historical_batch_renderer,
        },
        "writer": {
            "authority_module": "creamem3.memeye_matched_ingest_v19",
            "historical_bundle_sha256": bundle["prompt_bundle_sha256"],
            "additive_prompt_sha256": _sha256(
                bundle["mem0_additive_prompt"]
            ),
            "official_source_span_lines": source_span,
            "runtime_splitlines": len(
                str(bundle["mem0_additive_prompt"]).splitlines()
            ),
            "schema_adaptation": "creamem_typed_output_and_provenance_only",
            "summary_max_output_tokens": matched_protocol[
                "summary_max_output_tokens"
            ],
            "fact_max_output_tokens": matched_protocol[
                "fact_max_output_tokens"
            ],
            "max_provider_attempts": matched_protocol[
                "max_provider_attempts"
            ],
            "frozen_current_date": matched_protocol[
                "frozen_current_date"
            ],
            "matched_protocol_sha256": matched_protocol["protocol_sha256"],
        },
        "operation_context": {
            "policy_name": operation_context.POLICY_NAME,
            "runtime_mapping": dict(
                operation_context.OPERATION_TO_CONTEXT_POLICY
            ),
            "runtime_mapping_sha256": _sha256(
                operation_context.OPERATION_TO_CONTEXT_POLICY
            ),
            "locomo_result": copy.deepcopy(
                operation_protocol["locomo_development_result"]
            ),
        },
    }


def _module_surface_receipt(
    module_name: str, required: Sequence[str]
) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return {
            "module": module_name,
            "present": False,
            "surface_complete": False,
            "missing_symbols": list(required),
        }
    path = Path(spec.origin).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    missing = [name for name in required if name not in names]
    return {
        "module": module_name,
        "present": True,
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "surface_complete": not missing,
        "missing_symbols": missing,
    }


def _release_path(
    protocol: Mapping[str, Any], phase: str, output_root: Path
) -> Path:
    relative = protocol["capability_contracts"][
        f"{phase}_executor_release"
    ]
    return output_root / str(relative)


def _release_status(path: Path, phase: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "phase": phase,
            "path": str(path.resolve()),
            "present": False,
            "ready": False,
        }
    try:
        value = _read_mapping(path, f"{phase} executor release")
        ready = (
            value.get("schema_version") == RELEASE_SCHEMA
            and value.get("phase") == phase
            and value.get("status") == "ready"
            and isinstance(value.get("logical_call_hard_ceiling"), int)
            and int(value["logical_call_hard_ceiling"]) >= 0
        )
        return {
            "phase": phase,
            "path": str(path.resolve()),
            "present": True,
            "ready": ready,
            "file_sha256": _file_sha256(path),
            "module": value.get("module"),
            "entrypoint": value.get("entrypoint"),
            "logical_call_hard_ceiling": value.get(
                "logical_call_hard_ceiling"
            ),
        }
    except Exception as exc:
        return {
            "phase": phase,
            "path": str(path.resolve()),
            "present": True,
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_preflight(
    *,
    output_root: Path = DEFAULT_ROOT,
    caption_root: Path = DEFAULT_V7_CAPTION_ROOT,
) -> dict[str, Any]:
    """Validate the matrix without importing or calling a provider."""

    _require_no_provider_import("preflight/start")
    protocol = load_protocol()
    project = validate_project(output_root=output_root)
    captions = caption_inventory(caption_root)
    authorities = authority_receipt(protocol)
    g35 = _module_surface_receipt(
        str(protocol["capability_contracts"]["g35_runtime_module"]),
        (
            "MODEL_ID",
            "DEFAULT_CACHE_ROOT",
            "StageLimits",
            "DEFAULT_STAGE_LIMITS",
            "Gemini35Runtime",
            "text_part",
            "inline_image_part",
            "build_generate_content_request",
            "request_identity",
            "write_once_json",
            "protocol_manifest",
            "zero_call_preflight",
        ),
    )
    releases = {
        phase: _release_status(
            _release_path(protocol, phase, output_root), phase
        )
        for phase in sorted(LIVE_PHASES)
    }
    provider_smokes = {
        phase: {
            name: {
                "path": str(path.resolve()),
                "present": path.is_file(),
                "file_sha256": _file_sha256(path) if path.is_file() else None,
            }
            for name, path in provider_smoke_paths(
                phase, protocol=protocol, output_root=output_root
            ).items()
        }
        for phase in ("build", "predict")
    }
    zero_checks = [
        captions["status"] == "pass",
        authorities["status"] == "pass",
        project.get("model_calls") == 0,
        project.get("provider_calls") == 0,
        not _loaded_provider_modules(),
    ]
    zero_status = "pass" if all(zero_checks) else "fail"
    live_ready = (
        zero_status == "pass"
        and g35.get("surface_complete") is True
        and all(row.get("ready") is True for row in releases.values())
    )
    unresolved = [
        key
        for key, value in protocol["logical_call_ceiling"].items()
        if isinstance(value, str) and value.startswith("pending_")
    ]
    contract = {
        "schema_version": PREFLIGHT_SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_file_sha256": _file_sha256(PROTOCOL_PATH),
        "protocol_sha256": _sha256(protocol),
        "runner_path": str(Path(__file__).resolve()),
        "runner_file_sha256": _file_sha256(Path(__file__).resolve()),
        "output_root": str(output_root.resolve()),
        "project_manifest_path": str(
            (output_root / "project" / "manifest.json").resolve()
        ),
        "project_manifest_sha256": _file_sha256(
            output_root / "project" / "manifest.json"
        ),
        "project_contract_sha256": project["project_contract_sha256"],
        "matrix_cell_ids": matrix_cell_ids(),
        "counts": project["counts"],
        "caption_inventory": captions,
        "authority_receipt": authorities,
        "g35_runtime_surface": g35,
        "executor_releases": releases,
        "provider_availability_smokes": provider_smokes,
        "zero_call_protocol_status": zero_status,
        "live_execution_status": "ready" if live_ready else "blocked",
        "unresolved_static_budget_labels": unresolved,
        "model_calls": 0,
        "provider_calls": 0,
        "provider_modules_imported": _loaded_provider_modules(),
    }
    contract["preflight_contract_sha256"] = _sha256(contract)
    path = output_root / "preflight" / "preflight.json"
    _write_json_atomic(path, contract)
    _require_no_provider_import("preflight/end")
    return contract


def validate_release(
    path: Path,
    *,
    phase: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    release = _read_mapping(path, f"{phase} executor release")
    required = {
        "schema_version",
        "phase",
        "status",
        "protocol_sha256",
        "module",
        "module_file_sha256",
        "entrypoint",
        "logical_call_hard_ceiling",
        "stage_call_hard_ceilings",
        "provider_cache_root",
        "legacy_cache_readthrough",
    }
    if set(release) != required:
        raise MissingCapabilityError(f"{phase} release keyset drift")
    if (
        release["schema_version"] != RELEASE_SCHEMA
        or release["phase"] != phase
        or release["status"] != "ready"
        or release["protocol_sha256"] != _sha256(protocol)
        or release["entrypoint"]
        != protocol["capability_contracts"]["entrypoint"]
        or release["legacy_cache_readthrough"] is not False
        or not isinstance(release["logical_call_hard_ceiling"], int)
        or release["logical_call_hard_ceiling"] < 0
        or not isinstance(release["stage_call_hard_ceilings"], Mapping)
    ):
        raise MissingCapabilityError(f"{phase} executor release is not valid")
    cache_root = Path(str(release["provider_cache_root"])).resolve()
    if cache_root != DEFAULT_ROOT.resolve() and DEFAULT_ROOT.resolve() not in cache_root.parents:
        raise MissingCapabilityError(f"{phase} provider cache escaped v19 root")
    spec = importlib.util.find_spec(str(release["module"]))
    if spec is None or not spec.origin:
        raise MissingCapabilityError(f"{phase} executor module is unavailable")
    module_path = Path(spec.origin).resolve()
    if _file_sha256(module_path) != release["module_file_sha256"]:
        raise MissingCapabilityError(f"{phase} executor module digest drift")
    return release


def validate_independent_gate(
    gate_path: Path,
    *,
    phase: str,
    protocol: Mapping[str, Any],
    project: Mapping[str, Any],
    preflight: Mapping[str, Any],
    release_path: Path,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _read_mapping(gate_path, f"{phase} independent gate")
    required = {
        "schema_version",
        "decision",
        "reviewer_role",
        "reviewer",
        "phase",
        "protocol_sha256",
        "project_contract_sha256",
        "preflight_contract_sha256",
        "runner_file_sha256",
        "executor_release_sha256",
        "approved_logical_call_hard_ceiling",
        "approved_stage_call_hard_ceilings",
        "legacy_cache_readthrough",
        "score_airgap_reviewed",
    }
    if set(gate) != required:
        raise MissingCapabilityError("independent gate keyset drift")
    expected = {
        "schema_version": GATE_SCHEMA,
        "decision": "approve",
        "reviewer_role": "independent_protocol_auditor",
        "phase": phase,
        "protocol_sha256": _sha256(protocol),
        "project_contract_sha256": project["project_contract_sha256"],
        "preflight_contract_sha256": preflight[
            "preflight_contract_sha256"
        ],
        "runner_file_sha256": _file_sha256(Path(__file__).resolve()),
        "executor_release_sha256": _file_sha256(release_path),
        "approved_logical_call_hard_ceiling": release[
            "logical_call_hard_ceiling"
        ],
        "approved_stage_call_hard_ceilings": release[
            "stage_call_hard_ceilings"
        ],
        "legacy_cache_readthrough": False,
        "score_airgap_reviewed": True,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise MissingCapabilityError(
                f"independent gate binding mismatch: {key}"
            )
    if not str(gate.get("reviewer", "")).strip():
        raise MissingCapabilityError("independent gate lacks reviewer identity")
    return gate


def validate_prediction_complete(output_root: Path) -> dict[str, Any]:
    path = output_root / "predict" / "complete.json"
    value = _read_mapping(path, "full-matrix prediction completion")
    if value.get("schema_version") != PREDICTION_COMPLETE_SCHEMA:
        raise MatrixProtocolError("prediction completion schema drift")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_MATRIX_CELLS:
        raise MatrixProtocolError("prediction completion cell census drift")
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise MatrixProtocolError("prediction completion cell is not an object")
        cell_id = str(cell.get("cell_id", ""))
        if cell_id in seen or cell_id not in matrix_cell_ids():
            raise MatrixProtocolError("prediction completion cell identity drift")
        seen.add(cell_id)
        if cell.get("rows") != EXPECTED_QUESTIONS or cell.get("complete") is not True:
            raise MatrixProtocolError("prediction cell is incomplete")
        prediction_path = Path(str(cell.get("prediction_path", "")))
        if _file_sha256(prediction_path) != cell.get("prediction_sha256"):
            raise MatrixProtocolError("prediction artifact digest mismatch")
        prediction = _read_mapping(prediction_path, "prediction artifact")
        serialized_keys = {
            str(key).casefold()
            for row in prediction.get("rows", [])
            if isinstance(row, Mapping)
            for key in row
        }
        if serialized_keys & FORBIDDEN_PREDICTION_KEYS:
            raise MatrixProtocolError(
                "score-bearing fields entered prediction artifact"
            )
    if seen != set(matrix_cell_ids()):
        raise MatrixProtocolError("prediction completion misses matrix cells")
    return value


def _phase_prerequisite(phase: str, output_root: Path) -> None:
    if phase == "predict":
        build = _read_mapping(
            output_root / "build" / "complete.json", "build completion"
        )
        if build.get("complete") is not True:
            raise MatrixProtocolError("predict requires a complete build")
    elif phase == "score":
        validate_prediction_complete(output_root)


def provider_smoke_paths(
    phase: str,
    *,
    protocol: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Path]:
    availability = protocol["provider_availability_protocol"]
    if phase == "build":
        mapping = {
            "g25_summary_writer": "provider_smokes/g25_summary_1200_r1",
            "g35_caption": "provider_smokes/g35_caption_writer_view_r1",
            "embedding_batch": "provider_smokes/gemini_embedding_001_r1",
        }
    elif phase == "predict":
        # Qwen has no build-time role and is independently gated here.
        mapping = {
            "reranker": availability["reranker_prepredict_smoke"]["namespace"],
        }
    else:
        mapping = {}
    return {
        name: Path(output_root) / str(namespace) / "complete.json"
        for name, namespace in mapping.items()
    }


def _validate_official_build_smokes(
    *, protocol: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Authority-first strict readback of the three audited official smokes."""

    root = Path(output_root).resolve()
    bridge_path = root / "releases" / "official_provider_capability_bridge.json"
    bridge = _read_mapping(bridge_path, "official provider capability bridge")
    copy_bridge = dict(bridge)
    observed = copy_bridge.pop("bridge_sha256", None)
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
        or observed != _sha256(copy_bridge)
        or bridge.get("schema_version") != OFFICIAL_SMOKE_BRIDGE_SCHEMA
        or bridge.get("status") != "ready"
        or bridge.get("authorized") is not True
        or bridge.get("protocol_sha256") != _sha256(protocol)
        or bridge.get("runner_file_sha256") != OFFICIAL_SMOKE_RUNNER_SHA256
        or bridge.get("independent_audit_status") != "pass"
        or bridge.get("formal_cache_reuse_allowed") is not False
    ):
        raise MissingCapabilityError("official provider capability bridge drift")
    runner_path = (
        SCRIPT_DIR / "run_official_provider_availability_smokes_v19_1.py"
    ).resolve()
    if (
        Path(str(bridge.get("runner_path", ""))).resolve() != runner_path
        or _file_sha256(runner_path) != OFFICIAL_SMOKE_RUNNER_SHA256
    ):
        raise MissingCapabilityError("official smoke runner drift before import")
    fixed = {
        "g25_summary_writer": (
            root / "provider_smokes" / "g25_summary_1200_r1" / "complete.json",
            SCRIPT_DIR
            / "research_memeye_full_arch_g35_v19"
            / "official_g25_summary_writer_smoke_gate_v19_1.json",
            1,
        ),
        "g35_caption": (
            root
            / "provider_smokes"
            / "g35_caption_writer_view_r1"
            / "complete.json",
            SCRIPT_DIR
            / "research_memeye_full_arch_g35_v19"
            / "official_g35_caption_writer_view_smoke_gate_v19_1.json",
            3,
        ),
        "embedding_batch": (
            root
            / "provider_smokes"
            / "gemini_embedding_001_r1"
            / "complete.json",
            SCRIPT_DIR
            / "research_memeye_full_arch_g35_v19"
            / "official_embedding_batch_shape_smoke_gate_v19_1.json",
            1,
        ),
    }
    rows = bridge.get("smokes")
    if not isinstance(rows, Mapping) or set(rows) != set(fixed):
        raise MissingCapabilityError("official smoke bridge census drift")
    for smoke_id, (_receipt, gate, _calls) in fixed.items():
        if _file_sha256(gate) != OFFICIAL_SMOKE_GATE_SHA256[smoke_id]:
            raise MissingCapabilityError(f"official {smoke_id} gate drift")
    runner = importlib.import_module(
        "creamem3.run_official_provider_availability_smokes_v19_1"
    )
    if (
        Path(runner.__file__).resolve() != runner_path
        or _file_sha256(Path(runner.__file__)) != OFFICIAL_SMOKE_RUNNER_SHA256
    ):
        raise MissingCapabilityError("official smoke runner import drift")
    row_keys = {
        "receipt_path",
        "receipt_file_sha256",
        "gate_path",
        "gate_file_sha256",
        "provider_calls",
        "formal_cache_reuse_allowed",
    }
    receipts: dict[str, Any] = {}
    for smoke_id, (receipt_path, gate_path, calls) in fixed.items():
        raw = rows[smoke_id]
        if not isinstance(raw, Mapping) or set(raw) != row_keys:
            raise MissingCapabilityError(f"official {smoke_id} bridge row drift")
        row = dict(raw)
        if (
            Path(str(row["receipt_path"])).resolve() != receipt_path.resolve()
            or row["receipt_file_sha256"]
            != OFFICIAL_SMOKE_RECEIPT_SHA256[smoke_id]
            or Path(str(row["gate_path"])).resolve() != gate_path.resolve()
            or row["gate_file_sha256"] != OFFICIAL_SMOKE_GATE_SHA256[smoke_id]
            or row["provider_calls"] != calls
            or row["formal_cache_reuse_allowed"] is not False
        ):
            raise MissingCapabilityError(f"official {smoke_id} bridge binding drift")
        try:
            validated = runner.validate_official_smoke_receipt(
                smoke_id,
                receipt_path,
                expected_file_sha256=OFFICIAL_SMOKE_RECEIPT_SHA256[smoke_id],
                gate_path=gate_path,
            )
        except Exception as exc:
            raise MissingCapabilityError(
                f"official {smoke_id} evidence rejected: {exc}"
            ) from exc
        if validated.get("formal_cache_reuse_allowed") is not False:
            raise MissingCapabilityError(f"official {smoke_id} reuse drift")
        receipts[smoke_id] = {
            "path": str(receipt_path.resolve()),
            "sha256": OFFICIAL_SMOKE_RECEIPT_SHA256[smoke_id],
            "gate_path": str(gate_path.resolve()),
            "gate_sha256": OFFICIAL_SMOKE_GATE_SHA256[smoke_id],
            "provider_calls": calls,
            "formal_cache_reuse_allowed": False,
        }
    receipts["bridge"] = {
        "path": str(bridge_path.resolve()),
        "sha256": _file_sha256(bridge_path),
        "bridge_sha256": observed,
    }
    return receipts


def validate_phase_provider_smokes(
    phase: str,
    *,
    protocol: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Fail before executor import unless independent availability seals pass."""

    if phase == "build":
        return _validate_official_build_smokes(
            protocol=protocol, output_root=output_root
        )

    expected = {
        "g25": {
            "model": "gemini-2.5-flash-lite",
            "provider_calls": 1,
            "checks": {
                "stage": "summary",
                "max_output_tokens": 1200,
                "max_provider_attempts": 1,
            },
        },
        "g35": {
            "model": "gemini-3.5-flash-lite",
            "provider_calls": 3,
            "checks": {"stages": ["caption", "writer", "view"]},
        },
        "embedding": {
            "model": "gemini-embedding-001",
            "provider_calls": 1,
            "checks": {
                "dimension": 3072,
                "task_type": None,
                "input_type": None,
                "output_dimensionality": None,
            },
        },
        "reranker": {
            "model": "qwen3-rerank",
            "provider_calls": 1,
            "checks": {"stage": "rerank", "nonempty_ranking": True},
        },
    }
    receipts: dict[str, Any] = {}
    for smoke_id, path in provider_smoke_paths(
        phase, protocol=protocol, output_root=output_root
    ).items():
        value = _read_mapping(path, f"{smoke_id} provider smoke seal")
        row = expected[smoke_id]
        if (
            value.get("schema_version") != PROVIDER_SMOKE_SCHEMA
            or value.get("status") != "pass"
            or value.get("smoke_id") != smoke_id
            or value.get("protocol_sha256") != _sha256(protocol)
            or value.get("model") != row["model"]
            or value.get("provider_calls") != row["provider_calls"]
            or value.get("checks") != row["checks"]
            or value.get("matrix_cache_or_output_reusable") is not False
            or not str(value.get("independent_gate_sha256", ""))
        ):
            raise MissingCapabilityError(
                f"invalid {smoke_id} provider availability seal"
            )
        smoke_cache = Path(str(value.get("cache_root", ""))).resolve()
        root = Path(output_root).resolve()
        if root not in smoke_cache.parents:
            raise MissingCapabilityError(
                f"{smoke_id} provider smoke escaped r2 root"
            )
        if smoke_id == "reranker":
            runner_authority = protocol["authority_files"][
                "qwen_reranker_smoke_runner"
            ]
            expected_qwen_path = _resolve_authority_path(
                str(runner_authority["path"])
            )
            if _file_sha256(expected_qwen_path) != runner_authority["sha256"]:
                raise MissingCapabilityError(
                    "Qwen reranker smoke validator authority drift"
                )
            qwen_smoke = importlib.import_module(
                "creamem3.qwen_rerank_smoke_v19"
            )
            qwen_path = Path(qwen_smoke.__file__).resolve()
            if qwen_path != expected_qwen_path or (
                _file_sha256(qwen_path) != runner_authority["sha256"]
            ):
                raise MissingCapabilityError(
                    "Qwen reranker smoke validator import drift"
                )
            try:
                validated_qwen = qwen_smoke.validate_complete(
                    output_root=root,
                    allow_test_root=root != DEFAULT_ROOT.resolve(),
                )
            except qwen_smoke.QwenSmokeError as exc:
                raise MissingCapabilityError(
                    f"Qwen reranker smoke evidence rejected: {exc}"
                ) from exc
            if validated_qwen != value:
                raise MissingCapabilityError(
                    "Qwen reranker smoke validator/readback drift"
                )
            permutation = value.get("index_permutation")
            scores = value.get("scores")
            aligned_scores = value.get("aligned_scores")
            formal_cache = Path(
                str(value.get("formal_ranking_cache_root", ""))
            ).resolve()
            expected_formal_cache = (
                root
                / protocol["provider_availability_protocol"]
                ["reranker_prepredict_smoke"]
                ["formal_ranking_cache_namespace"]
            ).resolve()
            if (
                value.get("provider") != "dashscope"
                or value.get("endpoint")
                != protocol["provider_availability_protocol"]
                ["reranker_prepredict_smoke"]
                ["endpoint"]
                or value.get("max_provider_attempts") != 1
                or value.get("complete_index_permutation") is not True
                or value.get("finite_scores") is not True
                or not isinstance(permutation, list)
                or sorted(permutation) != [0, 1, 2]
                or not isinstance(scores, list)
                or len(scores) != 3
                or not all(
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                    for score in scores
                )
                or not isinstance(aligned_scores, list)
                or len(aligned_scores) != 3
                or not all(
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                    for score in aligned_scores
                )
                or any(
                    float(aligned_scores[index]) != float(score)
                    for index, score in zip(permutation, scores)
                )
                or value.get("key_or_fingerprint_persisted") is not False
                or formal_cache != expected_formal_cache
                or smoke_cache == formal_cache
                or smoke_cache in formal_cache.parents
                or formal_cache in smoke_cache.parents
            ):
                raise MissingCapabilityError(
                    "invalid Qwen reranker smoke isolation/shape seal"
                )
        receipts[smoke_id] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
        }
    return receipts


def run_live_phase(
    phase: str,
    *,
    gate_path: Path,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    if phase not in LIVE_PHASES:
        raise MatrixProtocolError(f"not a live matrix phase: {phase}")
    protocol = load_protocol()
    project = validate_project(output_root=output_root)
    preflight_path = output_root / "preflight" / "preflight.json"
    preflight = _read_mapping(preflight_path, "preflight contract")
    preflight_copy = dict(preflight)
    observed = preflight_copy.pop("preflight_contract_sha256", None)
    if observed != _sha256(preflight_copy):
        raise MatrixProtocolError("preflight contract digest mismatch")
    release_path = _release_path(protocol, phase, output_root)
    if not release_path.is_file():
        raise MissingCapabilityError(
            f"{phase} executor release is missing: {release_path}"
        )
    release = validate_release(release_path, phase=phase, protocol=protocol)
    validate_independent_gate(
        gate_path,
        phase=phase,
        protocol=protocol,
        project=project,
        preflight=preflight,
        release_path=release_path,
        release=release,
    )
    _phase_prerequisite(phase, output_root)
    # Provider availability is a distinct gate.  It is checked before the
    # provider-bearing executor module is imported, and smoke outputs/caches
    # are explicitly non-reusable by the matrix.
    provider_smokes = validate_phase_provider_smokes(
        phase, protocol=protocol, output_root=output_root
    )

    # Provider-bearing code may be imported only after every preceding check.
    module = importlib.import_module(str(release["module"]))
    entrypoint = getattr(module, str(release["entrypoint"]), None)
    if not callable(entrypoint):
        raise MissingCapabilityError(f"{phase} executor entrypoint is not callable")
    request: dict[str, Any] = {
        "phase": phase,
        "experiment_id": protocol["experiment_id"],
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": _sha256(protocol),
        "output_root": str(output_root.resolve()),
        "project_manifest_path": str(
            (output_root / "project" / "manifest.json").resolve()
        ),
        "query_manifest_path": project["query_path"],
        "memory_root": project["memory_root"],
        "matrix_cell_ids": matrix_cell_ids(),
        "logical_call_hard_ceiling": release[
            "logical_call_hard_ceiling"
        ],
        "stage_call_hard_ceilings": dict(
            release["stage_call_hard_ceilings"]
        ),
        "legacy_cache_readthrough": False,
        "provider_smoke_seals": provider_smokes,
    }
    if phase == "score":
        # The raw data root first becomes reachable after the ten-cell seal.
        request["score_data_root"] = str(DEFAULT_DATA_ROOT.resolve())
    result = entrypoint(request)
    if not isinstance(result, Mapping):
        raise MatrixProtocolError(f"{phase} executor returned no receipt object")
    calls = int(result.get("provider_calls", -1))
    if calls < 0 or calls > int(release["logical_call_hard_ceiling"]):
        raise MatrixProtocolError(f"{phase} provider call ceiling exceeded")
    receipt = {
        "schema_version": PHASE_RECEIPT_SCHEMA,
        "phase": phase,
        "protocol_sha256": _sha256(protocol),
        "project_contract_sha256": project["project_contract_sha256"],
        "preflight_contract_sha256": preflight[
            "preflight_contract_sha256"
        ],
        "executor_release_sha256": _file_sha256(release_path),
        "provider_calls": calls,
        "logical_call_hard_ceiling": release[
            "logical_call_hard_ceiling"
        ],
        "result": dict(result),
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    if result.get("status") == "waiting_dynamic_embedding_budget_gate":
        # The first build invocation intentionally stops after writer packets
        # and the zero-call embedding census.  Preserve that generation-only
        # receipt content-addressed; a later dynamically rebound release/gate
        # writes the sole final orchestrator receipt after BUILD_COMPLETE.
        _write_once_json(
            output_root
            / phase
            / "orchestrator_partial_receipts"
            / f"{receipt['receipt_sha256']}.json",
            receipt,
        )
    else:
        _write_once_json(output_root / phase / "orchestrator_receipt.json", receipt)
    return receipt


def status_report(output_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    rows: dict[str, Any] = {}
    for phase in PHASE_ORDER:
        candidates = {
            "project": output_root / "project" / "manifest.json",
            "preflight": output_root / "preflight" / "preflight.json",
            "build": output_root / "build" / "complete.json",
            "predict": output_root / "predict" / "complete.json",
            "score": output_root / "score" / "complete.json",
        }
        path = candidates[phase]
        rows[phase] = {
            "path": str(path.resolve()),
            "present": path.is_file(),
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
    rows["executor_releases"] = {
        phase: _release_status(
            _release_path(protocol, phase, output_root), phase
        )
        for phase in sorted(LIVE_PHASES)
    }
    return {
        "experiment_id": protocol["experiment_id"],
        "output_root": str(output_root.resolve()),
        "matrix_cells": matrix_cell_ids(),
        "phases": rows,
        "provider_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(*PHASE_ORDER, "status"),
    )
    parser.add_argument("--allow-model-calls", action="store_true")
    parser.add_argument("--gate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.phase in LIVE_PHASES:
        if not args.allow_model_calls:
            raise SystemExit(
                f"{args.phase} requires --allow-model-calls and an independent gate"
            )
        if args.gate is None:
            raise SystemExit(f"{args.phase} requires --gate")
        result = run_live_phase(args.phase, gate_path=args.gate)
    else:
        if args.allow_model_calls or args.gate is not None:
            raise SystemExit("zero-call phases reject live-call flags")
        if args.phase == "project":
            result = project_dataset()
        elif args.phase == "preflight":
            result = build_preflight()
        else:
            result = status_report()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "status": result.get(
                    "zero_call_protocol_status",
                    result.get("status", "complete"),
                ),
                "output_root": str(DEFAULT_ROOT),
                "contract_sha256": result.get(
                    "preflight_contract_sha256",
                    result.get(
                        "project_contract_sha256",
                        result.get("receipt_sha256"),
                    ),
                ),
                "provider_calls": result.get("provider_calls", 0),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Provider-cold matched MemEye ingest contract for the v19 matrix.

This module is the call-level authority that the old v9/v18 adapters could
not be: it keeps the historical MGTM/Mem0 prompt text and MemEye rendering,
but freezes the actual 0.4533 call contract at 1,200 summary tokens, 16,000
fact tokens, and three provider attempts per request.  No provider is
imported here.  Callers inject generation and embedding functions only after
the matrix gate has been approved.

Two representations are emitted from the same caption bank:

* CreaMem writes exactly one packet per complete physical session and adapts
  only serialization/provenance to the typed graph schema.
* MGTM keeps the native historical greedy 10-image/10-round batches and the
  native summary -> fact tree.  The historical pure batch/render functions
  are loaded from their pinned source with AST, rather than importing the
  provider-bearing evaluation program.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import memeye_adapter_v9 as legacy


SCHEMA_VERSION = "creamem3.memeye_matched_ingest.v19.0"
CREAMEM_PACKET_SCHEMA = "creamem3.memeye_matched_session_packet.v19.0"
NATIVE_PACKET_SCHEMA = "creamem3.memeye_matched_native_packet.v19.0"
NATIVE_TREE_SCHEMA = "creamem3.memeye_matched_native_tree.v19.0"

SUMMARY_MAX_OUTPUT_TOKENS = 1200
FACT_MAX_OUTPUT_TOKENS = 16000
MAX_PROVIDER_ATTEMPTS = 3
# The eight historical V9/V7 native tree/report pairs were all written on
# 2026-07-05, and eval_memeye.py passed date.today().isoformat() into every
# fact request.  Freezing that date removes an otherwise unintended prompt
# variable while comparing the two backbones.
FROZEN_CURRENT_DATE = "2026-07-05"
NATIVE_MAX_IMAGES = 10
NATIVE_MAX_ROUNDS = 10

WORK_MODELS = {
    "g25": "gemini-2.5-flash-lite",
    "g35": "gemini-3.5-flash-lite",
}
EMBED_MODEL = "gemini-embedding-001"

SOURCE_EVAL = Path(
    r"C:\Users\86139\Desktop\CreaMem-AutoResearch\multimodal\eval_memeye.py"
)
EXPECTED_SOURCE_EVAL_SHA256 = (
    "98f645b36f4766fded5500bcead7e7713edefceff070ac95058aaf873e5ee0ce"
)
EXPECTED_SOURCE_FUNCTION_SHA256 = {
    "summarize_batch": "c6cff990897a0001573952a39a245b2001bf975de2df0e7bae8eab8abcb0cb61",
    "ingest_batch": "8d00dc3e3a0b91e06b0e326132ce4c566d664cc2f40619816218bd163499ffe5",
    "make_batches": "dcf0e5f90c0caa1d8043452b853dc9f1475ee2ed7f9b1c19e9719b22af4dec01",
    "render_batch": "bac70b0da8034590799a4833d9069a4a7d1bfc52c62638ba3e5ed0cf448339a5",
}
EXPECTED_HISTORICAL_PROMPT_BUNDLE_SHA256 = (
    "c33b9350d6cf877ce962f68218b4a7699bef71a59f3f2260faf7dbf76d9e3081"
)

GenerateFn = Callable[..., Any]
EmbedFn = Callable[..., list[list[float]]]


class MatchedIngestError(RuntimeError):
    """A prompt, source, output, provenance, or graph invariant failed."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _function_segment(source: str, name: str) -> tuple[str, ast.FunctionDef]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            if not segment:
                raise MatchedIngestError(f"cannot recover source for {name}")
            return segment, node
    raise MatchedIngestError(f"missing source function {name}")


def source_authority_receipt() -> dict[str, Any]:
    """Bind exact historical calls plus every reused pure adapter function."""

    source = SOURCE_EVAL.read_text(encoding="utf-8")
    external_names = (
        "summarize_batch",
        "ingest_batch",
        "round_sort_key",
        "collect_rounds",
        "image_occurrence_key",
        "render_round",
        "render_batch",
        "make_batches",
        "build_mem0_user_prompt",
    )
    external = {
        name: _text_sha256(_function_segment(source, name)[0])
        for name in external_names
    }
    for name, expected in EXPECTED_SOURCE_FUNCTION_SHA256.items():
        if external[name] != expected:
            raise MatchedIngestError(f"historical authority drift: {name}")
    if _file_sha256(SOURCE_EVAL) != EXPECTED_SOURCE_EVAL_SHA256:
        raise MatchedIngestError("historical eval_memeye.py file drift")

    adapter_names = (
        "load_mgtm_prompt_sources",
        "build_records_from_cached_captions",
        "_session_batch",
        "_fact_user_prompt",
        "_parse_json",
        "audit_memeye_graph",
        "add_large_memeye_offline_event_views",
    )
    adapter = {
        name: _text_sha256(inspect.getsource(getattr(legacy, name)))
        for name in adapter_names
    }
    graph_names = ("_validate_session_packet", "_embed_checked", "audit_graph")
    graph = {
        name: _text_sha256(inspect.getsource(getattr(gm, name)))
        for name in graph_names
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "historical_eval_path": str(SOURCE_EVAL.resolve()),
        "historical_eval_file_sha256": _file_sha256(SOURCE_EVAL),
        "historical_external_function_sha256": external,
        "adapter_file_sha256": _file_sha256(Path(legacy.__file__).resolve()),
        "adapter_function_sha256": adapter,
        "graph_file_sha256": _file_sha256(Path(gm.__file__).resolve()),
        "graph_function_sha256": graph,
        "summary_max_output_tokens": SUMMARY_MAX_OUTPUT_TOKENS,
        "fact_max_output_tokens": FACT_MAX_OUTPUT_TOKENS,
        "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        "frozen_current_date": FROZEN_CURRENT_DATE,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def load_prompt_bundle() -> dict[str, str]:
    """Load the exact historical prompt text through the provider-cold v9 loader."""

    bundle = legacy.load_mgtm_prompt_sources()
    if bundle.get("prompt_bundle_sha256") != EXPECTED_HISTORICAL_PROMPT_BUNDLE_SHA256:
        raise MatchedIngestError("historical MGTM/Mem0 prompt bundle drift")
    return dict(bundle)


def build_summary_request(
    rendered_batch: str,
    prompt_bundle: Mapping[str, str],
    *,
    backbone: str,
    unit_id: str,
) -> dict[str, Any]:
    if backbone not in WORK_MODELS:
        raise MatchedIngestError(f"unknown backbone {backbone}")
    prompt = str(prompt_bundle["summary_prompt"]).replace(
        "{batch}", str(rendered_batch)
    )
    request = {
        "stage": "writer",
        "substage": "summary",
        "backbone": backbone,
        "model": WORK_MODELS[backbone],
        "unit_id": str(unit_id),
        "prompt": prompt,
        "system": None,
        "json_mode": True,
        "max_output_tokens": SUMMARY_MAX_OUTPUT_TOKENS,
        "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        "rendered_input_sha256": _text_sha256(rendered_batch),
        "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
    }
    request["request_contract_sha256"] = _canonical_sha256(request)
    return request


def build_creamem_fact_user_prompt(
    record: Mapping[str, Any],
    summary: str,
    prompt_bundle: Mapping[str, str],
    *,
    current_date: str = FROZEN_CURRENT_DATE,
) -> str:
    """Render v9's schema adapter with an explicit, resumable current date."""

    batch = legacy._session_batch(record)  # type: ignore[attr-defined]
    timestamp = str(record.get("timestamp", "") or "")
    legal_ids = [str(turn["dia_id"]) for turn in record.get("turns", [])]
    return "\n\n".join(
        [
            f"## Summary\n{str(summary).strip()}",
            "## Last k Messages\n",
            "## Recently Extracted Memories\n[]",
            "## Existing Memories\n[]",
            f"## New Messages\n{batch}",
            f"## Observation Date\n{timestamp}",
            f"## Current Date\n{str(current_date).strip()}",
            "## Custom Instructions\n"
            + str(prompt_bundle["visual_custom_instructions"]),
            "## LEGAL_SOURCE_DIA_IDS\n"
            + json.dumps(legal_ids, ensure_ascii=False),
            legacy.CREAMEM3_OUTPUT_OVERRIDE,
            "# Output:",
        ]
    )


def build_creamem_fact_request(
    record: Mapping[str, Any],
    summary: str,
    prompt_bundle: Mapping[str, str],
    *,
    backbone: str,
    unit_id: str,
    current_date: str = FROZEN_CURRENT_DATE,
) -> dict[str, Any]:
    if backbone not in WORK_MODELS:
        raise MatchedIngestError(f"unknown backbone {backbone}")
    system = "\n\n".join(
        [
            str(prompt_bundle["mem0_additive_prompt"]),
            str(prompt_bundle["visual_system_override"]),
            legacy.CREAMEM3_OUTPUT_OVERRIDE,
        ]
    )
    prompt = build_creamem_fact_user_prompt(
        record, summary, prompt_bundle, current_date=current_date
    )
    request = {
        "stage": "writer",
        "substage": "creamem_fact",
        "backbone": backbone,
        "model": WORK_MODELS[backbone],
        "unit_id": str(unit_id),
        "prompt": prompt,
        "system": system,
        "json_mode": True,
        "max_output_tokens": FACT_MAX_OUTPUT_TOKENS,
        "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        "current_date": str(current_date),
        "summary_sha256": _text_sha256(summary),
        "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
    }
    request["request_contract_sha256"] = _canonical_sha256(request)
    return request


def _reply_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("text", ""))
    return str(getattr(value, "text", ""))


def _invoke(generate_fn: GenerateFn, request: Mapping[str, Any]) -> Any:
    return generate_fn(
        str(request["stage"]),
        str(request["prompt"]),
        system=request.get("system"),
        json_mode=bool(request["json_mode"]),
        max_output_tokens=int(request["max_output_tokens"]),
        max_provider_attempts=int(request["max_provider_attempts"]),
        cache_metadata={
            "schema_version": SCHEMA_VERSION,
            "substage": request["substage"],
            "backbone": request["backbone"],
            "unit_id": request["unit_id"],
            "request_contract_sha256": request["request_contract_sha256"],
        },
    )


def extract_creamem_session_packet(
    record: Mapping[str, Any],
    generate_fn: GenerateFn,
    prompt_bundle: Mapping[str, str],
    *,
    backbone: str,
    task: str,
) -> dict[str, Any]:
    """Write one complete physical-session packet with exact call caps."""

    session_id = int(record["session_id"])
    unit_id = f"{task}::session::{session_id:04d}"
    rendered = legacy._session_batch(record)  # type: ignore[attr-defined]
    summary_request = build_summary_request(
        rendered, prompt_bundle, backbone=backbone, unit_id=unit_id
    )
    summary_reply = _invoke(generate_fn, summary_request)
    summary_value = legacy._parse_json(_reply_text(summary_reply))  # type: ignore[attr-defined]
    summary = str(summary_value.get("batch_summary", "")).strip()
    if not summary:
        raise MatchedIngestError(f"{unit_id} returned an empty batch_summary")

    fact_request = build_creamem_fact_request(
        record,
        summary,
        prompt_bundle,
        backbone=backbone,
        unit_id=unit_id,
    )
    fact_reply = _invoke(generate_fn, fact_request)
    fact_value = legacy._parse_json(_reply_text(fact_reply))  # type: ignore[attr-defined]
    if set(fact_value) != {"atomic_facts"}:
        raise MatchedIngestError(f"{unit_id} fact output schema drift")
    legal_ids = [str(turn["dia_id"]) for turn in record.get("turns", [])]
    normalized_summary, facts = gm._validate_session_packet(  # type: ignore[attr-defined]
        {"batch_summary": summary, "atomic_facts": fact_value["atomic_facts"]},
        legal_ids,
    )
    return {
        "schema_version": CREAMEM_PACKET_SCHEMA,
        "backbone": backbone,
        "model": WORK_MODELS[backbone],
        "task": task,
        "session_id": session_id,
        "source_session_id": str(record.get("source_session_id", "")),
        "timestamp": str(record.get("timestamp", "") or ""),
        "iso_date": legacy._iso_date(str(record.get("timestamp", "") or "")),  # type: ignore[attr-defined]
        "turn_count": len(legal_ids),
        "source_dia_ids": legal_ids,
        "batch_summary": normalized_summary,
        "atomic_facts": facts,
        "summary_request_contract_sha256": summary_request[
            "request_contract_sha256"
        ],
        "fact_request_contract_sha256": fact_request["request_contract_sha256"],
        "summary_response_sha256": _text_sha256(_reply_text(summary_reply)),
        "fact_response_sha256": _text_sha256(_reply_text(fact_reply)),
        "summary_logical_calls": 1,
        "fact_logical_calls": 1,
        "max_provider_attempts_per_call": MAX_PROVIDER_ATTEMPTS,
        "input_sha256": _text_sha256(rendered),
        "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
    }


def materialize_creamem_session_graph(
    packets: Sequence[Mapping[str, Any]],
    source_map: Mapping[str, Mapping[str, Any]],
    embed_fn: EmbedFn,
    *,
    backbone: str,
    task: str,
) -> dict[str, Any]:
    """Materialize the provider-free typed graph from validated packets."""

    ordered = sorted(packets, key=lambda item: int(item["session_id"]))
    if not ordered:
        raise MatchedIngestError("cannot materialize an empty session graph")
    summaries: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for packet in ordered:
        session_id = int(packet["session_id"])
        summary_id = f"S{session_id:03d}"
        summary_sources: list[str] = []
        for item in packet["atomic_facts"]:
            fact_id = f"F{len(facts) + 1:05d}"
            for source_id in item["source_dia_ids"]:
                if source_id not in summary_sources:
                    summary_sources.append(source_id)
            facts.append(
                {
                    "id": fact_id,
                    "node_type": "atomic_fact",
                    "text": str(item["fact"]),
                    "source_dia_ids": list(item["source_dia_ids"]),
                    "session_id": session_id,
                    "source_session_id": packet["source_session_id"],
                    "timestamp": packet["timestamp"],
                    "iso_date": packet["iso_date"],
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
                "text": str(packet["batch_summary"]),
                "session_ids": [session_id],
                "source_session_id": packet["source_session_id"],
                "timestamp": packet["timestamp"],
                "iso_date": packet["iso_date"],
                "source_dia_ids": summary_sources,
                "embedding": [],
            }
        )

    nodes = [*summaries, *facts]
    vectors = gm._embed_checked(  # type: ignore[attr-defined]
        [str(node["text"]) for node in nodes],
        embed_fn,
        tag="creamem3_memeye_v19_session_embed",
    )
    for node, vector in zip(nodes, vectors):
        node["embedding"] = vector
    bundle = load_prompt_bundle()
    graph: dict[str, Any] = {
        "schema_version": gm.GRAPH_SCHEMA_VERSION,
        "matched_ingest_schema_version": SCHEMA_VERSION,
        "graph_variant": "session",
        "backbone": backbone,
        "work_model": WORK_MODELS[backbone],
        "task": task,
        "architecture": {
            "representation": "typed_many_to_many_event_graph",
            "fact_policy": "immutable_atomic_facts_are_source_of_truth",
            "summary_policy": "derived_routing_views_do_not_own_facts",
            "write_unit": "one_complete_physical_memeye_session",
            "input_scope": "dialogue_plus_frozen_gemini_image_captions",
            "source_extraction_semantics": "mgtm_mem0_visual_high_recall",
            "source_output_adaptation": "creamem3_fact_plus_provenance_schema_only",
            "summary_max_output_tokens": SUMMARY_MAX_OUTPUT_TOKENS,
            "fact_max_output_tokens": FACT_MAX_OUTPUT_TOKENS,
            "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        },
        "prompt_schema_version": SCHEMA_VERSION,
        "prompt_sha256": {
            **dict(gm.PROMPT_SHA256),
            "memeye_mgtm_bundle": bundle["prompt_bundle_sha256"],
            "memeye_output_override": _text_sha256(
                legacy.CREAMEM3_OUTPUT_OVERRIDE
            ),
        },
        "session_packets": copy.deepcopy(list(ordered)),
        "summary_nodes": summaries,
        "fact_nodes": facts,
        "edges": edges,
        "memeye_source_map": copy.deepcopy(dict(source_map)),
        "stats": {
            "sessions": len(ordered),
            "session_summary_nodes": len(summaries),
            "event_view_nodes": 0,
            "facts": len(facts),
            "edges": len(edges),
            "summary_logical_calls": len(ordered),
            "fact_logical_calls": len(ordered),
            "max_provider_attempts_per_call": MAX_PROVIDER_ATTEMPTS,
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
        }
    )
    return graph


_PURE_EXTERNAL_NAMES = (
    "round_sort_key",
    "collect_rounds",
    "image_occurrence_key",
    "render_round",
    "render_batch",
    "make_batches",
    "build_mem0_user_prompt",
)


def load_native_pure_authority() -> dict[str, Callable[..., Any]]:
    """Compile only pinned provider-free function definitions from eval_memeye."""

    if _file_sha256(SOURCE_EVAL) != EXPECTED_SOURCE_EVAL_SHA256:
        raise MatchedIngestError("historical eval source drift")
    source = SOURCE_EVAL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected: list[ast.stmt] = []
    for name in _PURE_EXTERNAL_NAMES:
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(matches) != 1:
            raise MatchedIngestError(f"native pure authority missing {name}")
        selected.append(matches[0])
    literal_values: dict[str, Any] = {}
    for literal_name in (
        "BATCH_IMAGES",
        "BATCH_ROUNDS",
        "MEM0_VISUAL_CUSTOM_INSTRUCTIONS",
    ):
        matches: list[ast.AST] = []
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == literal_name
                for target in targets
            ):
                matches.append(node.value)
        if len(matches) != 1:
            raise MatchedIngestError(
                f"native literal authority missing {literal_name}"
            )
        literal_values[literal_name] = ast.literal_eval(matches[0])
    if (
        literal_values["BATCH_IMAGES"] != NATIVE_MAX_IMAGES
        or literal_values["BATCH_ROUNDS"] != NATIVE_MAX_ROUNDS
        or literal_values["MEM0_VISUAL_CUSTOM_INSTRUCTIONS"]
        != load_prompt_bundle()["visual_custom_instructions"]
    ):
        raise MatchedIngestError("native literal authority drift")
    namespace: dict[str, Any] = {"re": re, **literal_values}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=selected, type_ignores=[])
            ),
            str(SOURCE_EVAL),
            "exec",
        ),
        namespace,
    )
    return {name: namespace[name] for name in _PURE_EXTERNAL_NAMES}


def native_batch_plan(
    task: str,
    memory_payload: Mapping[str, Any],
    captions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[str]]:
    """Use the exact native collector, greedy 10/10 batcher, and renderer."""

    authority = load_native_pure_authority()
    raw = {
        "multi_session_dialogues": copy.deepcopy(
            list(memory_payload.get("multi_session_dialogues", []))
        )
    }
    rounds = authority["collect_rounds"](task, raw, Path("."))
    batches = authority["make_batches"](
        rounds, max_images=NATIVE_MAX_IMAGES, max_rounds=NATIVE_MAX_ROUNDS
    )
    rendered = [authority["render_batch"](batch, captions) for batch in batches]
    return rounds, batches, rendered


def parse_native_facts(raw: Any) -> list[str]:
    value = legacy._parse_json(_reply_text(raw))  # type: ignore[attr-defined]
    output: list[str] = []
    memories = value.get("memory", [])
    if isinstance(memories, list):
        for item in memories:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or item.get("fact")
            else:
                text = item
            normalized = " ".join(str(text or "").split())
            if normalized:
                output.append(normalized)
    if not output and isinstance(value.get("atomic_facts"), list):
        for item in value["atomic_facts"]:
            normalized = " ".join(str(item or "").split())
            if normalized:
                output.append(normalized)
    return output


def extract_native_batch_packet(
    *,
    task: str,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    rendered: str,
    generate_fn: GenerateFn,
    prompt_bundle: Mapping[str, str],
    backbone: str,
) -> dict[str, Any]:
    """Run one native MGTM summary/fact pair without CreaMem serialization."""

    unit_id = f"{task}::native::{batch_index:04d}"
    summary_request = build_summary_request(
        rendered, prompt_bundle, backbone=backbone, unit_id=unit_id
    )
    summary_reply = _invoke(generate_fn, summary_request)
    summary_value = legacy._parse_json(_reply_text(summary_reply))  # type: ignore[attr-defined]
    summary = str(summary_value.get("batch_summary", "")).strip()
    if not summary:
        summary = f"{task} rounds {batch[0]['round']}..{batch[-1]['round']}"

    authority = load_native_pure_authority()
    observation_date = str(batch[0].get("date", "") or FROZEN_CURRENT_DATE)
    system = "\n\n".join(
        [
            str(prompt_bundle["mem0_additive_prompt"]),
            str(prompt_bundle["visual_system_override"]),
        ]
    )
    fact_prompt = authority["build_mem0_user_prompt"](
        rendered, observation_date, FROZEN_CURRENT_DATE
    )
    fact_request = {
        "stage": "writer",
        "substage": "native_fact",
        "backbone": backbone,
        "model": WORK_MODELS[backbone],
        "unit_id": unit_id,
        "prompt": fact_prompt,
        "system": system,
        "json_mode": True,
        "max_output_tokens": FACT_MAX_OUTPUT_TOKENS,
        "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        "current_date": FROZEN_CURRENT_DATE,
        "rendered_input_sha256": _text_sha256(rendered),
        "prompt_bundle_sha256": prompt_bundle["prompt_bundle_sha256"],
    }
    fact_request["request_contract_sha256"] = _canonical_sha256(fact_request)
    fact_reply = _invoke(generate_fn, fact_request)
    facts = parse_native_facts(fact_reply)
    return {
        "schema_version": NATIVE_PACKET_SCHEMA,
        "backbone": backbone,
        "model": WORK_MODELS[backbone],
        "task": task,
        "batch_index": batch_index,
        "batch_id": f"NB{batch_index:04d}",
        "summary": summary,
        "facts": facts,
        "t_start": f"{batch[0]['date']} {batch[0]['round']}",
        "t_end": f"{batch[-1]['date']} {batch[-1]['round']}",
        "rounds": len(batch),
        "images": sum(len(row.get("images", [])) for row in batch),
        "summary_request_contract_sha256": summary_request[
            "request_contract_sha256"
        ],
        "fact_request_contract_sha256": fact_request["request_contract_sha256"],
        "summary_response_sha256": _text_sha256(_reply_text(summary_reply)),
        "fact_response_sha256": _text_sha256(_reply_text(fact_reply)),
        "summary_logical_calls": 1,
        "fact_logical_calls": 1,
        "max_provider_attempts_per_call": MAX_PROVIDER_ATTEMPTS,
        "rendered_input_sha256": _text_sha256(rendered),
    }


def materialize_native_tree(
    *,
    task: str,
    backbone: str,
    rounds: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
    embed_fn: EmbedFn,
) -> dict[str, Any]:
    """Materialize the historical MGTM tree shape with JSON-safe embeddings."""

    summaries: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    for packet in sorted(packets, key=lambda row: int(row["batch_index"])):
        summary_id = len(summaries)
        summaries.append(
            {
                "id": summary_id,
                "text": str(packet["summary"]),
                "t_start": str(packet["t_start"]),
                "t_end": str(packet["t_end"]),
                "apps": [task],
                "rounds": int(packet["rounds"]),
                "images": int(packet["images"]),
                "emb": [],
            }
        )
        for text in packet["facts"]:
            facts.append(
                {
                    "id": len(facts),
                    "parent_summary_id": summary_id,
                    "text": str(text),
                    "emb": [],
                }
            )
    if summaries:
        vectors = embed_fn(
            [str(row["text"]) for row in summaries],
            tag="memeye_v19_mgtm_summary_embed",
        )
        if len(vectors) != len(summaries):
            raise MatchedIngestError("native summary embedding count mismatch")
        for row, vector in zip(summaries, vectors):
            row["emb"] = [float(value) for value in vector]
    if facts:
        vectors = embed_fn(
            [str(row["text"]) for row in facts],
            tag="memeye_v19_mgtm_fact_embed",
        )
        if len(vectors) != len(facts):
            raise MatchedIngestError("native fact embedding count mismatch")
        for row, vector in zip(facts, vectors):
            row["emb"] = [float(value) for value in vector]
    safe_rounds: list[dict[str, Any]] = []
    for raw in rounds:
        row = {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in {"_sort", "image_root"}
        }
        safe_rounds.append(row)
    tree: dict[str, Any] = {
        "schema_version": NATIVE_TREE_SCHEMA,
        "backbone": backbone,
        "work_model": WORK_MODELS[backbone],
        "task": task,
        "architecture": {
            "representation": "native_mgtm_summary_fact_tree",
            "write_unit": "greedy_10_images_or_10_rounds",
            "summary_max_output_tokens": SUMMARY_MAX_OUTPUT_TOKENS,
            "fact_max_output_tokens": FACT_MAX_OUTPUT_TOKENS,
            "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
            "creamem_schema_adapter_applied": False,
        },
        "items": safe_rounds,
        "SUMMARIES": summaries,
        "FACTS": facts,
        "batch_packets": copy.deepcopy(list(packets)),
        "stats": {
            "rounds": len(safe_rounds),
            "write_batches": len(packets),
            "summaries": len(summaries),
            "facts": len(facts),
            "summary_logical_calls": len(packets),
            "fact_logical_calls": len(packets),
        },
    }
    tree["content_fingerprint"] = _canonical_sha256(
        {
            "summaries": [
                {key: value for key, value in row.items() if key != "emb"}
                for row in summaries
            ],
            "facts": [
                {key: value for key, value in row.items() if key != "emb"}
                for row in facts
            ],
        }
    )
    return tree


def build_records_from_captions(
    task: str,
    memory_payload: Mapping[str, Any],
    captions: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Expose the pinned v9 physical-session projector without drift."""

    return legacy.build_records_from_cached_captions(task, memory_payload, captions)


def protocol_manifest() -> dict[str, Any]:
    bundle = load_prompt_bundle()
    value = {
        "schema_version": SCHEMA_VERSION,
        "models": dict(WORK_MODELS),
        "embedding_model": EMBED_MODEL,
        "summary_max_output_tokens": SUMMARY_MAX_OUTPUT_TOKENS,
        "fact_max_output_tokens": FACT_MAX_OUTPUT_TOKENS,
        "max_provider_attempts": MAX_PROVIDER_ATTEMPTS,
        "frozen_current_date": FROZEN_CURRENT_DATE,
        "native_batching": {
            "max_images": NATIVE_MAX_IMAGES,
            "max_rounds": NATIVE_MAX_ROUNDS,
            "mode": "historical_greedy_exact_ast_authority",
        },
        "prompt_bundle_sha256": bundle["prompt_bundle_sha256"],
        "source_authority": source_authority_receipt(),
    }
    value["protocol_sha256"] = _canonical_sha256(value)
    return value


__all__ = [
    "CREAMEM_PACKET_SCHEMA",
    "EMBED_MODEL",
    "FACT_MAX_OUTPUT_TOKENS",
    "FROZEN_CURRENT_DATE",
    "MAX_PROVIDER_ATTEMPTS",
    "NATIVE_MAX_IMAGES",
    "NATIVE_MAX_ROUNDS",
    "NATIVE_PACKET_SCHEMA",
    "NATIVE_TREE_SCHEMA",
    "SCHEMA_VERSION",
    "SUMMARY_MAX_OUTPUT_TOKENS",
    "WORK_MODELS",
    "MatchedIngestError",
    "build_creamem_fact_request",
    "build_creamem_fact_user_prompt",
    "build_records_from_captions",
    "build_summary_request",
    "extract_creamem_session_packet",
    "extract_native_batch_packet",
    "load_native_pure_authority",
    "load_prompt_bundle",
    "materialize_creamem_session_graph",
    "materialize_native_tree",
    "native_batch_plan",
    "parse_native_facts",
    "protocol_manifest",
    "source_authority_receipt",
]

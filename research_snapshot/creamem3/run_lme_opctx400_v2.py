"""Run the frozen 400-question LongMemEval complement with OpCtx v2.

This campaign wrapper reuses the byte-frozen 100-question v2 execution core
without editing it.  It binds a separately audited 400-question complement,
uses fresh roots, and records every launch's concurrency settings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from creamem3 import lme_opctx_v2 as adapter
from creamem3 import run_lme_opctx100_v1 as legacy
from creamem3 import run_lme_opctx100_v2 as core


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "creamem3" / "audit" / "lme400_complement_20260831" / "lme400_manifest.json"
MANIFEST_SHA256 = "730750005e28bcc01ff50ab4f86ec0113c07a3d3d6daa8ae35e85e8babe9c64a"
MANIFEST_BUILDER = MANIFEST.with_name("build_lme400_complement_manifest.py")
MANIFEST_AUDIT = MANIFEST.with_name("lme400_manifest.audit.json")
EXCLUDED_MANIFEST = (
    ROOT / "creamem3" / "audit" / "lme100_stratified_20260826" / "lme100_manifest.json"
)
EXCLUDED_MANIFEST_SHA256 = "d4c446b1eb3bfb58e0cd54910b5453e0e352ca325199dae1fb3e26e1342e593c"
ORDERED_IDS_SHA256 = "f4b55e1d4dc19f0c3fdcd8cb4144e7ca07f7c9b632f4e890c5f774e82b0ba94e"
SORTED_ID_SET_SHA256 = "87b263bc532ebd8977d26efbe5a4992a6167aaa4593871c48f5b69a15ac342ce"
BINDING_ITEMS_SHA256 = "b2eb37e6fddfd63ca48edcad151cf23f1af26c73d6fd2c8a854b12798f0e8930"

DEFAULT_OUTPUT = ROOT / "creamem3" / "results_lme_opctx400_v2_b3_d768_q6w1"
DEFAULT_CACHE = Path(r"D:\creamem3_cache\lme_opctx400_v2_b3_d768_q6w1")
DEFAULT_PREFLIGHT = ROOT / "creamem3" / "audit" / "lme_opctx400_v2" / "preflight.json"
RUN_SCHEMA = "creamem3.lme_opctx400.runner.v2_b3_d768"
FROZEN_V2_CORE_SHA256 = "b331c56345ca28e38da0fcdea08897c4d0216e6e94605388753103ac9c8c718a"

_BASE_RUN_CONFIG = core._run_config


def _load_inputs(
    contract: adapter.PromptContract,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if core._sha_file(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("frozen LME400 manifest SHA mismatch")
    if core._sha_file(EXCLUDED_MANIFEST) != EXCLUDED_MANIFEST_SHA256:
        raise RuntimeError("excluded LME100 manifest SHA mismatch")
    data = core._load_json(legacy.LME_DATASET)
    manifest = core._load_json(MANIFEST)
    excluded_manifest = core._load_json(EXCLUDED_MANIFEST)
    if not isinstance(data, list) or len(data) != 500:
        raise RuntimeError("LongMemEval source must contain 500 records")
    if (
        manifest.get("count") != 400
        or len(manifest.get("ids", [])) != 400
        or len(manifest.get("items", [])) != 400
        or len(set(manifest.get("ids", []))) != 400
    ):
        raise RuntimeError("manifest must bind exactly 400 unique questions")
    if (
        excluded_manifest.get("count") != 100
        or len(excluded_manifest.get("ids", [])) != 100
        or len(set(excluded_manifest.get("ids", []))) != 100
    ):
        raise RuntimeError("excluded manifest must bind exactly 100 unique questions")
    if manifest.get("exclusion", {}).get("sha256") != EXCLUDED_MANIFEST_SHA256:
        raise RuntimeError("LME400 exclusion receipt differs from frozen LME100")
    invariants = manifest.get("invariants", {})
    if (
        invariants.get("ordered_ids_sha256") != ORDERED_IDS_SHA256
        or invariants.get("sorted_id_set_sha256") != SORTED_ID_SET_SHA256
        or invariants.get("binding_items_sha256") != BINDING_ITEMS_SHA256
        or core._canonical(manifest["ids"]) != ORDERED_IDS_SHA256
        or core._canonical(sorted(manifest["ids"])) != SORTED_ID_SET_SHA256
        or core._canonical(manifest["items"]) != BINDING_ITEMS_SHA256
    ):
        raise RuntimeError("LME400 deterministic complement invariants differ")

    source_ids = {str(item["question_id"]) for item in data}
    excluded_ids = {str(value) for value in excluded_manifest["ids"]}
    selected_ids = {str(value) for value in manifest["ids"]}
    if excluded_ids & selected_ids or excluded_ids | selected_ids != source_ids:
        raise RuntimeError("LME100/LME400 are not an exact disjoint partition of the source")

    selected: list[dict[str, Any]] = []
    for position, binding in enumerate(manifest["items"]):
        if int(binding.get("position", -1)) != position:
            raise RuntimeError(f"manifest position mismatch at {position}")
        source_index = int(binding["source_index"])
        item = data[source_index]
        qid = str(item["question_id"])
        if qid != str(binding["question_id"]) or qid != str(manifest["ids"][position]):
            raise RuntimeError(f"manifest/source id mismatch at {position}")
        if core._canonical(item) != str(binding["record_sha256"]):
            raise RuntimeError(f"source record hash mismatch for {qid}")
        if str(binding["question_type"]) != str(item["question_type"]):
            raise RuntimeError(f"question type mismatch for {qid}")
        if bool(binding["is_abstention"]) != qid.endswith("_abs"):
            raise RuntimeError(f"abstention binding mismatch for {qid}")
        answer_type = "integer" if isinstance(item["answer"], int) else "string"
        if str(binding["answer_json_type"]) != answer_type:
            raise RuntimeError(f"answer type mismatch for {qid}")
        if len(item["haystack_sessions"]) != int(binding["session_count"]):
            raise RuntimeError(f"session count mismatch for {qid}")
        if sum(len(session) for session in item["haystack_sessions"]) != int(binding["turn_count"]):
            raise RuntimeError(f"turn count mismatch for {qid}")
        if not (
            len(item["haystack_sessions"])
            == len(item["haystack_dates"])
            == len(item["haystack_session_ids"])
        ):
            raise RuntimeError(f"session arrays misaligned for {qid}")
        selected.append(item)
    if [str(row["question_id"]) for row in selected] != list(manifest["ids"]):
        raise RuntimeError("selected source order differs from frozen LME400 manifest")
    if contract.file_sha256["lme_dataset"] != core._sha_file(legacy.LME_DATASET):
        raise RuntimeError("dataset changed after prompt preflight")
    return data, manifest, selected


def _run_config(contract: adapter.PromptContract) -> dict[str, Any]:
    config = _BASE_RUN_CONFIG(contract)
    sources = config["implementation_source_sha256"]
    sources["frozen_v2_core"] = sources.pop("runner")
    sources["campaign_runner"] = core._sha_file(Path(__file__).resolve())
    sources["complement_manifest_builder"] = core._sha_file(MANIFEST_BUILDER)
    config["campaign"] = {
        "name": "lme400_complement_of_completed_lme100",
        "manifest": str(MANIFEST),
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_audit_sha256": core._sha_file(MANIFEST_AUDIT),
        "excluded_lme100_manifest_sha256": EXCLUDED_MANIFEST_SHA256,
        "ordered_ids_sha256": ORDERED_IDS_SHA256,
        "sorted_id_set_sha256": SORTED_ID_SET_SHA256,
        "binding_items_sha256": BINDING_ITEMS_SHA256,
        "complement_count": 400,
        "overlap_with_completed_lme100": 0,
        "union_source_count": 500,
    }
    config["cache_contract"]["campaign_manifest_sha256"] = MANIFEST_SHA256
    config["cache_contract"]["completed_lme100_cache_compatible"] = False
    return config


def _patch_core() -> None:
    """Patch campaign identity once, before any worker thread can start."""

    if core._sha_file(Path(core.__file__).resolve()) != FROZEN_V2_CORE_SHA256:
        raise RuntimeError("frozen LME100 v2 execution core SHA mismatch")
    core.MANIFEST = MANIFEST
    core.MANIFEST_SHA256 = MANIFEST_SHA256
    core.RUN_SCHEMA = RUN_SCHEMA
    core._run_config = _run_config


def run_preflight(
    output: Path = DEFAULT_PREFLIGHT,
    *,
    limit: int = 400,
    question_workers: int = 6,
    writer_workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= limit <= 400:
        raise ValueError("preflight limit must be 1..400")
    if not 1 <= question_workers <= 8 or not 1 <= writer_workers <= 8:
        raise ValueError("preflight worker counts must be 1..8")
    if question_workers * writer_workers > 8:
        raise ValueError("preflight worker product must not exceed 8")
    _patch_core()
    contract = adapter.assert_prompt_contract()
    _, manifest, selected_all = _load_inputs(contract)
    selected = selected_all[:limit]
    order_rows = [legacy._ordered_session_specs(item)[1] for item in selected]
    receipt = {
        "schema_version": "creamem3.lme_opctx400.preflight.v2_b3_d768",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "network_calls": 0,
        "model_calls": 0,
        "manifest": {
            "path": str(MANIFEST),
            "sha256": MANIFEST_SHA256,
            "count": manifest["count"],
            "selected_count": limit,
            "excluded_lme100_manifest_sha256": EXCLUDED_MANIFEST_SHA256,
        },
        "execution_plan": {
            "question_workers": question_workers,
            "writer_workers": writer_workers,
            "worker_product": question_workers * writer_workers,
            "generation_semaphore": 8,
            "qwen_semaphore": 4,
        },
        "prompt_contract": contract.receipt(),
        "prompt_hashes_unchanged_from_completed_lme100": {
            "write": contract.prompt_sha256["write_template"],
            "qa": contract.prompt_sha256["qa_template"],
            "judge": contract.prompt_sha256["judge_template"],
        },
        "causal_order": {
            "policy": "stable_timestamp_then_source_position_then_consecutive_chunk_3",
            "questions_reordered": sum(bool(row["reordered"]) for row in order_rows),
        },
        "call_estimate": core._call_estimate(selected),
        "run_config": _run_config(contract),
    }
    receipt["receipt_sha256"] = core._canonical(receipt)
    core._atomic_json(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--question-workers", type=int, default=6)
    parser.add_argument("--writer-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.limit <= 400:
        raise SystemExit("--limit must be 1..400")
    if not 1 <= args.question_workers <= 8 or not 1 <= args.writer_workers <= 8:
        raise SystemExit("worker counts must be 1..8")
    if args.question_workers * args.writer_workers > 8:
        raise SystemExit("question-workers x writer-workers must not exceed 8")
    _patch_core()
    preflight = run_preflight(
        args.preflight_output.resolve(),
        limit=args.limit,
        question_workers=args.question_workers,
        writer_workers=args.writer_workers,
    )
    if not args.allow_live:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    contract = adapter.assert_prompt_contract()
    _, manifest, selected_all = _load_inputs(contract)
    selected = selected_all[: args.limit]
    bindings = manifest["items"][: args.limit]
    legacy._load_live_credentials()
    release_clients, release_memory = legacy._load_release_runtime(contract)
    generation_type, retrieval_type = core._client_types(release_clients)
    config = _run_config(contract)
    core._assert_model_stack(release_clients.ModelStack.for_provider("gemini"), config)
    config_sha = core._canonical(config)
    output_root = args.output.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    config_path = output_root / "run_config.json"
    selection_path = output_root / "selection.json"
    existing = (output_root.exists() and any(output_root.iterdir())) or (
        cache_root.exists() and any(cache_root.iterdir())
    )
    if existing and not args.resume:
        raise SystemExit("LME400 requires fresh output/cache roots; use --resume only for this exact run")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    envelope = {"config": config, "config_sha256": config_sha}
    if config_path.exists() and core._load_json(config_path) != envelope:
        raise SystemExit("existing run_config differs from LME400 contract")
    if not config_path.exists():
        core._atomic_json(config_path, envelope)

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
    selection_envelope = {"selection": selection, "selection_sha256": core._canonical(selection)}
    if selection_path.exists():
        if core._load_json(selection_path) != selection_envelope:
            raise SystemExit("resume selection differs; use a fresh root instead of changing --limit")
    elif existing and args.resume:
        raise SystemExit("existing LME400 run has no sealed selection receipt")
    else:
        core._atomic_json(selection_path, selection_envelope)

    launch_receipt = core._sealed_payload(
        {
            "schema_version": f"{RUN_SCHEMA}.launch",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "resume": bool(args.resume),
            "question_workers": args.question_workers,
            "writer_workers": args.writer_workers,
            "worker_product": args.question_workers * args.writer_workers,
            "generation_semaphore": 8,
            "qwen_semaphore": 4,
            "selected_count": args.limit,
            "manifest_sha256": MANIFEST_SHA256,
            "config_sha256": config_sha,
            "selection_sha256": selection_envelope["selection_sha256"],
            "output_root": str(output_root),
            "cache_root": str(cache_root),
        }
    )
    core._atomic_json(
        output_root / "launches" / f"launch-{time.time_ns()}.json",
        launch_receipt,
    )

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
    if args.question_workers == 1:
        rows = [core._safe_process(**job) for job in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.question_workers) as executor:
            rows = list(executor.map(lambda job: core._safe_process(**job), jobs))

    rows_path = output_root / "rows.jsonl"
    rows_temporary = rows_path.with_suffix(".jsonl.tmp")
    rows_temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    rows_temporary.replace(rows_path)
    complete = [row for row in rows if row.get("status") == "complete"]
    timing_keys = sorted(
        {key for row in complete for key in row.get("timings_seconds", {})}
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
        "rows_sha256": core._sha_file(rows_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["summary_sha256"] = core._canonical(summary)
    core._atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

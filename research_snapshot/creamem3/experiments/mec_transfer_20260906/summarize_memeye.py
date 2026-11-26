"""Read-only paired downstream audit; --write creates only new analysis files.

No model/provider factory or selection is invoked. Gold is evaluator-only and
is opened only after both completed arms and the frozen selector bundle exist.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from creamem3.experiments.mec_transfer_20260906 import run_memeye as adapter

direct = adapter.direct
HISTORICAL_ROOT = Path(r"D:\creamem3_cache\memeye_v19\mec50_soft_bundle_v2_routing_only_gemini_r1_results")
TOKENS = ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens")


def read(path: Path) -> dict[str, Any]:
    return direct._read_object(path, "downstream audit input")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def seal(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    digest = body.pop(field, None)
    require(digest == direct._sha256(body), f"{field} seal drift")


def provider_audit(cache: Path) -> dict[str, Any]:
    """Bind every physical intent/response/plan/final; never print prompts/keys."""
    by_stage: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    parse_events: list[dict[str, Any]] = []
    first_times: list[float] = []
    final_by_request: dict[tuple[str, str], Any] = {}
    bad_markers = [str(path) for label in ("failures", "orphans")
                   for path in cache.rglob(f"{label}/**/*.json")]
    require(not bad_markers, f"failure/orphan receipts: {bad_markers}")
    for branch in (cache / "provider" / "predict" / "g25_reader", cache / "provider" / "score" / "g35_judge"):
        for stage_dir in sorted((branch / "attempts").iterdir()):
            if not stage_dir.is_dir():
                continue
            stage = stage_dir.name
            intents = list(stage_dir.glob("*/attempt_*_request.json"))
            responses = list(stage_dir.glob("*/attempt_*_response.json"))
            require(len(intents) == len(responses), f"unmatched durable attempts in {stage}")
            logical_keys: set[str] = set()
            status_counts: Counter[str] = Counter()
            usage = Counter({key: 0 for key in TOKENS})
            for intent_path in sorted(intents):
                first_times.append(intent_path.stat().st_mtime)
                intent = read(intent_path)
                plan = read(branch / "plans" / stage / f"{intent['resume_hash']}.json")
                require(intent["plan_sha256"] == direct._sha256(plan), "intent plan SHA drift")
                identity = {"resume_hash": intent["resume_hash"], "attempt_ordinal": intent["attempt_ordinal"],
                            "plan_sha256": intent["plan_sha256"]}
                require(intent["attempt_identity_sha256"] == direct._sha256(identity), "attempt identity drift")
                response_path = intent_path.with_name(intent_path.name.replace("_request.json", "_response.json"))
                response = read(response_path)
                for field in ("stage", "model", "protocol_sha256", "request_sha256", "resume_hash", "attempt_ordinal", "attempt_identity_sha256"):
                    require(response[field] == intent[field], f"attempt response {field} drift")
                metadata = plan["cache_metadata"]
                logical_key = str(metadata.get("row_id") or metadata.get("exact_triple_sha256") or plan["prompt_sha256"])
                logical_keys.add(logical_key)
                outcome = response["outcome_kind"]
                status_counts[outcome] += 1
                final_path = branch / "responses" / stage / f"{intent['resume_hash']}.json"
                final = read(final_path) if final_path.exists() else None
                if outcome == "raw_response":
                    require(response["response_sha256"] == direct._sha256(response["text"]), "raw response text SHA drift")
                    for key in TOKENS:
                        usage[key] += int(response.get("usage", {}).get(key, 0))
                    require(final is not None, "raw response lacks final cache")
                    require(final["response_sha256"] == response["response_sha256"] and final["text"] == response["text"], "final/raw body drift")
                    require(Path(final["cache_path"]).resolve() == final_path.resolve(), "final cache path drift")
                    require(final["attempt_ordinal"] == intent["attempt_ordinal"], "final ordinal drift")
                    require(final["request_sha256"] == intent["request_sha256"] and final["resume_hash"] == intent["resume_hash"], "final request binding drift")
                    final_by_request[(stage, intent["request_sha256"])] = final
                    if stage in {"focused", "structured_fused"}:
                        parsed = direct.predict._parse_json_object(final["text"])
                        invalid = not bool(parsed)
                        missing_answer = stage == "structured_fused" and not bool(parsed.get("final_answer"))
                        if invalid or missing_answer or parsed.get("_failed"):
                            parse_events.append({"row_id": metadata.get("row_id"), "stage": stage,
                                "invalid_or_empty_json_object": invalid, "missing_structured_final_answer": missing_answer,
                                "explicit_failed_marker": bool(parsed.get("_failed")), "response_path": str(final_path),
                                "output_tokens": final.get("usage", {}).get("output_tokens")})
                else:
                    require(outcome == "explicit_provider_error", f"unknown physical outcome {outcome}")
                    recovered = final is not None and int(final["attempt_ordinal"]) > int(intent["attempt_ordinal"])
                    event = {"row_id": metadata.get("row_id"), "stage": stage, "ordinal": intent["attempt_ordinal"],
                             "retryable": response.get("retryable"), "status_code": response.get("status_code"),
                             "error_message": response.get("error_message"), "recovered": recovered,
                             "request_path": str(intent_path), "response_path": str(response_path)}
                    errors.append(event)
                    require(response.get("retryable") is True and recovered, f"unrecovered/nonretryable provider error: {event}")
            by_stage[stage] = {"logical_calls": len(logical_keys), "physical_attempts": len(intents),
                               "extra_physical_attempts": len(intents) - len(logical_keys),
                               "outcomes": dict(status_counts), "usage": dict(usage)}
    totals = {key: sum(stage["usage"][key] for stage in by_stage.values()) for key in TOKENS}
    return {"status": "PASS", "by_stage": by_stage, "usage": totals,
            "logical_calls": sum(stage["logical_calls"] for stage in by_stage.values()),
            "physical_attempts": sum(stage["physical_attempts"] for stage in by_stage.values()),
            "retries": errors, "recovered_retries": sum(event["recovered"] for event in errors),
            "failure_orphan_receipts": bad_markers, "parse_events": parse_events,
            "first_request_utc": datetime.fromtimestamp(min(first_times), timezone.utc).isoformat(),
            "first_request_epoch": min(first_times), "_final_by_request": final_by_request}


def arm_audit(scope: Path, arm: str, gold: Mapping[str, Any]) -> dict[str, Any]:
    root = scope / arm / "results"
    cache = scope / arm / "provider_cache"
    gate = adapter.raw.preflight(selection_path=scope / "manifests" / f"{arm}.json",
                                 output_root=root, cache_root=cache, workers=4)
    prediction_complete, predictions = direct._load_prediction_complete(root, gate["contract"], gate["prepared"])
    score_complete = read(root / "score" / "complete.json")
    seal(score_complete, "complete_sha256")
    require(score_complete["complete"] is True and score_complete["contract_sha256"] == gate["contract"]["contract_sha256"], "score contract drift")
    require(score_complete["rows_sha256"] == direct._sha256(score_complete["rows"]), "score receipt list drift")
    require(len(predictions) == len(score_complete["rows"]) == 24, "completed arm row census mismatch")
    provider = provider_audit(cache)
    finals = provider.pop("_final_by_request")
    scored = {}
    for prediction, receipt in zip(predictions, score_complete["rows"], strict=True):
        row_id = prediction["row_id"]
        path = direct._score_path(root, prediction)
        require(receipt["row_id"] == row_id and Path(receipt["path"]).resolve() == path.resolve(), "score receipt path/ID drift")
        require(receipt["file_sha256"] == direct._file_sha256(path), "score row file SHA drift")
        row = direct._load_score_row(path, prediction, gold[row_id])
        require(receipt["row_sha256"] == row["row_sha256"], "score receipt row SHA drift")
        require(row["exact_triple_sha256"] == direct._sha256([gold[row_id]["question"], gold[row_id]["gold"], prediction["prediction"]]), "judge exact triple drift")
        final_path = Path(row["cache_path"])
        require(row["cache_file_sha256"] == direct._file_sha256(final_path), "judge cache file SHA drift")
        branch = cache / "provider" / "score" / "g35_judge"
        for key, linked in (("plan_file_sha256", branch / "plans" / "judge" / f"{row['resume_hash']}.json"),
                            ("attempt_request_file_sha256", branch / "attempts" / "judge" / row["resume_hash"] / f"attempt_{row['successful_attempt_ordinal']:03d}_request.json"),
                            ("attempt_response_file_sha256", branch / "attempts" / "judge" / row["resume_hash"] / f"attempt_{row['successful_attempt_ordinal']:03d}_response.json")):
            require(row[key] == direct._file_sha256(linked), f"judge {key} drift")
        for stage in prediction["reader_stages"]:
            require(finals[(stage["stage"], stage["request_sha256"])]["response_sha256"] == stage["response_sha256"], "prediction/reader cache binding drift")
        require(finals[("judge", row["request_sha256"])]["response_sha256"] == row["response_sha256"], "score/judge cache binding drift")
        scored[row_id] = {"row_id": row_id, "question": gold[row_id]["question"], "prediction": prediction["prediction"],
            "mapped_score": float(row["mapped_score"]), "raw_score": float(row["raw_score"]),
            "x": row["x"], "y": row["y"], "reasoning": row["reasoning"],
            "selected_fact_ids": prediction["selected_fact_ids"], "context_sha256": prediction["context_sha256"],
            "context_chars": prediction["context_stats"]["context_chars"], "prediction_path": str(direct._prediction_path(root, {"query": prediction})),
            "score_path": str(path)}
    mapped = sum(row["mapped_score"] for row in scored.values())
    raw_score = sum(row["raw_score"] for row in scored.values())
    require(math.isclose(mapped / 24, score_complete["mapped_mean"], abs_tol=1e-12), "summary mapped mean drift")
    require(math.isclose(raw_score / 24, score_complete["raw_mean"], abs_tol=1e-12), "summary raw mean drift")
    require(score_complete["qwen_provider_calls"] == 0, "Qwen calls found")
    for phase_complete, key in ((prediction_complete, "reader_usage"), (score_complete, "judge_usage")):
        for stage, receipt in phase_complete[key]["stages"].items():
            actual = provider["by_stage"].get(stage, {"physical_attempts": 0, "usage": {key: 0 for key in TOKENS}})
            require(actual["physical_attempts"] == receipt["provider_calls"], "physical attempt usage drift")
            require(actual["usage"] == receipt["usage"], "durable token usage drift")
    finished = (root / "score" / "complete.json").stat().st_mtime
    return {"output_root": str(root), "cache_root": str(cache), "rows": 24, "complete": 24,
            "pipeline_errors": 0, "mapped_sum": mapped, "mapped_percent": mapped / 24 * 100,
            "raw_mean": raw_score / 24, "mapped_distribution": dict(Counter(str(row["mapped_score"]) for row in scored.values())),
            "context_chars_mean": statistics.fmean(row["context_chars"] for row in scored.values()),
            "provider": provider, "finished_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
            "elapsed_from_first_request_seconds": finished - provider["first_request_epoch"],
            "score_complete_file_sha256": direct._file_sha256(root / "score" / "complete.json"),
            "prediction_complete_file_sha256": direct._file_sha256(root / "predict" / "complete.json"),
            "row_data": scored}


def summarize(root: Path) -> dict[str, Any]:
    scope = adapter._scope_root(root, False)
    selection_complete = adapter.validate_bundle(scope)
    # Explicit completion barrier before touching evaluator-only source files.
    for arm in adapter.LIVE_ARMS:
        require((scope / arm / "results" / "score" / "complete.json").is_file(), "both arm scores must complete before final analysis")
    gold = direct._gold_rows()
    arms = {arm: arm_audit(scope, arm, gold) for arm in adapter.LIVE_ARMS}
    dense, new = (arms[arm]["row_data"] for arm in adapter.LIVE_ARMS)
    require(set(dense) == set(new), "paired question IDs differ")
    pairs = []
    for row_id in sorted(dense):
        d, n = dense[row_id], new[row_id]
        delta = n["mapped_score"] - d["mapped_score"]
        same = n["prediction"] == d["prediction"]
        pairs.append({"row_id": row_id, "question": d["question"], "x": d["x"], "y": d["y"],
            "dense_mapped": d["mapped_score"], "new_mapped": n["mapped_score"], "delta": delta,
            "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
            "same_prediction": same, "same_prediction_judge_score_drift": same and delta != 0,
            "dense_prediction": d["prediction"], "new_prediction": n["prediction"],
            "dense_reasoning": d["reasoning"], "new_reasoning": n["reasoning"],
            "selected_overlap": len(set(d["selected_fact_ids"]) & set(n["selected_fact_ids"])),
            "dense_chars": d["context_chars"], "new_chars": n["context_chars"]})
    outcomes = Counter(row["outcome"] for row in pairs)
    historical_manifest = read(scope / "manifests" / "v2_historical.json")
    hist_rows = []
    historical_complete = read(HISTORICAL_ROOT / "score" / "complete.json")
    seal(historical_complete, "complete_sha256")
    historical_receipts = {row["row_id"]: row for row in historical_complete["rows"]}
    for manifest_row in historical_manifest["rows"]:
        row_id = manifest_row["row_id"]
        receipt = historical_receipts[row_id]
        score_path = Path(receipt["path"])
        require(receipt["file_sha256"] == direct._file_sha256(score_path), "historical score file drift")
        pred = read(HISTORICAL_ROOT / "predict" / "rows" / score_path.name)
        seal(pred, "row_sha256")
        historical = direct._load_score_row(score_path, pred, gold[row_id])
        require(pred["selected_fact_ids"] == manifest_row["selected_fact_ids"], "historical V2 selection mismatch")
        require(pred["context_sha256"] == manifest_row["trace"]["render"]["context_sha256"], "historical V2 raw context mismatch")
        hist_rows.append({"row_id": row_id, "mapped_score": historical["mapped_score"],
                          "context_chars": pred["context_stats"]["context_chars"], "score_path": str(score_path)})
    return {"schema_version": "creamem3.mec_transfer_memeye_downstream_audit.v1", "status": "PASS",
            "scope": str(scope), "summary_script_sha256": direct._file_sha256(Path(__file__)),
            "selector_complete_sha256": selection_complete["complete_sha256"],
            "analysis_only_no_model_calls": True, "selection_or_parameters_changed": False,
            "score_meaning": "MemEye mapped graded score; not binary accuracy",
            "arms": arms, "paired": {"n": len(pairs), "new_wins": outcomes["win"], "new_losses": outcomes["loss"],
                "ties": outcomes["tie"], "delta_mapped_points": sum(row["delta"] for row in pairs),
                "delta_pp": statistics.fmean(row["delta"] for row in pairs) * 100,
                "mean_selected_overlap": statistics.fmean(row["selected_overlap"] for row in pairs),
                "same_prediction_count": sum(row["same_prediction"] for row in pairs),
                "same_prediction_judge_drift_count": sum(row["same_prediction_judge_score_drift"] for row in pairs),
                "same_prediction_judge_drift_net_points": sum(row["delta"] for row in pairs if row["same_prediction_judge_score_drift"]),
                "rows": pairs},
            "historical_v2_only_not_main_paired_arm": {"root": str(HISTORICAL_ROOT), "n": len(hist_rows),
                "mapped_sum": sum(float(row["mapped_score"]) for row in hist_rows),
                "mapped_percent": statistics.fmean(float(row["mapped_score"]) for row in hist_rows) * 100,
                "budget_violations_against_new_Bq": sum(row["v2_over_budget"] for row in selection_complete["rows_diagnostic"]),
                "warning": "historical different run; frozen V2 often exceeds matched character budget; not evidence of paired treatment effect",
                "rows": hist_rows},
            "interpretation": "24-case predeclared engineering pilot; do not tune on these outcomes or claim confirmed improvement"}


def markdown(result: Mapping[str, Any]) -> str:
    paired = result["paired"]
    lines = ["# MemEye fixed 24-question pilot", "", "Frozen selection; no Qwen reranker; downstream read-only analysis.", "",
             "| Arm | Mapped score sum / 24 | Mapped % | Mean context chars | Physical calls | Retries |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, row in result["arms"].items():
        lines.append(f"| {arm} | {row['mapped_sum']:.1f}/24 | {row['mapped_percent']:.2f} | {row['context_chars_mean']:.2f} | {row['provider']['physical_attempts']} | {len(row['provider']['retries'])} |")
    lines += ["", "Mapped score is graded, not binary accuracy.", "",
              f"New versus Dense: {paired['new_wins']} wins, {paired['new_losses']} losses, {paired['ties']} ties; {paired['delta_pp']:+.2f} pp.",
              f"Same predictions: {paired['same_prediction_count']}; different Judge scores on identical predictions: {paired['same_prediction_judge_drift_count']} (net {paired['same_prediction_judge_drift_net_points']:+.1f} mapped points).", "",
              "## Changed cases", "", "| Row ID | Dense | New | Delta |", "|---|---:|---:|---:|"]
    for row in paired["rows"]:
        if row["delta"]:
            lines.append(f"| {row['row_id']} | {row['dense_mapped']:.1f} | {row['new_mapped']:.1f} | {row['delta']:+.1f} |")
    lines += ["", "## Process audit", ""]
    for arm, row in result["arms"].items():
        lines.append(f"- {arm}: {row['complete']}/24 complete; pipeline errors {row['pipeline_errors']}; parse events {len(row['provider']['parse_events'])}; recovered transport retries {row['provider']['recovered_retries']}; tokens {row['provider']['usage']}; first request to completion {row['elapsed_from_first_request_seconds']:.1f}s.")
    hist = result["historical_v2_only_not_main_paired_arm"]
    lines += ["", "## Historical V2 diagnostic only", "", f"Same 24 IDs: {hist['mapped_sum']:.1f}/24, {hist['mapped_percent']:.2f}%. {hist['budget_violations_against_new_Bq']}/24 exceed the new per-question character budget. Different historical run: not a same-budget paired treatment estimate.", "",
              "Input/code/graph/manifest/result/physical request-response-cache bindings passed. No model calls or selection changes were made by this audit. The 24-case pilot does not establish a stable general improvement.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=adapter.DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = summarize(args.root)
    if args.write:
        target = Path(result["scope"]) / "analysis"
        direct._write_once(target / "paired_metrics.json", result)
        report = target / "paired_report.md"
        payload = markdown(result)
        if report.exists():
            require(report.read_text(encoding="utf-8") == payload, "refusing to overwrite different analysis report")
        else:
            with report.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
    public = {"status": result["status"], "scope": result["scope"],
              "arms": {arm: {key: value for key, value in row.items() if key not in {"row_data", "provider"}}
                       for arm, row in result["arms"].items()},
              "paired": {key: value for key, value in result["paired"].items() if key != "rows"},
              "changed_cases": [row for row in result["paired"]["rows"] if row["delta"]],
              "historical_v2": {key: value for key, value in result["historical_v2_only_not_main_paired_arm"].items() if key != "rows"}}
    print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

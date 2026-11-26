"""Optional, separately registered recovery of exactly two MAX_TOKENS Readers.

No calls on import, draft or prepare. Live requires a user-authorization record.
The frozen caption_control.py, its registration and terminal outputs are untouched.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import time
from filelock import FileLock
import caption_control as c

ROOT = c.DEFAULT_ROOT
RECOVERY = ROOT / "focused5200_recovery_r1"
HERE = Path(__file__).resolve()
TARGETS = {
    "Card_Playlog_Test::Q46": "b99c631b1217ade2c4654a248402ba75de1b58de36d8c3952b7ad4fcdd787b70",
    "Home_Renovation_Interior_Design::Q18": "7f2c23309bb8eaeb500efb869fa0f17e63e36e1cf9abcb2b7f4e4ff725cab4e8",
}


def target_evidence(reg, rid, request_hash):
    task = rid.split("::", 1)[0]
    arm_root = ROOT / "tasks" / task / "arms/original47"
    key = c.b.sha(rid)
    context = arm_root / "contexts/original_reader" / (key + ".json")
    directory = arm_root / "calls" / request_hash[:2] / request_hash
    manifest_path = directory / "request.json.gz"
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    raw_path = directory / "attempt_001_provider.json"
    response = c.b.read(raw_path)["response"]
    outcome_path = directory / "attempt_001_outcome.json"
    outcome = c.b.read(outcome_path)
    item = c.b.read(context)
    c.require(manifest["stage"] == "focused" and manifest["model"] == "gemini-2.5-flash-lite", "Wrong failed stage/model")
    c.require(manifest["metadata"]["row_id"] == rid and item["query"]["row_id"] == rid, "Wrong row binding")
    c.require(manifest["metadata"]["context_sha256"] == item["context_sha256"], "Failed context changed")
    c.require(manifest["metadata"]["registration_sha256"] == reg["seal_sha256"], "Original registration drift")
    c.require(manifest["generation_config"]["maxOutputTokens"] == 2600, "Original cap changed")
    c.require(len(response.get("candidates", [])) == 1 and response["candidates"][0]["finishReason"] == "MAX_TOKENS", "Not an explicit truncation")
    c.require(not outcome["ok"] and not outcome["retryable"] and not (directory / "response.json").exists(), "Unexpected old call state")
    return {"row_id": rid, "task": task, "arm_root": str(arm_root), "context_path": str(context),
            "failed_request_path": str(manifest_path), "failed_request_hash": request_hash,
            "failed_provider_path": str(raw_path), "original_cap": 2600, "recovery_cap": 5200,
            "protected_files": {str(p): c.b.file_sha(p) for p in (context, manifest_path, raw_path, outcome_path)}}


def draft():
    reg = c.validate(ROOT)
    return {"schema": "opctx.public_caption.focused_cap_recovery.20261111.v1",
        "original_registration_sha256": reg["seal_sha256"],
        "original_registration_file_sha256": c.b.file_sha(ROOT / "registration.json"),
        "recovery_code_sha256": c.b.file_sha(HERE),
        "targets": {rid: target_evidence(reg, rid, req) for rid, req in TARGETS.items()},
        "scope": "only two incomplete original47 focused Reader requests; cap2600->5200; one physical attempt each",
        "model_prompt_input_order_algorithm_unchanged": True,
        "new_namespace": str(RECOVERY / "calls"), "missing_final_qa_uses_original500_cap": True,
        "old_failed_outputs_preserved": True, "no_scoring_or_gold_read": True}


def prepare(authorization):
    c.require(bool(authorization.strip()), "Explicit user-authorization reference required")
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RECOVERY / "registration.lock"), timeout=1):
        p = RECOVERY / "registration.json"
        if p.exists():
            return validate()
        value = draft()
        value["authorization"] = authorization
        value["prepared_at"] = time.time()
        value["protected_predictions"] = {str(p): c.b.file_sha(p) for p in ROOT.glob("tasks/*/arms/*/predictions/*/*.json")}
        c.require(len(value["protected_predictions"]) == 740, "Expected exactly740 existing successes before this recovery")
        for rid, info in value["targets"].items():
            c.require(not (Path(info["arm_root"]) / "predictions/original_reader" / (c.b.sha(rid) + ".json")).exists(), "Target already completed")
        return c.b.save(p, value)


def validate():
    original = c.validate(ROOT)
    reg = c.b.read(RECOVERY / "registration.json")
    c.require(reg["recovery_code_sha256"] == c.b.file_sha(HERE), "Recovery code changed")
    c.require(reg["original_registration_sha256"] == original["seal_sha256"], "Original experiment changed")
    c.require(reg["original_registration_file_sha256"] == c.b.file_sha(ROOT / "registration.json"), "Original registry changed")
    for info in reg["targets"].values():
        for path, digest in info["protected_files"].items():
            c.require(c.b.file_sha(path) == digest, "Original failed evidence/context changed")
    for path, digest in reg["protected_predictions"].items():
        c.require(c.b.file_sha(path) == digest, "Existing successful answer changed")
    return reg


class RecoveryRuntime:
    def __init__(self, info, recovery, allow):
        self.info, self.recovery = info, recovery
        self.model = "gemini-2.5-flash-lite"
        self.original = c.LoggedRuntime(Path(info["arm_root"]), self.model, allow, answer_cap=500)
        self.focused = c.LoggedRuntime(RECOVERY / "calls" / c.b.sha(info["row_id"]), self.model, allow)

    def generate(self, stage, prompt, **kwargs):
        if stage != "focused":
            return self.original.generate(stage, prompt, **kwargs)
        manifest = json.loads(gzip.decompress(Path(self.info["failed_request_path"]).read_bytes()))
        c.require(kwargs.get("cache_metadata") == manifest["metadata"], "Recovery request metadata changed")
        c.require(prompt == manifest["prompt"] and kwargs.get("system") == manifest["system"], "Recovery prompt changed")
        c.require(kwargs.get("max_output_tokens") == 2600 and kwargs.get("json_mode") is True, "Unexpected Reader request")
        config = c.rt.config(self.model, stage, 5200, True)
        reference = dict(manifest["generation_config"], maxOutputTokens=5200)
        c.require(config == reference, "Recovery changes more than the output cap")
        kwargs["max_output_tokens"] = 5200
        kwargs["max_provider_attempts"] = 1
        return self.focused.generate(stage, prompt, **kwargs)


def live():
    recovery = validate()
    original = c.validate(ROOT)
    c.rt.prior.disk_guard(ROOT)
    with FileLock(str(ROOT / "dispatcher.lock"), timeout=1):
        failures = []
        for rid, info in recovery["targets"].items():
            tr = ROOT / "tasks" / info["task"]
            with FileLock(str(tr / "task.lock"), timeout=1):
                target = Path(info["arm_root"]) / "predictions/original_reader" / (c.b.sha(rid) + ".json")
                receipt = RECOVERY / "completed" / (c.b.sha(rid) + ".json")
                if receipt.exists():
                    saved = c.b.read(receipt)
                    c.require(c.b.file_sha(target) == saved["prediction_file_sha256"], "Recovered answer changed")
                    continue
                c.require(not target.exists(), "Unreceipted new prediction requires audit")
                item = c.b.read(info["context_path"])
                runtime = RecoveryRuntime(info, recovery, True)
                try:
                    pred = c.old.predict(Path(info["arm_root"]), "original_reader", item, runtime, original)
                except Exception as exc:
                    failures.append(rid)
                    c.b.save(RECOVERY / "errors" / f"{time.time_ns()}.json", dict(row_id=rid,
                        error_type=type(exc).__name__, time=time.time(), recovery_registration_sha256=recovery["seal_sha256"]))
                    print("RECOVERY FAILED", rid, type(exc).__name__, flush=True)
                    continue
                c.b.save(receipt, dict(row_id=rid, prediction_file_sha256=c.b.file_sha(target),
                    prediction_seal_sha256=pred["seal_sha256"], recovery_registration_sha256=recovery["seal_sha256"],
                    original_failed_request_hash=info["failed_request_hash"], focused_output_cap=5200))
                print("RECOVERED", rid, flush=True)
        validate()
        c.require(not failures, "Unresolved bounded recovery rows: " + ", ".join(failures))
        c.require_all_predictions(ROOT, original)
        c.b.save(RECOVERY / "complete.json", dict(n=2, original_prediction_count=371, enhanced_prediction_count=371,
            recovery_registration_sha256=recovery["seal_sha256"], no_scores_read=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("draft", "prepare", "live", "status"))
    p.add_argument("--authorization", default="")
    p.add_argument("--allow-model-calls", action="store_true")
    args = p.parse_args()
    c.require((args.phase == "live") == args.allow_model_calls, "Only live requires explicit model-call flag")
    if args.phase == "draft":
        print(json.dumps(draft(), ensure_ascii=False, indent=2))
    elif args.phase == "prepare":
        print(json.dumps({"registered": prepare(args.authorization)["seal_sha256"]}))
    elif args.phase == "status":
        reg = validate()
        print(json.dumps({"registered": reg["seal_sha256"], "completed": len(list((RECOVERY / "completed").glob("*.json")))}, indent=2))
    else:
        live()


if __name__ == "__main__":
    main()

"""Score frozen enhanced public-caption predictions with the registered Lite Judge.

The Flash endpoint returned a different served model. Those outputs remain
unaccepted and untouched. This entry continues only the already-registered
3.5 Flash-Lite scoring branch; generation and successful scores are immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import threading
import time
from filelock import FileLock
import caption_control as c

ROOT = c.DEFAULT_ROOT
RECOVERY = ROOT / "enhanced_lite_score_resume_r1"
HERE = Path(__file__).resolve()
ARM, JUDGE, READER = "enhanced48", "g35lite", "enhanced_reader"


def prepare():
    original = c.validate(ROOT)
    c.require_all_predictions(ROOT, original)
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RECOVERY / "registration.lock"), timeout=1):
        if (RECOVERY / "registration.json").exists():
            return validate()
        predictions = list(ROOT.glob("tasks/*/arms/*/predictions/*/*.json"))
        original_scores = list(ROOT.glob("tasks/*/arms/original47/scores/g35lite/original_reader/*.json"))
        rejected = list(ROOT.glob("tasks/*/arms/enhanced48/judges/g35flash/calls/*/*/attempt_001_provider.json"))
        c.require(len(predictions) == 742 and len(original_scores) == 371 and len(rejected) == 371,
                  "Expected complete frozen generation/original scoring and 371 rejected Flash responses")
        evidence = []
        for path in rejected:
            raw = c.b.read(path)["response"]
            c.require(raw.get("modelVersion") == "gemini-3.6-flash", "Unexpected served Flash model")
            c.require(raw["candidates"][0]["finishReason"] == "STOP", "Additional Flash failure requires review")
            evidence.extend([path, path.with_name("attempt_001_outcome.json"), path.with_name("request.json.gz")])
        c.require(not list(ROOT.glob("tasks/*/arms/enhanced48/scores/*/*/*.json")),
                  "Enhanced scores already exist before this recovery registration")
        return c.b.save(RECOVERY / "registration.json", {
            "schema": "opctx.public_caption.enhanced_lite_score_resume.20261111.v1",
            "created_at": time.time(), "code_sha256": c.b.file_sha(HERE),
            "original_registration_sha256": original["seal_sha256"],
            "original_registration_file_sha256": c.b.file_sha(ROOT / "registration.json"),
            "scope": "Only registered enhanced48/g35lite score branch; 371 frozen predictions; no generation or Flash calls",
            "model": c.rt.JUDGES[JUDGE], "judge_prompt_and_mapping": "unchanged original Runtime.judge",
            "output_cap": 400, "workers": 4, "explicit_transient_attempt_cap": 3,
            "failure_policy": "stop pending jobs after any failure; preserve all raw responses; no automatic restart",
            "wrong_model_response_policy": "retain rejected gemini-3.6-flash responses; do not accept or score them",
            "protected_files": {str(p): c.b.file_sha(p) for p in predictions + original_scores + evidence},
        })


def validate():
    original = c.validate(ROOT)
    reg = c.b.read(RECOVERY / "registration.json")
    c.require(reg["code_sha256"] == c.b.file_sha(HERE), "Recovery source changed")
    c.require(reg["original_registration_sha256"] == original["seal_sha256"], "Original registration changed")
    c.require(reg["original_registration_file_sha256"] == c.b.file_sha(ROOT / "registration.json"), "Registry file changed")
    for path, sha in reg["protected_files"].items():
        c.require(c.b.file_sha(path) == sha, "Protected prediction/score/failure changed: " + path)
    c.require_all_predictions(ROOT, original)
    return reg


def live():
    registration = validate()
    original = c.validate(ROOT)
    c.rt.prior.disk_guard(ROOT)
    with FileLock(str(ROOT / "dispatcher.lock"), timeout=1):
        # No gold is read until both generation arms are complete and validated.
        gold_rows = c.b.read(c.SOURCE / "scoring/open.json")["rows"]
        stop = threading.Event()

        def score(q):
            if stop.is_set():
                return {"row_id": q["row_id"], "status": "not_attempted_after_stop"}
            task, key = q["task"], c.b.sha(q["row_id"])
            ar = ROOT / "tasks" / task / "arms" / ARM
            score_path = ar / "scores" / JUDGE / READER / (key + ".json")
            pred = c.b.read(c.prediction_paths(ROOT, task, ARM) / (key + ".json"))
            if score_path.exists():
                saved = c.b.read(score_path)
                c.require(saved["prediction_sha256"] == pred["seal_sha256"], "Previously scored answer changed")
                c.require(saved["judge"] == c.rt.JUDGES[JUDGE], "Existing Judge differs")
                return {"row_id": q["row_id"], "status": "already_complete"}
            try:
                gold = gold_rows[q["row_id"]]
                c.require(gold["question"] == q["question"], "Question/gold binding changed")
                judge = c.LoggedRuntime(RECOVERY / "tasks" / task / JUDGE, c.rt.JUDGES[JUDGE], True)
                out, verdict = judge.judge(q["question"], gold["gold"], pred["prediction"],
                    dict(arm=ARM, row_id=q["row_id"], prediction_sha256=pred["seal_sha256"]))
                c.b.save(score_path, dict(row_id=q["row_id"], task=task, arm=ARM, judge=judge.model,
                    x=gold["x"], y=gold["y"], xy=gold["x"] + "-" + gold["y"],
                    raw_score=verdict.raw_score, mapped_score=verdict.mapped_score, reasoning=verdict.reasoning,
                    generation=asdict(out), prediction_sha256=pred["seal_sha256"],
                    registration_sha256=original["seal_sha256"], recovery_registration_sha256=registration["seal_sha256"]))
                print("SCORED", ARM, JUDGE, q["row_id"], flush=True)
                return {"row_id": q["row_id"], "status": "complete"}
            except BaseException as exc:
                stop.set()
                error = dict(row_id=q["row_id"], status="failed", error_type=type(exc).__name__, time=time.time())
                c.b.save(RECOVERY / "errors" / (str(time.time_ns()) + ".json"), error)
                return error

        with ThreadPoolExecutor(max_workers=registration["workers"]) as pool:
            futures = [pool.submit(score, q) for q in original["questions"]]
            results = [f.result() for f in as_completed(futures)]
        validate()
        report = c.status(ROOT, original)
        final = dict(time=time.time(), jobs=results, results=report["results"],
                     recovery_registration_sha256=registration["seal_sha256"])
        c.b.save(RECOVERY / "reports" / (str(time.time_ns()) + ".json"), final)
        c.require(report["results"][ARM][JUDGE]["n"] == 371, "Enhanced Lite scoring incomplete; inspect preserved errors")
        print(json.dumps(report["results"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "status", "live"))
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    c.require((args.phase == "live") == args.allow_model_calls, "Only live requires explicit call flag")
    if args.phase == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif args.phase == "status":
        validate()
        print(json.dumps(c.status(ROOT, c.validate(ROOT))["results"], ensure_ascii=False, indent=2))
    else:
        live()


if __name__ == "__main__":
    main()

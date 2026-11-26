"""Record the user's Lite-only scoring decision; no model calls or raw edits."""
from pathlib import Path
from datetime import datetime, timezone
import json
from filelock import FileLock
import caption_control as c

PATH = c.DEFAULT_ROOT / "user_dispositions" / "20261111_lite_only.json"
QUOTE = "先只保留3.5 Flash-Lite评分"


def main():
    original = c.validate(c.DEFAULT_ROOT)
    report = c.status(c.DEFAULT_ROOT, original)
    assert report["results"]["original47"]["g35lite"]["n"] == 371
    assert report["results"]["enhanced48"]["g35lite"]["n"] == 371
    assert report["results"]["enhanced48"]["g35flash"]["n"] == 0
    with FileLock(str(c.DEFAULT_ROOT / "dispatcher.lock"), timeout=1):
        rejected = list(c.DEFAULT_ROOT.glob("tasks/*/arms/enhanced48/judges/g35flash/calls/*/*/attempt_001_provider.json"))
        assert len(rejected) == 371
        protected = []
        for path in rejected:
            raw = c.b.read(path)["response"]
            assert raw["modelVersion"] == "gemini-3.6-flash"
            protected.extend([path, path.with_name("attempt_001_outcome.json"), path.with_name("request.json.gz")])
        if PATH.exists():
            value = c.b.read(PATH)
            assert value["user_quote"] == QUOTE
            for path, sha in value["protected_rejected_artifacts"].items():
                assert c.b.file_sha(path) == sha
        else:
            value = c.b.save(PATH, dict(
                schema="opctx.public_caption.user_disposition.20261111.v1",
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                user_quote=QUOTE,
                original_registration_sha256=original["seal_sha256"],
                original_registration_file_sha256=c.b.file_sha(c.DEFAULT_ROOT / "registration.json"),
                decision="Retain only gemini-3.5-flash-lite Judge results; cancel the Flash Judge scoring branch for this control.",
                reason="Requests for gemini-3.5-flash returned served model gemini-3.6-flash, which was rejected by the frozen validation rule.",
                rejected_response_count=371,
                flash_branch_status="excluded_by_user_decision_not_pending",
                retained_branches={"original47/g35lite": 371, "enhanced48/g35lite": 371},
                retained_scoring_plan_complete=True,
                no_new_api_calls=True, no_generation_changes=True,
                no_score_zero_filling=True, original_registration_unchanged=True,
                protected_rejected_artifacts={str(p): c.b.file_sha(p) for p in protected},
                raw_retention="Keep all rejected requests/responses/outcomes; do not delete, relabel as3.5, or adopt their scores.",
            ))
    print(json.dumps({k: value[k] for k in ("seal_sha256", "user_quote", "flash_branch_status", "retained_scoring_plan_complete")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

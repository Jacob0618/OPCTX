"""Recovery transport; prior successful receipts are read-only, never rewritten."""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from creamem3.experiments.locomo_g35lite_20260908 import run as baseline
from creamem3.experiments.locomo_g35lite_20260908.runtime import *

ORIGINAL = baseline.RUN_ROOT
RESUMED = Path(r"D:\creamem3_cache\locomo_g35lite_full1540_20260908_r2")
OFFLINE = Path(r"D:\creamem3_cache\locomo_g35lite_offline6_full1540_20260908_r1")


def inspect_prior(root):
    """Refuse recovery through nonretryable failures, orphan calls or corrupt receipts."""
    manifest = read(root / "manifest.json")
    if manifest["source_hashes"] != baseline.source_manifest():
        raise StopRun("Historical experiment source drift")
    report = {"root": str(root), "intents": 0, "responses": 0, "errors": 0,
              "completed_requests": 0, "exhausted_retryable_requests": []}
    for directory in (root / "provider").glob("*/*"):
        identity = read(directory / "request.json")
        if sha(identity) != directory.name:
            raise StopRun("Prior request hash drift")
        error_count = 0
        for intent in directory.glob("*.intent.json"):
            record = read(intent)
            if record["request_sha256"] != directory.name:
                raise StopRun("Prior intent request mismatch")
            stem = intent.name.replace(".intent.json", "")
            response = directory / (stem + ".response.json")
            error = directory / (stem + ".error.json")
            report["intents"] += 1
            if response.exists() == error.exists():
                raise StopRun(f"Prior orphan or duplicate outcome: {intent}")
            if response.exists():
                read(response)
                report["responses"] += 1
            else:
                value = read(error)
                if not value["retryable"]:
                    raise StopRun(f"Nonretryable prior error: {error}")
                report["errors"] += 1
                error_count += 1
        if (directory / "final.json").exists():
            final = read(directory / "final.json")
            response = read(directory / f"attempt_{final['attempt']:02d}.response.json")
            if final["identity"] != identity or final["response_sha256"] != sha(response):
                raise StopRun("Prior final binding mismatch")
            report["completed_requests"] += 1
        elif error_count == 4:
            report["exhausted_retryable_requests"].append({"stage": identity["stage"],
                                                          "scope": identity["scope"],
                                                          "request_sha256": directory.name})
    return report


class ReuseRuntime(Runtime):
    def __init__(self, root, *, sources=(), concurrency=8, allow_model_calls=False):
        super().__init__(root, allow_model_calls=allow_model_calls, concurrency=concurrency)
        self.sources = tuple(Path(p) for p in sources)

    def post(self, stage, scope, model, method, body):
        identity = {"stage": stage, "scope": scope, "model": model, "method": method, "body": body}
        key = sha(identity)
        if not (self.root / "provider" / stage / key / "final.json").exists():
            for source in self.sources:
                directory = source / "provider" / stage / key
                final_path = directory / "final.json"
                if not final_path.exists():
                    continue
                final = read(final_path)
                response = read(directory / f"attempt_{final['attempt']:02d}.response.json")
                if final["request_sha256"] != key or final["identity"] != identity or sha(response) != final["response_sha256"]:
                    self._halt("Source cache contract drift")
                seal(self.root / "cache_links" / stage / f"{key}.json",
                     {"source": str(final_path), "source_final_sha256": sha(final),
                      "request_sha256": key, "response_sha256": sha(response)})
                self.log({"kind": "source_cache_hit", "stage": stage, "scope": scope,
                          "request": key, "source_root": str(source)})
                return response["response"]
        return super().post(stage, scope, model, method, body)


def shared_sources():
    files = [Path(__file__), Path(__file__).with_name("resume_baseline.py")]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in files}

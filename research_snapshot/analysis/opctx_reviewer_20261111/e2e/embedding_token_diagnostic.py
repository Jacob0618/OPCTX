"""Separate post-run native embedding-token measurement; never pipeline cost.

Reads only the cold-run batchEmbedContents request ledger. Sends one native
countTokens request per distinct content, never concatenates embedding items.
No generation, embedding, Writer, or QA calls are made by this diagnostic.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
MODEL = "gemini-embedding-001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plan(source):
    summary = read(source / "summary.json")
    if summary["status"] != "complete":
        raise RuntimeError("The pipeline must complete before token diagnostics")
    meta = read(Path(__file__).with_name("embedding_metadata.json"))
    # Accept the official metadata object directly or a response wrapper.
    model_meta = meta.get("response", meta.get("metadata", meta))
    if "countTokens" not in model_meta.get("supportedGenerationMethods", []):
        raise RuntimeError("Official model metadata does not advertise countTokens")
    items = {}
    for folder in sorted((source / "calls").iterdir()):
        outcome = read(folder / "outcome.json")
        if outcome["kind"] != "embedding" or not outcome["attempted"]:
            continue
        entry = read(folder / "request.json")
        if entry["model"] != MODEL:
            raise RuntimeError("Unexpected embedding model")
        for request in entry["request"]["requests"]:
            # The exact embedding content object, not a different model's
            # tokenizer or an aggregate concatenation of individual texts.
            content = request["content"]
            key = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            item = items.setdefault(key, dict(content=content, multiplicity=0, stages=Counter(), successful_stages=Counter()))
            item["multiplicity"] += 1
            item["stages"][outcome["stage"]] += 1
            if outcome["status"] == "ok":
                item["successful_stages"][outcome["stage"]] += 1
    return items, model_meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--allow-model-calls", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    items, model_meta = plan(args.input_root.resolve())
    brief = dict(model=MODEL, unique_contents=len(items), physical_embedding_items=sum(x["multiplicity"] for x in items.values()),
                 successful_embedding_items=sum(sum(x["successful_stages"].values()) for x in items.values()),
                 pipeline_calls=0, diagnostic_call_limit=len(items), model_input_token_limit=model_meta.get("inputTokenLimit"))
    if not args.allow_model_calls:
        print(json.dumps(brief, indent=2))
        return 0
    if not args.output_root or not 1 <= args.workers <= 8:
        p.error("--output-root required; workers must be 1..8")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Parent must supply credentials through environment")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from creamem3.experiments.memeye_resume_20260921 import network
    from creamem3.experiments.memeye_backbones_20260921.runtime import disk_guard
    stop, lock = threading.Event(), threading.Lock()
    results = []
    write(output / "registration.json", {**brief, "input_root": str(args.input_root.resolve()), "pid": os.getpid(),
        "scope": "post-run diagnostic only; excluded from pipeline calls, cost, and wall time",
        "interpretation": "Native count of exact input contents, not embedding billed usage; over-limit items are flagged, not silently clipped",
        "retry_policy": "one request per unique content; stop new work on first failure"})

    def count(pair):
        key, item = pair
        if stop.is_set():
            return
        start = time.perf_counter()
        row = dict(key=key, multiplicity=item["multiplicity"], stages=dict(item["stages"]),
                   successful_stages=dict(item["successful_stages"]), status="started")
        try:
            disk_guard(output)
            payload = {"contents": [item["content"]]}
            write(output / "calls" / key / "request.json", payload)
            response = network.client().post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:countTokens",
                content=network.g._canonical_json(payload).encode(),
                headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            row["http_status"] = response.status_code
            if response.status_code != 200:
                raise RuntimeError("native countTokens HTTP failure")
            raw = response.json()
            write(output / "calls" / key / "response.json", raw)
            tokens = raw.get("totalTokens")
            if not isinstance(tokens, int) or tokens < 0:
                raise RuntimeError("native countTokens missing totalTokens")
            row.update(status="ok", tokens=tokens, exceeds_advertised_input_limit=tokens > int(model_meta.get("inputTokenLimit", 2048)))
        except Exception as exc:
            stop.set()
            row.update(status="failed", error_type=type(exc).__name__)
        finally:
            row["diagnostic_wall_seconds"] = time.perf_counter() - start
            write(output / "calls" / key / "outcome.json", row)
            with lock:
                results.append(row)

    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(count, items.items()))
    finally:
        good = [x for x in results if x["status"] == "ok"]
        stages, successful_stages = defaultdict(int), defaultdict(int)
        for row in good:
            for stage, n in row["stages"].items():
                stages[stage] += row["tokens"] * n
            for stage, n in row["successful_stages"].items():
                successful_stages[stage] += row["tokens"] * n
        complete = len(good) == len(items)
        report = dict(status="complete" if complete else "incomplete", **brief,
            diagnostic_requests=len(results), successful_diagnostic_requests=len(good),
            exact_input_content_tokens_by_pipeline_stage=dict(stages),
            exact_successful_input_content_tokens_by_pipeline_stage=dict(successful_stages),
            exact_input_content_tokens=sum(row["tokens"] * row["multiplicity"] for row in good),
            over_limit_unique_contents=sum(row["exceeds_advertised_input_limit"] for row in good),
            native_embedding_billed_usage="unknown: countTokens measures input content, not truncation/billing metadata",
            diagnostic_parallel_wall_seconds=time.perf_counter() - start)
        write(output / "summary.json", report)
        print(json.dumps(report, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

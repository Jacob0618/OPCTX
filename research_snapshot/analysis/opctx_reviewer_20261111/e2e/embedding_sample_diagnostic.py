"""Post-run stratified native token-count estimate; never a pipeline stage.

The population is exact unique embedding content, weighted by its original
physical multiplicity. Fixed strata use stage membership and text length;
sampling does not inspect answers, scores, or token-count outcomes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import threading
import time
import embedding_token_diagnostic as shared
import lme_embedding_token_diagnostic as lme

SEED = 20261111
N_PER_STRATUM = 10


def char_bin(n):
    for limit in (128, 512, 2048, 8192):
        if n <= limit:
            return str(limit)
    return "above8192"


def prepare(dataset, source):
    items, meta = (lme.plan(source) if dataset == "lme" else shared.plan(source))
    strata = defaultdict(list)
    for key, item in sorted(items.items()):
        chars = sum(len(p.get("text", "")) for p in item["content"]["parts"])
        label = "+".join(sorted(item["stages"])) + "/chars_le_" + char_bin(chars)
        strata[label].append(key)
    rng = random.Random(SEED)
    selected = {}
    for label, keys in sorted(strata.items()):
        for key in rng.sample(keys, min(N_PER_STRATUM, len(keys))):
            selected[key] = dict(stratum=label, item=items[key])
    census = {
        "seed": SEED, "sample_per_stratum": N_PER_STRATUM,
        "unique_contents": len(items), "sample_contents": len(selected),
        "physical_embedding_items": sum(v["multiplicity"] for v in items.values()),
        "population_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
        "strata": {k: {"population_n": len(v), "sample_n": min(N_PER_STRATUM, len(v))}
                   for k, v in sorted(strata.items())},
        "sample_keys": {k: v["stratum"] for k, v in selected.items()},
        "scope": "Separate countTokens diagnostic; no generation or embedding calls; excluded from pipeline calls/wall.",
        "estimand": "Native tokens of exact input content, weighted by physical multiplicity; NOT provider billed/truncated usage.",
        "interval": "Approximate normal 95% sampling interval using stratified SRS finite-population variance, not workload uncertainty.",
    }
    return selected, census, meta


def estimate(rows, census, successful_only):
    stage_names = sorted({s for r in rows for s in r["stages"]})
    output = {}
    for stage in stage_names + ["ALL"]:
        estimate_total = variance = 0.0
        for label, numbers in census["strata"].items():
            group = [r for r in rows if r["stratum"] == label]
            if len(group) != numbers["sample_n"]:
                raise RuntimeError("Missing sampled counts: no partial-population estimate")
            values = []
            for r in group:
                weights = r["successful_stages"] if successful_only else r["stages"]
                weight = sum(weights.values()) if stage == "ALL" else weights.get(stage, 0)
                values.append(r["tokens"] * weight)
            n, population_n = len(values), numbers["population_n"]
            estimate_total += population_n * statistics.mean(values)
            if n > 1:
                variance += population_n ** 2 * (1 - n / population_n) * statistics.variance(values) / n
        margin = 1.96 * math.sqrt(variance)
        output[stage] = {"estimated_input_tokens": estimate_total,
                         "sampling_interval95": [max(0, estimate_total - margin), estimate_total + margin]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("locomo", "lme"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    selected, census, meta = prepare(args.dataset, args.input_root.resolve())
    if not args.allow_model_calls:
        print(json.dumps(census, indent=2))
        return 0
    if args.output_root is None or not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Output and process-local credentials required")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shared.write(output / "registration.json", dict(census, dataset=args.dataset,
        input_root=str(args.input_root.resolve()), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=shared.MODEL, request_attempts_per_content=1, workers=4))
    from creamem3.experiments.memeye_resume_20260921 import network
    from creamem3.experiments.memeye_backbones_20260921.runtime import disk_guard
    stop, lock = threading.Event(), threading.Lock()
    rows = []

    def count(pair):
        key, selection = pair
        if stop.is_set():
            return
        item = selection["item"]
        row = dict(key=key, stratum=selection["stratum"], stages=dict(item["stages"]),
                   successful_stages=dict(item["successful_stages"]), status="started")
        started = time.perf_counter()
        try:
            disk_guard(output)
            payload = {"contents": [item["content"]]}
            shared.write(output / "calls" / key / "request.json", payload)
            response = network.client().post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{shared.MODEL}:countTokens",
                content=network.g._canonical_json(payload).encode(),
                headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            row["http_status"] = response.status_code
            try:
                raw = response.json()
            except ValueError:
                raw = {"non_json_response": True}
            shared.write(output / "calls" / key / "response.json", raw)
            if response.status_code != 200:
                raise RuntimeError("Official countTokens failed; no automatic retry")
            tokens = raw.get("totalTokens")
            if not isinstance(tokens, int) or tokens < 0:
                raise RuntimeError("Invalid token count")
            row.update(status="ok", tokens=tokens,
                       exceeds_input_limit=tokens > int(meta.get("inputTokenLimit", 2048)))
        except Exception as exc:
            stop.set()
            row.update(status="failed", error_type=type(exc).__name__)
        finally:
            row["diagnostic_seconds"] = time.perf_counter() - started
            shared.write(output / "calls" / key / "outcome.json", row)
            with lock:
                rows.append(row)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(count, selected.items()))
    good = [r for r in rows if r["status"] == "ok"]
    complete = len(good) == len(selected)
    report = dict(status="complete" if complete else "incomplete", census=census,
                  diagnostics_attempted=len(rows), diagnostics_successful=len(good),
                  diagnostic_wall_seconds=time.perf_counter() - started,
                  sampled_over_limit_contents=sum(r["exceeds_input_limit"] for r in good),
                  unsampled_over_limit_status="unknown", native_billed_usage="unknown")
    if complete:
        report["all_attempted_input_estimate"] = estimate(good, census, False)
        report["successful_input_estimate"] = estimate(good, census, True)
    shared.write(output / "summary.json", report)
    print(json.dumps(report, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

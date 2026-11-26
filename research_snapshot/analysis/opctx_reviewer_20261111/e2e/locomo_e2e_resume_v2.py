"""Bounded recovery of this cold LoCoMo run only, preserving the frozen entry.

Reuses exact successful request responses produced by --source-run. No old
experiment output is admitted. Explicit transient HTTP errors get at most
three physical attempts total per exact request, including prior failures.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import types

BASE_PATH = Path(__file__).with_name("locomo_e2e.py")
spec = importlib.util.spec_from_file_location("locomo_e2e_frozen_base", BASE_PATH)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def request_key(model, payload):
    return b.digest([model, payload])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-model-calls", action="store_true")
    args = p.parse_args()
    prepared = b.preflight()
    sources = [x.resolve() for x in args.source_run]
    registrations = [b.read(x / "registration.json") for x in sources]
    old_summaries = [b.read(x / "summary.json") for x in sources]
    if any(x["status"] == "complete" for x in old_summaries):
        raise RuntimeError("A source is already complete; no recovery")
    if len(set(sources)) != len(sources):
        raise RuntimeError("Duplicate recovery source")
    for registration in registrations:
        if registration["script_sha256"] != b.file_sha(BASE_PATH):
            raise RuntimeError("Frozen entry changed after source run")
        for key in ("sample_id", "source_hashes", "raw_dataset_sha256", "qa_prompt_sha256"):
            if registration[key] != prepared[-1][key]:
                raise RuntimeError(f"Recovery binding mismatch: {key}")
    prior_calls, prior_attempts, cache, bindings = [], Counter(), {}, []
    successful_request_keys = set()
    for source in sources:
        for folder in sorted((source / "calls").iterdir()):
            outcome = b.read(folder / "outcome.json")
            entry = b.read(folder / "request.json")
            prior_calls.append(outcome)
            key = request_key(entry["model"], entry["request"])
            prior_attempts[key] += outcome["physical_calls"]
            if outcome["status"] == "ok":
                successful_request_keys.add(key)
                raw = b.read(folder / "response.json")
                payload = entry["request"]
                if outcome["kind"] == "generation":
                    if outcome["finish_reason"] != "STOP":
                        raise RuntimeError("Non-STOP response cannot be reused")
                    cache_key = hashlib.sha256(json.dumps(["chat.v1", entry["model"], payload], sort_keys=True).encode()).hexdigest()
                    cache[cache_key] = {"text": "".join(p.get("text", "") for p in raw["candidates"][0]["content"]["parts"])}
                else:
                    if len(payload["requests"]) != len(raw["embeddings"]):
                        raise RuntimeError("Prior embedding shape mismatch")
                    for req, vector in zip(payload["requests"], raw["embeddings"]):
                        parts = req["content"]["parts"]
                        if len(parts) != 1 or set(parts[0]) != {"text"}:
                            raise RuntimeError("Unexpected old embedding wire")
                        cache_key = hashlib.sha256(json.dumps(["emb.v1", entry["model"], req["taskType"], req["outputDimensionality"], parts[0]["text"]]).encode()).hexdigest()
                        cache[cache_key] = {"v": vector["values"]}
                bindings.append(dict(request=str(folder / "request.json"), response=str(folder / "response.json"),
                                     request_sha256=b.file_sha(folder / "request.json"), response_sha256=b.file_sha(folder / "response.json")))
            elif not (outcome.get("http_status") in (None, 408, 429, 500, 502, 503, 504)
                      and outcome.get("retryable") is True and outcome["status"] == "explicit_provider_error"):
                raise RuntimeError("Unapproved prior failure kind; do not automatically retry")
    policy = dict(source_runs=[str(x) for x in sources], source_bindings=[dict(root=str(x), registration_sha256=b.file_sha(x / "registration.json"), summary_sha256=b.file_sha(x / "summary.json")) for x in sources], recovered_successes=bindings,
        retry_policy="Only verified pre-send connection failure or explicit HTTP 408/429/500/502/503/504, max3 physical attempts per exact request including source; no retries for safety/truncation/post-connect indeterminate outcomes",
        missing_request_remaining_attempts={k: max(0, 3 - n) for k, n in prior_attempts.items() if k not in successful_request_keys},
        note="Logical cold run resumed from its own outputs; interrupted execution windows reported separately, never called an uninterrupted wall clock")
    prepared[-1]["recovery_policy"] = policy
    prepared[-1]["recovery_entry_sha256"] = b.file_sha(__file__)
    if not args.allow_model_calls:
        print(json.dumps(policy, indent=2))
        return 0

    original_ledger = b.Ledger

    class RecoveryLedger(original_ledger):
        def __init__(self, output):
            super().__init__(output)
            self.cache.update(cache)
            self.request_attempts = Counter(prior_attempts)

        def request(self, url, payload):
            stage, scope = self.local.stage, getattr(self.local, "scope", "shared")
            kind = "embedding" if ":batchEmbedContents" in url else "generation"
            model = b.EMBED_MODEL if kind == "embedding" else b.MODEL
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:" + ("batchEmbedContents" if kind == "embedding" else "generateContent")
            if url.split("?")[0] != endpoint:
                raise b.ExecutionStop("Unexpected endpoint/model")
            key = request_key(model, payload)
            while True:
                if self.stop.is_set():
                    raise b.ExecutionStop("Stopped before request")
                with self.lock:
                    if self.request_attempts[key] >= 3:
                        self.stop.set()
                        raise b.ExecutionStop("Exact request physical attempt cap reached")
                    self.request_attempts[key] += 1
                    attempt = self.request_attempts[key]
                    self.next_id += 1
                    number = self.next_id
                folder = self.output / "calls" / f"{number:06d}_{stage}"
                outcome = dict(id=number, scope=scope, stage=stage, kind=kind, model=model, status="pending",
                    attempted=False, physical_calls=0, request_key=key, logical_attempt=attempt, usage=None)
                start = time.perf_counter()
                retry = False
                try:
                    self.runtime.disk_guard(self.output)
                    b.save(folder / "request.json", dict(scope=scope, stage=stage, model=model, request=payload, request_sha256=b.digest(payload)))
                    outcome.update(attempted=True, physical_calls=1)
                    if kind == "generation":
                        raw = self.provider.generate_content(model=model, request=payload)
                    else:
                        response = self.network.client().post(endpoint, content=self.network.g._canonical_json(payload).encode(),
                            headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
                        if response.status_code != 200:
                            outcome["http_status"] = response.status_code
                            if response.status_code in (408, 429, 500, 502, 503, 504):
                                retry = attempt < 3
                                outcome.update(status="explicit_provider_error", retryable=True)
                            else:
                                outcome.update(status="explicit_provider_error", retryable=False)
                            raise b.ExecutionStop("Embedding HTTP failure")
                        raw = response.json()
                    b.save(folder / "response.json", raw)
                    outcome.update(http_status=200, usage=raw.get("usageMetadata"))
                    if kind == "generation":
                        version = str(raw.get("modelVersion", ""))
                        candidates = raw.get("candidates") or []
                        reason = candidates[0].get("finishReason") if candidates else None
                        outcome.update(model_version=version, finish_reason=reason)
                        if not version.startswith("gemini-2.5-flash") or "lite" in version:
                            raise b.ExecutionStop("Response model changed")
                        if not candidates or reason != "STOP":
                            raise b.ExecutionStop("Non-STOP response; no automatic retry")
                        if not outcome["usage"] or "promptTokenCount" not in outcome["usage"]:
                            raise b.ExecutionStop("Missing generation usage")
                    else:
                        requests, vectors = payload["requests"], raw.get("embeddings", [])
                        if len(vectors) != len(requests) or any(len(v.get("values", [])) != 768 for v in vectors):
                            raise b.ExecutionStop("Embedding shape mismatch")
                        outcome.update(input_items=len(requests), input_chars=sum(len(p.get("text", "")) for r in requests for p in r["content"]["parts"]),
                                       token_usage_known=bool(outcome["usage"]))
                    outcome["status"] = "ok"
                    return raw
                except self.network.g.ExplicitProviderError as exc:
                    retry = exc.status_code in (None, 408, 429, 500, 502, 503, 504) and exc.retryable and attempt < 3
                    outcome["pre_send_connection_failure"] = exc.status_code is None
                    outcome["provider_request_reached"] = None if exc.status_code is None else True
                    if exc.status_code is None:
                        cause = exc.__cause__
                        cause_text = str(cause).lower()
                        cause_type = type(cause).__name__
                        unsafe = any(x in cause_text for x in ("certificate", "ssl", "tls", "permission", "access denied", "forbidden"))
                        known_connect = cause_type in ("ConnectError", "ConnectTimeout")
                        outcome.update(connection_cause_type=cause_type,
                            connection_classification="tls_or_permission_stop" if unsafe else ("pre_send_connect_failure" if known_connect else "unclassified_stop"),
                            provider_request_reached=False if known_connect else None,
                            possible_indeterminate_billing=False if known_connect else True)
                        retry = retry and known_connect and not unsafe
                    outcome.update(status="explicit_provider_error", http_status=exc.status_code, retryable=exc.retryable)
                    if not retry:
                        self.stop.set()
                        raise b.ExecutionStop("Explicit provider failure at recovery limit") from None
                except b.ExecutionStop:
                    if outcome["status"] == "pending":
                        outcome["status"] = "stopped"
                    if not retry:
                        self.stop.set()
                        raise
                except BaseException as exc:
                    outcome.update(status="transport_or_local_failure", error_type=type(exc).__name__, possible_indeterminate_billing=outcome["attempted"])
                    self.stop.set()
                    raise b.ExecutionStop("Unknown failure; no automatic retry") from None
                finally:
                    outcome["service_seconds"] = time.perf_counter() - start
                    b.save(folder / "outcome.json", outcome)
                    with self.lock:
                        self.calls.append(outcome)
                if retry:
                    time.sleep(min(4, 2 ** (attempt - 1)))

    b.Ledger = RecoveryLedger
    code = b.run(args, prepared)
    summary = b.read(args.output.resolve() / "summary.json")
    new_calls = [b.read(p) for p in (args.output.resolve() / "calls").glob("*/outcome.json")]
    all_calls = prior_calls + new_calls
    stages = {}
    for stage in sorted({c["stage"] for c in all_calls}):
        calls = [c for c in all_calls if c["stage"] == stage]
        known = [c["usage"] for c in calls if c.get("usage") is not None]
        stages[stage] = dict(kind=calls[0]["kind"], physical_calls=sum(c["physical_calls"] for c in calls),
            physical_network_attempts=sum(c["physical_calls"] for c in calls),
            presend_connection_failures=sum(c.get("http_status") is None and c.get("retryable") is True and c["status"] == "explicit_provider_error" for c in calls),
            http_provider_failures=sum(c.get("http_status") is not None and c.get("http_status") != 200 for c in calls),
            successful_calls=sum(c["status"] == "ok" for c in calls), unknown_usage_calls=sum(c.get("usage") is None for c in calls),
            measured_input_tokens=sum(u.get("promptTokenCount", 0) for u in known),
            measured_output_tokens=sum(u.get("candidatesTokenCount", 0) for u in known),
            measured_thought_tokens=sum(u.get("thoughtsTokenCount", 0) for u in known),
            measured_cached_input_tokens=sum(u.get("cachedContentTokenCount", 0) for u in known),
            service_seconds=sum(c["service_seconds"] for c in calls),
            embedding_items=sum(c.get("input_items", 0) for c in calls), embedding_chars=sum(c.get("input_chars", 0) for c in calls))
    b.save(args.output.resolve() / "logical_run_summary.json", dict(status=summary["status"], completed=summary["completed"],
        source_runs=[str(x) for x in sources], resumed_run=str(args.output.resolve()), stages=stages,
        active_execution_windows_seconds=[x["wall_clock_seconds"] for x in old_summaries] + [summary["wall_clock_seconds"]],
        total_active_execution_window_seconds=sum(x["wall_clock_seconds"] for x in old_summaries) + summary["wall_clock_seconds"],
        wall_note="Sum of measured execution windows, excludes human recovery pause; not uninterrupted wall time.",
        source_memory_batches=[dict(root=str(root), batches=old["memory_batches"]) for root, old in zip(sources, old_summaries)], resumed_memory_batches=summary["memory_batches"],
        raw_full_population_counts=summary["raw_full_population_counts"],
        embedding_usage_note="Missing native embedding usage remains unknown; zeros in measured token totals mean no measurement, not zero tokens.",
        failed_call_note="HTTP error responses without usage are retained and counted; provider billing is unknown, not silently declared zero."))
    return code


if __name__ == "__main__":
    raise SystemExit(main())



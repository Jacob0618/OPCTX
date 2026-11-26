"""Cold end-to-end cost profile of the historical LoCoMo 81.75 configuration.

No old model responses, facts, embeddings, plans, or answers enter execution.
Original source hashes and raw dataset are validated before any model call.
Default is offline preflight; parent must explicitly enable model calls.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import types
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
FROZEN_GRAPH = ROOT / "creamem3/frozen_sources/locomo81_graph_memory_v2.py"
FROZEN_ONLINE = ROOT / "creamem3/results_top40_fallback_repairs_v5/.full_top40_v5_r1_fallback_repair_r2.staging-lfq8z1l9/source_snapshot/creamem3/online_memory_v3.py"
SOURCE = ROOT / "creamem3/results_autoresearch_v8/online_operation_context_v3/conv-26/online_operation_context_v3.json"
HIST_GRAPH = ROOT / "creamem3/results_locomo10_v3/conv-26/graph_online.json"
EXPECTED = {
    "graph": "6e60aa6f8a4123f848221ba180f703f12c540268d0e17981f435d8a81947b8ee",
    "online": "3af38e30881fc6ce64d393c388c4594e1a9adbecb196136bbc306659f14f72b9",
    "runner": "2a0d0329be6ba5142bf38a6ade669d8120b999f3f112c4fb2d5d7bbe5296930e",
    "client": "0222e860eccae9b0561038c27ca5aae439c6ea088bc0d8b71b1bddc4769612dd",
    "qa_prompt": "e84efdf3d7d115b04f547261a6683c60014606e39f41e33c13566c4edc71dfb5",
}
STAGES = {"creamem3_v2_write": "writer", "creamem3_v3_online_fuse": "online",
          "creamem3_v2_rerank": "reranker", "retrieve": "planner", "read": "qa",
          "creamem3_v2_embed_write": "write_embedding", "creamem3_v3_embed_online": "online_embedding",
          "embed_query": "query_embedding"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    if not isinstance(value, bytes):
        value = (value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)).encode()
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()


def load_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


def private_module(path, name, import_map=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (import_map or {}):
            node.module = import_map[node.module]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def preflight():
    from creamem3 import prompts_v2
    from creamem3.locomo_all import DEFAULT_DATASET_PATH, load_all_locomo
    from creamem3.online_operation_context_v8 import materialize_online_operation_context
    from creamem3.strict_semantic_planner_v8 import classify_strict
    source = read(SOURCE)
    authority = read(HIST_GRAPH)["runner_manifest"]
    protocol = authority["protocol"]
    paths = {"graph": FROZEN_GRAPH, "online": FROZEN_ONLINE,
             "runner": ROOT / "creamem3/run_conv26_v2.py", "client": ROOT / "eval/gemini_client.py"}
    for name, path in paths.items():
        if file_sha(path) != EXPECTED[name]:
            raise RuntimeError(f"Historical source hash mismatch: {name}")
    if prompts_v2.PROMPT_SHA256 != protocol["memory_prompt_sha256"]:
        raise RuntimeError("Historical memory prompt mismatch")
    for name in ("semantic_context", "strict_planner"):
        item = source["source_snapshot"][name]
        if file_sha(item["path"]) != item["sha256"]:
            raise RuntimeError(f"Historical reading source mismatch: {name}")
    all_data = load_all_locomo(DEFAULT_DATASET_PATH)
    all_data.sort(key=lambda x: int(x["sample_id"].split("-")[-1]))
    loaded = all_data[0]
    if loaded["sample_id"] != "conv-26" or len(loaded["records"]) != 19 or len(loaded["qa"]) != 152:
        raise RuntimeError("Predeclared sample identity/count mismatch")
    if loaded["hashes"] != authority["dataset"]["hashes"]:
        raise RuntimeError("Original raw dataset binding mismatch")
    tree = ast.parse((ROOT / "eval/prompts.py").read_text(encoding="utf-8-sig"))
    template = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "QA_PROMPT" for t in n.targets))
    if digest(template) != EXPECTED["qa_prompt"]:
        raise RuntimeError("QA prompt mismatch")
    graph = private_module(FROZEN_GRAPH, "_opctx_e2e_frozen_graph")
    online = private_module(FROZEN_ONLINE, "_opctx_e2e_frozen_online",
                            {"creamem3.graph_memory_v2": graph.__name__})
    # Existing Reader revision changed only the two-view label. It was previously
    # byte-verified on all 152 frozen contexts; run-time inputs here are new.
    metadata = dict(sample_id="conv-26", sample_n=152, sample_sessions=19, sample_online_frontiers=18,
        total_conversations=len(all_data), total_questions=sum(len(x["qa"]) for x in all_data),
        total_sessions=sum(len(x["records"]) for x in all_data),
        total_online_frontiers=sum(max(0, len(x["records"]) - 1) for x in all_data),
        sample_choice="lowest numeric ID, fixed before any new outcome",
        cold_input="original dialogue only; no old facts/embeddings/plans/contexts/model results reused",
        model=MODEL, embedding_model=EMBED_MODEL, embedding_dimension=768,
        write_unit="one complete physical session", thinking_budget=0,
        retrieval=dict(summary_top_k=10, dense_fact_top_k=20, bm25_fact_top_k=20, candidate_cap=120,
                       reranker="Gemini 2.5 Flash", rerank_top_k=20),
        selector="not present in original 81.75 pipeline; dense cap120, no mathematical Selector",
        reader="new question-only Planner -> original deterministic materializer -> original QA",
        judge="excluded; this is cost profiling, not scored QA evaluation",
        source_hashes={str(p): file_sha(p) for p in [*paths.values(), ROOT / "creamem3/prompts_v2.py",
            ROOT / "creamem3/online_operation_context_v8.py", ROOT / "creamem3/semantic_context_v8.py",
            ROOT / "creamem3/strict_semantic_planner_v8.py", ROOT / "creamem3/semantic_query_planner_v6.py"]},
        raw_dataset_sha256=file_sha(DEFAULT_DATASET_PATH), qa_prompt_sha256=digest(template),
        note="Original function bodies reused. Frozen Online import points only to exact-hash frozen graph helper. Private instrumentation changes logging only.")
    # Strip scoring data before any run stage. Only protocol verification used it.
    clean = dict(sample_id=loaded["sample_id"], records=loaded["records"],
                 legal_dia_ids=list(loaded["dia_id_map"]),
                 qa=[{"qa_id": q["qa_id"], "question": q["question"]} for q in loaded["qa"]])
    return clean, graph, online, materialize_online_operation_context, classify_strict, template, metadata


class ExecutionStop(BaseException):
    """Bypass legacy catch-and-fallback on provider/safety/access failure."""


class Ledger:
    def __init__(self, output):
        from creamem3.experiments.memeye_backbones_20260921 import runtime
        from creamem3.experiments.memeye_resume_20260921 import network
        self.output, self.runtime, self.network = output, runtime, network
        self.provider = network.VerifiedProvider()
        self.local, self.stop, self.lock = threading.local(), threading.Event(), threading.Lock()
        self.calls, self.cache, self.usage, self.next_id = [], {}, {}, 0

    def add(self, tag, prompt_toks, out_toks, calls=1, cached=False, model=None, thought_toks=0):
        with self.lock:
            item = self.usage.setdefault(tag, dict(calls=0, cache_hits=0, prompt_tokens=0, output_tokens=0, thought_tokens=0))
            if cached:
                item["cache_hits"] += 1
            else:
                for k, v in (("calls", calls), ("prompt_tokens", prompt_toks), ("output_tokens", out_toks), ("thought_tokens", thought_toks)):
                    item[k] += v

    def cache_get(self, key):
        with self.lock:
            return self.cache.get(key)

    def cache_put(self, key, value):
        with self.lock:
            self.cache[key] = value

    def request(self, url, payload):
        if self.stop.is_set():
            raise ExecutionStop("Stopped before request")
        stage, scope = self.local.stage, getattr(self.local, "scope", "shared")
        kind = "embedding" if ":batchEmbedContents" in url else "generation"
        expected = EMBED_MODEL if kind == "embedding" else MODEL
        suffix = "batchEmbedContents" if kind == "embedding" else "generateContent"
        allowed = f"https://generativelanguage.googleapis.com/v1beta/models/{expected}:{suffix}"
        if url.split("?")[0] != allowed:
            raise ExecutionStop("Unexpected endpoint or model")
        with self.lock:
            self.next_id += 1
            number = self.next_id
        folder = self.output / "calls" / f"{number:06d}_{stage}"
        outcome = dict(id=number, scope=scope, stage=stage, kind=kind, model=expected,
                       status="pending", attempted=False, physical_calls=0)
        start = time.perf_counter()
        try:
            self.runtime.disk_guard(self.output)
            save(folder / "request.json", dict(scope=scope, stage=stage, model=expected, request=payload,
                                               request_sha256=digest(payload)))
            outcome.update(attempted=True, physical_calls=1)
            if kind == "generation":
                raw = self.provider.generate_content(model=MODEL, request=payload)
            else:
                # Same verified TLS/proxy client as VerifiedProvider; original
                # native batchEmbedContents wire object, dimension/task unchanged.
                response = self.network.client().post(allowed,
                    content=self.network.g._canonical_json(payload).encode(),
                    headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
                if not 200 <= response.status_code < 300:
                    outcome["http_status"] = response.status_code
                    raise ExecutionStop("Embedding HTTP failure; no automatic retry")
                raw = response.json()
            save(folder / "response.json", raw)
            outcome["http_status"] = 200
            outcome["usage"] = raw.get("usageMetadata")
            if kind == "generation":
                version = str(raw.get("modelVersion", ""))
                outcome["model_version"] = version
                if not version.startswith("gemini-2.5-flash") or "lite" in version:
                    raise ExecutionStop("Response model version changed")
                candidates = raw.get("candidates") or []
                reason = candidates[0].get("finishReason") if candidates else None
                outcome["finish_reason"] = reason
                if not candidates or reason != "STOP":
                    raise ExecutionStop("Safety/blocked/incomplete provider response")
                if not outcome["usage"] or "promptTokenCount" not in outcome["usage"]:
                    raise ExecutionStop("Missing generation usage")
            else:
                requests = payload["requests"]
                vectors = raw.get("embeddings", [])
                if len(vectors) != len(requests) or any(len(v.get("values", [])) != 768 for v in vectors):
                    raise ExecutionStop("Embedding result shape mismatch")
                outcome.update(input_items=len(requests),
                    input_chars=sum(len(p.get("text", "")) for r in requests for p in r["content"]["parts"]),
                    token_usage_known=bool(outcome["usage"]))
            outcome["status"] = "ok"
            return raw
        except self.network.g.ExplicitProviderError as exc:
            outcome.update(status="explicit_provider_error", http_status=exc.status_code,
                           retryable=exc.retryable)
            self.stop.set()
            raise ExecutionStop("Explicit provider failure; preserved outcome") from None
        except ExecutionStop:
            outcome["status"] = "stopped"
            self.stop.set()
            raise
        except BaseException as exc:
            outcome.update(status="transport_or_local_failure", error_type=type(exc).__name__,
                           possible_indeterminate_billing=outcome["attempted"])
            self.stop.set()
            raise ExecutionStop("Unexpected failure; no automatic retry") from None
        finally:
            outcome["service_seconds"] = time.perf_counter() - start
            save(folder / "outcome.json", outcome)
            with self.lock:
                self.calls.append(outcome)

    def snapshots(self):
        with self.lock:
            return json.loads(json.dumps(self.calls)), json.loads(json.dumps(self.usage))


def run(args, prepared):
    clean, gm, online, materialize, classify, template, meta = prepared
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Parent must supply GEMINI_API_KEY")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Require new output directory; no implicit overwrite/resume")
    # mkdir(exist_ok=False) is the atomic single-owner claim; a competing run
    # loses before registration or any API call. No implicit resume is allowed.
    output.mkdir(parents=True, exist_ok=False)
    save(output / "owner_lock.json", dict(pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(),
                                         policy="atomic new-directory ownership; retained after completion"))
    meta.update(created_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers,
                api_retry_policy="one physical attempt; original writer/reranker semantic validation retries retained",
                cache_policy="cold in-memory cache; only results generated during this run may be reused",
                disk_floor_gib=15, script_sha256=file_sha(__file__))
    save(output / "registration.json", meta)
    ledger = Ledger(output)
    ns = dict(json=json, hashlib=hashlib, os=os, re=re, BACKBONE_MODEL=MODEL, EMBED_MODEL=EMBED_MODEL,
        BACKBONE_BASE_URL="", NO_CHAT_CACHE=False, USAGE=ledger,
        BASE="https://generativelanguage.googleapis.com/v1beta", API_KEY="", _post=ledger.request,
        _cache_get=ledger.cache_get, _cache_put=ledger.cache_put)
    load_function(ROOT / "eval/gemini_client.py", "_valid_judge_text", ns)
    load_function(ROOT / "eval/gemini_client.py", "_valid_cached_chat", ns)
    raw_chat = load_function(ROOT / "eval/gemini_client.py", "chat", ns)
    raw_embed = load_function(ROOT / "eval/gemini_client.py", "embed", ns)

    def chat(prompt, **kwargs):
        ledger.local.stage = STAGES[kwargs.get("tag", "read")]
        return raw_chat(prompt, **kwargs)

    def embed(texts, **kwargs):
        tag = kwargs.get("tag", "embed_query")
        ledger.local.stage = STAGES.get(tag, "online_embedding" if "online" in tag else "write_embedding")
        return raw_embed(texts, **kwargs)

    runner_ns = dict(Mapping=Mapping, Any=Any, Sequence=Sequence, chat=chat, embed=embed,
        FIXED_WRITER_READER_MODEL=MODEL, FIXED_EMBED_MODEL=EMBED_MODEL, EMBEDDING_DIM=768,
        CURRENT_SAMPLE_ID="conv-26", CURRENT_SAMPLE_INDEX=0, QA_PROMPT=template)
    model_chat = load_function(ROOT / "creamem3/run_conv26_v2.py", "_model_chat", runner_ns)
    embed_texts = load_function(ROOT / "creamem3/run_conv26_v2.py", "_embed_texts", runner_ns)
    answer_one = load_function(ROOT / "creamem3/run_conv26_v2.py", "_answer_one", runner_ns)
    batch_records, online_records, query_records, phases = [], [], [], {}
    mutex = threading.Lock()
    old_extract = gm.extract_session_packet

    def extract(record, *a, **kw):
        ledger.local.scope = f"session_{record['session_id']:03d}"
        t = time.perf_counter()
        packet = old_extract(record, *a, **kw)
        item = dict(session_id=record["session_id"], writer_wall_seconds=time.perf_counter() - t,
                    facts=len(packet["atomic_facts"]), writer_calls=packet["writer_attempts"])
        save(output / "batches" / f"session_{record['session_id']:03d}.json", {**item, "packet": packet})
        with mutex:
            batch_records.append(item)
        return packet

    gm.extract_session_packet = extract  # private in-memory wrapper, original body unchanged
    # Three _state_hash calls per original frontier. Start before the first;
    # original _chain_hash closes the entire frontier after validation/embedding.
    old_state, old_chain = online._state_hash, online._chain_hash
    frontier_state = {"calls": 0, "started": 0.0}
    sessions = [r["session_id"] for r in sorted(clean["records"], key=lambda r: r["session_id"])]

    def state(*a, **kw):
        if frontier_state["calls"] % 3 == 0:
            sid = sessions[frontier_state["calls"] // 3]
            ledger.local.scope = f"frontier_{sid:03d}"
            frontier_state["started"] = time.perf_counter()
        frontier_state["calls"] += 1
        return old_state(*a, **kw)

    def chain(previous, step):
        value = old_chain(previous, step)
        item = dict(session_id=step["session_id"], wall_seconds=time.perf_counter() - frontier_state["started"],
                    model_call_made=step["model_call_made"], accepted=len(step["accepted"]),
                    rejected=len(step["rejected"]), model_error=step["model_error"])
        online_records.append(item)
        save(output / "frontiers" / f"session_{step['session_id']:03d}.json", item)
        return value

    online._state_hash, online._chain_hash = state, chain
    started = time.perf_counter()
    status, failure = "started", None
    session_graph = online_graph = None
    try:
        ledger.local.scope = "build_shared"
        t = time.perf_counter()
        session_graph = gm.build_session_graph(clean["records"], model_chat, embed_texts, workers=args.workers)
        phases["writer_and_initial_embedding_wall_seconds"] = time.perf_counter() - t
        save(output / "graph_session.json", session_graph)
        check = gm.audit_graph(session_graph, legal_dia_ids=clean["legal_dia_ids"], expected_sessions=19)
        save(output / "session_audit.json", check)
        if check["status"] != "pass":
            raise ExecutionStop("New session graph invariant audit failed")
        t = time.perf_counter()
        online_graph = online.add_online_event_views(session_graph, model_chat, embed_texts)
        phases["online_with_embedding_wall_seconds"] = time.perf_counter() - t
        # The causal auditor replays these hash functions. Instrument only the
        # actual build, not the validation replay (which makes no model calls).
        online._state_hash, online._chain_hash = old_state, old_chain
        save(output / "graph_online.json", online_graph)
        check = online.audit_online_graph(session_graph, online_graph)
        save(output / "online_audit.json", check)
        if check["status"] != "pass":
            raise ExecutionStop("New Online graph causal audit failed")
        if frontier_state["calls"] != 19 * 3:
            raise ExecutionStop("Unexpected original frontier instrumentation count")
        ledger.local.scope = "all_queries"
        t = time.perf_counter()
        vectors = embed_texts([q["question"] for q in clean["qa"]], task_type="RETRIEVAL_QUERY", tag="embed_query")
        phases["query_embedding_wall_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        index = gm.GraphMemoryIndex(online_graph)
        phases["retrieval_index_build_wall_seconds"] = time.perf_counter() - t
        # Time exact original components while retaining GraphMemoryIndex.search.
        for name in ("_candidate_pool", "_rerank", "_assemble"):
            original = getattr(index, name)

            def timed(*a, _fn=original, _name=name, **kw):
                t = time.perf_counter()
                try:
                    return _fn(*a, **kw)
                finally:
                    ledger.local.timings[_name] = ledger.local.timings.get(_name, 0.0) + time.perf_counter() - t
            setattr(index, name, timed)

        def question_job(item):
            qa, vector = item
            qid = qa["qa_id"]
            ledger.local.scope, ledger.local.timings = qid, {}
            if ledger.stop.is_set():
                return dict(qa_id=qid, status="not_attempted_after_stop")
            t0 = time.perf_counter()
            result = dict(qa_id=qid, question=qa["question"], status="started")
            try:
                t = time.perf_counter()
                base_context, trace = index.search(qa["question"], vector, model_chat)
                result["retrieval_including_rerank_wall_seconds"] = time.perf_counter() - t
                t = time.perf_counter()
                plan, raw = classify(qa["question"], chat_fn=chat, model=MODEL)
                result["planner_wall_seconds"] = time.perf_counter() - t
                t = time.perf_counter()
                context, reader_audit = materialize(plan.plan.operation,
                    {"question": qa["question"], "retrieved": base_context, "retrieval_trace": trace}, online_graph)
                result["reader_local_wall_seconds"] = time.perf_counter() - t
                t = time.perf_counter()
                answer = answer_one({"qa": qa, "retrieved": context, "retrieval_trace": {}, "t_search": 0.0})
                result["qa_wall_seconds"] = time.perf_counter() - t
                if answer.get("qa_error"):
                    raise ExecutionStop("QA did not return an answer")
                result.update(status="ok", planner=asdict(plan), planner_raw=raw,
                    retrieval_trace=trace, reader_audit=reader_audit, retrieved=context,
                    context_sha256=digest(context), prediction=answer["response"])
            except (ExecutionStop, Exception) as exc:
                ledger.stop.set()
                result.update(status="failed", error_type=type(exc).__name__,
                              error="See stage request/outcome; no automatic restart",
                              traceback_frames=traceback.format_tb(exc.__traceback__))
            finally:
                result["query_wall_seconds"] = time.perf_counter() - t0
                result["local_component_timings"] = dict(ledger.local.timings)
                save(output / "questions" / f"{qid}.json", result)
                with mutex:
                    query_records.append(result)
            print(f"{qid}: {result['status']}", flush=True)
            return result

        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            returned = list(pool.map(question_job, zip(clean["qa"], vectors)))
        phases["all_queries_parallel_wall_seconds"] = time.perf_counter() - t
        status = "complete" if len(returned) == 152 and all(q["status"] == "ok" for q in returned) else "incomplete"
    except (ExecutionStop, Exception) as exc:
        ledger.stop.set()
        status, failure = "stopped", dict(error_type=type(exc).__name__,
            error="Inspect saved physical request outcomes; original successful records are retained",
            traceback_frames=traceback.format_tb(exc.__traceback__))
    finally:
        wall = time.perf_counter() - started
        calls, hook_usage = ledger.snapshots()
        stages = {}
        for stage in sorted({c["stage"] for c in calls}):
            cs = [c for c in calls if c["stage"] == stage]
            usages = [c["usage"] for c in cs if c.get("usage") is not None]
            known = len(usages) == len(cs)
            stages[stage] = dict(kind=cs[0]["kind"], physical_calls=sum(c["physical_calls"] for c in cs),
                successful_calls=sum(c["status"] == "ok" for c in cs), usage_complete=known,
                input_tokens=sum(u.get("promptTokenCount", 0) for u in usages) if known else None,
                output_tokens=sum(u.get("candidatesTokenCount", 0) for u in usages) if known else None,
                thought_tokens=sum(u.get("thoughtsTokenCount", 0) for u in usages) if known else None,
                cached_content_tokens=sum(u.get("cachedContentTokenCount", 0) for u in usages) if known else None,
                service_seconds=sum(c["service_seconds"] for c in cs),
                input_items=sum(c.get("input_items", 0) for c in cs), input_chars=sum(c.get("input_chars", 0) for c in cs))
        projected = {}
        factors = {"writer": meta["total_sessions"] / 19, "write_embedding": meta["total_sessions"] / 19,
                   "online": meta["total_online_frontiers"] / 18, "online_embedding": meta["total_online_frontiers"] / 18}
        if status == "complete":
            for stage, v in stages.items():
                factor = factors.get(stage, meta["total_questions"] / 152)
                projected[stage] = {"factor": factor, "scaling_unit": "sessions" if stage in ("writer", "write_embedding") else
                    ("online_frontiers" if stage in ("online", "online_embedding") else "questions"),
                    **{k: (v[k] * factor if v[k] is not None else None) for k in (
                        "physical_calls", "input_tokens", "output_tokens", "thought_tokens", "cached_content_tokens", "service_seconds", "input_items", "input_chars")}}
        summary = dict(status=status, failure=failure, sample_id="conv-26", n=152,
            completed=sum(q["status"] == "ok" for q in query_records), sessions_completed=len(batch_records),
            online_frontiers_completed=sum(x["model_call_made"] for x in online_records),
            online_append_steps_completed=len(online_records), wall_clock_seconds=wall, phases=phases,
            stages=stages, legacy_usage_hooks=hook_usage, memory_batches=batch_records,
            online_frontiers=online_records, new_graph_stats=online_graph.get("stats") if online_graph else None,
            total_generation_service_seconds=sum(c["service_seconds"] for c in calls if c["kind"] == "generation"),
            total_embedding_service_seconds=sum(c["service_seconds"] for c in calls if c["kind"] == "embedding"),
            mathematical_selector="not used by historical 81.75 configuration",
            raw_full_population_counts={k: meta[k] for k in ("total_conversations", "total_questions", "total_sessions", "total_online_frontiers")},
            estimated_full_population_stages=projected,
            extrapolation_note="Build scaled by true session/frontier counts, queries by 1540/152; one conversation is not representative evidence for exact full-run cost or wall time.",
            embedding_usage_note="Missing native batchEmbedContents usage is null, not zero; input items/chars and physical batch calls are measured.")
        save(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("status", "completed", "sessions_completed", "online_frontiers_completed", "wall_clock_seconds", "stages")}, indent=2))
    return 0 if status == "complete" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-model-calls", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error("workers must be 1..8")
    if a.allow_model_calls and not a.output:
        p.error("--output required")
    prepared = preflight()
    if not a.allow_model_calls:
        print(json.dumps(prepared[-1], ensure_ascii=False, indent=2))
        return 0
    return run(a, prepared)


if __name__ == "__main__":
    raise SystemExit(main())

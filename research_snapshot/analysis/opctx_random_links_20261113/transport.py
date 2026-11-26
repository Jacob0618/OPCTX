"""Audited native Gemini transport for the frozen LoCoMo link intervention.

The original chat/embed function bodies are loaded by AST without importing
their mutable global client or its historical response cache. This module
does not initiate calls on import. Credentials remain in the caller's process.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
CLIENT_SOURCE = ROOT / "eval/gemini_client.py"
CLIENT_SHA256 = "0222e860eccae9b0561038c27ca5aae439c6ea088bc0d8b71b1bddc4769612dd"
BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-3-flash-preview"
EMBED_MODEL = "gemini-embedding-001"
GENERATION_MODELS = frozenset((MODEL, JUDGE_MODEL))
MAX_PHYSICAL_ATTEMPTS = 2
RETRY_HTTP = frozenset((429, 503))
STAGES = {
    "creamem3_v2_rerank": "reranker",
    "retrieve": "planner",
    "read": "qa",
    "judge": "judge",
    "embed": "embedding",
    "embed_query": "query_embedding",
    "creamem3_v2_embed_question": "query_embedding",
}


class ExecutionStop(BaseException):
    """Must bypass historical catch-and-fallback and semantic retry loops."""


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False,
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_function(name, namespace):
    """Reuse the same AST-loading approach as the audited E2E entry."""
    source = CLIENT_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != CLIENT_SHA256:
        raise ExecutionStop("Frozen chat/embed source hash changed")
    tree = ast.parse(source.decode("utf-8-sig"))
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(CLIENT_SOURCE), "exec"), namespace)
    return namespace[name]


def _version_matches(expected, actual):
    # Permit provider numeric version suffixes, not Lite/image/other aliases.
    return bool(re.fullmatch(re.escape(expected) + r"(?:-\d+(?:-\d+)*)?", actual))


class AuditTransport:
    def __init__(self, output: Path):
        from creamem3.experiments.memeye_resume_20260921 import network

        self.output = Path(output).resolve()
        if not self.output.is_dir():
            raise ExecutionStop("Runner must create its owned output directory first")
        self.network = network
        self.provider = network.VerifiedProvider()
        self.local = threading.local()
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.calls, self.cache, self.usage = [], {}, {}
        self.request_locks, self.attempts, self.raw_success = {}, {}, {}
        self.next_id = 0
        calls_root = self.output / "calls"
        if calls_root.exists():
            for folder in sorted(calls_root.iterdir()):
                if not folder.is_dir():
                    continue
                match = re.fullmatch(r"(\d+)_[a-z_]+", folder.name)
                if not match:
                    raise ExecutionStop("Unexpected existing call directory")
                self.next_id = max(self.next_id, int(match.group(1)))
                req_path, out_path = folder / "request.json", folder / "outcome.json"
                if not req_path.exists() or not out_path.exists():
                    raise ExecutionStop("Unresolved prior request; explicit audit required")
                req = json.loads(req_path.read_text(encoding="utf-8"))
                out = json.loads(out_path.read_text(encoding="utf-8"))
                key = digest([req["model"], req["request"]])
                if key != out.get("request_key"):
                    raise ExecutionStop("Existing request/outcome binding mismatch")
                self.calls.append(out)
                self.attempts[key] = self.attempts.get(key, 0) + int(out["physical_calls"])
                if out.get("status") == "ok":
                    raw = json.loads((folder / "response.json").read_text(encoding="utf-8"))
                    if digest(raw) != out.get("response_sha256"):
                        raise ExecutionStop("Existing response hash mismatch")
                    self._validate_response(req["model"], req["kind"], req["request"], raw)
                    self.raw_success[key] = raw
                elif not out.get("retry_permitted"):
                    raise ExecutionStop("Prior terminal failure requires explicit review")

    def add(self, tag, prompt_toks, out_toks, calls=1, cached=False,
            model=None, thought_toks=0):
        # Logical client usage; physical outcome records are authoritative for
        # billing, transport retries, and exact-response reuse during resume.
        with self.lock:
            item = self.usage.setdefault(tag, dict(calls=0, cache_hits=0,
                prompt_tokens=0, output_tokens=0, thought_tokens=0, models={}))
            if cached:
                item["cache_hits"] += 1
            else:
                for k, v in (("calls", calls), ("prompt_tokens", prompt_toks),
                             ("output_tokens", out_toks), ("thought_tokens", thought_toks)):
                    item[k] += v
                if model:
                    item["models"][model] = item["models"].get(model, 0) + calls

    def cache_get(self, key):
        with self.lock:
            return copy.deepcopy(self.cache.get(key))

    def cache_put(self, key, value):
        with self.lock:
            self.cache[key] = copy.deepcopy(value)

    def _request_lock(self, key):
        with self.lock:
            return self.request_locks.setdefault(key, threading.Lock())

    def _endpoint(self, url, payload):
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "generativelanguage.googleapis.com":
            raise ExecutionStop("Unexpected provider host or protocol")
        if parts.query not in ("", "key=") or parts.fragment:
            raise ExecutionStop("Unexpected query/fragment; credentials must use native headers")
        prefix = "/v1beta/models/"
        if not parts.path.startswith(prefix):
            raise ExecutionStop("Unexpected provider API path")
        model, separator, operation = parts.path[len(prefix):].partition(":")
        if not separator:
            raise ExecutionStop("Provider operation missing")
        if operation == "generateContent" and model in GENERATION_MODELS:
            kind = "generation"
        elif operation == "batchEmbedContents" and model == EMBED_MODEL:
            kind = "embedding"
            requests = payload.get("requests", [])
            if not requests or any(r.get("model") != "models/" + EMBED_MODEL
                                   or r.get("outputDimensionality") != 768
                                   for r in requests):
                raise ExecutionStop("Unexpected embedding model/dimension")
        else:
            raise ExecutionStop("Unexpected provider model or operation")
        return model, kind, BASE + "/models/" + model + ":" + operation

    def _validate_response(self, model, kind, payload, raw):
        if not isinstance(raw, dict):
            raise ExecutionStop("Provider response is not an object")
        usage = raw.get("usageMetadata")
        if kind == "generation":
            if not _version_matches(model, str(raw.get("modelVersion", ""))):
                raise ExecutionStop("Provider returned a different model version")
            candidates = raw.get("candidates") or []
            if (raw.get("promptFeedback", {}).get("blockReason")
                    or len(candidates) != 1 or candidates[0].get("finishReason") != "STOP"):
                raise ExecutionStop("Blocked or non-STOP response; no automatic retry")
            text = "".join(p.get("text", "") for p in
                           candidates[0].get("content", {}).get("parts", []))
            if not text.strip():
                raise ExecutionStop("Empty generation; no automatic retry")
            if not isinstance(usage, dict) or "promptTokenCount" not in usage:
                raise ExecutionStop("Missing generation usage")
        else:
            vectors = raw.get("embeddings", [])
            if len(vectors) != len(payload["requests"]):
                raise ExecutionStop("Embedding result count mismatch")
            for entry in vectors:
                values = entry.get("values", [])
                if len(values) != 768 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                    raise ExecutionStop("Embedding vector shape/value mismatch")
            # Native embedding endpoint often omits a modelVersion field.
            if raw.get("modelVersion") and not _version_matches(model, raw["modelVersion"]):
                raise ExecutionStop("Embedding response model changed")

    def request(self, url, payload):
        try:
            model, kind, endpoint = self._endpoint(url, payload)
        except ExecutionStop:
            self.stop.set()
            raise
        key = digest([model, payload])
        with self._request_lock(key):
            if self.stop.is_set():
                raise ExecutionStop("Stopped before request")
            with self.lock:
                if key in self.raw_success:
                    return copy.deepcopy(self.raw_success[key])
            while True:
                if self.stop.is_set():
                    raise ExecutionStop("Stopped before request")
                stage = str(getattr(self.local, "stage", kind))
                if not re.fullmatch(r"[a-z_]+", stage):
                    self.stop.set()
                    raise ExecutionStop("Invalid ledger stage")
                scope = str(getattr(self.local, "scope", "shared"))
                with self.lock:
                    if self.attempts.get(key, 0) >= MAX_PHYSICAL_ATTEMPTS:
                        self.stop.set()
                        raise ExecutionStop("Exact-request physical attempt limit reached")
                    attempt = self.attempts.get(key, 0) + 1
                    self.next_id += 1
                    number = self.next_id
                folder = self.output / "calls" / f"{number:06d}_{stage}"
                outcome = dict(id=number, scope=scope, stage=stage, kind=kind, model=model,
                    request_key=key, logical_attempt=attempt, status="pending", attempted=False,
                    physical_calls=0, usage=None, retry_permitted=False,
                    started_utc=datetime.now(timezone.utc).isoformat())
                start = time.perf_counter()
                retry = False
                try:
                    if shutil.disk_usage(self.output.anchor).free < 15 * 1024**3:
                        raise ExecutionStop("Disk free below 15 GiB; no new calls")
                    if not os.environ.get("GEMINI_API_KEY"):
                        raise ExecutionStop("Gemini credential missing")
                    save(folder / "request.json", dict(scope=scope, stage=stage, kind=kind,
                        model=model, endpoint=endpoint, request=payload,
                        request_sha256=digest(payload)))
                    outcome.update(attempted=True, physical_calls=1)
                    with self.lock:
                        self.attempts[key] = attempt
                    if kind == "generation":
                        raw = self.provider.generate_content(model=model, request=payload)
                    else:
                        try:
                            response = self.network.client().post(endpoint,
                                content=self.network.g._canonical_json(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json",
                                         "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
                        except (self.network.httpx.ConnectError, self.network.httpx.ConnectTimeout) as exc:
                            raise self.network.g.ExplicitProviderError("Pre-send connection failure",
                                retryable=True, status_code=None) from exc
                        except self.network.httpx.TransportError as exc:
                            raise self.network.g.IndeterminateProviderError("Indeterminate transport outcome") from exc
                        if not 200 <= response.status_code < 300:
                            raise self.network.g.ExplicitProviderError("Embedding HTTP failure",
                                retryable=response.status_code in RETRY_HTTP,
                                status_code=response.status_code)
                        raw = response.json()
                    save(folder / "response.json", raw)
                    outcome.update(http_status=200, usage=raw.get("usageMetadata"),
                                   response_sha256=digest(raw))
                    if kind == "generation":
                        candidates = raw.get("candidates") or []
                        outcome.update(model_version=raw.get("modelVersion"),
                            finish_reason=candidates[0].get("finishReason") if candidates else None)
                    else:
                        outcome.update(input_items=len(payload["requests"]),
                            input_chars=sum(len(p.get("text", "")) for r in payload["requests"]
                                            for p in r["content"]["parts"]),
                            token_usage_known=bool(outcome["usage"]),
                            model_version_verifiable=bool(raw.get("modelVersion")))
                    self._validate_response(model, kind, payload, raw)
                    outcome["status"] = "ok"
                    with self.lock:
                        self.raw_success[key] = copy.deepcopy(raw)
                    return raw
                except self.network.g.ExplicitProviderError as exc:
                    known_connect, unsafe = False, False
                    if exc.status_code is None:
                        cause = exc.__cause__
                        text = str(cause).lower()
                        known_connect = type(cause).__name__ in ("ConnectError", "ConnectTimeout")
                        unsafe = any(word in text for word in
                            ("certificate", "ssl", "tls", "permission", "access denied", "forbidden"))
                        outcome.update(connection_cause_type=type(cause).__name__,
                            connection_classification="tls_or_permission_stop" if unsafe else
                            ("pre_send_connect_failure" if known_connect else "unclassified_stop"))
                    eligible = bool(exc.retryable) and (exc.status_code in RETRY_HTTP or
                                                       (known_connect and not unsafe))
                    retry = eligible and attempt < MAX_PHYSICAL_ATTEMPTS
                    outcome.update(status="explicit_provider_error", http_status=exc.status_code,
                        retryable=eligible, retry_permitted=retry,
                        possible_indeterminate_billing=not known_connect,
                        provider_request_reached=False if known_connect else None)
                    if not retry:
                        self.stop.set()
                        raise ExecutionStop("Provider failure; preserved evidence, no further retry") from None
                except ExecutionStop:
                    outcome["status"] = "stopped"
                    self.stop.set()
                    raise
                except BaseException as exc:
                    outcome.update(status="transport_or_local_failure", error_type=type(exc).__name__,
                                   possible_indeterminate_billing=outcome["attempted"])
                    self.stop.set()
                    raise ExecutionStop("Unexpected/indeterminate failure; no automatic retry") from None
                finally:
                    outcome["service_seconds"] = time.perf_counter() - start
                    try:
                        save(folder / "outcome.json", outcome)
                    except BaseException:
                        self.stop.set()
                        raise ExecutionStop("Unable to persist outcome; audit before resuming") from None
                    with self.lock:
                        self.calls.append(copy.deepcopy(outcome))
                if retry:
                    # This wait is bounded and wakes immediately on another worker's stop.
                    if self.stop.wait(1.0):
                        raise ExecutionStop("Stopped before the one permitted retry")

    def snapshots(self):
        with self.lock:
            return copy.deepcopy(self.calls), copy.deepcopy(self.usage)


def make_clients(transport: AuditTransport):
    """Return the frozen (chat, embed) bodies with only transport/cache bindings."""
    if os.environ.get("GEMINI3_THINKING_LEVEL", "minimal") != "minimal":
        raise ExecutionStop("Historical Judge requires the minimal thinking level")
    namespace = dict(json=json, hashlib=hashlib, os=os, re=re,
        BACKBONE_MODEL=MODEL, EMBED_MODEL=EMBED_MODEL, BACKBONE_BASE_URL="",
        NO_CHAT_CACHE=False, USAGE=transport, BASE=BASE, API_KEY="",
        _post=transport.request, _cache_get=transport.cache_get,
        _cache_put=transport.cache_put)
    _load_function("_valid_judge_text", namespace)
    _load_function("_valid_cached_chat", namespace)
    raw_chat = _load_function("chat", namespace)
    raw_embed = _load_function("embed", namespace)

    def chat(prompt, **kwargs):
        tag = kwargs.get("tag", "read")
        transport.local.stage = STAGES.get(tag, "generation")
        # The original body still controls prompt, output cap, thinking, and
        # bounded format retries. The transport never retries a non-STOP result.
        return raw_chat(prompt, **kwargs)

    def embed(texts, **kwargs):
        transport.local.stage = STAGES.get(kwargs.get("tag", "embed_query"), "embedding")
        return raw_embed(texts, **kwargs)

    return chat, embed

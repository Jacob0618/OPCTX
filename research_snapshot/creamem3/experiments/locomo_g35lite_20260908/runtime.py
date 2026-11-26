"""Isolated official Gemini transport: hashed write-once receipts, no secret URLs."""
from __future__ import annotations
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BACKBONE = "gemini-3.5-flash-lite"
JUDGE = "gemini-3.5-flash"
EMBED = "gemini-embedding-001"
BASE = "https://generativelanguage.googleapis.com/v1beta"
TRANSIENT = {429, 500, 502, 503, 504}

class StopRun(BaseException):
    """Bypasses legacy model-error fallbacks for transport/integrity stop conditions."""

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(record) != {"payload", "sha256"} or sha(record["payload"]) != record["sha256"]:
        raise StopRun(f"Artifact hash mismatch: {path}")
    return record["payload"]

def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise StopRun(f"Refusing to overwrite artifact: {path}")
        return
    temporary = path.with_name(path.name + ".pending")
    if temporary.exists():
        raise StopRun(f"Incomplete artifact requires inspection: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical({"payload": value, "sha256": sha(value)}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)

def request_body(prompt, max_tokens, json_mode=False, system=None):
    generation = {"maxOutputTokens": int(max_tokens), "thinkingConfig": {"thinkingLevel": "minimal"}}
    if json_mode:
        generation["responseMimeType"] = "application/json"
    result = {"contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
              "generationConfig": generation}
    if system:
        result["systemInstruction"] = {"parts": [{"text": str(system)}]}
    return result

def response_text(response):
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    return "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
                   if not part.get("thought", False))

class Runtime:
    def __init__(self, root, *, allow_model_calls=False, concurrency=8, timeout=180):
        self.root = Path(root)
        self.allow = allow_model_calls
        self.gate = threading.BoundedSemaphore(concurrency)
        self.timeout = timeout
        self.stopped = threading.Event()
        self.mutex = threading.Lock()
        self.locks = {}
        self.physical_count = 0
        self.events = self.root / "events.jsonl"

    def log(self, event):
        event = {"utc": now(), **event}
        with self.mutex:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical(event) + "\n")
                handle.flush()

    def _halt(self, message):
        self.stopped.set()
        raise StopRun(message)

    def post(self, stage, scope, model, method, body):
        identity = {"stage": stage, "scope": scope, "model": model, "method": method, "body": body}
        key = sha(identity)
        directory = self.root / "provider" / stage / key
        with self.mutex:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            final_path = directory / "final.json"
            if final_path.exists():
                final = read(final_path)
                if final["request_sha256"] != key or final["identity"] != identity:
                    self._halt("Cached request contract mismatch")
                raw = read(directory / f"attempt_{final['attempt']:02d}.response.json")
                if sha(raw) != final["response_sha256"]:
                    self._halt("Response receipt drift")
                self.log({"kind": "cache_hit", "stage": stage, "scope": scope, "request": key})
                return raw["response"]
            if not self.allow:
                self._halt(f"Model calls disabled; uncached {stage}/{scope}")
            if self.stopped.is_set():
                raise StopRun("Dispatch already stopped")
            seal(directory / "request.json", identity)
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                self._halt("No configured Gemini key")
            for attempt in range(1, 5):
                prefix = directory / f"attempt_{attempt:02d}"
                intent_path = prefix.with_suffix(".intent.json")
                response_path = prefix.with_suffix(".response.json")
                error_path = prefix.with_suffix(".error.json")
                if intent_path.exists():
                    if response_path.exists():
                        raw = read(response_path)
                        seal(final_path, {"identity": identity, "request_sha256": key, "attempt": attempt,
                                          "response_sha256": sha(raw)})
                        return raw["response"]
                    if error_path.exists():
                        error = read(error_path)
                        if not error["retryable"]:
                            self._halt(f"Prior nonretryable failure {stage}/{scope}")
                        continue
                    self._halt(f"Orphan physical request requires read-only investigation: {intent_path}")
                with self.gate:
                    if self.stopped.is_set():
                        raise StopRun("Dispatch stopped before next request")
                    with self.mutex:
                        self.physical_count += 1
                        if self.physical_count > 20000:
                            self.stopped.set()
                            raise StopRun("Physical-call safety ceiling reached")
                    seal(intent_path, {"request_sha256": key, "attempt": attempt, "utc": now()})
                    self.log({"kind": "physical_start", "stage": stage, "scope": scope,
                              "request": key, "attempt": attempt, "model": model})
                    started = time.perf_counter()
                    request = urllib.request.Request(
                        f"{BASE}/models/{model}:{method}",
                        data=canonical(body).encode("utf-8"),
                        headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
                    try:
                        with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                            response = json.loads(handle.read().decode("utf-8"))
                        raw = {"response": response, "elapsed_s": time.perf_counter()-started, "utc": now()}
                        seal(response_path, raw)
                        seal(final_path, {"identity": identity, "request_sha256": key, "attempt": attempt,
                                          "response_sha256": sha(raw)})
                        self.log({"kind": "physical_success", "stage": stage, "scope": scope,
                                  "request": key, "attempt": attempt, "model": model,
                                  "elapsed_s": raw["elapsed_s"], "usage": response.get("usageMetadata", {})})
                        return response
                    except urllib.error.HTTPError as exc:
                        retryable = exc.code in TRANSIENT
                        error = {"status": exc.code, "type": "HTTPError", "retryable": retryable,
                                 "elapsed_s": time.perf_counter()-started, "utc": now()}
                        seal(error_path, error)
                        self.log({"kind": "physical_error", "stage": stage, "scope": scope,
                                  "request": key, "attempt": attempt, **error})
                        if not retryable:
                            self._halt(f"Nonretryable HTTP {exc.code}: {stage}/{scope}; no model substitution")
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        error = {"status": None, "type": type(exc).__name__, "retryable": True,
                                 "acceptance_unknown": True, "elapsed_s": time.perf_counter()-started, "utc": now()}
                        seal(error_path, error)
                        self.log({"kind": "physical_error", "stage": stage, "scope": scope,
                                  "request": key, "attempt": attempt, **error})
                    except Exception as exc:
                        self._halt(f"Unexpected provider/receipt failure {type(exc).__name__} at {directory}")
                if attempt < 4:
                    if self.stopped.wait(min(30, 2**attempt)):
                        raise StopRun("Dispatch stopped during retry backoff")
            self._halt(f"Four physical attempts exhausted: {stage}/{scope}")

    def chat(self, scope, stage, prompt, *, model=BACKBONE, max_tokens=8192, json_mode=False, system=None):
        if stage == "judge" and model != JUDGE:
            self._halt("Judge model mismatch")
        if stage != "judge" and model != BACKBONE:
            self._halt("Non-Judge model mismatch")
        result = self.post(stage, scope, model, "generateContent",
                           request_body(prompt, max_tokens, json_mode, system))
        text = response_text(result)
        if not text.strip():
            self._halt(f"Empty model response: {stage}/{scope}")
        return text

    def chat_fn(self, scope, stage=None):
        def call(prompt, **kwargs):
            tag = kwargs.get("tag", "")
            mapped = stage or {"creamem3_v2_write": "writer", "creamem3_v3_online_fuse": "online",
                               "creamem3_v2_rerank": "rerank", "retrieve": "planner"}.get(tag)
            if mapped is None:
                self._halt(f"Unregistered generation stage: {tag}")
            return self.chat(scope, mapped, prompt, max_tokens=kwargs.get("max_tokens", 8192),
                             json_mode=kwargs.get("json_mode", False), system=kwargs.get("system"))
        return call

    def embed(self, texts, *, task_type="RETRIEVAL_DOCUMENT", tag="embedding", **_):
        vectors = []
        for start in range(0, len(texts), 100):
            chunk = [str(text) for text in texts[start:start+100]]
            body = {"requests": [{"model": f"models/{EMBED}", "content": {"parts": [{"text": text}]},
                                  "taskType": task_type, "outputDimensionality": 768} for text in chunk]}
            result = self.post("embedding", task_type + ":" + sha(chunk), EMBED, "batchEmbedContents", body)
            batch = result.get("embeddings", [])
            if len(batch) != len(chunk):
                self._halt("Embedding response count mismatch")
            for item in batch:
                vector = item.get("values", [])
                if len(vector) != 768 or not all(math.isfinite(v) for v in vector) or sum(v*v for v in vector) <= 0:
                    self._halt("Bad embedding: dimension/nonfinite/zero")
                vectors.append(vector)
        return vectors


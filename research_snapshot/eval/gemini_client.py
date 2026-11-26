"""Gemini REST client: retries, disk cache, token accounting, thread-pool map.

All systems (ours + baselines) must route through this module so that
backbone/judge/embedding usage is uniform and token-accounted.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.environ.get(
    "AGENT_MEMORY_CACHE_ROOT", os.path.join(ROOT, "results", "cache"))
BASE_CACHE_PATH = os.environ.get("AGENT_MEMORY_BASE_CACHE", "")


def _load_env():
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
BACKBONE_MODEL = os.environ.get("BACKBONE_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3-flash-preview")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
BASE = "https://generativelanguage.googleapis.com/v1beta"

# Alternate generative backbone (DeepSeek / Qwen, OpenAI-compatible). When set,
# every non-gemini chat() call routes here; judge (gemini) and embeddings
# (gemini-embedding-001) stay on Gemini so only the generative backbone varies.
BACKBONE_BASE_URL = os.environ.get("BACKBONE_BASE_URL", "").rstrip("/")
BACKBONE_API_KEY = os.environ.get("BACKBONE_API_KEY", "")
BACKBONE_THINKING = os.environ.get("BACKBONE_THINKING", "off")  # off|on
# Disable CHAT caching (embeddings still cached) for clean, contention-free
# cold runs with exact token accounting under high process concurrency.
NO_CHAT_CACHE = bool(os.environ.get("LLM_NO_CHAT_CACHE"))
_cache_lock = threading.Lock()
_cache_conn = None
_base_cache_conn = None


def _valid_judge_text(text):
    tags = re.findall(r"\[\[(yes|no)\]\]", (text or "").lower())
    return len(tags) == 1


def _valid_cached_chat(tag, hit):
    """Reject cached judge text that the audited binary parser cannot use."""
    if hit is None or tag != "judge":
        return hit is not None
    return _valid_judge_text(hit.get("text", ""))


def _cache():
    global _cache_conn
    if _cache_conn is None:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        _cache_conn = sqlite3.connect(
            os.path.join(CACHE_ROOT, "llm_cache.sqlite"),
            check_same_thread=False, timeout=60)
        _cache_conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        _cache_conn.commit()
    return _cache_conn


def _base_cache():
    """Open the optional immutable cache beneath the per-run overlay."""
    global _base_cache_conn
    if not BASE_CACHE_PATH:
        return None
    if _base_cache_conn is None:
        uri = Path(BASE_CACHE_PATH).resolve().as_uri() + "?mode=ro"
        _base_cache_conn = sqlite3.connect(
            uri, uri=True, check_same_thread=False, timeout=60)
    return _base_cache_conn


def _cache_get(key):
    with _cache_lock:
        row = _cache().execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        base = _base_cache()
        if row is None and base is not None:
            row = base.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(key, value):
    with _cache_lock:
        _cache().execute("INSERT OR REPLACE INTO kv VALUES (?,?)",
                         (key, json.dumps(value)))
        _cache().commit()


class Usage:
    """Global token accounting, bucketed by tag.

    The eval harness uses 'read' for final answer generation. ChronoHG's
    retrieval-time planner calls use 'retrieve' so QA cost can be reported
    separately from query expansion.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}

    def add(self, tag, prompt_toks, out_toks, calls=1, cached=False,
            model=None, thought_toks=0):
        with self.lock:
            d = self.data.setdefault(tag, {
                "prompt_tokens": 0, "output_tokens": 0,
                "thought_tokens": 0, "calls": 0, "cache_hits": 0,
                "models": {}})
            if model:
                d["models"][str(model)] = d["models"].get(str(model), 0) + calls
            if cached:
                d["cache_hits"] += 1
            else:
                d["prompt_tokens"] += prompt_toks
                d["output_tokens"] += out_toks
                d["thought_tokens"] += thought_toks
                d["calls"] += calls

    def snapshot(self):
        with self.lock:
            return json.dumps(self.data)


USAGE = Usage()


def _post(url, payload, max_retries=12):
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                detail = e.read().decode("utf-8")[:500]
            except Exception:
                detail = ""
            last_err = f"HTTP {code}: {detail}"
            if code in (429, 500, 502, 503, 504):
                sleep = min(60, (2 ** attempt) + random.random() * 2)
                time.sleep(sleep)
                continue
            raise RuntimeError(last_err)
        except Exception as e:  # network glitch / timeout
            last_err = str(e)
            time.sleep(min(60, (2 ** attempt) + random.random() * 2))
    raise RuntimeError(f"gemini call failed after {max_retries} retries: {last_err}")


def _post_auth(url, payload, api_key, max_retries=12):
    """POST to an OpenAI-compatible endpoint with a Bearer key (DeepSeek/Qwen)."""
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                detail = e.read().decode("utf-8")[:500]
            except Exception:
                detail = ""
            last_err = f"HTTP {code}: {detail}"
            if code in (429, 500, 502, 503, 504):
                time.sleep(min(60, (2 ** attempt) + random.random() * 2))
                continue
            raise RuntimeError(last_err)
        except Exception as e:  # network glitch / timeout
            last_err = str(e)
            time.sleep(min(60, (2 ** attempt) + random.random() * 2))
    raise RuntimeError(f"backbone call failed after {max_retries} retries: {last_err}")


def _apply_backbone_params(model, payload):
    """Disable provider 'thinking' (frozen no-think protocol) and clamp output."""
    m = model.lower()
    if BACKBONE_THINKING != "on":
        if "deepseek" in m:
            payload["thinking"] = {"type": "disabled"}
        elif "qwen3" in m:
            payload["enable_thinking"] = False
    if payload.get("max_tokens", 0) > 8192:  # small backbones cap output
        payload["max_tokens"] = 8192


def _openai_chat(prompt, model, system, temperature, max_tokens, tag,
                 use_cache, json_mode):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    use_cache = use_cache and not NO_CHAT_CACHE
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    _apply_backbone_params(model, payload)
    key = hashlib.sha256(json.dumps(
        ["ochat.v1", BACKBONE_BASE_URL, payload], sort_keys=True).encode()).hexdigest()
    if use_cache:
        hit = _cache_get(key)
        if _valid_cached_chat(tag, hit):
            USAGE.add(tag, 0, 0, cached=True)
            return hit["text"]
    resp = _post_auth(BACKBONE_BASE_URL + "/chat/completions", payload,
                      BACKBONE_API_KEY)
    text = ""
    try:
        text = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        text = ""
    um = resp.get("usage", {}) or {}
    USAGE.add(tag, um.get("prompt_tokens", 0), um.get("completion_tokens", 0),
              model=resp.get("model", model))
    if use_cache and text:
        _cache_put(key, {"text": text})
    return text


def chat(prompt, model=None, system=None, temperature=0.0, max_tokens=8192,
         thinking_budget=0, tag="read", use_cache=True, json_mode=False):
    """Single-turn generateContent. thinking_budget=0 disables thinking
    (applies to gemini-2.5-flash; judge model ignores unsupported budget 0
    by us simply not sending thinkingConfig for it). Non-gemini models route
    to the OpenAI-compatible alternate backbone when BACKBONE_BASE_URL is set."""
    model = model or BACKBONE_MODEL
    if BACKBONE_BASE_URL and not model.startswith("gemini"):
        return _openai_chat(prompt, model, system, temperature, max_tokens,
                            tag, use_cache, json_mode)
    use_cache = use_cache and not NO_CHAT_CACHE
    gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    if model.startswith("gemini-2.5") and thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    elif model.startswith("gemini-3") and thinking_budget is not None:
        # Gemini 3 does not support a literal zero-token thinking budget.
        # ``minimal`` is its documented lowest-effort setting and is the
        # closest supported equivalent to the frozen no-thinking protocol.
        gen["thinkingConfig"] = {
            "thinkingLevel": os.environ.get(
                "GEMINI3_THINKING_LEVEL", "minimal")
        }
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": gen}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    key = hashlib.sha256(json.dumps(
        ["chat.v1", model, payload], sort_keys=True).encode()).hexdigest()
    if use_cache:
        hit = _cache_get(key)
        if _valid_cached_chat(tag, hit):
            USAGE.add(tag, 0, 0, cached=True)
            return hit["text"]
    # A protocol-invalid binary verdict is a failed provider response, not a
    # negative label. Retry the identical prompt without touching valid calls.
    attempts = 5 if tag == "judge" else 1
    text = ""
    for attempt in range(attempts):
        if tag == "judge" and attempt:
            # Gemini 3 counts hidden thought tokens against maxOutputTokens.
            # Escalate only after a truncated/ambiguous binary response.
            gen["maxOutputTokens"] = min(8192, max_tokens * (2 ** attempt))
        resp = _post(f"{BASE}/models/{model}:generateContent?key={API_KEY}", payload)
        try:
            parts = resp["candidates"][0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            text = ""
        um = resp.get("usageMetadata", {})
        thought_toks = um.get("thoughtsTokenCount", 0) or 0
        USAGE.add(tag, um.get("promptTokenCount", 0),
                  um.get("candidatesTokenCount", 0) + thought_toks,
                  model=resp.get("modelVersion", model),
                  thought_toks=thought_toks)
        if tag != "judge" or _valid_judge_text(text):
            break
    if use_cache and text:
        _cache_put(key, {"text": text})
    return text


def count_tokens(prompt, model=None, system=None):
    """Count the exact Gemini input tokens without generating a response.

    This mirrors the content/system layout used by :func:`chat` and calls the
    provider's countTokens endpoint.  It deliberately does not touch the
    generative usage ledger because no completion is requested.
    """
    model = model or BACKBONE_MODEL
    if not model.startswith("gemini"):
        raise ValueError("exact prompt-only counting currently requires Gemini")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    resp = _post(f"{BASE}/models/{model}:countTokens?key={API_KEY}", payload)
    return int(resp.get("totalTokens", 0))


def embed(texts, model=None, tag="embed", task_type="RETRIEVAL_DOCUMENT",
          dim=768, use_cache=True):
    """Batch embedding with per-text caching. Returns list of vectors."""
    model = model or EMBED_MODEL
    out = [None] * len(texts)
    todo = []
    keys = []
    for i, t in enumerate(texts):
        k = hashlib.sha256(json.dumps(
            ["emb.v1", model, task_type, dim, t]).encode()).hexdigest()
        keys.append(k)
        hit = _cache_get(k) if use_cache else None
        if hit is not None:
            out[i] = hit["v"]
            USAGE.add(tag, 0, 0, cached=True)
        else:
            todo.append(i)
    for s in range(0, len(todo), 100):  # batchEmbedContents limit
        batch = todo[s:s + 100]
        payload = {"requests": [{
            "model": f"models/{model}",
            "content": {"parts": [{"text": texts[i]}]},
            "taskType": task_type,
            "outputDimensionality": dim} for i in batch]}
        resp = _post(f"{BASE}/models/{model}:batchEmbedContents?key={API_KEY}",
                     payload)
        for i, e in zip(batch, resp["embeddings"]):
            out[i] = e["values"]
            USAGE.add(tag, 0, 0)
            if use_cache:
                _cache_put(keys[i], {"v": e["values"]})
    return out


def pmap(fn, items, workers=8, desc=""):
    """Ordered concurrent map with simple progress printing."""
    results = [None] * len(items)
    done = [0]
    lock = threading.Lock()

    def run(i):
        results[i] = fn(items[i])
        with lock:
            done[0] += 1
            if desc and (done[0] % 25 == 0 or done[0] == len(items)):
                print(f"  [{desc}] {done[0]}/{len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, range(len(items))))
    return results

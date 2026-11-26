"""Small Gemini REST client; no embedded credentials or private endpoints."""
import base64
import json
import mimetypes
import os
from pathlib import Path
import numpy as np
import requests

class Gemini:
    def __init__(self, config):
        self.config = config
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise ValueError("Set GEMINI_API_KEY in your environment.")
        self.base = "https://generativelanguage.googleapis.com/v1beta"
        self.http = requests.Session()

    def _post(self, model, method, payload):
        response = self.http.post(
            f"{self.base}/models/{model}:{method}", json=payload,
            headers={"x-goog-api-key": self.key}, timeout=(20, 300))
        if not response.ok:
            # Do not print credentials, request bodies, or remote echo text.
            raise RuntimeError(f"Gemini {method}: HTTP {response.status_code}")
        return response.json()

    def generate(self, prompt, *, json_output=True, max_tokens=4096, image=None):
        parts = [{"text": prompt}]
        if image is not None:
            path = Path(image)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                raise ValueError("Supported images: JPEG, PNG, WebP")
            parts.append({"inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
        generation = {
            "temperature": self.config.temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": self.config.thinking_budget}}
        if json_output:
            generation["responseMimeType"] = "application/json"
        result = self._post(self.config.model, "generateContent", {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation})
        candidates = result.get("candidates", [])
        if not candidates:
            reason = result.get("promptFeedback", {}).get("blockReason", "NO_CANDIDATE")
            raise RuntimeError(f"Generation stopped: {reason}")
        candidate = candidates[0]
        if candidate.get("finishReason") != "STOP":
            raise RuntimeError(f"Generation stopped: {candidate.get('finishReason')}")
        text = "".join(p.get("text", "") for p in
                       candidate.get("content", {}).get("parts", [])
                       if not p.get("thought"))
        if not text.strip():
            raise RuntimeError("Empty model output")
        if not json_output:
            return text.strip()
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Expected one JSON object")
        return value

    def embed(self, texts, *, query=False):
        if not texts:
            return []
        model = self.config.embedding_model
        vectors = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            result = self._post(model, "batchEmbedContents", {"requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": text}]},
                 "taskType": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                 "outputDimensionality": self.config.embedding_dim}
                for text in batch]})
            rows = result.get("embeddings", [])
            if len(rows) != len(batch):
                raise ValueError("Embedding batch size mismatch")
            vectors.extend(row["values"] for row in rows)
        matrix = np.asarray(vectors, dtype=float)
        if matrix.shape != (len(texts), self.config.embedding_dim):
            raise ValueError("Embedding dimension mismatch")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Nonfinite or zero embedding")
        return (matrix / norms).tolist()

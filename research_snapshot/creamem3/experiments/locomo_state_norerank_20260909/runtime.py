"""Reject every model reranking call, including accidental legacy fallbacks."""
import json
from creamem3.experiments.locomo_online_only_20260909.runtime import (
    Runtime as ParentRuntime, BACKBONE, JUDGE, StopRun, read, seal, sha,
    file_sha, now, response_text)


class Runtime(ParentRuntime):
    def post(self, stage, scope, model, method, body):
        expected = {"qa": BACKBONE, "online": BACKBONE, "judge": JUDGE,
                    "embedding": "gemini-embedding-001"}
        if stage not in expected or model != expected[stage]:
            self._halt("Forbidden stage/model in no-reranker study: " + stage)
        result = super().post(stage, scope, model, method, body)
        if stage == "online":
            candidates = result.get("candidates", [])
            if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
                self._halt("Online incomplete generation: " + scope)
            try:
                value = json.loads(response_text(result))
            except (ValueError, TypeError):
                self._halt("Online malformed JSON: " + scope)
            if not isinstance(value, dict) or not isinstance(value.get("operations"), list):
                self._halt("Online invalid operation contract: " + scope)
        return result

    def structured_online(self, scope, prompt, schema):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16384,
                    "thinkingConfig": {"thinkingBudget": 0},
                    "responseMimeType": "application/json", "responseSchema": schema}}
        self.log({"kind": "logical_request", "stage": "online", "scope": scope,
                  "model": BACKBONE, "body_sha256": sha(body)})
        response = self.post("online", scope, BACKBONE, "generateContent", body)
        return json.loads(response_text(response))

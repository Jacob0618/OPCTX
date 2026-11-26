"""2.5-only generation over the already audited, write-once Gemini transport."""
from creamem3.experiments.locomo_g35lite_20260908.runtime import (
    Runtime as Transport, StopRun, canonical, file_sha, now, read, seal, sha, response_text)

BACKBONE = "gemini-2.5-flash"
JUDGE = "gemini-3-flash-preview"

class Runtime(Transport):
    def chat(self, scope, stage, prompt, *, model=None, max_tokens=8192, json_mode=False, system=None):
        target = JUDGE if stage == "judge" else BACKBONE
        if model is not None and model != target:
            self._halt("Unexpected model in isolated2.5 experiment")
        generation = {"temperature": 0.0, "maxOutputTokens": int(max_tokens),
                      "thinkingConfig": {"thinkingLevel": "minimal"} if stage == "judge" else {"thinkingBudget": 0}}
        if json_mode:
            generation["responseMimeType"] = "application/json"
        body = {"contents": [{"role": "user", "parts": [{"text": str(prompt)}]}], "generationConfig": generation}
        if system:
            body["systemInstruction"] = {"parts": [{"text": str(system)}]}
        # Identical rerank / QA / judge requests across arms share one result.
        # Explicit repeats use a separate Runtime root, not an arm-dependent
        # source of noise. All aliases are recorded in events.jsonl.
        physical_scope = "shared:" + sha(body) if stage in {"rerank", "qa", "judge"} else scope
        self.log({"kind": "logical_request", "scope": scope, "physical_scope": physical_scope,
                  "stage": stage, "model": target, "body_sha256": sha(body)})
        result = self.post(stage, physical_scope, target, "generateContent", body)
        text = response_text(result)
        if not text.strip():
            self._halt("Empty model response: " + stage + "/" + scope)
        return text

    def chat_fn(self, scope, stage=None):
        def call(prompt, **kwargs):
            tag = kwargs.get("tag", "")
            mapped = stage or {"creamem3_v2_write": "writer", "creamem3_v3_online_fuse": "online",
                               "creamem3_v2_rerank": "rerank", "retrieve": "planner"}.get(tag)
            if mapped is None:
                self._halt("Unregistered generation stage: " + tag)
            return self.chat(scope, mapped, prompt, max_tokens=kwargs.get("max_tokens", 8192),
                             json_mode=kwargs.get("json_mode", False), system=kwargs.get("system"))
        return call

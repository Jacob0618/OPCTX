"""Offline transport smoke test. Never opens a network connection."""
from pathlib import Path
import importlib.util
import json
import os
import tempfile
import types

path = Path(__file__).with_name("locomo_e2e.py")
spec = importlib.util.spec_from_file_location("locomo_cost_mock_target", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
prepared = m.preflight()
records = {r["session_id"]: r for r in prepared[0]["records"]}
base_ledger = m.Ledger

class MockLedger(base_ledger):
    def __init__(self, output):
        super().__init__(output)
        self.provider = self
        self.network = types.SimpleNamespace(client=lambda: self, g=self.network.g)

    def generate_content(self, *, model, request):
        stage = self.local.stage
        prompt = request["contents"][0]["parts"][0]["text"]
        if stage == "writer":
            sid = int(self.local.scope.rsplit("_", 1)[1])
            rec = records[sid]
            response = {"batch_summary": f"Offline fixture session {sid}", "atomic_facts": [
                {"fact": f"Offline fixture fact in session {sid}", "source_dia_ids": [rec["turns"][0]["dia_id"]]}]}
        elif stage == "online":
            response = {"operations": []}
        elif stage == "reranker":
            candidates = json.loads(prompt.split("Candidate facts JSON:\n", 1)[1])
            response = {"ranked_fact_ids": [r["fact_id"] for r in candidates[:20]]}
        elif stage == "planner":
            response = {"operation": "atomic_lookup", "scope": "one_entity_state", "evidence_cardinality": "one",
                        "temporal_mode": "none", "abstraction": "concrete", "answer_shape": "value", "confidence": 1.0}
        elif stage == "qa":
            response = "Offline fixture answer, not an experiment result."
        else:
            raise AssertionError(stage)
        text = json.dumps(response) if isinstance(response, dict) else response
        return {"modelVersion": "gemini-2.5-flash", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
                "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3, "totalTokenCount": 14}}

    def post(self, url, *, content, headers):
        assert url.endswith(":batchEmbedContents"), url
        payload = json.loads(content)
        raw = {"embeddings": [{"values": [1.0] + [0.0] * 767} for _ in payload["requests"]]}
        return types.SimpleNamespace(status_code=200, json=lambda: raw)

m.Ledger = MockLedger
os.environ["GEMINI_API_KEY"] = "OFFLINE_MOCK_NOT_A_KEY"
out = Path(tempfile.mkdtemp(prefix="opctx_e2e_offline_")) / "run"
code = m.run(types.SimpleNamespace(output=out, workers=4), prepared)
summary = m.read(out / "summary.json")
assert code == 0 and summary["completed"] == 152, summary
assert summary["sessions_completed"] == 19 and summary["online_frontiers_completed"] == 18
assert summary["stages"]["writer"]["physical_calls"] == 19
assert summary["stages"]["online"]["physical_calls"] == 18
assert summary["stages"]["write_embedding"]["input_tokens"] is None
print("OFFLINE_MOCK_PASS " + str(out))

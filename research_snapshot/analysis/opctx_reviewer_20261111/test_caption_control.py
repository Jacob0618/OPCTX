"""Offline tests. Provider transport is replaced; no credentials or model calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import caption_control as c


class OfflineControlTests(unittest.TestCase):
    def test_public_caption_census_and_verbatim(self):
        reg = c.b.read(c.SOURCE / "registration.json")
        total = 0
        for task, entry in reg["tasks"].items():
            self.assertEqual(c.b.file_sha(entry["path"]), entry["sha256"])
            history = c.b.read(entry["path"])
            captions = c.public_captions(history)
            self.assertEqual(set(captions), set(c.build._chronological_unique_images(task, history)))
            for session in history["multi_session_dialogues"]:
                for turn in session["dialogues"]:
                    for image, text in zip(turn.get("input_image", []), turn.get("image_caption", [])):
                        self.assertEqual(captions[image], text)
            total += len(captions)
        self.assertEqual(total, 438)
        fake = {"multi_session_dialogues": [{"dialogues": [{"input_image": ["a"], "image_caption": ["  original\n"]}]}]}
        self.assertEqual(c.public_captions(fake)["a"], "  original\n")

    def test_zero_call_hash_inventory_and_routes(self):
        hashes = c.code_hashes()
        self.assertIn(str(Path(c.old.__file__).resolve()), hashes)
        self.assertIn(str(Path(c.build.__file__).resolve()), hashes)
        self.assertEqual([c.build.matrix.reader_route_for_task(t) for t in c.SMOKE_TASKS],
                         ["two_stage", "structured_fused"])
        qs = c.b.read(c.SOURCE / "queries_open.json")["rows"]
        self.assertEqual(len(qs), 371)
        self.assertTrue(all(set(q) == {"row_id", "task", "question"} for q in qs))

    def test_original_predict_reuse_and_registered_caps(self):
        calls = []
        def fake_generate(runtime, stage, prompt, **kw):
            calls.append((runtime.model, stage, kw["max_output_tokens"]))
            text = '{"final_answer":"test answer"}' if stage == "structured_fused" else "test answer"
            return c.rt.prior.Reply(text, "request", "response", runtime.model, runtime.model,
                                   {"prompt_tokens": 10, "output_tokens": 2}, False)
        with tempfile.TemporaryDirectory(prefix="caption-control-offline-") as tmp:
            with patch.object(c.rt.Runtime, "generate", fake_generate), patch.object(c.rt.prior, "disk_guard"):
                for arm, cfg in c.ARMS.items():
                    for route in ("direct", "structured_fused"):
                        rid = arm + "-" + route
                        q = {"row_id": rid, "task": "test", "question": "question"}
                        item = {"query": q, "context": "fixed facts", "context_sha256": c.b.sha("fixed facts")}
                        reg = {"seal_sha256": "test", "routes": {rid: route}}
                        root = Path(tmp) / arm / route
                        gen = c.LoggedRuntime(root, cfg["model"], False, answer_cap=cfg["answer_cap"])
                        pred = c.old.predict(root, cfg["reader_arm"], item, gen, reg)
                        self.assertEqual(pred["prediction"], "test answer")
                        self.assertEqual(pred["model"], cfg["model"])
                        self.assertEqual(calls[-1][2], cfg["answer_cap"] if route == "direct" else 1200)
                        n = len(calls)
                        self.assertEqual(c.old.predict(root, cfg["reader_arm"], item, gen, reg), pred)
                        self.assertEqual(len(calls), n)

    def test_no_images_or_caption_calls(self):
        gen = c.LoggedRuntime(Path("unused-offline"), "gemini-3.5-flash-lite", False)
        with self.assertRaisesRegex(RuntimeError, "must not view raw images"):
            gen.generate("caption", "ignored")
        with self.assertRaisesRegex(RuntimeError, "must not view raw images"):
            gen.generate("answer", "ignored", image_path="never-read.png")

    def test_q0_and_extractively_anchored_channels(self):
        question = "What color is the cabinet?"
        p = c.selection.qp.validate_response(question, '{"entity":["cabinet"],"relation":[],"attribute":["color"],"scope":[]}')
        out = c.selection.query_texts(question, p, c.ARMS["enhanced48"]["query_policy"])
        self.assertEqual(out[0]["query"], question)
        self.assertGreater(len(out), 1)
        self.assertEqual(c.ARMS["original47"]["query_policy"], "q0_only")
        self.assertEqual(c.ARMS["original47"]["judges"], ["g35lite"])
        self.assertEqual(c.ARMS["enhanced48"]["judges"], ["g35flash", "g35lite"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

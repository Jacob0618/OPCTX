"""Public-caption OpCtx control: original functions, isolated artifacts, no key handling.

prepare/status are zero-call. run requires --allow-model-calls and inherits the
parent's configured provider environment. No dotenv loading or credential reads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import os
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from filelock import FileLock
from creamem3.experiments.memeye_lite_ablation_20260923 import run as old

b, build, rt, selection, memory = old.b, old.build, old.rt, old.selection, old.memory
SOURCE = Path(r"D:\me_backbone21\r1")
DEFAULT_ROOT = Path(r"D:\opctx_caption_control_20261111\r1")
PROTOCOL = Path(__file__).with_name("caption_control_protocol.md")
ARMS = {
    "original47": {"reader_arm": "original_reader", "model": "gemini-2.5-flash-lite",
                   "query_policy": "q0_only", "answer_cap": 500, "judges": ["g35lite"]},
    "enhanced48": {"reader_arm": "enhanced_reader", "model": "gemini-3.5-flash-lite",
                   "query_policy": "q0_plus_anchored_attributes", "answer_cap": 4096,
                   "judges": ["g35flash", "g35lite"]},
}
SMOKE_TASKS = ("Cartoon_Entertainment_Companion", "Outdoor_Navigation_Route_Memory_Assistant")
TASK_WORKERS, QA_WORKERS = 4, 4


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def public_captions(history):
    captions = {}
    for session in history["multi_session_dialogues"]:
        for turn in session["dialogues"]:
            images, descriptions = turn.get("input_image", []), turn.get("image_caption", [])
            require(len(images) == len(descriptions), "Public caption/image alignment")
            for image, caption in zip(images, descriptions, strict=True):
                require(isinstance(caption, str) and caption.strip(), "Missing public caption")
                require(image not in captions or captions[image] == caption, "Conflicting public captions")
                captions[image] = caption  # Deliberately no strip/paraphrase/augmentation.
    return captions


def code_hashes():
    values = old.code_hashes()
    values[str(Path(__file__).resolve())] = b.file_sha(__file__)
    values[str(PROTOCOL)] = b.file_sha(PROTOCOL)
    return values


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "registration.lock"), timeout=1):
        if (root / "registration.json").exists():
            return validate(root)
        src = b.read(SOURCE / "registration.json")
        questions = b.read(SOURCE / "queries_open.json")["rows"]
        require(len(questions) == 371 and len({q["row_id"] for q in questions}) == 371, "Expected 371 questions")
        require(all(set(q) == {"row_id", "task", "question"} for q in questions), "Queries must be score-free")
        paths = [SOURCE / "registration.json", SOURCE / "queries_open.json", SOURCE / "scoring/open.json"]
        caption_audit = {}
        for task, entry in src["tasks"].items():
            path = Path(entry["path"])
            require(b.file_sha(path) == entry["sha256"], "Source history hash drift: " + task)
            history = b.read(path)
            require(not set(history) - {"character_profile", "multi_session_dialogues", "seal_sha256"}, "Unexpected history fields")
            caps = public_captions(history)
            expected = build._chronological_unique_images(task, history)
            require(set(caps) == set(expected), "Caption coverage differs from ingest images")
            caption_audit[task] = {"images": len(caps), "sessions": len(history["multi_session_dialogues"]),
                                  "caption_map_sha256": b.sha(caps), "caption_chars": sum(map(len, caps.values()))}
            paths.append(path)
        require(sum(x["images"] for x in caption_audit.values()) == 438, "Expected 438 public captions")
        require(sum(x["sessions"] for x in caption_audit.values()) == 221, "Expected 221 sessions")
        routes = {q["row_id"]: build.matrix.reader_route_for_task(q["task"]) for q in questions}
        require(routes == src["opctx_routes"], "Frozen Reader routes changed")
        protected = [Path(r"C:\Users\86139\Desktop\OpCtx_保留结果_20260927") / name
                     for name in ("MemEye_47.44.zip", "MemEye_enhanced_reader.zip")]
        authority = b.authorities()[2]
        reg = dict(schema="opctx.public_caption.control.20261111.v1", arms=ARMS,
            questions=questions, n=371, tasks=src["tasks"], routes=routes,
            code_hashes=code_hashes(), source_hashes={str(p): b.file_sha(p) for p in paths},
            protected_result_hashes={str(p): b.file_sha(p) for p in protected}, caption_audit=caption_audit,
            caption_source="verbatim published image_caption; no new visual inference",
            shared_memory_model="gemini-3.5-flash-lite", shared_memory_variant="online_base",
            source_generation_reused=False, old_facts_reused=False, caption_model_calls=0,
            selector_max_facts=30, candidate_scope="all task-visible fresh Writer facts, matching historical selectors",
            char_budget="Unicode character length of fresh-memory q0 Dense50 raw rendering; selector budget=length+1",
            raw_reader_context="only selected source facts; no event-view prose or session-summary prose",
            embedding="gemini-embedding-001 native 3072, original omitted task/dimensionality fields",
            prompt_hashes={k: b.sha(authority[k]) for k in ("open", "focused", "fused")} |
                {"writer_bundle": build.ingest.load_prompt_bundle()["prompt_bundle_sha256"],
                 "query_projection": b.sha(selection.qp.PROMPT_PREFIX), "online": build.ONLINE_PROMPT_SHA256},
            smoke={"tasks": list(SMOKE_TASKS), "questions": "first question in frozen input order per task",
                   "complete_histories": True, "reuse_in_full": True, "scoring": False},
            task_workers=TASK_WORKERS, qa_workers=QA_WORKERS, disk_floor_gib=15,
            physical_generation_attempt_cap=3, no_automatic_semantic_retries=True,
            scoring_phase="after all 742 predictions are durably frozen", created_at=time.time())
        return b.save(root / "registration.json", reg)


def validate(root):
    reg = b.read(root / "registration.json")
    require(reg["arms"] == ARMS, "Arm configuration drift")
    require(reg["code_hashes"] == code_hashes(), "Frozen source code drift")
    for group in ("source_hashes", "protected_result_hashes"):
        for path, digest in reg[group].items():
            require(b.file_sha(path) == digest, "Frozen input/result changed: " + path)
    return reg


def event(root, **data):
    return b.save(root / "telemetry" / f"{time.time_ns()}_{os.getpid()}.json", data)


class LoggedRuntime(rt.Runtime):
    def __init__(self, root, model, allow, *, answer_cap=None):
        super().__init__(root / "calls", model, allow)
        self.audit_root, self.answer_cap = root, answer_cap

    def generate(self, stage, prompt, **kwargs):
        require(stage != "caption" and kwargs.get("image_path") is None, "Public-caption control must not view raw images")
        if stage == "answer" and self.answer_cap is not None:
            kwargs["max_output_tokens"] = self.answer_cap
        kwargs["max_provider_attempts"] = min(3, kwargs.get("max_provider_attempts", 3))
        rt.prior.disk_guard(self.audit_root)
        started, clock = time.time(), time.perf_counter()
        try:
            reply = super().generate(stage, prompt, **kwargs)
        except BaseException as exc:
            event(self.audit_root, kind="llm", stage=stage, model=self.model, ok=False,
                  start=started, end=time.time(), elapsed_s=time.perf_counter()-clock,
                  error_type=type(exc).__name__, prompt_sha256=b.sha(prompt),
                  requested_output_cap=kwargs.get("max_output_tokens", 500), metadata=kwargs.get("cache_metadata", {}))
            raise
        event(self.audit_root, kind="llm", stage=stage, model=self.model, ok=True,
              start=started, end=time.time(), elapsed_s=time.perf_counter()-clock,
              request_sha256=reply.request_sha256, response_sha256=reply.response_sha256,
              cache_hit=reply.cache_hit, usage=reply.usage,
              requested_output_cap=kwargs.get("max_output_tokens", 500), metadata=kwargs.get("cache_metadata", {}))
        return reply


def embedding_backend(root, reg, allow):
    class LoggedProvider(rt.prior.VerifiedEmbeddingProvider):
        def batch_embed_contents(self, *, request):
            rt.prior.disk_guard(root)
            start, clock = time.time(), time.perf_counter()
            try:
                result = super().batch_embed_contents(request=request)
            except BaseException as exc:
                event(root, kind="embedding_http", ok=False, start=start, end=time.time(),
                      elapsed_s=time.perf_counter()-clock, error_type=type(exc).__name__, request_sha256=b.sha(request))
                raise
            event(root, kind="embedding_http", ok=True, start=start, end=time.time(),
                  elapsed_s=time.perf_counter()-clock, request_sha256=b.sha(request))
            return result
    return build.OfficialNative3072EmbeddingBackend(root / "embedding", containment_root=root,
        approved_provider_attempt_hard_ceiling=6000, budget_binding_sha256=reg["seal_sha256"],
        allow_model_calls=allow, provider_factory=LoggedProvider)


def prepare_memory(root, task, reg, allow):
    build.ingest.WORK_MODELS["g35lite"] = reg["shared_memory_model"]
    gen = LoggedRuntime(root, reg["shared_memory_model"], allow)
    history = b.read(reg["tasks"][task]["path"])
    history = {k: history[k] for k in ("character_profile", "multi_session_dialogues")}
    caps = public_captions(history)
    require(b.sha(caps) == reg["caption_audit"][task]["caption_map_sha256"], "Caption map drift")
    images = build._chronological_unique_images(task, history)
    row = dict(task=task, memory=history, memory_sha256=b.sha(history), images=images,
               sessions=len(history["multi_session_dialogues"]))
    b.save(root / "input.json", dict(history=history, images=images, registration_sha256=reg["seal_sha256"]))
    b.save(root / "public_captions.json", dict(captions=caps, source="published image_caption verbatim",
        generated_caption_calls=0, caption_map_sha256=b.sha(caps)))
    embed = embedding_backend(root, reg, allow)
    start, clock = time.time(), time.perf_counter()
    session, session_ref = build.build_session_graph_for_task(row, caps, backbone="g35lite",
        generation=gen, embedding=embed, output_root=root)
    event(root, kind="stage", stage="writer_and_index", start=start, end=time.time(), elapsed_s=time.perf_counter()-clock)
    start, clock = time.time(), time.perf_counter()
    online, online_ref = build.build_online_graph_for_task(row, session, backbone="g35lite",
        generation=gen, embedding=embed, output_root=root)
    event(root, kind="stage", stage="online_and_index", start=start, end=time.time(), elapsed_s=time.perf_counter()-clock)
    require(online["fact_nodes"] == session["fact_nodes"], "Online mutated source facts")
    b.save(root / "memory_complete.json", dict(model=gen.model, task=task, images=len(images), sessions=row["sessions"],
        facts=len(session["fact_nodes"]), session=session_ref, online=online_ref, public_caption_sha256=b.sha(caps),
        registration_sha256=reg["seal_sha256"]))
    return session, embed


def context_for(root, arm, graph, index, facts, ids, q, gen, embed, reg):
    key = b.sha(q["row_id"])
    context_path = root / "contexts" / ARMS[arm]["reader_arm"] / (key + ".json")
    if context_path.exists():
        item = b.read(context_path)
        require(item["query"] == q and item["registration_sha256"] == reg["seal_sha256"], "Context resume drift")
        require(item["source_graph_sha256"] == b.sha(graph), "Selected context graph drift")
        return item
    if arm == "enhanced48":
        path = root / "projections" / (key + ".json")
        if path.exists():
            projection = b.read(path)["projection"]
        else:
            out = gen.generate("planner", selection.qp.build_prompt(q["question"]), json_mode=True,
                max_output_tokens=2048, cache_metadata=dict(row_id=q["row_id"], purpose="enhanced_query_projection"))
            projection = selection.qp.validate_response(q["question"], out.text)
            b.save(path, dict(projection=projection, generation=asdict(out)))
        queries = selection.query_texts(q["question"], projection, "q0_plus_anchored_attributes")
    else:
        queries = [dict(query_id="q0", query=q["question"], origin="original_complete_question")]
    start, clock = time.time(), time.perf_counter()
    vectors = dict(zip([x["query"] for x in queries], embed.embed([x["query"] for x in queries],
        tag="query", cache_metadata=dict(row_id=q["row_id"], purpose="query_channels")), strict=True))
    channels = old.channels_for(index, facts, queries, vectors)
    dense50 = selection.display_ids(ids, ids, channels[0]["dense_scores"])[:50]
    dense_text, _ = selection.adapter.render_selection(graph, dense50)
    item = old.select_context(graph, q, channels, len(dense_text))
    item.update(registration_sha256=reg["seal_sha256"], source_graph_sha256=b.sha(graph),
                query_policy=ARMS[arm]["query_policy"], dense50_reference_ids=dense50,
                active_query_texts=queries, candidate_count=len(facts))
    event(root, kind="stage", stage="retrieval_selection", row_id=q["row_id"], arm=arm,
        start=start, end=time.time(), elapsed_s=time.perf_counter()-clock, selected_n=len(item["selected_fact_ids"]))
    return b.save(context_path, item)


def run_predictions(root, task, reg, allow, *, smoke=False):
    task_root = root / "tasks" / task
    task_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(task_root / "task.lock"), timeout=1):
        graph, embed = prepare_memory(task_root / "common_memory", task, reg, allow)
        facts, _ = selection.adapter.adapt_facts(graph)
        ids = [f["id"] for f in facts]
        index = memory.GraphMemoryIndex(graph, candidate_cap=len(facts), rerank_top_k=50)
        require(ids == [f["id"] for f in index.facts], "Fact/vector ordering drift")
        queries = [q for q in reg["questions"] if q["task"] == task]
        if smoke:
            queries = queries[:1]
        failures = []
        for arm, cfg in ARMS.items():
            arm_root = task_root / "arms" / arm
            gen = LoggedRuntime(arm_root, cfg["model"], allow, answer_cap=cfg["answer_cap"])
            items = [context_for(arm_root, arm, graph, index, facts, ids, q, gen, embed, reg) for q in queries]
            with ThreadPoolExecutor(max_workers=QA_WORKERS) as pool:
                jobs = {pool.submit(old.predict, arm_root, cfg["reader_arm"], item, gen, reg): item["query"]["row_id"] for item in items}
                for future in as_completed(jobs):
                    try:
                        future.result()
                        print("PREDICTED", arm, jobs[future], flush=True)
                    except Exception as exc:
                        receipt = dict(stage="prediction", arm=arm, row_id=jobs[future], error_type=type(exc).__name__, time=time.time())
                        b.save(task_root / "errors" / f"{time.time_ns()}.json", receipt)
                        failures.append(receipt)
        require(not failures, "Unresolved prediction failures; immutable successes retained")
        marker = root / "smoke" / (task + ".json") if smoke else task_root / "generation_complete.json"
        return b.save(marker, dict(task=task, n_per_arm=len(queries), arms=list(ARMS),
            question_ids=[q["row_id"] for q in queries], complete_histories=True,
            registration_sha256=reg["seal_sha256"], no_scores_read=True))


def prediction_paths(root, task, arm):
    return root / "tasks" / task / "arms" / arm / "predictions" / ARMS[arm]["reader_arm"]


def require_all_predictions(root, reg):
    for arm in ARMS:
        found = []
        for task in reg["tasks"]:
            rows = [b.read(p) for p in prediction_paths(root, task, arm).glob("*.json")]
            found.extend(r["row_id"] for r in rows)
            require(all(r["registration_sha256"] == reg["seal_sha256"] for r in rows), "Prediction registration drift")
        require(len(found) == 371 and set(found) == {q["row_id"] for q in reg["questions"]}, "All predictions required before scoring: " + arm)


def score_one(root, task, arm, q, gold, reg, allow):
    cfg, key = ARMS[arm], b.sha(q["row_id"])
    arm_root = root / "tasks" / task / "arms" / arm
    pred = b.read(prediction_paths(root, task, arm) / (key + ".json"))
    require(gold["question"] == q["question"], "Gold/question binding drift")
    for judge_name in cfg["judges"]:
        score_path = arm_root / "scores" / judge_name / cfg["reader_arm"] / (key + ".json")
        if score_path.exists():
            require(b.read(score_path)["prediction_sha256"] == pred["seal_sha256"], "Scored answer changed")
            continue
        judge = LoggedRuntime(arm_root / "judges" / judge_name, rt.JUDGES[judge_name], allow)
        out, verdict = judge.judge(q["question"], gold["gold"], pred["prediction"],
            dict(arm=arm, row_id=q["row_id"], prediction_sha256=pred["seal_sha256"]))
        b.save(score_path, dict(row_id=q["row_id"], task=task, arm=arm, judge=judge.model,
            x=gold["x"], y=gold["y"], xy=gold["x"] + "-" + gold["y"],
            raw_score=verdict.raw_score, mapped_score=verdict.mapped_score, reasoning=verdict.reasoning,
            generation=asdict(out), prediction_sha256=pred["seal_sha256"], registration_sha256=reg["seal_sha256"]))
    print("SCORED", arm, q["row_id"], flush=True)


def score_task(root, task, reg, allow):
    require_all_predictions(root, reg)
    # This is the only semantic read of gold data; all generation is already frozen.
    gold_rows = b.read(SOURCE / "scoring/open.json")["rows"]
    task_root = root / "tasks" / task
    with FileLock(str(task_root / "task.lock"), timeout=1):
        failures = []
        with ThreadPoolExecutor(max_workers=QA_WORKERS) as pool:
            jobs = {pool.submit(score_one, root, task, arm, q, gold_rows[q["row_id"]], reg, allow): (arm, q["row_id"])
                    for arm in ARMS for q in reg["questions"] if q["task"] == task}
            for future in as_completed(jobs):
                try:
                    future.result()
                except Exception as exc:
                    arm, rid = jobs[future]
                    receipt = dict(stage="judge", arm=arm, row_id=rid, error_type=type(exc).__name__, time=time.time())
                    b.save(task_root / "errors" / f"{time.time_ns()}.json", receipt)
                    failures.append(receipt)
        require(not failures, "Unresolved scores; no missing-as-zero")
        b.save(task_root / "scoring_complete.json", dict(task=task, registration_sha256=reg["seal_sha256"]))


def status(root, reg):
    result = {"root": str(root), "tasks": {}, "results": {}}
    for task in reg["tasks"]:
        tr = root / "tasks" / task
        info = {"expected": sum(q["task"] == task for q in reg["questions"]),
                "memory_complete": (tr / "common_memory/memory_complete.json").exists(), "arms": {}}
        if info["memory_complete"]:
            info["memory"] = b.read(tr / "common_memory/memory_complete.json")
        for arm, cfg in ARMS.items():
            ar = tr / "arms" / arm
            info["arms"][arm] = {"contexts": len(list((ar / "contexts" / cfg["reader_arm"]).glob("*.json"))),
                "predictions": len(list(prediction_paths(root, task, arm).glob("*.json"))),
                "scores": {j: len(list((ar / "scores" / j / cfg["reader_arm"]).glob("*.json"))) for j in cfg["judges"]}}
        result["tasks"][task] = info
    for arm, cfg in ARMS.items():
        result["results"][arm] = {}
        for j in cfg["judges"]:
            rows = [b.read(p) for p in root.glob("tasks/*/arms/" + arm + "/scores/" + j + "/" + cfg["reader_arm"] + "/*.json")]
            summary = {"n": len(rows), "expected": 371}
            require(len({r["row_id"] for r in rows}) == len(rows), "Duplicate score identities")
            if len(rows) == 371:
                summary.update(raw_percent=100*statistics.mean(r["raw_score"] for r in rows),
                               mapped_percent=100*statistics.mean(r["mapped_score"] for r in rows))
                for axis in ("x", "y", "xy", "task"):
                    summary["by_" + axis] = {v: {"n": sum(r[axis] == v for r in rows),
                        "raw_percent": 100*statistics.mean(r["raw_score"] for r in rows if r[axis] == v),
                        "mapped_percent": 100*statistics.mean(r["mapped_score"] for r in rows if r[axis] == v)}
                        for v in sorted({r[axis] for r in rows})}
            result["results"][arm][j] = summary
    return result


def launch(root, phase, task):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / f"{phase}_{task}_{time.time_ns()}.log"
    cmd = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), phase,
           "--root", str(root), "--task", task, "--allow-model-calls"]
    with logfile.open("ab") as output:
        process = subprocess.Popen(cmd, cwd=REPO, stdout=output, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    b.save(root / "launches" / f"{process.pid}_{time.time_ns()}.json",
        dict(pid=process.pid, command=cmd, logfile=str(logfile), start=time.time()))
    return process.wait()


def run(root, reg):
    with FileLock(str(root / "dispatcher.lock"), timeout=1):
        started = time.time()
        for task in SMOKE_TASKS:
            if not (root / "smoke" / (task + ".json")).exists():
                require(launch(root, "smoke", task) == 0, "Smoke failed; full run not started")
        for phase in ("generate", "score"):
            failures = []
            with ThreadPoolExecutor(max_workers=TASK_WORKERS) as pool:
                jobs = {pool.submit(launch, root, phase, task): task for task in reg["tasks"]}
                for future in as_completed(jobs):
                    if future.result() != 0:
                        failures.append(jobs[future])
            require(not failures, phase + " incomplete: " + ", ".join(failures))
        report = status(root, reg)
        require(all(s["n"] == 371 for a in report["results"].values() for s in a.values()), "Incomplete final scores")
        b.save(root / "complete.json", dict(registration_sha256=reg["seal_sha256"],
            results=report["results"], started=started, ended=time.time(), elapsed_wall_s=time.time()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "generate", "score", "run", "status"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task")
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_relative_to(DEFAULT_ROOT.parent.resolve()) and root != DEFAULT_ROOT.parent.resolve(), "New experiment root must remain under isolated D:/opctx_caption_control_20261111/")
    live = args.phase in ("smoke", "generate", "score", "run")
    require(live == args.allow_model_calls, "Only live phases require --allow-model-calls")
    if args.phase == "prepare":
        print(json.dumps({"registered": prepare(root)["seal_sha256"], "root": str(root)})); return
    reg = validate(root)
    if args.phase == "status":
        print(json.dumps(status(root, reg), ensure_ascii=False, indent=2)); return
    rt.prior.disk_guard(root)
    if args.phase == "run":
        run(root, reg); return
    require(args.task in reg["tasks"], "A registered --task is required")
    try:
        if args.phase == "smoke":
            require(args.task in SMOKE_TASKS, "Only fixed smoke tasks allowed")
            run_predictions(root, args.task, reg, True, smoke=True)
        elif args.phase == "generate":
            require(all((root / "smoke" / (t + ".json")).exists() for t in SMOKE_TASKS), "Both smoke routes must pass first")
            run_predictions(root, args.task, reg, True)
        else:
            score_task(root, args.task, reg, True)
    except BaseException as exc:
        b.save(root / "stops" / f"{time.time_ns()}_{os.getpid()}.json", dict(phase=args.phase,
            task=args.task, error_type=type(exc).__name__, time=time.time()))
        raise


if __name__ == "__main__":
    main()

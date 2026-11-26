"""Preserve MGTM-style candidate routing; change only derived Online memory."""
import argparse
import copy
import json
from pathlib import Path
import time

from creamem3.experiments.locomo_online_only_20260909 import run as old
from creamem3.experiments.locomo_online_only_20260909 import audit as physical_audit
from creamem3.experiments.locomo_online_only_20260909.runtime import Runtime as OriginalRuntime
from creamem3.experiments.locomo_online_only_20260909.runtime import read, seal, sha, file_sha, StopRun, now
from creamem3.experiments.locomo_state_norerank_20260909 import revision
from creamem3.experiments.locomo_state_norerank_20260909 import revision_roles_v3 as roles

RUN = Path(r"D:\creamem3_cache\locomo_joint_revision_20260909_r1")
SOURCE = roles.BUILD_ROOT
ARMS = ["session_strong", "opctx_original", "revision_v3"]
OriginalIndex = old.graph_api.GraphMemoryIndex


class StrictIndex(OriginalIndex):
    """Unchanged successful selection; never silently accept cosine fallback."""
    def search(self, *args, **kwargs):
        context, trace = super().search(*args, **kwargs)
        if trace["rerank"]["fallback"]:
            raise StopRun("Reranker validation failed; cosine fallback is not an accepted result")
        return context, trace


class Runtime(OriginalRuntime):
    def post(self, stage, scope, model, method, body):
        if stage not in {"rerank", "qa", "judge"}:
            self._halt("Read-only-memory trial attempted a forbidden model stage: " + stage)
        return super().post(stage, scope, model, method, body)


def replay(sample, parent, registration):
    base_graph = old.get_source_graph(sample, "session_strong", parent)
    base = revision.legacy._base_parts(base_graph)
    state = revision.empty_state()
    digests = []
    for sid in base["session_ids"]:
        value, exposed, trace = revision.build_input(base, state, sid)
        checkpoint = read(SOURCE / "state_build/checkpoints" / sample / f"{sid:04d}.json")
        identity = {"base_sha256": sha(base_graph), "registration_sha256": sha(registration),
                    "session_id": sid, "input_sha256": sha(value), "before_sha256": sha(state)}
        if checkpoint["identity"] != identity or checkpoint["input"] != value or checkpoint["trace"] != trace:
            raise StopRun("Causal input replay mismatch: " + sample + "/" + str(sid))
        state, accepted, rejected = roles.apply_response(state, checkpoint["response"], exposed,
                                                        base["fact_by_id"], sid)
        if (checkpoint["state"] != state or checkpoint["state_sha256"] != sha(state)
                or checkpoint["accepted"] != accepted or checkpoint["rejected"] != rejected):
            raise StopRun("State transition replay mismatch: " + sample + "/" + str(sid))
        digests.append({"session_id": sid, "checkpoint_sha256": sha(checkpoint)})
    source_path = SOURCE / "graphs/revision" / (sample + ".json")
    source = read(source_path)
    graph = source["graph"]
    if (source["step_digests"] != digests or source["registration_sha256"] != sha(registration)
            or source["base_sha256"] != sha(base_graph) or graph["revision_memory"] != state
            or graph["fact_nodes"] != base_graph["fact_nodes"]):
        raise StopRun("Final Online graph differs from causal replay: " + sample)
    expected = copy.deepcopy(base_graph)
    views = revision.views_from_state(state, base["fact_by_id"], include_history=True)
    expected["summary_nodes"] += list(views.values())
    expected["edges"] += revision.legacy._view_edges(views)
    if graph["summary_nodes"] != expected["summary_nodes"] or graph["edges"] != expected["edges"]:
        raise StopRun("Derived view membership/materialization drift: " + sample)
    structural = old.graph_structure_audit(graph, sample)
    if structural["status"] != "pass":
        raise StopRun("Invalid graph structure: " + sample)
    return source, {"status": "pass", "checkpoints": len(digests), "immutable_writer": True,
                    "replayed_transitions": True, "source_path": str(source_path),
                    "source_sha256": sha(source)}, structural


def prepare():
    loaded, parent = old.prepare()
    ready = read(roles.run.ROOT / "revision_v3_ready.json")
    registration = read(SOURCE / "registration.json")
    if (ready["conversations"] != len(loaded) or ready["registration_sha256"] != sha(registration)
            or registration["source_sha256"] != file_sha(roles.__file__)
            or registration["base_compiler_sha256"] != file_sha(revision.__file__)):
        raise StopRun("Online build registration drift")
    paths = [Path(__file__), Path(old.__file__), Path(revision.__file__), Path(roles.__file__),
             Path(physical_audit.__file__), Path(old.ROOT / "creamem3/online_memory_v3.py")]
    study = {"schema": "locomo.original_joint_retrieval.revision.v1", "parent_protocol_sha256": sha(parent),
             "source_sha256": {str(p): file_sha(p) for p in paths}, "online_registration_sha256": sha(registration),
             "arms": ARMS, "fresh_matched_reader_calls": True, "shared_exact_request_bodies": True,
             "summary_and_event_top_k": 10, "dense_top_k": 20, "bm25_top_k": 20,
             "expand_all_routed_members_before_cap": True, "candidate_cap": 120, "final_facts": 20,
             "rerank_model": "gemini-2.5-flash", "rerank_fallback": "fatal",
             "no_new_writer_or_planner": True, "no_model_rerank_in_online_build": True,
             "no_full_run_until_dev_review": True,
             "clarification": "No-reranker global-cosine diagnostics are not the original joint retrieval ablation"}
    seal(RUN / "registration.json", study)
    protocol = {**parent, "joint_revision_study_sha256": sha(study)}
    seal(RUN / "protocol.json", protocol)
    variant = {"variant": "revision_v3", "online_registration": registration,
               "study_sha256": sha(study), "source_build": str(SOURCE),
               "read_side": "Original frozen joint routing/rerank/OpCtx renderer; successful behavior unchanged"}
    seal(RUN / "variants/revision_v3.json", variant)
    for conv in loaded:
        sample = conv["sample_id"]
        source, causal, structural = replay(sample, parent, registration)
        seal(RUN / "graphs/revision_v3" / (sample + ".json"), {
            "identity": {"variant_sha256": sha(variant), "session_sha256": source["base_sha256"]},
            "graph": source["graph"], "audit": causal, "standard_audit": structural,
            "source_record_sha256": sha(source), "status_counts": source["status_counts"],
            "operation_counts": source["operation_counts"]})
        old.emit("JOINT PREFLIGHT " + sample + " causal replay and immutable Writer PASS")
    return loaded, protocol


def audit(loaded, protocol):
    result = physical_audit.audit(RUN)
    questions = {q["qa_id"]: (conv["sample_id"], q) for conv in loaded for q in conv["qa"]}
    vectors = read(old.VECTOR_FILE)
    vector_map = dict(zip(vectors["ids"], vectors["vectors"]))
    indexes = {}
    for arm in ARMS:
        for conv in loaded:
            sample = conv["sample_id"]
            graph = (read(RUN / "graphs/revision_v3" / (sample + ".json"))["graph"] if arm == "revision_v3"
                     else old.get_source_graph(sample, arm, protocol))
            indexes[(arm, sample)] = (OriginalIndex(graph), sha(graph))
        for qid in protocol["dev_ids"]:
            sample, question = questions[qid]
            row = read(RUN / "rows" / arm / (qid + ".json"))
            retrieval = read(RUN / "retrieval" / arm / (qid + ".json"))
            index, graph_sha = indexes[(arm, sample)]
            _, candidate_trace = index._candidate_pool(question["question"], vector_map[qid])
            if any(retrieval["trace"][k] != v for k, v in candidate_trace.items()):
                raise StopRun("Joint-channel trace replay mismatch: " + arm + "/" + qid)
            if (row["identity"]["qa_sha256"] != sha(question) or row["identity"]["graph_sha256"] != graph_sha
                    or row["identity"]["vector_sha256"] != sha(vector_map[qid])
                    or row["question"] != question["question"] or row["answer"] != question["answer"]
                    or row["rerank_fallback"] or row["context"] != retrieval["context"]):
                raise StopRun("Question/graph/retrieval binding mismatch: " + arm + "/" + qid)
    if not result["pass"]:
        raise StopRun("Physical audit failed: " + json.dumps(result, ensure_ascii=False))
    result["joint_candidate_trace_replays"] = len(ARMS) * len(protocol["dev_ids"])
    seal(RUN / "completed_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "audit"])
    args = parser.parse_args()
    with old.Lease(RUN):
        loaded, protocol = prepare()
        if args.command == "preflight":
            return
        if args.command == "audit":
            print(json.dumps(audit(loaded, protocol), ensure_ascii=False)); return
        from dotenv import load_dotenv
        load_dotenv(old.ROOT / ".env", override=False)
        old.graph_api.GraphMemoryIndex = StrictIndex
        runtime = Runtime(RUN / "calls", allow_model_calls=True, concurrency=24, timeout=180)
        try:
            reports = old.evaluate(RUN, ARMS, loaded, protocol, runtime, qa_workers=24)
            checks = audit(loaded, protocol)
            seal(RUN / "complete.json", {"protocol_sha256": sha(protocol), "reports": reports, "audit": checks})
        except BaseException as exc:
            runtime.stopped.set()
            seal(RUN / ("stop_" + str(time.time_ns()) + ".json"),
                 {"utc": now(), "type": type(exc).__name__, "message": str(exc)[:500]})
            raise


if __name__ == "__main__":
    main()

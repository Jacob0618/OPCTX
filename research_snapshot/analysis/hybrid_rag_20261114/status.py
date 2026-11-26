"""Read-only live Hybrid-RAG status; no credentials, APIs, writes, or resume."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

DEFAULT_ROOTS = [Path("D:/hybrid_rag_20261114") / f"{name}_r1" for name in ("locomo", "lme")]
LOCOMO_POLICY_ROOT = Path("D:/hybrid_rag_20261114/locomo_policy_zero_20261114_r1")
LOCOMO_POLICY_ID = "locomo_invalid_rerank_id_zero_20261114"
LOCOMO_FINAL_ROOT = Path("D:/hybrid_rag_20261114/locomo_final_20261114_r1")
LME_RECOVERY_ROOT = Path("D:/hybrid_rag_20261114/lme_r2")
LME_RECOVERY_ROOT3 = Path("D:/hybrid_rag_20261114/lme_r3")
LME_RECOVERY_ROOT4 = Path("D:/hybrid_rag_20261114/lme_r4")
LME_RECOVERY_ROOT5 = Path("D:/hybrid_rag_20261114/lme_r5")
LME_JUDGE_ROOT6 = Path("D:/hybrid_rag_20261114/lme_judge_finish_r6")
LME_POLICY_QID = "29f2956b_abs"
LME_ADDITIONAL_POLICY_IDS = {"06878be2", "75832dbd"}


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def native_digest(value):
    """Same JSON-byte convention as the recorded native transport ledger."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def processes():
    """Never return raw command lines, environment variables, or credentials."""
    try:
        import psutil
    except ImportError:
        return [], "psutil_unavailable_process_state_unknown"
    rows = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            if str(process.info.get("name", "")).lower() not in ("python", "python.exe", "pythonw.exe", "python3"):
                continue
            cmd = process.cmdline() or []
            runner = next((Path(arg).name for arg in cmd if "hybrid_rag_20261114" in arg.replace("\\", "/")
                           and Path(arg).name in ("run.py", "lme_resume_policy_zero.py", "lme_resume_two_zeros.py", "lme_isolated_continue.py", "lme_isolated_continue_r5.py", "lme_judge_finish_r6.py")), None)
            if not runner:
                continue
            def value(name):
                return cmd[cmd.index(name)+1] if name in cmd and cmd.index(name)+1 < len(cmd) else None
            recovery = runner in ("lme_resume_policy_zero.py", "lme_resume_two_zeros.py", "lme_isolated_continue.py", "lme_isolated_continue_r5.py", "lme_judge_finish_r6.py") and "--run" in cmd
            default_recovery = {"lme_resume_two_zeros.py": LME_RECOVERY_ROOT3,
                                "lme_isolated_continue.py": LME_RECOVERY_ROOT4,
                                "lme_isolated_continue_r5.py": LME_RECOVERY_ROOT5,
                                "lme_judge_finish_r6.py": LME_JUDGE_ROOT6}.get(runner, LME_RECOVERY_ROOT)
            output = value("--output") or (str(default_recovery) if recovery else None)
            if output:
                rows.append(dict(pid=process.info["pid"], root=str(Path(output).resolve()),
                    phase="judge" if runner == "lme_judge_finish_r6.py" else "full" if recovery else value("--phase"), created_unix=process.create_time(),
                    recovery_entry=recovery))
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return rows, None


def snapshot(root, window=60, process_rows=None, process_error=None):
    root = Path(root).resolve()
    now, since = time.time(), time.time()-window
    errors, mtimes = [], {}
    def read(path):
        try:
            mtimes[str(path)] = path.stat().st_mtime
            with path.open(encoding="utf-8-sig") as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), error_type=type(exc).__name__,
                possibly_mid_write=now-mtimes.get(str(path), 0) < 2))
            return None
    config = read(root / "config.json")
    if not config:
        return dict(root=str(root), state="unreadable_config", read_errors=errors)
    expected_ids = set(config["question_ids"])
    registration_path = root / "recovery_registration.json"
    registration = read(registration_path) if registration_path.is_file() else None
    recovery_parent, recovery_ancestor, inherited_paths = None, None, set()
    isolated_mode, judge_only_mode = False, False
    index_root = root
    if (isinstance(registration, dict) and root == LME_RECOVERY_ROOT.resolve()
            and registration.get("schema") == "hybrid_rag.single_policy_recovery.v1"
            and Path(registration.get("output", "")).resolve() == root
            and Path(registration.get("parent", "")).resolve() == root.with_name("lme_r1")):
        recovery_parent = Path(registration["parent"]).resolve()
        inherited_paths = set(registration.get("copied_files_sha256", {}))
    elif (isinstance(registration, dict) and root == LME_RECOVERY_ROOT3.resolve()
            and registration.get("schema") == "hybrid_rag.two_more_policy_zeros.v1"
            and Path(registration.get("output", "")).resolve() == root
            and Path(registration.get("parent", "")).resolve() == root.with_name("lme_r2")
            and Path(registration.get("ancestor", "")).resolve() == root.with_name("lme_r1")):
        recovery_parent = Path(registration["parent"]).resolve()
        recovery_ancestor = Path(registration["ancestor"]).resolve()
        inherited_paths = set(registration.get("copied_files_sha256", {}))
    elif (isinstance(registration, dict) and root in (LME_RECOVERY_ROOT4.resolve(), LME_RECOVERY_ROOT5.resolve())
            and registration.get("schema") == ("hybrid_rag.isolated_qa500_missing.v1" if root == LME_RECOVERY_ROOT4.resolve()
                                               else "hybrid_rag.isolated_native_maxtokens_r5.v1")
            and Path(registration.get("output", "")).resolve() == root
            and [Path(p).resolve() for p in registration.get("parents", [])] ==
                [root.with_name(f"lme_r{i}") for i in range(1, 4 if root == LME_RECOVERY_ROOT4.resolve() else 5)]
            and registration.get("runtime_layout") == "per_question_isolated"
            and registration.get("new_missing_scored_zero") is False):
        recovery_parent = Path(registration["parents"][-1]).resolve()
        inherited_paths = set(registration.get("copied_files_sha256", {}))
        isolated_mode = True
    elif (isinstance(registration, dict) and root == LME_JUDGE_ROOT6.resolve()
            and registration.get("schema") == "hybrid_rag.judge_only_finish_r6.v1"
            and Path(registration.get("output", "")).resolve() == root
            and [Path(p).resolve() for p in registration.get("parents", [])] ==
                [root.with_name(f"lme_r{i}") for i in range(1, 6)]
            and Path(registration.get("source_output", "")).resolve() == LME_RECOVERY_ROOT5.resolve()
            and Path(registration.get("index_source_output", "")).resolve() == LME_RECOVERY_ROOT5.resolve()
            and registration.get("runtime_layout") == "judge_only_worker_isolated"):
        recovery_parent = LME_RECOVERY_ROOT5.resolve()
        index_root = recovery_parent
        inherited_paths = set(registration.get("copied_files_sha256", {}))
        isolated_mode, judge_only_mode = True, True
    prior_roots = ([Path(p).resolve() for p in reversed(registration["parents"])] if isolated_mode else
                   [p for p in (recovery_parent, recovery_ancestor) if p])
    policy_registration, policy_parent, policy_ancestor = registration, recovery_parent, recovery_ancestor
    registration_chain, registration_chain_valid = {root: registration}, True
    if isolated_mode:
        for prior_root in prior_roots:
            prior_registration = prior_root / "recovery_registration.json"
            if prior_registration.is_file():
                registration_chain[prior_root] = read(prior_registration) or {}
        policy_registration = registration_chain.get(root.with_name("lme_r3"), {})
        policy_parent, policy_ancestor = root.with_name("lme_r2"), root.with_name("lme_r1")
        for child, child_reg in registration_chain.items():
            parent_name = child_reg.get("parents", [child_reg.get("parent")])[-1]
            parent = Path(parent_name).resolve() if parent_name else None
            if parent in registration_chain:
                path = parent / "recovery_registration.json"
                registration_chain_valid = (registration_chain_valid and
                    child_reg.get("parent_files_sha256", child_reg.get("source_files_sha256", {})).get("recovery_registration.json") ==
                    hashlib.sha256(path.read_bytes()).hexdigest())
    def inherited(path):
        return not path.is_relative_to(root) or path.relative_to(root).as_posix() in inherited_paths
    inventory = read(index_root / "offline_inventory.json") or {}
    expected_contexts = {c["context_id"] for c in inventory.get("contexts", [])}
    active = [p for p in (process_rows or []) if os.path.normcase(p["root"]) == os.path.normcase(str(root))]
    full_start = max((p.get("created_unix") or 0 for p in active if p.get("phase") in ("full", "judge")), default=0)
    ids = {name: set() for name in ("prediction", "inference", "result", "failure", "retrieval", "selected", "context")}
    auxiliary_names = {"retrieval", "selected", "context"}
    recent = {name: set() for name in ids}
    bad, failure_rows, policy_result_rows, native_results, missing_rows = [], {}, {}, {}, {}
    for directory in sorted((root / "questions").glob("*")):
        if not directory.is_dir():
            continue
        qid = directory.name
        missing_path = directory / "missing.json"
        if missing_path.is_file():
            missing_rows[qid] = read(missing_path)
        for name in ids:
            path = directory / f"{name}.json"
            if not path.is_file():
                continue
            # Large retrieval traces and rendered contexts are not parsed on a
            # routine status poll. Core answer/failure records remain validated.
            if name in auxiliary_names:
                stat = path.stat()
                mtimes[str(path)] = stat.st_mtime
                if qid in expected_ids and stat.st_size:
                    ids[name].add(qid)
                    if stat.st_mtime >= since:
                        recent[name].add(qid)
                continue
            item = read(path)
            if item is None:
                continue
            if name == "result" and item.get("status") == "policy_zero":
                # A policy result is not a Judge result; validate separately.
                policy_result_rows[qid] = item
                continue
            valid = qid in expected_ids and item.get("question_id", qid) == qid
            if name in ("prediction", "inference", "result"):
                valid = valid and isinstance(item.get("prediction"), str) and bool(item["prediction"].strip())
            if name == "result":
                judge = item.get("judge") or {}
                tags = re.findall(r"\[\[(yes|no)\]\]", str(judge.get("raw", "")).lower())
                valid = (valid and type(item.get("correct")) in (int, bool) and item["correct"] in (0, 1)
                         and len(tags) == 1 and item["correct"] == judge.get("correct") == int(tags[0] == "yes"))
            if not valid:
                bad.append(dict(question_id=qid, checkpoint=name, path=str(path)))
                continue
            ids[name].add(qid)
            if mtimes[str(path)] >= since and not inherited(path):
                recent[name].add(qid)
            if name == "failure":
                failure_rows[qid] = item
            elif name == "result":
                native_results[qid] = item

    policy_ids, recent_policy_ids, policy_invalid, policy_times = set(), set(), [], []
    authorized_policy_outcome_paths = set()
    final_artifact = None
    policy_path = LOCOMO_POLICY_ROOT / "policy.json"
    policy_applies = (config.get("dataset") == "locomo" and
                      root == (LOCOMO_POLICY_ROOT.parent / "locomo_r1").resolve())
    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if policy_applies and policy_path.is_file():
        policy = read(policy_path)
        policy_valid = (isinstance(policy, dict) and policy.get("policy_id") == LOCOMO_POLICY_ID
            and Path(policy.get("source_run", "")).resolve() == root
            and policy.get("source_config_sha256") == file_hash(root / "config.json")
            and policy.get("dataset") == "locomo" and policy.get("score") == 0
            and policy.get("judge_called") is False and policy.get("extra_model_calls") == 0
            and policy.get("retains_all_original_evidence") is True)
        if not policy_valid:
            policy_invalid.append(dict(path=str(policy_path), reason="policy_contract_or_source_config_mismatch"))
        else:
            for path in sorted((LOCOMO_POLICY_ROOT / "decisions").glob("*.json")):
                decision = read(path)
                if not isinstance(decision, dict):
                    continue
                qid = decision.get("question_id")
                failure = failure_rows.get(qid)
                source = root / "questions" / str(qid) / "failure.json"
                retrieval = source.with_name("retrieval.json")
                audit = failure.get("audit") if isinstance(failure, dict) else None
                eligible = (qid in expected_ids and path.stem == qid and qid not in policy_ids
                    and decision.get("policy_id") == LOCOMO_POLICY_ID
                    and decision.get("status") == "user_policy_failure_zero"
                    and type(decision.get("correct")) is int and decision["correct"] == 0
                    and decision.get("prediction") is None and decision.get("judge") is None
                    and decision.get("judge_called") is False
                    and Path(decision.get("source_failure", "")).resolve() == source.resolve()
                    and source.is_file() and decision.get("source_failure_sha256") == file_hash(source)
                    and retrieval.is_file() and decision.get("source_retrieval_sha256") == file_hash(retrieval)
                    and isinstance(failure, dict) and failure.get("error_type") == "StageFailure"
                    and failure.get("error") == "Original Gemini reranker exhausted validation"
                    and isinstance(audit, dict) and audit.get("attempts") == 2 and audit.get("fallback") is True
                    and audit.get("error") == "GraphMemoryError: reranker returned an unknown fact ID"
                    and not any(qid in ids[n] for n in ("prediction", "inference", "result")))
                if not eligible:
                    policy_invalid.append(dict(question_id=qid, path=str(path), reason="decision_or_failure_binding_mismatch"))
                    continue
                policy_ids.add(qid)
                policy_times.append(mtimes[str(path)])
                if mtimes[str(path)] >= since:
                    recent_policy_ids.add(qid)
    if policy_applies and (LOCOMO_FINAL_ROOT / "new_policy.json").is_file():
        extra_policy_path = LOCOMO_FINAL_ROOT / "new_policy.json"
        extra_policy = read(extra_policy_path) or {}
        allowed_counts = {"conv-43-q002": 7, "conv-48-q127": 1}
        valid = (extra_policy.get("policy_id") == "locomo_two_count_failures_zero_20261114"
            and set(extra_policy.get("question_ids", [])) == set(allowed_counts)
            and extra_policy.get("returned_counts") == allowed_counts and extra_policy.get("expected_count") == 20
            and Path(extra_policy.get("source_run", "")).resolve() == root
            and extra_policy.get("source_config_sha256") == file_hash(root / "config.json")
            and extra_policy.get("correct") == 0 and extra_policy.get("judge_called") is False
            and extra_policy.get("new_model_calls") == 0
            and extra_policy.get("does_not_authorize_future_other_failures") is True)
        if not valid:
            policy_invalid.append(dict(path=str(extra_policy_path), reason="additional_count_policy_contract_mismatch"))
        else:
            for qid, returned_count in allowed_counts.items():
                path = LOCOMO_FINAL_ROOT / "new_decisions" / f"{qid}.json"
                decision = read(path) or {}
                source = root / "questions" / qid / "failure.json"
                failure = failure_rows.get(qid) or {}
                audit = failure.get("audit") or {}
                eligible = (decision.get("question_id") == qid
                    and decision.get("score_origin") == "authorized_count_failure_zero"
                    and decision.get("correct") == 0 and decision.get("prediction") is None
                    and decision.get("model_judge_called") is False
                    and Path(decision.get("source", "")).resolve() == source
                    and decision.get("source_sha256") == file_hash(source)
                    and failure.get("error_type") == "StageFailure" and audit.get("attempts") == 2
                    and audit.get("error") == f"GraphMemoryError: reranker returned {returned_count} IDs; expected 20"
                    and not any(qid in ids[n] for n in ("prediction", "inference", "result")))
                if not eligible:
                    policy_invalid.append(dict(question_id=qid, path=str(path), reason="count_failure_binding_mismatch"))
                    continue
                policy_ids.add(qid)
                policy_times.append(mtimes[str(path)])
                if mtimes[str(path)] >= since:
                    recent_policy_ids.add(qid)
        summary_path, rows_path = LOCOMO_FINAL_ROOT / "summary.json", LOCOMO_FINAL_ROOT / "settled_rows.json"
        if summary_path.is_file() and rows_path.is_file():
            summary, rows = read(summary_path) or {}, read(rows_path) or []
            rows_ok = isinstance(rows, list) and len(rows) == len(expected_ids)
            seen = set()
            for item in rows if isinstance(rows, list) else []:
                qid = item.get("question_id")
                if qid in seen or qid not in expected_ids:
                    rows_ok = False
                seen.add(qid)
                if qid in native_results:
                    native = native_results[qid]
                    rows_ok = rows_ok and item.get("score_origin") == "actual_judge" and item.get("correct") == native["correct"]
                    rows_ok = rows_ok and item.get("prediction") == native["prediction"]
                else:
                    rows_ok = rows_ok and qid in policy_ids and item.get("correct") == 0
            if (rows_ok and seen == expected_ids and summary.get("complete") is True
                    and summary.get("source_config_sha256") == file_hash(root / "config.json")
                    and summary.get("overall", {}).get("n") == len(expected_ids)
                    and summary.get("overall", {}).get("correct") == sum(item["correct"] for item in rows)
                    and summary.get("actual_judge_count") == len(ids["result"])
                    and summary.get("invalid_id_policy_zero_count", 0) + summary.get("count_policy_zero_count", 0) == len(policy_ids)):
                final_artifact = dict(summary=str(summary_path), settled_rows=str(rows_path),
                    overall=summary["overall"], actual_judge_count=len(ids["result"]), policy_zero_count=len(policy_ids))
            else:
                policy_invalid.append(dict(path=str(summary_path), reason="final_artifact_population_or_score_mismatch"))
    if recovery_parent:
        # Each LME exception is named in the independent recovery registration;
        # no unrelated/future failure inherits this scoring policy.
        for prior_root in prior_roots:
            for path in sorted((prior_root / "questions").glob("*/failure.json")):
                item = read(path)
                if isinstance(item, dict) and item.get("question_id") in expected_ids:
                    failure_rows.setdefault(item["question_id"], item)
        allowed_ids = {LME_POLICY_QID} | (LME_ADDITIONAL_POLICY_IDS if policy_ancestor else set())
        old_registration = read(policy_parent / "recovery_registration.json") if policy_ancestor else policy_registration
        config_valid = (config.get("dataset") == "lme" and isinstance(old_registration, dict)
            and old_registration.get("original_config_sha256") == file_hash(root / "config.json")
            == file_hash(policy_parent / "config.json")
            and set(registration.get("policy_zero_ids", [])) == allowed_ids
            and set(policy_registration.get("policy_zero_ids", [])) == allowed_ids)
        if isolated_mode:
            config_valid = config_valid and registration_chain_valid
        for qid in sorted(allowed_ids):
            decision = policy_result_rows.get(qid) or {}
            decision_path = root / "questions" / qid / "result.json"
            old_policy = qid == LME_POLICY_QID
            source_root = (policy_ancestor or policy_parent) if old_policy else policy_parent
            source_hashes = policy_registration.get("ancestor_files_sha256" if old_policy and policy_ancestor else "parent_files_sha256", {})
            if policy_ancestor:
                expected_result_hash = (policy_registration.get("copied_files_sha256", {}).get(f"questions/{qid}/result.json")
                    if old_policy else policy_registration.get("new_policy_results_sha256", {}).get(qid))
            else:
                expected_result_hash = policy_registration.get("policy_result_sha256")
            source_failure = source_root / "questions" / qid / "failure.json"
            original_call = Path(decision.get("original_failure", "")).resolve()
            outcome_path = original_call / "outcome.json"
            path_valid = original_call.parent == source_root / "runtime" / "calls" and outcome_path.is_file()
            original_outcome = (read(outcome_path) or {}) if path_valid else {}
            eligible = (config_valid and decision_path.is_file() and source_failure.is_file() and path_valid
                and expected_result_hash == file_hash(decision_path)
                and source_hashes.get(f"questions/{qid}/failure.json") == file_hash(source_failure)
                and source_hashes.get(outcome_path.relative_to(source_root).as_posix()) == file_hash(outcome_path)
                and decision.get("question_id") == qid and decision.get("status") == "policy_zero"
                and type(decision.get("correct")) is int and decision["correct"] == 0
                and decision.get("prediction") is None and decision.get("judge") is None
                and decision.get("model_judge_called") is False and decision.get("generation_valid") is False
                and decision.get("score_origin") == ("user_explicit_single_question_policy_2026-11-14" if old_policy
                    else "user_explicit_two_question_policy_2026-11-14")
                and original_outcome.get("scope") == qid and original_outcome.get("stage") == "qa"
                and original_outcome.get("status") == "stopped" and original_outcome.get("finish_reason") == "MAX_TOKENS"
                and original_outcome.get("usage", {}).get("candidatesTokenCount") == 500
                and decision.get("original_response_sha256") == original_outcome.get("response_sha256")
                and not any(qid in ids[n] for n in ("prediction", "inference", "result")))
            if eligible:
                policy_ids.add(qid)
                authorized_policy_outcome_paths.add(str(outcome_path))
                if not inherited(decision_path):
                    policy_times.append(mtimes[str(decision_path)])
                    if mtimes[str(decision_path)] >= since:
                        recent_policy_ids.add(qid)
            else:
                policy_invalid.append(dict(question_id=qid, path=str(decision_path), reason="lme_policy_source_binding_mismatch"))
    for qid in policy_result_rows.keys() - policy_ids:
        bad.append(dict(question_id=qid, checkpoint="result", reason="unrecognized_or_unverified_policy_zero"))
    missing_ids, recent_missing_ids, missing_outcome_paths, missing_times = set(), set(), set(), []
    missing_usage_anomalies = []
    for qid, marker in missing_rows.items():
        marker = marker or {}
        marker_path = root / "questions" / qid / "missing.json"
        evidence = marker.get("evidence") or {}
        is_known = isolated_mode and qid in registration.get("known_missing_ids", [])
        call = Path(marker.get("original_failure") or evidence.get("call", "")).resolve()
        owner_roots = prior_roots if is_known else [root]
        physical_owner = next((p for p in owner_roots if call.parent in (
            p / "runtime" / "calls", p / "runtime" / "questions" / qid / "calls")), None)
        path_valid = physical_owner is not None and (call / "outcome.json").is_file()
        out = (read(call / "outcome.json") or {}) if path_valid else {}
        req = (read(call / "request.json") or {}) if path_valid else {}
        # Only failed QA responses are inspected here; never embedding arrays.
        raw = (read(call / "response.json") or {}) if path_valid and out.get("stage") == "qa" else {}
        candidates = raw.get("candidates") or []
        observed_output = raw.get("usageMetadata", {}).get("candidatesTokenCount")
        native_valid = (out.get("model") == out.get("model_version") == raw.get("modelVersion") == "gemini-2.5-flash"
            and req.get("model") == "gemini-2.5-flash" and req.get("kind") == "generation"
            and not raw.get("error") and out.get("physical_calls") == 1
            and out.get("request_key") == native_digest([req.get("model"), req.get("request")])
            and out.get("response_sha256") == native_digest(raw)
            and (observed_output is None or type(observed_output) is int and observed_output >= 0)
            and len(candidates) == 1 and candidates[0].get("finishReason") == "MAX_TOKENS"
            and not raw.get("promptFeedback", {}).get("blockReason")
            and not any(r.get("blocked") for r in raw.get("promptFeedback", {}).get("safetyRatings", []))
            and not any(r.get("blocked") for r in candidates[0].get("safetyRatings", [])))
        marker_valid = (isolated_mode and qid in expected_ids and marker.get("question_id") == qid
            and marker.get("status") == "missing" and marker.get("correct") is None and marker.get("judge") is None
            and marker.get("model_judge_called") is False and qid not in policy_ids
            and not any(qid in ids[k] for k in ("prediction", "inference", "result"))
            and path_valid and out.get("scope") == qid and out.get("stage") == "qa"
            and out.get("status") == "stopped" and out.get("finish_reason") == "MAX_TOKENS"
            and out.get("http_status") == 200 and out.get("model") == "gemini-2.5-flash"
            and req.get("request", {}).get("generationConfig", {}).get("maxOutputTokens") == 500 and native_valid)
        if is_known:
            physical_hashes = next((r.get("parent_files_sha256", r.get("source_files_sha256", {})) for r in registration_chain.values()
                if Path(r.get("parents", [r.get("parent", "")])[-1]).resolve() == physical_owner), {}) if path_valid else {}
            known_marker_hash = registration.get("known_missing_sha256", {}).get(qid)
            if judge_only_mode:
                known_marker_hash = registration.get("copied_files_sha256", {}).get(f"questions/{qid}/missing.json")
            physical_bound = (path_valid and physical_hashes.get((call / "outcome.json").relative_to(physical_owner).as_posix()) == file_hash(call / "outcome.json")
                and physical_hashes.get((call / "request.json").relative_to(physical_owner).as_posix()) == file_hash(call / "request.json"))
            if judge_only_mode and physical_owner == recovery_parent:
                # Judge-only recovery deliberately does not copy/hash the huge
                # embedding ledger. The immutable missing marker binds the QA
                # response digest; native request/response digests were checked
                # above, and no index or generation output is rewritten.
                physical_bound = (registration.get("source_files_sha256", {}).get(f"questions/{qid}/missing.json")
                                  == file_hash(recovery_parent / "questions" / qid / "missing.json"))
            marker_valid = (marker_valid and registration_chain_valid
                and known_marker_hash == file_hash(marker_path) and physical_bound
                and (marker.get("original_response_sha256") or evidence.get("response_sha256")) == out.get("response_sha256"))
        else:
            marker_valid = (marker_valid and evidence.get("response_sha256") == out.get("response_sha256")
                and evidence.get("request_key") == out.get("request_key"))
        if not marker_valid:
            bad.append(dict(question_id=qid, checkpoint="missing", reason="missing_marker_evidence_mismatch"))
            continue
        missing_ids.add(qid)
        missing_outcome_paths.add(str(call / "outcome.json"))
        if type(observed_output) is int and observed_output > 500:
            missing_usage_anomalies.append(dict(question_id=qid, requested_max_output_tokens=500,
                reported_output_tokens=observed_output, disposition="missing_not_zero_not_retried"))
        if not is_known:
            missing_times.append(mtimes[str(marker_path)])
            if mtimes[str(marker_path)] >= since:
                recent_missing_ids.add(qid)
    unresolved_failure, resolved_failure = [], []
    policy_settled_failures, isolated_missing_failures, queued_judge_failures = [], [], []
    for qid, item in sorted(failure_rows.items()):
        stage = item.get("stage", "inference")
        recovered = qid in ids["result"] or (stage != "judge" and qid in ids["inference"])
        record = dict(question_id=qid, stage=stage, error_type=item.get("error_type"),
                      transport_stopped=item.get("transport_stopped"))
        queued = registration.get("source_failures_queued", {}).get(qid, {}) if judge_only_mode else {}
        queued_path = recovery_parent / "questions" / qid / "failure.json" if judge_only_mode else None
        queue_valid = (judge_only_mode and bool(queued) and stage == "judge" and qid in ids["inference"]
            and not (root / "questions" / qid / "failure.json").exists()
            and not (root / "questions" / qid / "local_judge_failure.json").exists()
            and queued_path.is_file() and Path(queued.get("path", "")).resolve() == queued_path
            and queued.get("sha256") == file_hash(queued_path)
            == registration.get("source_files_sha256", {}).get(f"questions/{qid}/failure.json")
            and queued.get("error_type") == item.get("error_type"))
        if qid in policy_ids:
            policy_settled_failures.append({**record, "resolution": "authorized_policy_zero_no_judge"})
        elif qid in missing_ids:
            isolated_missing_failures.append({**record, "resolution": "retained_missing_not_zero_not_retried"})
        elif not recovered and queue_valid:
            queued_judge_failures.append({**record, "resolution": "queued_for_registered_judge_recovery_not_yet_scored"})
        else:
            (resolved_failure if recovered else unresolved_failure).append(record)

    index_ids, recent_indices, index_invalid = set(), set(), []
    for path in sorted((index_root / "indices").glob("*/complete.json")):
        item = read(path)
        if not item:
            continue
        cid = item.get("context_id")
        if (cid not in expected_contexts or cid != path.parent.name or not isinstance(item.get("chunks"), int)
                or not (path.parent / "chunks.json").is_file() or not (path.parent / "vectors.npy").is_file()):
            index_invalid.append(str(path))
            continue
        index_ids.add(cid)
        if mtimes[str(path)] >= since and not inherited(path):
            recent_indices.add(cid)

    outcomes, pending, successful_keys = [], [], defaultdict(list)
    runtime_roots = [root / "runtime"] + [p / "runtime" for p in prior_roots]
    call_bases = [(runtime_root, runtime_root.parent == root) for runtime_root in runtime_roots]
    for runtime_root in runtime_roots:
        call_bases.extend((qdir, runtime_root.parent == root) for qdir in (runtime_root / "questions").glob("*") if qdir.is_dir())
        call_bases.extend((worker, runtime_root.parent == root) for worker in runtime_root.glob("worker_*") if worker.is_dir())
    for call_base, current_runtime in call_bases:
      for folder_name in ("calls", "qwen_calls"):
        for folder in sorted((call_base / folder_name).glob("*")):
            if not folder.is_dir():
                continue
            op, req = folder / "outcome.json", folder / "request.json"
            if op.exists():
                value = read(op)
                if value is None:
                    continue
                value = {**value, "_path": str(op), "_mtime": mtimes[str(op)], "_inherited": not current_runtime}
                outcomes.append(value)
                if value.get("status") == "ok" and value.get("request_key"):
                    successful_keys[value["request_key"]].append(value["_mtime"])
            elif req.is_file() and current_runtime:
                value = read(req) or {}
                age = max(0, now-mtimes.get(str(req), now))
                pending.append(dict(path=str(req), scope=value.get("scope"), stage=value.get("stage", "qwen_rerank"),
                    age_seconds=round(age, 1), state="orphan_no_runner" if not active and not process_error else
                    "long_pending_check_required" if age > 330 else "in_flight_or_recently_dispatched"))

    def downstream_resolved(item):
        scope, stage = str(item.get("scope", "")), item.get("stage")
        if scope.endswith("/index"):
            return scope[:-6] in index_ids
        if scope not in expected_ids:
            return False
        if stage == "judge":
            return scope in ids["result"]
        if stage in ("qa", "generation"):
            return scope in ids["prediction"] or scope in ids["inference"]
        if stage in ("reranker", "qwen_rerank"):
            if scope in ids["prediction"] or scope in ids["inference"]:
                return True
            selected = read(root/"questions"/scope/"selected.json") if scope in ids["selected"] else None
            return isinstance(selected, dict) and isinstance(selected.get("selected_ids"), list) and bool(selected["selected_ids"])
        if stage == "query_embedding":
            if scope in ids["prediction"] or scope in ids["inference"]:
                return True
            retrieved = read(root/"questions"/scope/"retrieval.json") if scope in ids["retrieval"] else None
            return isinstance(retrieved, dict) and retrieved.get("question_id") == scope and isinstance(retrieved.get("candidate_ids"), list)
        return False

    unresolved_outcomes, recovered_outcomes, policy_outcomes, isolated_outcomes = [], [], [], []
    stage_stats = defaultdict(lambda: dict(outcomes=0, physical_calls=0, statuses=Counter(),
        known_input_tokens=0, input_usage_missing_calls=0, known_output_tokens=0,
        known_thought_tokens=0, known_cached_input_tokens=0, summed_service_seconds=0.0))
    cost = dict(gemini_generation_known_usd=0.0, gemini_standard_input_known_usd=0.0,
                qwen_known_cny=0.0, embedding_usd=None, unknown_usage_nonjudge_calls=0,
                pricing_date="2026-11-11", judge_excluded=True)
    durable_call_times, recent_ok = [], Counter()
    for item in outcomes:
        stage, status = str(item.get("stage", "unknown")), str(item.get("status", "unknown"))
        physical = int(item.get("physical_calls", bool(item.get("attempted", True))))
        stat = stage_stats[stage]
        stat["outcomes"] += 1
        stat["physical_calls"] += physical
        stat["statuses"][status] += 1
        stat["summed_service_seconds"] += float(item.get("service_seconds") or 0)
        native = item.get("usage") or {}
        inp = native.get("promptTokenCount", native.get("input_tokens", native.get("total_tokens")))
        output = native.get("candidatesTokenCount", native.get("output_tokens"))
        thoughts, cached = int(native.get("thoughtsTokenCount") or 0), int(native.get("cachedContentTokenCount") or 0)
        if isinstance(inp, (int, float)):
            stat["known_input_tokens"] += inp
        else:
            stat["input_usage_missing_calls"] += physical
        if isinstance(output, (int, float)):
            stat["known_output_tokens"] += output
        stat["known_thought_tokens"] += thoughts
        stat["known_cached_input_tokens"] += cached
        if stage != "judge":
            if not isinstance(inp, (int, float)):
                cost["unknown_usage_nonjudge_calls"] += physical
            elif item.get("model") == "gemini-2.5-flash":
                if isinstance(output, (int, float)):
                    cost["gemini_generation_known_usd"] += ((inp-cached)*.30+cached*.03+(output+thoughts)*2.50)/1e6
                    cost["gemini_standard_input_known_usd"] += (inp*.30+(output+thoughts)*2.50)/1e6
                else:
                    cost["unknown_usage_nonjudge_calls"] += physical
            elif item.get("model") == "qwen3-rerank":
                cost["qwen_known_cny"] += inp*.50/1e6
        if status == "ok":
            if not item["_inherited"]:
                durable_call_times.append(item["_mtime"])
            if item["_mtime"] >= since and not item["_inherited"]:
                recent_ok[stage] += 1
            continue
        exact_recovered = any(t >= item["_mtime"] for t in successful_keys.get(item.get("request_key"), []))
        downstream = downstream_resolved(item)
        record = {k: item.get(k) for k in ("scope", "stage", "status", "http_status", "error_type", "retry_permitted", "finish_reason")}
        record["output_token_count"] = output
        record.update(path=item["_path"], recovered_by="later_exact_request_success" if exact_recovered else
                      "downstream_checkpoint" if downstream else None)
        if item["_path"] in authorized_policy_outcome_paths:
            policy_outcomes.append({**record, "resolution": "authorized_policy_zero_no_judge"})
        elif item["_path"] in missing_outcome_paths:
            isolated_outcomes.append({**record, "resolution": "retained_missing_not_zero_not_retried"})
        else:
            (recovered_outcomes if exact_recovered or downstream else unresolved_outcomes).append(record)

    timings = []
    for path in sorted((root / "execution_timing").glob("*.json")):
        item = read(path)
        if item and (item.get("phase") == "full" or isolated_mode):
            timings.append((mtimes[str(path)], path.name, item))
    execution_records = []
    for path in sorted((root / "executions").glob("*.json")):
        item = read(path)
        if isinstance(item, dict):
            execution_records.append((mtimes[str(path)], item))
    latest_execution = max(execution_records, default=(0, {}), key=lambda pair: pair[0])
    global_stop = ((root / "STOP").is_file() or bool(latest_execution[1].get("transport_stopped")
                   and latest_execution[0] >= full_start))
    current_outcomes = [x for x in outcomes if not x["_inherited"]]
    current_call_counts = Counter()
    for item in current_outcomes:
        current_call_counts[str(item.get("stage", "unknown"))] += int(item.get("physical_calls", bool(item.get("attempted", True))))
    judge_only_stage_violations = [{"path": x["_path"], "stage": x.get("stage"), "scope": x.get("scope")}
        for x in current_outcomes if judge_only_mode and x.get("stage") != "judge"]
    latest_stage = max(current_outcomes, key=lambda x: x["_mtime"], default={}).get("stage")
    latest_stage_time = max((x["_mtime"] for x in current_outcomes), default=0)
    inference_done_marker = max((t for t, name, _ in timings if name.endswith("_inference.json") and t >= full_start), default=0)
    settled_ids = ids["result"] | policy_ids
    accounted_ids = settled_ids | missing_ids
    if len(settled_ids) == len(expected_ids):
        phase = "complete"
    elif len(accounted_ids) == len(expected_ids):
        phase = "finished_with_unscored_missing"
    elif judge_only_mode:
        phase = "judge"
    elif latest_stage == "judge" and latest_stage_time >= full_start:
        phase = "judge"
    elif len(ids["inference"] | policy_ids | missing_ids) == len(expected_ids):
        phase = "generation_finished_judge_pending"
    elif inference_done_marker and not active:
        phase = "generation_halted_with_missing"
    else:
        phase = "generation"
    durable_paths = [root / "questions" / qid / f"{name}.json"
                     for name in ("prediction", "inference", "result") for qid in ids[name]]
    durable_times = [mtimes[str(path)] for path in durable_paths if not inherited(path)]
    durable_times += [mtimes[str(index_root/"indices"/cid/"complete.json")] for cid in index_ids
                     if not inherited(index_root/"indices"/cid/"complete.json")]
    durable_times += durable_call_times
    durable_times += policy_times
    durable_times += missing_times
    latest_durable = max(durable_times, default=None)
    latest_activity = max(mtimes.values(), default=None)
    blocking_errors = bool(unresolved_failure or unresolved_outcomes or bad or index_invalid or policy_invalid or judge_only_stage_violations)
    if global_stop:
        state = "global_stop_recorded"
    elif blocking_errors and phase in ("complete", "finished_with_unscored_missing"):
        state = "accounted_rows_with_unresolved_errors"
    elif phase == "complete":
        state = "complete_with_authorized_policy_zeros" if policy_ids else "complete"
    elif phase == "finished_with_unscored_missing":
        state = "complete_with_missing"
    elif not active and not process_error:
        state = "stopped_with_missing_results"
    elif blocking_errors:
        state = "running_with_unresolved_errors" if active else "process_unknown_with_errors"
    elif process_error:
        state = "process_state_unknown"
    elif latest_durable is not None and now-latest_durable > 330:
        state = "running_no_recent_durable_progress_check_required"
    elif isolated_mode and missing_ids:
        state = "running_with_isolated_missing"
    else:
        state = "running"
    return dict(root=str(root), dataset=config["dataset"], timestamp_utc=utc(now), state=state, phase=phase,
        expected_questions=len(expected_ids), active_processes=active, process_query_error=process_error,
        unique_valid_checkpoints={k: len(v) for k, v in ids.items() if k not in auxiliary_names},
        auxiliary_checkpoint_files_present={k: len(ids[k]) for k in sorted(auxiliary_names)},
        source_failure_count=len(failure_rows), policy_settled_count=len(policy_ids),
        policy_settled_ids=sorted(policy_ids), total_settled_count=len(settled_ids),
        native_judge_result_count=len(ids["result"]), policy_invalid_decisions=policy_invalid,
        missing_count=len(missing_ids), missing_ids=sorted(missing_ids),
        missing_usage_anomalies=missing_usage_anomalies,
        all_questions_accounted=accounted_ids == expected_ids, global_stop_recorded=global_stop,
        unaccounted_question_count=len(expected_ids - accounted_ids),
        final_settlement_artifact=final_artifact,
        index_complete=dict(n=len(index_ids), expected=len(expected_contexts), invalid_receipts=index_invalid,
            source_output=str(index_root), inherited_source=judge_only_mode,
            note="Completion receipt plus required files present; full vector hashes are not re-read on every status check."),
        last_window_seconds=window,
        new_durable_checkpoints={**{k: len(v) for k, v in recent.items() if k not in auxiliary_names},
                                "index_complete": len(recent_indices), "policy_settled": len(recent_policy_ids),
                                "missing": len(recent_missing_ids)},
        new_successful_physical_outcomes_by_stage=dict(recent_ok),
        last_durable_utc=utc(latest_durable), seconds_since_durable=round(now-latest_durable, 1) if latest_durable else None,
        latest_observed_activity_utc=utc(latest_activity), latest_completed_runtime_stage=latest_stage,
        unresolved_failures=unresolved_failure, resolved_historical_failure_records=resolved_failure,
        policy_settled_failure_records=policy_settled_failures,
        queued_judge_recovery_failure_records=queued_judge_failures,
        isolated_missing_failure_records=isolated_missing_failures,
        isolated_missing_non_ok_outcomes=isolated_outcomes,
        unresolved_non_ok_outcomes=unresolved_outcomes, recovered_historical_non_ok_outcomes=recovered_outcomes,
        pending_requests=pending, invalid_checkpoints=bad, read_errors=errors,
        recovery=dict(parent=str(recovery_parent), inherited_prediction_count=sum(
            f"questions/{qid}/prediction.json" in inherited_paths for qid in ids["prediction"]),
            new_prediction_count=sum(f"questions/{qid}/prediction.json" not in inherited_paths for qid in ids["prediction"]),
            new_inference_count=sum(f"questions/{qid}/inference.json" not in inherited_paths for qid in ids["inference"]),
            new_judge_result_count=sum(f"questions/{qid}/result.json" not in inherited_paths for qid in ids["result"]),
            inherited_judge_result_count=sum(f"questions/{qid}/result.json" in inherited_paths for qid in ids["result"]),
            judge_only=judge_only_mode,
            resumed_progress_confirmed=bool(any(call.get("status") == "ok" for call in current_outcomes) or any(
                f"questions/{qid}/{name}.json" not in inherited_paths for name in ("inference", "result") for qid in ids[name])) )
            if recovery_parent else None,
        policy_settled_non_ok_outcomes=policy_outcomes,
        physical_stage_usage=dict(stage_stats), current_known_inference_cost=cost,
        current_root_physical_calls_by_stage=dict(current_call_counts),
        judge_only_stage_violations=judge_only_stage_violations,
        note="Recent means valid files with modification time within the window, not an inferred counter delta. "
             "Counts are unique manifest-bound IDs. Inference/QA checkpoints and authorized policy zeros are not native Judge results. "
             "LoCoMo policy decisions are accepted only after source config/failure/retrieval SHA and exact failure-class checks. "
             "LME recovery distinguishes inherited checkpoints from new output; named policy results and stopped physical outcomes are source-hash bound. "
             "Large auxiliary contexts/traces are only counted as present; embedding response arrays are never read. "
             "Unknown token usage is not zero. Physical costs are partial observed subtotals, not full E2E cost. "
             "This script neither reads .env nor imports the runner/transport, calls APIs, writes state, or resumes work.")


def current_lme_root():
    """Choose the newest completed registration, not a merely created folder."""
    for root, schema, parent in ((LME_JUDGE_ROOT6, "hybrid_rag.judge_only_finish_r6.v1", LME_RECOVERY_ROOT5),
                                 (LME_RECOVERY_ROOT5, "hybrid_rag.isolated_native_maxtokens_r5.v1", LME_RECOVERY_ROOT4),
                                 (LME_RECOVERY_ROOT4, "hybrid_rag.isolated_qa500_missing.v1", LME_RECOVERY_ROOT3),
                                 (LME_RECOVERY_ROOT3, "hybrid_rag.two_more_policy_zeros.v1", LME_RECOVERY_ROOT),
                                 (LME_RECOVERY_ROOT, "hybrid_rag.single_policy_recovery.v1", DEFAULT_ROOTS[1])):
        try:
            registration = json.loads((root / "recovery_registration.json").read_text(encoding="utf-8-sig"))
            parent_valid = (Path(registration.get("parent", "")).resolve() == parent.resolve())
            if root in (LME_RECOVERY_ROOT4, LME_RECOVERY_ROOT5):
                parent_valid = [Path(p).resolve() for p in registration.get("parents", [])] == [
                    root.with_name(f"lme_r{i}").resolve() for i in range(1, 4 if root == LME_RECOVERY_ROOT4 else 5)]
            elif root == LME_JUDGE_ROOT6:
                parent_valid = [Path(p).resolve() for p in registration.get("parents", [])] == [
                    root.with_name(f"lme_r{i}").resolve() for i in range(1, 6)]
            if (registration.get("schema") == schema and Path(registration.get("output", "")).resolve() == root.resolve()
                    and parent_valid):
                return root
        except (OSError, ValueError):
            continue
    return DEFAULT_ROOTS[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", help="Repeat for multiple run roots; defaults to both current runs")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    procs, error = processes()
    roots = args.root or [DEFAULT_ROOTS[0], current_lme_root()]
    results = [snapshot(root, args.window, procs, error) for root in roots]
    disks = {}
    for drive in ("C:/", "D:/"):
        try:
            disks[drive] = dict(free_gib=round(shutil.disk_usage(drive).free/1024**3, 2))
        except OSError:
            disks[drive] = dict(free_gib=None)
    payload = dict(timestamp_utc=utc(time.time()), runs=results, disks=disks,
        disk_stop_required=disks["D:/"]["free_gib"] is not None and disks["D:/"]["free_gib"] < 15)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for run in results:
        if "unique_valid_checkpoints" not in run:
            print(f"{run['root']}: {run['state']}")
            continue
        counts, inc = run["unique_valid_checkpoints"], run["new_durable_checkpoints"]
        pids = ",".join(str(p["pid"]) for p in run["active_processes"]) or "none"
        print(f"{run['dataset']}: {run['state']} | phase={run['phase']} | PID={pids}")
        print(f"  prediction/inference/result={counts['prediction']}/{counts['inference']}/{counts['result']} "
              f"of {run['expected_questions']}; index={run['index_complete']['n']}/{run['index_complete']['expected']}")
        print(f"  source_failures={run['source_failure_count']}; policy_settled={run['policy_settled_count']}; "
              f"total_settled={run['total_settled_count']}/{run['expected_questions']} (native Judge={run['native_judge_result_count']})")
        if run.get("missing_count") or run.get("global_stop_recorded"):
            print(f"  unscored missing={run['missing_count']}; global_stop={run['global_stop_recorded']}; "
                  f"all_accounted={run['all_questions_accounted']}")
        print(f"  last{args.window}s +prediction/inference/result/index={inc['prediction']}/{inc['inference']}/{inc['result']}/{inc['index_complete']}; "
              f"unresolved failures/outcomes={len(run['unresolved_failures'])}/{len(run['unresolved_non_ok_outcomes'])}; "
              f"recovered old outcomes={len(run['recovered_historical_non_ok_outcomes'])}; pending={len(run['pending_requests'])}")
        for issue in run["unresolved_failures"][:5]:
            print(f"  failure: {issue['question_id']} stage={issue['stage']} type={issue['error_type']}")
        for issue in run["unresolved_non_ok_outcomes"][:5]:
            print(f"  runtime: scope={issue['scope']} stage={issue['stage']} status={issue['status']} HTTP={issue['http_status']}")
        if run.get("recovery"):
            recovery = run["recovery"]
            print(f"  recovery: inherited predictions={recovery['inherited_prediction_count']}; "
                  f"new prediction/inference={recovery['new_prediction_count']}/{recovery['new_inference_count']}; "
                  f"new durable progress={recovery['resumed_progress_confirmed']}")
            if recovery.get("judge_only"):
                print(f"  Judge-only: new valid Judge results={recovery['new_judge_result_count']}; "
                      f"indices inherited from {run['index_complete']['source_output']}")
        if run.get("final_settlement_artifact"):
            overall = run["final_settlement_artifact"]["overall"]
            print(f"  verified final: {overall['correct']}/{overall['n']} = {overall['accuracy_percent']:.2f}%")
    print(f"Free GiB: C={disks['C:/']['free_gib']}, D={disks['D:/']['free_gib']}; D<15 stop={payload['disk_stop_required']}")


if __name__ == "__main__":
    main()

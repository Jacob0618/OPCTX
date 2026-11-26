"""Offline, deterministic LME100 resampling for the clean Writer comparison.

The source and two excluded 100-question selections are byte-pinned. Selection
uses only question IDs, question type, and the ID's abstention suffix: neither
answer content nor any prior prediction/score influences selection or ordering.
Labels in this audit manifest must not be forwarded to inference components.

The default CLI only prints an audit summary. ``--freeze`` explicitly persists
the manifest without overwriting a different existing artifact. No model or
network client is imported by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path.cwd().resolve()
SEED = 20260905
SCHEMA_VERSION = "creamem3.lme_clean100.selection.v1"
SOURCE_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
FIRST_MANIFEST_SHA256 = "d4c446b1eb3bfb58e0cd54910b5453e0e352ca325199dae1fb3e26e1342e593c"
REPEAT_SELECTION_FILE_SHA256 = "0d1abf7122d9f58f933828e215ac544138bfb7cbf469e598c2a972c16bdda0af"
REPEAT_SELECTION_SHA256 = "32ff65be0a1c784dae2ce1588963b06a2a32df82c821cfabb4e5f3aa24a6029f"
DEFAULT_DATASET = ROOT / "data" / "longmemeval" / "longmemeval_s.json"
DEFAULT_FIRST_MANIFEST = ROOT / "creamem3" / "audit" / "lme100_stratified_20260826" / "lme100_manifest.json"
DEFAULT_REPEAT_SELECTION = ROOT / "creamem3" / "results_lme_opctx100_repeat_from400_v2_b3_d768_q6w1_20260902_r1" / "selection.json"
DEFAULT_MANIFEST = ROOT / "creamem3" / "audit" / "lme_clean100_20260905" / "manifest.json"

TYPE_QUOTAS = {
    "knowledge-update": 16,
    "multi-session": 27,
    "single-session-assistant": 11,
    "single-session-preference": 6,
    "single-session-user": 14,
    "temporal-reasoning": 26,
}
ABSTENTION_QUOTAS = {
    "knowledge-update": 1,
    "multi-session": 3,
    "single-session-assistant": 0,
    "single-session-preference": 0,
    "single-session-user": 1,
    "temporal-reasoning": 1,
}


def canonical_sha256(value: Any) -> str:
    """Match the old frozen manifest's exact record canonicalization."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rank(label: str, question_id: str, question_type: str = "", abstention: bool = False) -> str:
    payload = "\0".join((str(SEED), label, question_type, str(int(abstention)), question_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_pinned(path: Path, expected_sha256: str, label: str) -> Any:
    _require(sha256_file(path) == expected_sha256, f"{label} file SHA256 mismatch")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _validate_source(data: Any) -> list[dict[str, Any]]:
    _require(isinstance(data, list) and len(data) == 500, "source must contain exactly 500 records")
    seen = set()
    for index, item in enumerate(data):
        _require(isinstance(item, dict), f"source[{index}] must be an object")
        qid = item.get("question_id")
        _require(isinstance(qid, str) and bool(qid) and qid not in seen,
                 f"source[{index}] has missing or duplicate question ID")
        seen.add(qid)
        _require(item.get("question_type") in TYPE_QUOTAS, f"unknown type for {qid}")
        arrays = [item.get(key) for key in ("haystack_sessions", "haystack_dates", "haystack_session_ids")]
        _require(all(isinstance(value, list) for value in arrays)
                 and len(arrays[0]) > 0 and len({len(value) for value in arrays}) == 1,
                 f"unaligned source session arrays for {qid}")
        _require(all(isinstance(session, list) for session in arrays[0]), f"malformed sessions for {qid}")
    return data


def _verify_exclusion_items(items: Any, data: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    _require(isinstance(items, list) and len(items) == 100, f"{label} must bind 100 records")
    ids = []
    for position, item in enumerate(items):
        _require(isinstance(item, dict), f"{label} binding must be an object")
        index = item.get("source_index")
        _require(type(index) is int and 0 <= index < len(data), f"{label} source index invalid")
        record = data[index]
        _require(item.get("position") == position, f"{label} position mismatch")
        _require(item.get("question_id") == record["question_id"], f"{label} ID mismatch")
        _require(item.get("record_sha256") == canonical_sha256(record), f"{label} record SHA mismatch")
        ids.append(record["question_id"])
    _require(len(set(ids)) == 100, f"{label} contains duplicate IDs")
    return ids


def selected_records(data: Sequence[Mapping[str, Any]], excluded_ids: set[str]) -> list[tuple[int, Mapping[str, Any]]]:
    """Rank only ID/type/abstention metadata, never answers or old outcomes."""
    source_ids = [item["question_id"] for item in data]
    _require(len(source_ids) == len(set(source_ids)), "duplicate source IDs")
    _require(len(excluded_ids) == 200 and excluded_ids <= set(source_ids), "expected 200 source-bound exclusions")
    strata: dict[tuple[str, bool], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(data):
        qid = item["question_id"]
        if qid not in excluded_ids:
            strata[(item["question_type"], qid.endswith("_abs"))].append((index, item))
    _require(sum(map(len, strata.values())) == 300, "expected exactly 300 eligible questions")
    chosen = []
    for qtype, total in TYPE_QUOTAS.items():
        for is_abs, quota in ((True, ABSTENTION_QUOTAS[qtype]), (False, total - ABSTENTION_QUOTAS[qtype])):
            pool = sorted(strata[(qtype, is_abs)], key=lambda pair: (
                _rank("select", pair[1]["question_id"], qtype, is_abs), pair[1]["question_id"]))
            _require(len(pool) >= quota, f"insufficient eligible stratum {qtype}/{is_abs}: {len(pool)} < {quota}")
            chosen.extend(pool[:quota])
    chosen.sort(key=lambda pair: (_rank("order", pair[1]["question_id"]), pair[1]["question_id"]))
    _require(len(chosen) == len({record["question_id"] for _, record in chosen}) == 100,
             "selection must contain 100 distinct records")
    return chosen


def load_inputs(*, dataset_path: Path = DEFAULT_DATASET,
                first_manifest_path: Path = DEFAULT_FIRST_MANIFEST,
                repeat_selection_path: Path = DEFAULT_REPEAT_SELECTION,
                manifest_path: Path | None = DEFAULT_MANIFEST
                ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Verify pinned inputs and regenerate; if frozen, require exact equality.

    Returns source records and ordered selected records for existing runner
    compatibility. Callers must use their existing allowlisted inference
    projection; answers, types, and abstention labels remain evaluator-only.
    This function does not create or change any file.
    """
    data = _validate_source(_read_pinned(dataset_path, SOURCE_SHA256, "source dataset"))
    first = _read_pinned(first_manifest_path, FIRST_MANIFEST_SHA256, "first100 manifest")
    repeat_envelope = _read_pinned(repeat_selection_path, REPEAT_SELECTION_FILE_SHA256, "repeat100 selection")
    repeat = repeat_envelope.get("selection")
    _require(isinstance(repeat, dict) and repeat.get("limit") == 100
             and repeat_envelope.get("selection_sha256") == REPEAT_SELECTION_SHA256
             and canonical_sha256(repeat) == REPEAT_SELECTION_SHA256, "repeat100 selection canonical SHA mismatch")
    first_ids = _verify_exclusion_items(first.get("items"), data, "first100")
    _require(first.get("count") == 100 and first.get("ids") == first_ids, "first100 IDs mismatch")
    repeat_ids = _verify_exclusion_items(repeat.get("items"), data, "repeat100")
    _require(not set(first_ids).intersection(repeat_ids), "excluded selections must be disjoint")
    excluded = set(first_ids) | set(repeat_ids)
    chosen = selected_records(data, excluded)
    items = [{"position": position, "question_id": item["question_id"], "source_index": source_index,
              "record_sha256": canonical_sha256(item), "question_type": item["question_type"],
              "type": item["question_type"], "is_abstention": item["question_id"].endswith("_abs"),
              "abstention": item["question_id"].endswith("_abs")}
             for position, (source_index, item) in enumerate(chosen)]
    selection = {"seed": SEED, "method": "sha256_rank_within_type_and_abstention_then_separate_order_rank",
                 "rank_payload": "UTF8(NUL.join(seed,label,question_type,int(abstention),question_id))",
                 "selection_label": "select", "order_label": "order", "order_type": "", "order_abstention": False,
                 "tie_breaker": "question_id", "type_quotas": dict(TYPE_QUOTAS),
                 "abstention_quotas": dict(ABSTENTION_QUOTAS), "abstention_rule": "question_id.endswith('_abs')",
                 "eligible_count": 300, "limit": 100, "items": items}
    manifest = {
        "schema_version": SCHEMA_VERSION, "name": "lme_clean100_excluding_first100_and_same78_seed20260905",
        "source": {"dataset": "data/longmemeval/longmemeval_s.json", "sha256": SOURCE_SHA256, "record_count": 500},
        "exclusions": [
            {"path": "creamem3/audit/lme100_stratified_20260826/lme100_manifest.json",
             "file_sha256": FIRST_MANIFEST_SHA256, "count": 100, "ordered_ids_sha256": canonical_sha256(first_ids)},
            {"path": "creamem3/results_lme_opctx100_repeat_from400_v2_b3_d768_q6w1_20260902_r1/selection.json",
             "file_sha256": REPEAT_SELECTION_FILE_SHA256, "selection_sha256": REPEAT_SELECTION_SHA256,
             "count": 100, "ordered_ids_sha256": canonical_sha256(repeat_ids)}],
        "excluded_ids_sha256": canonical_sha256(sorted(excluded)), "selection": selection,
        "selection_sha256": canonical_sha256(selection), "count": 100,
        "ids": [item["question_id"] for item in items], "items": items,
        "runtime_constraints": {
            "selection_uses_only": ["question_id", "question_type", "question_id_suffix_abstention"],
            "answer_text_or_prior_outcomes_used_for_selection": False,
            "old_prediction_score_or_error_artifacts_read": False,
            "inference_must_not_receive": ["answer", "question_type", "type", "is_abstention", "abstention", "answer_session_ids"],
            "record_hash_includes_labels_for_integrity_only": True,
            "manifest_labels_are_audit_only": True,
            "test_status": "new_stratified_retest_subset_not_unseen_holdout_previous_full500_included_all_source_records",
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if manifest_path is not None and Path(manifest_path).exists():
        with Path(manifest_path).open(encoding="utf-8") as stream:
            frozen = json.load(stream)
        _require(frozen == manifest, "frozen clean100 manifest differs from deterministic regeneration")
    return data, manifest, [data[index] for index, _ in chosen]


def persist_manifest(manifest: Mapping[str, Any] | None = None, path: Path = DEFAULT_MANIFEST) -> Path:
    """Explicitly freeze a validated manifest; existing different data is refused."""
    expected = load_inputs(manifest_path=None)[1]
    if manifest is not None:
        _require(dict(manifest) == expected, "provided manifest differs from deterministic regeneration")
    target = Path(path)
    if target.exists():
        with target.open(encoding="utf-8") as stream:
            _require(json.load(stream) == expected, "refusing to overwrite a different existing manifest")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(expected, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Explicitly persist the verified manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    _, manifest, selected = load_inputs(manifest_path=args.manifest)
    if args.freeze:
        persist_manifest(manifest, args.manifest)
    print(json.dumps({"mode": "freeze" if args.freeze else "dry_no_write", "count": len(selected),
                      "seed": SEED, "manifest_sha256": manifest["manifest_sha256"],
                      "selection_sha256": manifest["selection_sha256"],
                      "type_counts": dict(Counter(item["question_type"] for item in selected)),
                      "abstention_counts": dict(Counter(item["question_type"] for item in selected if item["question_id"].endswith("_abs"))),
                      "manifest_path": str(args.manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

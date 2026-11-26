"""Validated loader for every conversation in LoCoMo's locomo10.json.

The source dataset contains a handful of malformed evidence annotations.  QA
rows are never dropped for that reason: uniquely recoverable ID formatting is
normalized, while unresolved evidence is retained in an explicit audit field
and excluded only from source-ID recall diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CATEGORY_NAMES: dict[int, str] = {
    1: "multi",
    2: "temporal",
    3: "open",
    4: "single",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "locomo" / "locomo10.json"

_SESSION_RE = re.compile(r"session_(\d+)\Z")
_TIMESTAMP_RE = re.compile(r"session_(\d+)_date_time\Z")
_DIA_ID_RE = re.compile(r"D(\d+):(\d+)\Z")
_DIA_ID_FIND_RE = re.compile(r"D\d+:\d+")
_MALFORMED_THREE_PART_RE = re.compile(r"D:(\d+):(\d+)\Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _session_ids(conversation: Mapping[str, Any]) -> list[int]:
    ids: list[int] = []
    for key, value in conversation.items():
        match = _SESSION_RE.fullmatch(key)
        if match is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{key} is not a turn list")
        ids.append(int(match.group(1)))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("conversation has no sessions or duplicate numeric IDs")
    return sorted(ids)


def _render_text_only(record: Mapping[str, Any]) -> str:
    session_id = int(record["session_id"])
    timestamp = str(record.get("timestamp", "") or "")
    iso_date = ""
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %B %d, %Y",
        "%d %B, %Y",
        "%B %d, %Y",
    ):
        try:
            iso_date = datetime.strptime(timestamp.strip(), fmt).date().isoformat()
            break
        except ValueError:
            continue
    header = f"[Session {session_id}"
    if timestamp:
        header += f" at {timestamp}"
    if iso_date:
        header += f"; ISO date {iso_date}"
    lines = [header + "]"]
    for turn in record["turns"]:
        speaker = re.sub(r"\s+", " ", str(turn.get("speaker", "") or "")).strip()
        text = re.sub(r"\s+", " ", str(turn.get("text", "") or "")).strip()
        lines.append(
            f"{speaker} ({turn['dia_id']}): {text}"
        )
    return "\n".join(lines)


def _build_records(
    conversation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    dia_id_map: dict[str, dict[str, Any]] = {}
    global_turn_index = 0
    for record_index, session_id in enumerate(_session_ids(conversation)):
        timestamp = str(
            conversation.get(f"session_{session_id}_date_time", "") or ""
        )
        turns = [_copy(turn) for turn in conversation[f"session_{session_id}"]]
        for turn_index, turn in enumerate(turns):
            dia_id = turn.get("dia_id")
            if not isinstance(dia_id, str):
                raise ValueError(
                    f"session {session_id} turn {turn_index + 1} lacks dia_id"
                )
            match = _DIA_ID_RE.fullmatch(dia_id)
            expected = f"D{session_id}:{turn_index + 1}"
            if match is None or dia_id != expected:
                raise ValueError(
                    f"dia_id {dia_id!r} does not match physical position {expected}"
                )
            if dia_id in dia_id_map:
                raise ValueError(f"duplicate dia_id {dia_id}")
            dia_id_map[dia_id] = {
                "session_id": session_id,
                "timestamp": timestamp,
                "record_index": record_index,
                "turn_index": turn_index,
                "turn_number": turn_index + 1,
                "global_turn_index": global_turn_index,
                "speaker": str(turn.get("speaker", "")),
                "has_image": bool(turn.get("img_url")),
                "has_caption": bool(turn.get("blip_caption")),
            }
            global_turn_index += 1
        record = {
            "session_id": session_id,
            "timestamp": timestamp,
            "turns": turns,
        }
        record["text_only_sha256"] = hashlib.sha256(
            _render_text_only(record).encode("utf-8")
        ).hexdigest()
        records.append(record)
    return records, dia_id_map


def _normalize_candidate(raw_id: str) -> str | None:
    match = _DIA_ID_RE.fullmatch(raw_id)
    if match is None:
        return None
    return f"D{int(match.group(1))}:{int(match.group(2))}"


def _normalize_evidence(
    raw_evidence: Any, valid_ids: Mapping[str, Any]
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    normalized: list[str] = []
    repairs: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    if raw_evidence is None:
        return normalized, repairs, unresolved
    if not isinstance(raw_evidence, list):
        return normalized, repairs, [
            {"raw": repr(raw_evidence), "reason": "evidence is not a list"}
        ]
    for item in raw_evidence:
        if not isinstance(item, str):
            unresolved.append(
                {"raw": repr(item), "reason": "evidence item is not a string"}
            )
            continue
        found = _DIA_ID_FIND_RE.findall(item)
        remainder = _DIA_ID_FIND_RE.sub("", item).strip(" ;,\t\r\n")
        item_resolved = False
        if found and not remainder:
            for raw_id in found:
                canonical = _normalize_candidate(raw_id)
                if canonical in valid_ids:
                    if canonical != raw_id:
                        repairs.append({"raw": raw_id, "normalized": canonical})
                    if canonical not in normalized:
                        normalized.append(canonical)
                    item_resolved = True
                else:
                    unresolved.append(
                        {"raw": raw_id, "reason": "source ID is absent"}
                    )
            continue
        malformed = _MALFORMED_THREE_PART_RE.fullmatch(item.strip())
        if malformed:
            canonical = f"D{int(malformed.group(1))}:{int(malformed.group(2))}"
            if canonical in valid_ids:
                repairs.append({"raw": item, "normalized": canonical})
                if canonical not in normalized:
                    normalized.append(canonical)
                item_resolved = True
        if not item_resolved:
            unresolved.append(
                {"raw": item, "reason": "unrecognized or unverifiable evidence ID"}
            )
    return normalized, repairs, unresolved


def _filtered_qa(
    sample_id: str, source_qa: Any, dia_id_map: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(source_qa, list):
        raise ValueError("sample qa must be a list")
    output: list[dict[str, Any]] = []
    occurrences: Counter[str] = Counter()
    for source_index, qa in enumerate(source_qa):
        if not isinstance(qa, dict):
            raise ValueError(f"QA row {source_index} is not an object")
        category = qa.get("category")
        if category not in CATEGORY_NAMES:
            continue
        raw_evidence = _copy(qa.get("evidence", []))
        ids, repairs, unresolved = _normalize_evidence(raw_evidence, dia_id_map)
        question = str(qa.get("question", ""))
        question_occurrence = occurrences[question]
        occurrences[question] += 1
        output.append(
            {
                "qa_id": f"{sample_id}-q{source_index:03d}",
                "source_index": source_index,
                "question": question,
                "question_occurrence": question_occurrence,
                "answer": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": int(category),
                "category_name": CATEGORY_NAMES[int(category)],
                "evidence": raw_evidence,
                "evidence_dia_ids": ids,
                "evidence_repairs": repairs,
                "evidence_unresolved": unresolved,
            }
        )
    return output


def _select_sample(data: Sequence[Any], selector: str | int) -> tuple[int, dict[str, Any]]:
    if isinstance(selector, int) or str(selector).isdigit():
        index = int(selector)
        if index < 0 or index >= len(data):
            raise IndexError(f"conversation index {index} is outside 0..{len(data)-1}")
        sample = data[index]
        if not isinstance(sample, dict):
            raise ValueError(f"sample index {index} is not an object")
        return index, sample
    target = str(selector)
    matches = [
        (index, sample)
        for index, sample in enumerate(data)
        if isinstance(sample, dict) and str(sample.get("sample_id", "")) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"sample selector {target!r} matched {len(matches)} rows")
    return matches[0]


def load_locomo_conversation(
    path: str | Path = DEFAULT_DATASET_PATH,
    selector: str | int = 0,
) -> dict[str, Any]:
    dataset_path = Path(path).resolve()
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LoCoMo root must be a list")
    sample_index, sample = _select_sample(data, selector)
    sample_id = str(sample.get("sample_id", "") or "")
    if not sample_id:
        raise ValueError(f"sample {sample_index} lacks sample_id")
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError(f"{sample_id} conversation is not an object")
    records, dia_id_map = _build_records(conversation)
    qa = _filtered_qa(sample_id, sample.get("qa", []), dia_id_map)
    timestamp_session_ids = {
        int(match.group(1))
        for key in conversation
        if (match := _TIMESTAMP_RE.fullmatch(key)) is not None
    }
    actual_session_ids = [int(record["session_id"]) for record in records]
    all_turns = [turn for record in records for turn in record["turns"]]
    hashes = {
        "file_sha256": file_sha256(dataset_path),
        "dataset_sha256": canonical_sha256(sample),
        "qa_sha256": canonical_sha256(qa),
    }
    stats = {
        "sessions": len(records),
        "session_ids": actual_session_ids,
        "turns": len(all_turns),
        "unique_dia_ids": len(dia_id_map),
        "qa_total": len(sample.get("qa", [])),
        "qa_kept_categories_1_to_4": len(qa),
        "qa_dropped": len(sample.get("qa", [])) - len(qa),
        "qa_by_category": {
            str(category): sum(1 for row in qa if row["category"] == category)
            for category in CATEGORY_NAMES
        },
        "qa_with_any_resolved_evidence": sum(
            bool(row["evidence_dia_ids"]) for row in qa
        ),
        "qa_with_unresolved_evidence": sum(
            bool(row["evidence_unresolved"]) for row in qa
        ),
        "evidence_repairs": sum(len(row["evidence_repairs"]) for row in qa),
        "evidence_unresolved": sum(len(row["evidence_unresolved"]) for row in qa),
        "timestamp_only_session_ids": sorted(
            timestamp_session_ids - set(actual_session_ids)
        ),
        "turns_with_image": sum(bool(turn.get("img_url")) for turn in all_turns),
        "turns_with_caption": sum(
            bool(turn.get("blip_caption")) for turn in all_turns
        ),
    }
    return {
        "schema_version": "creamem3.locomo_conversation.v2",
        "sample_id": sample_id,
        "sample_index": sample_index,
        "dataset_path": str(dataset_path),
        "speakers": {
            "speaker_a": str(conversation.get("speaker_a", "")),
            "speaker_b": str(conversation.get("speaker_b", "")),
        },
        "records": records,
        "qa": qa,
        "dia_id_map": dia_id_map,
        "hashes": hashes,
        "stats": stats,
    }


def load_all_locomo(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    dataset_path = Path(path).resolve()
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LoCoMo root must be a list")
    return [load_locomo_conversation(dataset_path, index) for index in range(len(data))]


def build_manifest(loaded: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": loaded["schema_version"],
        "sample_id": loaded["sample_id"],
        "sample_index": loaded["sample_index"],
        "dataset_path": loaded["dataset_path"],
        "hashes": dict(loaded["hashes"]),
        "stats": dict(loaded["stats"]),
        "sessions": [
            {
                "session_id": record["session_id"],
                "timestamp": record["timestamp"],
                "turn_count": len(record["turns"]),
                "first_dia_id": record["turns"][0]["dia_id"],
                "last_dia_id": record["turns"][-1]["dia_id"],
                "text_only_sha256": record["text_only_sha256"],
            }
            for record in loaded["records"]
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--conversation", default="all")
    args = parser.parse_args(list(argv) if argv is not None else None)
    loaded = (
        load_all_locomo(args.dataset)
        if args.conversation == "all"
        else [load_locomo_conversation(args.dataset, args.conversation)]
    )
    print(
        json.dumps(
            [build_manifest(item) for item in loaded],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATEGORY_NAMES",
    "DEFAULT_DATASET_PATH",
    "build_manifest",
    "canonical_sha256",
    "file_sha256",
    "load_all_locomo",
    "load_locomo_conversation",
]

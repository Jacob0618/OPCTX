"""Read-only authority gate and fast predict entry point for the thinking round.

The canonical build at ``isolated_full_arch_thinking512_low_r1`` has already
passed ``build_executor_v19.validate_build_complete``.  Replaying all 64 graph
semantics in every later process takes more than twenty CPU minutes.  This
isolated entry point replaces only that repeated replay with a byte-level,
fixed-path authority check.  It performs no provider call and writes nothing
while validating the gate.

Gate only (the safe default)::

    python -m creamem3.run_memeye_thinking_predict_fast_v19

Run predict after the gate succeeds::

    python -m creamem3.run_memeye_thinking_predict_fast_v19 \
        --run-predict --allow-model-calls --workers 32
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from creamem3 import build_executor_v19 as build
from creamem3 import graph_memory_v2 as graph_memory
from creamem3 import memeye_matched_ingest_v19 as ingest
from creamem3 import predict_executor_v19 as predict
from creamem3 import run_memeye_full_arch_matrix_v19 as matrix
from creamem3 import run_memeye_isolated_full_arch_retry5_v19 as base
from creamem3 import run_memeye_isolated_full_arch_thinking512_low_v19 as thinking


SCHEMA_VERSION = "creamem3.memeye_thinking_fast_build_authority.v19.0"

# These values bind this opt-in shortcut to the build that completed and then
# passed the canonical deep validator.  Any later build or source-code change
# must be deeply validated once and deliberately repinned here.
EXPECTED_BUILD_COMPLETE_CONTRACT_SHA256 = (
    "9a8fed66373d9865ca711e87abb709bbce40d323867bfacd1f53b577bea0d95e"
)
EXPECTED_BUILD_COMPLETE_FILE_SHA256 = (
    "1ed143cde18b50528fb7b9c7d4937542fec6a7152ec03135b3ab65b3c82c74fa"
)
EXPECTED_BUILD_EXECUTOR_SHA256 = (
    "d60319c2e9769196626457c4b1a5258d58775ac13540950a759e1090aea1904c"
)
EXPECTED_MATCHED_INGEST_SHA256 = (
    "1e10624ee8e90f4e557964b9227ca9eab4e9e6f6f00b90dd382788d1dada72ef"
)
EXPECTED_GRAPH_RETRIEVAL_SHA256 = (
    "c74806d7668aa539041cad0a4d8686fb4376ee7b2af54169135d17f04329ff96"
)
EXPECTED_PREDICT_EXECUTOR_SHA256 = (
    "511c9c8f7ac54b3e323ecca16c1a5133b9cfdf75eb5e06aca8c48960b5c8ff81"
)
EXPECTED_MATRIX_RUNNER_SHA256 = (
    "cc78dc6c7d78229cfac2a2658578a6e59122a7b8bf4c8f491b2f2722ef9dacad"
)
EXPECTED_BASE_RUNNER_SHA256 = (
    "0eb799cf1f37ae2d4d02a239affad97cd0b0e776b8fe0d0343c3a4e261a9eef0"
)
EXPECTED_THINKING_RUNNER_SHA256 = (
    "aa7f5d1d9dd11fef799201b5e8032e63dc55067dec9e5e9936b8a6e6e496e104"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "8852b5bd3642a11e9e6b2f01cbc54d05bb2da0f8a9097119f1115fac0d35c232"
)
EXPECTED_QUERY_FILE_SHA256 = (
    "e6fb2d7baba42670b3bb0c15bed2497393c44a21ab2d750e9d4bec87142ab71e"
)
EXPECTED_PROJECT_CONTRACT_SHA256 = (
    "c1f793d9779b06dbe13638bbebe486810f81ccbe6b316a10229a270c334ee5e7"
)
EXPECTED_PROTOCOL_SHA256 = (
    "053d1b3d32f594b6afa75ecab6bd60b8675152f1450c0ab06ffbd8933ed438d0"
)
EXPECTED_WRITER_COMPLETE_FILE_SHA256 = (
    "6e1ccf83ce22df4e72d8dec3a68c47f855c510823e03b469bd8ab0ad1643d41f"
)
EXPECTED_WRITER_EVIDENCE_FILE_SHA256 = (
    "3bea614de349af598bad5be5fe8e9a96f080f3a0e6c81eea8d902fa533fc07c2"
)
EXPECTED_WRITER_PROJECTION_FILE_SHA256 = (
    "e10e52a9c7556962464be1b8ef19325e3eda96ddc6cbafde975d959ef098b341"
)
EXPECTED_THINKING_CONTRACT_FILE_SHA256 = (
    "8f2eede95369f0c35badbb37e412d15f2d3898c8296503d51013a756f136966a"
)

BACKBONES = ("g25", "g35")
CATEGORIES = ("mgtm", "offline", "online", "session")


class FastBuildAuthorityError(RuntimeError):
    """The byte-level authority gate disagrees with the accepted build."""


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FastBuildAuthorityError(f"missing/unreadable authority file: {path}") from exc
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise FastBuildAuthorityError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FastBuildAuthorityError(f"{label} is not an object: {path}")
    return value


def _validate_digest(value: Mapping[str, Any], key: str, label: str) -> str:
    body = dict(value)
    observed = body.pop(key, None)
    if not isinstance(observed, str) or observed != _sha256(body):
        raise FastBuildAuthorityError(f"{label} internal digest drift")
    return observed


def _require_file_sha(path: Path, expected: str, label: str) -> str:
    observed = _file_sha256(path)
    if observed != expected:
        raise FastBuildAuthorityError(
            f"{label} byte drift: expected {expected}, observed {observed}"
        )
    return observed


def _validate_source_authority(source_root: Path, output_root: Path) -> tuple[list[str], dict[str, str]]:
    if source_root != thinking.DEFAULT_SOURCE_PROJECT_ROOT.resolve():
        raise FastBuildAuthorityError("fast gate is restricted to the canonical source root")
    manifest_path = source_root / "project" / "manifest.json"
    query_path = source_root / "project" / "queries_score_free.json"
    _require_file_sha(
        manifest_path, EXPECTED_SOURCE_MANIFEST_FILE_SHA256, "source project manifest"
    )
    _require_file_sha(query_path, EXPECTED_QUERY_FILE_SHA256, "source query bank")
    _require_file_sha(
        output_root / "project" / "queries_score_free.json",
        EXPECTED_QUERY_FILE_SHA256,
        "isolated query bank",
    )
    manifest = _read_object(manifest_path, "source project manifest")
    project_sha = _validate_digest(
        manifest, "project_contract_sha256", "source project manifest"
    )
    if (
        project_sha != EXPECTED_PROJECT_CONTRACT_SHA256
        or manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or manifest.get("query_file_sha256") != EXPECTED_QUERY_FILE_SHA256
        or manifest.get("raw_data_paths_exposed_to_build_or_predict") is not False
        or manifest.get("counts")
        != {
            "tasks": 8,
            "questions": 371,
            "physical_sessions": 221,
            "image_references": 438,
            "unique_images": 438,
            "two_stage_rows": 306,
            "structured_fused_rows": 65,
        }
    ):
        raise FastBuildAuthorityError("source project authority contract drift")
    task_rows = manifest.get("task_receipts")
    if not isinstance(task_rows, list) or len(task_rows) != 8:
        raise FastBuildAuthorityError("source task receipt census drift")
    tasks: list[str] = []
    memory_shas: dict[str, str] = {}
    expected_memory_paths: set[Path] = set()
    for row in task_rows:
        if not isinstance(row, Mapping):
            raise FastBuildAuthorityError("source task receipt shape drift")
        task = str(row.get("task", ""))
        memory_path = (source_root / "project" / "memory" / f"{task}.json").resolve()
        if (
            not task
            or Path(str(row.get("memory_path", ""))).resolve() != memory_path
            or not isinstance(row.get("memory_sha256"), str)
        ):
            raise FastBuildAuthorityError("source task memory authority drift")
        memory_sha = _require_file_sha(
            memory_path, str(row["memory_sha256"]), f"source memory {task}"
        )
        tasks.append(task)
        memory_shas[task] = memory_sha
        expected_memory_paths.add(memory_path)
    observed_memory_paths = {
        path.resolve()
        for path in (source_root / "project" / "memory").glob("*.json")
        if path.is_file()
    }
    if len(set(tasks)) != 8 or observed_memory_paths != expected_memory_paths:
        raise FastBuildAuthorityError("source memory filesystem census drift")
    return sorted(tasks), memory_shas


def _validate_runner_authorities() -> dict[str, str]:
    authorities = {
        "build_executor": (Path(build.__file__).resolve(), EXPECTED_BUILD_EXECUTOR_SHA256),
        "matched_ingest": (Path(ingest.__file__).resolve(), EXPECTED_MATCHED_INGEST_SHA256),
        "graph_retrieval": (
            Path(graph_memory.__file__).resolve(),
            EXPECTED_GRAPH_RETRIEVAL_SHA256,
        ),
        "predict_executor": (
            Path(predict.__file__).resolve(),
            EXPECTED_PREDICT_EXECUTOR_SHA256,
        ),
        "matrix_runner": (Path(matrix.__file__).resolve(), EXPECTED_MATRIX_RUNNER_SHA256),
        "base_runner": (Path(base.__file__).resolve(), EXPECTED_BASE_RUNNER_SHA256),
        "thinking_runner": (
            Path(thinking.__file__).resolve(),
            EXPECTED_THINKING_RUNNER_SHA256,
        ),
    }
    return {
        name: _require_file_sha(path, expected, f"{name} authority")
        for name, (path, expected) in authorities.items()
    }


def _validate_writer_seals(root: Path, complete: Mapping[str, Any]) -> dict[str, str]:
    writer_path = root / "build" / "writer_complete.json"
    evidence_path = root / "build" / "writer_generation_evidence.json"
    projection_path = root / "build" / "writer_projection_complete.json"
    contract_path = root / "build" / "thinking_round_contract.json"
    if (
        complete.get("writer_complete_file_sha256")
        != EXPECTED_WRITER_COMPLETE_FILE_SHA256
        or complete.get("writer_generation_evidence_file_sha256")
        != EXPECTED_WRITER_EVIDENCE_FILE_SHA256
    ):
        raise FastBuildAuthorityError("BUILD_COMPLETE writer seal binding drift")
    seals = {
        "writer_complete": _require_file_sha(
            writer_path, EXPECTED_WRITER_COMPLETE_FILE_SHA256, "writer complete"
        ),
        "writer_generation_evidence": _require_file_sha(
            evidence_path, EXPECTED_WRITER_EVIDENCE_FILE_SHA256, "writer evidence"
        ),
        "writer_projection_complete": _require_file_sha(
            projection_path, EXPECTED_WRITER_PROJECTION_FILE_SHA256, "writer projection"
        ),
        "thinking_round_contract": _require_file_sha(
            contract_path, EXPECTED_THINKING_CONTRACT_FILE_SHA256, "thinking contract"
        ),
    }
    writer = _read_object(writer_path, "writer complete")
    evidence = _read_object(evidence_path, "writer generation evidence")
    projection = _read_object(projection_path, "writer projection complete")
    thinking_contract = _read_object(contract_path, "thinking round contract")
    if (
        _validate_digest(writer, "writer_complete_sha256", "writer complete")
        != "76e9460a62057caa14be49f75b099b12540f88ee841572d98c924a8215f8dcc9"
        or writer.get("complete") is not True
        or writer.get("project_contract_sha256") != EXPECTED_PROJECT_CONTRACT_SHA256
        or writer.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        raise FastBuildAuthorityError("writer completion authority drift")
    if (
        _validate_digest(
            evidence, "generation_evidence_sha256", "writer generation evidence"
        )
        != "f4a4b272b6e8cb682b18ffb6dd28988ff18a76b5600dd29e223a2229e20fbf10"
        or evidence.get("writer_complete_file_sha256")
        != EXPECTED_WRITER_COMPLETE_FILE_SHA256
        or evidence.get("provider_calls_during_validation") != 0
    ):
        raise FastBuildAuthorityError("writer generation evidence drift")
    if (
        _validate_digest(projection, "complete_sha256", "writer projection complete")
        != "956ef2ee2c23f7d559f86cf856e86a6c23574522c84a6dedb01529d2c3b4da8b"
        or Path(str(projection.get("output_root", ""))).resolve() != root
        or projection.get("writer_complete_file_sha256")
        != EXPECTED_WRITER_COMPLETE_FILE_SHA256
        or projection.get("writer_generation_evidence_file_sha256")
        != EXPECTED_WRITER_EVIDENCE_FILE_SHA256
        or projection.get("raw_to_projection_to_packet_validated") is not True
        or projection.get("provider_finals_synthesized") != 0
        or projection.get("additional_provider_calls") != 0
        or projection.get("provider_calls") != 0
    ):
        raise FastBuildAuthorityError("writer projection seal drift")
    if (
        _validate_digest(thinking_contract, "contract_sha256", "thinking contract")
        != "4dbd0192b90bf9b93ce7bca0735dd630f535d9cffc1d0509178d62b020e9f436"
        or Path(str(thinking_contract.get("root", ""))).resolve() != root
    ):
        raise FastBuildAuthorityError("thinking round contract drift")
    return seals


def _validate_artifacts_and_receipts(
    root: Path, complete: Mapping[str, Any], tasks: Sequence[str]
) -> dict[str, Any]:
    rows = complete.get("artifacts")
    if not isinstance(rows, list) or len(rows) != 64:
        raise FastBuildAuthorityError("BUILD_COMPLETE artifact cardinality drift")
    expected_ids = {
        (backbone, task, category)
        for backbone in BACKBONES
        for task in tasks
        for category in CATEGORIES
    }
    observed_ids: set[tuple[str, str, str]] = set()
    artifact_paths: set[Path] = set()
    receipt_paths: set[Path] = set()
    receipt_file_rows: list[dict[str, str]] = []
    artifact_by_id: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FastBuildAuthorityError("BUILD_COMPLETE artifact row shape drift")
        identity = (
            str(row.get("backbone", "")),
            str(row.get("task", "")),
            str(row.get("category", "")),
        )
        if identity not in expected_ids or identity in observed_ids:
            raise FastBuildAuthorityError("BUILD_COMPLETE artifact identity drift")
        backbone, task, category = identity
        artifact_path = (
            root / "build" / "graphs" / backbone / category / f"{task}.json"
        ).resolve()
        receipt_path = (
            root / "build" / "receipts" / backbone / category / f"{task}.json"
        ).resolve()
        if Path(str(row.get("artifact_path", ""))).resolve() != artifact_path:
            raise FastBuildAuthorityError("BUILD_COMPLETE artifact fixed-path drift")
        artifact_sha = _require_file_sha(
            artifact_path, str(row.get("artifact_sha256", "")), "build graph artifact"
        )
        receipt = _read_object(receipt_path, "build graph receipt")
        receipt_contract_sha = _validate_digest(
            receipt, "receipt_sha256", "build graph receipt"
        )
        if (
            receipt_contract_sha != row.get("receipt_sha256")
            or receipt.get("backbone") != backbone
            or receipt.get("task") != task
            or receipt.get("category") != category
            or Path(str(receipt.get("artifact_path", ""))).resolve() != artifact_path
            or receipt.get("artifact_sha256") != artifact_sha
            or receipt.get("content_fingerprint") != row.get("content_fingerprint")
        ):
            raise FastBuildAuthorityError("build graph receipt binding drift")
        # Receipts are write-once canonical JSON.  Reconstructing those exact
        # bytes makes the recomputed file SHA meaningful, not merely reported.
        # ``Path.write_text`` used by the canonical writer translates every
        # newline to CRLF on this Windows build host.
        canonical_receipt_bytes = (
            json.dumps(receipt, ensure_ascii=False, indent=2).replace("\n", "\r\n")
            + "\r\n"
        ).encode("utf-8")
        receipt_file_sha = _file_sha256(receipt_path)
        if receipt_file_sha != _sha256(canonical_receipt_bytes):
            raise FastBuildAuthorityError("build graph receipt canonical-byte drift")
        observed_ids.add(identity)
        artifact_paths.add(artifact_path)
        receipt_paths.add(receipt_path)
        artifact_by_id[identity] = row
        receipt_file_rows.append(
            {
                "backbone": backbone,
                "task": task,
                "category": category,
                "artifact_file_sha256": artifact_sha,
                "receipt_file_sha256": receipt_file_sha,
                "receipt_contract_sha256": receipt_contract_sha,
            }
        )
    if observed_ids != expected_ids:
        raise FastBuildAuthorityError("build artifact identity census drift")
    observed_artifact_paths = {
        path.resolve()
        for path in (root / "build" / "graphs").rglob("*.json")
        if path.is_file()
    }
    observed_receipt_paths = {
        path.resolve()
        for path in (root / "build" / "receipts").rglob("*.json")
        if path.is_file()
    }
    if observed_artifact_paths != artifact_paths or observed_receipt_paths != receipt_paths:
        raise FastBuildAuthorityError("build artifact/receipt filesystem census drift")

    architecture = complete.get("architecture_bindings")
    if not isinstance(architecture, list) or len(architecture) != 16:
        raise FastBuildAuthorityError("BUILD_COMPLETE architecture cardinality drift")
    expected_architecture: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for task in tasks:
            mgtm = artifact_by_id[(backbone, task, "mgtm")]
            session = artifact_by_id[(backbone, task, "session")]
            online = artifact_by_id[(backbone, task, "online")]
            offline = artifact_by_id[(backbone, task, "offline")]
            expected_architecture.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "mgtm_matched_path": mgtm["artifact_path"],
                    "mgtm_matched_sha256": mgtm["artifact_sha256"],
                    "session_only_path": session["artifact_path"],
                    "session_only_sha256": session["artifact_sha256"],
                    "online_base_path": online["artifact_path"],
                    "online_base_sha256": online["artifact_sha256"],
                    "online_operation_context_v3_path": online["artifact_path"],
                    "online_operation_context_v3_sha256": online["artifact_sha256"],
                    "offline_base_path": offline["artifact_path"],
                    "offline_base_sha256": offline["artifact_sha256"],
                }
            )
    if architecture != expected_architecture:
        raise FastBuildAuthorityError("BUILD_COMPLETE architecture binding drift")
    receipt_file_rows.sort(key=lambda row: (row["backbone"], row["task"], row["category"]))
    return {
        "artifacts_verified": len(artifact_paths),
        "receipts_verified": len(receipt_paths),
        "architecture_bindings_verified": len(architecture),
        "artifact_receipt_file_rows_sha256": _sha256(receipt_file_rows),
    }


def validate_fast_build_authority(
    *,
    source_project_root: Path = thinking.DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path = thinking.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate the accepted build without semantic replay or external I/O."""

    root = Path(output_root).resolve()
    source_root = Path(source_project_root).resolve()
    if root != thinking.DEFAULT_OUTPUT_ROOT.resolve():
        raise FastBuildAuthorityError("fast gate is restricted to the canonical thinking root")
    complete_path = root / "build" / "complete.json"
    complete_file_sha = _require_file_sha(
        complete_path, EXPECTED_BUILD_COMPLETE_FILE_SHA256, "BUILD_COMPLETE"
    )
    complete = _read_object(complete_path, "BUILD_COMPLETE")
    complete_contract_sha = _validate_digest(
        complete, "complete_sha256", "BUILD_COMPLETE"
    )
    if (
        complete_contract_sha != EXPECTED_BUILD_COMPLETE_CONTRACT_SHA256
        or complete.get("complete") is not True
        or complete.get("executor_sha256") != EXPECTED_BUILD_EXECUTOR_SHA256
        or complete.get("matched_ingest_sha256") != EXPECTED_MATCHED_INGEST_SHA256
        or complete.get("project_contract_sha256") != EXPECTED_PROJECT_CONTRACT_SHA256
        or complete.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or complete.get("legacy_cache_readthrough") is not False
        or complete.get("predict_or_score_implemented") is not False
    ):
        raise FastBuildAuthorityError("BUILD_COMPLETE accepted authority drift")
    protocol = matrix.load_protocol()
    if _sha256(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise FastBuildAuthorityError("protocol authority drift")
    tasks, source_memory_shas = _validate_source_authority(source_root, root)
    runner_authorities = _validate_runner_authorities()
    writer_seals = _validate_writer_seals(root, complete)
    artifact_census = _validate_artifacts_and_receipts(root, complete, tasks)
    plan = _read_object(root / "build" / "plan.json", "sealed build plan")
    if (
        _validate_digest(plan, "plan_sha256", "sealed build plan")
        != complete.get("plan_sha256")
        or Path(str(plan.get("output_root", ""))).resolve() != root
        or plan.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        raise FastBuildAuthorityError("sealed build plan drift")
    for field, name in (
        ("embedding_budget_file_sha256", "embedding_budget.json"),
        ("dynamic_embedding_gate_file_sha256", "dynamic_embedding_budget_gate.json"),
    ):
        expected = str(complete.get(field, ""))
        if len(expected) != 64:
            raise FastBuildAuthorityError(f"BUILD_COMPLETE {field} drift")
        _require_file_sha(root / "build" / name, expected, name)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "canonical_thinking_build_fast_authority_approved",
        "decision": "approve",
        "output_root": str(root),
        "source_project_root": str(source_root),
        "build_complete_contract_sha256": complete_contract_sha,
        "build_complete_file_sha256": complete_file_sha,
        "source_memory_rows_sha256": _sha256(source_memory_shas),
        "runner_authorities": runner_authorities,
        "writer_seals": writer_seals,
        **artifact_census,
        "semantic_replay_performed": False,
        "provider_calls": 0,
        "network_calls": 0,
        "files_written": 0,
    }


@contextmanager
def _accepted_build_reuse_scope() -> Iterator[None]:
    """Route the base runner through its already-validated immutable fast path."""

    names = (
        "DEFAULT_OUTPUT_ROOT",
        "VALIDATED_BUILD_COMPLETE_FILE_SHA256",
        "VALIDATED_BUILD_COMPLETE_CONTRACT_SHA256",
        "VALIDATED_BUILD_EXECUTOR_SHA256",
    )
    previous = {name: getattr(base, name) for name in names}
    base.DEFAULT_OUTPUT_ROOT = thinking.DEFAULT_OUTPUT_ROOT
    base.VALIDATED_BUILD_COMPLETE_FILE_SHA256 = EXPECTED_BUILD_COMPLETE_FILE_SHA256
    base.VALIDATED_BUILD_COMPLETE_CONTRACT_SHA256 = (
        EXPECTED_BUILD_COMPLETE_CONTRACT_SHA256
    )
    base.VALIDATED_BUILD_EXECUTOR_SHA256 = EXPECTED_BUILD_EXECUTOR_SHA256
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run_predict(
    *,
    source_project_root: Path = thinking.DEFAULT_SOURCE_PROJECT_ROOT,
    output_root: Path = thinking.DEFAULT_OUTPUT_ROOT,
    workers: int = thinking.DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    gate = validate_fast_build_authority(
        source_project_root=source_project_root, output_root=output_root
    )
    with _accepted_build_reuse_scope():
        result = thinking.run_predict(
            source_project_root=source_project_root,
            output_root=output_root,
            workers=workers,
        )
    return {
        **result,
        "fast_build_authority": copy.deepcopy(gate),
        "fast_build_authority_used": True,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-project-root", type=Path, default=thinking.DEFAULT_SOURCE_PROJECT_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=thinking.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=thinking.DEFAULT_EVAL_WORKERS)
    parser.add_argument("--run-predict", action="store_true")
    parser.add_argument("--allow-model-calls", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.run_predict != args.allow_model_calls:
        raise SystemExit("--run-predict and --allow-model-calls must be supplied together")
    result = (
        run_predict(
            source_project_root=args.source_project_root,
            output_root=args.output_root,
            workers=args.workers,
        )
        if args.run_predict
        else validate_fast_build_authority(
            source_project_root=args.source_project_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

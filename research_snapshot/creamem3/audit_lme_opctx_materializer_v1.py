"""Zero-network parity audit for the LME Top-K OpCtx adapter.

The LME adapter is allowed to widen the frozen final selection from 20 to 30,
but it is not allowed to alter any other Online/OpCtx v3 rendering behavior.
This audit replays all 1,540 frozen LoCoMo development rows at Top-20 and
requires byte-identical contexts plus identical routing and selected/view IDs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from creamem3.graph_memory_v2 import canonical_sha256
from creamem3.lme_opctx_v1 import materialize_operation_context, sha256_text
from creamem3.online_operation_context_v8 import (
    materialize_online_operation_context,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "creamem3" / "results_autoresearch_v8" / "online_operation_context_v3"
DEFAULT_RECEIPT = ROOT / "creamem3" / "audit" / "lme_opctx_v1" / "locomo1540_materializer_parity.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def audit() -> dict[str, Any]:
    checks = 0
    mismatches: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    operation_counts: dict[str, int] = {}
    for frozen_path in sorted(FROZEN_ROOT.glob("conv-*/online_operation_context_v3.json")):
        frozen = _load(frozen_path)
        graph_path = Path(str(frozen["source_graph_path"]))
        source_path = Path(str(frozen["source_result_path"]))
        graph = _load(graph_path)
        source = _load(source_path)
        source_rows = {str(row["qa_id"]): row for row in source["rows"]}
        source_hashes[str(frozen["sample_id"])] = {
            "frozen_result": sha256_text(frozen_path.read_text(encoding="utf-8")),
            "source_result": sha256_text(source_path.read_text(encoding="utf-8")),
            "source_graph": sha256_text(graph_path.read_text(encoding="utf-8")),
        }
        for frozen_row in frozen["rows"]:
            qa_id = str(frozen_row["qa_id"])
            base_row = source_rows[qa_id]
            frozen_audit = frozen_row["retrieval_trace"]["online_operation_context_v8"]["materializer_audit"]
            operation = str(frozen_audit["operation"])
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            sanitized = {
                "question": str(base_row["question"]),
                "retrieved": str(base_row["retrieved"]),
                "retrieval_trace": base_row["retrieval_trace"],
            }
            official_context, official_audit = materialize_online_operation_context(
                operation, sanitized, graph
            )
            adapted_context, adapted_audit = materialize_operation_context(
                operation=operation,
                question=str(base_row["question"]),
                base_retrieved=str(base_row["retrieved"]),
                retrieval_trace=base_row["retrieval_trace"],
                graph=graph,
                selected_top_k=20,
            )
            checks += 1
            fields = {
                "context": official_context == adapted_context,
                "context_policy": official_audit["context_policy"] == adapted_audit["context_policy"],
                "selected_fact_ids": official_audit["selected_fact_ids"] == adapted_audit["selected_fact_ids"],
                "rendered_event_view_ids": official_audit["rendered_event_view_ids"] == adapted_audit["rendered_event_view_ids"],
                "rendered_session_summary_ids": official_audit["rendered_session_summary_ids"] == adapted_audit["rendered_session_summary_ids"],
            }
            if not all(fields.values()):
                mismatches.append(
                    {
                        "qa_id": qa_id,
                        "sample_id": str(frozen["sample_id"]),
                        "operation": operation,
                        "field_equality": fields,
                        "official_context_sha256": sha256_text(official_context),
                        "adapted_context_sha256": sha256_text(adapted_context),
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": "creamem3.lme_opctx_materializer_parity.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "network_calls": 0,
        "model_calls": 0,
        "source_root": str(FROZEN_ROOT),
        "rows_checked": checks,
        "expected_rows": 1540,
        "operation_counts": operation_counts,
        "source_hashes": source_hashes,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": checks == 1540 and not mismatches,
        "claim": "Top20 adapter is byte-identical to frozen Online/OpCtx v3; LME changes only final_k to 30",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("rows_checked", "mismatch_count", "passed", "receipt_sha256")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""B3 LongMemEval adapter with a per-physical-session Writer fact budget.

This is an isolated experimental variant of :mod:`creamem3.lme_opctx_v2`.
It keeps three consecutive physical sessions in each causal frontier and
changes only the Writer contract:

* each physical session contributes a soft target of 10--70 facts;
* a frontier containing ``n`` sessions therefore requests ``10*n``--``70*n``;
* coverage must be balanced across the physical sessions;
* the model must not pad the list with duplicates or unverifiable claims; and
* the Writer output allowance is 16,384 tokens so the 210-fact B3 ceiling is
  not accidentally truncated by the 8,192-token baseline allowance.

The JSON shape remains ``batch_summary`` plus one flat ``atomic_facts`` list.
No minimum fact count is enforced or retried; requested and observed counts
are recorded for later audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from creamem3 import graph_memory_v2 as gm
from creamem3 import lme_opctx_v2 as base


SCHEMA_VERSION = "creamem3.lme_online_opctx.v2_b3_scaled_facts.v1"
FRONTIER_SIZE = base.FRONTIER_SIZE
EMBEDDING_DIM = base.EMBEDDING_DIM
EMBEDDING_MODEL = base.EMBEDDING_MODEL
EMBEDDING_TASK_TYPE_SENT = base.EMBEDDING_TASK_TYPE_SENT
EMBEDDING_NORMALIZATION = base.EMBEDDING_NORMALIZATION

LME_DATASET = base.LME_DATASET
MGTM_ROOT = base.MGTM_ROOT
MGTM_PROMPT_SOURCE = base.MGTM_PROMPT_SOURCE
MGTM_MEMORY_SOURCE = base.MGTM_MEMORY_SOURCE
MGTM_CLIENT_SOURCE = base.MGTM_CLIENT_SOURCE
MGTM_RUNNER_SOURCE = base.MGTM_RUNNER_SOURCE
NonRetryableProviderError = base.NonRetryableProviderError
PromptContractError = base.PromptContractError
QwenGraphMemoryIndex = base.QwenGraphMemoryIndex
materialize_operation_context = base.materialize_operation_context
render_lme_frontier = base.render_lme_frontier
render_lme_session = base.render_lme_session
group_session_specs = base.group_session_specs
sha256_text = base.sha256_text
legacy = base.legacy

WRITE_MAX_ATTEMPTS = base.WRITE_MAX_ATTEMPTS
WRITE_RETRY_BACKOFF_S = base.WRITE_RETRY_BACKOFF_S
WRITE_MAX_TOKENS = 16_384
FACTS_PER_PHYSICAL_SESSION_SOFT_MIN = 10
FACTS_PER_PHYSICAL_SESSION_SOFT_MAX = 70

_BASE_FACT_RULE = (
    "For atomic_facts: 10-70 atomic statements, ~10-25 words each, resolved "
    "pronouns, preserving verbatim time phrases."
)
_SCALED_FACT_RULE = (
    "For atomic_facts: {fact_soft_min}-{fact_soft_max} atomic statements in total "
    "({per_session_soft_min}-{per_session_soft_max} per physical session), ~10-25 "
    "words each, resolved pronouns, preserving verbatim time phrases. This is a soft "
    "target, not a quota: do not invent, duplicate, split, or pad facts merely to reach "
    "the lower bound. Keep only distinct, verifiable facts grounded in the dialogue.\n\n"
    "Cover every one of the {unit_count} physical {unit_label} sufficiently and balance "
    "coverage across them; do not let a verbose session crowd out precise details from "
    "another session. Keep the single flat atomic_facts list and do not add per-session "
    "keys to the JSON."
)

WRITER_FACT_POLICY: dict[str, Any] = {
    "policy": "linear_soft_fact_capacity_per_physical_session_v1",
    "per_physical_session_soft_min": FACTS_PER_PHYSICAL_SESSION_SOFT_MIN,
    "per_physical_session_soft_max": FACTS_PER_PHYSICAL_SESSION_SOFT_MAX,
    "frontier_soft_min_formula": "10 * actual_unit_count",
    "frontier_soft_max_formula": "70 * actual_unit_count",
    "balanced_coverage_required": True,
    "minimum_enforced": False,
    "padding_or_duplicate_facts_forbidden": True,
    "output_shape": "unchanged_batch_summary_plus_flat_atomic_facts",
    "writer_max_tokens": WRITE_MAX_TOKENS,
    "all_non_writer_stages_unchanged": True,
}
WRITER_FACT_POLICY_SHA256 = gm.canonical_sha256(WRITER_FACT_POLICY)

# Filled after deriving this template from the byte-frozen MGTM authority.  The
# literals make accidental edits fail closed before a provider can be called.
EXPECTED_SCALED_WRITE_TEMPLATE_SHA256 = (
    "965f95376007c3971f65010e402b2fd64daf0da37dc2af678c0396a7ef398040"
)
EXPECTED_WRITER_FACT_POLICY_SHA256 = (
    "f311c77dbbb622e8e8f35ebe7469b2566ef13cd517d96cac339eb0a978511c0b"
)
EXPECTED_SCALED_PROMPT_CONTRACT_SHA256 = (
    "5e12007517516ba6e4db8902ac90bd83232be4420c007436c7bbc7ab79330350"
)

_SOURCE_ASSERT_PROMPT_CONTRACT = legacy.assert_prompt_contract


def _derive_scaled_write_template(source_template: str) -> str:
    if source_template.count(_BASE_FACT_RULE) != 1:
        raise PromptContractError("the frozen Writer fact rule cannot be derived exactly once")
    return source_template.replace(_BASE_FACT_RULE, _SCALED_FACT_RULE)


@dataclass(frozen=True)
class PromptContract(legacy.PromptContract):
    """Derived prompt contract with an explicit frozen source relationship."""

    source_prompt_contract_sha256: str

    def receipt(self) -> dict[str, Any]:
        receipt = super().receipt()
        receipt["schema_version"] = SCHEMA_VERSION
        receipt["source_prompt_contract_sha256"] = self.source_prompt_contract_sha256
        receipt["writer_variant"] = {
            "write_template_sha256": self.prompt_sha256["write_template"],
            "fact_policy": dict(WRITER_FACT_POLICY),
            "fact_policy_sha256": WRITER_FACT_POLICY_SHA256,
            "only_prompt_stage_changed": "writer",
            "qa_template_unchanged": True,
            "judge_template_unchanged": True,
            "qwen_instruction_unchanged": True,
        }
        receipt["write_call_contract"]["max_tokens"] = WRITE_MAX_TOKENS
        receipt["write_call_contract"]["baseline_max_tokens"] = base.WRITE_MAX_TOKENS
        receipt["write_call_contract"]["max_tokens_override_scope"] = "writer_only"
        return receipt


def _contract_payload(
    source: legacy.PromptContract,
    prompt_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_prompt_contract_sha256": source.contract_sha256,
        "file_sha256": dict(source.file_sha256),
        "prompt_sha256": dict(prompt_sha256),
        "writer_fact_policy_sha256": WRITER_FACT_POLICY_SHA256,
        "writer_max_tokens": WRITE_MAX_TOKENS,
        "render_policy": "source_template_with_single_fact_rule_replaced_then_direct_format",
        "qa_call": "byte-identical_to_source_contract",
        "judge_call": "byte-identical_to_source_contract",
        "qwen_instruction": "byte-identical_to_source_contract",
    }


def assert_prompt_contract() -> PromptContract:
    """Validate the MGTM authority and derive the isolated Writer variant."""

    source = _SOURCE_ASSERT_PROMPT_CONTRACT()
    write_template = _derive_scaled_write_template(source.write_template)
    write_sha256 = sha256_text(write_template)
    if write_sha256 != EXPECTED_SCALED_WRITE_TEMPLATE_SHA256:
        raise PromptContractError("scaled Writer template hash drift")
    if WRITER_FACT_POLICY_SHA256 != EXPECTED_WRITER_FACT_POLICY_SHA256:
        raise PromptContractError("scaled Writer fact-policy hash drift")
    prompt_sha256 = dict(source.prompt_sha256)
    prompt_sha256["write_template"] = write_sha256
    contract_sha256 = gm.canonical_sha256(_contract_payload(source, prompt_sha256))
    if contract_sha256 != EXPECTED_SCALED_PROMPT_CONTRACT_SHA256:
        raise PromptContractError("scaled Writer prompt-contract hash drift")
    return PromptContract(
        write_template=write_template,
        qa_template=source.qa_template,
        judge_template=source.judge_template,
        qwen_instruction=source.qwen_instruction,
        file_sha256=dict(source.file_sha256),
        prompt_sha256=prompt_sha256,
        contract_sha256=contract_sha256,
        source_prompt_contract_sha256=source.contract_sha256,
    )


def assert_live_prompt_contract(contract: PromptContract) -> None:
    """Fail closed if either the source authority or derived policy drifts."""

    current = assert_prompt_contract()
    if current != contract:
        raise PromptContractError("scaled Writer contract changed after preflight")


def requested_fact_range(unit_count: int) -> tuple[int, int]:
    if isinstance(unit_count, bool) or not 1 <= int(unit_count) <= FRONTIER_SIZE:
        raise ValueError("unit_count must be between one and three")
    return (
        FACTS_PER_PHYSICAL_SESSION_SOFT_MIN * int(unit_count),
        FACTS_PER_PHYSICAL_SESSION_SOFT_MAX * int(unit_count),
    )


def render_write_prompt(
    contract: PromptContract,
    batch: str,
    *,
    unit_count: int,
) -> str:
    fact_soft_min, fact_soft_max = requested_fact_range(unit_count)
    if contract.prompt_sha256.get("write_template") != EXPECTED_SCALED_WRITE_TEMPLATE_SHA256:
        raise PromptContractError("render received a non-scaled Writer contract")
    return contract.write_template.format(
        unit_count=int(unit_count),
        unit_label="session" if int(unit_count) == 1 else "sessions",
        batch=str(batch),
        fact_soft_min=fact_soft_min,
        fact_soft_max=fact_soft_max,
        per_session_soft_min=FACTS_PER_PHYSICAL_SESSION_SOFT_MIN,
        per_session_soft_max=FACTS_PER_PHYSICAL_SESSION_SOFT_MAX,
    )


ChatFn = Callable[..., str]


def write_one_frontier(
    *,
    contract: PromptContract,
    frontier: Mapping[str, Any],
    chat_fn: ChatFn,
    write_model: str = "gemini-2.5-flash",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one scaled Writer request without enforcing its soft count target."""

    assert_live_prompt_contract(contract)
    members = [dict(spec) for spec in frontier.get("members", [])]
    if not 1 <= len(members) <= FRONTIER_SIZE:
        raise gm.GraphMemoryError("writer frontier must contain one to three sessions")
    batch = render_lme_frontier(frontier)
    fact_soft_min, fact_soft_max = requested_fact_range(len(members))
    prompt = render_write_prompt(contract, batch, unit_count=len(members))
    last_error = "unknown writer failure"
    request_hashes: list[str] = []
    physical_attempts: list[dict[str, Any]] = []
    for attempt in range(WRITE_MAX_ATTEMPTS):
        request_hashes.append(sha256_text(prompt))
        try:
            raw = chat_fn(
                prompt,
                model=write_model,
                temperature=0.0,
                max_tokens=WRITE_MAX_TOKENS,
                thinking_budget=0,
                tag="write",
                use_cache=False,
                json_mode=True,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[-300:]}"
            physical_attempts.append(
                {
                    "ordinal": attempt + 1,
                    "canonical_request_sha256": request_hashes[-1],
                    "status": "transport_error",
                    "error_type": type(exc).__name__,
                }
            )
            if isinstance(exc, NonRetryableProviderError):
                raise
        else:
            physical_attempts.append(
                {
                    "ordinal": attempt + 1,
                    "canonical_request_sha256": request_hashes[-1],
                    "status": "response_received",
                    "response_sha256": sha256_text(str(raw or "")),
                }
            )
            value = legacy._parse_writer_json(raw)
            if value is None:
                last_error = "writer returned no valid JSON object"
            else:
                summary = str(value.get("batch_summary", "") or "").strip()
                raw_facts = value.get("atomic_facts")
                if not isinstance(raw_facts, list):
                    last_error = "writer returned atomic_facts with a non-list type"
                else:
                    facts = [legacy._fact_text(item) for item in raw_facts]
                    facts = [fact for fact in facts if fact]
                    if summary:
                        return {
                            "frontier_id": int(frontier["frontier_id"]),
                            "physical_sessions": [
                                {
                                    "source_position": int(spec["source_position"]),
                                    "source_session_id": str(spec["source_session_id"]),
                                    "date": str(spec["date"]),
                                    "turn_count": len(spec["turns"]),
                                }
                                for spec in members
                            ],
                            "batch_summary": summary,
                            "atomic_facts": facts,
                            "writer_attempts": attempt + 1,
                            "write_prompt_sha256": sha256_text(prompt),
                            "write_prompt_contract_sha256": contract.contract_sha256,
                            "writer_fact_policy_sha256": WRITER_FACT_POLICY_SHA256,
                            "writer_fact_budget": {
                                "physical_session_count": len(members),
                                "per_physical_session_soft_min": FACTS_PER_PHYSICAL_SESSION_SOFT_MIN,
                                "per_physical_session_soft_max": FACTS_PER_PHYSICAL_SESSION_SOFT_MAX,
                                "requested_soft_min": fact_soft_min,
                                "requested_soft_max": fact_soft_max,
                                "observed_atomic_facts": len(facts),
                                "minimum_enforced": False,
                                "below_soft_min": len(facts) < fact_soft_min,
                                "above_soft_max": len(facts) > fact_soft_max,
                            },
                            "request_sha256s": request_hashes,
                            "physical_attempts": physical_attempts,
                            "rendered_batch_sha256": sha256_text(batch),
                        }
                    last_error = "writer returned an empty batch_summary"
        if attempt + 1 < WRITE_MAX_ATTEMPTS:
            sleep_fn(WRITE_RETRY_BACKOFF_S)
    raise gm.GraphMemoryError(
        f"scaled frontier write failed after {WRITE_MAX_ATTEMPTS} calls: {last_error}"
    )


def build_frontier_graph_from_packets(
    packets: Sequence[Mapping[str, Any]],
    embed_fn: base.EmbedFn,
    *,
    contract: PromptContract,
    display_text_fn: base.DisplayTextFn | None = None,
) -> dict[str, Any]:
    """Build the unchanged graph while retaining Writer capacity audit data."""

    graph = base.build_frontier_graph_from_packets(
        packets,
        embed_fn,
        contract=contract,
        display_text_fn=display_text_fn,
    )
    graph["adapter_schema_version"] = SCHEMA_VERSION
    graph["architecture"]["writer_fact_capacity"] = (
        "soft_10_to_70_per_physical_session_linear_in_actual_frontier_size"
    )
    graph["architecture"]["writer_output_shape"] = (
        "unchanged_batch_summary_plus_flat_atomic_facts"
    )
    graph["architecture"]["writer_max_tokens"] = WRITE_MAX_TOKENS
    budget_rows: list[dict[str, Any]] = []
    for normalized, packet in zip(graph["session_packets"], packets, strict=True):
        budget = dict(packet.get("writer_fact_budget", {}))
        if not budget:
            raise gm.GraphMemoryError("scaled Writer packet is missing its fact-budget audit")
        normalized["writer_fact_policy_sha256"] = str(
            packet.get("writer_fact_policy_sha256", "")
        )
        normalized["writer_fact_budget"] = budget
        budget_rows.append(budget)
    graph["writer_fact_policy"] = dict(WRITER_FACT_POLICY)
    graph["writer_fact_policy_sha256"] = WRITER_FACT_POLICY_SHA256
    graph["stats"].update(
        {
            "writer_requested_soft_min_total": sum(
                int(row["requested_soft_min"]) for row in budget_rows
            ),
            "writer_requested_soft_max_total": sum(
                int(row["requested_soft_max"]) for row in budget_rows
            ),
            "writer_observed_atomic_facts_total": sum(
                int(row["observed_atomic_facts"]) for row in budget_rows
            ),
            "writer_frontiers_below_soft_min": sum(
                bool(row["below_soft_min"]) for row in budget_rows
            ),
            "writer_frontiers_above_soft_max": sum(
                bool(row["above_soft_max"]) for row in budget_rows
            ),
        }
    )
    graph["content_fingerprint"] = gm.canonical_sha256(
        {
            "base_content_fingerprint": graph["content_fingerprint"],
            "writer_fact_policy_sha256": WRITER_FACT_POLICY_SHA256,
            "writer_fact_budgets": budget_rows,
        }
    )
    return graph


__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EMBEDDING_NORMALIZATION",
    "EXPECTED_SCALED_PROMPT_CONTRACT_SHA256",
    "EXPECTED_SCALED_WRITE_TEMPLATE_SHA256",
    "FACTS_PER_PHYSICAL_SESSION_SOFT_MAX",
    "FACTS_PER_PHYSICAL_SESSION_SOFT_MIN",
    "FRONTIER_SIZE",
    "LME_DATASET",
    "NonRetryableProviderError",
    "PromptContract",
    "QwenGraphMemoryIndex",
    "SCHEMA_VERSION",
    "WRITER_FACT_POLICY",
    "WRITER_FACT_POLICY_SHA256",
    "WRITE_MAX_TOKENS",
    "assert_live_prompt_contract",
    "assert_prompt_contract",
    "build_frontier_graph_from_packets",
    "group_session_specs",
    "materialize_operation_context",
    "render_lme_frontier",
    "render_lme_session",
    "render_write_prompt",
    "requested_fact_range",
    "sha256_text",
    "write_one_frontier",
]

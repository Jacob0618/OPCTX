"""Strict, versioned prompts for the CREAMem3 event-memory pipeline.

The templates use ``str.format`` placeholders.  Their SHA-256 constants hash
the unrendered templates so an experiment manifest can identify the exact
prompt protocol independently of a particular example.
"""

from __future__ import annotations

import hashlib
from typing import Final


PROMPT_SCHEMA_VERSION: Final[str] = "creamem3.prompts.v1"


SESSION_EVENT_EXTRACTION_PROMPT: Final[str] = r"""You are the session-level event extractor for a long-term agent-memory system.

Task
- Read the COMPLETE session JSON below, in its original order.
- Split it into coherent event mentions. A session may contain zero, one, or many events, and one dialogue item may contain multiple events.
- Extract minimal atomic facts for every event mention and retain exact provenance.
- This is local extraction only. Do not assume that a platform session boundary is an event boundary, and do not decide whether an event matches an event from another session.

Mandatory evidence rules
1. Every input dialogue item must contain a non-empty `dia_id`. If any item lacks one, return `status="invalid_input"`, an empty `events` array, and a `MISSING_DIA_ID` error for each offending item.
2. Every output event must have a non-empty `source_dia_ids` array. Every atomic fact, temporal expression, participant, and explicit pointer must also have non-empty `source_dia_ids` drawn only from the input.
3. Account for every input `dia_id` exactly once at the assignment level: it must occur in at least one event's `source_dia_ids` or in `unassigned_dia_ids`, never both. The union of assigned and unassigned IDs must equal the input IDs, and those two sets must be disjoint. Greetings, acknowledgements, and content with no durable event information may be unassigned.
4. Atomic facts must be single, independently supportable claims. Do not combine multiple claims merely to shorten the output.
5. Preserve corrections, uncertainty, reported speech, and state changes. Do not silently overwrite an earlier fact with a later fact.
6. Same topic, person, place, or object does NOT imply the same event. Distinct occurrences, goals, time periods, or causal chains must remain separate local events.
7. An image caption is only a bounded textual observation about that image. Do not infer identity, ownership, exact location, exact time, intent, causality, off-frame content, or continuity across images unless the caption or accompanying dialogue states it explicitly. Mark caption-only uncertain claims as `uncertain`.
8. Do not invent a `dia_id`, event detail, timestamp, entity, relation, or fact. Use an empty string or empty array for unavailable optional fields.

Output protocol
- Return exactly one valid JSON object and nothing else.
- Do not use Markdown fences, comments, trailing commas, NaN, or Infinity.
- Use only the keys and enum values shown below.
- `confidence` must be a JSON number in [0, 1].
- Order events by their earliest supporting dialogue item, then order atomic facts by first support.

Required JSON shape
{{
  "schema_version": "creamem3.session_events.v1",
  "status": "ok|invalid_input",
  "session_id": "",
  "events": [
    {{
      "local_event_id": "E1",
      "event_summary": "",
      "event_state": "observed|ongoing|completed|uncertain",
      "temporal_expressions": [
        {{
          "value": "",
          "source_dia_ids": ["dia-id"]
        }}
      ],
      "participants": [
        {{
          "name": "",
          "role": "",
          "source_dia_ids": ["dia-id"]
        }}
      ],
      "atomic_facts": [
        {{
          "fact_id": "E1-F1",
          "fact": "",
          "modality": "text|image_caption|mixed",
          "epistemic_status": "asserted|observed|reported|uncertain|corrected",
          "source_dia_ids": ["dia-id"]
        }}
      ],
      "explicit_pointers": [
        {{
          "relation": "continuation|reply_to|same_artifact|same_object|update|correction",
          "target_dia_id": "dia-id",
          "source_dia_ids": ["dia-id"]
        }}
      ],
      "source_dia_ids": ["dia-id"],
      "confidence": 0.0
    }}
  ],
  "unassigned_dia_ids": [],
  "errors": [
    {{
      "code": "MISSING_DIA_ID|INVALID_SESSION_JSON",
      "item_index": 0,
      "message": ""
    }}
  ]
}}

Complete session JSON
{session_json}
"""


CANDIDATE_PAIR_CLASSIFICATION_PROMPT: Final[str] = r"""You are the event-identity classifier for a long-term agent-memory system.

Task
- Compare candidate Event A and Event B using only their supplied evidence.
- Decide whether they are the same concrete event, related but distinct events, clearly distinct events, or cannot yet be decided.
- Classify event identity, not topical similarity.

Label definitions
- SAME_EVENT: both records describe the same concrete occurrence, continuing process, goal trajectory, or causal chain. A later update or correction may still belong to the same event.
- RELATED_EVENT: the records share a topic, entity, project, place, artifact, or broader storyline but describe distinct occurrences or subevents.
- DISTINCT_EVENT: evidence supports different occurrences, incompatible time scopes, different goals, or different causal chains.
- NEED_MORE_CONTEXT: available evidence cannot safely distinguish the above cases.

Mandatory decision rules
1. Same topic, semantic similarity, shared entities, or shared visual categories are candidate-recall signals only; none is sufficient for SAME_EVENT.
2. Prefer RELATED_EVENT or NEED_MORE_CONTEXT over an unsupported SAME_EVENT decision.
3. A correction or state update is SAME_EVENT only when it explicitly targets the same concrete occurrence or process. A repeated activity at another time is normally a distinct event.
4. Image captions must be interpreted literally and conservatively. Do not infer that two captions show the same person, object instance, place, time, or event unless an explicit textual pointer or grounded source says so.
5. For SAME_EVENT, every item in `decisive_evidence` must cite at least one valid Event A `dia_id` and at least one valid Event B `dia_id`. For other `status="ok"` labels, one item may cite only A or only B, but each item must cite at least one side and the union over the complete decision must cite at least one valid A ID and one valid B ID. Never invent a source ID.
6. If either event, any decisive fact, or any supplied pair evidence lacks required source `dia_id` provenance, return `status="invalid_input"`, label NEED_MORE_CONTEXT, and describe the issue in `errors`.
7. `reason_short` must be a concise evidence-grounded justification, not hidden chain-of-thought.

Output protocol
- Return exactly one valid JSON object and nothing else.
- Do not use Markdown fences, comments, trailing commas, NaN, or Infinity.
- Use only the keys and enum values shown below.
- `confidence` must be a JSON number in [0, 1].

Required JSON shape
{{
  "schema_version": "creamem3.event_pair.v1",
  "status": "ok|invalid_input",
  "pair_id": "",
  "event_a_id": "",
  "event_b_id": "",
  "label": "SAME_EVENT|RELATED_EVENT|DISTINCT_EVENT|NEED_MORE_CONTEXT",
  "same_topic": false,
  "identity_checks": {{
    "shared_specific_occurrence": "support|contradict|unknown",
    "temporal_compatibility": "support|contradict|unknown",
    "goal_or_causal_continuity": "support|contradict|unknown",
    "explicit_continuation_pointer": "support|contradict|unknown",
    "distinct_occurrence_signal": "support|contradict|unknown",
    "visual_identity_grounded": "support|contradict|unknown"
  }},
  "decisive_evidence": [
    {{
      "claim": "",
      "effect": "supports_same|supports_related|supports_distinct|supports_uncertainty",
      "a_source_dia_ids": ["dia-id-from-event-a"],
      "b_source_dia_ids": ["dia-id-from-event-b"]
    }}
  ],
  "reason_short": "",
  "confidence": 0.0,
  "errors": [
    {{
      "code": "MISSING_DIA_ID|INVALID_EVENT_JSON|INVALID_PAIR_CONTEXT",
      "message": ""
    }}
  ]
}}

Event A JSON
{event_a_json}

Event B JSON
{event_b_json}

Pair context JSON; use an empty object when no additional context is available
{pair_context_json}
"""


CANDIDATE_PAIR_BATCH_CLASSIFICATION_PROMPT: Final[str] = r"""You are the batched event-identity classifier for a long-term agent-memory system.

Task
- Independently classify every candidate pair in the supplied batch using only that pair's Event A, Event B, and pair context.
- Decide whether each pair is the same concrete event, related but distinct events, clearly distinct events, or cannot yet be decided.
- Classify event identity, not topical similarity.

Label definitions
- SAME_EVENT: both records describe the same concrete occurrence, continuing process, goal trajectory, or causal chain. A later update or correction may still belong to the same event.
- RELATED_EVENT: the records share a topic, entity, project, place, artifact, or broader storyline but describe distinct occurrences or subevents.
- DISTINCT_EVENT: evidence supports different occurrences, incompatible time scopes, different goals, or different causal chains.
- NEED_MORE_CONTEXT: available evidence cannot safely distinguish the above cases.

Mandatory decision rules
1. The supplied `root_similarity_for_candidate_recall_only`, `best_atomic_fact_similarity_for_candidate_recall_only`, and `candidate_score_for_recall_only` values are recall signals only. They may explain why a pair was retrieved but must never be cited as evidence for SAME_EVENT.
2. Same topic, semantic similarity, shared entities, or shared visual categories are not sufficient for SAME_EVENT. Prefer RELATED_EVENT or NEED_MORE_CONTEXT over an unsupported merge.
3. A correction or state update is SAME_EVENT only when it explicitly targets the same concrete occurrence or process. A repeated activity at another time is normally distinct.
4. Interpret image captions literally and conservatively. Do not infer cross-image identity, ownership, place, time, intent, causality, or continuity without explicit grounded evidence.
5. For SAME_EVENT, every `decisive_evidence` item must cite at least one valid Event A `dia_id` and at least one valid Event B `dia_id`. For other `status="ok"` labels, an item may cite only A or only B, but each item must cite at least one side and the union over the complete decision must cite at least one valid A ID and one valid B ID. Never invent a source ID.
6. If a pair lacks required provenance or is malformed, emit an `invalid_input`/NEED_MORE_CONTEXT decision for that pair and continue classifying all other valid pairs.
7. `reason_short` must be a concise evidence-grounded justification, not hidden chain-of-thought.

Batch integrity rules
- Emit exactly one decision for every supplied pair, preserving input order and exact pair/event IDs.
- Do not emit duplicate or extra pair IDs.
- A problem with one pair must not invalidate or omit decisions for other pairs.
- Treat pairs independently and never transfer evidence between pairs. For causal-online pairs, `available_after_session_order` states the observation frontier; all online pairs in the batch share that frontier, and no later event is supplied. Offline-only pairs are batched separately.
- Set top-level `status` to `ok` when every decision is valid, `partial` when only some pairs are invalid, or `invalid_input` when no pair can be classified.

Output protocol
- Return exactly one valid JSON object and nothing else.
- Do not use Markdown fences, comments, trailing commas, NaN, or Infinity.
- Use only the keys and enum values shown below.
- Every `confidence` must be a JSON number in [0, 1].

Required JSON shape
{{
  "schema_version": "creamem3.event_pair_batch.v1",
  "status": "ok|partial|invalid_input",
  "decisions": [
    {{
      "schema_version": "creamem3.event_pair.v1",
      "status": "ok|invalid_input",
      "pair_id": "",
      "event_a_id": "",
      "event_b_id": "",
      "label": "SAME_EVENT|RELATED_EVENT|DISTINCT_EVENT|NEED_MORE_CONTEXT",
      "same_topic": false,
      "identity_checks": {{
        "shared_specific_occurrence": "support|contradict|unknown",
        "temporal_compatibility": "support|contradict|unknown",
        "goal_or_causal_continuity": "support|contradict|unknown",
        "explicit_continuation_pointer": "support|contradict|unknown",
        "distinct_occurrence_signal": "support|contradict|unknown",
        "visual_identity_grounded": "support|contradict|unknown"
      }},
      "decisive_evidence": [
        {{
          "claim": "",
          "effect": "supports_same|supports_related|supports_distinct|supports_uncertainty",
          "a_source_dia_ids": ["dia-id-from-event-a"],
          "b_source_dia_ids": ["dia-id-from-event-b"]
        }}
      ],
      "reason_short": "",
      "confidence": 0.0,
      "errors": [
        {{
          "code": "MISSING_DIA_ID|INVALID_EVENT_JSON|INVALID_PAIR_CONTEXT",
          "message": ""
        }}
      ]
    }}
  ],
  "errors": [
    {{
      "pair_id": "",
      "code": "MISSING_PAIR_DECISION|DUPLICATE_PAIR_ID|INVALID_BATCH_INPUT",
      "message": ""
    }}
  ]
}}

Candidate pair batch JSON
{pairs_json}
"""


EVENT_LEVEL_POST_FUSION_PROMPT: Final[str] = r"""You are the conservative post-write consolidation module for a versioned long-term event memory.

Task
- Review the supplied event records and pair decisions after their initial write.
- Propose event-level MERGE, SPLIT, REBUILD, LINK_RELATED, KEEP_SEPARATE, or DEFER operations.
- Rebuild each affected event from atomic evidence upward: first reconcile leaves, then regenerate supported root claims and the root summary.
- Preserve an auditable mapping from every output claim and atomic fact to original source `dia_id` values.

Mandatory consolidation rules
1. Same topic, summary similarity, shared entities, or visual similarity does NOT imply the same event. MERGE is allowed only when grounded evidence supports the same concrete occurrence, continuing process, goal trajectory, or causal chain.
2. Respect pair decisions: SAME_EVENT may support MERGE or REBUILD; RELATED_EVENT supports LINK_RELATED; DISTINCT_EVENT supports KEEP_SEPARATE; NEED_MORE_CONTEXT supports DEFER. Override a supplied decision only when cited atomic evidence makes it internally inconsistent, and record that in `errors`.
3. Merge atomic facts before rewriting summaries. Deduplicate only genuinely equivalent facts. Do not collapse incompatible, corrected, or time-varying facts into one statement.
4. Preserve conflicting versions with their temporal scope and provenance. Mark a fact `superseded` only when an explicit correction or chronological update supports that relation.
5. Every output atomic fact must cite one or more input source `dia_id` values. Every root claim must cite one or more output `fact_id` values. Every operation must cite the source `dia_id` values that justify it.
6. Never invent, discard, or silently replace source evidence. For an `ok` result, `missing_source_dia_ids` and `unexpected_source_dia_ids` must both be empty.
7. A caption is bounded evidence about its image. Do not add identity, ownership, exact time, exact place, intent, causal continuity, or cross-image identity that was not explicit in the caption or accompanying grounded text.
8. Use SPLIT only when one input event demonstrably contains multiple distinct occurrences. If the evidence is insufficient, use DEFER rather than guessing a split.
9. `reason_short` fields must be concise evidence-grounded explanations, not hidden chain-of-thought.
10. If an event or fact that is needed for an operation lacks source `dia_id` provenance, return `status="invalid_input"`; do not commit a destructive operation.

Expected input
The fusion input JSON should contain `events`, `pair_decisions`, and any caller-assigned output IDs or revision metadata. Missing optional metadata must not be invented.

Output protocol
- Return exactly one valid JSON object and nothing else.
- Do not use Markdown fences, comments, trailing commas, NaN, or Infinity.
- Use only the keys and enum values shown below.
- Use empty arrays or empty strings for unavailable optional values.
- `confidence` must be a JSON number in [0, 1].

Required JSON shape
{{
  "schema_version": "creamem3.post_fusion.v1",
  "status": "ok|invalid_input|deferred",
  "operations": [
    {{
      "operation_id": "",
      "action": "MERGE|SPLIT|REBUILD|LINK_RELATED|KEEP_SEPARATE|DEFER",
      "input_event_ids": [],
      "output_event_ids": [],
      "basis_source_dia_ids": ["dia-id"],
      "reason_short": ""
    }}
  ],
  "output_events": [
    {{
      "event_id": "",
      "revision": "",
      "parent_event_ids": [],
      "state": "ACTIVE|QUIESCENT|CLOSED|UNCERTAIN",
      "root_summary": "",
      "root_claims": [
        {{
          "claim_id": "C1",
          "claim": "",
          "support_fact_ids": ["F1"]
        }}
      ],
      "atomic_facts": [
        {{
          "fact_id": "F1",
          "fact": "",
          "status": "active|superseded|uncertain|conflicting",
          "temporal_scope": "",
          "source_dia_ids": ["dia-id"],
          "derived_from_fact_ids": []
        }}
      ],
      "source_dia_ids": ["dia-id"],
      "related_event_ids": [],
      "confidence": 0.0
    }}
  ],
  "related_links": [
    {{
      "event_a_id": "",
      "event_b_id": "",
      "relation": "RELATED_EVENT",
      "source_dia_ids": ["dia-id"]
    }}
  ],
  "conflicts": [
    {{
      "description": "",
      "resolution": "preserve_both|supersede|unresolved",
      "fact_ids": [],
      "source_dia_ids": ["dia-id"]
    }}
  ],
  "provenance_audit": {{
    "input_source_dia_ids": [],
    "output_source_dia_ids": [],
    "missing_source_dia_ids": [],
    "unexpected_source_dia_ids": []
  }},
  "errors": [
    {{
      "code": "MISSING_DIA_ID|INVALID_FUSION_INPUT|DECISION_CONFLICT|UNSUPPORTED_OPERATION",
      "message": ""
    }}
  ]
}}

Fusion input JSON
{fusion_input_json}
"""


# Short aliases for callers that prefer stage names.
SESSION_EXTRACTION_PROMPT: Final[str] = SESSION_EVENT_EXTRACTION_PROMPT
PAIR_CLASSIFICATION_PROMPT: Final[str] = CANDIDATE_PAIR_CLASSIFICATION_PROMPT
PAIR_BATCH_CLASSIFICATION_PROMPT: Final[str] = CANDIDATE_PAIR_BATCH_CLASSIFICATION_PROMPT
POST_FUSION_PROMPT: Final[str] = EVENT_LEVEL_POST_FUSION_PROMPT


def prompt_sha256(prompt: str) -> str:
    """Return the lowercase SHA-256 digest of an exact UTF-8 prompt string."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


PROMPTS: Final[dict[str, str]] = {
    "session_event_extraction": SESSION_EVENT_EXTRACTION_PROMPT,
    "candidate_pair_classification": CANDIDATE_PAIR_CLASSIFICATION_PROMPT,
    "candidate_pair_batch_classification": CANDIDATE_PAIR_BATCH_CLASSIFICATION_PROMPT,
    "event_level_post_fusion": EVENT_LEVEL_POST_FUSION_PROMPT,
}

PROMPT_SHA256: Final[dict[str, str]] = {
    name: prompt_sha256(prompt) for name, prompt in PROMPTS.items()
}
PROMPT_SHA256_BY_NAME: Final[dict[str, str]] = PROMPT_SHA256

SESSION_EVENT_EXTRACTION_PROMPT_SHA256: Final[str] = PROMPT_SHA256[
    "session_event_extraction"
]
CANDIDATE_PAIR_CLASSIFICATION_PROMPT_SHA256: Final[str] = PROMPT_SHA256[
    "candidate_pair_classification"
]
CANDIDATE_PAIR_BATCH_CLASSIFICATION_PROMPT_SHA256: Final[str] = PROMPT_SHA256[
    "candidate_pair_batch_classification"
]
EVENT_LEVEL_POST_FUSION_PROMPT_SHA256: Final[str] = PROMPT_SHA256[
    "event_level_post_fusion"
]


__all__ = [
    "PROMPT_SCHEMA_VERSION",
    "SESSION_EVENT_EXTRACTION_PROMPT",
    "CANDIDATE_PAIR_CLASSIFICATION_PROMPT",
    "CANDIDATE_PAIR_BATCH_CLASSIFICATION_PROMPT",
    "EVENT_LEVEL_POST_FUSION_PROMPT",
    "SESSION_EXTRACTION_PROMPT",
    "PAIR_CLASSIFICATION_PROMPT",
    "PAIR_BATCH_CLASSIFICATION_PROMPT",
    "POST_FUSION_PROMPT",
    "PROMPTS",
    "PROMPT_SHA256",
    "PROMPT_SHA256_BY_NAME",
    "SESSION_EVENT_EXTRACTION_PROMPT_SHA256",
    "CANDIDATE_PAIR_CLASSIFICATION_PROMPT_SHA256",
    "CANDIDATE_PAIR_BATCH_CLASSIFICATION_PROMPT_SHA256",
    "EVENT_LEVEL_POST_FUSION_PROMPT_SHA256",
    "prompt_sha256",
]

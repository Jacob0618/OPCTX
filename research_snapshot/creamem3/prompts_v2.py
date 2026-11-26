"""Prompts for the independent CreaMem3 v2 memory graph.

The session writer deliberately keeps MGTM's useful extraction rules while
adding source provenance.  The storage representation is defined elsewhere
and is not a two-level tree: summaries are routing views over immutable facts.
"""

from __future__ import annotations

import hashlib
from typing import Final


PROMPT_SCHEMA_VERSION: Final[str] = "creamem3.prompts.v2"


SESSION_PACKET_PROMPT: Final[str] = r"""You analyze one COMPLETE dialogue session containing {turn_count} turns and emit a compact long-term memory packet.

Output STRICT JSON with exactly this shape:
{{
  "batch_summary": "<3-4 sentence summary of the main events and topics>",
  "atomic_facts": [
    {{"fact": "<one atomic fact>", "source_dia_ids": ["D1:1"]}}
  ]
}}

CRITICAL - GROUNDING AND TIME
1. Use only the supplied dialogue text. No image, image URL, or image caption is supplied; do not infer visual information.
2. Every fact must be directly supported by the cited source_dia_ids. Cite only IDs present in this session and never invent an ID.
3. Preserve verbatim relative time expressions exactly as spoken in each atomic fact. Never replace them with an inferred absolute date: "last Friday" stays "last Friday", "two weeks ago" stays "two weeks ago", and "yesterday", "today", or "next Tuesday" stay as-is.
4. When a supporting dialogue date is available, also include its ISO anchor, for example: "Melanie attended the workshop last Friday (said on 2023-07-15)." The anchor supplements rather than replaces the original expression.
5. Preserve uncertainty, reported speech, plans, corrections, quantities, and state changes. Do not silently turn them into stronger claims.

For batch_summary: write 3-4 information-dense sentences capturing the narrative arc, distinctive entities, main events, and session date. Keep different occurrences distinguishable. The summary is a retrieval aid, so include specific names, activities, places, and goals when supported.

For atomic_facts: aim for {fact_guidance}, normally about 10-25 words each. Resolve pronouns to explicit entities when the source makes the resolution clear. Each item should express one independently supportable claim. Extract all useful details, but extract only supported facts; never invent or pad facts merely to meet the suggested range. Repeated facts should be emitted once with all supporting source IDs.

Session identifier: {session_id}
Original session timestamp: {timestamp}
ISO session date: {iso_date}

Complete text-only dialogue session:
{batch}
"""


OFFLINE_EVENT_VIEW_PROMPT: Final[str] = r"""You create non-destructive cross-session routing views for an independent long-term memory graph.

The input contains immutable session summaries and atomic facts. Facts are the source of truth. Do not rewrite, delete, combine, or invent facts. Create derived views only when evidence from at least two different sessions belongs to the same concrete event, a continuing process or goal, or an explicit update chain. Non-adjacent sessions may belong to the same view.

Do not group facts merely because they mention the same person, broad topic, hobby, place, or repeated activity. Distinct occurrences must remain distinguishable. A fact may support more than one view only when the scopes genuinely overlap. Omitting a weak or uncertain group is better than forcing a group.

Output STRICT JSON with exactly this shape:
{{
  "event_views": [
    {{
      "view_id": "V1",
      "view_type": "same_event|continuing_process|update_chain",
      "summary": "<2-4 evidence-grounded sentences with distinguishing dates or stages>",
      "member_fact_ids": ["F0001", "F0002"]
    }}
  ]
}}

Rules:
1. Every member_fact_id must be copied exactly from the input.
2. Every view must contain facts from at least two different sessions.
3. The summary may state only claims supported by its member facts. Preserve corrections and temporal scope.
4. Prefer compact, precise views that improve retrieval over broad profiles.
5. Return one JSON object only, without Markdown or commentary.

Memory packets:
{memory_packets_json}
"""


ONLINE_EVENT_UPDATE_PROMPT: Final[str] = r"""You update causal cross-session routing views after receiving dialogue session {frontier_session_id}.

The input is strictly limited to the current session, selected facts from earlier sessions, and selected active views created from that prefix. Atomic facts are immutable and are the source of truth. Return no operation when the evidence does not describe the same concrete event, a continuing process or goal, or an explicit update chain. Do not group items merely because they share a person, broad topic, hobby, place, or repeated activity.

Output STRICT JSON with exactly this shape:
{{
  "operations": [
    {{
      "operation": "create|extend",
      "target_view_id": "",
      "view_type": "same_event|continuing_process|update_chain",
      "summary_claims": [
        {{"text": "<one grounded claim>", "support_fact_ids": ["F0001"]}}
      ],
      "member_fact_ids": ["F0001", "F0002"]
    }}
  ]
}}

Rules:
1. Copy every fact ID and target view ID exactly from the input. Do not invent IDs or facts.
2. Every create operation must include at least one current-session fact and at least one earlier-session fact, spanning at least two sessions. Its target_view_id must be an empty string.
3. Every extend operation must target one supplied active view, preserve all of its existing member facts, and add at least one current-session fact. Keep the supplied view_type.
4. Each summary claim must cite supporting member facts. Every member fact must support at least one claim. Preserve dates, uncertainty, corrections, and distinct occurrences.
5. Membership may only grow. Never rewrite, delete, merge, or duplicate atomic facts. Return at most 12 operations.
6. Return one JSON object only, without Markdown or commentary. An empty operations list is valid and preferred to a weak grouping.

Causal prefix input JSON:
{online_input_json}
"""


FACT_RERANK_PROMPT: Final[str] = r"""Rank atomic long-term memory facts for the question.

Select exactly {top_k} distinct fact IDs when at least {top_k} candidates are supplied; otherwise select every candidate. Rank facts by how directly and completely they help answer the question. Favor complementary evidence when the answer needs multiple details, dates, comparisons, or counts. Keep facts that establish the relevant time anchor. Do not answer the question and do not invent IDs.

Output STRICT JSON only:
{{"ranked_fact_ids": ["F0001", "F0002"]}}

Question:
{question}

Candidate facts JSON:
{candidates_json}
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROMPT_SHA256: Final[dict[str, str]] = {
    "session_packet": _sha256(SESSION_PACKET_PROMPT),
    "offline_event_view": _sha256(OFFLINE_EVENT_VIEW_PROMPT),
    "online_event_update": _sha256(ONLINE_EVENT_UPDATE_PROMPT),
    "fact_rerank": _sha256(FACT_RERANK_PROMPT),
}


__all__ = [
    "FACT_RERANK_PROMPT",
    "OFFLINE_EVENT_VIEW_PROMPT",
    "ONLINE_EVENT_UPDATE_PROMPT",
    "PROMPT_SCHEMA_VERSION",
    "PROMPT_SHA256",
    "SESSION_PACKET_PROMPT",
]

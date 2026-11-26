"""Pre-registered query-independent Online interventions; Writer unchanged."""
from creamem3.prompts_v2 import ONLINE_EVENT_UPDATE_PROMPT

RELATIONAL_ADDENDUM = r"""
ADDITIONAL EVENT-MAINTENANCE INSTRUCTIONS
The purpose of a cross-session view is to make relationships or state evolution explicit, not to restate a collection of similar facts. Apply the following principles uniformly to every conversation:
- Identify the SAME concrete event or ongoing goal using its actor, action/object and distinguishing occurrence. Sharing a hobby, entity or keyword is not enough. Keep witnessing someone perform separate from personally performing; ownership, advice, intention and completed action are different relations.
- For a genuine continuation, summarize what is maintained, what has changed and how the evidence connects. Preserve a plan as a plan until a cited fact reports completion. Explicitly preserve the earlier and later scope when values change; do not treat every later statement as a correction.
- Write compact JOINT summary claims when several member facts establish one relation or progression, citing all and only the necessary support. Do not produce one paraphrase for every fact if one precise grounded claim can cover their shared meaning.
- When facts establish an explicit cause/result, state that relation. Mere chronological order or thematic similarity is NOT evidence of causation. Never infer an unstated cause.
- Preserve exact named objects, places, quantities, dates and actor roles needed to distinguish occurrences. Do not replace these with generic descriptions. Relative dates retain their source anchors.
- Keep the active view coherent. Prefer no operation to a weak broad profile. For an extension, retain the existing support membership and all grounded distinctions, but rewrite the derived claims to eliminate redundant retellings. Base atomic facts remain immutable.
These instructions do not change the output JSON schema or the original allowlist, grounding, cross-session, or causal-prefix rules.
"""

VARIANTS = {"relational_delta": ONLINE_EVENT_UPDATE_PROMPT + RELATIONAL_ADDENDUM}

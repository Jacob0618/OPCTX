"""Preserve already-selected relation endpoints through Focused compression.

No new evidence/retrieval call. Additional delivered text/cost is explicit.
This interface is not installed into any historical QA runner.
"""
from .state import render_package, required_support, digest


def final_context(state, facts, selected_raw_ids, selected_claim_ids,
                  aggregate_context, *, max_items, char_budget):
    original, original_receipt = render_package(state, facts, selected_raw_ids,
                      selected_claim_ids, max_items=max_items, char_budget=char_budget)
    raw = list(dict.fromkeys(fid for cid in selected_claim_ids for fid in required_support(state['claims'][cid])))
    protected, _ = render_package(state, facts, raw, selected_claim_ids,
                                 max_items=max_items, char_budget=char_budget)
    header = '\n\nSELECTED RELATIONS AND THEIR SOURCE EVIDENCE (not additional retrieval):\n'
    combined = aggregate_context + (header + protected if protected else '')
    fallback = len(combined) > char_budget
    result = original if fallback else combined
    return result, {'selected_evidence': original_receipt, 'protected_raw_fact_ids': raw,
                    'protected_derived_fact_ids': list(selected_claim_ids),
                    'raw_budget_fallback': fallback, 'chars': len(result),
                    'context_sha256': digest(result), 'new_model_calls': 0,
                    'not_assumed_equal_input_tokens_to_legacy': True}

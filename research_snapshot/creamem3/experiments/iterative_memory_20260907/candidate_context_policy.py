"""Matched candidate-anchor interventions; no extra evidence or model calls."""
from __future__ import annotations
from collections import Counter

POLICIES={'candidate_last','candidate_absent','candidate_absent_plus_raw'}
HEADER='AGGREGATED EVIDENCE:\n'
RAW_SEPARATOR='\n\n--- RAW SELECTED MEMORY (SAME SELECTED EVIDENCE) ---\n'


def rendered_context(base_render,aggregation,raw_context,mode,policy):
    if mode!='candidate_first' or policy not in POLICIES:
        raise ValueError('declared candidate-first authority and new policy required')
    original=base_render(aggregation,raw_context,mode)
    candidate=str(aggregation.get('candidate_answer') or '').strip()
    if aggregation.get('_failed') or original==raw_context:
        return original
    if not candidate:
        return original+RAW_SEPARATOR+raw_context if policy=='candidate_absent_plus_raw' else original
    block='CANDIDATE ANSWER: '+candidate
    prefix=HEADER+block
    if original==prefix:
        # Candidate-last has no other blocks to reorder. Removing the only
        # useful block invokes the unchanged raw-evidence fallback, not empty QA.
        return original if policy=='candidate_last' else raw_context
    if not original.startswith(prefix+'\n'):
        raise RuntimeError('pinned source renderer candidate prefix changed')
    remainder=original[len(prefix)+1:]
    if not remainder.strip():raise RuntimeError('ambiguous empty rendered evidence')
    if policy=='candidate_last':
        result=HEADER+remainder+'\n'+block
        if len(result)!=len(original) or Counter(result)!=Counter(original):
            raise RuntimeError('candidate move changed the visible character multiset')
        return result
    result=HEADER+remainder
    return result+RAW_SEPARATOR+raw_context if policy=='candidate_absent_plus_raw' else result

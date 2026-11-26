"""Rank utilities and validated character-budget inputs for the selector."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Mapping, Sequence
import re
import numpy as np
from scipy.stats import rankdata

STOPWORDS = frozenset("""a an and are as at be been being but by can could did do
does doing for from had has have having he her hers him his how i if in into is
it its may me might mine more most much my of on or our ours she should so than
that the their theirs them then there these they this those through to too under
up us was we were what when where which who why will with would you your yours
s t""".split())
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

def tokens(text):
    return {s.casefold() for s in TOKEN.findall(text)}

def percentile_utility(values: Sequence[float], *, positive_only: bool) -> np.ndarray:
    """Frozen midrank convention, zero-based; uninformative dense channel off."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be one-dimensional finite values")
    if not len(values):
        return values.copy()
    if positive_only:
        active = values > 0
        if not active.any():
            return np.zeros(len(values))
    elif np.ptp(values) == 0:
        return np.zeros(len(values))
    result = (rankdata(values, method="average") - 1) / max(len(values)-1, 1)
    if positive_only:
        result[values <= 0] = 0
        if len(values) == 1 and values[0] > 0:
            result[0] = 1
    return result

def validate_inputs(units: Sequence[Mapping[str, Any]], dense_scores: Sequence[float], bm25_scores: Sequence[float],
                    *, max_units: int, char_budget: int, group_costs: Mapping[str, int], min_units: int = 0) -> dict[str, Any]:
    for name, value in (("max_units", max_units), ("min_units", min_units), ("char_budget", char_budget)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not 0 <= min_units <= max_units <= 50:
        raise ValueError("require0 <= min_units <= max_units <=50")
    if len(units) != len(dense_scores) or len(units) != len(bm25_scores):
        raise ValueError("unit/score length mismatch")
    if len({str(u["id"]) for u in units}) != len(units):
        raise ValueError("duplicate unit ID")
    order = sorted(range(len(units)), key=lambda i: str(units[i]["id"]))
    clean = []
    for i in order:
        unit = units[i]
        if not isinstance(unit["text"], str) or not unit["text"].strip() or not str(unit["id"]):
            raise ValueError("unit requires nonempty ID/text")
        cost = unit["cost"]
        if isinstance(cost, (bool, np.bool_)) or not isinstance(cost, (int, np.integer)) or cost < 0:
            raise ValueError("unit costs must be nonnegative integer characters")
        span = unit.get("span")
        if span is not None:
            if not isinstance(span, Mapping) or set(span) != {"fact_id", "start", "end"} or not str(span["fact_id"]):
                raise ValueError("span must be {fact_id,start,end}")
            start, end = span["start"], span["end"]
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (start, end)) or not 0 <= start < end:
                raise ValueError("span must be a nonempty nonnegative half-open integer interval")
            if end-start != len(unit["text"]):
                raise ValueError("extractive span length must equal exact unit text length")
            span = {"fact_id": str(span["fact_id"]), "start": int(start), "end": int(end)}
        sources = unit.get("sources")
        if sources is None:
            sources = []
        if not isinstance(sources, (list, tuple)) or any(not isinstance(s, str) or not s for s in sources):
            raise ValueError("sources must be a sequence of nonempty strings")
        # Keys/spans bind to caller-provided raw provenance, not inferred entity
        # identity. Source text equality must also be checked by its adapter.
        clean.append({"id": str(unit["id"]), "text": unit["text"], "cost": int(cost),
                      "group": str(unit["group"]), "span": span, "sources": sorted(set(sources))})
    groups = sorted({u["group"] for u in clean})
    gc = {}
    for group in groups:
        if group not in group_costs:
            raise ValueError(f"missing shared cost for group: {group}")
        cost = group_costs[group]
        if isinstance(cost, (bool, np.bool_)) or not isinstance(cost, (int, np.integer)) or cost < 0:
            raise ValueError("group costs must be nonnegative integer characters")
        gc[group] = int(cost)
    dense = np.asarray(dense_scores, dtype=float)[order]
    bm25 = np.asarray(bm25_scores, dtype=float)[order]
    if not np.isfinite(dense).all() or not np.isfinite(bm25).all() or (bm25 < 0).any():
        raise ValueError("scores must be finite and BM25 nonnegative")
    by_parent = defaultdict(list)
    duplicate_lookup = defaultdict(list)
    for i, unit in enumerate(clean):
        if unit["span"] is not None:
            by_parent[unit["span"]["fact_id"]].append(i)
        if unit["sources"]:
            duplicate_lookup[(tuple(unit["sources"]), unit["text"])].append(i)
    overlaps = []
    for indices in by_parent.values():
        active = []
        for i in sorted(indices, key=lambda j: (clean[j]["span"]["start"], clean[j]["span"]["end"], clean[j]["id"])):
            start = clean[i]["span"]["start"]
            active = [j for j in active if clean[j]["span"]["end"] > start]
            overlaps.extend((min(i,j),max(i,j)) for j in active)
            active.append(i)
    duplicates = [indices for indices in duplicate_lookup.values() if len(indices)>1]
    return {"units": clean, "dense": dense, "bm25": bm25, "group_costs": gc,
            "groups": groups, "original_indices": order,
            "overlap_pairs": sorted(set(overlaps)), "duplicate_groups": duplicates}

def selected_cost(units: Sequence[Mapping[str, Any]], selected: Sequence[int], group_costs: Mapping[str, int]) -> int:
    return sum(int(units[i]["cost"]) for i in selected) + sum(int(group_costs[g]) for g in {str(units[i]["group"]) for i in selected})


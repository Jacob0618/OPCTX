"""Score-free set-selection objectives without mandatory lexical coverage.

The caller supplies immutable units, dense/BM25 scores and exact additive
render costs. No question text, reference answer, category or source-closure
reward is consumed. A certificate concerns only the stated finite surrogate,
within recorded numerical tolerance, never semantic answer completeness.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import rankdata

VERSION = "set_selectors.pareto_geometric.v1"
TOL = 1e-7
VARIANTS = ("pareto_relevance", "geometric_relevance")
OBJECTIVES = {
    "pareto_relevance": "max min_k(U_k(S)/I_k), then max sum_k(U_k(S)/I_k), then minimum integer rendering cost; each I_k is optimized over the same original feasible domain; no lexical coverage constraint",
    "geometric_relevance": "max sum_i in S geometric_mean_k(u_ki) over informative channels, then minimum integer rendering cost; geometric mean is a rank-agreement surrogate, not a probability",
}


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


def relevance_arrays(dense_scores: Sequence[float], bm25_scores: Sequence[float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if len(dense_scores) != len(bm25_scores):
        raise ValueError("channel length mismatch")
    channels = {"dense": percentile_utility(dense_scores, positive_only=False),
                "bm25": percentile_utility(bm25_scores, positive_only=True)}
    active = [v for v in channels.values() if np.any(v)]
    relevance = np.prod(np.vstack(active), axis=0) ** (1/len(active)) if active else np.zeros(len(dense_scores))
    return channels, relevance


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


def select_units(units: Sequence[Mapping[str, Any]], dense_scores: Sequence[float], bm25_scores: Sequence[float], *,
                 variant: str, max_units: int = 50, char_budget: int, group_costs: Mapping[str, int],
                 min_units: int = 0, time_limit: float = 5.0) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if not isinstance(time_limit, (float, int)) or isinstance(time_limit, bool) or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    started = time.perf_counter()
    data = validate_inputs(units, dense_scores, bm25_scores, max_units=max_units, char_budget=char_budget,
                           group_costs=group_costs, min_units=min_units)
    fs, dense, bm25 = data["units"], data["dense"], data["bm25"]
    groups, gc = data["groups"], data["group_costs"]
    n, ng = len(fs), len(groups)
    group_index = {g: j for j, g in enumerate(groups)}
    tau = n + ng
    nv = tau + 1
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[:n] = 1
    constraints, lows, highs = [], [], []
    def add(row, lo=-np.inf, hi=np.inf):
        constraints.append(dict(row)); lows.append(float(lo)); highs.append(float(hi))
    add({i: 1 for i in range(n)}, lo=min_units, hi=max_units)
    costs = np.zeros(nv)
    members = defaultdict(list)
    for i, unit in enumerate(fs):
        costs[i] = unit["cost"]
        members[unit["group"]].append(i)
    for group, indices in members.items():
        zi = n + group_index[group]
        costs[zi] = gc[group]
        for i in indices:
            add({i: 1, zi: -1}, hi=0)
        add({zi: 1, **{i: -1 for i in indices}}, hi=0)
    add({i: v for i, v in enumerate(costs) if v}, hi=char_budget)
    for i,j in data["overlap_pairs"]:
        add({i:1,j:1},hi=1)
    for indices in data["duplicate_groups"]:
        add({i:1 for i in indices},hi=1)
    utilities, geometric = relevance_arrays(dense, bm25)
    ideals = {}
    stages = []
    current = np.zeros(nv) if min_units == 0 else None
    all_optimal = True
    stopped = False
    returned_from = "known_empty_feasible_set" if current is not None else None

    def solve(name, objective, *, lock=False):
        nonlocal current, all_optimal, stopped, returned_from
        rr, cc, vv = [], [], []
        for row_index, row in enumerate(constraints):
            for column, value in row.items():
                rr.append(row_index); cc.append(column); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(constraints), nv)).tocsc()
        t0 = time.perf_counter()
        result = milp(-objective, integrality=integrality, bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lows, highs),
                      options={"time_limit": float(time_limit), "mip_rel_gap": 0.0, "presolve": True})
        reconstructed, violation = None, None
        feasible = result.x is not None
        if feasible:
            raw = np.asarray(result.x, dtype=float)
            feasible = bool(np.isfinite(raw).all() and np.max(abs(raw[:n]-np.rint(raw[:n])), initial=0) < 1e-5)
            if feasible:
                reconstructed = np.zeros(nv)
                reconstructed[:n] = np.rint(raw[:n])
                chosen = np.flatnonzero(reconstructed[:n] > .5)
                for group, indices in members.items():
                    reconstructed[n + group_index[group]] = any(reconstructed[i] > .5 for i in indices)
                reconstructed[tau] = min([1.] + [float(utilities[k][chosen].sum()/ideal) for k, ideal in ideals.items()]) if ideals else 0.
                actual = matrix @ reconstructed
                violation = max(0., float(np.max(np.asarray(lows)-actual, initial=0)), float(np.max(actual-np.asarray(highs), initial=0)))
                feasible = bool(violation <= 1e-10 and np.all(reconstructed >= 0) and np.all(reconstructed <= 1))
        optimal = bool(result.status == 0 and feasible)
        upper = getattr(result, "mip_dual_bound", None)
        gap = getattr(result, "mip_gap", None)
        stages.append({"name": name, "status": int(result.status), "message": str(result.message),
                       "feasible_incumbent": feasible, "optimal": optimal,
                       "seconds": time.perf_counter()-t0, "objective": float(objective @ reconstructed) if feasible else None,
                       "maximization_upper_bound": -float(upper) if upper is not None and math.isfinite(float(upper)) else None,
                       "mip_gap": float(gap) if gap is not None and math.isfinite(float(gap)) else None,
                       "reconstructed_max_constraint_violation": violation})
        if feasible:
            current, returned_from = reconstructed, name
        if not optimal:
            stopped, all_optimal = True, False
            return False
        if lock:
            add({i: value for i, value in enumerate(objective) if value}, lo=float(objective @ current)-TOL)
        return True

    if variant == "pareto_relevance":
        for channel, utility in utilities.items():
            if not np.any(utility):
                continue
            objective = np.zeros(nv); objective[:n] = utility
            if not solve(channel + "_ideal", objective):
                break
            value = float(objective @ current)
            if value > TOL:
                ideals[channel] = value
        normalized = np.zeros(nv)
        if not stopped and ideals:
            for channel, ideal in ideals.items():
                ratio = utilities[channel] / ideal
                add({tau: 1, **{i: -v for i, v in enumerate(ratio) if v}}, hi=0)
                normalized[:n] += ratio
            objective = np.zeros(nv); objective[tau] = 1
            solve("pareto_max_min_relevance", objective, lock=True)
        if not stopped and ideals:
            solve("pareto_sum_relevance_tiebreak", normalized, lock=True)
    else:
        objective = np.zeros(nv); objective[:n] = geometric
        solve("geometric_relevance_sum", objective, lock=True)
    if not stopped:
        solve("minimum_render_characters_tiebreak", -costs)
    if current is None:
        raise RuntimeError("solver did not produce a feasible incumbent; no set returned")
    chosen = list(np.flatnonzero(current[:n] > .5))
    cost = selected_cost(fs, chosen, gc)
    if not min_units <= len(chosen) <= max_units or cost > char_budget:
        raise RuntimeError("hard count or rendering-budget violation")
    chosen_set = set(chosen)
    if any(i in chosen_set and j in chosen_set for i,j in data["overlap_pairs"]) or any(len(chosen_set.intersection(g))>1 for g in data["duplicate_groups"]):
        raise RuntimeError("hard overlapping-span or duplicate-provenance violation")
    ordered = sorted(chosen, key=lambda i: (-float(dense[i]), fs[i]["id"]))
    return {"version": VERSION, "variant": variant, "objective_definition": OBJECTIVES[variant],
            "selected_ids": [fs[i]["id"] for i in ordered], "selected_count": len(chosen), "cost": cost,
            "min_units": min_units, "max_units": max_units, "char_budget": char_budget,
            "cost_unit": "caller-supplied additive Unicode characters including shared group costs, not tokens",
            "selected_unit_metadata": [{"id": fs[i]["id"], "group": fs[i]["group"], "span": fs[i]["span"], "sources": fs[i]["sources"]} for i in ordered],
            "overlap_conflict_pairs": len(data["overlap_pairs"]), "exact_duplicate_classes": len(data["duplicate_groups"]),
            "certified_optimal": all_optimal and not stopped,
            "certificate_scope": "finite stated surrogate with floating arithmetic and recorded lexicographic tolerance; not answer completeness",
            "termination": "all_stages_optimal" if not stopped else "partial_stage_incumbent",
            "returned_incumbent_from_stage": returned_from, "solver_stages": stages,
            "time_limit_seconds_per_stage": float(time_limit), "lexicographic_tolerance": TOL,
            "seconds": time.perf_counter()-started, "channel_ideal_values": ideals,
            "channel_utilities": {k: float(u[chosen].sum()) for k, u in utilities.items()},
            "channel_ratios": {k: float(utilities[k][chosen].sum()/v) for k, v in ideals.items()},
            "geometric_relevance_sum": float(geometric[chosen].sum()),
            "informative_channels": [k for k, u in utilities.items() if np.any(u)],
            "literal_coverage_priority": False, "source_closure_reward": False,
            "semantic_completeness_claim": False, "new_model_calls": 0}

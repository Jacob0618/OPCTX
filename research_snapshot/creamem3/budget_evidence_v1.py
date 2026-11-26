"""Frozen, gold-free lexicographic evidence selection (research baseline).

No learned/model reranker, no dataset-specific coefficients or branch rules.
Exactness is about the supplied finite surrogate, NOT answer correctness.
All costs are additive character accounting supplied by the renderer adapter.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import rankdata

VERSION = "budget_evidence.lexical_robust.v1.0"
# English function words only; no image/color/OCR/dataset vocabulary removed.
STOPWORDS = frozenset("""a an and are as at be been being but by can could did do
does doing for from had has have having he her hers him his how i if in into is
it its may me might mine more most much my of on or our ours she should so than
that the their theirs them then there these they this those through to too under
up us was we were what when where which who why will with would you your yours
s t""".split())
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
TOL = 1e-7  # numerical solver tolerance, not a retrieval hyperparameter


def tokens(text: str) -> set[str]:
    return {s.casefold() for s in TOKEN.findall(text)}


def percentile_utility(values: Sequence[float], *, positive_only: bool) -> np.ndarray:
    """Midrank empirical position, not a probability; zero for inactive channel."""
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("non-finite retrieval score")
    if len(a) == 0:
        return a
    if positive_only:
        active = a > 0
        if not active.any():
            return np.zeros(len(a))
    elif np.ptp(a) == 0:
        return np.zeros(len(a))  # flat semantic channel has no ranking evidence
    p = (rankdata(a, method="average") - 1) / max(len(a) - 1, 1)
    if positive_only:
        p[a <= 0] = 0
        if len(a) == 1 and a[0] > 0:
            p[0] = 1
    return p


def select_evidence(
    facts: Sequence[Mapping[str, Any]], question: str,
    dense_scores: Sequence[float], bm25_scores: Sequence[float],
    max_facts: int, char_budget: int, group_costs: Mapping[str, int],
    time_limit: float = 5.0,
    *, requirements: Sequence[Mapping[str, Any]] | None = None,
    packages: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Coverage -> balanced retrieval utility -> utility sum -> minimum cost.

    Optional explicit requirements: {label, supports:[fact_id,...]}.
    Optional AND packages: {label, requirements:[requirement_label,...]}.
    Default requirements are literal non-function query terms, with no maximum
    role count. Missing terms remain visible as unsupported; no gold is accepted.
    Duplicate exclusion only matches byte-equivalent Python text AND source set.
    On a solver time limit, return the feasible incumbent with a partial-stage
    label, never silently claim the full lexicographic optimum.
    """
    began = time.perf_counter()
    if len(facts) != len(dense_scores) or len(facts) != len(bm25_scores):
        raise ValueError("score/fact length mismatch")
    if max_facts < 0 or char_budget < 0 or time_limit <= 0:
        raise ValueError("invalid resource budget")
    incoming_ids = [str(f["id"]) for f in facts]
    if len(set(incoming_ids)) != len(incoming_ids):
        raise ValueError("duplicate fact ID")
    order = sorted(range(len(facts)), key=lambda i: incoming_ids[i])
    fs = [facts[i] for i in order]
    ds = np.asarray(dense_scores, dtype=float)[order]
    bs = np.asarray(bm25_scores, dtype=float)[order]
    ids = [str(f["id"]) for f in fs]
    lookup = {fid: i for i, fid in enumerate(ids)}
    n = len(fs)
    groups = sorted({str(f["group"]) for f in fs})
    gi = {g: i for i, g in enumerate(groups)}
    raw_costs = [f["cost"] for f in fs] + [group_costs[g] for g in groups]
    if any(not isinstance(c, (int, np.integer)) or c < 0 for c in raw_costs):
        raise ValueError("costs must be nonnegative integer characters")
    f_tokens = [tokens(str(f["text"])) for f in fs]
    if requirements is None:
        reqs = [{"label": t, "supports": [ids[i] for i, ts in enumerate(f_tokens) if t in ts]}
                for t in sorted(tokens(question) - STOPWORDS)]
    else:
        reqs = [dict(r) for r in requirements]
    req_labels = [str(r["label"]) for r in reqs]
    if len(set(req_labels)) != len(req_labels):
        raise ValueError("duplicate requirement label")
    ri = {label: j for j, label in enumerate(req_labels)}
    supports = [sorted({lookup[str(fid)] for fid in r["supports"]}) for r in reqs]
    pack_members = []
    for package in packages:
        members = sorted({ri[str(label)] for label in package["requirements"]})
        if not members:
            raise ValueError("empty AND package")
        pack_members.append(members)
    m, g, h = len(reqs), len(groups), len(pack_members)
    y0, z0, p0, robust = n, n + m, n + m + g, n + m + g + h
    nv = robust + 1
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[:n] = 1
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def constrain(row: Mapping[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        rows.append(dict(row)); lower.append(float(lo)); upper.append(float(hi))

    constrain({i: 1 for i in range(n)}, hi=max_facts)
    costs = np.zeros(nv)
    for i, f in enumerate(fs):
        costs[i] = int(f["cost"])
    for group, j in gi.items():
        costs[z0 + j] = int(group_costs[group])
        members = [i for i, f in enumerate(fs) if str(f["group"]) == group]
        for i in members:
            constrain({i: 1, z0 + j: -1}, hi=0)
        constrain({z0 + j: 1, **{i: -1 for i in members}}, hi=0)
    constrain({i: v for i, v in enumerate(costs) if v}, hi=char_budget)
    for j, ss in enumerate(supports):
        constrain({y0+j: 1, **{i: -1 for i in ss}}, hi=0)
        # OR equality prevents an AND package from inventing partial supports.
        for i in ss:
            constrain({i: 1, y0+j: -1}, hi=0)
    for j, members in enumerate(pack_members):
        for member in members:
            constrain({p0+j: 1, y0+member: -1}, hi=0)
        constrain({p0+j: 1, **{y0+k: -1 for k in members}}, lo=1-len(members))
    duplicate_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, f in enumerate(fs):
        sources = tuple(sorted(set(map(str, f.get("sources", [])))))
        # Without provenance equality there is no warrant to collapse events.
        key = (sources or ("NO_SOURCE", ids[i]), str(f["text"]))
        duplicate_groups[key].append(i)
    for members in duplicate_groups.values():
        if len(members) > 1:
            constrain({i: 1 for i in members}, hi=1)

    stages: list[dict[str, Any]] = []
    current = np.zeros(nv)
    all_optimal = True
    stopped = False
    utilities = {
        "dense": percentile_utility(ds, positive_only=False),
        "bm25": percentile_utility(bs, positive_only=True),
    }
    ideals: dict[str, float] = {}

    def solve(name: str, objective: np.ndarray, *, lock: bool = False,
              integer_objective: bool = False) -> bool:
        nonlocal current, all_optimal, stopped
        rr, cc, vv = [], [], []
        for row_idx, row in enumerate(rows):
            for col, value in row.items():
                rr.append(row_idx); cc.append(col); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), nv)).tocsc()
        start = time.perf_counter()
        result = milp(-objective, integrality=integrality, bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True})
        feasible = result.x is not None
        reconstructed = None
        max_violation = None
        if feasible:
            x = np.asarray(result.x)
            product = matrix @ x
            feasible = bool(np.max(np.abs(x[:n]-np.rint(x[:n])), initial=0) < 1e-5
                            and np.all(product >= np.asarray(lower)-1e-5)
                            and np.all(product <= np.asarray(upper)+1e-5)
                            and np.all(x >= -1e-5) and np.all(x <= 1+1e-5))
            if feasible:
                # Independently reconstruct every auxiliary from integer facts.
                # A loose solver feasibility tolerance must not weaken our
                # tighter lexicographic locks or certify a rounded bad set.
                reconstructed = np.zeros(nv)
                reconstructed[:n] = np.rint(x[:n])
                chosen_now = {i for i in range(n) if reconstructed[i] > .5}
                for j, ss in enumerate(supports):
                    reconstructed[y0+j] = bool(chosen_now.intersection(ss))
                for group, j in gi.items():
                    reconstructed[z0+j] = any(str(fs[i]["group"]) == group for i in chosen_now)
                for j, members in enumerate(pack_members):
                    reconstructed[p0+j] = all(reconstructed[y0+k] > .5 for k in members)
                reconstructed[robust] = min([1.0] + [
                    float(utilities[channel][list(chosen_now)].sum()/ideal)
                    for channel, ideal in ideals.items()]) if ideals else 0.0
                exact_product = matrix @ reconstructed
                max_violation = max(0.0, float(np.max(np.asarray(lower)-exact_product, initial=0)),
                                    float(np.max(exact_product-np.asarray(upper), initial=0)))
                # Locks already contain TOL; allow only floating arithmetic
                # roundoff here, not a second retrieval/solver tolerance.
                feasible = max_violation <= 1e-10
        optimal = bool(result.status == 0 and feasible)
        stage = {"name": name, "status": int(result.status), "message": result.message,
                 "feasible_incumbent": feasible, "optimal": optimal,
                 "seconds": time.perf_counter()-start,
                 "objective": float(objective @ reconstructed) if feasible else None,
                 "reconstructed_max_constraint_violation": max_violation,
                 "maximization_upper_bound": (-float(result.mip_dual_bound)
                     if getattr(result, "mip_dual_bound", None) is not None
                     and math.isfinite(float(result.mip_dual_bound)) else None),
                 "mip_gap": (float(result.mip_gap) if getattr(result, "mip_gap", None) is not None
                             and math.isfinite(float(result.mip_gap)) else None)}
        stages.append(stage)
        if feasible:
            current = reconstructed
        if not optimal:
            all_optimal = False
            stopped = True
            return False
        if lock:
            value = float(objective @ current)
            if integer_objective:
                value = round(value)
                constrain({i: v for i, v in enumerate(objective) if v}, lo=value, hi=value)
            else:
                constrain({i: v for i, v in enumerate(objective) if v}, lo=value-TOL)
        return True

    if h:
        obj = np.zeros(nv); obj[p0:p0+h] = 1
        solve("complete_packages", obj, lock=True, integer_objective=True)
    if not stopped:
        obj = np.zeros(nv); obj[y0:y0+m] = 1
        solve("literal_coverage", obj, lock=True, integer_objective=True)
    if not stopped:
        for channel, utility in utilities.items():
            if not np.any(utility):
                continue
            obj = np.zeros(nv); obj[:n] = utility
            if not solve(channel+"_ideal", obj):
                break
            value = float(obj @ current)
            if value > TOL:
                ideals[channel] = value
    normalized = np.zeros(nv)
    if not stopped and ideals:
        for channel, ideal in ideals.items():
            ratio = utilities[channel] / ideal
            constrain({robust: 1, **{i: -v for i, v in enumerate(ratio) if v}}, hi=0)
            normalized[:n] += ratio
        obj = np.zeros(nv); obj[robust] = 1
        solve("balanced_channel_utility", obj, lock=True)
    if not stopped and ideals:
        solve("sum_channel_utility_tiebreak", normalized, lock=True)
    if not stopped:
        solve("minimum_characters_tiebreak", -costs)

    chosen = {i for i in range(n) if current[i] > 0.5}
    selected = sorted(chosen, key=lambda i: (-float(ds[i]), ids[i]))
    opened = {str(fs[i]["group"]) for i in chosen}
    cost = sum(int(fs[i]["cost"]) for i in chosen) + sum(int(group_costs[gr]) for gr in opened)
    covered = {j for j, ss in enumerate(supports) if chosen.intersection(ss)}
    completed = [j for j, members in enumerate(pack_members) if set(members) <= covered]
    if cost > char_budget or len(chosen) > max_facts:
        raise RuntimeError("post-solve hard budget validation failed")
    return {"version": VERSION, "selected_ids": [ids[i] for i in selected],
            "selected_count": len(selected), "cost": cost, "char_budget": char_budget,
            "cost_unit": "adapter_defined_additive_characters_not_tokens",
            "certified_optimal": all_optimal and not stopped,
            "termination": "all_stages_optimal" if not stopped else "partial_stage_incumbent",
            "solver_stages": stages, "seconds": time.perf_counter()-began,
            "objectives": {"literal_coverage": len(covered), "requirements": len(reqs),
                           "complete_packages": len(completed), "packages": h,
                           "channel_ideal_values": ideals,
                           "channel_utility": {c: float(u[list(chosen)].sum()) for c,u in utilities.items()},
                           "channel_ratios": {c: float(utilities[c][list(chosen)].sum()/v) for c,v in ideals.items()}},
            "covered_requirements": [req_labels[j] for j in sorted(covered)],
            "unsupported_requirements": [req_labels[j] for j, ss in enumerate(supports) if not ss],
            "missing_requirements": [req_labels[j] for j in range(m) if j not in covered],
            "duplicate_classes": sum(len(v)>1 for v in duplicate_groups.values()),
            "semantic_completeness_claim": False,
            "zero_model_rerank_calls": True}

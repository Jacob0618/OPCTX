"""Query-independent memory units; local exact support-closed set packing.

The objective is a geometric surrogate, NOT a probability or QA guarantee.
Every selected fact occupies exactly one slot and can receive credit only once.
"""
import threading
import warnings
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

SOLVER_LOCK = threading.Lock()


def normalize(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim not in (1, 2) or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding matrix")
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding")
    return matrix / norms


def choose(fact_scores, units, unit_scores, k=20, mode="closed"):
    """units contain indices into fact_scores. Expose no gold/category inputs."""
    scores = np.maximum(0.0, np.asarray(fact_scores, dtype=np.float64))
    n = len(scores)
    if n < k or not np.isfinite(scores).all() or len(units) != len(unit_scores):
        raise ValueError("Invalid selection inputs")
    clean = []
    for i, (unit, score) in enumerate(zip(units, unit_scores)):
        members = sorted(set(unit["indices"]))
        if not members or min(members) < 0 or max(members) >= n or not np.isfinite(score):
            raise ValueError("Invalid memory-unit support")
        if len(members) <= k:
            clean.append((i, members, max(0.0, float(score))))
    if mode in {"cosine", "flat"}:
        utilities = scores.copy()
        if mode == "flat":
            for _, members, score in clean:
                utilities[members] = np.maximum(utilities[members], score)
        chosen = sorted(range(n), key=lambda i: (-float(utilities[i]), i))[:k]
        selected_units = []
        objective = float(utilities[chosen].sum())
        status = "pointwise_exact"
    elif mode == "closed":
        # Non-positive-gain packages cannot beat their singleton replacement.
        best_support = {}
        for ui, members, score in clean:
            if len(members) * score <= float(scores[members].sum()) + 1e-12:
                continue
            key = tuple(members)
            previous = best_support.get(key)
            if previous is None or score > previous[2]:
                best_support[key] = (ui, members, score)
        packages = [best_support[key] for key in sorted(best_support)]
        objective_values = np.concatenate([scores, [len(m) * s for _, m, s in packages]])
        row, col, data = list(range(1, n+1)), list(range(n)), [1.0] * n
        row.extend([0] * n); col.extend(range(n)); data.extend([1.0] * n)
        for j, (_, members, _) in enumerate(packages, n):
            row.append(0); col.append(j); data.append(float(len(members)))
            for member in members:
                row.append(member+1); col.append(j); data.append(1.0)
        matrix = coo_matrix((data, (row, col)), shape=(n+1, len(objective_values))).tocsc()
        with SOLVER_LOCK, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            result = milp(-objective_values, integrality=np.ones(len(objective_values)),
                bounds=Bounds(0, 1), constraints=LinearConstraint(matrix,
                    np.r_[k, np.zeros(n)], np.r_[k, np.ones(n)]),
                options={"time_limit": 15.0, "mip_rel_gap": 0.0, "threads": 1})
        if result.status != 0 or result.x is None:
            raise ValueError("Support packing did not prove optimality: " + str(result.message))
        binary = np.rint(result.x).astype(int)
        if np.max(np.abs(result.x-binary)) > 1e-5:
            raise ValueError("Non-integral solver result")
        chosen = list(np.flatnonzero(binary[:n]))
        selected_units = []
        for j, (ui, members, _) in enumerate(packages, n):
            if binary[j]:
                chosen.extend(members); selected_units.append(ui)
        objective = float(objective_values @ binary)
        status = "milp_optimal"
    else:
        raise ValueError(mode)
    if len(chosen) != k or len(set(chosen)) != k:
        raise ValueError("Fact budget or disjoint-support violation")
    # Common presentation order controls selection vs ordering effects.
    chosen.sort(key=lambda i: (-float(scores[i]), i))
    return {"indices": chosen, "unit_indices": selected_units,
            "objective": objective, "status": status, "mode": mode,
            "model_rerank_calls": 0, "facts": k}

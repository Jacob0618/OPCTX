"""Generic numerical gate for finite MILP proxy objectives, not retrieval tuning.

Disable the locally demonstrated presolve certificate inconsistency uniformly.
Record raw solver values; independently gate both raw and reconstructed primal,
dual, integrality, bounds, and constraints. Feasible partial sets remain useful
but must never be described as certified optimal.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import milp as scipy_milp

VERSION = 'milp.numerical_gate.v2'
TOLERANCE = 1e-7


def _finite(value):
    return value is not None and math.isfinite(float(value))


def milp(c, *, integrality, bounds, constraints, options):
    """One solver call, fixed no-presolve policy, no silent solver fallback."""
    if options.get('presolve') is not False or options.get('mip_rel_gap') != 0.0:
        raise ValueError('numerical v2 requires presolve=False and mip_rel_gap=0')
    if not np.isin(np.asarray(integrality), (0,1)).all():
        raise ValueError('numerical v2 supports continuous and ordinary integer variables only')
    result = scipy_milp(c, integrality=integrality, bounds=bounds,
                       constraints=constraints, options=options)
    objective = np.asarray(c, dtype=float)
    audit = {'schema': VERSION, 'presolve': False, 'tolerance': TOLERANCE,
             'raw_fun': float(result.fun) if _finite(result.fun) else None,
             'raw_maximization_upper_bound': -float(result.mip_dual_bound) if _finite(getattr(result, 'mip_dual_bound', None)) else None,
             'raw_max_objective': None, 'raw_constraint_violation': None,
             'raw_bound_violation': None, 'raw_integer_rounding_delta': None,
             'raw_checks_pass': False, 'raw_check_failures': []}
    failed = audit['raw_check_failures']
    if result.x is None:
        failed.append('no_raw_incumbent')
    else:
        x = np.asarray(result.x, dtype=float)
        if x.shape != objective.shape or not np.isfinite(x).all():
            failed.append('raw_shape_or_nonfinite')
        else:
            ax = constraints.A @ x
            cv = max(0., float(np.max(constraints.lb-ax, initial=0)), float(np.max(ax-constraints.ub, initial=0)))
            bv = max(0., float(np.max(bounds.lb-x, initial=0)), float(np.max(x-bounds.ub, initial=0)))
            integer = np.asarray(integrality) != 0
            iv = float(np.max(abs(x[integer]-np.rint(x[integer])), initial=0))
            primal = float(-objective @ x)
            audit.update(raw_max_objective=primal, raw_constraint_violation=cv,
                         raw_bound_violation=bv, raw_integer_rounding_delta=iv)
            if cv > TOLERANCE: failed.append('raw_constraint_violation')
            if bv > TOLERANCE: failed.append('raw_bound_violation')
            if iv > TOLERANCE: failed.append('raw_integrality_violation')
            if audit['raw_fun'] is None or abs(primal+audit['raw_fun']) > TOLERANCE:
                failed.append('raw_fun_objective_mismatch')
            upper = audit['raw_maximization_upper_bound']
            if upper is not None and primal-upper > TOLERANCE:
                failed.append('raw_primal_exceeds_dual')
    audit['raw_checks_pass'] = not failed
    result['numerical_raw_audit'] = audit
    return result


def audit_reconstruction(result, objective, reconstructed, feasible, *, integer_objective=False):
    """Necessary numerical consistency, not a mathematical/semantic oracle."""
    audit = dict(result.numerical_raw_audit)
    failures = list(audit['raw_check_failures'])
    obj = np.asarray(objective, dtype=float)
    if feasible:
        candidate = np.asarray(reconstructed, dtype=float)
        if candidate.shape != obj.shape or not np.isfinite(candidate).all() or not np.isfinite(obj).all():
            feasible = False
            failures.append('reconstructed_shape_or_nonfinite')
    primal = float(obj @ reconstructed) if feasible else None
    audit['reconstructed_max_objective'] = primal
    audit['reconstruction_objective_delta'] = None
    upper = audit['raw_maximization_upper_bound']
    audit['integer_objective'] = bool(integer_objective)
    effective_upper = upper
    if integer_objective and upper is not None:
        # Every achievable value is an integer (coverage/count or exact integer
        # character cost). An upper bound between integers proves the same
        # discrete optimum after flooring. This is not a retrieval tolerance.
        effective_upper = math.floor(upper+TOLERANCE)
    audit['effective_maximization_upper_bound'] = effective_upper
    if not feasible:
        failures.append('reconstructed_incumbent_infeasible_or_missing')
    else:
        raw = audit['raw_max_objective']
        if raw is None:
            failures.append('raw_objective_missing')
        else:
            audit['reconstruction_objective_delta'] = primal-raw
            allowance = TOLERANCE
            if integer_objective:
                obj = np.asarray(objective)
                if (np.max(abs(reconstructed[obj != 0]-np.asarray(result.x)[obj != 0]), initial=0) > TOLERANCE
                        or abs(primal-round(primal)) > 1e-10):
                    failures.append('integer_objective_reconstruction_not_small_roundoff')
                allowance *= max(1., float(abs(obj).sum()))
            audit['reconstruction_objective_roundoff_allowance'] = allowance
            if abs(primal-raw) > allowance:
                failures.append('raw_reconstructed_objective_mismatch')
        if effective_upper is not None and primal-effective_upper > TOLERANCE:
            failures.append('reconstructed_primal_exceeds_dual')
        if result.status == 0:
            if effective_upper is None:
                failures.append('optimal_without_finite_dual')
            elif effective_upper-primal > TOLERANCE:
                failures.append('optimal_dual_primal_gap')
    audit['check_failures'] = failures
    audit['numerical_checks_pass'] = not failures
    audit['certified_optimal'] = bool(result.status == 0 and feasible and not failures)
    audit['certificate_scope'] = 'Necessary finite surrogate floating-point consistency; no semantic answer-completeness claim'
    return audit


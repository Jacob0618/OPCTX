"""Exact ordinal evidence selection with alternative sufficient support sets.

No fitted relevance weights. Lexicographic action utility is a design choice,
not QA optimality. A solver must prove feasibility/infeasibility at every step;
timeouts never turn into cosine fallback or a claimed mathematical optimum.
"""
import threading
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from creamem3.experiments.locomo_online_semantics_20260909.proof_factorization import canonical_proofs

_SOLVER_LOCK=threading.Lock()


def choose(actions,proof_map,fallback,budget,*,time_limit=10.0):
    if not isinstance(actions,list) or any(not isinstance(a,str) or a not in proof_map for a in actions):raise ValueError('Unknown action')
    if len(fallback)!=budget or len(set(fallback))!=budget or budget<1:raise ValueError('Invalid ranked fallback')
    proofs={}
    for a,groups in proof_map.items():
        if not groups or any(not isinstance(p,list) or not p or any(not isinstance(f,str) for f in p) for p in groups):raise ValueError('Invalid proof group')
        proofs[a]=canonical_proofs(groups)
    ordered=list(dict.fromkeys(actions));universe=sorted(set(fallback)|{f for a in ordered for p in proofs[a] for f in p})
    positions={f:i for i,f in enumerate(universe)};n=len(universe);calls=0

    def solve(required,forced):
        nonlocal calls
        calls+=1
        groups=[(a,p) for a in required for p in proofs[a]]
        rows=1+len(required)+sum(len(p) for a,p in groups)
        matrix=lil_matrix((rows,n+len(groups)),dtype=np.float64);lo=[];hi=[];r=0
        matrix[r,:n]=1;lo.append(0);hi.append(budget);r+=1
        for a in required:
            for j,(b,p) in enumerate(groups):
                if a==b:matrix[r,n+j]=1
            lo.append(1);hi.append(1);r+=1
        for j,(a,p) in enumerate(groups):
            for f in p:
                matrix[r,n+j]=1;matrix[r,positions[f]]=-1
                lo.append(-np.inf);hi.append(0);r+=1
        lower=np.zeros(n+len(groups));upper=np.ones(n+len(groups))
        for f in forced:lower[positions[f]]=1
        costs=np.r_[np.ones(n),np.zeros(len(groups))]
        with _SOLVER_LOCK:
            result=milp(costs,integrality=np.ones(n+len(groups)),bounds=Bounds(lower,upper),
                        constraints=LinearConstraint(matrix.tocsc(),np.array(lo),np.array(hi)),
                        options={'time_limit':time_limit,'mip_rel_gap':0.0})
        if result.status==2:return None
        if result.status!=0 or result.x is None:raise RuntimeError('No proven optimal MILP result: '+str(result.message))
        if np.max(np.abs(result.x-np.rint(result.x)))>1e-6:raise RuntimeError('Nonintegral MILP evidence solution')
        selected={f for f,i in positions.items() if result.x[i]>.5}
        if len(selected)>budget or not set(forced)<=selected or any(not any(set(p)<=selected for p in proofs[a]) for a in required):
            raise RuntimeError('MILP result failed exact Boolean check')
        return selected

    accepted=[];skipped=[];forced=[]
    for a in ordered:
        if solve(accepted+[a],forced) is not None:accepted.append(a)
        else:skipped.append(a)
    # Prefer the pre-existing model-ranked facts, without freezing a particular
    # proof too early. Alternative proofs remain globally reassignable.
    for f in fallback:
        if solve(accepted,forced+[f]) is not None:forced.append(f)
    # Exact canonical tie-break only within declared proofs/ranked fallback.
    # This is not a cosine padding pool and introduces no unrelated facts.
    for f in universe:
        if len(forced)==budget:break
        if f not in forced and solve(accepted,forced+[f]) is not None:forced.append(f)
    if len(forced)!=budget:raise RuntimeError('DNF decoder failed exact budget fill')
    selected=set(forced)
    chosen=[f for f in fallback if f in selected]+sorted(selected-set(fallback))
    witnesses={a:next(p for p in proofs[a] if set(p)<=selected) for a in accepted}
    if any(any(set(p)<=selected for p in proofs[a]) for a in skipped):raise RuntimeError('Skipped action unexpectedly satisfied')
    return chosen,{'accepted_actions':accepted,'skipped_infeasible_actions':skipped,'selected_proofs':witnesses,
                   'solver_calls':calls,'per_solve_time_limit':time_limit,
                   'exact_budget':budget,'lexicographically_optimal_given_proposed_proofs_and_order':True,
                   'not_a_semantic_or_QA_optimality_guarantee':True,'no_cosine_padding':True}

"""Exact feasibility certificate only; never selects a deployment action."""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def can_complete(world, matrix, stats=None):
    """Greedy constructive certificate; if it fails, solve binary feasibility exactly.

    This is a feasibility check, never a candidate deployment action pool. No
    optimizer result is used as the learned action. Infeasible != solver timeout.
    """
    remaining = world.nodes[:, 1:3] - matrix.T @ world.stages[:, 3:5]
    if np.any(remaining < -1e-8): return False
    missing = np.flatnonzero(~matrix.any(1))
    if not len(missing): return True
    specs = world.stages[missing, 3:5]
    # A cheap sufficient certificate measured to remove repeated greedy packing
    # on generously provisioned instances. Each slot fits ANY remaining stage.
    slots=np.floor(np.maximum(remaining,0)/specs.max(0)).min(1).sum()
    if slots >= len(missing): return True
    order = np.argsort(-np.max(specs/np.maximum(remaining.max(0), 1e-9), axis=1))
    free = remaining.copy()
    for index in order:
        fits = np.flatnonzero((free+1e-9 >= specs[index]).all(1))
        if not len(fits): break
        v = fits[np.argmax((free[fits]/np.maximum(specs[index], 1e-9)).min(1))]
        free[v] -= specs[index]
    else: return True
    m, n = len(missing), world.c.n
    a = lil_matrix((m+2*n, m*n), dtype=float)
    for i in range(m):
        a[i, i*n:(i+1)*n] = 1
        for v in range(n):
            a[m+v, i*n+v] = specs[i, 0]
            a[m+n+v, i*n+v] = specs[i, 1]
    lower = np.r_[np.ones(m), np.full(2*n, -np.inf)]
    upper = np.r_[np.ones(m), remaining[:, 0], remaining[:, 1]]
    fit = (remaining[None, :, :]+1e-9 >= specs[:, None, :]).all(2).ravel()
    tick=time.perf_counter()
    result = milp(np.zeros(m*n), integrality=np.ones(m*n), bounds=Bounds(0, fit.astype(float)),
                  constraints=LinearConstraint(a.tocsr(), lower, upper))
    if stats is not None:
        stats["milp_calls"]+=1; stats["milp_seconds"]+=time.perf_counter()-tick
    if result.status == 2: return False
    if not result.success: raise RuntimeError(f"Feasibility solver could not certify: {result.message}")
    return True

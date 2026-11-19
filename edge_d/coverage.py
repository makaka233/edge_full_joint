"""Exact one-action effects on *coverage-stage* resource opportunities.

Only previews resource consequences of ADD(k,n) without changing the draft,
choosing an action, inferring reward, or postprocessing any deployment.

Output is six per-action observational features. It is NOT a guarantee of
physical CPU stability or future ability to finish all remaining placements.
"""
import numpy as np
import torch


def opportunity_effects(e, X):
    X = np.asarray(X, dtype=np.bool_)
    K, N = e.K, e.N
    assert X.shape == (K,N)
    assert e.phase == 1, 'Coverage feature must only be computed in coverage phase'
    covered = X.any(axis=1)
    missing = ~covered
    free = np.asarray(e.capacity - X.T @ e.spec, np.float64)
    spec = np.asarray(e.spec, np.float64)
    cpu = np.asarray(e.w.nodes[:,0], np.float64)
    demand = np.asarray(e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0], np.float64)
    possible = (free[None,:,:] + 1e-8 >= spec[:,None,:]).all(axis=2) & missing[:,None]
    counts = np.maximum(possible.sum(axis=1), 1)
    available_cpu = possible @ cpu
    # Mandatory load: any already placed single-replica stage has no choice of compute node.
    # During coverage every placed stage has exactly one replica.
    mandatory = np.einsum('kn,k->n',X.astype(np.float64),demand)
    before_max = float((mandatory / np.maximum(cpu,1e-9)).max())
    all_legal = e.legal()[:K*N].reshape(K,N)
    assert np.all(all_legal <= possible), 'Coverage actions must target currently uncovered stages'
    results = np.zeros((K,N,6), dtype=np.float32)
    risk_before = (demand[missing]/np.maximum(available_cpu[missing],1e-9)).max(initial=0.)
    total_future_demand = demand[missing].sum()
    for k,n in np.argwhere(all_legal):
        # One addition to node n; exact RAM/SSD impact on all other missing stages.
        next_free_n = free[n] - spec[k]
        next_eligible_n = (next_free_n[None,:] + 1e-8 >= spec).all(axis=1)
        loss = possible[:,n] & ~next_eligible_n & missing
        loss[k] = False  # the chosen stage is no longer missing
        after_counts = possible.sum(axis=1) - loss.astype(np.int64)
        after_cpu = available_cpu - loss.astype(float)*cpu[n]
        remaining = missing.copy(); remaining[k]=False
        if remaining.any():
            ratio_after = demand[remaining]/np.maximum(after_cpu[remaining],1e-9)
            worst_future = float(ratio_after.max())
        else:
            worst_future = 0.
        # Mandatory-load change is exact for this intermediate one-replica draft.
        new_max = max(before_max, float((mandatory[n]+demand[k])/max(cpu[n],1e-9)))
        lost_weighted = float((demand[loss] / np.maximum(counts[loss],1)).sum())
        # Continuous opportunity margin: also detects reduced room even when
        # the binary eligibility of other stages has not disappeared.
        # The margin uses only known RAM/SSD inventory and stage footprints.
        before_margin=np.maximum(0.,np.minimum(1.,((free[n][None,:]-spec)/np.maximum(e.capacity[n][None,:],1e-9)).min(axis=1)))
        after_margin=np.maximum(0.,np.minimum(1.,((next_free_n[None,:]-spec)/np.maximum(e.capacity[n][None,:],1e-9)).min(axis=1)))
        margin_loss=float(np.sum(demand[remaining]*(before_margin[remaining]-after_margin[remaining])))
        results[k,n] = [
            np.tanh(new_max - before_max),
            np.tanh((mandatory[n]+demand[k])/max(cpu[n],1e-9)),
            np.tanh(lost_weighted/max(total_future_demand,1e-9)*K),
            np.tanh(margin_loss/max(total_future_demand,1e-9)*K),
            np.tanh(worst_future - risk_before),
            float(covered.mean()),
        ]
    return torch.from_numpy(results), dict(mask=all_legal,eligible=possible,
        counts=possible.sum(axis=1),before_mandatory_max=before_max,
        before_optimistic_future_ratio=float(risk_before),
        candidate_count=int(all_legal.sum()))


def ground_truth_one_step(e, X, k, n):
    """Exact physical-inventory audit of one legal action, no reward or solver."""
    X=np.asarray(X,bool)
    free=np.asarray(e.capacity-X.T@e.spec,float)
    spec=np.asarray(e.spec,float)
    missing=~X.any(axis=1)
    eligible=(free[None]+1e-8>=spec[:,None]).all(2)&missing[:,None]
    after=X.copy();after[k,n]=True
    free2=np.asarray(e.capacity-after.T@e.spec,float)
    missing2=~after.any(axis=1)
    eligible2=(free2[None]+1e-8>=spec[:,None]).all(2)&missing2[:,None]
    cpu=np.asarray(e.w.nodes[:,0],float)
    dem=np.asarray(e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0],float)
    mandatory_before=X.T@dem; mandatory_after=after.T@dem
    return dict(lost_eligible_other=int((eligible&~eligible2&missing2[:,None]).sum()),
       lost_weighted=float((dem[:,None]*(eligible&~eligible2&missing2[:,None])/
                     np.maximum(eligible.sum(axis=1),1)[:,None]).sum()),
       mandatory_cpu_max_before=float((mandatory_before/cpu).max()),
       mandatory_cpu_max_after=float((mandatory_after/cpu).max()),
       legal=bool(e.legal()[k*e.N+n]))

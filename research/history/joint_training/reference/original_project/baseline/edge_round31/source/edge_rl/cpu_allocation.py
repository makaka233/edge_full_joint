"""Work-conserving CPU disciplines. Original KKT remains the default.
finish_pf maximizes weighted proportional-fair log utility at each event;
weights favor completion, regularized by the minimum catalog stage work.
Neither PS nor finish_pf is claimed to solve dynamic global E2E exactly.
"""
import numpy as np
MODES=('kkt','ps','finish_pf','fcfs','srpt')

def cpu_rates(work, capacity, mode='kkt', reference_work=None):
    w=np.asarray(work,dtype=float)
    if not np.isfinite(capacity) or capacity<=0 or not np.all(np.isfinite(w)) or np.any(w<=0):
        raise ValueError('positive finite work and capacity required')
    if mode not in MODES: raise ValueError(f'unknown CPU allocator: {mode}')
    if not len(w): return w.copy()
    if mode=='kkt': weight=np.sqrt(w)
    elif mode=='ps': weight=np.ones_like(w)
    elif mode=='finish_pf':
        if reference_work is None or not np.isfinite(reference_work) or reference_work<=0: raise ValueError('positive catalog work reference required')
        weight=1./(1.+w/reference_work)
    else:
        rates=np.zeros_like(w);rates[0 if mode=='fcfs' else int(np.argmin(w))]=capacity;return rates
    return capacity*weight/weight.sum()

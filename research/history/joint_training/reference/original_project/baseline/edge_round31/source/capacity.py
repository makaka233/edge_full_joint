"""Offline CPU feasibility audit; never invoked by a policy or action mask."""
import numpy as np
from scipy.optimize import linprog

def mat(row,K,N):
    return np.unpackbits(np.frombuffer(bytes.fromhex(row['matrix_hex']),dtype=np.uint8))[:K*N].reshape(K,N).astype(bool)

def cpu(w,m,d):
    K,N=m.shape;need=d[:,w.stage_service].sum(0)*w.stages[:,0]
    A=np.zeros((N,K*N+1));E=np.zeros((K,K*N+1))
    for k in range(K):
        E[k,k*N:(k+1)*N]=1
        for n in range(N):A[n,k*N+n]=1
    A[:,-1]=-w.nodes[:,0];b=[(0,None) if x else (0,0) for x in m.ravel()]+[(0,None)]
    c=np.zeros(K*N+1);c[-1]=1
    res=linprog(c,A_ub=A,b_ub=np.zeros(N),A_eq=E,b_eq=need,bounds=b,method='highs')
    if not res.success:raise RuntimeError(res.message)
    return float(res.fun),float(need.sum()/w.nodes[:,0].sum())

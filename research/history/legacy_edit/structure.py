"""Action-conditioned CPU load feasibility certificates, no action selection or repair.

One-stage and two-stage Hall-type necessary bounds for ideal CPU work splitting:
 max( sum_{stage in S} workload(stage) / sum_{node in union replicas(S)} CPU(node) )
for stage subsets S of cardinality 1 or 2. Both quantities are NECESSARY bounds,
not sufficient feasibility tests and not actual scheduler latency predictions.
Only available replicas are counted; uncovered stages are reported separately.
Designed as a vectorized preview for ALL (stage,node) action positions.
"""
import numpy as np
import torch


def structural_features(e, X):
    X=np.asarray(X, dtype=bool)
    assert X.shape==(e.K,e.N)
    K,N=e.K,e.N
    cap=np.asarray(e.w.nodes[:,0],dtype=np.float64)
    demand=np.asarray(e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0],dtype=np.float64)
    covered=X.any(axis=1)
    stagecap=X.astype(float)@cap
    union=X[:,None,:] | X[None,:,:]
    paircap=np.einsum('ikn,n->ik', union.astype(float),cap)
    diag=np.eye(K,dtype=bool)
    validpair=covered[:,None]&covered[None,:]&~diag
    pair=np.divide(demand[:,None]+demand[None,:],paircap,out=np.zeros((K,K)),where=validpair&(paircap>0))
    single=np.divide(demand,stagecap,out=np.zeros(K),where=covered&(stagecap>0))
    base=max(float(np.max(pair)),float(np.max(single)),float(demand[covered].sum()/cap.sum()))
    # Compute each existing cut unaffected by changing stage i once per i.
    unaffected=np.zeros(K)
    for i in range(K):
        remaining=np.ones(K,bool);remaining[i]=False
        unaffected[i]=max(float(single[remaining].max(initial=0.)),
                          float(pair[np.ix_(remaining,remaining)].max(initial=0.)),
                          float(demand[remaining&covered].sum()/cap.sum()))
    # prospective stage capacities (no mutation), candidate i->n
    newstagecap=stagecap[:,None]+cap[None,:]
    new_single=demand[:,None]/np.maximum(newstagecap,1e-12)
    after=np.maximum(unaffected[:,None],new_single)
    coverage_count=covered.sum()
    for i in range(K):
        others=np.flatnonzero(covered & (np.arange(K)!=i))
        if not len(others):continue
        # node n added to stage i: pair union gains cap[n] unless other stage has node n.
        cp=paircap[i,others,None]+cap[None,:]*(~X[others,:])
        joint=(demand[i]+demand[others,None])/np.maximum(cp,1e-12)
        after[i]=np.maximum(after[i],joint.max(axis=0))
    # Global bound applies to covered stages only; if stage i was not covered,
    # selecting this action increases accounted workload in the partial-state certificate.
    after=np.maximum(after,
        (demand[covered].sum()+(~covered).astype(float)[:,None]*demand[:,None])/cap.sum())
    # stage and union bottleneck terms are informative even when the maximum is unchanged.
    before_stage=single[:,None]
    marginal_stage=before_stage-new_single
    after_minus=after-base
    static_remaining=(e.capacity-X.T@e.spec).astype(float)
    legal=(~X)&(static_remaining[None]+1e-8>=e.spec[:,None]).all(axis=2)
    subset_scarcity=np.divide(demand[:,None],np.maximum(stagecap[:,None]+cap[None,:],1e-12))
    # Six normalized per-(stage,node) features. All finite, no labels from future.
    out=np.stack([
        np.broadcast_to(np.tanh(base),(K,N)),
        np.tanh(base-after),
        np.tanh(marginal_stage),
        np.tanh(new_single),
        np.tanh((after-1.).clip(-5,5)),
        np.broadcast_to((~covered)[:,None].astype(float),(K,N)),
    ],axis=-1).astype(np.float32)
    out[~legal]=0.
    return torch.from_numpy(out), dict(stage_pair_bound=float(base), after_bound=after,
                                     legal=legal, covered=int(coverage_count),
                                     minmax_after=float(after[legal].min()) if legal.any() else None)


def pair_bound(e,X):
    return structural_features(e,X)[1]['stage_pair_bound']


def bottleneck_cut(e, X, epsilon=1e-5):
    """Exact current ideal minmax CPU load LP + a tight Hall-type bottleneck cut.

    A cut is a *necessary* CPU capacity certificate; never a hard deployment mask
    or a replacement/suggested deployment matrix. Missing stages are excluded from
    this certificate until coverage finishes (caller should only use complete X).
    """
    import networkx as nx
    from edge_d.world import cpu_lower_bound
    X=np.asarray(X,bool)
    if not X.any(axis=1).all():
        return None
    lower=float(cpu_lower_bound(e,X))
    d=np.asarray(e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0],float)
    cpu=np.asarray(e.w.nodes[:,0],float)
    g=nx.DiGraph();src='src';dst='sink'
    total=float(d.sum());inf=total+sum(cpu)+1.
    for k in range(e.K):
        g.add_edge(src,f's{k}',capacity=float(d[k]))
        for n in np.flatnonzero(X[k]):g.add_edge(f's{k}',f'n{n}',capacity=inf)
    for n in range(e.N):g.add_edge(f'n{n}',dst,capacity=max(0.,lower*(1-epsilon)*float(cpu[n])))
    _,(left,_)=nx.minimum_cut(g,src,dst,flow_func=nx.algorithms.flow.preflow_push)
    stages=np.array([f's{k}' in left for k in range(e.K)],bool)
    nodes=np.any(X[stages],axis=0) if stages.any() else np.zeros(e.N,bool)
    if not stages.any():
        # Safe fallback for degenerate numeric scenario; not a proof of a tight set.
        return dict(value=lower,stages=stages,nodes=nodes,cut_bound=0.,tight=False,stage_count=0,node_count=0)
    ratio=float(d[stages].sum()/max(cpu[nodes].sum(),1e-12))
    return dict(value=lower,stages=stages,nodes=nodes,cut_bound=ratio,
                tight=bool(abs(ratio-lower)<1e-3),
                stage_count=int(stages.sum()),node_count=int(nodes.sum()))


def bottleneck_features(e,X):
    """Six actionable, physically grounded features for all 896 ADD positions.

    The cutoff only affects observations; every valid action remains available.
    No reward, future demand realizations or policy outcomes are consulted.
    """
    X=np.asarray(X,bool);K,N=X.shape
    cut=bottleneck_cut(e,X)
    features=np.zeros((K,N,6),np.float32)
    if cut is None:
        # Explicit phase flag, no fictitious perfect CPU capacity while uncovered.
        features[:,:,5]=float(X.any(axis=1).mean())
        return torch.from_numpy(features),None
    stage=cut['stages'];node=cut['nodes'];capacity=e.w.nodes[:,0]
    d=e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0]
    if stage.any():
        demand=float(d[stage].sum());available=float(capacity[node].sum())
        after=demand/(available+capacity[None,:])
        potential=np.maximum(cut['cut_bound']-after,0.)
        benefit=stage[:,None].astype(float)*(~node)[None,:].astype(float)
        features[:,:,0]=float(np.tanh(cut['value']))
        features[:,:,1]=benefit*np.tanh(potential[None,:])
        features[:,:,2]=np.broadcast_to(stage[:,None],(K,N))
        features[:,:,3]=np.broadcast_to((~node)[None,:],(K,N))
        features[:,:,4]=np.broadcast_to((capacity/capacity.mean())[None,:],(K,N))*benefit
    features[:,:,5]=float(X.any(axis=1).mean())
    return torch.from_numpy(features),cut

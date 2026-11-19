"""Full-option counterfactual experience, no action selection from outcome."""
import copy,numpy as np
from sac import Transition,batch,distribution
import torch

def alternative(pre,post,d,s,primary,info,seed,gs,epsilon,scale,rng_for):
    if not pre.old.any(1).all():return None,None
    z=copy.deepcopy(pre); st=z.state();m=z.legal().copy();a=z.BUILD if primary[0].a==z.KEEP else z.KEEP
    with torch.no_grad():
        x=batch([st],d.device);mm=d.tensor([m],torch.bool);_,lr=distribution(d.policy_logits(x,mm),mm);logratio=float(lr[0,a])
    path=[];mat=z.apply(a)
    if mat is None:path.append(Transition(st,m,a,0.,z.state(),z.legal().copy(),1.,False,False,logratio))
    rng=rng_for(seed,29071,gs)
    while mat is None:
        st=z.state();m=z.legal().copy();a,logratio,_=d.act(st,m,rng);mat=z.apply(a)
        if mat is None:path.append(Transition(st,m,a,0.,z.state(),z.legal().copy(),1.,False,False,logratio))
    ns,ai,_,term,trunc=z.commit(mat,s.scheduler(rng_for(seed,30,gs),epsilon),int(rng_for(seed,90,gs).integers(0,2**31-1)),gs,False)
    path.append(Transition(st,m,a,ai['reward']*scale,ns,z.legal().copy(),z.c.gamma_d,term,trunc,logratio))
    assert ai['trace_sha256']==info['trace_sha256'] and np.array_equal(ns.demand,post.state().demand)
    for t in path:t.partner_version=int(s.version);t.collection_step=gs
    log=dict(gs=gs,actual=info['decision'],alternative='KEEP' if path[0].a==z.KEEP else 'REDEPLOY',actual_reward=info['reward'],alternative_reward=ai['reward'],alternative_n=ai['n'],alternative_failed=ai['failed'],alternative_delay=ai['penalized_delay'],alternative_replicas=ai['replicas'],alternative_actions=len(path),trace_sha256=ai['trace_sha256'],alternative_matrix_hex=np.packbits(mat.ravel()).tobytes().hex())
    return path,log

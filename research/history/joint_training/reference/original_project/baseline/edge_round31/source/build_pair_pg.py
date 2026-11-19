"""Fresh conditional BUILD trajectory pairs for direct soft policy-gradient learning.
No candidate selection. The main trajectory is committed before the alternative is evaluated.
The two paths are IID given the macro state and BUILD, with common external requests.
"""
import copy, numpy as np, torch
from sac import Transition,batch,distribution

def collect_build_pair(pre,post,path,info,d,s,seed,gs,epsilon,scale,rng_for):
    if info['decision']!='REDEPLOY':return None,None
    z=copy.deepcopy(pre);z.apply(z.BUILD);rng=rng_for(seed,28371,gs);alt=[]
    while True:
        st=z.state();mask=z.legal().copy();a,lr,_=d.act(st,mask,rng);mat=z.apply(a)
        if mat is None:alt.append(Transition(st,mask,a,0.,z.state(),z.legal().copy(),1.,False,False,lr))
        else:
            ns,ai,_,term,trunc=z.commit(mat,s.scheduler(rng_for(seed,30,gs),epsilon),int(rng_for(seed,90,gs).integers(0,2**31-1)),gs,False)
            alt.append(Transition(st,mask,a,ai['reward']*scale,ns,z.legal().copy(),z.c.gamma_d,term,trunc,lr));break
    assert ai['trace_sha256']==info['trace_sha256'] and np.array_equal(ns.demand,post.state().demand)
    assert all(t.s.phase!=0 for t in path[1:]) and all(t.s.phase!=0 for t in alt)
    log=dict(gs=gs,actual_reward=info['reward'],alternative_reward=ai['reward'],actual_actions=len(path)-1,alternative_actions=len(alt),alternative_n=ai['n'],alternative_failed=ai['failed'],alternative_delay=ai['penalized_delay'],alternative_replicas=ai['replicas'],trace_sha256=ai['trace_sha256'],alternative_matrix_hex=np.packbits(mat.ravel()).tobytes().hex())
    return (path[1:],alt),log

@torch.no_grad()
def soft_return(d,path):
    last=path[-1];v=d.next_value(batch([last.ns],d.device),d.tensor([last.nm],torch.bool))[0]
    return float(last.r+(0. if last.terminated else last.discount*float(v))-sum(d.alpha_at(t.s.phase)*t.logratio for t in path))

def update_build_pg(d,pair):
    if pair is None:return {}
    a,b=pair;ga,gb=soft_return(d,a),soft_return(d,b);adv=.5*(ga-gb)
    # Constant scalar units; no length-dependent path weighting. Score sums
    # include every sampled internal action, even variable-length paths.
    # Chunked backward equals one summed loss at the same parameter snapshot.
    d.ao.zero_grad();total=0.
    for path,coeff in ((a,-adv),(b,adv)):
        for i in range(0,len(path),16):
            tr=path[i:i+16];x=batch([t.s for t in tr],d.device);m=d.tensor([t.mask for t in tr],torch.bool);act=d.tensor([t.a for t in tr],torch.long)
            p,lr=distribution(d.policy_logits(x,m),m);loss=coeff*lr.gather(1,act[:,None]).sum();loss.backward();total+=float(loss.detach())
    grad=float(torch.nn.utils.clip_grad_norm_(d.actor.parameters(),10,error_if_nonfinite=True));d.ao.step()
    return dict(d_build_pg_loss=total,d_build_pg_grad=grad,d_build_return_a=ga,d_build_return_b=gb,d_build_return_delta=ga-gb)

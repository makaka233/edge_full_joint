"""Pre-update diagnostic only: actual terminal decision and independent alternatives.
No branch outcome may change live action, replay, or optimizer. All branches use same request trace.
"""
import copy, json
import numpy as np,torch
from sac import batch,distribution
from capacity import cpu

def inspect_terminal(e,d,s,seed,gs,epsilon,st,mask,a,mat,actual_p,lr,rng_for):
    from train import rng_for as rfor
    x=batch([st],d.device);m=d.tensor([mask],torch.bool)
    with torch.no_grad():
        p,_=distribution(d.policy_logits(x,m),m);p=p[0].cpu().numpy();q1=d.q1(x)[0].cpu().numpy();q2=d.q2(x)[0].cpu().numpy();q=np.minimum(q1,q2)
    alt_mask=mask.copy();alt_mask[a]=False
    other=np.flatnonzero(alt_mask)
    rho,load=cpu(e.w,mat,st.demand)
    ret={'gs':gs,'phase':st.phase,'action':int(a),'type':'KEEP' if a==e.KEEP else 'FINISH','actual_p':float(actual_p),'legal':int(mask.sum()),'q1':float(q1[a]),'q2':float(q2[a]),'qmin':float(q[a]),'best_alt_q':float(q[other].max()) if len(other) else None,'mean_alt_q':float((q[other]*p[other]).sum()/p[other].sum()) if len(other) and p[other].sum()>0 else None,'alt_probability':float(p[other].sum()),'rho':rho,'total_load':load,'replicas':int(mat.sum()),'stage_counts':mat.sum(1).tolist(),'branches':[]}
    if len(other):
        cond=p[other].astype(float);cond/=cond.sum()
        kl=float(np.sum(cond*np.log(np.maximum(cond*len(other),1e-300))))
        vcont=float(np.dot(cond,q[other])-d.alpha*kl)
        gap=float(q[a]-vcont);ref=-np.log(len(other));z=float(np.log(float(p[a])/(1-float(p[a]))))
        target_logit=float(ref+gap/d.alpha)
        ret.update(conditional_kl=kl,conditional_soft_value=vcont,soft_q_gap=gap,actor_logit=z,reference_logit=float(ref),optimal_logit=target_logit,critic_implied_p=float(1/(1+np.exp(-np.clip(target_logit,-700,700)))),binary_gradient=float(p[a]*(1-p[a])*(d.alpha*(z-ref)-gap)))
    phi=float(d.potential(x)[0]);ret['potential']=phi
    qseed=int(rfor(seed,90,gs).integers(0,2**31-1))
    def physical(label,z,matrix,subsequent_entropy=0.,first_cond_entropy=0.):
        info,_=z.physical(matrix,s.scheduler(rfor(seed,30,gs),epsilon),qseed,gs,False)
        rr,ll=cpu(z.w,matrix,st.demand)
        old=z.old; spec=z.spec;cap=z.capacity;c=z.cost
        hold=c['holding_weight']*float((matrix.T@spec/cap).sum())/z.N*c['resource_normalizer'];add=matrix&~old;rem=old&~matrix
        switch=c['switch_weight']*float((add.T@spec/cap).sum()+c['removal_ratio']*(rem.T@spec/cap).sum())/z.N*c['resource_normalizer']
        delay=c['delay_weight']*info['penalized_delay']*(info['n']/(z.windows*z.window))/c['reference_rate']
        raw=-(delay+hold+switch)
        nxt=copy.deepcopy(z);nxt.old=matrix.copy();nxt.draft[:]=False;nxt.phase=0;nxt._mask=None;nxt.period+=1;nxt.w.advance()
        nx=batch([nxt.state()],d.device);nm=d.tensor([nxt.legal()],torch.bool)
        with torch.no_grad():v=float(d.next_value(nx,nm)[0]);pn=float(d.potential(nx)[0])
        target=.1*raw+z.c.gamma_d*(v+pn)-phi-subsequent_entropy
        ret['branches'].append(dict(label=label,replicas=int(matrix.sum()),rho=rr,load=ll,n=info['n'],failed=info['failed'],e2e=info['penalized_delay'],reward=raw,trace=info['trace_sha256'],bootstrap_v=v,initial_action_q_return=target,conditional_soft_return=target-first_cond_entropy,subsequent_entropy=subsequent_entropy,first_cond_entropy=first_cond_entropy))
    physical('actual',e,mat)
    # Same deployment and actual cohort, isolate scheduler behavior/exploration.
    from edge_rl.physics import HeuristicScheduler
    for name,scheduler in [('greedy',s.scheduler(rfor(seed,30,gs),0.)),('heuristic',HeuristicScheduler(e.w))]:
        info,_=e.physical(mat,scheduler,qseed,gs,False)
        ret[name]=dict(n=info['n'],failed=info['failed'],e2e=info['penalized_delay'],mean_completed=info['mean_completed'],p95=info['p95_completed'],trace=info['trace_sha256'])
    # Per-window actual cohort workload, versus the revealed expected rates.
    traces,_=e.trace(qseed);windows=[]
    for b,arr in traces:
        demand=np.zeros((e.N,len(e.c.services)))
        for req in arr:
            if b<=req.time<b+e.window:demand[req.access,req.service]+=1/e.window
        wrho,wload=cpu(e.w,mat,demand)
        windows.append(dict(n=int(round(demand.sum()*e.window)),rho=wrho,load=wload,rate=float(demand.sum())))
    ret['realized_cpu_windows']=windows
    # First alternative is conditioned on not stopping / not keeping. Later actions are the unchanged policy.
    if len(other):
      for rep in range(4):
        z=copy.deepcopy(e);z.old=st.old.copy();z.draft=st.draft.copy();z.phase=st.phase;z._mask=None
        rng=rfor(seed,180,gs,rep)
        probs=p[other].astype(float);probs/=probs.sum();ii=int(rng.choice(len(other),p=probs));act=int(other[ii]);matrix=z.apply(act)
        first_entropy=float(d.alpha*np.log(probs[ii]*len(other)));subsequent=0.
        while matrix is None:
            ss=z.state();mm=z.legal();act,lr,_=d.act(ss,mm,rng,False);matrix=z.apply(act);subsequent+=d.alpha*lr
        physical('continue_'+str(rep),z,matrix,subsequent,first_entropy)
    return ret

"""Training-only paired soft-return advantage for KEEP/FINISH.
One counterfactual physical interaction at most per GS, after actual action chosen.
No branch is used to select or replace a live deployment; no extra D/S replay.
"""
from dataclasses import dataclass
import copy,hashlib
import numpy as np, torch
from torch import nn
from torch.nn import functional as F

@dataclass
class BinaryPair:
    state: object
    delta: float
    kind: str
    gs: int

class BinaryAdvantage(nn.Module):
    def __init__(self,w):
        super().__init__();K,N=w.c.k,w.c.n;self.K=K;self.N=N
        self.register_buffer('cpu_cap',torch.tensor(w.nodes[:,0],dtype=torch.float32));self.register_buffer('work',torch.tensor(w.stages[:,0],dtype=torch.float32));self.register_buffer('svc',torch.tensor(w.stage_service,dtype=torch.long));self.register_buffer('isolated',torch.tensor(w.stages[:,1,None,None]*w.path_seconds_per_mbit[None,:,:]+w.stages[:,0,None,None]/w.nodes[None,None,:,0],dtype=torch.float32));self.register_buffer('cap',torch.tensor(w.nodes[:,1:3],dtype=torch.float32));self.register_buffer('use',torch.tensor(w.stages[:,3:5],dtype=torch.float32))
        C=w.context_dim+3;self.mlp=nn.Sequential(nn.Linear(C+4*K+N+9,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2));nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
    def forward(self,x):
        old,draft,dem,ctx=x;phase=ctx[:,-3:].argmax(-1);m=torch.where((phase==0)[:,None,None],old,draft)
        need=dem[:,:,self.svc].sum(1)*self.work;cov=m@self.cpu_cap;count=m.sum(-1);ratio=need/cov.clamp_min(1.)
        rho=(m*ratio[:,:,None]).sum(1);cost=self.isolated[None].masked_fill(~m.bool()[:,:,None,:],float('inf')).min(-1).values;cost=torch.where(m.any(-1)[:,:,None],cost,self.isolated.max(-1).values[None]);by=dem[:,:,self.svc].transpose(1,2);quality=(cost*by).sum(-1)/by.sum(-1).clamp_min(1e-6)
        usage=(m.transpose(1,2)@self.use)/self.cap
        agg=torch.stack([rho.max(-1).values,rho.mean(-1),ratio.max(-1).values,ratio.mean(-1),count.min(-1).values/self.N,count.sum(-1)/(self.K*self.N),need.sum(-1)/self.cpu_cap.sum(),usage[:,:,0].mean(1),usage[:,:,1].mean(1)],-1)
        features=torch.cat([ctx,torch.log1p(need),count/self.N,torch.log1p(ratio),torch.log1p(quality),torch.log1p(rho),torch.log1p(agg)],-1)
        return self.mlp(features).gather(1,(phase==2).long()[:,None]).squeeze(1)

def project_delta(logits,delta,mask,phase,alpha,strength):
    A=logits.shape[-1];idx=torch.arange(A,device=logits.device)[None,:]
    macro=(phase==0)&mask[:,A-3]&mask[:,A-2];finish=(phase==2)&mask[:,A-1]&mask[:,:A-3].any(-1);use=macro|finish
    if not use.any():return logits
    l=logits[use];special=torch.where(macro[use],torch.full_like(macro[use],A-3,dtype=torch.long),torch.full_like(macro[use],A-1,dtype=torch.long));rest=mask[use]&(idx!=special[:,None]);n=rest.sum(-1).float()
    partition=torch.logsumexp(l.masked_fill(~rest,-1e9),-1);sl=l.gather(1,special[:,None]).squeeze(1);z=sl-partition;aa=alpha if not torch.is_tensor(alpha) else alpha.reshape(-1)[use]
    zstar=-torch.log(n)+delta[use]/aa;result=logits.clone();result[use]=torch.where(idx==special[:,None],(sl+(strength if not torch.is_tensor(strength) else strength.reshape(-1)[use])*(zstar-z))[:,None],l);return result

def choose_pair(e,path,st,mask,a,lr,seed,gs,rng_for,strategy="geometric"):
    """Choose a training-only paired state before physical outcome is observed.
    Geometric: historical round-14 pre-drawn rank, no fallback.
    Stratified: pre-drawn rank from three bands, with actual terminal FINISH
    fallback when not reached and ADD remains legal. No reward-based selection.
    Macro selection remains identical; all choices use at most one extra probe.
    """
    rows=[(i,t.s,t.mask,t.a,t.logratio) for i,t in enumerate(path)]+[(len(path),st,mask,a,lr)]
    r=rng_for(seed,193,gs);u=float(r.random())
    if strategy=="geometric":index=int(r.geometric(.04)-1)
    elif strategy in ("stratified","balanced"):
        # Rank is drawn before looking at the trajectory outcome. Equal mass on
        # three rank bands; these are probe indices, never an action bound.
        lo,hi=((0,40),(40,100),(100,160))[int(r.integers(3))];index=int(r.integers(lo,hi))
    else:raise ValueError(strategy)
    if rows[0][2][e.KEEP] and (u<.5 if strategy=="balanced" else (rows[0][3]==e.KEEP or u<.25)):return rows[0]
    eligible=[t for t in rows if t[1].phase==2 and t[2][e.FINISH] and t[2][:e.K*e.N].any()]
    if index<len(eligible):return eligible[index]
    if strategy in ("stratified","balanced") and eligible and eligible[-1][3]==e.FINISH:
        # Only the actually visited terminal state, conditional on FINISH.
        # Its physical outcome is not observed at selection time. The other
        # arm is freshly sampled from the conditional continuation policy.
        return eligible[-1]
    if strategy=="balanced" and rows[0][2][e.KEEP]:return rows[0]
    return None

def paired_experience(pre_env,d,s,path,actual_info,chosen,seed,gs,epsilon,reward_scale,rng_for,return_path=False):
    from sac import batch,distribution,Transition
    i,st,mask,actual_action,_=chosen;special=pre_env.KEEP if st.phase==0 else pre_env.FINISH;kind='keep' if st.phase==0 else 'finish'
    future_origin=copy.deepcopy(pre_env) if (getattr(d,"macro_pair_horizon",getattr(d,"pair_horizon",1)) if st.phase==0 else getattr(d,"pair_horizon",1))==2 else None
    z=pre_env;z.old=st.old.copy();z.draft=st.draft.copy();z.phase=st.phase;z._mask=None;assert np.array_equal(z.legal(),mask)
    rest=mask.copy();rest[special]=False;assert rest.any() and mask[special]
    with torch.no_grad():
        xx=batch([st],d.device);mm=d.tensor([mask],torch.bool);logits=d.policy_logits(xx,mm);pp,ll=distribution(logits,mm);cp,cl=distribution(logits,d.tensor([rest],torch.bool));probs=cp[0].cpu().numpy().astype(float);probs/=probs.sum()
    actual_special=actual_action==special;rng=rng_for(seed,194,gs)
    alt_action=int(rng.choice(len(mask),p=probs)) if actual_special else special
    alt=[];mat=None;first=True
    while mat is None:
        sx=z.state();mx=z.legal().copy()
        if first:act=alt_action;logratio=float(ll[0,act]);first=False
        else:act,logratio,_=d.act(sx,mx,rng,False)
        mat=z.apply(act)
        if mat is None:alt.append(Transition(sx,mx,act,0.,z.state(),z.legal().copy(),1.,False,False,logratio))
        else:
            ns,info,_,term,trunc=z.commit(mat,s.scheduler(rng_for(seed,30,gs),epsilon),int(rng_for(seed,90,gs).integers(0,2**31-1)),gs,False)
            alt.append(Transition(sx,mx,act,info['reward']*reward_scale,ns,z.legal().copy(),z.c.gamma_d,term,trunc,logratio))
    assert info['trace_sha256']==actual_info['trace_sha256'] and np.array_equal(alt[-1].ns.demand,path[-1].ns.demand)
    ga=float(d.fresh_targets(path)[i]);gb=float(d.fresh_targets(alt)[0]);alpha=d.alpha_at(st.phase)
    if actual_special:gb-=alpha*float(cl[0,alt_action])
    else:ga-=alpha*float(cl[0,actual_action])
    delta=ga-gb if actual_special else gb-ga
    # Exact soft-return decomposition. First-action entropy is excluded for Q;
    # only the conditional entropy of the first continuation action is included.
    # Potential shaping cancels between alternatives with the same start state.
    def components(seq,begin,first_cond):
        last=seq[-1]
        with torch.no_grad():
            xx=batch([last.ns],d.device);mm=d.tensor([last.nm],torch.bool)
            boot=0. if last.terminated else last.discount*float(d.next_value(xx,mm)[0])
            potential=(0. if last.terminated else last.discount*float(d.potential(xx)[0]))-float(d.potential(batch([seq[begin].s],d.device))[0]) if d.potential_scale else 0.
        entropy=-sum(d.alpha_at(t.s.phase)*t.logratio for t in seq[begin+1:])-first_cond
        return dict(physical=float(last.r),bootstrap=boot,potential=potential,entropy=entropy)
    ca=components(path,i,0. if actual_special else alpha*float(cl[0,actual_action]))
    cb=components(alt,0,alpha*float(cl[0,alt_action]) if actual_special else 0.)
    sign=1. if actual_special else -1.
    parts={k:sign*(ca[k]-cb[k]) for k in ca}
    assert abs(sum(parts.values())-delta)<2e-5,(parts,delta)
    h2={}
    if future_origin is not None and not path[-1].truncated and not path[-1].terminated:
        # Sample new demand innovations from the revealed pre-commit world.
        # The live world's RNG state is never used for these successor rolls.
        from train import rollout
        ea,eb=copy.deepcopy(future_origin),copy.deepcopy(future_origin)
        for env,mat2 in ((ea,path[-1].ns.old),(eb,mat)):
            env.old=mat2.copy();env.draft[:]=False;env.phase=0;env._mask=None;env.period+=1
            env.w.rng=rng_for(seed,196,gs);env.w.advance()
        future_seed=int(rng_for(seed,197,gs).integers(0,2**31-1))
        pa,ia,_,_=rollout(ea,d,s,future_seed,gs+1,epsilon,False,False,reward_scale)
        pb,ib,_,_=rollout(eb,d,s,future_seed,gs+1,epsilon,False,False,reward_scale)
        assert ia['trace_sha256']==ib['trace_sha256'] and ia['demand_hash']==ib['demand_hash']
        def future_terms(p):
            last=p[-1]
            with torch.no_grad():
                v=0. if last.terminated else last.discount*float(d.next_value(batch([last.ns],d.device),d.tensor([last.nm],torch.bool))[0])
            return dict(physical=float(last.r),entropy=-sum(d.alpha_at(t.s.phase)*t.logratio for t in p),bootstrap=v)
        fa,fb=future_terms(pa),future_terms(pb)
        two={k:sign*(fa[k]-fb[k]) for k in fa}
        old_delta=delta
        ga=ga-ca['bootstrap']+z.c.gamma_d*sum(fa.values());gb=gb-cb['bootstrap']+z.c.gamma_d*sum(fb.values());delta=sign*(ga-gb)
        h2=dict(pair_horizon=2,one_period_delta=old_delta,future_components=two,future_actual={k:ia[k] for k in ('reward','failed','n','replicas','trace_sha256','demand_hash')},future_alternative={k:ib[k] for k in ('reward','failed','n','replicas','trace_sha256','demand_hash')},future_seed=future_seed)
        assert abs(delta-(parts['physical']+parts['entropy']+parts['potential']+z.c.gamma_d*sum(two.values())))<2e-5
    pair=BinaryPair(st.clone(),delta,kind,gs)
    def fields(r):return {k:r[k] for k in ('replicas','reward','penalized_delay','failed','n','trace_sha256')}
    rec=dict(gs=gs,kind=kind,path_index=i,actual_special=bool(actual_special),actual_action=int(actual_action),alternative_action=int(alt_action),draft_replicas=int(st.draft.sum()),old_replicas=int(st.old.sum()),actual=fields(actual_info),alternative=fields(info),actual_return=ga,alternative_return=gb,delta=delta,alternative_matrix_hex=np.packbits(mat.ravel()).tobytes().hex(),state_matrix_hex=np.packbits((st.old if st.phase==0 else st.draft).ravel()).tobytes().hex(),next_demand_hash=hashlib.sha256(alt[-1].ns.demand.tobytes()).hexdigest(),delta_components=parts,actual_components=ca,alternative_components=cb,continuation_actions=(len(alt) if actual_special else len(path)-i))
    rec.update(h2);return (pair,rec,alt) if return_path else (pair,rec)

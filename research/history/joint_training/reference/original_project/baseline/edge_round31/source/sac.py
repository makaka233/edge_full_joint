"""Masked discrete relative-entropy SAC on the actual construction MDP.
Uniform legal-action reference: V=E_pi[min Q - alpha log(pi/u)].
The KL regularizer avoids a positive entropy bonus for gratuitously long zero-time paths.
"""
from dataclasses import dataclass
from collections import deque
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from edge_rl.agents import polyak

@dataclass
class Transition:
    s: object
    mask: np.ndarray
    a: int
    r: float
    ns: object
    nm: np.ndarray
    discount: float
    terminated: bool
    truncated: bool
    logratio: float
    hazard: float = 0.0
    partner_version: int = -1
    collection_step: int = -1

def capacity_afterstates(rho,draft,need,cov,cpu_cap):
    """Full capacity-proportional load after each legal ADD; no LP or repair."""
    N=rho.shape[-1]
    curr=need/cov.clamp_min(1e-6)
    after=need[:,:,None]/(cov[:,:,None]+cpu_cap[None,None,:])
    delta=after-curr[:,:,None]
    rho_all=rho[:,None,None,:]+draft[:,:,None,:]*delta[:,:,:,None]
    eye=torch.eye(N,device=rho.device,dtype=rho.dtype)
    return rho_all+((1-draft)*after)[:,:,:,None]*eye[None,None]

class Net(nn.Module):
    def __init__(self,w,h=48,actor=False,afterstate=False):
        super().__init__(); self.K,self.N=w.c.k,w.c.n;K,N=self.K,self.N;self.afterstate=afterstate;self.is_actor=actor
        self.register_buffer('node',torch.tensor(w.nodes[:,:4]/[20,24,160,120],dtype=torch.float32))
        self.register_buffer('spec',torch.tensor(w.stages/[2,4,4,5,40],dtype=torch.float32))
        self.register_buffer('cap',torch.tensor(w.nodes[:,1:3],dtype=torch.float32));self.register_buffer('use',torch.tensor(w.stages[:,3:5],dtype=torch.float32))
        self.register_buffer('adj',torch.tensor(w.adjacency,dtype=torch.float32));self.register_buffer('svc',torch.tensor(w.stage_service,dtype=torch.long))
        self.register_buffer('region',F.one_hot(torch.tensor(w.regions),w.c.regions).float()); self.register_buffer('hops',torch.tensor(w.hops/max(1,w.hops.max()),dtype=torch.float32))
        C=w.context_dim+3;J=len(w.c.services);R=w.c.regions
        self.n1=nn.Linear(2*K+4+2+J+R+C,h);self.n2=nn.Linear(h,h)
        self.k1=nn.Linear(5+2+2+C,h);self.k2=nn.Linear(h,h)
        self.nb=nn.Linear(h,K);self.kb=nn.Linear(h,1);self.special=nn.Sequential(nn.Linear(2*h+C,64),nn.ReLU(),nn.Linear(64,3))
        self.local=nn.Linear(4,1);self.scale=h**-.5;self.shared=bool(getattr(w,"shared_cpu_features",False));self.exact_shared=bool(getattr(w,"exact_cpu_effect",False));self.factorized=actor and bool(getattr(w,"factorized_stage_policy",False));self.finish_features=bool(getattr(w,"finish_features",False));self.separate_finish=actor and bool(getattr(w,"separate_finish",False))
        if afterstate:
            self.register_buffer('isolated',torch.tensor(w.stages[:,1,None,None]*w.path_seconds_per_mbit[None,:,:]+w.stages[:,0,None,None]/w.nodes[None,None,:,0],dtype=torch.float32))
            self.register_buffer('cpu_cap',torch.tensor(w.nodes[:,0],dtype=torch.float32));self.register_buffer('work',torch.tensor(w.stages[:,0],dtype=torch.float32))
            self.action_effect=nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,1))
            self.common=nn.Sequential(nn.Linear(2*h+C+4,64),nn.ReLU(),nn.Linear(64,1))
            if actor:nn.init.normal_(self.action_effect[-1].weight,std=.012);nn.init.zeros_(self.action_effect[-1].bias)
            if self.shared:
                self.shared_effect=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,1))
                self.shared_global=nn.Sequential(nn.Linear(6,32),nn.SiLU(),nn.Linear(32,3))
                for net in (self.shared_effect,self.shared_global):nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        self.stage_blocks=bool(getattr(w,"stage_blocks",False))
        if self.stage_blocks:
            # Fork RNG so the old actions have exactly their old initialization.
            with torch.random.fork_rng(devices=[]):
                self.block_head=nn.Sequential(nn.Linear(h+5,32),nn.SiLU(),nn.Linear(32,1))
                nn.init.zeros_(self.block_head[-1].weight);nn.init.zeros_(self.block_head[-1].bias)
        self.chain_context=bool(getattr(w,"chain_context",False))
        if self.chain_context:
            assert afterstate
            prev=np.arange(K);nxt=np.arange(K);first=np.ones(K,bool);last=np.ones(K,bool)
            for ids in w.c.services:
                for a,b in zip(ids,ids[1:]):prev[b]=a;nxt[a]=b;first[b]=False;last[a]=False
            self.register_buffer('chain_prev',torch.tensor(prev));self.register_buffer('chain_next',torch.tensor(nxt));self.register_buffer('chain_first',torch.tensor(first));self.register_buffer('chain_last',torch.tensor(last));self.register_buffer('chain_path',torch.tensor(w.path_seconds_per_mbit,dtype=torch.float32))
            # Zero initialization preserves the original random policy and all masks.
            with torch.random.fork_rng(devices=[]):
                self.chain_head=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,1));nn.init.zeros_(self.chain_head[-1].weight);nn.init.zeros_(self.chain_head[-1].bias)
        if self.factorized:
            assert afterstate
            rng_state=torch.get_rng_state();self.stage_actor=nn.Sequential(nn.Linear(h+4,32),nn.SiLU(),nn.Linear(32,1));nn.init.zeros_(self.stage_actor[-1].weight);nn.init.zeros_(self.stage_actor[-1].bias);torch.set_rng_state(rng_state)
        self.retain_context=self.factorized and bool(getattr(w,"retain_context",False))
        if self.retain_context:
            with torch.random.fork_rng(devices=[]):
                self.retain_head=nn.Sequential(nn.Linear(h+7,32),nn.SiLU(),nn.Linear(32,1))
                nn.init.zeros_(self.retain_head[-1].weight);nn.init.zeros_(self.retain_head[-1].bias)
        self.count_context=bool(getattr(w,"count_context",False))
        if self.count_context:
            assert afterstate
            with torch.random.fork_rng(devices=[]):
                self.count_stage=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,1))
                self.count_finish=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,1))
                self.count_effect=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,1))
                for net in (self.count_stage,self.count_finish,self.count_effect):
                    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        if self.finish_features:
            assert afterstate
            rng_state=torch.get_rng_state()
            self.finish_head=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.finish_head[-1].weight);nn.init.zeros_(self.finish_head[-1].bias)
            torch.set_rng_state(rng_state)
        if actor:
            for m in [self.nb,self.kb,self.special[-1],self.local]:nn.init.normal_(m.weight,std=.012);nn.init.zeros_(m.bias)
        self.decompose_return=bool(getattr(w,"decompose_return",False)) and not actor
        if self.decompose_return:
            assert afterstate and self.shared
            from layout_reward import LayoutReward
            with torch.random.fork_rng(devices=[]):
                self.layout_reward=LayoutReward(w)
                self.period_stage=nn.Linear(h,1)
                self.period_node=nn.Linear(h,K)
                self.period_effect=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,1))
                self.period_special=nn.Linear(2*h+C,3)
                for layer in (self.period_stage,self.period_node,self.period_effect[-1],self.period_special):
                    nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,b,mask=None,return_immediate=False):
        old,draft,dem,ctx=b;B=len(old);K,N=self.K,self.N
        free=(self.cap[None]-draft.transpose(1,2)@self.use)/self.cap[None]
        h=F.relu(self.n1(torch.cat([old.transpose(1,2),draft.transpose(1,2),self.node[None].expand(B,-1,-1),free,dem,self.region[None].expand(B,-1,-1),ctx[:,None].expand(-1,N,-1)],-1)))
        h=F.relu(h+self.n2(self.adj@h))
        bystage=dem[:,:,self.svc].transpose(1,2);mass=bystage.sum(-1,keepdim=True);weights=bystage/mass.clamp_min(1e-5);distance=weights@self.hops
        k=F.relu(self.k1(torch.cat([self.spec[None].expand(B,-1,-1),old.sum(-1,keepdim=True)/8,draft.sum(-1,keepdim=True)/8,mass/4,weights.max(-1,keepdim=True).values,ctx[:,None].expand(-1,K,-1)],-1)))
        k=F.relu(self.k2(k));pair=torch.einsum('bkh,bnh->bkn',k,h)*self.scale+self.nb(h).transpose(1,2)+self.kb(k)
        # Direct, learned pair features (not a heuristic score or action restriction).
        pair=pair+self.local(torch.stack([bystage,distance,old,draft],-1)).squeeze(-1)
        global_h=torch.cat([h.mean(1),k.mean(1),ctx],-1);special=self.special(global_h)
        if self.afterstate:
            # Exact change in a cheap isolated-stage *descriptor*, not a reward,
            # hard rule or chosen action. The learned critic decides its value.
            cost=self.isolated[None];current=cost.masked_fill(~draft.bool()[:,:,None,:],float('inf')).min(-1).values
            current=torch.where(draft.any(-1)[:,:,None],current,self.isolated.max(-1).values[None])
            gain=((current[:,:,:,None]-cost).clamp_min(0)*weights[:,:,:,None]).sum(2)
            coverage=draft@self.cpu_cap;need=(mass.squeeze(-1)*self.work[None]);ratio=need/(coverage+1.)
            next_ratio=need[:,:,None]/(coverage[:,:,None]+self.cpu_cap[None,None,:]+1.)
            free_after=free[:,None]-self.use[None,:,None,:]/self.cap[None,None]
            # Every candidate's feature is computed, but none is simulated or preselected.
            effect=torch.stack([gain,ratio[:,:,None].expand(-1,-1,N),next_ratio,free_after[...,0],free_after[...,1],old,draft],-1)
            pair=pair+self.action_effect(effect).squeeze(-1)
            if self.chain_context:
                # Stage-conditioned traffic descriptors. Use revealed demand and
                # existing/draft replicas only. Approximate predecessor source
                # distribution by soft routing of access demand to its replicas.
                # No physical simulation, action restriction, or future demand.
                def sources(mat):
                    pred=mat[:,self.chain_prev,:];valid=pred>0
                    # Time-like proxy from access to predecessor, including CPU.
                    pcost=self.isolated[self.chain_prev]
                    logits=-pcost[None]/.1
                    logits=logits.masked_fill(~valid[:,:,None,:],-1e9)
                    p=logits.softmax(-1)*valid[:,:,None,:]
                    flow=(weights[:,:,:,None]*p).sum(2)
                    use=(valid.any(-1)&~self.chain_first[None])[:,:,None]
                    return torch.where(use,flow,weights)
                src=sources(draft);oldsrc=sources(old)
                incoming=(src@self.chain_path)*self.spec[None,:,1,None]*4
                oldin=(oldsrc@self.chain_path)*self.spec[None,:,1,None]*4
                succ=draft[:,self.chain_next,:];osucc=old[:,self.chain_next,:]
                def outgoing(mat):
                    pc=self.chain_path[None,None,:,:].expand(B,K,-1,-1).masked_fill(~mat.bool()[:,:,None,:],1e3).min(-1).values
                    return torch.where((mat.any(-1)&~self.chain_last[None])[:,:,None],pc,torch.zeros_like(pc))*self.spec[None,:,2,None]*4
                out=outgoing(succ);oldout=outgoing(osucc)
                fchain=torch.stack([src,oldsrc,incoming,oldin,out,oldout,bystage,need[:,:,None].expand(-1,-1,N)/10,old],-1)
                pair=pair+self.chain_head(fchain).squeeze(-1)
            if self.shared:
                # Capacity-proportional work allocation proxy. No routing/LP/repair;
                # uncovered stage demand is explicitly tracked, not silently ignored.
                def load(mat):
                    cov=mat@self.cpu_cap
                    per=need/cov.clamp_min(1e-6)
                    rho=(mat*per[:,:,None]).sum(1)
                    missing=(need*(cov<=0)).sum(-1)/self.cpu_cap.sum()
                    return cov,rho,missing
                cov,rho,missing=load(draft);_,oldrho,oldmissing=load(old)
                curr=need/cov.clamp_min(1e-6)
                after=need[:,:,None]/(cov[:,:,None]+self.cpu_cap[None,None,:])
                node_after=rho[:,None,:]+(1-draft)*after
                # For legal ADD: new node's load rises; existing replica loads fall.
                if self.exact_shared:
                    # Full capacity-proportional afterstate: reduce load on every
                    # existing replica, then add load on the candidate node.
                    # The mask remains exclusively physical; no repair or choice.
                    rho_all=capacity_afterstates(rho,draft,need,cov,self.cpu_cap)
                    before_excess=(rho-.8).clamp_min(0).sum(-1)[:,None,None]
                    delta_excess=before_excess-(rho_all-.8).clamp_min(0).sum(-1)
                    features=torch.stack([rho[:,None,:].expand(-1,K,-1),node_after,
                        rho_all.max(-1).values,delta_excess,after],-1)
                else:
                    features=torch.stack([rho[:,None,:].expand(-1,K,-1),node_after,
                        rho.max(-1).values[:,None,None].expand(-1,K,N),
                        after,missing[:,None,None].expand(-1,K,N)],-1)
                pair=pair+self.shared_effect(features).squeeze(-1)
                agg=torch.stack([rho.max(-1).values,rho.mean(-1),missing,oldrho.max(-1).values,oldrho.mean(-1),oldmissing],-1)
                special=special+self.shared_global(agg)
            quality=(current*weights).sum(-1);extra=torch.stack([quality.mean(-1),ratio.mean(-1),draft.sum((1,2))/200.,old.sum((1,2))/200.],-1)
            if self.count_context:
                # No capacity threshold or requested replica target is imposed.
                oldcov=old@self.cpu_cap
                shortage=need/(coverage+1.)
                oldshort=need/(oldcov+1.)
                addgain=shortage[:,:,None]-next_ratio
                fcount=torch.stack([need[:,:,None].expand(-1,-1,N)/10,
                    shortage[:,:,None].expand(-1,-1,N),oldshort[:,:,None].expand(-1,-1,N),
                    next_ratio,addgain,coverage[:,:,None].expand(-1,-1,N)/self.cpu_cap.sum(),
                    draft.sum(-1)[:,:,None].expand(-1,-1,N)/N,
                    free_after[...,0],free_after[...,1]],-1)
                pair=pair+self.count_effect(fcount).squeeze(-1)
                dfrac=draft.sum(-1)/N;ofrac=old.sum(-1)/N
                fstage=torch.stack([need/10,shortage,oldshort,dfrac,ofrac,
                    coverage/self.cpu_cap.sum(),oldcov/self.cpu_cap.sum(),quality,
                    (addgain*(1-draft)).max(-1).values],-1)
                stop=torch.stack([shortage.mean(-1),shortage.max(-1).values,
                    shortage.topk(4,-1).values.mean(-1),oldshort.mean(-1),
                    dfrac.mean(-1),ofrac.mean(-1),quality.mean(-1),
                    (need*shortage).sum(-1)/need.sum(-1).clamp_min(1e-6),
                    free.mean((1,2))],-1)
                z=self.count_finish(stop).squeeze(-1)
                special=special+torch.stack([torch.zeros_like(z),torch.zeros_like(z),z],-1)
            if self.finish_features:
                # Descriptors of the current FINISH afterstate, never a feasibility mask.
                need_exact=need/coverage.clamp_min(1e-6)
                rr=(draft*need_exact[:,:,None]).sum(1)
                counts=draft.sum(-1)
                ffinish=torch.stack([ratio.max(-1).values,ratio.mean(-1),ratio.topk(4,-1).values.mean(-1),
                    rr.max(-1).values,rr.mean(-1),(rr-1).clamp_min(0).mean(-1),
                    counts.min(-1).values/N,counts.sum(-1)/(K*N),quality.max(-1).values,quality.mean(-1)],-1)
                z=self.finish_head(ffinish).squeeze(-1)
                special=special+torch.stack([torch.zeros_like(z),torch.zeros_like(z),z],-1)
            if not self.is_actor:
                v=self.common(torch.cat([global_h,extra],-1));pair=pair+v[:,:,None];special=special+v
        if self.factorized:
            assert mask is not None
            # p(stage,node) = p(stage) p(node | stage); the log-count term
            # keeps the initial reference uniform over individual legal ADDs.
            # Every legal ADD and FINISH remains available with no quotas.
            valid=mask[:,:K*N].reshape(B,K,N);counts=valid.sum(-1).clamp_min(1)
            conditional=pair.masked_fill(~valid,-1e9)
            if self.retain_context:
                retain=valid & old.bool(); new=valid & ~old.bool()
                fgroup=torch.stack([ratio,quality,mass.squeeze(-1)/4,draft.sum(-1)/N,
                    old.sum(-1)/N,retain.sum(-1)/N,new.sum(-1)/N],-1)
                bias=self.retain_head(torch.cat([k,fgroup],-1)).squeeze(-1)
                conditional=conditional+torch.where(retain,bias[:,:,None],torch.zeros_like(conditional))
            conditional=conditional-torch.logsumexp(conditional,-1,keepdim=True)
            f=torch.stack([ratio,quality,mass.squeeze(-1)/4,draft.sum(-1)/N],-1)
            stage=self.stage_actor(torch.cat([k,f],-1)).squeeze(-1)+counts.float().log()
            if self.count_context:stage=stage+self.count_stage(fstage).squeeze(-1)
            pair=conditional+stage[:,:,None]
            if self.separate_finish:
                # A Bernoulli stop/continue marginal with the same uniform-legal
                # reference as the original joint SAC. The stage/node policy
                # is conditional on continuing; its logits cannot inflate or
                # deflate the FINISH odds through their aggregate partition.
                flat=pair.flatten(1);legal=mask[:,:K*N];nlegal=legal.sum(-1,keepdim=True)
                partition=torch.logsumexp(flat.masked_fill(~legal,-1e9),-1,keepdim=True)
                normalized=flat-partition+nlegal.clamp_min(1).float().log()
                use=mask[:,-1:] & (nlegal>0)
                pair=torch.where(use,normalized,flat).reshape(B,K,N)
        immediate=None
        if self.decompose_return:
            # Direct physical-period return conditional on the chosen action and
            # subsequent construction behavior. No future bootstrap in its label.
            im_pair=self.period_stage(k)+self.period_node(h).transpose(1,2)+self.period_effect(torch.cat([effect,features],-1)).squeeze(-1)
            im_special=self.period_special(global_h)
            keep=self.layout_reward(old,old,dem);finish=self.layout_reward(old,draft,dem)
            im_special=torch.stack([keep,im_special[:,1],finish],-1)
        if self.stage_blocks:
            remaining=old.bool() & ~draft.bool();count=remaining.sum(-1).clamp_min(1)
            pooled=(pair*remaining).sum(-1)/count
            feat=torch.stack([remaining.sum(-1)/N,old.sum(-1)/N,draft.sum(-1)/N,mass.squeeze(-1)/4,(free[:,None,:,0]*remaining).sum(-1)/count],-1)
            block=pooled+self.block_head(torch.cat([k,feat],-1)).squeeze(-1)
            residual=torch.cat([pair.flatten(1),block,special],-1)
            if self.decompose_return:
                im_block=(im_pair*remaining).sum(-1)/count
                immediate=torch.cat([im_pair.flatten(1),im_block,im_special],-1)
        else:
            residual=torch.cat([pair.flatten(1),special],-1)
            if self.decompose_return:immediate=torch.cat([im_pair.flatten(1),im_special],-1)
        if immediate is not None:
            total=residual+immediate.detach()
            return (total,immediate) if return_immediate else total
        return residual

def batch(states,device='cpu'):
    def t(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32,device=device)
    return t([s.old for s in states]),t([s.draft for s in states]),t([s.demand for s in states]),t([np.r_[s.context,np.eye(3)[s.phase]] for s in states])

def distribution(logits,m):
    assert m.any(-1).all();lp=F.log_softmax(logits.masked_fill(~m,-1e9),-1);p=lp.exp()*m;lp=lp.masked_fill(~m,0.)
    logratio=(lp+torch.log(m.sum(-1,keepdim=True).float())).masked_fill(~m,0.)
    return p,logratio

def soft_value(logits,q1,q2,m,alpha):
    p,lr=distribution(logits,m);return (p*(torch.minimum(q1,q2)-alpha*lr)).sum(-1)

def target(r,d,terminal,v):return r+d*(~terminal).float()*v

def binary_marginal_correction(logits,q,mask,phase,alpha,damping,scope="both"):
    """Damped Bernoulli Fisher preconditioning of SAC's existing marginal objective.

    At a binary decision (KEEP vs BUILD, or FINISH vs any legal ADD),
    the conditional continuation value includes its original relative entropy.
    The correction changes gradient conditioning, not the soft policy optimum.
    No legal action is removed; the critic, reward and behavior remain unchanged.
    """
    A=logits.shape[-1];idx=torch.arange(A,device=logits.device)[None,:]
    macro=(phase==0)&mask[:,A-3]&mask[:,A-2] if scope=="both" else torch.zeros_like(phase,dtype=torch.bool)
    finish=(phase==2)&mask[:,A-1]&mask[:,:A-3].any(-1)
    use=macro|finish
    if not use.any():return logits.sum()*0.,{}
    l=logits[use];qq=q[use];mm=mask[use];is_macro=macro[use]
    special=torch.where(is_macro,torch.full_like(is_macro,A-3,dtype=torch.long),torch.full_like(is_macro,A-1,dtype=torch.long))
    rest=mm & (idx!=special[:,None]);n=rest.sum(-1).float();assert (n>0).all()
    z=l.gather(1,special[:,None]).squeeze(1)-torch.logsumexp(l.masked_fill(~rest,-1e9),-1)
    prob=torch.sigmoid(z);logp=F.logsigmoid(z);lognot=F.logsigmoid(-z)
    aa=alpha if not torch.is_tensor(alpha) else alpha.reshape(-1)[use]
    logu=-torch.log1p(n);logu_rest=torch.log(n)-torch.log1p(n)
    with torch.no_grad():
        cp,clr=distribution(l.detach(),rest)
        vc=(cp*(qq.detach()- (aa if not torch.is_tensor(aa) else aa[:,None])*clr)).sum(-1)
        qs=qq.gather(1,special[:,None]).squeeze(1).detach()
        gain=qs-vc
        # Damping limits gradient amplification, not action probabilities.
        weight=(1./(prob.detach()*(1-prob.detach())+damping)-1.)
        target_logit=-torch.log(n)+gain/aa
    binary=prob*(aa*(logp-logu)-qs)+(1-prob)*(aa*(lognot-logu_rest)-vc)
    loss=(weight*binary).sum()/len(logits)
    stats=dict(d_binary_p=float(prob.mean().detach()),d_binary_gap=float(gain.mean()),d_binary_fisher_weight=float((weight+1).mean()),d_binary_logit_error=float((z.detach()-target_logit).abs().mean()),d_binary_states=int(use.sum()))
    return loss,stats

def binary_soft_projection(logits,q,mask,phase,alpha,strength):
    """Partial entropy-regularized policy improvement of only binary marginals.
    q is detached by caller. The conditional distribution of ADDs is unchanged.
    The resulting stochastic policy is used in rollout, evaluation, TD and MC.
    This is a log-odds interpolation toward the exact soft binary optimum, not
    an action filter, simulator lookahead or post-hoc choice.
    """
    A=logits.shape[-1];idx=torch.arange(A,device=logits.device)[None,:]
    macro=(phase==0)&mask[:,A-3]&mask[:,A-2]
    finish=(phase==2)&mask[:,A-1]&mask[:,:A-3].any(-1)
    use=macro|finish
    if not use.any():return logits
    l=logits[use];qq=q[use];mm=mask[use]
    special=torch.where(macro[use],torch.full_like(macro[use],A-3,dtype=torch.long),torch.full_like(macro[use],A-1,dtype=torch.long))
    rest=mm & (idx!=special[:,None]);n=rest.sum(-1).float()
    partition=torch.logsumexp(l.masked_fill(~rest,-1e9),-1)
    special_logit=l.gather(1,special[:,None]).squeeze(1);z=special_logit-partition
    cp,clr=distribution(l,rest);aa=alpha if not torch.is_tensor(alpha) else alpha.reshape(-1)[use]
    vc=(cp*(qq-(aa if not torch.is_tensor(aa) else aa[:,None])*clr)).sum(-1)
    qs=qq.gather(1,special[:,None]).squeeze(1)
    zstar=-torch.log(n)+(qs-vc)/aa
    adjusted=special_logit+strength*(zstar-z)
    result=logits.clone();result[use]=torch.where(idx==special[:,None],adjusted[:,None],l)
    return result

def compact_path(path):
    """Share identical immutable snapshots within one physical period.
    No data quantization, transition removal, ordering or sampling changes.
    """
    if not path:return
    anchor=path[0].s;prev=None
    for t in path:
        if prev is not None and prev.ns.phase==t.s.phase and all(np.array_equal(getattr(prev.ns,k),getattr(t.s,k)) for k in ('old','draft','demand','context')) and np.array_equal(prev.nm,t.mask):
            t.s=prev.ns;t.mask=prev.nm
        for st in (t.s,t.ns):
            for key in ('old','demand','context'):
                if np.array_equal(getattr(st,key),getattr(anchor,key)):setattr(st,key,getattr(anchor,key))
        prev=t

def macro_smdp_path(path,agent):
    """Replace only the macro BUILD transition by its completed option return.

    Internal action sequence is unchanged. The macro reward includes the realized
    construction relative-entropy regularization, but no macro-action KL term.
    All internal discounts are one; the final physical transition supplies the
    macro duration and continuation state. Actual physical reward/logging are
    unchanged. This is a semi-Markov option sample, not a counterfactual.
    """
    if len(path)<=1:return path
    first,last=path[0],path[-1]
    assert first.s.phase==0 and last.ns.phase==0 and first.a==len(first.mask)-2
    if first.ns.phase==0:return path  # already converted checkpoint
    assert all(t.discount==1 for t in path[:-1])
    assert all(t.s.phase!=0 for t in path[1:])
    tr=copy.copy(first)
    tr.r=float(sum(t.r-agent.alpha_at(t.s.phase)*t.logratio for t in path[1:]))
    tr.ns,tr.nm,tr.discount,tr.terminated,tr.truncated=last.ns,last.nm,last.discount,last.terminated,last.truncated
    tr.hazard=last.hazard
    return [tr]+list(path[1:])


def migrate_macro_smdp(replay,agent):
    """Convert real historical complete paths, without inventing any reward."""
    converted=0
    for groups in replay.paths:
        path=[t for group in groups for t in group]
        new=macro_smdp_path(path,agent)
        if new[0] is not path[0]:groups[0][0]=new[0];converted+=1
    return converted

class Replay:
    def __init__(self,capacity=192):self.paths=deque(maxlen=capacity);self.probes=deque(maxlen=capacity)
    def add(self,path):
        compact_path(path)
        groups=[[],[],[],[]]
        for tr in path:
            g=0 if tr.s.phase==0 else 1 if tr.s.phase==1 else 3 if tr.a==len(tr.mask)-1 else 2
            groups[g].append(tr)
        self.paths.append([g for g in groups if g])
    def compact(self):
        for groups in self.paths:compact_path([t for g in groups for t in g])
    def add_probe(self,t):
        if not hasattr(self,"probes"):self.probes=deque(maxlen=self.paths.maxlen)
        self.probes.append(t)
    def sample(self,n,rng,probe_fraction=0.):
        out=[]
        probes=getattr(self,"probes",[])
        npb=min(n-1,int(round(n*probe_fraction))) if probes and probe_fraction else 0
        for _ in range(n-npb):
            g=self.paths[int(rng.integers(len(self.paths)))];p=g[int(rng.integers(len(g)))];out.append(p[int(rng.integers(len(p)))])
        for _ in range(npb):out.append(probes[int(rng.integers(len(probes)))])
        return out
    def sample_global_phase(self,n,rng):
        """Choose a nonempty phase, then a stored path group, then transition.

        In complete paired-option replay, each macro state contributes one KEEP
        and one BUILD path. Sampling phase first preserves that macro balance
        instead of giving one-step KEEP paths four times the macro mass.
        No outcome-based sampling and no change to environment/action masks.
        """
        groups=[[],[],[],[]]
        for path in self.paths:
            for g in path:
                t=g[0]
                k=0 if t.s.phase==0 else 1 if t.s.phase==1 else 3 if t.a==len(t.mask)-1 else 2
                groups[k].append(g)
        active=[g for g in groups if g]
        out=[]
        for _ in range(n):
            phase=active[int(rng.integers(len(active)))];g=phase[int(rng.integers(len(phase)))];out.append(g[int(rng.integers(len(g)))])
        return out
    def phase_groups(self):
        groups=[[],[],[],[]];weights=[[],[],[],[]]
        for path in self.paths:
            for group in path:
                t=group[0];k=0 if t.s.phase==0 else 1 if t.s.phase==1 else 3 if t.a==len(t.mask)-1 else 2
                groups[k].append(group);weights[k].append(1./len(path))
        return groups,weights
    def sample_factorial(self,n,rng,phase_mass,macro_balance):
        """2x2 ablation: phase mass versus within-macro path weighting.
        path mass preserves the original path-first marginal exactly;
        global mass gives each nonempty phase equal probability.
        For macro, balance chooses paths uniformly; otherwise original
        inverse-group-count weighting is retained. For nonmacro, the
        global-phase scheme uses uniform path groups (as R30); the
        path scheme retains the original conditional weights. This also
        handles BUILD paths with a missing redundancy phase exactly.
        """
        groups,weights=self.phase_groups();active=[k for k in range(4) if groups[k]]
        masses=np.array([sum(weights[k]) if phase_mass=='path' else 1. for k in active]);masses/=masses.sum()
        probs=[]
        for k in active:
            w=np.ones(len(groups[k])) if (k==0 and macro_balance) or (k!=0 and phase_mass=='global') else np.asarray(weights[k]);probs.append(w/w.sum())
        out=[]
        for _ in range(n):
            j=int(rng.choice(len(active),p=masses));k=active[j];g=groups[k][int(rng.choice(len(groups[k]),p=probs[j]))];out.append(g[int(rng.integers(len(g)))])
        return out
    def __len__(self):return sum(sum(len(g) for g in p) for p in self.paths)

class SAC:
    def __init__(self,w,seed,hidden=48,alpha=.002,lr=.0005,tau=.015,device='cpu',potential_scale=0.,afterstate=False):
        torch.manual_seed(seed);self.device=device;self.alpha=alpha;self.tau=tau;self.construction_alpha=float(getattr(w,"construction_alpha",alpha));self.pair_horizon=int(getattr(w,"pair_horizon",1));self.fresh_weight=float(getattr(w,"fresh_weight",1.));self.critic_mse=bool(getattr(w,"critic_mse",False));self.decompose_return=bool(getattr(w,"decompose_return",False));self.build_pair_pg=bool(getattr(w,"build_pair_pg",False))
        self.actor=Net(w,hidden,True,afterstate).to(device);self.q1=Net(w,hidden,False,afterstate).to(device);self.q2=Net(w,hidden,False,afterstate).to(device)
        self.t1=copy.deepcopy(self.q1).requires_grad_(False);self.t2=copy.deepcopy(self.q2).requires_grad_(False)
        self.ao=torch.optim.Adam(self.actor.parameters(),lr=float(getattr(w,"actor_lr",lr)));self.qo=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.macro_forward_kl=bool(getattr(w,"macro_forward_kl",False));self.binary_damping=float(getattr(w,"binary_actor_damping",0.));self.binary_scope=getattr(w,"binary_actor_scope","both");self.binary_projection=float(getattr(w,"binary_projection",0.));self.binary_projection_warmup=int(getattr(w,"binary_projection_warmup",0));self.binary_advantage=bool(getattr(w,"binary_advantage",False));self.binary_pair_warmup=int(getattr(w,"binary_pair_warmup",16));self.binary_pair_count=0;self.binary_pair_updates=0
        if self.binary_advantage:
            from binary_advantage import BinaryAdvantage
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+77);self.binary_adv=BinaryAdvantage(w).to(device)
            self.binary_adv_opt=torch.optim.Adam(self.binary_adv.parameters(),lr=float(getattr(w,"binary_pair_lr",.001)));self.binary_pairs=deque(maxlen=256)
        self.auto_alpha=bool(getattr(w,"auto_alpha",False))
        if self.auto_alpha:
            self.log_alpha=nn.Parameter(torch.tensor(np.log(alpha),dtype=torch.float32,device=device));self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=float(getattr(w,"alpha_lr",.01)));self.alpha_target_kl=float(getattr(w,"alpha_target_kl",.3))
        self.risk_weight=float(getattr(w,'risk_weight',0.));self.risk_gamma=float(w.c.gamma_d)
        if self.risk_weight:
            self.risk=copy.deepcopy(self.q1);self.risk_target=copy.deepcopy(self.risk).requires_grad_(False)
            self.risk_opt=torch.optim.Adam(self.risk.parameters(),lr=float(getattr(w,'risk_lr',lr)))
        self.updates=0;self.potential_scale=potential_scale;self.cpu_potential=float(getattr(w,"shared_cpu_potential",0.));self.cost_potential=getattr(w,"cost_potential",None)
        if self.cost_potential is not None:assert self.potential_scale and "migration" not in self.cost_potential
        self.pcpu=torch.as_tensor(w.nodes[:,0],dtype=torch.float32,device=device);self.pwork=torch.as_tensor(w.stages[:,0],dtype=torch.float32,device=device)
        # Isolated, AP-to-replica cost surrogate, not an action constraint. For
        # downstream stages AP is only a proxy source; exact physical reward is
        # unchanged. Missing stages use worst one-node cost, not a hard penalty.
        cost=w.stages[:,1,None,None]*w.path_seconds_per_mbit[None,:,:]+w.stages[:,0,None,None]/w.nodes[None,None,:,0]
        self.isolated=torch.as_tensor(cost,dtype=torch.float32,device=device)
        self.service=torch.as_tensor(w.stage_service,dtype=torch.long,device=device)
    @torch.no_grad()
    def potential(self,x):
        old,draft,dem,ctx=x
        if not self.potential_scale:return torch.zeros(len(old),device=self.device)
        # [B,K,source,destination]; min over deployed nodes, max for uncovered.
        cost=self.isolated[None].masked_fill(~draft.bool()[:,:,None,:],float('inf')).min(-1).values
        worst=self.isolated.max(-1).values[None]
        cost=torch.where(draft.any(-1)[:,:,None],cost,worst)
        # Scale consistent with the base reward scaled by 0.1.
        phi=-(cost*dem[:,:,self.service].transpose(1,2)).sum((1,2))/20.
        if self.cpu_potential:
            need=dem[:,:,self.service].sum(1)*self.pwork
            cov=draft@self.pcpu;rho=(draft*(need/cov.clamp_min(1e-6))[:,:,None]).sum(1)
            missing=(need*(cov<=0)).sum(-1)/self.pcpu.sum()
            # Potential difference only, exactly zero at macro boundaries.
            # Shared load proxy, never a mask, repair, or terminal reward.
            excess=(rho-0.8).clamp_min(0)
            phi=phi-self.cpu_potential/self.potential_scale*(excess.max(-1).values+excess.mean(-1)+missing)
        phi=self.potential_scale*phi
        if self.cost_potential is not None:
            phi=phi+self.resource_cost_potential(x)
        return phi*(1-ctx[:,-3])
    def resource_cost_potential(self,x):
        """Exact construction cost difference vs removing old layout.
        At macro boundaries zero. ADDs pay real hold + added replica cost,
        or receive avoided removal cost. Final physical reward cancels the
        potential, so the total original objective is unchanged.
        """
        old,draft,_,ctx=x;p=self.cost_potential
        def resource(mat):return (mat.transpose(1,2)@self.actor.use/self.actor.cap).sum((1,2))*p['resource_normalizer']/self.actor.N
        delta=p['holding_weight']*resource(draft)+p['switch_weight']*(resource(draft*(1-old))-p['removal_ratio']*resource(draft*old))
        return -p['scale']*delta*(1-ctx[:,-3])
    def tensor(self,v,dtype=torch.float32):return torch.as_tensor(np.asarray(v),dtype=dtype,device=self.device)
    def projection_strength(self):
        if self.binary_advantage:return self.binary_projection*min(1.,self.binary_pair_count/max(1,self.binary_pair_warmup))
        return self.binary_projection*(min(1.,self.updates/self.binary_projection_warmup) if self.binary_projection_warmup else 1.)
    def policy_logits(self,x,m):
        logits=self.actor(x,m);strength=self.projection_strength()
        if strength:
            phase=x[3][:,-3:].argmax(-1)
            A=m.shape[-1];eligible=((phase==0)&m[:,A-3]&m[:,A-2])|((phase==2)&m[:,A-1]&m[:,:A-3].any(-1))
            if eligible.any():
                if self.binary_advantage:
                    from binary_advantage import project_delta
                    with torch.no_grad():delta=self.binary_adv(x)
                    logits=project_delta(logits,delta,m,phase,self.alpha_for(x[3]),self.proximal_strength(x[3],strength))
                else:
                    with torch.no_grad():q=torch.minimum(self.q1(x),self.q2(x))
                    logits=binary_soft_projection(logits,q,m,phase,self.alpha_for(x[3]),strength)
        return logits
    @torch.no_grad()
    def act(self,s,m,rng,deterministic=False,behavior_epsilon=0.):
        x=batch([s],self.device);mask=self.tensor([m],torch.bool);logits=self.policy_logits(x,mask);p,lr=distribution(logits,mask)
        p=p[0].cpu().numpy().astype(float);p/=p.sum()
        # Optional macro-gate behavior mixture. Q is conditioned on the actual
        # first action; all subsequent construction actions remain on-policy.
        behavior=(1-behavior_epsilon)*p+behavior_epsilon*m/m.sum() if behavior_epsilon else p
        a=int(logits.masked_fill(~mask,-1e9).argmax().item()) if deterministic else int(rng.choice(len(m),p=behavior))
        return a,float(lr[0,a]),float(p[a])
    def proximal_strength(self,ctx,strength):
        # Same proximal penalty in both time scales; prevents division by a tiny temperature.
        if self.construction_alpha==self.alpha:return strength
        a=self.alpha_for(ctx);prox=self.alpha*(1-strength)/strength
        return a/(a+prox)
    def alpha_at(self,phase):
        return float(self.log_alpha.detach().exp()) if self.auto_alpha and phase==2 else (self.alpha if phase==0 else self.construction_alpha)
    def alpha_for(self,ctx):
        if not self.auto_alpha:
            if self.construction_alpha==self.alpha:return self.alpha
            return torch.where(ctx[:,-3:-2]>0.5,self.alpha,self.construction_alpha)
        # Only redundant construction (phase 2) has adaptive KL temperature.
        return torch.where(ctx[:,-1:]>0.5,self.log_alpha.detach().exp(),self.alpha)
    def next_value(self,x,m):
        return soft_value(self.policy_logits(x,m),self.t1(x),self.t2(x),m,self.alpha_for(x[3]))
    @torch.no_grad()
    def fresh_targets(self,path):
        # Fresh on-policy soft Monte Carlo target: bootstrap after real commit,
        # not after an optimizer update block. Every internal transition has beta=1.
        last=path[-1];x=batch([last.ns],self.device);m=self.tensor([last.nm],torch.bool)
        v=float(self.next_value(x,m)[0])
        def a_at(t):
            return self.alpha_at(t.s.phase)
        if not self.potential_scale:
            z=last.r+(0 if last.terminated else last.discount*v);out=[z]
            for i in range(len(path)-2,-1,-1):
                z=path[i].r+path[i].discount*(z-a_at(path[i+1])*path[i+1].logratio);out.append(z)
            return list(reversed(out))
        ph=self.potential(batch([t.s for t in path],self.device)).cpu().numpy();pn=float(self.potential(x)[0]);z=last.r+last.discount*(0 if last.terminated else pn)-ph[-1]+(0 if last.terminated else last.discount*v)
        out=[z]
        for i in range(len(path)-2,-1,-1):
            z=path[i].r+path[i].discount*ph[i+1]-ph[i]+path[i].discount*(z-a_at(path[i+1])*path[i+1].logratio);out.append(z)
        return list(reversed(out))
    def risk_values(self,net,x):
        # Bounded normalized discounted hazard value, not a feasibility test.
        return torch.sigmoid(net(x)-3.)
    @torch.no_grad()
    def risk_value(self,x,m):
        p,_=distribution(self.policy_logits(x,m),m)
        return (p*self.risk_values(self.risk_target,x)).sum(-1)
    @torch.no_grad()
    def risk_fresh_targets(self,path):
        last=path[-1];x=batch([last.ns],self.device);m=self.tensor([last.nm],torch.bool)
        z=(1-self.risk_gamma)*float(last.hazard)+(0. if last.terminated else last.discount*float(self.risk_value(x,m)[0]))
        out=[z]
        for t in reversed(path[:-1]):
            z=(1-self.risk_gamma)*float(t.hazard)+t.discount*z;out.append(z)
        return list(reversed(out))
    def update(self,items,fresh,fy,rfy=None,period_r=None,layout_pair=None):
        x=batch([t.s for t in items],self.device);nx=batch([t.ns for t in items],self.device);m=self.tensor([t.mask for t in items],torch.bool);nm=self.tensor([t.nm for t in items],torch.bool)
        a=self.tensor([t.a for t in items],torch.long);r=self.tensor([t.r for t in items]);d=self.tensor([t.discount for t in items]);done=self.tensor([t.terminated for t in items],torch.bool)
        with torch.no_grad():
            shaped=r+d*(~done).float()*self.potential(nx)-self.potential(x) if self.potential_scale else r
            y=target(shaped,d,done,self.next_value(nx,nm))
        fx=batch([t.s for t in fresh],self.device);fa=self.tensor([t.a for t in fresh],torch.long);fyt=self.tensor(fy)
        imloss=None
        if self.decompose_return:
            assert period_r is not None
            q1,im1=self.q1(x,return_immediate=True);q2,im2=self.q2(x,return_immediate=True)
            fq1,fim1=self.q1(fx,return_immediate=True);fq2,fim2=self.q2(fx,return_immediate=True)
            # Fresh prefixes have real on-rollout terminal labels, not surrogate targets.
            direct=torch.full_like(fyt,float(period_r));terminal=nx[3][:,-3]>.5
            imloss=F.mse_loss(fim1.gather(1,fa[:,None]).squeeze(1),direct)+F.mse_loss(fim2.gather(1,fa[:,None]).squeeze(1),direct)
            # Old complete-layout transitions have known immediate physical rewards.
            # Historical incomplete prefixes would require off-policy correction, so excluded here.
            if terminal.any():
                imloss=imloss+F.mse_loss(im1.gather(1,a[:,None]).squeeze(1)[terminal],r[terminal])+F.mse_loss(im2.gather(1,a[:,None]).squeeze(1)[terminal],r[terminal])
            q1=q1.gather(1,a[:,None]).squeeze(1);q2=q2.gather(1,a[:,None]).squeeze(1)
            fq1=fq1.gather(1,fa[:,None]).squeeze(1);fq2=fq2.gather(1,fa[:,None]).squeeze(1)
        else:
            q1=self.q1(x).gather(1,a[:,None]).squeeze(1);q2=self.q2(x).gather(1,a[:,None]).squeeze(1)
            fq1=self.q1(fx).gather(1,fa[:,None]).squeeze(1);fq2=self.q2(fx).gather(1,fa[:,None]).squeeze(1)
        criterion=F.mse_loss if self.critic_mse else F.smooth_l1_loss
        td=criterion(q1,y)+criterion(q2,y);mc=criterion(fq1,fyt)+criterion(fq2,fyt);loss=td+self.fresh_weight*mc
        if imloss is not None:loss=loss+imloss
        pairloss=None
        if layout_pair is not None:
            # Same demand / exogenous trace / partner, actual measured rewards.
            # Pair loss targets relative layout quality; it cannot select actions.
            states,labels=layout_pair;px=batch(states,self.device);py=self.tensor(labels)
            pred1=self.q1.layout_reward(px[0],px[1],px[2]);pred2=self.q2.layout_reward(px[0],px[1],px[2])
            pairloss=.5*(F.mse_loss(pred1,py)+F.mse_loss(pred2,py))+(pred1[0]-pred1[1]-py[0]+py[1]).square()+(pred2[0]-pred2[1]-py[0]+py[1]).square()
            loss=loss+pairloss
        self.qo.zero_grad();loss.backward();qg=float(nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),10,error_if_nonfinite=True));self.qo.step()
        risk_stats={}
        if self.risk_weight:
            assert rfy is not None
            with torch.no_grad():
                hz=self.tensor([getattr(t,'hazard',0.) for t in items]);ry=(1-self.risk_gamma)*hz+d*(~done).float()*self.risk_value(nx,nm)
                ry=ry.clamp(0.,1.)
            rp=self.risk_values(self.risk,x).gather(1,a[:,None]).squeeze(1)
            rfp=self.risk_values(self.risk,fx).gather(1,fa[:,None]).squeeze(1)
            rl=F.binary_cross_entropy(rp,ry)+F.binary_cross_entropy(rfp,self.tensor(rfy).clamp(0.,1.))
            self.risk_opt.zero_grad();rl.backward();rg=float(nn.utils.clip_grad_norm_(self.risk.parameters(),10,error_if_nonfinite=True));self.risk_opt.step();polyak(self.risk_target,self.risk,self.tau)
            risk_stats=dict(d_risk_loss=float(rl.detach()),d_risk_grad=rg,d_risk_target=float(ry.mean()),d_risk_sample=float(rp.mean().detach()))
        logits=self.policy_logits(x,m);p,lr=distribution(logits,m)
        with torch.no_grad():
            q=torch.minimum(self.q1(x),self.q2(x))
            if self.risk_weight:q=q-self.risk_weight*self.risk_values(self.risk,x)
        per_state=(p*(self.alpha_for(x[3])*lr-q)).sum(-1);al=per_state.mean();bstats={}
        if self.build_pair_pg:
            # Construction actor is updated by actual fresh trajectory pairs.
            # The existing macro SAC actor objective and replay Q losses remain.
            macro=x[3][:,-3]>.5
            al=(per_state*macro.float()).sum()/len(per_state)
        if self.macro_forward_kl:
            # Exact soft policy improvement target over the legal macro actions.
            # Forward KL avoids p(a) suppression in the reverse-KL gradient.
            # Construction rows retain the original SAC objective unchanged.
            macro=(x[3][:,-3]>.5)&(m.sum(-1)>1)
            if macro.any():
                with torch.no_grad():
                    target_p=torch.softmax((q[macro]/self.alpha).masked_fill(~m[macro],-1e9),-1)
                logp=torch.log_softmax(logits[macro].masked_fill(~m[macro],-1e9),-1)
                ce=-(target_p*logp).sum(-1)
                # Use dimensionless cross entropy on macro rows to give their
                # policy fitting gradients a nonvanishing scale. No reward change.
                al=(per_state[~macro].sum()+ce.sum())/len(per_state)
                bstats=dict(d_macro_ce=float(ce.mean().detach()),d_macro_fit_rows=int(macro.sum()),d_macro_target_keep=float(target_p[:,-3].mean()),d_macro_actor_keep=float(p[macro,-3].mean().detach()))
        if self.binary_damping:
            correction,bstats=binary_marginal_correction(logits,q,m,x[3][:,-3:].argmax(-1),self.alpha_for(x[3]),self.binary_damping,self.binary_scope)
            al=al+correction
        self.ao.zero_grad();al.backward();ag=float(nn.utils.clip_grad_norm_(self.actor.parameters(),10,error_if_nonfinite=True));self.ao.step()
        alpha_stats={}
        if self.auto_alpha:
            phase2=x[3][:,-1]>0.5
            if phase2.any():
                kl=(p.detach()*lr.detach()).sum(-1)[phase2].mean()
                # Dual descent: raise temperature when KL exceeds target.
                alpha_loss=self.log_alpha*(self.alpha_target_kl-kl)
                self.alpha_opt.zero_grad();alpha_loss.backward();self.alpha_opt.step()
                alpha_stats=dict(d_phase2_kl=float(kl),d_alpha_phase2=float(self.log_alpha.detach().exp()))
        polyak(self.t1,self.q1,self.tau);polyak(self.t2,self.q2,self.tau);self.updates+=1
        return dict(d_critic=float(td.detach()),d_immediate_loss=float(imloss.detach()) if imloss is not None else 0.,d_layout_pair_loss=float(pairloss.detach()) if pairloss is not None else 0.,d_mc=float(mc.detach()),d_actor=float(al.detach()),d_qgrad=qg,d_agrad=ag,d_kl=float((p*lr).sum(-1).mean().detach()),d_target=float(y.mean()),**alpha_stats,**risk_stats,**bstats)
    def learn_binary_pairs(self,pairs,rng,updates=4):
        if not pairs:return dict(d_pair_count=self.binary_pair_count,d_pair_updates=self.binary_pair_updates)
        self.binary_pairs.extend(pairs);self.binary_pair_count+=len(pairs)
        for _ in range(updates):
            rr=[self.binary_pairs[int(i)] for i in rng.integers(len(self.binary_pairs),size=16)];x=batch([r.state for r in rr],self.device);y=self.tensor([r.delta for r in rr]);q=self.binary_adv(x);loss=F.smooth_l1_loss(q,y)
            self.binary_adv_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.binary_adv.parameters(),10,error_if_nonfinite=True);self.binary_adv_opt.step();self.binary_pair_updates+=1
        return dict(d_pair_count=self.binary_pair_count,d_pair_updates=self.binary_pair_updates,d_pair_loss=float(loss.detach()),d_pair_pred=float(q.mean().detach()),d_pair_target=float(y.mean()))
    def state_dict(self):
        result={k:getattr(self,k).state_dict() for k in ['actor','q1','q2','t1','t2','ao','qo']}|dict(updates=self.updates)
        if self.binary_advantage:result.update(binary_adv=self.binary_adv.state_dict(),binary_adv_opt=self.binary_adv_opt.state_dict(),binary_pairs=self.binary_pairs,binary_pair_count=self.binary_pair_count,binary_pair_updates=self.binary_pair_updates)
        if self.risk_weight:result.update(risk=self.risk.state_dict(),risk_target=self.risk_target.state_dict(),risk_opt=self.risk_opt.state_dict())
        if self.auto_alpha:result.update(log_alpha=self.log_alpha.detach().clone(),alpha_opt=self.alpha_opt.state_dict())
        return result
    def load_state_dict(self,x):
        for k,v in x.items():
            if k in ('updates','binary_pairs','binary_pair_count','binary_pair_updates'):setattr(self,k,v)
            elif k=='log_alpha':self.log_alpha.data.copy_(v)
            else:getattr(self,k).load_state_dict(v)

"""Current experimental Agent-D Actor and exact categorical sampling.

BUILD-only baseline: coverage ADD; after coverage, ADD / REMOVE /
RELOCATE / FINISH. This is a *baseline known not to have converged*,
not a claim of a new trainable algorithm.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .coverage import opportunity_effects
from .structure import structural_features
from edge_d.world import ResourceFeatures
class GlobalPolicy(nn.Module):
    def __init__(self,base_dim):
        super().__init__()
        self.core=nn.Sequential(nn.Linear(base_dim+20,64),nn.SiLU(),nn.Linear(64,48),nn.SiLU(),nn.Linear(48,1))
        self.stop_bias=nn.Parameter(torch.tensor(-3.6))
        self.stop=nn.Sequential(nn.Linear(5,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.stop[-1].weight);nn.init.zeros_(self.stop[-1].bias)
    def forward(self,state):
        b,d,a,x,_,_,_=state
        logits=self.core(torch.cat((b,d,a),-1)).reshape(-1)
        ctx=torch.stack([x.float().mean(),d[:,:,1].float().mean(),d[:,:,2].float().mean(),
                         (x.any(1)).float().mean(),a[:,:,0].max()])
        return logits,(self.stop_bias+self.stop(ctx).reshape(()))

class GlobalFinishPolicy(GlobalPolicy):
    """Learned FINISH context; does not change ADD logits or action legality.

    Critically, the descriptors are pooled over *deployed* or *currently legal*
    cells rather than averages over all 896 (mostly irrelevant) positions.
    Uncovered stages (if any) remain represented through already-legal coverage
    actions; FINISH itself is legal only when all stages have coverage.
    """
    def __init__(self,base_dim):
        super().__init__(base_dim)
        self.dim=base_dim+20
        self.stage_embed=nn.Sequential(nn.Linear(self.dim,16),nn.SiLU())
        self.stage_score=nn.Linear(16,1)
        self.node_score=nn.Linear(16,1)
        # stage attention, node attention, legal action context, max stage
        self.additional_stop=nn.Sequential(nn.Linear(16*4+6,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.additional_stop[-1].weight)
        nn.init.zeros_(self.additional_stop[-1].bias)
    def forward(self,state):
        logits,stop=super().forward(state)
        b,d,a,x,legal,phase,finish=state
        h=self.stage_embed(torch.cat((b,d,a),-1))
        existing=x.bool()
        feasible=torch.from_numpy(legal[:x.numel()].reshape(x.shape).copy())
        # Each stage currently has its own set of deployed replicas.  Express
        # pressure/competition as stage-level pooled representations.
        counts=existing.sum(1,keepdim=True).clamp(min=1).float()
        stages=(h*existing.unsqueeze(-1)).sum(1)/counts
        active=existing.any(1)
        if bool(active.any()):
            scores=self.stage_score(stages).reshape(-1)
            scores=scores.masked_fill(~active,-1e6)
            weighted_stages=(stages*F.softmax(scores,dim=0)[:,None]).sum(0)
            critical_stages=stages[active].amax(0)
        else:
            weighted_stages=h.new_zeros(16)
            critical_stages=h.new_zeros(16)
        # A node can host multiple stages which share a finite CPU capacity.
        nc=existing.sum(0,keepdim=False).clamp(min=1).float()
        nodes=(h*existing.unsqueeze(-1)).sum(0)/nc[:,None]
        active_nodes=existing.any(0)
        if bool(active_nodes.any()):
            nscores=self.node_score(nodes).reshape(-1)
            nscores=nscores.masked_fill(~active_nodes,-1e6)
            weighted_nodes=(nodes*F.softmax(nscores,dim=0)[:,None]).sum(0)
        else: weighted_nodes=h.new_zeros(16)
        legal_n=feasible.sum().clamp(min=1).float()
        legal_context=(h*feasible.unsqueeze(-1)).sum((0,1))/legal_n
        # No CPU hard threshold: continuous summaries only.
        node_rep=existing.float().sum(0)
        stage_rep=existing.float().sum(1)
        extra=torch.stack((
            existing.float().mean(),
            node_rep.max()/max(1,x.shape[0]),
            node_rep.std(unbiased=False)/max(1,x.shape[0]),
            stage_rep.min()/max(1,x.shape[1]),
            feasible.float().mean(),
            a[:,:,0].max(),
        ))
        feature=torch.cat((weighted_stages,weighted_nodes,legal_context,critical_stages,extra))
        return logits,stop+self.additional_stop(feature).reshape(())

class EditActor(GlobalFinishPolicy):
    def __init__(self,dim):
        super().__init__(dim)
        # Two new policy-selected action types; initial probabilities are small,
        # not hard-coded restrictions or proxy optimization decisions.
        self.kind_head=nn.Sequential(nn.Linear(6,16),nn.SiLU(),nn.Linear(16,2))
        nn.init.zeros_(self.kind_head[-1].weight)
        nn.init.constant_(self.kind_head[-1].bias,-2.5)
        self.remove_core=nn.Sequential(nn.Linear(dim+20,32),nn.SiLU(),nn.Linear(32,1))

    def edit_scores(self,s):
        b,d,a,x,mask,phase,finish=s
        stacked=torch.cat((b,d,a),-1)
        rem=self.remove_core(stacked).reshape(x.shape)
        # no future rewards, only present partial matrix and observed resources
        count=x.float().sum(1)
        used=x.float().sum(0)
        meta=torch.stack((x.float().mean(), count.min()/32.,count.max()/32.,
                          used.max()/28.,d[:,:,1].mean(),a[:,:,0].max()))
        return rem,self.kind_head(meta)

def lightweight_state(e,base,feature_maker):
    # Exact physical RAM/SSD legality comes from Env.legal(); no solver output
    # ever changes the policy's action or the submitted matrix.
    dyn=feature_maker.features(e.draft,'C')
    if e.phase==1:cut,_=opportunity_effects(e,e.draft)
    else:cut,_=structural_features(e,e.draft)  # fast 1- and 2-stage necessary CPU bounds
    return (base,dyn,cut,torch.from_numpy(e.draft.copy()),e.legal().copy(),e.phase,e.FINISH)

def legality(e):
    X=e.draft.copy();K,N=X.shape
    add=e.legal()[:K*N].reshape(K,N).copy()
    rem=X&(X.sum(1)[:,None]>1)
    free=e.capacity-X.T@e.spec
    dest=(~X)&(free[None]+1e-8>=e.spec[:,None]).all(axis=2)
    movable=X&dest.any(1)[:,None]
    return add,rem,movable,dest

def _entropy(logp):
    if logp.numel()==0:return logp.new_zeros(())
    return -(logp.exp()*logp).sum()

def choose_action(actor,s,e,arm,rng,compute_entropy=False,force_continue=False):
    """Return policy action, exact differentiable log-probability, entropy.

    For RELOCATE, p(k,from,to)=p(continue)*p(type)*p(k,from|type)
                                      *p(to|k,from,type).
    """
    logits,stop=actor(s);x=s[3];K,N=x.shape
    add,rem,mov,dest=legality(e)
    def categorical(ids,values):
        lps=F.log_softmax(values,dim=0)
        p=lps.detach().exp().cpu().numpy();p=p/p.sum()
        pos=int(rng.choice(len(ids),p=p))
        return int(ids[pos]),lps[pos],_entropy(lps),lps
    if e.phase==1:
        ids=np.flatnonzero(add.ravel())
        assert len(ids) and not bool(e.legal()[e.FINISH])
        idx,lp,ent,_=categorical(ids,logits[torch.as_tensor(ids,dtype=torch.long)])
        return ('ADD',idx//N,idx%N,None),lp,ent
    assert e.phase==2
    remscore,kind_extra=actor.edit_scores(s)
    can_add=bool(add.any());can_rem=bool(rem.any());can_move=bool(mov.any())
    kinds=['ADD']
    kinds_logits=[logits.new_zeros(())]
    if arm=='EDIT' and can_rem:
        kinds.append('REMOVE');kinds_logits.append(kind_extra[0])
    if arm=='EDIT' and can_move:
        kinds.append('RELOCATE');kinds_logits.append(kind_extra[1])
    if not can_add:
        kinds=[k for k in kinds if k!='ADD']
        kinds_logits=[kind_extra[0] if k=='REMOVE' else kind_extra[1] for k in kinds]
    if not kinds:
        if force_continue: raise ValueError('No legal continuation: FINISH is forced by capacity')
        return ('FINISH',None,None,None),stop.new_zeros(()),stop.new_zeros(())
    kindlog=F.log_softmax(torch.stack(kinds_logits),0)
    stoplp=F.logsigmoid(stop)
    contlp=F.logsigmoid(-stop)
    if compute_entropy:
        Hstop=-(stoplp.exp()*stoplp+contlp.exp()*contlp)
        Hkind=_entropy(kindlog)
    kindind=int(rng.choice(len(kinds),p=kindlog.detach().exp().cpu().numpy()))
    # For exact entropy, build conditional choice entropy for each allowed kind.
    def choice_entropy(kind):
        if kind=='ADD':
            ids=np.flatnonzero(add.ravel()); lp=F.log_softmax(logits[torch.as_tensor(ids)],0)
            return _entropy(lp)
        if kind=='REMOVE':
            ids=np.flatnonzero(rem.ravel());lp=F.log_softmax(remscore.reshape(-1)[torch.as_tensor(ids)],0)
            return _entropy(lp)
        ids=np.flatnonzero(mov.ravel());lp=F.log_softmax(remscore.reshape(-1)[torch.as_tensor(ids)],0)
        h=_entropy(lp)
        for t,src in enumerate(ids):
            k=src//N
            ds=np.flatnonzero(dest[k])
            h=h+lp[t].exp()*_entropy(F.log_softmax(logits.reshape(K,N)[k,torch.as_tensor(ds)],0))
        return h
    H=(Hstop+contlp.exp()*(Hkind+sum(kindlog[i].exp()*choice_entropy(k) for i,k in enumerate(kinds)))) if compute_entropy else stop.new_zeros(())
    # Forced CONTINUE is used ONLY for physical counterfactual sampling.
    # The ordinary on-policy implementation is unchanged.
    if not force_continue and rng.random()<float(stoplp.detach().exp()):
        return ('FINISH',None,None,None),stoplp,H
    kind=kinds[kindind]
    if kind=='ADD':
        ids=np.flatnonzero(add.ravel());idx,lp,_,_=categorical(ids,logits[torch.as_tensor(ids)])
        return ('ADD',idx//N,idx%N,None),contlp+kindlog[kindind]+lp,H
    if kind=='REMOVE':
        ids=np.flatnonzero(rem.ravel());idx,lp,_,_=categorical(ids,remscore.reshape(-1)[torch.as_tensor(ids)])
        return ('REMOVE',idx//N,idx%N,None),contlp+kindlog[kindind]+lp,H
    ids=np.flatnonzero(mov.ravel());idx,src_lp,_,_=categorical(ids,remscore.reshape(-1)[torch.as_tensor(ids)])
    k,from_node=divmod(idx,N)
    ds=np.flatnonzero(dest[k]);to,dlp,_,_=categorical(ds,logits.reshape(K,N)[k,torch.as_tensor(ds)])
    return ('RELOCATE',k,from_node,to),contlp+kindlog[kindind]+src_lp+dlp,H

def json_action(action):
    kind,k,n,to=action
    return {'kind':kind,'stage':None if k is None else int(k),
            'node':None if n is None else int(n),'destination':None if to is None else int(to)}

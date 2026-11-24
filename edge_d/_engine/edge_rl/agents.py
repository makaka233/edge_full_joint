"""Masked Dueling Double DQN for the actual S training path."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .replay import laber_selection

def soft_target(reward, discount, terminated, value):
    # Time-limit truncation bootstraps; true termination does not.
    return reward+discount*(~terminated).float()*value


def greedy_actions(online, mask):
    return online.masked_fill(~mask, -1e9).argmax(-1)


def double_dqn_value(online, target, mask):
    action = greedy_actions(online, mask)
    return target.gather(1, action[:,None]).squeeze(1)


def polyak(target, online, tau):
    with torch.no_grad():
        for t, s in zip(target.parameters(),online.parameters()): t.lerp_(s,tau)


class DuelingNetwork(nn.Module):
    def __init__(self, hidden=64, feature_dim=12, mask_center=False):
        super().__init__()
        self.mask_center=mask_center
        self.encoder=nn.Sequential(nn.Linear(feature_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.advantage=nn.Sequential(nn.Linear(2*hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.value=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def add_structured_branch(self,k=28):
        self.structured_k=k
        self.stage_encoder=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU())
        self.edge_encoder=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU())
        self.relational_head=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
        with torch.no_grad():
            self.relational_head[-1].weight.zero_();self.relational_head[-1].bias.zero_()

    def forward(self,x):
        relational=None
        if hasattr(self,'structured_k'):
            b,n,_=x.shape;k=self.structured_k
            tok=x[:,:,24:].reshape(b,n,k,4)
            ids=torch.arange(k,device=x.device,dtype=x.dtype)/max(1,k-1)
            ids=ids[None,None,:,None].expand(b,n,k,1)
            current=(torch.abs(ids-x[:,:,11,None,None])<.001).to(x.dtype)
            hstage=self.stage_encoder(torch.cat([tok,ids,current],dim=-1))
            # Node-to-stage messages over all candidate nodes. Deployment and live
            # work are explicit edge attributes; shared stage aggregation connects nodes.
            stage=hstage.mean(1,keepdim=True).expand_as(hstage)
            edge=self.edge_encoder(torch.cat([hstage,stage],dim=-1))
            # Only deployed edges carry stage work; global pooling preserves context.
            dep=tok[:,:,:,0:1];node=(edge*dep).sum(2)/dep.sum(2).clamp_min(1.)
            global_node=node.mean(1,keepdim=True).expand_as(node)
            relational=self.relational_head(torch.cat([node,global_node],dim=-1)).squeeze(-1)
            x=x[:,:,:24]
        h=self.encoder(x)
        global_h=h.mean(1)
        advantage=self.advantage(torch.cat([h,global_h[:,None].expand_as(h)],dim=-1)).squeeze(-1)
        if relational is not None:advantage=advantage+relational
        if self.mask_center:
            mask=x[:,:,7]
            center=(advantage*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1.)
        else: center=advantage.mean(1,keepdim=True)
        return self.value(global_h)+advantage-center


class SchedulingAgent:
    def __init__(self,c):
        self.c=c; self.device=torch.device(c.device)
        extended=getattr(c,'s_next_context',False)
        self.online=DuelingNetwork(c.hidden,16 if extended else 12,extended).to(self.device)
        if getattr(c,'s_competition_context',False):
            if not extended: raise ValueError('competition context requires next-stage context')
            # Start with the exact same function as the 16-feature baseline.
            old=self.online.encoder[0]
            with torch.random.fork_rng(devices=[]):
                new=nn.Linear(28 if getattr(c,"s_reservation_context",False) else 27 if getattr(c,"s_system_credit",False) else 24,c.hidden).to(self.device)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:,:16].copy_(old.weight); new.bias.copy_(old.bias)
            self.online.encoder[0]=new
        if getattr(c,"s_structured_context",False):
            with torch.random.fork_rng(devices=[]):self.online.add_structured_branch(c.k)
            self.online.to(self.device)
        self.local_auxiliary=getattr(c,'s_local_auxiliary',False)
        if self.local_auxiliary:
            self.local_online=copy.deepcopy(self.online)
            self.local_target=copy.deepcopy(self.local_online).requires_grad_(False)
            self.local_optimizer=torch.optim.Adam(self.local_online.parameters(),lr=c.learning_rate)
            # A trainable residual is initialized at zero; the combined Q is
            # learned on the global objective, not a hard-coded heuristic.
            with torch.no_grad():
                for head in (self.online.advantage,self.online.value):
                    head[-1].weight.zero_();head[-1].bias.zero_()
        self.target=copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=c.learning_rate)
        self.version=self.updates=0

    def team_q(self,x,target=False):
        q=(self.target if target else self.online)(x)
        if self.local_auxiliary:
            local=(self.local_target if target else self.local_online)(x).detach()
            if getattr(self.c,'s_center_local_advantage',False):
                # Only the action-dependent local advantage is useful as a prior.
                # The team network owns the entire system-level state value.
                mask=x[:,:,7];local=local-(local*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1.)
            q=q+local
        return q

    def update_local(self,batch):
        if not self.local_auxiliary or not batch:return {}
        def tensor(values,dtype=torch.float32):return torch.as_tensor(np.asarray(values),dtype=dtype,device=self.device)
        x=tensor([t.obs for t in batch]);nx=tensor([t.next_obs for t in batch]);nm=tensor([t.next_mask for t in batch],torch.bool)
        a=tensor([t.action for t in batch],torch.long);r=tensor([t.reward for t in batch]);done=tensor([t.terminated for t in batch],torch.bool)
        with torch.no_grad():y=soft_target(r,torch.full_like(r,.99),done,double_dqn_value(self.local_online(nx),self.local_target(nx),nm))
        q=self.local_online(x).gather(1,a[:,None]).squeeze(1);loss=F.smooth_l1_loss(q,y)
        self.local_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.local_online.parameters(),10,error_if_nonfinite=True);self.local_optimizer.step();polyak(self.local_target,self.local_online,self.c.tau)
        return dict(s_local_loss=float(loss.detach()))

    def scheduler(self,rng,epsilon=0.):
        @torch.no_grad()
        def act(obs,mask):
            if rng.random() < epsilon: return int(rng.choice(np.flatnonzero(mask)))
            q=self.team_q(torch.as_tensor(obs[None],device=self.device))
            return int(q.masked_fill(~torch.as_tensor(mask[None],device=self.device),-1e9).argmax(1).item())
        return act

    @torch.no_grad()
    def system_trace_targets(self,transitions,lam):
        """Watkins-style truncated lambda targets on fresh ordered system decisions.
        Stop the trace at any next behavior action that is not current-greedy.
        All window ends are true finite-cohort terminals.
        """
        if not transitions:return np.empty(0,np.float32)
        vals=[];greedy=[]
        for i in range(0,len(transitions),256):
            b=transitions[i:i+256]
            nx=torch.as_tensor(np.stack([t.next_obs for t in b]),device=self.device)
            nm=torch.as_tensor(np.stack([t.next_mask for t in b]),device=self.device)
            a=self.team_q(nx).masked_fill(~nm,-1e9).argmax(-1)
            vals.extend(self.team_q(nx,True).gather(1,a[:,None]).squeeze(1).cpu().numpy());greedy.extend(a.cpu().numpy())
        out=np.empty(len(transitions),np.float32)
        for i in range(len(transitions)-1,-1,-1):
            t=transitions[i]
            if t.terminated:out[i]=t.reward
            else:
                assert i+1<len(transitions) and transitions[i+1].window_index==t.window_index
                next_value=float(vals[i]);w=lam if transitions[i+1].action==greedy[i] else 0.
                out[i]=t.reward+self.c.gamma_s*((1-w)*next_value+w*float(out[i+1]))
        return out

    def update(self,batch,fresh=None,fresh_targets=None,*,laber_rng=None,laber_size=None,successors=None):
        if not batch: return {}
        if laber_rng is not None and (fresh is not None or laber_size is None):
            raise ValueError('Large-batch replay requires an ordinary replay pool and a minibatch size')
        if successors is not None and (len(successors)!=len(batch) or fresh is not None or laber_rng is not None):
            raise ValueError('Two-step DDQN requires one successor slot per ordinary replay item')
        nf=len(fresh) if fresh is not None else 0
        if nf:batch=list(batch)+list(fresh)
        def tensor(values,dtype=torch.float32): return torch.as_tensor(np.asarray(values),dtype=dtype,device=self.device)
        x=tensor([t.obs for t in batch]); nx=tensor([t.next_obs for t in batch])
        nm=tensor([t.next_mask for t in batch],torch.bool)
        action=tensor([t.action for t in batch],torch.long)
        reward=tensor([t.reward for t in batch]); terminal=tensor([t.terminated for t in batch],torch.bool)
        diagnostics={}
        with torch.no_grad():
            greedy=greedy_actions(self.team_q(nx),nm)
            value=self.team_q(nx,True).gather(1,greedy[:,None]).squeeze(1)
            gamma=self.c.gamma_s
            target=soft_target(reward,torch.full_like(reward,gamma),terminal,value)
            if successors is not None:
                # A non-greedy behavior action cuts the Watkins trace at s(t+1).
                available=sum(t is not None for t in successors)
                selected=[(i,t) for i,(t,a) in enumerate(zip(successors,greedy.tolist(),strict=True))
                          if t is not None and not batch[i].terminated and t.action==a]
                diagnostics.update(s_successor_share=available/len(batch),s_two_step_share=len(selected)/len(batch))
                if selected:
                    indices, following=zip(*selected)
                    idx=tensor(indices,torch.long)
                    endpoint=tensor([t.next_obs for t in following])
                    mask=tensor([t.next_mask for t in following],torch.bool)
                    second=tensor([t.reward for t in following])
                    ended=tensor([t.terminated for t in following],torch.bool)
                    continuation=double_dqn_value(self.team_q(endpoint),self.team_q(endpoint,True),mask)
                    target[idx]=reward[idx]+gamma*soft_target(second,torch.full_like(second,gamma),ended,continuation)
            if nf:target[-nf:]=tensor(fresh_targets)
        if laber_rng is None:
            q=self.team_q(x).gather(1,action[:,None]).squeeze(1)
            loss=F.smooth_l1_loss(q,target)
        else:
            with torch.no_grad():
                q_pool=self.team_q(x).gather(1,action[:,None]).squeeze(1)
                error=(target-q_pool).abs()
                # Magnitude of the derivative of the original Huber loss w.r.t. Q.
                picked,weights=laber_selection(error.clamp_max(1).cpu().numpy(),laber_size,laber_rng)
                idx=torch.as_tensor(picked,dtype=torch.long,device=self.device)
                w=torch.as_tensor(weights,dtype=q_pool.dtype,device=self.device)
                pressure=torch.expm1(x[torch.arange(len(x),device=self.device),action,16])>=.5
                diagnostics=dict(s_pool_td=float(error.mean()),s_sample_td=float(error[idx].mean()),
                                 s_pool_pressure=float(pressure.float().mean()),
                                 s_sample_pressure=float(pressure[idx].float().mean()),
                                 s_is_ess=float(w.sum().square()/w.square().sum()),
                                 s_weight_mean=float(w.mean()))
            q=self.team_q(x[idx]).gather(1,action[idx,None]).squeeze(1)
            loss=(F.smooth_l1_loss(q,target[idx],reduction='none')*w).mean()
        self.optimizer.zero_grad(); loss.backward(); grad_norm=nn.utils.clip_grad_norm_(self.online.parameters(),10,error_if_nonfinite=True)
        self.optimizer.step(); polyak(self.target,self.online,self.c.tau)
        self.updates+=1
        return dict(s_loss=float(loss.detach()),s_grad_norm=float(grad_norm),**diagnostics)

    def state_dict(self):
        out=dict(online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),version=self.version,updates=self.updates)
        if self.local_auxiliary:out.update(local_online=self.local_online.state_dict(),local_target=self.local_target.state_dict(),local_optimizer=self.local_optimizer.state_dict())
        return out

    def load_state_dict(self,state):
        for key,value in state.items():
            if key in ["version","updates"]: setattr(self,key,value)
            else: getattr(self,key).load_state_dict(value)

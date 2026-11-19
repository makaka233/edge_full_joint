"""Physical network, versioned stochastic demand splits, observation features.

IMPORTANT: The checked-in configuration reflects the audited legacy benchmark,
not the original 7200-CU/s / 300-Mbps research wish list.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from ._engine.edge_rl.config import Config
from ._engine.environment import Env

CONFIG_DIR=Path(__file__).resolve().parents[1]/"configs"
SPLITS=("train","fixed","holdout")

def stable_seed(*tokens):
    return int.from_bytes(hashlib.sha256("\0".join(map(str,tokens)).encode()).digest()[:4],"big")

def make_world(seed:int, scene:int, split:str="train"):
    """Independent population/demand by seed and split; deterministic per identity."""
    if split not in SPLITS: raise ValueError(f"unknown split: {split}")
    if scene<0: raise ValueError("scene must be nonnegative")
    cfg=json.loads((CONFIG_DIR/"experiment.json").read_text(encoding="utf8"))
    c=Config.load(CONFIG_DIR/"world.json")
    c.cpu_allocator=cfg["cpu_allocator"]
    c.s_next_context=bool(cfg.get("s_next_context",True))
    c.s_competition_context=bool(cfg.get("s_competition_context",False))
    # Original S simulator extensions must be configured with the network.
    for key in ('s_system_credit','s_system_credit_scale','s_resource_credit',
                's_structured_context','s_reservation_context','s_local_auxiliary',
                's_center_local_advantage','s_local_period_mean','s_local_reward_scale',
                's_objective_aligned','s_objective_reward_scale','s_mc_pg'):
        if key in cfg:setattr(c,key,cfg[key])
    c.device="cpu"
    c.torch_threads=1
    pop_split="holdout_population" if split=="holdout" else "training_population"
    pop_seed=stable_seed("edge-v2",seed,pop_split)
    e=Env(c,pop_seed,cfg["episode_steps"],cfg["windows"],cfg["window_seconds"],
          cfg["warm_seconds"],cfg["tail_seconds"],cfg["failure_penalty_seconds"],
          cfg.get("demand_model"),cfg.get("cost_model"),cfg.get("continuing",False))
    e.risk_threshold=cfg.get("d_risk_threshold",1.)
    e.reset(stable_seed("edge-v2",seed,split,"demand",scene),0)
    for _ in range(scene%12): e.w.advance(); e.period+=1
    return e

def pair_features(e):
 w=e.w;d=w.node_demand();stage=d[:,w.stage_service].T;K,N=e.K,e.N;R=np.eye(4)[w.regions];rg=stage@R;old=e.old.astype(float);util=old.T@e.spec/e.capacity
 fs=[np.broadcast_to(w.stages[:,j,None]/s,(K,N)) for j,s in [(0,2),(1,4),(2,4),(3,6),(4,50)]]
 fs += [np.broadcast_to(w.nodes[None,:,j]/s,(K,N)) for j,s in [(0,20),(1,24),(2,160)]]
 fs += [stage/1.5,np.broadcast_to(stage.sum(1)[:,None]/8,(K,N)),(rg@R.T)/4,old,np.broadcast_to(old.mean(1)[:,None],(K,N)),np.broadcast_to(old.mean(0)[None,:],(K,N)),np.broadcast_to(util[None,:,0],(K,N)),np.broadcast_to(util[None,:,1],(K,N)),np.broadcast_to(d.sum(0)[w.stage_service,None]/8,(K,N)),np.broadcast_to(d.sum(1)[None,:]/3,(K,N)),np.full((K,N),d.sum()/40)]
 return torch.tensor(np.stack(fs,-1),dtype=torch.float32)

def observe(e):
    """20 observed per-stage/per-node features; no future request outcomes."""
    x=pair_features(e)
    w=e.w; d=w.node_demand()[:,w.stage_service].T
    hops=np.asarray([[len(w.routes[i][j]) for j in range(e.N)] for i in range(e.N)],float)
    local=(d@hops)/np.maximum(d.sum(1)[:,None],1e-10)/8.
    return torch.cat((x,torch.tensor(local[...,None],dtype=torch.float32)),dim=-1)

class LoadAwareHeuristic:
    """Fixed deterministic scheduler based solely on existing S observation fields.
    No lookahead into random future requests and no access to evaluation rewards.
    """
    version=0
    def scheduler(self, rng=None, epsilon=0.):
        def pick(obs, mask):
            capacity=np.maximum(obs[:,0]*20,1e-6)
            queued=np.maximum(obs[:,2]*20,0.)
            stage_work=np.maximum(obs[:,8]*2,0.)
            path_sec=np.maximum(obs[:,5],0.)
            queued_hops=np.maximum(obs[:,4]*20,0.)
            # workload / processing capacity + route latency + measured route competition
            # 16-column S observations include downstream optimistic communication+execution.
            score=(queued+stage_work)/capacity+path_sec*(1+.06*queued_hops)
            if obs.shape[1]>=16:score=score+.45*np.maximum(obs[:,14],0.)
            return int(np.argmin(np.where(mask,score,np.inf)))
        return pick

class ResourceFeatures:
    def __init__(self,e):
        self.K,self.N=e.K,e.N
        self.cpu=e.w.nodes[:,0].astype(float)
        self.spec=e.spec.astype(float)
        self.cap=e.capacity.astype(float)
        self.work=e.w.stages[:,0].astype(float)
        self.service=e.w.stage_service.astype(int)
        self.dem=(e.w.node_demand()[:,self.service].T*self.work[:,None])
        route=e.w.path_seconds_per_mbit.astype(float)
        transfer=e.w.stages[:,1].astype(float)
        comm=route[None,:,:]*transfer[:,None,None]
        compute=self.work[:,None,None]/np.maximum(self.cpu[None,None,:],1e-8)
        self.cost=comm+compute
        # A fast, positive conditional dispatch proxy, not the real frozen S policy.
        z=-self.cost/.1;z=z-z.max(axis=2,keepdims=True)
        self.weights=np.maximum(np.exp(z),1e-12)
        self.dem_sum=self.dem.sum(1)
    def features(self,X,arm):
        K,N=self.K,self.N
        used=X.T@self.spec
        free=(self.cap-used)/self.cap
        reps=X.sum(1)
        common=np.stack([
            X.astype(float),
            np.broadcast_to(free[None,:,0],(K,N)),
            np.broadcast_to(free[None,:,1],(K,N)),
            np.broadcast_to(reps[:,None]/N,(K,N)),
            np.full((K,N),np.count_nonzero(reps)/K),
            np.broadcast_to(X.sum(0)[None,:]/K,(K,N)),
            np.broadcast_to((reps==0)[:,None],(K,N)),
        ],axis=-1)
        dynamic=np.zeros((K,N,7),dtype=float)
        if arm=='A':return torch.as_tensor(np.concatenate([common,dynamic],axis=-1),dtype=torch.float32)
        # Global load forecast under the current partially built matrix.
        norm=(self.weights*X[:,None,:]).sum(2)
        probs=np.divide(self.weights*X[:,None,:],norm[:,:,None],out=np.zeros_like(self.weights),where=norm[:,:,None]>0)
        src_load=probs*self.dem[:,:,None]
        per_stage=src_load.sum(1)
        total=per_stage.sum(0)
        rho=total/np.maximum(self.cpu,1e-8)
        stage_pressure=self.dem_sum/np.maximum(X@self.cpu,1e-8)
        dynamic[:,:,0]=np.broadcast_to(np.log1p(np.maximum(rho,0))[None,:],(K,N))
        dynamic[:,:,1]=np.broadcast_to(np.log1p(np.maximum(stage_pressure,0))[:,None],(K,N))
        dynamic[:,:,2]=np.broadcast_to(np.log1p(np.maximum(rho.max(),0)),(K,N))
        if arm=='C':
            # Counterfactual ADD(k,n): reweight only the selected stage's
            # workload shares, then measure effect across ALL nodes.
            fraction=np.divide(self.weights,norm[:,:,None]+self.weights,
                               out=np.ones_like(self.weights),where=(norm[:,:,None]+self.weights)>0)
            reduction=np.einsum('ksj,ksn->knj',src_load,fraction,optimize=True)
            added=np.einsum('ks,ksn->kn',self.dem,fraction,optimize=True)
            delta=-reduction
            ii=np.arange(N)
            delta[:,ii,ii]+=added
            future=(total[None,None,:]+delta)/np.maximum(self.cpu[None,None,:],1e-8)
            worst=np.max(future,axis=2)
            risk=lambda r:np.sum(np.maximum(r-.85,0.)**2,axis=-1)
            risk_base=float(risk(rho))
            risk_after=risk(future)
            dynamic[:,:,3]=np.tanh(np.max(rho)-worst)
            dynamic[:,:,4]=np.tanh((risk_base-risk_after)/5.)
            dynamic[:,:,5]=np.tanh(future[:,ii,ii]-rho[None,:])
            # Change of stage communication cost for the ADD action.
            old_comm=np.einsum('ksj,ksj->ks',probs,self.cost,optimize=True)
            new_comm=self.cost.transpose(0,1,2)
            dd=self.dem[:,:,None]*fraction*(new_comm-old_comm[:,:,None])
            comm_delta=np.sum(dd,axis=1)/np.maximum(self.dem_sum[:,None],1e-8)
            dynamic[:,:,6]=np.tanh(-comm_delta/.1)
        return torch.as_tensor(np.concatenate([common,dynamic],axis=-1),dtype=torch.float32)

def cpu_lower_bound(e,X):
    K,N=e.K,e.N
    d=e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0]
    ij=np.argwhere(X);count=len(ij)
    eq=np.zeros((K,count+1));ub=np.zeros((N,count+1))
    for j,(k,n) in enumerate(ij):eq[k,j]=1;ub[n,j]=1
    ub[:,-1]=-e.w.nodes[:,0]
    fit=linprog(np.r_[np.zeros(count),1.],A_ub=ub,b_ub=np.zeros(N),A_eq=eq,b_eq=d,
                bounds=[(0,None)]*(count+1),method='highs')
    if not fit.success:raise RuntimeError(fit.message)
    return float(fit.x[-1])

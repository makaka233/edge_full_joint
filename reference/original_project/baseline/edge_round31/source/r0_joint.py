"""R0: one true macro KEEP/BUILD decision, simultaneous structured row proposals,
legal global projection; on-policy n-step actor-critic D and original DDQN S.
No reference layouts or old replay in training. All random action draws logged.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse, copy, gzip, hashlib, json, pickle, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from train import config,environment,rng_for,digest,write,read_core
from edge_rl.agents import SchedulingAgent
from edge_rl.replay import SReplay
from p23_global import features as pair_features,Decode,valid,hexmat
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'r0_results'; CFG=ROOT/'configs/experiment_r0_joint.json'; K=28; N=32

# Current-state features only. Same full old matrix, per-stage demand and topology.
def observe(e):
    x=pair_features(e); w=e.w; d=w.node_demand()[:,w.stage_service].T
    hops=np.array([[len(w.routes[i][j]) for j in range(N)] for i in range(N)],float)
    local=(d@hops)/np.maximum(d.sum(1)[:,None],1e-10)/8.
    x=torch.cat((x,torch.tensor(local[...,None],dtype=torch.float32)),dim=-1)
    ctx=torch.tensor(w.context(),dtype=torch.float32)
    old=torch.tensor(e.old.copy(),dtype=torch.bool)
    # A capacity-derived reference density, NOT a quota; all 1..N counts have support.
    frac=float(np.min(e.capacity.sum(0)/(N*e.spec.sum(0))))
    frac=np.clip((N*frac-1)/(N-1),.02,.98)
    base=float(np.log(frac/(1-frac)))
    return x,ctx,old,base

class DPolicy(nn.Module):
    def __init__(self,ctxdim):
        super().__init__();self.se=nn.Embedding(K,8);self.ne=nn.Embedding(N,8)
        self.pair=nn.Sequential(nn.Linear(36,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.ctx=nn.Sequential(nn.Linear(ctxdim,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.row=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.node=nn.Linear(32,1);self.count=nn.Linear(32,1);self.refresh=nn.Linear(32,1)
        self.macro=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,2))
        self.value=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,obs):
        x,ctx,old,base=obs
        h=self.pair(torch.cat((x,self.se.weight[:,None].expand(K,N,8),self.ne.weight[None,:].expand(K,N,8)),dim=-1));r=h.mean(1);n=h.mean(0);g=self.ctx(ctx)
        z=self.edge(torch.cat((h,r[:,None].expand(K,N,32),n[None,:].expand(K,N,32),g[None,None].expand(K,N,32)),dim=-1))
        rr=self.row(torch.cat((z.mean(1),r,g[None].expand(K,32)),dim=-1));gg=torch.cat((z.mean((0,1)),r.mean(0),g))
        return self.macro(gg),self.refresh(rr).squeeze(-1),self.count(rr).squeeze(-1)+base,self.node(z).squeeze(-1),self.value(gg).squeeze(-1)

def sample_action(policy,obs,rng):
    with torch.no_grad():m,ref,c,loc,v=policy(obs)
    old=obs[2].numpy();covered=bool(old.any(1).all());mp=torch.softmax(m,0).numpy()
    mode=int(rng.choice(2,p=mp.astype(float)/mp.sum(dtype=float))) if covered else 1
    if mode==0:return {'mode':0,'refresh':[],'counts':[],'order':[],'jitter_seed':None},old.copy()
    refresh=(rng.random(K)<torch.sigmoid(ref).numpy()) if covered else np.ones(K,bool)
    prop=old.copy();counts=[0]*K;orders=[[] for _ in range(K)]
    cp=torch.sigmoid(c).numpy();logits=loc.numpy()
    for k in np.flatnonzero(refresh):
        count=1+int(rng.binomial(N-1,float(cp[k])))
        order=np.argsort(-(logits[k]+rng.gumbel(size=N)))[:count]
        prop[k]=False;prop[k,order]=True;counts[k]=count;orders[k]=order.tolist()
    return {'mode':1,'refresh':refresh.tolist(),'counts':counts,'order':orders,'jitter_seed':int(rng.integers(0,2**31-1))},prop

def logprob(policy,obs,a):
    m,ref,c,loc,v=policy(obs);covered=bool(obs[2].any(1).all());mode=a['mode'];lp=F.log_softmax(m,0)[mode] if covered else m.sum()*0
    if mode==0:return lp,v
    refresh=torch.tensor(a['refresh'],dtype=torch.bool)
    if covered:lp=lp+-F.binary_cross_entropy_with_logits(ref,refresh.float(),reduction='sum')
    ids=torch.where(refresh)[0];counts=torch.tensor(a['counts'],dtype=torch.float32)[ids]
    if len(ids):
        lp=lp+torch.distributions.Binomial(total_count=N-1,logits=c[ids]).log_prob(counts-1).sum()
        # Exact probability of each ordered without-replacement draw (Plackett-Luce).
        # Rows sampled simultaneously; no physical sequential resource consumption.
        orders=[a['order'][int(k)] for k in ids];maxcnt=max(map(len,orders));mask=torch.zeros((len(ids),N),dtype=torch.bool)
        z=loc[ids]
        for j in range(maxcnt):
            active=torch.tensor([j<len(o) for o in orders],dtype=torch.bool)
            ix=torch.tensor([o[j] if j<len(o) else 0 for o in orders],dtype=torch.long)
            row=torch.arange(len(ids));lp=lp+F.log_softmax(z.masked_fill(mask,-1e9),-1)[row[active],ix[active]].sum()
            mask=mask.clone();mask[row[active],ix[active]]=True
    return lp,v

def execute(e,policy,dec,rng):
    obs=observe(e);a,proposal=sample_action(policy,obs,rng)
    if a['mode']==0:X=proposal.copy();di={'solver_seconds':0.,'solver_gap':0.,'solver_status':0}
    else:
        # Minimize Hamming disagreement. Tiny, independently sampled tie break;
        # maximum aggregate perturbation <1, so exact optimum prioritizes Hamming.
        noise=np.random.default_rng(a['jitter_seed']).uniform(-1,1,(K,N))*.0001
        X,di=dec(e,np.where(proposal,1.,-1.)+noise)
    valid(e,X);return obs,a,X,dict(proposal_hex=hexmat(proposal),proposal_replicas=int(proposal.sum()),projection_hamming=int(np.count_nonzero(X!=proposal)),refreshed_rows=sum(a['refresh']),**di)

def setup(seed):
    c,x=config(CFG,seed);torch.set_num_threads(1);torch.manual_seed(seed);e=environment(c,x);s=SchedulingAgent(c);e.reset(seed+88000,0);torch.manual_seed(seed+17);p=DPolicy(len(e.w.context()));return c,x,e,s,p

def statehash(e):
    st=e.state();return hashlib.sha256(st.old.tobytes()+st.demand.tobytes()+st.context.tobytes()+np.asarray([e.period,e.w.t],dtype=np.float64).tobytes()).hexdigest()

def value_targets(rewards,values,bootstrap,gamma=.95,lam=.9):
    # Actual n-step generalized advantage, detached TD targets.
    adv=np.zeros(len(rewards),dtype=np.float32);last=0.;nxt=bootstrap
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+gamma*nxt-values[i];last=delta+gamma*lam*last;adv[i]=last;nxt=values[i]
    return adv,adv+np.array(values,dtype=np.float32)

def evaluate(p,s,c,x,seed,gs,out,arm):
    rows=[]
    for j,ds in enumerate((954101,954202)):
        e=environment(c,x);e.reset(ds,j*5);dec=Decode(e);rng=np.random.default_rng(900000+seed+j)
        for t in range(1,5):
            _,a,X,di=execute(e,p,dec,rng);ps=int(rng_for(ds,808,t).integers(0,2**31-1));old=hexmat(e.old);before=statehash(e)
            _,r,_,_,_=e.commit(X,s.scheduler(rng_for(ds,809,t),0.),ps,t,False)
            r.update(gs=gs,seed=seed,arm=arm,eval_seq=j,eval_step=t,mode=a['mode'],matrix_hex=hexmat(X),old_hex=old,state_hash=before,next_state_hash=statehash(e),d_hash=digest(p),s_hash=digest(s.online),**di);write(out,r);rows.append(r)
    print('EVAL',seed,arm,gs,np.mean([r['reward'] for r in rows]),sum(r['failed'] for r in rows),flush=True)
    return float(np.mean([r['reward'] for r in rows]))

def save(out,name,p,s,opt,e,replay,rng,gs,seed,arm,du,best):
    d=dict(d=p.state_dict(),s=s.state_dict(),d_opt=opt.state_dict(),env=e,rng=rng.bit_generator.state,gs=gs,seed=seed,arm=arm,d_updates=du,best=best,d_hash=digest(p),s_hash=digest(s.online))
    tmp=out/(name+'.tmp');torch.save(d,tmp);tmp.replace(out/name)
    if name=='latest.pt':
        with gzip.open(out/'latest_replay.pkl.gz.tmp','wb',compresslevel=1) as f:pickle.dump(replay,f,protocol=5)
        (out/'latest_replay.pkl.gz.tmp').replace(out/'latest_replay.pkl.gz')

def run(seed,arm,steps,resume=None):
    c,x,e,s,p=setup(seed);learnD=arm=='joint';learnS=arm in ('joint','s_only');out=OUT/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=bool(resume));opt=torch.optim.Adam(p.parameters(),lr=x['r0_d_lr']);dec=Decode(e);rng=np.random.default_rng(seed+771);replay=SReplay(c.s_replay_transitions,c.replay_max_age);du=0;best=-1e20;start=0
    initialD=digest(p);initialS=digest(s.online);roll=[]
    if resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);p.load_state_dict(ck['d']);s.load_state_dict(ck['s']);opt.load_state_dict(ck['d_opt']);e=ck['env'];rng.bit_generator.state=ck['rng'];du=ck['d_updates'];best=ck['best'];start=ck['gs'];dec=Decode(e)
        with gzip.open(out/'latest_replay.pkl.gz','rb') as f:replay=pickle.load(f)
        assert digest(p)==ck['d_hash'] and digest(s.online)==ck['s_hash'] and start%x['r0_rollout']==0
    else:
        best=evaluate(p,s,c,x,seed,0,out/'eval.jsonl',arm);save(out,'best.pt',p,s,opt,e,replay,rng,0,seed,arm,du,best)
    for gs in range(start+1,steps+1):
        obs,a,X,di=execute(e,p,dec,rng);old=hexmat(e.old);before=statehash(e);sver=s.version;dh=digest(p);sh=digest(s.online)
        # D and S fixed throughout a rollout, then both updated at the boundary.
        block_start=(gs-1)//x['r0_rollout']*x['r0_rollout'];eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-block_start/c.epsilon_decay_steps)
        ps=int(rng_for(seed,900,gs).integers(0,2**31-1));e.current_s_version=s.version
        _,r,ts,terminated,truncated=e.commit(X,s.scheduler(rng_for(seed,901,gs),eps),ps,gs,learnS)
        assert not terminated and not truncated and np.array_equal(e.old,X)
        if learnS:
            assert ts and all(t.mask[t.action] and t.collection_step==gs and np.isfinite(t.reward) for t in ts);replay.add(ts);replay.prune(gs)
        with torch.no_grad():lp,val=logprob(p,obs,a)
        roll.append((obs,a,.1*r['reward'],float(val),before,statehash(e)))
        r.update(gs=gs,seed=seed,arm=arm,mode=a['mode'],action=a,old_hex=old,matrix_hex=hexmat(X),state_hash=before,next_state_hash=statehash(e),d_hash=dh,s_hash=sh,s_version=sver,epsilon=eps,collected=len(ts),d_logp=float(lp),value=float(val),**di);write(out/'train.jsonl',r)
        if gs%x['r0_rollout']==0 or gs==steps:
            with torch.no_grad():bootstrap=float(p(observe(e))[-1])
            rewards=[b[2] for b in roll];values=[b[3] for b in roll];adv,target=value_targets(rewards,values,bootstrap,c.gamma_d,x['r0_lambda']);dinfo={}
            if learnD:
                lpvs=[logprob(p,b[0],b[1]) for b in roll];lp=torch.stack([a[0] for a in lpvs]);vv=torch.stack([a[1] for a in lpvs]);aa=torch.tensor(adv);tt=torch.tensor(target)
                # Exact full-action score; no division by action length, no post-hoc reward selection.
                aloss=-(lp*aa).mean();vloss=F.smooth_l1_loss(vv,tt);loss=aloss+vloss
                opt.zero_grad();loss.backward();gn=float(nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True));opt.step();du+=1
                dinfo=dict(d_loss=float(aloss.detach()),v_loss=float(vloss.detach()),grad_norm=gn)
            sl=[]
            if learnS:
                for ui in range(x['s_updates_per_period']*len(roll)):sl.append(s.update(replay.sample(x['s_batch'],gs,rng_for(seed,902,gs,ui))))
                s.version+=1
            write(out/'updates.jsonl',dict(gs=gs,seed=seed,arm=arm,rollout_gs=list(range(gs-len(roll)+1,gs+1)),rewards=rewards,values=values,bootstrap=bootstrap,advantages=adv.tolist(),targets=target.tolist(),gamma=c.gamma_d,lam=x['r0_lambda'],state_hashes=[b[4] for b in roll],next_hashes=[b[5] for b in roll],d_updates=du,s_updates=s.updates,d_hash=digest(p),s_hash=digest(s.online),s_loss=float(np.mean([a['s_loss'] for a in sl])) if sl else None,**dinfo));roll=[]
        print(seed,arm,gs,'KEEP' if a['mode']==0 else 'BUILD',round(r['reward'],3),round(r['penalized_delay'],3),r['failed'],r['replicas'],r['changed'],flush=True)
        if gs in (8,16,32,64):
            score=evaluate(p,s,c,x,seed,gs,out/'eval.jsonl',arm)
            if score>best:best=score;save(out,'best.pt',p,s,opt,e,replay,rng,gs,seed,arm,du,best)
        if gs%x['r0_rollout']==0:save(out,'latest.pt',p,s,opt,e,replay,rng,gs,seed,arm,du,best)
    if not learnD:assert digest(p)==initialD
    if not learnS:assert digest(s.online)==initialS
    (out/'completed.json').write_text(json.dumps(dict(gs=steps,d_updates=du,s_updates=s.updates,initial_d=initialD,initial_s=initialS,final_d=digest(p),final_s=digest(s.online)),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--arm',choices=['joint','s_only','frozen'],required=True);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--resume',action='store_true');a=ap.parse_args();assert a.steps%4==0;OUT.mkdir(exist_ok=True);run(a.seed,a.arm,a.steps,a.resume)

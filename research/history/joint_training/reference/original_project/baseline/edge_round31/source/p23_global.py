"""P2-3 global BUILD pilot: shared full-score network, correlated Gaussian policy, legal MILP.
Original physical env + frozen P2 PF S. No post-hoc selection, no critic, no old replay.
Score-space policy exact density; integer projection is environment action mapping.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from p2_s_train import setup,ROOT,OUT as P2OUT
from train import config,environment,rng_for,digest,write,read_core
OUT=ROOT/'p23_results'; K=28; N=32; SP=.35; SR=.30; SN=.25

def valid(e,X):
 assert X.shape==(e.K,e.N) and X.dtype==bool and X.any(1).all() and np.all(X.T@e.spec<=e.capacity+1e-7);return X

def hexmat(X):return np.packbits(X.ravel()).tobytes().hex()
def features(e):
 w=e.w;d=w.node_demand();stage=d[:,w.stage_service].T;K,N=e.K,e.N;R=np.eye(4)[w.regions];rg=stage@R;old=e.old.astype(float);util=old.T@e.spec/e.capacity
 fs=[np.broadcast_to(w.stages[:,j,None]/s,(K,N)) for j,s in [(0,2),(1,4),(2,4),(3,6),(4,50)]]
 fs += [np.broadcast_to(w.nodes[None,:,j]/s,(K,N)) for j,s in [(0,20),(1,24),(2,160)]]
 fs += [stage/1.5,np.broadcast_to(stage.sum(1)[:,None]/8,(K,N)),(rg@R.T)/4,old,np.broadcast_to(old.mean(1)[:,None],(K,N)),np.broadcast_to(old.mean(0)[None,:],(K,N)),np.broadcast_to(util[None,:,0],(K,N)),np.broadcast_to(util[None,:,1],(K,N)),np.broadcast_to(d.sum(0)[w.stage_service,None]/8,(K,N)),np.broadcast_to(d.sum(1)[None,:]/3,(K,N)),np.full((K,N),d.sum()/40)]
 return torch.tensor(np.stack(fs,-1),dtype=torch.float32)
class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(19,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1));self.pair=nn.Parameter(torch.randn(K,N)*.03);self.stage_emb=nn.Parameter(torch.randn(K,8)*.08);self.node_emb=nn.Parameter(torch.randn(N,8)*.08)
 def forward(self,x):return self.net(x).squeeze(-1)+self.pair+self.stage_emb@self.node_emb.T

def sample(mu,rng):return mu+SP*rng.normal(size=(K,N))+SR*rng.normal(size=(K,1))+SN*rng.normal(size=(1,N))
def quad(d):
 # Eigenvalues of sigma_p² I + sigma_r² (I_K⊗J_N) + sigma_n² (J_K⊗I_N).
 g=d.mean((-1,-2),keepdim=True);r=d.mean(-1,keepdim=True)-g;c=d.mean(-2,keepdim=True)-g;z=d-r-c-g
 return z.square().sum((-1,-2))/SP**2+r.square().sum((-1,-2))*N/(SP**2+N*SR**2)+c.square().sum((-1,-2))*K/(SP**2+K*SN**2)+g.square().sum((-1,-2))*K*N/(SP**2+N*SR**2+K*SN**2)
def logp(a,m):return -.5*quad(a-m) # constant independent of policy
class Decode:
 def __init__(self,e):
  K,N=e.K,e.N;A=lil_matrix((K+2*N,K*N));lo=np.r_[np.ones(K),np.full(2*N,-np.inf)];hi=np.r_[np.full(K,np.inf),e.capacity[:,0],e.capacity[:,1]]
  for k in range(K):
   A[k,k*N:(k+1)*N]=1
   for n in range(N):A[K+n,k*N+n]=e.spec[k,0];A[K+N+n,k*N+n]=e.spec[k,1]
  self.cons=LinearConstraint(A.tocsr(),lo,hi);self.bounds=Bounds(0,1);self.integ=np.ones(K*N)
 def __call__(self,e,W):
  t=time.monotonic()
  for lim in (.15,1.,5.):
   q=milp(-W.ravel(),integrality=self.integ,bounds=self.bounds,constraints=self.cons,options={'time_limit':lim,'mip_rel_gap':.03})
   if q.x is not None:
    try:X=valid(e,np.rint(q.x).reshape(K,N).astype(bool));return X,dict(solver_gap=float(q.mip_gap),solver_status=int(q.status),solver_seconds=time.monotonic()-t)
    except AssertionError:pass
  raise RuntimeError('no legal MILP solution')
def physical(e,s,X,ps,gs):
 before=hexmat(e.old);_,r,_,_,_=e.commit(valid(e,X),s.scheduler(rng_for(ps,2000),0.),ps,gs,False);r.update(matrix_hex=hexmat(X),old_hex=before);return r

def world(seed):
 c,x,e,s,old=setup('finish_pf',seed);p=torch.load(P2OUT/f'PF_learn_{seed}'/'latest.pt',weights_only=False,map_location='cpu');s.load_state_dict(p['s']);assert digest(s.online)==p['policy_hash'];return c,x,e,s,old

def eval_states(c,x,old):
 # two independent natural states, two historical old-layout variants, two high-load shock states.
 r30=read_core(ROOT/'../runs/pairs_balanced_20261114/latest.pt')['env'].old.copy()
 for name,ds,phase,ps,M in [('normal0',81200,0,92700,old),('normal3_r30',81203,3,92703,r30),('hot13',94313,1,95913,old),('hot17_r30',94317,5,95917,r30)]:
  e=environment(c,x);e.reset(ds,phase);e.old=M.copy();yield name,e,ps
 for gs in (41,44):
  e=environment(c,x);e.reset(20261114+88000,0);e.old=old.copy()
  for _ in range(gs-1):e.w.advance();e.period+=1
  ps=int(rng_for(20261114,900,gs).integers(0,2**31-1));yield f'shock{gs}',e,ps

def evaluate(p,s,c,x,old,seed,gs,out):
 q=[]
 for j,(name,e,ps) in enumerate(eval_states(c,x,old)):
  dec=Decode(e);mu=p(features(e)).detach().numpy();rng=np.random.default_rng(810000+j+seed)
  for k in range(3):
   W=mu if k==0 else sample(mu,rng);X,di=dec(e,W);r=physical(copy.deepcopy(e),s,X,ps,1);r.update(gs=gs,seed=seed,state=name,index=k,mode='mean' if k==0 else 'sample',policy_hash=digest(p),**di);write(out/'eval.jsonl',r);q.append(r)
 print(seed,'EVAL',gs,'mean',round(np.mean([r['reward'] for r in q if r['index']==0]),4),'sample',round(np.mean([r['reward'] for r in q if r['index']!=0]),4),flush=True);return float(np.mean([r['reward'] for r in q if r['index']!=0]))

def run(seed,arm,steps):
 c,x,e,s,old=world(seed);torch.manual_seed(seed+17);p=Policy();opt=torch.optim.Adam(p.parameters(),lr=.001);dec=Decode(e);out=OUT/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=False);e.reset(seed+88000,0);e.old=old.copy();rng=np.random.default_rng(seed+771);initial=digest(p);sh=digest(s.online);best=-np.inf;batch=[];updates=0
 def save(name,gs):torch.save(dict(actor=p.state_dict(),opt=opt.state_dict(),env=e,gs=gs,seed=seed,arm=arm,updates=updates,rng=rng.bit_generator.state,s_hash=sh,policy_hash=digest(p)),out/name)
 if arm=='learn':best=evaluate(p,s,c,x,old,seed,0,out);save('best.pt',0)
 elif arm in ('antithetic','preconditioned'):
  # Identical initial actor and independent noise evaluation; reuse IID GS0 reference, not new physical rows.
  q=[json.loads(l) for l in (OUT/f'learn_{seed}'/'eval.jsonl').read_text().splitlines()];q=[r for r in q if r['gs']==0 and r['index']!=0];assert len(q)==12 and all(r['policy_hash']==initial for r in q);best=float(np.mean([r['reward'] for r in q]));save('best.pt',0)
 else:save('initial.pt',0)
 for gs in range(1,steps+1):
  f=features(e);mu=p(f).detach().numpy();actions=[sample(mu,rng) for _ in range(4)]
  if arm=='antithetic':actions=[actions[0],2*mu-actions[0],actions[2],2*mu-actions[2]]
  base=copy.deepcopy(e);ps=int(rng_for(seed,900,gs).integers(0,2**31-1));rows=[]
  for i,W in enumerate(actions):
   env=e if i==0 else copy.deepcopy(base);X,di=dec(env,W);r=physical(env,s,X,ps,gs);r.update(gs=gs,seed=seed,arm=arm,index=i,policy_hash=digest(p),**di);write(out/'train.jsonl',r);rows.append(r)
  assert len({r['trace_sha256'] for r in rows})==1 and len({r['old_hex'] for r in rows})==1
  rewards=np.array([r['reward'] for r in rows]);adv0=.5*(rewards-rewards[[1,0,3,2]]) if arm=='antithetic' else rewards-(rewards.sum()-rewards)/3;batch.append((f,torch.tensor(np.stack(actions),dtype=torch.float32),torch.tensor(.1*adv0,dtype=torch.float32)))
  if gs%2==0:
   if arm!='frozen':
    fs=torch.stack([b[0] for b in batch]);a=torch.stack([b[1] for b in batch]);adv=torch.stack([b[2] for b in batch]);mu0=p(fs);loss=-(logp(a,mu0[:,None])*adv).mean() if arm!='preconditioned' else -(adv[...,None,None]*(a-mu0.detach()[:,None])*mu0[:,None]).sum((-1,-2)).mean();opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(p.parameters(),5.));opt.step();updates+=1;kl=float(.5*quad(p(fs).detach()-mu0.detach()).mean());write(out/'updates.jsonl',dict(gs=gs,updates=updates,loss=float(loss.detach()),grad_norm=gn,kl=kl,adv_std=float(adv.std()),policy_hash=digest(p)))
   batch=[]
  print(seed,arm,gs,round(rows[0]['reward'],3),round(rows[0]['penalized_delay'],3),rows[0]['failed'],round(float(rewards.mean()),3),flush=True)
  if arm!='frozen' and gs in (8,16,32,48,64):
   score=evaluate(p,s,c,x,old,seed,gs,out);save(f'gs{gs}.pt',gs)
   if score>best:best=score;save('best.pt',gs)
  if gs%8==0:save('latest.pt',gs)
 assert digest(s.online)==sh
 if arm=='frozen':assert digest(p)==initial
 save('latest.pt',steps);(out/'completed.json').write_text(json.dumps(dict(gs=steps,updates=updates,initial_hash=initial,final_hash=digest(p),s_hash=sh,best=best if arm!='frozen' else None),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['learn','frozen','antithetic','preconditioned'],required=True);p.add_argument('--steps',type=int,default=64);a=p.parse_args();OUT.mkdir(exist_ok=True);run(a.seed,a.arm,a.steps)

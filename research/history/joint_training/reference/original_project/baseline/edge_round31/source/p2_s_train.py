"""P2 fixed-placement S training, matched frozen and old CPU controls. No old replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,pickle,gzip
from pathlib import Path
import numpy as np,torch
from train import config,environment,rng_for,digest,write,read_core
from edge_rl.agents import SchedulingAgent
from edge_rl.replay import SReplay
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'p2_results';CFG=ROOT/'configs/experiment_p1_finish_pf.json'
def setup(mode,seed):
 c,x=config(CFG,seed);c.cpu_allocator=mode;x['cpu_allocator']=mode;torch.set_num_threads(1);torch.manual_seed(seed);e=environment(c,x);s=SchedulingAgent(c);old=read_core(ROOT/'../reference/round25_seed2_128_core.pt')['env'].old.copy();return c,x,e,s,old

def layouts(e,old):
 r30=read_core(ROOT/'../runs/pairs_balanced_20261114/latest.pt')['env'].old.copy();e.reset(94313,1);need=e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0];cand=[(float(need[k]/e.w.nodes[n,0]),k,int(n)) for k in range(e.K) for n in np.flatnonzero(old[k]) if old[k].sum()>1];res={'R25':old,'R30':r30}
 for target in (1.05,1.25):
  rho,k,n=min(cand,key=lambda t:abs(t[0]-target));m=old.copy();m[k]=False;m[k,n]=True;res[f'forced_{target}']=m
 for m in res.values():assert m.any(1).all() and np.all(m.T@e.spec<=e.capacity+1e-7)
 return res
STATES=[('normal0',81200,0,92700,'R25'),('normal0_r30',81200,0,92700,'R30'),('normal3',81203,3,92703,'R25'),('normal3_r30',81203,3,92703,'R30'),('hot13',94313,1,95913,'R25'),('hot13_light',94313,1,95913,'forced_1.05'),('hot13_heavy',94313,1,95913,'forced_1.25'),('hot17',94317,5,95917,'R25')]
def evaluate(c,x,s,old,mode,gs,seed,out,tag):
 e=environment(c,x);ms=layouts(e,old);rs=[]
 for name,ds,phase,ps,layout in STATES:
  e.reset(ds,phase);M=ms[layout];e.old=M.copy();_,r,_,_,_=e.commit(M,s.scheduler(rng_for(ps,2000),0.),ps,1,False);r.update(gs=gs,seed=seed,arm=tag,mode=mode,state=name,layout=layout,policy_hash=digest(s.online));write(out,r);rs.append(r)
 return rs

def checkpoint(path,s,e,replay,gs,seed,mode,arm,x,full=False):
 p=dict(s=s.state_dict(),gs=gs,seed=seed,mode=mode,arm=arm,config=x,env=e,policy_hash=digest(s.online),replay_count=len(replay));tmp=path.with_suffix('.tmp');torch.save(p,tmp);tmp.replace(path)
 if full:
  with gzip.open(path.parent/'latest_replay.pkl.gz.tmp','wb',compresslevel=1) as f:pickle.dump(replay,f,protocol=5)
  (path.parent/'latest_replay.pkl.gz.tmp').replace(path.parent/'latest_replay.pkl.gz')

def run(seed,arm,steps):
 mode='kkt' if arm=='KKT_learn' else 'finish_pf';learn=arm!='PF_frozen';c,x,e,s,old=setup(mode,seed);out=OUT/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=False);initial=digest(s.online);replay=SReplay(c.s_replay_transitions,c.replay_max_age);e.reset(seed+88000,0);e.old=old.copy();best=-float('inf');start=time.monotonic()
 def ev(gs):
  nonlocal best
  rows=evaluate(c,x,s,old,mode,gs,seed,out/'eval.jsonl',arm);score=float(np.mean([r['reward'] for r in rows]));print(arm,seed,'EVAL',gs,score,flush=True)
  if score>best:best=score;checkpoint(out/'best.pt',s,e,replay,gs,seed,mode,arm,x)
 if learn:ev(0)
 else:checkpoint(out/'initial.pt',s,e,replay,0,seed,mode,arm,x)
 for gs in range(1,steps+1):
  eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-(gs-1)/c.epsilon_decay_steps);ps=int(rng_for(seed,900,gs).integers(0,2**31-1));e.current_s_version=s.version;t=time.monotonic();_,r,ts,_,_=e.commit(old,s.scheduler(rng_for(seed,901,gs),eps),ps,gs,learn);u=[]
  if learn:
   assert ts and all(t.mask[t.action] and t.collection_step==gs and np.isfinite(t.reward) for t in ts);replay.add(ts);replay.prune(gs)
   for ui in range(x['s_updates_per_period']):u.append(s.update(replay.sample(x['s_batch'],gs,rng_for(seed,902,gs,ui))))
   s.version+=1
  r.update(gs=gs,arm=arm,seed=seed,mode=mode,epsilon=eps,s_updates=s.updates,s_version=s.version,collected=len(ts),replay_count=len(replay),policy_hash=digest(s.online),loss=float(np.mean([a['s_loss'] for a in u])) if u else None,grad_norm=float(np.mean([a['s_grad_norm'] for a in u])) if u else None,wall_s=time.monotonic()-t);write(out/'train.jsonl',r);print(arm,seed,gs,round(r['reward'],4),round(r['penalized_delay'],4),r['failed'],flush=True)
  if learn and gs in (16,32,48,64):ev(gs)
 if not learn:assert digest(s.online)==initial
 checkpoint(out/'latest.pt',s,e,replay,steps,seed,mode,arm,x,full=learn)
 (out/'completed.json').write_text(json.dumps(dict(gs=steps,updates=s.updates,seed=seed,arm=arm,mode=mode,initial_hash=initial,final_hash=digest(s.online),wall_s=time.monotonic()-start,best_eval=best if learn else None),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261114);p.add_argument('--arm',choices=['PF_learn','PF_frozen','KKT_learn','mature'],required=True);p.add_argument('--steps',type=int,default=64);a=p.parse_args();OUT.mkdir(exist_ok=True)
 if a.arm=='mature':
  for mode in ('kkt','finish_pf'):
   c,x,e,s,old=setup(mode,a.seed);s.load_state_dict(read_core(ROOT/'../reference/round25_seed2_128_core.pt')['s']);evaluate(c,x,s,old,mode,0,a.seed,OUT/'mature.jsonl','mature')
 else:run(a.seed,a.arm,a.steps)

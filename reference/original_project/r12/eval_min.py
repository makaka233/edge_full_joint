"""Frozen natural Markov evaluation; NEVER used for best checkpoint selection."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import r2b_joint as b
from r12 import joint as q

ROOT=Path(__file__).resolve().parent
INDEPENDENT=(160003,160029,160071,160109)

def load(seed,mode,model):
 out=ROOT/'experiments'/mode/'r51/results'/f'{seed}_D'
 path=out/(model+'.pt')
 if not path.exists():raise FileNotFoundError(f'No checkpoint {path}. Run train first.')
 ck=torch.load(path,map_location='cpu',weights_only=False)
 c,x,_,s,p=q.setup(seed,'D');p.load_state_dict(ck['d']);s.load_state_dict(ck['s'])
 assert b.digest(p)==ck['d_hash'] and b.digest(s.online)==ck['s_hash']
 return c,x,s,p,ck['gs']

def run(seed,mode,model):
 c,x,s,p,gs0=load(seed,mode,model)
 out=ROOT/'independent'/f'{seed}_{mode}_{model}_gs{gs0}.jsonl'
 out.parent.mkdir(exist_ok=True,parents=True)
 if out.exists():raise FileExistsError(f'Independent evaluation already exists: {out}')
 rows=[]
 for ds in INDEPENDENT:
  e=b.environment(c,x);e.reset(ds,ds%12);rng=b.rng_for(seed,160011,ds)
  for gs in range(1,9):
   tic=time.monotonic();o,a,X,_=b.execute(e,p,b.Decode(e),rng)
   ps=int(b.rng_for(ds,160055,gs).integers(0,2**31-1))
   ph=q.physical(e,o,a,X)
   r,_=q.rec(e,s,p,X,a,ps,gs,0,'A')
   row=dict(seed=seed,arm=mode,model=model,checkpoint_gs=gs0,demand_seed=ds,gs=gs,
       reward=float(r['reward']),e2e=float(r['penalized_delay']),n=int(r['n']),
       failed=int(r['failed']),holding=float(r['holding_cost']),switch=float(r['switch_cost']),
       replicas=int(X.sum()),action_mode=int(a['mode']),matrix_hex=b.hexmat(X),
       trace_sha256=r['trace_sha256'],wall_s=time.monotonic()-tic,projected_cpu_pressure=ph['pressure'])
   with out.open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   rows.append(row)
  print('INDEPENDENT',seed,mode,model,gs0,ds,flush=True)
 summary=dict(seed=seed,arm=mode,model=model,gs=gs0,eval_gs=len(rows),n=sum(t['n'] for t in rows),
    reward_per_gs=float(np.mean([t['reward'] for t in rows])),
    request_weighted_e2e=float(sum(t['e2e']*t['n'] for t in rows)/sum(t['n'] for t in rows)),
    unfinished=int(sum(t['failed'] for t in rows)),
    builds=int(sum(t['action_mode']==1 for t in rows)))
 out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True)
 a.add_argument('--arm',choices=['A','S'],default='A')
 a.add_argument('--model',choices=['best','latest'],default='latest')
 v=a.parse_args();run(v.seed,v.arm,v.model)

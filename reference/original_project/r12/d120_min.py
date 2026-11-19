"""Small authentic D120 regression for a frozen checkpoint; not training credit."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import r2b_joint as b
from r12 import joint as q
from r12.eval_min import load
ROOT=Path(__file__).resolve().parent

def run(seed,arm,model,periods):
 c,x,s,p,ck=load(seed,arm,model)
 out=ROOT/'d120'/f'{seed}_{arm}_{model}_gs{ck}.jsonl'
 out.parent.mkdir(exist_ok=True,parents=True)
 if out.exists():raise FileExistsError(out)
 ds=171003;e=b.environment(c,x);e.reset(ds,0);rng=b.rng_for(seed,171011,ds)
 for gs in range(1,periods+1):
  start=time.monotonic();o,a,X,_=b.execute(e,p,b.Decode(e),rng)
  ps=int(b.rng_for(ds,171055,gs).integers(0,2**31-1))
  eighty=q.measure(e,s,X,ps,0,duration=80)
  rr,_=q.rec(e,s,p,X,a,ps,gs,0,'D120')
  row=dict(seed=seed,arm=arm,model=model,checkpoint_gs=ck,eval_gs=gs,demand_seed=ds,
       d80_e2e=eighty['e2e'],d80_failed=eighty['failed'],d80_n=eighty['n'],
       d120_e2e=rr['d_penalized_delay'],d120_failed=rr['d_failed'],d120_n=rr['d_n'],
       d120_credit_reward=rr['d_reward'],d120_main_short_reward=rr['reward'],
       holding=rr['holding_cost'],switch=rr['switch_cost'],
       action_mode=int(a['mode']),replicas=int(X.sum()),matrix_hex=b.hexmat(X),
       wall_s=time.monotonic()-start)
  with out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print('D120',seed,arm,model,gs,round(row['d120_e2e'],4),row['d120_failed'],flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True)
 a.add_argument('--arm',choices=['A','S'],default='A');a.add_argument('--model',choices=['best','latest'],default='latest')
 a.add_argument('--periods',type=int,default=2);v=a.parse_args()
 assert 1<=v.periods<=8;run(v.seed,v.arm,v.model,v.periods)

"""Compare learned S with heuristic S, keeping the learned D and exogenous traces fixed."""
import argparse,json
from pathlib import Path
import torch
from checkpoint import read_core
from train import config,environment,agents,reset,rollout,summary
from edge_rl.physics import HeuristicScheduler

class LocalOnly:
    def __init__(self,s):self.s=s
    def scheduler(self,rng,epsilon=0.):
        @torch.no_grad()
        def act(obs,mask):
            q=self.s.local_online(torch.as_tensor(obs[None],device=self.s.device));m=torch.as_tensor(mask[None],device=self.s.device)
            return int(q.masked_fill(~m,-1e9).argmax(1).item())
        return act

class Heuristic:
    def __init__(self,w):self.w=w
    def scheduler(self,rng,epsilon=0.):return HeuristicScheduler(self.w)

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--out',required=True);p.add_argument('--eval-seed',type=int,default=171703);p.add_argument('--steps',type=int,default=8);p.add_argument('--local-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);ck=read_core(a.policy);name='experiment_round17_rate_weighted.json' if ck['config']['cost_model'].get('latency_objective')=='rate_weighted' else 'experiment_round17_centered.json' if ck['config'].get('s_center_local_advantage') else 'experiment_round17_residual.json';c,x=config('configs/'+name,ck['config']['seed']);assert x==ck['config'];d,s=agents(c,x,environment(c,x));d.load_state_dict(ck['d']);s.load_state_dict(ck['s']);arms={};raw={}
    for key in (('local_only',) if a.local_only else ('learned','heuristic','local_only')):
        e=environment(c,x);reset(e,a.eval_seed,0);sched=Heuristic(e.w) if key=='heuristic' else LocalOnly(s) if key=='local_only' else s;rows=[]
        for gs in range(1,a.steps+1):
            _,info,_,_=rollout(e,d,sched,a.eval_seed,gs,0.,False,False,x['reward_scale']);rows.append(info)
        arms[key]=summary(rows);raw[key]=rows;print(key,arms[key],flush=True)
    reference=next(iter(raw.values()));assert all([(z['trace_sha256'],z['matrix_hex']) for z in v]==[(z['trace_sha256'],z['matrix_hex']) for z in reference] for v in raw.values())
    Path(a.out).write_text(json.dumps(dict(trained_gs=ck['gs'],seed=a.eval_seed,steps=a.steps,arms=arms,trace_hashes=[z['trace_sha256'] for z in reference]),indent=2))
if __name__=='__main__':main()

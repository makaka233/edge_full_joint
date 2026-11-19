"""Isolated counterfactual scheduler credit experiment; no live action is replaced."""
import argparse,json,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from train import config,environment,agents,reset,rng_for,rollout,summary
from checkpoint import read_core
from edge_rl.physics import Simulator
from evaluate_round17_scheduler import LocalOnly

def window(e,m,b,arr,policy,change=None,record=False):
    rows=[];idx=0;focal=[None];end=b+e.window;limit=end+e.tail
    def act(obs,mask):
        nonlocal idx
        a=policy(obs,mask)
        if change is not None and idx==change[0]:
            assert np.array_equal(obs,change[3]) and np.array_equal(mask,change[4]) and a==change[1]
            a=change[2]
        if record:rows.append((idx,obs.copy(),mask.copy(),a,sim.now,focal[0]))
        idx+=1;return a
    sim=Simulator(e.w,m,act,b-e.warm,0,0,False,cohort=(b,end,limit))
    ready=sim.ready
    def tagged(r,previous=None):
        focal[0]=r.id;return ready(r,previous)
    sim.ready=tagged
    sim.add_arrivals(arr);sim.run_until(b);sim.run_until(end)
    while sim.now<limit-1e-8 and sim.result(b,end)['censored']:sim.run_until(min(limit,sim.now+1.))
    r=sim.result(b,end);own={q.id:(q.finish-q.arrival.time if q.finish is not None else sim.now-q.arrival.time+e.failure_penalty) for q in sim.requests if b<=q.arrival.time<end};return r['delay_lower_bound_sum']+e.failure_penalty*r['censored'],r['arrivals'],r['censored'],rows,own

def fit(s,pairs,rng,updates=128,external=False):
    losses=[]
    for _ in range(updates):
        rr=[pairs[i] for i in rng.integers(len(pairs),size=min(32,len(pairs)))];x=torch.tensor(np.stack([r[0] for r in rr]));a=torch.tensor([r[1] for r in rr]);b=torch.tensor([r[2] for r in rr]);y=torch.tensor([r[4] if external else r[3] for r in rr],dtype=torch.float32)
        q=s.online(x) if external else s.team_q(x);delta=q.gather(1,a[:,None]).squeeze(1)-q.gather(1,b[:,None]).squeeze(1);l=F.smooth_l1_loss(delta,y) + (0.05*(q-q.mean(1,keepdim=True)).square().mean() if external else 0.)
        s.optimizer.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(s.online.parameters(),10,error_if_nonfinite=True);s.optimizer.step();losses.append(float(l.detach()))
    return losses

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--periods',type=int,default=4);p.add_argument('--pairs-per-window',type=int,default=4);p.add_argument('--eval-steps',type=int,default=4);p.add_argument('--seed',type=int,default=181701);a=p.parse_args();torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=read_core(a.checkpoint);c,x=config('configs/experiment_round17_residual.json',ck['config']['seed']);assert x==ck['config'];d,s=agents(c,x,environment(c,x));d.load_state_dict(ck['d']);s.load_state_dict(ck['s']);old=copy.deepcopy(s)
    with torch.no_grad():
        for head in (s.online.advantage,s.online.value):head[-1].weight.zero_();head[-1].bias.zero_()
    s.optimizer=torch.optim.Adam(s.online.parameters(),lr=.0003)
    e=environment(c,x);reset(e,a.seed,0);pairs=[];rng=rng_for(a.seed,81);records=[]
    for gs in range(1,a.periods+1):
        pre=copy.deepcopy(e);_,info,_,_=rollout(e,d,LocalOnly(old),a.seed,gs,0.,False,False);m=e.old.copy();qseed=int(rng_for(a.seed,90,gs).integers(0,2**31-1));traces,h=pre.trace(qseed);assert h==info['trace_sha256'];n=sum(sum(b<=r.time<b+pre.window for r in arr) for b,arr in traces);ref=pre.cost['reference_rate']*pre.windows*pre.window
        for b,arr in traces:
            policy=LocalOnly(old).scheduler(None);cost,num,fail,rows,own=window(pre,m,b,arr,policy,record=True);eligible=[r for r in rows if r[4]>=b and r[2].sum()>1 and r[5] in own]
            for ix in rng.choice(len(eligible),min(len(eligible),a.pairs_per_window),replace=False):
                i,obs,mask,action,t,rid=eligible[int(ix)];alt=int(rng.choice(np.flatnonzero(mask & (np.arange(len(mask))!=action))))
                v,nn,ff,_,altown=window(pre,m,b,arr,policy,(i,action,alt,obs,mask));assert nn==num
                assert own.keys()==altown.keys(); own_delta=altown[rid]-own[rid];target=(v-cost)*ref/max(1,n); external=(v-cost-own_delta)*ref/max(1,n);pairs.append((obs,action,alt,target,external));records.append(dict(gs=gs,decision=i,action=action,alternative=alt,delta=target,own_delta=own_delta,external_delta=external,baseline_cost=cost,alternative_cost=v,n=num,failed_before=fail,failed_after=ff,trace_sha256=h))
        print('PAIRS',gs,len(pairs),flush=True)
    direct=copy.deepcopy(s);losses=fit(direct,pairs,rng);extloss=fit(s,pairs,rng,external=True);(out/'pairs.json').write_text(json.dumps(records,indent=2));torch.save(dict(s=s.state_dict(),d=d.state_dict(),config=x,gs=ck['gs'],source=str(a.checkpoint)),out/'paired_policy.pt')
    arms={};raw={}
    for name,pol in [('round17_team',old),('local_only',LocalOnly(old)),('paired_team',direct),('external_team',s)]:
        ev=environment(c,x);reset(ev,a.seed+10000,0);rs=[]
        for gs in range(1,a.eval_steps+1):
            _,info,_,_=rollout(ev,d,pol,a.seed+10000,gs,0.,False,False);rs.append(info)
        arms[name]=summary(rs);raw[name]=[(r['trace_sha256'],r['matrix_hex']) for r in rs];print(name,arms[name],flush=True)
    assert all(v==raw['local_only'] for v in raw.values());(out/'result.json').write_text(json.dumps(dict(seed=a.seed,source_gs=ck['gs'],pairs=len(pairs),loss_first=losses[0],loss_last=losses[-1],external_loss_first=extloss[0],external_loss_last=extloss[-1],arms=arms,trace_hashes=[v[0] for v in raw['local_only']]),indent=2))
if __name__=='__main__':main()

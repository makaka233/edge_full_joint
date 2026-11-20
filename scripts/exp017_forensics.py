"""Recreate EXP016 early policies, instrument actual decisions, and follow the shadow drain."""
import argparse
import base64
from copy import copy,deepcopy
from functools import partial
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from edge_d._engine import environment
from edge_d._engine.edge_rl.physics import Simulator
from edge_d.placement import decode
from edge_joint.exploration import ARMS,EarlyExplorer,probabilities,settings
from edge_joint.inventory_joint import Config,SEEDS,fingerprint,initialize,parameters,physical,step,world
from scripts.exp009 import rows,write_json


CASES = {20261113: {1:'train'}, 20261114:{2:'train',3:'train',7:'antithetic'}}
BASE = Path('research/EXP016/raw')
CHECK = ('matrix_packed_hex','trace_sha256','nested_trace_hash','s_policy_fingerprint',
         's_version','reward','censored','s_decisions')


def base_rows(seed,role):
    return rows(BASE/f'seed{seed}'/f'{role}.jsonl.gz')


def prepare(seed,out):
    out.mkdir(parents=True,exist_ok=False)
    cfg=Config(seed)
    policy,s,optimizer,replay=initialize(cfg)
    base={k:base_rows(seed,k) for k in ('train','antithetic')}
    matched=[]
    for gs in range(1,max(CASES[seed])+1):
        if gs in CASES[seed]:
            torch.save(dict(s=s.online.state_dict(),version=s.version),out/f's{gs}.pt')
        main,other,update=step(cfg,policy,s,optimizer,replay,gs)
        for role,actual in (('train',main),('antithetic',other)):
            expected=base[role][gs-1]
            if any(actual[k]!=expected[k] for k in CHECK):
                raise RuntimeError(f'EXP016 prefix differs at GS{gs}/{role}')
        matched.append(dict(gs=gs,online=main['reward'],antithetic=other['reward'],s_updates=update['s_updates']))
    write_json(out/'prefix.json',matched)


class ForensicAgent(EarlyExplorer):
    def scheduler(self,rng,epsilon=0.):
        self.rng=rng
        return super().scheduler(rng,epsilon)


class TapeSimulator(Simulator):
    def __init__(self,*args,tape,agent,**kwargs):
        super().__init__(*args,trace=True,**kwargs)
        self.phase='official'
        self.agent=agent
        self.tape=tape
        self.behavior=self.scheduler
        self.scheduler=self.observe_action
        self.deadline=self.cohort[2]

    def ready(self,task,previous=None):
        self.ready_task=task
        super().ready(task,previous)

    @torch.no_grad()
    def observe_action(self,obs,mask):
        action=self.behavior(obs,mask)
        q=self.agent.team_q(torch.as_tensor(obs[None],device=self.agent.device))[0].numpy()
        valid=np.flatnonzero(mask)
        task=self.ready_task
        eps,weight,temp=self.agent.active_settings
        probabilities_at_state=probabilities(q[valid].astype(float),eps,weight,temp)
        details=dict(window=self.window_index,phase=self.phase,time=self.now,request=task.id,
                     stage_index=task.index,stage=int(self.c.services[task.arrival.service][task.index]),
                     source=task.source,chosen=action,greedy=int(valid[q[valid].argmax()]),legal=valid.tolist(),
                     q_f32le=base64.b64encode(q.astype('<f4').tobytes()).decode(),
                     chosen_probability=float(probabilities_at_state[np.flatnonzero(valid==action)[0]]),
                     active_requests=self.active_requests,
                     cpu_jobs=[len(r.jobs) for r in self.cpu],
                     wired_jobs=[[i,len(r.jobs)] for i,r in enumerate(self.wired) if r.jobs])
        self.tape.write(json.dumps(details,separators=(',',':'))+'\n')
        return action


def report_queue(sim,start,stop):
    stat=sim.result(start,stop)
    return dict(time=sim.now,active_requests=sim.active_requests,completed=stat['completed'],
                censored=stat['censored'],active_jobs=stat['active_jobs'],
                work_residual=stat['max_work_conservation_error'])


def shadow(windows,agent,out):
    timeline=[]
    with gzip.open(out/'requests.jsonl.gz','wt',encoding='utf8',compresslevel=4) as tape:
        for sim,start,stop,rng_state in windows:
            agent.rng.bit_generator.state=rng_state
            tasks=[q for q in sim.requests if start<=q.arrival.time<stop]
            official={q.id:q.finish for q in tasks}
            origin=sim.now
            ledger=[report_queue(sim,start,stop)]
            sim.phase='shadow'
            for seconds in (1.,2.,4.,8.,16.,32.,64.):
                if not ledger[-1]['censored']:
                    break
                sim.run_until(origin+seconds)
                ledger.append(report_queue(sim,start,stop))
            timeline.append(dict(window=sim.window_index,official_stop=origin,steps=ledger))
            for request in sim.requests:
                in_cohort=request.id in official
                valid_finish=official.get(request.id)
                row=dict(window=sim.window_index,cohort=in_cohort,request=request.id,user=request.arrival.user,
                         service=request.arrival.service,arrival=request.arrival.time,
                         official_finish=valid_finish,final_finish=request.finish,
                         official_censored=in_cohort and valid_finish is None,
                         final_censored=in_cohort and request.finish is None,events=request.trace)
                tape.write(json.dumps(row,separators=(',',':'))+'\n')
    return timeline


def run(seed,gs,arm,checkpoint,out):
    out.mkdir(parents=True,exist_ok=False)
    role=CASES[seed][gs]
    original=base_rows(seed,role)[gs-1]
    cfg=Config(seed)
    env=world(cfg,original['scene'],'train')
    matrix=decode(env,np.array(original['latent'],dtype=np.float32),original['layout_seed'],True)
    s=ForensicAgent(seed,arm)
    saved=torch.load(checkpoint/f's{gs}.pt',weights_only=True)
    s.online.load_state_dict(saved['s']); s.version=saved['version']
    policy_id=fingerprint(parameters(s.online))
    if np.packbits(matrix.matrix).tobytes().hex()!=original['matrix_packed_hex'] or policy_id!=original['s_policy_fingerprint']:
        raise RuntimeError('Did not recover the registered deployment and S weights')
    windows=[]

    def hook(sim,start,stop):
        # The environment advances after evaluation. Keep this window's existing world.
        sim.w=copy(sim.w)
        windows.append((sim,start,stop,deepcopy(s.rng.bit_generator.state)))

    with gzip.open(out/'decisions.jsonl.gz','wt',encoding='utf8',compresslevel=4) as tape:
        with patch.object(environment,'Simulator',partial(TapeSimulator,tape=tape,agent=s)):
            row,_=physical(cfg,env,matrix,s,'train',original['scene'],gs,original['epsilon'],window_hook=hook)
        row['s_policy_fingerprint']=policy_id
        if not all(row[k]==original[k] for k in CHECK[:5]):
            raise RuntimeError('Counterfactual scenario was not matched')
        if arm=='A' and any(row[k]!=original[k] for k in CHECK):
            raise RuntimeError('The registered spike did not replay exactly')
        timeline=shadow(windows,s,out)
    write_json(out/'result.json',dict(seed=seed,registered_role=role,original_reward=original['reward'],
                                      exact_original=arm=='A',**row,shadow=timeline))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','case'))
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gs',type=int)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--checkpoints',type=Path)
    a=parser.parse_args()
    torch.set_num_threads(1)
    if a.action=='prepare':
        prepare(a.seed,a.out)
    else:
        if a.gs not in CASES[a.seed] or a.arm is None or a.checkpoints is None:
            parser.error('Provide a registered GS, an arm and recovered S snapshots')
        run(a.seed,a.gs,a.arm,a.checkpoints,a.out)

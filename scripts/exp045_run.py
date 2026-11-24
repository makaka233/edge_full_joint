"""EXP045 preregistered numerical control and bounded paired physical probes."""

import argparse
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json

import numpy as np
import torch
from torch import nn

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, run_lock
from scripts.exp044_gate import FOCAL, fit, fixed, gate_files, network, read_resume, replay_case
from scripts.exp044_run import ROOT as OLD, frozen, same_flow, write
from scripts.exp045_audit import ROOT, audit, numeric, stats
from scripts.exp043_run import ROOT as BASE, SEEDS


def register():
    evidence=json.loads((ROOT/'p0.audit.json').read_bytes())
    if len(evidence['probes'])!=12 or evidence['credit_max_abs_error']>1e-9:
        raise ValueError('Complete P0 audit before modifying Gate')
    parent=json.loads((OLD/'protocol.json').read_bytes())['plan']
    base=json.loads((BASE/'protocol.json').read_bytes())['plan']
    training=json.loads((OLD/'gate.protocol.json').read_bytes())['plan']
    plan=dict(experiment='EXP045',seeds=list(SEEDS),mean=evidence['calibration']['mean'],
              scale=np.maximum(evidence['calibration']['std'],.05).tolist(),floor=.05,
              calibration_n=evidence['calibration_n'],old_models=parent['model_sha256'],
              probes=[dict(seed=s,profile=name,interval=interval,loads=loads,
                           environment=stable_seed('EXP045','diagnostic',s,name,'population'),
                           scene=stable_seed('EXP045','diagnostic',s,name,'traffic'))
                      for s,(name,interval,loads) in product((20261113,20261116),
                          (('natural_young',1,[None]*24),('burst_old',8,[None]*11+[.70]*3+[None]*10)))],
              p1=dict(saturation_drop=.20,span=.10,min_seeds=3,gain_margin=.03,min_orientation=.50))
    file=ROOT/'protocol.json'
    if file.exists():
        saved=json.loads(file.read_bytes())
        if saved['plan']!=plan:
            raise ValueError('EXP045 registered calibration or cases changed')
    else:
        write(file,dict(plan=plan,source=source()))
    return plan,parent,base,training,evidence


class Standardize(nn.Module):
    def __init__(self,plan):
        super().__init__()
        self.register_buffer('mean',torch.tensor(plan['mean'],dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(plan['scale'],dtype=torch.float32))

    def forward(self,x):
        return (x-self.mean)/self.scale


class NormalizedGate(nn.Module):
    def __init__(self,plan):
        super().__init__()
        self.standardize=Standardize(plan)
        self.body=network()

    def forward(self,x):
        return self.body(self.standardize(x))


def make_gate(plan,arm):
    return network() if arm=='raw' else NormalizedGate(plan)


@torch.inference_mode()
def physical_probe(parent,base,case):
    torch.set_num_threads(1)
    file=ROOT/f'probe.{case["profile"]}.seed{case["seed"]}.json'
    if file.exists():
        data=json.loads(file.read_bytes())
        if data['case']!=case:
            raise ValueError('Stale EXP045 physical probe')
        return file.name
    model=frozen(base,parent,case['seed'],FOCAL)
    prefix,opening,_,_=replay_case(base,case,model,fixed(case['interval']),capture=True)
    extension=replace(model[0],periods=FOCAL+3),*model[1:]
    # The live opening already includes the load applied at this boundary.
    branch_case=dict(case,loads=[*case['loads'][:FOCAL-1],None,*case['loads'][FOCAL:]])
    arms=[]; scores=[]
    for action in (0,1):
        branch,_,lag,weight=replay_case(base,branch_case,extension,
                                        lambda x,p:(action if p==FOCAL else 0,None,False),opening=opening)
        scores.append(sum(.99**i*r['reward_raw'] for i,r in enumerate(branch[:3]))-weight*(lag or 0.)/.99)
        arms.append(branch)
    same_flow(*arms)
    d=prefix[-1]['timing']
    if arms[0][0]['timing']['state']!=d['state'] or arms[1][0]['timing']['state']!=d['state']:
        raise AssertionError('Physical probe pair did not share its actual starting state')
    write(file,dict(case=case,seed=case['seed'],profile=case['profile'],period=FOCAL,state=d['state'],age_hours=d['age_hours'],
                    gain=(scores[1]-scores[0])/3,branches=arms,source=source()))
    return file.name


@torch.no_grad()
def evaluate_states(net,probes):
    return [net(torch.tensor(p['state'],dtype=torch.float32)).tolist() for p in probes]


def same_data(plan,settings,evidence):
    """Keep the EXP044 rollout, replay sample order, initialization and optimizer fixed."""
    torch.set_num_threads(1)
    probes=[*evidence['probes'],*(json.loads((ROOT/f'probe.{c["profile"]}.seed{c["seed"]}.json').read_bytes())
                                for c in plan['probes'])]
    probes=[{k:p[k] for k in ('seed','profile','period','age_hours','state','gain')} for p in probes]
    outcomes={}
    for seed in SEEDS:
        old=read_resume(gate_files(seed)/'resume_latest.pt'); data=old['data']; out={}
        for arm in ('raw','normalized'):
            torch.manual_seed(stable_seed('EXP044','q-initial',seed))
            net=make_gate(plan,arm); target=deepcopy(net).requires_grad_(False)
            opt=torch.optim.Adam(net.parameters(),lr=settings['lr'])
            points=[p for p in probes if p['seed']==seed]
            initial=evaluate_states(net,points)
            metrics=[]
            for block in range(settings['blocks']):
                samples=data['samples'][:(block+1)*22]+data['auxiliary']
                rng=np.random.default_rng(stable_seed('EXP044',seed,block,'replay'))
                metrics.extend(fit(net,target,opt,samples,rng,settings))
            original=dict(net.state_dict())
            if arm=='raw' and not all(torch.equal(v,old['q'][k]) for k,v in original.items()):
                raise AssertionError(f'Identical EXP044 replay did not reproduce weights for seed {seed}')
            else:
                result_samples=samples
                body,body_target=net,target
                if arm=='normalized':
                    def scaled(x):
                        return net.standardize(torch.tensor(x,dtype=torch.float32)).tolist()
                    result_samples=[{**r,'x':scaled(r['x']),'next':scaled(r['next'])} for r in samples]
                    body,body_target=net.body,target.body
                probes_q=evaluate_states(net,points)
                out[arm]=dict(diagnostics=numeric(body,body_target,result_samples),initial_q=initial,
                              probes_q=probes_q,gradient=stats([m['gradient'] for m in metrics]),
                              clipped=sum(m['gradient']>5 for m in metrics),
                              td_update=stats([m['td_mae'] for m in metrics]),updates=metrics,
                              validation_q=evaluate_states(net,[r['timing'] for r in data['validation'][-1][1:]]),
                              state_response_span=float(np.ptp([q[1]-q[0] for q in probes_q])) if probes_q else None)
        outcomes[str(seed)]=dict(arms=out,probes=points)
    return outcomes


def verdict(plan,outcomes):
    g=plan['p1']; scores={}
    for arm in ('raw','normalized'):
        counts={-1:[0,0],1:[0,0]}
        for r in outcomes.values():
            for p,q in zip(r['probes'],r['arms'][arm]['probes_q'],strict=True):
                if abs(p['gain'])>g['gain_margin']:
                    sign=1 if p['gain']>0 else -1
                    counts[sign][0]+=int(sign*(q[1]-q[0])>0)
                    counts[sign][1]+=1
        orientation=float(np.mean([hits/n for hits,n in counts.values()])) if all(n for _,n in counts.values()) else None
        scores[arm]=dict(counts=counts,balanced_orientation=orientation)
    good=sum(r['arms']['raw']['diagnostics']['saturation']-r['arms']['normalized']['diagnostics']['saturation']
             >=g['saturation_drop'] and r['arms']['normalized']['diagnostics']['delta']['max']-
             r['arms']['normalized']['diagnostics']['delta']['min']>=g['span'] for r in outcomes.values())
    n=scores['normalized']['balanced_orientation']; raw=scores['raw']['balanced_orientation']
    advance=good>=g['min_seeds'] and n is not None and raw is not None and n>g['min_orientation'] and n>=raw
    return dict(advance=advance,numerical_seeds=good,physical_orientation=scores)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('audit','register','p1'),default='p1')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    if args.stage=='audit':
        audit(); return
    with run_lock(RunFiles(ROOT/'experiment')):
        plan,parent,base,settings,evidence=register()
        if args.stage=='register':return
        jobs(physical_probe,[(parent,base,c) for c in plan['probes']],min(args.workers,len(plan['probes'])))
        outcomes=same_data(plan,settings,evidence)
        result=verdict(plan,outcomes)
        write(ROOT/'p1.summary.json',dict(result=result,seeds=outcomes,source=source()))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

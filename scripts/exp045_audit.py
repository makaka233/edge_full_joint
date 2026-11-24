"""Reconcile EXP044 value learning against archived physical ledgers without rerunning simulation."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from edge_joint.inventory_joint import load_scale
from scripts.exp044_gate import case_env, gate_files, network, read_resume, transition
from scripts.exp044_run import ROOT as OLD, write


ROOT = Path('runs/results/EXP045')
NAMES = ('log offered requests / 4','log offered CPU ratio','existing deployment','log mean stage pressure',
         'log max stage pressure','log previous-slot demand drift','log active requests / 5',
         'log max wired clearing seconds','deployed matrix density','maximum RAM usage','maximum SSD usage',
         'clock sin','clock cos','log age slots / 3','log demand drift since activation',
         'log max CPU clearing seconds','holding cost')


def stats(data):
    x=np.asarray(data,dtype=float)
    return {k:v.tolist() for k,v in dict(min=x.min(0),max=x.max(0),mean=x.mean(0),std=x.std(0),
                p05=np.quantile(x,.05,axis=0),p50=np.quantile(x,.5,axis=0),p95=np.quantile(x,.95,axis=0)).items()}


def actual_credit(rows,gamma,weight):
    # t=0 is the forced initial deployment; t=T-1 has no observed successor.
    cost=np.asarray([r['holding_cost']+r['switch_cost'] for r in rows])
    late=np.asarray([weight*(r['e2e_penalized'] or 0.) for r in rows])
    t=np.arange(len(rows)); factor=gamma**t
    posted=-cost[1:-1]-late[:-2]/gamma
    original=float(sum(factor*np.asarray([r['reward_raw'] for r in rows])))
    remainder=-cost[0]-factor[-1]*cost[-1]-float(factor[-2:]@late[-2:])
    return posted, abs(original-(factor[1:-1]@posted+remainder))


def archived_probes(base):
    """Recover the old 17 causal inputs at the exact EXP044 P0 focal states."""
    result=[]
    for path in sorted(OLD.glob('p0.*.seed*.json')):
        data=json.loads(path.read_bytes()); case=data['case']; env=case_env(base,case)
        focal={p['period']:p for p in data['probes']}
        previous=anchor=None
        last=None
        for p,r in enumerate(data['fixed']['2']['rows'],1):
            load=case['loads'][p-1]
            if load is not None:
                env.w.scale*=load/load_scale(env)
            demand=env.w.node_demand()
            if not np.allclose(demand.sum(0),r['demand_by_service'],rtol=1e-10,atol=1e-10):
                raise AssertionError('Archived P0 demand reconstruction differs')
            if p in focal:
                old=data['fixed']['2']['rows'][p-2]; probe=focal[p]
                kept,changed=(probe['cells'][k]['rows'][0] for k in ('keep','redeploy'))
                if not kept['old_matrix']==changed['old_matrix']==old['active_matrix']:
                    raise AssertionError('Old matrix differs from physical fork')
                x=r['macro_state'].copy()
                x[5]=float(np.log1p(np.abs(demand-previous).sum()/previous.sum()))
                angle=2*np.pi*((p-1)%12)/12
                x[-1]=float(np.sin(angle))
                x += [float(np.cos(angle)),float(np.log1p(p-last)/3),
                      float(np.log1p(np.abs(demand-anchor).sum()/anchor.sum())),
                      float(np.log1p(max(n['work_at_start']/f for n,f in zip(r['cpu'],env.w.nodes[:,0],strict=True)))),
                      old['holding_cost']]
                result.append(dict(seed=case['seed'],profile=case['profile'],period=p,state=x,age_hours=2*(p-last),
                                   gain=probe['reward_delta'],opening=probe['opening'],
                                   proposal_change_bits=(int(kept['active_matrix'],16)^int(changed['active_matrix'],16)).bit_count(),
                                   proposal_switch_cost=changed['switch_cost']))
            if p==1 or r['activated_at'] is not None:
                anchor=demand.copy(); last=p
            previous=demand
            env.w.advance()
            env.w.t=r['end']
    return result


@torch.no_grad()
def numeric(net,target,samples):
    x=torch.tensor([r['x'] for r in samples]); nx=torch.tensor([r['next'] for r in samples]); q=net(x)
    z=net[0](x); h=net[1](z); delta=q[:,1]-q[:,0]
    a=torch.tensor([r['action'] for r in samples]); r=torch.tensor([r['reward'] for r in samples])
    target_q=target(nx)
    y=r+.99*target_q.gather(1,net(nx).argmax(1,keepdim=True)).squeeze(1)
    td=y-q.gather(1,a[:,None]).squeeze(1)
    dqdx=((net[2].weight[1]-net[2].weight[0])*(1-h.square())) @ net[0].weight
    return dict(q=stats(q),delta=stats(delta),pre_activation=stats(z.flatten()),hidden_std=stats(h.std(0,unbiased=False)),
                saturation=float((h.abs()>.99).float().mean()),input_gradient=stats(dqdx.abs()),
                common_offset_ratio=float(z.mean(0).norm()/z.std(0,unbiased=False).norm()),
                target=stats(y),td=stats(td),target_online_gap=stats((target_q-net(nx)).abs()),lagged_rewards=stats(r))


def audit():
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('runs/results/EXP043/protocol.json').read_bytes())['plan']
    plan=json.loads((OLD/'gate.protocol.json').read_bytes())['plan']
    gamma=plan['gamma']
    env=case_env(base,plan['training']['20261113'][0]); weight=env.cost['delay_weight']
    if not 0<env.tail<env.w.c.period_seconds:
        raise AssertionError('Lag-one cost publication is invalid for this cutoff')
    probes=archived_probes(base)
    inputs=[]; calibration=[]; groups={}; evidence={}; max_residual=0.; samples_count=0
    for key,cases in plan['training'].items():
        seed=int(key); saved=read_resume(gate_files(seed)/'resume_latest.pt'); d=saved['data']
        net=network(); target=network(); net.load_state_dict(saved['q']); target.load_state_dict(saved['target'])
        s=d['samples']; offset=0; visits={}
        for b,block in enumerate(d['blocks']):
            rows=block['rows']
            if rows[0]['action']!='BOOTSTRAP' or rows[0]['timing'] is not None or len(rows)!=24:
                raise AssertionError('Unexpected forced bootstrap or trajectory length')
            posted,residual=actual_credit(rows,gamma,weight); max_residual=max(max_residual,residual)
            if residual>1e-9:
                raise AssertionError('Discounted cost was lost or duplicated')
            batch=s[offset:offset+len(posted)]; offset+=len(posted)
            for j,entry in enumerate(batch,1):
                rebuilt=transition(rows[j],rows[j+1],rows[j-1]['e2e_penalized'],weight,gamma)
                if rebuilt!=entry or abs(entry['reward']-posted[j-1])>1e-10:
                    raise AssertionError('Saved training transition differs from the physical ledger')
                t=rows[j]['timing']; age='old' if t['age_hours']>=8 else 'young'
                profile='burst' if any(x is not None for x in cases[b]['loads']) else 'natural'
                groups.setdefault(f'{profile}.{age}',[]).append(entry['x'])
                visits.setdefault(f'{profile}.{age}',Counter())[str(entry['action'])]+=1
            if b==0:
                calibration.extend(e['x'] for e in batch)
        if offset!=len(s) or len(d['auxiliary'])!=2 or d['auxiliary'][0]['x']!=d['auxiliary'][1]['x']:
            raise AssertionError('Missing original samples or unmatched auxiliary state')
        samples_count+=len(s); inputs.extend(e['x'] for e in s)
        updates=[u for b in d['blocks'] for u in b['updates']]
        evidence[key]=dict(numeric=numeric(net,target,s),coverage=visits,updates=len(updates),
                           clipped=sum(u['gradient']>5 for u in updates),gradient=stats([u['gradient'] for u in updates]))
        for p in probes:
            if p['seed']==seed:
                p['old_q']=net(torch.tensor(p['state'],dtype=torch.float32)).tolist()
    result=dict(features=NAMES,inputs=stats(inputs),scenarios={k:stats(v) for k,v in groups.items()},
                calibration=stats(calibration),calibration_n=len(calibration),credit_max_abs_error=max_residual,
                raw_transitions=samples_count,auxiliary_transition_pairs=5,seeds=evidence,probes=probes,
                proposal_features_in_old_state=False,observation_tail_seconds=env.tail)
    write(ROOT/'p0.audit.json',result)
    print(json.dumps(dict(raw=result['raw_transitions'],credit_error=max_residual,probes=len(probes),
                          saturation={s:round(v['numeric']['saturation'],3) for s,v in evidence.items()},
                          common_offset_ratio={s:round(v['numeric']['common_offset_ratio'],2) for s,v in evidence.items()},
                          clipped={s:v['clipped'] for s,v in evidence.items()})))
    return result


if __name__=='__main__':
    audit()

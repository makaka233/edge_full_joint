"""Audit same-state effects and fit the registered A/B/C value controls."""

import argparse
from copy import deepcopy
import json
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from edge_d.world import stable_seed
from edge_joint.adaptive_runner import source
from scripts.exp043_run import SEEDS
from scripts.exp044_gate import BATCH, UPDATES, fit, gate_files, network, read_resume
from scripts.exp044_run import write
from scripts.exp045_audit import NAMES
from scripts.exp046_collect import FOCUS, ROOT, file_for, register


LABELS=('A','B','C')
HORIZONS=(1,3,5)


def read_cases(plan,part):
    return [json.loads(file_for(c).read_bytes()) for c in plan['case'][part]]


def credit_tail_audit():
    result={}
    for seed in SEEDS:
        archive=read_resume(gate_files(seed)/'resume_latest.pt')['data']
        rows=[r for b in archive['blocks'] for r in b['rows']]
        worst=min(rows,key=lambda r:r['reward_raw']); cpu=max(enumerate(worst['cpu']),key=lambda p:p[1]['work_pending'])
        old_usage=[]
        for i in range(4):
            pool=archive['samples'][:22*(i+1)]+archive['auxiliary']
            rng=np.random.default_rng(stable_seed('EXP044',seed,i,'replay'))
            old_usage.extend(pool[j]['reward'] for j in rng.integers(len(pool),size=UPDATES*BATCH))
        result[str(seed)]=dict(negative_draws=sum(r<=-20 for r in old_usage),draws=len(old_usage),
                               worst=dict(period=worst['period'],scene=worst['scene'],reward=worst['reward_raw'],
                                          trace=worst['trace_sha256'],censored=worst['censored'],cpu=cpu,
                                          conservation_error=worst['conservation_error']),
                               td_by_block=[mean(x['td_mae'] for x in b['updates']) for b in archive['blocks']])
    return result


def held_points(plan):
    points=[]
    for value in read_cases(plan,'held'):
        probe,=value['points']
        measured=np.asarray([r['deltas'] for r in probe['repeats']])
        immediate=[r['arms'][1]['rows'][0]['reward_raw']-r['arms'][0]['rows'][0]['reward_raw']
                   for r in probe['repeats']]
        points.append(dict(case=value['case'],**probe['point'],opening=probe['opening'],deltas=measured.tolist(),
                           gain=measured.mean(0).tolist(),repeat_error=np.abs(np.diff(measured,axis=0))[0].tolist(),
                           immediate_delta=immediate,
                           decode_seconds=[r['profile']['placement_policy_decode_seconds'] for r in value['stem'][1:]],
                           credit_max_error=value['credit_max_error'],wall_seconds=value['wall_seconds']))
    return points


def audit(plan):
    points=held_points(plan); eps=plan['margin']
    repeated=np.asarray([p['deltas'] for p in points]); gains=repeated.mean(1)
    stable=np.where((repeated>eps).all(1),1,np.where((repeated< -eps).all(1),-1,0))
    scaler=plan['base_scaler']; original=(np.asarray([p['state'][:17] for p in points])-scaler['mean'])/scaler['scale']
    nearest=[]
    for i,a in enumerate(original):
        other=[j for j in range(len(points)) if stable[i,2]*stable[j,2]==-1]
        if not other:continue
        j=min(other,key=lambda j:float(np.sqrt(np.mean((a-original[j])**2))))
        nearest.append(dict(left=i,right=j,rms=float(np.sqrt(np.mean((a-original[j])**2)))))
    correlations={k:[float(spearmanr([p['features'][k] for p in points],gains[:,i]).statistic)
                     for i in range(len(HORIZONS))] for k in FOCUS}
    stats=dict(snapshot_count=len(points),repeated_pairs=len(points)*2,stable_support=(stable==1).sum(0).tolist(),
               stable_keep=(stable==-1).sum(0).tolist(),remaining=(stable==0).sum(0).tolist(),
               reversed_repeats=((repeated.min(1)<-eps)&(repeated.max(1)>eps)).sum(0).tolist(),
               median_noise=np.median(np.abs(repeated[:,0]-repeated[:,1]),axis=0).tolist(),
               median_absolute_gain=np.median(np.abs(gains),axis=0).tolist(),
               horizon_3_to_5_reversals=int(np.sum((np.abs(gains[:,1:])>eps).all(1)&(gains[:,1]*gains[:,2]<0))),
               extra_decode_mean_seconds=mean(x for p in points for x in p['decode_seconds']),
               historical_tail=credit_tail_audit(),nearest_opposite=nearest,
               spearman_exploratory=correlations,points=points,source=source())
    write(ROOT/'p0-p1.audit.json',stats)
    print(json.dumps({k:stats[k] for k in ('snapshot_count','stable_support','stable_keep','reversed_repeats',
                                          'median_noise','median_absolute_gain','horizon_3_to_5_reversals')},ensure_ascii=False))
    return stats


class Value(nn.Module):
    def __init__(self,center,scale):
        super().__init__()
        self.register_buffer('center',torch.tensor(center,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.body=network()
        if len(center)>17:
            old=self.body[0]
            first=nn.Linear(len(center),32)
            with torch.no_grad():
                first.weight.zero_(); first.weight[:,:17].copy_(old.weight); first.bias.copy_(old.bias)
            self.body[0]=first

    def forward(self,x):
        return self.body((x-self.center)/self.scale)


def transform_pool(pool,arm):
    distinct=list(dict.fromkeys(tuple(r[k]) for r in pool for k in ('x','next')))
    perm=np.random.default_rng(stable_seed('EXP046','permutation','training-only')).permutation(len(distinct))
    replacements={x:distinct[i][17:] for x,i in zip(distinct,perm,strict=True)}
    def state(x):
        return x[:17] if arm=='A' else x if arm=='B' else x[:17]+list(replacements[tuple(x)])
    return [{**s,'x':state(s['x']),'next':state(s['next'])} for s in pool]


def evaluation_states(points,arm):
    x=[p['state'] for p in points]
    perm=np.random.default_rng(stable_seed('EXP046','permutation','held-only')).permutation(len(x))
    return torch.tensor([s[:17] if arm=='A' else s if arm=='B' else s[:17]+x[i][17:]
                         for s,i in zip(x,perm,strict=True)],dtype=torch.float32)


@torch.no_grad()
def model_stats(net,target,pool):
    x=torch.tensor([r['x'] for r in pool]); nxt=torch.tensor([r['next'] for r in pool]); q=net(x)
    hidden=net.body[1](net.body[0]((x-net.center)/net.scale))
    r=torch.tensor([r['reward'] for r in pool]); action=torch.tensor([r['action'] for r in pool]);
    y=r+.99*target(nxt).gather(1,net(nxt).argmax(1,keepdim=True)).squeeze(1)
    td=y-q.gather(1,action[:,None]).squeeze(1)
    return dict(q_min=float(q.min()),q_max=float(q.max()),hidden_sat=float((hidden.abs()>.99).float().mean()),
                target_quantiles=torch.quantile(y,torch.tensor([0.,.5,.95,1.])).tolist(),
                td_quantiles=torch.quantile(td,torch.tensor([0.,.5,.95,1.])).tolist(),
                absolute_td=float(td.abs().mean()))


def fit_control(plan,points,pool,held_cases):
    center=plan['base_scaler']['mean']; scale=plan['base_scaler']['scale']
    v=np.asarray([r[k][17:] for r in pool for k in ('x','next')])
    extra_mean=v.mean(0).tolist(); extra_scale=np.maximum(v.std(0),.0001).tolist()
    data={}
    q_expected=None
    for arm in LABELS:
        train=transform_pool(pool,arm); x=evaluation_states(points,arm)
        a={}
        for seed in SEEDS:
            torch.manual_seed(stable_seed('EXP044','q-initial',seed))
            net=Value(center if arm=='A' else center+extra_mean,scale if arm=='A' else scale+extra_scale)
            target=deepcopy(net).requires_grad_(False)
            opt=torch.optim.Adam(net.parameters(),lr=plan['q']['lr'])
            with torch.no_grad():initial=net(x).tolist()
            if arm=='A':
                if q_expected is None:q_expected={}
                q_expected[seed]=initial
            elif not np.allclose(initial,q_expected[seed],rtol=0.,atol=1e-6):
                raise AssertionError('Added or scrambled features changed initial Q')
            meters=[]; extreme=0
            for block in range(plan['q']['blocks']):
                key=stable_seed('EXP046',seed,block,'fixed-batches')
                meters.extend(fit(net,target,opt,train,np.random.default_rng(key),plan['q']))
                draws=np.random.default_rng(key).integers(len(train),size=BATCH*UPDATES)
                extreme+=sum(train[i]['reward']<=-20 for i in draws)
            with torch.no_grad():pred=net(x).tolist()
            paths={}
            if arm!='C':
                for case in held_cases:
                    if case['case']['seed']!=seed:continue
                    path=case['stem'][1:]
                    view=torch.tensor([r['timing']['state'][:len(net.center)] for r in path],dtype=torch.float32)
                    with torch.no_grad():scores=net(view).tolist()
                    paths[case['case']['profile']]=[dict(hour=2*(r['period']-1),age=r['timing']['age_hours'],
                        demand=r['demand_rate'],action=r['action'],q=q) for r,q in zip(path,scores,strict=True)]
            a[str(seed)]=dict(q=pred,initial=initial,updates=meters,diagnostics=model_stats(net,target,train),
                              extreme_draws=extreme,held_range=float(np.ptp(np.diff(pred,axis=1))),paths=paths)
        data[arm]=a
    return data,dict(mean=extra_mean,scale=extra_scale)


def evaluate(plan,data,points):
    observed=np.asarray([p['deltas'] for p in points]); eps=plan['margin']; gate=plan['p4_gate']
    stable=np.where((observed>eps).all(1),1,np.where((observed< -eps).all(1),-1,0))
    losses={}; results={}
    for arm in LABELS:
        matched=np.asarray([data[arm][str(p['case']['seed'])]['q'][i] for i,p in enumerate(points)])
        delta=matched[:,1]-matched[:,0]; action=delta>0
        regret=np.maximum(observed,0)-action[:,None,None]*observed
        scores=[]
        for i in range(len(HORIZONS)):
            classes=[float(np.mean((delta[stable[:,i]==j]*j)>0)) for j in (-1,1) if (stable[:,i]==j).any()]
            scores.append(mean(classes) if len(classes)==2 else None)
        losses[arm]=regret
        results[arm]=dict(regret=regret.mean((0,1)).tolist(),orientation=scores,actions=action.tolist(),delta=delta.tolist(),
                          per_point_regret=regret.mean(1).tolist(),
                          rank_spearman=[float(spearmanr(delta,observed.mean(1)[:,i]).statistic) for i in range(3)])
    b=results['B']; others=[results['A'],results['C']]
    wins={s:all(losses['B'][[p['case']['seed']==s for p in points],:,2].mean()<
                losses[m][[p['case']['seed']==s for p in points],:,2].mean() for m in ('A','C')) for s in SEEDS}
    checks=dict(stable_classes=all((stable[:,2]==j).sum()>=gate['minimum_stable_each'] for j in (-1,1)),
                orientation=b['orientation'][2] is not None and all(x['orientation'][2] is not None and
                    b['orientation'][2]-x['orientation'][2]>=gate['orientation_gain'] for x in others),
                regret=all(x['regret'][h]-b['regret'][h]>=gate['regret_gain'] for x in others for h in (1,2)),
                seed_wins=sum(wins.values())>=gate['minimum_seed_wins'])
    return results,dict(checks=checks,advance=all(checks.values()),seed_wins=wins)


def run_value(plan):
    started=perf_counter(); evidence=audit(plan); cases=read_cases(plan,'train')
    points=evidence['points']; pool=[r for c in cases for r in c['transitions']]
    if len(pool)!=130 or any(len(r['x'])!=20 or len(r['next'])!=20 for r in pool):
        raise AssertionError('Fixed pooled experience missing')
    data,scaler=fit_control(plan,points,pool,read_cases(plan,'held'))
    metrics,decision=evaluate(plan,data,points)
    x=np.asarray([r['x'] for r in pool])
    result=dict(outcomes=metrics,decision=decision,models=data,scaler=scaler,training_transitions=len(pool),
                training_moments={k:v.tolist() for k,v in dict(min=x.min(0),max=x.max(0),mean=x.mean(0),std=x.std(0)).items()},
                feature_names=[*NAMES,*FOCUS],training_physical_worker_seconds=sum(c['wall_seconds'] for c in cases),
                training_coverage={str(a):sum(r['action']==a for r in pool) for a in (0,1)},
                held=evidence,run_seconds=perf_counter()-started,source=source())
    write(ROOT/'p2-p3.summary.json',result)
    print(json.dumps(dict(decision=decision,outcomes={k:{a:m[a] for a in ('regret','orientation')} for k,m in metrics.items()})))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('audit','value'),default='value')
    args=parser.parse_args()
    torch.set_num_threads(1)
    plan,_,_=register()
    if args.stage=='audit':audit(plan)
    else:run_value(plan)


if __name__=='__main__':
    main()

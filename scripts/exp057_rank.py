"""Observe common-state agent rankings and run a same-flow prior-only comparator."""

from dataclasses import replace
from functools import partial
from time import perf_counter

import numpy as np
import torch

from comparison.run import MeasuredMultistageSimulator, world
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from edge_s.agent import resource_advantage
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import audit_rows, config
from scripts.exp056_run import compare
from scripts.exp057_run import ROOT, projections


class RankCapture(MeasuredMultistageSimulator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sampled = []
        self.multi = {}
        self.by_period = {}

    def ready(self,req,previous=None):
        real_scheduler = self.scheduler
        def observe(obs,mask):
            selected = real_scheduler(obs,mask)
            legal = np.flatnonzero(mask).tolist()
            if len(legal)<2:
                return selected
            p = self.window_index+1
            visit = self.multi.get(p,0)
            self.multi[p] = visit+1
            if visit%13 or self.by_period.get(p,0)>=32:
                return selected
            self.by_period[p] = self.by_period.get(p,0)+1
            chain = self.c.services[req.arrival.service]
            self.sampled.append(dict(period=p,request=req.id,service=req.arrival.service,stage=chain[req.index],
                time=self.now,selected=selected,legal=legal,obs=obs.copy(),
                cpu_capacity=[self.cpu[n].capacity for n in legal],
                cpu_work=[self.cpu[n].outstanding() for n in legal],
                cpu_jobs=[len(self.cpu[n].jobs) for n in legal],
                cpu_inflight=self.inflight_cpu[legal].tolist(),
                paths=[[[j,self.wired[j].outstanding(),self.wired[j].capacity,len(self.wired[j].jobs)]
                         for j in self.w.routes[req.source][n]] for n in legal],
                next_stage=None if req.index+1==len(chain) else chain[req.index+1],
                estimated_next=np.expm1(obs[legal,22]).tolist(),
                route_time=np.expm1(obs[legal,18]).tolist()))
            return selected
        self.scheduler = observe
        try:
            return super().ready(req,previous)
        finally:
            self.scheduler = real_scheduler


def ranking(events,older,newer):
    if not events:
        return []
    obs = torch.as_tensor(np.stack([r['obs'] for r in events]))
    prior = resource_advantage(obs,newer.config.latency_ref_seconds).numpy()
    residual = [a.online(obs).numpy() for a in (older,newer)]
    out = []
    for i,r in enumerate(events):
        legal = r['legal']
        p = prior[i,legal]
        q = [v[i,legal]+p for v in residual]
        ranks = [np.argsort(-x,kind='stable').tolist() for x in (p,*q)]
        top = [legal[order[0]] for order in ranks]
        if top[2]!=r['selected']:
            raise ValueError('Frozen S1008 action differs from its recorded greedy rank')
        work = np.asarray(r['cpu_work'])/r['cpu_capacity']
        o,n = ranks[1][0],ranks[2][0]
        next_time = r['estimated_next']
        out.append(dict(r,obs=r['obs'].tolist(),prior=p.tolist(),
                        residual504=residual[0][i,legal].tolist(),residual1008=residual[1][i,legal].tolist(),
                        q504=q[0].tolist(),q1008=q[1].tolist(),
                        ranks=[[legal[j] for j in order] for order in ranks],top=top,
                        top_gaps=[float(x[j[0]]-x[j[1]]) for x,j in zip((p,*q),ranks,strict=True)],
                        late_prior_loss=float(p[ranks[0][0]]-p[n]),
                        late_vs_early_queue=float(work[n]-work[o]),
                        late_vs_early_route=r['route_time'][n]-r['route_time'][o],
                        late_vs_early_next=next_time[n]-next_time[o]))
    return out


@torch.inference_mode()
def rank_seed(base,plan,protocol,seed,variant,profile):
    torch.set_num_threads(1)
    time = perf_counter()
    stamp = fingerprint(protocol)
    path = ROOT/f'rank.seed{seed}.json'
    if path.is_file():
        if read(path)['stamp']!=stamp:
            raise ValueError('Stale EXP057 rank diagnosis')
        return f'{path.name} reused'
    cross = read(ROOT/f'cross.seed{seed}.{variant}.{profile}.json')
    if cross['stamp']!=stamp or not cross['layout_match']['BD']:
        raise ValueError('Rank comparisons must share the mature D deployment sequence')
    p,q = projections(protocol,seed)
    cfg = replace(config(base,seed),freeze_ds=True)
    older = components(cfg)[1]
    older.online.load_state_dict(p['s_online'])
    d,newer,_,_,gate,_ = components(cfg)
    load_model(q,d,newer,gate)
    profile_world = partial(world,profile=workload(base['workload']),
                             demand_overrides=plan['persistence'] if variant=='persistent' else None)
    key,pop = (cross['meta'][k] for k in ('scene','population'))
    loads = tuple(.65 if profile=='pulse' and i==3 else None for i in range(cfg.periods))
    ctl = RolloutControl(environment_seed=pop,epsilon=0.,sample_d=False,loads=loads)
    captured = []
    def factory(*args,**kwargs):
        sim = RankCapture(*args,**kwargs)
        captured.append(sim)
        return sim
    env = profile_world(pop,key,'holdout')
    rows = episode(cfg,env,d,newer,gate,key,'holdout',126,control=ctl,simulator_factory=factory)[0]
    audit_rows(env,rows)
    scored = dict(rows=rows,metrics=summarize(rows))
    if not compare(cross['arms']['D'],scored)['exact']:
        raise ValueError('Instrumented trajectory diverged from its registered carrier')
    events = ranking(captured[0].sampled,older,newer)
    def prior_only(obs,mask):
        vals = resource_advantage(torch.as_tensor(obs[None]),newer.config.latency_ref_seconds)[0].numpy()
        return int(np.argmax(np.where(mask,vals,-np.inf)))
    newer.scheduler = lambda rng,epsilon:prior_only
    env = profile_world(pop,key,'holdout')
    prior = episode(cfg,env,d,newer,gate,key,'holdout',126,control=ctl,
                    simulator_factory=MeasuredMultistageSimulator)[0]
    audit_rows(env,prior)
    same_flow(rows,prior)
    if any(a['active_matrix']!=b['active_matrix'] for a,b in zip(rows,prior,strict=True)):
        raise ValueError('Prior changed the actual deployed matrix')
    write(path,dict(stamp=stamp,seed=seed,variant=variant,profile=profile,meta=cross['meta'],events=events,
                    multi=captured[0].multi,metrics={'S504':cross['arms']['B']['metrics'],
                    'S1008':scored['metrics'],'prior':summarize(prior)},prior_rows=prior,
                    carrier_events=sum(r['events'] for r in rows),wall_seconds=perf_counter()-time,source=source()))
    return f'{path.name} checked'

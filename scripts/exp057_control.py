"""Replay the registered unchanged control and open the sealed holdout once."""

from functools import partial
from statistics import mean
from time import perf_counter

import torch

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, world
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp044_run import write
from scripts.exp047_run import read
from scripts.exp055_joint import audit_rows, config
from scripts.exp056_long import destination, initialize, verify_prefix
from scripts.exp056_run import evaluate
from scripts.exp057_run import CASES, CONTROLS, FOCUS, ROOT, PREVIOUS, SEEDS, projections


def decision(protocol):
    d = read(ROOT/'decision.json')
    if d['stamp']!=fingerprint(protocol) or d['algorithm']!='unchanged':
        raise ValueError('Commit the no-change decision before running the unchanged control')
    return d


def seal_unchanged():
    """Record the mechanism decision before opening any new request streams."""
    protocol = read(ROOT/'protocol.json')['plan']
    stamp = fingerprint(protocol)
    path = ROOT/'decision.json'
    if path.is_file():
        decision(protocol)
        return
    if any(ROOT.glob('holdout.seed*.json')):
        raise ValueError('Holdout already opened')
    history = {seed:[ROOT/f'cross.seed{seed}.{v}.{c}.json' for v,c in CASES] for seed in SEEDS}
    rank = [ROOT/f'rank.seed{seed}.json' for seed in FOCUS]
    def evidence(path):
        r = read(path)
        if r['stamp']!=stamp:
            raise ValueError(f'Stale decision evidence: {path}')
        return r
    for seed in CONTROLS:
        pairs = [evidence(p)['effects'] for p in history[seed]]
        for e in ('B-A','D-C'):
            if mean(r[e]['reward'] for r in pairs)>=0. or mean(r[e]['e2e'] for r in pairs)<=0.:
                raise ValueError('Archived late-D attribution did not reproduce')
        if mean(r['D-B']['reward'] for r in pairs)<=0.:
            raise ValueError('Archived mitigating S effect did not reproduce')
    for f in rank:
        m = evidence(f)['metrics']
        if m['S1008']['reward']<=m['prior']['reward'] or m['S1008']['e2e']>=m['prior']['e2e']:
            raise ValueError('Archived prior-only comparison did not reproduce')
    files = [f for group in history.values() for f in group]+rank
    write(path,dict(stamp=stamp,algorithm='unchanged',evidence={p.name:digest(p) for p in files},
                    reason='Late D caused regression in target seeds; same-D late S mitigated it and beat the prior. '
                           'Frozen layouts identify policy effects but no verified training-mechanism change.',
                    source=source()))


def control_path(seed):
    return RunFiles(ROOT/f'control.seed{seed}',nested=True)


def control_seed(base,plan,protocol,seed):
    torch.set_num_threads(1)
    verdict = decision(protocol)
    target = protocol['control_episode']
    out = control_path(seed)
    old = read(PREVIOUS/'protocol.json')['plan']
    tick = perf_counter()
    with run_lock(out):
        imported = initialize(seed,plan,old,out)
        train(config(base,seed),out,target,resume=True,environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator)
        state = read_checkpoint(out/'resume_latest.pt')
        if (state['episode'],state['d_updates'],state['s']['updates'])!=(target,target,64*target):
            raise ValueError('Control did not advance its committed D/S state')
        for f,n in imported['files'].items():
            if f in state['offsets']:
                verify_prefix(out/f,n['size'],n['sha256'])
        rows = [r for r in records(out/'train.jsonl') if r['gs']>504]
        updates = [r for r in records(out/'updates.jsonl') if r['gs']>504]
        archived = [r for r in records(destination(seed)/'train.jsonl') if 504<r['gs']<=target*8]
        if ([r['gs'] for r in rows]!=list(range(505,target*8+1)) or
                [r['gs'] for r in updates]!=list(range(512,target*8+1,8)) or
                any(min(r[k] for k in ('d_change','s_change','s_target_change'))<=0. for r in updates)):
            raise ValueError('Missing control updates or physical periods')
        fields = ('trace_sha256','active_matrix','s_dispatch_counts','cpu','wired','reward_raw','e2e_completed',
                  'p95','p99','censored','residual_requests')
        diff = {k:[r['gs'] for r,t in zip(rows,archived,strict=True) if r.get(k)!=t.get(k)] for k in fields}
        physical = []
        for i in range(0,len(rows),8):
            r = rows[i]
            env = world(seed,r['scene'],'train',profile=workload(base['workload']))
            physical.append(audit_rows(env,rows[i:i+8]))
        write(ROOT/f'control.seed{seed}.json',dict(stamp=fingerprint(protocol),decision=fingerprint(verdict),
                    differences=diff,work_error=max(physical),last=digest(out/'last.pt'),best=digest(out/'best.pt'),
                    d=state['d_updates'],s=state['s']['updates'],history=state['source_history'],
                    wall_seconds=perf_counter()-tick,source=source()))
    return f'Control seed {seed} checked through GS{target*8}'


@torch.inference_mode()
def holdout_seed(base,plan,protocol,seed):
    torch.set_num_threads(1)
    verdict = decision(protocol)
    out = control_path(seed)
    with run_lock(out):
        proof = read(ROOT/f'control.seed{seed}.json')
        if (proof['decision']!=fingerprint(verdict) or any(digest(out/f'{k}.pt')!=proof[k] for k in ('last','best'))):
            raise ValueError('Control checkpoint differs from committed evaluation identity')
        models = {k:read_checkpoint(out/f'{k}.pt') for k in ('last','best')}
    previous,_ = projections(protocol,seed)
    result = []
    for case in protocol['holdout']:
        if case['seed']!=seed:
            continue
        v,c,key,pop = (case[k] for k in ('variant','profile','scene','population'))
        path = ROOT/f'holdout.seed{seed}.{v}.{c}.json'
        stamp = fingerprint(dict(case=case,decision=fingerprint(verdict),last=proof['last'],best=proof['best']))
        if path.is_file():
            if read(path)['stamp']!=stamp:
                raise ValueError(f'Existing holdout provenance differs: {path}')
            result.append(f'{path.name} reused')
            continue
        arms = evaluate(base,plan,seed,v,c,key,pop,{'GS504':previous,**models})
        write(path,dict(stamp=stamp,case=case,best_gs=models['best']['episode']*8,arms=arms,source=source()))
        result.append(f'{path.name} checked')
    return '\n'.join(result)


if __name__=='__main__':
    seal_unchanged()

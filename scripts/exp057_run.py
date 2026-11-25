"""Frozen GS504/1008 component swaps for EXP057."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from comparison.run import jobs
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import read_checkpoint, source
from edge_joint.adaptive_storage import RunFiles, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import SEEDS, inputs, output, projection_matches
from scripts.exp056_long import destination
from scripts.exp056_report import effects
from scripts.exp056_run import CASES, ROOT as PREVIOUS, evaluate


ROOT = Path('runs/results/EXP057')
PARENT = 'cc20528c051bcf508db01b061b91dd93625f4e86'
FOCUS = {20261113:('original','natural'),20261115:('persistent','natural'),20261114:('original','pulse')}
CONTROLS = (20261113,20261115)


def endpoint(seed,gs):
    return output(seed,False) if gs==504 else destination(seed)


def register(plan):
    old = read(PREVIOUS/'protocol.json')['plan']
    if old['exp055']!=fingerprint(plan):
        raise ValueError('Baseline training or evaluation protocol changed')
    owners = {}
    for seed in SEEDS:
        weights,states = {},{}
        for gs in (504,1008):
            p = endpoint(seed,gs)
            last,state = (read_checkpoint(p/f'{k}.pt') for k in ('last','resume_latest'))
            if (not projection_matches(last,state) or last['episode']*8!=gs or state['episode']*8!=gs or
                    state['d_updates']*8!=gs or state['s']['updates']!=8*gs or
                    any(m['origin']!={'kind':'random'} or m['config']!=plan['config'][str(seed)] or
                        m['environment_id']!=old['exp055'] for m in (state,last)) or
                    not state['d_optimizer']['state'] or not state['s']['optimizer']['state'] or
                    not state['rng'] or not state['replay']['columns'] or
                    any((p/k).stat().st_size!=n for k,n in state['offsets'].items())):
                raise ValueError(f'Wrong endpoint checkpoint or ledger for seed {seed} GS{gs}')
            weights[str(gs)] = dict(last=digest(p/'last.pt'),resume=digest(p/'resume_latest.pt'),source=state['source_history'])
            states[gs] = state
        origin = read(endpoint(seed,1008)/'continuation.json')['import']
        if (weights['504']['last']!=old['files'][str(seed)]['model']['last'] or
                weights['504']['resume']!=old['files'][str(seed)]['resume'] or
                weights['504']['resume']!=origin['files']['resume_latest.pt']['sha256']):
            raise ValueError(f'Unexpected source checkpoint for seed {seed}')
        def norm(name,x,y):
            return float(sum((p-y[name][k]).square().sum() for k,p in x[name].items()).sqrt())
        criteria = {k:norm(k,states[504],states[1008]) for k in ('d','s_online')}
        criteria['target'] = norm('target',states[504]['s'],states[1008]['s'])
        cases = {}
        for v,c in CASES:
            a = PREVIOUS/f'ablation.seed{seed}.{v}.{c}.json'
            d = PREVIOUS/f'long-eval.seed{seed}.{v}.{c}.episode126.json'
            meta = old['files'][str(seed)]['case'][f'{v}.{c}']
            late = read(d)
            expected = fingerprint(dict(model=weights['1008']['last'],meta=meta,episode=126,registered=fingerprint(old)))
            early = read(a)
            if (late['stamp']!=expected or late['seed']!=seed or early['stamp']!=fingerprint(old) or
                    early['seed']!=seed or early['meta']!=meta or not early['match']['D']['exact']):
                raise ValueError('Historical endpoint ledger changed')
            cases[f'{v}.{c}'] = dict(meta,early=str(a),late=str(d),early_sha=digest(a),late_sha=digest(d))
        owners[str(seed)] = dict(weights=weights,changes=criteria,cases=cases)
    holdout = []
    for seed in CONTROLS:
        for v,c in (('original','natural'),('persistent','pulse')):
            key = stable_seed('EXP057-test-v1',seed,v,c,'scene')
            pop = stable_seed('EXP057-test-v1',seed,v,c,'population')
            holdout.append(dict(seed=seed,variant=v,profile=c,scene=key,population=pop))
    prior_keys = {r[k] for s in owners.values() for r in s['cases'].values() for k in ('scene','population')}
    fresh = [r[k] for r in holdout for k in ('scene','population')]
    training_keys = {stable_seed('EXP019',s,'train-scene',i) for s in SEEDS for i in range(1,127)}
    if len(set(fresh))!=len(fresh) or set(fresh)&(prior_keys|training_keys|set(SEEDS)|{19001,19002}):
        raise ValueError('Reserved holdout keys overlap a registered selection flow')
    protocol = dict(parent=PARENT,exp056=fingerprint(old),owners=owners,focus={str(k):list(v) for k,v in FOCUS.items()},
                    horizon=8,holdout=holdout,control_episode=79,code={p:digest(Path(p)) for p in
                      ('scripts/exp057_run.py','research/EXP057/PROTOCOL.md','edge_joint/adaptive_training.py',
                       'edge_joint/adaptive_physics.py','edge_s/agent.py','edge_d/_engine/edge_rl/agents.py')})
    path = ROOT/'protocol.json'
    if path.is_file() and read(path)['plan']!=protocol:
        raise ValueError('Existing EXP057 registration changed')
    if not path.is_file():
        write(path,dict(plan=protocol,source=source()))
    return protocol


def projections(protocol,seed):
    models = []
    for gs in (504,1008):
        path = endpoint(seed,gs)/'last.pt'
        if digest(path)!=protocol['owners'][str(seed)]['weights'][str(gs)]['last']:
            raise ValueError('Registered model changed')
        models.append(read_checkpoint(path))
    return models


def historical(meta):
    a,d = (Path(meta[k]) for k in ('early','late'))
    if digest(a)!=meta['early_sha'] or digest(d)!=meta['late_sha']:
        raise ValueError('Frozen endpoint evidence changed')
    return read(a)['arms']['D'],read(d)['last']


@torch.inference_mode()
def cross_seed(base,plan,protocol,seed):
    torch.set_num_threads(1)
    stamp = fingerprint(protocol)
    p,q = projections(protocol,seed)
    results = []
    for v,c in CASES:
        path = ROOT/f'cross.seed{seed}.{v}.{c}.json'
        if path.exists():
            if read(path)['stamp']!=stamp:
                raise ValueError(f'Stale EXP057 evidence: {path}')
            results.append(f'{path.name} reused')
            continue
        tick = perf_counter()
        meta = protocol['owners'][str(seed)]['cases'][f'{v}.{c}']
        a,d = historical(meta)
        mixed = evaluate(base,plan,seed,v,c,meta['scene'],meta['population'],
                         {'B':dict(q,s_online=p['s_online']),'C':dict(p,s_online=q['s_online'])})
        b,cc = (mixed[k] for k in ('B','C'))
        arms = dict(A=a,B=b,C=cc,D=d)
        for x in arms.values():
            same_flow(a['rows'],x['rows'])
            m = x['metrics']
            layout = x['rows']
            m['mean_replicas'] = sum(sum(r['active_replicas']) for r in layout)/len(layout)
            m['changed_bits'] = sum((int(r['old_matrix'],16)^int(r['active_matrix'],16)).bit_count() for r in layout)
        layout_match = {}
        for x,y in (('A','C'),('B','D')):
            layout_match[x+y] = all(p['active_matrix']==q['active_matrix']
                                     for p,q in zip(arms[x]['rows'],arms[y]['rows'],strict=True))
        write(path,dict(stamp=stamp,seed=seed,variant=v,profile=c,arms=arms,layout_match=layout_match,
                        effects=effects({k:x['metrics'] for k,x in arms.items()}),source=source(),
                        meta=meta,wall_seconds=perf_counter()-tick))
        results.append(f'{path.name} checked')
    return '\n'.join(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('register','priority','cross','rank','control','holdout','report'))
    p.add_argument('--workers',type=int,default=3)
    args = p.parse_args()
    if args.workers<1:
        p.error('Workers must be positive')
    base,plan = inputs()
    with run_lock(RunFiles(ROOT/'experiment')):
        if args.stage=='register':
            register(plan)
            print(ROOT/'protocol.json')
            return
        protocol = read(ROOT/'protocol.json')['plan']
        if args.stage in ('priority','cross'):
            seeds = tuple(FOCUS) if args.stage=='priority' else SEEDS
            jobs(cross_seed,[(base,plan,protocol,s) for s in seeds],min(len(seeds),args.workers))
        elif args.stage=='rank':
            from scripts.exp057_rank import rank_seed
            jobs(rank_seed,[(base,plan,protocol,s,v,c) for s,(v,c) in FOCUS.items()],min(3,args.workers))
        elif args.stage in ('control','holdout'):
            from scripts.exp057_control import control_seed, holdout_seed
            fn = control_seed if args.stage=='control' else holdout_seed
            jobs(fn,[(base,plan,protocol,s) for s in CONTROLS],min(len(CONTROLS),args.workers))
        else:
            from scripts.exp057_report import report
            report(protocol)


if __name__=='__main__':
    main()

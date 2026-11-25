"""Train the registered structured Actor comparison and collect frozen physical evidence."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_d.world import stable_seed
from edge_joint.adaptive_runner import FIXED, read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.adaptive_physics import capacity_ok
from edge_joint.adaptive_training import RolloutControl, components, decode_placement, episode, load_model
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp043_run import LongConfig
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import audit_rows, config, inputs, output, projection_matches
from scripts.exp056_run import CASES
from scripts.exp058_run import FIELDS, UPDATE_FIELDS, discrepancies, read_span


ROOT = Path('runs/results/EXP060')
SEEDS = (20261113,20261115,20261114)
ARMS = ('A','B')
TARGET = 63
PARENT = '20f9ac8b3939c55b698d0c59edbc07f250d16345'
EVAL = (*CASES,('original','sustained055'),('original','sustained070'))


@dataclass(frozen=True)
class StructuredConfig(LongConfig):
    d_actor: str = 'structured'


def settings(base,seed,arm):
    cfg = replace(config(base,seed),profile_periods=True)
    return StructuredConfig(**asdict(cfg)) if arm=='B' else cfg


def paths(seed,arm,label=''):
    return RunFiles(ROOT/f'{label}{arm}.seed{seed}',nested=True)


def weights_digest(data):
    h = sha256()
    for k,v in sorted(data.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(repr((k,a.dtype.str,a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def registration(base,env):
    used = set(FIXED)|set(int(s) for s in env['config'])
    used.update(stable_seed('EXP019',int(s),'train-scene',i) for s in env['config'] for i in range(1,127))
    for r in read(ROOT.parent/'EXP056'/'protocol.json')['plan']['files'].values():
        used.update(c[k] for c in r['case'].values() for k in ('scene','population'))
    for old in ('EXP057','EXP058','EXP059'):
        suite = read(ROOT.parent/old/'protocol.json')['plan']['holdout']
        if isinstance(suite,dict):
            suite = [r for group in suite.values() for r in group]
        used.update(r[k] for r in suite for k in ('scene','population'))
    cases = []
    for s,(v,c) in product(SEEDS,EVAL):
        key,pop = (stable_seed('EXP060-independent-v1',s,v,c,k) for k in ('scene','population'))
        if key==pop or key in used or pop in used:
            raise ValueError('EXP060 held-out random stream overlaps an existing registration')
        used.update((key,pop))
        cases.append(dict(seed=s,variant=v,profile=c,scene=key,population=pop))
    baseline = {str(s):{f:digest(output(s,False)/f) for f in ('train.jsonl','updates.jsonl','last.pt')}
                for s in SEEDS}
    spec = dict(parent=PARENT,environment=fingerprint(env),target=TARGET,seeds=list(SEEDS),cases=cases,
                baseline=baseline,configs={f'{a}.{s}':asdict(settings(base,s,a)) for s,a in product(SEEDS,ARMS)},
                code={f:digest(Path(f)) for f in ('edge_d/placement.py','edge_joint/adaptive_training.py',
                      'edge_joint/adaptive_runner.py','scripts/exp060_run.py','research/EXP060/PROTOCOL.md')})
    path = ROOT/'protocol.json'
    if path.exists():
        if read(path)['plan']!=spec:
            raise ValueError('The registered EXP060 inputs changed')
    else:
        write(path,dict(plan=spec,source=source()))
    return spec


def initial_proof(base,seed,arm):
    d,s,_,replay,_gate,_opt = components(settings(base,seed,arm))
    return dict(seed=seed,arm=arm,actor=weights_digest(d.state_dict()),global_d=weights_digest(d.net.state_dict()),
                s_online=weights_digest(s.online.state_dict()),s_target=weights_digest(s.target.state_dict()),
                replay=len(replay),d_parameters=sum(p.numel() for p in d.parameters()),
                s_parameters=sum(p.numel() for p in s.online.parameters()))


@torch.inference_mode()
def control_evidence(base):
    seed = SEEDS[0]
    a,b = (settings(base,seed,k) for k in ARMS)
    actor,*_ = components(b)
    profile = workload(base['workload'])
    e = world(seed,16029,'holdout',profile=profile)
    mean = actor(e).numpy()
    next_e = world(seed,16030,'holdout',profile=profile)
    shift = float(np.linalg.norm(mean[8:]-actor(next_e).numpy()[8:]))
    variation = np.std(mean[8:].reshape(28,3),axis=0)
    if shift<=0 or np.any(variation<=0):
        raise ValueError('Stage head does not respond to input or stage identity')
    neutral = np.r_[mean[:8],np.zeros(84)]
    if not np.array_equal(decode_placement(a,e,neutral[:8],83).matrix,decode_placement(b,e,neutral,83).matrix):
        raise ValueError('Neutral stage controls must recover the original decoder')
    cells = []
    for c,k in product(range(3),(0,9,18)):
        mat = []
        for sign in (-1,1):
            z = neutral.copy()
            z[8+3*k+c] = sign*3.
            mat.append(decode_placement(b,e,z,83).matrix)
        left,right = mat
        cells.append(dict(channel=c,stage=k,target_bits=int((left[k]^right[k]).sum()),
                          matrix_bits=int((left^right).sum()),before=int(left[k].sum()),after=int(right[k].sum())))
    for c in range(3):
        if not any(x['channel']==c and x['target_bits'] for x in cells):
            raise ValueError(f'Stage control {c} did not change targeted physical placements')
    if not all(x['after']>x['before'] for x in cells if x['channel']==2):
        raise ValueError('Targeted redundancy intervention did not increase certified redundancy')
    rng = np.random.default_rng(16092)
    paired = []
    for key in range(110,122):
        latent = mean + .5*rng.normal(size=mean.shape)
        record = {}
        for label,cfg,z in zip(ARMS,(a,b),(latent[:8],latent),strict=True):
            tick = perf_counter()
            x = decode_placement(cfg,e,z,key).matrix
            record[label] = dict(seconds=perf_counter()-tick,replicas=int(x.sum()))
            if not x.any(1).all() or not capacity_ok(x,e.spec,e.capacity):
                raise ValueError('Invalid structured decode in the registered perturbation probe')
        paired.append(record)
    if float(np.median([r['B']['replicas'] for r in paired]))<=28.:
        raise ValueError('Candidate failed the cold redundancy probe')
    return dict(scene_shift_l2=shift,stage_mean_std=variation.tolist(),interventions=cells,random_decodes=paired)


def run_arm(base,spec,seed,arm,target=TARGET,label=''):
    torch.set_num_threads(1)
    begin = perf_counter()
    out = paths(seed,arm,label)
    proj = out/'resume_latest.pt'
    proof_path = ROOT/f'{out.prefix.name}.gs{target*8}.json'
    stamp = fingerprint(spec)
    with run_lock(out):
        if proof_path.is_file():
            prev = read(proof_path)
            if prev['stamp']!=stamp:
                raise ValueError('Mismatched training publication')
            if digest(proj)==prev['model_sha']['resume_latest']:
                return prev
            raise ValueError(f'Existing EXP060 endpoint checksum changed: {out.prefix}')
        cfg = settings(base,seed,arm)
        origin = initial_proof(base,seed,arm)

        def snapshot(initial):
            if initial['episode'] or len(initial['d_optimizer']['state']) or initial['s']['updates']:
                raise ValueError('A random initialization contains trained state')
            for k,expected in (('d',origin['actor']),('s_online',origin['s_online'])):
                if weights_digest(initial[k])!=expected:
                    raise ValueError('Actual initialization differs from the recorded model')
            write(out/'initial.json',origin)

        train(cfg,out,target,resume=proj.exists(),environment_id=stamp,
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator,initial_snapshot=snapshot)
        if read(out/'initial.json')!=origin:
            raise ValueError('Random initialization provenance mismatch')
        saved = read_checkpoint(proj)
        last,best = (read_checkpoint(out/f'{k}.pt') for k in ('last','best'))
        if ((saved['episode'],saved['d_updates'],saved['s']['updates'])!=(target,target,64*target) or
                saved['origin']!={'kind':'random'} or not projection_matches(last,saved) or
                not projection_matches(best,saved['best'])):
            raise ValueError('Incorrect endpoint, model projection, origin or training update budget')
        for f,n in saved['offsets'].items():
            if (out/f).stat().st_size!=n:
                raise ValueError('Uncommitted log at EXP060 endpoint')
        rows,u = (read_span(out/f,first=0,last=target*8) for f in ('train.jsonl','updates.jsonl'))
        if ([r['gs'] for r in rows]!=list(range(1,target*8+1)) or
                [r['gs'] for r in u]!=list(range(8,target*8+1,8)) or
                not all(min(x['d_change'],x['s_change'],x['s_target_change'])>0. for x in u)):
            raise ValueError('Missing actual joint updates or physical slots')
        if arm=='B' and not all(np.isfinite([x['d_stage_gradient'],x['d_stage_change']]).all() and
                                min(x['d_stage_gradient'],x['d_stage_change'])>0. for x in u):
            raise ValueError('Structured Actor did not learn on both its heads')
        errors = []
        for i in range(0,len(rows),8):
            e = world(seed,rows[i]['scene'],'train',profile=workload(base['workload']))
            errors.append(audit_rows(e,rows[i:i+8]))
        historical = {}
        if arm=='A':
            archive = output(seed,False)
            historical = {f:discrepancies(actual,read_span(archive/f,first=0,last=target*8),fields)
                          for f,actual,fields in (('train.jsonl',rows,FIELDS),('updates.jsonl',u,UPDATE_FIELDS))}
        proof = dict(seed=seed,arm=arm,gs=target*8,initial=origin,source=source(),stamp=stamp,
                     elapsed=saved['wall_seconds'],worker_seconds=perf_counter()-begin,history_match=historical,
                     d=saved['d_updates'],s=saved['s']['updates'],best_gs=best['episode']*8,
                     max_resource_error=max(errors),source_history=saved['source_history'],
                     model_sha={k:digest(out/f'{k}.pt') for k in ('best','last','resume_latest')})
        write(proof_path,proof)
    print(f'EXP060 {arm} {seed} verified through GS{target*8}',flush=True)
    return proof


@torch.inference_mode()
def smoke_evaluation(base):
    seed,key = SEEDS[0],16029
    original = None
    cells = {}
    for arm in ARMS:
        cfg = replace(settings(base,seed,arm),periods=2,freeze_ds=True)
        d,s,_,_,g,_ = components(cfg)
        e = world(seed,key,'holdout',profile=workload(base['workload']))
        control = RolloutControl(environment_seed=seed,epsilon=0.,sample_d=False,loads=(.55,.55))
        rows = episode(cfg,e,d,s,g,key,'holdout',0,control=control,profile=True,
                       simulator_factory=MeasuredMultistageSimulator)[0]
        error = audit_rows(e,rows)
        if original is None:
            original = rows
        else:
            same_flow(original,rows)
        cells[arm] = dict(resource_error=error,trace=[r['trace_sha256'] for r in rows],
                          decode_seconds=[r['profile']['physical_decoder_seconds'] for r in rows])
    return cells


def smoke(base,spec):
    torch.set_num_threads(1)
    seed = SEEDS[0]
    previous = ROOT/'smoke.json'
    expected = fingerprint(spec)
    if previous.exists():
        if read(previous)['stamp']!=expected:
            raise ValueError('Smoke belongs to a different experiment')
        return 'Registered smoke already verified'
    design = control_evidence(base)
    split,full = (paths(seed,'B',k) for k in ('split-','full-'))
    if not (split/'resume_latest.pt').exists() or read_checkpoint(split/'resume_latest.pt')['episode']<2:
        run_arm(base,spec,seed,'B',1,'split-')
    resumed = run_arm(base,spec,seed,'B',2,'split-')
    direct = run_arm(base,spec,seed,'B',2,'full-')
    for f,fields in (('train.jsonl',FIELDS+('d_credit',)),
                     ('updates.jsonl',UPDATE_FIELDS+('d_stage_gradient','d_stage_change'))):
        a,b = (read_span(p/f,first=0,last=16) for p in (split,full))
        if discrepancies(a,b,fields):
            raise ValueError('Structured continuation failed to reproduce actions and updates')
    if resumed['model_sha']['last']!=direct['model_sha']['last']:
        raise ValueError('Structured continuation failed to reproduce weights')
    u = read_span(full/'updates.jsonl',first=0,last=16)
    write(previous,dict(resumed=resumed,direct=direct,stamp=expected,gs=16,evaluation=smoke_evaluation(base),
                        design=design,
                        stage_gradient=[r['d_stage_gradient'] for r in u],
                        stage_change=[r['d_stage_change'] for r in u]))
    return 'Structured split/continuous physical smoke verified through GS16'


def checked_training(spec,seed):
    proof = {a:read(ROOT/f'{a}.seed{seed}.gs504.json') for a in ARMS}
    if any(p['stamp']!=fingerprint(spec) for p in proof.values()):
        raise ValueError('Registered model origin does not match')
    a,b = [proof[k]['initial'] for k in ARMS]
    if any(a[k]!=b[k] for k in ('global_d','s_online','s_target','replay')) or a['replay']!=0:
        raise ValueError('Random initialization is not paired')
    same_flow(*(read_span(paths(seed,k)/'train.jsonl',first=0,last=504) for k in ARMS))
    return proof


@torch.inference_mode()
def holdout(base,env,spec,seed):
    torch.set_num_threads(1)
    proof = checked_training(spec,seed)
    for case in spec['cases']:
        if case['seed']!=seed:
            continue
        v,c,key,pop = (case[k] for k in ('variant','profile','scene','population'))
        path = ROOT/f'holdout.seed{seed}.{v}.{c}.json'
        stamp = fingerprint(dict(case=case,run=fingerprint(spec),models={k:p['model_sha']['last'] for k,p in proof.items()}))
        if path.exists():
            if read(path)['stamp']!=stamp:
                raise ValueError('A stale EXP060 evaluation already exists')
            continue
        begin = perf_counter()
        variants = {}
        for a,zero in product(ARMS,(True,False)):
            label = f'{a}0' if zero else a
            cfg = replace(settings(base,seed,a),freeze_ds=True)
            d,s,_,_,gate,_ = components(cfg)
            checkpoint = paths(seed,a)/'last.pt'
            if not zero:
                if digest(checkpoint)!=proof[a]['model_sha']['last']:
                    raise ValueError('Frozen evaluation checkpoint changed')
                load_model(read_checkpoint(checkpoint),d,s,gate)
            else:
                init = proof[a]['initial']
                if weights_digest(d.state_dict())!=init['actor'] or weights_digest(s.online.state_dict())!=init['s_online']:
                    raise ValueError('Reconstructed GS0 differs from actual training origin')
            e = world(pop,key,'holdout',profile=workload(base['workload']),
                      demand_overrides=env['persistence'] if v=='persistent' else None)
            sustained = {'sustained055':.55,'sustained070':.70}.get(c)
            loads = (sustained,)*8 if sustained is not None else tuple(.65 if c=='pulse' and i==3 else None for i in range(8))
            control = RolloutControl(environment_seed=pop,epsilon=0.,sample_d=False,loads=loads)
            rows = episode(cfg,e,d,s,gate,key,'holdout',0 if zero else TARGET,control=control,
                           simulator_factory=MeasuredMultistageSimulator,profile=True)[0]
            error = audit_rows(e,rows)
            if variants:
                same_flow(next(iter(variants.values()))['rows'],rows)
            variants[label] = dict(rows=rows,metrics=summarize(rows),resource_error=error)
        write(path,dict(stamp=stamp,case=case,arms=variants,seconds=perf_counter()-begin,source=source()))
    return f'EXP060 seed {seed}: six independent cases complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','smoke','train','evaluate','report','unattended'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--third',action='store_true',help='Enable registered positive control only after a passing initial analysis')
    args = parser.parse_args()
    if args.workers<1:
        parser.error('Positive workers required')
    ROOT.mkdir(parents=True,exist_ok=True)
    base,env = inputs()
    seeds = (SEEDS[2],) if args.third else SEEDS[:2]
    with run_lock(RunFiles(ROOT/'experiment')):
        spec = registration(base,env)
        if args.third:
            result = read(ROOT/'summary.initial.json')
            if not result['eligible'] or result['stamp']!=fingerprint(spec):
                raise ValueError('Both primary seeds must pass before running Seed14')
        if args.action=='register':
            print(ROOT/'protocol.json')
        if args.action=='smoke':
            with ProcessPoolExecutor(max_workers=1) as pool:
                print(pool.submit(smoke,base,spec).result(),flush=True)
        if args.action in ('train','unattended'):
            if read(ROOT/'smoke.json')['stamp']!=fingerprint(spec):
                raise ValueError('Complete the registered smoke before training')
            tasks = [(base,spec,s,a) for s,a in product(seeds,ARMS)]
            jobs(run_arm,tasks,min(args.workers,len(tasks)))
        if args.action in ('evaluate','unattended'):
            jobs(holdout,[(base,env,spec,s) for s in seeds],min(args.workers,len(seeds)))
        if args.action=='unattended':
            write(ROOT/('third-complete.json' if args.third else 'initial-complete.json'),dict(
                stamp=fingerprint(spec),seeds=seeds,gs=504,cases=len(seeds)*len(EVAL),source=source()))
        if args.action=='report':
            from scripts.exp060_report import report
            report(base,spec,SEEDS if args.third else SEEDS[:2])


if __name__=='__main__':
    main()

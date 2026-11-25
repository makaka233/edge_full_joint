"""Branch complete EXP055 states and resume the existing trainer."""

import argparse
from functools import partial
from hashlib import sha256
from itertools import product
from pathlib import Path
from shutil import copyfileobj

import torch

from comparison.report import records
from comparison.run import MeasuredMultistageSimulator, jobs, world
from edge_joint.adaptive_runner import read_checkpoint, source, train
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp037_run import workload
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import read
from scripts.exp055_joint import SEEDS, audit_rows, config, inputs, output
from scripts.exp056_run import CASES, ROOT, evaluate


FILES = ('train.jsonl','fixed.jsonl','updates.jsonl','manifest.json','best.pt','last.pt','resume_latest.pt')
START = 63


def destination(seed,smoke=False):
    return RunFiles(ROOT/f'{"resume-smoke" if smoke else "long"}.seed{seed}',nested=True)


def verify_prefix(path,size,expected):
    h = sha256()
    with path.open('rb') as f:
        while size:
            block = f.read(min(size,2**20))
            if not block:
                raise ValueError(f'Truncated inherited ledger: {path}')
            h.update(block)
            size -= len(block)
    if h.hexdigest()!=expected:
        raise ValueError(f'Changed inherited ledger: {path}')


def initialize(seed,plan,registered,out):
    src = output(seed,False)
    with run_lock(src):
        state = read_checkpoint(src/'resume_latest.pt')
        expected = registered['files'][str(seed)]
        if (digest(src/'resume_latest.pt')!=expected['resume'] or state['episode']!=START or
                state['environment_id']!=fingerprint(plan) or state['config']!=plan['config'][str(seed)]):
            raise ValueError(f'EXP055 seed does not match its registered state: {seed}')
        names = {f:dict(size=(src/f).stat().st_size,sha256=digest(src/f)) for f in FILES}
        if (any(names[k]['size']!=n for k,n in state['offsets'].items()) or
                any(names[f'{k}.pt']['sha256']!=h for k,h in expected['model'].items())):
            raise ValueError(f'EXP055 source projection or committed log differs: {seed}')
        manifest = dict(seed=seed,episode=START,source=str(src.prefix),plan=fingerprint(plan),files=names)
        info = out/'continuation.json'
        if info.is_file():
            if read(info)['import']!=manifest:
                raise ValueError(f'Continuation prefix belongs to a different import: {out.prefix}')
        else:
            if any((out/f).exists() for f in FILES):
                raise ValueError(f'Unregistered files already exist in {out.prefix}')
            write(info,{'import':manifest,'source':source()})
        if not (out/'resume_latest.pt').is_file():
            for name,metadata in names.items():
                def transfer(handle, path=src/name):
                    with path.open('rb') as old:
                        copyfileobj(old,handle)
                atomic(out/name,transfer)
                if digest(out/name)!=metadata['sha256']:
                    raise ValueError(f'Copy failed verification: {out/name}')
        current = read_checkpoint(out/'resume_latest.pt')
        if current['episode']<START or (current['episode']==START and
                                       digest(out/'resume_latest.pt')!=expected['resume']):
            raise ValueError(f'Invalid imported recovery state: {out.prefix}')
        for f,size in state['offsets'].items():
            verify_prefix(out/f,size,names[f]['sha256'])
    return manifest


def continue_seed(base,plan,registered,seed,episodes=None,smoke=False):
    torch.set_num_threads(1)
    out = destination(seed,smoke)
    with run_lock(out):
        manifest = initialize(seed,plan,registered,out)
        if episodes is None:
            return f'{out.prefix} has verified EXP055 history'
        train(config(base,seed),out,episodes,resume=True,environment_id=fingerprint(plan),
              world_factory=partial(world,profile=workload(base['workload'])),
              simulator_factory=MeasuredMultistageSimulator)
        saved = read_checkpoint(out/'resume_latest.pt')
        if (saved['episode']!=episodes or saved['origin']!={'kind':'random'} or
                saved['environment_id']!=fingerprint(plan) or not saved['rng'] or not saved['replay']['columns'] or
                not saved['s']['optimizer']['state'] or not saved['d_optimizer']['state']):
            raise ValueError('Continuation state lost existing optimizer, replay or RNG')
        offsets = manifest['files']
        for f in ('train.jsonl','fixed.jsonl','updates.jsonl'):
            verify_prefix(out/f,offsets[f]['size'],offsets[f]['sha256'])
        if smoke:
            rows = list(records(out/'train.jsonl'))[-8:]
            update = list(records(out/'updates.jsonl'))[-1]
            if ([r['gs'] for r in rows]!=list(range(505,513)) or saved['d_updates']!=64 or
                    saved['s']['updates']!=4096 or
                    min(update[k] for k in ('d_change','s_change','s_target_change'))<=0 or
                    saved['source_history'][-1]['start_episode']!=START):
                raise ValueError('Resume smoke did not continue both D/S updates from GS504')
            env = world(seed,rows[0]['scene'],'train',profile=workload(base['workload']))
            error = audit_rows(env,rows)
            write(ROOT/'resume-smoke.json',dict(seed=seed,episode=saved['episode'],gs=512,d=saved['d_updates'],
                  s=saved['s']['updates'],max_work_error=error,history=saved['source_history'],
                  wall_seconds=saved['wall_seconds'],source=source()))
    return f'{out.prefix} verified through episode {episodes}'


@torch.inference_mode()
def evaluate_long(base,plan,registered,seed,target):
    torch.set_num_threads(1)
    out = destination(seed)
    with run_lock(out):
        state = read_checkpoint(out/'last.pt')
        h = digest(out/'last.pt')
        imported = read(out/'continuation.json')['import']
        if (state['episode']!=target or state['environment_id']!=fingerprint(plan) or
                imported['files']['resume_latest.pt']['sha256']!=registered['files'][str(seed)]['resume']):
            raise ValueError(f'Incomplete or incompatible continuation for seed {seed}')
    results = []
    for variant,profile in CASES:
        c = registered['files'][str(seed)]['case'][f'{variant}.{profile}']
        path = ROOT/f'long-eval.seed{seed}.{variant}.{profile}.episode{target}.json'
        stamp = fingerprint(dict(model=h,meta=c,episode=target,registered=fingerprint(registered)))
        if path.is_file():
            if read(path)['stamp']!=stamp:
                raise ValueError(f'Changed long checkpoint evidence: {path}')
            results.append(f'{path.name} reused')
            continue
        old = read(Path(c['path']))['arms']['last']
        trial = evaluate(base,plan,seed,variant,profile,c['scene'],c['population'],{'last':state})['last']
        same_flow(old['rows'],trial['rows'])
        write(path,dict(stamp=stamp,seed=seed,variant=variant,profile=profile,episode=target,last=trial,source=source()))
        results.append(f'{path.name} checked')
    return '\n'.join(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','smoke','train','evaluate'))
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--episodes',type=int,default=126)
    parser.add_argument('--seed',type=int,action='append',choices=SEEDS)
    args = parser.parse_args()
    if args.workers<1 or args.episodes<=START:
        parser.error('Positive workers and an absolute target above 63 episodes are required')
    base,plan = inputs()
    registered = read(ROOT/'protocol.json')['plan']
    if registered['exp055']!=fingerprint(plan):
        raise ValueError('EXP055 continuation configuration changed')
    if args.stage=='smoke':
        print(continue_seed(base,plan,registered,SEEDS[0],64,True))
        return
    seeds = list(dict.fromkeys(args.seed or SEEDS))
    tasks = [(base,plan,registered,s) for s in seeds]
    target = None if args.stage=='prepare' else args.episodes
    fn = evaluate_long if args.stage=='evaluate' else continue_seed
    jobs(fn,[(*task,target) for task in tasks],min(len(tasks),args.workers))


if __name__=='__main__':
    main()

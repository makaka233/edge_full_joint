"""Train the optional capped three-step Gate target against one-step Double DQN."""

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path

import torch

from comparison.run import jobs
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043, files as ds_files
from scripts.exp044_gate import fit
from scripts.exp044_run import write
from scripts.exp047_run import ARMS, GS, PERIODS, SETTINGS, GateRun, case, read, training
from scripts.exp049_run import ROOT as EXP049, SEEDS, evaluate
from scripts.exp050_audit import ROOT


def replay_key(row):
    return (row['branch'], row['gs']) if 'branch' in row else row['gs']


def replay_links(replay):
    # Auxiliary continuation nodes only supply returns. The focal transition is
    # the only member of its branch added to the uniform-sampling population.
    records = [node for row in replay for node in (row, *row.get('continuation', ()))]
    by_key = {replay_key(r): r for r in records}
    if len(by_key) != len(records):
        raise ValueError('Duplicate physical GS in Gate replay')
    links = {}
    for key, r in by_key.items():
        next_key = (r['branch'], r['gs']+1) if 'branch' in r else r['gs']+1
        after = by_key.get(next_key)
        if after is not None:
            if r['next'] != after['x'] or r['owner'] != after['owner']:
                raise ValueError('Gate return would cross a discontinuous replay state')
            links[key] = after
    return links


@torch.no_grad()
def three_step_targets(online, target, batch, links, gamma):
    """Extend returns only across observed next actions greedy under current online Q."""
    tips = list(batch)
    lengths = [1] * len(batch)
    rewards = torch.tensor([r['reward'] for r in batch], dtype=torch.float32)
    for step in (1, 2):
        next_states = torch.tensor([r['next'] for r in tips], dtype=torch.float32)
        greedy = online(next_states).argmax(1).tolist()
        increments = []
        for i, (r, action) in enumerate(zip(tips, greedy, strict=True)):
            after = links.get(replay_key(r)) if lengths[i] == step else None
            if after is not None and after['action'] == action:
                tips[i] = after
                lengths[i] += 1
                increments.append(after['reward'])
            else:
                increments.append(0.)
        rewards += gamma**step * rewards.new_tensor(increments)
    tail = torch.tensor([r['next'] for r in tips], dtype=torch.float32)
    greedy = online(tail).argmax(1, keepdim=True)
    value = target(tail).gather(1, greedy).squeeze(1)
    y = rewards + rewards.new_tensor([gamma**h for h in lengths])*value
    return y, dict(target_horizons=[lengths.count(h) for h in (1,2,3)])


class ThreeStepGate(GateRun):
    def improve(self):
        links = replay_links(self.replay)

        def build(online, target, batch):
            return three_step_targets(online, target, batch, links, SETTINGS['gamma'])

        updates = fit(self.online, self.target, self.optimizer, self.replay, self.replay_rng, SETTINGS,
                      target_builder=build)
        self.updates += len(updates)
        return updates


def train(root, plan, base, arm, seed, gs):
    return training(root, plan, base, arm, seed, gs, feature_arm='A', rng_arm='A',
                    gate_type=ThreeStepGate if arm == 'B' else GateRun)


def registration():
    evidence = read(ROOT / 'audit.json')
    if not evidence['gate']['three_step_candidate']:
        raise ValueError('The registered physical target diagnostic does not support training the candidate')
    old = read(EXP049 / 'protocol.json')['plan']
    base = read(EXP043 / 'protocol.json')['plan']
    if fingerprint(base) != old['exp043']:
        raise ValueError('EXP043 provenance differs')
    for seed in SEEDS:
        path = ds_files(seed) / 'last.pt'
        if digest(path) != old['model_sha256'][str(seed)]:
            raise ValueError(f'EXP043 frozen checkpoint differs: {path}')
    plan = deepcopy(old)
    plan.pop('phase_training')
    plan.update(experiment='EXP050', exp049=fingerprint(old),
                audit_sha256=fingerprint({k:evidence[k] for k in ('gate','values')}),
                training_by_arm={a:deepcopy(old['training_by_arm']['A']) for a in ARMS},
                target_protocol={'A':'one-step DDQN','B':'capped three-step DDQN with Watkins cut'},
                acceptance=dict(reward=.03,minimum_seed_wins=2),
                code={p:digest(Path(p)) for p in ('scripts/exp050_train.py','scripts/exp047_run.py',
                                                'scripts/exp044_gate.py','scripts/exp049_run.py',
                                                'scripts/exp049_report.py','edge_joint/adaptive_training.py',
                                                'edge_joint/adaptive_physics.py')})
    # Preserve workload profiles and vary only the external holdout keys.
    sources = [old['test'], old['training_by_arm']['A'], {k:[v] for k,v in old['validation'].items()}]
    original47 = read(Path('runs/results/EXP047/protocol.json'))['plan']
    sources.extend((original47['test'], original47['training'], {k:[v] for k,v in original47['validation'].items()}))
    existing = {c[k] for s in sources for seq in s.values() for c in seq for k in ('scene','environment')}
    for seed in SEEDS:
        test = [case('EXP050 independent',seed,i,c['profile'],c['loads'])
                for i,c in enumerate(old['test'][str(seed)])]
        for row in test:
            for key in ('scene','environment'):
                if row[key] in existing:
                    raise ValueError('New evaluation stream collides with an archived stream')
                existing.add(row[key])
        plan['test'][str(seed)] = test
    return plan,base


def load_protocol(root):
    with run_lock(RunFiles(root/'protocol')):
        plan,base = registration()
        path = root/'protocol.json'
        if path.exists():
            if read(path)['plan'] != plan:
                raise ValueError('Existing EXP050 data has different registered provenance')
        else:
            write(path,dict(plan=plan,source=source()))
    return plan,base


def formal_report(root):
    from scripts.exp049_report import collect, figures, plt, raw_axis, save, summarize
    with run_lock(RunFiles(root/'formal-report')):
        plan = read(root/'protocol.json')['plan']
        runs,cases = collect(root,plan)
        data = summarize(plan,runs,cases)
        data.pop('decision')
        margins = plan['acceptance']
        lines = ['# EXP050 两种子正式结果','','A 为一步 DDQN，B 为至多三步 DDQN。']
        passed = []
        for seed, result in data['by_seed'].items():
            d = result['delta']
            valid = (d['reward'] >= margins['reward'] and d['e2e'] < 0.
                     and all(d[k] <= 0 for k in ('e2e_penalized','p99','censored','terminal','spikes','dropped')))
            result['value_gate'] = valid
            passed.append(valid)
            lines.append(f'- {seed} 的 B−A Reward {d["reward"]:+.4f}，完成 E2E {d["e2e"]:+.4f} s，'
                         f'P99 {d["p99"]:+.4f} s，截尾 {d["censored"]:+.2f}。')
        data['candidate_pass'] = sum(passed) >= margins['minimum_seed_wins']
        for (arm, seed), run in runs.items():
            if arm == 'B':
                counts = [sum(r['target_horizons'][i] for r in run['updates']) for i in range(3)]
                data['training'][f'{arm}.seed{seed}']['target_horizons'] = counts
        lines.extend(['', f'两种子晋级判据 {"通过" if data["candidate_pass"] else "未通过"}。'
                      '逐轨迹和固定周期对比、固定验证、真实目标长度及尾部失败见 formal-summary.json。'])
        figures(root,runs,cases,data)
        fig,axes = plt.subplots(1,len(SEEDS),figsize=(12,4),layout='constrained')
        for ax,seed in zip(axes,SEEDS,strict=True):
            for arm in ARMS:
                raw_axis(ax,runs[arm,seed]['train'],'p99',arm,lw=.8)
            ax.set_title(str(seed))
            ax.legend()
        save(root,fig,9,'raw-p99')
        write(root/'formal-summary.json',data)
        atomic(root/'formal-report.md',lambda f:f.write(('\n'.join(lines)+'\n').encode('utf-8')))
    print(root/'formal-report.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','smoke','train','evaluate','report'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--gs',type=int,choices=range(PERIODS,GS+1,PERIODS),default=GS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage != 'train' and (args.arm or args.gs != GS):
        parser.error('--arm and --gs require train')
    if args.seed and args.stage not in ('train','evaluate'):
        parser.error('--seed requires train or evaluate')
    if args.stage == 'smoke':
        from scripts.exp050_smoke import smoke
        smoke()
        return
    if args.stage == 'report':
        formal_report(ROOT)
        return
    plan,base = load_protocol(ROOT)
    seeds = (args.seed,) if args.seed else SEEDS
    if args.stage == 'train':
        tasks = [(ROOT,plan,base,a,s,args.gs) for s,a in product(seeds,(args.arm,) if args.arm else ARMS)]
        jobs(train,tasks,min(args.workers,len(tasks)))
    elif args.stage == 'evaluate':
        tasks = [(ROOT,plan,base,s,c) for s in seeds for c in plan['test'][str(s)]]
        jobs(evaluate,tasks,min(args.workers,len(tasks)))
    else:
        print(ROOT/'protocol.json')


if __name__ == '__main__':
    main()

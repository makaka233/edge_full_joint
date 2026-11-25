"""Train EXP050's Gate with a small number of paired physical experiences."""

import argparse
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from time import perf_counter

from comparison.run import jobs
from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import RunFiles, run_lock
from edge_joint.inventory_joint import load_scale
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043, files as ds_files
from scripts.exp044_gate import learned, transition
from scripts.exp044_run import same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import ARMS, GS, PERIODS, SETTINGS, case, physical, read, training
from scripts.exp049_run import SEEDS, evaluate
from scripts.exp050_audit import ROOT as EXP050
from scripts.exp050_train import ThreeStepGate


ROOT = Path('runs/results/EXP051')
EXTRA_TRANSITIONS = 3
BRANCH_SLOTS = EXTRA_TRANSITIONS+1
DEFAULT_FORKS = 2
STRATUM = dict(load=.55, age_hours=8, cpu_backlog=25.)


class Openings:
    """Take the first causal high-pressure state in each window, with a last-slot fallback."""

    def __init__(self, periods, limit):
        self.windows = []
        self.reasons = {}
        through = periods-EXTRA_TRANSITIONS
        count = through-1  # eligible physical slots 2 through 'through'
        if limit > count:
            raise ValueError('Physical rollout too short for the registered openings')
        for i in range(limit):
            self.windows.append((2+i*count//limit, 1+(i+1)*count//limit))
        self.current = 0

    def __call__(self, env, sim, p, clock):
        if self.current == len(self.windows):
            return False
        low, high = self.windows[self.current]
        if not low <= p <= high:
            return False
        load = load_scale(env)
        age = 2*(p-clock.since)
        cpu = max(r.outstanding() for r in sim.cpu)
        pressure = ((load >= STRATUM['load']-1e-12 and age >= STRATUM['age_hours'])
                    or cpu >= STRATUM['cpu_backlog'])
        if not pressure and p < high:
            return False
        self.reasons[p] = dict(reason='pressure' if pressure else 'window_end',
                               load=load, age_hours=age, cpu_backlog=cpu, window=[low,high])
        self.current += 1
        return True


def collect_block(base, scene, model, run, block, epsilon, *, limit):
    start = perf_counter()
    periods = model[0].periods
    selector = Openings(periods,limit)
    openings, main_work = {}, {}
    main, weight = physical(base,scene,model,'A',learned(run.online,run.policy_rng,epsilon),
                            capture=selector,captured=openings,event_counts=main_work)
    main_seconds = perf_counter()-start
    if len(openings) != limit:
        raise AssertionError('Training capture count differs from registration')
    pairs, aux = [], []
    branch_start = perf_counter()
    actor_rng = deepcopy(run.policy_rng.bit_generator.state)
    greedy = learned(run.online)
    gamma = SETTINGS['gamma']
    for p, opening in openings.items():
        planned = main[p-1]['timing']['action']
        other = 1-planned
        env,sim = opening[1]
        opening_state = sim.snapshot()
        initial = fingerprint(opening_state)

        def choose(state, slot):
            act,q,_ = greedy(state,slot)
            return (other if slot == p else act),q,False

        work = {}
        rows,_,lag,branch_weight = rollout(base,scene,shorten(model,p+EXTRA_TRANSITIONS),choose,opening=opening,
                                           with_proposal=True,audit_policy=True,event_counts=work)
        factual = main[p-1:p-1+BRANCH_SLOTS]
        same_flow(factual,rows)
        if (branch_weight != weight or lag is None or fingerprint(sim.snapshot()) != initial
                or rows[0]['timing']['state'][:17] != factual[0]['timing']['state']
                or rows[0]['old_matrix'] != factual[0]['old_matrix']
                or rows[0]['backlog_in'] != factual[0]['backlog_in']
                or rows[0]['timing']['action'] != other):
            raise AssertionError('Counterfactual does not start at the factual decision state')
        for resource in ('cpu','wired','wireless'):
            if [r['work_at_start'] for r in rows[0][resource]] != [r['work_at_start'] for r in factual[0][resource]]:
                raise AssertionError('Copied opening resource work differs')
        proposal = rows[0]['timing']['proposal']
        if planned == 1 and proposal != factual[0]['active_matrix']:
            raise AssertionError('Factual and counterfactual decoders did not produce the same unique proposal')
        # The proposal is recorded for audit only. Gate sees the original 17 features.
        for row in rows:
            row['timing']['state'] = row['timing']['state'][:17]
        branch = f'b{block}.p{p}'
        path = []
        prior_e2e = lag['e2e_penalized']
        previous_deadline = (p-1)*model[0].period_seconds+env.tail
        for row,nxt in zip(rows[:-1],rows[1:],strict=True):
            if row['active_matrix'] != nxt['old_matrix'] or previous_deadline > row['end']:
                raise AssertionError('Auxiliary delayed cohort or physical successor is not observed')
            t = transition(row,nxt,prior_e2e,weight,gamma)
            path.append(dict(t, owner=run.identity, branch=branch, gs=(block-1)*periods+row['period']))
            prior_e2e = row['e2e_penalized']
            previous_deadline = row['score_deadline']
        sample = dict(path[0],continuation=path[1:])
        # Main r_t uses its own earlier cohort score, potentially changed by its action.
        main_t = transition(factual[0],factual[1],main[p-2]['e2e_penalized'],weight,gamma)
        pairs.append(dict(owner=run.identity, block=block, gs=sample['gs'], branch=branch, opening=initial,
                          snapshot=opening_state, env_rng=env.w.rng.bit_generator.state, selection=selector.reasons[p],
                          proposal=proposal, factual=dict(transition=main_t,rows=factual),
                          auxiliary=dict(transition=sample,rows=rows,previous_cohort=lag,events=work['events'])))
        aux.append(sample)
    if actor_rng != run.policy_rng.bit_generator.state:
        raise AssertionError('Auxiliary physics consumed the factual Gate RNG')
    snapshot_seconds = main_work['capture_seconds']
    accounting = dict(main_gs=len(main), main_events=main_work['events'], main_wall_seconds=main_seconds-snapshot_seconds,
                      auxiliary_gs=sum(len(r['auxiliary']['rows']) for r in pairs),
                      auxiliary_events=sum(r['auxiliary']['events'] for r in pairs),
                      auxiliary_wall_seconds=(perf_counter()-branch_start+snapshot_seconds if pairs else 0.),
                      auxiliary_transitions=len(aux),
                      auxiliary_continuations=sum(len(s['continuation']) for s in aux))
    return main,weight,aux,pairs,accounting


def registration(limit):
    source_plan = read(EXP050/'protocol.json')['plan']
    base = read(EXP043/'protocol.json')['plan']
    if (source_plan['experiment'] != 'EXP050' or source_plan['training_by_arm']['A'] != source_plan['training_by_arm']['B']
            or fingerprint(base) != source_plan['exp043']):
        raise ValueError('Unexpected EXP050 workload or EXP043 checkpoint registry')
    for seed in SEEDS:
        if digest(ds_files(seed)/'last.pt') != source_plan['model_sha256'][str(seed)]:
            raise ValueError(f'EXP043 model changed for seed {seed}')
    plan = deepcopy(source_plan)
    plan.update(experiment='EXP051', exp050=fingerprint(source_plan),
                treatment=dict(max_forks=limit,stratum=STRATUM,branch_slots=BRANCH_SLOTS,
                               sampling='first pressure event per window, else last eligible slot',
                               branch='opposite first action, then greedy before block optimization'),
                target_protocol={arm:'EXP050 capped three-step DDQN' for arm in ARMS},
                acceptance=dict(reward_margin=.03,e2e_ratio=.01,probe_margin=.10,auxiliary_gs_ratio=1/3),
                code={p:digest(Path(p)) for p in ('scripts/exp051_run.py','scripts/exp051_report.py',
                                                'scripts/exp050_train.py','scripts/exp049_run.py','scripts/exp049_report.py',
                                                'scripts/exp047_run.py','scripts/exp047_report.py',
                                                'scripts/exp046_collect.py','scripts/exp044_gate.py',
                                                'edge_joint/adaptive_training.py','edge_joint/adaptive_physics.py')})
    history = [read(Path(f'runs/results/EXP{n}/protocol.json'))['plan'] for n in ('047','049','050')]
    used = set()
    for old in history:
        trains = old['training_by_arm']['A'] if 'training_by_arm' in old else old['training']
        for series in (old['test'],trains,{s:[x] for s,x in old['validation'].items()}):
            used.update(r[k] for series_for_seed in series.values() for r in series_for_seed for k in ('scene','environment'))
    for seed in SEEDS:
        test = [case('EXP051 independent',seed,i,s['profile'],s['loads'])
                for i,s in enumerate(source_plan['test'][str(seed)])]
        for row in test:
            for key in ('scene','environment'):
                if row[key] in used:
                    raise ValueError('EXP051 test key overlaps another external stream')
                used.add(row[key])
        plan['test'][str(seed)] = test
    return plan,base


def register(root, requested):
    with run_lock(RunFiles(root/'protocol')):
        path = root/'protocol.json'
        saved = read(path)['plan'] if path.exists() else None
        limit = requested if requested is not None else (saved['treatment']['max_forks'] if saved else DEFAULT_FORKS)
        plan,base = registration(limit)
        if saved is None:
            write(path,dict(plan=plan,source=source()))
        elif saved != plan:
            raise ValueError('EXP051 saved protocol differs from the current source, fork limit or frozen dependencies')
    return plan,base


def train(root,plan,base,arm,seed,gs):
    limit = plan['treatment']['max_forks'] if arm == 'B' else 0
    return training(root,plan,base,arm,seed,gs,feature_arm='A',rng_arm='A',gate_type=ThreeStepGate,
                    collect_block=partial(collect_block,limit=limit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('p0','register','smoke','train','evaluate','report'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--gs',type=int,choices=range(PERIODS,GS+1,PERIODS),default=GS)
    parser.add_argument('--max-forks',type=int,choices=(1,2))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage != 'train' and (args.arm or args.gs != GS):
        parser.error('--arm and --gs require train')
    if args.seed and args.stage not in ('train','evaluate'):
        parser.error('--seed requires train or evaluate')
    if args.max_forks and args.stage not in ('register','train'):
        parser.error('--max-forks requires register or train')
    if args.stage == 'smoke':
        from scripts.exp051_smoke import smoke
        smoke()
        return
    if args.stage in ('report','p0'):
        from scripts.exp051_report import p0, report
        (report if args.stage == 'report' else p0)(ROOT)
        return
    plan,base = register(ROOT,args.max_forks)
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

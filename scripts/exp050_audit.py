"""Reconstruct EXP049 value learning and audit seven paired physical states."""

import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter

import torch

from comparison.run import jobs
from edge_joint.adaptive_runner import source
from scripts.exp027_run import fingerprint
from scripts.exp029_run import digest
from scripts.exp043_run import ROOT as EXP043
from scripts.exp044_gate import learned
from scripts.exp044_run import frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import (ARMS, ROOT as EXP047, VALIDATE, GateRun, experiences,
                                read, read_model, rows_at, run_files)
from scripts.exp047_smoke import equivalent
from scripts.exp048_audit import ROOT as EXP048, slot
from scripts.exp049_report import collect, coverage
from scripts.exp049_run import ROOT as EXP049, SEEDS


ROOT = Path('runs/results/EXP050')
FOCALS = (
    ('old-burst', 47, 20261117, 'burst', 8),
    ('shift-first', 49, 20261117, 'shift', 2),
    ('shift-onset', 49, 20261117, 'shift', 7),
    ('early-onset', 49, 20261117, 'burst_early', 6),
    ('late-first', 49, 20261117, 'burst_late', 3),
    ('natural-control', 49, 20261113, 'natural', 2),
    ('burst-control', 49, 20261113, 'burst_early', 6),
)
HORIZONS = (1, 3, 6)
METRICS = ('reward', 'e2e', 'e2e_penalized', 'p99', 'censored', 'terminal', 'cpu', 'wire', 'actual_changes')


def ledger_case(experiment, seed, profile, plan):
    root = EXP047 if experiment == 47 else EXP049
    path = root / f'eval.{profile}.seed{seed}.json'
    case = read(path)
    model = run_files(root, 'A', seed) / 'last.pt'
    if (case['identity']['gate_sha256']['A'] != digest(model) or case['identity']['stamp'] != fingerprint(plan)):
        raise ValueError(f'Archived evaluation does not own its source model: {path}')
    return case, model


def load_network(path, arm, seed):
    saved = read_model(path)
    net = GateRun(arm, seed, saved['identity']['stamp'], feature_arm='A', rng_arm='A').online
    if saved['identity']['seed'] != seed or saved['identity']['arm'] != arm:
        raise ValueError(f'Wrong seed or algorithm arm in checkpoint: {path}')
    net.load_state_dict(saved['q'])
    return net


def chronology(data):
    a,b = (data['methods'][arm]['rows'] for arm in ARMS)
    same_flow(a, b)
    pair = list(zip(a, b, strict=True))
    first = next((r['period'] for r,s in pair if r['action'] != s['action']), None)
    if first is None:
        raise ValueError('The registered EXP049 paired case unexpectedly has no action difference')
    opening = pair[first-1]
    for key in ('old_matrix', 'backlog_in', 'cpu', 'wired', 'wireless'):
        x,y = (r[key] for r in opening)
        if key in ('cpu', 'wired', 'wireless'):
            x,y = ([i['work_at_start'] for i in j] for j in (x,y))
        if x != y:
            raise ValueError('Earlier physical states differ before the first policy divergence')
    segments = {}
    for label, subset in (('before', pair[:first-1]), ('from_first', pair[first-1:])):
        segments[label] = [physical_metrics([p[i] for p in subset]) for i in range(2)]
    a_score,b_score,ref = (data['methods'][m]['metrics'] for m in ('A','B','fixed2'))
    return dict(first=first,
                first_change=next((r['period'] for r,s in pair if r['active_matrix'] != s['active_matrix']), None),
                first_loss=next((r['period'] for r,s in pair if s['reward_raw']-r['reward_raw'] >= 1), None),
                first_censored={arm: next((r['period'] for r in rows if r['censored']), None)
                                for arm, rows in zip(ARMS,(a,b),strict=True)},
                delta={k:b_score[k]-a_score[k] for k in METRICS},
                metrics=dict(A=a_score, B=b_score, fixed2=ref), segments=segments,
                rows=dict(A=[slot(r) for r in a],B=[slot(r) for r in b]))


def marks(run):
    return deepcopy(dict(online=run.online.state_dict(), target=run.target.state_dict()))


def reconstruct(plan, arm, seed, ledger):
    paths = run_files(EXP049, arm, seed)
    saved = read_model(paths / 'resume_latest.pt')['gate']
    run = GateRun(arm, seed, fingerprint(plan), feature_arm='A', rng_arm='A')
    archived = rows_at(paths / 'updates.jsonl')
    snapshots, per_mark = {0:marks(run)}, {}
    for block in ledger['blocks']:
        gs = block['gs']
        rows = ledger['train'][gs-24:gs]
        # Reconstruct the private epsilon stream and verify the model's logged values
        # on identical states, before any of this block's experiences are added.
        policy = learned(run.online, run.policy_rng, block['epsilon'])
        with torch.no_grad():
            for r in rows[1:]:
                stored = r['timing']
                action,q,explored = policy(stored['state'], r['period'])
                if action != stored['action'] or explored != stored['explored'] or q != stored['q']:
                    raise ValueError('Could not exactly reconstruct the archived behavior policy')
        if gs == 24:
            run.fit_scaler(rows)
        run.collect(experiences(rows, 10., run.identity))
        start = run.updates
        updates = run.improve()
        if any(any(row[k] != stored[k] for k in ('loss','td_mae','gradient'))
               for row,stored in zip(updates,archived[start:run.updates],strict=True)):
            raise ValueError('Reconstructed gradient ledger does not match the original')
        if gs in VALIDATE:
            snapshots[gs] = marks(run)
            counts = coverage(ledger['train'][:gs], run.replay, run.replay_rng.bit_generator.state, seed)
            per_mark[gs] = dict(coverage=counts, mean_td=sum(r['td_mae'] for r in updates)/len(updates),
                                mean_loss=sum(r['loss'] for r in updates)/len(updates),
                                fixed=physical_metrics(ledger['fixed'][gs]))
    current = run.snapshot()
    for k in ('online','target','optimizer','replay','policy_rng','replay_rng','updates','scaler_fitted'):
        if not equivalent(current[k], saved[k]):
            raise ValueError(f'Reconstruction differs from committed training state at {k}')
    restored = GateRun(arm, seed, fingerprint(plan), feature_arm='A', rng_arm='A')
    restored.restore(saved)
    if not equivalent(restored.snapshot(), saved):
        raise ValueError('The archived recovery state does not round-trip')
    return snapshots, per_mark


@torch.inference_mode()
def evaluate_targets(model, target, cells, gamma):
    result = {}
    for name, data in cells.items():
        rows = data['rows']
        x = torch.tensor([r['timing']['state'][:17] for r in rows], dtype=torch.float32)
        q = model(x)
        action = q.argmax(1, keepdim=True)
        tail = target(x).gather(1, action)[:,0].tolist()
        prev = data['previous_cohort']['e2e_penalized']
        rewards = []
        for r in rows[:-1]:
            rewards.append(-r['holding_cost']-r['switch_cost']-10.*(prev or 0.)/gamma)
            prev = r['e2e_penalized']
        matched = [r['timing']['action'] == a for r,a in zip(rows,action[:,0].tolist(),strict=True)]
        records = {}
        for h in HORIZONS:
            delayed = sum(gamma**i*rewards[i] for i in range(h))
            pending = gamma**(h-1)*10.*(rows[h-1]['e2e_penalized'] or 0.)
            if abs(delayed-data['aligned'][str(h)]-pending) > 1e-9:
                raise AssertionError('Delayed credit does not reconcile with the original cohort horizon')
            bootstrap = gamma**h*tail[h]
            records[h] = dict(delayed=delayed, bootstrap=bootstrap, target=delayed+bootstrap,
                              eligible=all(matched[1:h]), bootstrap_action=int(action[h,0]))
        result[name] = dict(targets=records, q=q[0].tolist())
    return dict(cells=result, gap=result['keep']['q'][1]-result['keep']['q'][0],
                y_delta={h:result['redeploy']['targets'][h]['target']-result['keep']['targets'][h]['target']
                         for h in HORIZONS},
                matched={h:all(x['targets'][h]['eligible'] for x in result.values()) for h in HORIZONS})


def run_offline(plan, local):
    ledgers,cases = collect(EXP049, plan)
    paired = [dict(seed=seed, profile=profile, **chronology(case)) for (seed,profile),case in cases.items()]
    evolution, outputs = {}, {}
    for arm,seed in product(ARMS,SEEDS):
        snapshots, per_mark = reconstruct(plan, arm, seed, ledgers[arm,seed])
        evolution[f'{arm}.seed{seed}'] = per_mark
        online,target = (GateRun(arm,seed,fingerprint(plan),feature_arm='A',rng_arm='A').online for _ in range(2))
        for f in local:
            if f['seed'] != seed:
                continue
            ratings = outputs.setdefault(f['id'], {}).setdefault(arm, {})
            for gs,saved in snapshots.items():
                online.load_state_dict(saved['online'])
                target.load_state_dict(saved['target'])
                if gs == 0:
                    for net in (online,target):
                        for field in ('center','scale'):
                            getattr(net,field).copy_(snapshots[288]['online'][field])
                ratings[gs] = evaluate_targets(online,target,f['cells'],plan['q']['gamma'])
        # Check reconstructed historical milestone weights against original recorded state values.
        for gs in VALIDATE:
            online.load_state_dict(snapshots[gs]['online'])
            with torch.no_grad():
                for p,s in zip(plan['historical_probes'][str(seed)],ledgers[arm,seed]['fixed'][gs][0]['historical_q'],
                               strict=True):
                    if online(torch.tensor(p['state'],dtype=torch.float32)).tolist() != s['q']:
                        raise ValueError('Archived historical milestone Q did not reproduce')
    return paired, evolution, outputs


@torch.inference_mode()
def fork_group(seed, profile, experiment, points, plans, base):
    torch.set_num_threads(1)
    started = perf_counter()
    plan = plans[experiment]
    case,path = ledger_case(experiment,seed,profile,plan)
    scene = case['identity']['case']
    continuation = run_files(EXP049,'A',seed) / 'last.pt'
    origin = learned(load_network(path,'A',seed))
    policy = learned(load_network(continuation,'A',seed))
    stamp = fingerprint(dict(code=digest(Path(__file__)),case=scene,origin=digest(path),
                             continuation=digest(continuation),exp043=plan['model_sha256'][str(seed)]))
    paths = [ROOT / f'fork.{name}.json' for name,_ in points]
    if all(p.exists() and read(p)['stamp'] == stamp for p in paths):
        return f'{seed} {profile} reused'
    last = max(p for _,p in points)
    ds = frozen(base,plan,seed,last+6)
    stem,openings,_,weight = rollout(base,scene,shorten(ds,last),origin,
                                     capture=[p for _,p in points],with_proposal=False,audit_policy=True)
    archived = case['methods']['A']['rows']
    same_flow(stem,archived[:last])
    for r,s in zip(stem,archived,strict=False):
        for k in ('action','active_matrix','old_matrix','cpu','wired','wireless','backlog_in','timing'):
            if r[k] != s[k]:
                raise ValueError(f'Could not reconstruct {k} at slot {r["period"]}')
    for (name,p),out in zip(points,paths,strict=True):
        opening = openings[p]
        env,sim = opening[1]
        state = sim.snapshot()
        initial = fingerprint(state)
        if experiment == 47 and initial != read(EXP048 / 'fork.A.20261117.burst.p8.json')['initial']:
            raise ValueError('Archived EXP048 physical opening changed')
        cells = {}
        for a,arm in enumerate(('keep','redeploy')):
            def choose(s,t):
                decision = policy(s,t)
                return (a,decision[1],False) if t == p else decision

            rows,_,lag,_ = rollout(base,scene,shorten(ds,p+6),choose,opening=opening,
                                    with_proposal=True,audit_policy=True)
            if (fingerprint(sim.snapshot()) != initial or rows[0]['timing']['state'][:17] != stem[p-1]['timing']['state']
                    or rows[0]['old_matrix'] != stem[p-1]['old_matrix']):
                raise ValueError('Branches do not share the original physical start')
            for k in ('cpu','wired','wireless'):
                if [r['work_at_start'] for r in rows[0][k]] != [r['work_at_start'] for r in stem[p-1][k]]:
                    raise ValueError('Opening resource queue changed')
            gamma = plans[49]['q']['gamma']
            scores = {h:sum(gamma**i*r['reward_raw'] for i,r in enumerate(rows[:h])) for h in HORIZONS}
            aligned = {h:v-weight*(lag['e2e_penalized'] or 0.)/gamma for h,v in scores.items()}
            cells[arm] = dict(rows=rows,previous_cohort=lag,raw=scores,aligned=aligned,
                              six=physical_metrics(rows[:6]))
        a,b = (cells[k]['rows'] for k in ('keep','redeploy'))
        same_flow(a,b)
        if a[0]['timing']['proposal'] != b[0]['timing']['proposal']:
            raise ValueError('The two arms used different proposals')
        write(out,dict(id=name,seed=seed,profile=profile,experiment=experiment,period=p,stamp=stamp,
                       initial=initial,snapshot=state,node_demand=env.w.node_demand().tolist(),
                       cells=cells,delta={h:cells['redeploy']['aligned'][h]-cells['keep']['aligned'][h] for h in HORIZONS},
                       raw_delta={h:cells['redeploy']['raw'][h]-cells['keep']['raw'][h] for h in HORIZONS},
                       wall_seconds=perf_counter()-started,source=source()))
    return f'{seed} {profile} physical openings {points} verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fork','report'))
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plans = {n:read(root / 'protocol.json')['plan'] for n,root in ((47,EXP047),(49,EXP049))}
    if args.stage == 'fork':
        groups = defaultdict(list)
        for name,experiment,seed,profile,p in FOCALS:
            groups[seed,profile,experiment].append((name,p))
        base = read(EXP043 / 'protocol.json')['plan']
        jobs(fork_group,[(*key,points,plans,base) for key,points in groups.items()],args.workers)
        return
    from scripts.exp050_report import figures, gate_diagnostic
    started = perf_counter()
    local = [read(ROOT / f'fork.{f[0]}.json') for f in FOCALS]
    paired,evolution,outputs = run_offline(plans[49],local)
    gate = gate_diagnostic(local,outputs)
    figures(ROOT,paired,evolution,local,outputs)
    write(ROOT / 'audit.json',dict(chronology=paired,checkpoints=evolution,values=outputs,gate=gate,
                                   source=source(),wall_seconds=perf_counter()-started))
    print(ROOT / 'audit.json')


if __name__ == '__main__':
    main()

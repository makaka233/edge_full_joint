"""Run one small physical regression on the EXP049 launcher and recovery path."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import append
from scripts.exp027_run import fingerprint
from scripts.exp044_gate import fixed
from scripts.exp044_run import frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import rollout, shorten
from scripts.exp047_run import GateRun, read_model, rows_at, run_files
from scripts.exp047_smoke import equivalent, rejects
from scripts.exp049_report import coverage, figures
from scripts.exp049_run import ARMS, METHODS, ROOT, SEEDS, finite_probe, paired_training, registration, train


def smoke():
    started = perf_counter()
    torch.set_num_threads(1)
    plan, base = registration()
    for seed in SEEDS:
        key = str(seed)
        original = plan['training_by_arm']['A'][key]
        assert paired_training(seed, original) == (original, plan['training_by_arm']['B'][key],
                                                   plan['phase_training'][key])
        for a, b in zip(original, plan['training_by_arm']['B'][key], strict=True):
            assert {k:v for k,v in a.items() if k != 'loads'} == {k:v for k,v in b.items() if k != 'loads'}
            if a['profile'] != 'burst':
                assert a == b
            else:
                assert a['loads'].count(.70) == b['loads'].count(.70) == 6
                assert a['loads'][0] == b['loads'][0] == .32
    seed = SEEDS[0]
    key = str(seed)
    mini = deepcopy(plan)
    mini.update(experiment='EXP049 smoke', slots=3, blocks=2, fixed_at=[0])
    for arm in ARMS:
        scene = mini['training_by_arm'][arm][key][0]
        scene.update(loads=[.32]*3, profile='burst')
        second = mini['training_by_arm'][arm][key][1]
        second.update(loads=[.32, .70, .32] if arm == 'B' else [.32, .32, .70], profile='burst')
    mini['validation'][key]['loads'] = [.32, .32, .32]
    stamp = fingerprint(mini)
    a, b, other = (GateRun(arm, s, stamp, feature_arm='A', rng_arm='A')
                   for arm,s in (('A', seed), ('B', seed), ('A', SEEDS[1])))
    assert equivalent(a.online.state_dict(), b.online.state_dict())
    assert a.policy_rng is not b.policy_rng and a.replay_rng is not b.replay_rng
    assert a.policy_rng.bit_generator.state == b.policy_rng.bit_generator.state
    assert not equivalent(a.online.state_dict(), other.online.state_dict())
    assert a.online.center.data_ptr() != b.online.center.data_ptr()
    assert a.target.body[0].weight.data_ptr() != b.target.body[0].weight.data_ptr()
    assert a.optimizer.param_groups[0]['params'][0] is not b.optimizer.param_groups[0]['params'][0]
    ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='smoke-', dir=ROOT) as folder:
        root = Path(folder)
        for arm in ARMS:
            train(root, mini, base, arm, seed, 3)
        paths = {arm: run_files(root, arm, seed) for arm in ARMS}
        snap = {arm: read_model(p / 'resume_latest.pt') for arm,p in paths.items()}
        left, right = (snap[arm]['gate'] for arm in ARMS)
        for field in ('online', 'target', 'optimizer', 'policy_rng', 'replay_rng'):
            assert equivalent(left[field], right[field])
        assert left['updates'] == right['updates'] == 48
        assert not equivalent(left['online'], a.online.state_dict())
        assert not equivalent(left['target'], a.target.state_dict())
        assert left['replay'][0]['owner'] != right['replay'][0]['owner']
        rejects(lambda: b.restore(left))
        rejects(lambda: other.restore(left))
        rejects(lambda: b.collect(left['replay']))
        b.restore(right)
        assert equivalent(right, b.snapshot())
        log = paths['B'] / 'train.jsonl'
        initial = rows_at(log)
        same_flow(rows_at(paths['A'] / 'train.jsonl'), initial)
        append(log, [{'uncommitted': True}])
        train(root, mini, base, 'B', seed, 6)
        actual = rows_at(log)
        assert actual[:3] == initial and len(actual) == 6
        result = read_model(paths['B'] / 'resume_latest.pt')
        assert result['resumes'] == 1 and result['blocks'] == 2 and result['gate']['updates'] == 96
        assert 'already committed GS6' in train(root, mini, base, 'B', seed, 6)
        assert read_model(paths['B'] / 'last.pt')['gs'] == 6
        assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient'])
                   for p in paths.values() for r in rows_at(p / 'updates.jsonl'))

        runs, evaluation = {}, {}
        for arm,p in paths.items():
            train_rows, fixed_rows = rows_at(p / 'train.jsonl'), rows_at(p / 'fixed.jsonl')
            committed = read_model(p / 'resume_latest.pt')['gate']
            observed = coverage(train_rows, committed['replay'], committed['replay_rng'], seed, periods=3)
            runs[arm, seed] = dict(train=train_rows, fixed={0: fixed_rows}, blocks=rows_at(p / 'blocks.jsonl'),
                                   coverage=observed)
            evaluation[arm] = dict(rows=fixed_rows, metrics=physical_metrics(fixed_rows))
        b.restore(result['gate'])
        scene = mini['training_by_arm']['B'][key][1]
        ds = frozen(base, plan, seed, 3)
        prefix, points, _, _ = rollout(base, scene, shorten(ds, 2), fixed(2), capture=(2,),
                                       with_proposal=False, audit_policy=True)
        p = finite_probe(base, scene, ds, points[2], {'A': a.online, 'B': b.online}, prefix[1], horizon=1)
        keep, redeploy = (p['cells'][arm]['rows'][0] for arm in ('keep', 'redeploy'))
        assert keep['action'] == 'KEEP' and keep['old_matrix'] == keep['active_matrix']
        assert redeploy['action'] == 'REDEPLOY' and redeploy['timing']['proposal'] == redeploy['active_matrix']
        # Use the real three-slot smoke ledgers solely to exercise the figure and reporting interface.
        methods = {m: evaluation[m if m in ARMS else 'A'] for m in METHODS}
        summary = dict(fixed={f'{a}.seed{seed}': {0: x['metrics']} for a,x in evaluation.items()},
                       historical_q={f'{a}.seed{seed}': {0: x['rows'][0]['historical_q']} for a,x in evaluation.items()},
                       by_seed={key: dict(metrics={m:x['metrics'] for m,x in methods.items()})})
        figures(root, runs, {(seed, 'burst_late'): dict(methods=methods, probe=p)}, summary)
        assert len(list(root.glob('fig*.png'))) == 8
    write(ROOT / 'smoke.json', dict(source=source(), wall_seconds=perf_counter()-started, physical_slots=21,
                                    validated_seeds=list(SEEDS), physical_seed=seed, paired_first_updates=48,
                                    resumed_updates=96, figures=8, isolation=True, recovery=True,
                                    matched_opening_probe=True, schedules=plan['phase_training']))
    print(ROOT / 'smoke.json')

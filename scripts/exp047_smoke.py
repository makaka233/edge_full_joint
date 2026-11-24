"""Minimal real-physics and checkpoint regression for EXP047."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from edge_joint.adaptive_storage import append, publish, random_state
from edge_joint.adaptive_runner import source
from scripts.exp044_gate import case_env
from scripts.exp044_run import TimingSimulator, frozen, physical_metrics, same_flow, write
from scripts.exp046_collect import proposal_features
from scripts.exp047_report import METRICS, figures
from scripts.exp047_run import (ROOT, SEEDS, GateRun, case, experiences, physical, projection,
                                publish_models, read_model, recover_logs, registration, rows_at, run_files)


def equivalent(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b


def rejects(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError('Foreign replay or checkpoint was accepted')


def smoke():
    started = perf_counter()
    torch.set_num_threads(1)
    plan, base = registration()
    seed, other_seed = SEEDS[:2]
    a, b, other, repeat = [GateRun(arm, s, 'smoke') for arm, s in
                           (('A', seed), ('B', seed), ('A', other_seed), ('A', seed))]
    assert equivalent(a.online.state_dict(), repeat.online.state_dict())
    assert not equivalent(a.online.state_dict(), other.online.state_dict())
    assert torch.equal(a.online.body[0].weight, b.online.body[0].weight[:, :17])
    assert torch.equal(a.online.body[-1].weight, b.online.body[-1].weight)
    for left, right in ((a, b), (a, other), (a, repeat)):
        assert left.replay is not right.replay
        assert left.optimizer.param_groups[0]['params'][0] is not right.optimizer.param_groups[0]['params'][0]
        assert left.online.center.data_ptr() != right.online.center.data_ptr()
        assert left.target.body[0].weight.data_ptr() != right.target.body[0].weight.data_ptr()
        assert left.policy_rng is not right.policy_rng and left.replay_rng is not right.replay_rng
    assert a.policy_rng.bit_generator.state != b.policy_rng.bit_generator.state
    assert a.replay_rng.bit_generator.state != other.replay_rng.bit_generator.state
    assert a.policy_rng.bit_generator.state == repeat.policy_rng.bit_generator.state
    ds = frozen(base, plan, seed, 3)
    before = [{k: v.detach().clone() for k, v in nn.state_dict().items()} for nn in (ds[1], ds[2].online)]
    scene = case('smoke', seed, 0, 'shift', [.32, .32, .55])

    def forced(run):
        def choose(state, period):
            q = run.online(torch.tensor(state, dtype=torch.float32)).tolist()
            return int(period == 3), q, False
        return choose

    rng = random_state()
    ra, weight = physical(base, scene, ds, 'A', forced(a))
    rb, _ = physical(base, scene, ds, 'B', forced(b))
    again, _ = physical(base, scene, ds, 'A', forced(repeat))
    assert equivalent(rng, random_state()), 'Proposal or rollout changed ambient RNG'
    assert all(equivalent(left, right.state_dict()) for left, right in zip(before, (ds[1], ds[2].online)))
    assert ds[2].updates == 0
    same_flow(ra, rb)
    same_flow(ra, again)
    for rows in (ra, rb, again):
        assert [r['action'] for r in rows] == ['BOOTSTRAP', 'KEEP', 'REDEPLOY']
        assert rows[1]['active_matrix'] == rows[1]['old_matrix'] and rows[1]['switch_cost'] == 0
    for first, second, third in zip(ra, rb, again, strict=True):
        for key in ('reward_raw', 'active_matrix', 'e2e_penalized', 'backlog_out', 'switch_cost'):
            assert first[key] == second[key] == third[key]
        if first['timing']:
            assert first['timing']['state'] == second['timing']['state'][:17]
            assert np.allclose(first['timing']['q'], second['timing']['q'], rtol=0., atol=1e-7)
    assert rb[2]['timing']['proposal'] == rb[2]['active_matrix']
    assert rb[2]['profile']['proposal_decode_seconds'] > 0.
    assert rb[2]['active_matrix'] != rb[2]['old_matrix']

    env = case_env(base, scene)
    bits = np.frombuffer(bytes.fromhex(ra[0]['active_matrix']), dtype=np.uint8)
    layout = np.unpackbits(bits).reshape(env.K, env.N).astype(bool)
    assert proposal_features(env, layout, layout, env.w.node_demand())['switch'] == 0.
    sim = TimingSimulator(env, layout, None, 1, ds[2].version, False)
    result = sim.redeploy(layout)
    assert result['switch_cost'] == 0. and result['activated_at'] is None

    for rows in (ra, rb):
        for row in rows:
            row['gs'] = row['period']
    for run, rows in ((a, ra), (b, rb)):
        run.fit_scaler(rows)
        samples = experiences(rows, weight, run.identity)
        run.collect(samples)
        assert len(samples) == 1 and samples[0]['gs'] == 2 and samples[0]['action'] == 0
        assert samples[0]['next'] == rows[2]['timing']['state']
        rejects(lambda: other.collect(samples))
    assert not other.replay and not other.scaler_fitted
    rejects(lambda: b.collect(a.replay))
    rejects(lambda: a.collect(b.replay))
    initial = a.snapshot()
    online_before = initial['online']['body.0.weight'].clone()
    target_before = initial['target']['body.0.weight'].clone()
    updates = a.improve()
    assert len(updates) == a.updates and all(np.isfinite(value) for row in updates for value in row.values())
    assert not torch.equal(a.online.body[0].weight, online_before)
    assert not torch.equal(a.target.body[0].weight, target_before)
    assert not other.optimizer.state and not b.optimizer.state

    ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='smoke-', dir=ROOT) as folder:
        folder = Path(folder)
        paths = run_files(folder, 'A', seed)
        path = paths / 'train.jsonl'
        offset = append(path, ra)
        append(path, [{'uncommitted': True}])
        checkpoint = paths / 'resume_latest.pt'
        projected = projection(a, 3)
        committed = dict(gate=a.snapshot(), offsets={'train': offset}, last=projected, best=projected)
        publish(checkpoint, committed)
        publish_models(paths, committed)
        saved = read_model(checkpoint)
        assert equivalent(committed, saved)
        rejects(lambda: b.restore(saved['gate']))
        rejects(lambda: other.restore(saved['gate']))
        repeat.restore(saved['gate'])
        recover_logs(paths, saved['offsets'])
        assert len(rows_at(path)) == 3
        assert equivalent(saved['gate'], repeat.snapshot())
        assert a.policy_rng.random() == repeat.policy_rng.random()
        assert a.improve() == repeat.improve()
        assert equivalent(a.snapshot(), repeat.snapshot())
        actual = {'A': rows_at(path), 'B': rb}
        cells = {arm: dict(rows=rows, metrics=physical_metrics(rows)) for arm, rows in actual.items()}
        chart_runs = {(arm, seed): dict(train=data['rows'], fixed={0: data['rows']}, updates=[
            dict(row, update=i) for i, row in enumerate(updates, 1)] if arm == 'A' else [],
            blocks=[dict(gs=3, epsilon=.6, metrics=data['metrics'])]) for arm, data in cells.items()}
        chart_summary = dict(fixed={arm: {str(seed): {0: data['metrics']}} for arm, data in cells.items()},
                             mean_and_population_sd={arm: {k: dict(mean=data['metrics'][k], sd=0.)
                                                          for k in METRICS} for arm, data in cells.items()})
        figures(folder, chart_runs, {(seed, 'burst'): dict(methods=cells)}, chart_summary)
        assert len(list(folder.glob('fig*.png'))) == 7
    evidence = dict(physical_slots=9, decisions=['BOOTSTRAP', 'KEEP', 'REDEPLOY'], optimizer_steps=len(updates),
                    seeds_checked=list(SEEDS[:2]), weights_registered=list(SEEDS), isolated_arms=['A', 'B'],
                    raw_rewards=[r['reward_raw'] for r in ra], fixed_external_trace=[r['trace_sha256'] for r in ra],
                    proposal_decode_ms=rb[2]['profile']['proposal_decode_seconds']*1000,
                    replay_per_arm=1, checkpoint_and_resume=True, continued_updates_identical=True,
                    plotted_figures=7, wall_seconds=perf_counter()-started, source=source())
    write(ROOT / 'smoke.json', evidence)
    print('EXP047 smoke passed', ROOT / 'smoke.json')

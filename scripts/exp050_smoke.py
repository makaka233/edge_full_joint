"""Run a minimal EXP050 target, isolation, and real-physics continuation test."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import append
from scripts.exp027_run import fingerprint
from scripts.exp044_run import same_flow, write
from scripts.exp047_run import GateRun, read, read_model, rows_at, run_files
from scripts.exp047_smoke import equivalent, rejects
from scripts.exp049_run import ROOT as EXP049, SEEDS
from scripts.exp050_audit import FOCALS, ROOT
from scripts.exp050_train import ThreeStepGate, registration, replay_links, three_step_targets, train


def math_smoke(gamma):
    # The first action can be off-policy. A later off-policy action or the
    # last observed transition ends the prefix before any further reward.
    replay = [dict(gs=gs, x=[float(gs)], next=[gs+1.], reward=float(i), action=a, owner='test')
              for i,(gs,a) in enumerate(((2,1),(3,0),(4,0),(5,1),(8,0)),1)]
    links = replay_links(replay)
    assert set(links) == {2,3,4}
    broken = deepcopy(replay)
    broken[1]['x'] = [-1.]
    rejects(lambda: replay_links(broken))

    def online(x):
        return x.new_tensor([1.,0.]).expand(len(x),2)

    def target(x):
        return x.new_tensor([2.,-9.]).expand(len(x),2)

    actual,details = three_step_targets(online,target,replay,links,gamma)
    expect = torch.tensor([1.+gamma*2.+gamma**2*3.+gamma**3*2., 2.+gamma*3.+gamma**2*2.,
                           3.+gamma*2., 4.+gamma*2., 5.+gamma*2.])
    torch.testing.assert_close(actual,expect)
    assert details['target_horizons'] == [3,1,1]


def ledger_smoke(gamma):
    audit = read(ROOT/'audit.json')
    trained_horizons = None
    for name,_,seed,_,_ in FOCALS:
        r = read(ROOT/f'fork.{name}.json')
        ckpt = read_model(run_files(EXP049,'A',seed)/'resume_latest.pt')['gate']
        run = GateRun('A',seed,ckpt['identity']['stamp'],feature_arm='A',rng_arm='A')
        run.restore(ckpt)
        values = []
        for cell in r['cells'].values():
            rows = cell['rows']
            prev = cell['previous_cohort']['e2e_penalized']
            replay = []
            for row,nxt in zip(rows,rows[1:],strict=False):
                replay.append(dict(gs=row['period'],owner=run.identity,x=row['timing']['state'][:17],
                                   next=nxt['timing']['state'][:17],action=row['timing']['action'],
                                   reward=-row['holding_cost']-row['switch_cost']-10.*(prev or 0.)/gamma))
                prev = row['e2e_penalized']
            y,meta = three_step_targets(run.online,run.target,[replay[0]],replay_links(replay),gamma)
            assert meta['target_horizons'] == [0,0,1]
            values.append(float(y[0]))
        expected = audit['values'][name]['A']['288']['y_delta']['3']
        assert np.isclose(values[1]-values[0],expected,rtol=0.,atol=3e-5)
        if name == 'shift-onset':
            # Exercise a real extended gradient on the committed physical
            # sequence, even if the tiny fresh rollout happens to always cut.
            control = GateRun('A',seed,'target smoke',feature_arm='A',rng_arm='A')
            candidate = ThreeStepGate('B',seed,'target smoke',feature_arm='A',rng_arm='A')
            for arm in (control,candidate):
                arm.online.load_state_dict(ckpt['online'])
                arm.target.load_state_dict(ckpt['target'])
                arm.collect([dict(row,owner=arm.identity) for row in replay])
            baseline = control.improve()
            updates = candidate.improve()
            trained_horizons = [sum(r['target_horizons'][i] for r in updates) for i in range(3)]
            assert trained_horizons[2] > 0
            assert not equivalent(ckpt['online'],candidate.online.state_dict())
            # Targets can differ while saturated Huber gradients have equal sign.
            assert any(a['loss'] != b['loss'] for a,b in zip(baseline,updates,strict=True))
    return trained_horizons


def smoke():
    started = perf_counter()
    torch.set_num_threads(1)
    plan,base = registration()
    gamma = plan['q']['gamma']
    math_smoke(gamma)
    archived_horizons = ledger_smoke(gamma)
    seed = SEEDS[0]
    mini = deepcopy(plan)
    mini.update(experiment='EXP050 smoke',slots=4,blocks=2,fixed_at=[0])
    mini['validation'][str(seed)]['loads'] = [.32]*4
    stamp = fingerprint(mini)
    with TemporaryDirectory(dir=ROOT,prefix='smoke-') as folder:
        root = Path(folder)
        train(root,mini,base,'A',seed,4)
        train(root,mini,base,'B',seed,4)
        pa,pb = (run_files(root,arm,seed) for arm in ('A','B'))
        a, b = (read_model(p/'resume_latest.pt')['gate'] for p in (pa,pb))
        assert a['updates'] == b['updates'] == 48
        assert a['online']['center'].equal(b['online']['center'])
        fresh = GateRun('A',seed,stamp,feature_arm='A',rng_arm='A')
        for fitted in (a,b):
            assert not equivalent(fresh.online.body.state_dict(),
                                  {k.removeprefix('body.'):v for k,v in fitted['online'].items() if k.startswith('body.')})
            assert fitted['optimizer']['state']
        ra, rb = (rows_at(p/'train.jsonl') for p in (pa,pb))
        same_flow(ra,rb)
        assert ra[0]['active_matrix'] == rb[0]['active_matrix']
        assert all(r['active_matrix']==r['old_matrix'] and not r['switch_cost']
                   for r in (*ra,*rb) if r['action']=='KEEP')
        candidate = ThreeStepGate('B',seed,stamp,feature_arm='A',rng_arm='A')
        other = ThreeStepGate('B',SEEDS[1],stamp,feature_arm='A',rng_arm='A')
        rejects(lambda: candidate.restore(a))
        rejects(lambda: candidate.collect(a['replay']))
        rejects(lambda: other.restore(b))
        candidate.restore(b)
        assert equivalent(candidate.snapshot(),b)
        append(pb/'train.jsonl',[{'uncommitted':True}])
        train(root,mini,base,'B',seed,8)
        after = read_model(pb/'resume_latest.pt')
        assert after['resumes']==1 and after['last']['gs']==8 and after['gate']['updates']==96
        assert len(rows_at(pb/'train.jsonl'))==8
        assert 'already committed GS8' in train(root,mini,base,'B',seed,8)
        updates = rows_at(pb/'updates.jsonl')
        horizons = [sum(r['target_horizons'][i] for r in updates) for i in range(3)]
        assert sum(horizons) == 96*16
        assert all(np.isfinite(r['gradient']) and np.isfinite(r['td_mae']) for r in updates)
    write(ROOT/'smoke.json',dict(source=source(),wall_seconds=perf_counter()-started,physical_slots=20,
                                probe_target_checks=len(FOCALS),paired_initial_updates=48,resumed_updates=96,
                                target_horizons=horizons,archived_gradient_horizons=archived_horizons,
                                recovery=True,isolated_replay=True))
    print(ROOT/'smoke.json')

"""Isolated EXP019 continuity, probability credit and recoverability checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from edge_d._engine.edge_rl.model import Arrival
from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint.adaptive_training import Config, components, credit_targets
from edge_joint.adaptive_physics import PeriodSimulator, capacity_ok, retirement_plan
from edge_joint.adaptive_runner import train
from edge_joint.adaptive_storage import RunFiles


class ContinuousDeploymentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_capacity_planner_retires_only_unpinned_and_keeps_coverage(self):
        old = np.array([[1, 1], [0, 1]], dtype=bool)
        target = np.array([[0, 1], [1, 0]], dtype=bool)
        capacity = np.array([[1., 1.], [2., 2.]])
        spec = np.ones((2, 2))
        pins = np.zeros_like(old)
        live = retirement_plan(old, target, pins, spec, capacity)
        np.testing.assert_array_equal(live, [[0, 1], [0, 1]])
        self.assertTrue(capacity_ok(live | target, spec, capacity))
        pins[0, 0] = True
        self.assertIsNone(retirement_plan(old, target, pins, spec, capacity))

    def test_image_transfer_keeps_dispatched_stage_and_releases_old_footprint(self):
        env = make_world(20261113, 0, 'fixed')
        old = decode(env, np.zeros(8), 8, True, include_optional=False).matrix
        service = 0
        stage = env.c.services[service][0]
        source = int(np.flatnonzero(old[stage])[0])
        for node in range(env.N):
            if old[stage, node]:
                continue
            target = old.copy()
            target[stage, source] = False
            target[stage, node] = True
            if capacity_ok(old | target, env.spec, env.capacity):
                break
        else:
            self.fail('No test migration with feasible overlap')
        sim = PeriodSimulator(env, old, lambda obs, mask: source if mask[source] else int(np.flatnonzero(mask)[0]),
                              1, 0, True)
        sim.add_arrivals([Arrival(0., 0, source, service)])
        uploaded = env.w.stages[stage, 1] / sim.wireless[source].capacity
        cpu_duration = env.w.stages[stage, 0] / sim.cpu[source].capacity
        sim.run_until(uploaded + cpu_duration * .01)
        self.assertEqual(sim.pins[stage, source], 1)
        now = sim.now
        change = sim.redeploy(target, 1e-10)
        self.assertEqual(change['status'], 'preparing')
        self.assertGreater(change['migration_cost'], 0)
        self.assertTrue(sim.image_jobs)
        mask_before = sim.deployment.copy()
        self.assertTrue(mask_before[stage, source])
        self.assertFalse(mask_before[stage, node])
        sim.verify()
        sim.run_until(now + cpu_duration * .05)
        self.assertIsNone(sim.target)
        self.assertEqual(sim.requests[0].destination, source)
        self.assertFalse(sim.deployment[stage, source])
        self.assertTrue(sim.occupied()[stage, source])
        sim.run_until(now + 20.)
        self.assertIsNotNone(sim.requests[0].finish)
        self.assertFalse(sim.occupied()[stage, source])
        self.assertGreater(sim.image_submitted, 0.)
        self.assertEqual(len(sim.transitions), sim.decisions)
        sim.verify()

    def test_credit_includes_only_earlier_cohorts_with_open_deadlines(self):
        records = [dict(start=i*10., score_deadline=i*10.+16., reward_net=-i-1.) for i in range(3)]
        torch.testing.assert_close(credit_targets(records, .5), torch.tensor([-2.75, -5.5, -7.]))

    def test_fixed_macro_modes_still_train_d_and_s(self):
        with tempfile.TemporaryDirectory(prefix='exp019-test-') as directory:
            flows = []
            for mode, action, probability, d_samples in (
                    ('keep','KEEP',0.,1), ('redeploy','REDEPLOY',1.,2)):
                with self.subTest(mode=mode):
                    cfg = Config(seed=20261113, periods=2, period_seconds=.4, s_batch=1,
                                 s_updates_per_gs=1, macro_mode=mode)
                    _, _, _, _, initial_gate, _ = components(cfg)
                    out = RunFiles(Path(directory)/mode)
                    train(cfg, out, 1)
                    for name in ('train.jsonl','fixed.jsonl'):
                        records=list(map(json.loads,(out/name).read_text().splitlines()))
                        selectable=[r for r in records if r['period']==2]
                        self.assertTrue(selectable)
                        for r in selectable:
                            self.assertEqual((r['mode'],r['action'],r['redeploy_probability']),
                                             (mode,action,probability))
                            self.assertIsNone(r['gate_logp'])
                        if name=='train.jsonl':
                            flows.append([r['trace_sha256'] for r in records])
                    update=json.loads((out/'updates.jsonl').read_text())
                    self.assertEqual((update['macro_updates'],update['macro_policy_samples'],
                                      update['d_samples']), (0,0,d_samples))
                    self.assertGreater(update['d_change'],0.)
                    self.assertGreater(update['s_change'],0.)
                    self.assertGreater(update['value_change'],0.)
                    self.assertEqual(update['macro_change'],0.)
                    state=torch.load(out/'resume_latest.pt',weights_only=True)
                    self.assertEqual(state['config']['macro_mode'],mode)
                    for key,value in initial_gate.state_dict().items():
                        if not key.startswith('value.'):
                            torch.testing.assert_close(state['gate'][key],value,rtol=0.,atol=0.)
            self.assertEqual(flows[0],flows[1])

    def test_tiny_training_is_exact_across_resume_and_saves_models(self):
        cfg = Config(seed=20261113, periods=2, period_seconds=.4, eval_every=1, s_batch=1, s_updates_per_gs=1)
        with tempfile.TemporaryDirectory(prefix='exp019-test-') as directory:
            base = Path(directory)
            a, b = RunFiles(base/'continuous'), RunFiles(base/'resumed')
            train(cfg, a, 2)
            train(cfg, b, 1)
            older = torch.load(b/'resume_latest.pt',weights_only=True)
            del older['config']['macro_mode']
            torch.save(older,b/'resume_latest.pt')
            with (b/'train.jsonl').open('ab') as file:
                file.write(b'{interrupted}\n')
            train(cfg, b, 2, resume=True)
            self.assertTrue(list(base.glob('resumed.train.jsonl.interrupted-*')))
            for projection in ('best.pt', 'last.pt', 'resume_latest.pt'):
                self.assertTrue((b/projection).is_file())
            left, right = (torch.load(p/'resume_latest.pt', weights_only=True) for p in (a, b))
            self.assertEqual(right['episode'], 2)
            self.assertGreater(right['s']['updates'], 0)
            self.assertEqual(right['d_updates'], 2)
            self.assertEqual(left['offsets'].keys(), right['offsets'].keys())
            for key in ('d', 'gate', 's_online'):
                for name in left[key]:
                    torch.testing.assert_close(left[key][name], right[key][name], rtol=0., atol=0.)
            records = [list(map(json.loads, (path/'train.jsonl').read_text().splitlines())) for path in (a, b)]
            for aa, bb in zip(*records, strict=True):
                for field in ('gs', 'epsilon', 'trace_sha256', 'active_matrix', 'reward_raw', 'reward_net',
                              'e2e_penalized', 'backlog_in', 'backlog_out'):
                    self.assertEqual(aa[field], bb[field])
            updates = list(map(json.loads, (b/'updates.jsonl').read_text().splitlines()))
            self.assertTrue(all(row['d_change']>0 and row['macro_change']>0 and row['s_change']>0 for row in updates))


if __name__ == '__main__':
    unittest.main()

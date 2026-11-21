"""Isolated EXP019 continuity, probability credit and recoverability checks."""
import json
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from edge_d._engine.edge_rl.model import Arrival
from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint.adaptive_training import Config, components, credit_targets, episode
from edge_joint.adaptive_physics import PeriodSimulator, capacity_ok, solo_delay_bounds
from edge_joint.adaptive_runner import evaluate, train
from edge_joint.adaptive_storage import RunFiles
from edge_s.agent import QUANTILES, QuantileSchedulingAgent, SchedulingAgent, quantile_huber, quantile_score, quantile_target


class ContinuousDeploymentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_solo_bound_matches_brute_force_and_ideal_physics(self):
        env = make_world(20261113,0,'fixed')
        masks = np.zeros((env.K,env.N),dtype=bool)
        masks[:,:3] = True
        world = env.w
        service, access = 0, 4
        chain = env.c.services[service]

        def time_for(destinations):
            previous = (access,*destinations[:-1])
            elapsed = world.stages[chain[0],1]/world.nodes[access,3]
            for stage,src,dst in zip(chain,previous,destinations,strict=True):
                elapsed += world.stages[stage,0]/world.nodes[dst,0]
                elapsed += world.stages[stage,1]*world.path_seconds_per_mbit[src,dst]
            return elapsed

        enumeration = min(time_for(path) for path in product(range(3),repeat=len(chain)))
        self.assertAlmostEqual(solo_delay_bounds(world,masks)[access,service],enumeration)
        placement = decode(env,np.zeros(8),8,True,include_optional=False).matrix
        destinations = [int(np.flatnonzero(placement[k])[0]) for k in chain]
        scheduled = iter(destinations)
        sim = PeriodSimulator(env,placement,lambda obs,mask:next(scheduled),1,0,False)
        sim.add_arrivals([Arrival(0.,0,access,service)])
        sim.run_until(20.)
        self.assertAlmostEqual(sim.requests[0].finish,time_for(destinations),places=7)
        self.assertLessEqual(solo_delay_bounds(world,placement)[access,service],sim.requests[0].finish+1e-8)

    def test_quantile_target_uses_online_legal_risk_choice_and_terminal_reward(self):
        online = torch.zeros((2,3,QUANTILES))
        online[:,0] = 100.
        online[:,1,:8] = -4.
        online[:,1,8:] = 4./3.
        self.assertLess(quantile_score(online)[0,1],quantile_score(online)[0,2])
        target = torch.stack((torch.ones(QUANTILES)*90.,torch.ones(QUANTILES)*9.,
                              torch.linspace(-2,2,QUANTILES))).repeat(2,1,1)
        legal = torch.tensor([[False,True,True],[False,False,False]])
        terminal = torch.tensor([False,True])
        r = torch.tensor([-.75,-.375])
        actual = quantile_target(online,target,legal,r,terminal,.99)
        torch.testing.assert_close(actual[0],r[0]+.99*target[0,2])
        torch.testing.assert_close(actual[1],r[1].expand(QUANTILES))
        with self.assertRaisesRegex(ValueError,'nonterminal'):
            quantile_target(online,target,legal,r,~terminal,.99)

    def test_quantile_huber_weights_and_scale(self):
        q = torch.zeros((1,2),requires_grad=True)
        loss = quantile_huber(q,torch.full((1,2),2.))
        torch.testing.assert_close(loss,torch.tensor(.75))
        loss.backward()
        torch.testing.assert_close(q.grad,torch.tensor([[-.125,-.375]]))

    def test_quantile_initial_values_and_exploration_match_original(self):
        original = SchedulingAgent(20261114)
        candidate = QuantileSchedulingAgent(20261114)
        mask = np.arange(original.c.n)%4==0
        obs = np.random.default_rng(9).normal(size=(original.c.n,original.features)).astype(np.float32)
        obs[:,7] = mask
        features = torch.from_numpy(obs[None])
        torch.testing.assert_close(original.team_q(features),candidate.team_q(features),rtol=0.,atol=0.)
        for eps in (0.,.25,1.):
            old = original.scheduler(np.random.default_rng(10),eps)
            new = candidate.scheduler(np.random.default_rng(10),eps)
            choices = [new(obs,mask) for _ in range(256)]
            self.assertEqual(choices,[old(obs,mask) for _ in range(256)])
            self.assertTrue(all(mask[a] for a in choices))
            if eps==1.:
                self.assertEqual(set(choices),set(np.flatnonzero(mask)))

    def test_quantile_joint_resume_and_sustained_evaluation(self):
        cfg = Config(seed=20261113,periods=2,period_seconds=.4,s_batch=1,s_updates_per_gs=1,
                     macro_mode='redeploy',s_algorithm='quantile')
        with tempfile.TemporaryDirectory(prefix='exp019-quantile-test-') as directory:
            run = RunFiles(Path(directory)/'quantile')
            train(cfg,run,1)
            train(cfg,run,2,resume=True)
            state = torch.load(run/'resume_latest.pt',weights_only=True)
            updates = list(map(json.loads,(run/'updates.jsonl').read_text().splitlines()))
            self.assertEqual((state['episode'],state['s']['updates'],state['d_updates']), (2,4,2))
            self.assertTrue(all(r['s_change']>0 and r['d_change']>0 and np.isfinite(r['s_loss']) for r in updates))
            self.assertTrue(torch.count_nonzero(state['s_online']['value_offset.weight']))
            evaluate(run,cfg.seed,sustained=True)
            heldout = list(map(json.loads,(run/'holdout-sustained.jsonl').read_text().splitlines()))
            self.assertEqual(len(heldout),32)
            self.assertEqual({r['offered_cpu_target'] for r in heldout},{.55,.70})
            self.assertTrue(all(np.isclose(r['offered_cpu_load'],r['offered_cpu_target']) for r in heldout))
        d,s,_,_,gate,_ = components(cfg)
        with torch.no_grad():
            rows,_,replay = episode(cfg,make_world(cfg.seed,19301,'holdout'),d,s,gate,19301,
                                    'holdout',0,offered_load=.55)
        self.assertFalse(replay)
        for row in rows:
            self.assertAlmostEqual(row['offered_cpu_load'],.55)
            self.assertEqual(row['epsilon'],0.)

    def test_legal_matrices_switch_both_directions_despite_infeasible_union(self):
        env = make_world(20261113, 0, 'fixed')
        rng = np.random.default_rng(42)
        old = decode(env, np.zeros(8), 8, True).matrix
        for i in range(12):
            target = decode(env, rng.normal(size=8), i+1, True).matrix
            if not capacity_ok(old | target, env.spec, env.capacity):
                break
        else:
            self.fail('Fixture needs two individually feasible matrices with infeasible overlap')
        for matrix in (old, target):
            self.assertEqual(matrix.shape, (28, 32))
            self.assertTrue(matrix.any(1).all())
            self.assertTrue(capacity_ok(matrix, env.spec, env.capacity))
        sim = PeriodSimulator(env, old, None, 1, 0, False)
        for before, after in ((old,target), (target,old)):
            change = sim.redeploy(after)
            self.assertEqual((change['status'], change['activated_at']), ('activated', sim.now))
            np.testing.assert_array_equal(sim.deployment, after)
            added = (after & ~before).T @ env.spec / env.capacity
            removed = (before & ~after).T @ env.spec / env.capacity
            cost_scale = env.cost['resource_normalizer'] / env.N
            expected = env.cost['switch_weight'] * cost_scale * (
                added.sum() + env.cost['removal_ratio'] * removed.sum())
            self.assertAlmostEqual(change['switch_cost'], expected)
            self.assertAlmostEqual(sim.holding_cost(), env.cost['holding_weight'] * cost_scale *
                                   ((after.T @ env.spec) / env.capacity).sum())
            self.assertEqual(sim.redeploy(after), dict(status='unchanged', switch_cost=0., activated_at=None))
            sim.verify()
        for invalid in (np.zeros_like(old), np.ones_like(old), old.astype(int)*2, old[:-1]):
            with self.assertRaises(ValueError):
                sim.redeploy(invalid)
            np.testing.assert_array_equal(sim.deployment, old)

    def test_dispatched_cpu_and_wired_jobs_survive_switch_and_future_masks_change(self):
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
            if capacity_ok(target, env.spec, env.capacity):
                break
        else:
            self.fail('No feasible test target')
        sim = PeriodSimulator(env, old, lambda obs, mask: source if mask[source] else int(np.flatnonzero(mask)[0]),
                              1, 0, True)
        sim.trace_enabled = True
        payload = env.w.stages[stage, 1]
        uploaded = payload / sim.wireless[source].capacity
        cpu_duration = env.w.stages[stage, 0] / sim.cpu[source].capacity
        boundary = 1.+uploaded+cpu_duration*.25
        access = next(n for n in range(env.N) if n!=source)
        link = env.w.routes[access][source][0]
        wire_dispatch = boundary - min(.05, payload / sim.wired[link].capacity*.25)
        wire_arrival = wire_dispatch - payload / sim.wireless[access].capacity
        sim.add_arrivals([Arrival(1., 0, source, service), Arrival(wire_arrival, 1, access, service),
                          Arrival(boundary+.1, 2, source, service)])
        sim.run_until(boundary)
        self.assertEqual((len(sim.cpu[source].jobs),len(sim.wired[link].jobs)), (1,1))
        before = [tuple(r.jobs) for r in sim.resources]
        self.assertEqual(sim.redeploy(target)['activated_at'], boundary)
        self.assertEqual(before,[tuple(r.jobs) for r in sim.resources])
        self.assertEqual([r.destination for r in sim.requests],[source,source])
        self.assertFalse(sim.deployment[stage,source])
        sim.verify()
        sim.run_until(boundary+60.)
        for request in sim.requests:
            first = source if request.arrival.user in (0,1) else node
            chain = env.c.services[service]
            self.assertIsNotNone(request.finish)
            for index, stage_id in enumerate(chain):
                finishes = [event for event in request.trace if event[0]==index and event[1]=='finish']
                self.assertEqual(len(finishes),1)
                self.assertEqual(finishes[0][3],first if index==0 else int(np.flatnonzero(target[stage_id])[0]))
        self.assertEqual(len(sim.transitions), sim.decisions)
        self.assertEqual(len({(r.request_id,r.stage_index) for r in sim.transitions}),sim.decisions)
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
                            if action == 'REDEPLOY':
                                self.assertEqual(r['active_matrix'],r['proposed_matrix'])
                                if r['active_matrix'] != r['old_matrix']:
                                    self.assertEqual(r['activated_at'],r['start'])
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

    def test_historical_checkpoint_cannot_be_resumed_or_republished(self):
        cfg = Config(seed=20261113)
        with tempfile.TemporaryDirectory(prefix='exp019-test-') as directory:
            out = RunFiles(Path(directory)/'historical')
            torch.save(dict(format=1),out/'resume_latest.pt')
            before = (out/'resume_latest.pt').read_bytes()
            with self.assertRaisesRegex(ValueError,'historical deployment model'):
                train(cfg,out,1,resume=True)
            with self.assertRaisesRegex(ValueError,'historical deployment model'):
                evaluate(out,cfg.seed)
            self.assertEqual((out/'resume_latest.pt').read_bytes(),before)
            self.assertFalse((out/'last.pt').exists())


if __name__ == '__main__':
    unittest.main()

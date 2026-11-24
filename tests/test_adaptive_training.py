"""Isolated EXP019 continuity, probability credit and recoverability checks."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from edge_d._engine.edge_rl.model import Arrival
from edge_d._engine.edge_rl.physics import STransition
from edge_d._engine.edge_rl.replay import SReplay
from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint import adaptive_cli
from edge_joint.adaptive_training import Config, components, credit_targets
from edge_joint.adaptive_physics import PeriodSimulator, capacity_ok, solo_delay_bounds
from edge_joint.adaptive_runner import evaluate, train
from edge_joint.adaptive_storage import RunFiles
from edge_s.agent import SchedulingAgent, resource_advantage
from edge_d._engine.edge_rl.replay import laber_selection


class ContinuousDeploymentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_laber_weights_preserve_conditional_uniform_objective(self):
        class DrawAll:
            def choice(self,n,*,size,replace,p):
                self.probability=p
                self.assertion=(n,size,replace)
                return np.arange(n)
        rng=DrawAll()
        q=np.array([0.,.2,1.,.05])
        picked,weights=laber_selection(q,4,rng)
        np.testing.assert_array_equal(picked,np.arange(4))
        self.assertEqual(rng.assertion,(4,4,True))
        np.testing.assert_allclose(rng.probability*weights,np.full(4,.25),rtol=0.,atol=1e-15)
        self.assertLessEqual(weights.max(),10.)
        _,weights=laber_selection(np.zeros(4),4,rng)
        np.testing.assert_array_equal(weights,np.ones(4))

    def test_watkins_target_masks_cuts_and_terminates(self):
        s=SchedulingAgent(20261113)
        gamma=s.c.gamma_s
        middle=np.array([[1,8],[4,5],[80,90]],dtype=np.float32)
        end=np.array([[2,11],[99,88],[1,7]],dtype=np.float32)
        m1=np.array([True,True,False]); m2=np.array([True,False,True])
        def transition(action,reward,nx,mask,done=False,stage=0,step=1,window=0):
            return STransition(middle,m1,action,reward,nx,mask,done,False,0,step,0,stage,window)
        first=transition(0,1.,middle,m1)
        continued=transition(1,2.,end,m2,stage=1,step=2,window=1)
        exploratory=replace(continued,action=0)
        terminal=replace(continued,terminated=True)
        index=SReplay(4,2,index_successors=True)
        index.add([first])
        self.assertEqual(index.successors([first]),[None])
        index.add([continued,replace(continued,collection_step=9,window_index=0)])
        self.assertIs(index.successors([first])[0],continued)
        index.prune(5)
        self.assertEqual(index.successors([first]),[None])

        parameter=next(s.online.parameters())[0,0]
        def q(x,target=False):
            return x[:,:,1 if target else 0]+(parameter-parameter.detach())
        with patch.object(s,'team_q',side_effect=q), patch(
                'edge_d._engine.edge_rl.agents.F.smooth_l1_loss',wraps=torch.nn.functional.smooth_l1_loss) as loss:
            result=s.update([first]*4,successors=[continued,exploratory,terminal,None])
        expected=[1+gamma*(2+gamma*11),1+gamma*5,1+gamma*2,1+gamma*5]
        torch.testing.assert_close(loss.call_args.args[1],torch.tensor(expected))
        self.assertEqual(result['s_successor_share'],.75)
        self.assertEqual(result['s_two_step_share'],.5)

    def test_watkins_resume_reconstructs_successors(self):
        from scripts.exp042_run import WatkinsConfig

        cfg=WatkinsConfig(seed=20261115,macro_mode='redeploy',periods=2,period_seconds=2.,
                          eval_every=1,s_batch=4,s_update_timing='period',s_resource_q_prior=True,
                          s_two_step_watkins=True)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            a,b=(RunFiles(Path(directory)/name) for name in ('continuous','resumed'))
            train(cfg,a,2)
            train(cfg,b,1)
            train(cfg,b,2,resume=True)
            check=('reward_raw','active_matrix','s_boundary','trace_sha256')
            rows=[list(map(json.loads,(out/'train.jsonl').read_text(encoding='utf-8').splitlines())) for out in (a,b)]
            self.assertEqual(*[[[r[k] for k in check] for r in source] for source in rows])
            self.assertTrue(any(r['s_boundary']['s_two_step_share']>0 for r in rows[0]))
            left,right=[torch.load(out/'resume_latest.pt',weights_only=True,map_location='cpu') for out in (a,b)]
            for name in ('d','s','d_optimizer','replay'):
                torch.testing.assert_close(left[name],right[name],atol=0.,rtol=0.)
            l_rng,r_rng=(state['rng'] for state in (left,right))
            torch.testing.assert_close(l_rng['torch'],r_rng['torch'],atol=0.,rtol=0.)
            torch.testing.assert_close(l_rng['numpy'][1],r_rng['numpy'][1],atol=0.,rtol=0.)
            self.assertEqual(l_rng['numpy'][0],r_rng['numpy'][0])
            self.assertEqual(l_rng['numpy'][2:],r_rng['numpy'][2:])
            self.assertEqual(l_rng['python'],r_rng['python'])

    def test_laber_resume_preserves_actual_training(self):
        from scripts.exp040_run import ReplayConfig

        cfg=ReplayConfig(seed=20261115,macro_mode='redeploy',periods=2,period_seconds=2.,
                         eval_every=1,s_batch=4,s_update_timing='period',s_resource_q_prior=True,
                         s_laber_pool_factor=4)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            a,b=(RunFiles(Path(directory)/name) for name in ('continuous','resumed'))
            train(cfg,a,2)
            train(cfg,b,1)
            train(cfg,b,2,resume=True)
            check=['reward_raw','active_matrix','s_boundary','trace_sha256']
            rows=[(out/'train.jsonl').read_text(encoding='utf-8').splitlines() for out in (a,b)]
            left,right=[[{k:r[k] for k in check} for r in map(json.loads,source)] for source in rows]
            self.assertEqual(left,right)
            self.assertTrue(all(r['s_boundary']['optimizer_steps']==8 and
                                0<r['s_boundary']['s_is_ess']<=4.00001 for r in left))
            x,y=[torch.load(out/'resume_latest.pt',weights_only=True,map_location='cpu') for out in (a,b)]
            for name in ('d','s','d_optimizer','replay'):
                torch.testing.assert_close(x[name],y[name],atol=0.,rtol=0.)
            torch.testing.assert_close(x['rng']['torch'],y['rng']['torch'],atol=0.,rtol=0.)
            torch.testing.assert_close(x['rng']['numpy'][1],y['rng']['numpy'][1],atol=0.,rtol=0.)
            self.assertEqual(x['rng']['numpy'][0],y['rng']['numpy'][0])
            self.assertEqual(x['rng']['numpy'][2:],y['rng']['numpy'][2:])
            self.assertEqual(x['rng']['python'],y['rng']['python'])

    def test_centered_resource_advantage_uses_current_legal_resource_time(self):
        obs=torch.zeros((2,4,24))
        obs[0,:,0]=.5
        obs[0,:,8]=1.
        obs[0,:3,7]=1
        obs[0,1,17]=torch.log1p(torch.tensor(1.))
        obs[0,2,18]=torch.log1p(torch.tensor(.5))
        advantage=resource_advantage(obs,1.)
        self.assertTrue(torch.isfinite(advantage).all())
        self.assertGreater(advantage[0,0],advantage[0,2])
        self.assertGreater(advantage[0,2],advantage[0,1])
        torch.testing.assert_close(advantage[0].sum(),torch.tensor(0.))
        self.assertEqual(advantage[0,3].item(),0.)
        self.assertFalse(advantage[1].any())
        obs[0,1:3,7]=0
        self.assertFalse(resource_advantage(obs,1.).any())

    def test_resource_prior_preserves_random_weights_and_enters_both_q_functions(self):
        baseline=SchedulingAgent(20261116)
        candidate=SchedulingAgent(20261116,resource_q_prior=True)
        for left,right in zip(baseline.online.parameters(),candidate.online.parameters(),strict=True):
            torch.testing.assert_close(left,right,rtol=0.,atol=0.)
        obs=torch.zeros((1,32,24))
        obs[...,0]=.5
        obs[...,8]=1.
        obs[0,1:3,7]=1.
        obs[0,2,16]=2.
        expected=resource_advantage(obs,candidate.config.latency_ref_seconds)
        for target in (False,True):
            torch.testing.assert_close(candidate.team_q(obs,target)-baseline.team_q(obs,target),expected)
        mask=obs[0,:,7].bool().numpy()
        select=candidate.scheduler(np.random.default_rng(17),epsilon=1.)
        self.assertEqual({select(obs[0].numpy(),mask) for _ in range(100)},{1,2})

    def test_cross_period_transition_retains_dispatch_version_and_bootstrap(self):
        env = make_world(20261114, 0)
        env.w.t = 0.
        matrix = decode(env, np.zeros(8), 8, True).matrix
        sim = PeriodSimulator(env, matrix, lambda obs, mask: int(np.flatnonzero(mask)[0]),
                              1, 0, True, track_dispatch=True)
        stage = env.c.services[0][0]
        upload = env.w.stages[stage, 1]/env.w.nodes[0, 3]
        sim.add_arrivals([Arrival(0., 0, 0, 0)])
        sim.run_until(upload + 1e-7)
        self.assertEqual(len(sim.transitions), 0)
        sim.version, sim.collection_step, sim.window_index = 1, 2, 1
        sim.run_until(20.)
        sim.verify()
        first, *rest = sim.transitions
        self.assertEqual((first.version, first.collection_step, first.window_index), (0, 1, 0))
        self.assertFalse(first.terminated)
        self.assertTrue(first.next_mask.any())
        self.assertTrue(rest[-1].terminated)
        self.assertTrue(all((r.version, r.collection_step, r.window_index) == (1, 2, 1) for r in rest))

    def test_period_ddqn_budget_flow_and_resume(self):
        base = Config(seed=20261115, macro_mode='redeploy', periods=3, period_seconds=1.5,
                      s_batch=1, s_updates_per_gs=2, eval_every=1)
        candidate = replace(base, s_update_timing='period')

        def assert_tree(left, right):
            if isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_tree(left[key], right[key])
            elif isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0., atol=0.)
            else:
                self.assertEqual(left, right)

        def ledger(out, key):
            return [json.loads(line) for line in (out/f'{key}.jsonl').read_text(encoding='utf8').splitlines()]

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            control, continuous, resumed = [RunFiles(Path(directory)/name) for name in ('base','continuous','resume')]
            train(base, control, 2)
            train(candidate, continuous, 2)
            train(candidate, resumed, 1)
            train(candidate, resumed, 2, resume=True)
            rollout, ref, continued = (ledger(out, 'train') for out in (control, continuous, resumed))
            self.assertEqual([r['trace_sha256'] for r in rollout], [r['trace_sha256'] for r in ref])
            for first, second in zip(ref, continued, strict=True):
                keep = ('active_matrix', 'trace_sha256', 'reward_raw', 'completed_delays', 's_boundary')
                self.assertEqual([first[k] for k in keep], [second[k] for k in keep])
                self.assertLess(first['conservation_error'], 1e-6)
            self.assertLess(ref[0]['s_policy_version'], ref[1]['s_policy_version'])
            for i, row in enumerate(ledger(continuous, 'updates')):
                periods = ref[3*i:3*i+3]
                self.assertEqual((row['d_updates'], row['s_updates_this_episode']), (i+1, 6))
                self.assertEqual(sum(p['s_boundary']['optimizer_steps'] for p in periods), 6)
                self.assertEqual(sum(p['s_boundary']['completed_transitions'] for p in periods), row['admitted_to_replay'])
                self.assertEqual(sum(p['s_decisions']+p.get('drain_decisions',0) for p in periods),
                                 row['admitted_to_replay']+row['s_unfinished_expiry_excluded'])
            self.assertEqual(ledger(control, 'updates')[-1]['s_updates'], ledger(continuous, 'updates')[-1]['s_updates'])
            left, right = [torch.load(out/'resume_latest.pt', weights_only=True) for out in (continuous, resumed)]
            for key in ('d', 'd_optimizer', 'gate', 'gate_optimizer', 's', 'replay'):
                assert_tree(left[key], right[key])

    def test_batch_train_resume_and_evaluate(self):
        parser = argparse.ArgumentParser()
        adaptive_cli.add_arguments(parser)
        seeds = [str(seed) for seed in range(20261113, 20261118)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root/'runs/results/EXP023/instant.redeploy.batch'
            training = '--macro-mode redeploy --periods 2 --period-seconds 1 --eval-every 1 --s-batch 1'.split()

            def run(action, selector, out, episodes, *options):
                args = parser.parse_args([action, *selector, '--experiment', 'EXP023', '--prefix', str(out),
                                          '--episodes', str(episodes), *options])
                return adaptive_cli.run(parser, args)

            with patch.object(adaptive_cli, '__file__', str(root/'edge_joint/adaptive_cli.py')), redirect_stdout(StringIO()) as stdout:
                first = RunFiles(prefix.with_name(f'{prefix.name}.seed{seeds[0]}'))
                self.assertEqual(run('train', ['--seed', seeds[0]], first.prefix, 1, *training), 0)
                previous = (first/'manifest.json').read_bytes()
                selector = ['--seeds', *seeds]
                self.assertEqual(run('train', selector, prefix, 1, *training, '--resume', '--evaluate-after'), 0)
                self.assertEqual((first/'manifest.json').read_bytes(), previous)
                messages = stdout.getvalue().splitlines()
                last = prefix.with_name(f'{prefix.name}.seed{seeds[-1]}').resolve()
                first_evaluation = f'evaluate: seed={seeds[0]} prefix={first.prefix.resolve()}'
                self.assertLess(messages.index(str(last)), messages.index(first_evaluation))
                holdout = (first/'holdout.jsonl').read_bytes()
                with redirect_stderr(StringIO()) as stderr:
                    self.assertEqual(run('evaluate', selector, prefix, 2), 1)
                self.assertIn('expected 2 completed episodes', stderr.getvalue())
                self.assertEqual((first/'holdout.jsonl').read_bytes(), holdout)
                self.assertFalse((root/'runs/results/EXP019').exists())
                for seed in seeds:
                    out = RunFiles(prefix.with_name(f'{prefix.name}.seed{seed}'))
                    manifest = json.loads((out/'manifest.json').read_text(encoding='utf8'))
                    self.assertEqual(manifest['config']['seed'], int(seed))
                    self.assertEqual(manifest['origin']['kind'], 'random')
                    self.assertEqual(len((out/'train.jsonl').read_text(encoding='utf8').splitlines()), 2)
                    self.assertEqual(len((out/'holdout.jsonl').read_text(encoding='utf8').splitlines()), 24)

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

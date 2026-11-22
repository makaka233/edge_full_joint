"""Causal resource snapshot, capacity control, and EXP025 training/recovery checks."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from edge_d._engine.edge_rl.model import Arrival
from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint.adaptive_physics import PeriodSimulator, capacity_ok
from edge_joint.adaptive_runner import train
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import Config, components
from edge_joint.placement_state import capture_resources, resource_features
from scripts.exp019_report import check_period


class PlacementObservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_observation_is_causal_and_stage_work_reconciles(self):
        env = make_world(20261113, 7)
        env.w.t = 0.
        env.old = decode(env, np.zeros(8), 10, True).matrix
        initial = capture_resources(env, None)
        stage = env.c.services[0][0]
        node = int(np.flatnonzero(env.old[stage])[0])
        sim = PeriodSimulator(env, env.old, lambda obs, mask: int(np.flatnonzero(mask)[0]),
                              0, 0, False, observe_placement=True)
        sim.add_arrivals([Arrival(0., 0, node, 0)])
        sim.run_until(env.w.stages[stage,1]/env.w.nodes[node,3]+1e-6)
        frozen_now, heap_size, events = sim.now, len(sim.heap), sim.events
        snapshot = capture_resources(env, sim)
        self.assertEqual((sim.now,len(sim.heap),sim.events),(frozen_now,heap_size,events))
        self.assertEqual(snapshot.cpu_jobs[stage,node],1)
        self.assertEqual(snapshot.cpu_jobs.sum(),1)
        np.testing.assert_allclose(snapshot.cpu_work.sum(0),[r.outstanding() for r in sim.cpu])
        sim.add_arrivals([Arrival(30., 1, node, 0)])
        self.assertEqual(snapshot.as_log(),capture_resources(env,sim).as_log())
        vector = resource_features(env, snapshot, env.w.node_demand())
        empty = resource_features(env, initial, env.w.node_demand())
        self.assertEqual((len(vector), vector.dtype), (372,torch.float32))
        self.assertTrue(torch.isfinite(vector).all().item())
        self.assertTrue((vector >= 0).all().item())
        self.assertGreater(float((vector-empty).norm()),0.)
        initial.cpu_work[stage,node] = 5.*env.w.nodes[node,0]
        initial.cpu_jobs[stage,node] = 16
        initial.wired_work[0] = 2.*env.w.directed[0][2]
        calibrated = resource_features(env,initial,env.w.node_demand())
        self.assertEqual(calibrated[[4*node,4*node+1,268]].tolist(),[1.,1.,1.])
        self.assertTrue(all(x<1e-6 for x in sim.verify()))

    def test_initial_equivalence_and_masked_capacity_control(self):
        cfg = Config(seed=20261113, macro_mode='redeploy')
        env = make_world(cfg.seed, 7)
        base, orig_s, *_ = components(cfg)
        candidate, cand_s, *_ = components(replace(cfg,d_observation='resource'))
        blind, *_ = components(replace(cfg,d_observation='blind'))
        self.assertEqual(sum(p.numel() for p in candidate.parameters()),sum(p.numel() for p in blind.parameters()))
        for name, p in orig_s.online.state_dict().items():
            torch.testing.assert_close(p,cand_s.online.state_dict()[name],rtol=0,atol=0)
        state = base.features(env)
        live = torch.linspace(0, 1.0, candidate.resource_weight.shape[1])
        a = base.mean_from_features(state)
        b = candidate.mean_from_features(state,live)
        z = blind.mean_from_features(state,torch.zeros_like(live))
        for mean in (b,z):
            torch.testing.assert_close(a,mean,rtol=0,atol=0)
        b.sum().backward()
        z.sum().backward()
        self.assertGreater(float(candidate.resource_weight.grad.norm()),0.)
        self.assertEqual(float(blind.resource_weight.grad.norm()),0.)
        matrix = decode(env,b.detach().numpy(),2,True).matrix
        self.assertEqual(matrix.shape,(28,32))
        self.assertTrue(matrix.any(1).all() and capacity_ok(matrix,env.spec,env.capacity))

    def test_real_training_diagnostics_and_resume(self):
        for timing in ('episode', 'period'):
            with self.subTest(timing=timing):
                self.check_training_diagnostics_and_resume(timing)

    def check_training_diagnostics_and_resume(self, timing):
        cfg = Config(seed=20261113, macro_mode='redeploy', periods=2, period_seconds=2., s_batch=1,
                     d_observation='resource', d_diagnostics=True, eval_every=1, s_update_timing=timing)

        def log(out, kind):
            return [json.loads(line) for line in (out/f'{kind}.jsonl').read_text().splitlines()]

        def same(left,right):
            if isinstance(left,torch.Tensor):
                torch.testing.assert_close(left,right,rtol=0,atol=0)
            elif isinstance(left,dict):
                self.assertEqual(left.keys(),right.keys())
                for key in left:
                    same(left[key],right[key])
            else:
                self.assertEqual(left,right)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            a,b,control = (RunFiles(Path(directory)/name) for name in ('continuous','resumed','control'))
            train(cfg,a,2)
            train(cfg,b,1)
            train(cfg,b,2,resume=True)
            train(replace(cfg,d_observation='blind'),control,2)
            states = [torch.load(out/'resume_latest.pt',weights_only=True) for out in (a,b,control)]
            for key in ('d','d_optimizer','s','replay','gate','gate_optimizer'):
                same(states[0][key],states[1][key])
            self.assertGreater(float(states[0]['d']['resource_weight'].norm()),0.)
            self.assertEqual(float(states[2]['d']['resource_weight'].norm()),0.)
            source = make_world(cfg.seed,0)
            train_rows = [log(out,'train') for out in (a,b,control)]
            if timing == 'period':
                self.assertEqual([r['s_policy_version'] for r in train_rows[0]],[0,1,2,3])
                self.assertTrue(all(r['s_boundary']['optimizer_steps']==8 for rows in train_rows for r in rows))
                self.assertEqual([r['s_boundary'] for r in train_rows[0]], [r['s_boundary'] for r in train_rows[1]])
            for x,y,z in zip(*train_rows,strict=True):
                for field in ('trace_sha256','reward_net','active_matrix','d_credit','d_observation'):
                    self.assertEqual(x[field],y[field])
                self.assertEqual(x['trace_sha256'],z['trace_sha256'])
                check_period(source,x,True)
                self.assertAlmostEqual(x['d_credit']['discounted_return']-x['d_credit']['value'],
                                       x['d_credit']['advantage'],places=5)
            for updates in (log(a,'updates'),log(control,'updates')):
                self.assertEqual((updates[-1]['d_updates'],updates[-1]['s_updates']),(2,32))
                self.assertTrue(all(x['d_audit_decodes']==2 and 'd_anchor_changed_bits' in x for x in updates))
            self.assertTrue(any(r['d_anchor_live_mean_delta']>0. for r in log(a,'updates')))
            self.assertTrue(all(r['d_anchor_live_mean_delta']==0. for r in log(control,'updates')))

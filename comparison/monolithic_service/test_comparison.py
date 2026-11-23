"""Physical and learning regression tests for the independent comparison."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from edge_d._engine.edge_rl.model import Arrival
from edge_d.physical import hasharr
from edge_d.world import make_world
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import components as multistage_components, decode_placement
from edge_joint.adaptive_physics import capacity_ok, footprint, switch_cost
from edge_joint.inventory_joint import load_scale
from scripts.exp029_long import config

from comparison import run
from comparison.run import rollout
from comparison.profile_audit import payload_bytes, registered_profiles
from . import training
from .environment import MeasuredMonolithicSimulator, MeasuredMultistageSimulator, world


SEED=20261113


class ComparisonTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg=replace(config(SEED),periods=2,period_seconds=2.,s_batch=1,eval_every=1)
        cls.multi=world(SEED,1)
        cls.mono=world(SEED,1,mono=True)
        cls.layouts=[decode_placement(cls.cfg,e,np.zeros(8),17).matrix for e in (cls.multi,cls.mono)]
        cls.service=next(i for i,chain in enumerate(cls.multi.c.services) if len(chain)>1)
        cls.physical=[]
        for env,layout,factory in zip((cls.multi,cls.mono),cls.layouts,
                (MeasuredMultistageSimulator,MeasuredMonolithicSimulator),strict=True):
            env.w.t=0.
            # Restrict each of this request's stages to a distinct, legal remote node.
            chain=env.c.services[cls.service]
            fixed=layout.copy()
            free=env.capacity-(fixed.T @ env.spec)
            for k in chain:
                free+=fixed[k,:,None]*env.spec[k]
                fixed[k]=False
            used=set()
            for k in chain:
                dest=next(n for n in range(1,env.N) if n not in used and np.all(free[n]>=env.spec[k]-1e-8))
                used.add(dest)
                fixed[k,dest]=True
                free[dest]-=env.spec[k]
            sim=factory(env,fixed,lambda obs,mask: int(np.flatnonzero(mask)[0]),1,0,True)
            user=int(env.w.users_by_access[0][0])
            sim.add_arrivals([Arrival(0.1,user,0,cls.service)])
            sim.run_until(10.)
            sim.verify()
            cls.physical.append((sim,chain,fixed))

    def test_01_cpu_work_is_conserved_for_every_service(self):
        np.testing.assert_allclose(self.mono.w.stages[:,0],self.multi.service_work,rtol=0,atol=0)
        expected=self.multi.service_work[self.service]
        for sim,_,_ in self.physical:
            self.assertAlmostEqual(sim.arrival_cpu_work,expected)
            self.assertAlmostEqual(sum(r.served for r in sim.cpu),expected)

    def test_02_ram_ssd_are_summed_for_each_full_replica(self):
        for s,group in enumerate(self.multi.c.services):
            np.testing.assert_allclose(self.mono.spec[s],self.multi.spec[group].sum(0),rtol=0,atol=1e-10)

    def test_03_atomic_cpu_job_is_on_one_node(self):
        sim,chain,x=self.physical[1]
        self.assertEqual(len(chain),1)
        nodes=[i for i,r in enumerate(sim.cpu) if r.submitted]
        self.assertEqual(nodes,np.flatnonzero(x[self.service]).tolist())
        self.assertEqual(len(nodes),1)

    def test_04_overloaded_or_uncovered_monolithic_deployment_is_rejected(self):
        sim,_,x=self.physical[1]
        overloaded=np.ones_like(x)
        self.assertFalse(capacity_ok(overloaded,self.mono.spec,self.mono.capacity))
        with self.assertRaises(ValueError):
            sim.redeploy(overloaded)
        x=x.copy()
        x[self.service]=False
        with self.assertRaises(ValueError):
            sim.redeploy(x)

    def test_05_split_stage_matrix_cannot_pass_as_a_full_replica(self):
        _,chain,x=self.physical[0]
        self.assertFalse(x[chain].all(0).any())
        self.assertTrue(capacity_ok(x,self.multi.spec,self.multi.capacity))
        with self.assertRaises(ValueError):
            self.physical[1][0].redeploy(x)

    def test_06_one_legal_dispatch_and_one_terminal_transition(self):
        sim,_,x=self.physical[1]
        self.assertEqual(sim.decisions,1)
        self.assertEqual(len(sim.transitions),1)
        decision=sim.transitions[0]
        self.assertTrue(decision.terminated)
        np.testing.assert_array_equal(decision.mask,x[self.service])
        self.assertTrue(decision.mask[decision.action])

    def test_07_wireless_entry_and_final_output_preserved(self):
        for env,entry in zip((self.multi,self.mono),self.physical,strict=True):
            sim,chain,_=entry
            self.assertAlmostEqual(sum(r.submitted for r in sim.wireless),env.w.stages[chain[0],1])
        for s,group in enumerate(self.multi.c.services):
            self.assertEqual(self.mono.c.stages[s][1],self.multi.c.stages[group[0]][1])
            self.assertEqual(self.mono.c.stages[s][2],self.multi.c.stages[group[-1]][2])

    def test_08_no_internal_transmission_but_entry_hops_remain(self):
        for (sim,chain,x),env in zip(self.physical,(self.multi,self.mono),strict=True):
            dest=[int(np.flatnonzero(x[k])[0]) for k in chain]
            links=[env.w.routes[a][b] for a,b in zip([0,*dest[:-1]],dest,strict=True)]
            volume=[len(route)*env.w.stages[k,1] for route,k in zip(links,chain,strict=True)]
            self.assertGreater(sim.entry_wire,0)
            self.assertAlmostEqual(sim.entry_wire,volume[0])
            self.assertAlmostEqual(sim.internal_wire,sum(volume[1:]))
            self.assertAlmostEqual(sum(r.submitted for r in sim.wired),sum(volume))
        self.assertGreater(self.physical[0][0].internal_wire,0)
        self.assertEqual(self.physical[1][0].internal_wire,0)

    def test_09_intermediate_scales_only_matched_internal_fields(self):
        e=world(SEED,1,scenario=('intermediate',4.))
        for chain in e.c.services:
            for a,b in zip(chain,chain[1:]):
                self.assertEqual(e.w.stages[a,2],4*self.multi.w.stages[a,2])
                self.assertEqual(e.w.stages[a,2],e.w.stages[b,1])
            self.assertEqual(e.w.stages[chain[0],1],self.multi.w.stages[chain[0],1])
            self.assertEqual(e.w.stages[chain[-1],2],self.multi.w.stages[chain[-1],2])
        np.testing.assert_array_equal(e.w.stages[:,[0,3,4]],self.multi.w.stages[:,[0,3,4]])
        same=world(SEED,1,mono=True,scenario=('intermediate',4.))
        np.testing.assert_array_equal(same.w.stages,self.mono.w.stages)

    def test_10_original_allocators_and_all_resource_ledgers(self):
        for sim,_,_ in self.physical:
            self.assertEqual(sim.cpu_allocator,'finish_pf')
            self.assertEqual(sim.cpu_reference_work,min(self.multi.w.stages[:,0]))
            for r in sim.resources:
                self.assertAlmostEqual(r.submitted,r.served+r.outstanding())
                self.assertLessEqual(r.max_capacity_error,1e-8)
        e=world(SEED,1,scenario=('wired',.5))
        self.assertEqual(e.w.routes,self.multi.w.routes)
        np.testing.assert_allclose(e.w.path_seconds_per_mbit,2*self.multi.w.path_seconds_per_mbit)

    def test_11_reward_and_cost_use_resource_weighting(self):
        for arm,initializer in (('multistage',multistage_components),('monolithic',training.components)):
            d,s,_,_,gate,_=initializer(self.cfg)
            case=dict(environment=20261117,scene=35237)
            result=rollout(self.cfg,d,s,gate,arm,('load',.40),case,0)
            for r in result['rows']:
                self.assertAlmostEqual(r['reward_raw'],-10*r['e2e_penalized']-r['holding_cost']-r['switch_cost'])
        a,b=(env.w.stages[:,3:5] for env in (self.multi,self.mono))
        complete=self.layouts[1]
        expanded=complete[self.multi.w.stage_service]
        self.assertAlmostEqual(footprint(self.multi,expanded),footprint(self.mono,complete))
        self.assertAlmostEqual(switch_cost(self.multi,np.zeros_like(expanded),expanded),
                               switch_cost(self.mono,np.zeros_like(complete),complete))
        np.testing.assert_allclose(expanded.T@a,complete.T@b)

    def test_12_request_flows_pair_under_each_physical_intervention(self):
        def sample(env):
            rows=[]
            for i in range(2):
                rows.append(hasharr(env.w.arrivals(24*i,24*(i+1),np.random.default_rng(i))))
                env.w.advance()
            return rows
        natural=sample(world(SEED,5))
        for case in (('load',.55),('cpu',.7),('wired',.5),('intermediate',4.)):
            with self.subTest(case=case):
                a,b=[world(SEED,5,mono=m,scenario=case) for m in (False,True)]
                np.testing.assert_array_equal(a.w.user_access,b.w.user_access)
                np.testing.assert_array_equal(a.w.user_group,b.w.user_group)
                if case[0]=='load':
                    self.assertAlmostEqual(load_scale(a),.55)
                    self.assertAlmostEqual(load_scale(b),.55)
                if case[0]=='cpu':
                    self.assertAlmostEqual(load_scale(a),load_scale(world(SEED,5))/.7)
                trace=sample(a)
                self.assertEqual(trace,sample(b))
                if case[0]!='load':
                    self.assertEqual(trace,natural)

    def test_13_initial_parameters_optimizer_and_replay_are_separate(self):
        a=multistage_components(self.cfg)
        b=training.components(self.cfg)
        for x,y in zip(a[0].parameters(),b[0].parameters(),strict=True):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertNotEqual(x.data_ptr(),y.data_ptr())
        self.assertIsNot(a[2],b[2])
        self.assertIsNot(a[3],b[3])
        self.assertIsNot(a[1].optimizer,b[1].optimizer)
        self.assertIsNot(b[1].online,b[1].target)

    def test_14_checkpoint_resume_reproduces_continuous_training(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            a,b,fork=(RunFiles(Path(directory)/name) for name in ('continuous','resumed','fork'))
            training.train(self.cfg,a,2)
            training.train(self.cfg,b,1)
            artifacts=('resume_latest.pt','train.jsonl','fixed.jsonl','updates.jsonl')
            original={name:(b/name).read_bytes() for name in artifacts}
            training.fork_resume(self.cfg,b,fork,1)
            training.train(self.cfg,fork,2,resume=True)
            self.assertEqual({name:(b/name).read_bytes() for name in artifacts},original)
            training.train(self.cfg,b,2,resume=True)
            first=training.read_checkpoint(a/'resume_latest.pt')
            for path in (b,fork):
                state=training.read_checkpoint(path/'resume_latest.pt')
                for key in ('d','s','d_optimizer','gate_optimizer','gate','replay','d_credit_offset','s_actions'):
                    torch.testing.assert_close(first[key],state[key],rtol=0,atol=0)
                self.assertEqual(state['d_updates'],2)
                self.assertEqual(state['s']['updates'],32)

    def test_15_state_and_checkpoint_dimensions_do_not_mix(self):
        self.assertEqual(self.layouts[0].shape,(28,32))
        self.assertEqual(self.layouts[1].shape,(10,32))
        self.assertEqual(self.physical[0][0].transitions[0].obs.shape,(32,24))
        self.assertEqual(self.physical[1][0].transitions[0].obs.shape,(32,19))
        d,s,*_=training.components(self.cfg)
        self.assertEqual(d.features(self.mono).shape,(57,))
        with self.assertRaises(ValueError):
            s.scheduler(np.random.default_rng(1))(np.ones((32,24)),np.ones(32,bool))
        with TemporaryDirectory() as dir:
            path=Path(dir)/'wrong.pt'
            torch.save({'format':2},path)
            with self.assertRaises(ValueError):
                training.read_checkpoint(path)

    def test_16_inference_population_preserves_stationary_network_rate(self):
        _,profiles=registered_profiles()
        original=make_world(SEED,7,'holdout')
        profile=profiles['population_control']
        modified=world(SEED,7,'holdout',profile=profile)
        self.assertEqual(len(modified.w.user_access),3500)
        self.assertEqual(modified.w.user_counts.sum(),3500)
        self.assertAlmostEqual(modified.c.users*modified.c.user_rate,20.)
        for _ in range(8):
            np.testing.assert_allclose(modified.w.rates(modified.w.t).sum(),
                                       original.w.rates(original.w.t).sum(),rtol=0,atol=1e-12)
            modified.w.advance()
            original.w.advance()

    def test_17_inference_profile_only_changes_declared_workload(self):
        _,profiles=registered_profiles()
        profile=profiles['compute_with_compact_payload']
        multi=world(SEED,1,profile=profile)
        mono=world(SEED,1,mono=True,profile=profile)
        np.testing.assert_array_equal(multi.w.nodes,self.multi.w.nodes)
        self.assertEqual(multi.w.directed,self.multi.w.directed)
        self.assertEqual(multi.reference_work,self.multi.reference_work)
        self.assertEqual(multi.cost,self.multi.cost)
        np.testing.assert_array_equal(multi.spec,self.multi.spec)
        np.testing.assert_array_equal(multi.service_work,2*self.multi.service_work)
        np.testing.assert_array_equal(mono.w.stages[:,0],multi.service_work)
        expected=(2720,928,8288,512,8480,544,8224,64,1580,224,24608,112,65568,2336,1472,104,4064,64)
        pairs=[(a,b) for chain in multi.c.services for a,b in zip(chain,chain[1:])]
        self.assertEqual([profile['internal_payload_bytes'][edge] for edge in pairs],list(expected))
        for chain in multi.c.services:
            self.assertEqual(multi.w.stages[chain[0],1],self.multi.w.stages[chain[0],1])
            self.assertEqual(multi.w.stages[chain[-1],2],self.multi.w.stages[chain[-1],2])
        for (a,b),size in zip(pairs,expected,strict=True):
            self.assertEqual(multi.w.stages[a,2],size*8/1_000_000)
            self.assertEqual(multi.w.stages[a,2],multi.w.stages[b,1])

    def test_18_catalog_rejects_duplicate_and_nonadjacent_messages(self):
        plan,profiles=registered_profiles()
        duplicated=dict(plan,messages=[*plan['messages'],plan['messages'][0]])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            payload_bytes(duplicated)
        bad=dict(plan,messages=[dict(plan['messages'][0],edge=[0,2]),*plan['messages'][1:]])
        profile=dict(profiles['payload_only'],internal_payload_bytes=payload_bytes(bad))
        with self.assertRaisesRegex(ValueError,'adjacent stage pair'):
            world(SEED,1,profile=profile)

    def test_19_catalog_bytes_reach_real_internal_link_queues(self):
        _,profiles=registered_profiles()
        profile=profiles['compute_with_compact_payload']
        env=world(SEED,1,profile=profile)
        layout=np.eye(env.K,env.N,dtype=bool)
        self.assertTrue(capacity_ok(layout,env.spec,env.capacity))
        env.w.t=0.
        sim=MeasuredMultistageSimulator(env,layout,lambda obs,mask:int(np.flatnonzero(mask)[0]),1,0,False)
        access=31
        user=int(env.w.users_by_access[access][0])
        sim.add_arrivals([Arrival(.1,user,access,s) for s in range(len(env.c.services))])
        sim.run_until(60.)
        sim.verify()
        self.assertTrue(all(r.finish is not None for r in sim.requests))
        self.assertEqual(sim.decisions,28)
        expected=sum(size*8/1_000_000*len(env.w.routes[a][b])
                     for (a,b),size in profile['internal_payload_bytes'].items())
        self.assertAlmostEqual(sim.internal_wire,expected)

    def test_20_comprised_exp1_trains_evaluates_and_resumes_for_both_architectures(self):
        suite=run.INFERENCE_SUITE
        plan=run.protocol(suite)
        self.assertEqual(plan['settings']['seeds'],list(range(20261113,20261118)))
        self.assertEqual(plan['settings']['episodes']*plan['settings']['periods'],5000)
        plan['settings'].update(seeds=[SEED],episodes=2,periods=2,period_seconds=2.,eval_every=1)
        plan['configs']={SEED:dict(plan['configs'][SEED],periods=2,period_seconds=2.,eval_every=1,s_batch=1)}
        plan['futures']=plan['futures'][:1]
        plan['evaluation']['periods']=2
        args=['comparison.run','--suite',suite,'--workers','1']
        with TemporaryDirectory() as directory, patch.object(run,'INFERENCE_ROOT',Path(directory)), \
                patch.object(run,'protocol',return_value=plan), patch('sys.argv',args), redirect_stdout(StringIO()):
            run.main()
            root=Path(directory)
            review=json.loads((root/f'{suite}.review.json').read_bytes())
            for arm,name in run.INFERENCE_ARMS.items():
                folder=root/name/suite/f'seed{SEED}'
                for file in ('best.pt','last.pt','resume_latest.pt','train.raw.png','fixed.jsonl','updates.jsonl'):
                    self.assertTrue((folder/file).is_file(),file)
                self.assertEqual(len(list(folder.glob('holdout.*.json'))),4)
                self.assertEqual(review['training'][str(SEED)][arm]['last_gs'],4)
                self.assertEqual(review['training'][str(SEED)][arm]['s_updates'],32)
                altered=dict(plan,training_profile='compute_only')
                with self.assertRaisesRegex(ValueError,'independent training arm'):
                    run.load_projection(json.loads(json.dumps(altered)),arm,SEED,'best')
            multi=root/'proposed'/suite/f'seed{SEED}'
            mono=root/'MSP'/suite/f'seed{SEED}'
            name=next(multi.glob('holdout.*.json')).name
            before=(multi/name).read_bytes()
            (mono/name).unlink()
            run.main()
            self.assertEqual((multi/name).read_bytes(),before)
            self.assertTrue((mono/name).is_file())

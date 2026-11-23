"""Protocol, paired interventions, and prequential diagnostic checks."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from scripts.exp030_audit import demand_summary, load_conditioned_bias
from scripts.exp030_frozen import SEEDS, PAIRS, registry, verify_pairs
from edge_joint.adaptive_runner import read_checkpoint, train
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import Config, D_CREDIT_CALIBRATION_RATE
from scripts.exp024_report import read


class FrozenCases(TestCase):
    def test_registry(self):
        cases = registry()
        main = [c for c in cases if not c['historical'] and not c['sample'] and c['seed']==SEEDS[0]]
        self.assertEqual(set(Counter(c['profile'] for c in main).values()),{4})
        self.assertEqual(len({(c['environment'],c['scene']) for c in main}),16)
        self.assertEqual(len(cases),32)
        for case in cases:
            self.assertEqual(len(case['loads']),24)
        first = {c['id']: (c['environment'],c['scene']) for c in main if c['id'].endswith('.1')}
        for case in cases:
            if not case['historical'] and (case['seed']!=SEEDS[0] or case['sample']):
                self.assertEqual(first[case['id']],(case['environment'],case['scene']))

    def test_pair_checks_continuous_origin_and_shared_exploration(self):
        row = dict(scene=1,period=1,start=0,end=24,trace_sha256='same',requests=2,
                   demand_by_service=[2.],old_matrix='0f',backlog_in=4,d_latent=[1.5],d_mean=[1.])
        cells = {key:dict(rows=[deepcopy(row)]) for key in PAIRS}
        cells['B']['rows'][0].update(d_latent=[2.5],d_mean=[2.])
        verify_pairs(cells)
        cells['B']['rows'][0]['backlog_in'] = 3
        with self.assertRaises(AssertionError):
            verify_pairs(cells)
        cells['B']['rows'][0].update(backlog_in=4,d_latent=[2.4])
        with self.assertRaises(AssertionError):
            verify_pairs(cells)

    def test_coverage_does_not_join_episodes(self):
        row = dict(offered_cpu_load=.70,demand_by_service=[2.],hot_region=0)
        self.assertEqual(demand_summary([row]*16,8)['longest_at_065'],8)
        self.assertEqual(demand_summary([row]*24)['longest_at_065'],24)

    def test_offline_baseline_probe(self):
        offset = torch.tensor([-5., -2.])
        first = torch.tensor([.1, .2])
        square = first.square() + .01
        cross = first*offset-.1
        torch.testing.assert_close(load_conditioned_bias(offset,first,square,cross,first),
                                   torch.zeros_like(first))
        self.assertTrue(torch.all(load_conditioned_bias(offset,first,square,cross,first+.05)<0.))

    def test_conditioned_credit_matches_history_and_resumes_exactly(self):
        cfg = Config(seed=20261113,macro_mode='redeploy',periods=2,period_seconds=2.,s_batch=1,
                     d_decoder='link-priced',d_diagnostics=True,d_advantage_calibration=True,
                     d_load_conditioned_credit=True)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            continuous,resumed = [RunFiles(Path(temp)/name) for name in ('continuous','resumed')]
            train(cfg,continuous,2)
            train(cfg,resumed,1)
            start = read_checkpoint(resumed/'resume_latest.pt')
            train(cfg,resumed,2,resume=True)
            a,b = (read_checkpoint(out/'resume_latest.pt') for out in (continuous,resumed))
            for key in ('d','s','replay','d_optimizer','d_credit_offset','d_load_credit_moments'):
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            rows = read(resumed/'train.jsonl')
            updates = read(resumed/'updates.jsonl')
            self.assertEqual(updates[0]['d_credit_load_bias'],[0.,0.])
            loads = torch.tensor([r['offered_cpu_load']-.20 for r in rows[:2]])
            adv = torch.tensor([r['d_credit']['advantage'] for r in rows[:2]])
            moments = torch.stack((loads,loads.square(),loads*adv))*D_CREDIT_CALIBRATION_RATE
            torch.testing.assert_close(torch.tensor(start['d_load_credit_moments']),moments,rtol=0,atol=0)
            second_load = torch.tensor([r['offered_cpu_load']-.20 for r in rows[2:]])
            expected = load_conditioned_bias(torch.tensor(start['d_credit_offset']),*moments,second_load)
            torch.testing.assert_close(torch.tensor(updates[1]['d_credit_load_bias']),expected,rtol=0,atol=0)

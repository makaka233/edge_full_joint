from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from edge_joint.adaptive_runner import read_checkpoint, train
from edge_joint.adaptive_storage import RunFiles
from edge_joint.adaptive_training import Config, D_CREDIT_CALIBRATION_RATE
from scripts.exp024_report import read
from scripts.exp029_report import factorial, intervals, layout_difference
from scripts.exp029_run import jobs


class Exp029Tests(TestCase):
    def test_preregistration_uses_32_unique_profiles_with_separate_reference_cohorts(self):
        plan = jobs()
        historical = [x for x in plan if x['historical']]
        self.assertEqual(len(historical), 16)
        independent = [x for x in plan if not x['historical']]
        unique = {x['id']: x for x in independent}
        self.assertEqual(len(unique), 32)
        self.assertEqual(len({x['environment'] for x in unique.values()}), 32)
        self.assertEqual(set(Counter(x['profile'] for x in unique.values()).values()), {8})
        by_cohort = Counter(x['cohort'] for x in independent)
        self.assertEqual(by_cohort, {'priced14': 16, 'priced17': 16, 'cpu14': 8, 'priced13': 8})

    def test_direction_and_interaction(self):
        self.assertEqual(factorial(10, 7, 11, 5), {'D': -3, 'S': 1, 'joint': -5, 'interaction': -3})

    def test_stratified_pairing_retains_the_number_of_independent_trajectories(self):
        result = intervals({'natural': [-2., -2.], 'burst': [4., 4.]}, 'constant')
        self.assertEqual(result, dict(n=4, estimate=1., lower=1., upper=1., positive=2, negative=2))

    def test_matrix_support_bit_order(self):
        zero = {'active_matrix': '00'*112}
        two = {'active_matrix': '80'+'00'*110+'01'}
        result = layout_difference([zero], [two])
        self.assertEqual(result['mean_hamming'], 2.)
        self.assertEqual(result['support_changed'], [1]+[0]*26+[1])

    def test_causal_credit_and_exact_resume_with_live_updates(self):
        cfg = Config(seed=20261114, macro_mode='redeploy', d_diagnostics=True,
                     periods=2, period_seconds=2., s_batch=1, d_advantage_calibration=True)
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            continuous, resumed, baseline = (RunFiles(Path(tmp)/name) for name in ('a', 'b', 'c'))
            train(cfg, continuous, 2)
            train(cfg, resumed, 1)
            start = read_checkpoint(resumed/'resume_latest.pt')
            train(replace(cfg, d_advantage_calibration=False), baseline, 1)
            control = read_checkpoint(baseline/'resume_latest.pt')
            self.assertTrue(all(torch.equal(value, control['d'][key]) for key, value in start['d'].items()))
            train(cfg, resumed, 2, resume=True)
            a, b = (read_checkpoint(out/'resume_latest.pt') for out in (continuous, resumed))
            self.assertEqual(a['d_credit_offset'], b['d_credit_offset'])
            self.assertEqual((a['d_updates'], a['s']['updates']), (2, 32))
            for key in ('d', 'd_optimizer', 's', 'replay', 'gate', 'gate_optimizer'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
            rows = read(resumed/'train.jsonl')
            update = read(resumed/'updates.jsonl')
            self.assertEqual(update[0]['d_credit_offset'], [0., 0.])
            for j, row in enumerate(rows[:2]):
                prior = row['d_credit']['advantage']*D_CREDIT_CALIBRATION_RATE
                self.assertAlmostEqual(prior, update[1]['d_credit_offset'][j], places=6)
                credit = rows[j+2]['d_credit']
                self.assertAlmostEqual(credit['advantage']-prior, credit['actor_advantage'], places=5)

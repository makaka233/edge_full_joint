"""Lightweight checks for the EXP031 causal diagnostics."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from scripts.exp029_report import factorial
from scripts.exp030_continue import independent_registry
from scripts.exp031_frozen import PAIRS, registry, stage_statistics
from scripts.exp031_report import observation_alias, regression_gate


def effect(values):
    contrasts = factorial(*values)
    contrasts['D_given_S1'] = contrasts['D']+contrasts['interaction']
    contrasts['S_given_D1'] = contrasts['S']+contrasts['interaction']
    return {k:dict(estimate=v, lower=v-.001, upper=v+.001) for k, v in contrasts.items()}


class DeploymentAttributionTests(TestCase):
    def test_registered_independence_and_archive_controls(self):
        history, *independent = registry()
        old = independent_registry()[-1]
        self.assertEqual((history['scene'], history['environment']), (old['scene'], old['environment']))
        self.assertTrue(history['historical'])
        self.assertFalse(history['sample'])
        self.assertEqual(Counter(x['profile'] for x in independent),
                         {p:4 for p in ('natural','burst','sustained055','sustained070')})
        self.assertEqual(len({(x['environment'], x['scene']) for x in independent}),16)
        self.assertTrue(all(x['sample'] and len(x['loads']) == 24 for x in independent))
        self.assertEqual(PAIRS['B'], ('conditioned','exp029'))
        self.assertEqual(PAIRS['C'], ('exp029','conditioned'))

    def test_dominance_requires_a_replicated_joint_degradation(self):
        effects = dict(reward=effect((0.,-.1,0.,-.1)), e2e=effect((0.,.1,0.,.1)),
                       p99=effect((0.,.1,0.,.1)))
        self.assertEqual(regression_gate(effects)['attribution'],'D')
        effects['e2e'] = effect((0.,0.,.1,.1))
        self.assertIsNone(regression_gate(effects)['attribution'])
        effects['reward'] = effect((0.,0.,0.,0.))
        self.assertEqual(regression_gate(effects)['attribution'],'S')

    def test_interaction_and_tail_only_are_distinguished(self):
        result = dict(reward=effect((0.,0.,0.,-.1)), e2e=effect((0.,0.,0.,.1)),
                      p99=effect((0.,0.,0.,.1)))
        self.assertEqual(regression_gate(result)['attribution'],'interaction')
        result['reward'] = result['e2e'] = effect((0.,0.,0.,0.))
        self.assertIsNone(regression_gate(result)['attribution'])

    def test_equal_inputs_can_have_different_live_work(self):
        row = dict(period=2,input=[1.]*57,cpu=[0.,0.],wire=[0.])
        cells = {p:dict(observations=[deepcopy(row)]) for p in PAIRS}
        cells['C']['observations'][0]['cpu'][1]=.5
        cells['D']['observations'][0]['input'][0]=2.
        alias = observation_alias(dict(cells=cells))
        self.assertEqual(alias['exact_input_pairs'],1)
        self.assertEqual(alias['max_cpu_example']['node'],1)
        self.assertEqual(alias['max_cpu_example']['max_node_gap_cu'],.5)

    def test_stage_residence_and_hop_bins_preserve_cross_period_work(self):
        done = dict(ready=(1., 1), stage=7, compute=(3., 1), finish=(3.5, 1),
                    cpu_cu=2., stage_mbit=5., hops=[(8, 1., 3.)])
        carried = dict(ready=(2.5, 2), stage=7, stage_mbit=5., hops=[(9, 2.5, None)])
        sim = SimpleNamespace()
        with patch('scripts.exp031_frozen.stage_trace', return_value=[done, carried]):
            stats = stage_statistics(sim,[dict(period=1),dict(period=2)],2.)
        self.assertEqual(stats['stages'],[[1,7,1,1,1,2.,.5,2.],[2,7,2,1,0,0.,0.,0.]])
        self.assertEqual(stats['hops'],[[1,7,8,1,1,5.,2.],[2,7,9,1,0,5.,0.]])

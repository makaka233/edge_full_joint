"""Check the causal comparison and its pre-registered decision rule."""

from unittest import TestCase

from scripts.exp032_report import contrasts, inference, pooled_interactions
from scripts.exp032_run import future_ids, suffix_loads


PLAN = dict(futures=4, minimum_reward=.03, minimum_e2e_relative=.01,
            minimum_consistent_futures=3, bootstrap_draws=1000)


def comparison(reward, delay):
    return dict(reward=reward, e2e=-delay*.5, e2e_advantage=delay, p95=0., p99=0.)


class CounterfactualTests(TestCase):
    def test_future_random_sources_are_distinct_and_repeatable(self):
        draws = future_ids('one')
        self.assertEqual(draws, future_ids('one'))
        values = [r[k] for case in ('one','two') for r in future_ids(case)
                  for k in ('request_seed','demand_seed')]
        self.assertEqual(len(set(values)),16)

    def test_branch_keeps_the_current_load_and_extends_sustained_profile(self):
        spec = dict(period=3,case=dict(profile='sustained070',loads=[.70]*2))
        self.assertEqual(suffix_loads(spec,4),(None,None,None,.70,.70,.70))
        spec['case'] = dict(profile='burst',loads=[None,.70,.70,None])
        self.assertEqual(suffix_loads(spec,5),(None,None,None,.70,None,None,None))

    def test_ranking_flip_requires_both_metrics_and_a_real_margin(self):
        samples = [[comparison(.07,.03),comparison(-.06,-.04)] for _ in range(4)]
        result = inference(samples,PLAN,'genuine')
        self.assertTrue(result['confirmed'])
        self.assertEqual(result['consistent_futures'],4)
        self.assertEqual(result['orientation'],[1,-1])
        self.assertAlmostEqual(result['interaction']['reward'][0],.13)
        near_zero = [[comparison(.02,.03),comparison(-.03,-.002)] for _ in range(4)]
        conflict = [[comparison(.06,.04),comparison(-.07,.03)] for _ in range(4)]
        self.assertFalse(inference(near_zero,PLAN,'tiny')['confirmed'])
        self.assertFalse(inference(conflict,PLAN,'conflict')['confirmed'])

    def test_crossing_at_one_future_does_not_pass(self):
        samples = [[comparison(.10,.04),comparison(-.08,-.04)]]
        samples += [[comparison(.04,.03),comparison(.02,.03)] for _ in range(3)]
        self.assertFalse(inference(samples,PLAN,'inconsistent')['confirmed'])

    def test_contrasts_and_pooled_interaction_keep_state_pairs_together(self):
        scores = {'Q1.X1':dict(reward=-4.,e2e=.4,p95=.5,p99=.6),
                  'Q1.X2':dict(reward=-5.,e2e=.5,p95=.65,p99=.8),
                  'Q2.X1':dict(reward=-6.,e2e=.6,p95=.7,p99=.9),
                  'Q2.X2':dict(reward=-5.,e2e=.5,p95=.65,p99=.8)}
        paired = contrasts(scores)
        self.assertEqual([x['reward'] for x in paired],[1.,-1.])
        self.assertGreater(paired[0]['e2e_advantage'],0.)
        values = pooled_interactions([dict(profile='natural',samples=[paired]*4)],PLAN)
        self.assertEqual(values['reward'],[2.,2.,2.])
        self.assertAlmostEqual(values['e2e'][0],-.2)

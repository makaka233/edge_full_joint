import unittest

from scripts.research_artifacts import MODE, ROOT, load_run, normalize, rows


class ResearchArtifactsTests(unittest.TestCase):
    def test_archived_runs_account_for_every_reward(self):
        counts = {}
        for input_dir in (ROOT/'evidence/raw').glob('EXP*/placement_*'):
            experiment = input_dir.parent.name
            run = input_dir.name
            cfg, submissions, scores = load_run(
                run, ROOT/'evidence/manifests'/experiment/run,
                input_dir, ROOT/'evidence/summaries'/experiment/run, compressed=True)
            raw = rows(input_dir/'train.jsonl.gz')
            self.assertEqual([item['reward_raw'] for item in submissions], [item['reward'] for item in raw])
            self.assertEqual([item['gs'] for item in submissions], list(range(1, 65)))
            self.assertEqual(len({item['physical_submission_id'] for item in submissions}), 64)
            self.assertEqual(len([item for item in scores if item['split'] == 'fixed']), 9)
            self.assertTrue(any(item['censored_requests'] for item in submissions))
            counts[(cfg['seed'], cfg['mode'])] = len(submissions)
        self.assertEqual(set(counts), {(seed, mode) for seed in (20261113, 20261114) for mode in MODE})
        self.assertEqual(sum(counts.values()), 512)

    def test_no_synthetic_rewards_for_errors_and_missing_submissions(self):
        config = {'mode': 'd-only', 'gs': 2, 'seed': 100}
        errors = [{'gs': 1, 'reason': 'no physical result'}]
        output = normalize([], [], errors, config, 'test', completed=False)
        self.assertEqual([row['reward_raw'] for row in output], [None, None])
        self.assertEqual([row['status'] for row in output], ['evaluation_error', 'not_observed'])
        self.assertEqual(output[0]['reason'], 'no physical result')
        with self.assertRaises(ValueError):
            normalize([], [], errors, config, 'test', completed=True)

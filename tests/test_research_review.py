import unittest

from scripts.research_review import csv_block, csv_tables
from scripts.review_exp007 import EXACT, _compact, build


class ResearchReviewTests(unittest.TestCase):
    def test_csv_preserves_identities_and_exact_metrics(self):
        row = {'id': 'a,b', 'reward_raw': -1.2345678901234567, 'e2e_penalized_s': 0.12345678912345678,
               'queue': None, 'cpu_pressure': 1.123456789}
        compressed = _compact([row])[0]
        actual, = csv_tables(csv_block('test', tuple(row), [compressed]))[0]
        self.assertEqual(actual['id'], row['id'])
        self.assertEqual(actual['queue'], 'N/A')
        for key in row.keys() & EXACT:
            self.assertEqual(float(actual[key]), row[key])
        self.assertAlmostEqual(float(actual['cpu_pressure']), row['cpu_pressure'], places=6)

    def test_exp007_complete_single_document(self):
        text = build()
        train, fixed, holdout, branches, history = csv_tables(text)
        self.assertEqual([len(periods) for periods in (train, fixed, holdout, branches)], [480, 160, 384, 12])
        self.assertEqual(len({(row['seed'], row['gs']) for row in history}), 16)
        self.assertEqual({row['split'] for row in holdout}, {'holdout'})
        self.assertEqual(sum(row['selection'] == 'forced_at_fork' for row in train), 80)
        self.assertLess(len(text.encode('utf8')), 500_000)

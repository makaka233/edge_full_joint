import unittest

from deepseek_workspace.system_credit_ddqn.evaluate import action_relation


class EvaluationTests(unittest.TestCase):
    def test_action_relation_reports_exact_first_divergence(self):
        relation = action_relation([1, 2, 3, 4], [1, 2, 9, 8, 7])
        self.assertEqual(relation["left_count"], 4)
        self.assertEqual(relation["right_count"], 5)
        self.assertEqual(relation["first_divergence"], 2)
        self.assertEqual(relation["common_prefix"], 2)

    def test_identical_action_prefix_reports_length_difference(self):
        relation = action_relation([1, 2], [1, 2, 3])
        self.assertEqual(relation["first_divergence"], 2)
        self.assertEqual(relation["common_prefix"], 2)


if __name__ == "__main__":
    unittest.main()

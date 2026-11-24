import unittest

from deepseek_workspace.state_value_audit.counterfactual import action_set


class ActionSetTests(unittest.TestCase):
    def test_action_set_is_legal_distinct_and_bounded(self):
        decision = {
            "legal": [2, 5, 7, 9],
            "prior_node": 5,
            "full_node": 9,
            "residual_node": 2,
            "q": [0.0] * 32,
        }
        decision["q"][5] = -1
        decision["q"][9] = 2
        decision["q"][2] = 1
        actions = action_set(decision)
        self.assertEqual(len(actions), 3)
        self.assertEqual(len(set(actions)), 3)
        self.assertTrue(set(actions) <= set(decision["legal"]))
        self.assertEqual(actions[:2], [5, 9])


if __name__ == "__main__":
    unittest.main()

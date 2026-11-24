import unittest

from deepseek_workspace.state_value_audit.offline_predictor import split_groups


class OfflineSplitTests(unittest.TestCase):
    def test_complete_trajectories_are_split_without_leakage(self):
        groups = [
            {"id": f"{seed}.{stage}.{profile}", "seed": seed,
             "stage": stage, "profile": profile}
            for seed in (20261113, 20261115, 20261117)
            for stage in ("GS0", "GS16", "GS64")
            for profile in ("natural", "sustained070")
        ]
        train, validation, test = split_groups(groups)
        self.assertFalse(set(item["id"] for item in train) & set(item["id"] for item in test))
        self.assertFalse(set(item["id"] for item in validation) & set(item["id"] for item in test))
        self.assertFalse(set(item["id"] for item in train) & set(item["id"] for item in validation))
        self.assertTrue(all(item["seed"] == 20261117 for item in test))
        self.assertEqual(len(train) + len(validation) + len(test), len(groups))


if __name__ == "__main__":
    unittest.main()

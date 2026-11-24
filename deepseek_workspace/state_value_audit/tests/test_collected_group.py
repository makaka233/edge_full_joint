import json
from pathlib import Path
import unittest


class CollectedGroupTests(unittest.TestCase):
    def test_real_group_has_common_state_actions_and_no_future_features(self):
        files = list(Path("deepseek_workspace/state_value_audit/results/groups").glob("*.json"))
        if not files:
            self.skipTest("Collected smoke group is not available")
        for path in files:
            group = json.loads(path.read_text(encoding="utf8"))["group"]
            self.assertEqual(len(group["states"]), 4)
            for state in group["states"]:
                self.assertGreaterEqual(len(state["actions"]), 2)
                self.assertLessEqual(len(state["actions"]), 3)
                initial = {item["initial_state_sha256"] for item in state["actions"]}
                self.assertEqual(len(initial), 1)
                traces = {
                    tuple(row["trace_sha256"] for row in action["rows"])
                    for action in state["actions"]
                }
                self.assertEqual(len(traces), 1)
                for action in state["actions"]:
                    self.assertEqual(len(action["features_a"]), 97)
                    self.assertEqual(len(action["features_b"]), 139)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from deepseek_workspace.state_value_audit.features import (
    OBSERVATION_COLUMNS,
    action_features,
    state_signature,
)


class FeatureTests(unittest.TestCase):
    def test_observation_schema_and_feature_shapes(self):
        self.assertEqual(len(OBSERVATION_COLUMNS), 24)
        obs = np.zeros((32, 24), dtype=np.float32)
        mask = np.zeros(32, dtype=bool)
        mask[[3, 9]] = True
        obs[:, 7] = mask
        extended = np.ones(42, dtype=np.float32)
        original = action_features(obs, mask, 3, extended, include_extension=False)
        enlarged = action_features(obs, mask, 3, extended, include_extension=True)
        self.assertEqual(original.shape, (97,))
        self.assertEqual(enlarged.shape, (139,))
        np.testing.assert_array_equal(enlarged[:97], original)
        np.testing.assert_array_equal(enlarged[97:], extended)
        self.assertEqual(state_signature(obs, extended).shape, (73,))


if __name__ == "__main__":
    unittest.main()

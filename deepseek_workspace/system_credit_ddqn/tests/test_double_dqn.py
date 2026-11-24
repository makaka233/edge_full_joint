import unittest

import numpy as np

from deepseek_workspace.system_credit_ddqn.credit_ledger import SystemTransition
from deepseek_workspace.system_credit_ddqn.scheduling_agent import SystemCreditScheduler


class DoubleDQNTests(unittest.TestCase):
    def test_event_time_discount_and_online_update(self):
        scheduler = SystemCreditScheduler(20261113)
        mask = np.zeros(32, dtype=bool)
        mask[3] = True
        config = scheduler.config
        transitions = []
        for i, elapsed in enumerate((0.5, 2.0, 8.0, 24.0)):
            obs = np.zeros((config.n, scheduler.features), dtype=np.float32)
            obs[:, 7] = mask
            transitions.append(SystemTransition(
                obs=obs,
                mask=mask.copy(),
                action=3,
                reward=-float(i + 1),
                next_obs=obs.copy(),
                next_mask=mask.copy(),
                terminated=i == 3,
                truncated=False,
                version=0,
                collection_step=1,
                request_id=i,
                stage_index=0,
                window_index=0,
                elapsed_seconds=elapsed,
                decision_index=i,
                decision_time=float(i),
            ))
        before = [parameter.detach().clone() for parameter in scheduler.online.parameters()]
        metrics = scheduler.update_event_time(transitions)
        self.assertTrue(np.isfinite(metrics["s_loss"]))
        self.assertAlmostEqual(metrics["min_discount"], 0.99 ** 24, places=6)
        self.assertEqual(scheduler.updates, 1)
        self.assertTrue(scheduler.resource_q_prior)
        self.assertTrue(any(
            not np.array_equal(old.numpy(), new.detach().numpy())
            for old, new in zip(before, scheduler.online.parameters(), strict=True)
        ))

    def test_true_terminal_has_no_bootstrap(self):
        scheduler = SystemCreditScheduler(20261113)
        mask = np.ones(32, dtype=bool)
        obs = np.zeros((32, scheduler.features), dtype=np.float32)
        transition = SystemTransition(
            obs=obs,
            mask=mask,
            action=5,
            reward=-7.25,
            next_obs=np.zeros_like(obs),
            next_mask=mask.copy(),
            terminated=True,
            truncated=False,
            version=0,
            collection_step=1,
            request_id=1,
            stage_index=0,
            window_index=0,
            elapsed_seconds=4.0,
            decision_index=0,
            decision_time=10.0,
        )
        target, discount = scheduler.event_time_targets([transition])
        self.assertAlmostEqual(float(target[0]), -7.25)
        self.assertAlmostEqual(float(discount[0]), 0.99 ** 4, places=6)


if __name__ == "__main__":
    unittest.main()

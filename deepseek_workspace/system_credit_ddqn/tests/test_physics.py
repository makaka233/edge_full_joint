from dataclasses import replace
from functools import partial
import unittest

import numpy as np
import torch

from edge_d.world import make_world
from edge_joint.adaptive_training import Config, components, episode
from deepseek_workspace.system_credit_ddqn.scheduling_agent import SystemCreditScheduler
from deepseek_workspace.system_credit_ddqn.simulator_adapter import (
    SystemCreditSimulator,
    cohort_windows,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self):
        return Config(
            seed=20261113,
            periods=1,
            period_seconds=0.4,
            s_batch=1,
            s_updates_per_gs=1,
            eval_every=1,
            macro_mode="redeploy",
        )

    def test_system_ledger_matches_original_penalized_delays(self):
        cfg = self.config()
        d, _, _, _, gate, _ = components(cfg)
        scheduler = SystemCreditScheduler(cfg.seed)
        env = make_world(cfg.seed, 0, "train")
        windows = cohort_windows(
            start=0.0,
            periods=cfg.periods,
            period_seconds=cfg.period_seconds,
            tail_seconds=env.tail,
        )
        holder = {}

        def factory(*args, **kwargs):
            sim = SystemCreditSimulator(*args, cohort_windows=windows, **kwargs)
            holder["sim"] = sim
            return sim

        rows, _, transitions = episode(
            cfg,
            env,
            d,
            scheduler,
            gate,
            0,
            "train",
            1,
            simulator_factory=factory,
        )
        sim = holder["sim"]
        self.assertGreater(rows[0]["requests"], 0)
        self.assertEqual(len(transitions), sim.decisions)
        self.assertAlmostEqual(
            sim.credit_audit.total_punished_delay,
            sum(rows[0]["penalized_delays"]),
            places=9,
        )
        self.assertLess(sim.credit_audit.identity_error, 1e-9)
        self.assertTrue(all(item.mask[item.action] for item in transitions))

    def test_adapter_does_not_change_frozen_physical_result(self):
        cfg = replace(self.config(), freeze_ds=True)
        d, _, _, _, gate, _ = components(cfg)
        windows = cohort_windows(
            start=0.0,
            periods=cfg.periods,
            period_seconds=cfg.period_seconds,
            tail_seconds=make_world(cfg.seed, 0, "fixed").tail,
        )
        base = episode(
            cfg,
            make_world(cfg.seed, 0, "fixed"),
            d,
            SystemCreditScheduler(cfg.seed),
            gate,
            0,
            "fixed",
            0,
        )[0]
        adapted = episode(
            cfg,
            make_world(cfg.seed, 0, "fixed"),
            d,
            SystemCreditScheduler(cfg.seed),
            gate,
            0,
            "fixed",
            0,
            simulator_factory=partial(
                SystemCreditSimulator,
                cohort_windows=windows,
                credit_enabled=False,
            ),
        )[0]
        fields = (
            "active_matrix",
            "trace_sha256",
            "reward_raw",
            "completed_delays",
            "penalized_delays",
            "cpu",
            "wired",
            "wireless",
            "conservation_error",
            "capacity_error",
        )
        for left, right in zip(base, adapted, strict=True):
            for field in fields:
                if isinstance(left[field], float):
                    self.assertAlmostEqual(left[field], right[field], places=12)
                else:
                    self.assertEqual(left[field], right[field])


if __name__ == "__main__":
    unittest.main()

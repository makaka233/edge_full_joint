from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

import torch

from edge_d.world import make_world
from edge_joint.adaptive_training import Config
from deepseek_workspace.system_credit_ddqn.training import train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_system_credit_training_updates_ds_and_saves_models(self):
        cfg = Config(
            seed=20261113,
            periods=2,
            period_seconds=0.4,
            s_batch=2,
            s_updates_per_gs=1,
            eval_every=1,
            macro_mode="redeploy",
        )
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "system"
            result = train(cfg, out, 1, world_factory=make_world)
            self.assertEqual(result["d_updates"], 1)
            self.assertGreater(result["s_updates"], 0)
            for name in ("best.pt", "last.pt", "resume_latest.pt", "train.jsonl",
                         "updates.jsonl", "fixed.jsonl", "decisions.jsonl", "manifest.json"):
                self.assertTrue((out / name).is_file(), name)
            updates = [json.loads(line) for line in
                       (out / "updates.jsonl").read_text(encoding="utf8").splitlines()]
            self.assertEqual(len(updates), 1)
            self.assertGreater(updates[0]["d_change"], 0)
            self.assertGreater(updates[0]["s_change"], 0)
            self.assertLess(updates[0]["credit_identity_error"], 1e-9)

    def test_resume_matches_uninterrupted_training(self):
        cfg = Config(
            seed=20261115,
            periods=2,
            period_seconds=0.3,
            s_batch=2,
            s_updates_per_gs=1,
            eval_every=2,
            macro_mode="redeploy",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = root / "continuous"
            resumed = root / "resumed"
            train(cfg, continuous, 2, world_factory=make_world)
            train(cfg, resumed, 1, world_factory=make_world)
            train(cfg, resumed, 2, world_factory=make_world, resume=True)
            left = torch.load(continuous / "resume_latest.pt", weights_only=True)
            right = torch.load(resumed / "resume_latest.pt", weights_only=True)
            for key in ("d", "gate"):
                for name, value in left[key].items():
                    self.assertTrue(torch.equal(value, right[key][name]))
            for tree in ("s",):
                for group in ("online", "target", "optimizer"):
                    for name, value in left[tree][group].items():
                        if torch.is_tensor(value):
                            self.assertTrue(torch.equal(value, right[tree][group][name]))
            self.assertEqual(left["episode"], right["episode"])
            self.assertEqual(left["d_updates"], right["d_updates"])
            self.assertEqual(left["s_actions"], right["s_actions"])
            for name, values in left["replay"]["columns"].items():
                self.assertTrue(torch.equal(values, right["replay"]["columns"][name]))
            left_rows = [json.loads(line) for line in
                         (continuous / "train.jsonl").read_text(encoding="utf8").splitlines()]
            right_rows = [json.loads(line) for line in
                          (resumed / "train.jsonl").read_text(encoding="utf8").splitlines()]
            self.assertEqual(len(left_rows), len(right_rows))
            fields = ("gs", "trace_sha256", "active_matrix", "reward_raw",
                      "completed_delays", "penalized_delays", "s_decisions")
            for a, b in zip(left_rows, right_rows, strict=True):
                self.assertEqual({key: a[key] for key in fields},
                                 {key: b[key] for key in fields})


if __name__ == "__main__":
    unittest.main()

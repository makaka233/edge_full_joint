import base64
import unittest

import numpy as np
import torch

from edge_d.placement import decode
from edge_d.world import make_world
from edge_joint.placement_training import PlacementConfig
from scripts.exp009 import create, train_step
from scripts.exp010 import TRAINABLE, configure, step
from scripts.exp010_diagnostics import add_prefix, ranked_additions


class Exp010Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_mask_score_and_resource_conservation(self):
        cfg = PlacementConfig()
        actor = create(cfg.seed, 'spatial')
        batch, _, _, _ = train_step(actor, cfg, 'spatial', 1, trace_probabilities=True)
        env = make_world(cfg.seed, 0)
        for row in batch:
            latent = np.asarray(row['latent'])
            core = decode(env, latent, row['layout_seed'], True, include_optional=False)
            matrix = core.matrix.copy()
            free = env.capacity-matrix.T@env.spec
            eligible = np.unpackbits(np.frombuffer(bytes.fromhex(row['eligible_mask']), np.uint8)).reshape(matrix.shape)
            stored = np.frombuffer(base64.b64decode(row['eligible_logits_f32le']), '<f4')
            logits = actor.logits(env, latent).detach()
            np.testing.assert_allclose(logits.numpy()[eligible.astype(bool)], stored, atol=1e-6)
            rng = np.random.default_rng(row['bit_draw_seed'])
            order = rng.permutation(matrix.size)
            draws = rng.random(matrix.shape)
            prob = logits.sigmoid().numpy()
            for entry in order:
                stage,node = divmod(int(entry),env.N)
                allow = not matrix[stage,node] and bool(np.all(env.spec[stage] <= free[node]+1e-9))
                self.assertEqual(allow, bool(eligible[stage,node]))
                if allow and draws[stage,node] < prob[stage,node]:
                    matrix[stage,node] = True
                    free[node] -= env.spec[stage]
            self.assertEqual(np.packbits(matrix).tobytes().hex(), row['matrix_packed_hex'])
            self.assertTrue(matrix.any(1).all())
            self.assertTrue(np.all(matrix.T@env.spec <= env.capacity+1e-7))
            self.assertEqual(int(matrix[eligible.astype(bool)].sum()), row['selected_bits'])
            self.assertAlmostEqual(row['spatial_log_prob'], row['selected_log_prob']+row['rejected_log_prob'], places=3)
            for window in row['windows']:
                self.assertLess(window['conservation'], 1e-6)

    def test_independent_head_updates(self):
        cfg = PlacementConfig()
        for arm in ('gaussian','spatial','joint'):
            with self.subTest(arm=arm):
                actor = create(cfg.seed, 'spatial')
                groups, optimizers = configure(actor, arm)
                before = {k: [value.detach().clone() for value in module.parameters()] for k,module in groups.items()}
                _, stats = step(cfg, actor, arm, groups, optimizers, 1)
                for head,module in groups.items():
                    unchanged = all(torch.equal(old, current) for old,current in zip(before[head],module.parameters()))
                    self.assertEqual(unchanged, head not in TRAINABLE[arm])
                    self.assertEqual(f'{head}_grad_norm' in stats, not unchanged)

    def test_count_counterfactual_prefix_legal(self):
        cfg = PlacementConfig()
        env = make_world(cfg.seed, 0)
        core = decode(env, np.zeros(8), 37, True, include_optional=False).matrix
        priority = np.random.default_rng(14).logistic(size=core.shape)
        prefix = ranked_additions(env, core, priority, 100)
        self.assertGreaterEqual(len(prefix), 100)
        matrix = add_prefix(core, prefix[:100])
        self.assertEqual(int(matrix.sum()-core.sum()), 100)
        self.assertTrue(np.all(matrix.T@env.spec <= env.capacity+1e-7))

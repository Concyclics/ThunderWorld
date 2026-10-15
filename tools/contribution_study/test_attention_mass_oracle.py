import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'packages/fp4-repair/src'))
from attention_mass_oracle import attention_tile_mass, mass_capture, oracle_mask, paired_gap, selector_cost
from structure_repair import mask_cost, random_mask, window_mask


class TestMassOracle(unittest.TestCase):
    def test_full_dense_gqa_reference_and_tails(self):
        for m, n, h, hk in [(129, 257, 4, 2), (416, 547, 4, 1), (17, 133, 2, 2)]:
            gen = torch.Generator().manual_seed(m)
            q = torch.randn(1, m, h, 16, generator=gen).bfloat16()
            k = torch.randn(1, n, hk, 16, generator=gen).bfloat16()
            got = attention_tile_mass(q, k)
            for head in range(h):
                p = torch.softmax(q[0, :, head].float() @ k[0, :, head // (h // hk)].float().T / 4, -1)
                expected = torch.stack([torch.stack([p[i:i+128, j:j+128].sum(1).mean()
                    for j in range(0, n, 128)]) for i in range(0, m, 128)])
                torch.testing.assert_close(got[head], expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(got.sum(-1), torch.ones_like(got[..., 0]), rtol=1e-6, atol=1e-7)

    def test_uniform_zero_and_tail_keys(self):
        got = attention_tile_mass(torch.zeros(1, 129, 4, 16), torch.zeros(1, 257, 2, 16)).numpy()
        np.testing.assert_allclose(got, np.broadcast_to(np.array([128, 128, 1]) / 257, got.shape), rtol=1e-6)
        mask = np.zeros_like(got, dtype=bool)
        mask[:, -1] = True
        self.assertAlmostEqual(mass_capture(got, mask, 129), 1/129, places=7)

    def test_exhaustive_maximum_mass(self):
        scores = np.array([[[.6, .3, .1]], [[.05, .75, .2]]])
        mask = oracle_mask(scores, .5, 5)
        optimum = max(scores.reshape(-1)[list(c)].sum() for c in itertools.combinations(range(6), 3))
        self.assertAlmostEqual(float((scores * mask).sum()), optimum)

    def test_matched_issued_mma_and_gqa(self):
        for m, n in [(129, 257), (416, 547)]:
            scores = np.random.default_rng(m).random((4, (m+127)//128, (n+127)//128))
            scores /= scores.sum(-1, keepdims=True)
            for coverage in [0, .25, .5, .75, 1]:
                masks = [oracle_mask(scores, coverage, 71), random_mask(scores.shape, coverage, 7),
                         window_mask(scores.shape, coverage, m, n)]
                costs = [mask_cost(mask, m, n, 2)['total_mma'] for mask in masks]
                self.assertEqual(len(set(costs)), 1)
                self.assertGreaterEqual(mass_capture(scores, masks[0], m) + 1e-12,
                                        max(mass_capture(scores, x, m) for x in masks))
                if coverage in (0, 1):
                    self.assertEqual(masks[0].all() if coverage else not masks[0].any(), True)

    def test_oracle_cost_not_hidden(self):
        cost = selector_cost(129, 257, 4, 128)
        self.assertEqual(cost['fp32_qk_flops'], 2*129*257*4*128)
        self.assertFalse(cost['included_in_matched_fp4_budget'])

    def test_benefit_recovery_is_not_clipped(self):
        result = paired_gap([[1, 1]], [[2, 2]], [[4, 4]], replicates=50)
        self.assertEqual(result['oracle_benefit_recovered_pct'], 150)
        self.assertEqual(result['nmse_gap_pct'], -50)
        self.assertEqual(result['nmse_gap_ci95_pct'], [-50, -50])

    def test_incomplete_mass_rejected(self):
        with self.assertRaises(ValueError):
            oracle_mask(np.ones((2, 3, 4)), .5, 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

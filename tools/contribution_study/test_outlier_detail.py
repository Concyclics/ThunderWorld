"""Independent checks for the newly introduced candidate search/patch logic."""
import unittest
import torch
from thunderworld_fp4.cosmos.calibration import _encode, decode
from outlier_detail import search, losses, group_indices, patch_asset


class OutlierDetailTest(unittest.TestCase):
    def test_covariance_matches_direct_three_term_error(self):
        gen = torch.Generator().manual_seed(472)
        z = torch.randn(3, 16, generator=gen)
        a = torch.randn(40, 3, 16, generator=gen)
        a0, ar = .95*a, .035*a
        features = torch.cat((a-a0-ar, a0+ar, ar), -1)
        cov = torch.einsum('mgd,mge->gde', features, features)
        w0, w1 = .91*z, .08*z
        expected = torch.einsum('mgk,gk->mg', a, z)-torch.einsum('mgk,gk->mg', a0+ar, w0)-torch.einsum('mgk,gk->mg', a0, w1)
        torch.testing.assert_close(losses(z, w0[None], w1[None], cov)[0], expected.square().sum(0), atol=1e-5, rtol=2e-5)

    def test_exhaustive_search_and_actual_residual(self):
        gen = torch.Generator().manual_seed(748)
        z = torch.randn(4, 16, generator=gen)*9
        z[0] = 0
        c0, s0 = _encode(z, 1.)
        c1, s1 = _encode(z-decode(c0, s0, 1.), 1.)
        f = torch.randn(4, 48, 48, generator=gen)
        cov = f.transpose(1, 2)@f
        saved = z.clone()
        found = search(z, c0, s0, c1, s1, cov)
        candidates = []
        for position in range(-1, 16):
            inp = z.clone()
            if position >= 0:
                inp[:, position] = 0
            for bs in (.75, .875, 1., 1.5):
                cc0, ss0 = _encode(inp, bs)
                v0 = decode(cc0, ss0, 1.)
                for rs in (.875, 1., 1.125):
                    cc1, ss1 = _encode(z-v0, rs)
                    candidates.append(losses(z, v0[None], decode(cc1, ss1, 1.)[None], cov)[0])
        torch.testing.assert_close(found['joint']['loss'], torch.stack(candidates).min(0).values)
        self.assertTrue(torch.equal(z, saved))
        for method in found.values():
            self.assertTrue(torch.isfinite(method['loss']).all())
        self.assertEqual(int(found['joint']['removed'][0]), -1)
        naive = found['max_fixed']
        residual = z-decode(naive['c0'], naive['s0'], 1.)
        ec, es = _encode(residual, 1.)
        self.assertTrue(torch.equal(ec, naive['c1']))
        self.assertTrue(torch.equal(es.view(torch.uint8), naive['s1'].view(torch.uint8)))

    def test_nonmax_removal_fixed_scale_preserves_retained_base(self):
        z = torch.tensor([[6., .3, .9, 1.3, -1.7, 2., 0., 0.]*2])
        c0, s0 = _encode(z, 1.)
        changed = z.clone(); changed[0, 1] = 0
        c1, s1 = _encode(changed, 1.)
        self.assertTrue(torch.equal(s0.view(torch.uint8), s1.view(torch.uint8)))
        keep = torch.ones_like(z, dtype=torch.bool); keep[0, 1] = False
        self.assertTrue(torch.equal(decode(c0, s0, 1.)[keep], decode(c1, s1, 1.)[keep]))

    def test_patch_only_paid_group_and_mandatory_support(self):
        z = torch.randn(256, 256, generator=torch.Generator().manual_seed(10))
        c0, s0 = _encode(z, 1.)
        c1, s1 = _encode(z-decode(c0, s0, 1.), 1.)
        old = dict(codes=torch.stack((c0, c1)), scales=torch.stack((s0, s1)))
        blocks = torch.tensor([2])
        nr, kg = group_indices(blocks, 256)
        patch = dict(c0=torch.zeros(1024, 8, dtype=torch.uint8), c1=torch.ones(1024, 8, dtype=torch.uint8),
                     s0=torch.ones(1024, 1).to(torch.float8_e4m3fn), s1=torch.ones(1024, 1).to(torch.float8_e4m3fn),
                     removed=torch.zeros(1024, dtype=torch.long))
        new = patch_asset(old, patch, nr, kg, blocks)
        self.assertEqual(new['mandatory_w_blocks'], (2,))
        self.assertTrue(torch.equal(new['codes'][:, :128], old['codes'][:, :128]))
        self.assertTrue(torch.equal(new['codes'][:, 128:, 64:], old['codes'][:, 128:, 64:]))
        self.assertTrue(torch.equal(old['codes'][0], c0))


if __name__ == '__main__':
    unittest.main()

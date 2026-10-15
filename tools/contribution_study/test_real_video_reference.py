"""CPU regression checks for the recorded-video evaluation protocol."""
import unittest

import numpy as np
import torch
from PIL import Image

from real_video_reference import nearest_indices, resize_crop
from real_video_metrics import combine_full


class RealVideoProtocolTests(unittest.TestCase):
    def test_timestamp_alignment_uses_pts_and_rejects_padding(self):
        # VFR: a missing frame makes ordinal frame matching incorrect.
        timestamps = [0., .02, .08, .11, .18]
        np.testing.assert_array_equal(nearest_indices(timestamps, [0., .04, .08, .12, .16]),
                                      [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'no padding'):
            nearest_indices(timestamps, [0., .20])
        with self.assertRaisesRegex(ValueError, 'Non-increasing'):
            nearest_indices([0., .1, .1], [0., .1])

    def test_exact_timestamp_tie_chooses_earlier(self):
        np.testing.assert_array_equal(nearest_indices([0., .25, .5], [.125, .375]), [0, 1])

    def test_conditioning_preprocess_agrees_at_extreme_aspect_ratios(self):
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _preprocess_conditioning_image
        rng = np.random.default_rng(5)
        for h, w in [(300, 500), (501, 299), (241, 319), (192, 320)]:
            with self.subTest(shape=(h, w)):
                frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                actual = _preprocess_conditioning_image(Image.fromarray(frame), height=192, width=320)
                actual = ((actual[0].permute(1, 2, 0)+1)*127.5).round().byte().numpy()
                np.testing.assert_array_equal(resize_crop(frame), actual)

    def test_including_conditioning_frame_uses_global_mse_not_mean_psnr(self):
        # First-frame equality must not turn the full clip's finite PSNR infinite.
        reference = np.ones((61, 2, 2, 3), dtype=np.float32)
        video = reference.copy()
        video[1:] -= .5
        actual = combine_full({'ssim_mean': .3, 'lpips': .6},
                              {'ssim_mean': 1., 'lpips': 0.}, video, reference)
        self.assertAlmostEqual(actual['mse'], .25*60/61)
        self.assertAlmostEqual(actual['psnr_db'], 10*np.log10(61/(.25*60)))
        self.assertAlmostEqual(actual['ssim_mean'], (60*.3+1)/61)
        self.assertAlmostEqual(actual['lpips'], .6*60/61)
        self.assertFalse(actual['psnr_exact'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

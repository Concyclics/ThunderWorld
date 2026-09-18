import pytest
from evidence_outlier_gate import mean_guard, paired_rows


def test_psnr_improvement_cannot_replace_failed_lpips_guard():
    reference = dict(lpips=.1, ssim_mean=.9, relative_l2=.1, psnr_db=25.)
    candidate = dict(lpips=.11, ssim_mean=.91, relative_l2=.09, psnr_db=26.)
    assert mean_guard(candidate, reference) == ['lpips']
    assert mean_guard(reference, reference) == ['lpips']


def test_missing_pairs_fail_closed():
    with pytest.raises(ValueError, match='paired'):
        paired_rows({('scene', 1): {}}, {('scene', 2): {}})


def test_worst_direction_and_counts():
    ref = {('a', 1): dict(lpips=.1, ssim_mean=.9, relative_l2=.1, psnr_db=25.),
           ('b', 2): dict(lpips=.1, ssim_mean=.9, relative_l2=.1, psnr_db=25.)}
    new = {('a', 1): dict(lpips=.09, ssim_mean=.91, relative_l2=.09, psnr_db=26.),
           ('b', 2): dict(lpips=.12, ssim_mean=.87, relative_l2=.13, psnr_db=24.)}
    result = paired_rows(new, ref)
    assert result['lpips']['mean_delta'] == pytest.approx(.005)
    assert result['ssim_mean']['worst_delta'] == pytest.approx(-.03)
    assert result['psnr_db']['worst_seed'] == 2
    assert all(row['degraded_seeds'] == 1 for row in result.values())

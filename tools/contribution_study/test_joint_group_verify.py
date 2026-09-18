"""Independent checks of the offline three-term proposal algebra."""
import torch
from joint_group_verify import group_loss, proposals
from thunderworld_fp4.cosmos.calibration import _encode, decode


def test_three_and_four_covariance_match_explicit_current_input_errors():
    torch.manual_seed(193)
    t, a0, ar = [torch.randn(29, 5, 16, dtype=torch.float64) for _ in range(3)]
    w, w0, wr = [torch.randn(5, 16, dtype=torch.float64) for _ in range(3)]
    features = torch.cat((t-a0-ar, a0+ar, ar), -1)
    cov = torch.einsum('mgd,mge->gde', features, features)
    for cross in (False, True):
        actual = torch.einsum('mgk,gk->mg', t, w)-torch.einsum('mgk,gk->mg', a0+ar, w0)-torch.einsum('mgk,gk->mg', a0, wr)
        if cross:
            actual -= torch.einsum('mgk,gk->mg', ar, wr)
        torch.testing.assert_close(group_loss(w, w0, wr, cov, cross), actual.square().sum(0))


def test_group_search_keeps_noop_and_reconstructs_actual_residual():
    torch.manual_seed(42)
    z = torch.randn(11, 16)*7
    c0, s0 = _encode(z, 1.)
    c1, s1 = _encode(z-decode(c0, s0, 1.), 1.)
    features = torch.randn(19, 11, 48)
    cov = torch.einsum('mgd,mge->gde', features, features)
    original = group_loss(z, decode(c0, s0, 1.), decode(c1, s1, 1.), cov)
    result = proposals(z, c0, s0, c1, s1, cov)
    for label, row in result.items():
        assert torch.all(row['loss'] <= original)
        if label == 'scale_only':
            assert (row['removed'] == -1).all()
        b = decode(row['c0'], row['s0'], 1.)
        r = decode(row['c1'], row['s1'], 1.)
        torch.testing.assert_close(group_loss(z, b, r, cov), row['loss'])
        # Every stored residual must be an actual encoding of z - decoded W0.
        options = [_encode(z-b, scale) for scale in (.875, 1., 1.125)]
        covered = torch.zeros(len(z), dtype=torch.bool)
        for cc, ss in options:
            covered |= (cc == row['c1']).all(-1) & (ss.view(torch.uint8) == row['s1'].view(torch.uint8)).all(-1)
        assert covered.all()
    assert torch.all(result['joint']['loss'] <= result['scale_only']['loss'])


def test_equal_reconstructed_weight_does_not_imply_equal_three_term():
    a0, ar, w0, wr, shift = [torch.tensor(x) for x in (2., .25, 3., .5, 1.)]
    three = a0*(w0+wr)+ar*w0
    shifted = a0*(w0+shift+wr-shift)+ar*(w0+shift)
    assert shifted-three == ar*shift


def test_temporal_coverage_and_complete_aggregation():
    import pytest
    from schedule_calibrate import phase_sample_steps, aggregate_temporal_samples
    assert phase_sample_steps(1) == [0, 7, 18, 28]
    assert phase_sample_steps(3) == [0, 3, 6, 7, 12, 17, 18, 22, 27, 28, 31, 34]
    rows = {str(step): {'attention|0|0|0': dict(step=step, losses={'base': float(step)}, baseline_bitwise_equal=True)} for step in (0, 3, 6)}
    result = aggregate_temporal_samples(rows, {'attention|0|0|0': {}}, 3)
    assert result['attention|0|0|0']['losses']['base'] == 3.
    with pytest.raises(ValueError, match='Incomplete within-phase'):
        aggregate_temporal_samples({'0': rows['0']}, {'attention|0|0|0': {}}, 3)

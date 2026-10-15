import pytest
import torch
from importance_joint_calibrate import objective, top_weight_tiles


def test_objective_balances_datasets_and_penalizes_worst_panel():
    assert objective([1., 1., 5.], ['r', 'r', 'p']) == 4.25


def test_weight_contribution_uses_activation_and_a_sensitivity_uses_w0():
    w = torch.full((256, 256), 1.5, dtype=torch.bfloat16)
    codes = torch.stack([torch.full((256, 128), 0x22, dtype=torch.uint8),
                         torch.full((256, 128), 0x11, dtype=torch.uint8)])
    scales = torch.ones((2, 256, 16)).to(torch.float8_e4m3fn)
    rms = torch.cat([torch.ones(128), torch.full((128,), 2.)])
    ids, gain, sensitivity = top_weight_tiles(w, codes, scales, torch.tensor(1.), torch.ones(256), rms, 1)
    assert ids == [1]
    torch.testing.assert_close(gain[:, 1], 4 * gain[:, 0])
    torch.testing.assert_close(sensitivity, torch.full((256,), 256.))
    assert top_weight_tiles(w, codes, scales, torch.tensor(1.), torch.ones(256), rms, 0)[0] == []
    with pytest.raises(ValueError):
        top_weight_tiles(w, codes, scales, torch.tensor(1.), torch.ones(256), rms, -1)

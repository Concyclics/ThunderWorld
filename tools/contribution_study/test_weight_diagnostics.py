import torch
from evidence_weight_diagnostics import counts


def test_extreme_ties_do_not_create_false_nonextreme_errors():
    w = torch.tensor([[6., -6., 2., *([0.]*13)], [6., -6., 2., *([0.]*13)], [0.]*16])
    base = w.clone()
    base[0, 0] -= 1
    base[0, 2] -= 1  # tied error includes an extreme, so not excluded
    base[1, 2] -= 1  # unique worst error is an interior element
    result = counts(w, base)
    assert result['groups'] == 3 and result['nonzero_error_groups'] == 2
    assert result['worst_error_excludes_absolute_max'] == 1
    assert result['worst_error_excludes_signed_min_and_max'] == 1

import pytest
import torch
from awq_marlin import exact_codes_from_codebook
from thunderworld_fp4.cosmos.calibration import decode


def test_exact_codebook_extraction_keeps_all_codes_and_signed_zero():
    ids = torch.arange(16, dtype=torch.uint8).repeat(3, 2)
    codes = ids[:, 0::2] | (ids[:, 1::2] << 4)
    scales = torch.tensor([[1., .5], [2., 1.], [.25, .125]], dtype=torch.float8_e4m3fn)
    factor = torch.tensor(.173)
    values = decode(codes, scales.float()*factor, 1.).bfloat16()
    restored = exact_codes_from_codebook(values, scales, factor)
    assert torch.equal(restored, codes)
    assert torch.equal(decode(restored, scales.float()*factor, 1.).bfloat16().view(torch.uint8), values.view(torch.uint8))


def test_inexact_value_is_rejected_without_refitting_scale():
    scales = torch.ones(1, 1, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='outside'):
        exact_codes_from_codebook(torch.full((1, 16), .625, dtype=torch.bfloat16), scales, torch.tensor(1.))

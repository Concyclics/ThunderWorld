import pytest
import torch

from thunderworld_fp4.cosmos.settings import RunConfig


def test_selector_config_contract():
    config = RunConfig(activation_score='represented_gain')
    assert RunConfig.from_dict(config.to_dict()) == config
    assert RunConfig().activation_score == 'residual_energy'
    with pytest.raises(ValueError):
        RunConfig(activation_score='represented_gain', share_pack=True)
    with pytest.raises(ValueError):
        RunConfig(activation_score='unknown')
    with pytest.raises(ValueError):
        RunConfig(activation_score='represented_gain', precision='bf16', attention_backend='bf16')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 GPU')
@pytest.mark.parametrize('m,k,zero', [(129, 2048, False), (416, 9216, False), (129, 2048, True)])
@torch.inference_mode()
def test_gain_selector_independent_decode_and_current_graph(m, k, zero):
    from thunderworld_fp4._sparse_pack import pack_sparse
    from thunderworld_fp4._calibrated import pack
    from thunderworld_fp4.cosmos.calibration import decode
    torch.manual_seed(18118)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    if zero:
        x.zero_()
    d = torch.exp(torch.linspace(-1, 1, k, device='cuda'))
    sensitivity = torch.linspace(.1, 2., k, device='cuda')
    count = k // 128 * 3 // 4
    mask = torch.zeros(((m+127)//128, k//128), device='cuda', dtype=torch.bool)

    def check(packed):
        codes, scales, factor, mp = packed
        # Full packing has no online mask/selector. Decode its actual FP4
        # representation with an independent Torch lookup-table reference.
        full = pack(x, d, True, True, mask, base=.875, res=1.125)
        represented = decode(full[0], full[1], full[2])[:, :m]
        residual = (x.float()*d).bfloat16().float()-represented[0]
        remaining = residual-represented[1]
        gain = (residual.square()-remaining.square()).clamp_min(0)*sensitivity
        padded = torch.zeros((mask.shape[0]*128, k), device='cuda')
        padded[:m] = gain
        score = padded.reshape(mask.shape[0], 128, k//128, 128).sum((1, 3))
        indices = torch.argsort(score, dim=1, descending=True, stable=True)[:, :count]
        reference = torch.zeros_like(mask).scatter_(1, indices, True)
        assert torch.equal(mask, reference)
        assert torch.equal(codes[0], full[0][0])
        assert torch.equal(scales[0].view(torch.uint8), full[1][0].view(torch.uint8))
        selected = mask.repeat_interleave(128, 0)[:mp]
        assert torch.equal(codes[1][selected.repeat_interleave(64, 1)],
                           full[0][1][selected.repeat_interleave(64, 1)])
        assert torch.equal(scales[1].view(torch.uint8)[selected.repeat_interleave(8, 1)],
                           full[1][1].view(torch.uint8)[selected.repeat_interleave(8, 1)])
        assert torch.equal(factor, full[2])

    def invoke():
        return pack_sparse(x, d, sensitivity, mask, count, base=.875, res=1.125,
                           score='represented_gain')

    check(invoke())
    for _ in range(3):
        invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invoke()
    x.mul_(.25).add_(.375)
    graph.replay()
    check(captured)

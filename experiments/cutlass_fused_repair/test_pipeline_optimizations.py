"""GPU checks for unchanged-input semantics, ownership, and exact RMSNorm math."""
import pytest
import torch
from thunderworld_fp4.cosmos.norm_fusion import nemotron_norm
from thunderworld_fp4.cosmos.codec_graph import CodecGraphs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(1, 32, 128), (1, 4096, 128), (1, 256, 2048), (1, 3840, 2048)])
@pytest.mark.parametrize('scale', [0., 1., 64.])
@torch.inference_mode()
def test_norm_exact(shape, scale):
    torch.manual_seed(891)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale
    w = torch.randn(shape[-1], device='cuda', dtype=torch.bfloat16)
    xf = x.float()
    ref = (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6))).to(x.dtype)
    assert torch.equal(nemotron_norm(x, w, 1e-6), ref)


class CausalCodec(torch.nn.Module):
    """A tiny causal loop whose Python cache must reset each request."""
    def _encode(self, x):
        state = torch.zeros_like(x[0])
        rows = []
        for frame in x:
            state = state * .5 + frame
            rows.append(state)
        return torch.stack(rows)

    def _decode(self, x, return_dict=True):
        from diffusers.models.autoencoders.vae import DecoderOutput
        result = self._encode(x).square()
        return DecoderOutput(sample=result) if return_dict else (result,)


@torch.inference_mode()
def test_codec_current_inputs_ownership_and_signature_guard():
    # Decode uses its own implementation so it does not nest encoder captures.
    vae = CausalCodec().eval().cuda()
    encode = vae._encode
    vae._decode = lambda x, return_dict=False: (encode(x).square(),)
    graph = CodecGraphs(vae)
    x = torch.randn(8, 128, device='cuda')
    previous = None
    for kind in ('encode', 'decode'):
        for delta in (0., 3., -2.):
            current = x + delta
            expected = encode(current)
            if kind == 'decode': expected = expected.square()
            saved = previous.clone() if previous is not None else None
            actual = graph.run(kind, current)
            assert torch.equal(actual, expected)
            if saved is not None: assert torch.equal(previous, saved)
            previous = actual
    graph.warming = False
    with pytest.raises(RuntimeError, match='Unwarmed'):
        graph.run('encode', torch.randn(7, 128, device='cuda'))
    vae.use_tiling = True
    with pytest.raises(RuntimeError, match='Unwarmed'):
        graph.run('encode', x)

@pytest.mark.parametrize('tokens,heads,dim', [(37,16,128),(3840,16,128),(3840,8,128)])
@torch.inference_mode()
def test_pointwise_exact(tokens,heads,dim):
    from thunderworld_fp4.cosmos.pipeline_tuning import fused_rope, relu2
    from thunderworld_fp4.cosmos.ar_cache import rotate
    torch.manual_seed(891)
    x=torch.randn(tokens,heads,dim,device='cuda',dtype=torch.bfloat16)
    cos=torch.randn(tokens,1,dim,device='cuda',dtype=torch.bfloat16)
    sin=torch.randn_like(cos)
    assert torch.equal(fused_rope(x,cos,sin), x*cos+rotate(x)*sin)
    assert torch.equal(relu2(x),torch.relu(x).square())

@pytest.mark.parametrize('shape', [(33,2048),(3840,8192)])
@pytest.mark.parametrize('scale',[0.,1.,64.])
@torch.inference_mode()
def test_calibrated_factor_exact(shape,scale):
    from thunderworld_fp4.cosmos.calibrated_fusion import activate
    from thunderworld_fp4._linear_kernels import scaled_amax, finish_amax
    import triton
    torch.manual_seed(891)
    x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)*scale
    d=torch.randn(shape[-1],device='cuda',dtype=torch.float32)*3
    expected=torch.relu(x).square()
    partial=torch.empty((triton.cdiv(x.numel(),4096),),device='cuda');ref=torch.empty((),device='cuda')
    scaled_amax[(partial.numel(),)](expected,d,partial,*shape,4096,num_warps=4)
    finish_amax[(1,)](partial,ref,partial.numel(),triton.next_power_of_2(partial.numel()),num_warps=4)
    y,factor=activate(x,d)
    assert torch.equal(y,expected)
    assert torch.equal(factor,ref)

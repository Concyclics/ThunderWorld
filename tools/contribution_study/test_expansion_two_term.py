"""Qualify asymmetric two-product FP4 GEMM and streaming attention variants."""
import math
import pytest
import torch

from expansion_kernels import attention, decode, linear, pack_attention, pack_linear, term_mask


def probability_planes(value, planes):
    """Torch E4M3/E2M1 encode/decode, independent of the native packing kernel."""
    factor = (value.abs().amax(-1, keepdim=True)/1536).clamp_min(1e-12)
    residual = value/factor
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=value.device)
    code = torch.arange(8, device=value.device)
    rank = (code % 2)*8 + code  # nearest, ties to even
    result = []
    for _ in range(planes):
        groups = residual.reshape(*residual.shape[:-1], 8, 16)
        scale = (groups.abs().amax(-1)*(1/6)).clamp_min(1/512).to(torch.float8_e4m3fn).float()
        normalized = groups/scale[..., None]
        distances = (normalized.abs()[..., None]-levels).abs()
        tied = distances == distances.amin(-1, keepdim=True)
        index = torch.where(tied, rank, 100).argmin(-1)
        represented = (levels[index]*normalized.sign()*scale[..., None]).reshape_as(residual)
        result.append(represented*factor)
        residual = residual-represented
    return result


def decoded_attention_reference(q, k, v, wp, ap):
    packed, mp, npad = pack_attention(q, k, v, wp, ap)
    qc, qs, qf, kc, ks, kf, vc, vs, vf = packed
    m, n, h, hk = q.shape[1], k.shape[1], q.shape[2], k.shape[2]
    qd = decode(qc, qs, qf[None, :, None]).reshape(ap, h, mp, 128)
    kd = decode(kc, ks, kf[None, :, None]).reshape(wp, hk, npad, 128)
    nt = npad//128
    vc = vc.reshape(wp, hk, nt, 64, 128).transpose(-1, -2).contiguous()
    vs = vs.reshape(wp, hk, nt, 8, 128).transpose(-1, -2).contiguous()
    vd = decode(vc, vs, vf.reshape(1, hk, nt, 128, 1))
    output = torch.empty_like(q, dtype=torch.float32)
    for head in range(h):
        kh = head//(h//hk)
        scores = sum(qd[a, head, :m] @ kd[w, kh, :n].T for w in range(wp) for a in range(ap))/math.sqrt(128)
        p = torch.exp(scores-scores.amax(-1, keepdim=True))
        total = torch.zeros((m, 128), device=q.device)
        for block in range(nt):
            pt = torch.nn.functional.pad(p[:, block*128:min((block+1)*128, n)],
                                         (0, max(0, (block+1)*128-n)))
            pd = probability_planes(pt, ap)
            total += sum(pd[a] @ vd[w, kh, block].T for w in range(wp) for a in range(ap))
        output[0, :, head] = total/p.sum(-1, keepdim=True)
    return output


@pytest.mark.parametrize('m', [129, 416])
@pytest.mark.parametrize('wp,ap', [(1, 2), (2, 1)])
@torch.inference_mode()
def test_two_term_linear_decoded_reference_and_graph(m, wp, ap):
    torch.manual_seed(9222300+m+wp)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(m, 256, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
    wc, ws, wg, _ = pack_linear(w, wp)
    ac, asc, ag, _ = pack_linear(x, ap)
    mask = term_mask(wp, ap)
    assert mask == (3 if ap == 2 else 17) and mask.bit_count() == 2
    a, b = decode(ac, asc, ag)[:, :m], decode(wc, ws, wg)
    ref = sum(a[j] @ b[i].T for i in range(wp) for j in range(ap))
    y = linear(x, (wc, ws, wg), wp, ap, mask)
    torch.testing.assert_close(y.float(), ref, rtol=.004, atol=.004)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = linear(x, (wc, ws, wg), wp, ap, mask)
    x.mul_(.5).add_(.125)
    graph.replay()
    assert torch.equal(output, linear(x, (wc, ws, wg), wp, ap, mask))
    x.zero_()
    graph.replay()
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize('m,n,h,hk', [(129, 145, 4, 2), (416, 257, 2, 2)])
@pytest.mark.parametrize('wp,ap', [(1, 2), (2, 1)])
@torch.inference_mode()
def test_two_term_attention_decoded_reference_gqa_tails(m, n, h, hk, wp, ap):
    torch.manual_seed(9222400+m+wp)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn((1, m, h, 128), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((1, n, hk, 128), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = term_mask(wp, ap)
    actual = attention(q, k, v, wp, ap, mask)
    reference = decoded_attention_reference(q, k, v, wp, ap)
    error = torch.linalg.vector_norm(actual.float()-reference)/torch.linalg.vector_norm(reference)
    assert error < .004, float(error)
    if ap == 2:
        from thunderworld_fp4 import RepairConfig
        from thunderworld_fp4.runtime import _forward
        cfg = RepairConfig(coverage=1., terms=2, query_tile=32, splits=2, transpose_v=True, fast_p=True)
        assert torch.equal(actual, _forward(q, k, v, cfg)[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attention(q, k, v, wp, ap, mask)
    q.mul_(.75); k.add_(.125); v.mul_(.5)
    graph.replay()
    assert torch.equal(output, attention(q, k, v, wp, ap, mask))
    q.zero_(); k.zero_(); v.zero_()
    graph.replay()
    assert torch.count_nonzero(output) == 0

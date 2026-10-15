"""Offline export of the pinned official dynamic fake-quant block scales.

ModelOpt's Python real-export division and its Triton fake-quant division can
choose different E4M3 codes at a tie. Import the exact official helper used in
the frozen calibration. No scale is fitted from decoded output values.
"""
import torch
import triton
import triton.language as tl
from modelopt.torch.kernels.quantization.common.nvfp4_quant import fp8_quantize_scale


@triton.jit
def _scales(W, G, S, N: tl.constexpr, K: tl.constexpr):
    nr = tl.program_id(0)*16+tl.arange(0, 16)
    kc = tl.program_id(1)*64+tl.arange(0, 64)
    w = tl.load(W+nr[:, None]*K+kc[None, :], (nr[:, None] < N) & (kc[None, :] < K), 0).to(tl.float32)
    group = tl.reshape(tl.abs(w), (16, 4, 16))
    amax = tl.max(group, 2)
    g = tl.load(G).to(tl.float32)
    g = tl.where(g > 0., g, 1.e-12)
    effective = fp8_quantize_scale(amax, g)
    # This inverse only recovers an already selected E4M3 value; its rounding
    # margin is far wider than FP32 multiplication/division error.
    code = (effective/g).to(tl.float8e4nv)
    column = tl.program_id(1)*4+tl.arange(0, 4)
    tl.store(S+nr[:, None]*(K//16)+column[None, :], code,
             (nr[:, None] < N) & (column[None, :] < K//16))


def export_scales(weight, global_scale):
    if weight.ndim != 2 or weight.shape[1] % 16 or not weight.is_cuda:
        raise ValueError('Expected contiguous CUDA N/K16 weight')
    weight = weight.contiguous()
    n, k = weight.shape
    scales = torch.empty((n, k//16), device=weight.device, dtype=torch.float8_e4m3fn)
    _scales[(triton.cdiv(n, 16), triton.cdiv(k, 64))](weight, global_scale, scales, n, k, num_warps=4)
    return scales

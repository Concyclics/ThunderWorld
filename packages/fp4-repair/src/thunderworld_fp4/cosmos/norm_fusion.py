"""Reduction-preserving pointwise fusion for Cosmos3 Edge's Nemotron RMSNorm.

Keep PyTorch's FP32 mean reduction to preserve its summation order. Fuse the
surrounding casts and pointwise math, with FMA disabled. This is inference only.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def norm_square(X, S, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(S + i, x * x, i < N)


@triton.jit
def norm_finish(X, V, W, Y, N: tl.constexpr, D: tl.constexpr,
                EPS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    v = tl.load(V + i // D, i < N, 0)
    w = tl.load(W + i % D).to(tl.float32)
    normalized = x * libdevice.rsqrt(v + EPS)
    tl.store(Y + i, w * normalized, i < N)


def nemotron_norm(x, weight, eps):
    if torch.is_grad_enabled():
        raise RuntimeError('Fused Nemotron RMSNorm requires inference mode')
    if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('Fused Nemotron RMSNorm requires contiguous CUDA tensors')
    if x.dtype != torch.bfloat16 or x.shape[-1] != weight.numel():
        raise ValueError('Fused Nemotron RMSNorm supports BF16 inputs matching the weight')
    squared = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    norm_square[(triton.cdiv(x.numel(), 512),)](x, squared, x.numel(), 512, enable_fp_fusion=False)
    variance = squared.mean(-1, keepdim=True)
    norm_finish[(triton.cdiv(x.numel(), 512),)](
        x, variance, weight, y, x.numel(), x.shape[-1], eps, 512, enable_fp_fusion=False)
    return y


def install_norm_fusion(transformer):
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm
    names = []
    for name, module in transformer.named_modules():
        if type(module) is Cosmos3NemotronRMSNorm:
            def forward(x, _module=module):
                if _module.training:
                    raise RuntimeError('Fused Nemotron RMSNorm requires eval mode')
                return nemotron_norm(x, _module.weight, _module.eps)
            module.forward = forward
            names.append(name)
    if not names:
        raise ValueError('This model has no supported Cosmos3 Nemotron RMSNorm modules')
    return names

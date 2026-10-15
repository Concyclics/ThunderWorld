"""Exact activation quantization with CUTLASS base-scale stores in the producer.

Keep row-major scales for sparse repair and shared native consumers. The added
stores replace the separate base-scale read/permutation launch; no MMA changes.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from .._linear_kernels import scaled_amax, finish_amax
from .._quantize import quantize_linear
from .._calibrated import quantize_calibrated
from .._sparse_pack import choose, residual_only


@dataclass
class CutlassPacked:
    native: tuple
    scales: torch.Tensor
    alpha: torch.Tensor


@triton.jit
def scale_offset(row, col, COLS: tl.constexpr):
    return (row // 128) * (128 * COLS) + (col // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + col % 4


@triton.jit
def pack_kernel(X, D, F, WG, MASK, SENS, C, S, CS, ALPHA, E,
                M: tl.constexpr, K: tl.constexpr, MP: tl.constexpr,
                REPAIR: tl.constexpr, FULL: tl.constexpr, SPARSE: tl.constexpr,
                CALIBRATED: tl.constexpr, BASE: tl.constexpr, RES: tl.constexpr):
    rb = tl.program_id(0); kb = tl.program_id(1)
    mi = rb * 16 + tl.arange(0, 16); ki = kb * 128 + tl.arange(0, 128)
    x = tl.load(X + mi[:, None] * K + ki[None, :], mi[:, None] < M, 0).to(tl.float32)
    d = tl.load(D + ki); x = (x * d[None, :]).to(tl.bfloat16).to(tl.float32)
    f = tl.load(F)
    if SPARSE:
        p, s, r, rs, _ = quantize_calibrated(x, f, 16, 128, False, BASE, 1., 1.)
        code = tl.reshape(tl.join(p & 15, p >> 4), (16, 128)).to(tl.int32)
        mag = code & 7
        value = tl.where(mag < 4, mag * .5, tl.where(mag < 6, mag - 2., 2. * mag - 8.))
        value = tl.where(code >= 8, -value, value)
        decoded = tl.reshape(tl.reshape(value, (16, 8, 16)) * s.to(tl.float32)[:, :, None], (16, 128)) * f
        residual = x - decoded
        sensitivity = tl.load(SENS + ki)
        score = tl.sum(tl.sum(residual * residual * sensitivity[None, :], 1), 0)
        tl.store(E + rb * (K // 128) + kb, score)
    else:
        if REPAIR and not FULL:
            active = tl.load(MASK + (rb // 8) * (K // 128) + kb)
        else:
            active = REPAIR
        if active:
            if CALIBRATED:
                p, s, r, rs, _ = quantize_calibrated(x, f, 16, 128, True, BASE, RES, 1.)
            else:
                p, s, r, rs, _ = quantize_linear(x, f, 16, 128, True)
        else:
            if CALIBRATED:
                p, s, r, rs, _ = quantize_calibrated(x, f, 16, 128, False, BASE, RES, 1.)
            else:
                p, s, r, rs, _ = quantize_linear(x, f, 16, 128, False)
            r = tl.full((16, 64), 0, tl.uint8); rs = tl.full((16, 8), 0, tl.float8e4nv)
    ci = mi[:, None] * (K // 2) + kb * 64 + tl.arange(0, 64)[None, :]
    col = kb * 8 + tl.arange(0, 8)
    si = mi[:, None] * (K // 16) + col[None, :]
    tl.store(C + ci, p, mi[:, None] < MP)
    tl.store(S + si, s, mi[:, None] < MP)
    tl.store(CS + scale_offset(mi[:, None], col[None, :], K // 16), s, mi[:, None] < MP)
    if REPAIR and not SPARSE:
        tl.store(C + MP * (K // 2) + ci, r, mi[:, None] < MP)
        tl.store(S + MP * (K // 16) + si, rs, mi[:, None] < MP)
    # CUTLASS TMA reads M128 scale tiles, whereas native packing pads to M16.
    # Give only the last row producer responsibility for the extra zero padding.
    if MP % 128 != 0:
        if rb == tl.cdiv(M, 16) - 1:
            tail = MP + tl.arange(0, 128)
            tl.store(CS + scale_offset(tail[:, None], col[None, :], K // 16), 0.,
                     tail[:, None] < tl.cdiv(MP, 128) * 128)
    if rb == 0 and kb == 0:
        tl.store(ALPHA, f * tl.load(WG))


def pack_input(x, op, *, sensitivity=None, base=1., res=1., factor=None,energy=1.):
    """Mirror the native/calibrated/sparse producer without changing its factor."""
    m, k = x.shape; mp = triton.cdiv(m, 16) * 16
    sparse = sensitivity is not None and 0 < op.na < k // 128
    if factor is None:
        partial = torch.empty((triton.cdiv(m * k, 4096),), device=x.device, dtype=torch.float32)
        factor = torch.empty((), device=x.device, dtype=torch.float32)
        scaled_amax[(partial.numel(),)](x, op.d, partial, m, k, 4096, num_warps=4)
        finish_amax[(1,)](partial, factor, partial.numel(), triton.next_power_of_2(partial.numel()), num_warps=4)
    planes = 2 if op.na else 1
    codes = torch.empty((planes, mp, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((planes, mp, k // 16), device=x.device, dtype=torch.float8_e4m3fn)
    cutlass_scales = torch.empty((triton.cdiv(m, 128) * 128, k // 16), device=x.device, dtype=scales.dtype)
    alpha = torch.empty((), device=x.device, dtype=torch.float32)
    summary = torch.empty((triton.cdiv(m, 16), k // 128), device=x.device, dtype=torch.float32) if sparse else factor
    pack_kernel[(triton.cdiv(m, 16), k // 128)](
        x, op.d, factor, op.wg, op.am, sensitivity if sparse else op.d,
        codes, scales, cutlass_scales, alpha, summary, m, k, mp,
        op.na > 0, op.na == k // 128, sparse, (base, res) != (1., 1.), base, res,
        num_warps=4, enable_fp_fusion=not sparse)
    if sparse:
        choose[(triton.cdiv(m, 128),)](summary, op.am, m, k, triton.next_power_of_2(k // 128), op.na, energy, num_warps=4)
        residual_only[(triton.cdiv(m, 16), k // 128)](
            x, op.d, factor, op.am, codes, scales, m, k, mp, base, res, num_warps=4)
    return CutlassPacked((codes, scales, factor, mp), cutlass_scales, alpha)

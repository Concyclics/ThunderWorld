"""Quota-sized A1 storage; identical quantization and K-order accumulation.

Only the representation changes. A0 retains its existing row-major layout.
Each M128 row stores at most ``na`` residual K128 tiles, in ascending K order.
The current GPU mask creates the current address map on every call/replay.
"""
import torch
import triton
import triton.language as tl
from .cutlass.packing import CutlassPacked, pack_kernel
from ._linear_kernels import scaled_amax, finish_amax
from ._sparse_pack import choose
from ._calibrated import quantize_calibrated
from ._quantize import quantize_linear


def compact_bytes(m, k, na):
    mb, mp = triton.cdiv(m, 128), triton.cdiv(m, 16) * 16
    dense = mp * k * 9 // 16 if na else 0
    compact = mb * na * 128 * 72 + mb * (k // 128) * 4 if na else 0
    use = 0 < na < k // 128 and compact < dense
    return dict(enabled=use, dense_residual_bytes=dense,
                residual_bytes=mb * na * 128 * 72 if use else dense,
                address_map_bytes=mb * (k // 128) * 4 if use else 0)


@triton.jit
def address_map(MASK, MAP, KT: tl.constexpr, NA: tl.constexpr, BT: tl.constexpr):
    mb = tl.program_id(0); kb = tl.arange(0, BT)
    selected = tl.load(MASK + mb * KT + kb, kb < KT, 0).to(tl.int32)
    rank = tl.cumsum(selected) - 1
    # The prepared selector's integer quota is an API invariant. Check under
    # TRITON_DEBUG in qualification; normal inference has no host readback.
    tl.device_assert(tl.sum(selected, 0) <= NA, "A support exceeds prepared quota")
    slot = tl.where(selected != 0, mb * NA + rank, -1)
    tl.store(MAP + mb * KT + kb, slot, kb < KT)


@triton.jit
def compact_residual(X, D, F, MAP, C, S,
                     M: tl.constexpr, K: tl.constexpr,
                     CALIBRATED: tl.constexpr, BASE: tl.constexpr, RES: tl.constexpr):
    rb = tl.program_id(0); kb = tl.program_id(1)
    slot = tl.load(MAP + (rb // 8) * (K // 128) + kb)
    if slot >= 0:
        rows = rb * 16 + tl.arange(0, 16)
        cols = kb * 128 + tl.arange(0, 128)
        x = tl.load(X + rows[:, None] * K + cols[None, :], rows[:, None] < M, 0).to(tl.float32)
        d = tl.load(D + cols)
        x = (x * d[None, :]).to(tl.bfloat16).to(tl.float32)
        if CALIBRATED:
            _, _, r, rs, _ = quantize_calibrated(x, tl.load(F), 16, 128, True, BASE, RES, 1.)
        else:
            _, _, r, rs, _ = quantize_linear(x, tl.load(F), 16, 128, True)
        local_rows = (rb % 8) * 16 + tl.arange(0, 16)
        tl.store(C + slot * 8192 + local_rows[:, None] * 64 + tl.arange(0, 64)[None, :], r)
        tl.store(S + slot * 1024 + local_rows[:, None] * 8 + tl.arange(0, 8)[None, :], rs)


def pack_compact(x, op, *, sensitivity=None, base=1., res=1., factor=None, energy=1.):
    m, k = x.shape
    mb, mp = triton.cdiv(m, 128), triton.cdiv(m, 16) * 16
    sparse = sensitivity is not None and 0 < op.na < k // 128
    if not compact_bytes(m, k, op.na)['enabled']:
        raise ValueError('Use the qualified dense layout when compaction cannot save storage')
    if factor is None:
        partial = torch.empty((triton.cdiv(m * k, 4096),), device=x.device, dtype=torch.float32)
        factor = torch.empty((), device=x.device, dtype=torch.float32)
        scaled_amax[(partial.numel(),)](x, op.d, partial, m, k, 4096, num_warps=4)
        finish_amax[(1,)](partial, factor, partial.numel(), triton.next_power_of_2(partial.numel()), num_warps=4)
    codes = torch.empty((1, mp, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((1, mp, k // 16), device=x.device, dtype=torch.float8_e4m3fn)
    cs = torch.empty((mb * 128, k // 16), device=x.device, dtype=scales.dtype)
    alpha = torch.empty((), device=x.device, dtype=torch.float32)
    summary = torch.empty((triton.cdiv(m, 16), k // 128), device=x.device, dtype=torch.float32) if sparse else factor
    pack_kernel[(triton.cdiv(m, 16), k // 128)](
        x, op.d, factor, op.wg, op.am, sensitivity if sparse else op.d,
        codes, scales, cs, alpha, summary, m, k, mp,
        False, False, sparse, (base, res) != (1., 1.), base, res,
        num_warps=4, enable_fp_fusion=not sparse)
    if sparse:
        choose[(mb,)](summary, op.am, m, k, triton.next_power_of_2(k // 128), op.na, energy, num_warps=4)
    locations = torch.empty((mb, k // 128), device=x.device, dtype=torch.int32)
    address_map[(mb,)](op.am, locations, k // 128, op.na, triton.next_power_of_2(k // 128), num_warps=4)
    rc = torch.empty((mb * op.na, 128, 64), device=x.device, dtype=torch.uint8)
    rs = torch.empty((mb * op.na, 128, 8), device=x.device, dtype=scales.dtype)
    compact_residual[(mb * 8, k // 128)](
        x, op.d, factor, locations, rc, rs, m, k,
        sparse or (base, res) != (1., 1.), base, res, num_warps=4)
    result = CutlassPacked((codes, scales, factor, mp), cs, alpha)
    result.compact_repair = (rc, rs, locations)
    return result

"""K16 NVFP4 base/residual quantizers; FP32 outer factors."""
import triton
import triton.language as tl


@triton.jit
def quantize(x, R: tl.constexpr, C: tl.constexpr, REPAIR: tl.constexpr):
    # A common outer factor for base and residual permits accumulator fusion.
    factor = tl.maximum(tl.max(tl.abs(x), 1) / 1536.0, 1.0e-12)
    z = x / factor[:, None]
    z3 = tl.reshape(z, (R, C // 16, 16))
    scale = tl.maximum(tl.max(tl.abs(z3), 2) / 6.0, 0.001953125).to(tl.float8e4nv)
    normalized = tl.reshape(z3 / scale.to(tl.float32)[:, :, None], (R, C))
    pairs = tl.reshape(normalized, (R, C // 2, 2))
    lo, hi = tl.split(pairs)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    if REPAIR:
        code = tl.reshape(tl.join(packed & 15, packed >> 4), (R, C)).to(tl.int32)
        mag = code & 7
        val = tl.where(mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2., 2. * mag - 8.))
        val = tl.where(code >= 8, -val, val)
        residual = z3 - tl.reshape(val, (R, C // 16, 16)) * scale.to(tl.float32)[:, :, None]
        rs = tl.maximum(tl.max(tl.abs(residual), 2) / 6.0, 0.001953125).to(tl.float8e4nv)
        rn = tl.reshape(residual / rs.to(tl.float32)[:, :, None], (R, C // 2, 2))
        rlo, rhi = tl.split(rn)
        rp = tl.inline_asm_elementwise(
            "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }",
            constraints="=r,f,f", args=[rlo, rhi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    else:
        rp = packed
        rs = scale
    return packed, scale, rp, rs, factor


@triton.jit
def reciprocal(x):
    # All callers clamp scales/factors positive and normal before reaching here.
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def quantize_fast(x, R: tl.constexpr, C: tl.constexpr, REPAIR: tl.constexpr):
    # A common outer factor for base and residual permits accumulator fusion.
    factor = tl.maximum(tl.max(tl.abs(x), 1) * (1.0 / 1536.0), 1.0e-12)
    z = x * reciprocal(factor)[:, None]
    z3 = tl.reshape(z, (R, C // 16, 16))
    scale = tl.maximum(tl.max(tl.abs(z3), 2) * (1.0 / 6.0), 0.001953125).to(tl.float8e4nv)
    normalized = tl.reshape(z3 * reciprocal(scale.to(tl.float32))[:, :, None], (R, C))
    pairs = tl.reshape(normalized, (R, C // 2, 2))
    lo, hi = tl.split(pairs)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    if REPAIR:
        code = tl.reshape(tl.join(packed & 15, packed >> 4), (R, C)).to(tl.int32)
        mag = code & 7
        val = tl.where(mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2., 2. * mag - 8.))
        val = tl.where(code >= 8, -val, val)
        residual = z3 - tl.reshape(val, (R, C // 16, 16)) * scale.to(tl.float32)[:, :, None]
        rs = tl.maximum(tl.max(tl.abs(residual), 2) * (1.0 / 6.0), 0.001953125).to(tl.float8e4nv)
        rn = tl.reshape(residual * reciprocal(rs.to(tl.float32))[:, :, None], (R, C // 2, 2))
        rlo, rhi = tl.split(rn)
        rp = tl.inline_asm_elementwise(
            "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }",
            constraints="=r,f,f", args=[rlo, rhi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    else:
        rp = packed
        rs = scale
    return packed, scale, rp, rs, factor


@triton.jit
def quantize_p(x, R:tl.constexpr, C:tl.constexpr, REPAIR:tl.constexpr, FAST:tl.constexpr):
    if FAST:return quantize_fast(x,R,C,REPAIR)
    else:return quantize(x,R,C,REPAIR)


@triton.jit
def quantize_linear(x, factor, R: tl.constexpr, C: tl.constexpr, REPAIR: tl.constexpr):
    # A common outer factor for base and residual permits accumulator fusion.
    z = x / factor
    z3 = tl.reshape(z, (R, C // 16, 16))
    scale = tl.maximum(tl.max(tl.abs(z3), 2) / 6.0, 0.001953125).to(tl.float8e4nv)
    normalized = tl.reshape(z3 / scale.to(tl.float32)[:, :, None], (R, C))
    pairs = tl.reshape(normalized, (R, C // 2, 2))
    lo, hi = tl.split(pairs)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }",
        constraints="=r,f,f", args=[lo, hi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    if REPAIR:
        code = tl.reshape(tl.join(packed & 15, packed >> 4), (R, C)).to(tl.int32)
        mag = code & 7
        val = tl.where(mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2., 2. * mag - 8.))
        val = tl.where(code >= 8, -val, val)
        residual = z3 - tl.reshape(val, (R, C // 16, 16)) * scale.to(tl.float32)[:, :, None]
        rs = tl.maximum(tl.max(tl.abs(residual), 2) / 6.0, 0.001953125).to(tl.float8e4nv)
        rn = tl.reshape(residual / rs.to(tl.float32)[:, :, None], (R, C // 2, 2))
        rlo, rhi = tl.split(rn)
        rp = tl.inline_asm_elementwise(
            "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }",
            constraints="=r,f,f", args=[rlo, rhi], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)
    else:
        rp = packed
        rs = scale
    return packed, scale, rp, rs, factor

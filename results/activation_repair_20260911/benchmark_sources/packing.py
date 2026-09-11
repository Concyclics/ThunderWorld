"""Experimental scale calibration and direct interleaved A0/A1 packing.

BASE/RES scale the block max before actual E4M3 rounding and E2M1 conversion.
Residuals are always recomputed from the rounded base. Shared outer factor is
unchanged; this preserves one FP32 accumulation scale. No runtime grid search.
"""
import torch,triton
import triton.language as tl
from thunderworld_fp4._linear_kernels import scaled_amax,finish_amax
@triton.jit
def quantize_calibrated(x, factor, R: tl.constexpr, C: tl.constexpr, REPAIR: tl.constexpr, BASE:tl.constexpr, RES:tl.constexpr):
    # A common outer factor for base and residual permits accumulator fusion.
    z = x / factor
    z3 = tl.reshape(z, (R, C // 16, 16))
    scale = tl.minimum(tl.maximum(tl.max(tl.abs(z3), 2) * BASE / 6.0, 0.001953125),448.).to(tl.float8e4nv)
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
        rs = tl.minimum(tl.maximum(tl.max(tl.abs(residual), 2) * RES / 6.0, 0.001953125),448.).to(tl.float8e4nv)
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
def pack_kernel(X,D,F,U,C,S,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,REPAIR:tl.constexpr,FULL:tl.constexpr,PAIRED:tl.constexpr,BASE:tl.constexpr,RES:tl.constexpr):
 mi=tl.program_id(0)*16+tl.arange(0,16);kb=tl.program_id(1);ki=kb*128+tl.arange(0,128)
 x=tl.load(X+mi[:,None]*K+ki[None,:],mi[:,None]<M,0).to(tl.float32);d=tl.load(D+ki)
 x=(x*d[None,:]).to(tl.bfloat16).to(tl.float32);f=tl.load(F)
 if REPAIR and not FULL:
  active=tl.load(U+(tl.program_id(0)//8)*(K//128)+kb)
  if active:p,s,r,rs,_=quantize_calibrated(x,f,16,128,True,BASE,RES)
  else:
   p,s,r,rs,_=quantize_calibrated(x,f,16,128,False,BASE,RES)
   r=tl.full((16,64),0,tl.uint8);rs=tl.full((16,8),0,tl.float8e4nv)
 else:p,s,r,rs,_=quantize_calibrated(x,f,16,128,REPAIR,BASE,RES)
 if PAIRED:
  ci=mi[:,None]*K+kb*128+tl.arange(0,64)[None,:];si=mi[:,None]*(K//8)+kb*16+tl.arange(0,8)[None,:];co=64;so=8
 else:
  ci=mi[:,None]*(K//2)+kb*64+tl.arange(0,64)[None,:];si=mi[:,None]*(K//16)+kb*8+tl.arange(0,8)[None,:];co=MP*(K//2);so=MP*(K//16)
 tl.store(C+ci,p,mi[:,None]<MP);tl.store(S+si,s,mi[:,None]<MP)
 if REPAIR:
  tl.store(C+co+ci,r,mi[:,None]<MP);tl.store(S+so+si,rs,mi[:,None]<MP)

def pack(x,d,repair,full,union,*,paired=False,base=1.,res=1.):
 if paired and not repair:raise ValueError('Paired format requires residual plane')
 if not (0<base<=1 and 0<res<=1.5):raise ValueError('Unsupported calibration scale')
 m,k=x.shape;mp=triton.cdiv(m,16)*16
 partial=torch.empty((triton.cdiv(m*k,4096),),device=x.device,dtype=torch.float32);f=torch.empty((),device=x.device,dtype=torch.float32)
 scaled_amax[(partial.numel(),)](x,d,partial,m,k,4096,num_warps=4)
 finish_amax[(1,)](partial,f,partial.numel(),triton.next_power_of_2(partial.numel()),num_warps=4)
 planes=2 if repair else 1
 c=torch.empty((planes,mp,k//2),device=x.device,dtype=torch.uint8);s=torch.empty((planes,mp,k//16),device=x.device,dtype=torch.float8_e4m3fn)
 pack_kernel[(triton.cdiv(m,16),k//128)](x,d,f,union,c,s,m,k,mp,repair,full,paired,base,res,num_warps=4)
 return c,s,f,mp

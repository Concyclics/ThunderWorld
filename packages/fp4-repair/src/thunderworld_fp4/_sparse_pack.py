"""Current-residual weighted A selection. Only selected residual codes are written.

The base pass emits 16-row residual summaries; no full residual-code materialization
or selector GEMM. A second pass re-encodes selected blocks with the same factor.
"""
import torch
import triton
import triton.language as tl
from ._linear_kernels import scaled_amax,finish_amax
from ._calibrated import quantize_calibrated


@triton.jit
def base_summary(X,D,F,SENS,C,S,E,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,BASE:tl.constexpr,
                 RES:tl.constexpr=1.,GAIN_SCORE:tl.constexpr=False):
    rb=tl.program_id(0);kb=tl.program_id(1)
    mi=rb*16+tl.arange(0,16);ki=kb*128+tl.arange(0,128)
    x=tl.load(X+mi[:,None]*K+ki[None,:],mi[:,None]<M,0).to(tl.float32)
    d=tl.load(D+ki);x=(x*d[None,:]).to(tl.bfloat16).to(tl.float32)
    f=tl.load(F)
    p,s,r,rs,_=quantize_calibrated(x,f,16,128,GAIN_SCORE,BASE,RES,1.)
    code=tl.reshape(tl.join(p&15,p>>4),(16,128)).to(tl.int32)
    mag=code&7
    value=tl.where(mag<4,mag*.5,tl.where(mag<6,mag-2.,2.*mag-8.))
    value=tl.where(code>=8,-value,value)
    decoded=tl.reshape(tl.reshape(value,(16,8,16))*s.to(tl.float32)[:,:,None],(16,128))*f
    residual=x-decoded
    benefit=residual*residual
    if GAIN_SCORE:
        # Evaluate the represented residual, not a hypothetical exact repair.
        # Codes stay in registers; only selected residual codes are written.
        rc=tl.reshape(tl.join(r&15,r>>4),(16,128)).to(tl.int32)
        rm=rc&7
        rv=tl.where(rm<4,rm*.5,tl.where(rm<6,rm-2.,2.*rm-8.))
        rv=tl.where(rc>=8,-rv,rv)
        represented=tl.reshape(tl.reshape(rv,(16,8,16))*rs.to(tl.float32)[:,:,None],(16,128))*f
        remaining=residual-represented
        benefit=tl.maximum(benefit-remaining*remaining,0.)
    sensitivity=tl.load(SENS+ki)
    score=tl.sum(tl.sum(benefit*sensitivity[None,:],1),0)
    tl.store(E+rb*(K//128)+kb,score)
    tl.store(C+mi[:,None]*(K//2)+kb*64+tl.arange(0,64)[None,:],p,mi[:,None]<MP)
    tl.store(S+mi[:,None]*(K//16)+kb*8+tl.arange(0,8)[None,:],s,mi[:,None]<MP)


@triton.jit
def choose(E,MASK,M:tl.constexpr,K:tl.constexpr,BT:tl.constexpr,NA:tl.constexpr,ENERGY:tl.constexpr=1.):
    mb=tl.program_id(0);kt=tl.arange(0,BT)
    rows=mb*8+tl.arange(0,8)
    scores=tl.load(E+rows[:,None]*(K//128)+kt[None,:],(rows[:,None]<tl.cdiv(M,16))&(kt[None,:]<K//128),0.)
    energy=tl.where(kt<K//128,tl.sum(scores,0),-float('inf'))
    rank=tl.sum(((energy[None,:]>energy[:,None])|((energy[None,:]==energy[:,None])&(kt[None,:]<kt[:,None]))).to(tl.int32),1)
    active=rank<NA
    if ENERGY < 1.:
        # Stop at the smallest stable prefix explaining ENERGY of the current
        # residual sensitivity proxy, subject to the existing per-tile cap.
        # Zero residual needs no repair. No host decision or additional GEMM.
        values=tl.where(kt<K//128,energy,0.)
        before=tl.sum(tl.where(rank[None,:]<rank[:,None],values[None,:],0.),1)
        active=active & (before < ENERGY*tl.sum(values,0))
    tl.store(MASK+mb*(K//128)+kt,active,kt<K//128)


@triton.jit
def residual_only(X,D,F,MASK,C,S,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,BASE:tl.constexpr,RES:tl.constexpr):
    rb=tl.program_id(0);kb=tl.program_id(1)
    active=tl.load(MASK+(rb//8)*(K//128)+kb)
    if active:
        mi=rb*16+tl.arange(0,16);ki=kb*128+tl.arange(0,128)
        x=tl.load(X+mi[:,None]*K+ki[None,:],mi[:,None]<M,0).to(tl.float32)
        d=tl.load(D+ki);x=(x*d[None,:]).to(tl.bfloat16).to(tl.float32)
        _,_,r,rs,_=quantize_calibrated(x,tl.load(F),16,128,True,BASE,RES,1.)
        tl.store(C+MP*(K//2)+mi[:,None]*(K//2)+kb*64+tl.arange(0,64)[None,:],r,mi[:,None]<MP)
        tl.store(S+MP*(K//16)+mi[:,None]*(K//16)+kb*8+tl.arange(0,8)[None,:],rs,mi[:,None]<MP)


def pack_sparse(x,d,sensitivity,mask,count,*,base=1.,res=1.,factor=None,energy=1.,score='residual_energy'):
    if score not in ('residual_energy','represented_gain'):raise ValueError('Unknown sparse activation score')
    m,k=x.shape;mp=triton.cdiv(m,16)*16
    if not 0<count<k//128:raise ValueError('Sparse packing requires a partial integer quota')
    if factor is None:
        partial=torch.empty((triton.cdiv(m*k,4096),),device=x.device,dtype=torch.float32)
        factor=torch.empty((),device=x.device,dtype=torch.float32)
        scaled_amax[(partial.numel(),)](x,d,partial,m,k,4096,num_warps=4)
        finish_amax[(1,)](partial,factor,partial.numel(),triton.next_power_of_2(partial.numel()),num_warps=4)
    c=torch.empty((2,mp,k//2),device=x.device,dtype=torch.uint8)
    s=torch.empty((2,mp,k//16),device=x.device,dtype=torch.float8_e4m3fn)
    summary=torch.empty((triton.cdiv(m,16),k//128),device=x.device,dtype=torch.float32)
    base_summary[(triton.cdiv(m,16),k//128)](x,d,factor,sensitivity,c,s,summary,m,k,mp,base,res,
        score=='represented_gain',num_warps=4,enable_fp_fusion=False)
    choose[(triton.cdiv(m,128),)](summary,mask,m,k,triton.next_power_of_2(k//128),count,energy,num_warps=4)
    residual_only[(triton.cdiv(m,16),k//128)](x,d,factor,mask,c,s,m,k,mp,base,res,num_warps=4)
    return c,s,factor,mp

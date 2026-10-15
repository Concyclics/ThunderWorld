"""One-to-four nested NVFP4 residual planes, native SM120 quality experiments.

Every product uses dot_scaled(E2M1,E4M3) and a shared FP32 accumulator.
The outer factor stays fixed across residual planes, matching the existing
two-plane quantization. The representable-scale floor may cause convergence
to plateau; increasing the number of planes does not promise monotone quality.
"""
import torch
import triton
import triton.language as tl

from thunderworld_fp4._linear_kernels import scaled_amax, finish_amax
from thunderworld_fp4._quantize import reciprocal
from thunderworld_fp4._kernels import combine


def term_mask(weight_planes, activation_planes, dual=False):
    if not 1 <= weight_planes <= 4 or not 1 <= activation_planes <= 4:
        raise ValueError('Supported plane count: 1..4')
    if dual and (weight_planes, activation_planes) != (2, 2):
        raise ValueError('Three-term dual requires two planes on both sides')
    return sum(1 << (w*4+a) for w in range(weight_planes) for a in range(activation_planes)
               if not (dual and w==a==1))


@triton.jit
def component(z, R:tl.constexpr, C:tl.constexpr, FAST:tl.constexpr=False):
    groups=tl.reshape(z,(R,C//16,16))
    if FAST:scale=tl.maximum(tl.max(tl.abs(groups),2)*(1./6.),.001953125).to(tl.float8e4nv)
    else:scale=tl.maximum(tl.max(tl.abs(groups),2)/6.,.001953125).to(tl.float8e4nv)
    if FAST:normalized=groups*reciprocal(scale.to(tl.float32))[:,:,None]
    else:normalized=groups/scale.to(tl.float32)[:,:,None]
    lo,hi=tl.split(tl.reshape(normalized,(R,C//2,2)))
    packed=tl.inline_asm_elementwise(
        '{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }',
        constraints='=r,f,f',args=[lo,hi],dtype=tl.uint32,is_pure=True,pack=1).to(tl.uint8)
    code=tl.reshape(tl.join(packed&15,packed>>4),(R,C)).to(tl.int32)
    mag=code&7
    value=tl.where(mag<4,mag*.5,tl.where(mag<6,mag-2.,2.*mag-8.))
    value=tl.where(code>=8,-value,value)
    residual=groups-tl.reshape(value,(R,C//16,16))*scale.to(tl.float32)[:,:,None]
    return packed,scale,tl.reshape(residual,(R,C))


@triton.jit
def pack_linear_kernel(X,D,F,C,S,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,P:tl.constexpr):
    rows=tl.program_id(0)*16+tl.arange(0,16)
    cols=tl.program_id(1)*128+tl.arange(0,128)
    x=tl.load(X+rows[:,None]*K+cols[None,:],rows[:,None]<M,0).to(tl.float32)
    d=tl.load(D+cols)
    z=(x*d[None,:]).to(tl.bfloat16).to(tl.float32)/tl.load(F)
    ci=rows[:,None]*(K//2)+tl.program_id(1)*64+tl.arange(0,64)[None,:]
    si=rows[:,None]*(K//16)+tl.program_id(1)*8+tl.arange(0,8)[None,:]
    for plane in tl.static_range(P):
        codes,scales,z=component(z,16,128)
        tl.store(C+plane*MP*(K//2)+ci,codes,rows[:,None]<MP)
        tl.store(S+plane*MP*(K//16)+si,scales,rows[:,None]<MP)


def pack_linear(x, planes, prescale=None):
    if x.ndim!=2 or x.dtype!=torch.bfloat16 or not x.is_cuda or x.shape[1]%128:
        raise ValueError('Expected CUDA BF16 [M,K], K multiple of 128')
    if not 1 <= planes <= 4:
        raise ValueError('Supported plane count: 1..4')
    x=x.contiguous();m,k=x.shape;mp=triton.cdiv(m,16)*16
    if not m:raise ValueError('Empty matrix')
    d=torch.ones(k,device=x.device) if prescale is None else prescale
    partial=torch.empty((triton.cdiv(m*k,4096),),device=x.device,dtype=torch.float32)
    factor=torch.empty((),device=x.device,dtype=torch.float32)
    scaled_amax[(len(partial),)](x,d,partial,m,k,4096,num_warps=4)
    finish_amax[(1,)](partial,factor,len(partial),triton.next_power_of_2(len(partial)),num_warps=4)
    codes=torch.empty((planes,mp,k//2),device=x.device,dtype=torch.uint8)
    scales=torch.empty((planes,mp,k//16),device=x.device,dtype=torch.float8_e4m3fn)
    pack_linear_kernel[(mp//16,k//128)](x,d,factor,codes,scales,m,k,mp,planes,num_warps=4)
    return codes,scales,factor,mp


@triton.jit
def gemm_kernel(AC,AS,WC,WS,AG,WG,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                MP:tl.constexpr,WP:tl.constexpr,AP:tl.constexpr,MASK:tl.constexpr,BM:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*128+tl.arange(0,128)
    dk=tl.arange(0,64);ds=tl.arange(0,8)
    acc=tl.full((BM,128),0,tl.float32)
    # Same K/term order as full-support Native: W0 first (A0,A1,...), then W1, ...
    for wp in tl.static_range(WP):
        for kb in range(K//128):
            wi=wp*N*(K//2)+ni[:,None]*(K//2)+kb*64+dk[None,:]
            wsi=wp*N*(K//16)+ni[:,None]*(K//16)+kb*8+ds[None,:]
            w=tl.load(WC+wi);s=tl.load(WS+wsi)
            for ap in tl.static_range(AP):
                if MASK & (1 << (wp*4+ap)):
                    ai=ap*MP*(K//2)+mi[:,None]*(K//2)+kb*64+dk[None,:]
                    asi=ap*MP*(K//16)+mi[:,None]*(K//16)+kb*8+ds[None,:]
                    a=tl.load(AC+ai,mi[:,None]<MP,0);asc=tl.load(AS+asi,mi[:,None]<MP,0.)
                    acc=tl.dot_scaled(a,asc,'e2m1',w.T,s,'e2m1',acc)
    tl.store(Y+mi[:,None]*N+ni[None,:],acc*(tl.load(WG)*tl.load(AG)),mi[:,None]<M)


def linear(x, weight, weight_planes, activation_planes, mask, *, bm=128, prescale=None):
    ac,asc,ag,mp=pack_linear(x,activation_planes,prescale)
    wc,ws,wg=weight
    m,k=x.shape;n=wc.shape[1]
    if n%128 or wc.shape!=(weight_planes,n,k//2) or ws.shape!=(weight_planes,n,k//16):
        raise ValueError('Packed weight shape mismatch')
    y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    gemm_kernel[(triton.cdiv(m,bm),n//128)](ac,asc,wc,ws,ag,wg,y,m,n,k,mp,
        weight_planes,activation_planes,mask,bm,num_warps=4,num_stages=2)
    return y


@triton.jit
def pack_qk_kernel(X,C,S,F,M:tl.constexpr,H:tl.constexpr,SX:tl.constexpr,SH:tl.constexpr,
                   MP:tl.constexpr,P:tl.constexpr):
    r=tl.program_id(0)*16+tl.arange(0,16);h=tl.program_id(1);d=tl.arange(0,128)
    x=tl.load(X+r[:,None]*SX+h*SH+d[None,:],r[:,None]<M,0).to(tl.float32)
    f=tl.maximum(tl.max(tl.abs(x),1)/1536.,1.e-12)
    z=x/f[:,None]
    ci=(h*MP+r[:,None])*64+tl.arange(0,64)[None,:]
    si=(h*MP+r[:,None])*8+tl.arange(0,8)[None,:]
    for plane in tl.static_range(P):
        c,s,z=component(z,16,128)
        tl.store(C+plane*H*MP*64+ci,c,r[:,None]<MP)
        tl.store(S+plane*H*MP*8+si,s,r[:,None]<MP)
    tl.store(F+h*MP+r,f,r<MP)


@triton.jit
def pack_v_kernel(X,C,S,F,N:tl.constexpr,H:tl.constexpr,SX:tl.constexpr,SH:tl.constexpr,
                  NT:tl.constexpr,P:tl.constexpr):
    di=tl.program_id(0)*16+tl.arange(0,16);nt=tl.program_id(1);h=tl.program_id(2)
    ni=nt*128+tl.arange(0,128)
    x=tl.load(X+ni[None,:]*SX+h*SH+di[:,None],ni[None,:]<N,0).to(tl.float32)
    f=tl.maximum(tl.max(tl.abs(x),1)/1536.,1.e-12);z=x/f[:,None]
    ci=(h*NT+nt)*128*64+tl.arange(0,64)[None,:]*128+di[:,None]
    si=(h*NT+nt)*128*8+tl.arange(0,8)[None,:]*128+di[:,None]
    for plane in tl.static_range(P):
        c,s,z=component(z,16,128)
        tl.store(C+plane*H*NT*128*64+ci,c)
        tl.store(S+plane*H*NT*128*8+si,s)
    tl.store(F+(h*NT+nt)*128+di,f)


def pack_attention(q,k,v,wp,ap):
    from thunderworld_fp4.runtime import _validate
    _validate(q,k,v)
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2]
    mp=triton.cdiv(m,16)*16;np=triton.cdiv(n,128)*128
    values=[]
    for kind,x,heads,length,padded,planes in [('q',q,h,m,mp,ap),('k',k,hk,n,np,wp),('v',v,hk,n,np,wp)]:
        codes=torch.empty((planes,heads*padded,64),device=x.device,dtype=torch.uint8)
        scales=torch.empty((planes,heads*padded,8),device=x.device,dtype=torch.float8_e4m3fn)
        factors=torch.empty(heads*padded,device=x.device,dtype=torch.float32)
        if kind=='v':
            pack_v_kernel[(8,np//128,heads)](x,codes,scales,factors,length,heads,
                x.stride(1),x.stride(2),np//128,planes,num_warps=4)
        else:
            pack_qk_kernel[(padded//16,heads)](x,codes,scales,factors,length,heads,
                x.stride(1),x.stride(2),padded,planes,num_warps=4)
        values.extend((codes,scales,factors))
    return values,mp,np


@triton.jit
def attention_kernel(QC,QS,QF,KC,KS,KF,VC,VS,VF,O,STATS,M:tl.constexpr,N:tl.constexpr,
        HQ:tl.constexpr,HK:tl.constexpr,MP:tl.constexpr,NP:tl.constexpr,WP:tl.constexpr,
        AP:tl.constexpr,MASK:tl.constexpr,BM:tl.constexpr,SPLITS:tl.constexpr,FAST_P:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);h=tl.program_id(1);kh=h//(HQ//HK)
    dd=tl.arange(0,128);dk=tl.arange(0,64);ds=tl.arange(0,8)
    qi=(h*MP+mi[:,None])*64+dk[None,:]
    qsi=(h*MP+mi[:,None])*8+ds[None,:]
    qf=tl.load(QF+h*MP+mi,mi<MP,0)
    acc=tl.full((BM,128),0,tl.float32)
    mx=tl.full((BM,),-float('inf'),tl.float32);denom=tl.full((BM,),0,tl.float32)
    split=tl.program_id(2)
    for block in range(split*tl.cdiv(tl.cdiv(N,128),SPLITS),
                       tl.minimum((split+1)*tl.cdiv(tl.cdiv(N,128),SPLITS),tl.cdiv(N,128))):
        ni=block*128+tl.arange(0,128)
        ki=(kh*NP+ni[:,None])*64+dk[None,:];ksi=(kh*NP+ni[:,None])*8+ds[None,:]
        kf=tl.load(KF+kh*NP+ni)
        score=tl.full((BM,128),0,tl.float32)
        for wp in tl.static_range(WP):
            k=tl.load(KC+wp*HK*NP*64+ki);ks=tl.load(KS+wp*HK*NP*8+ksi)
            for ap in tl.static_range(AP):
                if MASK & (1 << (wp*4+ap)):
                    q=tl.load(QC+ap*HQ*MP*64+qi,mi[:,None]<MP,0)
                    qs=tl.load(QS+ap*HQ*MP*8+qsi,mi[:,None]<MP,0.)
                    score=tl.dot_scaled(q,qs,'e2m1',k.T,ks,'e2m1',score)
        score=score*qf[:,None]*kf[None,:]*.08838834764831845
        score=tl.where(ni[None,:]<N,score,-float('inf'))
        nm=tl.maximum(mx,tl.max(score,1));alpha=tl.exp(mx-nm)
        p=tl.exp(score-nm[:,None]);denom=denom*alpha+tl.sum(p,1)
        if FAST_P:pf=tl.maximum(tl.max(tl.abs(p),1)*(1./1536.),1.e-12);pn=p*reciprocal(pf)[:,None]
        else:pf=tl.maximum(tl.max(tl.abs(p),1)/1536.,1.e-12);pn=p/pf[:,None]
        vi=(kh*tl.cdiv(N,128)+block)*128*64+dk[None,:]*128+dd[:,None]
        vsi=(kh*tl.cdiv(N,128)+block)*128*8+ds[None,:]*128+dd[:,None]
        vf=tl.load(VF+(kh*tl.cdiv(N,128)+block)*128+dd)
        tile=tl.full((BM,128),0,tl.float32)
        for wp in tl.static_range(WP):
            v=tl.load(VC+wp*HK*NP*64+vi);vs=tl.load(VS+wp*HK*NP*8+vsi)
            residual=pn
            # Recompute identical P planes for each V plane to bound registers.
            # Extra scalar quantization is documented; no latency claim is made.
            for ap in tl.static_range(AP):
                pc,ps,residual=component(residual,BM,128,FAST_P)
                if MASK & (1 << (wp*4+ap)):
                    tile=tl.dot_scaled(pc,ps,'e2m1',v.T,vs,'e2m1',tile)
        acc=acc*alpha[:,None]+tile*pf[:,None]*vf[None,:];mx=nm
    if SPLITS==1:
        tl.store(O+(mi[:,None]*HQ+h)*128+dd[None,:],acc/denom[:,None],mi[:,None]<M)
    else:
        oi=((split*HQ+h)*M+mi[:,None])*128+dd[None,:]
        tl.store(O+oi,acc,mi[:,None]<M)
        si=((split*HQ+h)*M+mi)*2
        tl.store(STATS+si,mx,mi<M);tl.store(STATS+si+1,denom,mi<M)


def attention(q,k,v,wp,ap,mask,*,bm=32,splits=2,fast_p=True):
    packed,mp,np=pack_attention(q,k,v,wp,ap)
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2]
    nt=triton.cdiv(n,128);splits=min(splits,nt);chunk=triton.cdiv(nt,splits);splits=triton.cdiv(nt,chunk)
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    parts=out if splits==1 else torch.empty((splits,h,m,128),device=q.device,dtype=torch.float32)
    stats=out if splits==1 else torch.empty((splits,h,m,2),device=q.device,dtype=torch.float32)
    attention_kernel[(triton.cdiv(m,bm),h,splits)](*packed,parts,stats,m,n,h,hk,mp,np,
        wp,ap,mask,bm,splits,fast_p,num_warps=4,num_stages=1)
    if splits>1:
        combine[(triton.cdiv(m,4),h)](parts,stats,out,m,h,splits,triton.next_power_of_2(splits),
            num_warps=4,enable_fp_fusion=False)
    return out


def decode(codes,scales,factor):
    """Independent Torch unpacking; used only by correctness/representation tests."""
    table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=codes.device)
    values=torch.stack((codes&15,codes>>4),-1).long()
    return table[values].flatten(-2)*scales.float().repeat_interleave(16,-1)*factor

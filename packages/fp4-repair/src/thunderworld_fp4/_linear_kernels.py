"""Portable native FP4 FFN/projection kernels; weights are already packed."""
import triton
import triton.language as tl
from ._quantize import quantize_linear


@triton.jit
def scaled_amax(X, D, PARTIAL, M:tl.constexpr, K:tl.constexpr, B:tl.constexpr):
    idx=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+idx,idx<M*K,0).to(tl.float32)
    d=tl.load(D+idx%K)
    x=(x*d).to(tl.bfloat16).to(tl.float32)
    tl.store(PARTIAL+tl.program_id(0),tl.max(tl.abs(x),0))


@triton.jit
def finish_amax(PARTIAL, FACTOR, COUNT:tl.constexpr, B:tl.constexpr):
    idx=tl.arange(0,B);value=tl.max(tl.load(PARTIAL+idx,idx<COUNT,0),0)
    tl.store(FACTOR,tl.where(value>0,value/2688.,1.))


@triton.jit
def activation_rank(X,D,RANK,M:tl.constexpr,K:tl.constexpr,BT:tl.constexpr):
    mb=tl.program_id(0);kt=tl.arange(0,BT)
    # Eight sampled rows/columns within each 128x128 activation block.
    rows=tl.minimum(mb*128+tl.arange(0,8)*16,M-1)
    cols=kt[:,None]*128+tl.arange(0,8)[None,:]*16
    x=tl.load(X+rows[None,:,None]*K+cols[:,None,:],cols[:,None,:]<K,0).to(tl.float32)
    d=tl.load(D+cols,cols<K,0)
    x=(x*d[:,None,:]).to(tl.bfloat16).to(tl.float32)
    energy=tl.sum(tl.sum(x*x,2),1)
    energy=tl.where(kt<K//128,energy,-float('inf'))
    rank=tl.sum(((energy[None,:]>energy[:,None])|((energy[None,:]==energy[:,None])&(kt[None,:]<kt[:,None]))).to(tl.int32),1)
    tl.store(RANK+mb*(K//128)+kt,rank,kt<K//128)


@triton.jit
def pack_activation(X,D,FACTOR,UNION,CODES,SCALES,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,
                    REPAIR:tl.constexpr,FULL:tl.constexpr):
    mi=tl.program_id(0)*16+tl.arange(0,16);ki=tl.program_id(1)*128+tl.arange(0,128)
    x=tl.load(X+mi[:,None]*K+ki[None,:],mi[:,None]<M,0).to(tl.float32)
    d=tl.load(D+ki);x=(x*d[None,:]).to(tl.bfloat16).to(tl.float32)
    factor=tl.load(FACTOR)
    if REPAIR and not FULL:
        active=tl.load(UNION+(tl.program_id(0)*16//128)*(K//128)+tl.program_id(1))
        if active:p,s,rp,rs,_=quantize_linear(x,factor,16,128,True)
        else:
            p,s,rp,rs,_=quantize_linear(x,factor,16,128,False)
            rp=tl.full((16,64),0,tl.uint8);rs=tl.full((16,8),0,tl.float8e4nv)
    else:p,s,rp,rs,_=quantize_linear(x,factor,16,128,REPAIR)
    pi=mi[:,None]*(K//2)+tl.program_id(1)*64+tl.arange(0,64)[None,:]
    si=mi[:,None]*(K//16)+tl.program_id(1)*8+tl.arange(0,8)[None,:]
    tl.store(CODES+pi,p,mi[:,None]<MP);tl.store(SCALES+si,s,mi[:,None]<MP)
    if REPAIR:
        tl.store(CODES+MP*(K//2)+pi,rp,mi[:,None]<MP)
        tl.store(SCALES+MP*(K//16)+si,rs,mi[:,None]<MP)


@triton.jit
def linear_gemm(AC,AS,WC,WS,AF,WG,BIAS,MASK,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                MP:tl.constexpr,BIAS_ON:tl.constexpr,REPAIR:tl.constexpr,
                FULL:tl.constexpr,BM:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*128+tl.arange(0,128)
    dk=tl.arange(0,64);ds=tl.arange(0,8);acc=tl.full((BM,128),0,tl.float32)
    for kt in range(K//128):
        ai=mi[:,None]*(K//2)+kt*64+dk[None,:]
        asi=mi[:,None]*(K//16)+kt*8+ds[None,:]
        wi=ni[:,None]*(K//2)+kt*64+dk[None,:]
        wsi=ni[:,None]*(K//16)+kt*8+ds[None,:]
        a=tl.load(AC+ai,mi[:,None]<MP,0);s=tl.load(AS+asi,mi[:,None]<MP,0.)
        w=tl.load(WC+wi);ws=tl.load(WS+wsi)
        acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
        if REPAIR:
            if FULL:selected=True
            else:selected=tl.load(MASK+((tl.program_id(0)*BM//128)*(N//128)+tl.program_id(1))*(K//128)+kt)
            if selected:
                r=tl.load(AC+MP*(K//2)+ai,mi[:,None]<MP,0)
                rs=tl.load(AS+MP*(K//16)+asi,mi[:,None]<MP,0.)
                acc=tl.dot_scaled(r,rs,'e2m1',w.T,ws,'e2m1',acc)
    y=acc*(tl.load(AF)*tl.load(WG))
    if BIAS_ON:y=y+tl.load(BIAS+ni)[None,:].to(tl.float32)
    tl.store(Y+mi[:,None]*N+ni[None,:],y,mi[:,None]<M)

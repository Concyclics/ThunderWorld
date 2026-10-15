"""Thor-native NVFP4 attention; explicit SM110 tcgen05, transposed Q32/64/128 tiles.

K/Q and V/P operands are transposed so the native MMA M dimension is
always 128; its N dimension is the smaller query tile. This removes query
padding without emulating FP4 in BF16.

The original selectors, packing and split combine remain in use. This kernel
changes the hardware accumulation path from compiler-emulated BF16 to native
NVFP4. It requires a separate numerical qualification and is not bitwise-equivalent
to the emulation. No base blocks or scheduled repair terms are removed.
"""
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, TensorMemoryScalesLayout, allocate_tensor_memory,
    fence_async_shared, tcgen05_mma_scaled, tcgen05_commit, mbarrier,
)


@gluon.jit
def reciprocal(x):
    return gl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;',constraints='=f,f',
        args=[x],dtype=gl.float32,is_pure=True,pack=1)


@gluon.jit
def quantize_p(x, REPAIR:gl.constexpr, FAST:gl.constexpr):
    R:gl.constexpr=x.shape[0]
    if FAST:
        factor=gl.maximum(gl.max(gl.abs(x),1)*(1./1536.),1.e-12)
        z=x*reciprocal(factor)[:,None]
    else:
        factor=gl.maximum(gl.max(gl.abs(x),1)/1536.,1.e-12)
        z=x/factor[:,None]
    z3=gl.reshape(z,(R,8,16))
    if FAST:scale=gl.maximum(gl.max(gl.abs(z3),2)*(1./6.),.001953125).to(gl.float8e4nv)
    else:scale=gl.maximum(gl.max(gl.abs(z3),2)/6.,.001953125).to(gl.float8e4nv)
    if FAST:normalized=gl.reshape(z3*reciprocal(scale.to(gl.float32))[:,:,None],(R,128))
    else:normalized=gl.reshape(z3/scale.to(gl.float32)[:,:,None],(R,128))
    lo,hi=gl.split(gl.reshape(normalized,(R,64,2)))
    packed=gl.inline_asm_elementwise(
        '{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }',
        constraints='=r,f,f',args=[lo,hi],dtype=gl.uint32,is_pure=True,pack=1).to(gl.uint8)
    if REPAIR:
        code=gl.reshape(gl.join(packed&15,packed>>4),(R,128)).to(gl.int32)
        mag=code&7
        val=gl.where(mag<4,mag*.5,gl.where(mag<6,mag-2.,2.*mag-8.))
        val=gl.where(code>=8,-val,val)
        residual=z3-gl.reshape(val,(R,8,16))*scale.to(gl.float32)[:,:,None]
        if FAST:
            rs=gl.maximum(gl.max(gl.abs(residual),2)*(1./6.),.001953125).to(gl.float8e4nv)
            rn=gl.reshape(residual*reciprocal(rs.to(gl.float32))[:,:,None],(R,64,2))
        else:
            rs=gl.maximum(gl.max(gl.abs(residual),2)/6.,.001953125).to(gl.float8e4nv)
            rn=gl.reshape(residual/rs.to(gl.float32)[:,:,None],(R,64,2))
        rlo,rhi=gl.split(rn)
        rp=gl.inline_asm_elementwise(
            '{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u32.u8 $0, tmp; }',
            constraints='=r,f,f',args=[rlo,rhi],dtype=gl.uint32,is_pure=True,pack=1).to(gl.uint8)
    else:rp=packed;rs=scale
    return packed,scale,rp,rs,factor


@gluon.jit
def dot(a,asc,b,bsc,asmem,bsmem,atmem,btmem,acc,bar,phase,USE:gl.constexpr):
    asmem.store(b);bsmem.store(a)
    atmem.store(gl.convert_layout(bsc,atmem.get_reg_layout()))
    btmem.store(gl.convert_layout(asc,btmem.get_reg_layout()))
    fence_async_shared()
    tcgen05_mma_scaled(asmem,bsmem.permute((1,0)),acc,atmem,btmem,'e2m1','e2m1',use_acc=USE)
    tcgen05_commit(bar)
    mbarrier.wait(bar,phase)
    return phase^1


@gluon.jit
def streaming_attention(QC,QS,QF,KC,KS,KF,VC,VS,VF,FLAGS,O,STATS,
    M:gl.constexpr,N:gl.constexpr,HQ:gl.constexpr,HK:gl.constexpr,
    MP:gl.constexpr,NP:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,
    REPAIR:gl.constexpr,FULL:gl.constexpr,TERMS:gl.constexpr,SPLITS:gl.constexpr,
    LAZY_P:gl.constexpr,FAST_P:gl.constexpr=False,TRANSPOSE_V:gl.constexpr=False,
    RELOAD_Q:gl.constexpr=False,PVFLAGS=None,SEPARATE:gl.constexpr=False,
    PV_REPAIR:gl.constexpr=False,PV_FULL:gl.constexpr=False,PV_TERMS:gl.constexpr=3):
    gl.static_assert((BM==32 or BM==64 or BM==128) and BN==128)
    L:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[gl.num_warps(),1],[1,0])
    LS:gl.constexpr=gl.BlockedLayout([1,1],[4,8],[gl.num_warps(),1],[1,0])
    R:gl.constexpr=gl.SliceLayout(1,L)
    C:gl.constexpr=gl.SliceLayout(0,L)
    SL:gl.constexpr=gl.NVMMASharedLayout.get_default_for([128,64],gl.uint8)
    asmem=gl.allocate_shared_memory(gl.uint8,[128,64],SL)
    bsmem=gl.allocate_shared_memory(gl.uint8,[BM,64],SL)
    atmem=allocate_tensor_memory(gl.float8e4nv,[128,8],TensorMemoryScalesLayout())
    btmem=allocate_tensor_memory(gl.float8e4nv,[BM,8],TensorMemoryScalesLayout())
    tilemem=allocate_tensor_memory(gl.float32,[128,BM],TensorMemoryLayout([128,BM],col_stride=1))
    bar=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(bar,count=1)
    phase=0
    mb=gl.program_id(0);h=gl.program_id(1);kh=h//(HQ//HK)
    mi=mb*BM+gl.arange(0,BM,layout=R)
    mis=mb*BM+gl.arange(0,BM,layout=gl.SliceLayout(1,LS))
    dd=gl.arange(0,128,layout=C)
    dk=gl.arange(0,64,layout=C)
    ds=gl.arange(0,8,layout=gl.SliceLayout(0,LS))
    qi=(h*MP+mi[:,None])*64+dk[None,:]
    qsi=(h*MP+mis[:,None])*8+ds[None,:]
    acc=gl.full((BM,128),0,gl.float32,L)
    mx=gl.full((BM,),-float('inf'),gl.float32,R)
    denom=gl.full((BM,),0,gl.float32,R)
    split=gl.program_id(2)
    for start in range(split*gl.cdiv(gl.cdiv(N,128),SPLITS),gl.minimum((split+1)*gl.cdiv(gl.cdiv(N,128),SPLITS),gl.cdiv(N,128))):
        # Q is small and cache-resident. Reloading releases these registers
        # across the softmax/P-quantization/PV region of each key tile.
        q0=gl.load(QC+qi,mi[:,None]<MP,0);qs=gl.load(QS+qsi,mis[:,None]<MP,0.)
        qf=gl.load(QF+h*MP+mi,mi<MP,0.)
        ni=start*128+gl.arange(0,128,layout=R)
        nis=start*128+gl.arange(0,128,layout=gl.SliceLayout(1,LS))
        ki=(kh*NP+ni[:,None])*64+dk[None,:]
        ksi=(kh*NP+nis[:,None])*8+ds[None,:]
        k0=gl.load(KC+ki);ks=gl.load(KS+ksi)
        kf=gl.load(KF+kh*NP+start*128+dd)
        phase=dot(q0,qs,k0,ks,asmem,bsmem,atmem,btmem,tilemem,bar,phase,False)
        if REPAIR:
            selected=True if FULL else gl.load(FLAGS+(h*gl.cdiv(M,128)+mb*BM//128)*gl.cdiv(N,128)+start)>1
            if selected:
                q1=gl.load(QC+HQ*MP*64+qi,mi[:,None]<MP,0)
                qr=gl.load(QS+HQ*MP*8+qsi,mis[:,None]<MP,0.)
                phase=dot(q1,qr,k0,ks,asmem,bsmem,atmem,btmem,tilemem,bar,phase,True)
                if TERMS>=3:
                    k1=gl.load(KC+HK*NP*64+ki);kr=gl.load(KS+HK*NP*8+ksi)
                    phase=dot(q0,qs,k1,kr,asmem,bsmem,atmem,btmem,tilemem,bar,phase,True)
                    if TERMS==4:phase=dot(q1,qr,k1,kr,asmem,bsmem,atmem,btmem,tilemem,bar,phase,True)
        score=gl.convert_layout(gl.permute(tilemem.load(),(1,0)),L)
        score=score*qf[:,None]*kf[None,:]*.08838834764831845
        score=gl.where(start*128+dd[None,:]<N,score,-float('inf'))
        nm=gl.maximum(mx,gl.max(score,1));alpha=gl.exp(mx-nm)
        p=gl.exp(score-nm[:,None]);denom=denom*alpha+gl.sum(p,1)
        if SEPARATE:
            if PV_FULL:pv_selected=True
            elif PV_REPAIR:pv_selected=gl.load(PVFLAGS+(h*gl.cdiv(M,128)+mb*BM//128)*gl.cdiv(N,128)+start)>1
            else:pv_selected=False
        elif REPAIR:pv_selected=selected
        else:pv_selected=False
        if (PV_REPAIR if SEPARATE else REPAIR) and LAZY_P and not (PV_FULL if SEPARATE else FULL):
            if pv_selected:p0,ps,p1,pr,pf=quantize_p(p,True,FAST_P)
            else:p0,ps,p1,pr,pf=quantize_p(p,False,FAST_P)
        else:p0,ps,p1,pr,pf=quantize_p(p,PV_REPAIR if SEPARATE else REPAIR,FAST_P)
        vr=gl.arange(0,128,layout=R)
        vrs=gl.arange(0,128,layout=gl.SliceLayout(1,LS))
        if TRANSPOSE_V:
            vi=(kh*gl.cdiv(N,128)+start)*128*64+dk[None,:]*128+vr[:,None]
            vsi=(kh*gl.cdiv(N,128)+start)*128*8+ds[None,:]*128+vrs[:,None]
        else:
            vi=((kh*gl.cdiv(N,128)+start)*128+vr[:,None])*64+dk[None,:]
            vsi=((kh*gl.cdiv(N,128)+start)*128+vrs[:,None])*8+ds[None,:]
        v0=gl.load(VC+vi);vs=gl.load(VS+vsi)
        vf=gl.load(VF+(kh*gl.cdiv(N,128)+start)*128+dd)
        phase=dot(p0,ps,v0,vs,asmem,bsmem,atmem,btmem,tilemem,bar,phase,False)
        if PV_REPAIR if SEPARATE else REPAIR:
            if pv_selected:
                phase=dot(p1,pr,v0,vs,asmem,bsmem,atmem,btmem,tilemem,bar,phase,True)
                if (PV_TERMS if SEPARATE else TERMS)>=3:
                    v1=gl.load(VC+HK*NP*64+vi);vs1=gl.load(VS+HK*NP*8+vsi)
                    phase=dot(p0,ps,v1,vs1,asmem,bsmem,atmem,btmem,tilemem,bar,phase,True)
                    if (PV_TERMS if SEPARATE else TERMS)==4:phase=dot(p1,pr,v1,vs1,asmem,bsmem,atmem,btmem,tilemem,bar,phase,True)
        tile=gl.convert_layout(gl.permute(tilemem.load(),(1,0)),L)
        acc=acc*alpha[:,None]+tile*pf[:,None]*vf[None,:];mx=nm
    mbarrier.invalidate(bar)
    if SPLITS==1:
        gl.store(O+(mi[:,None]*HQ+h)*128+dd[None,:],acc/denom[:,None],mi[:,None]<M)
    else:
        oi=((split*HQ+h)*M+mi[:,None])*128+dd[None,:]
        gl.store(O+oi,acc,mi[:,None]<M)
        si=((split*HQ+h)*M+mi)*2
        gl.store(STATS+si,mx,mi<M);gl.store(STATS+si+1,denom,mi<M)

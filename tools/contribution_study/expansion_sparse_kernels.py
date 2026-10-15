"""Budgeted three-term ablation. Parent packing and arithmetic order stay fixed.

This quality mechanism path deliberately packs full residual planes. Only actual
MMA tile execution is sparse; packing, storage and latency savings are NOT claimed.
"""
import torch
import triton
import triton.language as tl
from expansion_kernels import component, pack_linear, pack_attention
from thunderworld_fp4._kernels import combine
from thunderworld_fp4._quantize import reciprocal


@triton.jit
def represented(c, s, R: tl.constexpr):
    code=tl.reshape(tl.join(c&15,c>>4),(R,128)).to(tl.int32)
    mag=code&7
    value=tl.where(mag<4,mag*.5,tl.where(mag<6,mag-2.,2.*mag-8.))
    value=tl.where(code>=8,-value,value)
    return tl.reshape(tl.reshape(value,(R,8,16))*s.to(tl.float32)[:,:,None],(R,128))


@triton.jit
def energy_kernel(C,S,F,E,M:tl.constexpr,K:tl.constexpr,H:tl.constexpr,
                  R:tl.constexpr,ROW_FACTOR:tl.constexpr):
    rt,kb,ph=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    plane,head=ph//H,ph%H
    r=rt*R+tl.arange(0,R)
    ci=((plane*H+head)*M+r[:,None])*(K//2)+kb*64+tl.arange(0,64)[None,:]
    si=((plane*H+head)*M+r[:,None])*(K//16)+kb*8+tl.arange(0,8)[None,:]
    c=tl.load(C+ci,r[:,None]<M,0);s=tl.load(S+si,r[:,None]<M,0.)
    values=represented(c,s,R)
    if ROW_FACTOR:
        f=tl.load(F+head*M+r,r<M,0.)
        values=values*f[:,None]
    else:
        values=values*tl.load(F)
    total=tl.sum(tl.sum(values*values,1),0)
    tl.store(E+(ph*tl.cdiv(M,R)+rt)*(K//128)+kb,total)


@triton.jit
def v_energy_kernel(C,S,F,E,H:tl.constexpr,NT:tl.constexpr):
    block,ph=tl.program_id(0),tl.program_id(1)
    plane,head=ph//H,ph%H
    d=tl.arange(0,128)
    ci=((plane*H+head)*NT+block)*128*64+tl.arange(0,64)[None,:]*128+d[:,None]
    si=((plane*H+head)*NT+block)*128*8+tl.arange(0,8)[None,:]*128+d[:,None]
    values=represented(tl.load(C+ci),tl.load(S+si),128)
    f=tl.load(F+(head*NT+block)*128+d)
    values=values*f[:,None]
    tl.store(E+ph*NT+block,tl.sum(tl.sum(values*values,1),0))


def energies(c,s,f,*,heads=1,rows=128,row_factor=False):
    planes,total,half_k=c.shape
    m,k=total//heads,half_k*2
    out=torch.empty((planes,heads,triton.cdiv(m,rows),k//128),device=c.device,dtype=torch.float32)
    energy_kernel[(triton.cdiv(m,rows),k//128,planes*heads)](
        c,s,f,out,m,k,heads,rows,row_factor,num_warps=4)
    return out


def choose(scores,count):
    """Fixed integer quota; stable index tie-break; no GPU -> host transfers."""
    if not 0<=count<=scores.shape[-1]:raise ValueError('Invalid repair quota')
    mask=torch.zeros_like(scores,dtype=torch.bool)
    if count:
        order=torch.argsort(scores,dim=-1,descending=True,stable=True)
        mask.scatter_(-1,order[...,:count],True)
    return mask.contiguous()


def weight_support(weight,numerator,denominator=4):
    wc,ws,wg=weight
    e=energies(wc,ws,wg)[:,0]
    wm=choose(e[1],(wc.shape[-1]//64)*numerator//denominator)
    # Offline sensitivity of each K128 channel block; does not change W0.
    sensitivity=e[0].sum(dim=0)
    return wm,sensitivity


def linear(x,weight,wm,sensitivity,numerator,denominator=4,*,prescale=None,am=None):
    ac,asc,ag,mp=pack_linear(x,2,prescale)
    wc,ws,wg=weight
    m,k=x.shape;n=wc.shape[1]
    if n%128 or wc.shape!=(2,n,k//2):raise ValueError('Invalid sparse weight shape')
    if am is None:
        energy=energies(ac,asc,ag)[1,0]
        am=choose(energy*sensitivity[None,:],(k//128)*numerator//denominator)
    if am.shape!=(triton.cdiv(m,128),k//128) or wm.shape!=(n//128,k//128):
        raise ValueError('Invalid sparse linear masks')
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    sparse_gemm_kernel[(triton.cdiv(m,128),n//128)](
        ac,asc,wc,ws,ag,wg,am,wm,y,m,n,k,mp,2,2,19,128,num_warps=4,num_stages=2)
    return y


def attention_support(packed,mp,npad,m,h,hk,numerator,denominator=4,bm=32):
    qc,qs,qf,kc,ks,kf,vc,vs,vf=packed
    qe=energies(qc,qs,qf,heads=h,rows=bm,row_factor=True)[...,0]
    ke=energies(kc,ks,kf,heads=hk,rows=128,row_factor=True)[...,0]
    nt=npad//128
    ve=torch.empty((2,hk,nt),device=qc.device,dtype=torch.float32)
    v_energy_kernel[(nt,2*hk)](vc,vs,vf,ve,hk,nt,num_warps=4)
    ke=ke.repeat_interleave(h//hk,dim=1);ve=ve.repeat_interleave(h//hk,dim=1)
    # Norm-product proxies, no selector matrix multiplication. Q/P are A-side;
    # K/V are W-side. P does not yet exist: use V norms as a PV proxy, not
    # an assertion about the actual post-softmax correction benefit.
    qscores=torch.stack((qe[1,:,:,None]*ke[0,:,None,:],qe[0,:,:,None]*ke[1,:,None,:]))
    pscores=ve[:,:,None,:].expand(2,h,triton.cdiv(m,bm),nt)
    quota=nt*numerator//denominator
    return choose(qscores,quota),choose(pscores,quota)


def attention(q,k,v,numerator,denominator=4,*,masks=None,bm=32,splits=2,fast_p=True):
    packed,mp,npad=pack_attention(q,k,v,2,2)
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2]
    if masks is None:masks=attention_support(packed,mp,npad,m,h,hk,numerator,denominator,bm)
    qm,pm=masks
    expected=(2,h,triton.cdiv(m,bm),triton.cdiv(n,128))
    if qm.shape!=expected or pm.shape!=expected:raise ValueError('Invalid attention masks')
    nt=triton.cdiv(n,128);splits=min(splits,nt);chunk=triton.cdiv(nt,splits);splits=triton.cdiv(nt,chunk)
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    parts=out if splits==1 else torch.empty((splits,h,m,128),device=q.device,dtype=torch.float32)
    stats=out if splits==1 else torch.empty((splits,h,m,2),device=q.device,dtype=torch.float32)
    sparse_attention_kernel[(triton.cdiv(m,bm),h,splits)](
        *packed,qm,pm,parts,stats,m,n,h,hk,mp,npad,2,2,19,bm,splits,fast_p,num_warps=4,num_stages=1)
    if splits>1:
        combine[(triton.cdiv(m,4),h)](parts,stats,out,m,h,splits,triton.next_power_of_2(splits),
            num_warps=4,enable_fp_fusion=False)
    return out

@triton.jit
def sparse_gemm_kernel(AC,AS,WC,WS,AG,WG,AM,WM,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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
                    if (wp == 0 and ap == 0):
                        acc=tl.dot_scaled(a,asc,'e2m1',w.T,s,'e2m1',acc)
                    elif wp == 0:
                        if tl.load(AM+tl.program_id(0)*(K//128)+kb):
                            acc=tl.dot_scaled(a,asc,'e2m1',w.T,s,'e2m1',acc)
                    else:
                        if tl.load(WM+tl.program_id(1)*(K//128)+kb):
                            acc=tl.dot_scaled(a,asc,'e2m1',w.T,s,'e2m1',acc)
    tl.store(Y+mi[:,None]*N+ni[None,:],acc*(tl.load(WG)*tl.load(AG)),mi[:,None]<M)


@triton.jit
def sparse_attention_kernel(QC,QS,QF,KC,KS,KF,VC,VS,VF,QM,PM,O,STATS,M:tl.constexpr,N:tl.constexpr,
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
        sid=(h*tl.cdiv(M,BM)+tl.program_id(0))*tl.cdiv(N,128)+block
        plane_stride=HQ*tl.cdiv(M,BM)*tl.cdiv(N,128)
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
                    if wp == 0 and ap == 0:
                        score=tl.dot_scaled(q,qs,'e2m1',k.T,ks,'e2m1',score)
                    else:
                        if tl.load(QM+(wp if wp else 0)*plane_stride+sid):
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
                    if wp == 0 and ap == 0:
                        tile=tl.dot_scaled(pc,ps,'e2m1',v.T,vs,'e2m1',tile)
                    else:
                        if tl.load(PM+(wp if wp else 0)*plane_stride+sid):
                            tile=tl.dot_scaled(pc,ps,'e2m1',v.T,vs,'e2m1',tile)
        acc=acc*alpha[:,None]+tile*pf[:,None]*vf[None,:];mx=nm
    if SPLITS==1:
        tl.store(O+(mi[:,None]*HQ+h)*128+dd[None,:],acc/denom[:,None],mi[:,None]<M)
    else:
        oi=((split*HQ+h)*M+mi[:,None])*128+dd[None,:]
        tl.store(O+oi,acc,mi[:,None]<M)
        si=((split*HQ+h)*M+mi)*2
        tl.store(STATS+si,mx,mi<M);tl.store(STATS+si+1,denom,mi<M)

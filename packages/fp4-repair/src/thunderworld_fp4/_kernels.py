"""Online Q/K/V packing plus native FP4 tile-streaming attention.

Packing is performed afresh for every input. No weights, old activations,
scores or probability matrices are cached. P is quantized only on chip.
"""
import triton
import triton.language as tl
from ._quantize import quantize, quantize_p


@triton.jit
def pack_qk(X, CODES, SCALES, FACTORS, M: tl.constexpr, H: tl.constexpr,
            SX: tl.constexpr, SH: tl.constexpr, MP: tl.constexpr, REPAIR: tl.constexpr):
    r=tl.program_id(0)*16+tl.arange(0,16)
    h=tl.program_id(1)
    d=tl.arange(0,128)
    x=tl.load(X+r[:,None]*SX+h*SH+d[None,:],r[:,None]<M,0).to(tl.float32)
    p,s,rp,rs,f=quantize(x,16,128,REPAIR)
    idx=(h*MP+r[:,None])*64+tl.arange(0,64)[None,:]
    si=(h*MP+r[:,None])*8+tl.arange(0,8)[None,:]
    tl.store(CODES+idx,p,r[:,None]<MP);tl.store(SCALES+si,s,r[:,None]<MP)
    if REPAIR:
        tl.store(CODES+H*MP*64+idx,rp,r[:,None]<MP)
        tl.store(SCALES+H*MP*8+si,rs,r[:,None]<MP)
    tl.store(FACTORS+h*MP+r,f,r<MP)


@triton.jit
def pack_v(X, CODES, SCALES, FACTORS, ENERGY, N:tl.constexpr,H:tl.constexpr,
           SX:tl.constexpr,SH:tl.constexpr,NT:tl.constexpr,REPAIR:tl.constexpr,NEED_ENERGY:tl.constexpr,TRANSPOSE:tl.constexpr=False):
    di=tl.program_id(0)*16+tl.arange(0,16)
    nt=tl.program_id(1);h=tl.program_id(2)
    ni=nt*128+tl.arange(0,128)
    x=tl.load(X+ni[None,:]*SX+h*SH+di[:,None],ni[None,:]<N,0).to(tl.float32)
    if NEED_ENERGY:tl.store(ENERGY+(h*NT+nt)*8+tl.program_id(0),tl.sum(tl.sum(x*x,1),0)/16384.)
    p,s,rp,rs,f=quantize(x,16,128,REPAIR)
    row=(h*NT+nt)*128+di
    idx=row[:,None]*64+tl.arange(0,64)[None,:]
    si=row[:,None]*8+tl.arange(0,8)[None,:]
    if TRANSPOSE:
        idx=(h*NT+nt)*128*64+tl.arange(0,64)[None,:]*128+di[:,None]
        si=(h*NT+nt)*128*8+tl.arange(0,8)[None,:]*128+di[:,None]
    tl.store(CODES+idx,p);tl.store(SCALES+si,s)
    if REPAIR:
        tl.store(CODES+H*NT*128*64+idx,rp)
        tl.store(SCALES+H*NT*128*8+si,rs)
    tl.store(FACTORS+row,f)


@triton.jit
def streaming_attention(QC,QS,QF,KC,KS,KF,VC,VS,VF,FLAGS,O,STATS,
                     M:tl.constexpr,N:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                     MP:tl.constexpr,NP:tl.constexpr,
                     BM:tl.constexpr,BN:tl.constexpr,REPAIR:tl.constexpr,FULL:tl.constexpr,TERMS:tl.constexpr,SPLITS:tl.constexpr,LAZY_P:tl.constexpr,FAST_P:tl.constexpr=False,TRANSPOSE_V:tl.constexpr=False):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);h=tl.program_id(1);kh=h//(HQ//HK)
    dd=tl.arange(0,128);dk=tl.arange(0,64);ds=tl.arange(0,8)
    qi=(h*MP+mi[:,None])*64+dk[None,:]
    qsi=(h*MP+mi[:,None])*8+ds[None,:]
    q0=tl.load(QC+qi,mi[:,None]<MP,0);qs=tl.load(QS+qsi,mi[:,None]<MP,0.0)
    qf=tl.load(QF+h*MP+mi,mi<MP,0)
    if REPAIR:
        q1=tl.load(QC+HQ*MP*64+qi,mi[:,None]<MP,0)
        qr=tl.load(QS+HQ*MP*8+qsi,mi[:,None]<MP,0.0)
    acc=tl.full((BM,128),0,tl.float32)
    mx=tl.full((BM,),-float('inf'),tl.float32);denom=tl.full((BM,),0,tl.float32)
    split=tl.program_id(2)
    for start in range(split*tl.cdiv(tl.cdiv(N,BN),SPLITS),tl.minimum((split+1)*tl.cdiv(tl.cdiv(N,BN),SPLITS),tl.cdiv(N,BN))):
        ni=start*BN+tl.arange(0,BN)
        ki=(kh*NP+ni[:,None])*64+dk[None,:]
        ksi=(kh*NP+ni[:,None])*8+ds[None,:]
        k0=tl.load(KC+ki);ks=tl.load(KS+ksi);kf=tl.load(KF+kh*NP+ni)
        score=tl.dot_scaled(q0,qs,'e2m1',k0.T,ks,'e2m1')
        if REPAIR:
            if FULL:selected=True
            else:selected=tl.load(FLAGS+(h*tl.cdiv(M,128)+tl.program_id(0)*BM//128)*tl.cdiv(N,128)+start*BN//128)>1
            if selected:
                score=tl.dot_scaled(q1,qr,'e2m1',k0.T,ks,'e2m1',score)
                if TERMS>=3:
                    k1=tl.load(KC+HK*NP*64+ki);kr=tl.load(KS+HK*NP*8+ksi)
                    score=tl.dot_scaled(q0,qs,'e2m1',k1.T,kr,'e2m1',score)
                    if TERMS==4:score=tl.dot_scaled(q1,qr,'e2m1',k1.T,kr,'e2m1',score)
        score=score*qf[:,None]*kf[None,:]*0.08838834764831845
        score=tl.where(ni[None,:]<N,score,-float('inf'))
        nm=tl.maximum(mx,tl.max(score,1));alpha=tl.exp(mx-nm)
        p=tl.exp(score-nm[:,None]);denom=denom*alpha+tl.sum(p,1)
        if REPAIR and LAZY_P and not FULL:
            if selected:
                p0,ps,p1,pr,pf=quantize_p(p,BM,BN,True,FAST_P)
            else:
                p0,ps,p1,pr,pf=quantize_p(p,BM,BN,False,FAST_P)
        else:
            p0,ps,p1,pr,pf=quantize_p(p,BM,BN,REPAIR,FAST_P)
        vi=((kh*tl.cdiv(N,128)+start*BN//128)*128+dd[:,None])*64+(start*BN%128)//2+tl.arange(0,BN//2)[None,:]
        vsi=((kh*tl.cdiv(N,128)+start*BN//128)*128+dd[:,None])*8+(start*BN%128)//16+tl.arange(0,BN//16)[None,:]
        if TRANSPOSE_V:
            vi=(kh*tl.cdiv(N,128)+start*BN//128)*128*64+((start*BN%128)//2+tl.arange(0,BN//2)[None,:])*128+dd[:,None]
            vsi=(kh*tl.cdiv(N,128)+start*BN//128)*128*8+((start*BN%128)//16+tl.arange(0,BN//16)[None,:])*128+dd[:,None]
        v0=tl.load(VC+vi);vs=tl.load(VS+vsi)
        vf=tl.load(VF+(kh*tl.cdiv(N,128)+start*BN//128)*128+dd)
        tile=tl.dot_scaled(p0,ps,'e2m1',v0.T,vs,'e2m1')
        if REPAIR:
            if selected:
                tile=tl.dot_scaled(p1,pr,'e2m1',v0.T,vs,'e2m1',tile)
                if TERMS>=3:
                    v1=tl.load(VC+HK*NP*64+vi);vr=tl.load(VS+HK*NP*8+vsi)
                    tile=tl.dot_scaled(p0,ps,'e2m1',v1.T,vr,'e2m1',tile)
                    if TERMS==4:tile=tl.dot_scaled(p1,pr,'e2m1',v1.T,vr,'e2m1',tile)
        acc=acc*alpha[:,None]+tile*pf[:,None]*vf[None,:];mx=nm
    if SPLITS==1:
        tl.store(O+(mi[:,None]*HQ+h)*128+dd[None,:],acc/denom[:,None],mi[:,None]<M)
    else:
        oi=((split*HQ+h)*M+mi[:,None])*128+dd[None,:]
        tl.store(O+oi,acc,mi[:,None]<M)
        si=((split*HQ+h)*M+mi)*2
        tl.store(STATS+si,mx,mi<M);tl.store(STATS+si+1,denom,mi<M)


@triton.jit
def combine(PARTS,STATS,OUT,M:tl.constexpr,H:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr):
    r=tl.program_id(0)*4+tl.arange(0,4);h=tl.program_id(1)
    # Keep split reduction local to each output element. A cross-warp tl.sum
    # lowering in the first prototype produced shared-memory WAW race reports.
    top=tl.full((4,),-float('inf'),tl.float32)
    for s in tl.static_range(SPLITS):
        mx=tl.load(STATS+((s*H+h)*M+r)*2,r<M,-float('inf'))
        top=tl.maximum(top,mx)
    top=tl.where(r<M,top,0.)
    di=tl.arange(0,128)
    accum=tl.full((4,128),0,tl.float32);total=tl.full((4,),0,tl.float32)
    for s in tl.static_range(SPLITS):
        si=((s*H+h)*M+r)*2
        mx=tl.load(STATS+si,r<M,-float('inf'));den=tl.load(STATS+si+1,r<M,0.)
        alpha=tl.exp(mx-top);total=total+den*alpha
        idx=((s*H+h)*M+r[:,None])*128+di[None,:]
        value=tl.load(PARTS+idx,r[:,None]<M,0.)
        accum=accum+value*alpha[:,None]
    tl.store(OUT+(r[:,None]*H+h)*128+di[None,:],accum/total[:,None],r[:,None]<M)


def pack_inputs(q,k,v,repair=True,need_energy=False,terms=3,transpose_v=False):
    import torch
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2]
    mp=triton.cdiv(m,16)*16;np=triton.cdiv(n,128)*128;planes=2 if repair else 1
    outputs=[];energy=torch.empty((hk,np//128,8),device=q.device,dtype=torch.float32) if need_energy else q
    for kind,x,hh,nn,padded in [('q',q,h,m,mp),('k',k,hk,n,np),('v',v,hk,n,np)]:
        local_repair=repair and (kind=='q' or terms>=3)
        planes=2 if local_repair else 1
        codes=torch.empty((planes,hh*padded,64),device=q.device,dtype=torch.uint8)
        scales=torch.empty((planes,hh*padded,8),device=q.device,dtype=torch.float8_e4m3fn)
        factors=torch.empty((hh*padded,),device=q.device,dtype=torch.float32)
        if kind=='v':
            pack_v[(8,np//128,hh)](x,codes,scales,factors,energy,nn,hh,x.stride(1),x.stride(2),np//128,local_repair,need_energy,transpose_v,num_warps=4)
        else:
            pack_qk[(triton.cdiv(padded,16),hh)](x,codes,scales,factors,nn,hh,x.stride(1),x.stride(2),padded,local_repair,num_warps=4)
        outputs.extend([codes,scales,factors])
    return (outputs,mp,np,energy) if need_energy else (outputs,mp,np)


@triton.jit
def select_tiles(Q,K,ENERGY,FLAGS,M:tl.constexpr,N:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                 Q_S:tl.constexpr,Q_H:tl.constexpr,K_S:tl.constexpr,K_H:tl.constexpr,
                 COUNT:tl.constexpr,TERMS:tl.constexpr,BT:tl.constexpr):
    h=tl.program_id(0);mt=tl.program_id(1);kh=h//(HQ//HK)
    # Sixteen MMA rows duplicate the eight fixed query samples.
    qi=tl.minimum(mt*128+(tl.arange(0,16)%8)*16,M-1)
    sample=tl.arange(0,BT*8);ki=sample//8*128+sample%8*16
    dd=tl.arange(0,128)
    q=tl.load(Q+qi[:,None]*Q_S+h*Q_H+dd[None,:])
    k=tl.load(K+tl.minimum(ki[:,None],N-1)*K_S+kh*K_H+dd[None,:])
    sim=tl.dot(q,k.T)*0.08838834764831845
    sim=tl.where(ki[None,:]<N,sim,-float('inf'))
    weights=tl.exp(sim-tl.max(sim,1)[:,None]);weights=weights/tl.sum(weights,1)[:,None]
    mass=tl.sum(tl.sum(tl.reshape(weights,(16,BT,8)),2),0)/16.
    nt=tl.cdiv(N,128);ti=tl.arange(0,BT)
    energy=tl.sqrt(tl.sum(tl.load(ENERGY+(kh*nt+ti[:,None])*8+tl.arange(0,8)[None,:],ti[:,None]<nt,0.),1))
    score=tl.where(ti<nt,mass*energy,-float('inf'))
    # Rank the small number of key tiles; deterministic lower-index tie break.
    rank=tl.sum(((score[None,:]>score[:,None])|((score[None,:]==score[:,None])&(ti[None,:]<ti[:,None]))).to(tl.int32),1)
    tl.store(FLAGS+(h*tl.cdiv(M,128)+mt)*nt+ti,tl.where(rank<COUNT,TERMS,1),ti<nt)


@triton.jit
def select_tiles_head(Q,K,ENERGY,FLAGS,M:tl.constexpr,N:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                 Q_S:tl.constexpr,Q_H:tl.constexpr,K_S:tl.constexpr,K_H:tl.constexpr,
                 COUNTS,TERMS:tl.constexpr,BT:tl.constexpr):
    h=tl.program_id(0);mt=tl.program_id(1);kh=h//(HQ//HK)
    COUNT=tl.load(COUNTS+h)
    # Sixteen MMA rows duplicate the eight fixed query samples.
    qi=tl.minimum(mt*128+(tl.arange(0,16)%8)*16,M-1)
    sample=tl.arange(0,BT*8);ki=sample//8*128+sample%8*16
    dd=tl.arange(0,128)
    q=tl.load(Q+qi[:,None]*Q_S+h*Q_H+dd[None,:])
    k=tl.load(K+tl.minimum(ki[:,None],N-1)*K_S+kh*K_H+dd[None,:])
    sim=tl.dot(q,k.T)*0.08838834764831845
    sim=tl.where(ki[None,:]<N,sim,-float('inf'))
    weights=tl.exp(sim-tl.max(sim,1)[:,None]);weights=weights/tl.sum(weights,1)[:,None]
    mass=tl.sum(tl.sum(tl.reshape(weights,(16,BT,8)),2),0)/16.
    nt=tl.cdiv(N,128);ti=tl.arange(0,BT)
    energy=tl.sqrt(tl.sum(tl.load(ENERGY+(kh*nt+ti[:,None])*8+tl.arange(0,8)[None,:],ti[:,None]<nt,0.),1))
    score=tl.where(ti<nt,mass*energy,-float('inf'))
    # Rank the small number of key tiles; deterministic lower-index tie break.
    rank=tl.sum(((score[None,:]>score[:,None])|((score[None,:]==score[:,None])&(ti[None,:]<ti[:,None]))).to(tl.int32),1)
    tl.store(FLAGS+(h*tl.cdiv(M,128)+mt)*nt+ti,tl.where(rank<COUNT,TERMS,1),ti<nt)

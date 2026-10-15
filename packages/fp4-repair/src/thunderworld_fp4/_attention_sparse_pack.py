"""Post-selector Q/K/V residual packing over exact current GQA support unions."""
import torch,triton
import triton.language as tl
from ._quantize import quantize


@triton.jit
def support_union(QFLAGS,PFLAGS,QMASK,KMASK,VMASK,MT:tl.constexpr,NT:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                  QREPAIR:tl.constexpr,QFULL:tl.constexpr,PREPAIR:tl.constexpr,PFULL:tl.constexpr,
                  QTERMS:tl.constexpr,PTERMS:tl.constexpr,BT:tl.constexpr,BR:tl.constexpr):
    kh=tl.program_id(0);ki=tl.arange(0,BT);r=tl.arange(0,BR)
    head=kh*(HQ//HK)+r//MT;row=r%MT
    valid=(r[:,None]<(HQ//HK)*MT)&(ki[None,:]<NT)
    if QFULL:q=tl.full((BR,BT),True,tl.int1)&valid
    elif QREPAIR:q=tl.load(QFLAGS+(head[:,None]*MT+row[:,None])*NT+ki[None,:],valid,1)>1
    else:q=tl.full((BR,BT),False,tl.int1)
    if PFULL:p=tl.full((BR,BT),True,tl.int1)&valid
    elif PREPAIR:p=tl.load(PFLAGS+(head[:,None]*MT+row[:,None])*NT+ki[None,:],valid,1)>1
    else:p=tl.full((BR,BT),False,tl.int1)
    tl.store(QMASK+head*MT+row,tl.sum(q.to(tl.int32),1)>0,r<(HQ//HK)*MT)
    tl.store(KMASK+kh*NT+ki,(tl.sum(q.to(tl.int32),0)>0)&(QTERMS>=3),ki<NT)
    tl.store(VMASK+kh*NT+ki,(tl.sum(p.to(tl.int32),0)>0)&(PTERMS>=3),ki<NT)


@triton.jit
def residual_qk(X,C,S,MASK,M:tl.constexpr,H:tl.constexpr,SX:tl.constexpr,SH:tl.constexpr,MP:tl.constexpr):
    rb=tl.program_id(0);h=tl.program_id(1)
    if tl.load(MASK+h*tl.cdiv(M,128)+rb//8):
        r=rb*16+tl.arange(0,16);d=tl.arange(0,128)
        x=tl.load(X+r[:,None]*SX+h*SH+d[None,:],r[:,None]<M,0).to(tl.float32)
        _,_,rp,rs,_=quantize(x,16,128,True)
        idx=(h*MP+r[:,None])*64+tl.arange(0,64)[None,:]
        si=(h*MP+r[:,None])*8+tl.arange(0,8)[None,:]
        tl.store(C+H*MP*64+idx,rp,r[:,None]<MP)
        tl.store(S+H*MP*8+si,rs,r[:,None]<MP)


@triton.jit
def residual_v(X,C,S,MASK,N:tl.constexpr,H:tl.constexpr,SX:tl.constexpr,SH:tl.constexpr,NT:tl.constexpr,TRANSPOSE:tl.constexpr):
    block=tl.program_id(1);h=tl.program_id(2)
    if tl.load(MASK+h*NT+block):
        di=tl.program_id(0)*16+tl.arange(0,16);ni=block*128+tl.arange(0,128)
        x=tl.load(X+ni[None,:]*SX+h*SH+di[:,None],ni[None,:]<N,0).to(tl.float32)
        _,_,rp,rs,_=quantize(x,16,128,True)
        row=(h*NT+block)*128+di
        idx=row[:,None]*64+tl.arange(0,64)[None,:]
        si=row[:,None]*8+tl.arange(0,8)[None,:]
        if TRANSPOSE:
            idx=(h*NT+block)*128*64+tl.arange(0,64)[None,:]*128+di[:,None]
            si=(h*NT+block)*128*8+tl.arange(0,8)[None,:]*128+di[:,None]
        tl.store(C+H*NT*128*64+idx,rp)
        tl.store(S+H*NT*128*8+si,rs)


def pack_selected(q,k,v,packed,mp,np,qflags,pflags,*,qrepair,qfull,prepair,pfull,qterms,pterms,transpose):
    m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2];mt,nt=triton.cdiv(m,128),triton.cdiv(n,128)
    qm=torch.empty((h,mt),device=q.device,dtype=torch.bool)
    km=torch.empty((hk,nt),device=q.device,dtype=torch.bool);vm=torch.empty_like(km)
    support_union[(hk,)](qflags,pflags,qm,km,vm,mt,nt,h,hk,qrepair,qfull,prepair,pfull,qterms,pterms,
                         triton.next_power_of_2(nt),triton.next_power_of_2((h//hk)*mt),num_warps=4)
    if qrepair:residual_qk[(triton.cdiv(mp,16),h)](q,*packed[:2],qm,m,h,q.stride(1),q.stride(2),mp,num_warps=4)
    if qrepair and qterms>=3:residual_qk[(triton.cdiv(np,16),hk)](k,*packed[3:5],km,n,hk,k.stride(1),k.stride(2),np,num_warps=4)
    if prepair and pterms>=3:residual_v[(8,nt,hk)](v,*packed[6:8],vm,n,hk,v.stride(1),v.stride(2),nt,transpose,num_warps=4)
    return dict(q_union=qm,k_union=km,v_union=vm)

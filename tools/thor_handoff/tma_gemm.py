"""Thor K128 repair GEMM with TMA operand prefetch and persistent TMEM.

This module is opt-in. It never patches the installed inference package. The
base/A interleave, W CSR order, scaling and final BF16 rounding are unchanged.
"""
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, TensorMemoryScalesLayout, allocate_tensor_memory,
    fence_async_shared, tcgen05_mma_scaled, tcgen05_commit, mbarrier, tma,
)


@gluon.jit
def mma(a,b,acc,asc,bsc,bar,phase,use_acc):
    tcgen05_mma_scaled(a,b.permute((1,0)),acc,asc,bsc,'e2m1','e2m1',use_acc=use_acc)
    tcgen05_commit(bar)
    mbarrier.wait(bar,phase)
    return phase^1


@gluon.jit
def load_operand(C, S, smem, stmem, rows, kb, K: gl.constexpr,
                 MP: gl.constexpr, PLANE: gl.constexpr, MASK: gl.constexpr,
                 COMPACT: gl.constexpr = False):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    LS: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    rr = gl.convert_layout(rows, gl.SliceLayout(1, L))
    rrs = gl.convert_layout(rows, gl.SliceLayout(1, LS))
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, L))
    ss = gl.arange(0, 8, layout=gl.SliceLayout(0, LS))
    if COMPACT:
        offsets = kb*128*64 + rr[:, None]*64 + kk[None, :]
        soffsets = kb*128*8 + rrs[:, None]*8 + ss[None, :]
    else:
        offsets = PLANE*MP*(K//2) + rr[:, None]*(K//2) + kb*64 + kk[None, :]
        soffsets = PLANE*MP*(K//16) + rrs[:, None]*(K//16) + kb*8 + ss[None, :]
    if MASK:
        values = gl.load(C+offsets, rr[:, None]<MP, 0)
        scales = gl.load(S+soffsets, rrs[:, None]<MP, 0.)
    else:
        values = gl.load(C+offsets)
        scales = gl.load(S+soffsets)
    smem.store(values)
    stmem.store(gl.convert_layout(scales, stmem.get_reg_layout()))


@gluon.jit
def kernel(AC, AS, WC, WS, WG, AG, AI, AM, WR, WRS, WI, OFF, Y,
         M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, MP: gl.constexpr,
         NA: gl.constexpr, HASW: gl.constexpr, CROSS: gl.constexpr,
         B: gl.constexpr, BM: gl.constexpr, INTERLEAVE: gl.constexpr,
         SPLIT_K: gl.constexpr = 1, DENSE_W: gl.constexpr = False,
         WLOC=None, SHARED_BANK: gl.constexpr = False, AD=None, RD=None, WD=None):
    gl.static_assert(B == 128 and BM == 128 and SPLIT_K == 1)
    gl.static_assert(INTERLEAVE or NA == 0 or NA == K//B)
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    SL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 64], gl.uint8)
    SL3:gl.constexpr=SL
    aa=gl.allocate_shared_memory(gl.uint8,[2,128,64],SL3)
    ar=gl.allocate_shared_memory(gl.uint8,[2,128,64],SL3)
    ww=gl.allocate_shared_memory(gl.uint8,[2,128,64],SL3)
    smem_a=aa.index(0);smem_w=ww.index(0)
    scale_r=allocate_tensor_memory(gl.float8e4nv,[128,8],TensorMemoryScalesLayout())
    loads=gl.allocate_shared_memory(gl.int64,[2,1],mbarrier.MBarrierLayout())
    mbarrier.init(loads.index(0),count=1);mbarrier.init(loads.index(1),count=1)
    scale_a = allocate_tensor_memory(gl.float8e4nv, [128, 8], TensorMemoryScalesLayout())
    scale_w = allocate_tensor_memory(gl.float8e4nv, [128, 8], TensorMemoryScalesLayout())
    acc = allocate_tensor_memory(gl.float32, [128, 128], TensorMemoryLayout([128, 128], col_stride=1))
    bar=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(bar,count=1)
    phase=0
    mb = gl.program_id(0)
    nb = gl.program_id(1)
    rr = gl.arange(0, 128, layout=gl.SliceLayout(1, L))
    mi = mb*128+rr
    ni = nb*128+rr
    first_selected=True if NA==K//128 else gl.load(AM+mb*(K//128)) if NA>0 else False
    if first_selected:mbarrier.expect(loads.index(0),24576)
    else:mbarrier.expect(loads.index(0),16384)
    tma.async_load(AD,[mb*128,0],loads.index(0),aa.index(0))
    tma.async_load(WD,[nb*128,0],loads.index(0),ww.index(0))
    if first_selected:tma.async_load(RD,[mb*128,0],loads.index(0),ar.index(0))
    selected=first_selected
    LS:gl.constexpr=gl.BlockedLayout([1,1],[4,8],[4,1],[1,0])
    mi_s=mb*128+gl.arange(0,128,layout=gl.SliceLayout(1,LS))
    ni_s=nb*128+gl.arange(0,128,layout=gl.SliceLayout(1,LS))
    ds=gl.arange(0,8,layout=gl.SliceLayout(0,LS))
    for kb in range(K//128):
        slot=kb%2
        mbarrier.wait(loads.index(slot),(kb//2)%2)
        next_selected=False
        if kb+1<K//128:
            ns=1-slot
            next_selected=True if NA==K//128 else gl.load(AM+mb*(K//128)+kb+1) if NA>0 else False
            if next_selected:mbarrier.expect(loads.index(ns),24576)
            else:mbarrier.expect(loads.index(ns),16384)
            tma.async_load(AD,[mb*128,(kb+1)*64],loads.index(ns),aa.index(ns))
            tma.async_load(WD,[nb*128,(kb+1)*64],loads.index(ns),ww.index(ns))
            if next_selected:tma.async_load(RD,[mb*128,(kb+1)*64],loads.index(ns),ar.index(ns))
        asc=gl.load(AS+mi_s[:,None]*(K//16)+kb*8+ds[None,:],mi_s[:,None]<MP,0.)
        wsc=gl.load(WS+ni_s[:,None]*(K//16)+kb*8+ds[None,:])
        scale_a.store(gl.convert_layout(asc,scale_a.get_reg_layout()))
        scale_w.store(gl.convert_layout(wsc,scale_w.get_reg_layout()))
        tcgen05_mma_scaled(aa.index(slot),ww.index(slot).permute((1,0)),acc,scale_a,scale_w,'e2m1','e2m1',use_acc=kb!=0)
        if NA>0:
            if selected:
                asc=gl.load(AS+MP*(K//16)+mi_s[:,None]*(K//16)+kb*8+ds[None,:],mi_s[:,None]<MP,0.)
                scale_r.store(gl.convert_layout(asc,scale_r.get_reg_layout()))
                tcgen05_mma_scaled(ar.index(slot),ww.index(slot).permute((1,0)),acc,scale_r,scale_w,'e2m1','e2m1',use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar,phase)
        phase=phase^1
        if kb+1<K//128:selected=next_selected
    mbarrier.invalidate(loads.index(0));mbarrier.invalidate(loads.index(1))
    if HASW:
        begin = nb*(K//128) if DENSE_W else gl.load(OFF+nb)
        end = (nb+1)*(K//128) if DENSE_W else gl.load(OFF+nb+1)
        for slot in range(begin, end):
            kb = slot-begin if DENSE_W else gl.load(WI+slot)
            storage_slot = gl.load(WLOC+slot) if SHARED_BANK else slot
            load_operand(AC, AS, smem_a, scale_a, mi, kb, K, MP, 0, True)
            load_operand(WR, WRS, smem_w, scale_w, rr, storage_slot, K, N, 0, False, True)
            fence_async_shared()
            phase=mma(smem_a,smem_w,acc,scale_a,scale_w,bar,phase,True)
            if CROSS and NA > 0:
                selected = gl.load(AM+mb*(K//128)+kb)
                if selected:
                    load_operand(AC, AS, smem_a, scale_a, mi, kb, K, MP, 1, True)
                    fence_async_shared()
                    phase=mma(smem_a,smem_w,acc,scale_a,scale_w,bar,phase,True)
    mbarrier.invalidate(bar)
    values = acc.load()
    OL: gl.constexpr = values.type.layout
    om = mb*128+gl.arange(0, 128, layout=gl.SliceLayout(1, OL))
    on = nb*128+gl.arange(0, 128, layout=gl.SliceLayout(0, OL))
    values = values*(gl.load(WG)*gl.load(AG))
    gl.store(Y+om[:, None]*N+on[None, :], values, om[:, None]<M)


class Launcher:
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
            if kwargs.get('num_warps',4)!=4:raise ValueError('TMA repair GEMM is qualified with four warps')
            # The double buffer and barrier phases are explicit in this kernel.
            kwargs['num_stages']=1
            ac=args[0];wc=args[2]
            layout=gl.NVMMASharedLayout.get_default_for([128,64],gl.uint8)
            kwargs.update(AD=TensorDescriptor.from_tensor(ac[0],[128,64],layout),
                RD=TensorDescriptor.from_tensor(ac[1] if ac.shape[0]>1 else ac[0],[128,64],layout),
                WD=TensorDescriptor.from_tensor(wc,[128,64],layout))
            return kernel[grid](*args,**kwargs)
        return launch

gemm=Launcher()

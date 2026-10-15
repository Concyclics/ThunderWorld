"""Thor K128 repair GEMM with explicit prefetch/MMA overlap.

This module is opt-in. It never patches the installed inference package. The
base/A interleave, W CSR order, scaling and final BF16 rounding are unchanged.
"""
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, TensorMemoryScalesLayout, allocate_tensor_memory,
    fence_async_shared, tcgen05_mma_scaled, tcgen05_commit, mbarrier,
)


@gluon.jit
def mma(a,b,acc,asc,bsc,bar,phase,use_acc):
    tcgen05_mma_scaled(a,b.permute((1,0)),acc,asc,bsc,'e2m1','e2m1',use_acc=use_acc)
    tcgen05_commit(bar)
    mbarrier.wait(bar,phase)
    return phase^1


@gluon.jit
def read_operand(C, S, rows, kb, K: gl.constexpr,
                 MP: gl.constexpr, PLANE: gl.constexpr, MASK: gl.constexpr,
                 COMPACT: gl.constexpr = False):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [gl.num_warps(), 1], [1, 0])
    LS: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [gl.num_warps(), 1], [1, 0])
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
    return values,scales


@gluon.jit
def load_operand(C,S,smem,stmem,rows,kb,K:gl.constexpr,MP:gl.constexpr,PLANE:gl.constexpr,MASK:gl.constexpr,COMPACT:gl.constexpr=False):
    values,scales=read_operand(C,S,rows,kb,K,MP,PLANE,MASK,COMPACT)
    smem.store(values)
    stmem.store(gl.convert_layout(scales,stmem.get_reg_layout()))


@gluon.jit
def gemm(AC, AS, WC, WS, WG, AG, AI, AM, WR, WRS, WI, OFF, Y,
         M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, MP: gl.constexpr,
         NA: gl.constexpr, HASW: gl.constexpr, CROSS: gl.constexpr,
         B: gl.constexpr, BM: gl.constexpr, INTERLEAVE: gl.constexpr,
         SPLIT_K: gl.constexpr = 1, DENSE_W: gl.constexpr = False,
         WLOC=None, SHARED_BANK: gl.constexpr = False):
    gl.static_assert(B == 128 and BM == 128 and SPLIT_K == 1)
    gl.static_assert(INTERLEAVE or NA == 0 or NA == K//B)
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [gl.num_warps(), 1], [1, 0])
    SL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 64], gl.uint8)
    smem_a = gl.allocate_shared_memory(gl.uint8, [128, 64], SL)
    smem_w = gl.allocate_shared_memory(gl.uint8, [128, 64], SL)
    smem_r = gl.allocate_shared_memory(gl.uint8, [128,64], SL)
    scale_r = allocate_tensor_memory(gl.float8e4nv,[128,8],TensorMemoryScalesLayout())
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
    av,asc=read_operand(AC,AS,mi,0,K,MP,0,True)
    wv,wsc=read_operand(WC,WS,ni,0,K,N,0,False)
    for kb in range(K//128):
        smem_a.store(av);smem_w.store(wv)
        scale_a.store(gl.convert_layout(asc,scale_a.get_reg_layout()))
        scale_w.store(gl.convert_layout(wsc,scale_w.get_reg_layout()))
        fence_async_shared()
        tcgen05_mma_scaled(smem_a,smem_w.permute((1,0)),acc,scale_a,scale_w,'e2m1','e2m1',use_acc=kb!=0)
        if NA>0:
            selected=True if NA==K//128 else gl.load(AM+mb*(K//128)+kb)
            if selected:
                # Separate operand storage allows CUDA-core loads and packing
                # into SMEM/TMEM to run while the base MMA is still in flight.
                load_operand(AC,AS,smem_r,scale_r,mi,kb,K,MP,1,True)
                fence_async_shared()
                tcgen05_mma_scaled(smem_r,smem_w.permute((1,0)),acc,scale_r,scale_w,'e2m1','e2m1',use_acc=True)
        tcgen05_commit(bar)
        if kb+1<K//128:
            # Fetch the next dense operands before waiting for the current
            # accumulator. Never overwrite storage consumed by in-flight MMA.
            av,asc=read_operand(AC,AS,mi,kb+1,K,MP,0,True)
            wv,wsc=read_operand(WC,WS,ni,kb+1,K,N,0,False)
        mbarrier.wait(bar,phase)
        phase=phase^1
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

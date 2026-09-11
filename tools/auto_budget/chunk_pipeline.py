"""Two-slot M-chunk packing/GEMM pipeline with one full-input scale and mask.

Preserves K accumulation order. Chunks start at M128 boundaries. The packing
stream waits for a slot's previous GEMM before reuse; the compute stream waits
for each chunk's producer. No cross-step caching or per-chunk global scaling.
"""
import torch
import triton
from dynamic import TIERS,load_native
from thunderworld_fp4._linear_kernels import scaled_amax,finish_amax,pack_activation


class ChunkPipeline:
    def __init__(self,linear,x,tier,chunk):
        if chunk not in (128,256,512):raise ValueError('M128-aligned chunk required')
        self.linear=linear;self.tier=tier;self.chunk=chunk;self.shape=tuple(x.shape)
        self.side=torch.cuda.Stream();a,_=TIERS[tier];planes=2 if a else 1;k=x.shape[1]
        self.slots=[(torch.empty((planes,chunk,k//2),device=x.device,dtype=torch.uint8),
                     torch.empty((planes,chunk,k//16),device=x.device,dtype=torch.float8_e4m3fn)) for _ in range(2)]
        self.ready=[torch.cuda.Event() for _ in range(2)];self.free=[torch.cuda.Event() for _ in range(2)]

    def __call__(self,x):
        if tuple(x.shape)!=self.shape or not x.is_contiguous():raise ValueError('Pipeline shape/layout mismatch')
        linear=self.linear;linear.prepare(x,self.tier);op=linear.op;m,k=x.shape;a,_=TIERS[self.tier]
        partial=torch.empty((triton.cdiv(m*k,4096),),device=x.device,dtype=torch.float32)
        factor=torch.empty((),device=x.device,dtype=torch.float32)
        scaled_amax[(partial.numel(),)](x,op.d,partial,m,k,4096,num_warps=4)
        finish_amax[(1,)](partial,factor,partial.numel(),triton.next_power_of_2(partial.numel()),num_warps=4)
        out=torch.empty((m,op.n),device=x.device,dtype=x.dtype)
        main=torch.cuda.current_stream();self.side.wait_stream(main)
        for j,start in enumerate(range(0,m,self.chunk)):
            slot=j%2;size=min(self.chunk,m-start);mp=triton.cdiv(size,16)*16
            codes,scales=self.slots[slot];mask=op.am[start//128:]
            with torch.cuda.stream(self.side):
                if j>=2:self.side.wait_event(self.free[slot])
                pack_activation[(triton.cdiv(size,16),k//128)](x[start:],op.d,factor,mask,codes,scales,size,k,mp,a>0,a==1,num_warps=4)
                self.ready[slot].record()
            main.wait_event(self.ready[slot])
            load_native().gemm[(triton.cdiv(size,op.bm),op.n//128,1)](
                codes,scales,op.wc,op.ws,op.wg,factor,op.ai,mask,op.wr,op.wrs,op.wi,op.off,out[start:],
                size,op.n,k,mp,op.na,op.hasw,False,128,op.bm,True,1,num_warps=op.num_warps,num_stages=op.num_stages)
            self.free[slot].record()
        main.wait_stream(self.side)
        return out

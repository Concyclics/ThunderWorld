"""Experimental full-W residual GEMM layouts; no requantization or cross term.

Plane layout retains the original K order. Pair layout folds two weight planes
into an expanded reduction tile and aliases A0 loads, without a 2K activation
buffer. Both retain 2K base+W work and an optional A1W0 loop contribution.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def merged_gemm(AC,AS,WC,WS,AG,WG,AM,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,
                BM:tl.constexpr,A:tl.constexpr,PAIR:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*128+tl.arange(0,128)
    acc=tl.full((BM,128),0,tl.float32)
    if PAIR:
        dc=tl.arange(0,128);ds=tl.arange(0,16)
        for kb in range(K//128):
            ac=tl.load(AC+mi[:,None]*(K//2)+kb*64+dc[None,:]%64,mi[:,None]<MP,0)
            asc=tl.load(AS+mi[:,None]*(K//16)+kb*8+ds[None,:]%8,mi[:,None]<MP,0.)
            wc=tl.load(WC+ni[:,None]*K+kb*128+dc[None,:])
            ws=tl.load(WS+ni[:,None]*(K//8)+kb*16+ds[None,:])
            acc=tl.dot_scaled(ac,asc,'e2m1',wc.T,ws,'e2m1',acc)
            if A>0:
                if A==2:selected=True
                else:selected=tl.load(AM+(tl.program_id(0)*BM//128)*(K//128)+kb)
                if selected:
                    d0=tl.arange(0,64);s0=tl.arange(0,8)
                    ar=tl.load(AC+MP*(K//2)+mi[:,None]*(K//2)+kb*64+d0[None,:],mi[:,None]<MP,0)
                    ars=tl.load(AS+MP*(K//16)+mi[:,None]*(K//16)+kb*8+s0[None,:],mi[:,None]<MP,0.)
                    w0=tl.load(WC+ni[:,None]*K+kb*128+d0[None,:]);w0s=tl.load(WS+ni[:,None]*(K//8)+kb*16+s0[None,:])
                    acc=tl.dot_scaled(ar,ars,'e2m1',w0.T,w0s,'e2m1',acc)
    else:
        dc=tl.arange(0,64);ds=tl.arange(0,8)
        for vk in range(2*K//128):
            kb=vk%(K//128)
            ac=tl.load(AC+mi[:,None]*(K//2)+kb*64+dc[None,:],mi[:,None]<MP,0)
            asc=tl.load(AS+mi[:,None]*(K//16)+kb*8+ds[None,:],mi[:,None]<MP,0.)
            wc=tl.load(WC+ni[:,None]*K+vk*64+dc[None,:]);ws=tl.load(WS+ni[:,None]*(K//8)+vk*8+ds[None,:])
            acc=tl.dot_scaled(ac,asc,'e2m1',wc.T,ws,'e2m1',acc)
            if A>0:
                if vk<K//128:
                    if A==2:selected=True
                    else:selected=tl.load(AM+(tl.program_id(0)*BM//128)*(K//128)+kb)
                    if selected:
                        ar=tl.load(AC+MP*(K//2)+mi[:,None]*(K//2)+kb*64+dc[None,:],mi[:,None]<MP,0)
                        ars=tl.load(AS+MP*(K//16)+mi[:,None]*(K//16)+kb*8+ds[None,:],mi[:,None]<MP,0.)
                        acc=tl.dot_scaled(ar,ars,'e2m1',wc.T,ws,'e2m1',acc)
    tl.store(Y+mi[:,None]*N+ni[None,:],acc*(tl.load(WG)*tl.load(AG)),mi[:,None]<M)

class MergedWeight:
    """Offline layout transform, full W repair only; current A supports are read."""
    def __init__(self,op,pair=False):
        if not op.hasw or op.cross or op.b!=128 or op.split_k!=1 or not op.interleave:
            raise ValueError('Requires full W, K128, interleaved A, no cross or split-K')
        n,k=op.n,op.k
        expected=torch.arange(k//128,device=op.wi.device).repeat(n//128)
        if not torch.equal(op.wi,expected) or not torch.equal(op.off,torch.arange(n//128+1,device=op.off.device)*(k//128)):
            raise ValueError('Sparse W supports need a different layout')
        w1=op.wr.reshape(n//128,k//128,128,64).permute(0,2,1,3).reshape(n,k//2)
        s1=op.wrs.view(torch.uint8).reshape(n//128,k//128,128,8).permute(0,2,1,3).reshape(n,k//16)
        if pair:
            self.wc=torch.stack([op.wc.reshape(n,k//128,64),w1.reshape(n,k//128,64)],2).reshape(n,k).contiguous()
            self.ws=torch.stack([op.ws.view(torch.uint8).reshape(n,k//128,8),s1.reshape(n,k//128,8)],2).reshape(n,k//8).contiguous().view(torch.float8_e4m3fn)
        else:
            self.wc=torch.cat([op.wc,w1],1).contiguous();self.ws=torch.cat([op.ws.view(torch.uint8),s1],1).contiguous().view(torch.float8_e4m3fn)
        self.op=op;self.pair=pair;self.bm=op.bm;self.stages=op.num_stages
    def run_packed(self,x,packed):
        ac,sc,ag,mp=packed;m=x.shape[0];op=self.op
        if not op.hasw or op.cross:raise ValueError('Full W repair required')
        a=0 if op.na==0 else 2 if op.na==op.k//128 else 1
        y=torch.empty((m,op.n),device=x.device,dtype=x.dtype)
        merged_gemm[(triton.cdiv(m,self.bm),op.n//128)](ac,sc,self.wc,self.ws,ag,op.wg,op.am,y,m,op.n,op.k,mp,self.bm,a,self.pair,num_warps=4,num_stages=self.stages)
        return y
    def __call__(self,x):return self.run_packed(x,self.op.pack(x))

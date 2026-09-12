"""Native compact-index repair loops; offline packed weight CSR tiles.

All repairs use the same outer factors as the dense base, permitting one
FP32 accumulator. Residual W storage is only the selected N128/K_B tiles.
"""
import torch,triton
import triton.language as tl
from .linear import _pack

@triton.jit
def gemm(AC,AS,WC,WS,WG,AG,AI,AM,WR,WRS,WI,OFF,Y,
         M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,
         NA:tl.constexpr,HASW:tl.constexpr,CROSS:tl.constexpr,B:tl.constexpr,BM:tl.constexpr,INTERLEAVE:tl.constexpr,SPLIT_K:tl.constexpr=1,DENSE_W:tl.constexpr=False,WLOC=None,SHARED_BANK:tl.constexpr=False):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);nb=tl.program_id(1);ni=nb*128+tl.arange(0,128)
    dk=tl.arange(0,B//2);ds=tl.arange(0,B//16);acc=tl.full((BM,128),0,tl.float32)
    part=tl.program_id(2)
    if SPLIT_K==1:begin_k=0;end_k=K//B
    else:begin_k=part*tl.cdiv(K//B,SPLIT_K);end_k=tl.minimum((part+1)*tl.cdiv(K//B,SPLIT_K),K//B)
    for kb in range(begin_k,end_k):
        ai=mi[:,None]*(K//2)+kb*(B//2)+dk[None,:];asi=mi[:,None]*(K//16)+kb*(B//16)+ds[None,:]
        wi=ni[:,None]*(K//2)+kb*(B//2)+dk[None,:];wsi=ni[:,None]*(K//16)+kb*(B//16)+ds[None,:]
        a=tl.load(AC+ai,mi[:,None]<MP,0);s=tl.load(AS+asi,mi[:,None]<MP,0.)
        w=tl.load(WC+wi);ws=tl.load(WS+wsi)
        acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
        if NA==K//B or (INTERLEAVE and NA>0):
            if NA==K//B:selected=True
            else:selected=tl.load(AM+(tl.program_id(0)*BM//128)*(K//B)+kb)
            if selected:
                a=tl.load(AC+MP*(K//2)+ai,mi[:,None]<MP,0);s=tl.load(AS+MP*(K//16)+asi,mi[:,None]<MP,0.)
                acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
    if NA>0 and NA<K//B and not INTERLEAVE:
        for j in range(NA):
            kb=tl.load(AI+(tl.program_id(0)*BM//128)*NA+j)
            if SPLIT_K==1 or (kb>=begin_k and kb<end_k):
                ai=MP*(K//2)+mi[:,None]*(K//2)+kb*(B//2)+dk[None,:]
                asi=MP*(K//16)+mi[:,None]*(K//16)+kb*(B//16)+ds[None,:]
                wi=ni[:,None]*(K//2)+kb*(B//2)+dk[None,:];wsi=ni[:,None]*(K//16)+kb*(B//16)+ds[None,:]
                a=tl.load(AC+ai,mi[:,None]<MP,0);s=tl.load(AS+asi,mi[:,None]<MP,0.)
                w=tl.load(WC+wi);ws=tl.load(WS+wsi)
                acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
    if HASW:
        if DENSE_W:begin=nb*(K//B);end=(nb+1)*(K//B)
        else:begin=tl.load(OFF+nb);end=tl.load(OFF+nb+1)
        for slot in range(begin,end):
            if DENSE_W:kb=slot-begin
            else:kb=tl.load(WI+slot)
            if SPLIT_K==1 or (kb>=begin_k and kb<end_k):
                storage_slot=tl.load(WLOC+slot) if SHARED_BANK else slot
                ai=mi[:,None]*(K//2)+kb*(B//2)+dk[None,:];asi=mi[:,None]*(K//16)+kb*(B//16)+ds[None,:]
                a=tl.load(AC+ai,mi[:,None]<MP,0);s=tl.load(AS+asi,mi[:,None]<MP,0.)
                w=tl.load(WR+storage_slot*128*(B//2)+tl.arange(0,128)[:,None]*(B//2)+dk[None,:])
                ws=tl.load(WRS+storage_slot*128*(B//16)+tl.arange(0,128)[:,None]*(B//16)+ds[None,:])
                acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
                if CROSS and NA>0:
                    selected=tl.load(AM+(tl.program_id(0)*BM//128)*(K//B)+kb)
                    if selected:
                        a=tl.load(AC+MP*(K//2)+ai,mi[:,None]<MP,0);s=tl.load(AS+MP*(K//16)+asi,mi[:,None]<MP,0.)
                        acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
    if SPLIT_K==1:
        tl.store(Y+mi[:,None]*N+ni[None,:],acc*(tl.load(WG)*tl.load(AG)),mi[:,None]<M)
    else:
        tl.store(Y+part*M*N+mi[:,None]*N+ni[None,:],acc,mi[:,None]<M)


@triton.jit
def reduce_split_k(PARTS,Y,WG,AG,SIZE:tl.constexpr,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.full((BLOCK,),0,tl.float32)
    for part in tl.static_range(SPLITS):
        acc+=tl.load(PARTS+part*SIZE+idx,idx<SIZE,0.)
    tl.store(Y+idx,acc*(tl.load(WG)*tl.load(AG)),idx<SIZE)

class Native:
    def __init__(self,wc,ws,wg,d,am,wm,b=128,cross=False,bm=32,interleave=False,num_warps=4,num_stages=1,split_k=1):
        if b not in (64,128,256) or bm not in (32,64,128):
            raise ValueError("validated experimental tiles: K64/128/256, M32/64/128")
        if not wc.is_cuda or torch.cuda.get_device_capability(wc.device)!=(12,0):
            raise ValueError("native experiment requires SM120")
        if am.dtype!=torch.bool or wm.dtype!=torch.bool:
            raise ValueError("repair supports must be boolean")
        if num_warps not in (4,8) or num_stages not in (1,2,3,4):
            raise ValueError("supported launch settings: warps 4/8, stages 1/2/3/4")
        self.num_warps=num_warps;self.num_stages=num_stages
        if type(split_k) is not int or split_k not in (1,2,4,8):
            raise ValueError("split_k must be 1,2,4,8")
        self.split_k=split_k
        self.b=b;self.bm=bm;self.interleave=interleave;self.cross=cross;self.d=d;self.wc=wc[0].contiguous().clone();self.ws=ws[0].contiguous().clone();self.wg=wg
        self.n,self.k=self.wc.shape[0],self.wc.shape[1]*2
        # All host inspection and sparse weight gathering occurs offline.
        self.am=am.reshape(-1,self.k//b,b).any(-1).contiguous()
        counts=self.am.sum(-1)
        assert bool((counts==counts[0]).all())
        self.na=int(counts[0]);self.ai=self.am.nonzero()[:,1].reshape(self.am.shape[0],self.na).int().contiguous()
        wm=wm.reshape(self.n//128,128,self.k//b,b).any((1,3));pairs=wm.nonzero();self.hasw=bool(pairs.numel())
        self.dense_w=bool(wm.all())
        self.off=torch.cat([torch.zeros(1,device=wc.device,dtype=torch.int32),wm.sum(-1).int().cumsum(0).int()])
        self.wi=pairs[:,1].int().contiguous()
        self.wloc=self.wi;self.shared_bank=False
        tiles=wc[1].reshape(self.n//128,128,self.k//b,b//2).permute(0,2,1,3)
        scales=ws[1].view(torch.uint8).reshape(self.n//128,128,self.k//b,b//16).permute(0,2,1,3)
        self.wr=tiles[pairs[:,0],pairs[:,1]].contiguous()
        self.wrs=scales[pairs[:,0],pairs[:,1]].contiguous().view(torch.float8_e4m3fn)
        self.union=self.am.repeat_interleave(b,1).reshape(self.am.shape[0],self.k//128,128).any(-1).contiguous()
    def pack(self,x):
        return _pack(x,self.d,self.na>0,self.na==self.k//self.b,self.union)
    def run_packed(self,x,packed):
        ac,sc,ag,mp=packed;m=x.shape[0]
        out=torch.empty((m,self.n),device=x.device,dtype=x.dtype)
        parts=out if self.split_k==1 else torch.empty((self.split_k,m,self.n),device=x.device,dtype=torch.float32)
        gemm[(triton.cdiv(m,self.bm),self.n//128,self.split_k)](ac,sc,self.wc,self.ws,self.wg,ag,self.ai,self.am,self.wr,self.wrs,self.wi,self.off,parts,m,self.n,self.k,mp,self.na,self.hasw,self.cross,self.b,self.bm,self.interleave,self.split_k,self.dense_w,WLOC=self.wloc,SHARED_BANK=self.shared_bank,num_warps=self.num_warps,num_stages=self.num_stages)
        if self.split_k>1:
            reduce_split_k[(triton.cdiv(m*self.n,512),)](parts,out,self.wg,ag,m*self.n,self.split_k,512,num_warps=4)
        return out
    def __call__(self,x):
        if x.ndim!=2 or x.shape[1]!=self.k or x.shape[0]<=0 or triton.cdiv(x.shape[0],128)!=self.am.shape[0]:
            raise ValueError("activation shape must match the prepared M128 support and K")
        if x.dtype!=torch.bfloat16 or x.device!=self.wc.device or not x.is_contiguous():
            raise ValueError("expected contiguous BF16 activation on the packed-weight device")
        return self.run_packed(x,self.pack(x))
    @property
    def residual_weight_bytes(self):return sum(t.numel()*t.element_size() for t in [self.wr,self.wrs,self.wi,self.off]) if self.hasw else 0

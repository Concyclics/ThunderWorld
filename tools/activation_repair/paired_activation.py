"""Combine A0/A1 inside K256 scaled dot; W0 loads alias, no duplicated W buffer."""
import torch,triton
import triton.language as tl
from packing import pack

@triton.jit
def gemm(AC,AS,WC,WS,WR,WRS,WI,OFF,AM,AG,WG,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,BM:tl.constexpr,FULL:tl.constexpr,HASW:tl.constexpr,PACKED:tl.constexpr):
 mi=tl.program_id(0)*BM+tl.arange(0,BM);nb=tl.program_id(1);ni=nb*128+tl.arange(0,128)
 dc=tl.arange(0,128);ds=tl.arange(0,16);d0=tl.arange(0,64);s0=tl.arange(0,8)
 acc=tl.full((BM,128),0,tl.float32)
 for kb in range(K//128):
  if FULL:active=True
  else:active=tl.load(AM+(tl.program_id(0)*BM//128)*(K//128)+kb)
  if active:
   if PACKED:
    ai=mi[:,None]*K+kb*128+dc[None,:];si=mi[:,None]*(K//8)+kb*16+ds[None,:]
   else:
    ai=(dc[None,:]//64)*MP*(K//2)+mi[:,None]*(K//2)+kb*64+dc[None,:]%64
    si=(ds[None,:]//8)*MP*(K//16)+mi[:,None]*(K//16)+kb*8+ds[None,:]%8
   a=tl.load(AC+ai,mi[:,None]<MP,0);s=tl.load(AS+si,mi[:,None]<MP,0.)
   w=tl.load(WC+ni[:,None]*(K//2)+kb*64+dc[None,:]%64);ws=tl.load(WS+ni[:,None]*(K//16)+kb*8+ds[None,:]%8)
   acc=tl.dot_scaled(a,s,'e2m1',w.T,ws,'e2m1',acc)
  else:
   if PACKED:ai_base=mi[:,None]*K+kb*128+d0[None,:];si_base=mi[:,None]*(K//8)+kb*16+s0[None,:]
   else:ai_base=mi[:,None]*(K//2)+kb*64+d0[None,:];si_base=mi[:,None]*(K//16)+kb*8+s0[None,:]
   a_base=tl.load(AC+ai_base,mi[:,None]<MP,0);s_base=tl.load(AS+si_base,mi[:,None]<MP,0.)
   w_base=tl.load(WC+ni[:,None]*(K//2)+kb*64+d0[None,:]);ws_base=tl.load(WS+ni[:,None]*(K//16)+kb*8+s0[None,:])
   acc=tl.dot_scaled(a_base,s_base,'e2m1',w_base.T,ws_base,'e2m1',acc)
 if HASW:
  begin=tl.load(OFF+nb);end=tl.load(OFF+nb+1)
  for slot in range(begin,end):
   kb=tl.load(WI+slot)
   if PACKED:ai_weight=mi[:,None]*K+kb*128+d0[None,:];si_weight=mi[:,None]*(K//8)+kb*16+s0[None,:]
   else:ai_weight=mi[:,None]*(K//2)+kb*64+d0[None,:];si_weight=mi[:,None]*(K//16)+kb*8+s0[None,:]
   a_weight=tl.load(AC+ai_weight,mi[:,None]<MP,0);s_weight=tl.load(AS+si_weight,mi[:,None]<MP,0.)
   w_weight=tl.load(WR+slot*128*64+tl.arange(0,128)[:,None]*64+d0[None,:]);ws_weight=tl.load(WRS+slot*128*8+tl.arange(0,128)[:,None]*8+s0[None,:])
   acc=tl.dot_scaled(a_weight,s_weight,'e2m1',w_weight.T,ws_weight,'e2m1',acc)
 tl.store(Y+mi[:,None]*N+ni[None,:],acc*(tl.load(AG)*tl.load(WG)),mi[:,None]<M)

class PairedActivation:
 def __init__(self,op,packed=False,base=1.,res=1.,gain=1.):
  if op.b!=128 or op.cross or op.split_k!=1 or not op.interleave:raise ValueError('K128 interleaved, no cross/split-K required')
  self.op=op;self.packed=packed;self.base=base;self.res=res;self.gain=gain;self.bm=op.bm;self.stages=op.num_stages
 def pack(self,x):
  op=self.op
  if not self.packed and self.base==self.res==self.gain==1.:return op.pack(x)
  return pack(x,op.d,True,op.na==op.k//128,op.am,paired=self.packed,base=self.base,res=self.res,gain=self.gain)
 def run_packed(self,x,p):
  op=self.op
  if op.na<=0:raise ValueError('A repair required')
  c,s,f,mp=p;m=x.shape[0];y=torch.empty((m,op.n),device=x.device,dtype=x.dtype)
  gemm[(triton.cdiv(m,self.bm),op.n//128)](c,s,op.wc,op.ws,op.wr,op.wrs,op.wi,op.off,op.am,f,op.wg,y,m,op.n,op.k,mp,self.bm,op.na==op.k//128,op.hasw,self.packed,num_warps=4,num_stages=self.stages)
  return y
 def __call__(self,x):return self.run_packed(x,self.pack(x))

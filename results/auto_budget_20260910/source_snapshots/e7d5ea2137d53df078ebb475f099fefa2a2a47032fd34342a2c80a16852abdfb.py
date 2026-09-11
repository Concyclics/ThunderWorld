"""Bounded phase/branch budgets with GPU-resident current-input tile selection."""
import math
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/cosmos_e2e'))
from adapter import load_native
from thunderworld_fp4.linear import _pack

TIERS={'base':(0.,False),'a50':(.5,False),'act2':(1.,False),
       'wgt2':(0.,True),'a50w':(.5,True),'dual':(1.,True)}
UNIFORM={'uniform25':'a50','uniform50':'act2','uniform75':'a50w',
         'ffn_base':'base','ffn_act2':'act2','ffn_wgt2':'wgt2','ffn_dual':'dual'}
BOUNDS=(0.,.2,.5,.8,1.)


def phase(step,steps):
    return min(3,sum(step>=math.ceil(steps*x) for x in BOUNDS[1:-1]))


@triton.jit
def select_tiles(X,D,MASK,M:tl.constexpr,K:tl.constexpr,BT:tl.constexpr,NA:tl.constexpr):
    mb=tl.program_id(0);kt=tl.arange(0,BT)
    rows=tl.minimum(mb*128+tl.arange(0,8)*16,M-1)
    cols=kt[:,None]*128+tl.arange(0,8)[None,:]*16
    x=tl.load(X+rows[None,:,None]*K+cols[:,None,:],cols[:,None,:]<K,0).to(tl.float32)
    d=tl.load(D+cols,cols<K,0)
    x=(x*d[:,None,:]).to(tl.bfloat16).to(tl.float32)
    energy=tl.sum(tl.sum(x*x,2),1)
    energy=tl.where(kt<K//128,energy,-float('inf'))
    rank=tl.sum(((energy[None,:]>energy[:,None])|((energy[None,:]==energy[:,None])&(kt[None,:]<kt[:,None]))).to(tl.int32),1)
    tl.store(MASK+mb*(K//128)+kt,rank<NA,kt<K//128)


class BudgetLinear(torch.nn.Module):
    def __init__(self,source,controller=None,name=''):
        super().__init__();self.controller=controller;self.name=name
        if source.bias is not None:raise ValueError('Bias-free FFNs required')
        self.in_features=source.in_features;self.out_features=source.out_features
        w=source.weight.detach().cuda().contiguous();n,k=w.shape
        d=torch.ones(k,device=w.device)
        c,s,g,_=_pack(w,d,True,True,w)
        self.op=load_native().Native(c,s,g,d,torch.ones((1,k),device=w.device,dtype=torch.bool),
            torch.ones((n,k),device=w.device,dtype=torch.bool),bm=128,interleave=True,num_stages=2)
        # One shared W0/W1 allocation for every tier; no per-step re-packing.
        self.masks={};self.warmed_shapes=set();self.tier='dual';self.calls=0;self.selected_extra_flops=0;self.base_flops=0

    def prepare(self,x,tier):
        if tier not in TIERS:raise ValueError(tier)
        a,w=TIERS[tier];m,k=x.shape;mb=triton.cdiv(m,128)
        if mb not in self.masks:self.masks[mb]=torch.empty((mb,k//128),device=x.device,dtype=torch.bool)
        self.op.na=int(a*(k//128));self.op.hasw=w
        self.op.am=self.op.union=self.masks[mb]
        # INTERLEAVE=True does not read compact AI; no dynamic nonzero/CPU sync.
        if 0<self.op.na<k//128:
            select_tiles[(mb,)](x,self.op.d,self.op.am,m,k,triton.next_power_of_2(k//128),self.op.na,num_warps=4)

    def forward(self,x):
        shape=x.shape;flat=x.reshape(-1,self.in_features).contiguous()
        tier=self.tier if self.controller is None else self.controller.resolve(self.name)
        if self.controller is not None and tuple(flat.shape) not in self.warmed_shapes:
            if not self.controller.warming:raise RuntimeError('Unwarmed budget input shape')
            for warmtier in TIERS:
                self.prepare(flat,warmtier);self.op(flat)
            self.warmed_shapes.add(tuple(flat.shape))
        self.prepare(flat,tier)
        self.calls+=1
        flops=2*flat.shape[0]*self.in_features*self.out_features
        self.base_flops+=flops;self.selected_extra_flops+=flops*(TIERS[tier][0]+TIERS[tier][1])
        return self.op(flat).reshape(*shape[:-1],self.out_features)


class BudgetController:
    def __init__(self,study,policy):
        self.study=study;self.policy=policy;self.mode='uniform50';self.task=None;self.warming=False

    def resolve(self,name):
        if self.mode in UNIFORM:return UNIFORM[self.mode]
        key=f'{self.task}|{self.study.branch}|{phase(self.study.step,self.study.steps)}|{name}'
        return self.policy['policies'][self.mode]['assignments'][key]


def install(transformer,controller):
    modules=[]
    for i,layer in enumerate(transformer.layers):
        for name,child in list(layer.mlp_moe_gen.named_children()):
            if isinstance(child,torch.nn.Linear):
                replacement=BudgetLinear(child,controller,f'{i}.{name}')
                setattr(layer.mlp_moe_gen,name,replacement);modules.append(replacement)
        if i%8==0:print('Budget FFN layer',i,flush=True)
    return modules

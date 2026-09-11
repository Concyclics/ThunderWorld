"""Exact BF16 pointwise output plus global-amax partials for the next FFN.

Keep Nano SiLU in PyTorch to preserve its transcendental/rounding behavior.
Only fuse ReLU-square (Edge) or BF16 multiply (Nano) with the subsequent read
for amax. The final global factor uses the original finish_amax kernel.
"""
import types
import torch
import triton
import triton.language as tl
from thunderworld_fp4._linear_kernels import finish_amax,pack_activation


@triton.jit
def pointwise_amax(A,B,Y,PARTIAL,SIZE:tl.constexpr,RELU2:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(A+idx,idx<SIZE,0).to(tl.float32)
    if RELU2:
        a=tl.maximum(a,0.);y=a*a
    else:
        b=tl.load(B+idx,idx<SIZE,0).to(tl.float32);y=a*b
    y=y.to(tl.bfloat16)
    tl.store(Y+idx,y,idx<SIZE)
    tl.store(PARTIAL+tl.program_id(0),tl.max(tl.abs(y.to(tl.float32)),0))


def activate_with_factor(a,b=None):
    assert a.dtype==torch.bfloat16 and a.is_contiguous()
    if b is not None:assert b.shape==a.shape and b.dtype==a.dtype and b.is_contiguous()
    y=torch.empty_like(a);partial=torch.empty((triton.cdiv(a.numel(),4096),),device=a.device,dtype=torch.float32)
    factor=torch.empty((),device=a.device,dtype=torch.float32)
    pointwise_amax[(partial.numel(),)](a,a if b is None else b,y,partial,a.numel(),b is None,4096,num_warps=4)
    finish_amax[(1,)](partial,factor,partial.numel(),triton.next_power_of_2(partial.numel()),num_warps=4)
    return y,factor


def pack_with_factor(x,d,repair,full,mask,factor):
    m,k=x.shape;mp=triton.cdiv(m,16)*16;planes=2 if repair else 1
    codes=torch.empty((planes,mp,k//2),device=x.device,dtype=torch.uint8)
    scales=torch.empty((planes,mp,k//16),device=x.device,dtype=torch.float8_e4m3fn)
    pack_activation[(triton.cdiv(m,16),k//128)](x,d,factor,mask,codes,scales,m,k,mp,repair,full,num_warps=4)
    return codes,scales,factor,mp


def install(transformer):
    from dynamic import BudgetLinear
    count=0
    for layer in transformer.layers:
        mlp=layer.mlp_moe_gen
        if not isinstance(mlp.down_proj,BudgetLinear):raise ValueError('Fusion requires budget FFNs')
        if not bool((mlp.down_proj.op.d==1).all()):raise ValueError('Fused factor requires identity prescale')
        def forward(self,x):
            if self.hidden_act=='relu2':z,factor=activate_with_factor(self.up_proj(x))
            elif self.hidden_act=='silu':z,factor=activate_with_factor(self.act_fn(self.gate_proj(x)),self.up_proj(x))
            else:raise ValueError(self.hidden_act)
            return self.down_proj(z,factor=factor)
        mlp.forward=types.MethodType(forward,mlp);count+=1
    return count

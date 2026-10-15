"""Exact calibrated ReLU²/amax fusion, preserving prescale and FP4 packing.

The output activation remains unscaled BF16. Its calibrated amax is computed
from BF16(output * per-channel prescale), exactly as the existing producer.
Global reduction, FP4 base/residual quantization and sparse selection stay intact.
"""
import torch
import triton
import triton.language as tl
from .._linear_kernels import finish_amax


@triton.jit
def calibrated_relu_amax(X,D,Y,P,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.maximum(tl.load(X+i,i<N,0).to(tl.float32),0.)
    y=(x*x).to(tl.bfloat16)
    tl.store(Y+i,y,i<N)
    d=tl.load(D+i%K)
    scaled=(y.to(tl.float32)*d).to(tl.bfloat16).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(scaled),0))


def activate(x,d):
    if x.dtype!=torch.bfloat16 or not x.is_contiguous():raise ValueError('Expected contiguous BF16 FFN input')
    y=torch.empty_like(x);p=torch.empty((triton.cdiv(x.numel(),4096),),device=x.device,dtype=torch.float32)
    f=torch.empty((),device=x.device,dtype=torch.float32)
    calibrated_relu_amax[(p.numel(),)](x,d,y,p,x.numel(),x.shape[-1],4096,num_warps=4,enable_fp_fusion=False)
    finish_amax[(1,)](p,f,p.numel(),triton.next_power_of_2(p.numel()),num_warps=4)
    return y,f


def pack_native(x,op,factor,base,res):
    from .._calibrated import pack_kernel
    from .._fusion import pack_with_factor
    m,k=x.shape
    if (base,res)==(1.,1.):return pack_with_factor(x,op.d,op.na>0,op.na==k//128,op.am,factor)
    mp=triton.cdiv(m,16)*16;planes=2 if op.na else 1
    c=torch.empty((planes,mp,k//2),device=x.device,dtype=torch.uint8)
    s=torch.empty((planes,mp,k//16),device=x.device,dtype=torch.float8_e4m3fn)
    pack_kernel[(triton.cdiv(m,16),k//128)](x,op.d,factor,op.am,c,s,m,k,mp,op.na>0,op.na==k//128,False,base,res,1.,num_warps=4)
    return c,s,factor,mp


def install(model):
    from .linears import RepairLinear,select_tiles
    names=[]
    for name,mlp in model.named_modules():
        if not name.endswith('mlp_moe_gen') or mlp.hidden_act!='relu2' or not isinstance(mlp.down_proj,RepairLinear):continue
        down=mlp.down_proj
        down.register_buffer('activation_fusion_prescale',down.asset['prescale'].cuda())
        original=down._compute
        def compute(x,factor=None,packed=None,_down=down,_original=original):
            if factor is None:return _original(x,factor,packed)
            if packed is not None:raise ValueError('Cannot combine supplied packed input and fused factor')
            op=_down.bound;m,k=x.shape;base,res=_down.asset.get('activation_scale',(1.,1.))
            sparse=_down.asset.get('selector')=='residual_sensitivity' and 0<op.na<k//128
            if not sparse and 0<op.na<k//128:
                select_tiles[(triton.cdiv(m,128),)](x,op.d,op.am,m,k,triton.next_power_of_2(k//128),op.na,num_warps=4)
            if getattr(op,'direct_scale_pack',False):
                from ..cutlass.packing import pack_input
                packed=pack_input(x,op,sensitivity=_down.sensitivity if sparse else None,base=base,res=res,factor=factor,energy=_down.repair_energy)
            elif sparse:
                from .._sparse_pack import pack_sparse
                packed=pack_sparse(x,op.d,_down.sensitivity,op.am,op.na,base=base,res=res,factor=factor,
                                   energy=_down.repair_energy,score=_down.repair_score)
            else:packed=pack_native(x,op,factor,base,res)
            y=op.run_packed(x,packed)
            return y if _down.bias is None else y+_down.bias
        down._compute=compute
        def forward(x,_mlp=mlp):
            y,factor=activate(_mlp.up_proj(x),_mlp.down_proj.activation_fusion_prescale)
            return _mlp.down_proj(y,factor=factor)
        mlp.forward=forward;names.append(name)
    return names

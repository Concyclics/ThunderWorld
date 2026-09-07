"""Bounded sanitizer coverage for tail attention and sparse FFN repair."""
import torch
from thunderworld_fp4 import FP4Linear, LinearRepairConfig, RepairConfig, fp4_attention

with torch.inference_mode():
    q=torch.randn(1,33,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,257,2,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    out=fp4_attention(q,k,v,RepairConfig(head_budgets=((0,0),(1,1)),splits=8))
    layer=FP4Linear.from_float(torch.randn(256,384,device='cuda',dtype=torch.bfloat16),
                              config=LinearRepairConfig(coverage=.5,output_budgets=((0,1),)))
    y=layer(torch.randn(145,384,device='cuda',dtype=torch.bfloat16))
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(y).all()
print('sanitizer smoke passed')

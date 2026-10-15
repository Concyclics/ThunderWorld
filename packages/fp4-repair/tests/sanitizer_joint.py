"""Bounded sanitizer coverage of modified dense-W and Q-reload kernels."""
from types import SimpleNamespace
import torch
from thunderworld_fp4.cosmos.linears import RepairLinear
from thunderworld_fp4.cosmos.settings import TIERS
from thunderworld_fp4.cosmos.tuning import LaunchTable
from thunderworld_fp4 import RepairConfig, fp4_attention

with torch.inference_mode():
    controller=SimpleNamespace(warming=True,table=LaunchTable(),
        linear_spec=lambda name,default:default,record_linear=lambda *args:None)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    for tier in ('base','a50w','dual','four'):
        layer=RepairLinear(source,controller,'test',TIERS[tier])
        for m in (129,416):
            for zero in (False,True):
                x=torch.randn(m,384,device='cuda',dtype=torch.bfloat16)
                if zero:x.zero_()
                y=layer(x)
                assert torch.isfinite(y).all()
                if zero:assert not y.any()
    q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,257,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    for coverage in (0.,.5,1.):
        for terms in (2,3,4):
            y=fp4_attention(q,k,v,RepairConfig(coverage=coverage,terms=terms,
                reload_q=True,transpose_v=True,splits=2))
            assert torch.isfinite(y).all()
    torch.cuda.synchronize()
print('Joint modified kernels sanitizer passed')

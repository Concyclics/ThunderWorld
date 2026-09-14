"""Mixed execution choices preserve phase ownership, masks and frozen profiles."""
import os
from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from thunderworld_fp4.cosmos.linears import RepairLinear
from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec
from thunderworld_fp4.cosmos.tuning import LaunchTable


@pytest.mark.parametrize('repair',[LinearSpec(0,0),LinearSpec(0,.5),LinearSpec(.5,.5)])
@pytest.mark.parametrize('m,frozen',[(129,False),(129,True),(256,True)])
@torch.inference_mode()
def test_joint_candidates_and_changed_inputs(repair,m,frozen):
    config=RunConfig(linear_backend='cutlass-fused',cutlass_library=os.environ['TW_CUTLASS_LIBRARY'],
        cutlass_small_library=os.environ['TW_CUTLASS_OTHER_LIBRARY'],joint_tuning=True)
    class Table:
        key=staticmethod(LaunchTable.key)
        tune=True
        entries={}
        def search(self,key,candidates,apply,fn,*,probes):
            assert len(probes)==2
            assert len(candidates)==((6 if frozen and m==129 else 10) if repair.activation==0 else 4)
            apply(candidates[0]);reference=fn().clone()
            refs=[f().clone() for f in probes]
            accepted=[]
            for c in candidates:
                apply(c)
                if torch.equal(reference,fn()) and all(torch.equal(r,f()) for r,f in zip(refs,probes)):
                    accepted.append(c)
            assert sum(c['backend']=='cutlass-fused' for c in accepted)==4
            if repair.activation==0:assert any(c['backend']=='triton' for c in accepted)
            self.selected=accepted[-1]
        def get(self,key,default):return self.selected
    table=Table()
    plan=SimpleNamespace(linears={'0|0|test':{'w_blocks':list(range(8))}}) if frozen else None
    controller=SimpleNamespace(config=config,budget_plan=plan,table=table,warming=True,
        linear_spec=lambda name,default:repair,record_linear=lambda *args:None)
    source=torch.nn.Linear(512,256,bias=False,device='cuda',dtype=torch.bfloat16)
    layer=RepairLinear(source,controller,'test',repair)
    x=torch.randn(m,512,device='cuda',dtype=torch.bfloat16)
    layer(x);selected=layer.bound
    for v in selected.tuning_variants:
        if hasattr(v,'tile_n'):assert v.bm==128
    controller.warming=False
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=layer(x)
    x.mul_(.9).add_(.13);graph.replay()
    anchor=selected.tuning_variants[0];layer.bound=anchor
    assert torch.equal(out,layer._compute(x))
    layer.bound=selected
    assert torch.equal(out,layer(x))


def test_joint_config_contract():
    with pytest.raises(ValueError):RunConfig(joint_tuning=True)
    config=RunConfig(linear_backend='cutlass-fused',cutlass_library='offline.so',joint_tuning=True)
    assert RunConfig.from_dict(config.to_dict())==config
    with pytest.raises(ValueError):replace(config,share_pack=True)

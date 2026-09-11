import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fused_pointwise_factor_and_all_tiers():
    from fusion import activate_with_factor
    from dynamic import BudgetLinear,TIERS
    from thunderworld_fp4.linear import _pack
    with torch.inference_mode():
        torch.manual_seed(9301)
        linear=BudgetLinear(torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16))
        for m in [1,257]:
            x=torch.randn(m,256,device='cuda',dtype=torch.bfloat16);other=torch.randn_like(x)
            for relu in [True,False]:
                expected=torch.relu(x).square() if relu else x*other
                actual,f=activate_with_factor(x,None if relu else other)
                assert torch.equal(actual,expected)
                old=_pack(expected,linear.op.d,False,False,linear.op.am)
                assert torch.equal(f,old[2])
                for tier in TIERS:
                    linear.tier=tier;reference=linear(expected);out=linear(actual,factor=f)
                    assert torch.equal(reference,out),(m,relu,tier)
        actual,f=activate_with_factor(torch.zeros(257,256,device='cuda',dtype=torch.bfloat16))
        assert not bool(actual.any()) and f.item()==1.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_two_slots_tail_and_replay_changed_input():
    from dynamic import BudgetLinear,TIERS
    from chunk_pipeline import ChunkPipeline
    with torch.inference_mode():
        torch.manual_seed(9302)
        x=torch.randn(641,256,device='cuda',dtype=torch.bfloat16)
        linear=BudgetLinear(torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16))
        for tier in TIERS:
            linear.tier=tier
            for chunk in [128,256,512]:
                pipe=ChunkPipeline(linear,x,tier,chunk)
                assert torch.equal(pipe(x),linear(x)),(tier,chunk)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):y=pipe(x)
                x.mul_(.875);graph.replay();torch.cuda.synchronize()
                assert torch.equal(y,linear(x)),(tier,chunk)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_phase_graph_current_inputs_and_phase_boundaries():
    from types import SimpleNamespace
    from dynamic import BudgetLinear,phase
    from phase_graph import PhaseGraph
    with torch.inference_mode():
        torch.manual_seed(9303)
        study=SimpleNamespace(mode='ffn_auto50',counts={'full':0},step=0,steps=35,branch=0,call=0)
        c=SimpleNamespace(study=study,mode='auto50',task='i2v',warming=True,pack_hits=0,pack_cache={},share_pack=False,policy={})
        c.resolve=lambda name:['base','a50','act2','dual'][phase(study.step,study.steps)]
        linear=BudgetLinear(torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16),c,'0.up_proj')
        def forward(x):
            study.counts['full']+=1
            return (linear(x),)
        wrapper=PhaseGraph(forward,study,c,lambda:[linear]);x=torch.randn(257,256,device='cuda',dtype=torch.bfloat16)
        wrapper.warming=True;wrapper(x=x);wrapper.warming=False;c.warming=False
        assert wrapper.captures==4
        saved=[]
        for step in [0,6,7,17,18,27,28,34,0]:
            study.step=step;x.mul_(.9375);expected=forward(x)[0].clone();before=linear.calls
            got=wrapper(x=x)[0];assert linear.calls==before+1 and torch.equal(got,expected)
            saved.append((got,got.clone()))
        assert all(torch.equal(a,b) for a,b in saved)
        study.branch=1
        with pytest.raises(RuntimeError):wrapper(x=x)

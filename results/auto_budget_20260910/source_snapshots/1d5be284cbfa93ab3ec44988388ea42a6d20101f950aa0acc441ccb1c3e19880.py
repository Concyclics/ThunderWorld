import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch
import pytest
from allocate import allocate


def test_budget_ceiling_and_nonmonotone_options():
    units=[dict(key=str(i),weight=1,importance=1,options={
        'base':dict(extra=0,loss=1,ms=1),'a':dict(extra=1,loss=.9,ms=1.1),
        'w':dict(extra=1,loss=.2,ms=1.2),'dual':dict(extra=2,loss=.3,ms=2)}) for i in range(4)]
    for f in [0,.25,.5,.75,1]:
        result=allocate(units,f)
        assert result['extra_flops']<=8*f
        assert result['predicted_loss']<=1
        assert 'dual' not in result['assignments'].values()
    assert set(allocate(units,.5)['assignments'].values())=={'w'}


def decode(c,s,g):
    codes=torch.stack((c&15,c>>4),-1).flatten(-2).long()
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=c.device)
    return lut[codes&7]*torch.where(codes>=8,-1.,1.)*s.float().repeat_interleave(16,-1)*g


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120')
def test_dynamic_terms_tail_and_graph_current_input():
    from dynamic import BudgetLinear,TIERS
    from thunderworld_fp4.linear import _pack
    with torch.inference_mode():
        torch.manual_seed(808)
        src=torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16)
        mod=BudgetLinear(src)
        wc,ws,wg,_=_pack(src.weight,torch.ones(256,device='cuda'),True,True,src.weight)
        weights=decode(wc,ws,wg)
        x=torch.randn(257,256,device='cuda',dtype=torch.bfloat16)
        for tier,(a,w) in TIERS.items():
            mod.tier=tier;actual=mod(x).float()
            ac,asc,ag,mp=mod.op.pack(x);acts=decode(ac,asc,ag)[:,:len(x)]
            ref=acts[0]@weights[0].T
            if a:ref+=acts[1]@weights[0].T
            if w:ref+=acts[0]@weights[1].T
            assert (actual-ref).norm()/ref.norm()<.004
            if a==.5:assert torch.equal(mod.op.am.sum(-1),torch.ones(3,device='cuda',dtype=torch.int64))
        mod.tier='a50';mod(x);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):y=mod(x)
        x.zero_();x[:,:128]=torch.randn_like(x[:,:128])*8
        graph.replay();torch.cuda.synchronize()
        assert bool(mod.op.am[:,0].all()) and not bool(mod.op.am[:,1].any())
        assert torch.equal(y,mod(x))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120')
def test_shared_gate_up_all_tiers_exact():
    from dynamic import BudgetLinear,BudgetController,TIERS
    from types import SimpleNamespace
    with torch.inference_mode():
        torch.manual_seed(809);x=torch.randn(257,256,device='cuda',dtype=torch.bfloat16)
        gate=BudgetLinear(torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16),name='0.gate_proj')
        up=BudgetLinear(torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16),name='0.up_proj')
        c=object.__new__(BudgetController);c.study=SimpleNamespace(call=0);c.share_pack=True;c.warming=False
        c.paired_names={'0.gate_proj','0.up_proj'};c.policy={};c.reset_pack()
        for a in TIERS:
            for b in TIERS:
                gate.controller=up.controller=None;gate.tier=a;up.tier=b
                ref=(gate(x).clone(),up(x).clone())
                c.resolve=lambda name: a if name=='0.gate_proj' else b
                for m in [gate,up]:m.warmed_shapes.add(tuple(x.shape));m.controller=c
                actual=(gate(x),up(x))
                assert all(torch.equal(v,w) for v,w in zip(actual,ref)),(a,b)
                assert not c.pack_cache
        assert c.pack_hits==36

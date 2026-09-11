import pytest
import torch
from thunderworld_fp4 import RepairConfig, LinearRepairConfig, FP4Attention, FP4Linear, fp4_attention, make_repair_mask
from thunderworld_fp4.linear import _pack
from oracle import reference
pytestmark=[pytest.mark.cuda,pytest.mark.skipif(not torch.cuda.is_available(),reason='requires SM120 CUDA')]

def inputs(m=33,n=257):
    torch.manual_seed(17)
    return tuple(torch.randn((1,s,h,128),device='cuda',dtype=torch.bfloat16) for s,h in [(m,4),(n,2),(n,2)])

@pytest.mark.parametrize('terms,coverage,n,splits',[(3,0,1,8),(2,1,129,4),(3,.5,257,8),(4,1,129,2),(3,1,257,2)])
@torch.inference_mode()
def test_attention_oracle(terms,coverage,n,splits):
    q,k,v=inputs(n=n)
    cfg=RepairConfig(terms=terms,coverage=coverage,splits=splits)
    y,mask=fp4_attention(q,k,v,cfg,return_mask=True)
    ref=reference(q,k,v,mask.int()*2+1,coverage>0,terms,splits)
    assert torch.isfinite(y).all()
    assert (y.float()-ref).norm()/ref.norm()<.012

@pytest.mark.parametrize('terms,coverage',[(3,0),(2,1),(3,.5),(3,1),(4,1)])
@torch.inference_mode()
def test_transposed_v_layout(terms,coverage):
    from dataclasses import replace
    q,k,v=inputs(m=129,n=257)
    cfg=RepairConfig(terms=terms,coverage=coverage,head_budgets=((0,0),(1,1)) if coverage==.5 else ())
    baseline,mask=fp4_attention(q,k,v,cfg,return_mask=True)
    changed,other=fp4_attention(q,k,v,replace(cfg,transpose_v=True),return_mask=True)
    assert torch.equal(mask,other)
    torch.testing.assert_close(changed,baseline,rtol=0,atol=0)

@pytest.mark.parametrize('transpose_v',[False,True])
@torch.inference_mode()
def test_heads_graph_and_masks(transpose_v):
    q,k,v=inputs(m=145,n=385)
    cfg=RepairConfig(coverage=.5,head_budgets=((0,0),(1,1),(2,.25)),transpose_v=transpose_v)
    engine=FP4Attention(cfg)
    a,mask=engine(q,k,v,return_mask=True)
    assert mask.sum(-1).cpu().tolist()==[[0,0],[4,4],[1,1],[2,2]]
    old=a.clone()
    k=k*.7;v=v+1
    b=engine(q,k,v)
    torch.testing.assert_close(b,fp4_attention(q,k,v,cfg),rtol=0,atol=0)
    assert torch.equal(a,old) and not torch.equal(a,b)
    explicit=make_repair_mask(145,385,4,{0:[(0,3)],2:[(1,0),(0,1)]},device=q.device)
    c,selected=engine(q,k,v,repair_mask=explicit,return_mask=True)
    assert torch.equal(explicit,selected)
    explicit.logical_not_()
    d=engine(q,k,v,repair_mask=explicit,return_mask=True)[0]
    torch.testing.assert_close(d,fp4_attention(q,k,v,cfg,repair_mask=explicit),rtol=0,atol=0)
    assert not torch.equal(c,d)
    engine.clear()

def decode(codes,scales):
    ids=torch.stack((codes&15,codes>>4),-1).flatten(-2).long()
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=codes.device)
    return lut[ids&7]*torch.where(ids>=8,-1,1)*scales.float().repeat_interleave(16,-1)

@pytest.mark.parametrize('coverage,explicit',[(0,False),(1,False),(.5,False),(.5,True)])
@torch.inference_mode()
def test_linear_decoded(coverage,explicit):
    torch.manual_seed(42)
    w=torch.randn(256,384,device='cuda',dtype=torch.bfloat16)
    d=torch.linspace(.5,2,384,device='cuda')
    bias=torch.randn(256,device='cuda',dtype=torch.bfloat16)
    layer=FP4Linear.from_float(w,prescale=d,bias=bias,config=LinearRepairConfig(coverage=coverage,output_budgets=((1,1),) if coverage==.5 else ()))
    before=layer.weight_codes.clone()
    x=torch.randn(145,384,device='cuda',dtype=torch.bfloat16)
    mask=torch.zeros((2,2,3),device='cuda',dtype=torch.bool)
    if explicit:mask[0,0,2]=True;mask[1,1,0]=True
    elif coverage==1:mask.fill_(True)
    elif coverage==.5:
        from thunderworld_fp4._linear_kernels import activation_rank
        rank=torch.empty((2,3),device='cuda',dtype=torch.int32)
        activation_rank[(2,)](x,layer.prescale,rank,145,384,4,num_warps=4)
        mask=(rank[:,None,:]<layer._counts[None,:,None]).contiguous()
    y=layer(x,repair_mask=mask if explicit else None)
    ac,sc,af,mp=_pack(x,layer.prescale,True,True,x)
    a=decode(ac,sc)[:,:145]*af
    wd=decode(layer.weight_codes,layer.weight_scales)*layer.weight_global_scale
    ref=a[0]@wd.T
    for mb in range(2):
        for nb in range(2):
            rows=slice(mb*128,min(145,(mb+1)*128));cols=slice(nb*128,(nb+1)*128)
            for kb in range(3):
                if mask[mb,nb,kb]:
                    ks=slice(kb*128,(kb+1)*128)
                    ref[rows,cols]+=a[1,rows,ks]@wd[cols,ks].T
    ref+=bias.float()
    assert (y.float()-ref).norm()/ref.norm()<.004
    assert torch.equal(before,layer.weight_codes)
    assert torch.isfinite(y).all()

@torch.inference_mode()
def test_linear_graph_current_input():
    layer=FP4Linear.from_float(torch.randn(128,256,device='cuda',dtype=torch.bfloat16))
    x=torch.randn(32,256,device='cuda',dtype=torch.bfloat16)
    mask=torch.ones((1,1,2),device='cuda',dtype=torch.bool)
    for _ in range(3):layer(x,repair_mask=mask)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):y=layer(x,repair_mask=mask)
    x.mul_(2);mask.zero_();graph.replay()
    torch.testing.assert_close(y,layer(x,repair_mask=mask),rtol=0,atol=0)

@torch.inference_mode()
def test_all_zero_mask_and_streams():
    q,k,v=inputs(m=17,n=129)
    # Sequence strides need not be contiguous; only D is contiguous.
    q=q.repeat_interleave(2,dim=1)[:,::2]
    engine=FP4Attention(RepairConfig(coverage=0),graph_cache_size=2)
    mask=torch.zeros((4,1,2),device='cuda',dtype=torch.bool)
    base=engine(q,k,v)
    selected=engine(q,k,v,repair_mask=mask)
    torch.testing.assert_close(base,selected,rtol=0,atol=0)
    parent=torch.cuda.current_stream()
    other=torch.cuda.Stream();other.wait_stream(parent)
    with torch.cuda.stream(other):out=engine(q,k,v)
    parent.wait_stream(other)
    torch.testing.assert_close(out,base,rtol=0,atol=0)
    engine.clear()

@torch.inference_mode()
def test_linear_config_immutable():
    layer=FP4Linear.from_float(torch.zeros(128,128,device='cuda',dtype=torch.bfloat16))
    with pytest.raises(AttributeError):layer.config=LinearRepairConfig(coverage=0)
    out=layer(torch.zeros(1,128,device='cuda',dtype=torch.bfloat16))
    assert torch.equal(out,torch.zeros_like(out))

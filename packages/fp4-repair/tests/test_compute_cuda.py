from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from thunderworld_fp4 import RepairConfig,fp4_attention,FP4Attention

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 GPU required')


@pytest.mark.parametrize('m',[129,416])
@torch.inference_mode()
def test_sparse_activation_codes_current_residual_and_graph(m):
    from thunderworld_fp4._sparse_pack import pack_sparse
    from thunderworld_fp4._calibrated import pack
    from thunderworld_fp4.cosmos.calibration import decode,quantize_weight
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.settings import LinearSpec
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    torch.manual_seed(42)
    x=torch.randn((m,384),device='cuda',dtype=torch.bfloat16)
    d=torch.linspace(.5,2,384,device='cuda');sensitivity=torch.linspace(.1,4,384,device='cuda')
    mask=torch.empty(((m+127)//128,3),device='cuda',dtype=torch.bool)
    sparse=pack_sparse(x,d,sensitivity,mask,1,base=.875,res=1.125)
    dense=pack(x,d,True,True,x,base=.875,res=1.125)
    for a,b in zip(sparse[:2],dense[:2]):
        assert torch.equal(a[0].view(torch.uint8),b[0].view(torch.uint8))
        width=a.shape[-1]//3
        for row in range(m):
            for block in range(3):
                if mask[row//128,block]:
                    assert torch.equal(a[1,row,block*width:(block+1)*width].view(torch.uint8),b[1,row,block*width:(block+1)*width].view(torch.uint8))
    residual=(x.float()*d).bfloat16().float()-decode(dense[0][0,:m],dense[1][0,:m],dense[2])
    for block in range((m+127)//128):
        score=(residual[block*128:(block+1)*128].square()*sensitivity).reshape(-1,3,128).sum((0,2))
        assert mask[block].nonzero().item()==score.argmax().item()
    w=torch.randn((256,384),device='cuda',dtype=torch.bfloat16)
    wc,ws,wg=quantize_weight(w,d)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    source.weight.copy_(w)
    spec=LinearSpec(0,0,True,activation_blocks=1,weight_support=(0,2,4))
    plan=SimpleNamespace(linears={'0|0|test':dict(w_blocks=[0,2,4]),'0|1|test':dict(w_blocks=[0,1,2,4,5])})
    controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:spec,
                               record_linear=lambda *a:None,budget_plan=plan)
    asset=dict(codes=wc.cpu(),scales=ws.cpu(),factor=wg.cpu(),prescale=d.cpu(),score=torch.ones(2,3),bias=None,
               selector='residual_sensitivity',activation_sensitivity=sensitivity.cpu(),activation_scale=(.875,1.125))
    layer=RepairLinear(source,controller,'test',spec,asset)
    layer(x);controller.warming=False
    expected=layer(x)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=layer(x)
    x.add_(.125);graph.replay()
    assert torch.equal(out,layer(x)) and not torch.equal(out,expected)
    # Shared union indirection must preserve exactly the original compact-bank result.
    controller2=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:spec,record_linear=lambda *a:None)
    standalone=RepairLinear(source,controller2,'test',spec,asset)
    assert torch.equal(out,standalone(x))


@pytest.mark.parametrize('m,n',[ (129,129),(416,416)])
@pytest.mark.parametrize('qk,pv,qt,pt',[(0,1,2,3),(1,0,4,2),(1,1,3,4)])
@torch.inference_mode()
def test_independent_qk_pv_gqa_sparse_pack(m,n,qk,pv,qt,pt):
    from oracle import reference
    torch.manual_seed(5)
    q=torch.randn((1,m,4,128),device='cuda',dtype=torch.bfloat16)
    k=torch.randn((1,n,2,128),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    cfg=RepairConfig(coverage=0,head_quotas=(qk,0,qk,qk),pv_head_quotas=(0,pv,pv,pv),terms=qt,pv_terms=pt,
                     sparse_pack=True,transpose_v=True,fast_p=False)
    y,masks=fp4_attention(q,k,v,cfg,return_mask=True)
    legacy=fp4_attention(q,k,v,replace(cfg,sparse_pack=False))
    assert torch.equal(y,legacy)
    ref=reference(q,k,v,torch.where(masks[0],qt,1),True,terms=qt,
                  pvflags=torch.where(masks[1],pt,1),pvterms=pt)
    torch.testing.assert_close(y.float(),ref,atol=.004,rtol=.025)
    graph=FP4Attention(cfg)
    graph(q,k,v)
    q.add_(.25)
    assert torch.equal(graph(q,k,v),fp4_attention(q,k,v,cfg))


@torch.inference_mode()
def test_zero_full_attention_and_native_weight_codec():
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    from thunderworld_fp4._calibrated import pack
    torch.manual_seed(23)
    w=torch.randn(128,256,device='cuda',dtype=torch.bfloat16)
    d=torch.linspace(.25,4,256,device='cuda')
    for base in (.75,.875,1.):
        for residual in (.875,1.,1.125):
            c,s,g=quantize_weight(w,d,base=base,residual=residual)
            native=pack((w.float()/d).bfloat16().contiguous(),torch.ones_like(d),True,True,w,base=base,res=residual)
            assert torch.equal(c,native[0]) and torch.equal(s.view(torch.uint8),native[1].view(torch.uint8))
            assert torch.equal(g,native[2])
    q=torch.zeros((1,129,4,128),device='cuda',dtype=torch.bfloat16)
    k=torch.zeros((1,416,2,128),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    for count in (0,4):
        cfg=RepairConfig(head_quotas=(count,)*4,pv_head_quotas=(count,)*4,terms=4,pv_terms=4,sparse_pack=True)
        out=fp4_attention(q,k,v,cfg)
        assert torch.isfinite(out).all()
        assert torch.equal(out,fp4_attention(q,k,v,replace(cfg,sparse_pack=False)))


@torch.inference_mode()
def test_joint_budget_preflight_phase_cfg_and_output_lifetime(tmp_path):
    import copy,json
    import diffusers.utils.import_utils as imports
    imports._xformers_available=False
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3PackedMoTAttention,Cosmos3VLTextMLP
    from thunderworld_fp4.cosmos.engine import JointEngine,generation_targets
    from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    from thunderworld_fp4.cosmos.budget import PHASES
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();layer=torch.nn.Module()
            layer.self_attn=Cosmos3PackedMoTAttention(256,128,2,1,False,1e-6)
            layer.self_attn.processor._attention_backend='_native_flash'
            layer.mlp_moe_gen=Cosmos3VLTextMLP(256,384,'silu')
            self.layers=torch.nn.ModuleList([layer]);self.config=SimpleNamespace(num_embodiment_domains=2)
        def forward(self,x,t):
            cos=torch.ones((len(x),128),device=x.device,dtype=x.dtype);sin=torch.zeros_like(cos)
            _,y=self.layers[0].self_attn(x,x,(cos,sin,cos,sin))
            return (self.layers[0].mlp_moe_gen(y)+t,)
    torch.manual_seed(813)
    model=Model().cuda().bfloat16();other=copy.deepcopy(model)
    plan=dict(schema_version=1,budget=1.5,steps=35,phases=list(PHASES),task='i2v',linears={},attention={})
    for branch in (0,1):
        for ph in range(4):
            for name in generation_targets(model):
                a=int(ph==1 and name.endswith('up_proj'))
                w=[0] if ph==3 and name.endswith('down_proj') else []
                if w:a=1
                plan['linears'][f'{branch}|{ph}|{name}']=dict(a_blocks=a,w_blocks=w,cross=bool(a and w))
            plan['attention'][f'{branch}|{ph}|0']=dict(qk_counts=[int(ph==2),0],pv_counts=[0,int(ph==2)],qk_terms=2,pv_terms=2)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    cfg=RunConfig(compute_budget=1.5,budget_plan=str(path),ffn=LinearSpec(0,0),projection=LinearSpec(0,0))
    graph=JointEngine(model,cfg,LaunchTable())
    eager=JointEngine(other,replace(cfg,execution='eager',fuse_ffn=False),LaunchTable())
    x=torch.randn(129,256,device='cuda',dtype=torch.bfloat16);t=torch.ones((),device='cuda',dtype=torch.bfloat16)
    for engine in (graph,eager):
        engine.reset_request(35,2,'i2v');engine.warming=True;engine.discovering=True
        for _ in range(2):engine(x=x,t=t)
        assert not engine.cache
        ledger=engine.compile_budget();assert 1.<ledger['ratio']<=1.5
        engine.discovering=False;engine.reset_request(35,2,'i2v')
        for _ in range(2):engine(x=x,t=t)
        engine.warming=False;engine.reset_request(35,2,'i2v')
    saved=[]
    for call in range(70):
        x.mul_(.99);t.add_(.125)
        a=graph(x=x,t=t)[0];b=eager(x=x,t=t)[0]
        assert torch.equal(a,b),(call//2,call%2)
        if call in (0,14,36,56):saved.append((a,a.clone()))
    assert all(torch.equal(a,b) for a,b in saved)
    for engine in (graph,eager):engine.reset_request(35,2,'i2v')
    x.add_(.25)
    assert torch.equal(graph(x=x,t=t)[0],eager(x=x,t=t)[0])


@torch.inference_mode()
def test_explicit_mask_overrides_independent_quotas_with_current_graph_input():
    from oracle import reference
    torch.manual_seed(991)
    q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,257,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    mask=torch.zeros((4,2,3),device='cuda',dtype=torch.bool);mask[1,:,1]=True
    cfg=RepairConfig(head_quotas=(0,)*4,pv_head_quotas=(0,)*4,terms=2,pv_terms=4,sparse_pack=True,fast_p=False)
    backend=FP4Attention(cfg)
    out=backend(q,k,v,repair_mask=mask)
    ref=reference(q,k,v,torch.where(mask,2,1),True,terms=2,pvflags=torch.where(mask,4,1),pvterms=4)
    torch.testing.assert_close(out.float(),ref,atol=.004,rtol=.025)
    mask.logical_not_();q.add_(.125)
    assert torch.equal(backend(q,k,v,repair_mask=mask),fp4_attention(q,k,v,cfg,repair_mask=mask))

import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec,TIERS


def test_config_roundtrip_and_baseline_contract():
    config=RunConfig()
    assert config==RunConfig.from_dict(config.to_dict())
    assert config.debug=='off' and config.execution=='graph'
    with pytest.raises(ValueError):RunConfig(precision='fp8')
    with pytest.raises(ValueError):LinearSpec(float('nan'))
    with pytest.raises(ValueError):LinearSpec(0,1,True)


def test_debug_off_does_not_allocate_events(monkeypatch):
    from thunderworld_fp4.cosmos.diagnostics import Diagnostics
    def forbidden(*a,**kw):raise AssertionError('Diagnostic event in off mode')
    monkeypatch.setattr(torch.cuda,'Event',forbidden)
    diag=Diagnostics('off',70)
    with diag.begin():diag.end(None)
    assert diag.finish()=={'mode':'off'} and not diag.events and not diag.trace


def test_offline_support_storage_roundtrip_and_rejects_expansion():
    from thunderworld_fp4.cosmos.assets import compact, expand
    asset=dict(codes=torch.randint(0,256,(2,256,192),dtype=torch.uint8),
        scales=torch.rand(2,256,24).to(torch.float8_e4m3fn),score=torch.arange(6).reshape(2,3))
    restored=expand(compact(asset,1.),1.)
    for name in ('codes','scales'):
        assert torch.equal(restored[name].view(torch.uint8),asset[name].view(torch.uint8))
    packed=compact(asset,.5)
    assert len(packed['residual_indices'])==3
    with pytest.raises(ValueError,match='insufficient'):expand(packed,1.)
    restored=expand(packed,.5)
    assert torch.equal(restored['codes'][0],asset['codes'][0])
    assert not restored['codes'][1,:128].any()


def test_host_schedule_matches_reference():
    from thunderworld_fp4.cosmos.scheduler import HostScheduleUniPC
    from diffusers import UniPCMultistepScheduler
    import numpy as np
    for shift in (3.,10.):
        reference=UniPCMultistepScheduler(use_flow_sigmas=True,flow_shift=shift)
        host=HostScheduleUniPC.from_config(reference.config)
        sigmas=np.linspace(.999,0,36)[:-1]
        reference.set_timesteps(35,device='cpu',sigmas=sigmas)
        host.set_timesteps(35,device='cuda',sigmas=sigmas)
        assert host.timesteps.device.type=='cpu'
        assert torch.equal(host.timesteps,reference.timesteps)
        assert torch.equal(host.sigmas,reference.sigmas)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 GPU')
@torch.inference_mode()
def test_packed_linear_all_terms_and_graph():
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    from thunderworld_fp4._native import Native
    torch.manual_seed(120)
    controller=SimpleNamespace(warming=True,table=LaunchTable(),
        linear_spec=lambda name,default:default,record_linear=lambda *args:None)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    for tier,spec in TIERS.items():
        op=RepairLinear(source,controller,'test',spec)
        for m in (129,416):
            x=torch.randn(m,384,device='cuda',dtype=torch.bfloat16)
            y=op(x)
            native=op.bound
            def decode(c,s,g):
                codes=torch.stack((c&15,c>>4),-1).flatten(-2).long()
                lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=c.device)
                return lut[codes&7]*torch.where(codes>=8,-1.,1.)*s.float().repeat_interleave(16,-1)*g
            ac,asc,ag,mp=native.pack(x)
            acts=decode(ac,asc,ag)[:,:m]
            weights=decode(op.asset['codes'].cuda(),op.asset['scales'].cuda(),op.asset['factor'].cuda())
            reference=acts[0]@weights[0].T
            if native.na:reference+=acts[1]@weights[0].T
            if native.hasw:reference+=acts[0]@weights[1].T
            if spec.cross:reference+=acts[1]@weights[1].T
            assert (y.float()-reference).norm()/reference.norm()<.004
            native.dense_w=False
            gold=op(x)
            assert torch.equal(y,gold),tier
            native.dense_w=spec.weight==1
            for _ in range(3):op(x)
            controller.warming=False
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=op(x)
            saved=result.clone();x.mul_(.5).add_(.1);graph.replay()
            assert torch.equal(result,op(x)) and not torch.equal(result,saved)
            controller.warming=True


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 GPU')
@torch.inference_mode()
def test_fp8_decoded_reference_and_serialization():
    from thunderworld_fp4.cosmos.linears import FP8Linear
    from vllm import _custom_ops as ops
    torch.manual_seed(121)
    source=torch.nn.Linear(256,256,bias=False,device='cuda',dtype=torch.bfloat16)
    op=FP8Linear(source)
    x=torch.randn(129,256,device='cuda',dtype=torch.bfloat16)
    a,s=ops.scaled_fp8_quant(x,scale=None,use_per_token_if_dynamic=True)
    reference=(a.float()*s)@(op.weight.float()*op.weight_scale)
    actual=op(x).float()
    assert (actual-reference).norm()/reference.norm()<.004
    restored=FP8Linear(source,op.export())
    assert torch.equal(restored(x),op(x))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 GPU')
@torch.inference_mode()
def test_sparse_offline_asset_preserves_actual_gemm():
    from thunderworld_fp4.cosmos.assets import compact, expand
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    controller=SimpleNamespace(warming=True,table=LaunchTable(),
        linear_spec=lambda name,default:default,record_linear=lambda *args:None)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    spec=LinearSpec(.5,.5,True)
    op=RepairLinear(source,controller,'test',spec)
    restored=RepairLinear(source,controller,'test',spec,expand(compact(op.export(),.5),.5))
    x=torch.randn(416,384,device='cuda',dtype=torch.bfloat16)
    assert torch.equal(op(x),restored(x))


def test_quality_gate_is_paired_and_requires_all_modalities():
    from thunderworld_fp4.cosmos.quality import gate
    def rows(error):
        return [dict(seed=s,metrics={'video':dict(lpips=error,ssim_mean=1-error,relative_l2=error),
            'action':dict(relative_l2=error,rmse=error)}) for s in range(91001,91009)]
    assert gate(rows(.1),rows(.2),'policy')['passed']
    assert not gate(rows(.2),rows(.2),'policy')['passed']
    assert not gate(rows(.1),rows(.2),'av')['passed']
    bad=rows(.1);bad[0]['metrics']['action']['rmse']=5
    assert not gate(bad,rows(.2),'policy')['passed']
    with pytest.raises(ValueError):gate(rows(.1)[:-1],rows(.2),'policy')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 GPU')
@pytest.mark.parametrize('share_pack',[False,True])
@torch.inference_mode()
def test_joint_graph_phases_heads_masks_and_owned_outputs(share_pack):
    import copy
    import diffusers.utils.import_utils as imports
    imports._xformers_available=False
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3PackedMoTAttention,Cosmos3VLTextMLP
    from thunderworld_fp4.cosmos.engine import JointEngine
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    from thunderworld_fp4.config import RepairConfig,RepairPolicy,RepairRule
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer=torch.nn.Module()
            layer.self_attn=Cosmos3PackedMoTAttention(256,128,2,1,False,1e-6)
            layer.self_attn.processor._attention_backend='_native_flash'
            layer.mlp_moe_gen=Cosmos3VLTextMLP(256,384,'silu')
            self.layers=torch.nn.ModuleList([layer])
            self.config=SimpleNamespace(num_embodiment_domains=2)
        def forward(self,x,t):
            cos=torch.ones((len(x),128),device=x.device,dtype=x.dtype)
            sin=torch.zeros_like(cos)
            _,y=self.layers[0].self_attn(x,x,(cos,sin,cos,sin))
            return (self.layers[0].mlp_moe_gen(y)+t,)
    torch.manual_seed(122)
    model=Model().cuda().bfloat16();eager_model=copy.deepcopy(model)
    attn=RepairPolicy(RepairConfig(coverage=.5,head_budgets=((1,1.),),query_tile=16),
        (RepairRule(RepairConfig(coverage=1,terms=4,query_tile=16),steps=(2,4)),))
    from dataclasses import replace
    cfg=RunConfig(attention=attn,share_pack=share_pack,ffn=LinearSpec(.5,1,True))
    graph=JointEngine(model,cfg,LaunchTable())
    eager=JointEngine(eager_model,replace(cfg,execution='eager',share_pack=False,fuse_ffn=False),LaunchTable())
    x=torch.randn(33,256,device='cuda',dtype=torch.bfloat16);t=torch.ones((),device='cuda',dtype=torch.bfloat16)
    for engine in (graph,eager):
        engine.reset_request(4,1,'i2v');engine.warming=True
        engine(x=x,t=t);engine.warming=False
        engine.reset_request(4,1,'i2v')
    saved=[]
    for step in range(4):
        x.mul_(.875);t.add_(1)
        a=graph(x=x,t=t)[0];b=eager(x=x,t=t)[0]
        assert torch.equal(a,b)
        saved.append((a,a.clone()))
    assert len(graph.cache)==2
    assert all(torch.equal(a,b) for a,b in saved)
    graph.reset_request(4,1,'new_task')
    with pytest.raises(RuntimeError,match='Unwarmed'):graph(x=x,t=t)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 GPU')
@pytest.mark.parametrize('coverage',[0.,.5,1.])
@pytest.mark.parametrize('terms',[2,3,4])
@torch.inference_mode()
def test_attention_reload_residual_is_exact(coverage,terms):
    from dataclasses import replace
    from thunderworld_fp4.config import RepairConfig
    from thunderworld_fp4.runtime import _forward
    torch.manual_seed(124)
    q=torch.randn(1,33,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,129,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    cfg=RepairConfig(coverage=coverage,terms=terms,query_tile=16,transpose_v=True)
    a=_forward(q,k,v,cfg)[0]
    b=_forward(q,k,v,replace(cfg,reload_q=True))[0]
    assert torch.equal(a,b)

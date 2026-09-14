"""Run explicitly on SM120 after building the packaged fused.cu; includes current Graph inputs."""
import os
import pytest
import torch
import triton
from fused import CutlassFused as CutlassBridge
from thunderworld_fp4.cutlass.layout import swizzle
from thunderworld_fp4._native import Native
from thunderworld_fp4.linear import _pack
from thunderworld_fp4.cosmos.calibration import decode

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires SM120')


@pytest.mark.parametrize('m,na,full_w,cross,zero',[
    (129,0,False,False,False),(416,1,False,True,False),
    (416,3,True,True,False),(129,1,False,True,True),
    (1024,0,False,False,False)])
@torch.inference_mode()
def test_bridge(m,na,full_w,cross,zero):
    torch.manual_seed(7137);torch.backends.cuda.matmul.allow_tf32=False
    n,k=256,384;d=torch.linspace(.5,2,k,device='cuda')
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    wc,ws,wg,_=_pack(w,d,True,True,w)
    am=torch.zeros(triton.cdiv(m,128),k//128,device='cuda',dtype=torch.bool)
    am[:,:na]=True
    wm=torch.zeros(n,k,device='cuda',dtype=torch.bool)
    if full_w:wm.fill_(True)
    elif na:wm[:128,:128]=True;wm[128:,256:]=True
    op=Native(wc,ws,wg,d,am.repeat_interleave(128,1),wm,bm=128,interleave=True,num_stages=2,cross=cross)
    op.union=op.am
    if op.hasw and not full_w:
        op.wr=op.wr.flip(0);op.wrs=op.wrs.view(torch.uint8).flip(0).view(torch.float8_e4m3fn)
        op.wloc=torch.arange(len(op.wi)-1,-1,-1,device='cuda',dtype=torch.int32);op.shared_bank=True
    bridge=CutlassBridge(op,m,os.environ['TW_CUTLASS_LIBRARY'])
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    if zero:x.zero_()
    def run():return bridge.run_packed(x,op.pack(x))
    packed=op.pack(x);ac,ass,ag,_=packed
    a0=decode(ac[0],ass[0],torch.ones_like(ag))[:m]
    w0=decode(wc[0],ws[0],torch.ones_like(wg))
    ref=a0@w0.T
    if na:
        a1=decode(ac[1],ass[1],torch.ones_like(ag))[:m]
        a1*=op.am.repeat_interleave(128,0).repeat_interleave(128,1)[:m]
        ref+=a1@w0.T
    if op.hasw:
        w1=decode(wc[1],ws[1],torch.ones_like(wg))*wm
        ref+=a0@w1.T
        if cross and na:ref+=a1@w1.T
    ref=(ref*(ag*wg)).bfloat16()
    first=run();saved=first.clone()
    relative=float((first.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-20))
    assert relative<.002
    for _ in range(3):run()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    x.mul_(.73).add_(.1)
    if 0<na<3:op.am.copy_(op.am.roll(1,1))
    graph.replay()
    torch.testing.assert_close(out,run(),rtol=0,atol=0)
    assert torch.equal(first,saved)
    native=op(x)
    assert float((out.float()-native.float()).norm()/native.float().norm().clamp_min(1e-20))<.002


def test_scale_layout_is_exact():
    # Independent indexing reference for a partial M128 scale tile.
    s=torch.arange(144*24,device='cuda').remainder(126).byte().view(torch.float8_e4m3fn).reshape(144,24)
    padded=torch.zeros(256,24,device='cuda',dtype=torch.uint8);padded[:144]=s.view(torch.uint8)
    reference=padded.reshape(2,4,32,6,4).permute(0,3,2,1,4).contiguous().reshape(256,24)
    assert torch.equal(swizzle(s).view(torch.uint8),reference)


@torch.inference_mode()
@pytest.mark.parametrize('direct',[False,True])
def test_runtime_shared_assets_and_phase_switch(direct):
    from types import SimpleNamespace
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec
    from thunderworld_fp4.cutlass.artifact import verify_library
    library=os.environ['TW_CUTLASS_LIBRARY']
    config=RunConfig(linear_backend='cutlass-fused',cutlass_library=library,cutlass_direct_pack=direct)
    controller=SimpleNamespace(config=config,budget_plan=None)
    torch.manual_seed(1871)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    layer=RepairLinear(source,controller,'test',LinearSpec())
    x=torch.randn(416,384,device='cuda',dtype=torch.bfloat16)
    specs=[LinearSpec(0,0),LinearSpec(.34,.5,True),LinearSpec(1,1,True)]
    first=None
    for spec in specs:
        layer.prepare_shape(len(x),spec);wrapper=layer.bound
        if first is None:first=wrapper
        assert wrapper.ws.shape==(256,24)  # Native row-major scales must remain visible.
        assert wrapper.wc.data_ptr()==first.wc.data_ptr()
        assert wrapper.w0_scales.data_ptr()==first.w0_scales.data_ptr()
        result=layer._compute(x)
        layer.bound=wrapper.op;reference=layer._compute(x);layer.bound=wrapper
        assert float((result.float()-reference.float()).norm()/reference.float().norm())<.002
    assert verify_library(library)['tile'] in ([128,128,128],[128,64,128])


@torch.inference_mode()
def test_n64_n128_bitwise():
    other=os.environ.get('TW_CUTLASS_OTHER_LIBRARY')
    if not other:pytest.skip('Both offline tile variants required')
    torch.manual_seed(82)
    for m in (129,416,960,1024):
        n,k=384,512;d=torch.ones(k,device='cuda')
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        wc,ws,wg,_=_pack(w,d,True,True,w)
        for na,full,cross in [(0,False,False),(1,False,True),(4,True,True)]:
            am=torch.zeros(triton.cdiv(m,128),k//128,device='cuda',dtype=torch.bool);am[:,:na]=True
            wm=torch.zeros(n,k,device='cuda',dtype=torch.bool)
            if full:wm.fill_(True)
            elif na:wm[128:256,256:384]=True
            op=Native(wc,ws,wg,d,am.repeat_interleave(128,1),wm,bm=128,interleave=True,num_stages=2,cross=cross)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);packed=op.pack(x)
            a=CutlassBridge(op,m,os.environ['TW_CUTLASS_LIBRARY']);b=CutlassBridge(op,m,other)
            assert torch.equal(a.run_packed(x,packed),b.run_packed(x,packed))


@torch.inference_mode()
@pytest.mark.parametrize('direct',[False,True])
def test_runtime_variant_profile_and_current_input(direct):
    other=os.environ.get('TW_CUTLASS_OTHER_LIBRARY')
    if not other:pytest.skip('Both offline tile variants required')
    from types import SimpleNamespace
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    primary=os.environ['TW_CUTLASS_LIBRARY']
    config=RunConfig(linear_backend='cutlass-fused',cutlass_library=other,cutlass_small_library=primary,cutlass_direct_pack=direct)
    class Table:
        key=staticmethod(LaunchTable.key)
        bad=False
        def search(self,key,candidates,apply,fn):
            assert len(candidates)==2
            apply(candidates[0]);reference=fn().clone()
            apply(candidates[-1]);assert torch.equal(reference,fn())
            self.chosen=candidates[-1]
        def get(self,key,default):return dict(tile=[32,64,128]) if self.bad else self.chosen
    table=Table();spec=LinearSpec(.34,.5,True)
    controller=SimpleNamespace(config=config,budget_plan=None,warming=True,table=table,
        linear_spec=lambda name,default:spec,record_linear=lambda *args:None)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    layer=RepairLinear(source,controller,'test',spec)
    x=torch.randn(416,384,device='cuda',dtype=torch.bfloat16)
    result=layer(x)
    assert layer.ops[(416,spec.key)] is layer.bound
    controller.warming=False;x.mul_(.8).add_(.3)
    assert not torch.equal(result,layer(x))
    controller.warming=True;table.bad=True
    with pytest.raises(ValueError,match='launch profile'):layer(x)

"""Run explicitly on SM120 after building base.cu; includes current Graph inputs."""
import os
import pytest
import torch
import triton
from bridge import CutlassBridge,swizzle
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
    bridge=CutlassBridge(op,m,os.environ.get('TW_CUTLASS_LIBRARY','/tmp/thunderworld_cutlass_fp32_base.so'))
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    if zero:x.zero_()
    def run():return bridge.run_packed(x,op.pack(x))
    packed=op.pack(x);ac,ass,ag,_=packed
    base=bridge.base(packed)
    a0=decode(ac[0],ass[0],torch.ones_like(ag))[:m]
    w0=decode(wc[0],ws[0],torch.ones_like(wg))
    ref=a0@w0.T
    torch.testing.assert_close(base,ref,rtol=2e-5,atol=.1)
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

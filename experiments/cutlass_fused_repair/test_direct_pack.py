"""Direct-scale producer must preserve native bytes, masks and Graph outputs."""
import os
import pytest
import torch
import triton
from thunderworld_fp4._native import Native
from thunderworld_fp4.linear import _pack
from thunderworld_fp4._calibrated import pack
from thunderworld_fp4._sparse_pack import pack_sparse
from thunderworld_fp4._fusion import pack_with_factor
from thunderworld_fp4.cutlass.packing import pack_input
from thunderworld_fp4.cutlass.backend import CutlassFused
from thunderworld_fp4.cutlass.layout import swizzle

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 required')


@pytest.mark.parametrize('m',[1,129,416,960,1024])
@pytest.mark.parametrize('kind',['base','full','partial','sparse','calibrated','factor'])
@torch.inference_mode()
def test_direct_pack(m,kind):
    torch.manual_seed(914);n,k=256,512
    d=torch.linspace(.5,2,k,device='cuda')
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    c,s,g,_=_pack(w,d,True,True,w)
    na=0 if kind=='base' else (4 if kind in ('full','calibrated','factor') else 2)
    mask=torch.zeros(triton.cdiv(m,128),k//128,device='cuda',dtype=torch.bool);mask[:,:na]=True
    wm=torch.zeros(n,k,device='cuda',dtype=torch.bool);wm[:128,:128]=True
    op=Native(c,s,g,d,mask.repeat_interleave(128,1),wm,bm=128,interleave=True,num_stages=2,cross=na>0)
    op.am=op.union=mask;op.na=na
    bridge=CutlassFused(op,m,os.environ['TW_CUTLASS_LIBRARY'])
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
    sensitivity=torch.linspace(.1,2,k,device='cuda') if kind=='sparse' else None
    base,res=(.875,1.25) if kind in ('sparse','calibrated') else (1.,1.)
    factor=_pack(x,d,False,False,x)[2] if kind=='factor' else None
    def reference():
        if kind=='sparse':return pack_sparse(x,d,sensitivity,op.am,na,base=base,res=res)
        if kind=='calibrated':return pack(x,d,True,True,op.am,base=base,res=res)
        if factor is not None:return pack_with_factor(x,d,True,True,op.am,factor)
        return op.pack(x)
    def direct():return pack_input(x,op,sensitivity=sensitivity,base=base,res=res,factor=factor)
    def check():
        old=reference();old_mask=op.am.clone();new=direct()
        assert torch.equal(old_mask,op.am)
        assert torch.equal(old[2],new.native[2])
        assert torch.equal(old[0][0],new.native[0][0])
        assert torch.equal(old[1][0].view(torch.uint8),new.native[1][0].view(torch.uint8))
        if na:
            active=op.am.repeat_interleave(128,0).repeat_interleave(64,1)[:old[3]]
            assert torch.equal(old[0][1][active],new.native[0][1][active])
            active_s=op.am.repeat_interleave(128,0).repeat_interleave(8,1)[:old[3]]
            assert torch.equal(old[1][1].view(torch.uint8)[active_s],new.native[1][1].view(torch.uint8)[active_s])
        assert torch.equal(swizzle(old[1][0]).view(torch.uint8),new.scales.view(torch.uint8))
        assert torch.equal(old[2]*op.wg,new.alpha)
        assert torch.equal(bridge.run_packed(x,old),bridge.run_packed(x,new))
    check()
    for _ in range(2):bridge.run_packed(x,direct())
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=bridge.run_packed(x,direct())
    x.mul_(.73).add_(.11);op.am.copy_(op.am.roll(1,1))
    graph.replay()
    assert torch.equal(out,bridge.run_packed(x,reference()))
    check()
    x.zero_();check()

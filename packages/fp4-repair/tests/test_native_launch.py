"""Pipelined experimental FFN: tail, mutable graph input and output ownership."""
import importlib.util
from pathlib import Path
import pytest
import torch
from thunderworld_fp4.linear import _pack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires SM120 CUDA')


@pytest.mark.parametrize('bm,stages', [(32,2),(32,3),(64,2),(64,3),(128,2),(128,3)])
@pytest.mark.parametrize('split_k',[1,4,8])
@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('interleave',[False,True])
@torch.inference_mode()
def test_native_pipeline_graph(bm, stages,split_k,cross,interleave):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('native_launch_test', root/'experiments/ffn_block_repair_20260907/native.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(5090)
    weight = torch.randn((256,384), device='cuda', dtype=torch.bfloat16)
    d = torch.ones(384, device='cuda')
    c,s,g,_ = _pack(weight,d,True,True,weight)
    am = torch.zeros((2,384),device='cuda',dtype=torch.bool)
    am[0,:128]=True;am[1,256:]=True
    wm = torch.zeros_like(weight,dtype=torch.bool)
    wm[:128,:256]=True
    op = module.Native(c,s,g,d,am,wm,bm=bm,interleave=interleave,num_stages=stages,split_k=split_k,cross=cross)
    baseline = module.Native(c,s,g,d,am,wm,bm=32,interleave=True,cross=cross)
    x = torch.randn((129,384),device='cuda',dtype=torch.bfloat16)
    first = op(x);saved = first.clone()
    gold=baseline(x).float()
    assert (first.float()-gold).norm()/gold.norm()<.004
    for _ in range(3):op(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result = op(x)
    x.mul_(.73).add_(.1)
    graph.replay()
    torch.testing.assert_close(result,op(x),rtol=0,atol=0)
    gold=baseline(x).float()
    assert (result.float()-gold).norm()/gold.norm()<.004
    assert torch.equal(first,saved) and not torch.equal(first,result)

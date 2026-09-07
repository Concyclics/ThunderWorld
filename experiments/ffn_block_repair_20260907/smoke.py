"""Tail rows, sparse offline W blocks, cross term, current-input graph replay."""
import os
import torch
from study import quant
from native import Native
assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','3')
with torch.inference_mode():
    torch.manual_seed(9)
    w=torch.randn(256,1024,device='cuda',dtype=torch.bfloat16);d=torch.linspace(.5,2,1024,device='cuda')
    wp,wc,ws,wg=quant(w,torch.ones_like(d));x=torch.randn(145,1024,device='cuda',dtype=torch.bfloat16)
    for b,bm in [(128,32),(256,64),(256,128)]:
        am=torch.zeros((2,1024),device='cuda',dtype=torch.bool);am[0,::2*b]=True;am[1,b::2*b]=True
        am=am.reshape(2,-1,b).any(-1).repeat_interleave(b,1)
        wm=torch.zeros((256,1024),device='cuda',dtype=torch.bool);wm[:128,b:2*b]=True;wm[128:,2*b:3*b]=True
        op=Native(wc,ws,wg,d,am,wm,b,True,bm,True)
        before=(op.wc.clone(),op.wr.clone());ap,*_=quant(x,d)
        expanded=am.repeat_interleave(128,0)[:145]
        ref=ap[0]@wp[0].T+(ap[1]*expanded)@wp[0].T+ap[0]@(wp[1]*wm).T+(ap[1]*expanded)@(wp[1]*wm).T
        assert (op(x).float()-ref).norm()/ref.norm()<.004
        for _ in range(3):op(x)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=op(x)
        x.mul_(.7);graph.replay();assert torch.equal(out,op(x))
        assert torch.equal(before[0],op.wc) and torch.equal(before[1],op.wr)
    torch.cuda.synchronize()
print('3 sparse native tail/cross/graph checks passed')

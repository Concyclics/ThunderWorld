"""Numerical, tail, changing-input graph, and structured-mask boundary checks."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/auto_budget'),str(ROOT/'tools/kernel_handoff')]
from dynamic import BudgetLinear
from bench import decode
from merged_weight import MergedWeight
from structured_attention import compile_mask
from thunderworld_fp4.linear import _pack
from thunderworld_fp4 import RepairConfig,fp4_attention

@torch.inference_mode()
def main():
 torch.manual_seed(8512);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 for m in [1,257]:
  src=torch.nn.Linear(256,128,bias=False,device='cuda',dtype=torch.bfloat16);x=torch.randn(m,256,device='cuda',dtype=torch.bfloat16);layer=BudgetLinear(src)
  wc,ws,wg,_=_pack(src.weight,torch.ones(256,device='cuda'),True,True,src.weight);wp=decode(wc,ws)*wg
  for tier in ['wgt2','a50w','dual']:
   layer.prepare(x,tier);ac,asc,ag,mp=layer.op.pack(x);ap=decode(ac,asc)[:,:m]*ag
   ma=layer.op.am.repeat_interleave(128,0)[:m].repeat_interleave(128,1) if tier=='a50w' else torch.full_like(x,tier=='dual',dtype=torch.bool)
   ref=ap[0]@wp[0].T+ap[0]@wp[1].T+(ap[1]*ma)@wp[0].T if tier!='wgt2' else ap[0]@wp[0].T+ap[0]@wp[1].T
   for pair in [False,True]:
    op=MergedWeight(layer.op,pair);y=op(x);err=float((y.float()-ref).norm()/ref.norm());assert err<.004
    for _ in range(2):op(x)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=op(x)
    x.mul_(.9375);g.replay();torch.cuda.synchronize();assert torch.equal(out,op(x))
    rows.append(dict(m=m,tier=tier,pair=pair,decoded_ref_l2=err,graph_current_input=True))
    # Recompute the independent reference after mutating the current input.
    ac,asc,ag,mp=layer.op.pack(x);ap=decode(ac,asc)[:,:m]*ag
    ref=ap[0]@wp[0].T+ap[0]@wp[1].T+(ap[1]*ma)@wp[0].T if tier!='wgt2' else ap[0]@wp[0].T+ap[0]@wp[1].T
 density=np.ones((4,6,6));qr=np.array([2]*128+[4]);kr=np.array([0]*29+[2]*228)
 masks={c:compile_mask(density,qr,kr,c) for c in [0,.5,1]}
 assert [int(m.sum()) for m in masks.values()]==[0,12,24]
 assert np.array_equal(masks[.5],compile_mask(density,qr,kr,.5))
 q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(1,257,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 for c in [0,1]:
  got=fp4_attention(q,k,v,RepairConfig(coverage=c,terms=2),repair_mask=torch.from_numpy(masks[c]).cuda())
  expected=fp4_attention(q,k,v,RepairConfig(coverage=c,terms=2));assert torch.equal(got,expected)
 report=dict(passed=True,ffn_cases=rows,attention_mask_full_empty_equivalence=True,mixed_role_tail_quota=True)
 (ROOT/'results/repair_design_20260911/validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

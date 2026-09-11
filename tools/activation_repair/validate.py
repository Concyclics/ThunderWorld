from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/auto_budget'),str(ROOT/'tools/kernel_handoff')]
from dynamic import BudgetLinear
from bench import decode
from paired_activation import PairedActivation
from packing import pack
@torch.inference_mode()
def main():
 torch.manual_seed(8611);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 for m in [1,257]:
  src=torch.nn.Linear(256,128,bias=False,device='cuda',dtype=torch.bfloat16);x=torch.randn(m,256,device='cuda',dtype=torch.bfloat16);layer=BudgetLinear(src)
  wc,ws,wf,_=pack(src.weight,layer.op.d,True,True,src.weight);w=decode(wc,ws)*wf
  for tier in ['a50','act2','a50w','dual']:
   layer.prepare(x,tier);old=layer.op.pack(x);new=pack(x,layer.op.d,True,layer.op.na==2,layer.op.am)
   assert all(torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8)) for a,b in zip(old[:3],new[:3]))
   for direct in [False,True]:
    op=PairedActivation(layer.op,direct);p=op.pack(x);c,s,f,mp=p
    if direct:
     c=c.reshape(mp,2,2,64).permute(2,0,1,3).reshape(2,mp,128);s=s.view(torch.uint8).reshape(mp,2,2,8).permute(2,0,1,3).reshape(2,mp,16).contiguous().view(torch.float8_e4m3fn)
     assert torch.equal(c,old[0]) and torch.equal(s.view(torch.uint8),old[1].view(torch.uint8))
    a=decode(c,s)[:,:m]*f
    ref=a[0]@w[0].T+a[1]@w[0].T
    if layer.op.hasw:ref+=a[0]@w[1].T
    y=op(x);err=float((y.float()-ref).norm()/ref.norm());assert err<.004,(tier,direct,err)
    for _ in range(2):op(x)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=op(x)
    x.mul_(.9375);layer.prepare(x,tier);g.replay();torch.cuda.synchronize();assert torch.equal(out,op(x))
    rows.append(dict(m=m,tier=tier,direct=direct,reference_l2=err,graph_updated_input=True))
    old=layer.op.pack(x)
 # Calibrated scales and gains must also agree with a decoded reference.
 layer.prepare(x,'dual')
 for direct in [False,True]:
  op=PairedActivation(layer.op,direct,base=.875,res=.95,gain=1.0625);c,s,f,mp=op.pack(x)
  if direct:
   c=c.reshape(mp,2,2,64).permute(2,0,1,3).reshape(2,mp,128)
   s=s.view(torch.uint8).reshape(mp,2,2,8).permute(2,0,1,3).reshape(2,mp,16).contiguous().view(torch.float8_e4m3fn)
  a=decode(c,s)[:,:len(x)]*f;ref=a[0]@w[0].T+a[1]@w[0].T+a[0]@w[1].T
  err=float((op(x).float()-ref).norm()/ref.norm());assert err<.004
  rows.append(dict(m=len(x),tier='dual_calibrated',direct=direct,reference_l2=err))
 r=dict(passed=True,rows=rows,baseline_packing_bitwise=True,direct_layout_exact=True)
 (ROOT/'results/activation_repair_20260911/validation.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()

import json
from pathlib import Path
import torch
from awq_marlin import make_linear
from thunderworld_fp4.cosmos.calibration import decode
from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
from baselines import write
r=Path('/home/yufeng/thunderworld/full_quality_ready_20260919/nano');root=r/'awq_native'
manifest=json.loads((root/'manifest.json').read_text())
rows=[]
with shared_worker():
 select_shared_device();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 with torch.inference_mode():
  for name,e in manifest['modules'].items():
   a=torch.load(root/e['file'],weights_only=True);op=make_linear(a)
   scales=a['scales'].cuda().float()*a['factor'].cuda();gold=decode(a['codes'].cuda(),scales,1.).bfloat16()
   for m in (129,416,960,1024):
    x=torch.randn(m,op.in_features,device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(180918+m))
    for _ in range(3):op(x)
    graph=torch.cuda.CUDAGraph();fixed=x.clone();torch.cuda.synchronize()
    with torch.cuda.graph(graph):out=op(fixed)
    for label,z in [('random',x),('changed',x*.75+.125),('zero',torch.zeros_like(x))]:
     fixed.copy_(z);graph.replay();eager=op(z)
     ref=torch.nn.functional.linear(z*op.prescale,gold)
     err=float((out.float()-ref.float()).square().sum()/ref.float().square().sum().clamp_min(1e-20))
     row=dict(module=name,m=m,n=op.out_features,k=op.in_features,case=label,nmse=err,graph_equal=torch.equal(out,eager),finite=bool(out.isfinite().all()),ref_finite=bool(ref.isfinite().all()),max_abs=float((out.float()-ref.float()).abs().max()),ref_max=float(ref.abs().max()),factor=a['factor'].tolist(),prescale_range=[float(a['prescale'].min()),float(a['prescale'].max())])
     rows.append(row)
     if err>1e-4 or not row['finite'] or not row['graph_equal']:
      print(json.dumps(row),flush=True);write(r/'awq_mismatch.json',dict(complete=True,passed=False,rows=rows));raise SystemExit(1)
    del graph,out
   if len(rows)%120==0:print(json.dumps(dict(modules=len(rows)//12)),flush=True)
  write(r/'awq_mismatch.json',dict(complete=True,passed=True,rows=rows))

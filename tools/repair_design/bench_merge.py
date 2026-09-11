"""Compare already-fused native residual loops with offline merged layouts."""
import argparse,hashlib,json,random,statistics,sys
from pathlib import Path
import torch,triton.testing
from triton.runtime.errors import OutOfResources
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/auto_budget'),str(ROOT/'tools/kernel_handoff')]
from dynamic import BudgetLinear
from bench import decode
from merged_weight import MergedWeight

def measure(fn):return triton.testing.do_bench_cudagraph(fn,rep=15)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 report={'complete':False,'gpu':torch.cuda.get_device_name(),'scope':'Real model weights, repeated captured 64-row probes at actual M; local quality only. Online packing included. Each winner retimed in randomized order for five rounds.','rows':[],'sources':{}}
 payloads=[(model,f) for model in ['edge','nano'] for f in sorted((ROOT/f'runs5090/auto_budget_{model}_bench').glob('*.pt'))]
 if a.limit:payloads=payloads[:a.limit]
 for model,path in payloads:
  report['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  data=torch.load(path,weights_only=True,map_location='cpu');x=data['x'].cuda();w=data['weight'].cuda()
  src=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16);src.weight=torch.nn.Parameter(w,requires_grad=False)
  layer=BudgetLinear(src);base=layer.op
  merged={name:MergedWeight(base,pair=name=='pair') for name in ['plane','pair']}
  launch=json.loads((path.parent/'benchmark.json').read_text())['launches'];shape=f'{len(x)}:{w.shape[0]}:{w.shape[1]}'
  for tier in ['wgt2','a50w','dual']:
   layer.prepare(x,tier);cfg=launch[shape+'|'+tier];base.bm=cfg['bm'];base.num_stages=cfg['num_stages'];packed=base.pack(x);ref=base.run_packed(x,packed);gold=torch.nn.functional.linear(x,w).float()
   candidates=[]
   for name,op in merged.items():
    for bm in [32,64,128]:
     for stages in [1,2,3]:
      op.bm=bm;op.stages=stages
      try:y=op.run_packed(x,packed)
      except OutOfResources as exc:
       candidates.append(dict(layout=name,bm=bm,stages=stages,unsupported=str(exc)));continue
      err=float((y.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
      assert torch.isfinite(y).all() and err<.004,(shape,tier,name,bm,stages,err)
      if name=='plane':assert torch.equal(y,ref),(shape,tier,bm,stages,'plane order changed')
      candidates.append(dict(layout=name,bm=bm,stages=stages,error=err,bitwise=torch.equal(y,ref),ms=measure(lambda:op(x))))
    win=min([c for c in candidates if c['layout']==name and 'ms' in c],key=lambda c:c['ms']);op.bm=win['bm'];op.stages=win['stages']
   fns={'original':lambda:base(x),**{name:(lambda op=op:op(x)) for name,op in merged.items()}}
   samples={k:[] for k in fns};rng=random.Random(8511)
   for _ in range(5):
    order=list(fns);rng.shuffle(order)
    for key in order:samples[key].append(measure(fns[key]))
   med={k:statistics.median(v) for k,v in samples.items()}
   row=dict(model=model,shape=shape,tier=tier,original_bf16_l2=float((ref.float()-gold).norm()/gold.norm()),candidates=candidates,samples_ms=samples,median_ms=med,speedups={k:med['original']/med[k] for k in merged})
   report['rows'].append(row);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
   print(model,shape,tier,row['speedups'],flush=True)
  del layer,base,merged,x,w,src;torch.cuda.empty_cache()
 report['complete']=True
 for path in [Path(__file__),Path(__file__).with_name('merged_weight.py')]:report['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

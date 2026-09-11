"""Matched launch search incl. current-input selection and direct paired packing."""
from pathlib import Path
import sys,json,hashlib,random,statistics,argparse
import torch,triton.testing
from triton.runtime.errors import OutOfResources
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/auto_budget'))
from dynamic import BudgetLinear
from paired_activation import PairedActivation
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 r=dict(complete=False,rows=[],sources={},scope='Real model weights and repeated captured probes. Current-input partial selector, online packing and GEMM included. Matched 3-launch search and five randomized 10-ms CUDA Graph retimings. M416 reshapes development probes; it is a shape timing test, not evaluation-policy quality.')
 payloads=[]
 for model in ['edge','nano']:
  for f in sorted((ROOT/f'runs5090/auto_budget_{model}_bench').glob('*.pt')):
   payloads.append((model,f,None))
   if f.name.startswith('316_'):payloads.append((model,f,416))
 if args.limit:payloads=payloads[:args.limit]
 for model,f,newm in payloads:
  d=torch.load(f,weights_only=True,map_location='cpu');x=d['x'].cuda();w=d['weight'].cuda();r['sources'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  if newm:x=x.repeat(triton.cdiv(newm,len(x)),1)[:newm].contiguous()
  src=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16);src.weight=torch.nn.Parameter(w,requires_grad=False);layer=BudgetLinear(src)
  shape=f'{len(x)}:{w.shape[0]}:{w.shape[1]}'
  for tier in ['a50','act2','a50w','dual']:
   layer.prepare(x,tier);ref=layer.op(x).float();ops={'original':layer.op,'gather':PairedActivation(layer.op),'direct':PairedActivation(layer.op,True)};samples=[];winners={}
   for name,op in ops.items():
    for bm,stages in [(32,3),(64,3),(128,2)]:
     op.bm=bm
     if name=='original':op.num_stages=stages
     else:op.stages=stages
     try:
      layer.prepare(x,tier);y=op(x).float();err=float((y-ref).norm()/ref.norm());assert torch.isfinite(y).all() and err<.004,(shape,tier,name,err)
      def fn():layer.prepare(x,tier);return op(x)
      ms=triton.testing.do_bench_cudagraph(fn,rep=10)
      samples.append(dict(layout=name,bm=bm,stages=stages,ms=ms,relative_l2=err,bitwise=torch.equal(y,ref)))
     except OutOfResources as e:samples.append(dict(layout=name,bm=bm,stages=stages,unsupported=str(e)))
    winners[name]=min((v for v in samples if v['layout']==name and 'ms' in v),key=lambda v:v['ms'])
   timings={k:[] for k in ops};rng=random.Random(8612)
   for _ in range(5):
    order=list(ops);rng.shuffle(order)
    for name in order:
     op=ops[name];cfg=winners[name];op.bm=cfg['bm']
     if name=='original':op.num_stages=cfg['stages']
     else:op.stages=cfg['stages']
     def fn():layer.prepare(x,tier);return op(x)
     timings[name].append(triton.testing.do_bench_cudagraph(fn,rep=10))
   med={k:statistics.median(v) for k,v in timings.items()}
   row=dict(model=model,shape=shape,tier=tier,candidates=samples,winners=winners,samples_ms=timings,median_ms=med,speedups={k:med['original']/med[k] for k in ['gather','direct']});r['rows'].append(row);args.output.write_text(json.dumps(r,indent=2)+'\n');print(model,shape,tier,row['speedups'],flush=True)
  del ops,layer,src,x,w;torch.cuda.empty_cache()
 r['complete']=True
 for p in Path(__file__).parent.glob('*.py'):r['sources'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 args.output.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()

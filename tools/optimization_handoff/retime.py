"""Retime frozen A/W merge choices without selecting new winners on audit data."""
from pathlib import Path
import argparse,json,hashlib,random,statistics,sys
import torch,triton,triton.testing
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/activation_repair'),str(ROOT/'tools/repair_design'),str(ROOT/'tools/auto_budget')]
from paired_activation import PairedActivation
from merged_weight import MergedWeight
from dynamic import BudgetLinear
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--payload-root',type=Path,default=ROOT/'runs5090');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 specs=[('A',ROOT/'results/activation_repair_20260911/benchmark.json'),('W',ROOT/'results/repair_design_20260911/merged_weight_benchmark.json')]
 report=dict(complete=False,rows=[],sources={},gpu=torch.cuda.get_device_name(),scope='Frozen previous launch choices, five randomized 10-ms CUDA Graph rounds. A includes current-input selection; W reproduces its original fixed-support packing+GEMM scope. Repeated captured rows are timing inputs, not new quality samples.')
 for family,source in specs:
  data=json.loads(source.read_text());assert data['complete'];report['sources'][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
  groups={}
  for row in data['rows']:groups.setdefault((row['model'],row['shape']),[]).append(row)
  for (model,shape),records in groups.items():
   m,n,k=map(int,shape.split(':'));folder=a.payload_root/f'auto_budget_{model}_bench';payload=folder/(shape.replace(':','_')+'.pt')
   if not payload.exists() and m==416:payload=folder/f'316_{n}_{k}.pt'
   raw=torch.load(payload,weights_only=True,map_location='cpu');x=raw['x'].cuda();x=x.repeat(triton.cdiv(m,len(x)),1)[:m].contiguous();w=raw['weight'].cuda()
   report['sources'][str(payload)]=hashlib.sha256(payload.read_bytes()).hexdigest()
   src=torch.nn.Linear(k,n,bias=False,dtype=torch.bfloat16);src.weight=torch.nn.Parameter(w,requires_grad=False);layer=BudgetLinear(src);op=layer.op
   old=json.loads((ROOT/f'results/auto_budget_20260910/records/runs5090/auto_budget_{model}_bench/benchmark.json').read_text())
   for record in records:
    tier=record['tier'];layer.prepare(x,tier)
    if family=='A':
     choices=record['winners'];bank={'original':op,'gather':PairedActivation(op),'direct':PairedActivation(op,True)}
    else:
     cfg=old['launches'][shape+'|'+tier];choices={'original':dict(bm=cfg['bm'],stages=cfg['num_stages'])}
     for layout in ['plane','pair']:choices[layout]=min((v for v in record['candidates'] if v['layout']==layout and 'ms' in v),key=lambda v:v['ms'])
     bank={'original':op,'plane':MergedWeight(op),'pair':MergedWeight(op,True)}
    def select(name):
     obj=bank[name];cfg=choices[name];obj.bm=cfg['bm']
     if name=='original':obj.num_stages=cfg['stages']
     else:obj.stages=cfg['stages']
     return obj
    ref=select('original')(x).float();errors={}
    for name in bank:
     y=select(name)(x).float();err=float((y-ref).norm()/ref.norm().clamp_min(1e-12));assert torch.isfinite(y).all() and err<.004
     if family=='A' or name=='plane':assert torch.equal(y,ref)
     errors[name]=dict(relative_l2=err,bitwise=torch.equal(y,ref))
    samples={name:[] for name in bank};rng=random.Random(8701)
    for _ in range(5):
     order=list(bank);rng.shuffle(order)
     for name in order:
      obj=select(name)
      def fn():
       if family=='A':layer.prepare(x,tier)
       return obj(x)
      samples[name].append(triton.testing.do_bench_cudagraph(fn,rep=10))
    med={key:statistics.median(v) for key,v in samples.items()};speeds={key:med['original']/v for key,v in med.items() if key!='original'}
    report['rows'].append(dict(family=family,model=model,shape=shape,tier=tier,samples_ms=samples,median_ms=med,speedups=speeds,previous_speedups=record['speedups'],errors=errors));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(family,model,shape,tier,speeds,flush=True)
   del bank,layer,op,src,w,x;torch.cuda.empty_cache()
 report['complete']=True
 for directory in ['tools/optimization_handoff','tools/activation_repair','tools/repair_design','tools/auto_budget']:
  for path in (ROOT/directory).glob('*.py'):report['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

"""Output-aware block-scale calibration; development phases held out from fitting.

Scope: three layers per model, each FFN projection. Fit pooled projection policies
on steps 1/11; inspect steps 23/33 separately. Original BF16 captures are teacher
inputs, so this is not quantized-trajectory distribution calibration.
"""
from pathlib import Path
import sys,json,hashlib,gc
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/auto_budget'),str(ROOT/'tools/cosmos_e2e')]
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers import Cosmos3OmniTransformer
from adapter import load_native
from packing import pack
GRID=[(b,r) for b in [0.75,0.875,1.] for r in [0.75,1.,1.25]]
@torch.inference_mode()
def main():
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'results/activation_repair_20260911/calibration.json'
 if out.exists():raise FileExistsError(out)
 report=dict(complete=False,rows=[],sources={},scope=__doc__)
 for name,revision,layers in [('edge','Cosmos3-Edge-a9d944e',[0,14,27]),('nano','Cosmos3-Nano-411f42a',[0,18,35])]:
  directory=ROOT/f'data_local/auto_budget_{name}_capture';manifest=json.loads((directory/'manifest.json').read_text());assert manifest['complete']
  report['sources'][str((directory/'manifest.json').relative_to(ROOT))]=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
  model=Cosmos3OmniTransformer.from_pretrained(str(ROOT/'models'/revision/'transformer'),torch_dtype=torch.bfloat16,local_files_only=True)
  metas=[r for r in manifest['rows'] if int(r['name'].split('.')[0]) in layers]
  for module in dict.fromkeys(r['name'] for r in metas):
   li,part=module.split('.');weight=getattr(model.layers[int(li)].mlp_moe_gen,part).weight.detach().cuda().contiguous();n,k=weight.shape;d=torch.ones(k,device='cuda');am=torch.ones((1,k),device='cuda',dtype=torch.bool);wm=torch.ones((n,k),device='cuda',dtype=torch.bool)
   wb=pack(weight,d,True,True,weight);data=[]
   for meta in [m for m in metas if m['name']==module]:
    path=directory/meta['file'];payload=torch.load(path,weights_only=True,map_location='cpu');x=payload['x'].cuda();gold=payload['y'].cuda().float()
    report['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    data.append((meta,x,gold,{g:pack(x,d,True,True,x,base=g[0],res=g[1]) for g in GRID}))
   for side in ['A','W']:
    for b,r in GRID:
     wp=wb if side=='A' else pack(weight,d,True,True,weight,base=b,res=r)
     op=load_native().Native(wp[0],wp[1],wp[2],d,am,wm,bm=64,interleave=True,num_stages=2)
     for mode in ['act2','dual'] if side=='A' else ['dual']:
      op.hasw=mode=='dual'
      for meta,x,gold,aps in data:
       ap=aps[(b,r)] if side=='A' else aps[(1.,1.)];y=op.run_packed(x,ap).float();loss=float((y-gold).square().sum()/gold.square().sum().clamp_min(1e-20));assert torch.isfinite(y).all()
       report['rows'].append(dict(model=name,module=module,part=part,mode=mode,side=side,base=b,res=r,task=meta['task'],branch=meta['branch'],step=meta['step'],split='fit' if meta['step']<20 else 'held_phase',loss=loss))
     del op
   del data,weight,wb;gc.collect();torch.cuda.empty_cache();out.write_text(json.dumps(report,indent=2)+'\n');print(name,module,'rows',len(report['rows']),flush=True)
  del model;gc.collect()
 # Choose only with fit data. Held phases never influence the policy.
 policies={}
 for model in ['edge','nano']:
  policies[model]={}
  for mode in ['act2','dual']:
   policies[model][mode]={}
   for part in sorted({r['part'] for r in report['rows'] if r['model']==model}):
    rows=[r for r in report['rows'] if (r['model'],r['mode'],r['part'])==(model,mode,part)];keys=sorted({(r['side'],r['base'],r['res']) for r in rows});scores=[]
    for key in keys:
     v={split:sum(r['loss'] for r in rows if r['split']==split and (r['side'],r['base'],r['res'])==key)/sum(1 for r in rows if r['split']==split and (r['side'],r['base'],r['res'])==key) for split in ['fit','held_phase']}
     scores.append(dict(side=key[0],base=key[1],res=key[2],**v))
    best=min(scores,key=lambda x:x['fit']);baseline=next(x for x in scores if (x['side'],x['base'],x['res'])==('A',1.,1.))
    policies[model][mode][part]=dict(selected=best,baseline=baseline,fit_gain=1-best['fit']/baseline['fit'],held_gain=1-best['held_phase']/baseline['held_phase'])
 report.update(complete=True,policies=policies)
 for p in Path(__file__).parent.glob('*.py'):report['sources'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(policies,indent=2))
if __name__=='__main__':main()

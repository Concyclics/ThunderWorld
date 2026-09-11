"""Full-trajectory calibration/activation-pair screen; process-local adapters."""
from pathlib import Path
import sys,json,math,hashlib,argparse
from collections import Counter
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/finding_resample'),str(ROOT/'tools/auto_budget')]
import run as runner
import ffn
from packing import pack
from paired_activation import PairedActivation
options=argparse.ArgumentParser(add_help=False)
options.add_argument('--pair-benchmark',type=Path,default=ROOT/'results/activation_repair_20260911/benchmark.json')
options.add_argument('--calibration-policy',type=Path,default=ROOT/'results/activation_repair_20260911/calibration_gain.json')
extra,remaining=options.parse_known_args();sys.argv=[sys.argv[0],*remaining]
model=sys.argv[sys.argv.index('--model')+1];out=Path(sys.argv[sys.argv.index('--output')+1]);calpath=extra.calibration_policy;bpath=extra.pair_benchmark
cal=json.loads(calpath.read_text());bench=json.loads(bpath.read_text());assert cal['complete'] and bench['complete'];policy=cal['policies'][model]
launches={}
for r in bench['rows']:
 if r['model']!=model or r['tier'] not in ('act2','dual'):continue
 best=max(r['speedups'],key=r['speedups'].get)
 if r['speedups'][best]>1.05:launches[r['shape'],r['tier']]=r['winners'][best]
receipts=Counter();orig_config=ffn.TermLinear.configure

def configure(self,mode):
 tier='dual' if 'dual' in mode else 'act2';actual='ffn_'+tier if mode in ('ffn_act_cal','ffn_dual_cal','ffn_act_pair','ffn_dual_pair','ffn_dual_cal_pair') else mode
 orig_config(self,actual);self._mode=mode;self._tier=tier;self._a_scale=(1.,1.,1.);self._pair_cache={}
 if 'cal' in mode:
  cfg=policy[tier][self._part]['selected']
  if cfg['side']=='A':self._a_scale=(cfg['base'],cfg.get('res',1.),cfg.get('gain',1.))
  else:
   wc,ws,wg=self._cal_w[tier];n,k=self.out_features,self.in_features
   self.op=self.native_cls(wc.cuda(),ws.cuda(),wg.cuda(),torch.ones(k,device='cuda'),torch.ones((1,k),device='cuda',dtype=torch.bool),torch.ones((n,k),device='cuda',dtype=torch.bool),bm=128,interleave=True,num_stages=2)
   self.am,self.ai,self.union=self.op.am,self.op.ai,self.op.union;self.support_cache={}

def install(transformer):
 result=[]
 for li,layer in enumerate(transformer.layers):
  for part,child in list(layer.mlp_moe_gen.named_children()):
   if not isinstance(child,torch.nn.Linear):continue
   op=ffn.TermLinear(child);op._part=part;op._cal_w={}
   for tier in ['act2','dual']:
    cfg=policy[tier][part]['selected']
    if cfg['side']=='W':
     w=child.weight.detach().cuda().contiguous();d=torch.ones(w.shape[1],device='cuda');c,s,f,_=pack(w,d,True,True,w,base=cfg['base'],res=cfg.get('res',1.),gain=cfg.get('gain',1.));op._cal_w[tier]=(c.cpu(),s.cpu(),f.cpu());del w,c,s,f
   setattr(layer.mlp_moe_gen,part,op);result.append(op)
  if li%8==0:print('Prepared calibrated FFN',li,flush=True)
 return result

def forward(self,x):
 self.calls+=1;shape=x.shape;flat=x.reshape(-1,self.in_features).contiguous();rows=math.ceil(len(flat)/128)
 if rows not in self.support_cache:self.support_cache[rows]=tuple(t.expand(rows,-1).contiguous() for t in (self.am,self.ai,self.union))
 self.op.am,self.op.ai,self.op.union=self.support_cache[rows]
 key=f'{len(flat)}:{self.out_features}:{self.in_features}';cfg=launches.get((key,self._tier)) if self._mode.endswith('_pair') else None
 receipts[self._mode,self._part,key,'native' if cfg is None else cfg['layout']]+=1
 if cfg:
  if key not in self._pair_cache:
   p=PairedActivation(self.op,cfg['layout']=='direct',*self._a_scale);p.bm=cfg['bm'];p.stages=cfg['stages'];self._pair_cache[key]=p
  y=self._pair_cache[key](flat)
 elif self._a_scale!=(1.,1.,1.):
  p=pack(flat,self.op.d,True,True,self.op.am,base=self._a_scale[0],res=self._a_scale[1],gain=self._a_scale[2]);y=self.op.run_packed(flat,p)
 else:y=self.op(flat)
 return y.reshape(*shape[:-1],self.out_features)
ffn.TermLinear.configure=configure;ffn.TermLinear.forward=forward;ffn.install=install
if __name__=='__main__':
 snap=out.with_name(out.name+'_sources');snap.mkdir(parents=True,exist_ok=False)
 for p in Path(__file__).parent.glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
 try:runner.main()
 finally:
  f=out/'result.json'
  if f.exists():
   r=json.loads(f.read_text());r['activation_repair']=dict(calibration_path=str(calpath),benchmark_path=str(bpath),calibration_sha256=hashlib.sha256(calpath.read_bytes()).hexdigest(),benchmark_sha256=hashlib.sha256(bpath.read_bytes()).hexdigest(),sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.glob('*.py')},receipts=[dict(mode=k[0],part=k[1],shape=k[2],layout=k[3],calls=v) for k,v in receipts.items()],scope='One fresh seed on existing evaluation scenes; FFN-only changes and BF16 attention. Receipts include warmup; eager times are diagnostic, not throughput. Global projection settings fitted on three layers and early phases only.')
   f.write_text(json.dumps(r,indent=2)+'\n')

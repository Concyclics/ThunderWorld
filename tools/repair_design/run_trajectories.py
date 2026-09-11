"""Research wrapper: new A/W assignments and feature-only attention candidates.

Uses the established full-trajectory runner. Timing is exploratory eager, not
production throughput. Overrides exist only in this process; defaults unchanged.
"""
from pathlib import Path
import sys,json,hashlib,math,argparse
from collections import Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/finding_resample'),str(ROOT/'tools/auto_budget')]
import run as runner
import ffn
from runtime import AttentionStudy,config,fp4
from thunderworld_fp4.config import RepairConfig
from dynamic import phase
from structured_attention import compile_mask,roles_from_kwargs
from merged_weight import MergedWeight

options=argparse.ArgumentParser(add_help=False)
options.add_argument('--merge-benchmark',type=Path,default=ROOT/'results/repair_design_20260911/merged_weight_benchmark.json')
options.add_argument('--attention-prior',type=Path)
extra,remaining=options.parse_known_args();sys.argv=[sys.argv[0],*remaining]
model=sys.argv[sys.argv.index('--model')+1]
out=Path(sys.argv[sys.argv.index('--output')+1])
prior_path=extra.attention_prior or ROOT/f'results/repair_design_20260911/{model}_attention_prior.npz'
with np.load(prior_path) as z:prior=z['density']
bench=json.loads(extra.merge_benchmark.read_text())
assert bench['complete']
launches={}
for row in bench['rows']:
 if row['model']!=model:continue
 best=max(row['speedups'],key=row['speedups'].get)
 if row['speedups'][best]>1.05:
  configs=[c for c in row['candidates'] if c['layout']==best and 'ms' in c]
  win=min(configs,key=lambda c:c['ms']);launches[(row['shape'],row['tier'])]=(best,win['bm'],win['stages'])
execution_receipts=Counter();mask_receipts={}
orig_install=ffn.install;orig_configure=ffn.TermLinear.configure;orig_forward=ffn.TermLinear.forward

def configure(self,mode):
 hybrid=mode in ('ffn_split_aw','ffn_split_aw_merged')
 actual=('ffn_wgt2' if self._repair_part=='down_proj' else 'ffn_act2') if hybrid else mode
 orig_configure(self,actual)
 self._merged_enabled=mode=='ffn_split_aw_merged';self._layout_cache={};self._actual_tier='wgt2' if actual=='ffn_wgt2' else 'act2'

def install(transformer):
 modules=orig_install(transformer)
 for layer in transformer.layers:
  for part,child in layer.mlp_moe_gen.named_children():
   if isinstance(child,ffn.TermLinear):child._repair_part=part
 return modules

def forward(self,x):
 if not getattr(self,'_merged_enabled',False) or self._actual_tier!='wgt2':return orig_forward(self,x)
 shape=x.shape;flat=x.reshape(-1,shape[-1]).contiguous();key=f'{len(flat)}:{self.out_features}:{self.in_features}'
 cfg=launches.get((key,'wgt2'))
 execution_receipts[(self._repair_part,key,str(cfg))]+=1
 if cfg is None:return orig_forward(self,x)
 self.calls+=1;rows=math.ceil(len(flat)/128)
 if rows not in self.support_cache:self.support_cache[rows]=tuple(t.expand(rows,-1).contiguous() for t in (self.am,self.ai,self.union))
 self.op.am,self.op.ai,self.op.union=self.support_cache[rows]
 layout,bm,stages=cfg
 if layout not in self._layout_cache:self._layout_cache[layout]=MergedWeight(self.op,pair=layout=='pair')
 op=self._layout_cache[layout];op.bm=bm;op.stages=stages
 return op(flat).reshape(*shape[:-1],self.out_features)
# Constructor calls configure('ffn_act2') before projection names are assigned;
# only hybrid modes use _repair_part.
ffn.TermLinear.configure=configure;ffn.TermLinear.forward=forward;ffn.install=install
orig_dispatch=AttentionStudy.dispatch

@torch.inference_mode()
def dispatch(self,q,k,v,**kwargs):
 if self.mode not in ('attn_struct_a50','attn_energy_a50'):return orig_dispatch(self,q,k,v,**kwargs)
 if kwargs.get('is_causal',False):return orig_dispatch(self,q,k,v,**kwargs)
 if kwargs.get('attn_mask') is not None or kwargs.get('parallel_config') is not None:raise ValueError('Only unmasked noncausal Generator attention')
 if not hasattr(self,'_prior_cache'):self._prior_cache={};self._layout_keys={};self._quota_cache={};self._prior_receipts={}
 h=q.shape[2];nt=math.ceil(k.shape[1]/128);mb=math.ceil(q.shape[1]/128)
 if self.mode=='attn_energy_a50':
  key=(h,nt)
  if key not in self._quota_cache:
   # Exact half of global tile count, same as structured candidate (H is even).
   counts=[nt//2+(int(nt%2 and head<h//2)) for head in range(h)]
   cfg=RepairConfig(coverage=0,terms=2,head_budgets=tuple((i,(c-.25)/nt if c else 0.) for i,c in enumerate(counts)),query_tile=32,splits=2,transpose_v=True)
   assert list(cfg.head_counts(h,k.shape[1]))==counts
   self._quota_cache[key]=(cfg,torch.tensor(counts,device=q.device,dtype=torch.int32))
  cfg,counts=self._quota_cache[key]
  out=fp4(q,k,v,cfg,counts_tensor=counts)[0]
 else:
  if self.step==0 and self.layer==0 or self.branch not in self._layout_keys:
   qr,kr=roles_from_kwargs(self.kwargs);assert len(qr)==q.shape[1] and len(kr)==k.shape[1]
   key=hashlib.sha256(qr.tobytes()+kr.tobytes()).hexdigest();self._layout_keys[self.branch]=key
   if key not in self._prior_cache:
    masks={};receipts=[]
    for ph in range(4):
     for layer in range(prior.shape[1]):
      mask=compile_mask(prior[ph,layer],qr,kr,.5);assert int(mask.sum())==mask.size//2
      masks[ph,layer]=torch.from_numpy(mask).to(q.device)
      receipts.append(dict(phase=ph,layer=layer,selected=int(mask.sum()),tiles=mask.size))
    self._prior_cache[key]=masks;self._prior_receipts[key]=receipts;mask_receipts[key]=receipts
  key=self._layout_keys[self.branch];mask=self._prior_cache[key][phase(self.step,self.steps),self.layer]
  out=fp4(q,k,v,config(.5,2),repair_mask=mask)[0]
 self.counts['full']+=1
 return out
AttentionStudy.dispatch=dispatch

if __name__=='__main__':
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
 out.mkdir(parents=True,exist_ok=False)
 snapshot=out/'repair_sources';snapshot.mkdir()
 for p in Path(__file__).parent.glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
 # The established runner owns output creation. Preserve snapshots beside it.
 snapshot.rename(out.with_name(out.name+'_sources'));out.rmdir()
 runner.main()
 f=out/'result.json';r=json.loads(f.read_text());r['repair_design']={'scope':'Exploratory full-trajectory screen, one fresh seed. FFN split: W-only down, A-only up/gate. Merged layout enabled only above 1.05 local tuning ratio; optional layouts coexist during this screen so memory is not deployment memory. Attention priors are descriptive, not repair-benefit calibrated; both half-budget variants execute exactly half the padded tiles.','sources':sources,'prior_path':str(prior_path),'merge_benchmark_path':str(extra.merge_benchmark),'prior_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),'merge_benchmark_sha256':hashlib.sha256(extra.merge_benchmark.read_bytes()).hexdigest(),'mask_receipts':mask_receipts,'execution_receipts':[dict(part=k[0],shape=k[1],launch=k[2],calls=v) for k,v in execution_receipts.items()],'selected_launches':{f'{k[0]}|{k[1]}':v for k,v in launches.items()}}
 f.write_text(json.dumps(r,indent=2)+'\n')

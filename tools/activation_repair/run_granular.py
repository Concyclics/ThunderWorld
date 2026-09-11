"""Exploratory Edge layer/task/CFG gain policy, fit on early development steps."""
from pathlib import Path
import csv,json,hashlib
import run_trajectories as base
from runtime import AttentionStudy
ROOT=base.ROOT;out=base.out
assert base.model=='edge'
path=ROOT/'results/activation_repair_20260911/granular_calibration.csv'
pol={}
for r in csv.DictReader(path.open()):
 if r['model']!='edge' or r['mode']!='act2':continue
 side,b,res,gain=r['candidate'].split('|');assert side=='A'
 pol[r['module'],r['task'],int(r['branch'])]=(float(b),float(res),float(gain))
ctx={};old_pre=AttentionStudy.pre;old_args=base.runner.call_args;old_install=base.ffn.install;old_config=base.ffn.TermLinear.configure;old_forward=base.ffn.TermLinear.forward

def pre(self,module,args,kwargs):old_pre(self,module,args,kwargs);ctx['branch']=self.branch

def call_args(case,seed,steps):ctx['task']=case['task'];return old_args(case,seed,steps)

def install(transformer):
 result=old_install(transformer)
 for li,layer in enumerate(transformer.layers):
  for part,child in layer.mlp_moe_gen.named_children():
   if isinstance(child,base.ffn.TermLinear):child._name=f'{li}.{part}'
 return result

def configure(self,mode):
 old_config(self,'ffn_act2' if mode=='ffn_act_granular' else mode);self._mode=mode

def forward(self,x):
 if self._mode=='ffn_act_granular':self._a_scale=pol.get((self._name,ctx['task'],ctx['branch']),(1.,1.,1.))
 return old_forward(self,x)
AttentionStudy.pre=pre;base.runner.call_args=call_args;base.ffn.install=install;base.ffn.TermLinear.configure=configure;base.ffn.TermLinear.forward=forward
if __name__=='__main__':
 snap=out.with_name(out.name+'_sources');snap.mkdir(parents=True,exist_ok=False)
 for p in Path(__file__).parent.glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
 try:base.runner.main()
 finally:
  f=out/'result.json'
  if f.exists():
   r=json.loads(f.read_text());r['granular_calibration']=dict(policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),policy={str(k):v for k,v in pol.items()},sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.glob('*.py')},scope='All five fit-selected non-unit gains retained, including the one that regressed on held phases. Only sampled layers adjusted, remaining layers unchanged. No selection using fresh evaluation outputs.')
   f.write_text(json.dumps(r,indent=2)+'\n')

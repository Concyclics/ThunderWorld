"""Codec-only alternating timings plus changed-input and full-request equality checks.

Run from the repository root after the GPU regression tests. Pilot timings do not
replace the randomized complete-request benchmark matrix.
"""
import json,sys,time,statistics
from pathlib import Path
import torch
from thunderworld_fp4.cosmos.codec_graph import CodecGraphs
from thunderworld_fp4.cosmos.runner import CosmosRunner
from thunderworld_fp4.cosmos.settings import RunConfig
r=Path(__file__).resolve().parents[2];base=r/'runs5090/joint_e2e_20260914';out=r/'runs5090/pipeline_5090_20260914/codec_probe.json'
@torch.inference_mode()
def main():
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 cp=r/'tools/joint_inference/cases/robot_observe.json';case=json.loads(cp.read_text());case['image']=str((cp.parent/case['image']).resolve())
 run=CosmosRunner(r/'models/Cosmos3-Edge-a9d944e',RunConfig.from_dict(json.loads((base/'configs/joint125.json').read_text())),tuning=base/'profiles/joint125.json',assets=r/'data_local/cutlass5090_rebuilt_20260914/i2v_b125')
 run.prepare();run.warmup(case,35);run.warm_request(case,35)
 inputs={};originals={}
 for kind in ('encode','decode'):
  fn=getattr(run.pipe.vae,'_'+kind);originals[kind]=fn
  def observe(x,*args,_fn=fn,_kind=kind,**kw):inputs[_kind]=x.clone();return _fn(x,*args,**kw)
  setattr(run.pipe.vae,'_'+kind,observe)
 run.infer(case,7103,35)
 for kind,fn in originals.items():setattr(run.pipe.vae,'_'+kind,fn)
 cg=CodecGraphs(run.pipe.vae);rows=[]
 for kind,x in inputs.items():
  fn=originals[kind] if kind=='encode' else lambda x:originals['decode'](x,return_dict=False)[0]
  for i,v in enumerate((x,x*.875+.03125)):
   ref=fn(v).clone();y=cg.run(kind,v);torch.cuda.synchronize();print(kind,i,'equal',torch.equal(ref,y),flush=True)
   assert torch.equal(ref,y)
  times={'original':[],'graph':[]}
  for repeat in range(5):
   for label in (['original','graph'] if repeat%2==0 else ['graph','original']):
    torch.cuda.synchronize();t=time.perf_counter();fn(x) if label=='original' else cg.run(kind,x);torch.cuda.synchronize();times[label].append(time.perf_counter()-t)
  rows.append(dict(kind=kind,shape=list(x.shape),times=times));print(rows[-1],flush=True)
 cg.warming=False
 _,row=run.infer(case,7103,35)
 old=json.loads((base/'trajectory_00_joint125/result.json').read_text())['rows'][0]
 assert row['sha256']==old['sha256'];print('Full request identical',row['wall_seconds'],flush=True)
 out.write_text(json.dumps(dict(rows=rows,codec=cg.receipt(),request=row),indent=2)+'\n')
if __name__=='__main__':main()

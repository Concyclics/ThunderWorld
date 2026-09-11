"""Fit a model-local phase policy using BF16 captures and native measured costs."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch
import triton.testing
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers import Cosmos3OmniTransformer
from dynamic import ROOT,BudgetLinear,TIERS,phase,BOUNDS
from allocate import allocate


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['edge','nano'],required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((a.capture/'manifest.json').read_text());assert manifest['complete']
    modelpath=ROOT/'models'/('Cosmos3-Edge-a9d944e' if a.model=='edge' else 'Cosmos3-Nano-411f42a')
    model,info=Cosmos3OmniTransformer.from_pretrained(str(modelpath/'transformer'),torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    records=[];costs={};units=[]
    names=list(dict.fromkeys(r['name'] for r in manifest['rows']))
    for name in names:
        layer,part=name.split('.');src=getattr(model.layers[int(layer)].mlp_moe_gen,part)
        linear=BudgetLinear(src)
        for meta in (r for r in manifest['rows'] if r['name']==name):
            data=torch.load(a.capture/meta['file'],map_location='cpu',weights_only=True)
            x=data['x'].cuda();gold=data['y'].cuda().float()
            ph=phase(meta['step'],meta['steps']);key=f'{meta["task"]}|{meta["branch"]}|{ph}|{name}'
            length=math.ceil(meta['steps']*BOUNDS[ph+1])-math.ceil(meta['steps']*BOUNDS[ph])
            u=dict(key=key,task=meta['task'],name=name,phase=ph,branch=meta['branch'],
                   weight=2*meta['m']*meta['n']*meta['k']*length,importance=length,options={})
            shape=f'{meta["m"]}:{meta["n"]}:{meta["k"]}'
            for tier,(ac,wc) in TIERS.items():
                linear.tier=tier;y=linear(x).float()
                loss=float((y-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
                ck=shape+'|'+tier
                if ck not in costs:
                    full=x.repeat(math.ceil(meta['m']/len(x)),1)[:meta['m']].contiguous()
                    linear(full);torch.cuda.synchronize()
                    samples=[triton.testing.do_bench_cudagraph(lambda:linear(full),rep=10) for _ in range(3)]
                    costs[ck]=dict(ms=statistics.median(samples),samples_ms=samples,shape=shape,tier=tier,
                        bm=linear.op.bm,num_stages=linear.op.num_stages,
                        scope='Current captured rows repeated to real M only for latency; not extra quality samples')
                    del full
                u['options'][tier]=dict(extra=ac+int(wc),loss=loss,ms=costs[ck]['ms'])
            units.append(u);records.append(dict(**meta,unit_key=key,options=u['options']))
        del linear;gc.collect();torch.cuda.empty_cache()
        print('Calibrated',a.model,name,'units',len(units),'cost shapes',len(costs),flush=True)
    assert len({u['key'] for u in units})==len(units),'Use one seed/case per task for this compiler'
    policies={}
    for fraction in [.25,.5,.75]:
        bytask={task:allocate([u for u in units if u['task']==task],fraction) for task in dict.fromkeys(u['task'] for u in units)}
        assignments={k:v for result in bytask.values() for k,v in result['assignments'].items()}
        policies['auto'+str(int(fraction*100))]=dict(assignments=assignments,by_task=bytask)
    report=dict(schema_version=1,model=a.model,model_revision=manifest['model_revision'],complete=True,
        calibration_manifest_sha256=hashlib.sha256((a.capture/'manifest.json').read_bytes()).hexdigest(),
        calibration_cases=sorted({r['case'] for r in manifest['rows']}),calibration_seeds=sorted({r['seed'] for r in manifest['rows']}),
        scope='Greedy local error improvement per measured incremental ms, subject to per-task additional-FP4-product FLOP ceiling. Not a global optimum or quality guarantee.',
        runtime='Layer/CFG/normalized-step tier table; partial A supports recomputed on GPU from current-input energy. No output reuse. W1 kept dense to share storage across tiers.',
        phases=BOUNDS,policies=policies,costs=costs,units=units)
    (a.output/'policy.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'local_errors.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Saved',a.output/'policy.json',flush=True)


if __name__=='__main__':main()

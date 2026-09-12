"""Matched-output ablation; probe runs are excluded from formal leaderboards."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import statistics
import numpy as np
import torch
from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
from thunderworld_fp4.cosmos.cli import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--tuning',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=5)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    idle_guard();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    config=RunConfig.from_dict(json.loads(a.config.read_text()))
    case=json.loads(a.case.read_text())
    if case.get('image'):case['image']=str((a.case.parent/case['image']).resolve())
    result=dict(complete=False,case=case,rows=[],scope='Matched policies; simulated historical probes, not historical whole-run timing')
    rng=random.Random(120);gold={}
    for repeat in range(a.rounds):
        variants=['graph','eager','historical_probes','unfused','timing','trace'];rng.shuffle(variants)
        for variant in variants:
            idle_guard()
            cfg=replace(config,execution='eager' if variant=='eager' else 'graph',
                debug=variant if variant in ('timing','trace') else 'off',strict_tuning=True,
                fuse_ffn=variant!='unfused')
            runner=CosmosRunner(a.model,cfg,tuning=a.tuning,audio=case['task']=='av').prepare()
            runner.warmup(case)
            runner.warm_request(case)
            sketches=[]
            if variant=='historical_probes':
                # The previous research entry unconditionally sampled every
                # output and synchronized via cpu().numpy() at each call.
                def hook(module,args,output):
                    values=[]
                    for value in output:
                        if value is None:values.append(np.zeros(256,np.float32));continue
                        tensor=value[0].flatten()
                        indices=torch.linspace(0,len(tensor)-1,256,device=tensor.device).long()
                        values.append(tensor[indices].float().cpu().numpy())
                    sketches.append(np.stack(values))
                runner.pipe.transformer.register_forward_hook(hook)
                from diffusers import UniPCMultistepScheduler
                original=runner._context
                def legacy_context(*args,**kwargs):
                    original(*args,**kwargs)
                    runner.pipe.scheduler=UniPCMultistepScheduler.from_config(runner.pipe.scheduler.config)
                runner._context=legacy_context
            for seed in (93001,93002,93003):
                sketches.clear();idle_guard()
                _,row=runner.infer(case,seed)
                idle_guard()
                if seed in gold and gold[seed]!=row['sha256']:raise AssertionError('Ablation changed output')
                gold[seed]=row['sha256']
                result['rows'].append(dict(variant=variant,round=repeat,**row))
                a.output.write_text(json.dumps(result,indent=2)+'\n')
            runner.engine.clear();del runner
            original=None;legacy_context=None;hook=None
            gc.collect();torch.cuda.empty_cache()
    result['medians']={v:statistics.median(r['wall_seconds'] for r in result['rows'] if r['variant']==v)
                       for v in variants}
    result['complete']=True;a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

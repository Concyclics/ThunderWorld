"""Verify frozen launch optimization against a complete untuned quality trajectory."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import torch
from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
from thunderworld_fp4.cosmos.cli import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--quality-matrix',type=Path,required=True)
    p.add_argument('--tuning-root',type=Path,required=True)
    p.add_argument('--jobs',nargs='+',required=True,help='case:config pairs')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    matrix=json.loads(a.quality_matrix.read_text())
    if not matrix['complete']:raise ValueError('Completed quality trajectories required')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    result=dict(complete=False,rows=[])
    for pair in a.jobs:
        case,config=pair.split(':')
        job=next(j for j in matrix['jobs'] if j['case']==case and j['config']==config)
        reference=json.loads((Path(job['output'])/'result.json').read_text())
        gold=next(r for r in reference['rows'] if r['seed']==91001)
        cfg=replace(RunConfig.from_dict(reference['config']),strict_tuning=True)
        idle_guard()
        runner=CosmosRunner(a.model,cfg,tuning=a.tuning_root/f'{case}_{config}.json',audio=case=='av_pottery').prepare()
        runner.warmup(reference['case'])
        _,row=runner.infer(reference['case'],91001)
        if row['sha256']!=gold['sha256']:raise AssertionError(f'Tuned full trajectory changed: {pair}')
        result['rows'].append(dict(case=case,config=config,seed=91001,steps=35,bitwise=True,sha256=row['sha256']))
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        runner.engine.clear();del runner;gc.collect();torch.cuda.empty_cache()
    result['complete']=True;a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

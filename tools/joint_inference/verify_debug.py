"""Full-model output equivalence across execution/debug modes; not a speed test."""
import argparse,gc,json
from pathlib import Path
from dataclasses import replace
import torch
from thunderworld_fp4.cosmos import CosmosRunner,RunConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=RunConfig.from_dict(json.loads(a.config.read_text()));case=json.loads(a.case.read_text())
    if case.get('image'):case['image']=str((a.case.parent/case['image']).resolve())
    report=dict(complete=False,rows=[]);reference=None
    for execution in ('eager','graph'):
        for debug in ('off','timing','trace'):
            runner=CosmosRunner(a.model,replace(cfg,execution=execution,debug=debug),audio=case['task']=='av').prepare()
            runner.warmup(case,steps=2)
            _,row=runner.infer(case,90991,steps=2)
            if reference is None:reference=row['sha256']
            assert row['sha256']==reference,(execution,debug)
            if debug=='off':assert row['diagnostics']=={'mode':'off'}
            else:assert len(row['diagnostics']['transformer_step_ms'])==row['calls']
            report['rows'].append(dict(execution=execution,debug=debug,sha256=row['sha256'],bitwise=True))
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            runner.engine.clear();del runner;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

"""Full-request CUPTI trace, separate from uninstrumented latency measurements."""
import argparse
import json
from pathlib import Path
import torch
from thunderworld_fp4.cosmos.settings import RunConfig
from thunderworld_fp4.cosmos.runner import CosmosRunner
from thunderworld_fp4.cosmos.cli import idle_guard

ROOT=Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',choices=['bf16','joint125','cutlass150'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--codec-graph',action='store_true')
    p.add_argument('--ar-cache',action='store_true')
    p.add_argument('--fuse-norm',action='store_true')
    for flag in ('first-frame','fuse-pointwise','static-rotary','fuse-calibrated'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'runs5090/joint_e2e_20260914'
    config=RunConfig.from_dict(json.loads((previous/f'configs/{a.name}.json').read_text()))
    casepath=ROOT/'tools/joint_inference/cases/robot_observe.json'
    case=json.loads(casepath.read_text());case['image']=str((casepath.parent/case['image']).resolve())
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    idle_guard()
    runner=CosmosRunner(ROOT/'models/Cosmos3-Edge-a9d944e',config,
        tuning=previous/f'profiles/{a.name}.json',codec_graph=a.codec_graph,fuse_norm=a.fuse_norm,ar_cache=a.ar_cache,first_frame=a.first_frame,fuse_pointwise=a.fuse_pointwise,static_rotary=a.static_rotary,fuse_calibrated=a.fuse_calibrated,
        assets=None if a.name=='bf16' else ROOT/f'data_local/cutlass5090_rebuilt_20260914/i2v_b{a.name[-3:]}')
    runner.prepare();runner.warmup(case,35);runner.warm_request(case,35)
    for obj,method,label in [(runner.pipe.transformer,'forward','transformer'),
                             (runner.pipe,'_encode_video','video_encode'),
                             (runner.pipe.vae,'decode','video_decode')]:
        original=getattr(obj,method)
        def measured(*args,_fn=original,_label=label,**kwargs):
            with torch.profiler.record_function('TW/'+_label):return _fn(*args,**kwargs)
        setattr(obj,method,measured)
    if runner.ar_cache:
        original_prepare=runner.ar_cache.prepare_call
        def measured_prefill(kwargs):
            if runner.engine.call < runner.engine.branches:
                with torch.profiler.record_function('TW/ar_prefill'):return original_prepare(kwargs)
            return original_prepare(kwargs)
        runner.ar_cache.prepare_call=measured_prefill
    idle_guard()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function('TW/request'):
            _,row=runner.infer(case,7103,35)
    prof.export_chrome_trace(str(a.output/'trace.json'))
    reference=json.loads((previous/f'trajectory_00_{a.name}/result.json').read_text())['rows'][0]
    assert row['sha256']==reference['sha256']
    row['performance_eligible']=False
    (a.output/'receipt.json').write_text(json.dumps(dict(complete=True,row=row,**runner.receipt(),
        scope='Instrumented CUPTI request; not a performance leaderboard sample; output matches uninstrumented seed 7103'),indent=2)+'\n')
    print(a.output,flush=True)


if __name__=='__main__':main()

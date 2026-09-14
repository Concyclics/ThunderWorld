"""Installed entry point; local assets only, fresh outputs, explicit GPU selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def idle_guard():
    device=os.environ.get('CUDA_VISIBLE_DEVICES')
    if device not in ('0','3'):raise RuntimeError('Select physical GPU0 or GPU3 explicitly')
    rows=subprocess.check_output(['nvidia-smi','-i',device,
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    others=[s.strip() for s in rows if s.strip() and s.strip()!=str(os.getpid())]
    if others:raise RuntimeError('Formal timing requires an idle GPU; competing PIDs: '+','.join(others))


def policy_overrides(config,*,attention_policy=None,ffn_policy=None,ffn_policy_mode=None,projection_config=None):
    """Parse independent policy files through the same schema as RunConfig."""
    from dataclasses import replace
    from .settings import RunConfig, LinearSpec
    if attention_policy:
        attention=RunConfig.from_dict({'attention':json.loads(Path(attention_policy).read_text())}).attention
        config=replace(config,attention=attention)
    if ffn_policy:config=replace(config,ffn_policy=str(Path(ffn_policy).resolve()))
    if ffn_policy_mode is not None:config=replace(config,ffn_policy_mode=ffn_policy_mode)
    if projection_config:
        config=replace(config,projection=LinearSpec(**json.loads(Path(projection_config).read_text())))
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','warmup','infer','benchmark'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--case',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path)
    p.add_argument('--tuning',type=Path)
    p.add_argument('--tune',action='store_true')
    p.add_argument('--resource-mode',choices=['exclusive','shared-calibration'])
    p.add_argument('--compute-budget',type=float)
    p.add_argument('--budget-plan',type=Path)
    p.add_argument('--linear-backend',choices=['triton','cutlass-fused'])
    p.add_argument('--cutlass-library',type=Path)
    p.add_argument('--cutlass-small-library',type=Path)
    p.add_argument('--cutlass-direct-pack',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--joint-tuning',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--fuse-calibrated',action=argparse.BooleanOptionalAction,default=False,help='Fuse calibrated FP4 FFN ReLU-squared and prescaled amax')
    p.add_argument('--first-frame',action=argparse.BooleanOptionalAction,default=False,help='Encode only the single I2V conditioning frame')
    p.add_argument('--fuse-pointwise',action=argparse.BooleanOptionalAction,default=False,help='Exact DM RoPE and ReLU-squared fusion; requires AR cache')
    p.add_argument('--static-rotary',action=argparse.BooleanOptionalAction,default=False,help='Refresh rotary tables once per I2V request and CFG branch')
    p.add_argument('--ar-cache',action=argparse.BooleanOptionalAction,default=False,
                   help='Request-local AR prefill for static-text I2V; includes prefill in request timing')
    p.add_argument('--codec-graph',action=argparse.BooleanOptionalAction,default=False,
                   help='Experimental resident VAE CUDA Graphs; warm every request shape first')
    p.add_argument('--fuse-norm',action=argparse.BooleanOptionalAction,default=False,
                   help='Experimental Edge Nemotron RMSNorm pointwise fusion; validate trajectories before use')
    p.add_argument('--debug',choices=['off','timing','trace'])
    p.add_argument('--execution',choices=['graph','eager'])
    p.add_argument('--attention-policy',type=Path,help='Independent RepairPolicy JSON')
    p.add_argument('--ffn-policy',type=Path,help='Independent layer/phase FFN policy JSON')
    p.add_argument('--ffn-policy-mode',help='Named policy in the FFN allocation file')
    p.add_argument('--projection-config',type=Path,help='Independent LinearSpec JSON')
    p.add_argument('--cohort',type=Path)
    p.add_argument('--cohort-role',choices=['calibration','trajectory','validation','final'])
    p.add_argument('--steps',type=int,default=35)
    p.add_argument('--seeds',type=int,nargs='+',default=[91001])
    p.add_argument('--audio-output',choices=['latent','waveform'],default='waveform')
    p.add_argument('--save-outputs',action='store_true')
    a=p.parse_args()
    from contextlib import ExitStack
    resources=ExitStack()
    if a.steps<1 or not a.seeds:p.error('Positive steps and seeds required')
    cohort=None
    if a.cohort:
        from .cohorts import Cohort
        if a.cohort_role is None:p.error('--cohort-role is required with --cohort')
        cohort=Cohort.load(a.cohort);cohort.validate(a.cohort_role,a.seeds)
        if a.steps!=cohort.value['steps']:p.error('Cohort step mismatch')
    elif a.cohort_role:p.error('--cohort is required with --cohort-role')
    if a.operation!='prepare' and a.case is None:p.error('--case required')
    from dataclasses import replace
    from .settings import RunConfig
    config=RunConfig.from_dict(json.loads(a.config.read_text()))
    config=policy_overrides(config,attention_policy=a.attention_policy,ffn_policy=a.ffn_policy,
                            ffn_policy_mode=a.ffn_policy_mode,projection_config=a.projection_config)
    backend_updates={}
    if a.linear_backend is not None:backend_updates['linear_backend']=a.linear_backend
    if a.cutlass_library is not None:backend_updates['cutlass_library']=str(a.cutlass_library.resolve())
    if a.cutlass_small_library is not None:backend_updates['cutlass_small_library']=str(a.cutlass_small_library.resolve())
    if a.cutlass_direct_pack is not None:backend_updates['cutlass_direct_pack']=a.cutlass_direct_pack
    if a.joint_tuning is not None:backend_updates['joint_tuning']=a.joint_tuning
    if backend_updates:config=replace(config,**backend_updates)
    for field in ('debug','execution','resource_mode','compute_budget','budget_plan'):
        if getattr(a,field) is not None:config=replace(config,**{field:str(getattr(a,field)) if field=='budget_plan' else getattr(a,field)})
    if config.resource_mode=='shared-calibration':
        if a.operation=='benchmark' or a.tune:p.error('Shared calibration prohibits benchmark and tuning')
        from .resources import shared_worker, select_shared_device
        resources.enter_context(shared_worker())
        select_shared_device()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','3'):p.error('Set CUDA_VISIBLE_DEVICES=0 or 3')
    if a.operation=='benchmark':
        if config.debug!='off':p.error('Benchmark rejects diagnostic probes')
        config=replace(config,strict_tuning=True)
        idle_guard()
    if a.tune:idle_guard()
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,operation=a.operation,rows=[],config=config.to_dict())
    if cohort:result.update(cohort_sha256=cohort.fingerprint,cohort_role=a.cohort_role)
    def save():
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    runner=None
    try:
        import torch
        import numpy as np
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        from .runner import CosmosRunner
        case=json.loads(a.case.read_text()) if a.case else None
        if case and case.get('image'):
            image=Path(case['image'])
            if not image.is_absolute():image=(a.case.parent/image).resolve()
            case['image']=str(image)
            result['image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
        result['case']=case
        runner=CosmosRunner(a.model,config,tuning=a.tuning,tune=a.tune,assets=a.assets,
                            audio=case is not None and case['task']=='av',codec_graph=a.codec_graph,fuse_norm=a.fuse_norm,ar_cache=a.ar_cache,first_frame=a.first_frame,fuse_pointwise=a.fuse_pointwise,static_rotary=a.static_rotary,fuse_calibrated=a.fuse_calibrated)
        if a.tune:runner.table.checkpoint_path=a.output/'tuning.partial.json'
        runner.prepare(a.output/'assets' if a.operation=='prepare' else None)
        if a.operation!='prepare':runner.warmup(case,a.steps,a.audio_output)
        if a.operation=='benchmark':
            idle_guard();runner.warm_request(case,a.steps,a.audio_output);idle_guard()
        if os.environ.get('THUNDERWORLD_COLD_START'):
            result['cold_ready_seconds']=time.perf_counter()-float(os.environ['THUNDERWORLD_COLD_START'])
            result['cold_scope']='Fresh process through prepared Graph; existing disk kernel cache retained'
        runner.table.save(a.output/'tuning.json')
        result.update(runner.receipt());save()
        if a.operation in ('infer','benchmark'):
            for seed in a.seeds:
                if a.operation=='benchmark':idle_guard()
                signals,row=runner.infer(case,seed,a.steps,a.audio_output)
                if a.operation=='benchmark':idle_guard()
                if a.save_outputs or a.operation=='infer':
                    path=a.output/f'seed{seed}.npz';np.savez_compressed(path,**signals)
                    row['outputs']=path.name
                result['rows'].append(row);save()
        result.update(runner.receipt())
        result['complete']=True;save()
        print(a.output/'result.json',flush=True)
    except Exception as error:
        if runner is not None and a.tune:runner.table.save(a.output/'tuning.partial.json')
        result['error']=repr(error);save();raise
    finally:
        resources.close()


if __name__=='__main__':main()

"""Measure the Cosmos3 body with a fixed BF16 VAE boundary and no resident codec.

Encode each fixed image outside measurement, verify latent equivalence, remove
the codec, then measure fresh AR + all 35 denoising steps. No video quality is
evaluated here and no old denoising/AR result is reused in timed requests.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import weakref
from expand_worker import PROFILES,sha,write,idle_guard,refresh_protected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[301017,301018,301031])
    p.add_argument('--require-idle',action='store_true')
    a=p.parse_args()
    profile={**PROFILES[a.profile],'asset_dir':str(a.asset_dir)}
    if profile.get('vae_planes',0):raise ValueError('Body comparison always uses the same BF16 VAE boundary')
    variant=json.loads((a.variant/'variant.json').read_text())
    for name,digest in variant['variant_hashes'].items():
        if sha(a.variant/'thunderworld_fp4'/name)!=digest:raise ValueError('Changed runtime')
    fixed=json.loads(a.manifest.read_text());spec=copy.deepcopy(fixed['methods'][a.method])
    for key,field in [('assets','assets_sha256'),('budget_plan','plan_sha256')]:
        path=spec.get(key) if key=='assets' else spec['config'].get(key)
        if path and sha(Path(path)/'manifest.json' if key=='assets' else path)!=spec[field]:raise ValueError('Changed asset/plan')
    for row in fixed['cases']:
        if sha(row['case']['image'])!=row['image_sha256']:raise ValueError('Changed input')
    if a.require_idle:idle_guard()
    sys.path.insert(0,str(a.variant))
    import torch
    import numpy as np
    from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
    from thunderworld_fp4._expanded_runtime import finalize,tensor_bytes
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,model=fixed['model_key'],method=a.method,profile_name=a.profile,
        profile=profile,pid=os.getpid(),rows=[],performance_eligible=False,
        manifest_sha256=sha(a.manifest),variant_sha256=sha(a.variant/'variant.json'),worker_sha256=sha(__file__),
        device=torch.cuda.get_device_name(),quality_evaluation=False,
        scope='Cosmos3 body with cached BF16 image encoding; fresh AR/CFG + 35 DM steps + scheduler and latent outputs',
        excluded='VAE weights, encode/decode compute/workspace, video output conversion; not end-to-end video latency')
    def checkpoint(stage):
        report.update(stage=stage,updated=time.time());write(a.output/'result.json',report)
        print(json.dumps(dict(stage=stage,model=fixed['model_key'],method=a.method,profile=a.profile)),flush=True)
    try:
        checkpoint('prepare')
        external = a.method in fixed.get('adapters', {})
        spec['config'].update(debug='off',execution='eager' if external else 'graph',strict_tuning=False,resource_mode='shared-calibration')
        runner=CosmosRunner(fixed['model'],RunConfig.from_dict(spec['config']),assets=spec.get('assets'),
            ar_cache=True,first_frame=True,expansion=profile).prepare()
        from body_baselines import install
        report['baseline_adapter'] = install(runner, fixed, a.method)
        refresh_protected(runner)
        if profile.get('ar_scope') == 'ffn':
            quantized = runner.expansion_report['quantized']
            if not quantized or any('.mlp.' not in r['module'] for r in quantized):
                raise AssertionError('FFN-only expansion changed another AR component')
        conditions={};references={};active=[None]
        original_encode=runner.pipe._encode_video
        def collect(x):
            z=original_encode(x)
            conditions[active[0]]=z.detach().clone()
            return z
        checkpoint('prepare_graphs_and_bf16_conditions')
        with torch.inference_mode():
            for entry in fixed['cases']:
                case=entry['case'];active[0]=case['id'];runner.warmup(case,35)
            runner.pipe._encode_video=collect
            for entry in fixed['cases']:
                case=entry['case'];active[0]=case['id']
                runner._context(case,35,'waveform')
                args=runner._args(case,a.seeds[0],35);args['output_type']='latent'
                output=runner.pipe(**args)
                references[case['id']]=output.video.detach().cpu().clone()
                del output
        runner.pipe._encode_video=original_encode
        # Keep only the config required to size latents. The stub has no tensors
        # and rejects accidental codec execution instead of falling back to CPU.
        class CodecBoundary(torch.nn.Module):
            def __init__(self,config,dtype):super().__init__();self.config=config;self.dtype=dtype
            def encode(self,*args,**kwargs):raise RuntimeError('VAE excluded from body measurement')
            def decode(self,*args,**kwargs):raise RuntimeError('VAE excluded from body measurement')
        vae=runner.pipe.vae
        report['excluded_vae_state_bytes']=tensor_bytes(vae)
        ref=weakref.ref(vae)
        vae.clear_cache()
        runner.pipe.vae=CodecBoundary(vae.config,vae.dtype)
        del vae
        def cached_encode(x):
            z=conditions[active[0]]
            if x.shape[2]!=61 or z.device!=x.device:raise ValueError('Changed conditioning shape/device')
            # Give every request fresh storage; never let pipeline writes modify
            # the immutable BF16 boundary tensor or another request's input.
            return z.clone()
        runner.pipe._encode_video=cached_encode
        finalize(runner)
        if hasattr(runner,'release_host_assets'):runner.release_host_assets()
        else:
            for op in runner.engine.linears.values():
                if hasattr(op,'asset'):
                    op.asset={k:v for k,v in op.asset.items() if k in
                        ('selector','activation_scale','identity_prescale','mandatory_w_blocks')}
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        if ref() is not None:raise AssertionError('A hidden VAE reference is still retained')
        report['vae_fully_released']=True
        report['condition_tensor_bytes']=sum(z.numel()*z.element_size() for z in conditions.values())
        report['condition_sha256']={k:__import__('hashlib').sha256(z.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,z in conditions.items()}
        with torch.inference_mode():
            for entry in fixed['cases']:
                case=entry['case'];active[0]=case['id'];runner._context(case,35,'waveform')
                args=runner._args(case,a.seeds[0],35);args['output_type']='latent'
                result=runner.pipe(**args)
                if not torch.equal(result.video.cpu(),references[case['id']]):raise AssertionError('Codec isolation changed latent output')
                del result
        report['latent_isolation_bitwise_equal']=True
        del references
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        receipt=runner.receipt();resident=receipt['resident_model_state_bytes']
        ready=torch.cuda.memory_allocated()
        report.update(body_resident_model_bytes=resident,body_ready_allocated_bytes=ready,
            body_ready_non_model_bytes=ready-resident,runner=receipt)
        from body_inventory import inventory
        report['resident_inventory'] = inventory(runner)
        if report['resident_inventory']['total_bytes'] != resident:
            raise AssertionError('Resident inventory does not match runner receipt')
        checkpoint('measure_body')
        with torch.inference_mode():
            for entry in fixed['cases']:
                case=entry['case'];active[0]=case['id']
                for seed in a.seeds:
                    if a.require_idle:idle_guard()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    runner._context(case,35,'waveform')
                    args=runner._args(case,seed,35);args['output_type']='latent'
                    captures=len(runner.engine.cache)
                    result=runner.pipe(**args)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                    if len(runner.engine.cache)!=captures or runner.engine.call!=70:raise AssertionError('Body execution contract violated')
                    value=result.video.detach().float().cpu().numpy()
                    if not np.isfinite(value).all():raise AssertionError('Nonfinite latent')
                    name=case['id']+'__'+str(seed)+'.npz';np.savez(a.output/name,latent=value)
                    report['rows'].append(dict(case=case['id'],benchmark=entry['benchmark'],seed=seed,
                        body_seconds=seconds,body_peak_allocated_bytes=peak,body_peak_reserved_bytes=reserved,
                        body_non_model_peak_bytes=peak-resident,output=name,npz_sha256=sha(a.output/name),
                        latent_dtype=str(result.video.dtype),storage_dtype=str(value.dtype),
                        latent_sha256=__import__('hashlib').sha256(value.tobytes()).hexdigest(),
                        ar_prefills=runner.ar_cache.request_prefills,calls=runner.engine.call,performance_eligible=False))
                    if profile.get('ar', 'bf16') != 'bf16':
                        lengths=[key[1] for key in runner.ar_cache.seen]
                        prepared=runner.expansion_report.get('ar_prepared_shapes', [])
                        report['rows'][-1]['expanded_ar_compute']=dict(
                            prefix_lengths=lengths,
                            fp4_issued_flops=sum(2*r['padded_m']*r['n']*r['k']*r['terms']*lengths.count(r['m'])
                                for r in prepared),
                            arithmetic=profile['ar'], scope=profile.get('ar_scope', 'all'),
                            outside_frozen_dm_budget=True)
                    del result,value
                    if a.require_idle:idle_guard()
                    checkpoint('measure_body')
        report.update(complete=True,finished=time.time(),
            caveat='Body peak includes transformer weights, AR/KV, Graph pools, live latents/activations and workspace. It is not pure activation memory or physical system RAM.')
        checkpoint('complete')
    except Exception as exc:
        report.update(error=repr(exc));checkpoint('failed');raise


if __name__=='__main__':main()

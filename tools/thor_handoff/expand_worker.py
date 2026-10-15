"""One isolated expanded-I2V run; quality and memory evidence, diagnostic timing."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess


PROFILES = {
    'original': dict(trim=False),
    'trim': dict(trim=True),
    'ar_ffn1': dict(trim=True, ar='fp4_base', ar_scope='ffn'),
    'ar_ffn4': dict(trim=True, ar='fp4_four', ar_scope='ffn'),
    'ar_ffn8': dict(trim=True, ar='fp8', ar_scope='ffn'),
    'ar4_embed2': dict(trim=True, ar='fp4_four', embedding_planes=2),
    'ar3_embed2': dict(trim=True, ar='fp4_three', embedding_planes=2),
    'transformer2': dict(trim=True, ar='fp4_four', embedding_planes=2, auxiliary_planes=2),
    'full2': dict(trim=True, ar='fp4_four', embedding_planes=2, auxiliary_planes=2, vae_planes=2),
    'full1': dict(trim=True, ar='fp4_base', embedding_planes=1, auxiliary_planes=1, vae_planes=1),
    'vae1': dict(trim=True, vae_planes=1),
    'vae2': dict(trim=True, vae_planes=2),
}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def idle_guard():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    pids=[int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    others=[pid for pid in pids if pid!=os.getpid()]
    if others:raise RuntimeError('PAUSED: other GPU compute processes '+str(others))


def refresh_protected(runner):
    # install() runs before JointEngine replaces generation weights. Refresh
    # after prepare so that its initial scan cannot mislabel replaced DM weights.
    runner.expansion_report['protected']=[dict(module=component+'.'+name,shape=list(p.shape),
        dtype=str(p.dtype),bytes=p.numel()*p.element_size())
        for component,obj in (('transformer',runner.pipe.transformer),('vae',runner.pipe.vae))
        for name,p in obj.named_parameters()]
    runner.expansion_report['protected_scan_stage']='after_generation_replacement'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[301017,301018,301031])
    p.add_argument('--benchmarks',nargs='+',default=['rbench'])
    p.add_argument('--diagnose',action='store_true')
    p.add_argument('--require-idle',action='store_true')
    a=p.parse_args()
    variant=json.loads((a.variant/'variant.json').read_text())
    for name,digest in variant['variant_hashes'].items():
        if sha(a.variant/'thunderworld_fp4'/name)!=digest: raise ValueError('Variant changed: '+name)
    fixed=json.loads(a.manifest.read_text())
    spec=copy.deepcopy(fixed['methods'][a.method])
    for key,field in [('assets','assets_sha256'),('budget_plan','plan_sha256')]:
        path=spec.get(key) if key=='assets' else spec['config'].get(key)
        if path and sha(Path(path)/'manifest.json' if key=='assets' else path)!=spec[field]:
            raise ValueError('Changed asset/plan: '+str(path))
    cases=[c for c in fixed['cases'] if c['benchmark'] in a.benchmarks]
    if len(cases)!=len(a.benchmarks): raise ValueError('Expected one fixed input per benchmark')
    for row in cases:
        if sha(row['case']['image'])!=row['image_sha256']:raise ValueError('Changed input')
    if a.require_idle:
        idle_guard()
    sys.path.insert(0,str(a.variant))
    import torch
    import numpy as np
    from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
    from thunderworld_fp4._expanded_runtime import finalize,ARRepairLinear
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    profile={**PROFILES[a.profile],'asset_dir':str(a.asset_dir)}
    report=dict(complete=False,model=fixed['model_key'],method=a.method,profile_name=a.profile,
        profile=profile,manifest_sha256=sha(a.manifest),variant_sha256=sha(a.variant/'variant.json'),
        worker_sha256=sha(__file__),pid=os.getpid(),started=time.time(),rows=[],memory={},
        performance_eligible=False,quality_evaluation=not a.require_idle,
        timing_scope='Complete request, output conversion and VAE decode included; saving excluded',
        timing_qualification='Diagnostic only, fixed safe launch without performance selection',
        device=dict(name=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability()),
        measurements={},host_memory_note='CUDA allocations and host RSS overlap on Thor and must not be added')
    def checkpoint(stage):
        report.update(stage=stage,updated=time.time());write(a.output/'result.json',report)
        print(json.dumps(dict(stage=stage,model=fixed['model_key'],method=a.method,profile=a.profile)),flush=True)
    def mem(stage):
        report['memory'][stage]=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    try:
        checkpoint('prepare');t=time.perf_counter()
        spec['config'].update(debug='off',execution='graph',strict_tuning=False,resource_mode='shared-calibration')
        runner=CosmosRunner(fixed['model'],RunConfig.from_dict(spec['config']),assets=spec.get('assets'),
            ar_cache=True,first_frame=True,expansion=profile).prepare()
        refresh_protected(runner)
        torch.cuda.synchronize();report['prepare_seconds']=time.perf_counter()-t;mem('prepared')
        checkpoint('warmup')
        for row in cases:runner.warmup(row['case'],35)
        for row in cases:runner.warm_request(row['case'],35)
        finalize(runner)
        if hasattr(runner,'release_host_assets'):report['host_release']=runner.release_host_assets()
        else:
            # This isolated process will not prepare more shapes after finalization.
            for op in runner.engine.linears.values():
                if hasattr(op,'asset'):
                    op.asset={k:v for k,v in op.asset.items() if k in
                        ('selector','activation_scale','identity_prescale','mandatory_w_blocks')}
        gc.collect();mem('ready');checkpoint('measure')
        for entry in cases:
            case=entry['case']
            for seed in a.seeds:
                if a.require_idle:idle_guard()
                captures=len(runner.engine.cache)
                signals,row=runner.infer(case,seed,35)
                if len(runner.engine.cache)!=captures:raise AssertionError('Capture during measurement')
                row['performance_eligible']=False
                row['generated_fps']=case['frames']/row['wall_seconds']
                row['benchmark']=entry['benchmark']
                # Actual current-request AR prefix lengths, including both CFG
                # branches; this is separate from the frozen generation ledger.
                lengths=[key[1] for key in runner.ar_cache.seen]
                ar_ops=[op for op in runner.pipe.transformer.modules() if isinstance(op,ARRepairLinear)]
                row['expanded_ar_compute']=dict(prefix_lengths=lengths,
                    fp4_issued_flops=sum(2*((m+127)//128*128)*op.in_features*op.out_features*
                        (4 if op.default.cross else 1+int(op.default.activation>0)+int(op.default.weight>0))
                        for op in ar_ops for m in lengths),
                    fp4_linears=len(ar_ops),scope='AR projections/FFN only; excludes BF16 causal attention; outside generation C0')
                name=case['id']+'__'+str(seed)+'.npz'
                np.savez(a.output/name,**signals)
                row['output']=name;row['npz_sha256']=sha(a.output/name)
                report['rows'].append(row);del signals
                checkpoint('measure')
                if a.require_idle:idle_guard()
        report['runner']=runner.receipt()
        resident=report['runner']['resident_model_state_bytes']
        ready=report['memory']['ready']['allocated']
        report['measurements']=dict(resident_model_bytes=resident,ready_allocated_bytes=ready,
            ready_non_model_bytes=ready-resident,
            request_peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in report['rows']),
            peak_non_model_remainder_bytes=max(row['peak_allocated_bytes'] for row in report['rows'])-resident,
            caveat='Non-model remainder includes Graph pools, inputs, live outputs, activations and workspaces; not pure activation memory')
        if a.diagnose:
            checkpoint('diagnose')
            stages=[]
            # Separate request: probes cannot contaminate the timings above.
            boundary={'encode_exit':None}
            def wrap(obj,name,label):
                original=getattr(obj,name)
                def observed(*args,**kwargs):
                    torch.cuda.synchronize()
                    before=torch.cuda.memory_allocated();prior_peak=torch.cuda.max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    if label=='decode_video' and boundary['encode_exit'] is not None:
                        stages.append(dict(stage='ar_denoise_scheduler_interval',
                            seconds=start-boundary['encode_exit'],peak_allocated=prior_peak,
                            note='Boundary interval includes fresh AR, 35 DM steps and scheduler'))
                    if label=='decode_video':
                        latent=args[0] if args else kwargs['z']
                        torch.save(latent.detach().cpu(),a.output/'decode_latent.pt')
                    result=original(*args,**kwargs)
                    torch.cuda.synchronize()
                    stages.append(dict(stage=label,seconds=time.perf_counter()-start,entry_allocated=before,
                        exit_allocated=torch.cuda.memory_allocated(),peak_allocated=torch.cuda.max_memory_allocated(),
                        prior_interval_peak=prior_peak,peak_growth=torch.cuda.max_memory_allocated()-before))
                    torch.cuda.reset_peak_memory_stats()
                    if label=='encode_video':boundary['encode_exit']=time.perf_counter()
                    return result
                setattr(obj,name,observed)
                return original
            encode=wrap(runner.pipe,'_encode_video','encode_video')
            decode=wrap(runner.pipe.vae,'decode','decode_video')
            _,diagnostic=runner.infer(cases[0]['case'],a.seeds[0],35)
            runner.pipe._encode_video=encode;runner.pipe.vae.decode=decode
            matched=next(r for r in report['rows'] if r['seed']==a.seeds[0] and r['case']==cases[0]['case']['id'])
            if diagnostic['sha256']!=matched['sha256']:raise AssertionError('Memory probes changed output')
            report['diagnostics']=dict(stages=stages,output_unchanged=True,performance_eligible=False,
                caveat='Peak growth includes operator workspaces and transient decoded weights, not only activation tensors')
        # A complete asset inventory binds cold preparation and deployable storage.
        report['expanded_asset_files']={Path(r['asset_file']).name:dict(bytes=Path(r['asset_file']).stat().st_size,
            sha256=sha(r['asset_file'])) for r in report['runner']['expansion']['quantized'] if r.get('asset_file')}
        report.update(complete=True,finished=time.time());checkpoint('complete')
    except Exception as exc:
        report.update(error=repr(exc),finished=time.time());checkpoint('failed');raise


if __name__=='__main__':main()

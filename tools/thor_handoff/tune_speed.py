"""Thor-only profiling and bitwise-equivalent launch tuning for frozen repairs."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time
import sys

from build_baselines import write,sha
from compare_speed import idle_guard


def memory_snapshot():
    import psutil
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM'):result[key+'_bytes']=int(value.split()[0])*1024
    vm=psutil.virtual_memory()
    return dict(system_total_bytes=vm.total,system_available_bytes=vm.available,
        process_rss_bytes=psutil.Process().memory_info().rss,**result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('profile','tune','measure'))
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--manifest',type=Path,help='Frozen model/asset/input manifest; defaults to the historical Edge campaign')
    p.add_argument('--method',choices=('bf16','fp8','fp4_base','b1.5_protected_n','b2_joint','awq_marlin','arcquant','arcquant100','sharq_perf'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tuning',type=Path)
    p.add_argument('--variant',type=Path)
    p.add_argument('--attention-backend',choices=('bf16','fp4'),
        help='Explicit attribution ablation for fp4_base only; changes attention arithmetic, not a matched-quality optimization')
    p.add_argument('--cases',type=int,nargs='+',default=[0,1])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',help='Explicit measurement seeds; legacy default remains 301017 + repeat')
    p.add_argument('--common-runtime',action='store_true',
        help='Attribution only: disable method-specific norm/rotary/calibrated/codec fusions; retain Transformer Graph')
    p.add_argument('--component',choices=('all','linear','attention'),default='all')
    a=p.parse_args()
    if a.seeds is not None and (len(a.seeds)!=a.repeats or len(set(a.seeds))!=len(a.seeds)):
        p.error('--seeds must contain exactly --repeats distinct seeds')
    if a.attention_backend and a.method!='fp4_base':
        p.error('--attention-backend is restricted to the unbudgeted fp4_base attribution experiment')
    if a.operation=='tune' and a.method in ('awq_marlin','arcquant','arcquant100','sharq_perf'):
        p.error('External baseline tuning is outside this study')
    lock=(a.root/'logs/gpu-test.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle_guard()
    variant=None
    if a.variant:
        variant=json.loads((a.variant/'variant.json').read_text())
        for name,digest in variant['variant_hashes'].items():
            if sha(a.variant/'thunderworld_fp4'/name)!=digest:raise ValueError(f'Variant changed: {name}')
        if 'library_sha256' in variant:
            if sha(a.variant/'cutlass_sm110.so')!=variant['library_sha256']:raise ValueError('Variant library changed')
            if sha(a.variant/'cutlass_sm110.build.json')!=variant['build_receipt_sha256']:raise ValueError('Variant build receipt changed')
            if a.operation=='tune':raise ValueError('This CUTLASS variant uses a fixed qualified launch; use measure/profile')
        sys.path.insert(0,str(a.variant))
    import torch
    import triton
    from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability()!=(11,0):raise ValueError('Thor only')
    fixed=a.manifest or a.root/'comparisons/edge_20260922/manifest.json'
    manifest=json.loads(fixed.read_text())
    spec=copy.deepcopy(manifest['methods'][a.method])
    package=Path(manifest.get('package',a.root/'data/edge'))
    if sha(package/'package.json')!=manifest['package_sha256']:
        raise ValueError('Frozen package changed')
    for key,field in (('assets','assets_sha256'),('budget_plan','plan_sha256')):
        path=spec.get(key) if key=='assets' else spec['config'].get(key)
        if path and sha(Path(path)/'manifest.json' if key=='assets' else path)!=spec[field]:
            raise ValueError('Frozen assets/plan changed')
    for row in manifest['cases']:
        if sha(row['case']['image'])!=row['image_sha256']:raise ValueError('Frozen input changed')
    for receipt in manifest.get('asset_receipts',{}).values():
        if sha(Path(receipt['path'])/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('External asset manifest changed')
    if manifest.get('arc_build_sha256') and sha(Path(manifest['arc_build'])/'build.json')!=manifest['arc_build_sha256']:
        raise ValueError('ARC build changed')
    if a.attention_backend:spec['config']['attention_backend']=a.attention_backend
    requested_runtime_options=copy.deepcopy(spec.get('runtime_options',{}))
    if a.common_runtime:spec['runtime_options']={}
    spec['config']['resource_mode']='exclusive'
    spec['config']['strict_tuning']=False
    if spec['config']['debug']!='off':raise ValueError('Debug must stay off')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,method=a.method,operation=a.operation,started=time.time(),
        rows=[],performance_eligible=False,quality_evaluation=False,
        fixed_manifest_sha256=sha(fixed),fixed_manifest=str(fixed),model_key=manifest.get('model_key','edge'),
        effective_config=spec['config'],runtime_options=spec.get('runtime_options',{}),
        runtime_attribution=None if not a.common_runtime else dict(
            common_runtime=True,original_runtime_options=requested_runtime_options),
        attribution_ablation=None if a.attention_backend is None else dict(attention_backend=a.attention_backend),
        tuning_input=None if a.tuning is None else dict(path=str(a.tuning),sha256=sha(a.tuning)))
    report['memory']={'before_prepare':memory_snapshot()}
    if report['memory']['before_prepare']['system_available_bytes']<12*1024**3:
        raise RuntimeError('PAUSED: less than 12 GiB available unified memory')
    report['variant']=None if variant is None else dict(path=str(a.variant),receipt_sha256=sha(a.variant/'variant.json'),kernel_sha256=variant['kernel_sha256'])
    if variant and 'backend' in variant:
        report['effective_backend_override']=dict(linear=variant['backend'],attention=variant['attention'],
            library_sha256=variant['library_sha256'],accumulation_note=variant['accumulation_note'])
    write(a.output/'result.json',report)
    try:
        start=time.perf_counter()
        runner=CosmosRunner(manifest['model'],RunConfig.from_dict(spec['config']),
            assets=spec.get('assets'),ar_cache=True,first_frame=True,tuning=a.tuning,
            tune=a.operation=='tune',**spec.get('runtime_options',{})).prepare()
        from baseline_adapter import install
        report['adapter']=install(runner,a.root,a.method,manifest.get('external_assets'),manifest.get('arc_build'))
        if manifest.get('targets') and set(runner.engine.targets)!=set(manifest['targets']):
            raise ValueError('Model projection names differ from frozen manifest')
        report['prepare_seconds']=time.perf_counter()-start
        report['memory']['after_prepare']=memory_snapshot()
        if a.operation=='tune':
            table=runner.table
            table.checkpoint_path=a.output/'tuning.json'
            search=table.search
            def constrained(key,candidates,apply,fn,*,probes=()):
                kind,shape,settings=json.loads(key)
                if a.component!='all' and kind!=a.component:return
                if kind=='linear':
                    # Preserve the original request-budget denominator and
                    # the K128 repair lattice while changing CTA execution.
                    m=shape[0]
                    candidates=[v for v in candidates if triton.cdiv(m,v['bm'])*v['bm']==triton.cdiv(m,128)*128]
                elif kind=='attention':
                    # SM110 tcgen05 supports larger row tiles. No split count,
                    # summation order, quotas or repair terms are changed.
                    anchor=candidates[0]
                    candidates=[anchor]+[v for v in candidates[1:] if v['query_tile'] in (32,64)]
                return search(key,candidates,apply,fn,probes=probes)
            table.search=constrained
        # Prepare every request template before any measured request.
        for index in a.cases:
            case=manifest['cases'][index]['case']
            idle_guard()
            runner.warmup(case,35)
            if a.operation=='tune':
                runner.table.save(a.output/'tuning.json')
                runner.table.tune=False
            runner.warm_request(case,35)
            report['memory']['after_warmup_'+str(index)]=memory_snapshot()
            if a.operation=='tune':runner.table.tune=True
        report['postwarm_inventory']=runner.receipt()
        report['ready_to_measure_unix']=time.time()
        if report['adapter'] and a.method=='awq_marlin':
            report['postwarm_inventory']['resident_model_state_bytes']-=report['adapter']['scratch_bytes']
            report['postwarm_inventory']['inventory_note']='Marlin registered workspace excluded from resident weights; included in peak/scratch.'
        for index in a.cases:
            case=manifest['cases'][index]['case']
            for repeat in range(1 if a.operation in ('profile','tune') else a.repeats):
                idle_guard()
                if memory_snapshot()['system_available_bytes']<12*1024**3:
                    raise RuntimeError('PAUSED: less than 12 GiB available unified memory')
                captures=len(runner.engine.cache)
                if a.operation=='profile':torch.cuda.profiler.start()
                seed=a.seeds[repeat] if a.seeds is not None else 301017+repeat
                try:signals,row=runner.infer(case,seed,35)
                finally:
                    if a.operation=='profile':
                        torch.cuda.synchronize();torch.cuda.profiler.stop()
                idle_guard()
                if len(runner.engine.cache)!=captures:raise ValueError('Measured request created a new Graph')
                report.setdefault('first_measured_output_unix',time.time())
                import numpy as np
                shapes={k:list(v.shape) for k,v in signals.items()}
                if not shapes or not all(bool(np.isfinite(v).all()) for v in signals.values()):
                    raise ValueError('Missing/non-finite output')
                if 'video' not in shapes or case['frames'] not in shapes['video']:
                    raise ValueError('Incomplete video output')
                row.update(case_index=index,repeat=repeat,new_frame_fps=(case['frames']-1)/row['wall_seconds'],
                    frames=case['frames'],new_frames=case['frames']-1,output_shapes=shapes,
                    no_other_compute_at_boundaries=True,captures_before=captures,
                    performance_eligible=False,profiled=a.operation=='profile',memory_after=memory_snapshot())
                report['rows'].append(row)
                write(a.output/'result.json',report)
                print(json.dumps(dict(method=a.method,operation=a.operation,case=index,
                    seconds=row['wall_seconds'],fps=row['new_frame_fps'],hashes=row['sha256'])),flush=True)
                del signals
        if a.operation=='tune':runner.table.save(a.output/'tuning.json')
        report['runner']=runner.receipt()
        report['timings']=runner.timings
        report.update(complete=True,finished=time.time())
    except Exception as e:
        report['error']=repr(e)
        raise
    finally:write(a.output/'result.json',report)


if __name__=='__main__':main()

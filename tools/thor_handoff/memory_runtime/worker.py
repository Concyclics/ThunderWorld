"""One process / one model / one shape; full-request memory and latency evidence."""
import argparse
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shape',type=int,nargs=3,required=True,metavar=('WIDTH','HEIGHT','FRAMES'))
    p.add_argument('--mode',choices=('legacy','release','compact'),default='compact')
    p.add_argument('--seeds',type=int,nargs='+',default=[301017,301018,301031])
    p.add_argument('--verify-release',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from build_baselines import write,sha
    from compare_speed import idle_guard
    from tune_speed import memory_snapshot
    lock=(a.root/'logs/gpu-test.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle_guard()
    variant=json.loads((a.variant/'variant.json').read_text())
    if a.mode!='legacy' and not variant.get('qualified'):raise ValueError('Memory variant is not qualified')
    for name,digest in variant['variant_hashes'].items():
        if sha(a.variant/'thunderworld_fp4'/name)!=digest:raise ValueError('Variant changed: '+name)
    if sha(a.variant/'cutlass_sm110.so')!=variant['library_sha256']:raise ValueError('Library changed')
    sys.path.insert(0,str(a.variant))
    import torch
    from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
    torch.set_num_threads(4)
    if a.mode=='release':
        # Use the new loader/lifetime behavior with the original full A1
        # buffer, isolating host release from compact activation storage.
        import thunderworld_fp4.cutlass.packing as packing
        packing.pack_input=packing.pack_input_dense
        import thunderworld_fp4._sm110_cutlass as backend
        backend.pack_input=packing.pack_input_dense
    fixed=json.loads(a.manifest.read_text())
    if sha(Path(fixed['package'])/'package.json')!=fixed['package_sha256']:raise ValueError('Frozen package changed')
    spec=copy.deepcopy(fixed['methods'][a.method])
    for key,field in [('assets','assets_sha256'),('budget_plan','plan_sha256')]:
        path=spec.get(key) if key=='assets' else spec['config'].get(key)
        if path and sha(Path(path)/'manifest.json' if key=='assets' else path)!=spec[field]:raise ValueError('Frozen asset/plan changed')
    candidates=[r for r in fixed['cases'] if r['benchmark']=='rbench']
    if len(candidates)!=1:raise ValueError('Expected one frozen RBench input')
    case=copy.deepcopy(candidates[0]['case'])
    if sha(case['image'])!=candidates[0]['image_sha256']:raise ValueError('Frozen input changed')
    case.update(width=a.shape[0],height=a.shape[1],frames=a.shape[2])
    spec['config'].update(resource_mode='exclusive',debug='off',execution='graph',strict_tuning=False)
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,status='prepare',pid=os.getpid(),started=time.time(),model=fixed['model_key'],
        method=a.method,mode=a.mode,shape=a.shape,case=case,config=spec['config'],runtime_options={},
        manifest_sha256=sha(a.manifest),variant_receipt_sha256=sha(a.variant/'variant.json'),
        library_sha256=variant['library_sha256'],worker_sha256=sha(__file__),rows=[],memory={},
        quality_evaluation=False,performance_eligible=False,
        scope='Single fixed RBench input; one process per model/method/shape; no quality evaluation; external host memory sampler',
        timing_scope='Complete request including encoding, 35 denoising steps, decoding and CPU output; saving excluded')
    def checkpoint(status):
        report['status']=status;report['updated']=time.time();write(a.output/'result.json',report)
        write(a.output/'phase.json',dict(stage=status,pid=os.getpid(),updated=report['updated']))
    def memory(stage):
        row=memory_snapshot()
        row.update(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['memory'][stage]=row
    try:
        if memory_snapshot()['system_available_bytes']<12*2**30:raise RuntimeError('PAUSED: available RAM below 12 GiB')
        torch.cuda.reset_peak_memory_stats();memory('before_prepare');checkpoint('prepare')
        start=time.perf_counter()
        runner=CosmosRunner(fixed['model'],RunConfig.from_dict(spec['config']),assets=spec.get('assets'),
                            ar_cache=True,first_frame=True).prepare()
        report['prepare_seconds']=time.perf_counter()-start;memory('after_prepare');checkpoint('warmup')
        runner.warmup(case,35);memory('after_graph_warmup')
        runner.warm_request(case,35);memory('after_request_warmup')
        old_hash=None
        if a.verify_release:
            checkpoint('verify_before_release')
            signals,row=runner.infer(case,a.seeds[0],35);old_hash=row['sha256'];del signals
        if a.mode!='legacy':
            checkpoint('release_host_assets')
            report['host_release']=runner.release_host_assets()
            # Explicitly verify the fixed-shape lifetime contract.
            rejected=False
            try:runner.warmup({**case,'frames':case['frames']+4},35)
            except RuntimeError as exc:rejected='fresh runner' in str(exc)
            if not rejected:raise AssertionError('Unwarmed shape did not reject released assets')
            report['unwarmed_shape_rejected']=True
        else:report['host_release']=dict(released_host_tensor_bytes=0,scope='Legacy retained assets')
        gc.collect();memory('ready');checkpoint('measure')
        for seed in a.seeds:
            idle_guard()
            if memory_snapshot()['system_available_bytes']<12*2**30:raise RuntimeError('PAUSED: available RAM below 12 GiB')
            captures=len(runner.engine.cache)
            signals,row=runner.infer(case,seed,35)
            import numpy as np
            if not all(np.isfinite(value).all() for value in signals.values()):raise AssertionError('Nonfinite generated output')
            if len(runner.engine.cache)!=captures:raise AssertionError('Graph capture during measured request')
            if old_hash is not None and seed==a.seeds[0]:
                if row['sha256']!=old_hash:raise AssertionError('Host release changed output')
                report['host_release_bitwise_equal']=True
            row.update(generated_fps=case['frames']/row['wall_seconds'],memory_after=memory_snapshot(),
                       captures_before=captures,performance_eligible=False)
            report['rows'].append(row);del signals
            checkpoint('measure');idle_guard()
            print(json.dumps(dict(model=fixed['model_key'],method=a.method,mode=a.mode,shape=a.shape,
                                  seed=seed,seconds=row['wall_seconds'],peak_gib=row['peak_allocated_bytes']/2**30)),flush=True)
        report['runner']=runner.receipt();report['timings']=runner.timings
        if a.mode=='compact' and a.method not in ('bf16','fp8'):
            from thunderworld_fp4._sm110_compact_pack import compact_bytes
            footprints=[]
            for name,op in runner.engine.linears.items():
                for (m,spec_key),native in op.ops.items():
                    row=dict(module=name,m=m,k=op.in_features,a_blocks=native.na,**compact_bytes(m,op.in_features,native.na))
                    footprints.append(row)
            report['packing_footprints']=footprints
            report['packing_footprint_scope']='Per prepared invocation capacity; buffers can share graph pools, so these are not additive resident allocations'
        report.update(complete=True,finished=time.time());checkpoint('complete')
    except Exception as exc:
        report.update(error=repr(exc),finished=time.time())
        text=str(exc)
        if isinstance(exc,torch.OutOfMemoryError) or 'PAUSED:' in text:
            checkpoint('paused_resource')
        elif 'Compute budget exceeded' in text:
            checkpoint('infeasible_budget')
        else:checkpoint('failed')
        raise


if __name__=='__main__':main()

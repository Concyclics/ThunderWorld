"""Small, serial Nano speed/profile supplement using existing PRO6000 paths."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from bundle import ROOT
from baselines import sha, write
from optimize import source_identity

METHODS = ['fp4_base', 'awq_marlin', 'arcquant', 'arcquant100', 'sharq_perf',
           'b1.5_protected_n', 'b2_joint']
PROFILES = METHODS[2:]
CASE = 'paibench__av_660aa819-4a75-45d0-b79a-e6c2f2c321dc'


def prepare(a):
    workspace = ROOT.parent
    original = workspace/'judge_full_i2v_20260919/nano/final/manifest.json'
    controls = workspace/'judge_full_baselines_20260922/nano/final/manifest.json'
    m = copy.deepcopy(json.loads(original.read_text()))
    base = json.loads(controls.read_text())
    m['methods']['fp4_base'] = base['methods']['fp4_base']
    assets = workspace/'full_quality_ready_20260919/nano/awq_native'
    awq = copy.deepcopy(base['methods']['bf16'])
    awq['baseline_adapter'] = dict(method='marlin_awq', assets=str(assets),
        assets_sha256=sha(assets/'manifest.json'),
        implementation_sha256=sha(ROOT/'tools/contribution_study/awq_marlin.py'))
    awq['scope'] = 'Existing NVFP4 AWQ-lite codes, native Marlin W4A16; speed-only, numerical discrepancy retained'
    m['methods']['awq_marlin'] = awq
    m['methods'] = {k:m['methods'][k] for k in METHODS}
    m['cases'] = [next(c for c in m['cases'] if c['case']['id'] == CASE)]
    m['cases'][0]['seeds'] = [301017,301018]
    m.pop('source_output_hashes', None)
    m.update(scope='Single-scene two-seed speed supplement, five single-seed profiles; no quality evaluation',
        source_manifests={str(p):sha(p) for p in [original,controls]},
        frozen_source=source_identity(), worker_sha256=sha(__file__),
        methods_order=METHODS, profile_methods=PROFILES, physical_gpu=a.gpu,
        performance_eligible=False, quality_evaluation=False)
    from baseline_runtime import verify
    for method in METHODS:
        verify(m, method)
    assert m['steps']==35 and m['cases'][0]['case']['frames']==61
    assert sha(m['cases'][0]['case']['image'])==m['cases'][0]['image_sha256']
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'manifest.json', m)


def worker(a):
    import numpy as np
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    from thunderworld_fp4.cosmos.cli import idle_guard
    from baseline_runtime import make_runner
    m=json.loads((a.output/'manifest.json').read_text())
    if m['frozen_source']!=source_identity() or m['worker_sha256']!=sha(__file__):
        raise ValueError('Frozen execution source changed')
    dest=a.output/a.operation/a.method
    dest.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False, method=a.method, model='nano', started=time.time(), rows=[],
        operation=a.operation, manifest_sha256=sha(a.output/'manifest.json'),
        performance_eligible=False, quality_evaluation=False,
        numerical_status='unqualified_speed_only' if a.method=='awq_marlin' else 'existing_path_no_new_quality_evaluation')
    write(dest/'result.json',state)
    try:
        with shared_worker():
            state['resource']=select_shared_device()
            idle_guard()
            import torch
            torch.set_num_threads(8)
            torch.backends.cuda.matmul.allow_tf32=False
            state['gpu']=dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                torch=torch.__version__, cuda=torch.version.cuda)
            runner,adapter=make_runner(m,a.method)
            state['adapter']=adapter
            case=m['cases'][0]['case']
            runner.warmup(case,35)
            runner.warm_request(case,35)
            state['prepared']=runner.receipt()
            state['preparation_timings']=runner.timings
            captures=len(runner.engine.cache)
            for seed in ([301017] if a.operation=='profile' else [301017,301018]):
                idle_guard()
                if a.operation=='profile':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        signals,row=runner.infer(case,seed,35)
                    prof.export_chrome_trace(str(dest/'request.trace.json'))
                else:
                    signals,row=runner.infer(case,seed,35)
                idle_guard()
                assert len(runner.engine.cache)==captures
                assert signals and all(np.isfinite(v).all() for v in signals.values())
                assert 'video' in signals and 61 in signals['video'].shape
                ledger=row.get('compute_ledger')
                if ledger and not ledger['feasible']:raise ValueError('Repair budget exceeded')
                row.update(profiled=a.operation=='profile', new_frame_fps=60/row['wall_seconds'],
                    frames=61, no_other_compute_at_boundaries=True)
                state['rows'].append(row)
                write(dest/'result.json',state)
                del signals
            state.update(complete=True, finished=time.time())
    except Exception as e:
        state.update(error=repr(e),finished=time.time())
        raise
    finally:
        write(dest/'result.json',state)


def report(a):
    sys.path.insert(0,str(ROOT/'tools/thor_handoff'))
    from section54_protocol import write_csv
    from section54_profiles import category
    speeds=[];requests=[];breakdowns=[];kernels=[]
    for method in METHODS:
        f=a.output/'speed'/method/'result.json'
        if not f.exists():continue
        d=json.loads(f.read_text())
        if not d['complete']:continue
        rows=d['rows'];med=statistics.median(r['wall_seconds'] for r in rows)
        speeds.append(dict(device='PRO6000',model='nano',method=method,requests=len(rows),
            median_seconds=med,fps=60/med,
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
            peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/2**30,
            numerical_status=d['numerical_status'],performance_eligible=False))
        for row in rows:requests.append(dict(row,method=method,source=str(f),source_sha256=sha(f)))
    for method in PROFILES:
        trace=a.output/'profile'/method/'request.trace.json'
        result=trace.parent/'result.json'
        if not result.exists() or not json.loads(result.read_text())['complete']:continue
        totals={k:0. for k in ['preparation_selection','projection_ffn','attention','combine','other']}
        groups={}
        for e in json.loads(trace.read_text())['traceEvents']:
            if e.get('cat')!='kernel' or e.get('ph')!='X':continue
            name=e['name'];kind=category(name)
            # PRO uses existing Triton/mangled CUTLASS paths; Thor uses demangled Nsight names.
            if name=='gemm' or name.startswith('_ZN7cutlass13device_kernel') or 'void marlin::Marlin<' in name:
                kind='projection_ffn'
            elif 'pytorch_flash::flash_' in name:kind='attention'
            elif 'reorder_x_kernel<' in name or 'fused_sparse_residual_x_kernel<' in name:kind='preparation_selection'
            duration=float(e['dur'])/1e6
            totals[kind]+=duration
            g=groups.setdefault(name,dict(kernel=name,category=kind,calls=0,seconds=0.))
            g['calls']+=1;g['seconds']+=duration
        assert sum(totals.values())>0
        for r in groups.values():kernels.append(dict(device='PRO6000',model='nano',method=method,**r))
        breakdowns.append(dict(device='PRO6000',model='nano',method=method,profiles=1,seed=301017,case=CASE,
            **{k+'_seconds':v for k,v in totals.items()},total_kernel_seconds=sum(totals.values()),
            source=str(trace),source_sha256=sha(trace),performance_eligible=False))
    write_csv(a.output/'speed.csv',speeds);write_csv(a.output/'requests.csv',requests)
    write_csv(a.output/'breakdown.csv',breakdowns);write_csv(a.output/'kernel_details.csv',kernels)
    write(a.output/'summary.json',dict(complete=len(speeds)==7 and len(breakdowns)==5,
        speed_rows=speeds,breakdown_rows=breakdowns,requests=len(requests),quality_evaluation=False,
        note='Single-scene speed diagnostic; two timings and one profile per applicable method; not a latency frontier'))


def run(a):
    state=dict(complete=False,completed=[],started=time.time(),gpu=a.gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),TW_SHARED_CALIBRATION_GPUS=str(a.gpu),
        TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1')
    try:
        for op,methods in [('speed',METHODS),('profile',PROFILES)]:
            for method in methods:
                state.update(operation=op,method=method);write(a.output/'status.json',state)
                f=a.output/op/method/'result.json'
                if f.exists():
                    if not json.loads(f.read_text())['complete']:raise RuntimeError('Incomplete attempt retained: '+str(f))
                else:
                    log=a.output/(op+'-'+method+'.log')
                    cmd=[sys.executable,str(Path(__file__).resolve()),op,'--method',method,'--output',str(a.output),'--gpu',str(a.gpu)]
                    with log.open('x') as stream:subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                state['completed'].append(op+':'+method);write(a.output/'status.json',state)
                report(a)
        state.update(complete=True,finished=time.time())
    except Exception as e:
        state.update(error=repr(e),finished=time.time());raise
    finally:write(a.output/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','run','speed','profile','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=[0,3],default=3)
    parser.add_argument('--method',choices=METHODS)
    args=parser.parse_args()
    if args.operation in ('speed','profile'):worker(args)
    else:globals()[args.operation](args)

"""Thor request latency / unified-memory diagnostics using the existing runner.

These initial SM110 portability measurements are not a tuned performance table.
Graph warmup, model loading and offline preparation are reported separately.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/fp4-repair/src'), str(ROOT/'tools/fp4_handoff')]
from bundle import check_package


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def memory():
    import psutil
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            status[key+'_bytes'] = int(value.split()[0])*1024
    vm = psutil.virtual_memory()
    return dict(system_total_bytes=vm.total, system_available_bytes=vm.available,
                process_rss_bytes=psutil.Process().memory_info().rss, **status)


def command(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return dict(command=args, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--model', choices=('edge','nano'), required=True)
    p.add_argument('--method', choices=('bf16','fp8','fp4_base','b1.5_protected_n','b2_joint',
                                      'b2.5_ffn_guarded','fp4_three','fp4_four'), required=True)
    p.add_argument('--case', type=int, choices=(0,1), default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seeds', type=int, nargs='+', default=[301017,301018,301031])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-outputs', action='store_true')
    p.add_argument('--execution', choices=('graph','eager'))
    p.add_argument('--tuning', type=Path)
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    if platform.machine() != 'aarch64':
        raise RuntimeError('Thor harness requires ARM; desktop results use the existing entry point')
    lock = (a.root/'logs/gpu-test.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = memory()
    if before['system_available_bytes'] < 12*1024**3:
        raise RuntimeError('PAUSED: less than 12 GiB available unified memory')
    import numpy as np
    import torch
    import triton
    from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
    if torch.cuda.get_device_capability() != (11,0):
        raise RuntimeError('This entry point is for Thor SM110')
    torch.set_num_threads(4)
    package_dir = a.root/'data'/a.model
    package = check_package(package_dir)
    source = json.loads((package_dir/'study.json').read_text())
    expected = package['model']
    model_files = json.loads((a.root/'data/model_files.json').read_text())
    matches = [(name,v) for name,v in model_files.items()
               if (v['repo'],v['revision']) == (expected['repo'],expected['revision'])]
    if len(matches) != 1:
        raise ValueError('No exact model revision in transfer manifest')
    folder, files = matches[0]
    model = a.root/'models'/folder
    verified = json.loads((model/'thunderworld_verified.json').read_text())
    if not verified['complete'] or verified['files'] != files['files']:
        raise ValueError('Model download/hash verification incomplete')
    spec = copy.deepcopy(source['methods'][a.method])
    cfg = spec['config']
    if cfg['linear_backend'] != 'triton' or cfg['debug'] != 'off':
        raise ValueError('Only debug-off portable native kernels are supported here')
    assets = None
    if spec.get('assets'):
        restored = a.root/'assets'/a.model
        receipt = json.loads((restored/'restored.json').read_text())
        if not receipt['complete'] or not receipt['all_tensor_hashes_match']:
            raise ValueError('Exact repair asset reconstruction incomplete')
        if receipt['package_sha256'] != hashlib.sha256((package_dir/'package.json').read_bytes()).hexdigest():
            raise ValueError('Assets are from a different recipe package')
        assets = restored/a.method/'assets'
        cfg['budget_plan'] = str(package_dir/a.method/'plan.json')
    if a.execution:
        cfg['execution'] = a.execution
    cfg['resource_mode'] = 'shared-calibration'
    cfg['strict_tuning'] = False
    case = copy.deepcopy(source['cases'][a.case]['case'])
    case['image'] = str(package_dir/case['image'])
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='thor-request-diagnostic-v1', complete=False, model=a.model,
        method=a.method, model_revision=expected, case=case, steps=source['steps'], debug='off',
        performance_eligible=False, status='preparing',
        limitations=['SM110 portability setup; not a tuned formal latency result.',
                     'Repair budgets are frozen logical policies; SM110 issued MMA padding needs separate qualification.',
                     'Unified RAM, Torch allocator and RSS overlap; never add these columns.'],
        environment=dict(machine=platform.machine(), torch=torch.__version__, triton=triton.__version__,
                         gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                         nvpmodel=command(['nvpmodel','-q']), gpu_state=command(['nvidia-smi'])),
        memory=dict(before_prepare=before), rows=[])
    write(a.output/'result.json', report)
    try:
        started = time.perf_counter()
        runner = CosmosRunner(model, RunConfig.from_dict(cfg), assets=assets, tuning=a.tuning,
                              ar_cache=True, first_frame=True, **spec.get('runtime_options',{})).prepare()
        report['prepare_wall_seconds'] = time.perf_counter()-started
        report['memory']['after_prepare'] = memory()
        report['status'] = 'warming'
        write(a.output/'result.json', report)
        runner.warmup(case, source['steps'])
        runner.warm_request(case, source['steps'])
        report['memory']['after_warmup'] = memory()
        report['runner'] = runner.receipt()
        report['status'] = 'running'
        write(a.output/'result.json', report)
        for repeat in range(a.repeats):
            seed = a.seeds[repeat % len(a.seeds)]
            start_memory = memory()
            signals, row = runner.infer(case, seed, source['steps'])
            row.update(repeat=repeat, memory_before=start_memory, memory_after=memory(),
                       generated_fps=case.get('frames',61)/row['wall_seconds'],
                       performance_eligible=False, platform='agx-thor-sm110')
            # Serialization is outside the existing runner's full-request timing.
            if a.save_outputs:
                file = f'output-{repeat:03d}-{seed}.npz'
                np.savez_compressed(a.output/file, **signals)
                row['npz'] = file
            report['rows'].append(row)
            write(a.output/'result.json', report)
            print(json.dumps(dict(method=a.method, repeat=repeat, seconds=row['wall_seconds'],
                                  fps=row['generated_fps'], peak_allocated=row['peak_allocated_bytes'])), flush=True)
        times = [r['wall_seconds'] for r in report['rows']]
        report['summary'] = dict(request_median_seconds=float(np.median(times)),
            request_p90_seconds=float(np.percentile(times,90)),
            median_generated_fps=case.get('frames',61)/float(np.median(times)),
            max_torch_allocated_bytes=max(r['peak_allocated_bytes'] for r in report['rows']),
            max_torch_reserved_bytes=max(r['peak_reserved_bytes'] for r in report['rows']))
        report['runner'] = runner.receipt()
        report.update(complete=True, status='complete')
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(a.output/'result.json', report)


if __name__ == '__main__':
    main()

"""Serial, matched Edge speed comparison on Thor; no generation-quality study.

Run `prepare`, then `run`. Every method/round uses a fresh process, identical
inputs and seeds, and complete warmup. Official eager-only baselines stay eager.
The resulting table measures this SM110 port, not a tuned hardware ceiling.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from build_baselines import sha, write

METHODS = ['bf16','fp8','fp4_base','awq_marlin','arcquant50','arcquant100',
           'sharq','b1.5_protected_n','b2_joint']
LABELS = dict(bf16='BF16',fp8='FP8 W8A8',fp4_base='FP4 W4A4',
              awq_marlin='NVFP4-AWQ W4A16',arcquant50='ARCQuant 50%',
              arcquant100='ARCQuant 100%',sharq='SharQ',
              **{'b1.5_protected_n':'Ours 1.5×','b2_joint':'Ours 2×'})


def compute_pids():
    r = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    try:
        return [int(line.split(',')[0]) for line in rows]
    except ValueError as e:
        raise RuntimeError('Cannot establish absence of other GPU compute: '+r.stdout) from e


def idle_guard():
    others = [pid for pid in compute_pids() if pid != os.getpid()]
    if others:
        raise RuntimeError('PAUSED: other GPU compute processes '+str(others))


def prepare(a):
    package = a.root/'data/edge'
    study = json.loads((package/'study.json').read_text())
    methods = {name:copy.deepcopy(study['methods'][name]) for name in METHODS if name in study['methods']}
    for name in METHODS:
        if name not in methods:
            methods[name] = copy.deepcopy(study['methods']['bf16'])
            if name in ('arcquant50','arcquant100','sharq'):
                methods[name]['config']['execution'] = 'eager'
    for name,spec in methods.items():
        cfg = spec['config']
        if cfg['debug'] != 'off':raise ValueError('Debug must be off')
        cfg['resource_mode'] = 'shared-calibration'
        cfg['strict_tuning'] = False
        if spec.get('assets'):
            spec['assets'] = str(a.root/'assets/edge'/name/'assets')
            spec['assets_sha256'] = sha(Path(spec['assets'])/'manifest.json')
            cfg['budget_plan'] = str(package/name/'plan.json')
            spec['plan_sha256'] = sha(cfg['budget_plan'])
    cases = copy.deepcopy(study['cases'])
    for row in cases:
        row['case']['image'] = str(package/row['case']['image'])
        if sha(row['case']['image']) != row['image_sha256']:
            raise ValueError('Input changed')
    rng = random.Random(220926)
    rounds = []
    for i in range(5):
        order = list(METHODS)
        rng.shuffle(order)
        rounds.append(dict(round=i,order=order,seeds=[301017,301018,301031]))
    model = a.root/'models/Cosmos3-Edge-a9d944e'
    record = dict(schema='thor-edge-speed-v1',created=time.time(),model=str(model),
        methods=methods,cases=cases,steps=35,rounds=rounds,concurrency=1,batch=1,
        package_sha256=sha(package/'package.json'),quality_evaluation=False,
        generation=dict(width=320,height=192,frames=61,steps=35,cfg=6,flow_shift=3,fps=24),
        timing_scope='Full request: host input setup + encode + 35 denoise steps + decode + CPU output; serialization excluded',
        limitations=['No per-device tuning was performed; no hardware-ceiling claim.',
                     'MAXN, clocks not locked; GPU process checks at request boundaries.',
                     'ARCQuant and SharQ retain official eager execution; other presets use Graph.',
                     'Policies are unchanged, but logical budget names are not SM110 padded MMA measurements.',
                     'Two fixed representative benchmark inputs measure speed, not dataset-wide quality.'])
    a.output.mkdir(parents=True,exist_ok=True)
    path = a.output/'manifest.json'
    if path.exists():raise FileExistsError('Do not overwrite frozen speed manifest')
    write(path,record)
    print(json.dumps(dict(manifest=str(path),methods=len(methods),requests=5*3*len(cases)*len(methods))))


def runner_for(a, manifest):
    from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
    from baseline_adapter import install
    spec = manifest['methods'][a.method]
    if sha(a.root/'data/edge/package.json') != manifest['package_sha256']:
        raise ValueError('Package changed')
    for key,field in (('assets','assets_sha256'),('budget_plan','plan_sha256')):
        path = spec.get(key) if key == 'assets' else spec['config'].get(key)
        if path and sha(Path(path)/'manifest.json' if key == 'assets' else path) != spec[field]:
            raise ValueError('Frozen assets/plan changed')
    verified = json.loads((Path(manifest['model'])/'thunderworld_verified.json').read_text())
    if not verified['complete']:raise ValueError('Model verification incomplete')
    for row in manifest['cases']:
        if sha(row['case']['image']) != row['image_sha256']:raise ValueError('Image changed')
    runner = CosmosRunner(manifest['model'],RunConfig.from_dict(spec['config']),
        assets=spec.get('assets'),ar_cache=True,first_frame=True,**spec.get('runtime_options',{})).prepare()
    adapter = install(runner,a.root,a.method)
    return runner,adapter


def worker(a):
    from run import memory,command
    import numpy as np
    import torch
    import triton
    lock = (a.root/'logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    idle_guard()
    if torch.cuda.get_device_capability() != (11,0):raise ValueError('Expected Thor')
    torch.set_num_threads(4)
    manifest = json.loads((a.output/'manifest.json').read_text())
    destination = a.output/'runs'/f'round-{a.round:02d}'/a.method
    destination.mkdir(parents=True,exist_ok=False)
    record = dict(complete=False,method=a.method,round=a.round,rows=[],
        manifest_sha256=sha(a.output/'manifest.json'),debug='off',quality_evaluation=False,
        performance_eligible=False,measurement_class='exclusive-checked untuned Thor port',
        environment=dict(torch=torch.__version__,triton=triton.__version__,
            gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
            nvpmodel=command(['nvpmodel','-q']),gpu_before=command(['nvidia-smi'])),
        memory_before=memory(),started=time.time())
    write(destination/'result.json',record)
    try:
        started = time.perf_counter()
        runner,adapter = runner_for(a,manifest)
        record['prepare_wall_seconds'] = time.perf_counter()-started
        record['adapter'] = adapter
        if record['memory_before']['system_available_bytes'] < 12*1024**3:
            raise RuntimeError('PAUSED: less than 12 GiB available RAM')
        for row in manifest['cases']:
            idle_guard()
            runner.warmup(row['case'],manifest['steps'])
            runner.warm_request(row['case'],manifest['steps'])
        record['warmup'] = runner.timings.copy()
        record['memory_after_warmup'] = memory()
        record['status'] = 'timing'
        write(destination/'result.json',record)
        for seed in manifest['rounds'][a.round]['seeds']:
            for row in manifest['cases']:
                idle_guard()
                captures = len(runner.engine.cache)
                signals,stats = runner.infer(row['case'],seed,manifest['steps'])
                idle_guard()
                if len(runner.engine.cache) != captures:
                    raise ValueError('A measured request captured a new Graph')
                stats.update(round=a.round,benchmark=row['benchmark'],
                    new_frame_fps=(row['case']['frames']-1)/stats['wall_seconds'],
                    no_other_compute_at_boundaries=True,memory_after=memory())
                # Check outputs are produced, but neither save/evaluate quality
                # nor compare them to a reference for this speed-only request.
                stats['output_shapes'] = {k:list(v.shape) for k,v in signals.items()}
                record['rows'].append(stats)
                write(destination/'result.json',record)
                print(json.dumps(dict(method=a.method,round=a.round,benchmark=row['benchmark'],
                    seed=seed,seconds=stats['wall_seconds'],fps=stats['new_frame_fps'])),flush=True)
                del signals
        record['runner'] = runner.receipt()
        record['environment']['gpu_after'] = command(['nvidia-smi'])
        record.update(complete=True,finished=time.time())
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        write(destination/'result.json',record)


def run(a):
    manifest = json.loads((a.output/'manifest.json').read_text())
    setup = json.loads((a.root/'logs/setup-status.json').read_text())
    if not setup['complete']:raise RuntimeError('Wait for migration verification before timing')
    for name in ('arcquant','sharq','awq_marlin'):
        probe = json.loads((a.output/(name+'-probe.json')).read_text())
        if not probe['complete']:raise RuntimeError('Native operator qualification missing: '+name)
    state = dict(complete=False,started=time.time(),total_runs=45,finished_runs=0)
    write(a.output/'status.json',state)
    for order in manifest['rounds']:
        for method in order['order']:
            path = a.output/'runs'/f'round-{order["round"]:02d}'/method
            previous = path/'result.json'
            if previous.exists():
                old = json.loads(previous.read_text())
                if old['complete'] and old['manifest_sha256'] == sha(a.output/'manifest.json'):
                    state['finished_runs'] += 1
                    continue
                raise RuntimeError('Incomplete run retained; inspect before retry: '+str(path))
            idle_guard()
            state.update(method=method,round=order['round'],updated=time.time())
            write(a.output/'status.json',state)
            log = a.output/'logs'/f'round-{order["round"]:02d}-{method}.log'
            log.parent.mkdir(exist_ok=True)
            cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'worker',
                '--root',str(a.root),'--output',str(a.output),'--method',method,'--round',str(order['round'])]
            with log.open('w') as stream:
                result = subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(error='Worker failed; see '+str(log),exit_code=result.returncode)
                write(a.output/'status.json',state)
                raise RuntimeError(state['error'])
            state['finished_runs'] += 1
            write(a.output/'status.json',state)
            report(a)
    state.update(complete=True,finished=time.time())
    write(a.output/'status.json',state)
    report(a)


def report(a):
    import numpy as np
    manifest = json.loads((a.output/'manifest.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((a.output/'runs').glob('*/*/result.json'))]
    complete = [r for r in records if r['complete'] and r['manifest_sha256'] == sha(a.output/'manifest.json')]
    rows = []
    for method in METHODS:
        for benchmark in ('all','rbench','paibench'):
            runs = [r for r in complete if r['method']==method]
            data = [row for r in runs for row in r['rows'] if benchmark=='all' or row['benchmark']==benchmark]
            if not data:continue
            times = [r['wall_seconds'] for r in data]
            rows.append(dict(method=method,benchmark=benchmark,requests=len(data),rounds=len(runs),
                median_seconds=float(np.median(times)),p90_seconds=float(np.percentile(times,90)),
                new_frame_fps=60/float(np.median(times)),
                max_allocated_gib=max(r['peak_allocated_bytes'] for r in data)/1024**3,
                max_reserved_gib=max(r['peak_reserved_bytes'] for r in data)/1024**3,
                prepare_median_seconds=float(np.median([r['prepare_wall_seconds'] for r in runs]))))
    summary = dict(complete=len(complete)==45,completed_runs=len(complete),total_runs=45,
        rows=rows,quality_evaluation=False,performance_eligible=False,
        limitations=manifest['limitations'],updated=time.time())
    write(a.output/'summary.json',summary)
    lines = ['# Thor Edge 速度对比','',
        f'完成 {len(complete)}/45 个方法轮次；只比较速度，不评估质量。', '',
        '同一 Edge 权重、两个固定 RBench/PAI-Bench 输入；320×192，61 帧，35 步，CFG=6；单进程、batch=1、debug=off。',
        '每轮独立加载/预热，随机交错方法，5 轮，每轮每个输入 3 个固定种子。',
        '延迟包含编码、生成、解码与 CPU 输出；加载、编译、建图另计。FPS=60/完整请求秒数（只计新生成帧）。', '',
        '| 方法 | 轮数/请求数 | 中位延迟 s | P90 s | 生成 FPS | Torch allocated GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['benchmark']=='all':
            lines.append(f'| {LABELS[row["method"]]} | {row["rounds"]}/{row["requests"]} | {row["median_seconds"]:.3f} | {row["p90_seconds"]:.3f} | {row["new_frame_fps"]:.3f} | {row["max_allocated_gib"]:.2f} |')
    lines += ['', '限制：Thor 移植版尚未做设备调优；不代表吞吐上限。MAXN，未锁频；每次计时前后检查无其他计算进程。',
        'ARCQuant/SharQ 使用官方 eager 路径，其余使用现有 Graph 配置；表中是这些完整实现的速度，不能单独归因为 GEMM。',
        'Torch 分配、进程 RSS 与系统共享内存不可相加。逻辑预算标签不代表已测定的 SM110 padding 后 MMA 比例。',
        '相对 BF16 的倍数请优先使用相同输入、轮次与种子的配对原始记录；样本不足时不作最终排名。','']
    (a.output/'report.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','run','worker','report'))
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--output',type=Path,default=Path('/home/chenhan/thunderworld/comparisons/edge_20260922'))
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--round',type=int,default=0)
    a = p.parse_args()
    globals()[a.operation](a)


if __name__=='__main__':main()

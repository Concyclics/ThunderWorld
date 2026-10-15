"""Speed-only Thor throughput: isolated runners on distinct CUDA streams.

Preparation and capture are serial. Each runner serves one request at a time.
Measured requests include host setup, all 35 steps, video decode and conversion.
Only request boundaries synchronize the owning stream. No quality score is run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
from pathlib import Path
import sys
import threading
import time
from build_baselines import write,sha
from compare_speed import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--method',choices=('b1.5_protected_n','b2_joint'),required=True)
    p.add_argument('--slots',type=int,choices=(1,2,4),required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--case',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    lock=(a.root/'logs/gpu-test.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle_guard()
    variant=json.loads((a.variant/'variant.json').read_text())
    for name,digest in variant['variant_hashes'].items():
        if sha(a.variant/'thunderworld_fp4'/name)!=digest:raise ValueError(f'Variant changed: {name}')
    sys.path.insert(0,str(a.variant))
    import numpy as np
    import psutil
    import torch
    from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
    assert torch.cuda.get_device_capability()==(11,0)
    torch.set_num_threads(4)
    manifest=json.loads((a.root/'comparisons/edge_20260922/manifest.json').read_text())
    spec=copy.deepcopy(manifest['methods'][a.method]);spec['config']['resource_mode']='exclusive'
    spec['config']['strict_tuning']=False
    assert spec['config']['debug']=='off'
    case=manifest['cases'][a.case]['case']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(complete=False,method=a.method,slots=a.slots,repeats=a.repeats,case=case,
        variant_receipt_sha256=sha(a.variant/'variant.json'),quality_evaluation=False,
        performance_eligible=False,concurrency='single process, separate model/graph storage and CUDA stream per runner',
        rows=[],preparation_seconds=[],debug='off',execution='graph')
    write(a.output,report)
    runners=[];streams=[]
    try:
        for index in range(a.slots):
            if psutil.virtual_memory().available<20*1024**3:raise RuntimeError('Less than 20 GiB available; skip larger concurrency')
            start=time.perf_counter();stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                runner=CosmosRunner(manifest['model'],RunConfig.from_dict(spec['config']),assets=spec['assets'],
                    ar_cache=True,first_frame=True,**spec.get('runtime_options',{})).prepare()
                runner.warmup(case,35);runner.warm_request(case,35)
            stream.synchronize();runners.append(runner);streams.append(stream)
            report['preparation_seconds'].append(time.perf_counter()-start)
            write(a.output,report)
        idle_guard();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        barrier=threading.Barrier(a.slots+1)
        def work(index):
            runner=runners[index];stream=streams[index];rows=[]
            with torch.inference_mode(),torch.cuda.device(0),torch.cuda.stream(stream):
                barrier.wait()
                for repeat in range(a.repeats):
                    stream.synchronize();start=time.perf_counter()
                    runner._context(case,35,'waveform')
                    output=runner.pipe(**runner._args(case,301017+index*100+repeat,35))
                    video=np.asarray(output.video)
                    stream.synchronize();end=time.perf_counter()
                    if runner.engine.call!=70 or not np.isfinite(video).all():raise RuntimeError('Invalid completed request')
                    rows.append(dict(slot=index,repeat=repeat,start=start,end=end,seconds=end-start,
                        frames=case.get('frames',61),calls=runner.engine.call,
                        compute_ratio=runner.engine.budget_ledger['ratio']))
                    del video,output
            return rows
        with ThreadPoolExecutor(max_workers=a.slots) as pool:
            futures=[pool.submit(work,i) for i in range(a.slots)]
            begin=time.perf_counter();barrier.wait()
            for future in futures:report['rows'].extend(future.result())
            torch.cuda.synchronize();end=time.perf_counter()
        elapsed=end-begin;frames=sum(row['frames'] for row in report['rows'])
        report.update(complete=True,group_wall_seconds=elapsed,frames=frames,generated_fps=frames/elapsed,
            requests_per_second=len(report['rows'])/elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_available_bytes=psutil.virtual_memory().available,
            timing_scope='steady group, host request setup + generated video decode and NumPy conversion; no encoding to files')
        print(json.dumps({k:report[k] for k in ('method','slots','group_wall_seconds','generated_fps','peak_allocated_bytes')}),flush=True)
    except Exception as error:
        report['error']=repr(error);raise
    finally:write(a.output,report)


if __name__=='__main__':main()

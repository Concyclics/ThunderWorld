"""Resource-only GPU2/3 dispatch for the already frozen Nano supplement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pro6000_nano_section54 as frozen
from baselines import sha, write


def pids(gpu):
    text=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True)
    return [int(v.strip()) for v in text.splitlines() if v.strip()]


def checked_idle():
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    if gpu not in ('2','3'):raise RuntimeError('Only explicitly assigned GPU2/3 allowed')
    others=[pid for pid in pids(gpu) if pid!=os.getpid()]
    if others:raise RuntimeError('Competing GPU compute: '+str(others))


def main(a):
    if a.operation=='worker':
        # The historical CLI guard hard-codes 0/3. Extend only this host guard
        # in the isolated process; model/config/kernel files remain untouched.
        from thunderworld_fp4.cosmos import cli
        cli.idle_guard=checked_idle
        frozen.worker(SimpleNamespace(output=a.output,operation='profile',method=a.method,gpu=a.gpu))
        write(a.output/'profile'/a.method/'resource_override.json',dict(
            physical_gpu=a.gpu,controller_sha256=sha(__file__),
            change='Host idle guard accepts explicit GPU2/3; same frozen model, kernels, assets and request'))
        return
    state=json.loads((a.output/'serial_status_before_parallel.json').read_text())
    state.update(complete=False,gpus=[2,3],parallel_profiles=True,controller_sha256=sha(__file__),
        assignment={'2':['arcquant100','b1.5_protected_n'],'3':['arcquant','sharq_perf','b2_joint']})
    lock=threading.Lock()
    write(a.output/'parallel_status.json',state)

    def finish(method):
        d=json.loads((a.output/'profile'/method/'result.json').read_text())
        if not d['complete']:raise RuntimeError('Incomplete profile '+method)
        with lock:
            state['completed'].append('profile:'+method)
            state['updated']=time.time()
            write(a.output/'parallel_status.json',state)

    def lane(gpu, methods):
        if gpu==3:
            # Let the in-flight GPU3 request and its CUDA teardown finish.
            path=a.output/'profile/arcquant/result.json'
            while True:
                d=json.loads(path.read_text()) if path.exists() else {}
                if d.get('error'):raise RuntimeError(d['error'])
                if d.get('complete') and 4114603 not in pids(3):break
                time.sleep(5)
            # The old scheduler is stopped, so it cannot start a duplicate.
            pid=4029634
            cmd=Path('/proc')/str(pid)/'cmdline'
            if cmd.exists():
                if 'pro6000_nano_section54.py' not in cmd.read_text():raise RuntimeError('Supervisor identity changed')
                os.kill(pid,signal.SIGTERM)
                os.kill(pid,signal.SIGCONT)
            finish('arcquant')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TW_SHARED_CALIBRATION_GPUS=str(gpu),
            TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1')
        for method in methods:
            f=a.output/'profile'/method/'result.json'
            if not f.exists():
                while pids(gpu):time.sleep(10)
                cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--output',str(a.output),
                     '--gpu',str(gpu),'--method',method]
                with (a.output/('profile-'+method+'.log')).open('x') as log:
                    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finish(method)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,2,['arcquant100','b1.5_protected_n']),
                     pool.submit(lane,3,['sharq_perf','b2_joint'])]
            for future in futures:future.result()
        frozen.report(a)
        state.update(complete=True,finished=time.time(),operation='complete')
    except Exception as e:
        state.update(error=repr(e),finished=time.time())
        raise
    finally:
        write(a.output/'parallel_status.json',state)
        write(a.output/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['run','worker'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=[2,3])
    p.add_argument('--method',choices=frozen.PROFILES)
    main(p.parse_args())

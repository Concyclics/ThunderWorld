"""Two independent, bounded vLLM judge replicas; never modifies generation jobs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def run(a):
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / 'media').mkdir(exist_ok=True)
    if (a.root / 'PAUSE_JUDGE.json').exists():
        raise RuntimeError('Judge is paused by the user pending memory investigation')
    stop = a.root / 'STOP_SERVERS'
    if stop.exists():
        raise RuntimeError('Remove STOP_SERVERS explicitly before restarting this pool')
    children = []
    ending = False

    def terminate(signum, frame):
        nonlocal ending
        ending = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with ExitStack() as stack:
        job_lock = stack.enter_context((a.root / 'pool.lock').open('a+'))
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock = Path(f'/tmp/thunderworld-shared-calibration-{os.getuid()}.lock')
        global_file = stack.enter_context(lock.open('a+'))
        fcntl.flock(global_file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            for i, gpu in enumerate(a.gpus):
                dev = stack.enter_context(lock.with_name(lock.name + f'.gpu{gpu}').open('a+'))
                fcntl.flock(dev, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True)
                free, total = map(int, info.strip().split(','))
                if free < total * a.memory_fraction + 8192:
                    raise RuntimeError(f'GPU{gpu}: {free} MiB free is insufficient for judge plus 8 GiB reserve')
                port = a.port + i
                cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                    '--model', str(a.model), '--served-model-name', 'qwen3-vl-8b-fp8-judge',
                    '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                    '--max-model-len', '8192', '--gpu-memory-utilization', str(a.memory_fraction),
                    '--max-num-seqs', '32', '--max-num-batched-tokens', '8192',
                    '--limit-mm-per-prompt', json.dumps({'image': {'count': 1, 'width': 640, 'height': 576},
                                                       'video': {'count': 1, 'num_frames': 16, 'width': 320, 'height': 192}}),
                    '--media-io-kwargs', json.dumps({'video': {'num_frames': 16, 'video_backend': 'opencv'}}),
                    '--allowed-local-media-path', str(a.root.parent),
                    '--mm-processor-cache-gb', '2', '--seed', '9232026', '--no-enable-log-requests']
                env = os.environ.copy()
                env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
                ninja = importlib.util.find_spec('ninja')
                if ninja is not None:
                    for parent in Path(ninja.origin).parents:
                        ninja_bin = parent / 'bin'
                        if (ninja_bin / 'ninja').exists():
                            env['PATH'] = str(ninja_bin) + os.pathsep + env['PATH']
                            break
                if (a.root / 'deps').exists():
                    env['PYTHONPATH'] = str(a.root / 'deps') + os.pathsep + env.get('PYTHONPATH', '')
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    VLLM_PLUGINS='',
                    VLLM_CACHE_ROOT=str(a.root / f'vllm_cache_gpu{gpu}'),
                    TORCHINDUCTOR_CACHE_DIR=str(a.root / f'inductor_gpu{gpu}'))
                log = stack.enter_context((a.root / f'server_gpu{gpu}.log').open('a'))
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                    start_new_session=True, pass_fds=(global_file.fileno(), dev.fileno()))
                children.append((proc, gpu, port, cmd))
            def state(status):
                return dict(status=status, pid=os.getpid(), started_unix=started, control_root=str(a.root.resolve()),
                    replicas=[dict(pid=p.pid, gpu=g, port=t, command=c,
                                   endpoint=f'http://127.0.0.1:{t}/v1') for p, g, t, c in children])
            started = time.time()
            write(a.root / 'servers.json', state('starting'))
            ready = set()
            while not ending and not stop.exists():
                for proc, gpu, port, cmd in children:
                    if proc.poll() is not None:
                        raise RuntimeError(f'vLLM GPU{gpu} exited {proc.returncode}; see server_gpu{gpu}.log')
                    if port not in ready:
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                                if r.status == 200:
                                    ready.add(port)
                                    write(a.root / 'servers.json', state('ready' if len(ready) == len(children) else 'starting'))
                        except (OSError, TimeoutError):
                            pass
                if len(ready) < len(children) and time.time() - started > 1800:
                    raise TimeoutError('vLLM startup exceeded 30 minutes')
                time.sleep(3)
        except Exception as exc:
            write(a.root / 'server_failure.json', dict(error=repr(exc), unix=time.time()))
            raise
        finally:
            for proc, *_ in children:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            for proc, *_ in children:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            write(a.root / 'servers_stopped.json', dict(unix=time.time(), owner_pid=os.getpid()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 2])
    p.add_argument('--port', type=int, default=18860)
    p.add_argument('--memory-fraction', type=float, default=0.28)
    a = p.parse_args()
    if len(a.gpus) != len(set(a.gpus)) or not 0.1 <= a.memory_fraction <= 0.6:
        p.error('Use unique GPUs and a bounded memory fraction')
    run(a)

"""Independent-process inference under an inherited, exclusive GPU pool lease.

The coordinator owns the existing global/shared and device/exclusive locks.
Children inherit those open descriptions instead of competing for the device
lock again. The frozen numerical worker and all of its resume checks are used
unchanged. This is a resource adapter, not a different inference implementation.
"""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from baselines import sha, write
import judge_generation as frozen

HERE = Path(__file__).resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def lock_paths(gpu):
    base = Path(f'/tmp/thunderworld-shared-calibration-{os.getuid()}.lock')
    return base, base.with_name(base.name+f'.gpu{gpu}')


@contextmanager
def device_leases(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError('Distinct physical GPU indices required')
    with ExitStack() as stack:
        global_file = stack.enter_context(lock_paths(gpus[0])[0].open('a+'))
        fcntl.flock(global_file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        devices = {}
        for gpu in sorted(gpus):
            handle = stack.enter_context(lock_paths(gpu)[1].open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            devices[gpu] = handle.fileno()
        yield global_file.fileno(), devices


def verify_descriptors(gpu, global_fd, device_fd):
    if global_fd == device_fd:
        raise ValueError('Global and device leases must be distinct')
    for fd, path in zip((global_fd, device_fd), lock_paths(gpu)):
        actual, expected = os.fstat(fd), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Wrong inherited resource lease')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Inherited device lease differs from CUDA_VISIBLE_DEVICES')


def infer(a):
    job = frozen.verify_job(a.root)
    if a.gpu not in job['gpus']:
        raise ValueError('GPU not qualified for this frozen job')
    verify_descriptors(a.gpu, a.global_fd, a.device_fd)
    from thunderworld_fp4.cosmos import resources
    original = resources.shared_worker

    @contextmanager
    def inherited_worker(lock_path=None):
        if lock_path is not None:
            raise ValueError('Custom lock path cannot replace the inherited pool lease')
        verify_descriptors(a.gpu, a.global_fd, a.device_fd)
        yield

    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'concurrency_invocations.jsonl').open('a') as log:
        log.write(json.dumps(dict(utc=now(), pid=os.getpid(), gpu=a.gpu,
            coordinator_pid=a.coordinator_pid, source_sha256=sha(HERE),
            lock_protocol='Inherited global shared / physical-device exclusive lease',
            numerical_worker_sha256=sha(HERE.with_name('dataset_infer.py')),
            performance_eligible=False))+'\n')
    resources.shared_worker = inherited_worker
    try:
        from dataset_infer import run
        role = 'qualification' if a.qualification else 'final'
        return run(a.root/a.model/role/'manifest.json', a.method, a.output,
                   smoke=a.qualification, cases=a.cases,
                   require_source_hash=a.qualification)
    finally:
        resources.shared_worker = original


def child_command(root, model, method, output, gpu, leases, *, cases=None, qualification=False):
    global_fd, devices = leases
    command = [sys.executable, str(HERE), 'infer', '--root', str(root),
        '--model', model, '--method', method, '--output', str(output), '--gpu', str(gpu),
        '--global-fd', str(global_fd), '--device-fd', str(devices[gpu]),
        '--coordinator-pid', str(os.getpid())]
    if qualification:
        command.append('--qualification')
    if cases:
        command.extend(['--cases', *cases])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        TW_SHARED_CALIBRATION_GPUS=','.join(map(str, devices)),
        TW_SHARED_CALIBRATION_LOCK_SCOPE='device', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    return command, env, (global_fd, devices[gpu])


def stop_children(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def probe_group(a, stage, width, leases):
    # Always two identical short blocks, run sequentially or concurrently.
    # Each block reproduces four frozen outputs plus its A-B-A check.
    pending = list(range(2))
    active = []
    all_processes = []
    outputs = []
    start = time.monotonic()
    with ExitStack() as stack:
        try:
            while pending or active:
                if time.monotonic()-start > 1200:
                    raise TimeoutError('Concurrency probe exceeded 20 minutes')
                while pending and len(active) < width:
                    slot = pending.pop(0)
                    folder = a.output/f'{a.method}/stage{stage}_w{width}/block{slot}'
                    folder.mkdir(parents=True, exist_ok=False)
                    command, env, fds = child_command(a.root, 'edge', a.method,
                        folder/'generation', a.gpu, leases, qualification=True)
                    log = stack.enter_context((folder/'generation.log').open('w'))
                    process = subprocess.Popen(command, cwd=frozen.REPO, env=env,
                        pass_fds=fds, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active.append((process, folder))
                    all_processes.append(process)
                for process, folder in list(active):
                    code = process.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f'Probe child exited {code}: {folder}')
                    progress = json.loads((folder/'generation/progress.json').read_text())
                    if not progress['complete'] or not progress.get('roundtrip_bitwise_equal'):
                        raise ValueError('Incomplete qualification under concurrent execution')
                    rows = [row for p in (folder/'generation').glob('*/result.json')
                            for row in json.loads(p.read_text())['rows']]
                    if len(rows) != 4 or not all(r['source_signal_hashes_match'] for r in rows):
                        raise ValueError('Concurrent output differs from frozen reference')
                    outputs.append(dict(folder=str(folder), videos=len(rows),
                        request_seconds=[r['wall_seconds'] for r in rows],
                        progress_sha256=sha(folder/'generation/progress.json')))
                    active.remove((process, folder))
                if active:
                    time.sleep(.2)
        finally:
            stop_children(all_processes)
    seconds = time.monotonic()-start
    return dict(stage=stage, width=width, seconds=seconds, videos=8,
        videos_per_hour=8*3600/seconds, outputs=outputs, performance_eligible=False)


def probe(a):
    job = frozen.verify_job(a.root)
    if a.gpu not in job['gpus']:
        raise ValueError('Unqualified GPU')
    a.output.mkdir(parents=True, exist_ok=False)
    config = dict(job_sha256=sha(a.root/'manifest.json'), source_sha256=sha(HERE),
        methods=a.methods, gpu=a.gpu, widths=[1, 2, 1], jobs_per_stage=2,
        started_utc=now(), performance_eligible=False,
        scope='Operational short-block throughput including load, warmup, four saved videos, '
              'A-B-A check and serialization per child. Shared load; not a latency leaderboard.')
    write(a.output/'manifest.json', config)
    result = dict(complete=False, methods={}, source_sha256=sha(HERE), performance_eligible=False)
    write(a.output/'progress.json', result)
    with device_leases([a.gpu]) as leases:
        from thunderworld_fp4.cosmos.resources import free_memory
        available = free_memory()[a.gpu]
        if available < (12+2*16)*1024:
            raise RuntimeError('Insufficient memory for two Edge replicas plus 12 GiB reserve')
        try:
            for method in a.methods:
                a.method = method
                stages = []
                result['methods'][method] = dict(stages=stages, complete=False)
                for stage, width in enumerate(config['widths']):
                    print(json.dumps(dict(event='probe_start', method=method, width=width, utc=now())), flush=True)
                    row = probe_group(a, stage, width, leases)
                    stages.append(row)
                    write(a.output/'progress.json', result)
                    print(json.dumps(dict(event='probe_complete', method=method,
                        width=width, videos_per_hour=row['videos_per_hour'], utc=now())), flush=True)
                serial = [s['videos_per_hour'] for s in stages if s['width'] == 1]
                concurrent = next(s['videos_per_hour'] for s in stages if s['width'] == 2)
                result['methods'][method].update(complete=True,
                    serial_videos_per_hour=statistics.mean(serial),
                    concurrent_videos_per_hour=concurrent,
                    ratio_to_fastest_serial=concurrent/max(serial),
                    candidate_for_live_pilot=concurrent/max(serial) >= 1.10,
                    all_source_hashes_match=True)
                write(a.output/'progress.json', result)
            result.update(complete=True, completed_utc=now())
        except BaseException as error:
            result.update(paused=True, error=repr(error), stopped_utc=now())
            raise
        finally:
            write(a.output/'progress.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('infer', 'probe'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model', choices=('edge', 'nano'))
    p.add_argument('--method')
    p.add_argument('--methods', nargs='+', default=['arcquant100', 'b1.5_protected_n', 'b2_joint', 'arcquant', 'sharq_perf'])
    p.add_argument('--cases', nargs='+')
    p.add_argument('--qualification', action='store_true')
    p.add_argument('--global-fd', type=int)
    p.add_argument('--device-fd', type=int)
    p.add_argument('--coordinator-pid', type=int)
    a = p.parse_args()
    a.root, a.output = a.root.resolve(), a.output.resolve()
    if a.operation == 'infer':
        if None in (a.model, a.method, a.global_fd, a.device_fd, a.coordinator_pid):
            p.error('infer requires model/method and inherited lease descriptors')
        infer(a)
    else:
        probe(a)


if __name__ == '__main__':
    main()

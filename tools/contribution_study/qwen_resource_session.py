"""Resume a frozen judge campaign inside an enforced, monitored memory cgroup.

This changes only concurrency and process supervision. Questions, frame sampling,
quantization, response parsing, and existing valid answers stay hash-verified.
Launch using systemd-run --user with MemoryHigh=48G, MemoryMax=64G,
MemorySwapMax=0 and TasksMax=1024; refuse to run without those memory bounds.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from qwen_task_judge import verify_manifest
from qwen_task_protocol import now, read, sha, write


def cgroup_path():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            return Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    raise RuntimeError('A cgroup v2 memory boundary is required')


def resource_snapshot(group):
    def integer(name):
        value = (group / name).read_text().strip()
        return int(value) if value != 'max' else None

    def pairs(path):
        return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}

    processes = []
    for pid in sorted(set((group / 'cgroup.procs').read_text().split()), key=int):
        proc = Path('/proc') / pid
        try:
            status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
            smaps = {line.split(':', 1)[0]: int(line.split()[1])
                     for line in (proc / 'smaps_rollup').read_text().splitlines()
                     if ':' in line and len(line.split()) > 1 and line.split()[1].isdigit()}
            processes.append(dict(pid=int(pid), name=status['Name'].strip(),
                rss_bytes=int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
                pss_bytes=smaps.get('Pss', 0) * 1024,
                private_bytes=(smaps.get('Private_Clean', 0) + smaps.get('Private_Dirty', 0)) * 1024,
                swap_bytes=int(status.get('VmSwap', '0 kB').split()[0]) * 1024,
                threads=int(status['Threads'])))
        except (OSError, ValueError, KeyError):
            # A child can exit between cgroup enumeration and its smaps read.
            continue
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(utc=now(), unix=time.time(),
        cgroup_current_bytes=integer('memory.current'), cgroup_peak_bytes=integer('memory.peak'),
        cgroup_max_bytes=integer('memory.max'), cgroup_events=pairs(group / 'memory.events'),
        cgroup_memory_stat=pairs(group / 'memory.stat'),
        system_available_bytes=int(meminfo['MemAvailable'].split()[0]) * 1024,
        processes=processes,
        totals={key: sum(row[key] for row in processes)
                for key in ('rss_bytes', 'pss_bytes', 'private_bytes', 'swap_bytes', 'threads')})


def stop_owned(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def run(args):
    root = args.root.resolve()
    scripts = Path(__file__).resolve().parent
    if (root / 'PAUSE_JUDGE.json').exists():
        raise RuntimeError('Judge is paused by the user; finish the memory investigation before resuming')
    group = cgroup_path()
    bounds = {name: (group / name).read_text().strip()
              for name in ('memory.high', 'memory.max', 'memory.swap.max', 'pids.max')}
    if (bounds['memory.max'] == 'max' or int(bounds['memory.max']) > 64 * 1024**3
            or bounds['memory.high'] == 'max' or int(bounds['memory.high']) > 48 * 1024**3
            or bounds['memory.swap.max'] != '0'):
        raise RuntimeError(f'Refusing an unbounded judge session: {bounds}')
    _, manifest_hash = verify_manifest(root)
    for name in ('pool.lock', 'client.lock'):
        with (root / name).open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    run_id = str(time.time_ns())
    history = root / 'history' / ('resource_resume_' + run_id)
    history.mkdir(parents=True)
    for name in ('progress.json', 'campaign.json', 'servers.json', 'pool_launch.json',
                 'recovery_launch.json', 'recovery.json', 'resource_session.json'):
        if (root / name).exists():
            shutil.copy2(root / name, history / name)
    for name in ('STOP_SERVERS', 'campaign_failure.json', 'server_failure.json',
                 'servers_stopped.json', 'unresolved_judge_answers.json'):
        if (root / name).exists():
            (root / name).rename(history / name)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'deps') + os.pathsep + env.get('PYTHONPATH', '')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VLLM_PLUGINS='', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
        OPENCV_FFMPEG_THREADS='1', VLLM_MEDIA_LOADING_THREAD_COUNT='4')
    state = dict(utc=now(), run_id=run_id, pid=os.getpid(), cgroup=str(group),
        bounds=bounds, manifest_sha256=manifest_hash, supervisor_sha256=sha(Path(__file__)),
        video_workers_per_replica=args.video_workers, media_loading_threads_per_replica=4,
        ffmpeg_threads_per_decoder=1, protocol_unchanged=True,
        performance_eligible=False, history=str(history),
        samples=str(root / ('resource_usage_' + run_id + '.jsonl')),
        peak_sampled_pss_bytes=0, peak_sampled_rss_bytes=0, status='starting')
    write(root / 'resource_session.json', state)
    write(history / 'settings.json', state)
    pool = client = None
    try:
        pool_cmd = [sys.executable, str(scripts / 'qwen_judge_pool.py'),
            '--root', str(root), '--model', str(args.model.resolve()),
            '--gpus', *map(str, args.gpus)]
        with (root / 'pool.log').open('a') as log:
            pool = subprocess.Popen(pool_cmd, env=env, stdout=log, stderr=log, start_new_session=True)
        write(root / 'pool_launch.json', dict(pid=pool.pid, command=pool_cmd, utc=now(), run_id=run_id))
        for _ in range(100):
            if pool.poll() is not None:
                raise RuntimeError('Pool exited during startup; see pool.log')
            if read(root / 'servers.json')['pid'] == pool.pid:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('New pool did not write its identity')
        client_cmd = [sys.executable, str(scripts / 'qwen_task_resume.py'),
            '--root', str(root), '--video-workers', str(args.video_workers)]
        with (root / 'recovery.log').open('a') as log:
            client = subprocess.Popen(client_cmd, env=env, stdout=log, stderr=log, start_new_session=True)
        write(root / 'recovery_launch.json', dict(pid=client.pid, command=client_cmd, utc=now(),
            runner_sha256=sha(scripts / 'qwen_task_resume.py'), run_id=run_id))
        state.update(pool_pid=pool.pid, client_pid=client.pid, status='running')
        with Path(state['samples']).open('a') as samples:
            while True:
                measurement = resource_snapshot(group)
                samples.write(json.dumps(measurement) + '\n')
                samples.flush()
                state.update(updated_utc=now(), latest=measurement)
                for key in ('rss', 'pss'):
                    peak = 'peak_sampled_' + key + '_bytes'
                    state[peak] = max(state[peak], measurement['totals'][key + '_bytes'])
                write(root / 'resource_session.json', state)
                if client.poll() is not None:
                    break
                if pool.poll() is not None and not (root / 'STOP_SERVERS').exists():
                    raise RuntimeError('Judge pool exited before scoring finished')
                time.sleep(5)
        state['status'] = 'complete' if client.returncode == 0 else 'incomplete'
        state['client_returncode'] = client.returncode
        return client.returncode
    except BaseException as exc:
        state.update(status='error', error=repr(exc))
        raise
    finally:
        stop_owned(client)
        stop_owned(pool)
        state['ended_utc'] = now()
        write(root / 'resource_session.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 2])
    parser.add_argument('--video-workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.video_workers <= 12:
        parser.error('video workers per replica must be between 1 and 12')
    with (args.root / 'resource_session.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(args))

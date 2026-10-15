"""Bounded independent-process slots for the existing frozen judge queue.

Primary slots handle both models and every method. Secondary slots only handle
explicitly admitted Edge methods, by measurement or user request. All slots
keep the same manifests, seeds, numerical worker, task output paths and resume
validation. Device leases exclude unrelated calibration workers while child
processes in this pool share the inherited lease.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from baselines import sha, write
import judge_generation as frozen
import judge_concurrent_infer as runtime

HERE = Path(__file__).resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def active(pid, root):
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    names = {'judge_generation.py', 'judge_device_control.py', HERE.name, runtime.HERE.name}
    return str(root).encode() in args and any(Path(os.fsdecode(x)).name in names for x in args if x)


def policy(root):
    job = frozen.verify_job(root)
    value = json.loads((root/'parallel_policy.json').read_text())
    if value['job_sha256'] != sha(root/'manifest.json'):
        raise ValueError('Parallel policy refers to another frozen job')
    if value['source_sha256'] != sha(HERE) or value['runtime_sha256'] != sha(runtime.HERE):
        raise ValueError('Parallel resource controller changed after policy freeze')
    if not value['gpus'] or not set(value['gpus']) <= set(job['gpus']):
        raise ValueError('Parallel policy contains unqualified GPUs')
    if not 1 <= value['slots_per_gpu'] <= 4:
        raise ValueError('One to four bounded slots per GPU required')
    if not set(value['secondary_methods']) <= set(frozen.METHODS):
        raise ValueError('Unknown secondary method')
    if value.get('probe_report') and sha(value['probe_report']) != value['probe_report_sha256']:
        raise ValueError('Concurrency admission evidence changed')
    if value.get('probe_manifest') and sha(value['probe_manifest']) != value['probe_manifest_sha256']:
        raise ValueError('Concurrency probe cohort changed')
    return job, value


def allowed(task, slot, value):
    return slot == 0 or (task['model'] == 'edge' and task['method'] in value['secondary_methods'])


def claim(root, job, value, gpu, slot):
    def take(queue):
        for task in job['tasks']:
            row = queue['tasks'][task['id']]
            if row['status'] == 'pending' and allowed(task, slot, value):
                start = now()
                attempt = dict(gpu=gpu, slot=slot, pid=os.getpid(), started_utc=start,
                               parallel_policy_sha256=sha(root/'parallel_policy.json'))
                row.update(status='running', **attempt)
                row['attempts'].append(attempt)
                return task
    return frozen.queue_update(root, take)


def worker(a):
    job, value = policy(a.root)
    if a.gpu not in value['gpus'] or not 0 <= a.slot < value['slots_per_gpu']:
        raise ValueError('Unconfigured worker slot')
    runtime.verify_descriptors(a.gpu, a.global_fd, a.device_fd)
    folder = a.root/'parallel_workers'/f'gpu{a.gpu}.slot{a.slot}'
    folder.mkdir(parents=True, exist_ok=True)
    state = dict(complete=False, pid=os.getpid(), gpu=a.gpu, slot=a.slot,
        phase='qualification', qualified={}, completed_tasks=[], started_utc=now())
    current = None

    def event(kind, **extra):
        data = dict(utc=now(), event=kind, gpu=a.gpu, slot=a.slot, **extra)
        with (folder/'events.jsonl').open('a') as log:
            log.write(json.dumps(data)+'\n')
        print(json.dumps(data), flush=True)

    def generate(model, method, output, log_path, *, cases=None, qualification=False):
        command, env, fds = runtime.child_command(a.root, model, method, output,
            a.gpu, (a.global_fd, {a.gpu:a.device_fd}), cases=cases, qualification=qualification)
        env['TW_SHARED_CALIBRATION_GPUS'] = ','.join(map(str, value['gpus']))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event('command_start', command=command, log=str(log_path))
        with log_path.open('a') as log:
            log.write('\n'+json.dumps(dict(utc=now(), command=command))+'\n')
            log.flush()
            # Inherit this worker's process group so a pool stop also stops its
            # active CUDA child. No other slot shares this process group.
            subprocess.run(command, cwd=frozen.REPO, env=env, pass_fds=fds,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            combinations = [(model, method) for model in ('edge', 'nano') for method in frozen.METHODS
                            if a.slot == 0 or (model == 'edge' and method in value['secondary_methods'])]
            for model, method in combinations:
                # Primary slots reuse unchanged per-GPU qualification receipts;
                # additional slots generate fresh reference-checked outputs.
                dest = (a.root/f'gpu{a.gpu}' if a.slot == 0 else folder)/'qualification'/model/method
                generate(model, method, dest, dest/'generation.log', qualification=True)
                receipt = json.loads((dest/'progress.json').read_text())
                if not receipt['complete'] or not receipt.get('roundtrip_bitwise_equal'):
                    raise ValueError('Parallel slot qualification incomplete')
                state['qualified'][model+'/'+method] = sha(dest/'progress.json')
                write(folder/'status.json', state)
                event('qualified', model=model, method=method)
            state['phase'] = 'generation'
            write(folder/'status.json', state)
            while True:
                policy(a.root)
                current = claim(a.root, job, value, a.gpu, a.slot)
                if current is None:
                    break
                taskroot = a.root/'tasks'/current['id']
                event('task_start', task=current['id'], videos=current['videos'])
                generate(current['model'], current['method'], taskroot/'generation',
                         taskroot/'generation.log', cases=current['cases'])
                path = taskroot/'generation/progress.json'
                receipt = json.loads(path.read_text())
                if not receipt['complete'] or set(receipt['completed']) != set(current['cases']):
                    raise ValueError('Incomplete parallel task')
                def finish(queue):
                    row = queue['tasks'][current['id']]
                    if row['pid'] != os.getpid() or row['slot'] != a.slot:
                        raise ValueError('Task ownership changed')
                    row.update(status='complete', completed_utc=now(), progress_sha256=sha(path))
                frozen.queue_update(a.root, finish)
                state['completed_tasks'].append(current['id'])
                write(folder/'status.json', state)
                event('task_complete', task=current['id'])
                current = None
            state.update(complete=True, phase='complete', completed_utc=now())
        except BaseException as error:
            state.update(paused=True, error=repr(error), traceback=traceback.format_exc(), stopped_utc=now())
            if current:
                def fail(queue):
                    queue['tasks'][current['id']].update(status='failed', error=repr(error), stopped_utc=now())
                frozen.queue_update(a.root, fail)
            event('paused', error=repr(error))
            raise
        finally:
            write(folder/'status.json', state)


def snapshot(root, job, value):
    state = frozen.snapshot(root, dict(job, gpus=[]))
    workers = {}
    for gpu in value['gpus']:
        slots = {}
        for slot in range(value['slots_per_gpu']):
            path = root/'parallel_workers'/f'gpu{gpu}.slot{slot}'/'status.json'
            slots[str(slot)] = json.loads(path.read_text()) if path.exists() else dict(phase='starting')
        complete = all(s.get('complete', False) for s in slots.values())
        workers[str(gpu)] = dict(gpu=gpu, phase='complete' if complete else 'generation',
            complete=complete, slots=slots, pid=slots['0'].get('pid'))
        write(root/f'gpu{gpu}/status.json', workers[str(gpu)])
    state.update(workers=workers, active_gpus=value['gpus'], slots_per_gpu=value['slots_per_gpu'],
        secondary_methods=value['secondary_methods'], scheduler='independent_process_pool',
        parallel_policy_sha256=sha(root/'parallel_policy.json'))
    return state


def stop_pool_children(children):
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in children:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def supervise(a):
    job, value = policy(a.root)
    children = []
    state = {}

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    with (a.root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def recover(queue):
            for row in queue['tasks'].values():
                if row['status'] in ('running', 'failed'):
                    if active(row.get('pid', 0), a.root):
                        raise RuntimeError('Previous task owner is still active')
                    row.update(status='pending')
        frozen.queue_update(a.root, recover)
        try:
            with runtime.device_leases(value['gpus']) as (global_fd, devices):
                from thunderworld_fp4.cosmos.resources import free_memory
                available = free_memory()
                # Additional slots only admit Edge; one primary can load Nano.
                required = (28+12*(value['slots_per_gpu']-1)+12)*1024
                if any(available[g] < required for g in value['gpus']):
                    raise RuntimeError(f'PAUSED: pool needs {required} MiB free per GPU before loading: {available}')
                write(a.root/'parallel_memory_admission.json', dict(utc=now(), free_mib=available,
                    required_mib_per_gpu=required, primary_reservation_mib=28*1024,
                    secondary_edge_reservation_mib=12*1024, reserve_mib=12*1024,
                    performance_eligible=False))
                try:
                    for model in job['models']:
                        env = os.environ.copy()
                        env['CUDA_VISIBLE_DEVICES'] = ''
                        with (a.root/model/'exporter.log').open('ab') as log:
                            proc = subprocess.Popen([sys.executable, str(frozen.HERE/'judge_video_export.py'),
                                '--root', str(a.root/model), '--follow'], env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(('exporter_'+model, proc))
                    for gpu in value['gpus']:
                        for slot in range(value['slots_per_gpu']):
                            folder = a.root/'parallel_workers'/f'gpu{gpu}.slot{slot}'
                            folder.mkdir(parents=True, exist_ok=True)
                            command = [sys.executable, str(HERE), 'worker', '--root', str(a.root),
                                '--gpu', str(gpu), '--slot', str(slot), '--global-fd', str(global_fd),
                                '--device-fd', str(devices[gpu])]
                            env = os.environ.copy()
                            env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                                TW_SHARED_CALIBRATION_GPUS=','.join(map(str, devices)),
                                TW_SHARED_CALIBRATION_LOCK_SCOPE='device', PYTHONUNBUFFERED='1',
                                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
                            with (folder/'supervisor.log').open('ab') as log:
                                proc = subprocess.Popen(command, cwd=frozen.REPO, env=env,
                                    pass_fds=(global_fd, devices[gpu]), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
                            children.append((f'gpu{gpu}.slot{slot}', proc))
                    write(a.root/'children.json', dict(processes=[dict(name=n, pid=p.pid) for n,p in children]))
                    while True:
                        state = snapshot(a.root, job, value)
                        for model, info in state['models'].items():
                            write(a.root/model/'workflow.json', info)
                        write(a.root/'pool_status.json', state)
                        failed = [name for name,proc in children if proc.poll() is not None and proc.returncode != 0]
                        if failed:
                            raise RuntimeError('Parallel worker failed: '+str(failed))
                        if state['complete']:
                            from judge_video_export import jsonlines
                            rows, blind = [], []
                            for model in job['models']:
                                rows += [json.loads(line) for line in (a.root/model/'judge_index.jsonl').read_text().splitlines()]
                                blind += [json.loads(line) for line in (a.root/model/'judge_blind/inputs.jsonl').read_text().splitlines()]
                            jsonlines(a.root/'judge_index.jsonl', rows)
                            jsonlines(a.root/'judge_inputs.jsonl', sorted(blind, key=lambda x:x['sample_id']))
                            break
                        if all(proc.poll() is not None for _,proc in children):
                            raise RuntimeError('Pool exited with incomplete outputs')
                        time.sleep(5)
                finally:
                    stop_pool_children([proc for _,proc in children])
        except BaseException as error:
            state.update(complete=False, paused=True, error=repr(error), traceback=traceback.format_exc(), updated_utc=now())
            for model in job['models']:
                write(a.root/model/'workflow.json', dict(state.get('models', {}).get(model, {}),
                    paused=True, error=repr(error)))
            raise
        finally:
            write(a.root/'pool_status.json', state)


def validate_admission(job, gpus, slots, methods, report=None, admission='measured'):
    if not gpus or len(gpus) != len(set(gpus)) or not set(gpus) <= set(job['gpus']):
        raise ValueError('Select distinct qualified GPUs')
    if not 1 <= slots <= 4 or not set(methods) <= set(frozen.METHODS):
        raise ValueError('Unsupported slots or methods')
    if admission not in ('measured', 'requested'):
        raise ValueError('Unknown concurrency admission mode')
    if slots > 1:
        if not methods:
            raise ValueError('Secondary slots require an explicit method list')
        # Explicit user-selected concurrency may proceed before its throughput
        # comparison is complete. Every slot still passes numerical reference
        # qualification before generating any final outputs.
        if admission == 'requested':
            return
        if slots > 2:
            raise ValueError('Measured probe only establishes one versus two processes')
        if not methods or not report or not report.get('complete'):
            raise ValueError('Completed probe and nonempty method admission required')
        for name in methods:
            row = report['methods'].get(name, {})
            if not (row.get('complete') and row.get('all_source_hashes_match') and row.get('candidate_for_live_pilot')):
                raise ValueError('Method did not pass measured concurrency admission: '+name)


def launch(a):
    job = frozen.verify_job(a.root)
    with (a.root/'device_control.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_path = a.root/'launch.json'
        if launch_path.exists():
            previous = json.loads(launch_path.read_text())
            if active(previous['pid'], a.root):
                if a.gpus is None and a.slots_per_gpu is None and a.secondary_methods is None:
                    policy(a.root)
                    print(json.dumps(previous))
                    return
                raise RuntimeError('Stop the current pool before changing concurrency policy')
        with (a.root/'supervisor.lock').open('a+') as gate:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue = json.loads((a.root/'queue.json').read_text())
            if any(row['status']=='running' and active(row.get('pid',0), a.root) for row in queue['tasks'].values()):
                raise RuntimeError('Previous task worker still active')
            path = a.root/'parallel_policy.json'
            existing = json.loads(path.read_text()) if path.exists() else {}
            gpus = a.gpus if a.gpus is not None else existing.get('gpus')
            slots = a.slots_per_gpu if a.slots_per_gpu is not None else existing.get('slots_per_gpu', 1)
            methods = a.secondary_methods if a.secondary_methods is not None else existing.get('secondary_methods', [])
            report_path = a.probe_report or (Path(existing['probe_report']) if existing.get('probe_report') else None)
            report = json.loads(report_path.read_text()) if report_path else None
            admission = a.admission or existing.get('admission', 'measured')
            validate_admission(job, gpus, slots, methods, report, admission)
            if report and report.get('source_sha256') != sha(runtime.HERE):
                raise ValueError('Probe used a different concurrency resource adapter')
            probe_manifest = report_path.parent/'manifest.json' if report_path else None
            if probe_manifest:
                evidence = json.loads(probe_manifest.read_text())
                if (evidence['job_sha256'] != sha(a.root/'manifest.json') or
                    evidence['source_sha256'] != sha(runtime.HERE)):
                    raise ValueError('Probe used a different frozen job or resource adapter')
            value = dict(gpus=gpus, slots_per_gpu=slots, secondary_methods=methods,
                admission=admission, numerical_qualification_required=True,
                secondary_models=['edge'], job_sha256=sha(a.root/'manifest.json'),
                source_sha256=sha(HERE), runtime_sha256=sha(runtime.HERE),
                probe_report=str(report_path.resolve()) if report_path else None,
                probe_report_sha256=sha(report_path) if report_path else None,
                probe_manifest=str(probe_manifest.resolve()) if probe_manifest else None,
                probe_manifest_sha256=sha(probe_manifest) if probe_manifest else None,
                requested_utc=now(), performance_eligible=False)
            if existing:
                with (a.root/'parallel_policy_history.jsonl').open('a') as log:
                    log.write(json.dumps(existing)+'\n')
            write(path, value)
            device_path = a.root/'device_policy.json'
            if device_path.exists():
                with (a.root/'device_policy_history.jsonl').open('a') as log:
                    log.write(json.dumps(json.loads(device_path.read_text()))+'\n')
            write(device_path, dict(active_gpus=gpus, qualified_gpus=job['gpus'],
                job_sha256=sha(a.root/'manifest.json'), controller_sha256=sha(HERE),
                parallel_policy_sha256=sha(path), requested_utc=now(),
                reason='User-authorized bounded independent-process concurrency'))
        command = [sys.executable, str(HERE), 'supervise', '--root', str(a.root)]
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        with (a.root/'supervisor.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=frozen.REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=process.pid, command=command, started_utc=now(), active_gpus=gpus,
            slots_per_gpu=slots, parallel_policy_sha256=sha(path), log=str(a.root/'supervisor.log'))
        write(launch_path, value)
        print(json.dumps(value), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('launch', 'supervise', 'worker', 'status'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int)
    p.add_argument('--slots-per-gpu', type=int)
    p.add_argument('--secondary-methods', nargs='*')
    p.add_argument('--probe-report', type=Path)
    p.add_argument('--admission', choices=('measured', 'requested'))
    p.add_argument('--gpu', type=int)
    p.add_argument('--slot', type=int)
    p.add_argument('--global-fd', type=int)
    p.add_argument('--device-fd', type=int)
    a = p.parse_args()
    a.root = a.root.resolve()
    if a.operation == 'status':
        job, value = policy(a.root)
        print(json.dumps(snapshot(a.root, job, value), indent=2))
    elif a.operation == 'worker':
        if None in (a.gpu, a.slot, a.global_fd, a.device_fd):
            p.error('worker requires slot and inherited lease descriptors')
        worker(a)
    elif a.operation == 'launch':
        launch(a)
    else:
        supervise(a)


if __name__ == '__main__':
    main()

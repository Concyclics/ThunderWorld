"""Resume the frozen baseline queue with two Nano processes per selected GPU.

Only resource scheduling changes. The original model manifests, numerical
worker, output identities, generation logs and video exporter are retained.
New device/slot pairs reproduce historical outputs before taking final work.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from baselines import sha, write
import judge_baseline_extension as extension
import judge_generation as frozen
import judge_parallel_queue as pool
import judge_concurrent_infer as runtime

HERE = Path(__file__).resolve()
METHODS = extension.METHODS
RESERVATION_MIB = 36*1024
HEADROOM_MIB = 12*1024


def now():
    return datetime.now(timezone.utc).isoformat()


def active(pid, root):
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return False
    return str(root).encode() in args and any(
        Path(os.fsdecode(a)).name.startswith('judge_') and a.endswith(b'.py') for a in args if a)


def hardware():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    return {int(i): dict(uuid=u.strip(), name=n.strip(), total_mib=int(t), free_mib=int(f))
        for i, u, n, t, f in csv.reader(io.StringIO(output), skipinitialspace=True)}


def validate_policy(root, job, value):
    devices = value['gpus']
    if (len(devices) != 2 or len(set(devices)) != 2 or
            any(type(g) is not int or g < 0 for g in devices) or value['slots_per_gpu'] != 2):
        raise ValueError('Two distinct GPUs, with exactly two slots each, required')
    if (value['models'] != ['nano'] or value['secondary_models'] != ['nano'] or
            value['secondary_methods'] != list(METHODS) or value['numerical_qualification_required'] is not True):
        raise ValueError('Every slot must qualify and admit the four Nano methods')
    if (value['job_sha256'] != sha(root/'manifest.json') or
            value['source_sha256'] != sha(HERE) or value['runtime_sha256'] != sha(runtime.__file__)):
        raise ValueError('Resource policy or frozen controller changed')
    if not set(value['models']) <= set(job['models']):
        raise ValueError('Unrecognized model')
    return value


def policy(root):
    job = extension.verify_job(root)
    value = validate_policy(root, job, json.loads((root/'parallel_policy.json').read_text()))
    # This is an execution view. Do not rewrite the original job's GPU list or
    # any model manifest: their hashes form the existing NPZ resume identities.
    return dict(job, gpus=value['gpus']), value


def view_job(root):
    return policy(root)[0]


def configure_runtime():
    frozen.METHODS = METHODS
    frozen.verify_job = view_job
    runtime.HERE = HERE


def claim(root, job, value, gpu, slot):
    if gpu not in value['gpus'] or slot not in range(value['slots_per_gpu']):
        raise ValueError('Unconfigured GPU/slot')
    def take(queue):
        for task in job['tasks']:
            row = queue['tasks'][task['id']]
            if (row['status'] == 'pending' and task['model'] in value['models'] and
                    task['method'] in value['secondary_methods']):
                attempt = dict(gpu=gpu, slot=slot, pid=os.getpid(), started_utc=now(),
                    parallel_policy_sha256=sha(root/'parallel_policy.json'))
                row.update(status='running', **attempt)
                row['attempts'].append(attempt)
                return task
    return frozen.queue_update(root, take)


def worker(a):
    job, value = policy(a.root)
    if a.gpu not in value['gpus'] or a.slot not in range(value['slots_per_gpu']):
        raise ValueError('Unconfigured worker')
    runtime.verify_descriptors(a.gpu, a.global_fd, a.device_fd)
    folder = a.root/'parallel_workers'/f'gpu{a.gpu}.slot{a.slot}'
    folder.mkdir(parents=True, exist_ok=True)
    state = dict(complete=False, pid=os.getpid(), gpu=a.gpu, slot=a.slot,
        phase='qualification', qualified={}, completed_tasks=[], started_utc=now(),
        models=value['models'], parallel_policy_sha256=sha(a.root/'parallel_policy.json'))
    current = None

    def event(kind, **extra):
        line = json.dumps(dict(utc=now(), event=kind, gpu=a.gpu, slot=a.slot, **extra))
        with (folder/'events.jsonl').open('a') as log:
            log.write(line+'\n')
        print(line, flush=True)

    def generate(model, method, output, log_path, *, cases=None, qualification=False):
        command, env, fds = runtime.child_command(a.root, model, method, output, a.gpu,
            (a.global_fd, {a.gpu: a.device_fd}), cases=cases, qualification=qualification)
        env['TW_SHARED_CALIBRATION_GPUS'] = ','.join(map(str, value['gpus']))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event('command_start', command=command, log=str(log_path))
        with log_path.open('a') as log:
            log.write('\n'+json.dumps(dict(utc=now(), command=command))+'\n')
            log.flush()
            subprocess.run(command, cwd=frozen.REPO, env=env, pass_fds=fds,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(folder/'status.json', state)
        try:
            for model in value['models']:
                for method in METHODS:
                    dest = folder/'device_runs'/value['run_id']/'qualification'/model/method
                    generate(model, method, dest, dest/'generation.log', qualification=True)
                    result = json.loads((dest/'progress.json').read_text())
                    if not result['complete'] or not result.get('roundtrip_bitwise_equal'):
                        raise ValueError('New device/slot qualification incomplete')
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
                dest = a.root/'tasks'/current['id']
                event('task_start', task=current['id'], videos=current['videos'])
                generate(current['model'], current['method'], dest/'generation',
                    dest/'generation.log', cases=current['cases'])
                receipt = dest/'generation/progress.json'
                result = json.loads(receipt.read_text())
                if not result['complete'] or set(result['completed']) != set(current['cases']):
                    raise ValueError('Incomplete resumed task')
                def finish(queue):
                    row = queue['tasks'][current['id']]
                    if row['pid'] != os.getpid() or row['gpu'] != a.gpu or row['slot'] != a.slot:
                        raise ValueError('Task ownership changed')
                    row.update(status='complete', completed_utc=now(), progress_sha256=sha(receipt))
                frozen.queue_update(a.root, finish)
                state['completed_tasks'].append(current['id'])
                write(folder/'status.json', state)
                event('task_complete', task=current['id'])
                current = None
            state.update(complete=True, phase='complete', completed_utc=now())
        except BaseException as error:
            state.update(paused=True, error=repr(error), traceback=traceback.format_exc(), stopped_utc=now())
            if current:
                frozen.queue_update(a.root, lambda queue: queue['tasks'][current['id']].update(
                    status='failed', error=repr(error), stopped_utc=now()))
            event('paused', error=repr(error))
            raise
        finally:
            write(folder/'status.json', state)


def snapshot(root, job, value):
    state = pool.snapshot(root, job, value)
    state.update(active_models=value['models'], resource_controller=str(HERE),
        scheduler='two_nano_processes_per_gpu', device_run_id=value['run_id'])
    return state


def supervise(a):
    job, value = policy(a.root)
    children, state = [], {}
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
                cards = hardware()
                required = RESERVATION_MIB*value['slots_per_gpu'] + HEADROOM_MIB
                for gpu in value['gpus']:
                    if (cards[gpu]['uuid'] != value['gpu_hardware'][str(gpu)]['uuid'] or
                            cards[gpu]['free_mib'] < required):
                        raise RuntimeError(f'PAUSED: GPU{gpu} requires {required} MiB free: {cards[gpu]}')
                write(a.root/'parallel_memory_admission.json', dict(utc=now(),
                    required_mib_per_gpu=required, nano_reservation_mib_per_slot=RESERVATION_MIB,
                    slots_per_gpu=value['slots_per_gpu'], reserve_mib=HEADROOM_MIB,
                    gpu_hardware=cards, performance_eligible=False))
                try:
                    initial = frozen.snapshot(a.root, dict(job, gpus=[]))
                    for model, info in initial['models'].items():
                        write(a.root/model/'workflow.json', info)
                        if info['complete']:
                            continue
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
                        with (a.root/model/'exporter.log').open('ab') as log:
                            process = subprocess.Popen([sys.executable,
                                str(HERE.with_name('judge_video_export.py')), '--root', str(a.root/model), '--follow'],
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(('exporter_'+model, process))
                    for gpu in value['gpus']:
                        for slot in range(value['slots_per_gpu']):
                            folder = a.root/'parallel_workers'/f'gpu{gpu}.slot{slot}'
                            folder.mkdir(parents=True, exist_ok=True)
                            command = [sys.executable, str(HERE), 'worker', '--root', str(a.root),
                                '--gpu', str(gpu), '--slot', str(slot), '--global-fd', str(global_fd),
                                '--device-fd', str(devices[gpu])]
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                                TW_SHARED_CALIBRATION_GPUS=','.join(map(str, devices)),
                                TW_SHARED_CALIBRATION_LOCK_SCOPE='device', PYTHONUNBUFFERED='1',
                                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
                            with (folder/'supervisor.log').open('ab') as log:
                                process = subprocess.Popen(command, cwd=frozen.REPO, env=env,
                                    pass_fds=(global_fd, devices[gpu]), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
                            children.append((f'gpu{gpu}.slot{slot}', process))
                    write(a.root/'children.json', dict(processes=[dict(name=n, pid=p.pid) for n, p in children]))
                    while True:
                        state = snapshot(a.root, job, value)
                        for model, info in state['models'].items():
                            write(a.root/model/'workflow.json', info)
                        write(a.root/'pool_status.json', state)
                        failed = [name for name, proc in children if proc.poll() is not None and proc.returncode != 0]
                        if failed:
                            raise RuntimeError('Relocated worker failed: '+str(failed))
                        if state['complete']:
                            from judge_video_export import jsonlines
                            rows, blind = [], []
                            for model in job['models']:
                                rows += [json.loads(s) for s in (a.root/model/'judge_index.jsonl').read_text().splitlines()]
                                blind += [json.loads(s) for s in (a.root/model/'judge_blind/inputs.jsonl').read_text().splitlines()]
                            jsonlines(a.root/'judge_index.jsonl', rows)
                            jsonlines(a.root/'judge_inputs.jsonl', sorted(blind, key=lambda r: r['sample_id']))
                            break
                        if all(proc.poll() is not None for _, proc in children):
                            raise RuntimeError('Workers exited with incomplete outputs')
                        time.sleep(5)
                finally:
                    pool.stop_pool_children([proc for _, proc in children])
        except BaseException as error:
            state.update(complete=False, paused=True, error=repr(error), traceback=traceback.format_exc(), updated_utc=now())
            for model in job['models']:
                write(a.root/model/'workflow.json', dict(state.get('models', {}).get(model, {}),
                    paused=True, error=repr(error)))
            raise
        finally:
            write(a.root/'pool_status.json', state)


def launch(a):
    job = extension.verify_job(a.root)
    with (a.root/'device_control.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = a.root/'launch.json'
        if path.exists() and active(json.loads(path.read_text())['pid'], a.root):
            current = json.loads((a.root/'parallel_policy.json').read_text())
            if current.get('source_sha256') == sha(HERE) and current['gpus'] == a.gpus and current['slots_per_gpu'] == a.slots_per_gpu:
                policy(a.root)
                print(path.read_text())
                return
            raise RuntimeError('Previous pool still active; drain its own processes first')
        with (a.root/'supervisor.lock').open('a+') as gate:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue = json.loads((a.root/'queue.json').read_text())
            if any(active(row.get('pid', 0), a.root) for row in queue['tasks'].values() if row['status'] in ('running', 'failed')):
                raise RuntimeError('Previous task process is still active')
            if any(t['model'] != 'nano' and queue['tasks'][t['id']]['status'] != 'complete' for t in job['tasks']):
                raise ValueError('This resource transition requires Edge generation to be complete')
            cards = hardware()
            required = RESERVATION_MIB*a.slots_per_gpu + HEADROOM_MIB
            if any(g not in cards or cards[g]['free_mib'] < required for g in a.gpus):
                raise RuntimeError(f'PAUSED: each selected GPU requires {required} MiB free')
            value = dict(gpus=a.gpus, slots_per_gpu=a.slots_per_gpu, models=['nano'], secondary_models=['nano'],
                secondary_methods=list(METHODS), admission='requested', numerical_qualification_required=True,
                job_sha256=sha(a.root/'manifest.json'), source_sha256=sha(HERE), runtime_sha256=sha(runtime.__file__),
                gpu_hardware={str(g): cards[g] for g in a.gpus}, requested_utc=now(),
                run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'), performance_eligible=False)
            validate_policy(a.root, job, value)
            old_path = a.root/'parallel_policy.json'
            old = json.loads(old_path.read_text()) if old_path.exists() else None
            if old and all(old.get(k) == value[k] for k in ('gpus', 'slots_per_gpu', 'models', 'source_sha256', 'job_sha256')):
                value = validate_policy(a.root, job, old)
            else:
                archive = a.root/'resource_transitions'/value['run_id']
                archive.mkdir(parents=True, exist_ok=True)
                for name in ('launch.json', 'children.json', 'pool_status.json', 'parallel_policy.json',
                             'parallel_memory_admission.json', 'device_policy.json', 'queue.json'):
                    file = a.root/name
                    if file.exists():
                        (archive/name).write_bytes(file.read_bytes())
                if old:
                    with (a.root/'parallel_policy_history.jsonl').open('a') as log:
                        log.write(json.dumps(old)+'\n')
                write(old_path, value)
                write(a.root/'device_policy.json', dict(active_gpus=a.gpus, slots_per_gpu=a.slots_per_gpu,
                    models=['nano'], controller=str(HERE), controller_sha256=sha(HERE),
                    job_sha256=sha(a.root/'manifest.json'), parallel_policy_sha256=sha(old_path),
                    qualification_required=True, reason='User requested GPU0/GPU2 with two processes per GPU'))
        command = [sys.executable, str(HERE), 'supervise', '--root', str(a.root)]
        with (a.root/'supervisor.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=frozen.REPO, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(pid=process.pid, command=command, started_utc=now(), active_gpus=a.gpus,
            slots_per_gpu=a.slots_per_gpu, models=['nano'], log=str(a.root/'supervisor.log'),
            parallel_policy_sha256=sha(a.root/'parallel_policy.json'))
        write(path, result)
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('launch', 'supervise', 'worker', 'infer', 'status'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', nargs=2, type=int, default=[0, 2])
    p.add_argument('--slots-per-gpu', type=int, choices=(2,), default=2)
    p.add_argument('--gpu', type=int)
    p.add_argument('--slot', type=int)
    p.add_argument('--global-fd', type=int)
    p.add_argument('--device-fd', type=int)
    p.add_argument('--coordinator-pid', type=int)
    p.add_argument('--model', choices=('nano',))
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--output', type=Path)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--qualification', action='store_true')
    a = p.parse_args()
    a.root = a.root.resolve()
    configure_runtime()
    if a.operation == 'launch':
        launch(a)
    elif a.operation == 'supervise':
        supervise(a)
    elif a.operation == 'worker':
        worker(a)
    elif a.operation == 'infer':
        runtime.infer(a)
    else:
        job, value = policy(a.root)
        print(json.dumps(snapshot(a.root, job, value), indent=2))


if __name__ == '__main__':
    main()

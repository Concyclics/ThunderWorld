"""Resume a frozen judge queue on an explicit subset of its qualified GPUs.

The frozen inference, exporter, manifests, and output identities stay intact.
Only the supervisor's device list and status reporting are adapted here.
Drain existing workers before using launch with a different device set.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import judge_generation as frozen
from baselines import sha, write

HERE = Path(__file__).resolve()
VERIFY = frozen.verify_job
SNAPSHOT = frozen.snapshot


def now(): return datetime.now(timezone.utc).isoformat()


def active(pid, root):
    path = Path(f'/proc/{pid}/cmdline')
    try: args = path.read_bytes().split(b'\0')
    except FileNotFoundError: return False
    return str(root).encode() in args and any(
        Path(os.fsdecode(a)).name in ('judge_generation.py', 'judge_device_control.py')
        for a in args if a)


def validate_policy(root, job, policy):
    selected = policy['active_gpus']
    if (not selected or any(type(g) is not int for g in selected) or
        len(set(selected)) != len(selected) or not set(selected) <= set(job['gpus'])):
        raise ValueError('Device policy must select nonempty, distinct qualified GPUs')
    if policy['job_sha256'] != sha(root/'manifest.json'):
        raise ValueError('Device policy refers to another frozen job')
    if policy['controller_sha256'] != sha(HERE):
        raise ValueError('Device controller changed after policy was saved')
    return policy


def selected_job(root):
    job = VERIFY(root)
    policy = validate_policy(root, job, json.loads((root/'device_policy.json').read_text()))
    return dict(job, gpus=list(policy['active_gpus']))


def runtime_snapshot(root, job):
    state = SNAPSHOT(root, job)
    state.update(active_gpus=list(job['gpus']), device_policy=str(root/'device_policy.json'),
        device_policy_sha256=sha(root/'device_policy.json'))
    return state


def launch(a):
    job = VERIFY(a.root)
    with (a.root/'device_control.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_file = a.root/'launch.json'
        if launch_file.exists():
            previous = json.loads(launch_file.read_text())
            if active(previous['pid'], a.root):
                policy_file = a.root/'device_policy.json'
                if policy_file.exists():
                    current = validate_policy(a.root, job, json.loads(policy_file.read_text()))
                    if a.gpus is None or current['active_gpus'] == a.gpus:
                        print(json.dumps(previous)); return
                raise RuntimeError('Drain the current supervisor/workers before changing devices')
        # Check the actual supervisor lock too, rather than relying on a PID file.
        with (a.root/'supervisor.lock').open('a+') as gate:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for gpu in job['gpus']:
                path = a.root/f'gpu{gpu}/status.json'
                if path.exists() and active(json.loads(path.read_text()).get('pid', 0), a.root):
                    raise RuntimeError('Previous GPU worker is still active')
            path = a.root/'device_policy.json'
            existing = json.loads(path.read_text()) if path.exists() else None
            selected = a.gpus if a.gpus is not None else (existing['active_gpus'] if existing else None)
            if selected is None: raise ValueError('--gpus required for first controlled launch')
            policy = dict(active_gpus=selected, qualified_gpus=job['gpus'],
                job_sha256=sha(a.root/'manifest.json'), controller_sha256=sha(HERE),
                requested_utc=now(), reason='User-selected device subset; inference configuration unchanged')
            validate_policy(a.root, job, policy)
            if existing:
                with (a.root/'device_policy_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(existing)+'\n')
            write(path, policy)
        command = [sys.executable, str(HERE), 'supervise', '--root', str(a.root)]
        env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
        with (a.root/'supervisor.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=frozen.REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(pid=process.pid, started_utc=now(), command=command,
            active_gpus=selected, log=str(a.root/'supervisor.log'),
            device_policy_sha256=sha(path))
        write(launch_file, result); print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('launch', 'supervise', 'status'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int)
    a = p.parse_args(); a.root = a.root.resolve()
    if a.operation == 'launch': return launch(a)
    if a.operation == 'status':
        print(json.dumps(runtime_snapshot(a.root, selected_job(a.root)), indent=2)); return
    frozen.verify_job = selected_job
    frozen.snapshot = runtime_snapshot
    frozen.active = active
    frozen.supervise(a)


if __name__ == '__main__': main()

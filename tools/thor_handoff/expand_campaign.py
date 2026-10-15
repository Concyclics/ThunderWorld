"""Serial, resumable expanded-I2V experiments. Never overlaps GPU workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl
from expand_worker import write,idle_guard


JOBS = [
    ('edge','bf16','original'),
    ('edge','bf16','trim'),
    ('edge','b1.5_protected_n','original'),
    ('edge','b1.5_protected_n','trim'),
    ('edge','b1.5_protected_n','ar4_embed2'),
    ('edge','b1.5_protected_n','transformer2'),
    ('edge','b1.5_protected_n','full2'),
    ('edge','bf16','vae1'),
    ('edge','bf16','vae2'),
    ('edge','b1.5_protected_n','full1'),
    ('edge','b2_joint','original'),
    ('edge','b2_joint','full2'),
    ('nano','bf16','original'),
    ('nano','bf16','trim'),
    ('nano','b1.5_protected_n','original'),
    ('nano','b1.5_protected_n','trim'),
    ('nano','b1.5_protected_n','ar4_embed2'),
    ('nano','b1.5_protected_n','full2'),
    ('nano','b2_joint','original'),
    ('nano','b2_joint','full2'),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--jobs',type=Path)
    p.add_argument('--require-idle',action='store_true')
    p.add_argument('--wait-idle',action='store_true',help='Wait for 30 seconds without other compute before each worker')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.wait_idle:a.require_idle=True
    jobs=json.loads(a.jobs.read_text()) if a.jobs else JOBS
    status=dict(started=time.time(),jobs=jobs,completed=[],pid=os.getpid(),performance_eligible=False)
    for model,method,profile in jobs:
        name='-'.join((model,method,profile));dst=a.root/'runs'/name
        result=dst/'result.json'
        if result.exists():
            old=json.loads(result.read_text())
            if old.get('complete'):status['completed'].append(name);continue
            raise RuntimeError('Incomplete run retained; inspect and use a new campaign path: '+str(dst))
        status.update(current=name,updated=time.time());write(a.root/'campaign.json',status)
        if a.wait_idle:
            quiet_since=None
            while True:
                try:
                    idle_guard()
                    if quiet_since is None:quiet_since=time.monotonic()
                    if time.monotonic()-quiet_since>=30:break
                    status['waiting']='confirming_idle'
                except RuntimeError as exc:
                    if 'PAUSED: other GPU compute' not in str(exc):raise
                    quiet_since=None;status['waiting']=str(exc)
                status['updated']=time.time();write(a.root/'campaign.json',status)
                time.sleep(15)
            status.pop('waiting',None)
            status['updated']=time.time();write(a.root/'campaign.json',status)
        cmd=[sys.executable,str(Path(__file__).with_name('expand_worker.py')),
            '--variant',str(a.variant),'--manifest',str(a.root/'manifests'/(model+'.json')),
            '--method',method,'--profile',profile,'--output',str(dst),'--asset-dir',str(a.root/'assets'/model),
            '--benchmarks','rbench','paibench','--diagnose']
        if a.require_idle:cmd+=['--require-idle']
        with (a.root/(name+'.log')).open('w') as f: rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
        if rc:
            status.update(failed=name,returncode=rc,updated=time.time());write(a.root/'campaign.json',status)
            raise SystemExit(rc)
        status['completed'].append(name)
    status.update(complete=True,finished=time.time());write(a.root/'campaign.json',status)


if __name__=='__main__':main()

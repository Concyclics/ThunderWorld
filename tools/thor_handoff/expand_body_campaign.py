"""Run matched body-only comparisons serially after the current full campaign."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from expand_worker import write,idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--after',type=Path)
    p.add_argument('--require-idle',action='store_true')
    p.add_argument('--jobs',type=Path)
    a=p.parse_args()
    lock=(a.root/'body-campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(complete=False,completed=[],performance_eligible=False)
    while a.after:
        prior=json.loads(a.after.read_text())
        if prior.get('complete'):break
        if prior.get('failed'):raise RuntimeError('Preceding campaign failed: '+str(prior['failed']))
        status.update(waiting_for=str(a.after),updated=time.time());write(a.root/'body-campaign.json',status)
        time.sleep(10)
    status.pop('waiting_for',None)
    jobs=json.loads(a.jobs.read_text()) if a.jobs else [(m,method,profile) for m in ('edge','nano') for method,profile in
        [('bf16','trim'),('fp8','trim'),('fp4_base','trim'),('b1.5_protected_n','trim'),('b2_joint','trim'),
         ('b1.5_protected_n','transformer2'),('b2_joint','transformer2')]]
    status['jobs']=jobs
    for model,method,profile in jobs:
        name='-'.join((model,method,profile));dest=a.root/'body-runs'/name
        if (dest/'result.json').exists():
            old=json.loads((dest/'result.json').read_text())
            if old.get('complete'):status['completed'].append(name);continue
            raise RuntimeError('Inspect incomplete body result before retry: '+str(dest))
        status.update(current=name,updated=time.time());write(a.root/'body-campaign.json',status)
        if a.require_idle:
            while True:
                try:idle_guard();break
                except RuntimeError as exc:
                    if 'PAUSED: other GPU compute' not in str(exc):raise
                    status.update(waiting=str(exc),updated=time.time());write(a.root/'body-campaign.json',status)
                    time.sleep(15)
            status.pop('waiting',None)
        command=[sys.executable,str(Path(__file__).with_name('expand_body.py')),
            '--variant',str(a.variant),'--manifest',str(a.root/'manifests'/(model+'.json')),
            '--method',method,'--profile',profile,'--output',str(dest),'--asset-dir',str(a.root/'assets'/model)]
        if a.require_idle:command.append('--require-idle')
        with (a.root/('body-'+name+'.log')).open('w') as f:rc=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
        if rc:
            status.update(failed=name,returncode=rc,updated=time.time());write(a.root/'body-campaign.json',status)
            raise SystemExit(rc)
        status['completed'].append(name)
    status.update(complete=True,finished=time.time());write(a.root/'body-campaign.json',status)


if __name__=='__main__':main()

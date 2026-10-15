"""Copy remote benchmark receipts locally; never starts another GPU job."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='chenhan@thor')
    p.add_argument('--remote',default='/home/chenhan/thunderworld/comparisons/edge_20260922')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--follow',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=25']
    previous = None
    while True:
        result = subprocess.run([*ssh,a.host,'cat '+shlex.quote(a.remote+'/status.json')],
            capture_output=True,text=True,timeout=60)
        if result.returncode:
            print(json.dumps(dict(sync_error=result.stderr[-600:],time=time.time())),flush=True)
            if not a.follow:raise RuntimeError('Cannot read remote status')
            time.sleep(60)
            continue
        status = json.loads(result.stdout)
        complete = status.get('complete',False)
        # Only small top-level receipts during measurement. Copy all raw rows
        # and logs once the GPU campaign ends, to avoid concurrent bulk I/O.
        filters = [] if complete or status.get('error') else [
            '--include=/*.json','--include=/*.md','--include=/*.log','--exclude=*']
        subprocess.run(['rsync','-a','-e',shlex.join(ssh),*filters,
            a.host+':'+a.remote+'/',str(a.output)+'/'],check=True,timeout=1200)
        current = (status.get('finished_runs'),status.get('method'),status.get('round'),status.get('error'),complete)
        if current != previous:
            print(json.dumps(dict(time=time.time(),**status)),flush=True)
            previous = current
        if complete or status.get('error') or not a.follow:
            return
        # Host-side receipt polling only; no CUDA events/hooks/profilers.
        for _ in range(3):time.sleep(60)


if __name__=='__main__':main()

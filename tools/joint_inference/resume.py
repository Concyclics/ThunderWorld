"""Resume an interrupted formal matrix, preserving every previous attempt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from thunderworld_fp4.cosmos.cli import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('matrix',type=Path);a=p.parse_args()
    data=json.loads(a.matrix.read_text())
    if data['complete']:raise ValueError('Matrix already complete')
    if data['suite']!='performance':raise ValueError('Only formal performance matrices can be resumed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=data['device']:raise ValueError('Resume on the same physical GPU')
    idle_guard()
    pending=[j for j in data['jobs'] if j.get('exit_code')!=0]
    if not pending:raise ValueError('No pending jobs')
    restart=min(j['round'] for j in pending)
    def save():a.matrix.write_text(json.dumps(data,indent=2)+'\n')
    # Repeat the entire interrupted round in its original randomized order.
    # This avoids keeping a half-round measured before a long external pause.
    for job in data['jobs']:
        if job['round']<restart:continue
        previous={k:v for k,v in job.items() if k!='attempts'}
        attempts=job.setdefault('attempts',[]);attempts.append(previous)
        destination=Path(str(previous['output'])+f'_retry{len(attempts)}')
        command=list(previous['command']);command[command.index('--output')+1]=str(destination)
        job.update(command=command,output=str(destination));job.pop('exit_code',None)
    save()
    for job in data['jobs']:
        if job['round']<restart:continue
        idle_guard();print('Resuming',job['output'],flush=True)
        start=time.perf_counter()
        with open(job['output']+'.log','w') as log:
            job['exit_code']=subprocess.run(job['command'],cwd=Path(__file__).resolve().parents[2],
                stdout=log,stderr=subprocess.STDOUT).returncode
        job['process_seconds']=time.perf_counter()-start;save()
        if job['exit_code']:raise RuntimeError('Resumed job failed: '+job['output'])
    data['complete']=True;save()


if __name__=='__main__':main()

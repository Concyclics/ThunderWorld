"""Reproducible serialized validation -> freeze -> final quality workflow.

Offline assets and development-selected jobs must already exist. Failures stop the
workflow; --resume never lowers precision, changes inputs, or selects on final data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','case','cohort','jobs','output','lpips-cache'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--metric-workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    here=Path(__file__).resolve().parent
    identity={key:hashlib.sha256(getattr(a,key).read_bytes()).hexdigest() for key in ('case','cohort','jobs')}
    identity['model']=str(a.model.resolve())
    protocol=a.output/'workflow.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=identity:raise ValueError('Resume workflow inputs changed')
    else:protocol.write_text(json.dumps(identity,indent=2)+'\n')
    def run(script,args,*,cpu=False):
        env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([sys.executable,str(here/script),*map(str,args)],env=env,check=True)
    def complete(path):
        if not path.exists():return False
        if not a.resume:raise FileExistsError(path)
        return bool(json.loads(path.read_text()).get('complete'))
    def infer(role,jobs):
        folder=a.output/role
        # The runner validates the frozen config/assets and resume provenance even
        # when every request is already complete.
        run('run_shared.py',['--model',a.model,'--case',a.case,'--cohort',a.cohort,
            '--jobs',jobs,'--role',role,'--output',folder,*(['--resume'] if a.resume and folder.exists() else [])])
        quality=folder/'quality.json'
        ready=complete(quality)
        if ready and json.loads(quality.read_text())['matrix_sha256']!=hashlib.sha256((folder/'matrix.json').read_bytes()).hexdigest():
            raise ValueError('Saved quality belongs to a different matrix')
        if not ready:
            if quality.exists():
                # Preserve interrupted metric evidence. GPU outputs are untouched.
                attempt=1
                while quality.with_name(f'quality.partial.{attempt}.json').exists():attempt+=1
                quality.rename(quality.with_name(f'quality.partial.{attempt}.json'))
            run('evaluate.py',[folder/'matrix.json','--cohort',a.cohort,'--workers',a.metric_workers,
                '--lpips-cache',a.lpips_cache,'--output',quality],cpu=True)
        return folder,quality
    validation,quality=infer('validation',a.jobs)
    selection=a.output/'frozen_final_jobs.json'
    frozen=complete(selection)
    if frozen and json.loads(selection.read_text())['validation_sha256']!=hashlib.sha256(quality.read_bytes()).hexdigest():
        raise ValueError('Frozen selection belongs to different validation results')
    if not frozen:
        run('select_budget.py',['--quality',quality,'--jobs',a.jobs,'--cohort',a.cohort,'--output',selection],cpu=True)
    final,quality=infer('final',selection)
    report=a.output/'report'
    if not complete(report/'report.json'):
        run('final_compute_report.py',['--quality',quality,'--matrix',final/'matrix.json',
            '--selection',selection,'--cohort',a.cohort,'--output',report],cpu=True)
    print(report/'report.json',flush=True)


if __name__=='__main__':main()

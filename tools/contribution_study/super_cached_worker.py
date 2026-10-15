"""One resident DM model serves several unchanged frozen generation chunks."""
import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path

from baselines import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    from super_cached_campaign import verify_deployment
    spec=verify_deployment(a.root,a.deployment)
    job=json.loads(a.job.read_text())
    from super_cached_runtime import activate
    activate(spec['bank'],dict(path=str(a.deployment/'manifest.json'),sha256=sha(a.deployment/'manifest.json')))
    import baseline_runtime
    import dataset_infer
    from thunderworld_fp4.cosmos import resources
    original_make=baseline_runtime.make_runner
    resident=[]
    def make(study,name):
        if name!=job['method']:raise ValueError('Resident worker cannot change quantization method')
        if not resident:resident.append(original_make(study,name))
        return resident[0]
    baseline_runtime.make_runner=make
    with resources.shared_worker():
        resource=resources.select_shared_device()
        # Keep the lease for the complete resident session. The unchanged
        # dataset worker's nested admission refers to our already-owned GPU.
        resources.shared_worker=lambda *args,**kwargs:nullcontext()
        resources.select_shared_device=lambda *args,**kwargs:dict(resource,resident_session=True)
        for task in job['tasks']:
            verify_deployment(a.root,a.deployment)
            out=a.root/'tasks'/task['id']/'generation'
            with (a.root/('generate_'+task['id']+'.log')).open('a',buffering=1) as log:
                with redirect_stdout(log),redirect_stderr(log):
                    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),method=job['method'],
                        execution='resident_single_GPU_DM_offline_AR',job=str(a.job))),flush=True)
                    dataset_infer.run(a.root/'super/final/manifest.json',job['method'],out,cases=task['cases'])
            print(json.dumps(dict(completed_chunk=task['id'],physical_gpu=resource['device'])),flush=True)


if __name__=='__main__':main()

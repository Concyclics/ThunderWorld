"""Wait for prefixes, qualify both workers, then continue the original corpus."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

from baselines import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();deployment=a.deployment.resolve();bank=deployment/'prefix_bank'
    from super_cached_campaign import qualification,seal,run,reconcile
    from thunderworld_fp4.cosmos.resources import free_memory
    tasks,queue=reconcile(root)
    state=json.loads((root/'progress.json').read_text())
    state.update(generated_videos=sum(2*len(t['cases']) for t in tasks if t['id'] in queue['completed']),
        phase='preparing_offline_AR_prefixes',deployment_preparation=str(deployment))
    state.pop('paused',None);state.pop('error',None)
    def update(**kwargs):
        state.update(kwargs,updated_utc=datetime.now(timezone.utc).isoformat());write(root/'progress.json',state)
    try:
        deadline=time.monotonic()+4*3600
        while True:
            m=json.loads((bank/'manifest.json').read_text())
            update(phase='preparing_offline_AR_prefixes',prepared_prefix_cases=len(m['cases']),prepared_prefix_assets=len(m['prefixes']))
            if m['complete']:break
            unit=json.loads((deployment/'prefix_launch.json').read_text())['unit']
            active=subprocess.check_output(['systemctl','--user','show',unit,'--property=ActiveState','--value'],text=True).strip()
            if active not in ('active','activating'):raise RuntimeError('Prefix preparation exited before completion')
            if time.monotonic()>deadline:raise TimeoutError('Prefix preparation exceeded four hours')
            time.sleep(15)
        while not all(free_memory().get(g,0)>=88*1024 for g in (0,2)):
            update(phase='waiting_resources_for_cached_qualification');time.sleep(15)
        update(phase='qualifying_offline_AR_two_workers')
        qualification(root,deployment,bank)
        seal(root,deployment,bank)
        run(root,deployment)
    except BaseException as error:
        latest=json.loads((root/'progress.json').read_text())
        latest.update(paused=True,error=repr(error),updated_utc=datetime.now(timezone.utc).isoformat())
        write(root/'progress.json',latest)
        raise


if __name__=='__main__':main()

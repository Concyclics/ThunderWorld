"""Finish the persistent quality job after its serial generation worker exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from baselines import write


def run(root, gpu):
    here=Path(__file__).resolve().parent
    state=dict(complete=False,stage='waiting_for_generation',performance_eligible=False)
    status=root/'completion.json'
    write(status,state)
    try:
        while True:
            flow=json.loads((root/'workflow.json').read_text())
            if flow.get('paused'):
                raise RuntimeError('Generation paused: '+str(flow.get('error')))
            if flow['complete']:
                break
            time.sleep(10)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TW_SHARED_CALIBRATION_GPUS='0,1',
                 TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1')
        for stage,script in [('metrics','selected_linear_metrics.py'),('report','selected_linear_report.py')]:
            state.update(stage=stage);write(status,state)
            with (root/'logs'/(stage+'.log')).open('a') as log:
                subprocess.run([sys.executable,str(here/script),'--root',str(root)],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(stage='waiting_for_video_export');write(status,state)
        while True:
            path=root/'edge/export_progress.json'
            if path.exists() and json.loads(path.read_text())['complete']:
                break
            time.sleep(10)
        state.update(complete=True,stage='complete',report=str(root/'report/report.md'))
        print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(paused=True,error=repr(error));raise
    finally:
        write(status,state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,choices=(0,1),default=1)
    a=p.parse_args();run(a.root.resolve(),a.gpu)

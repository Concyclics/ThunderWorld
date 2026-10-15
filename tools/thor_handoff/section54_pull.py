"""Pull the completed Thor cohort; leave large Nsight traces on Thor.

With --watch only a small status JSON is polled while inference is active.
Bulk result transfer starts after completion or a stopped/review state.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();args.destination.mkdir(parents=True,exist_ok=True)
    host='chenhan@thor'
    remote='/home/chenhan/thunderworld/comparisons/section54_fp4_retest_20260923'
    while True:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
                               'cat '+remote+'/campaign_status.json'],capture_output=True,text=True)
        if result.returncode:
            print(json.dumps(dict(time=time.time(),error=result.stderr.strip())),flush=True)
            if not args.watch:raise RuntimeError('Status unavailable')
            time.sleep(60);continue
        state=json.loads(result.stdout)
        if state.get('complete'):
            supplement=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
                'if test -f '+remote+'/nano-awq-speed-only/status.json; then cat '+remote+'/nano-awq-speed-only/status.json; else echo null; fi'],
                capture_output=True,text=True)
            if supplement.returncode:
                if not args.watch:raise RuntimeError('Supplement status unavailable')
                time.sleep(60);continue
            pending=json.loads(supplement.stdout)
            if pending:
                state['speed_only_supplement']=pending
                if not pending.get('complete') and pending.get('stage')!='stopped_requires_review':
                    state['complete']=False
        temp=args.destination/'remote_status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(args.destination/'remote_status.json')
        finished=state.get('complete') or state.get('state')=='stopped_requires_review'
        if finished or not args.watch:break
        time.sleep(60)
    subprocess.run(['rsync','-az','--exclude=*.nsys-rep','--exclude=*.sqlite','--exclude=*.qdstrm',
                    '--exclude=*.tmp',host+':'+remote+'/',str(args.destination/'measurements')+'/'],check=True)
    print(json.dumps(dict(status=state,local=str(args.destination/'measurements'))),flush=True)


if __name__=='__main__':main()

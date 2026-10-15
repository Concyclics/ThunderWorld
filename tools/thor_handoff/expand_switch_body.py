"""Switch our own Thor supervisor at an idle boundary; never interrupt its worker."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from expand_worker import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True)
    a=p.parse_args()
    while True:
        status=json.loads((a.root/'campaign.json').read_text())
        if status.get('complete') or status.get('failed'):break
        pid=status['pid'];proc=Path('/proc')/str(pid)
        if not proc.exists():break
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'expand_campaign.py' not in cmd or str(a.root) not in cmd:
            raise RuntimeError('Supervisor PID does not belong to this experiment')
        # Some Thor kernels omit CONFIG_PROC_CHILDREN; use portable procps.
        children=subprocess.run(['ps','-o','pid=','--ppid',str(pid)],capture_output=True,text=True).stdout.strip()
        if status.get('waiting') and not children:
            # Close the race with launching the next worker: stop only this
            # verified supervisor, then recheck that it has no child.
            os.kill(pid,signal.SIGSTOP)
            children=subprocess.run(['ps','-o','pid=','--ppid',str(pid)],capture_output=True,text=True).stdout.strip()
            if children:
                os.kill(pid,signal.SIGCONT);time.sleep(5);continue
            os.kill(pid,signal.SIGTERM)
            os.kill(pid,signal.SIGCONT)
            status.update(stopped_for_body_scope=True,stopped=time.time(),
                          note='User requested Cosmos3 body comparison; finished full-request evidence preserved')
            write(a.root/'campaign.json',status)
            break
        time.sleep(5)
    script=Path(__file__).with_name('expand_body_campaign.py')
    os.execv(sys.executable,[sys.executable,str(script),'--root',str(a.root),
             '--variant',str(a.variant),'--require-idle'])


if __name__=='__main__':main()

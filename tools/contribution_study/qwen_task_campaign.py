"""Run preparation, visual qualification, category pilot, repeatability, full queue."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from qwen_task_protocol import now, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reuse-from', type=Path)
    a = p.parse_args()
    a.root = a.root.resolve()
    with (a.root / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        while True:
            state = read(a.root / 'servers.json')
            ready = state['status'] == 'ready'
            if ready:
                for r in state['replicas']:
                    try:
                        with urllib.request.urlopen(r['endpoint'].removesuffix('/v1') + '/health', timeout=3) as response:
                            ready = ready and response.status == 200
                    except OSError:
                        ready = False
            if ready:
                break
            if not Path(f'/proc/{state["pid"]}').exists():
                raise RuntimeError('Judge pool exited before readiness; inspect pool.log')
            if time.time() - started > 1800:
                raise TimeoutError('Judge pool readiness exceeded 30 minutes')
            time.sleep(3)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VLLM_PLUGINS='', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
        env['PYTHONPATH'] = str(a.root / 'deps') + os.pathsep + env.get('PYTHONPATH', '')
        script = Path(__file__).with_name('qwen_task_judge.py')
        stages = [('prepare', []), ('qualify', []), ('pilot', ['score', '--pilot']),
                  ('repeat', []), ('full', ['score', '--shutdown-servers'])]
        if a.reuse_from:
            stages.insert(1, ('reuse', ['reuse', '--from-root', str(a.reuse_from.resolve())]))
        try:
            for name, args in stages:
                cmd = [sys.executable, str(script), *(args or [name]), '--root', str(a.root)]
                write(a.root / 'campaign.json', dict(stage=name, utc=now(), pid=os.getpid(), command=cmd,
                    started_unix=started, complete=False))
                with (a.root / (name + '.log')).open('a') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=log, env=env)
                if result.returncode:
                    raise RuntimeError(f'Judge stage {name} failed ({result.returncode}); see {name}.log')
            write(a.root / 'campaign.json', dict(stage='complete', utc=now(), pid=os.getpid(),
                  started_unix=started, complete=True))
        except BaseException as exc:
            write(a.root / 'campaign_failure.json', dict(utc=now(), error=repr(exc)))
            # Preserve all checkpoints and release only this campaign's judge resources.
            from qwen_task_judge import stop_servers
            stop_servers(a.root)
            raise


if __name__ == '__main__':
    main()

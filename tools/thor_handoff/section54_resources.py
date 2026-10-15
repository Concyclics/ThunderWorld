"""External host telemetry. Never inject CUDA events/hooks into inference."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_baselines import write
from compare_speed import compute_pids


def monitored_run(command, stream, log):
    monitor=None
    telemetry=Path(log).with_suffix('.tegrastats.log')
    with telemetry.open('w') as thermal:
        if shutil.which('tegrastats'):
            monitor=subprocess.Popen(['tegrastats','--interval','1000'],stdout=thermal,stderr=subprocess.STDOUT)
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        samples=[];overlap=[]
        try:
            while child.poll() is None:
                pids=compute_pids();other=[pid for pid in pids if pid!=child.pid]
                sample=dict(time=time.time(),other_compute_pids=other,worker_pid=child.pid)
                samples.append(sample)
                if other:overlap.append(sample)
                time.sleep(3)
        except Exception:
            child.terminate();child.wait(timeout=60)
            raise
        finally:
            if monitor:
                monitor.terminate();monitor.wait(timeout=15)
        write(Path(log).with_suffix('.compute_monitor.json'),dict(samples=samples,overlap=overlap,
              sample_interval_seconds=3,complete=True,scope='External process samples, not continuous GPU activity proof'))
        if overlap:
            stream.write('\nPAUSED: other GPU compute appeared during this run; all its timings are excluded.\n')
            return subprocess.CompletedProcess(command,75)
        return subprocess.CompletedProcess(command,child.returncode)

"""Reproducible kernel tests and CUDA sanitizers before freezing a fresh study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from baselines import sha,write


def run(root):
    if (root/'implementation.lock.json').exists():
        raise ValueError('Study already frozen; retain its qualification receipts')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    from expansion_runtime import source_identity
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    folder=root/'qualification';folder.mkdir(parents=True,exist_ok=True)
    commands={'pytest':[sys.executable,'-m','pytest',str(here/'test_expansion_kernels.py'),'-x','-q']}
    for tool in ('memcheck','racecheck'):
        commands[tool]=['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',
                        sys.executable,str(here/'expansion_experiment.py'),'smoke']
    results=dict(source_sha256=source_identity(),qualifier_sha256=sha(__file__),
                 tests_sha256=sha(here/'test_expansion_kernels.py'))
    with shared_worker():
        results['resource']=select_shared_device()
        for name,command in commands.items():
            log=folder/(name+'.log')
            with log.open('w') as stream:
                code=subprocess.run(command,cwd=repo,stdout=stream,stderr=subprocess.STDOUT).returncode
            results[name]=dict(exit_code=code,log=str(log),sha256=sha(log),command=command)
            write(folder/'results.json',results)
            print(json.dumps({name:results[name]}),flush=True)
            if code:raise RuntimeError('Qualification failed: '+name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root.resolve())

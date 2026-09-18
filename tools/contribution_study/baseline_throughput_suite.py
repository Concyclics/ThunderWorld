"""Serial GPU workers for a predeclared complete-request throughput staircase."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from baselines import sha, write


def prepare(study, output, reserve_mib=12*1024):
    if reserve_mib < 2048:raise ValueError('Keep at least 2048 MiB reserved')
    manifest = json.loads((study/'manifest.json').read_text())
    cases = [next(e for e in manifest['cases'] if e['benchmark'] == b)
             for b in ('rbench', 'paibench')]
    methods = manifest['batch_protocol']['methods']
    jobs = []
    rng = random.Random(180918)
    cells = [(name, case['case']['id']) for name in methods for case in cases]
    rng.shuffle(cells)
    for method, case in cells:
        jobs.append(dict(method=method, case=case, batches=manifest['batch_protocol']['sizes'],
                         file=f'{method}__{case}.json'))
    value = dict(frozen=True, study=str(study.resolve()), manifest_sha256=sha(study/'manifest.json'),
        source_sha256=sha(__file__), worker_sha256=sha(Path(__file__).with_name('baseline_batch.py')),
        cases=cases, methods=methods, jobs=jobs, rounds=5, waves=3,
        ordering_seed=180918, performance_eligible=False, reserve_mib=reserve_mib,
        concurrent_quality_policy='User requested speed only for concurrency. No batch > 1 output-equivalence gate; quality is separately measured at batch 1.',
        scope='Each resident method/case worker follows the common 1/2/4/8 staircase, avoiding repeated model loading. '
              'Fixed random method/case order; five within-configuration rounds, three waves each. '
              'Not the interleaved-independent-load formal latency protocol. '
              'Batch = independent complete requests; frozen existing implementations, no tuning.')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'suite.json', value)


def execute(root):
    suite_path = root/'suite.json'
    suite = json.loads(suite_path.read_text())
    if suite['worker_sha256'] != sha(Path(__file__).with_name('baseline_batch.py')):
        raise ValueError('Worker changed after suite freeze')
    study = Path(suite['study'])
    if suite['manifest_sha256'] != sha(study/'manifest.json'):
        raise ValueError('Study changed after suite freeze')
    state = dict(complete=False, suite_sha256=sha(suite_path), completed=[], skipped=[], performance_eligible=False)
    write(root/'progress.json', state)
    root.joinpath('logs').mkdir(exist_ok=True)
    try:
        for job in suite['jobs']:
            target = root/job['file']
            if target.exists():
                done = json.loads(target.read_text())
                if (not done['complete'] or done['manifest_sha256'] != suite['manifest_sha256'] or
                    done['source_sha256'] != suite['worker_sha256']):
                    raise ValueError('Inspect incomplete or changed worker: '+str(target))
                state['completed'].append(job['file'])
                continue
            cmd = [sys.executable, str(Path(__file__).with_name('baseline_batch.py')),
                   '--study', str(study), '--method', job['method'], '--case', job['case'],
                   '--output', str(target), '--batches', *map(str, job['batches']),
                   '--rounds', str(suite['rounds']), '--waves', str(suite['waves']),
                   '--reserve-mib', str(suite.get('reserve_mib',12*1024))]
            write(root/'logs'/job['file'], dict(command=cmd))
            logfile = root/'logs'/job['file'].replace('.json', '.log')
            state['active'] = job
            write(root/'progress.json', state)
            print(json.dumps(dict(starting=job, completed=len(state['completed']))), flush=True)
            with logfile.open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed'].append(job['file'])
            state.pop('active', None)
            write(root/'progress.json', state)
        state['complete'] = True
        write(root/'progress.json', state)
    except Exception as error:
        state.update(paused=True, error=repr(error))
        write(root/'progress.json', state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['prepare', 'run'])
    p.add_argument('--study', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reserve-mib', type=int, default=12*1024)
    a = p.parse_args()
    if a.operation == 'prepare':
        if a.study is None:
            p.error('prepare requires --study')
        prepare(a.study, a.output, a.reserve_mib)
    else:
        execute(a.output)


if __name__ == '__main__':
    main()

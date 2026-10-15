"""Finish the bounded experiment after the independent-load speed sweep."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bundle import ROOT
from baselines import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--reference-study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluation-deps', type=Path,
                   help='Optional directory containing evaluation-only Python dependencies')
    a = p.parse_args()
    if a.evaluation_deps is not None and not a.evaluation_deps.is_dir():
        p.error('--evaluation-deps must be an existing directory')
    state = dict(complete=False, completed=[])
    status = a.output/'followup_progress.json'
    write(status, state)
    suite = str(Path(__file__).with_name('optimization_suite.py'))
    metrics = str(Path(__file__).with_name('optimization_metrics.py'))
    jobs = []
    for sanitizer in ('memcheck', 'racecheck'):
        jobs.append((sanitizer, ['/usr/local/cuda/bin/compute-sanitizer', '--tool', sanitizer,
            '--error-exitcode', '1', sys.executable, '-m', 'pytest', '-q',
            'packages/fp4-repair/tests/test_represented_selector.py'], False))
    jobs.append(('quality_execution', [sys.executable, suite, 'quality', '--study', str(a.study),
                 '--output', str(a.output), '--variants', 'norm'], False))
    jobs.append(('quality_selector', [sys.executable, suite, 'quality', '--study', str(a.study),
                 '--output', str(a.output/'selector'), '--variants', 'gain',
                 '--methods', 'b1.5_protected_n', 'b2_joint'], False))
    for method in ('b1.5_protected_n', 'b2_joint'):
        jobs.append(('metrics_'+method, [sys.executable, metrics, '--candidate',
            str(a.output/'selector/quality'/f'{method}_gain'), '--reference-study', str(a.reference_study),
            '--output', str(a.output/'selector'/f'{method}_metrics.json')], True))
    jobs.append(('controls', [sys.executable, suite, 'benchmark', '--study', str(a.reference_study),
                 '--output', str(a.output/'controls'), '--variants', 'frozen', '--methods', 'bf16', 'fp8'], False))
    for variant in ('frozen', 'norm'):
        jobs.append(('profile_'+variant, [sys.executable, str(Path(__file__).with_name('optimize.py')),
            'profile', '--study', str(a.study), '--method', 'b2_joint', '--variant', variant,
            '--cases', 'rbench__visual_reasoning_0010', '--output', str(a.output/'profile'/variant)], False))
    try:
        while True:
            progress = json.loads((a.output/'benchmark_progress.json').read_text())
            if progress.get('error'):
                raise RuntimeError('Benchmark failed: '+progress['error'])
            if progress['complete']:
                break
            state['active'] = 'waiting_for_benchmark'; write(status, state)
            time.sleep(10)
        for label, command, cpu in jobs:
            state['active'] = label; write(status, state)
            print(json.dumps(dict(starting=label)), flush=True)
            env = dict(os.environ)
            if cpu:
                env['CUDA_VISIBLE_DEVICES'] = ''
                if a.evaluation_deps is not None:
                    env['PYTHONPATH'] = os.pathsep.join(filter(None, (
                        str(a.evaluation_deps.resolve()), env.get('PYTHONPATH'))))
            with (a.output/'logs'/('followup_'+label+'.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed'].append(label); write(status, state)
        state.pop('active', None); state['complete'] = True; write(status, state)
    except Exception as error:
        state['error'] = repr(error); write(status, state)
        raise


if __name__ == '__main__':
    main()

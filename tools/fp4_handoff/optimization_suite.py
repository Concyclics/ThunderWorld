"""Serial, resumable experiments; never overlap GPU experiment workers."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from bundle import ROOT
from baselines import sha, write


METHODS = ['b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['qualify', 'benchmark', 'quality'])
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variants', nargs='+')
    p.add_argument('--methods', nargs='+', default=METHODS)
    a = p.parse_args()
    worker = Path(__file__).with_name('optimize.py')
    defaults = {'qualify': ['frozen', 'calibrated', 'codec', 'norm'],
                'benchmark': ['frozen', 'norm'], 'quality': ['norm']}
    variants = a.variants or defaults[a.stage]
    jobs = [(method, variant, repeat) for repeat in range(5 if a.stage == 'benchmark' else 1)
            for method in a.methods for variant in variants]
    if a.stage == 'benchmark':
        rng = random.Random(181918)
        for repeat in range(5):
            cells = [(m, v, repeat) for m in a.methods for v in variants]
            rng.shuffle(cells)
            offset = repeat*len(cells)
            jobs[offset:offset+len(cells)] = cells
    protocol = dict(stage=a.stage, methods=a.methods, variants=variants, jobs=jobs,
                    manifest_sha256=sha(a.study/'manifest.json'), worker_sha256=sha(worker),
                    performance_eligible=False, scope='Serial workers; each benchmark round independently loads each method/variant; fixed randomized order; 3 seeds/case/round.')
    protocol_file = a.output/(a.stage+'_protocol.json')
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('Changed frozen protocol')
    else:
        write(protocol_file, protocol)
    state = dict(complete=False, jobs=len(jobs), completed=[], protocol_sha256=sha(protocol_file))
    progress = a.output/(a.stage+'_progress.json')
    logs = a.output/'logs'; logs.mkdir(parents=True, exist_ok=True)
    try:
        for method, variant, repeat in jobs:
            label = f'{method}_{variant}'+(f'_r{repeat}' if a.stage == 'benchmark' else '')
            target = a.output/a.stage/label
            result = target/'result.json'
            if result.exists() and json.loads(result.read_text()).get('complete'):
                state['completed'].append(label)
                continue
            if a.stage == 'benchmark' and variant != 'frozen':
                qual = a.output/'qualify'/f'{method}_{variant}'/'result.json'
                q = json.loads(qual.read_text())
                if not q['complete'] or not all(r['source_hashes_match'] for c in q['cases'].values() for r in c['rows']):
                    raise ValueError('Execution variant has not passed output qualification: '+label)
            state['active'] = label; write(progress, state)
            cmd = [sys.executable, str(worker), a.stage, '--study', str(a.study),
                   '--method', method, '--variant', variant, '--output', str(target), '--rounds', '1']
            print(json.dumps(dict(starting=label, finished=len(state['completed']), total=len(jobs))), flush=True)
            with (logs/(a.stage+'_'+label+'.log')).open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed'].append(label); state.pop('active', None); write(progress, state)
        state['complete'] = True; write(progress, state)
    except Exception as error:
        state['error'] = repr(error); write(progress, state)
        raise


if __name__ == '__main__':
    main()

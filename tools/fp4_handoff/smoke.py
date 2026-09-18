"""Serial full-request verification on the two frozen throughput inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from bundle import ROOT, METHODS
from baselines import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--require-source-hash', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.study/'manifest.json').read_text())
    cases = [next(e['case']['id'] for e in manifest['cases'] if e['benchmark']==b)
             for b in ('rbench','paibench')]
    logs = args.study/'smoke_logs';logs.mkdir(exist_ok=True)
    result = dict(complete=False, performance_eligible=False, manifest_sha256=sha(args.study/'manifest.json'),
                  cases=cases, methods=args.methods, rows=[])
    for method in args.methods:
        command = [sys.executable,str(Path(__file__).with_name('infer.py')),'--study',str(args.study),
                   '--method',method,'--cases',*cases]
        if args.require_source_hash:command.append('--require-source-hash')
        with (logs/(method+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        for case in cases:
            path = args.study/'generation'/method/case/'result.json'
            receipt = json.loads(path.read_text())
            if not receipt['complete']:raise ValueError('Incomplete request')
            result['rows'].append(dict(method=method, case=case, receipt_sha256=sha(path),
                seeds=[r['seed'] for r in receipt['rows']],
                all_source_hashes_match=all(r['source_signal_hashes_match'] for r in receipt['rows'])))
        write(args.study/'smoke.json',result)
        print(json.dumps(dict(method=method, complete=True)),flush=True)
    result['complete'] = True
    write(args.study/'smoke.json',result)


if __name__ == '__main__':main()

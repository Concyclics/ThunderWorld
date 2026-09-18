"""Serial baseline attribution with unchanged kernels and frozen full outputs."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from bundle import ROOT
from baselines import sha, write
from optimize import source_identity


METHODS = ['bf16', 'fp8', 'fp4_base', 'awq_marlin', 'arcquant', 'arcquant100',
           'sharq_perf', 'b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded']
CASE = 'rbench__visual_reasoning_0010'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-study', type=Path, required=True)
    p.add_argument('--compact-orders', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--benchmark-baselines', action='store_true')
    a = p.parse_args()
    source = a.reference_study/'manifest.json'
    manifest = copy.deepcopy(json.loads(source.read_text()))
    orders_path = a.compact_orders/'manifest.json'
    orders = json.loads(orders_path.read_text())
    if (not orders['complete'] or orders['schema'] != 'arc-channel-order-v1' or
            orders['source_capture_sha256'] != manifest['external']['capture_sha256']):
        raise ValueError('ARC compact orders must come from the same frozen panels')
    hashes, receipts = {}, {}
    for method in METHODS:
        hashes[method] = {}
        for entry in manifest['cases']:
            case = entry['case']['id']
            path = a.reference_study/'generation'/method/case/'result.json'
            d = json.loads(path.read_text())
            if not d['complete']:
                raise ValueError('Incomplete source generation')
            hashes[method][case] = {str(r['seed']): r['sha256'] for r in d['rows']}
            receipts[str(path)] = sha(path)
    folder = a.output/'study'
    folder.mkdir(parents=True, exist_ok=True)
    oracle = folder/'source_output_hashes.json'
    write(oracle, hashes)
    manifest['source_output_hashes'] = str(oracle.resolve())
    manifest['external']['capture'] = str(a.compact_orders.resolve())
    manifest['external']['capture_sha256'] = sha(orders_path)
    manifest['breakdown_provenance'] = dict(reference_manifest_sha256=sha(source),
        source_receipts=receipts, compact_orders_sha256=sha(orders_path),
        scope='Unchanged method definitions. ARC teacher maxima are stored as exact precomputed orders; no online algorithm or kernel optimization.')
    target = folder/'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Changed frozen breakdown inputs')
    write(target, manifest)
    state = dict(complete=False, completed=[], profile_case=CASE,
        source_sha256=sha(__file__), performance_eligible=False)
    progress = a.output/'progress.json'
    logs = a.output/'logs'; logs.mkdir(exist_ok=True)
    try:
        for method in a.methods:
            variant = 'norm' if method.startswith('b1.') or method in ('b2_joint', 'b2.5_ffn_guarded') else 'frozen'
            output = a.output/'profile'/method
            file = output/'result.json'
            if file.exists() and json.loads(file.read_text())['complete']:
                result = json.loads(file.read_text())
                if result['identity']['source'] != source_identity():
                    raise ValueError('Reused profiler source changed')
            else:
                state['active'] = method; write(progress, state)
                command = [sys.executable, str(Path(__file__).with_name('optimize.py')), 'profile',
                    '--study', str(folder), '--method', method, '--variant', variant,
                    '--cases', CASE, '--rounds', '1', '--output', str(output)]
                print(json.dumps(dict(starting=method)), flush=True)
                with (logs/(method+'.log')).open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads(file.read_text())
            if (set(result['cases']) != {CASE} or result['identity']['method'] != method or
                    result['identity']['variant'] != variant or
                    result['identity']['operation'] != 'profile'):
                raise ValueError('Unmatched profile')
            rows = result['cases'][CASE]['rows']
            if len(rows) != 2 or any(r['sha256'] != hashes[method][CASE][str(r['seed'])] for r in rows):
                raise ValueError('Profile changed a frozen output')
            state['completed'].append(method); write(progress, state)
        if a.benchmark_baselines:
            state['active'] = 'baseline_request_timings'; write(progress, state)
            command = [sys.executable, str(Path(__file__).with_name('optimization_suite.py')), 'benchmark',
                '--study', str(folder), '--output', str(a.output/'timing'), '--variants', 'frozen',
                '--methods', 'fp4_base', 'awq_marlin', 'arcquant', 'arcquant100', 'sharq_perf']
            with (logs/'timing.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.pop('active', None); state['complete'] = True; write(progress, state)
    except Exception as error:
        state['error'] = repr(error); write(progress, state)
        raise


if __name__ == '__main__':
    main()

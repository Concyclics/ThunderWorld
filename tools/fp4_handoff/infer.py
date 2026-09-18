"""Run any frozen repair/baseline with receiver-local assets and output receipts."""
import argparse
import gc
import json
from pathlib import Path

from bundle import ROOT
from baselines import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--cases', nargs='+', help='Default: all 16 frozen development scenes')
    parser.add_argument('--require-source-hash', action='store_true', help='Strict original-device byte equivalence')
    args = parser.parse_args()
    manifest_path = args.study/'manifest.json'
    study = json.loads(manifest_path.read_text())
    if not study.get('frozen') or study['steps'] != 35 or args.method not in study['methods']:
        raise ValueError('Frozen matching 35-step method required')
    cases = [e for e in study['cases'] if not args.cases or e['case']['id'] in args.cases]
    if not cases or (args.cases and {e['case']['id'] for e in cases} != set(args.cases)):
        raise ValueError('Unknown case')
    source = json.loads(Path(study['source_output_hashes']).read_text())[args.method]
    folder = args.study/'generation'/args.method
    identity = dict(manifest_sha256=sha(manifest_path), method=args.method, source_sha256=sha(__file__),
                    runtime_sha256=sha(ROOT/'tools/contribution_study/baseline_runtime.py'),
                    require_source_hash=args.require_source_hash)
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        resource = select_shared_device()
        import numpy as np
        import torch
        from baseline_runtime import make_runner
        from dataset_infer import clear_case
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        runner, adapter = make_runner(study, args.method)
        write(folder/'prepared.json', dict(identity=identity, resource=resource, adapter=adapter, runtime=runner.receipt()))
        try:
            for entry in cases:
                case = entry['case']
                if sha(case['image']) != entry['image_sha256']:
                    raise ValueError('Changed input image')
                output = folder/case['id']
                path = output/'result.json'
                case_identity = dict(**identity, case=case, image_sha256=entry['image_sha256'], seeds=entry['seeds'])
                result = dict(identity=case_identity, complete=False, rows=[], performance_eligible=False)
                if path.exists():
                    result = json.loads(path.read_text())
                    if result['identity'] != case_identity:
                        raise ValueError('Changed resume identity; use a new materialized study')
                    for row in result['rows']:
                        if sha(output/row['outputs']) != row['outputs_sha256']:
                            raise ValueError('Corrupt resumed output')
                    if result['complete']:continue
                output.mkdir(parents=True, exist_ok=True)
                clear_case(runner)
                runner.warmup(case, study['steps'])
                for seed in entry['seeds']:
                    if any(r['seed'] == seed for r in result['rows']):continue
                    signals, row = runner.infer(case, seed, study['steps'])
                    if not all(np.isfinite(x).all() for x in signals.values()):
                        raise ValueError('Nonfinite output')
                    matches = row['sha256'] == source[case['id']][str(seed)]
                    if args.require_source_hash and not matches:
                        raise ValueError('Output differs from source-device receipt')
                    ledger = row.pop('compute_ledger')
                    if ledger:
                        if not ledger['feasible']:raise ValueError('Request exceeds compute budget')
                        write(output/'compute.json', ledger)
                    filename = f'seed{seed}.npz'
                    np.savez_compressed(output/filename, **signals)
                    row.update(outputs=filename, outputs_sha256=sha(output/filename), source_signal_hashes_match=matches,
                               actual_mma_ratio=ledger['ratio'] if ledger else None,
                               compute_sha256=sha(output/'compute.json') if ledger else None)
                    result['rows'].append(row)
                    write(path, result)
                result.update(complete=True, adapter=adapter, model_identity=runner.identity, targets=runner.engine.targets)
                write(path, result)
                print(json.dumps(dict(method=args.method, case=case['id'], complete=True,
                    all_source_hashes_match=all(r['source_signal_hashes_match'] for r in result['rows']))), flush=True)
        finally:
            del runner
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':main()

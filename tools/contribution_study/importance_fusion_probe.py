"""Qualify existing prescale-aware FFN fusion against frozen joint-asset outputs.

Only an execution-equivalence probe: two development scenes per budget, both
seeds, full 35 steps. Shared timings are not promoted to a speed result.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from baselines import sha, write
from dataset_infer import clear_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['b1.5_joint', 'b2_joint', 'b2.5_joint'])
    a = p.parse_args()
    while True:
        status = json.loads((a.run / 'workflow.json').read_text()) if (a.run / 'workflow.json').exists() else {}
        if status.get('paused'):
            raise RuntimeError('Generation experiment paused')
        if status.get('complete'):
            break
        time.sleep(5)
    manifest = json.loads((a.run / 'manifest.json').read_text())
    if len(set(a.methods)) != len(a.methods) or any(name not in manifest['methods'] for name in a.methods):
        raise ValueError('Probe methods must be distinct frozen configurations')
    result = dict(complete=False, manifest_sha256=sha(a.run / 'manifest.json'), source_sha256=sha(__file__),
                  performance_eligible=False, rows=[], methods={}, scope='Full-output bitwise execution equivalence on one RBench and one PAI development scene per budget, both seeds. No speed or final-quality claim.')
    # Exercise a long prompt as well as a different dataset. Actual token shapes
    # are fixed by the existing reference receipts and checked below.
    cases = [max((c for c in manifest['cases'] if c['benchmark'] == b), key=lambda c: len(c['case']['prompt']))
             for b in ('rbench', 'paibench')]
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        result['resource'] = select_shared_device()
        import torch
        import numpy as np
        from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
        from thunderworld_fp4.cosmos.calibrated_fusion import install
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for name in a.methods:
                method = manifest['methods'][name]
                if sha(Path(method['assets']) / 'manifest.json') != method['assets_sha256']:
                    raise ValueError('Changed quantized assets')
                runner = CosmosRunner(manifest['model'], RunConfig.from_dict(method['config']), assets=method['assets'], ar_cache=True, first_frame=True).prepare()
                names = install(runner.pipe.transformer)
                if len(names) != 28:
                    raise ValueError('Incomplete calibrated FFN fusion')
                result['methods'][name] = dict(fused_layers=names, config=method['config'], assets_sha256=method['assets_sha256'])
                for case in cases:
                    folder = a.run / 'generation' / name / case['case']['id']
                    reference = json.loads((folder / 'result.json').read_text())
                    if not reference['complete'] or reference['model_identity'] != runner.identity:
                        raise ValueError('Unmatched frozen reference')
                    clear_case(runner)
                    runner.warmup(case['case'], 35)
                    for seed in case['seeds']:
                        expected = next(row for row in reference['rows'] if row['seed'] == seed)
                        if sha(folder / expected['outputs']) != expected['outputs_sha256']:
                            raise ValueError('Changed reference output')
                        signals, row = runner.infer(case['case'], seed, 35)
                        if row['compute_ledger']['ratio'] != expected['actual_mma_ratio']:
                            raise ValueError('Fusion changed issued MMA counts')
                        with np.load(folder / expected['outputs']) as old:
                            if set(signals) != set(old.files):
                                raise ValueError('Changed output keys')
                            for key, value in signals.items():
                                if value.dtype != old[key].dtype or value.shape != old[key].shape or hashlib.sha256(value.tobytes()).hexdigest() != expected['sha256'][key]:
                                    raise AssertionError(f'Fusion changed {name}/{case["case"]["id"]}/{seed}/{key}')
                        result['rows'].append(dict(method=name, case=case['case']['id'], seed=seed, bitwise_equal=True,
                                                    reference_sha256=sha(folder / 'result.json'), output_sha256=row['sha256'],
                                                    actual_mma_ratio=row['compute_ledger']['ratio'], performance_eligible=False))
                        write(a.output, result)
                clear_case(runner)
                del runner
                gc.collect()
                torch.cuda.empty_cache()
            result['complete'] = True
            write(a.output, result)
        except Exception as error:
            result.update(paused=True, error=repr(error))
            write(a.output, result)
            raise


if __name__ == '__main__':
    main()

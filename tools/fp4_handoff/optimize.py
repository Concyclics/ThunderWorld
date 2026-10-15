"""Bounded execution/selector experiments against frozen handoff assets.

Every worker independently prepares one method/variant. Diagnostic timing is
only collected with an idle GPU and never promoted to the formal ranking.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess

from bundle import ROOT
from baselines import sha, write


VARIANTS = {
    'frozen': {},
    'calibrated': {'fuse_calibrated': True},
    'pointwise': {'fuse_calibrated': True, 'fuse_pointwise': True},
    'rotary': {'fuse_calibrated': True, 'fuse_pointwise': True, 'static_rotary': True},
    'codec': {'fuse_calibrated': True, 'fuse_pointwise': True, 'static_rotary': True,
              'codec_graph': True},
    'norm': {'fuse_calibrated': True, 'fuse_pointwise': True, 'static_rotary': True,
             'codec_graph': True, 'fuse_norm': True},
    'gain': {'fuse_calibrated': True},
}


def source_identity():
    paths = sorted((ROOT/'packages/fp4-repair/src/thunderworld_fp4').rglob('*.py'))
    paths += [Path(__file__), ROOT/'tools/contribution_study/baseline_runtime.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['qualify', 'benchmark', 'quality', 'profile'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('rounds must be positive')
    manifest_file = args.study/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    if not manifest.get('frozen') or manifest['steps'] != 35:
        raise ValueError('A frozen 35-step study is required')
    original = copy.deepcopy(manifest['methods'][args.method])
    spec = manifest['methods'][args.method]
    spec['runtime_options'] = VARIANTS[args.variant]
    if args.variant == 'gain':
        spec['config']['activation_score'] = 'represented_gain'
    cases = manifest['cases']
    if args.cases:
        cases = [c for c in cases if c['case']['id'] in args.cases]
        if {c['case']['id'] for c in cases} != set(args.cases):
            raise ValueError('Unknown case')
    elif args.operation != 'quality':
        cases = [max((c for c in cases if c['benchmark'] == b),
                     key=lambda c: len(c['case']['prompt'])) for b in ('rbench', 'paibench')]
    expected = None
    if manifest.get('source_output_hashes'):
        expected = json.loads(Path(manifest['source_output_hashes']).read_text()).get(args.method)
    if args.operation == 'qualify' and expected is None:
        raise ValueError('Qualification needs frozen source hashes')
    identity = dict(source=source_identity(), manifest_sha256=sha(manifest_file),
                    method=args.method, variant=args.variant, operation=args.operation,
                    cases=[c['case']['id'] for c in cases], rounds=args.rounds,
                    original_spec=original, effective_spec=spec)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'result.json'
    result = dict(identity=identity, complete=False, cases={}, performance_eligible=False,
                  timing_scope='Complete debug-off request including fresh AR prefill, decode, CPU output; no loading/capture/serialization',
                  scope='Frozen-asset development experiment. No final-test, formal-speed or hardware-ceiling claim.')
    if path.exists():
        old = json.loads(path.read_text())
        if old['identity'] != identity:
            raise ValueError('Changed experiment identity; use a fresh output directory')
        if old['complete']:
            return
        result = old
        result.pop('error', None)
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        result['resource'] = select_shared_device()
        import numpy as np
        import torch
        from baseline_runtime import make_runner
        from dataset_infer import clear_case
        from thunderworld_fp4.cosmos.cli import idle_guard
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        runner = None
        try:
            if args.operation == 'benchmark':
                idle_guard()
            runner, adapter = make_runner(manifest, args.method)
            result['prepared'] = runner.receipt()
            result['gpu'] = dict(name=torch.cuda.get_device_name(),
                                 capability=torch.cuda.get_device_capability(),
                                 torch=torch.__version__, cuda=torch.version.cuda)
            for entry in cases:
                case = entry['case']; case_id = case['id']
                if result['cases'].get(case_id, {}).get('complete'):
                    continue
                if sha(case['image']) != entry['image_sha256']:
                    raise ValueError('Changed input image')
                clear_case(runner)
                runner.warmup(case, 35)
                runner.warm_request(case, 35)
                captures = len(runner.engine.cache)
                detail = dict(complete=False, benchmark=entry['benchmark'], rows=[],
                              preparation=dict(runner.timings), captures=captures)
                result['cases'][case_id] = detail
                write(path, result)
                seeds = entry['seeds']
                if args.operation == 'benchmark':
                    seeds = [*seeds[:2], 6099]
                for repeat in range(args.rounds if args.operation == 'benchmark' else 1):
                    for seed in seeds:
                        if args.operation == 'benchmark':
                            idle_guard()
                        if args.operation == 'profile':
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                                signals, row = runner.infer(case, seed, 35)
                            prof.export_chrome_trace(str(args.output/(case_id+f'_{seed}.trace.json')))
                            events = [dict(name=e.key, count=e.count, cpu_us=e.self_cpu_time_total,
                                           cuda_us=e.self_device_time_total) for e in prof.key_averages()]
                            write(args.output/(case_id+f'_{seed}.profile.json'), sorted(events, key=lambda e: -e['cuda_us']))
                        else:
                            signals, row = runner.infer(case, seed, 35)
                        if args.operation == 'benchmark':
                            idle_guard()
                        if len(runner.engine.cache) != captures:
                            raise ValueError('A measured request captured a new Transformer graph')
                        ledger = row.pop('compute_ledger')
                        if ledger:
                            if not ledger['feasible']:
                                raise ValueError('Repair budget exceeded')
                            if not repeat and seed == seeds[0]:
                                write(args.output/(case_id+'.compute.json'), ledger)
                        source = expected.get(case_id, {}).get(str(seed)) if expected else None
                        row.update(repeat=repeat, actual_mma_ratio=ledger['ratio'] if ledger else None,
                                   source_hashes_match=(row['sha256'] == source) if source else None,
                                   profiled=args.operation == 'profile')
                        if args.operation == 'quality':
                            output = args.output/case_id/f'seed{seed}.npz'
                            output.parent.mkdir(exist_ok=True)
                            np.savez_compressed(output, **signals)
                            row.update(outputs=str(output.relative_to(args.output)), outputs_sha256=sha(output))
                        detail['rows'].append(row)
                        write(path, result)
                        print(json.dumps(dict(method=args.method, variant=args.variant, case=case_id,
                                              seed=seed, repeat=repeat, seconds=row['wall_seconds'],
                                              equal=row['source_hashes_match'])), flush=True)
                detail['complete'] = True
                write(path, result)
            if source_identity() != identity['source']:
                raise ValueError('Runtime source changed while experiment was running')
            result['complete'] = True
            write(path, result)
        except Exception as error:
            result['error'] = repr(error)
            write(path, result)
            raise
        finally:
            if runner is not None:
                del runner
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

"""Unified debug-off quality worker: native adapters, fresh seeds, safe resume.

Historical hashes are optional unless explicitly required. Missing hashes never
count as matches. Shared calibration results are never performance-qualified.
"""
import argparse
import gc
import json
from pathlib import Path
from baselines import sha, write


def clear_case(runner):
    """Drop captured input-dependent state before a different prompt/image shape."""
    import torch
    runner.engine.clear()
    for key in ('attention_prepared', 'attention_shapes', 'shapes', 'trace_masks', 'pair_masks'):
        getattr(runner.engine, key).clear()
    runner.engine.mask_history.clear()
    runner.engine.budget_ledger = None
    runner._warmed.clear()
    if runner.ar_cache:
        runner.ar_cache.active = None
        runner.ar_cache.graphs.clear()
        runner.ar_cache.slots.clear()
        runner.ar_cache.seen.clear()
    gc.collect()
    torch.cuda.empty_cache()


def selected_cases(study, names=None, smoke=False):
    cases = study['cases']
    ids = [e['case']['id'] for e in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case ID')
    if names:
        cases = [e for e in cases if e['case']['id'] in names]
        if {e['case']['id'] for e in cases} != set(names):
            raise ValueError('Unknown case')
    if not cases:
        raise ValueError('Empty case selection')
    if smoke:
        first = cases[0]
        second = next((e for e in cases if e.get('benchmark') != first.get('benchmark')), None)
        if second is None:
            raise ValueError('Smoke roundtrip needs two different dataset scenes')
        cases = [first, second]
    for entry in cases:
        cid = entry['case']['id']
        if not cid or Path(cid).name != cid or cid in ('.', '..'):
            raise ValueError('Unsafe case ID')
        if not entry['seeds'] or len(set(entry['seeds'])) != len(entry['seeds']):
            raise ValueError('Empty or duplicate seed list')
        if sha(entry['case']['image']) != entry['image_sha256']:
            raise ValueError('Changed input image')
    return cases


def source_hashes(study, method, cases, required=False):
    source = {}
    if study.get('source_output_hashes'):
        source = json.loads(Path(study['source_output_hashes']).read_text()).get(method, {})
    for entry in cases:
        if required and any(str(s) not in source.get(entry['case']['id'], {}) for s in entry['seeds']):
            raise ValueError('Required historical hash absent; fresh seeds cannot be source-qualified')
    return source


def validate_rows(result, folder, identity):
    if result['identity'] != identity:
        raise ValueError('Changed resume identity; use a new study/output directory')
    seeds = [r['seed'] for r in result['rows']]
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(identity['seeds']):
        raise ValueError('Unexpected or duplicate saved seeds')
    for row in result['rows']:
        path = (folder/row['outputs']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != row['outputs_sha256']:
            raise ValueError('Corrupt resumed output')
        if row.get('compute_sha256') and sha(folder/'compute.json') != row['compute_sha256']:
            raise ValueError('Changed request budget ledger')
    if result['complete'] and set(seeds) != set(identity['seeds']):
        raise ValueError('Complete receipt has missing seeds')


def run(manifest, method, output, *, smoke=False, cases=None, require_source_hash=False):
    from baseline_runtime import verify
    study = json.loads(manifest.read_text())
    if not study.get('frozen') or study['steps'] != 35 or method not in study['methods']:
        raise ValueError('Frozen matching 35-step method required')
    cfg = study['methods'][method]['config']
    if cfg['debug'] != 'off' or cfg['resource_mode'] != 'shared-calibration':
        raise ValueError('Quality requires debug off / shared-calibration')
    verify(study, method)
    entries = selected_cases(study, cases, smoke)
    source = source_hashes(study, method, entries, require_source_hash)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(manifest_sha256=sha(manifest), method=method, source_sha256=sha(__file__),
        runtime_sha256=sha(Path(__file__).with_name('baseline_runtime.py')), smoke=smoke,
        cases=[e['case']['id'] for e in entries], require_source_hash=require_source_hash,
        source_output_hashes_sha256=sha(study['source_output_hashes']) if study.get('source_output_hashes') else None)
    progress = output/'progress.json'
    if progress.exists() and json.loads(progress.read_text())['identity'] != identity:
        raise ValueError('Changed resume identity')
    state = dict(identity=identity, complete=False, completed=[], performance_eligible=False)
    pending = []
    for entry in entries:
        folder = output/entry['case']['id']
        ci = dict(**identity, case=entry['case'], image_sha256=entry['image_sha256'], seeds=entry['seeds'])
        result = dict(identity=ci, complete=False, rows=[], performance_eligible=False)
        if (folder/'result.json').exists():
            result = json.loads((folder/'result.json').read_text())
            validate_rows(result, folder, ci)
        if result['complete']:
            state['completed'].append(entry['case']['id'])
        else:
            pending.append((entry, folder, result))
    old = json.loads(progress.read_text()) if progress.exists() else {}
    if not pending and (not smoke or old.get('roundtrip_bitwise_equal')):
        state['complete'] = True
        if smoke:
            state['roundtrip_bitwise_equal'] = True
        write(progress, state)
        return state
    write(progress, state)
    runner = None
    try:
        from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
        with shared_worker():
            resource = select_shared_device()
            import numpy as np
            import torch
            from baseline_runtime import make_runner
            torch.set_num_threads(8)
            torch.backends.cuda.matmul.allow_tf32 = False
            runner, adapter = make_runner(study, method)
            write(output/'prepared.json', dict(identity=identity, resource=resource, adapter=adapter,
                runtime=runner.receipt(), performance_eligible=False))
            for entry, folder, result in pending:
                folder.mkdir(exist_ok=True)
                clear_case(runner)
                runner.warmup(entry['case'], study['steps'])
                if runner.engine.budget_ledger and not runner.engine.budget_ledger['feasible']:
                    raise ValueError('Current input exceeds compute cap')
                for seed in entry['seeds']:
                    if any(r['seed'] == seed for r in result['rows']):
                        continue
                    signals, row = runner.infer(entry['case'], seed, study['steps'])
                    if not signals or not all(np.isfinite(x).all() for x in signals.values()):
                        raise ValueError('Empty or nonfinite output')
                    expected = source.get(entry['case']['id'], {}).get(str(seed))
                    matches = row['sha256'] == expected if expected is not None else None
                    if require_source_hash and matches is not True:
                        raise ValueError('Output differs from historical receipt')
                    ledger = row.pop('compute_ledger')
                    if cfg.get('compute_budget') is not None and not ledger:
                        raise ValueError('Budgeted request has no compute ledger')
                    if ledger:
                        if not ledger['feasible']:
                            raise ValueError('Request exceeds compute cap')
                        ledger_path = folder/'compute.json'
                        if ledger_path.exists() and json.loads(ledger_path.read_text()) != ledger:
                            raise ValueError('Compute bound changed between paired requests')
                        write(ledger_path, ledger)
                    filename = f'seed{seed}.npz'
                    temp = folder/(filename+'.partial')
                    with temp.open('wb') as stream:
                        np.savez_compressed(stream, **signals)
                    temp.replace(folder/filename)
                    row.update(outputs=filename, outputs_sha256=sha(folder/filename),
                        source_signal_hashes_match=matches, performance_eligible=False,
                        compute_sha256=sha(folder/'compute.json') if ledger else None,
                        actual_mma_ratio=ledger['ratio'] if ledger else None)
                    result['rows'].append(row)
                    write(folder/'result.json', result)
                result.update(complete=True, adapter=adapter, model_identity=runner.identity,
                    targets=runner.engine.targets, ar_cache=runner.ar_cache.receipt() if runner.ar_cache else None)
                write(folder/'result.json', result)
                state['completed'].append(entry['case']['id'])
                write(progress, state)
                print(json.dumps(dict(method=method, completed=len(state['completed']), total=len(entries),
                                      case=entry['case']['id'])), flush=True)
            if smoke:
                first = entries[0]
                clear_case(runner)
                runner.warmup(first['case'], study['steps'])
                _, row = runner.infer(first['case'], first['seeds'][0], study['steps'])
                receipt = json.loads((output/first['case']['id']/'result.json').read_text())
                expected = next(r['sha256'] for r in receipt['rows'] if r['seed'] == first['seeds'][0])
                if row['sha256'] != expected:
                    raise ValueError('A -> different scene -> A changed output')
                state['roundtrip_bitwise_equal'] = True
            state['complete'] = True
            write(progress, state)
    except Exception as error:
        state.update(paused=True, error=repr(error))
        write(progress, state)
        raise
    finally:
        if runner is not None:
            runner.engine.clear()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--method', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--cases', nargs='+')
    p.add_argument('--require-source-hash', action='store_true')
    a = p.parse_args()
    run(a.manifest, a.method, a.output, smoke=a.smoke, cases=a.cases, require_source_hash=a.require_source_hash)


if __name__ == '__main__':
    main()

"""Matched 1.5x / ARC100 quality supplement, retaining frozen baseline outputs."""
import argparse
import copy
import gc
import json
from pathlib import Path

from baselines import sha, write


def prepare(a):
    sources = {key: a.previous/key for key in ('cross', 'external', 'awq_native', 'outlier_development')}
    studies = {key: json.loads((path/'manifest.json').read_text()) for key, path in sources.items()}
    old = studies['cross']
    if any(v['model'] != old['model'] or v['cases'] != old['cases'] for v in studies.values()):
        raise ValueError('Unmatched frozen development cohorts')
    origins = {name: 'cross' for name in ('bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint')}
    origins.update(awq_marlin='awq_native', arcquant='external', sharq='external',
                   b2_5_placeholder='outlier_development')
    origins['b2.5_ffn_guarded'] = origins.pop('b2_5_placeholder')
    methods = {name: copy.deepcopy(studies[key]['methods'][name]) for name, key in origins.items()}
    methods['arcquant100'] = copy.deepcopy(methods['arcquant'])
    methods['arcquant100'].update(actual_generation_precision='ARCQuant-100%',
        scope='All activation residual channels; official native ARC kernels with BF16 attention.',
        limitations=['Eager host-alpha/default-stream API; fixed full-channel policy'])
    methods['sharq_perf'] = copy.deepcopy(methods['sharq'])
    methods['sharq_perf'].update(actual_generation_precision='SharQ-native-existing-epilogue-fusion',
        limitations=['Official extra_fusion=True; no Cosmos RMSNorm fusion port; eager/default-stream kernels'])
    value = dict(schema=1, frozen=True, model=old['model'], cases=old['cases'], methods=methods,
        steps=35, role='development', selection_eligible=False, performance_eligible=False,
        debug='off', resource_mode='shared-calibration', cohort_sha256=old['cohort_sha256'],
        imports={name: str(sources[key].resolve()) for name, key in origins.items()},
        parent_manifests={key: sha(path/'manifest.json') for key, path in sources.items()},
        external=studies['external']['external'],
        adapters=dict(arcquant=dict(method='arcquant', fraction=.5),
                      arcquant100=dict(method='arcquant', fraction=1.),
                      sharq=dict(method='sharq', extra_fusion=False),
                      sharq_perf=dict(method='sharq', extra_fusion=True)),
        scope='Same frozen sixteen development scenes and two seeds; 1.5x is protected_n, no refitting. '
              'New ARC100 and official SharQ epilogue performance variant receive full generation metrics. '
              'Historical outputs are hash-verified imports, never counted as new videos.',
        batch_protocol=dict(kind='independent_complete_request_concurrency', sizes=[1, 2, 4, 8],
            maximum_slots=8, rounds=5, waves_per_round=3, seed_base=301017,
            stop_rule='Two consecutive doublings with <5% throughput increase, or resource/correctness limit; observed plateau only',
            methods=[name for name in methods if name != 'sharq'],
            baseline_policy='Pinned existing kernels/flags, no baseline-specific tuning',
            scope='Cosmos upstream pipeline is single-sample; no merged-M full-pipeline batch claim. '
                  'Report full decoded videos/s, frames/s, completion latency and memory.'))
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'manifest.json', value)
    from importance_workflow import import_method
    for name, source in value['imports'].items():
        import_method(Path(source), a.output, name, a.output/'manifest.json')
    write(a.output/'quality_status.json', dict(complete=False, imported_methods=list(origins),
        new_methods=['arcquant100', 'sharq_perf'], performance_eligible=False))


def quality(a):
    manifest_path = a.output/'manifest.json'
    study = json.loads(manifest_path.read_text())
    if a.method not in ('arcquant100', 'sharq_perf'):
        raise ValueError('Only the two new variants generate fresh quality outputs')
    root = a.output/'generation'/a.method
    identity = dict(manifest_sha256=sha(manifest_path), method=a.method, source_sha256=sha(__file__),
                    runtime_sha256=sha(Path(__file__).with_name('baseline_runtime.py')))
    progress = root/'progress.json'
    if progress.exists() and json.loads(progress.read_text())['identity'] != identity:
        raise ValueError('Changed quality resume identity')
    state = dict(identity=identity, complete=False, completed=[], performance_eligible=False)
    write(progress, state)
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        resource = select_shared_device()
        import torch
        import numpy as np
        from baseline_runtime import make_runner
        from dataset_infer import clear_case
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        runner, adapter = make_runner(study, a.method)
        write(root/'prepared.json', dict(identity=identity, resource=resource, adapter=adapter,
                                        runtime=runner.receipt()))
        try:
            for entry in study['cases']:
                case = entry['case']
                if sha(case['image']) != entry['image_sha256']:
                    raise ValueError('Changed input image')
                folder = root/case['id']
                path = folder/'result.json'
                ci = dict(**identity, case=case, image_sha256=entry['image_sha256'], seeds=entry['seeds'])
                result = dict(identity=ci, complete=False, rows=[], performance_eligible=False, adapter=adapter)
                if path.exists():
                    result = json.loads(path.read_text())
                    if result['identity'] != ci:
                        raise ValueError('Changed case identity')
                    for row in result['rows']:
                        if sha(folder/row['outputs']) != row['outputs_sha256']:
                            raise ValueError('Changed resumed output')
                    if result['complete']:
                        state['completed'].append(case['id'])
                        continue
                folder.mkdir(parents=True, exist_ok=True)
                clear_case(runner)
                runner.warmup(case, 35)
                for seed in entry['seeds']:
                    if any(r['seed'] == seed for r in result['rows']):
                        continue
                    signals, row = runner.infer(case, seed, 35)
                    filename = f'seed{seed}.npz'
                    np.savez_compressed(folder/filename, **signals)
                    row.pop('compute_ledger')
                    row.update(outputs=filename, outputs_sha256=sha(folder/filename),
                               actual_mma_ratio=None, compute_sha256=None)
                    result['rows'].append(row)
                    write(path, result)
                result.update(complete=True, model_identity=runner.identity, targets=runner.engine.targets)
                write(path, result)
                state['completed'].append(case['id'])
                write(progress, state)
                print(json.dumps(dict(method=a.method, completed=len(state['completed']),
                                      total=len(study['cases']), case=case['id'])), flush=True)
            state['complete'] = True
            write(progress, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(progress, state)
            raise
        finally:
            del runner
            gc.collect()
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['prepare', 'quality'])
    p.add_argument('--previous', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method')
    a = p.parse_args()
    if a.operation == 'prepare' and a.previous is None:
        p.error('prepare requires --previous')
    (prepare if a.operation == 'prepare' else quality)(a)


if __name__ == '__main__':
    main()

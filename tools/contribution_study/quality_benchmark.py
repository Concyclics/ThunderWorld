"""Prepare and run the frozen two-model QUALITY matrix; no performance mode."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from baselines import sha, write
from full_benchmark_inventory import PRIMARY, ANCHORS, OURS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def quality_sources():
    files = [HERE/name for name in ('quality_benchmark.py','dataset_infer.py','dataset_suite.py',
        'baseline_runtime.py','external.py','awq_marlin.py','dataset_metrics.py','quality_benchmark_report.py',
        'dataset_report.py','awq_multiscene_report.py','arc_orders.py','portable.py','quality_shape_preflight.py',
        'importance_workflow.py')]
    files += list((REPO/'packages/fp4-repair/src/thunderworld_fp4').rglob('*.py'))
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(files)}


def immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Immutable preparation changed: '+str(path))
    write(path, value)


def copy_checked(source, target, digest):
    if sha(source) != digest:
        raise ValueError('Changed source: '+str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != digest:
            raise ValueError('Changed persisted asset: '+str(target))
    else:
        temp = target.with_name(target.name+'.partial')
        shutil.copyfile(source, temp)
        if sha(temp) != digest:
            raise ValueError('Asset copy checksum mismatch')
        temp.replace(target)


def persist_bank(source, target, digest):
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Incomplete offline asset bank')
    for entry in manifest['modules'].values():
        path = (source/entry['file']).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError('Asset path escapes bank')
        copy_checked(path, target/entry['file'], entry['sha256'])
    copy_checked(source/'manifest.json', target/'manifest.json', digest)


def prepare(a):
    inventory_file = REPO/'results/full_benchmark_inventory_20260919/inventory.json'
    inventory = json.loads(inventory_file.read_text())
    asset_root = (a.asset_root or a.root)/a.model
    cohort_path = a.workspace/'repair_redesign_20260917/cohort.json' if a.model == 'edge' else asset_root/'cohort.json'
    cohort = json.loads(cohort_path.read_text())
    root = a.root/a.model
    root.mkdir(parents=True, exist_ok=True)
    external = json.loads((REPO/'results/full_benchmark_inventory_20260919/edge_external.json').read_text())
    methods = {}; adapters = {}
    nano = None
    if a.model == 'nano':
        catalog = json.loads((asset_root/'ffn_guarded/candidates.json').read_text())
        if not catalog['complete']:
            raise ValueError('Incomplete Nano model-specific assets')
        nano = {c['name']: c for c in catalog['candidates']}
        external.update(arc_build=str(asset_root/'arc_build'), arc_build_sha256=sha(asset_root/'arc_build/build.json'),
                        capture=str(asset_root/'arc_orders'), capture_sha256=sha(asset_root/'arc_orders/manifest.json'))
    for name in (*PRIMARY, *ANCHORS):
        original = inventory['methods'][name]
        spec = copy.deepcopy(original['edge']['spec'])
        if a.model == 'nano':
            if name in OURS:
                spec = copy.deepcopy(nano[name])
                # Edge's ReLU2 norm preset is not silently applied to SwiGLU.
                spec['runtime_options'] = {}
            else:
                spec = dict(config=spec['config'])
            if name == 'awq_marlin' and a.awq_mode == 'marlin':
                bank = asset_root/'awq_native'
                spec['baseline_adapter'] = dict(method='marlin_awq', assets=str(bank),
                    assets_sha256=sha(bank/'manifest.json'), implementation_sha256=sha(HERE/'awq_marlin.py'))
        if spec.get('assets'):
            bank = asset_root/'deployment'/name/'assets'
            persist_bank(spec['assets'], bank, spec['assets_sha256'])
            spec['assets'] = str(bank)
        if spec['config'].get('budget_plan'):
            target = asset_root/'deployment'/name/'plan.json'
            copy_checked(Path(spec['config']['budget_plan']), target, spec['plan_sha256'])
            spec['config']['budget_plan'] = str(target)
        if name == 'awq_marlin' and a.awq_mode == 'reference':
            bank = a.workspace/'contribution_awq_20260917/assets' if a.model == 'edge' else asset_root/'awq_reference'
            spec = dict(config=spec['config'], baseline_adapter=dict(method='awq', assets=str(bank),
                assets_sha256=sha(bank/'manifest.json'), implementation_sha256=sha(HERE/'external.py')),
                scope='Official ModelOpt NVFP4 AWQ-lite exact decoded W4A16 quality reference. '
                      'Decoded BF16 storage/latency are ineligible as packed W4 deployment measurements.',
                performance_eligible=False, storage_comparison_eligible=False)
        if spec.get('baseline_adapter'):
            adapter = spec['baseline_adapter']
            bank = asset_root/'deployment'/('awq_reference' if name=='awq_marlin' and a.awq_mode=='reference' else name)/'assets'
            persist_bank(adapter['assets'], bank, adapter['assets_sha256'])
            adapter['assets'] = str(bank)
        if original.get('edge_external_adapter'):
            adapters[name] = original['edge_external_adapter']
        methods['awq_reference' if name=='awq_marlin' and a.awq_mode=='reference' else name] = spec
    common = dict(schema=1, frozen=True, model=cohort['model'], model_name=a.model, methods=methods,
        adapters=adapters, external=external, steps=35, debug='off', resource_mode='shared-calibration',
        performance_eligible=False, selection_eligible=False, cohort_sha256=sha(cohort_path),
        inventory_sha256=sha(inventory_file), statistics=dict(replicates=100000, bootstrap_seed=190919,
            family_size=12, correction='Bonferroni', primary_reference='arcquant100',
            family='3 budgets x 2 models x 2 datasets; separate FP8 gate family'),
        awq_mode=a.awq_mode,
        primary_comparisons=[dict(candidate=n, reference=r, inferential=r in ('fp8','arcquant100'))
                             for n in OURS for r in ('arcquant100','fp8',('awq_reference' if a.awq_mode=='reference' else 'awq_marlin'),'sharq_perf','fp4_base')],
        scope='Frozen 72-scene RBench/PAI I2V subset. BF16 consistency, not official leaderboard scores. '
              'Shared-GPU quality only. Model-specific calibration; final samples never used to fit assets.')
    dev = cohort['cases']['development']
    first = dev[0]
    second = next(e for e in dev if e['benchmark'] != first['benchmark'])
    for role, entries in [('smoke', [first, second]), ('development', dev), ('final', cohort['cases']['final'])]:
        value = dict(common, role=role, cases=copy.deepcopy(entries))
        # Smoke retains both paired seeds and uses only previously assigned dev inputs.
        if role == 'smoke' and a.model == 'edge':
            src = a.workspace/'fp4_next_20260918/optimized-study/manifest.json'
            old = json.loads(src.read_text())
            hashes = Path(old['source_output_hashes'])
            target = asset_root/'deployment/source_output_hashes.json'
            copy_checked(hashes, target, sha(hashes))
            value['source_output_hashes'] = str(target)
        immutable(root/role/'manifest.json', value)
    write(root/'prepared.json', dict(complete=True, model=a.model, methods=list(methods),
        final_manifest_sha256=sha(root/'final/manifest.json'), smoke_manifest_sha256=sha(root/'smoke/manifest.json'),
        quality_qualified=False, performance_eligible=False, final_started=False))
    print(json.dumps(dict(prepared=a.model, methods=len(methods), final_requests=144*len(methods))), flush=True)


def command(log, *args):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable, *map(str,args)], stdout=stream, stderr=subprocess.STDOUT, check=True)


def qualify(a):
    root = a.root/a.model
    manifest = root/'smoke/manifest.json'
    study = json.loads(manifest.read_text())
    path = root/'qualification.json'
    state = dict(complete=False, model=a.model, manifest_sha256=sha(manifest), methods={},
                 performance_eligible=False, scope='Two development scenes, two seeds, scene roundtrip; not a final quality claim')
    write(path, state)
    try:
        for name in study['methods']:
            output = root/'smoke/generation'/name
            command(root/'smoke'/f'{name}.log', HERE/'dataset_infer.py', manifest,
                    '--method', name, '--output', output, '--smoke')
            receipt = json.loads((output/'progress.json').read_text())
            if not receipt['complete'] or not receipt.get('roundtrip_bitwise_equal'):
                raise ValueError('Incomplete roundtrip qualification')
            # Deliberately exercise completed resume. The worker must validate
            # hashes and return before resource admission / model loading.
            command(root/'smoke'/f'{name}.log', HERE/'dataset_infer.py', manifest,
                    '--method', name, '--output', output, '--smoke')
            prepared = json.loads((output/'prepared.json').read_text())
            for file,digest in prepared['runtime']['source_sha256'].items():
                if sha(REPO/'packages/fp4-repair/src/thunderworld_fp4'/file) != digest:
                    raise ValueError('Runtime changed after generation; use a new qualification study')
            state['methods'][name] = dict(progress_sha256=sha(output/'progress.json'), completed_resume=True)
            write(path, state)
            print(json.dumps(dict(model=a.model, qualified=name, complete=len(state['methods']), total=len(study['methods']))), flush=True)
        command(root/'smoke/metrics.log', HERE/'dataset_metrics.py', manifest,
                '--generation', root/'smoke/generation', '--output', root/'smoke/metrics')
        metrics = json.loads((root/'smoke/metrics/progress.json').read_text())
        if not metrics['complete']:
            raise ValueError('Incomplete paired quality metrics')
        from quality_benchmark_report import report
        report(root/'smoke')
        from quality_shape_preflight import preflight
        preflight(root)
        state.update(complete=True, metrics_sha256=sha(root/'smoke/metrics/progress.json'),
                     final_manifest_sha256=sha(root/'final/manifest.json'), quality_sources=quality_sources(),
                     shape_preflight_sha256=sha(root/'shape_preflight.json'))
    except Exception as error:
        state.update(paused=True, error=repr(error))
        raise
    finally:
        write(path, state)


def execute(a):
    root = a.root/a.model
    qualification = json.loads((root/'qualification.json').read_text())
    manifest = root/'final/manifest.json'
    if not qualification['complete'] or qualification['final_manifest_sha256'] != sha(manifest):
        raise ValueError('Matching complete quality qualification required before final')
    if qualification.get('quality_sources') != quality_sources():
        raise ValueError('Quality implementation changed after qualification')
    if qualification.get('shape_preflight_sha256') != sha(root/'shape_preflight.json'):
        raise ValueError('Missing or changed final shape/budget preflight')
    from importance_workflow import run
    run(manifest, root/'final')
    from quality_benchmark_report import report
    report(root/'final')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare','qualify','run'))
    p.add_argument('--model', choices=('edge','nano'), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, help='Existing persistent calibration/deployment root')
    p.add_argument('--awq-mode', choices=('reference','marlin'), default='reference',
                   help='Quality defaults to exact official AWQ decoding; native Marlin needs separate scale-range qualification')
    p.add_argument('--workspace', type=Path, default=Path('/home/yufeng/thunderworld'))
    a = p.parse_args(); a.root = a.root.resolve()
    {'prepare':prepare,'qualify':qualify,'run':execute}[a.operation](a)


if __name__ == '__main__':
    main()

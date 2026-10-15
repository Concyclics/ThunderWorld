"""Portable orchestration of the existing calibration and quality workers.

Every GPU task is a separate process; this entry point never tunes on shared load.
Outputs are immutable or use the original content-checked resume protocol.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from common import ROOT, STUDY, immutable, read, sha, write

RELEASE_METHODS = {'b1.5_protected_n': '1.5x', 'b2_joint': '2x'}
DEFAULT_METHODS = ['bf16', 'fp8', 'fp4_base', *RELEASE_METHODS]


def execute(cmd, dry_run=False):
    cmd = list(map(str, cmd))
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def worker(model, script, *args):
    return [sys.executable, ROOT / 'tools/public/worker.py', '--model', model,
            '--script', script, '--', *args]


def reference_capture(capture, output):
    """A manifest-only view; no duplicate full-M panels or BF16 weights."""
    capture, output = Path(capture).resolve(), Path(output).resolve()
    spec = read(capture / 'manifest.json')
    if not spec.get('complete'):
        raise ValueError('Complete calibration capture required')
    rows, weights = [], {}
    def checked_path(item):
        file = capture / item['file']
        if sha(file) != item['sha256']:
            raise ValueError(f'Changed capture: {file}')
        return os.path.relpath(file, output)
    for name, entry in spec['weights'].items():
        weights[name] = checked_path(entry)
        own = [r for r in spec['rows'] if r['name'] == name]
        if len(own) != 16 or {r['scene_index'] for r in own} != set(range(16)):
            raise ValueError('Expected one full-M panel from each calibration scene')
        rows.extend(dict(r, kind='linear', file=checked_path(r)) for r in own)
    view = dict(complete=True, split='calibration', model_identity=spec['model_identity'],
        weights=weights, rows=rows, source_capture_sha256=sha(capture / 'manifest.json'),
        scope='16 calibration scenes, one phase/CFG per scene. Fresh calibration protocol, not the historical 16-panel AWQ experiment.')
    immutable(output / 'manifest.json', view)
    return output


def source_identity():
    roots = [ROOT / 'packages/fp4-repair/src', STUDY, ROOT / 'tools/public']
    return {str(f.relative_to(ROOT)): sha(f) for folder in roots for f in sorted(folder.rglob('*.py'))}


def check_sources(study):
    for name, digest in study['source_files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source changed after freezing: ' + name)


def released_candidate(model, profile, directory=None):
    from thunderworld_fp4.hub import resolve_release, registry
    root = resolve_release(model, profile, directory)
    # Reuse the release's checksum/identity validation, without loading CUDA.
    loader = importlib.util.spec_from_file_location('thunderworld_release_verifier', root / 'infer.py')
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    release, spec, cfg, _ = module.read_profile(profile, verify_weights=True)
    return dict(config=cfg, assets=str(root / spec['assets']),
        assets_sha256=spec['assets_manifest_sha256'], plan_sha256=spec['plan_sha256'],
        runtime_options=spec['runtime_options'], release_revision=registry()['models'][model]['revision'],
        release_manifest_sha256=sha(root / 'release.json'), execution_type='native_fp4_cosmos')


def prepare_manifest(args):
    from thunderworld_fp4.cosmos.settings import RunConfig, LinearSpec
    from thunderworld_fp4.config import RepairConfig, RepairPolicy
    cohort = read(args.work / 'cohort.json')
    if cohort['model_name'] != args.model or not cohort['frozen']:
        raise ValueError('Model/cohort mismatch')
    if args.split not in cohort['cases'] or not cohort['cases'][args.split]:
        raise ValueError('Missing selected cohort; use data.py --full for the full benchmark')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('Duplicate method name')
    supplied = {} if args.candidates is None else {c['name']: c for c in read(args.candidates)['candidates']}
    methods, adapters, external = {}, {}, {}
    for name in args.methods:
        if name in supplied:
            methods[name] = copy.deepcopy(supplied[name])
        elif name in RELEASE_METHODS:
            methods[name] = released_candidate(args.model, RELEASE_METHODS[name], args.release_dir)
        elif name in ('bf16', 'fp8', 'fp4_base'):
            fp4 = name == 'fp4_base'
            cfg = RunConfig(precision='fp4' if fp4 else name, resource_mode='shared-calibration',
                ffn=LinearSpec(0, 0), projection=LinearSpec(0, 0),
                attention_backend='fp4' if fp4 else 'bf16',
                attention=RepairPolicy(RepairConfig(coverage=0, terms=2, query_tile=32, transpose_v=True)))
            methods[name] = dict(config=cfg.to_dict(), assets=None,
                execution_type='native_w4a4' if fp4 else ('native_w8a8_online_per_token' if name == 'fp8' else 'bf16'))
        elif name in ('arcquant', 'arcquant100', 'sharq_perf', 'awq_reference'):
            cfg = RunConfig(precision='bf16', attention_backend='bf16', execution='eager',
                            resource_mode='shared-calibration')
            spec = dict(config=cfg.to_dict(), assets=None, execution_type='official_native_cosmos_adapter')
            if name == 'awq_reference':
                bank = args.work / 'awq'
                if not read(bank / 'manifest.json')['complete']:
                    raise ValueError('Complete ModelOpt AWQ assets required')
                spec.update(execution_type='offline_decoded_quality_reference', baseline_adapter=dict(
                    method='awq', assets=str(bank), assets_sha256=sha(bank / 'manifest.json'),
                    implementation_sha256=sha(STUDY / 'external.py')))
            else:
                lock = args.work / 'baselines/sources.lock.json'
                external.update(sources=str(lock), sources_sha256=sha(lock))
                if name.startswith('arcquant'):
                    build = args.work / 'arc-build'; capture = args.work / 'reference-capture'
                    external.update(arc_build=str(build), arc_build_sha256=sha(build / 'build.json'),
                        capture=str(capture), capture_sha256=sha(capture / 'manifest.json'))
                    adapters[name] = dict(method='arcquant', fraction=1. if name == 'arcquant100' else .5)
                else:
                    build = Path(read(lock)['sources']['sharq']['path']) / 'kernels/build_cmake_sm120a'
                    external.update(sharq_build=str(build), sharq_build_sha256=sha(build / 'contribution-build.json'))
                    adapters[name] = dict(method='sharq', extra_fusion=True)
            methods[name] = spec
        else:
            raise ValueError('Unknown method or missing --candidates: ' + name)
    if 'bf16' not in methods or 'fp8' not in methods:
        raise ValueError('Include bf16 and fp8 for paired reporting')
    from thunderworld_fp4.hub import registry
    base = registry()['models'][args.model]['base_required']
    for filename, digest in base['identity'].items():
        if sha(Path(cohort['model']) / filename) != digest:
            raise ValueError('Wrong base-model revision: ' + filename)
    study = dict(schema_version=1, frozen=True, model=cohort['model'], model_name=args.model,
        steps=35, methods=methods, cases=cohort['cases'][args.split], adapters=adapters, external=external,
        cohort_sha256=sha(args.work / 'cohort.json'), split=args.split, source_files=source_identity(),
        statistics=dict(replicates=10000, bootstrap_seed=170917, family_size=max(1, 2*(len(methods)-2))),
        performance_eligible=False, scope='Same-input/seed consistency to BF16 at 320x192, 61 frames, 35 steps, CFG 6. '
        'Fresh public reproduction; full-corpus evaluation overlaps calibration/development. No latency or task-success claim.')
    from baseline_runtime import verify
    for name in methods:
        RunConfig.from_dict(methods[name]['config']); verify(study, name)
    immutable(args.manifest, study)
    print(json.dumps(dict(manifest=str(args.manifest), methods=list(methods), scenes=len(study['cases']),
                         videos=sum(len(e['seeds']) for e in study['cases']) * len(methods)), indent=2))


def export_videos(manifest, generation):
    import numpy as np
    from diffusers.utils import export_to_video
    study = read(manifest)
    index = []
    for name in study['methods']:
        for entry in study['cases']:
            folder = generation / name / entry['case']['id']
            receipt = read(folder / 'result.json')
            if not receipt['complete']:
                raise ValueError('Incomplete generation')
            for row in receipt['rows']:
                source = folder / row['outputs']
                if sha(source) != row['outputs_sha256']:
                    raise ValueError('Changed NPZ')
                dest = source.with_suffix('.mp4')
                if not dest.exists():
                    with np.load(source, allow_pickle=False) as data:
                        video = data['video']
                        if video.ndim == 5 and video.shape[0] == 1: video = video[0]
                        if video.ndim != 4: raise ValueError('Expected one video')
                        export_to_video(video, str(dest), fps=24)
                index.append(dict(method=name, case=entry['case']['id'], seed=row['seed'],
                    video=str(dest.relative_to(generation)), video_sha256=sha(dest), npz_sha256=sha(source)))
    write(generation / 'videos.json', dict(manifest_sha256=sha(manifest), videos=index,
        scope='MP4 for viewing/judge only; pixel metrics use original NPZ float outputs.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('calibrate', 'capture', 'plan', 'joint', 'reference-capture',
        'collect-baselines', 'build-baselines', 'awq', 'prepare', 'generate', 'metrics', 'report', 'videos'))
    p.add_argument('--model', choices=('edge', 'nano', 'super'), default='edge')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--evaluation-dir', type=Path, help='Separate generation/metrics/report outputs for a different frozen manifest')
    p.add_argument('--split', choices=('calibration', 'development', 'validation', 'final', 'full'), default='final')
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    p.add_argument('--candidates', type=Path)
    p.add_argument('--release-dir', type=Path)
    p.add_argument('--budgets', type=float, nargs='+', choices=(1.25, 1.5, 2., 2.5), default=[1.25, 1.5, 2., 2.5])
    p.add_argument('--names', nargs='+', default=['arcquant', 'sharq', 'modelopt'])
    p.add_argument('--dry-run', action='store_true', help='Print expensive worker/build commands without launching')
    a = p.parse_args(); a.work = a.work.resolve()
    a.manifest = a.manifest.resolve() if a.manifest else a.work / 'evaluation.json'
    evaluation = a.evaluation_dir.resolve() if a.evaluation_dir else a.work
    cohort = a.work / 'cohort.json'
    def run(script, *rest, cpu=False):
        execute(worker('edge' if cpu else a.model, script, *rest), a.dry_run)
    if a.operation == 'calibrate':
        run('redesign_calibrate.py', cohort, '--output', a.work / 'calibration')
    elif a.operation == 'capture':
        run('importance_capture.py', '--cohort', cohort, '--output', a.work / 'capture')
    elif a.operation == 'plan':
        run('redesign_plan.py', '--cohort', cohort, '--calibration', a.work / 'calibration',
            '--output', a.work / 'controls', '--budgets', *a.budgets, '--skip-legacy-control', cpu=True)
    elif a.operation == 'joint':
        run('importance_priors.py', '--previous', a.work, '--output', a.work / 'priors', cpu=True)
        run('importance_joint_calibrate.py', '--study', a.work, '--previous', a.work,
            '--controls', a.work / 'controls/candidates.json', '--budgets', *a.budgets)
    elif a.operation == 'reference-capture':
        if a.dry_run: p.error('reference-capture is a CPU operation; omit --dry-run')
        reference_capture(a.work / 'capture', a.work / 'reference-capture')
    elif a.operation == 'collect-baselines':
        run('run.py', 'collect-baselines', '--output', a.work / 'baselines', '--names', *a.names,
            '--lock', ROOT / 'configs/public/baselines.lock.json', cpu=True)
    elif a.operation == 'build-baselines':
        import pybind11
        lock = a.work / 'baselines/sources.lock.json'; sources = read(lock)['sources']
        for name in a.names:
            if name not in ('arcquant', 'sharq'): continue
            execute(['git', '-C', sources[name]['path'], 'submodule', 'update', '--init', '--depth', '1',
                     'third-party/cutlass'], a.dry_run)
            common = ['--sources', lock, '--pybind11', pybind11.get_cmake_dir()]
            if name == 'arcquant':
                widths = sorted({r['shape'][2] for r in read(a.work / 'capture/manifest.json')['rows']})
                run('build_arcquant.py', *common, '--output', a.work / 'arc-build', '--widths', *widths, cpu=True)
                if max(widths) > 16384:
                    run('super_arc_build.py', '--build', a.work / 'arc-build', cpu=True)
            else: run('build_sharq.py', *common, cpu=True)
    elif a.operation == 'awq':
        run('awq_prepare.py', '--capture', a.work / 'reference-capture', '--sources',
            a.work / 'baselines/sources.lock.json', '--output', a.work / 'awq', '--resume', '--require-pinned-source')
    elif a.operation == 'prepare':
        if a.dry_run: p.error('prepare writes a frozen CPU manifest; omit --dry-run')
        prepare_manifest(a)
    elif a.operation == 'generate':
        study = read(a.manifest); check_sources(study)
        if study['model_name'] != a.model: raise ValueError('Model flag differs from manifest')
        for name in study['methods']:
            run('dataset_infer.py', a.manifest, '--method', name, '--output', evaluation / 'generation' / name)
    elif a.operation == 'metrics':
        run('dataset_metrics.py', a.manifest, '--generation', evaluation / 'generation', '--output', evaluation / 'metrics', cpu=True)
    elif a.operation == 'report':
        run('dataset_report.py', a.manifest, '--generation', evaluation / 'generation', '--metrics',
            evaluation / 'metrics', '--output', evaluation / 'report', cpu=True)
    elif a.operation == 'videos':
        if a.dry_run: p.error('videos is an export operation; omit --dry-run')
        export_videos(a.manifest, evaluation / 'generation')


if __name__ == '__main__':
    main()

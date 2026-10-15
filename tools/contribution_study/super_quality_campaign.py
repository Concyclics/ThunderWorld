"""Resumable full Super quality extension; never modifies Edge/Nano evidence.

prepare freezes the same 1,694 scenes and paired seeds. run verifies the official
checkpoint, independently calibrates Super using only the old calibration split,
qualifies all nine methods, checks every request budget, then generates/exports
the full corpus. Resource or numerical failures pause with a durable receipt.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from baselines import sha, write
from quality_benchmark import immutable, quality_sources

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
METHODS = ('bf16', 'fp8', 'awq_reference', 'fp4_base', 'arcquant', 'arcquant100',
           'sharq_perf', 'b1.5_protected_n', 'b2_joint')
EXPECTED_VIDEOS = 1694 * 2 * len(METHODS)
REVISION = 'f543c56225b2e04d0ad141e29655be3a45d9c455'


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    names = ('super_runtime.py', 'super_quality_campaign.py', 'super_quality_metrics.py', 'super_task_judge.py',
        'test_super_runtime.py', 'importance_capture.py', 'redesign_calibrate.py',
        'redesign_ffn_calibrate.py', 'redesign_plan.py', 'importance_priors.py',
        'importance_joint_calibrate.py', 'importance_ffn_guard.py', 'nano_quality_prepare.py',
        'build_arcquant.py', 'super_arc_build.py', 'super_arc_codec.py', 'arc_probe.py', 'external_probe.py', 'awq_prepare.py',
        'judge_video_export.py', 'full_video_metrics.py')
    return dict(quality_sources(), **{str((HERE / n).relative_to(REPO)): sha(HERE / n) for n in names})


def full_cohort(path):
    from judge_baseline_extension import validate_cases
    value = json.loads(path.read_text())
    validate_cases(value)
    if not value.get('frozen') or not value.get('complete'):
        raise ValueError('Frozen complete full corpus required')
    return value


def prepare(a):
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = json.loads((root / 'model.lock.json').read_text())
    if lock['repo'] != 'nvidia/Cosmos3-Super' or lock['revision'] != REVISION:
        raise ValueError('Wrong Super checkpoint')
    dependencies = json.loads((root / 'dependencies_verified.json').read_text())
    if not dependencies['complete']:
        raise ValueError('Qualify persistent dependencies before preparing the campaign')
    if shutil.disk_usage(root).free < 1200 * 2**30:
        raise RuntimeError('At least 1,200 GiB free space required for the new model, calibration and full outputs')
    original = WORKSPACE / 'full_i2v_data_20260919/cohort.json'
    full = full_cohort(original)
    calibration_path = WORKSPACE / 'full_quality_ready_20260919/nano/cohort.json'
    calibration = json.loads(calibration_path.read_text())
    calibration['model'] = lock['model']
    calibration['model_name'] = 'super'
    calibration['source_cohort_sha256'] = sha(calibration_path)
    calibration['scope'] = ('Same frozen scene/seed split as Edge/Nano; all weights, '
        'teacher panels, channel orders, repair budgets and AWQ assets recalibrated for Super. '
        'Full-corpus aggregates include previously designated calibration/development scenes; '
        'held-out split is identified separately, not a claim of new-scene generalization.')
    immutable(root / 'super/cohort.json', calibration)
    immutable(root / 'full_cohort.json', full)
    counts = {b: sum(len(e['seeds']) for e in full['cases'] if e['benchmark'] == b) * len(METHODS)
              for b in ('rbench', 'paibench')}
    manifest = dict(schema=1, kind='super_full_quality_extension', frozen=True,
        model=lock['model'], model_revision=REVISION, model_lock_sha256=sha(root / 'model.lock.json'),
        methods=list(METHODS), expected_videos=EXPECTED_VIDEOS, expected_by_benchmark=counts,
        full_cohort_sha256=sha(root / 'full_cohort.json'),
        dependencies_sha256=sha(root / 'dependencies_verified.json'),
        calibration_cohort_sha256=sha(root / 'super/cohort.json'), source_sha256=sources(),
        gpus=[0, 2], workers=1, placement='AR on physical GPU2, DM on GPU0',
        protocol=full['protocol'], chunk_scenes=12, performance_eligible=False,
        metrics=dict(reference='Super BF16 same scene/seed', names=['lpips', 'ssim_mean', 'psnr_db', 'relative_l2'],
            all_frames=True, resize=False, task_judge='same frozen Qwen3-VL-8B-Instruct-FP8 protocol'),
        generation_sources=[str(WORKSPACE / 'judge_full_i2v_20260919'), str(WORKSPACE / 'judge_full_baselines_20260922')],
        outputs_reused_from_other_models=False,
        scope=calibration['scope'])
    immutable(root / 'campaign.json', manifest)
    from super_task_judge import freeze, DEFAULT_PREVIOUS
    freeze(root / 'judge', DEFAULT_PREVIOUS)
    if not (root / 'progress.json').exists():
        write(root / 'progress.json', dict(complete=False, phase='prepared', stages={},
            expected_videos=EXPECTED_VIDEOS, generated_videos=0, performance_eligible=False))
    print(json.dumps(dict(prepared=True, expected_videos=EXPECTED_VIDEOS, by_benchmark=counts)))


def check_campaign(root):
    campaign = json.loads((root / 'campaign.json').read_text())
    if campaign['source_sha256'] != sources():
        raise ValueError('Frozen Super implementation changed; explicit requalification required')
    for file, key in [('model.lock.json', 'model_lock_sha256'), ('full_cohort.json', 'full_cohort_sha256'),
                      ('super/cohort.json', 'calibration_cohort_sha256'), ('dependencies_verified.json', 'dependencies_sha256')]:
        if sha(root / file) != campaign[key]:
            raise ValueError('Changed Super campaign input: ' + file)
    return campaign


def stream_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(root):
    lock = json.loads((root / 'model.lock.json').read_text())
    model = Path(lock['model'])
    path = root / 'model_verified.json'
    result = json.loads(path.read_text()) if path.exists() else dict(complete=False, files={})
    for row in lock['files']:
        file = model / row['path']
        if not file.exists() or file.stat().st_size != row['bytes']:
            raise FileNotFoundError('Incomplete model download: ' + str(file))
        old = result['files'].get(row['path'])
        if old and old['mtime_ns'] == file.stat().st_mtime_ns and old['bytes'] == file.stat().st_size:
            continue
        digest = stream_hash(file)
        if row['sha256'] and digest != row['sha256']:
            raise ValueError('Official weight checksum mismatch: ' + str(file))
        result['files'][row['path']] = dict(sha256=digest, bytes=file.stat().st_size, mtime_ns=file.stat().st_mtime_ns)
        write(path, result)
    result.update(complete=True, model_lock_sha256=sha(root / 'model.lock.json'))
    write(path, result)


def build_studies(root, *, bootstrap=False):
    asset_root = root / 'super'
    cohort = json.loads((asset_root / 'cohort.json').read_text())
    corpus = json.loads((root / 'full_cohort.json').read_text())
    inventory_path = REPO / 'results/full_benchmark_inventory_20260919/inventory.json'
    inventory = json.loads(inventory_path.read_text())['methods']
    external = json.loads((REPO / 'results/full_benchmark_inventory_20260919/edge_external.json').read_text())
    methods = {}
    adapters = {}
    if not bootstrap:
        catalog = json.loads((asset_root / 'ffn_guarded/candidates.json').read_text())
        if not catalog['complete']:
            raise ValueError('Incomplete independent Super repair calibration')
        candidates = {c['name']: c for c in catalog['candidates']}
        external.update(arc_build=str(asset_root / 'arc_build'), arc_build_sha256=sha(asset_root / 'arc_build/build.json'),
            capture=str(asset_root / 'arc_orders'), capture_sha256=sha(asset_root / 'arc_orders/manifest.json'))
    for name in (('bf16',) if bootstrap else METHODS):
        if name in ('b1.5_protected_n', 'b2_joint'):
            spec = copy.deepcopy(candidates[name])
            spec['runtime_options'] = {}
        else:
            source = 'awq_marlin' if name == 'awq_reference' else name
            spec = dict(config=copy.deepcopy(inventory[source]['edge']['spec']['config']))
            if name == 'awq_reference':
                bank = asset_root / 'awq_reference'
                spec.update(baseline_adapter=dict(method='awq', assets=str(bank),
                    assets_sha256=sha(bank / 'manifest.json'), implementation_sha256=sha(HERE / 'external.py')),
                    performance_eligible=False, storage_comparison_eligible=False,
                    scope='Official ModelOpt NVFP4 AWQ-lite decoded W4A16 quality reference; no native speed/storage claim')
            if inventory[source].get('edge_external_adapter'):
                adapters[name] = inventory[source]['edge_external_adapter']
        methods[name] = spec
    encoding = dict(fps=24, ffmpeg='/usr/bin/ffmpeg', ffprobe='/usr/bin/ffprobe',
        mp4=dict(codec='libx264', pixel_format='yuv420p', crf=15, preset='medium'),
        lossless=dict(codec='ffv1', pixel_format='bgr0', level=3, verify_rgb_roundtrip=True),
        float_to_rgb='round(clip(video,0,1)*255) to uint8; original float32 NPZ retained')
    common = dict(schema=1, frozen=True, model=cohort['model'], model_name='super', methods=methods,
        adapters=adapters, external=external, steps=35, debug='off', resource_mode='shared-calibration',
        performance_eligible=False, selection_eligible=False, protocol=corpus['protocol'],
        cohort_sha256=sha(root / 'full_cohort.json'), calibration_sha256=sha(asset_root / 'cohort.json'),
        source_sha256=sources(), super_runtime_sha256=sha(HERE / 'super_runtime.py'),
        inventory_sha256=sha(inventory_path), primary_comparisons=[], video_export=encoding,
        scope=cohort['scope'], source_output_hashes=None,
        input_copies={e['case']['id']: e['case']['image'] for e in corpus['cases']},
        gpu_policy=dict(allowed=[0, 2], placement='AR/DM branch split', workers=1))
    dev = cohort['cases']['development']
    smoke = [dev[0], next(e for e in dev if e['benchmark'] != dev[0]['benchmark'])]
    if bootstrap:
        immutable(asset_root / 'bootstrap/manifest.json', dict(common, role='development_bootstrap', cases=smoke))
    else:
        immutable(asset_root / 'smoke/manifest.json', dict(common, role='development_qualification', cases=smoke))
        immutable(asset_root / 'final/manifest.json', dict(common, role='full_I2V_corpus', cases=corpus['cases'],
            prior_scene_role_counts=corpus.get('prior_scene_role_counts')))


def run(a):
    root = a.root.resolve()
    campaign = check_campaign(root)
    lock_file = (root / 'campaign.lock').open('a+')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / 'progress.json'
    state = json.loads(state_path.read_text())
    if state.get('complete'):
        return
    state.pop('paused', None)
    state.pop('error', None)
    state['started_or_resumed_utc'] = now()
    asset_root = root / 'super'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,2', TW_SHARED_CALIBRATION_GPUS='0,2',
        OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', VLLM_PLUGINS='')

    def command(name, script, args=(), split=True):
        check_campaign(root)
        state.update(phase=name, updated_utc=now())
        write(state_path, state)
        full_model = script in ('dataset_infer.py', 'importance_capture.py', 'redesign_calibrate.py', 'redesign_ffn_calibrate.py')
        operator_only = script in ('importance_joint_calibrate.py', 'importance_ffn_guard.py',
            'arc_probe.py', 'super_arc_codec.py', 'external_probe.py', 'awq_prepare.py')
        needs_pair = full_model or script in ('super_quality_metrics.py', 'super_task_judge.py')
        launch_env = dict(env)
        if full_model or operator_only or needs_pair:
            from thunderworld_fp4.cosmos.resources import free_memory
            while True:
                free = free_memory()
                required = (88*1024, 72*1024) if full_model else (
                    (64*1024, 64*1024) if script == 'super_task_judge.py' else (12288, 12288))
                eligible = [d for d in (0, 2) if free.get(d, 0) >= 12288]
                ready = all(free.get(d, 0) >= n for d, n in zip((0, 2), required)) if needs_pair else bool(eligible)
                if ready:
                    if operator_only:
                        chosen = max(eligible, key=lambda d: free[d])
                        launch_env.update(CUDA_VISIBLE_DEVICES=str(chosen), TW_SHARED_CALIBRATION_LOCK_SCOPE='device')
                    break
                state.update(phase='waiting_resources', pending_stage=name, free_mib=free, updated_utc=now())
                write(state_path, state)
                time.sleep(30)
            state.update(phase=name, updated_utc=now())
            state.pop('pending_stage', None)
            write(state_path, state)
        cmd = [sys.executable]
        if full_model and split:
            cmd += [str(HERE / 'super_runtime.py'), 'run']
        cmd += [str(HERE / script), *map(str, args)]
        with (root / (name + '.log')).open('ab') as log:
            log.write((json.dumps(dict(command=cmd, utc=now())) + '\n').encode())
            log.flush()
            subprocess.run(cmd, env=launch_env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def stage(name, receipt, script, *args, split=True):
        if receipt.exists() and json.loads(receipt.read_text()).get('complete'):
            if name in state['stages'] and state['stages'][name] != sha(receipt):
                raise ValueError('Changed completed preparation stage: ' + name)
        else:
            command(name, script, args, split=split)
        if not json.loads(receipt.read_text()).get('complete'):
            raise ValueError('Incomplete stage: ' + name)
        state['stages'][name] = sha(receipt)
        write(state_path, state)

    export_process = None
    try:
        state['phase'] = 'waiting_model_download'
        write(state_path, state)
        deadline = time.monotonic() + 24 * 3600
        while True:
            lock = json.loads((root / 'model.lock.json').read_text())
            missing = [r['path'] for r in lock['files'] if not (Path(lock['model']) / r['path']).exists()
                       or (Path(lock['model']) / r['path']).stat().st_size != r['bytes']]
            if not missing:
                break
            state.update(download_missing_files=len(missing), updated_utc=now())
            write(state_path, state)
            if time.monotonic() > deadline:
                raise TimeoutError('Model download did not finish within 24 hours; resume after checking download.log')
            time.sleep(30)
        state['phase'] = 'verifying_official_weight_checksums'
        write(state_path, state)
        verify_model(root)
        build_studies(root, bootstrap=True)
        stage('bf16_bootstrap', asset_root / 'bootstrap/generation/bf16/progress.json', 'dataset_infer.py',
            asset_root / 'bootstrap/manifest.json', '--method', 'bf16', '--output', asset_root / 'bootstrap/generation/bf16', '--smoke')
        stage('capture', asset_root / 'capture/manifest.json', 'importance_capture.py',
            '--cohort', asset_root / 'cohort.json', '--output', asset_root / 'capture')
        stage('calibration', asset_root / 'calibration/manifest.json', 'redesign_calibrate.py',
            asset_root / 'cohort.json', '--output', asset_root / 'calibration')
        stage('ffn_calibration', asset_root / 'ffn_calibration/manifest.json', 'redesign_ffn_calibrate.py',
            '--cohort', asset_root / 'cohort.json', '--assets', asset_root / 'calibration/rtn_assets', '--output', asset_root / 'ffn_calibration')
        template = asset_root / 'plan_source.json'
        immutable(template, dict(methods={}, scope='No imported model-specific quotas'))
        common = ['--cohort', asset_root / 'cohort.json', '--calibration', asset_root / 'calibration',
                  '--previous', template, '--skip-legacy-control']
        stage('protected_plan', asset_root / 'protected_candidates/candidates.json', 'redesign_plan.py',
            *common, '--output', asset_root / 'protected_candidates', '--budgets', 1.5, 2.)
        stage('ffn_plan', asset_root / 'ffn_candidates/candidates.json', 'redesign_plan.py',
            *common, '--output', asset_root / 'ffn_candidates', '--ffn-calibration', asset_root / 'ffn_calibration', '--budgets', 2.5)
        if not (asset_root / 'joint_calibration/progress.json').exists():
            immutable(asset_root / 'candidates/candidates.json', json.loads((asset_root / 'protected_candidates/candidates.json').read_text()))
        ffn = json.loads((asset_root / 'ffn_candidates/candidates.json').read_text())
        for c in ffn['candidates']:
            if c['name'] == 'b2.5_ffn_aware_n':
                c['name'] = 'b2.5_ffn_aware'
        immutable(asset_root / 'ffn_followup/candidates/candidates.json', ffn)
        stage('priors', asset_root / 'priors/manifest.json', 'importance_priors.py', '--previous', asset_root, '--output', asset_root / 'priors')
        stage('joint', asset_root / 'joint_calibration/progress.json', 'importance_joint_calibrate.py', '--study', asset_root, '--previous', asset_root)
        stage('ffn_guard', asset_root / 'ffn_guarded/progress.json', 'importance_ffn_guard.py',
            '--study', asset_root, '--previous', asset_root, '--after', asset_root / 'joint_calibration/progress.json')
        from nano_quality_prepare import legacy_panels, arc_orders
        view = legacy_panels(asset_root)
        if not (asset_root / 'arc_orders/manifest.json').exists():
            arc_orders(asset_root)
        build = json.loads((asset_root / 'arc_build/build.json').read_text())
        if not build['complete'] or build['requested_widths'] != [5120, 8192, 25600]:
            raise ValueError('Complete Super-specific ARC build required')
        stage('arc_codec_probe', asset_root / 'arc_codec_probe.json', 'super_arc_codec.py',
            '--build', asset_root / 'arc_build', '--output', asset_root / 'arc_codec_probe.json')
        if json.loads((asset_root / 'arc_codec_probe.json').read_text())['build_sha256'] != sha(asset_root / 'arc_build/build.json'):
            raise ValueError('ARC codec qualification belongs to a different build')
        sanitizer = json.loads((asset_root / 'arc_sanitizers.json').read_text())
        if not sanitizer['complete'] or sanitizer['build_sha256'] != sha(asset_root / 'arc_build/build.json'):
            raise ValueError('The wide-K ARC port must pass memcheck and racecheck first')
        shapes = []
        for m, n, k in [(129, 1024, 5120), (416, 25600, 5120), (960, 5120, 25600), (416, 5120, 8192)]:
            shapes.extend(['--shape', m, n, k])
        stage('arc_probe', asset_root / 'arc_probe.json', 'arc_probe.py',
            '--build', asset_root / 'arc_build', '--output', asset_root / 'arc_probe.json', *shapes)
        sources_path = WORKSPACE / 'contribution_baselines_20260917/sources.lock.json'
        stage('sharq_probe', asset_root / 'sharq_probe.json', 'external_probe.py', '--all-shapes',
            '--capture', view, '--sources', sources_path, '--output', asset_root / 'sharq_probe.json')
        stage('awq_reference', asset_root / 'awq_reference/manifest.json', 'awq_prepare.py',
            '--capture', view, '--sources', sources_path, '--output', asset_root / 'awq_reference', '--resume')
        build_studies(root)
        for name in METHODS:
            out = asset_root / 'smoke/generation' / name
            stage('qualify_' + name, out / 'progress.json', 'dataset_infer.py',
                asset_root / 'smoke/manifest.json', '--method', name, '--output', out, '--smoke')
            if not json.loads((out / 'progress.json').read_text()).get('roundtrip_bitwise_equal'):
                raise ValueError('Missing A-B-A qualification: ' + name)
        command('all_request_shape_budget_preflight', 'quality_shape_preflight.py', [asset_root], split=False)
        write(asset_root / 'qualification.json', dict(complete=True,
            final_manifest_sha256=sha(asset_root / 'final/manifest.json'), source_sha256=sources(),
            shape_preflight_sha256=sha(asset_root / 'shape_preflight.json'),
            methods={n: sha(asset_root / 'smoke/generation' / n / 'progress.json') for n in METHODS},
            performance_eligible=False))
        corpus = json.loads((asset_root / 'final/manifest.json').read_text())
        queue_path = root / 'generation_queue.json'
        tasks = [dict(id=f'{b}__{start:04d}__{name}', method=name,
                      cases=[e['case']['id'] for e in corpus['cases'] if e['benchmark'] == b][start:start+12])
                 for b, count in [('rbench', 650), ('paibench', 1044)]
                 for start in range(0, count, 12) for name in METHODS]
        immutable(root / 'generation_tasks.json', dict(tasks=tasks, manifest_sha256=sha(asset_root / 'final/manifest.json')))
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else dict(complete=False, completed={})
        write(asset_root / 'workflow.json', dict(paused=False, generation_complete=False))
        with (root / 'video_export.log').open('ab') as log:
            export_process = subprocess.Popen([sys.executable, str(HERE / 'judge_video_export.py'),
                '--root', str(asset_root), '--follow'], env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
        for task in tasks:
            tid, name = task['id'], task['method']
            out = root / 'tasks' / tid / 'generation'
            if tid in queue['completed']:
                if sha(out / 'progress.json') != queue['completed'][tid]:
                    raise ValueError('Changed completed generation task: ' + tid)
                continue
            if export_process.poll() is not None and export_process.returncode != 0:
                raise RuntimeError('Video exporter failed; see video_export.log')
            if shutil.disk_usage(root).free < 150 * 2**30:
                raise RuntimeError('PAUSED: less than 150 GiB free disk; outputs retained')
            for cid in task['cases']:
                folder = out / cid
                folder.mkdir(parents=True, exist_ok=True)
                canonical = asset_root / 'final/generation' / name / cid
                canonical.parent.mkdir(parents=True, exist_ok=True)
                if not canonical.exists():
                    canonical.symlink_to(folder, target_is_directory=True)
                if canonical.resolve() != folder.resolve():
                    raise ValueError('Changed generation alias')
                log = folder / 'generation.log'
                if not log.exists():
                    log.symlink_to(root / ('generate_' + tid + '.log'))
            stage('generate_' + tid, out / 'progress.json', 'dataset_infer.py',
                asset_root / 'final/manifest.json', '--method', name, '--output', out, '--cases', *task['cases'])
            queue['completed'][tid] = sha(out / 'progress.json')
            write(queue_path, queue)
            state['generated_videos'] = sum(2*len(t['cases']) for t in tasks if t['id'] in queue['completed'])
            write(state_path, state)
        queue['complete'] = True
        write(queue_path, queue)
        write(asset_root / 'workflow.json', dict(paused=False, generation_complete=True))
        if export_process.wait() != 0:
            raise RuntimeError('Video exporter failed')
        if not json.loads((asset_root / 'export_progress.json').read_text())['complete']:
            raise RuntimeError('Missing exported videos')
        stage('full_pixel_metrics', root / 'metrics/progress.json', 'super_quality_metrics.py', '--root', root, split=False)
        stage('full_task_judge', root / 'judge/report/complete.json', 'super_task_judge.py',
            'run', '--campaign', root, split=False)
        state.update(complete=True, phase='generation_metrics_and_task_judge_complete', completed_utc=now(),
            task_judge_complete=True)
        write(state_path, state)
    except BaseException as error:
        state.update(complete=False, paused=True, error=repr(error), updated_utc=now())
        write(state_path, state)
        write(asset_root / 'workflow.json', dict(paused=True, error=repr(error)))
        raise
    finally:
        if export_process is not None and export_process.poll() is None:
            export_process.terminate()
            export_process.wait()
        lock_file.close()


def status(a):
    print((a.root / 'progress.json').read_text())
    for sub in ('capture', 'calibration', 'ffn_calibration'):
        p = a.root / 'super' / sub / 'manifest.json'
        if p.exists():
            x = json.loads(p.read_text())
            print(json.dumps(dict(stage=sub, complete=x['complete'], cases=len(x.get('cases', {})), paused=x.get('paused'))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'run', 'status', 'verify-model'))
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    {'prepare': prepare, 'run': run, 'status': status, 'verify-model': lambda a: verify_model(a.root)}[a.operation](a)


if __name__ == '__main__':
    main()

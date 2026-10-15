"""Add the four frozen quality baselines to an existing full-corpus judge run.

This entry point configures the existing queue, inherited GPU leases, numerical
worker and video exporter. It does not modify those frozen implementations or
the original run. The configuration adapter is installed in every child process
so all prepare, resume, qualification and source checks use this extension's
manifest. No numerical execution options are overridden.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import statistics

from baselines import sha, write
from quality_benchmark import immutable, quality_sources
import judge_generation as frozen
import judge_concurrent_infer as runtime
import judge_parallel_queue as pool

HERE = Path(__file__).resolve()
METHODS = ('bf16', 'fp8', 'awq_reference', 'fp4_base')
BASE_SOURCES = frozen.sources
BASE_VERIFY = frozen.verify_job
EXPECTED_PROTOCOL = dict(width=320, height=192, frames=61, fps=24, steps=35,
    guidance_scale=6.0, flow_shift=3.0, scheduler='HostScheduleUniPC',
    seeds_per_scene=2, add_resolution_template=False, add_duration_template=False,
    enable_sound=False)


def sources():
    paths = [HERE, HERE.with_name('judge_parallel_queue.py'),
             HERE.with_name('judge_concurrent_infer.py')]
    return dict(BASE_SOURCES(), **{str(p.relative_to(frozen.REPO)): sha(p) for p in paths})


def validate_cases(study):
    if any(study['protocol'].get(k) != v for k, v in EXPECTED_PROTOCOL.items()):
        raise ValueError('Original full-benchmark generation protocol changed')
    entries = study['cases']
    if Counter(e['benchmark'] for e in entries) != {'rbench': 650, 'paibench': 1044}:
        raise ValueError('Expected all 650 RBench and 1044 PAI-Bench scenes')
    if len({e['case']['id'] for e in entries}) != 1694:
        raise ValueError('Duplicate scene ID')
    for e in entries:
        case = e['case']
        if (len(e['seeds']) != 2 or len(set(e['seeds'])) != 2 or
                case['task'] != 'i2v' or
                any(case.get(k) != EXPECTED_PROTOCOL[k] for k in ('width', 'height', 'frames')) or
                case.get('negative_prompt') != study['protocol']['negative_prompt']):
            raise ValueError('Scene differs from frozen I2V protocol: ' + case['id'])


def validate_methods(study):
    from baseline_runtime import verify
    for name in METHODS:
        verify(study, name)
        spec = study['methods'][name]
        cfg = spec['config']
        if (cfg['debug'] != 'off' or cfg['resource_mode'] != 'shared-calibration' or
                cfg['execution'] != 'graph' or cfg.get('compute_budget') is not None or
                cfg.get('budget_plan') or spec.get('runtime_options') or spec.get('assets') or
                name in study.get('adapters', {})):
            raise ValueError('Unexpected baseline execution configuration: ' + name)
        expected = {'bf16': ('bf16', 'bf16'), 'fp8': ('fp8', 'bf16'),
                    'awq_reference': ('bf16', 'bf16'), 'fp4_base': ('fp4', 'fp4')}[name]
        if (cfg['precision'], cfg['attention_backend']) != expected:
            raise ValueError('Changed baseline precision: ' + name)
        if name == 'awq_reference':
            adapter = spec.get('baseline_adapter', {})
            if (adapter.get('method') != 'awq' or spec.get('performance_eligible') is not False or
                    spec.get('storage_comparison_eligible') is not False):
                raise ValueError('Expected the qualified ModelOpt NVFP4 AWQ W4A16 quality reference')
        elif spec.get('baseline_adapter'):
            raise ValueError('Unexpected external baseline adapter')
        if name == 'fp4_base':
            if any(cfg[k]['activation'] != 0 or cfg[k]['weight'] != 0 or cfg[k]['cross']
                   for k in ('ffn', 'projection')):
                raise ValueError('Plain W4A4 must have no linear repair')
            attn = cfg['attention']
            if (attn['rules'] or attn['default']['coverage'] != 0 or
                    attn['default'].get('head_budgets') or
                    attn['default'].get('head_quotas') is not None or
                    attn['default'].get('pv_head_quotas') is not None):
                raise ValueError('Plain W4A4 must have no attention repair')


def prepare(a):
    reference = a.reference.resolve()
    if a.root == reference or not 1 <= a.chunk_scenes <= 24:
        raise ValueError('Use a new run directory and 1-24 scenes per task')
    original = BASE_VERIFY(reference)
    if a.gpus != [0, 1] or not set(a.gpus) <= set(original['gpus']):
        raise ValueError('This supplement uses the original GPU0/1 pair')
    a.root.mkdir(parents=True, exist_ok=True)
    job = dict(schema=1, kind='full_I2V_four_baseline_extension', methods=list(METHODS),
        models={}, tasks=[], gpus=a.gpus, steps=35, performance_eligible=False,
        source_sha256=sources(), reference_root=str(reference),
        reference_manifest_sha256=sha(reference/'manifest.json'),
        cohort=original['cohort'], cohort_sha256=original['cohort_sha256'],
        video_export=original['video_export'], codec_tools=original['codec_tools'],
        scheduling=dict(kind='Existing shared longest-estimated-task-first pool',
            chunk_scenes=a.chunk_scenes, estimates='Historical smoke times for queue order only'),
        scope='Full Edge/Nano RBench/PAI-Bench baseline supplement; identical original '
              'inputs and seeds. Native FP8 W8A8 with BF16 attention; decoded ModelOpt '
              'NVFP4 AWQ W4A16 quality reference; unrepaired FP4 projections/attention. '
              'Shared concurrency is not performance evidence.')
    for model in ('edge', 'nano'):
        parent = a.source.resolve()/model
        previous_path = reference/model/'final/manifest.json'
        previous = json.loads(previous_path.read_text())
        validate_cases(previous)
        qualified_path = parent/'qualification.json'
        qualified = json.loads(qualified_path.read_text())
        source_path = parent/'final/manifest.json'
        if (not qualified['complete'] or qualified['final_manifest_sha256'] != sha(source_path) or
                qualified['shape_preflight_sha256'] != sha(parent/'shape_preflight.json')):
            raise ValueError('Unqualified source baseline matrix')
        current = quality_sources()
        changes = {k: dict(previous=qualified['quality_sources'].get(k), current=current.get(k))
            for k in set(current) | set(qualified['quality_sources'])
            if current.get(k) != qualified['quality_sources'].get(k)}
        if set(changes) - {frozen.RESOURCE_SOURCE}:
            raise ValueError('Numerical source changed since qualification: ' + str(changes))
        study = json.loads(source_path.read_text())
        validate_methods(study)
        if study['model'] != previous['model'] or study['steps'] != previous['steps']:
            raise ValueError('Baseline model revision or step count differs from original run')
        base = copy.deepcopy(previous)
        base.pop('reuse', None)
        base.update(methods={name: study['methods'][name] for name in METHODS}, adapters={},
            external=study.get('external', {}), awq_mode='reference', primary_comparisons=[],
            source_qualification=str(qualified_path), source_qualification_sha256=sha(qualified_path),
            source_output_hashes=None, scope=job['scope'],
            gpu_policy=dict(allowed=a.gpus, lock_scope='device', assignment='dynamic'))
        for e in base['cases']:
            if (sha(e['case']['image']) != e['image_sha256'] or
                    sha(base['input_copies'][e['case']['id']]) != e['image_sha256']):
                raise ValueError('Original input image changed')
        final = a.root/model/'final/manifest.json'
        immutable(final, base)
        smoke = json.loads((parent/'smoke/manifest.json').read_text())
        hashes, estimates, peaks = {}, {}, {}
        for name in METHODS:
            if smoke['methods'][name] != base['methods'][name]:
                raise ValueError('Baseline changed after historical smoke')
            hashes[name], estimates[name] = {}, {}
            peaks[name] = []
            for e in smoke['cases']:
                folder = parent/'smoke/generation'/name/e['case']['id']
                result = json.loads((folder/'result.json').read_text())
                if not result['complete'] or {r['seed'] for r in result['rows']} != set(e['seeds']):
                    raise ValueError('Incomplete historical smoke')
                for row in result['rows']:
                    if sha(folder/row['outputs']) != row['outputs_sha256']:
                        raise ValueError('Corrupt historical smoke NPZ')
                    peaks[name].append(row['peak_reserved_bytes'])
                hashes[name][e['case']['id']] = {str(r['seed']): r['sha256'] for r in result['rows']}
                estimates[name][e['benchmark']] = statistics.mean(r['wall_seconds'] for r in result['rows'])
        hp = a.root/model/'qualification/source_output_hashes.json'
        immutable(hp, hashes)
        qualification = copy.deepcopy(base)
        qualification.update(role='development_gpu_qualification', cases=smoke['cases'],
            source_output_hashes=str(hp), source_output_hashes_sha256=sha(hp))
        immutable(a.root/model/'qualification/manifest.json', qualification)
        preflight_path = reference/model/'shape_preflight.json'
        preflight = json.loads(preflight_path.read_text())
        shape_rows = next(iter(preflight['checks'].values()))['cases']
        if {r['case'] for r in shape_rows} != {e['case']['id'] for e in base['cases']}:
            raise ValueError('Incomplete original full-corpus shape coverage')
        immutable(a.root/model/'shape_preflight.json', dict(complete=True,
            manifest_sha256=sha(final), source=str(preflight_path), source_sha256=sha(preflight_path),
            methods=list(METHODS), cases=[{k: r[k] for k in ('case', 'm', 'n_by_cfg', 'key_tiles_by_cfg')}
                for r in shape_rows], compute_budget=None,
            scope='Same full inputs and latent/token shapes as original qualified run; '
                  'baseline methods have no repair budget. Each current input still warms before replay.'))
        job['models'][model] = dict(source_model_root=str(parent),
            source_manifest_sha256=sha(source_path), reference_model_manifest_sha256=sha(previous_path),
            final_manifest_sha256=sha(final),
            smoke_manifest_sha256=sha(a.root/model/'qualification/manifest.json'),
            shape_preflight_sha256=sha(a.root/model/'shape_preflight.json'),
            source_admission_changes=changes, expected_videos=1694*2*len(METHODS), reused_videos=0,
            expected_per_benchmark={b: sum(len(e['seeds']) for e in base['cases'] if e['benchmark'] == b)*len(METHODS)
                for b in ('rbench', 'paibench')},
            historical_peak_reserved_bytes={name: max(values) for name, values in peaks.items()},
            qualification='Each eligible GPU/slot reproduces all four historical outputs per '
                          'method and A-B-A before claiming final tasks')
        for name in METHODS:
            for benchmark in ('rbench', 'paibench'):
                entries = [e for e in base['cases'] if e['benchmark'] == benchmark]
                for start in range(0, len(entries), a.chunk_scenes):
                    part = entries[start:start+a.chunk_scenes]
                    tid = f'{model}__{name}__{benchmark}__{start:04d}'
                    job['tasks'].append(dict(id=tid, model=model, method=name, benchmark=benchmark,
                        cases=[e['case']['id'] for e in part], videos=sum(len(e['seeds']) for e in part),
                        estimated_seconds=len(part)*(2+4/35)*estimates[name][benchmark]))
                    for e in part:
                        folder = a.root/'tasks'/tid/'generation'/e['case']['id']
                        folder.mkdir(parents=True, exist_ok=True)
                        alias = a.root/model/'final/generation'/name/e['case']['id']
                        alias.parent.mkdir(parents=True, exist_ok=True)
                        if not alias.is_symlink():
                            alias.symlink_to(folder, target_is_directory=True)
                        if alias.resolve() != folder.resolve():
                            raise ValueError('Generation alias changed')
                        log = folder/'generation.log'
                        target = a.root/'tasks'/tid/'generation.log'
                        if not log.is_symlink():
                            log.symlink_to(target)
                        if log.resolve() != target.resolve():
                            raise ValueError('Log alias changed')
    job['tasks'].sort(key=lambda t: (-t['estimated_seconds'], t['id']))
    immutable(a.root/'manifest.json', job)
    if not (a.root/'queue.json').exists():
        write(a.root/'queue.json', dict(job_sha256=sha(a.root/'manifest.json'),
            tasks={t['id']: dict(status='pending', attempts=[]) for t in job['tasks']}))
    print(json.dumps(dict(prepared=True, root=str(a.root), videos=27104,
        tasks=len(job['tasks']), methods=METHODS, gpus=a.gpus)), flush=True)


def verify_job(root):
    job = json.loads((root/'manifest.json').read_text())
    if (job.get('kind') != 'full_I2V_four_baseline_extension' or job['methods'] != list(METHODS) or
            job['source_sha256'] != sources()):
        raise ValueError('Changed baseline supplement or frozen implementation')
    if (sha(Path(job['reference_root'])/'manifest.json') != job['reference_manifest_sha256'] or
            sha(job['cohort']) != job['cohort_sha256']):
        raise ValueError('Original benchmark or cohort changed')
    for entry in job['codec_tools'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed video encoder')
    for model, spec in job['models'].items():
        for role, key in [('final', 'final_manifest_sha256'), ('qualification', 'smoke_manifest_sha256')]:
            if sha(root/model/role/'manifest.json') != spec[key]:
                raise ValueError('Changed baseline model manifest')
        if sha(root/model/'shape_preflight.json') != spec['shape_preflight_sha256']:
            raise ValueError('Changed full-corpus shape check')
    return job


def configure_existing_queue():
    """Explicit, process-local dependency injection; original source files stay frozen."""
    frozen.METHODS = METHODS
    frozen.verify_job = verify_job
    pool.HERE = HERE
    runtime.HERE = HERE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'launch', 'supervise', 'worker', 'infer', 'status'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('/home/yufeng/thunderworld/full_quality_matrix_20260919'))
    p.add_argument('--reference', type=Path, default=Path('/home/yufeng/thunderworld/judge_full_i2v_20260919'))
    p.add_argument('--chunk-scenes', type=int, default=12)
    p.add_argument('--gpus', nargs='+', type=int)
    p.add_argument('--slots-per-gpu', type=int)
    p.add_argument('--secondary-methods', nargs='*')
    p.add_argument('--admission', choices=('measured', 'requested'), default='requested')
    p.add_argument('--probe-report', type=Path)
    p.add_argument('--gpu', type=int)
    p.add_argument('--slot', type=int)
    p.add_argument('--global-fd', type=int)
    p.add_argument('--device-fd', type=int)
    p.add_argument('--coordinator-pid', type=int)
    p.add_argument('--model', choices=('edge', 'nano'))
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--output', type=Path)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--qualification', action='store_true')
    a = p.parse_args()
    a.root = a.root.resolve()
    if a.operation == 'prepare':
        a.gpus = a.gpus or [0, 1]
        prepare(a)
        return
    configure_existing_queue()
    if a.operation == 'launch':
        # Historical Nano BF16/AWQ has a larger resident model than repair FP4.
        # Reserve 36 GiB for the primary Nano slot, 12 GiB for each Edge slot,
        # and 12 GiB free headroom; retain the established one-Nano-per-GPU rule.
        from thunderworld_fp4.cosmos.resources import free_memory
        slots = a.slots_per_gpu or 4
        devices = a.gpus or [0, 1]
        available = free_memory()
        required = (36 + 12*(slots-1) + 12)*1024
        if any(available[g] < required for g in devices):
            raise RuntimeError(f'PAUSED: baseline pool needs {required} MiB/GPU; available={available}')
        write(a.root/'baseline_memory_admission.json', dict(free_mib=available,
            required_mib_per_gpu=required, primary_reservation_mib=36*1024,
            secondary_edge_reservation_mib=12*1024, reserve_mib=12*1024,
            performance_eligible=False))
        a.gpus, a.slots_per_gpu = devices, slots
        a.secondary_methods = a.secondary_methods or list(METHODS)
        pool.launch(a)
    elif a.operation == 'supervise':
        pool.supervise(a)
    elif a.operation == 'worker':
        if None in (a.gpu, a.slot, a.global_fd, a.device_fd):
            p.error('worker requires a slot and inherited resource leases')
        pool.worker(a)
    elif a.operation == 'infer':
        if None in (a.model, a.method, a.output, a.gpu, a.global_fd, a.device_fd):
            p.error('infer requires model/method/output and inherited resource leases')
        runtime.infer(a)
    else:
        job, value = pool.policy(a.root)
        print(json.dumps(pool.snapshot(a.root, job, value), indent=2))


if __name__ == '__main__':
    main()

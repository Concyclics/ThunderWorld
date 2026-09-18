"""Freeze and emit exclusive full-request retests, including native AWQ.

No GPU is used by freeze/emit. Workers reuse CosmosRunner and its idle/debug/
tuning guards. New device tuning must preserve frozen full-generation outputs
before its timings become eligible. This is not a shared-calibration benchmark.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from baselines import sha, write

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def implementation():
    paths = list((REPO/'packages/fp4-repair/src').rglob('*.py'))
    paths += [HERE/'awq_marlin.py', HERE/'awq_scales.py', Path(__file__)]
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def freeze(root, output):
    if not json.loads((root/'awq_native/orchestrator.json').read_text())['complete']:
        raise ValueError('Full native AWQ qualification must finish first')
    operators = json.loads((root/'awq_native/operators.json').read_text())
    assets_path = root/'awq_native/assets/manifest.json'
    assets = json.loads(assets_path.read_text())
    if (not operators['complete'] or not assets.get('complete_model') or
        operators['assets_sha256'] != sha(assets_path) or operators['source_sha256'] != sha(HERE/'awq_marlin.py')):
        raise ValueError('Missing native AWQ operator qualification')
    shapes = {(row['name'], row['m']) for row in operators['rows']}
    expected_shapes = {(name, m) for name in assets['modules'] for m in (129, 416, 960, 1024)}
    if shapes != expected_shapes or len(operators['rows']) != 3*len(expected_shapes):
        raise ValueError('Incomplete native AWQ shape/current-input coverage')
    studies = {name: root/name for name in ('cross', 'outlier_development', 'awq_native')}
    manifests = {name: json.loads((path/'manifest.json').read_text()) for name, path in studies.items()}
    base = manifests['cross']
    if any(m['model'] != base['model'] or m['cases'] != base['cases'] for m in manifests.values()):
        raise ValueError('Unmatched speed/quality input cohorts')
    sources = {name: 'cross' for name in ('bf16', 'fp8', 'fp4_base', 'fp4_three', 'fp4_four', 'b1.5_protected_n', 'b2_joint')}
    sources.update({'b2.5_ffn_guarded': 'outlier_development', 'awq_marlin': 'awq_native'})
    jobs = []
    for benchmark in ('rbench', 'paibench'):
        entry = next(e for e in base['cases'] if e['benchmark'] == benchmark)
        for name, study in sources.items():
            spec = copy.deepcopy(manifests[study]['methods'][name])
            folder = studies[study]/'generation'/name/entry['case']['id']
            receipt = json.loads((folder/'result.json').read_text())
            if not receipt['complete'] or any(sha(folder/r['outputs']) != r['outputs_sha256'] for r in receipt['rows']):
                raise ValueError('Incomplete or changed frozen output')
            expected = {str(r['seed']): r['sha256'] for r in receipt['rows']}
            if set(expected) != set(map(str, entry['seeds'])):
                raise ValueError('Incomplete quality seeds')
            jobs.append(dict(name=benchmark+'_'+name, method=name, case=entry, spec=spec,
                expected=expected, quality_receipt_sha256=sha(folder/'result.json'),
                quality_manifest_sha256=sha(studies[study]/'manifest.json')))
    value = dict(frozen=True, model=base['model'], jobs=jobs, implementation=implementation(),
        native_awq_qualification_sha256=sha(root/'awq_native/orchestrator.json'),
        performance_eligible=False, quality_recommendation=False,
        timing_seeds=[91001, 91002, 91003], rounds=5, ordering_seed=180918,
        scope='Two fixed development scenes, one per dataset, all methods matched within each scene. '
              'Includes failed-quality budget controls. Timing does not confer a quality recommendation. '
              'Five randomized rounds x three complete requests, fresh process/prepare/warmup for each configuration/round. '
              'PRO 6000 and 5090 must have independent tuning, qualification and output directories.')
    if output.exists():
        if json.loads(output.read_text()) != value:
            raise ValueError('Changed frozen retest package')
    else:
        write(output, value)


def emit(manifest_path, output, device):
    manifest = json.loads(manifest_path.read_text())
    if not manifest['frozen']:
        raise ValueError('Freeze quality assets and outputs first')
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    for operation in ('prepare', 'verify'):
        for job in manifest['jobs']:
            commands.append(dict(operation=operation, job=job['name'], output=str(output/(operation+'_'+job['name']))))
    rng = random.Random(manifest['ordering_seed'])
    for round_id in range(manifest['rounds']):
        jobs = list(manifest['jobs'])
        rng.shuffle(jobs)
        commands += [dict(operation='benchmark', round=round_id, job=j['name'],
                          output=str(output/f'round{round_id}_{j["name"]}')) for j in jobs]
    for row in commands:
        row['command'] = [sys.executable, str(Path(__file__).resolve()), 'worker', '--manifest', str(manifest_path.resolve()),
                          '--job', row['job'], '--operation', row['operation'], '--output', row['output'], '--suite', str(output.resolve())]
        row['environment'] = dict(CUDA_VISIBLE_DEVICES=device)
    write(output/'commands.json', dict(manifest_sha256=sha(manifest_path), commands=commands,
        scope='Emitted only; execute sequentially on an exclusive GPU. No command is run by emit.'))


def worker(a):
    from dataclasses import replace
    from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
    from thunderworld_fp4.cosmos.cli import idle_guard
    import torch
    manifest = json.loads(a.manifest.read_text())
    if not manifest['frozen'] or manifest['implementation'] != implementation():
        raise ValueError('Changed implementation requires full quality requalification')
    job = next(j for j in manifest['jobs'] if j['name'] == a.job)
    spec = job['spec']
    if sha(job['case']['case']['image']) != job['case']['image_sha256']:
        raise ValueError('Changed benchmark input')
    for file, digest in ((Path(spec['assets'])/'manifest.json', spec['assets_sha256']),) if spec.get('assets') else ():
        if sha(file) != digest:
            raise ValueError('Changed quantization assets')
    if spec['config'].get('budget_plan') and sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Changed repair plan')
    adapter = spec.get('baseline_adapter')
    if adapter and (adapter['method'] != 'marlin_awq' or
                    sha(Path(adapter['assets'])/'manifest.json') != adapter['assets_sha256'] or
                    sha(HERE/'awq_marlin.py') != adapter['implementation_sha256']):
        raise ValueError('Only qualified native AWQ can be timed')
    idle_guard()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = RunConfig.from_dict(spec['config'])
    config = replace(config, resource_mode='exclusive', debug='off', execution='graph', strict_tuning=a.operation != 'prepare')
    tuning = None if a.operation == 'prepare' else a.suite/('prepare_'+a.job)/'tuning.json'
    result = dict(complete=False, operation=a.operation, job=a.job, rows=[],
                  manifest_sha256=sha(a.manifest), job_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest())
    start = time.perf_counter()
    runner = None
    try:
        runner = CosmosRunner(manifest['model'], config, tuning=tuning,
            tune=a.operation == 'prepare' and config.precision == 'fp4',
            assets=spec.get('assets'), ar_cache=True, first_frame=True).prepare()
        if adapter:
            from awq_marlin import install
            t = time.perf_counter()
            result['adapter'] = install(runner, adapter['assets'])
            torch.cuda.synchronize()
            runner.timings['offline_adapter_prepare_seconds'] = time.perf_counter()-t
        case = job['case']['case']
        runner.warmup(case, 35)
        runner.table.save(a.output/'tuning.json')
        result.update(runtime=runner.receipt(), ready_seconds=time.perf_counter()-start,
            tuning_sha256=sha(a.output/'tuning.json'), performance_eligible=False)
        if a.operation == 'verify':
            for seed, expected in job['expected'].items():
                idle_guard()
                _, row = runner.infer(case, int(seed), 35)
                idle_guard()
                if row['sha256'] != expected:
                    raise RuntimeError('Device/tuning changed full outputs; stop timing and requalify complete quality')
                row['performance_eligible'] = False
                result['rows'].append(row)
            result['full_output_bitwise_equal'] = True
        elif a.operation == 'benchmark':
            qualification = json.loads((a.suite/('verify_'+a.job)/'result.json').read_text())
            if (not qualification['complete'] or not qualification['full_output_bitwise_equal'] or
                qualification['manifest_sha256'] != sha(a.manifest) or qualification['job_sha256'] != result['job_sha256'] or
                qualification['tuning_sha256'] != result['tuning_sha256'] or
                qualification['runtime']['environment'] != result['runtime']['environment']):
                raise ValueError('Matching device/tuning full-output qualification required')
            result['rows'] = runner.benchmark(case, manifest['timing_seeds'], 35)
            result['performance_eligible'] = True
        idle_guard()
        result.update(complete=True, runtime=runner.receipt())
        write(a.output/'result.json', result)
    except Exception as error:
        result.update(error=repr(error), performance_eligible=False)
        if runner is not None:
            runner.table.save(a.output/'tuning.partial.json')
        write(a.output/'result.json', result)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--root', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('emit')
    e.add_argument('--manifest', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--device', choices=('0', '3'), default='0')
    w = sub.add_parser('worker')
    for key in ('manifest', 'output', 'suite'):
        w.add_argument('--'+key, type=Path, required=True)
    w.add_argument('--job', required=True)
    w.add_argument('--operation', choices=('prepare', 'verify', 'benchmark'), required=True)
    a = p.parse_args()
    if a.action == 'freeze':
        freeze(a.root, a.output)
    elif a.action == 'emit':
        emit(a.manifest, a.output, a.device)
    else:
        worker(a)


if __name__ == '__main__':
    main()

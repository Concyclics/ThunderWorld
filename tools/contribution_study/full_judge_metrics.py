"""Frozen, resumable all-method scoring of the two complete judge cohorts."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from baselines import sha, write

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
METHODS = {
    'bf16': 'BF16 reference', 'fp8': 'FP8 W8A8',
    'awq_reference': 'AWQ NVFP4 W4A16 (decoded quality reference)',
    'fp4_base': 'FP4 W4A4', 'arcquant': 'ARCQuant 50%',
    'arcquant100': 'ARCQuant 100%', 'sharq_perf': 'SharQ',
    'b1.5_protected_n': 'Ours 1.5x', 'b2_joint': 'Ours 2.0x',
}
CANDIDATES = [m for m in METHODS if m != 'bf16']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    files = [HERE, HERE.with_name('full_video_metrics.py'),
             HERE.with_name('full_judge_metrics_report.py'),
             REPO/'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py']
    return {str(p): sha(p) for p in files}


def verify(root):
    manifest = read(root/'manifest.json')
    if manifest['source_sha256'] != sources():
        raise ValueError('Metric source changed; use a new run root')
    return manifest


def index_rows(root):
    result, ids = {}, set()
    with (root/'judge_index.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            key = (row['model'], row['method'], row['case_id'], row['seed'])
            if key in result or row['sample_id'] in ids:
                raise ValueError('Duplicate source index sample')
            result[key] = row
            ids.add(row['sample_id'])
    return result


def prepare(a):
    from full_video_metrics import environment
    if (a.root/'manifest.json').exists():
        verify(a.root)
        raise ValueError('Already prepared; run qualify/run/status on the existing root')
    a.root.mkdir(parents=True, exist_ok=True)
    roots = [a.baselines.resolve(), a.original.resolve()]
    indexes = {str(r): index_rows(r) for r in roots}
    models, source_inputs, tasks, counts = {}, {}, [], Counter()
    audit = dict(started_at=now(), errors=[], reused_scenes={}, execution={}, adapter={})
    for root in roots:
        for path in [root/'manifest.json', root/'judge_index.jsonl']:
            source_inputs[str(path)] = sha(path)
    for model in ('edge', 'nano'):
        studies = {str(r): read(r/model/'final/manifest.json') for r in roots}
        base = studies[str(roots[0])]
        original = studies[str(roots[1])]
        for field in ('cases', 'protocol', 'model', 'steps', 'debug', 'statistics'):
            if base[field] != original[field]:
                raise ValueError(f'Source cohorts differ: {model}/{field}')
        if base['debug'] != 'off' or base['steps'] != 35:
            raise ValueError('Unexpected source generation definition')
        if Counter(x['benchmark'] for x in base['cases']) != {'rbench': 650, 'paibench': 1044}:
            raise ValueError('Expected the full 650/1044 scene cohorts')
        if len({x['case']['id'] for x in base['cases']}) != 1694:
            raise ValueError('Duplicate case ID')
        owner = {}
        for root in roots:
            s = studies[str(root)]
            path = root/model/'final/manifest.json'
            source_inputs[str(path)] = sha(path)
            for method in s['methods']:
                if method in owner:
                    raise ValueError('Duplicate method across source cohorts')
                owner[method] = root
        if set(owner) != set(METHODS):
            raise ValueError('Unexpected full-cohort method inventory')
        models[model] = dict(model=base['model'], protocol=base['protocol'],
            statistics=base['statistics'], prior_scene_role_counts=base.get('prior_scene_role_counts'),
            methods={m: studies[str(owner[m])]['methods'][m] for m in METHODS},
            model_identity=None, targets=None)
        for entry in base['cases']:
            cid = entry['case']['id']
            if len(entry['seeds']) != 2 or len(set(entry['seeds'])) != 2:
                raise ValueError('Each scene must have its original two seeds')
            task = dict(model=model, case_id=cid, benchmark=entry['benchmark'], category=entry['category'],
                source_id=entry['source_id'], seeds=entry['seeds'], case=entry['case'],
                image_sha256=entry['image_sha256'], prior_scene_roles=entry.get('prior_scene_roles', {}),
                records={})
            gold = None
            for method in METHODS:
                root = owner[method]
                study = studies[str(root)]
                folder = root/model/'final/generation'/method/cid
                path = folder/'result.json'
                receipt_sha = sha(path)
                receipt = read(path)
                reuse = study.get('reuse', {})
                reused = cid in reuse.get('case_ids', [])
                expected_manifest = reuse['manifest_sha256'] if reused else source_inputs[str(root/model/'final/manifest.json')]
                if reused and receipt_sha != reuse['receipt_sha256'][method+'/'+cid]:
                    raise ValueError('Reused scene receipt changed')
                ident = receipt['identity']
                if (not receipt['complete'] or ident['manifest_sha256'] != expected_manifest or
                    ident['method'] != method or ident['case'] != entry['case'] or
                    ident['image_sha256'] != entry['image_sha256'] or ident['seeds'] != entry['seeds']):
                    raise ValueError(f'Unmatched/incomplete receipt: {path}')
                if method == 'bf16':
                    gold = receipt
                    if models[model]['model_identity'] is None:
                        models[model]['model_identity'] = receipt['model_identity']
                        models[model]['targets'] = receipt['targets']
                if (receipt['model_identity'] != models[model]['model_identity'] or
                    receipt['targets'] != models[model]['targets']):
                    raise ValueError(f'Model/projection scope mismatch: {path}')
                for k in ('enabled', 'qkv_streams', 'slots', 'scope'):
                    if receipt['ar_cache'][k] != gold['ar_cache'][k]:
                        raise ValueError('AR cache configuration differs')
                rows = {row['seed']: row for row in receipt['rows']}
                if len(rows) != len(receipt['rows']) or set(rows) != set(entry['seeds']):
                    raise ValueError('Missing or duplicate paired seeds')
                task['records'][method] = {}
                audit['adapter'].setdefault(model+'/'+method, receipt.get('adapter'))
                execution = audit['execution'].setdefault(model+'/'+method, set())
                for seed in entry['seeds']:
                    row = rows[seed]
                    if row['steps'] != base['steps'] or row['debug'] != 'off' or row['task'] != 'i2v':
                        raise ValueError('Generation protocol mismatch')
                    execution.add(row['execution'])
                    idx = indexes[str(root)].pop((model, method, cid, seed))
                    if any(idx[k] != entry[k] for k in ('benchmark', 'category', 'source_id')) or idx['prompt'] != entry['case']['prompt']:
                        raise ValueError('Index content differs from generation')
                    source = folder/row['outputs']
                    if Path(idx['raw_outputs']).resolve() != source.resolve() or source.stat().st_size <= 0:
                        raise ValueError('Output archive missing or index mismatch')
                    if set(row['sha256']) != {'video'}:
                        raise ValueError('Unexpected signal inventory')
                    task['records'][method][str(seed)] = dict(path=str(source),
                        npz_sha256=row['outputs_sha256'], video_sha256=row['sha256']['video'],
                        receipt=str(path), receipt_sha256=receipt_sha, sample_id=idx['sample_id'],
                        video_path=idx['video_path'], metadata=idx['metadata'],
                        execution=row['execution'], actual_mma_ratio=row.get('actual_mma_ratio'))
                    counts[model+'/'+entry['benchmark']+'/'+method] += 1
            target = a.root/'tasks'/model/(cid+'.json')
            write(target, task)
            tasks.append(dict(path=str(target.relative_to(a.root)), sha256=sha(target),
                result=f'results/{model}/{cid}.json', model=model, benchmark=entry['benchmark'],
                category=entry['category'], case_id=cid, pairs=len(CANDIDATES)*len(entry['seeds'])))
        audit['reused_scenes'][model] = len(original.get('reuse', {}).get('case_ids', []))
        print(json.dumps(dict(event='prepared_model', model=model, scenes=1694)), flush=True)
    if any(indexes.values()):
        raise ValueError('Unexpected unpaired samples left in source index')
    audit['execution'] = {k: sorted(v) for k, v in audit['execution'].items()}
    audit.update(completed_at=now(), complete=True, decoded_hash_checks='performed by worker before metric computation',
        expected_candidate_pairs=sum(t['pairs'] for t in tasks), source_videos=sum(counts.values()),
        counts=dict(counts), comparison_scope='Same-input/seed BF16 consistency; not real-world ground truth or task success')
    write(a.root/'input_audit.json', audit)
    manifest = dict(schema=1, frozen=True, created_at=now(), source_sha256=sources(),
        source_inputs_sha256=source_inputs, input_audit_sha256=sha(a.root/'input_audit.json'),
        environment=environment(), methods=METHODS, models=models, tasks=tasks,
        expected_pairs=54208, expected_scenes=3388, source_videos=60984,
        metric_definition=dict(reference='same model/scene/seed BF16', input='original float32 NPZ video',
            frames='all 61 including conditioned first frame', resize=False, data_range=1,
            lpips='official AlexNet LPIPS v0.1; FP32; batch=8; reference features cached',
            ssim='skimage default: uniform7x7, sample covariance 49/48, K1=.01, K2=.03; RGB/frame mean',
            psnr='-10 log10(clip-global float64 MSE); exact equality stored null plus psnr_exact=true',
            aggregation=['category_macro: seed mean -> scene mean -> equal categories',
                         'scene_micro: seed mean -> equal scenes'],
            bootstrap='100000 paired scene-cluster replicates, stratified by category; both seeds stay together'),
        performance_eligible=False, selection_eligible=False, source_roots=[str(r) for r in roots])
    if len(tasks) != manifest['expected_scenes'] or audit['expected_candidate_pairs'] != manifest['expected_pairs']:
        raise ValueError('Incorrect full cohort size')
    write(a.root/'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, root=str(a.root), pairs=54208, videos=60984)), flush=True)


def load_task(root, item):
    path = root/item['path']
    if sha(path) != item['sha256']:
        raise ValueError('Task changed')
    task = read(path)
    seen = {}
    for records in task['records'].values():
        for record in records.values():
            seen[record['receipt']] = record['receipt_sha256']
    if any(sha(path) != digest for path, digest in seen.items()):
        raise ValueError('Source receipt changed')
    return task


def qualify(a):
    import numpy as np
    import torch
    from full_video_metrics import VideoMetrics, load_video, environment
    from thunderworld_fp4.cosmos.quality import compare
    from judge_concurrent_infer import device_leases
    manifest = verify(a.root)
    if environment() != manifest['environment']:
        raise ValueError('Metric environment changed')
    qualification = dict(started_at=now(), manifest_sha256=sha(a.root/'manifest.json'),
        backend_sha256=sha(HERE.with_name('full_video_metrics.py')), gpu=a.gpu,
        tolerances=dict(lpips=1e-6, ssim_mean=1e-6, psnr_db=1e-10, relative_l2=1e-12, rmse=1e-12, mse=1e-12),
        checks=[], complete=False, performance_eligible=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    with device_leases([a.gpu]):
        evaluator = VideoMetrics()
        qualification['device_name'] = torch.cuda.get_device_name()
        def check(name, x, y, *, cpu=False):
            evaluator.set_reference(y)
            begin = time.perf_counter()
            actual = evaluator.compare(x)
            elapsed = time.perf_counter()-begin
            gold = compare({'video': x}, {'video': y}, evaluator.model)['video']
            diffs = {}
            for key, tolerance in qualification['tolerances'].items():
                av, bv = actual['video'][key], gold[key]
                if av is None or bv is None:
                    if av != bv: raise ValueError('Non-finite comparison mismatch')
                    diffs[key] = 0
                else:
                    diffs[key] = abs(av-bv)
                    if diffs[key] > tolerance:
                        raise ValueError(f'Metric mismatch {name}/{key}: {av} != {bv}')
            if actual['video']['psnr_exact'] != gold['psnr_exact']:
                raise ValueError('Exact match handling differs')
            row = dict(name=name, max_abs_difference=diffs, cached_metric_seconds=elapsed)
            if cpu:
                evaluator.model.cpu()
                cpu_result = compare({'video': x}, {'video': y}, evaluator.model)['video']
                evaluator.model.to(evaluator.device)
                row['cpu_lpips_abs_difference'] = abs(cpu_result['lpips']-actual['video']['lpips'])
                if row['cpu_lpips_abs_difference'] > 1e-6:
                    raise ValueError('Official CPU LPIPS and GPU LPIPS differ beyond tolerance')
            qualification['checks'].append(row)
            print(json.dumps(dict(event='qualified_pair', gpu=a.gpu, **row)), flush=True)
        rng = np.random.default_rng(20260923)
        z = np.zeros((9, 64, 64, 3), np.float32)
        r = rng.random((9, 64, 64, 3), dtype=np.float32)
        for name, x, y in [('zeros', z, z), ('identical', r, r),
                            ('constants', z+.5, z+.51), ('opposites', z, z+1),
                            ('random', r, rng.random(r.shape, dtype=np.float32)),
                            ('small_noise', r, np.clip(r+1e-4, 0, 1))]:
            check(name, x, y)
        # Fixed, outcomes-independent selection: two scenes per model/benchmark.
        for model in ('edge', 'nano'):
            for benchmark in ('rbench', 'paibench'):
                items = sorted((t for t in manifest['tasks'] if t['model']==model and t['benchmark']==benchmark), key=lambda t:t['case_id'])
                chosen = [items[0], items[len(items)//2]]
                for i, item in enumerate(chosen):
                    task = load_task(a.root, item)
                    seed = str(task['seeds'][i])
                    y = load_video(task['records']['bf16'][seed])
                    for method in CANDIDATES:
                        x = load_video(task['records'][method][seed])
                        check('/'.join((model, benchmark, task['case_id'], seed, method)), x, y,
                              cpu=i==0 and method=='fp8')
        qualification.update(complete=True, completed_at=now(), cases_checked=8, real_video_pairs=64)
    write(a.root/f'qualification_gpu{a.gpu}.json', qualification)


def completed_result(root, item, identity):
    path = root/item['result']
    if not path.exists(): return None
    result = read(path)
    if result.get('identity') != identity or result.get('task_sha256') != item['sha256']:
        raise ValueError('Metric resume identity mismatch')
    return result if result.get('complete') else None


def result_identity(root):
    return dict(manifest_sha256=sha(root/'manifest.json'))


def worker(a):
    from full_video_metrics import VideoMetrics, load_video, environment
    from judge_concurrent_infer import verify_descriptors
    manifest = verify(a.root)
    verify_descriptors(a.gpu, a.global_fd, a.device_fd)
    if environment() != manifest['environment']:
        raise ValueError('Changed metric environment')
    items = manifest['tasks'][:]
    random.Random(190919).shuffle(items)
    items = items[a.slot::a.slots]
    identity = result_identity(a.root)
    progress = a.root/f'workers/slot{a.slot}.json'
    count = 0
    for item in items:
        completed_result(a.root, item, identity)  # validate partial and complete resume identities
        if (a.root/item['result']).exists():
            count += len(read(a.root/item['result'])['rows'])
    state = dict(identity=identity, slot=a.slot, gpu=a.gpu, pid=os.getpid(), started_at=now(),
        expected=sum(item['pairs'] for item in items), completed_pairs=count, resumed_pairs=count,
        completed_scenes=0, complete=False, failed=False, performance_eligible=False)
    write(progress, state)
    try:
        evaluator = VideoMetrics()
        for item in items:
            if completed_result(a.root, item, identity):
                state['completed_scenes'] += 1
                continue
            task = load_task(a.root, item)
            path = a.root/item['result']
            result = dict(identity=identity, task_sha256=item['sha256'], complete=False,
                model=task['model'], benchmark=task['benchmark'], category=task['category'],
                case_id=task['case_id'], source_id=task['source_id'], prior_scene_roles=task['prior_scene_roles'],
                rows=[], gpu=a.gpu, started_at=now(), performance_eligible=False)
            if path.exists():
                result = read(path)
            already = {(r['method'], r['seed']) for r in result['rows']}
            state.update(current_scene=task['model']+'/'+task['case_id'], updated_at=now())
            write(progress, state)
            for seed in task['seeds']:
                missing = [m for m in CANDIDATES if (m, seed) not in already]
                if not missing: continue
                reference_record = task['records']['bf16'][str(seed)]
                reference = load_video(reference_record)
                evaluator.set_reference(reference)
                for method in missing:
                    record = task['records'][method][str(seed)]
                    begin = time.perf_counter()
                    candidate = load_video(record)
                    measured = evaluator.compare(candidate)
                    result['rows'].append(dict(method=method, seed=seed, sample_id=record['sample_id'],
                        reference_sample_id=reference_record['sample_id'], metrics={'video': measured['video']},
                        per_frame=measured['per_frame'], candidate_npz_sha256=record['npz_sha256'],
                        reference_npz_sha256=reference_record['npz_sha256'],
                        decoded_hashes_verified=True, metric_seconds_shared=time.perf_counter()-begin))
                    write(path, result)
                    count += 1
                    state.update(completed_pairs=count, updated_at=now())
                    write(progress, state)
            expected = {(m, seed) for m in CANDIDATES for seed in task['seeds']}
            actual = [(r['method'], r['seed']) for r in result['rows']]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError('Incomplete scene result or duplicate pair')
            result.update(complete=True, completed_at=now())
            write(path, result)
            state.update(completed_pairs=count, completed_scenes=state['completed_scenes']+1, updated_at=now())
            write(progress, state)
            print(json.dumps(dict(event='scene_complete', slot=a.slot, model=task['model'], case=task['case_id'], completed_pairs=count)), flush=True)
        state.update(complete=True, completed_at=now(), updated_at=now())
        if count != state['expected']: raise ValueError('Worker coverage mismatch')
        write(progress, state)
    except BaseException as exc:
        state.update(failed=True, error=repr(exc), traceback=traceback.format_exc(), updated_at=now())
        write(progress, state)
        raise


def status(root, *, print_result=True):
    manifest = read(root/'manifest.json')
    workers = [read(p) for p in sorted((root/'workers').glob('slot*.json'))]
    n = sum(w['completed_pairs'] for w in workers)
    result = dict(updated_at=now(), completed_pairs=n, expected_pairs=manifest['expected_pairs'],
        percent=100*n/manifest['expected_pairs'], workers=workers,
        metrics_complete=bool(workers) and all(w['complete'] for w in workers) and n==manifest['expected_pairs'],
        report_complete=(root/'report/complete.json').exists(), failed=any(w.get('failed') for w in workers))
    if workers:
        start = min(datetime.fromisoformat(w['started_at']) for w in workers)
        elapsed = (datetime.now(timezone.utc)-start).total_seconds()
        new = n-sum(w['resumed_pairs'] for w in workers)
        result['elapsed_seconds'] = elapsed
        result['pairs_per_second'] = new/elapsed if elapsed else None
        result['eta_seconds'] = (manifest['expected_pairs']-n)*elapsed/new if new>0 else None
    if print_result:
        print(json.dumps({k:v for k,v in result.items() if k!='workers'}, ensure_ascii=False), flush=True)
    return result


def run(a):
    from judge_concurrent_infer import device_leases, stop_children
    manifest = verify(a.root)
    identity = result_identity(a.root)
    for path, digest in manifest['source_inputs_sha256'].items():
        if sha(path) != digest: raise ValueError('Source cohort/index changed')
    for gpu in a.gpus:
        q = read(a.root/f'qualification_gpu{gpu}.json')
        if not q['complete'] or q['manifest_sha256'] != identity['manifest_sha256']:
            raise ValueError('Each metric GPU must pass independent qualification')
    if not a.gpus or len(set(a.gpus)) != len(a.gpus) or a.workers_per_gpu < 1:
        raise ValueError('Invalid GPU pool')
    with (a.root/'run.lock').open('a+') as handle, ExitStack() as stack:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases = stack.enter_context(device_leases(a.gpus))
        global_fd, devices = leases
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.strip().splitlines()}
        if any(free[g] < 12288 for g in a.gpus):
            raise RuntimeError('Less than 12 GiB free on a requested GPU; pause without changing workload')
        slots = len(a.gpus)*a.workers_per_gpu
        invocation = dict(started_at=now(), pid=os.getpid(), gpus=a.gpus, workers_per_gpu=a.workers_per_gpu,
            slots=slots, identity=identity, free_mib=free, performance_eligible=False)
        old = a.root/'invocation.json'
        if old.exists() and read(old)['slots'] != slots:
            raise ValueError('Resume with the same worker count to preserve progress accounting')
        write(old, invocation)
        processes = []
        try:
            for slot in range(slots):
                gpu = a.gpus[slot//a.workers_per_gpu]
                log_path = a.root/f'workers/slot{slot}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = stack.enter_context(log_path.open('a'))
                command = [sys.executable, str(HERE), 'worker', '--root', str(a.root), '--gpu', str(gpu),
                    '--slot', str(slot), '--slots', str(slots), '--global-fd', str(global_fd), '--device-fd', str(devices[gpu])]
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                process = subprocess.Popen(command, cwd=REPO, env=env, pass_fds=(global_fd,devices[gpu]),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
            while any(p.poll() is None for p in processes):
                for p in processes:
                    if p.poll() not in (None, 0): raise RuntimeError(f'Metric worker {p.pid} exited {p.returncode}; see workers logs')
                write(a.root/'progress.json', status(a.root, print_result=False))
                time.sleep(10)
            if any(p.returncode for p in processes): raise RuntimeError('Metric worker failed')
        finally:
            stop_children(processes)
        state = status(a.root)
        write(a.root/'progress.json', state)
        if not state['metrics_complete']: raise ValueError('Incomplete metric coverage')
    from full_judge_metrics_report import report
    report(a.root)
    write(a.root/'progress.json', status(a.root, print_result=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare_p = sub.add_parser('prepare'); prepare_p.add_argument('--root', type=Path, required=True)
    prepare_p.add_argument('--original', type=Path, default=Path('/home/yufeng/thunderworld/judge_full_i2v_20260919'))
    prepare_p.add_argument('--baselines', type=Path, default=Path('/home/yufeng/thunderworld/judge_full_baselines_20260922'))
    for name in ('qualify', 'run', 'worker', 'status', 'report'):
        q = sub.add_parser(name); q.add_argument('--root', type=Path, required=True)
        if name in ('qualify', 'worker'): q.add_argument('--gpu', type=int, required=True)
        if name == 'run':
            q.add_argument('--gpus', type=int, nargs='+', default=[0,2]); q.add_argument('--workers-per-gpu', type=int, default=2)
        if name == 'worker':
            for flag in ('slot','slots','global-fd','device-fd'): q.add_argument('--'+flag, type=int, required=True)
    a = p.parse_args(); a.root = a.root.resolve()
    if a.command == 'status': status(a.root)
    elif a.command == 'report':
        from full_judge_metrics_report import report
        report(a.root)
    else: globals()[a.command](a)


if __name__ == '__main__':
    main()

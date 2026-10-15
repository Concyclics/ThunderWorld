"""Coverage-checked full-cohort tables and paired scene-cluster bootstrap."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from baselines import sha, write
from full_judge_metrics import CANDIDATES, METHODS, completed_result, now, read, result_identity, verify

METRICS = ('lpips', 'psnr_db', 'ssim_mean', 'relative_l2', 'rmse')


def number(value):
    return float(value) if np.isfinite(value) else None


def save_csv(path, rows):
    if not rows: raise ValueError('Empty report table')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap(matrices, replicates, seed):
    """Keep methods, metrics and both seed-averages paired at scene level.

    Each category resamples its own N scenes. Return both equal-category and
    equal-scene estimates from the same draws. Multinomial counts implement
    ordinary with-replacement cluster bootstrap without a huge gathered array.
    """
    rng = np.random.default_rng(seed)
    shape = next(iter(matrices.values())).shape[1:]
    macro = np.zeros((replicates, *shape), np.float64)
    micro = np.zeros_like(macro)
    total = sum(len(x) for x in matrices.values())
    for category in sorted(matrices):
        matrix = matrices[category]
        n = len(matrix)
        flat = matrix.reshape(n, -1)
        if np.isnan(flat).any() or np.isneginf(flat).any():
            raise ValueError('Invalid bootstrap values')
        infinite = np.isposinf(flat)
        finite = np.where(infinite, 0, flat)
        for start in range(0, replicates, 512):
            stop = min(start+512, replicates)
            counts = rng.multinomial(n, np.full(n, 1/n), size=stop-start)
            value = counts @ finite / n
            if infinite.any(): value[counts @ infinite.astype(float) > 0] = np.inf
            value = value.reshape(stop-start, *shape)
            macro[start:stop] += value/len(matrices)
            micro[start:stop] += value*n/total
    return macro, micro


def interval(values, alpha=.05):
    # Nearest ranks handle exact-match +inf without inf-inf interpolation.
    return [number(v) for v in np.quantile(values, [alpha/2, 1-alpha/2], method='inverted_cdf')]


def report(root):
    manifest = verify(root)
    identity = result_identity(root)
    out = root/'report'; out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    seen = set(); input_hashes = {}; video_rows = []; scene_rows = []
    with (out/'per_video.jsonl.tmp').open('w') as raw:
        for item in manifest['tasks']:
            result = completed_result(root, item, identity)
            if result is None: raise ValueError('Full report refuses partial coverage: '+item['path'])
            if sha(root/item['path']) != item['sha256']: raise ValueError('Task changed')
            task = read(root/item['path'])
            rows = {(r['method'], r['seed']): r for r in result['rows']}
            expected = {(m, seed) for m in CANDIDATES for seed in task['seeds']}
            if set(rows) != expected or len(rows) != len(result['rows']): raise ValueError('Wrong scene coverage')
            matrix = np.empty((len(CANDIDATES), len(METRICS)), np.float64)
            exact_by_method = {}
            for mi, method in enumerate(CANDIDATES):
                values = []
                for seed in task['seeds']:
                    row = rows[(method, seed)]
                    key = (task['model'], method, task['case_id'], seed)
                    if key in seen: raise ValueError('Duplicate evaluation')
                    seen.add(key)
                    m = row['metrics']['video']
                    record = task['records'][method][str(seed)]
                    gold = task['records']['bf16'][str(seed)]
                    if (row['candidate_npz_sha256'] != record['npz_sha256'] or
                        row['reference_npz_sha256'] != gold['npz_sha256'] or
                        not row.get('decoded_hashes_verified')):
                        raise ValueError('Metric input provenance mismatch')
                    if any(len(row['per_frame'][k]) != task['case']['frames'] for k in ('lpips','ssim')):
                        raise ValueError('Frame sampling or incomplete frame metrics')
                    if (m['psnr_db'] is None) != m['psnr_exact']:
                        raise ValueError('Non-finite PSNR must represent exact equality')
                    values.append([np.inf if k=='psnr_db' and m['psnr_exact'] else m[k] for k in METRICS])
                    public = dict(model=task['model'], benchmark=task['benchmark'], category=task['category'],
                        case_id=task['case_id'], source_id=task['source_id'], method=method, seed=seed,
                        sample_id=row['sample_id'], reference_sample_id=row['reference_sample_id'],
                        **{k:m[k] for k in METRICS}, mse=m['mse'], psnr_exact=m['psnr_exact'],
                        candidate_npz=record['path'], reference_npz=gold['path'],
                        candidate_video=record['video_path'], reference_video=gold['video_path'])
                    video_rows.append(public)
                    raw.write(json.dumps(dict(**public, per_frame=row['per_frame'],
                        candidate_npz_sha256=record['npz_sha256'], reference_npz_sha256=gold['npz_sha256']), allow_nan=False)+'\n')
                matrix[mi] = np.mean(values, axis=0)
                exact_by_method[method] = sum(rows[(method,s)]['metrics']['video']['psnr_exact'] for s in task['seeds'])
                scene_rows.append(dict(model=task['model'], benchmark=task['benchmark'], category=task['category'],
                    case_id=task['case_id'], method=method, seeds=len(task['seeds']),
                    **{k:number(matrix[mi, ki]) for ki,k in enumerate(METRICS)},
                    psnr_exact_pairs=exact_by_method[method]))
            grouped[(task['model'],task['benchmark'])].append(dict(category=task['category'],
                case_id=task['case_id'], matrix=matrix, exact=exact_by_method, rows=rows))
            input_hashes[item['result']] = sha(root/item['result'])
    if len(seen) != manifest['expected_pairs']: raise ValueError('Full pair count mismatch')
    (out/'per_video.jsonl.tmp').replace(out/'per_video.jsonl')
    save_csv(out/'per_video.csv', video_rows)
    save_csv(out/'per_scene.csv', scene_rows)
    del video_rows
    tables, contrasts, category_rows = [], [], []
    summary = dict(identity=identity, created_at=now(), complete=False,
        reference='Same model/input/seed BF16 output, not a real-video reference',
        paired_videos=len(seen), bf16_reference_videos=6776, groups={}, performance_eligible=False,
        aggregation='Category macro is the primary historical summary; scene micro also reported.',
        statistics='Two seeds averaged within each scene; resample scenes within categories, jointly across all methods.',
        limitations=['Existing scenes include previously used calibration/development/validation scenes; no held-out or cross-scene generalization claim.',
            'AWQ is the ModelOpt NVFP4 W4A16 decoded BF16 quality reference, not a qualified packed performance result.',
            'ARCQuant and SharQ are the recorded Cosmos adaptations; see input_audit.json for adapter details.',
            'BF16 is a mathematical self-reference: LPIPS=0, SSIM=1, PSNR=+inf (JSON null); not a measured competing quality score.',
            'No latency, throughput or model-storage conclusions are drawn from this metric run.'])
    for gi, ((model, benchmark), scenes) in enumerate(sorted(grouped.items())):
        stats = manifest['models'][model]['statistics']
        replicates, seed = stats['replicates'], stats['bootstrap_seed']+gi
        cats = sorted({s['category'] for s in scenes})
        matrices = {c:np.stack([s['matrix'] for s in scenes if s['category']==c]) for c in cats}
        macro, micro = bootstrap(matrices, replicates, seed)
        point_macro = np.mean([m.mean(axis=0) for m in matrices.values()], axis=0)
        point_micro = np.mean([s['matrix'] for s in scenes], axis=0)
        for c, matrix in matrices.items():
            for mi, method in enumerate(CANDIDATES):
                category_rows.append(dict(model=model, benchmark=benchmark, category=c, method=method,
                    scenes=len(matrix), videos=2*len(matrix), **{k:number(matrix[:,mi,ki].mean()) for ki,k in enumerate(METRICS)}))
        summaries = {}
        for aggregation, draws, point in [('category_macro',macro,point_macro),('scene_micro',micro,point_micro)]:
            method_rows = {}
            for mi, method in enumerate(CANDIDATES):
                row = dict(model=model, benchmark=benchmark, aggregation=aggregation, method=method,
                    label=METHODS[method], scenes=len(scenes), videos=2*len(scenes), categories=len(cats))
                for ki,k in enumerate(METRICS):
                    low, high = interval(draws[:,mi,ki])
                    row.update({k:number(point[mi,ki]), k+'_ci95_low':low, k+'_ci95_high':high})
                row['psnr_exact_pairs'] = sum(s['exact'][method] for s in scenes)
                method_rows[method] = row; tables.append(row)
            summaries[aggregation] = method_rows
        summary['groups'][model+'/'+benchmark] = dict(scenes=len(scenes), videos_per_method=2*len(scenes),
            category_scene_counts={c:len(v) for c,v in matrices.items()}, bootstrap_replicates=replicates,
            bootstrap_seed=seed, summaries=summaries,
            bf16=dict(lpips=0, ssim_mean=1, psnr_db=None, psnr_exact=True, role='reference'))
        for method in ('b1.5_protected_n','b2_joint'):
            mi = CANDIDATES.index(method)
            for baseline in ('arcquant100','arcquant','sharq_perf','fp8','awq_reference','fp4_base'):
                bi = CANDIDATES.index(baseline)
                differences = macro[:,mi]-macro[:,bi]
                delta = point_macro[mi]-point_macro[bi]
                row = dict(model=model, benchmark=benchmark, method=method, baseline=baseline,
                    aggregation='category_macro', scenes=len(scenes), paired_videos=2*len(scenes),
                    planned_family=baseline in ('arcquant100','fp8'), family_size=stats['family_size'])
                for ki,k in enumerate(METRICS):
                    if np.isnan(differences[:,ki]).any():
                        row.update({k+'_difference':None,k+'_ci95_low':None,k+'_ci95_high':None})
                    else:
                        lo, hi = interval(differences[:,ki])
                        row.update({k+'_difference':number(delta[ki]),k+'_ci95_low':lo,k+'_ci95_high':hi})
                lo,hi=interval(differences[:,0],alpha=.05/stats['family_size'])
                row.update(lpips_familywise_low=lo,lpips_familywise_high=hi,
                    passes_lpips95_and_ssim_l2=bool(row['lpips_ci95_high']<0 and delta[2]>=0 and delta[3]<=0),
                    passes_familywise_and_ssim_l2=bool(hi<0 and delta[2]>=0 and delta[3]<=0))
                # Worst seed is descriptive, not an independent inference unit.
                worst = max(((s['rows'][(method,seed)]['metrics']['video']['lpips']-
                    s['rows'][(baseline,seed)]['metrics']['video']['lpips'],s['case_id'],seed)
                    for s in scenes for meth,seed in s['rows'] if meth==method), key=lambda v:v[0])
                row.update(worst_seed_lpips_degradation=worst[0], worst_case_id=worst[1], worst_seed=worst[2])
                contrasts.append(row)
        print(json.dumps(dict(event='report_group',model=model,benchmark=benchmark,scenes=len(scenes))),flush=True)
    save_csv(out/'summary.csv', tables); save_csv(out/'per_category.csv',category_rows)
    save_csv(out/'paired_comparisons.csv', contrasts)
    summary.update(complete=True, completed_at=now(), paired_comparisons=contrasts)
    write(out/'summary.json', summary)
    lines = ['# Full Edge/Nano BF16-consistency metrics', '',
        'Coverage: RBench 650 scenes and PAI-Bench 1,044 scenes, two original seeds per scene, '
        'both models and all nine generated methods. 60,984 source videos; 54,208 candidate–BF16 pairs.', '',
        'All 61 original float32 frames at 320×192 are evaluated. Generation used 35 steps. '
        'LPIPS ↓ (AlexNet 0.1), PSNR ↑ (range=1, clip-global MSE) and SSIM ↑ (7×7 uniform, RGB) '
        'measure consistency with the same-input/seed BF16 output, not physical correctness or task success.', '',
        'Tables below use the historical category-macro average: average the two seeds per scene, '
        'then average scenes within each category, then weight categories equally. '
        'Scene-micro averages and 95% scene-cluster bootstrap intervals are also in `summary.csv`. '
        'Each model/benchmark is reported separately.', '']
    for group, content in summary['groups'].items():
        lines += [f'## {group} — {content["scenes"]} scenes', '',
            '| Method | LPIPS ↓ | PSNR (dB) ↑ | SSIM ↑ |', '|---|---:|---:|---:|',
            '| BF16 reference | 0 (self) | reference | 1 (self) |']
        for method in CANDIDATES:
            row=content['summaries']['category_macro'][method]
            psnr = f'{row["psnr_db"]:.4f}' if row['psnr_db'] is not None else '+∞ (exact pair present)'
            lines.append(f'| {METHODS[method]} | {row["lpips"]:.6f} | {psnr} | {row["ssim_mean"]:.6f} |')
        lines.append('')
    lines += ['## Interpretation and provenance', '', *['- '+x for x in summary['limitations']], '',
        '- Paired comparisons keep both seeds and all methods together within each resampled scene. '
        'The original 100,000 bootstrap replicates and conservative Bonferroni family size 12 are preserved. '
        'ARCQuant 100% primary comparisons and FP8 gates are separate families. Other contrasts are exploratory.',
        '- PSNR is supplementary; the pre-existing FP8 gate still requires lower LPIPS with its confidence '
        'interval below zero, non-decreasing SSIM and non-increasing video relative L2.',
        '- `per_video.csv` / `per_video.jsonl`: all paired scores and original NPZ/video paths; '
        '`per_scene.csv` / `per_category.csv`: aggregation inputs; `paired_comparisons.csv`: paired deltas, '
        'confidence intervals and worst-seed regressions.',
        '- Frozen source manifests, metric versions, weights, GPU qualifications and input hashes '
        'are in the parent directory. BF16 self-reference is omitted as a finite PSNR value.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    write(out/'result_hashes.json',input_hashes)
    write(out/'complete.json',dict(complete=True, completed_at=now(), identity=identity,
        pairs=len(seen), scenes=len(manifest['tasks']), files={p.name:sha(p) for p in out.iterdir()
        if p.is_file() and p.name!='complete.json'}))
    print(json.dumps(dict(report_complete=True,root=str(out),pairs=len(seen))),flush=True)

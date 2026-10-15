"""Full-frame paired Super metrics, using the same qualified metric definitions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    from super_quality_campaign import check_campaign, METHODS
    from super_runtime import pair_worker, admit
    from full_video_metrics import VideoMetrics, load_video, environment
    from thunderworld_fp4.cosmos.quality import compare
    root = a.root.resolve()
    check_campaign(root)
    manifest = root / 'super/final/manifest.json'
    study = json.loads(manifest.read_text())
    out = root / 'metrics'
    out.mkdir(exist_ok=True)
    identity = dict(manifest_sha256=sha(manifest), source_sha256=sha(__file__),
                    backend_sha256=sha(Path(__file__).with_name('full_video_metrics.py')), environment=environment())
    path = out / 'progress.json'
    if path.exists() and json.loads(path.read_text())['identity'] != identity:
        raise ValueError('Metric identity changed')
    state = dict(identity=identity, complete=False, completed_pairs=0,
        expected_pairs=1694*2*8, performance_eligible=False)
    write(path, state)
    with pair_worker():
        resource = admit(minimum_mib=12288)
        evaluator = VideoMetrics(device='cuda:0', threads=4)
        qualification = dict(complete=False, identity=identity, checks=[], resource=resource)

        def qualify(label, x, y):
            evaluator.set_reference(y)
            actual = evaluator.compare(x)['video']
            reference = compare({'video': x}, {'video': y}, evaluator.model)['video']
            differences = {}
            for name, tolerance in dict(lpips=1e-6, ssim_mean=1e-6, psnr_db=1e-10,
                                       relative_l2=1e-12, mse=1e-12, rmse=1e-12).items():
                v, r = actual[name], reference[name]
                if v is None or r is None:
                    if v != r:
                        raise ValueError('Exact-match PSNR mismatch')
                    differences[name] = 0
                else:
                    differences[name] = abs(v-r)
                    if differences[name] > tolerance:
                        raise ValueError(f'Independent metric mismatch: {label}/{name}')
            qualification['checks'].append(dict(label=label, max_abs_difference=differences))

        rng = np.random.default_rng(20260928)
        z = np.zeros((9, 64, 64, 3), dtype=np.float32)
        r = rng.random(z.shape, dtype=np.float32)
        for label, x, y in [('zeros', z, z), ('identical', r, r), ('constants', z+.5, z+.51),
                            ('opposites', z, z+1), ('random', r, rng.random(z.shape, dtype=np.float32)),
                            ('small_noise', r, np.clip(r+1e-4, 0, 1))]:
            qualify(label, x, y)

        def record(method, entry, seed):
            folder = root / 'super/final/generation' / method / entry['case']['id']
            receipt_path = folder / 'result.json'
            receipt = json.loads(receipt_path.read_text())
            if (not receipt['complete'] or receipt['identity']['manifest_sha256'] != identity['manifest_sha256']
                or receipt['identity']['case'] != entry['case'] or receipt['identity']['seeds'] != entry['seeds']):
                raise ValueError('Mismatched generation record')
            row = next(r for r in receipt['rows'] if r['seed'] == seed)
            return dict(path=str(folder / row['outputs']), npz_sha256=row['outputs_sha256'],
                video_sha256=row['sha256']['video'], receipt_sha256=sha(receipt_path))

        summaries = defaultdict(list)
        real_qualified = set()
        for entry in study['cases']:
            cid = entry['case']['id']
            target = out / 'scenes' / (cid + '.json')
            result = dict(complete=False, identity=identity, benchmark=entry['benchmark'],
                category=entry['category'], case_id=cid, rows=[])
            if target.exists():
                result = json.loads(target.read_text())
                if result['identity'] != identity:
                    raise ValueError('Changed metric scene')
            for seed in entry['seeds']:
                gold_record = record('bf16', entry, seed)
                needed = [m for m in METHODS if m != 'bf16' and not any(r['method']==m and r['seed']==seed for r in result['rows'])]
                if needed:
                    gold = load_video(gold_record)
                    evaluator.set_reference(gold)
                for method in needed:
                    row = record(method, entry, seed)
                    video = load_video(row)
                    if entry['benchmark'] not in real_qualified:
                        qualify(entry['benchmark']+'/'+method, video, gold)
                        real_qualified.add(entry['benchmark'])
                        evaluator.set_reference(gold)
                    metrics = evaluator.compare(video)
                    result['rows'].append(dict(method=method, seed=seed, metrics=metrics['video'],
                        per_frame=metrics['per_frame'], candidate=row, reference=gold_record))
                    write(target, result)
            if len(result['rows']) != 16 or len({(r['method'], r['seed']) for r in result['rows']}) != 16:
                raise ValueError('Incomplete or duplicate paired metric scene')
            for row in result['rows']:
                if row['candidate'] != record(row['method'], entry, row['seed']) or row['reference'] != record('bf16', entry, row['seed']):
                    raise ValueError('Changed metric input on resume')
                summaries[(entry['benchmark'], row['method'])].append(row['metrics'])
            result['complete'] = True
            write(target, result)
            state['completed_pairs'] += len(result['rows'])
            write(path, state)
            print(json.dumps(dict(case=cid, completed_pairs=state['completed_pairs'], expected_pairs=state['expected_pairs'])), flush=True)
        qualification['complete'] = True
        write(out / 'qualification.json', qualification)
        summary = []
        for (benchmark, method), rows in sorted(summaries.items()):
            item = dict(benchmark=benchmark, method=method, videos=len(rows))
            for name in ('lpips', 'ssim_mean', 'psnr_db', 'relative_l2'):
                finite = [r[name] for r in rows if r[name] is not None]
                item[name] = float(np.mean(finite)) if finite else None
                if name == 'psnr_db':
                    item['psnr_exact_videos'] = sum(r['psnr_exact'] for r in rows)
            summary.append(item)
        write(out / 'summary.json', dict(complete=True, identity=identity, reference='same-model BF16',
            aggregation='mean over all paired videos; category counts and raw scene rows retained',
            rows=summary, performance_eligible=False))
        lines = ['# Cosmos3-Super full generation consistency', '',
            'Reference: Super BF16, matched scene and seed. All 61 frames, native 320×192. '
            'These are consistency metrics, not real-world task success or latency evidence.', '',
            '| Benchmark | Method | Videos | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in summary:
            psnr = 'exact' if r['psnr_db'] is None else f"{r['psnr_db']:.4f}"
            lines.append(f"| {r['benchmark']} | {r['method']} | {r['videos']} | {r['lpips']:.6f} | {r['ssim_mean']:.6f} | {psnr} | {r['relative_l2']:.6f} |")
        lines.extend(['', 'BF16 self-comparison is not assigned a finite PSNR. '
            'Finite PSNR means omit exact matches; their counts are separately recorded in summary.json.',
            'No selection or calibration uses these full-corpus output scores.'])
        (out / 'report.md').write_text('\n'.join(lines)+'\n')
        state['complete'] = state['completed_pairs'] == state['expected_pairs']
        write(path, state)


if __name__ == '__main__':
    main()

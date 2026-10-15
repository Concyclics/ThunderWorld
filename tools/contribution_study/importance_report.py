"""Report paired development quality, actual budgets and storage without speed claims."""
import argparse
import collections
import gzip
import json
from pathlib import Path

from baselines import sha, write
from dataset_report import average


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    manifest = json.loads((a.run / 'manifest.json').read_text())
    progress = json.loads((a.run / 'metrics/progress.json').read_text())
    if not progress['complete'] or progress['identity']['manifest_sha256'] != sha(a.run / 'manifest.json'):
        raise ValueError('Complete matched metrics required')
    fields = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')
    summaries, raw, costs = {}, [], {}
    for benchmark in ('rbench', 'paibench'):
        summaries[benchmark] = {}
        for name in manifest['methods']:
            if name == 'bf16':
                continue
            rows = []
            for entry in manifest['cases']:
                if entry['benchmark'] != benchmark:
                    continue
                file = a.run / 'metrics' / name / (entry['case']['id'] + '.json')
                result = json.loads(file.read_text())
                if not result['complete'] or result['provenance']['identity'] != progress['identity']:
                    raise ValueError('Changed metric implementation')
                for side, method in (('candidate', name), ('reference', 'bf16')):
                    receipt = a.run / 'generation' / method / entry['case']['id'] / 'result.json'
                    if result['provenance'][side + '_sha256'] != sha(receipt):
                        raise ValueError('Changed generation')
                rows.append({key: average([r['metrics']['video'][key] for r in result['rows']]) for key in fields})
                raw.append(result)
            summaries[benchmark][name] = {key: average([r[key] for r in rows]) for key in fields}
    for name, method in manifest['methods'].items():
        ratios, allocated, reserved = [], [], []
        representative = None
        for entry in manifest['cases']:
            folder = a.run / 'generation' / name / entry['case']['id']
            receipt = json.loads((folder / 'result.json').read_text())
            if not receipt['complete']:
                raise ValueError('Incomplete request')
            for row in receipt['rows']:
                if row['performance_eligible'] or row['debug'] != 'off':
                    raise ValueError('Unexpected measurement conditions')
                if row.get('actual_mma_ratio') is not None:
                    if row['actual_mma_ratio'] > method['config']['compute_budget']:
                        raise ValueError('Exceeded actual request budget')
                    ratios.append(row['actual_mma_ratio'])
                allocated.append(row['peak_allocated_bytes'])
                reserved.append(row['peak_reserved_bytes'])
            if (folder / 'compute.json').exists():
                if any(r['compute_sha256'] != sha(folder / 'compute.json') for r in receipt['rows']):
                    raise ValueError('Changed cost ledger')
                if representative is None:
                    ledger = json.loads((folder / 'compute.json').read_text())
                    representative = {k: v for k, v in ledger.items() if k not in ('rows', 'w1_support_union_blocks')}
        costs[name] = dict(mma_ratio_range=[min(ratios), max(ratios)] if ratios else None,
                           peak_allocated_bytes=[min(allocated), max(allocated)], peak_reserved_bytes=[min(reserved), max(reserved)],
                           offline_cost=method.get('cost'), representative_request_cost=representative)
    pairs = {}
    by_seed = collections.defaultdict(dict)
    for result in raw:
        for row in result['rows']:
            by_seed[result['benchmark'], result['name']][result['case_id'], row['seed']] = row['metrics']['video']
    for (benchmark, name), values in by_seed.items():
        references = ['fp8']
        if name == 'linear_a_only':
            references.append('linear_w_only')
        if name.startswith('b'):
            cap = name.split('_')[0]
            references.append({'b1.5': 'b1.5_protected_n', 'b2': 'b2_protected_n', 'b2.5': 'b2.5_ffn_aware'}.get(cap, 'b2.5_marginal'))
        for reference in references:
            if name == reference or (benchmark, reference) not in by_seed:
                continue
            other = by_seed[benchmark, reference]
            if values.keys() != other.keys():
                raise ValueError('Unmatched metric pairs')
            comparisons = {}
            for field in fields:
                if any(v[field] is None or other[k][field] is None for k, v in values.items()):
                    comparisons[field] = dict(exact_match_psnr_present=True)
                    continue
                diffs = [(k, v[field] - other[k][field]) for k, v in values.items()]
                low = field in ('lpips', 'relative_l2')
                key, worst = (max if low else min)(diffs, key=lambda r: r[1])
                comparisons[field] = dict(mean_delta=average([v for _, v in diffs]), worst_delta=worst,
                                          worst_case=key[0], worst_seed=key[1],
                                          degraded_seeds=sum(v > 0 if low else v < 0 for _, v in diffs), seeds=len(diffs))
            pairs[f'{benchmark}|{name}|{reference}'] = comparisons
    result = dict(complete=True, manifest_sha256=sha(a.run / 'manifest.json'), summaries=summaries, costs=costs,
                  pairs=pairs, seed_metrics=raw, performance_eligible=False, final_quality_qualified=False,
                  scope='16 development scenes (9 RBench, 7 PAI), two seeds each, one scene/category. Descriptive paired means only; validation/final untouched.')
    report = a.run / 'report'
    report.mkdir(exist_ok=True)
    write(report / 'quality.json', result)
    lines = ['# 激活重要性与混合修复开发实验', '', result['scope'], '',
             '指标均相对同输入 BF16。linear_base/w_only/a_only/three 四个归因配置固定 BF16 attention，线性层分别执行 1/2/2/3 项 FP4 MMA；它们不是联合 FP4 预算候选。带预算的候选均使用 FP4 attention，并计入当前请求的完整预算。']
    for benchmark, methods in summaries.items():
        lines += ['', f'## {benchmark}', '', '| 配置 | LPIPS ↓ | SSIM ↑ | L2 ↓ | PSNR ↑ |', '|---|---:|---:|---:|---:|']
        for name, value in methods.items():
            psnr = f'{value["psnr_db"]:.3f}' if value['psnr_db'] is not None else '+∞'
            lines.append(f'| {name} | {value["lpips"]:.6f} | {value["ssim_mean"]:.6f} | {value["relative_l2"]:.6f} | {psnr} |')
    lines += ['', '## 实际预算和存储', '', '| 配置 | 请求 MMA/C0 范围 | W1 字节 | allocated 峰值 GiB 范围 |', '|---|---:|---:|---:|']
    for name, cost in costs.items():
        ratio = cost['mma_ratio_range']
        mma = '—' if ratio is None else f'{ratio[0]:.6f}–{ratio[1]:.6f}'
        w1 = (cost['representative_request_cost'] or {}).get('w1_support_union_bytes', '—')
        memory = '–'.join(f'{v / 2**30:.3f}' for v in cost['peak_allocated_bytes'])
        lines.append(f'| {name} | {mma} | {w1} | {memory} |')
    lines += ['', '无共享 GPU 加速结论。非 MMA 工作、packing 流量模型、FP4/BF16 MMA 和 reserved 峰值在原始 JSON 中单列；W1 字节不等于整个模型大小。']
    (report / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    a.archive.mkdir(parents=True, exist_ok=True)
    (a.archive / (a.label.upper() + '.md')).write_text('\n'.join(lines) + '\n')
    for label, file in [('quality', report / 'quality.json'), ('manifest', a.run / 'manifest.json')]:
        (a.archive / f'{a.label}_{label}.json.gz').write_bytes(gzip.compress(file.read_bytes(), mtime=0))
    print(json.dumps(dict(complete=True, report=str(a.archive / (a.label.upper() + '.md')))))


if __name__ == '__main__':
    main()

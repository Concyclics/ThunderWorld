"""Post-hoc three-versus-four-term evidence audit; never a new selection gate."""
import argparse
import gzip
import json
from pathlib import Path

from awq_multiscene_report import paired_contrast
from baselines import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(gzip.decompress(args.quality.read_bytes()))
    if not data['complete']:
        raise ValueError('Complete existing cohort required')
    statistics = dict(replicates=10000, bootstrap_seed=180918, family_size=2)
    comparisons = {}
    for benchmark in ('rbench', 'paibench'):
        methods = {name: {} for name in ('fp4_three', 'fp4_four')}
        categories = {}
        for record in data['seed_metrics']:
            if record['benchmark'] != benchmark or record['name'] not in methods:
                continue
            if not record['complete'] or len(record['rows']) != 2:
                raise ValueError('Expected paired two-seed scenes')
            case = record['case_id']
            if categories.setdefault(case, record['category']) != record['category']:
                raise ValueError('Changed scene stratum')
            for row in record['rows']:
                key = case, row['seed']
                if key in methods[record['name']]:
                    raise ValueError('Duplicate scene/seed')
                methods[record['name']][key] = row['metrics']['video']
        if len(categories) != 36 or any(len(value) != 72 for value in methods.values()):
            raise ValueError('Expected 36 scenes / 72 videos per method and dataset')
        paired = paired_contrast(methods['fp4_three'], methods['fp4_four'], categories, statistics)
        # Do not repurpose the helper's superiority gate into a post-hoc promotion.
        comparisons[benchmark] = dict(candidate='fp4_three', reference='fp4_four', metrics=paired['metrics'],
            candidate_mean=data['summaries'][benchmark]['fp4_three']['category_macro'],
            reference_mean=data['summaries'][benchmark]['fp4_four']['category_macro'])
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(complete=True, comparisons=comparisons, statistics=statistics,
        source=dict(path=str(args.quality.resolve()), sha256=sha(args.quality),
                    manifest_sha256=data['manifest_sha256']),
        implementation={name: sha(Path(__file__).with_name(name)) for name in
                        ('paper_cross_audit.py', 'awq_multiscene_report.py', 'dataset_report.py')},
        selection_eligible=False, performance_eligible=False,
        scope='Exploratory post-hoc contrasts on the already evaluated 72-scene cohort. '
              'Three minus four terms; both use native FP4 linears and attention. '
              'Two LPIPS contrasts have Bonferroni intervals; other metrics are supplementary. '
              'No predeclared noninferiority margin, new generation, or latency measurement.')
    write(args.output / 'cross_term.json', result)
    lines = ['# 三项与四项：既有72场景的直接配对复核', '', result['scope'], '',
        '| 数据集 | 指标 | 三项−四项均值 | 配对95%区间 | 两项校正区间 |',
        '|---|---|---:|---|---|']
    for benchmark, row in comparisons.items():
        for field in ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db'):
            value = row['metrics'][field]
            lines.append(f'| {benchmark} | {field} | {value["mean_delta"]:.6f} | '
                         f'{value["ci95"]} | {value["familywise_ci"] if field == "lpips" else "补充指标"} |')
    lines += ['', '三项和四项都超过FP8的历史结论不等于三项与四项等价。本复核没有预注册非劣界限，不能事后将差异描述为无损。',
        '差异同时涉及投影、FFN、QK和PV的交叉项，不能归因于某一种模块。按模块隔离的交叉项消融仍需补做。']
    (args.output / 'CROSS_TERM.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({name: {field: {key: row['metrics'][field][key]
                         for key in ('mean_delta', 'ci95', 'familywise_ci')}
                         for field in ('lpips', 'psnr_db')}
                      for name, row in comparisons.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Predeclared development gates for the fixed-support representation study.

These descriptive gates authorize a later, frozen validation run. They are not
significance tests or evidence of held-out quality. Both datasets must pass.
"""
import argparse
import gzip
import json
from pathlib import Path

from baselines import sha, write

FIELDS = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')
DATASETS = ('rbench', 'paibench')


def mean_guard(candidate, reference):
    return [field for field, failed in (
        ('lpips', candidate['lpips'] >= reference['lpips']),
        ('ssim_mean', candidate['ssim_mean'] < reference['ssim_mean']),
        ('relative_l2', candidate['relative_l2'] > reference['relative_l2']),
    ) if failed]


def paired_rows(candidate, reference):
    if not candidate or candidate.keys() != reference.keys():
        raise ValueError('All paired scene/seed outputs are required')
    detail = {}
    for field in FIELDS:
        deltas = [(key, value[field]-reference[key][field]) for key, value in candidate.items()]
        lower = field in ('lpips', 'relative_l2')
        key, worst = (max if lower else min)(deltas, key=lambda x: x[1])
        detail[field] = dict(mean_delta=sum(v for _, v in deltas)/len(deltas),
            worst_delta=worst, worst_case=key[0], worst_seed=key[1],
            degraded_seeds=sum(v > 0 if lower else v < 0 for _, v in deltas), seeds=len(deltas))
    return detail


def report(run, output):
    manifest = json.loads((run/'manifest.json').read_text())
    quality_file = run/'report/quality.json'
    quality = json.loads(quality_file.read_text())
    if (manifest['role'] != 'development' or len(manifest['cases']) != 16 or
        not quality['complete'] or quality['manifest_sha256'] != sha(run/'manifest.json')):
        raise ValueError('Complete frozen sixteen-scene development study required')
    seeds = {}
    for row in quality['seed_metrics']:
        key = row['benchmark'], row['name']
        values = seeds.setdefault(key, {})
        for item in row['rows']:
            sample = row['case_id'], item['seed']
            if sample in values:
                raise ValueError('Duplicate scene/seed metric')
            values[sample] = item['metrics']['video']
    for dataset in DATASETS:
        expected = {(r['case']['id'], seed) for r in manifest['cases']
                    if r['benchmark'] == dataset for seed in r['seeds']}
        for name in manifest['methods']:
            if name != 'bf16' and seeds.get((dataset, name), {}).keys() != expected:
                raise ValueError('Missing scene/seed metrics')
    comparisons, selections = {}, {}
    for pair in manifest['primary_comparisons']:
        name, reference = pair['candidate'], pair['reference']
        spec, control = manifest['methods'][name], manifest['methods'][reference]
        parent = spec['origin']
        if spec['cost']['tensor_bytes'] != control['cost']['tensor_bytes']:
            raise ValueError('Representation comparison changed tensor storage')
        for entry in manifest['cases']:
            case = entry['case']['id']
            ledgers = []
            for method in (name, reference):
                folder = run/'generation'/method/case
                receipt = json.loads((folder/'result.json').read_text())
                ledger = json.loads((folder/'compute.json').read_text())
                if (not receipt['complete'] or not ledger['feasible'] or
                    ledger['ratio'] > pair['budget'] or
                    any(r['compute_sha256'] != sha(folder/'compute.json') for r in receipt['rows'])):
                    raise ValueError('Incomplete, changed or over-budget request')
                ledgers.append(ledger)
            for field in ('base_mma', 'total_mma', 'mma_by_dtype', 'non_mma_work',
                          'packing_traffic', 'w1_support_union_bytes', 'w1_support_union_blocks'):
                if ledgers[0][field] != ledgers[1][field]:
                    raise ValueError('Unmatched execution/support: '+field)
        entry = dict(budget=pair['budget'], reference=reference, parent=parent,
                     cost_and_tensor_storage_matched=True, comparisons={})
        failures = {}
        for other in (reference, parent, 'fp8'):
            failures[other] = {}
            for dataset in DATASETS:
                entry['comparisons'][dataset+'|'+other] = paired_rows(seeds[dataset, name], seeds[dataset, other])
                failures[other][dataset] = mean_guard(quality['summaries'][dataset][name], quality['summaries'][dataset][other])
        entry.update(mean_guard_failures=failures,
            improves_matched_rtn=not any(failures[reference].values()),
            improves_existing_parent=not any(failures[parent].values()),
            mean_exceeds_fp8=not any(failures['fp8'].values()),
            validation_eligible=not any(failures[reference].values()) and not any(failures[parent].values()),
            statistical_quality_qualified=False)
        comparisons[name] = entry
    for cap in sorted({r['budget'] for r in comparisons.values()}):
        eligible = [name for name, item in comparisons.items() if item['budget'] == cap and item['validation_eligible']]
        # Candidate count is bounded in advance; do not tune again on validation.
        selections[str(cap)] = dict(candidates=eligible, count=len(eligible),
            action='Freeze candidate assets and run reserved validation' if eligible else 'Stop this mechanism; keep the existing parent',
            final_split_consumed=False)
    result = dict(complete=True, comparisons=comparisons, selections=selections,
        manifest_sha256=sha(run/'manifest.json'), quality_sha256=sha(quality_file), source_sha256=sha(__file__),
        performance_eligible=False, statistics=None,
        scope='Predeclared descriptive development guards only, 16 scenes x 2 seeds. '
              'LPIPS must improve and SSIM/L2 must not regress on either dataset versus both matched RTN and existing parent. '
              'PSNR and worst-seed changes are reported, not substituted for failed guards. Validation and final remain unconsumed.')
    output.mkdir(parents=True, exist_ok=True)
    write(output/'gate.json', result)
    lines = ['# 固定预算表示优化：完整生成验收', '', result['scope'], '',
        '| 候选 | 相对 RTN 的 R/P LPIPS 差值 | 相对 RTN 通过 | 相对原方案通过 | 进入验证 |',
        '|---|---:|---|---|---|']
    for name, item in comparisons.items():
        delta = ' / '.join(f"{item['comparisons'][b+'|'+item['reference']]['lpips']['mean_delta']:+.6f}" for b in DATASETS)
        lines.append(f"| {name} | {delta} | {item['improves_matched_rtn']} | {item['improves_existing_parent']} | {item['validation_eligible']} |")
    lines += ['', '完整 LPIPS/SSIM/L2/PSNR、退化种子比例及最差种子差值见 gate.json；原始逐视频指标见压缩文件。',
              '失败候选保留，不通过更换指标或继续查看保留集寻找正结果。']
    (output/'GATE.md').write_text('\n'.join(lines)+'\n')
    (output/'quality.json.gz').write_bytes(gzip.compress(quality_file.read_bytes(), mtime=0))
    print(json.dumps(selections))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report(a.run, a.output)


if __name__ == '__main__':
    main()

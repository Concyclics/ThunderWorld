"""Report matched FP4-attention comparisons without claiming a latency frontier."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from aligned_sides import NEW_METHODS
from baselines import sha, write

FIELDS = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    manifest = json.loads((root / 'manifest.json').read_text())
    progress = json.loads((root / 'metrics/progress.json').read_text())
    if not json.loads((root / 'workflow.json').read_text())['complete'] or not progress['complete']:
        raise ValueError('Complete generation and metrics required')
    if progress['identity']['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('Changed cohort/configuration')
    values = collections.defaultdict(dict)
    summaries, costs, raw, receipts, ledgers = {}, {}, [], [], []
    base_by_case = {}
    prepared = {}
    for method in NEW_METHODS:
        file = root / 'generation' / method / 'prepared.json'
        prepared[method] = json.loads(file.read_text())
        if prepared[method]['identity']['manifest_sha256'] != sha(root / 'manifest.json'):
            raise ValueError('Changed preparation identity')
    sources = prepared[NEW_METHODS[0]]['runtime']['source_sha256']
    if any(value['runtime']['source_sha256'] != sources for value in prepared.values()):
        raise ValueError('New methods used different runtime implementations')
    for method, spec in manifest['methods'].items():
        ratios, memory, reserved, wbytes = [], [], [], []
        request_costs = []
        for entry in manifest['cases']:
            name = entry['case']['id']
            folder = root / 'generation' / method / name
            receipt = json.loads((folder / 'result.json').read_text())
            if not receipt['complete'] or receipt['identity']['case'] != entry['case']:
                raise ValueError('Incomplete or changed request')
            if {row['seed'] for row in receipt['rows']} != set(entry['seeds']):
                raise ValueError('Changed request seeds')
            for row in receipt['rows']:
                if row['steps'] != 35 or row['debug'] != 'off' or row['execution'] != 'graph' or row['performance_eligible']:
                    raise ValueError('Changed execution conditions')
                memory.append(row['peak_allocated_bytes'])
                reserved.append(row['peak_reserved_bytes'])
            receipts.append(dict(method=method, case=name, receipt=receipt,
                                 path=str(folder / 'result.json'), sha256=sha(folder / 'result.json')))
            if (folder / 'compute.json').exists():
                file = folder / 'compute.json'
                if any(row['compute_sha256'] != sha(file) for row in receipt['rows']):
                    raise ValueError('Changed compute ledger')
                ledger = json.loads(file.read_text())
                if not ledger['feasible'] or ledger['ratio'] > spec['config']['compute_budget']:
                    raise ValueError('Exceeded current-request budget')
                if method in NEW_METHODS:
                    for row in ledger['rows']:
                        if row['kind'] == 'attention':
                            terms = row['mma_by_dtype']['fp4']
                            if (row['total_mma'] != 2 * row['base_mma'] or
                                    any(terms[side + '_' + term] for side in ('qk', 'pv') for term in ('w', 'cross'))):
                                raise ValueError('Common attention is not full ACT2')
                    if method.startswith('single_') and ledger['total_mma'] != 2 * ledger['base_mma']:
                        raise ValueError('Single-sided baseline is not exactly 2x')
                prior = base_by_case.setdefault(name, ledger['base_mma'])
                if prior != ledger['base_mma']:
                    raise ValueError('Compared different base MMA denominators')
                ratios.append(ledger['ratio'])
                wbytes.append(ledger['w1_support_union_bytes'])
                request_costs.append({key: value for key, value in ledger.items() if key not in ('rows', 'w1_support_union_blocks')})
                ledgers.append(dict(method=method, case=name, ledger=ledger, sha256=sha(file)))
            if method == 'bf16':
                continue
            result = json.loads((root / 'metrics' / method / (name + '.json')).read_text())
            if not result['complete'] or result['provenance']['identity'] != progress['identity']:
                raise ValueError('Changed metric implementation')
            if result['provenance']['candidate_sha256'] != sha(folder / 'result.json') or result['provenance']['reference_sha256'] != sha(root / 'generation/bf16' / name / 'result.json'):
                raise ValueError('Changed generated/reference outputs')
            if {row['seed'] for row in result['rows']} != set(entry['seeds']) or len(result['rows']) != len(entry['seeds']):
                raise ValueError('Unpaired metric rows')
            raw.append(result)
            for row in result['rows']:
                metrics = row['metrics']['video']
                if not all(np.isfinite(metrics[field]) for field in FIELDS):
                    raise ValueError('Nonfinite quality metric')
                values[entry['benchmark'], method][name, row['seed']] = metrics
        costs[method] = dict(mma_ratio_range=[min(ratios), max(ratios)] if ratios else None,
            w1_support_bytes=sorted(set(wbytes)), allocated_peak_range=[min(memory), max(memory)],
            reserved_peak_range=[min(reserved), max(reserved)], request_costs=request_costs)
    for (benchmark, method), rows in values.items():
        summaries.setdefault(benchmark, {})[method] = {
            field: float(np.mean([row[field] for row in rows.values()])) for field in FIELDS}
    comparisons = list(manifest['primary_comparisons'])
    comparisons += [dict(candidate=target, reference=source) for target, source in (
        ('single_a_act2', 'linear_a_only'), ('single_w_act2', 'linear_w_only'),
        ('joint2_act2', 'b2_joint'), ('joint1.5_act2', 'b1.5_protected_n'),
        ('b1.5_protected_n', 'b2_joint'), ('single_a_act2', 'single_w_act2'))]
    pairs, decisions = {}, {}
    for comparison in comparisons:
        candidate, reference = comparison['candidate'], comparison['reference']
        checks = {}
        for benchmark in ('rbench', 'paibench'):
            left, right = values[benchmark, candidate], values[benchmark, reference]
            if left.keys() != right.keys():
                raise ValueError('Unmatched paired videos')
            detail = {}
            for field in FIELDS:
                diffs = [(key, row[field] - right[key][field]) for key, row in left.items()]
                lower = field in ('lpips', 'relative_l2')
                key, worst = (max if lower else min)(diffs, key=lambda row: row[1])
                old = summaries[benchmark][reference][field]
                new = summaries[benchmark][candidate][field]
                detail[field] = dict(mean_delta=new - old, relative_change=(new / old - 1) if old else None,
                    worst_case=key[0], worst_seed=key[1], worst_delta=worst,
                    degraded_videos=sum(value > 0 if lower else value < 0 for _, value in diffs), videos=len(diffs))
            pairs[f'{benchmark}|{candidate}|{reference}'] = detail
            checks[benchmark] = dict(mean_guard=detail['lpips']['mean_delta'] < 0 and
                detail['ssim_mean']['mean_delta'] >= 0 and detail['relative_l2']['mean_delta'] <= 0,
                all_four_means_improve=all(detail[field]['mean_delta'] < 0 if field in ('lpips', 'relative_l2')
                    else detail[field]['mean_delta'] > 0 for field in FIELDS))
        decisions[candidate + '|' + reference] = dict(by_dataset=checks,
            both_datasets_mean_guard=all(row['mean_guard'] for row in checks.values()))
    result = dict(complete=True, manifest_sha256=sha(root / 'manifest.json'), summaries=summaries,
        costs=costs, pairs=pairs, decisions=decisions, raw=raw, performance_eligible=False,
        scope='Descriptive paired means on 16 development scenes / 32 videos per method; one scene/category. '
              'No significance or independent-validation claim. Identical whole-request C0. Budget is a cap, not equal realized MMA. '
              'Joint policies also use different offline W0/W1 calibration; common attention alone does not isolate tile allocation.')
    report = root / 'report'
    report.mkdir(exist_ok=True)
    write(report / 'quality.json', result)
    lines = ['# 单侧修复与联合修复：对齐 FP4 attention', '',
        '16 个开发场景（RBench 9、PAI-Bench 7），每场景两个固定种子。Edge I2V，35 steps，CFG 6，Graph/debug off。指标衡量相对同输入 BF16 的一致性；不是任务成功率。', '',
        'single_a/w 表示生成投影和 FFN 的单侧修复。四个新配置都采用全量 ACT2 attention：Q0K0+Q1K0，P0V0+P1V0；K/V 是激活，不称为模型权重。', '',
        'single_a_act2、single_w_act2 的整请求 MMA 都严格为 2×；联合方案使用相同上限或更低上限，实际 MMA 单独列出。保留原联合 attention 和旧 BF16 attention 结果用于归因。']
    for benchmark, methods in summaries.items():
        lines += ['', '## ' + benchmark, '', '| 配置 | LPIPS ↓ | SSIM ↑ | L2 ↓ | PSNR dB ↑ |', '|---|---:|---:|---:|---:|']
        for name, metrics in methods.items():
            lines.append('| ' + name + ' | ' + ' | '.join(f'{metrics[field]:.6f}' for field in FIELDS) + ' |')
    lines += ['', '## 同预算上限比较', '', '| 候选 | 对照 | RBench 均值门槛 | PAI 均值门槛 | 两边均通过 |', '|---|---|---|---|---|']
    for pair in manifest['primary_comparisons']:
        key = pair['candidate'] + '|' + pair['reference']
        decision = decisions[key]
        states = [decision['by_dataset'][b]['mean_guard'] for b in ('rbench', 'paibench')]
        lines.append(f'| {pair["candidate"]} | {pair["reference"]} | ' + ' | '.join(
            '通过' if value else '未通过' for value in states + [decision['both_datasets_mean_guard']]) + ' |')
    lines += ['', '门槛为 LPIPS 下降且 SSIM/L2 均值不退化；PSNR 与最差种子另报。开发集没有独立显著性检验，不根据这些结果重新搜索或切换每场景配置。', '',
        '## 计算与显存', '', '| 配置 | 请求 MMA/C0 | W1 payload MiB | allocated 峰值 GiB |', '|---|---:|---:|---:|']
    for method, row in costs.items():
        mma = '—' if not row['mma_ratio_range'] else '–'.join(f'{v:.6f}' for v in row['mma_ratio_range'])
        storage = '—' if not row['w1_support_bytes'] else ', '.join(f'{v / 2**20:.3f}' for v in row['w1_support_bytes'])
        memory = '–'.join(f'{v / 2**30:.3f}' for v in row['allocated_peak_range'])
        lines.append(f'| {method} | {mma} | {storage} | {memory} |')
    lines += ['', 'W1 payload 不等于整个模型大小。分类型 MMA、padding、非 MMA、packing、reserved 显存和各请求账目均保留。共享 PRO 6000 的耗时不用于加速结论。', '',
        '共同 ACT2 attention 的比较保留了原联合方案的线性配额与离线资产，因此还存在 W0 校准和 W1 支持的差异，不能把收益全部归因于在线 tile 分配。']
    (report / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    args.archive.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report / 'REPORT.md', args.archive / 'RESULTS.md')
    for name, value in [('quality', result), ('manifest', manifest), ('generation_index', receipts), ('compute_ledgers', ledgers)]:
        (args.archive / (name + '.json.gz')).write_bytes(gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0))
    for name in ('probe.json', 'budget_checks.json'):
        shutil.copyfile(root / name, args.archive / name)
    for path in (root / 'policies').glob('*/plan.json'):
        target = args.archive / 'policies' / path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        (target / 'plan.json.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    write(args.archive / 'prepared.json', prepared)
    repo = Path(__file__).resolve().parents[2]
    package = repo / 'packages/fp4-repair/src/thunderworld_fp4'
    archived_sources = {}
    for relative, digest in sources.items():
        source = package / relative
        if sha(source) != digest:
            raise ValueError('Runtime source changed after generation: ' + relative)
        destination = args.archive / 'source' / source.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        archived_sources[str(source.relative_to(repo))] = digest
    for name in ('aligned_sides.py', 'aligned_sides_probe.py', 'aligned_sides_report.py',
                 'dataset_suite.py', 'dataset_infer.py', 'dataset_metrics.py', 'importance_workflow.py',
                 'baselines.py', 'schedule_envelope.py', 'schedule_common.py'):
        source = Path(__file__).with_name(name)
        destination = args.archive / 'source/tools/contribution_study' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        archived_sources[str(source.relative_to(repo))] = sha(source)
    write(args.archive / 'implementation.json', dict(complete=True, sources=archived_sources,
        scenarios=len(manifest['cases']), methods=len(manifest['methods']),
        videos=sum(len(row['receipt']['rows']) for row in receipts),
        new_videos=sum(len(row['receipt']['rows']) for row in receipts if row['method'] in NEW_METHODS),
        validation_run=False, final_run=False, performance_eligible=False))
    print(json.dumps(decisions))


if __name__ == '__main__':
    main()

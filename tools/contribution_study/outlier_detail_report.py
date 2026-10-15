"""Report fixed-budget encoding ablations, including ungated negative results."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from baselines import sha, write

LABELS = ('rtn', 'scale_only', 'max_fixed', 'max_scale', 'removal_only', 'joint')
NAMES = dict(rtn='RTN＋实际残差', scale_only='只调两平面尺度', max_fixed='强制移出最大值（固定尺度）',
             max_scale='最大值可选移出＋尺度', removal_only='任意元素可选移出（固定尺度）', joint='联合选择元素＋尺度')


def relative(base, candidate):
    return 100*(candidate/base-1) if base else None


def layer_bootstrap(ops, values, reference, candidate):
    """Paired layer-cluster resampling; descriptive, not an independent test."""
    layers = sorted({int(op['name'].split('.')[1]) for op in ops})
    index = [np.array([i for i, op in enumerate(ops) if int(op['name'].split('.')[1]) == layer]) for layer in layers]
    ref = np.array(values[reference]); cand = np.array(values[candidate])
    totals = np.array([[ref[i].sum(), cand[i].sum(), len(i)] for i in index])
    sampled = np.random.default_rng(20260924).integers(0, len(layers), (10000, len(layers)))
    sums = totals[sampled].sum(1)
    differences = (sums[:, 1]-sums[:, 0])/sums[:, 2]
    changes = 100*(sums[:, 1]/sums[:, 0]-1)
    return dict(layers=len(layers), candidate=candidate, reference=reference,
                nmse_difference_ci95=np.quantile(differences, [.025, .975]).tolist(),
                relative_change_ci95=np.quantile(changes, [.025, .975]).tolist(),
                scope='Descriptive paired layer bootstrap on reused development scenes, not an independent generalization test')


def report(root):
    protocol = json.loads((root/'protocol.json').read_text())
    progress = json.loads((root/'progress.json').read_text())
    if not progress['complete'] or progress['protocol_sha256'] != sha(root/'protocol.json'):
        raise ValueError('A complete, unchanged experiment is required')
    ops = list(progress['modules'].values())
    if set(progress['modules']) != set(protocol['modules']):
        raise ValueError('Incomplete operator cohort')
    aggregates = {mode: {label: dict(groups=0, removed_groups=0, removed_nonmax=0,
                                   energy=0., base_sse=0., dual_sse=0., inlier_sse=0.,
                                   inlier_rtn_sse=0., inlier_scale_sse=0.,
                                   weight_operator_nmse=[], three_review=[], four_review=[], three_fit=[])
                         for label in LABELS} for mode in ('raw', 'accepted')}
    strata = {label: {band: dict(groups=0, removed_groups=0, energy=0., dual_sse=0.,
                                rtn_dual_sse=0., scale_dual_sse=0.)
                       for band in ('max_rms_lt2', 'max_rms_2to3', 'max_rms_ge3')}
              for label in LABELS}
    removed_slices = {label: dict(groups=0, energy=0., dual_sse=0., rtn_dual_sse=0.,
                                 scale_dual_sse=0., inlier_sse=0., inlier_rtn_sse=0., inlier_scale_sse=0.) for label in LABELS}
    rows = []
    for op in ops:
        for art in op['artifacts']:
            if sha(root/art['file']) != art['sha256']:
                raise ValueError('Changed experiment artifact')
        file = root/'operators'/op['name']/'groups.npz'
        with np.load(file) as arrays:
            for mode in ('raw', 'accepted'):
                for label in LABELS:
                    selected = label if mode == 'raw' else op['chosen'][label]
                    prefix = selected+'__'
                    target = aggregates[mode][label]
                    mask = arrays[prefix+'removed'] >= 0
                    target['groups'] += len(mask)
                    target['removed_groups'] += int(mask.sum())
                    target['removed_nonmax'] += int((mask & ~arrays[prefix+'removed_is_max']).sum())
                    for key in ('energy', 'base_sse', 'dual_sse', 'inlier_sse', 'inlier_rtn_sse', 'inlier_scale_sse'):
                        target[key] += float(arrays[prefix+key].astype(np.float64).sum())
                    metric = op['metrics'][selected]
                    target['weight_operator_nmse'].append(metric['dual_nmse'])
                    native = op['raw_native'][selected]
                    target['three_review'].append(native['three']['review_mean'])
                    target['four_review'].append(native['four']['review_mean'])
                    target['three_fit'].append(native['three']['fit_mean'])
                    rows.append(dict(mode=mode, method=label, selected=selected, operator=op['name'],
                                     weight_nmse=metric['dual_nmse'], three_review=native['three']['review_mean'],
                                     four_review=native['four']['review_mean'], removed_groups=int(mask.sum())))
                    if mode != 'raw':
                        continue
                    ratio = arrays[prefix+'outlier_ratio']
                    for band, take in (('max_rms_lt2', ratio < 2), ('max_rms_2to3', (ratio >= 2) & (ratio < 3)), ('max_rms_ge3', ratio >= 3)):
                        entry = strata[label][band]
                        entry['groups'] += int(take.sum()); entry['removed_groups'] += int((take & mask).sum())
                        for field, source in (('energy', prefix+'energy'), ('dual_sse', prefix+'dual_sse'),
                                              ('rtn_dual_sse', 'rtn__dual_sse'), ('scale_dual_sse', 'scale_only__dual_sse')):
                            entry[field] += float(arrays[source][take].astype(np.float64).sum())
                    target = removed_slices[label]
                    target['groups'] += int(mask.sum())
                    for field, source in (('energy', prefix+'energy'), ('dual_sse', prefix+'dual_sse'),
                                          ('rtn_dual_sse', 'rtn__dual_sse'), ('scale_dual_sse', 'scale_only__dual_sse'),
                                          ('inlier_sse', prefix+'inlier_sse'), ('inlier_rtn_sse', prefix+'inlier_rtn_sse'),
                                          ('inlier_scale_sse', prefix+'inlier_scale_sse')):
                        target[field] += float(arrays[source][mask].astype(np.float64).sum())
    summary = {}
    intervals = {}
    for mode, methods in aggregates.items():
        summary[mode] = {}
        for label, stats in methods.items():
            result = dict(dual_nmse_pooled=stats['dual_sse']/stats['energy'],
                          base_nmse_pooled=stats['base_sse']/stats['energy'],
                          dual_nmse_operator_mean=float(np.mean(stats['weight_operator_nmse'])),
                          three_review_mean=float(np.mean(stats['three_review'])),
                          four_review_mean=float(np.mean(stats['four_review'])),
                          three_fit_mean=float(np.mean(stats['three_fit'])),
                          groups=stats['groups'], removed_groups=stats['removed_groups'],
                          removed_group_percent=100*stats['removed_groups']/stats['groups'],
                          removed_nonmax=stats['removed_nonmax'],
                          native_choices=dict(Counter(op['chosen'][label] for op in ops)))
            reference = methods['rtn']; scale = methods['scale_only']
            for metric in ('weight_operator_nmse', 'three_review', 'four_review'):
                result[metric+'_vs_rtn_pct'] = relative(np.mean(reference[metric]), np.mean(stats[metric]))
                result[metric+'_vs_scale_pct'] = relative(np.mean(scale[metric]), np.mean(stats[metric]))
                paired = np.array(stats[metric])-np.array(scale[metric])
                result[metric+'_operators_better_equal_worse_vs_scale'] = [int((paired < 0).sum()), int((paired == 0).sum()), int((paired > 0).sum())]
            summary[mode][label] = result
        vals = {label: data['three_review'] for label, data in methods.items()}
        intervals[mode] = [layer_bootstrap(ops, vals, baseline, 'joint') for baseline in ('rtn', 'scale_only')]
    for target in list(strata.values())+[removed_slices]:
        for entry in target.values():
            if entry['energy']:
                entry['dual_nmse'] = entry['dual_sse']/entry['energy']
                entry['dual_change_vs_rtn_pct'] = relative(entry['rtn_dual_sse'], entry['dual_sse'])
                entry['dual_change_vs_scale_pct'] = relative(entry['scale_dual_sse'], entry['dual_sse'])
            if 'inlier_sse' in entry and entry['groups']:
                entry['inlier_change_vs_rtn_pct'] = relative(entry['inlier_rtn_sse'], entry['inlier_sse'])
                entry['inlier_change_vs_scale_pct'] = relative(entry['inlier_scale_sse'], entry['inlier_sse'])
    result = dict(complete=True, protocol_sha256=sha(root/'protocol.json'),
                  scope=protocol['scope'], summary=summary, raw_outlier_strata=strata,
                  raw_removed_group_subsets=removed_slices, descriptive_layer_intervals=intervals,
                  sampled_groups=summary['raw']['rtn']['groups'], sampled_blocks=sum(len(op['sampled_blocks']) for op in ops),
                  paid_blocks=sum(op['support_blocks'] for op in ops), operators=len(ops),
                  operator_kinds=dict(Counter(op['name'].split('.')[-1] for op in ops)),
                  fixed_asset_cost=protocol['spec']['cost'],
                  cost=dict(three_fp4_mma=sum(op['cost_three']['total_mma'] for op in ops),
                            four_fp4_mma_upper_bound=sum(op['cost_four']['total_mma'] for op in ops),
                            scope='One call per studied operator at captured shapes; not a full request or speed estimate'),
                  independent_final_test=False, performance_eligible=False)
    out = root/'report'
    if (out/'encoding_audit.json').exists():
        result['encoding_audit'] = json.loads((out/'encoding_audit.json').read_text())
    if (out/'boundary_check.json').exists():
        check = json.loads((out/'boundary_check.json').read_text())
        if check['complete']:
            result['boundary_sensitivity_check'] = check['summary']
    write(out/'results.json', result)
    with (out/'operators.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    s = summary['accepted']; raw = summary['raw']
    lines = ['# Outlier 留给 W1：固定计算与支持的局部验证', '',
             f'完成 {len(ops)} 个 Edge attention 投影，固定抽样 {result["sampled_blocks"]} 个已付费 N128/K128 块、{result["sampled_groups"]:,} 个 K16 组。', '',
             '## 实验边界', '',
             '- 原始 2× 策略的 prescale、W1 支持、A 选择、激活尺度、执行 tile 完全固定。只改抽中的 W0/W1 编码。',
             '- 每个算子 16 个完整 M=960 教师面板：8 个用于本次搜索与接受，另 8 个复核；两组分别覆盖四阶段、双 CFG。',
             '- 这些都是已有 development 场景，基础资产曾用过这些数据。复核仅对本次新搜索隔离，不是独立最终测试。',
             '- 当前付费 W1 在 attention 投影中，不能据此声称 FFN、QK/PV 或完整生成质量提升。',
             '- 每组至多移出一项；所有 16 个位置与不移出竞争。实际 E4M3/E2M1 编解码，残差从实际 W0 重算。',
             '- 除强制最大值对照外，候选包含原编码；整体算子只按搜索面板三项输出误差接受。保留未接受提案结果。', '',
             '## 复核结果：通过搜索面板接受的编码', '',
             'NMSE 越低越好；权重是抽样组 W0+W1 的算子均值，算子输出使用完整输入和完整权重矩阵。', '',
             '| 方法 | 权重双平面 NMSE | 三项输出 NMSE | 四项输出 NMSE | 三项相对 RTN |',
             '|---|---:|---:|---:|---:|']
    for label in LABELS:
        row = s[label]
        lines.append(f'| {NAMES[label]} | {row["dual_nmse_operator_mean"]:.8g} | {row["three_review_mean"]:.8g} | {row["four_review_mean"]:.8g} | {row["three_review_vs_rtn_pct"]:+.5f}% |')
    lines += ['', f'联合方法相对“只调尺度”的三项输出变化：**{s["joint"]["three_review_vs_scale_pct"]:+.6f}%**；',
              f'63 个算子中改善／相同／变差：{s["joint"]["three_review_operators_better_equal_worse_vs_scale"]}。',
              f'四项输出相对只调尺度：{s["joint"]["four_review_vs_scale_pct"]:+.6f}%；双平面权重 NMSE 相对只调尺度：{s["joint"]["weight_operator_nmse_vs_scale_pct"]:+.6f}%。', '',
              '四项仅用于诊断遗漏交叉项的影响；其计算量更高，不把它算作原 2× 策略的同预算结果。', '',
              '## 未经整体接受筛选的原始提案', '',
              '| 提案 | 权重双平面相对 RTN | 三项输出相对 RTN | 三项输出相对只调尺度 | 移出组比例 |',
              '|---|---:|---:|---:|---:|']
    for label in LABELS:
        row = raw[label]
        lines.append(f'| {NAMES[label]} | {row["weight_operator_nmse_vs_rtn_pct"]:+.4f}% | {row["three_review_vs_rtn_pct"]:+.5f}% | {row["three_review_vs_scale_pct"]:+.5f}% | {row["removed_group_percent"]:.3f}% |')
    chosen = s['joint']['native_choices']
    removed = removed_slices['joint']
    lines += ['', f'联合方法最终按算子选中：`{json.dumps(chosen, ensure_ascii=False)}`。', '',
              '## 移出元素到底改善了什么', '',
              f'原始联合提案在 {raw["joint"]["removed_groups"]:,} 个组移出元素，其中非最大幅值为 {raw["joint"]["removed_nonmax"]:,} 个。',
              f'仅在实际移出的同一批组、用相同排除位置比较，保留元素的 W0 SSE 相对 RTN 为 {removed.get("inlier_change_vs_rtn_pct", 0):+.5f}%，相对只调尺度为 {removed.get("inlier_change_vs_scale_pct", 0):+.5f}%。',
              f'同一批组的完整 W0+W1 SSE 相对 RTN 为 {removed.get("dual_change_vs_rtn_pct", 0):+.5f}%，相对只调尺度为 {removed.get("dual_change_vs_scale_pct", 0):+.5f}%。',
              '这些条件子集用于解释机制；总体效应必须以全部预定抽样组和全部算子为准。', '',
              '| 原始最大值/RMS | 组数 | 联合提案双平面误差相对 RTN | 相对只调尺度 |',
              '|---|---:|---:|---:|']
    for band, data in strata['joint'].items():
        lines.append(f'| {band} | {data["groups"]:,} | {data.get("dual_change_vs_rtn_pct", 0):+.4f}% | {data.get("dual_change_vs_scale_pct", 0):+.4f}% |')
    if 'encoding_audit' in result:
        audit = result['encoding_audit']
        lines += ['', '## 舍入边界审计', '',
                  f'- {audit["exact_native_control_panels"]}/{audit["all_native_control_panels"]} 个原生对照面板与历史结果完全一致。',
                  f'- 移出组中，{audit["same_base_scale_vs_scale_only"]:,} 个与尺度对照具有相同基础尺度，{audit["different_base_scale_vs_scale_only"]:,} 个换了尺度。',
                  f'- 相同尺度组的保留元素，{audit["same_scale_retained_codes_equal"]:,} 组编码完全不变；另外 {audit["boundary_elements"]} 处差异均已定位到存量 RTN 与重新编码的舍入边界，最大归一化距离 {audit["boundary_max_distance"]:.4g}。',
                  '- 因此，不能把联合方法的保留元素误差改善单独归因于移出非最大元素；基础尺度的重新选择参与了收益。']
    if 'boundary_sensitivity_check' in result:
        check = result['boundary_sensitivity_check']
        lines += [f'- 将涉及边界差异的 {check["restored_groups"]} 个完整组恢复为尺度对照，重新执行原生算子并沿用 fit-only 接受规则后，权重 NMSE 相对只调尺度为 {check["accepted_weight_vs_scale_pct"]:+.6f}%，三项输出为 {check["accepted_three_vs_scale_pct"]:+.6f}%。这是按字节差异定义的事后稳健性检查，未用于重新搜索。']
    lines += ['', '## 成本与正确性', '',
              f'- 所有方法使用相同的 {result["paid_blocks"]} 个 W1 支持块；只抽样修改其中 {result["sampled_blocks"]} 个。',
              f'- 原资产张量总字节 {protocol["spec"]["cost"]["tensor_bytes"]:,}，W1 {protocol["spec"]["cost"]["w1_bytes"]:,}；逐算子检查编码、尺度、支持索引张量字节不变。',
              '- 移出元素涉及的 W1 块变为 mandatory，不能在之后的调度中关掉；必要支持的元数据另存，不能说所有序列化文件字节完全一样。',
              '- 同一三项/四项设置内，FP4 MMA、在线 activation packing、selector 和 scratch 形状相同；新搜索只增加离线工作。',
              '- 检查包括协方差代理对独立直接输出公式、穷举选择、实际残差、全零组、固定非最大值移出、仅修改付费组、补丁重载、压缩重载。',
              '- 使用已有原生 Triton 修复 GEMM，没有新增内核；共享 PRO 6000 数据不用于速度或延迟结论。', '',
              '## 可用于论文的结论边界', '',
              f'**局部编码小点部分成立，主要生成质量贡献尚未由此支持。** 固定尺度的选择性移出使抽样权重重构 NMSE 下降 {-s["removal_only"]["weight_operator_nmse_vs_rtn_pct"]:.3f}%；联合搜索相对只调尺度额外下降 {-s["joint"]["weight_operator_nmse_vs_scale_pct"]:.3f}%，但完整算子三项输出额外改善只有 {-s["joint"]["three_review_vs_scale_pct"]:.6f}%。',
              '更准确的表述是“在已分配 W1 的块内，联合选择可选的保留元素及两平面尺度，按实际修复表示优化误差”，而不是“移出最大 outlier 普遍改善量化”。全部已选元素为非最大幅值，强制移出最大值反而明显恶化。',
              '必须区分只调尺度的收益、移出元素的额外收益、以及整体输出效应。基础平面保留元素误差降低不保证完整双平面或三项输出误差降低。',
              '这里选择只覆盖 252 个块，不能将完整算子变化直接外推为全支持重编码收益，更不能替代完整生成指标。',
              '建议作为 residual-aware group encoding 的小消融写入方法细节，保留“不移出”候选，并明确 mandatory W1 约束；不作为解释主要生成质量提升的证据。', '',
              '完整原始数据：`../operators/*/groups.npz`、`patches.pt`、`result.json`；汇总 `results.json`、`operators.csv`。', '']
    (out/'REPORT.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(out/'REPORT.zh.md'), summary=summary), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    report(p.parse_args().root)

"""Paired AWQ comparisons on the unchanged historical 72-scene cohort."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from baselines import sha, write
from dataset_report import METRICS, average, stratified_interval


def paired_contrast(candidate, reference, categories, statistics):
    if set(candidate) != set(reference) or not candidate:
        raise ValueError('Nonempty, identical paired scene/seed keys required')
    detail = {}
    for field in METRICS:
        scene_deltas = collections.defaultdict(list)
        seed_deltas = []
        for (case, seed), values in candidate.items():
            left, right = values[field], reference[case, seed][field]
            delta = None if left is None or right is None else left - right
            scene_deltas[case].append(delta)
            seed_deltas.append(dict(case_id=case, seed=seed, delta=delta))
        if any(row['delta'] is None for row in seed_deltas):
            detail[field] = dict(status='infinite_exact_match_psnr', seed_deltas=seed_deltas,
                                 ci95=None, familywise_ci=None, mean_delta=None)
            continue
        strata = collections.defaultdict(list)
        scenes = []
        for case, deltas in scene_deltas.items():
            delta = float(np.mean(deltas))
            strata[categories[case]].append(delta)
            scenes.append(dict(case_id=case, delta=delta))
        result = stratified_interval(strata, replicates=statistics['replicates'],
            seed=statistics['bootstrap_seed'], family=statistics['family_size'])
        lower = field not in ('ssim_mean', 'psnr_db')
        result.update(higher_is_better=not lower, seed_deltas=seed_deltas,
            worst_seeds=sorted(seed_deltas, key=lambda row: row['delta'], reverse=lower)[:5],
            worst_scenes=sorted(scenes, key=lambda row: row['delta'], reverse=lower)[:5],
            worse_scene_count=sum(row['delta'] > 0 if lower else row['delta'] < 0 for row in scenes))
        detail[field] = result
    return dict(metrics=detail,
        candidate_improves=detail['lpips']['familywise_ci'][1] < 0 and
            detail['ssim_mean']['mean_delta'] >= 0 and detail['relative_l2']['mean_delta'] <= 0,
        reference_improves=detail['lpips']['familywise_ci'][0] > 0 and
            detail['ssim_mean']['mean_delta'] <= 0 and detail['relative_l2']['mean_delta'] >= 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    progress = json.loads((root / 'metrics/progress.json').read_text())
    if not json.loads((root / 'workflow.json').read_text())['complete'] or not progress['complete']:
        raise ValueError('Complete generation and metrics required')
    if progress['identity']['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Changed manifest')
    expected = (len(manifest['methods']) - 1) * len(manifest['cases'])
    if progress['completed'] != expected or progress['expected'] != expected:
        raise ValueError('Incomplete metric coverage')
    values, scenes, receipts, metric_rows, costs = collections.defaultdict(dict), [], [], [], []
    categories = {entry['case']['id']: entry['category'] for entry in manifest['cases']}
    for entry in manifest['cases']:
        case = entry['case']['id']
        reference_path = root / 'generation/bf16' / case / 'result.json'
        reference = json.loads(reference_path.read_text())
        for method, spec in manifest['methods'].items():
            receipt_path = root / 'generation' / method / case / 'result.json'
            receipt = json.loads(receipt_path.read_text())
            if not receipt['complete'] or receipt['identity']['manifest_sha256'] != sha(manifest_path):
                raise ValueError('Incomplete or changed generation manifest')
            for key in ('case', 'image_sha256', 'seeds'):
                wanted = entry['case'] if key == 'case' else entry[key]
                if receipt['identity'][key] != wanted or reference['identity'][key] != wanted:
                    raise ValueError('Changed paired input: ' + key)
            if receipt['model_identity'] != reference['model_identity'] or receipt['targets'] != reference['targets']:
                raise ValueError('Changed model or generation scope')
            if len(receipt['targets']) != 168 or len(receipt['rows']) != 2 or {r['seed'] for r in receipt['rows']} != set(entry['seeds']):
                raise ValueError('Changed target scope or paired seeds')
            for row in receipt['rows']:
                if row['steps'] != 35 or row['debug'] != 'off' or row['execution'] != 'graph' or row['performance_eligible']:
                    raise ValueError('Changed execution conditions')
            if method == 'awq':
                adapter = receipt.get('adapter', {})
                if (adapter.get('assets_sha256') != spec['baseline_adapter']['assets_sha256'] or
                        adapter.get('execution_type') != 'offline_decoded_quality_reference' or
                        receipt.get('actual_generation_precision') != 'NVFP4-AWQ-lite-W4A16' or
                        receipt.get('storage_comparison_eligible') is not False):
                    raise ValueError('Missing or changed AWQ adapter qualification')
            receipts.append(dict(method=method, case=case, path=str(receipt_path), sha256=sha(receipt_path), receipt=receipt))
            costs.append(dict(method=method, case=case, benchmark=entry['benchmark'],
                mma_ratios=[r['actual_mma_ratio'] for r in receipt['rows']],
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipt['rows']),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in receipt['rows']),
                performance_eligible=False, storage_comparison_eligible=method != 'awq'))
            if method == 'bf16':
                continue
            metric_path = root / 'metrics' / method / (case + '.json')
            result = json.loads(metric_path.read_text())
            wanted = dict(identity=progress['identity'], candidate_sha256=sha(receipt_path), reference_sha256=sha(reference_path))
            if not result['complete'] or result['provenance'] != wanted:
                raise ValueError('Changed metrics or generated outputs')
            if len(result['rows']) != 2 or {r['seed'] for r in result['rows']} != set(entry['seeds']):
                raise ValueError('Missing or repeated metric seeds')
            metric_rows.append(result)
            for row in result['rows']:
                video = row['metrics']['video']
                if not all(np.isfinite(video[key]) for key in METRICS if key != 'psnr_db'):
                    raise ValueError('Nonfinite quality metric')
                if video['psnr_db'] is not None and not np.isfinite(video['psnr_db']):
                    raise ValueError('PSNR must be finite or explicitly exact')
                if video['psnr_db'] is None and not video['psnr_exact']:
                    raise ValueError('Missing PSNR without exact-match marker')
                values[entry['benchmark'], method][case, row['seed']] = video
            scenes.append(dict(method=method, benchmark=entry['benchmark'], category=entry['category'], case=case,
                metrics={key: average([row['metrics']['video'][key] for row in result['rows']]) for key in METRICS}))
    methods = [name for name in manifest['methods'] if name != 'bf16']
    summaries, contrasts = {}, []
    for benchmark in ('rbench', 'paibench'):
        summaries[benchmark] = {}
        for method in methods:
            own = [row for row in scenes if row['method'] == method and row['benchmark'] == benchmark]
            grouped = {category: {key: average([row['metrics'][key] for row in own if row['category'] == category])
                                 for key in METRICS} for category in sorted({row['category'] for row in own})}
            summaries[benchmark][method] = dict(n_scenes=len(own), n_videos=2 * len(own), categories=grouped,
                category_macro={key: average([row[key] for row in grouped.values()]) for key in METRICS})
        for pair in manifest['primary_comparisons']:
            contrasts.append(dict(benchmark=benchmark, **pair, **paired_contrast(
                values[benchmark, pair['candidate']], values[benchmark, pair['reference']], categories, manifest['statistics'])))
    result = dict(complete=True, manifest_sha256=sha(manifest_path), summaries=summaries, contrasts=contrasts,
        statistics=manifest['statistics'], scenes=scenes, seed_metrics=metric_rows, costs=costs,
        scope=manifest['scope'], performance_eligible=False, selection_eligible=False)
    report = root / 'report'
    report.mkdir(exist_ok=True)
    write(report / 'quality.json', result)
    lines = ['# AWQ 补充：同一 72 场景的质量对比', '',
        'RBench 36 场景、PAI-Bench 36 场景，每场景两个相同种子；新增144段AWQ输出，复用1,296段经哈希核对的历史输出。Edge、192×320、61帧、35 steps、CFG6、Graph、debug off，保留相同新鲜AR预填充与首帧条件。', '',
        'AWQ为官方ModelOpt NVFP4 AWQ-lite W4A16，校准前关闭激活量化，BF16 attention；现有168层资产保持冻结。运行时使用离线解码BF16权重与校准prescale，仅评价量化质量。不能把其耗时或显存当成原生packed W4A16部署结果。', '',
        '指标均相对同输入BF16输出。场景内先平均两个种子，再等权平均类别；PSNR使用固定data_range=1的整段视频RGB MSE。这里不是任务成功率或官方榜单评分。', '',
        '低预算对照为历史72场景中冻结的marginal/head_coordinate配置，不是后续16场景开发集上的b2_joint或b1.5_protected_n。']
    order = ['awq'] + [name for name in methods if name != 'awq']
    for benchmark in ('rbench', 'paibench'):
        lines += ['', '## ' + benchmark, '', '| 配置 | LPIPS ↓ | SSIM ↑ | 视频相对 L2 ↓ | PSNR dB ↑ |', '|---|---:|---:|---:|---:|']
        for method in order:
            v = summaries[benchmark][method]['category_macro']
            psnr = '+∞' if v['psnr_db'] is None else f'{v["psnr_db"]:.3f}'
            lines.append(f'| {method} | {v["lpips"]:.6f} | {v["ssim_mean"]:.6f} | {v["relative_l2"]:.6f} | {psnr} |')
        lines += ['', '| 候选 − AWQ | LPIPS 差值 | 配对95%区间 | 16项校正区间 | 质量门槛结论 |', '|---|---:|---|---|---|']
        for row in contrasts:
            if row['benchmark'] != benchmark:
                continue
            d = row['metrics']['lpips']
            decision = '候选优于AWQ' if row['candidate_improves'] else 'AWQ优于候选' if row['reference_improves'] else '未通过全部约束'
            lines.append(f'| {row["candidate"]} | {d["mean_delta"]:.6f} | {d["ci95"]} | {d["familywise_ci"]} | {decision} |')
    lines += ['', '## 结论边界与原始证据', '',
        '本次为已使用历史cohort的AWQ补充评价，不是新预留的最终测试。AWQ未在这72场景重新校准或选参；历史候选结果在补测前已知。8种配置对AWQ、两个数据集共16项补充比较，按类别分层、以场景为配对簇bootstrap并作Bonferroni校正。质量门槛同时要求SSIM/L2均值不退化；PSNR为补充指标。历史FP8验收保持原结果。',
        'AWQ/FP8采用BF16 attention；FP4配置采用各自冻结的FP4 attention。完整方案比较包含该差异，不能归因成单一线性层激活量化效应。',
        '逐种子指标、场景均值、最差种子、最差场景、所有配对差值和诊断内存记录见quality.json.gz。生成输出及原始receipt索引见generation_index.json.gz；数据与配置见manifest.json，正确性检查见probe.json。',
        '本轮没有测量原生AWQ推理性能；共享PRO6000结果不用于速度排名或5090性能结论。']
    (report / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    args.archive.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report / 'REPORT.md', args.archive / 'REPORT.md')
    for name, value in [('quality', result), ('generation_index', receipts)]:
        (args.archive / (name + '.json.gz')).write_bytes(gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0))
    for filename in ('manifest.json', 'probe.json', 'assets_audit.json', 'PROTOCOL.md'):
        shutil.copyfile(root / filename, args.archive / filename)
    prepared = json.loads((root / 'generation/awq/prepared.json').read_text())
    write(args.archive / 'prepared.json', prepared)
    repo = Path(__file__).resolve().parents[2]
    package = repo / 'packages/fp4-repair/src/thunderworld_fp4'
    sources = {package / name: digest for name, digest in prepared['runtime']['source_sha256'].items()}
    for filename in ('awq_multiscene.py', 'awq_multiscene_report.py', 'awq_prepare.py', 'external.py',
                     'dataset_infer.py', 'dataset_suite.py', 'dataset_metrics.py', 'dataset_report.py',
                     'importance_workflow.py', 'baselines.py'):
        path = Path(__file__).with_name(filename)
        sources[path] = sha(path)
    exported = {}
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError('Runtime source changed after generation: ' + str(path))
        relative = path.relative_to(repo)
        target = args.archive / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        exported[str(relative)] = digest
    write(args.archive / 'implementation.json', dict(complete=True, sources=exported,
        new_videos=144, reused_videos=1296, total_videos=1440, performance_eligible=False))
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    main()

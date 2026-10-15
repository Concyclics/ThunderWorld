"""Archive matched single-request quality and speed-only concurrency diagnostics."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

from baselines import sha, write


LABELS = {
    'bf16': 'BF16', 'fp8': '原生 FP8', 'fp4_base': '普通 W4A4',
    'awq_marlin': 'NVFP4 AWQ-lite / Marlin', 'arcquant': 'ARCQuant 50%',
    'arcquant100': 'ARCQuant 100%', 'sharq_perf': 'SharQ 现成融合模式',
    'sharq': 'SharQ 原质量模式', 'b1.5_protected_n': '我们的 1.5×',
    'b2_joint': '我们的 2×', 'b2.5_ffn_guarded': '我们的 2.5×',
}
ORDER = ['bf16', 'fp8', 'fp4_base', 'awq_marlin', 'arcquant', 'arcquant100',
         'sharq_perf', 'b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded']
FIELDS = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')


def paired_quality(quality):
    """Descriptive scene-paired intervals; seeds remain nested inside scenes."""
    import numpy as np
    raw = defaultdict(dict)
    for result in quality['seed_metrics']:
        for row in result['rows']:
            raw[result['benchmark'], result['name']][result['case_id'], row['seed']] = row['metrics']['video']
    comparisons = [(name, other) for name in ('b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded')
                   for other in ('awq_marlin', 'arcquant100', 'sharq_perf', 'fp8')]
    comparisons.append(('arcquant100', 'arcquant'))
    result = {}
    for benchmark in ('rbench', 'paibench'):
        for candidate, reference in comparisons:
            a, b = raw[benchmark, candidate], raw[benchmark, reference]
            if not a or a.keys() != b.keys():
                raise ValueError('Unmatched quality pair')
            stats = {}
            for field in FIELDS:
                by_scene = defaultdict(list)
                differences = []
                for key in sorted(a):
                    delta = a[key][field] - b[key][field]
                    by_scene[key[0]].append(delta)
                    differences.append((key, delta))
                values = np.array([np.mean(v) for v in by_scene.values()])
                rng = np.random.default_rng(180918)
                draws = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
                worst = (max if field in ('lpips', 'relative_l2') else min)(differences, key=lambda v: v[1])
                stats[field] = dict(mean_delta=float(values.mean()),
                    descriptive_scene_bootstrap95=np.quantile(draws, [.025, .975]).tolist(),
                    scenes=len(values), seeds=len(differences), worst_seed_case=worst[0], worst_delta=worst[1])
            result[f'{benchmark}|{candidate}|{reference}'] = stats
    return result


def archive_json(path, value):
    path.write_bytes(gzip.compress((json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--throughput', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.study/'manifest.json').read_text())
    quality = json.loads((a.study/'report/quality.json').read_text())
    fresh = json.loads((a.study/'fresh_1p5_verification.json').read_text())
    suite = json.loads((a.throughput/'suite.json').read_text())
    progress = json.loads((a.throughput/'progress.json').read_text())
    digest = sha(a.study/'manifest.json')
    if not all(r.get('complete') for r in (quality, fresh, progress)):
        raise ValueError('Quality, fresh 1.5 verification and all throughput jobs must finish')
    if any(r['manifest_sha256'] != digest for r in (quality, fresh, suite)):
        raise ValueError('Mixed study identities')
    if progress['suite_sha256'] != sha(a.throughput/'suite.json'):
        raise ValueError('Changed throughput suite')
    if set(progress['completed']) != {job['file'] for job in suite['jobs']}:
        raise ValueError('Throughput progress omits declared jobs')
    if not fresh['all_bitwise_equal'] or fresh['cases'] != len(manifest['cases']):
        raise ValueError('Fresh 1.5x verification is incomplete')
    curves, raw, receipts = {}, [], {}
    for job in suite['jobs']:
        path = a.throughput/job['file']
        value = json.loads(path.read_text())
        if (not value['complete'] or value['manifest_sha256'] != digest or
                value['source_sha256'] != suite['worker_sha256'] or value['performance_eligible']):
            raise ValueError('Incomplete/changed/ineligible worker receipt')
        method, case_id = job['method'], job['case']
        quality_path = a.study/'generation'/method/case_id/'result.json'
        if sha(quality_path) != value['quality_receipt_sha256']:
            raise ValueError('Changed matching quality receipt')
        entry = next(e for e in manifest['cases'] if e['case']['id'] == case_id)
        for row in value['rows']:
            if len(row['measurements']) != suite['rounds'] * suite['waves']:
                raise ValueError('Wrong number of measured groups')
            if row['batch'] == 1 and row['bitwise_equal'] is not True:
                raise ValueError('Single-request quality path must be checked')
            if row['batch'] > 1 and row['numerical_checked']:
                raise ValueError('Unexpected concurrency quality qualification')
            for measurement in row['measurements']:
                if not math.isclose(measurement['videos_per_second'], row['batch']/measurement['group_seconds']):
                    raise ValueError('Throughput does not count complete requests')
        if not value['rows']:
            raise ValueError('No admitted batch in worker')
        rows = [{k: v for k, v in row.items() if k != 'measurements'} for row in value['rows']]
        best = max(rows, key=lambda v: v['median_videos_per_second'])
        adapter = value.get('adapter') or {}
        prepared_bytes = value['runtime']['prepared_weight_bytes']
        prepared_source = 'runtime unique generation deployment tensors'
        if adapter.get('generic_prepared_weight_bytes_qualified') is False:
            prepared_bytes = adapter['resident_generation_weight_bytes']
            prepared_source = 'native adapter unique weights/scales/prescale; excludes workspace'
        curves[f"{entry['benchmark']}|{method}"] = dict(method=method, benchmark=entry['benchmark'],
            case_id=case_id, stream_policy=value['stream_policy'], stop_reason=value['stop_reason'],
            rows=rows, best=best, source_sha256=sha(path),
            prepared_weight_bytes=prepared_bytes, prepared_weight_accounting=prepared_source,
            resident_model_state_bytes=value['runtime']['resident_model_state_bytes'],
            graph_capture_policy=value['graph_capture_policy'],
            attention_dispatch_policy=value['attention_dispatch_policy'],
            execution=value['runtime']['config']['execution'],
            configured_linear_backend=value['runtime']['config']['linear_backend'],
            quality_applies_to='single request only', peak_is='best observed fixed implementation; no hardware-maximum claim')
        raw.append(value)
    if len({r['runtime_source_sha256'] for r in raw}) != 1:
        raise ValueError('Runtime adapter changed during the sweep')
    if len({json.dumps(r['runtime']['source_sha256'], sort_keys=True) for r in raw}) != 1:
        raise ValueError('Inference implementation changed during the sweep')
    for method in manifest['methods']:
        receipts[method] = [json.loads((a.study/'generation'/method/e['case']['id']/'result.json').read_text())
                            for e in manifest['cases']]
    fresh_receipts = []
    for item in fresh['rows']:
        path = a.study/'fresh_1p5'/item['case']/'result.json'
        if sha(path) != item['fresh_receipt_sha256'] or not item['bitwise_equal']:
            raise ValueError('Changed fresh 1.5x receipt')
        fresh_receipts.append(json.loads(path.read_text()))
    pairs = paired_quality(quality)
    result = dict(complete=True, manifest_sha256=digest, performance_eligible=False,
        quality_scope=quality['scope'], quality=quality['summaries'], costs=quality['costs'],
        descriptive_quality_pairs=pairs, throughput=curves, fresh_1p5=fresh,
        concurrency_quality='Not evaluated, per user request. Initial unisolated-capture two-slot pilot failed equivalence; the isolated-capture implementation is measured for speed only.',
        hardware='Local RTX PRO 6000 Blackwell Server Edition, GPU0/3 selection; actual device receipt retained per worker',
        formal_benchmark=False, baseline_tuning=False,
        report_source_sha256=sha(__file__), throughput_suite_sha256=sha(a.throughput/'suite.json'))
    lines = ['# 1.5×、ARCQuant 100% 质量与完整请求吞吐补测', '',
        '质量与并发速度分别验收。质量使用 16 个已冻结开发场景（RBench 9、PAI-Bench 7），'
        '每场景 2 个种子；不是预留的 72 个最终场景测试。指标均为相对同输入 BF16 的一致性。', '',
        '新生成：1.5×、ARCQuant 100%、SharQ 现成融合模式各 32 段视频。1.5× 的 32 个输出与冻结版本逐位一致。'
        '其他方法使用哈希核验的同条件历史输出。未重新选参，最终验证/测试集未使用。', '',
        '共同条件：同版本 Edge，35 步、CFG 6、192×320、61 帧，BF16 AR 缓存并计入每请求预填充，debug off。'
        '各量化方法使用冻结的生成投影范围。我们的联合方案使用 FP4 attention；外部线性量化对照保留 BF16 attention。', '',
        '本轮三个修复档保持质量实验的 `linear_backend=triton`、`kernel_variant=optimized`，'
        '使用原生 FP4 Tensor Core 计算及联合 Graph；没有在计时时另换 CUTLASS 累加实现。'
        'FP8、AWQ、ARC、SharQ 的实际原生算子由各自适配器提供，不能用配置中的默认 backend 字段推断它们的计算精度。', '',
        'AWQ 行具体使用 ModelOpt 官方 `NVFP4_AWQ_LITE_CFG` 离线权重量化，保持 A16，'
        '以原始 E2M1/E4M3 codes/scales 调用 Marlin。文中的 AWQ 比较均限定为该固定实现。', '',
        '## 单请求质量', '']
    for benchmark in ('rbench', 'paibench'):
        lines += [f'### {benchmark}', '', '| 方法 | LPIPS ↓ | SSIM ↑ | 相对 L2 ↓ | PSNR dB ↑ |',
                  '|---|---:|---:|---:|---:|']
        for name in ORDER[1:]:
            v = quality['summaries'][benchmark][name]
            lines.append(f"| {LABELS[name]} | {v['lpips']:.6f} | {v['ssim_mean']:.6f} | {v['relative_l2']:.6f} | {v['psnr_db']:.3f} |")
        lines += ['']
    lines += ['1.5× 在这批场景上未全面超过 AWQ，也未超过 ARCQuant 100%。'
              '2× 的 RBench LPIPS 优于 ARC100，PAI LPIPS 接近；2.5× 的两组 LPIPS 均更低。'
              '这些是开发集描述结果，不能据此宣布最终质量门槛通过。'
              '2.5× 也没有满足超过 FP8 的原门槛：RBench 相对 L2 略高，PAI 的 LPIPS 仍高于 FP8。', '',
              'SharQ 的现成 extra_fusion=True 会改变残差合并舍入，因此重新测量该模式质量，'
              '没有将原 extra_fusion=False 的质量与融合模式速度拼接。未移植 Cosmos RMSNorm 融合。', '',
              '### 配对差值', '',
              '下表为候选 LPIPS − 对照 LPIPS；负值更好。区间按场景成对 bootstrap 20,000 次，'
              '每场景的两种子先平均。这些开发集区间仅用于描述；没有作多预算显著性或泛化主张。', '',
              '| 数据集 | 候选 | 对照 | ΔLPIPS | 描述性 95% 区间 |', '|---|---|---|---:|---|']
    for benchmark in ('rbench', 'paibench'):
        for candidate, reference in [('b1.5_protected_n', 'awq_marlin'), ('b1.5_protected_n', 'arcquant100'),
                                      ('arcquant100', 'arcquant'), ('b2_joint', 'arcquant100'),
                                      ('b2.5_ffn_guarded', 'arcquant100')]:
            v = pairs[f'{benchmark}|{candidate}|{reference}']['lpips']
            lo, hi = v['descriptive_scene_bootstrap95']
            lines.append(f"| {benchmark} | {LABELS[candidate]} | {LABELS[reference]} | {v['mean_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## 完整请求吞吐诊断', '',
        '每组计时包含输入读取、噪声生成、AR 预填充、35 步降噪、视频解码和 CPU 输出转换；'
        '不包含加载/建图、磁盘序列化和计时后的哈希检查。每档 5 轮，每轮 3 波；'
        '表中为 15 波的中位吞吐，帧/秒 = 视频/秒 × 61。', '',
        '原始记录的 `additional_prepare_seconds` 单列每档新增副本的准备、建图和计时前核验；'
        '离线校准沿用冻结资产，不计入暖机请求吞吐。', '',
        '当前上游 Cosmos pipeline 只支持单样本，prompt 列表会被截为首项。这里的 B 是独立完整请求的并发数，'
        '每路独立模型和 Graph，权重被复制；不是合并 GEMM 的真正大 batch，也不是共享权重服务的显存上限。', '',
        '所有方法统一使用独立 Graph 捕获流，并保护 SDPA 进程级后端开关的主机调用。'
        '这两项处理请求并发兼容性，不改变选中的 attention 算法，也不增加 GPU 同步。', '',
        '按用户要求，B>1 只测速度，没有输出一致性准入；隔离捕获流之前的双路预检曾出现不一致。'
        '所以并发速度不能与单请求质量合并成“同质量加速”结论。B1 路径仍核验完整质量输出。', '',
        'GPU 在每波边界检查无外部计算进程；这是固定现成实现的诊断测试，performance_eligible=false，'
        '未更新正式延迟排行榜。未专项调优任何 baseline。ARC/SharQ 保留官方默认流和 eager 主机调度；'
        '其封装成本包含在计时内，不能把差距全部归因于算法 MMA。', '']
    for benchmark in ('rbench', 'paibench'):
        lines += [f'### {benchmark}', '',
            '| 方法 | B1 视频/s | B2 | B4 | B8 | 最佳 B | 最佳/B1 | 最佳档 P90 秒 | 最佳档 allocated/reserved GiB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name in ORDER:
            curve = curves[f'{benchmark}|{name}']
            rows = {v['batch']: v for v in curve['rows']}
            speeds = [f"{rows[b]['median_videos_per_second']:.4f}" if b in rows else '—' for b in (1, 2, 4, 8)]
            best = curve['best']
            gain = best['median_videos_per_second']/rows[1]['median_videos_per_second']
            lines.append(f"| {LABELS[name]} | {' | '.join(speeds)} | {best['batch']} | {gain:.3f}× | "
                         f"{best['request_p90_seconds']:.3f} | {best['peak_allocated_bytes']/2**30:.2f} / {best['peak_reserved_bytes']/2**30:.2f} |")
        lines += ['', '| 方法 | 停止条件 |', '|---|---|']
        for name in ORDER:
            curve = curves[f'{benchmark}|{name}']
            lines.append(f"| {LABELS[name]} | {curve['stop_reason']} |")
        lines += ['']
    lines += ['连续两次翻倍吞吐提升不足 5% 时停止，或到 B8 / 预留显存限制停止。'
              '观测到的平台只代表本实现和这两类输入；不能称为 Tensor Core 已打满或硬件吞吐上限。', '',
              '### 各自最佳并发档的吞吐比', '',
              '下表是本轮每种方法各自最高已测吞吐之比，可能对应不同并发数。'
              '它不是单请求延迟加速比，也不包含并发质量保证。', '',
              '| 输入 | 候选 | / BF16 | / FP8 | / AWQ | / ARC100 | / SharQ |',
              '|---|---|---:|---:|---:|---:|---:|']
    for benchmark in ('rbench', 'paibench'):
        for candidate in ('b1.5_protected_n', 'b2_joint', 'b2.5_ffn_guarded'):
            speed = curves[f'{benchmark}|{candidate}']['best']['median_videos_per_second']
            ratios = [speed/curves[f'{benchmark}|{baseline}']['best']['median_videos_per_second']
                      for baseline in ('bf16', 'fp8', 'awq_marlin', 'arcquant100', 'sharq_perf')]
            lines.append(f"| {benchmark} | {LABELS[candidate]} | " +
                         ' | '.join(f'{ratio:.3f}×' for ratio in ratios) + ' |')
    lines += ['',
              '## 单实例存储与预算', '',
              'prepared weight 只统计替换生成投影的部署张量；模型 state 还包含原生 AR、嵌入及编解码器等，'
              '二者都不是运行期 Graph/scratch 峰值。BF16 没有替换层，其 prepared 栏不适用。'
              'AWQ 使用适配器去重统计的完整 weights/scales/prescale，排除 workspace；'
              '通用计数遗漏其外部 Parameter，不用于该行比较。', '',
              '| 方法 | 请求 MMA/C0 范围 | prepared weight MiB | 模型 state GiB | W1 支持 MiB |',
              '|---|---:|---:|---:|---:|']
    for name in ORDER:
        curve = curves[f'rbench|{name}']
        cost = quality['costs'][name]
        ratio = cost['mma_ratio_range']
        mma = '—' if ratio is None else f'{ratio[0]:.6f}–{ratio[1]:.6f}'
        prepared = '—' if name == 'bf16' else f"{curve['prepared_weight_bytes']/2**20:.2f}"
        w1 = (cost.get('offline_cost') or {}).get('w1_bytes')
        w1_text = '—' if w1 is None else f'{w1/2**20:.2f}'
        lines.append(f"| {LABELS[name]} | {mma} | {prepared} | {curve['resident_model_state_bytes']/2**30:.3f} | {w1_text} |")
    lines += ['',
              '## 原始证据', '',
              '- `summary.json`：质量、描述性配对区间、各档吞吐和停止原因。',
              '- `throughput_raw.json.gz`：逐波边界计时、显存、硬件/运行资产及内核记录。',
              '- `quality_raw.json.gz`、`generation_receipts.json.gz`：逐场景/种子指标与输出索引/哈希。',
              '- `manifest.json.gz`、`throughput_suite.json.gz`：输入与执行协议；复测命令见仓库文档。',
              '- `fresh_1p5_verification.json.gz`：1.5× 独立重跑的 32 个输出核验。', '']
    a.archive.mkdir(parents=True, exist_ok=True)
    write(a.archive/'summary.json', result)
    (a.archive/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (a.archive/'throughput.csv').open('w', newline='') as stream:
        fields = ['benchmark', 'case', 'method', 'batch', 'videos_per_second', 'frames_per_second',
                  'request_p50_seconds', 'request_p90_seconds', 'peak_allocated_bytes',
                  'peak_reserved_bytes', 'numerical_checked', 'performance_eligible']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for curve in curves.values():
            for row in curve['rows']:
                writer.writerow(dict(benchmark=curve['benchmark'], case=curve['case_id'],
                    method=curve['method'], batch=row['batch'],
                    videos_per_second=row['median_videos_per_second'],
                    frames_per_second=row['median_videos_per_second']*61,
                    **{k: row[k] for k in fields[6:]}))
    with (a.archive/'quality.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['benchmark', 'method', *FIELDS])
        writer.writeheader()
        for dataset, methods in quality['summaries'].items():
            for method, metrics in methods.items():
                writer.writerow(dict(benchmark=dataset, method=method, **metrics))
    for name, value in [('throughput_raw', raw), ('quality_raw', quality), ('manifest', manifest),
                        ('throughput_suite', suite), ('generation_receipts', receipts),
                        ('fresh_1p5_verification', fresh), ('fresh_1p5_receipts', fresh_receipts)]:
        archive_json(a.archive/(name+'.json.gz'), value)
    print(json.dumps(dict(complete=True, report=str(a.archive/'REPORT.md'))))


if __name__ == '__main__':
    main()

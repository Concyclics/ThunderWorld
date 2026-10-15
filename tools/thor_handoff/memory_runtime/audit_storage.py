"""Reconcile target weights, resident model state and request peaks from receipts.

CPU-only: reads frozen memory-v8 results and a hash-bound backend source snapshot.
Does not run inference, modify model assets, or estimate a new latency result.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

GIB = 2**30
MIB = 2**20
METHODS = ['bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint']
LABELS = {'bf16': 'BF16', 'fp8': 'FP8', 'fp4_base': 'W4A4',
          'b1.5_protected_n': 'Repair 1.5×', 'b2_joint': 'Repair 2×'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scope_path = args.campaign.parent / 'checkpoint-scope.json'
    scopes = json.loads(scope_path.read_text())
    models, rows, sources = {}, [], {str(scope_path): sha(scope_path)}
    for model in ['edge', 'nano']:
        receipts, paths = {}, {}
        for method in METHODS:
            matches = list((args.campaign / 'runs').glob(f'*-{model}-compact-{method}-320x192x61/result.json'))
            if len(matches) != 1:
                raise ValueError(f'Expected one frozen result: {model}/{method}')
            path = matches[0]
            receipts[method] = json.loads(path.read_text())
            if not receipts[method]['complete']:
                raise ValueError('Incomplete source result')
            paths[method] = str(path)
            sources[str(path)] = sha(path)

        baseline = receipts['bf16']['runner']['resident_model_state_bytes']
        baseline_peak = max(r['peak_allocated_bytes'] for r in receipts['bf16']['rows'])
        reference = receipts['fp4_base']['runner']
        for name in ['cosmos/runner.py', 'cosmos/linears.py', '_native.py',
                     '_sm110_cutlass.py', 'cutlass/layout.py']:
            source = args.backend / name
            if sha(source) != reference['source_sha256'][name]:
                raise ValueError('Backend snapshot does not match measured receipt: ' + name)
            sources[str(source)] = sha(source)

        # Shapes come from executed projections. Deduplicate CFG/phase variants.
        dims = {}
        for r in reference['observed_shapes']:
            shape = r['n'], r['k']
            if r['module'] in dims and dims[r['module']] != shape:
                raise ValueError('Changing weight shape')
            if any(x % 128 for x in shape):
                raise ValueError('Unexpected weight-tail padding; extend this audit first')
            dims[r['module']] = shape
        if set(dims) != set(reference['targets']):
            raise ValueError('Missing target projection')
        params = sum(n * k for n, k in dims.values())
        bf16_target = params * 2
        if bf16_target != scopes[model]['quantized_generation_weight_scope_bf16_bytes']:
            raise ValueError('Observed shapes disagree with checkpoint-header inventory')
        common = baseline - bf16_target
        codes = params // 2
        scale = params // 16  # One E4M3 byte per K16 group, no N/K padding here.
        prescale = sum(k for n, k in dims.values()) * 4
        sensitivity = prescale
        prepared_base = reference['prepared_weight_bytes']
        remaining_base_metadata = prepared_base - codes - 2 * scale - prescale
        if remaining_base_metadata < 0:
            raise ValueError('Inconsistent base weight packing inventory')
        details = dict(target_tensors=len(dims), target_parameters=params,
            bf16_target_bytes=bf16_target, bf16_resident_bytes=baseline,
            target_fraction_of_bf16_resident=bf16_target / baseline,
            target_fraction_of_transformer_checkpoint=scopes[model]['generation_fraction'],
            unchanged_resident_bytes=common, fp4_code_bytes=codes,
            native_scale_bytes=scale, cutlass_scale_layout_bytes=scale,
            prescale_bytes=prescale, sensitivity_bytes=sensitivity,
            other_base_metadata_bytes=remaining_base_metadata,
            single_scale_payload_effective_bits=4.5,
            retained_two_scale_payload_effective_bits=5.0,
            ideal_4bit_only_max_resident_reduction_at_current_scope=0.75 * bf16_target / baseline,
            exact_resident_reconciliation=True)
        models[model] = details
        for method in METHODS:
            raw = receipts[method]
            runner = raw['runner']
            resident = runner['resident_model_state_bytes']
            peak = max(r['peak_allocated_bytes'] for r in raw['rows'])
            ledger = raw['rows'][0].get('compute_ledger') or {}
            w1 = ledger.get('w1_support_union_bytes', 0)
            metadata_extra = 0
            if method in ('bf16', 'fp8'):
                target = bf16_target if method == 'bf16' else runner['prepared_weight_bytes']
            else:
                metadata_extra = runner['prepared_weight_bytes'] - prepared_base - w1
                if metadata_extra < 0:
                    raise ValueError('W1 payload exceeds prepared storage increment')
                target = codes + 2 * scale + prescale + sensitivity + remaining_base_metadata + w1 + metadata_extra
                if resident - common - runner['prepared_weight_bytes'] != sensitivity:
                    raise ValueError('Registered sensitivity buffer reconciliation failed')
            if resident != common + target:
                raise ValueError(f'Resident bytes do not reconcile: {model}/{method}')
            rows.append(dict(model=model, method=method, target_bf16_bytes=bf16_target,
                quantized_target_resident_bytes=target,
                target_storage_reduction_pct=100 * (1 - target / bf16_target),
                unchanged_resident_bytes=common, resident_model_state_bytes=resident,
                resident_reduction_pct=100 * (1 - resident / baseline),
                request_peak_allocated_bytes=peak,
                peak_reduction_pct=100 * (1 - peak / baseline_peak),
                non_model_allocated_remainder_at_peak_bytes=peak - resident,
                w1_code_scale_payload_bytes=w1, incremental_w1_metadata_bytes=metadata_extra,
                source=paths[method], source_sha256=sources[paths[method]]))
    sources[str(Path(__file__).resolve())] = sha(__file__)
    result = dict(scope='Thor memory-v8, 320x192x61, postwarm, three requests per method/model',
        methodology='CPU reconciliation of frozen postwarm receipts, independent observed N/K shapes, checkpoint-header totals and hash-matched backend source. No new live-tensor walk or GPU experiment.',
        models=models, rows=rows, sources=sources,
        caveats=['Target resident includes codes, both scale layouts, prescale, sensitivity, residual payload and indices.',
                 'Non-model allocated remainder is subtraction at request peak, not a separate activation-only peak.',
                 'Other base metadata is the small measured remainder; this audit does not assign each metadata tensor a name.',
                 'Disk checkpoints/packages are not measured by CUDA resident or peak counters.',
                 'Two scale layouts are currently consumed by different mainloop/repair paths; removing one requires a validated backend change.',
                 'These are memory measurements; no new speed or quality claim.'])
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    with (args.output / 'breakdown.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# FP4 存储核对：权重、整模型常驻与请求峰值', '',
        '结论：最新 Thor 记录的整模型字节与请求峰值没有发现算术错误；此前的 20%–28% 指完整请求峰值节省，不是量化权重的压缩率。下面三种分母必须分开。', '',
        'CPU 只读核对冻结结果、实际执行的 N/K 形状、checkpoint 文件头清单及与测量 hash 一致的后端源码；没有重新运行 GPU，也没有改变论文主表。', '',
        '| 模型 | 方法 | 目标权重常驻 GiB，含元数据 | 目标权重减少 | 整模型常驻 GiB | 常驻减少 | 请求峰值 GiB | 峰值减少 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {LABELS[r['method']]} | {r['quantized_target_resident_bytes']/GIB:.3f} | {r['target_storage_reduction_pct']:.2f}% | {r['resident_model_state_bytes']/GIB:.3f} | {r['resident_reduction_pct']:.2f}% | {r['request_peak_allocated_bytes']/GIB:.3f} | {r['peak_reduction_pct']:.2f}% |")
    lines += ['', '## 1. 量化覆盖范围是主要原因', '',
        '| 模型 | 原目标 BF16 权重 GiB | 未改动常驻部分 GiB | 目标占整模型常驻 | 目标占 Transformer checkpoint |',
        '|---|---:|---:|---:|---:|']
    for model, d in models.items():
        lines.append(f"| {model} | {d['bf16_target_bytes']/GIB:.3f} | {d['unchanged_resident_bytes']/GIB:.3f} | {100*d['target_fraction_of_bf16_resident']:.2f}% | {100*d['target_fraction_of_transformer_checkpoint']:.2f}% |")
    lines += ['', '当前替换的是生成分支投影与 FFN。AR/理解分支的其他投影与 FFN、embedding/lm_head 和公共组件仍有大量高精度存储；此处未把“非目标部分”全部叫作 AR，也未假设全模型已量化。', '',
        '即使量化范围不变且仅存理想四位编码、不存 scale/残差/索引，整模型常驻减少的上界也只有 Edge **25.91%**、Nano **32.77%**。这不是将全模型压成四分之一的实验。', '',
        '## 2. 四位编码之外还有实际存储', '',
        '| 项目 | Edge MiB | Nano MiB |', '|---|---:|---:|']
    for key, name in [('fp4_code_bytes','W0 FP4 codes'), ('native_scale_bytes','W0 原布局 E4M3 scales'),
                      ('cutlass_scale_layout_bytes','W0 CUTLASS 布局 scales'), ('prescale_bytes','Prescale'),
                      ('sensitivity_bytes','激活选择 sensitivity'), ('other_base_metadata_bytes','其余基础元数据')]:
        lines.append(f"| {name} | {models['edge'][key]/MIB:.4f} | {models['nano'][key]/MIB:.4f} |")
    lines += ['', '每 16 个 FP4 元素有一个一字节 scale；只保留一份时，编码加 scale 为 4.5 bit/weight。当前 Thor 同时保留 Native 与 CUTLASS 两种 scale 布局，合计为 5 bit/weight，再加少量 prescale/sensitivity/索引。第二份 scale 分别为 **84/414 MiB**，属于真实后端占用，不能在统计时删掉。', '',
        '源码中 `_native.py` 持有 `ws`；`_sm110_cutlass.py` 创建独立 `w0_scales=swizzle(op.ws)`；主 GEMM 使用 CUTLASS 布局，修复参数仍传入原布局。需要统一读取布局后才能安全释放副本，不能直接清空 `ws`。', '',
        '## 3. 稀疏 W1 的增量比公共高精度部分小', '',
        '| 模型 | 修复 | W1 codes+scales MiB | 额外 W1 索引等 MiB |', '|---|---|---:|---:|']
    for r in rows:
        if r['method'] in ('b1.5_protected_n', 'b2_joint'):
            lines.append(f"| {r['model']} | {LABELS[r['method']]} | {r['w1_code_scale_payload_bytes']/MIB:.4f} | {r['incremental_w1_metadata_bytes']/MIB:.4f} |")
    lines += ['', '这些是离线权重支持并集。在线 A1 修复不等于再常驻一整份模型权重；其缓冲随其他激活、Graph 分配一起进入请求峰值。', '',
        '## 4. 统计与表述应怎样调整', '',
        '- 论文并列报告“量化目标常驻字节”“完整模型常驻字节”“完整请求 peak allocated”，分别标注相对 BF16 的减少，不能都叫作模型压缩率。',
        '- 当前 inventory 按 storage 地址去重；同一权重跨阶段/CFG 的别名不重复计数。不同布局的真实存储必须分别计入。最新账本可精确重建全部十行整模型常驻字节。',
        '- 旧 PRO 6000 的 prewarm 记录曾漏掉惰性创建的量化权重，已在前次数据整理中标为不完整；不能恢复使用那批值。当前判断只针对 memory-v8 完成预热后的记录。',
        '- allocated 与 reserved 分列；Thor 进程 RSS 和 CUDA allocated 不能相加。磁盘原 BF16 checkpoint 仍保留，CUDA inventory 不能证明整个部署目录缩小了多少。',
        '- 要明显增加整模型存储优势，优先研究未量化的 AR/理解分支权重和 embedding/lm_head，再验证质量；统一 scale 布局可减少真实冗余，但只削减稀疏 W1 的收益有限。', '',
        '## 核对公式', '',
        '`常驻 = 未改动部分 + W0 codes + 原 scales + CUTLASS scales + prescale + sensitivity + 基础元数据 + W1 payload + 额外索引`', '',
        'FP4 三种方法按上述分项复算，与每行原始 `resident_model_state_bytes` **逐字节相等**。请求峰值减常驻所得余量是非模型活跃分配的余量，不能命名为独立测得的 activation 峰值。来源、源码 hash、未解释到单张量的少量元数据余量在 `audit.json`；十行字节明细在 `breakdown.csv`。', '']
    (args.output / 'README.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(args.output), rows=len(rows), resident_reconciled=True)))


if __name__ == '__main__':
    main()

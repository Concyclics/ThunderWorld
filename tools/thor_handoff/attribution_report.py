"""Compare matched Thor FP4/FP8 requests without mixing profiler timings.

This is a speed-only diagnostic, not a generation-quality or hardware-ceiling
claim. An explicitly selected BF16-attention ablation changes arithmetic.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


LABELS = {
    'fp8': 'FP8 W8A8 + BF16 attention',
    'fp4-original': 'W4A4：原 attention',
    'fp4-native': 'W4A4：原生 FP4 attention',
    'fp4-bf16-attention': 'W4A4 投影 + BF16 attention（归因）',
    'fp4-row-major': 'W4A4：原生 FP4 attention + 连续 V 布局',
}


def percentile(values, p):
    values = sorted(values)
    pos = (len(values)-1)*p
    i = int(pos)
    return values[i] + (values[min(i+1,len(values)-1)]-values[i])*(pos-i)


def profile_breakdown(root):
    profiles = {}
    for name in ('native','fp8'):
        path = root/f'kernels-{name}_cuda_gpu_kern_sum.csv'
        if not path.exists():
            return None
        request = json.loads((root/f'profile-request-{name}'/'result.json').read_text())
        if not request['complete'] or len(request['rows']) != 1 or not request['rows'][0]['profiled']:
            raise ValueError(f'Incomplete profiler capture: {name}')
        bins = {k:dict(seconds=0.,calls=0) for k in ('projection_gemm','activation_pack','generation_attention','other')}
        for row in csv.DictReader(path.open()):
            kernel = row['Name']
            group = 'other'
            if name == 'native':
                if kernel == 'gemm':group = 'projection_gemm'
                elif kernel in ('scaled_amax','finish_amax','pack_activation','pointwise_amax'):group='activation_pack'
                elif kernel in ('streaming_attention','pack_qk','pack_v','combine'):group='generation_attention'
            else:
                if 'cutlass::device_kernel<' in kernel:group='projection_gemm'
                elif 'dynamic_per_token_scaled_fp8_quant_kernel' in kernel:group='activation_pack'
                elif 'pytorch_flash::flash_fwd_kernel<' in kernel and int(row['Instances']) == 1960:group='generation_attention'
            bins[group]['seconds'] += int(row['Total Time (ns)'])/1e9
            bins[group]['calls'] += int(row['Instances'])
        if bins['projection_gemm']['calls'] != 11760:
            raise ValueError('Unexpected projection scope in profiler')
        profiles[name] = dict(groups=bins, source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            profiled_request_key=[request['rows'][0][k] for k in ('case','seed','steps')])
    if profiles['native']['profiled_request_key'] != profiles['fp8']['profiled_request_key']:
        raise ValueError('Profiler inputs do not match')
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-measurements', type=Path,
        help='Optional v2 repair results, used only for matched v2/v3 comparisons')
    args = parser.parse_args()
    rows = []
    manifest = None
    keys = None
    for name, label in LABELS.items():
        path = args.measurements/name/'result.json'
        if not path.exists():
            if name == 'fp4-row-major':
                continue
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        if not data['complete'] or data['operation'] != 'measure':
            raise ValueError(f'Incomplete or instrumented result: {path}')
        current_keys = sorted((r['case'], r['seed'], r['steps']) for r in data['rows'])
        if len(set(current_keys)) != len(current_keys) or len(current_keys) != 6:
            raise ValueError(f'Expected six distinct paired requests: {path}')
        if keys is None:
            keys = current_keys
            manifest = data['fixed_manifest_sha256']
        if current_keys != keys or data['fixed_manifest_sha256'] != manifest:
            raise ValueError(f'Unmatched inputs or manifest: {path}')
        for r in data['rows']:
            if (r['debug'] != 'off' or r['execution'] != 'graph' or
                    r['steps'] != 35 or r.get('profiled') or r['calls'] != 70):
                raise ValueError(f'Different measurement contract: {path}')
        elapsed = [r['wall_seconds'] for r in data['rows']]
        rows.append(dict(name=name, label=label, requests=len(elapsed),
            median_seconds=statistics.median(elapsed), p90_seconds=percentile(elapsed,.9),
            generated_fps=61/statistics.median(elapsed),
            allocated_gib=max(r['peak_allocated_bytes'] for r in data['rows'])/1024**3,
            reserved_gib=max(r['peak_reserved_bytes'] for r in data['rows'])/1024**3,
            per_case={str(i):statistics.median(r['wall_seconds'] for r in data['rows']
                      if r['case_index']==i) for i in (0,1)},
            variant=data.get('variant'), config=data['effective_config'],
            runtime_options=data['runtime_options'],
            attribution_ablation=data.get('attribution_ablation'),
            source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    indexed = {r['name']:r for r in rows}
    before = indexed['fp4-original']['median_seconds']
    after = indexed['fp4-native']['median_seconds']
    fp8 = indexed['fp8']['median_seconds']
    report = dict(complete=True, performance_eligible=False, quality_evaluation=False,
        fixed_manifest_sha256=manifest, paired_requests=keys, rows=rows,
        native_attention_speedup=before/after,
        native_attention_latency_reduction=1-after/before,
        remaining_latency_excess_over_fp8=after/fp8-1,
        timing_scope='Full request including decode and CPU output, excluding model preparation and file serialization',
        limitations=['Two inputs, three seeds; MAXN without locked clocks.',
                     'Sequential diagnostic measurements, not five randomized rounds.',
                     'BF16-attention ablation changes arithmetic; no generation-quality conclusion.',
                     'Profiler kernel sums are not request latency and may include overlapping work.'])
    report['profile_breakdown'] = profile_breakdown(args.measurements)
    report['repair_layout_comparisons'] = []
    if args.previous_measurements:
        for tag,budget in [('15',1.5),('20',2.)]:
            oldpath=args.previous_measurements/f'e2e-transposed{tag}'/'result.json'
            newpath=args.measurements/f'repair{tag}-row-major'/'result.json'
            old=json.loads(oldpath.read_text());new=json.loads(newpath.read_text())
            if not old['complete'] or not new['complete']:
                raise ValueError('Incomplete repair comparison')
            pairs=lambda data:{(r['case'],r['seed']):r for r in data['rows']}
            oldrows=pairs(old);newrows=pairs(new)
            if oldrows.keys()!=newrows.keys() or old['fixed_manifest_sha256']!=new['fixed_manifest_sha256']:
                raise ValueError('Unmatched repair comparison')
            if old['fixed_manifest_sha256']!=manifest:raise ValueError('Different repair manifest')
            for key,row in newrows.items():
                if (row['debug']!='off' or row['execution']!='graph' or row['steps']!=35 or
                    row.get('profiled') or not row['compute_ledger']['feasible'] or
                    row['compute_ledger']['ratio']!=oldrows[key]['compute_ledger']['ratio']):
                    raise ValueError('Repair budget/measurement contract changed')
            before=statistics.median(r['wall_seconds'] for r in old['rows'])
            after=statistics.median(r['wall_seconds'] for r in new['rows'])
            report['repair_layout_comparisons'].append(dict(budget=budget,
                before_seconds=before,after_seconds=after,latency_reduction=1-after/before,
                generated_fps=61/after,allocated_gib=max(r['peak_allocated_bytes'] for r in new['rows'])/1024**3,
                ratio_max=max(r['compute_ledger']['ratio'] for r in new['rows']),
                output_hash_matches=sum(newrows[k]['sha256']==oldrows[k]['sha256'] for k in newrows),
                requests=len(newrows),variant=new['variant'],
                sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (oldpath,newpath)}))
        before=indexed['fp4-original']['median_seconds']
        after=indexed['fp4-native']['median_seconds']
    lines = ['# Thor W4A4 / FP8 速度归因', '',
        'Edge，同一模型版本、两个固定输入各三个种子，320×192、61 帧、35 步、CFG=6；Graph、debug off。',
        '计时包括编码、生成、解码和 CPU 输出；不含加载、建图、MP4 保存。只测速度，不评估生成质量。',
        '这是独占检查下的顺序诊断，MAXN 未锁频，不是五轮随机交错的正式性能排行榜。', '',
        '| 实现 | 中位秒 | P90 秒 | 生成 FPS | allocated GiB |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['label']} | {row['median_seconds']:.3f} | {row['p90_seconds']:.3f} | {row['generated_fps']:.3f} | {row['allocated_gib']:.3f} |")
    lines += ['',
        f'仅替换 attention 后，W4A4 延迟下降 {100*(1-after/before):.1f}%，加速 {before/after:.3f}×；仍比本轮 FP8 请求慢 {100*(after/fp8-1):.1f}%。',
        '',
        '旧 attention 的 PTX 使用 BF16 MMA，且承担 FP4 Q/K/V 和 P 量化成本。新 attention 显式发射原生 NVFP4 tcgen05。GEMM 路径没有在这项替换中改变。',
        'FP8 投影使用 vLLM 的原生 CUTLASS scaled_mm，在线 per-token E4M3 量化；其 attention 是 BF16。普通 W4A4 没有残差修复，不应把差距归因于修复预算。',
        'W4A4 投影的在线 global amax、尺度归约、K16 E4M3 scale / E2M1 编码与 GEMM 是独立执行阶段；权重在准备期量化。',
        'BF16 attention 行仅用于定位当前 attention 实现的成本，改变了算术，不能当作全 FP4 的优化结果。',
        '']
    if report['profile_breakdown']:
        profiles=report['profile_breakdown']
        lines += ['## 同输入 Nsight 分解', '',
            '单个 PAI-Bench 请求的 GPU kernel 累计时间，独立于上面的无探针请求计时。FP4 列为原生 attention v2；这里没有使用含修复预算的 profile。', '',
            '| 部分 | W4A4 秒 | FP8 秒 |', '|---|---:|---:|']
        for group,label in [('projection_gemm','生成投影 GEMM（各 11,760 次）'),
                            ('activation_pack','激活量化相关（FP4 含 FFN pointwise 融合）'),
                            ('generation_attention','生成 attention（FP4 含 QKV 打包与 combine）'),
                            ('other','其余内核')]:
            lines.append(f"| {label} | {profiles['native']['groups'][group]['seconds']:.3f} | {profiles['fp8']['groups'][group]['seconds']:.3f} |")
        lines += ['',
            '当前 W4A4 GEMM 本身慢于 FP8 的 CUTLASS GEMM；FP4 激活量化相关累计时间没有更高。应优先改进 GEMM 数据流水、scale 布局与片上累加，其次优化 attention 的加载和 P 量化流水。',
            '“更多量化步骤”是代码事实，但不能替代测量，不能据此声称量化开销解释了主要差距。其余时间含 VAE、布局转换、归一化、逐元素计算和未量化部分。', '']
    if report['repair_layout_comparisons']:
        lines += ['## 布局优化对修复版本的收益', '',
            '只修改 packed-V 物理布局；W0/W1、在线选择和逐请求计算比率不变，未移除修复项。', '',
            '| 预算 | v2 秒 | 连续 V 布局 v3 秒 | 延迟下降 | 生成 FPS |',
            '|---|---:|---:|---:|---:|']
        for r in report['repair_layout_comparisons']:
            lines.append(f"| {r['budget']:g}× | {r['before_seconds']:.3f} | {r['after_seconds']:.3f} | {r['latency_reduction']:.1%} | {r['generated_fps']:.3f} |")
        qualification=json.loads((args.measurements/'row-major-qualified.json').read_text())
        if not qualification['complete'] or len(qualification['tests'])!=24 or not all(r['native_transposed_v_bitwise_equal'] for r in qualification['tests']):
            raise ValueError('V-layout bitwise qualification missing')
        for tool,marker in [('memcheck','ERROR SUMMARY: 0 errors'),('racecheck','0 errors, 0 warnings')]:
            if marker not in (args.measurements/f'row-major-{tool}.log').read_text():
                raise ValueError('V-layout sanitizer qualification missing')
        lines += ['', '24 项算子检查与原生 v2 逐位一致，memcheck/racecheck 通过。完整请求哈希匹配数与计算比率见 JSON；未做生成质量评分。', '']
    densepath=args.measurements/'dense-cutlass.json'
    if densepath.exists():
        dense=json.loads(densepath.read_text())
        report['dense_cutlass_probe']=dict(source=str(densepath),sha256=hashlib.sha256(densepath.read_bytes()).hexdigest(),result=dense)
        lines += ['## 原生 CUTLASS FP4 探测', '',
            '算子探测使用相同的 FP4 codes/scales，仅转换 scale 布局；未接入修复版本或以上请求计时。']
        if dense.get('complete') and dense.get('rows'):
            lines += ['', '| M/N/K | Triton GEMM ms | CUTLASS GEMM ms | Triton 含打包 ms | CUTLASS 含打包及布局转换 ms |',
                '|---|---:|---:|---:|---:|']
            for r in dense['rows']:
                lines.append(f"| {'/'.join(map(str,r['shape']))} | {r['fp4_triton_gemm_ms']:.4f} | {r['fp4_cutlass_gemm_ms']:.4f} | {r['fp4_triton_with_pack_ms']:.4f} | {r['fp4_cutlass_with_pack_and_swizzle_ms']:.4f} |")
        else:lines += ['', '当前探测状态：'+str(dense.get('skipped',dense.get('error','未完成')))]
        lines += ['', '算子收益不能直接外推到完整修复请求；不能通过多次独立 BF16 输出 GEMM 替换共享 FP32 累加器而不重新验证语义。', '']
    lines += ['各阶段收益不可相乘；吞吐上限需进一步进行内核优化和独占复测。', '']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ('native_attention_speedup','native_attention_latency_reduction','remaining_latency_excess_over_fp8')}))


if __name__ == '__main__':
    main()

"""Summarize completed full-step comparisons, never substitute smoke results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edge', type=Path, required=True)
    p.add_argument('--nano', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    reports = {}
    for name, path in [('Edge', a.edge), ('Nano', a.nano)]:
        d = json.loads(path.read_text())
        assert d['complete'] and d['args']['steps'] == 35 and d['args']['rounds'] >= 3, path
        assert not any(d['transformer_loading_info'].values()), path
        assert len(d['rows']) == len(d['args']['modes'])*(d['args']['rounds']+1), path
        reports[name] = d
        shutil.copy2(path, a.output/f'{name.lower()}_result.json')
        medians = {mode: s['wall_median_seconds'] for mode, s in d['summary'].items()}
        for mode in d['args']['modes']:
            rows = [r for r in d['rows'] if r['mode'] == mode]
            timed = [r for r in rows if r['timing_sample']]
            quality = [r for r in rows if r['round'] in (0, d['args']['rounds'])]
            old = mode.replace('optimized', 'original')
            repeat_hashes = {r['video_sha256'] for r in timed}
            records.append({'model': name, 'mode': mode, 'offload': d['offload_policy'],
                            'wall_median_s': medians[mode],
                            'wall_min_s': min(r['wall_seconds'] for r in timed),
                            'wall_max_s': max(r['wall_seconds'] for r in timed),
                            'transformer_median_s': statistics.median(r['transformer_seconds'] for r in timed),
                            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in timed)/2**30,
                            'bf16_over_candidate': medians['bf16']/medians[mode],
                            'original_over_optimized': medians[old]/medians[mode] if mode.endswith('optimized') else None,
                            'bf16_ssim_two_seeds': [r['vs_bf16']['ssim_mean'] for r in quality] if mode != 'bf16' else [1., 1.],
                            'original_fp4_identical': all(r['vs_original_fp4']['identical'] for r in rows) if mode.endswith('optimized') else None,
                            'same_seed_repeats_identical': len(repeat_hashes)==1,
                            'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for key in ('steps', 'frames', 'height', 'width', 'seed', 'quality_seed', 'cuda_graph'):
        assert reports['Edge']['args'][key] == reports['Nano']['args'][key], key
    assert reports['Edge']['image_sha256'] == reports['Nano']['image_sha256']
    output = {'complete': True, 'cuda_graph': reports['Edge']['args']['cuda_graph'],
              'scope': 'Full Generator at one image/prompt and two seeds, not task success or model quality ranking',
              'timing_method': 'Three same-seed requests per mode, sequential modes after warmup; media export excluded', 'rows': records}
    (a.output/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    with (a.output/'latency.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    spec = reports['Edge']['args']
    execution = '全部方案共用外层 CUDA Graph' if spec['cuda_graph'] else '全部方案使用 eager 调用'
    lines = ['# Cosmos3 Generator 端到端实测', '',
             f"同一图像/提示词，{spec['width']}×{spec['height']}、{spec['frames']} 帧、{spec['steps']} 步、CFG 6；{execution}。每方案同 seed 至少三次计时，另一个 seed 做完整轨迹对照。包含图像编码、分词、去噪、VAE 解码；不含加载、JIT、权重准备、safety checker、MP4 编码。", '',
             '| 模型 | 方案 | 总延迟中位数 / 范围 s | Transformer s | PyTorch 峰值 GiB | 相对 BF16 | 相对原始 FP4 | BF16 SSIM，两 seed |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        speed = f"{r['original_over_optimized']:.3f}×" if r['original_over_optimized'] is not None else '—'
        q = ' / '.join(f'{x:.4f}' for x in r['bf16_ssim_two_seeds'])
        lines.append(f"| {r['model']} | {r['mode']} | {r['wall_median_s']:.3f} / [{r['wall_min_s']:.3f}, {r['wall_max_s']:.3f}] | {r['transformer_median_s']:.3f} | {r['peak_allocated_gib']:.2f} | {r['bf16_over_candidate']:.3f}× | {speed} | {q} |")
    lines += ['', '倍数大于 1 表示候选更快。PyTorch 显存不含桌面、驱动及其他进程占用。FFN 为模型自身 RTN W4 + full ACT2；joint 另使用 generation attention 75% 双侧修复。没有迁移 Nano layer-0 的 AWQ prescale、校准支持或 head atlas。', '',
              '| 模型 / 优化方案 | 与原始 FP4 视频逐位一致 | 同 seed 三次输出一致 | offload |',
              '|---|---|---|---|']
    for r in records:
        if r['mode'].endswith('optimized'):
            lines.append(f"| {r['model']} / {r['mode']} | {r['original_fp4_identical']} | {r['same_seed_repeats_identical']} | {r['offload']} |")
    lines += ['', '原始计时、调用计数、加载一致性、源码哈希及视频路径见 [Edge](edge_result.json) / [Nano](nano_result.json)。SSIM 只衡量与同模型参考的一致性，不代表语义、物理正确性或机器人任务成功率。']
    (a.output/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

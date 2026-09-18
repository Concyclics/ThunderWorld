"""Export audited output FPS and single-request PSNR/SSIM/LPIPS means."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from baseline_sweep_report import LABELS, ORDER
from baselines import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = args.archive
    summary = json.loads((root/'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Complete measurements required')
    if (root/'verification.json').exists():
        verification = json.loads((root/'verification.json').read_text())
        if not verification['complete'] or verification['summary_sha256'] != sha(root/'summary.json'):
            raise ValueError('Changed or incomplete audited measurements')
    manifest = json.loads(gzip.decompress((root/'manifest.json.gz').read_bytes()))
    frames = {e['case']['id']: e['case']['frames'] for e in manifest['cases']}
    rows = []
    for benchmark in ('rbench', 'paibench'):
        for method in ORDER:
            curve = summary['throughput'][benchmark+'|'+method]
            single = next(r for r in curve['rows'] if r['batch'] == 1)
            best = curve['best']
            metrics = summary['quality'][benchmark].get(method, {})
            rows.append(dict(benchmark=benchmark, method=method, case=curve['case_id'],
                output_frames=frames[curve['case_id']], single_fps=single['median_videos_per_second']*frames[curve['case_id']],
                best_total_fps=best['median_videos_per_second']*frames[curve['case_id']],
                best_batch=best['batch'], single_request_seconds=single['median_group_seconds'],
                psnr_db=metrics.get('psnr_db'), ssim=metrics.get('ssim_mean'), lpips=metrics.get('lpips'),
                quality_scope='single request development cohort; BF16 reference omitted',
                performance_eligible=summary['performance_eligible']))
    lookup = {(r['benchmark'], r['method']): r for r in rows}
    lines = ['# 输出 FPS 与 PSNR / SSIM / LPIPS', '',
        '由已完成并核验的同一批测量换算，没有重新生成或计时。输出 FPS = 每视频输出帧数 × 视频/秒。'
        '本轮每段输出 61 帧；单路列是 61 / 单请求完整耗时，并发列合计所有请求输出。'
        '若按排除条件首帧的 60 帧计数，所有 FPS 乘 60/61，速度比和排序不变。', '',
        '本机 PRO 6000，Edge，35 步，CFG 6，192×320，debug off，含 AR 预填充、视频解码和 CPU 输出转换。'
        '每数据集各一个固定输入，每档 15 组计时。并发使用独立模型副本；B>1 只测速度，未检查结果一致性。'
        '这是固定实现诊断吞吐，performance_eligible=false。ARC/SharQ 保留现有 eager 适配，包含主机调度成本。', '',
        '## 生成速度', '',
        '| 方法 | RBench 单路 FPS | RBench 最佳总 FPS | PAI 单路 FPS | PAI 最佳总 FPS | 最佳并发数 R / P |',
        '|---|---:|---:|---:|---:|---:|']
    for method in ORDER:
        r, p = (lookup[d, method] for d in ('rbench', 'paibench'))
        lines.append(f"| {LABELS[method]} | {r['single_fps']:.2f} | {r['best_total_fps']:.2f} | "
                     f"{p['single_fps']:.2f} | {p['best_total_fps']:.2f} | {r['best_batch']} / {p['best_batch']} |")
    lines += ['', '## 单请求质量', '',
        '指标均与相同输入、种子的 BF16 输出比较：PSNR、SSIM 越高越好，LPIPS 越低越好。'
        '这衡量与 BF16 的一致性。BF16 是参考本身，不重复列为测得的质量结果。'
        '每场景先平均两个种子，再对场景平均；16 个冻结开发场景（RBench 9、PAI-Bench 7），不包含预留的 72 场景最终测试。', '']
    for benchmark in ('rbench', 'paibench'):
        lines += [f'### {benchmark}', '', '| 方法 | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |', '|---|---:|---:|---:|']
        for method in ORDER[1:]:
            row = lookup[benchmark, method]
            lines.append(f"| {LABELS[method]} | {row['psnr_db']:.2f} | {row['ssim']:.4f} | {row['lpips']:.4f} |")
        lines += ['']
    lines += ['1.5× 未全面超过 AWQ-lite，也未超过 ARCQuant 100%；2× 在两个数据集的三项均值均优于 AWQ-lite，'
              '对 ARC100 的 PAI LPIPS 接近。2.5× 在两个数据集的三项均值均优于 AWQ-lite、ARC100 和当前 SharQ。'
              '2.5× 尚未通过超过 FP8 的质量门槛。这些是开发集描述结果，配对区间与限制见 REPORT.md。', '',
              'AWQ 为 ModelOpt 官方 NVFP4_AWQ_LITE_CFG + 原生 Marlin W4A16。质量均值与并发速度分别报告。', '']
    (root/'FPS_QUALITY.md').write_text('\n'.join(lines)+'\n')
    with (root/'fps_quality.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(root/'fps_quality.json', dict(complete=True, source_summary_sha256=sha(root/'summary.json'),
        source_sha256=sha(__file__), frames_definition='all output frames',
        concurrent_quality_checked=False, performance_eligible=False, rows=rows))
    print(json.dumps(dict(complete=True, report=str(root/'FPS_QUALITY.md'), rows=len(rows))))


if __name__ == '__main__':
    main()

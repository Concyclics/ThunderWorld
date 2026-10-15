"""Matched-output quality and storage report for the expanded-I2V pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from expand_worker import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metrics',action='store_true')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    records={}
    for path in sorted((a.root/'runs').glob('*/result.json')):
        row=json.loads(path.read_text())
        if row.get('complete'):records[(row['model'],row['method'],row['profile_name'])]=(path,row)
    metrics_model=None
    if a.metrics:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'contribution_study'))
        from full_video_metrics import VideoMetrics
        metrics_model=VideoMetrics(device=a.device)
    rows=[];checks=[]
    for (model,method,profile),(path,run) in records.items():
        base_key=(model,'bf16','original')
        own_key=(model,method,'original')
        if base_key not in records:continue
        base_path,base=records[base_key]
        base_rows={(r['case'],r['seed']):r for r in base['rows']}
        own=records.get(own_key,(None,{}))[1]
        own_rows={(r['case'],r['seed']):r for r in own.get('rows',[])}
        samples=[]
        for r in run['rows']:
            key=r['case'],r['seed'];ref=base_rows[key]
            sample=dict(case=r['case'],seed=r['seed'],sha256=r['sha256'],bf16_equal=r['sha256']==ref['sha256'])
            if profile=='trim' and own_rows:
                sample['trim_bitwise_equal']=r['sha256']==own_rows[key]['sha256']
                if not sample['trim_bitwise_equal']:raise AssertionError('Trimming changed output: '+str((model,method,key)))
            if a.metrics:
                cache=path.parent/(r['output']+'.metrics.json')
                identity=dict(candidate_sha256=r['sha256'],reference_sha256=ref['sha256'])
                if cache.exists():
                    result=json.loads(cache.read_text())
                    if result['identity']!=identity:raise ValueError('Changed metric inputs')
                else:
                    for root,entry in ((path.parent,r),(base_path.parent,ref)):
                        with (root/entry['output']).open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
                        if d!=entry['npz_sha256']:raise ValueError('Changed NPZ')
                    candidate=np.load(path.parent/r['output'])['video'];reference=np.load(base_path.parent/ref['output'])['video']
                    for array,entry in ((candidate,r),(reference,ref)):
                        if hashlib.sha256(array.tobytes()).hexdigest()!=entry['sha256']['video']:raise ValueError('Changed video')
                    metrics_model.set_reference(reference)
                    result=dict(identity=identity,metrics=metrics_model.compare(candidate))
                    write(cache,result)
                sample['metrics']=result['metrics']['video']
            samples.append(sample)
        lat=[r['wall_seconds'] for r in run['rows']]
        entry=dict(model=model,method=method,profile=profile,samples=samples,n=len(samples),
            median_seconds=statistics.median(lat),p90_seconds=float(np.quantile(lat,.9)),
            generated_fps=61/statistics.median(lat),**run['measurements'],
            diagnostic_stages=run.get('diagnostics'),evidence=str(path),performance_eligible=False)
        if a.metrics:
            entry['mean_metrics']={key:float(np.mean([s['metrics'][key] for s in samples]))
                for key in ('lpips','ssim_mean','relative_l2','rmse')}
            values=[s['metrics']['psnr_db'] for s in samples]
            entry['mean_metrics']['psnr_db']=float(np.mean(values)) if all(v is not None for v in values) else None
        if profile=='trim' and own_rows:checks.append(dict(model=model,method=method,bitwise_equal=True,n=len(samples)))
        rows.append(entry)
    report=dict(scope='2 reused scenes (RBench/PAI-Bench), 3 seeds per scene, Edge/Nano, 35 steps, 320x192, 61 frames',
        quality_scope='BF16 consistency pilot; no full-benchmark or statistical-superiority claim',
        time_scope='Diagnostic local-device timing; no Thor/5090 extrapolation, no ranking update',
        rows=rows,trim_checks=checks,performance_eligible=False,
        metadata_note='Early raw expansion.protected scans precede DM replacement unless protected_scan_stage is present; do not use them as a final weight inventory. Resident bytes are measured after replacement and are unaffected.')
    write(a.output/'report.json',report)
    lines=['# I2V 裁剪与扩大量化验证','',report['scope'],'',
        '常驻模型与请求峰值分别列账。峰值减去常驻模型仍包含 Graph 池、输入、输出、激活和工作区，不是纯激活。',
        '所有耗时均为诊断数据；每行只覆盖两个已使用场景，不能据此宣布质量门槛通过。','',
        '| 模型 | DM 方法 | 扩展配置 | 常驻 GiB | 峰值 GiB | 非模型峰值 GiB | 中位秒 | LPIPS ↓ | SSIM ↑ | PSNR ↑ |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r.get('mean_metrics',{})
        def v(key):
            value=m.get(key);return '—' if value is None else f'{value:.5f}'
        lines.append(f"| {r['model']} | {r['method']} | {r['profile']} | {r['resident_model_bytes']/2**30:.3f} | {r['request_peak_allocated_bytes']/2**30:.3f} | {r['peak_non_model_remainder_bytes']/2**30:.3f} | {r['median_seconds']:.3f} | {v('lpips')} | {v('ssim_mean')} | {v('psnr_db')} |")
    lines+=['','AR FP4 四项是额外的 prefill 修复计算，1.5×/2×仍指原 DM/attention 预算，不能当作扩展后全请求预算。',
        'full2 = I2V 裁剪 + AR FP4 四项 + 双平面 embedding/辅助权重/VAE。VAE 仍使用 BF16 卷积，逐算子解码权重，不声称全模型 W4A4。',
        'full1 = 无修复 AR FP4 + 单平面 embedding/辅助权重/VAE，用于显示极限压缩的质量代价。',
        '— 表示未计算或与 BF16 逐位一致时 PSNR=+∞；JSON 通过 psnr_exact 区分。','']
    (a.output/'README.zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(rows=len(rows),trim_checks=checks,output=str(a.output))))


if __name__=='__main__':main()

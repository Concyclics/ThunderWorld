"""Audit and summarize the final interleaved three-way comparison."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted(a.results.glob('*.json')):
        d=json.loads(path.read_text())
        if not isinstance(d,dict) or 'latency_ms' not in d:continue
        assert d['verification'] is not None,path
        lat=d['latency_ms'];assert all(len(lat[k]['samples'])>=5 for k in ['bf16','fp4','original_fp4'])
        c=d['config'];shape=d['shape']
        rows.append({'kind':c['kind'],'asset':c['asset'],'preset':c['preset'],'M':shape['M'],
                     'shape':shape,'original_fp4_us':lat['original_fp4']['median']*1000,
                     'optimized_fp4_us':lat['fp4']['median']*1000,'bf16_us':lat['bf16']['median']*1000,
                     'original_over_optimized':lat['original_fp4']['median']/lat['fp4']['median'],
                     'bf16_over_optimized':lat['bf16']['median']/lat['fp4']['median'],
                     'latency_ms':lat,'verification':d['verification'],'source_sha256':d['source_sha256'],
                     'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'file':str(path),'config':c,
                     'actual_coverage':d.get('cost',{'activation':d.get('actual_activation_coverage'),'weight':d.get('actual_weight_coverage')})})
    if len(rows)!=60:raise ValueError(f'Expected 60 final cases, found {len(rows)}')
    report={'scope':'One-operator CUDA Graph replay; online packing included; not full-model speedups',
            'complete':True,'cases':rows,'summary':{}}
    for kind in ['ffn','attention']:
        group=[r for r in rows if r['kind']==kind]
        speed=[r['original_over_optimized'] for r in group]
        report['summary'][kind]={'cases':len(group),'speedup_min':min(speed),'speedup_median':statistics.median(speed),'speedup_max':max(speed),
                                'bf16_speedup_min':min(r['bf16_over_optimized'] for r in group),'bf16_speedup_max':max(r['bf16_over_optimized'] for r in group),
                                'decoded_error_max':max(r['verification']['decoded_reference_relative_l2'] for r in group),
                                'regressions_over_3_percent':[r['file'] for r in group if r['original_over_optimized']<1/1.03]}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['kind','asset','M','preset','original_fp4_us','optimized_fp4_us','bf16_us','original_over_optimized','bf16_over_optimized']
    with (a.output/'latency.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# RTX 5090 算子实测表','',
           '每项为同一进程中五轮随机交错 CUDA Graph 计时的中位数，包含当前输入的在线量化。完整样本范围、数值误差和源码哈希见 summary.json。',
           '', '| 算子/输入 | M | 档位 | 原始 FP4 μs | 优化 FP4 μs | BF16 μs | 对原始 FP4 | 对 BF16 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {Path(r['asset']).stem} | {r['M']} | {r['preset']} | {r['original_fp4_us']:.2f} | {r['optimized_fp4_us']:.2f} | {r['bf16_us']:.2f} | {r['original_over_optimized']:.2f}× | {r['bf16_over_optimized']:.2f}× |")
    (a.output/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()

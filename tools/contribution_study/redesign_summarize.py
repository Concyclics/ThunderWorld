"""Exploratory new-scene development results; never promote a final quality claim."""
import argparse,collections,gzip,json
from pathlib import Path
import numpy as np
from baselines import sha,write
from dataset_report import average


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--archive-label',default='development');a=p.parse_args()
    root=a.study/'development';manifest=json.loads((root/'manifest.json').read_text());progress=json.loads((root/'metrics/progress.json').read_text())
    if not progress['complete'] or progress['identity']['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('Complete matched development metrics required')
    fields=('lpips','ssim_mean','relative_l2','psnr_db');raw=[];summaries={}
    for benchmark in ('rbench','paibench'):
        summaries[benchmark]={}
        for name in manifest['methods']:
            if name=='bf16':continue
            values=[]
            for entry in manifest['cases']:
                if entry['benchmark']!=benchmark:continue
                f=root/'metrics'/name/(entry['case']['id']+'.json');row=json.loads(f.read_text())
                if not row['complete'] or row['provenance']['identity']!=progress['identity']:raise ValueError('Changed metric implementation')
                for side,method in (('candidate',name),('reference','bf16')):
                    receipt=root/'generation'/method/entry['case']['id']/'result.json'
                    if row['provenance'][side+'_sha256']!=sha(receipt):raise ValueError('Changed generation receipt')
                values.append({field:average([r['metrics']['video'][field] for r in row['rows']]) for field in fields})
                raw.append(row)
            summaries[benchmark][name]={field:average([v[field] for v in values]) for field in fields}
    result=dict(complete=True,manifest_sha256=sha(root/'manifest.json'),summaries=summaries,seed_metrics=raw,
        performance_eligible=False,final_quality_qualified=False,
        scope='Exploratory development means; one scene per category and two seeds per scene. No significance/promotion claim. Validation and final remain untouched.')
    lines=['# 低预算修复重设计：独立开发场景','',result['scope'],'',
        '所有新候选保持 RTN 三项质量锚点的量化语义。最初 protected_n、uniform_n、energy_n 保留完整线性 A 修复，使用跨阶段固定的 N128×完整 K 权重支持；三个位置对照的各算子 N 块数量完全相同。']
    if a.archive_label!='development':lines+=['',
        '后续候选用完整 FFN（包括 ReLU² 与当前下投影输入）的开关干预调整 W1 分配。带 `_k` 的候选使用 K128×完整 N 支持、激活 RMS 加权的残差尺度能量提出 K 排名；这只是候选排序代理，不等于实际 K 块修复收益。其他 N 候选保留 N128×完整 K。精确预算和存储分别查看各 candidate/compute 文件。']
    if any('balanced' in name for name in manifest['methods']):lines+=['',
        'balanced_k 保留 87.5% 线性 A 块、约 62.46% 静态 W1；QK/PV 分别按校准干预收益分配固定 head 配额（校准最短 key 长度下平均 75%，实际长度下单独计数）。它不是完整 A/attention 修复方案。']
    for benchmark,methods in summaries.items():
        lines+=['',f'## {benchmark}','','| 配置 | LPIPS ↓ | SSIM ↑ | 相对 L2 ↓ | PSNR ↑ |','|---|---:|---:|---:|---:|']
        for name,v in methods.items():
            psnr=f'{v["psnr_db"]:.3f}' if v['psnr_db'] is not None else '+∞'
            lines.append(f'| {name} | {v["lpips"]:.6f} | {v["ssim_mean"]:.6f} | {v["relative_l2"]:.6f} | {psnr} |')
    lines+=['','## 解释边界','',
        'b2.5_old_quota_rtn 保持旧配额和支持，改回 RTN，检验表示改变的影响。新 protected_n 同时调整支持结构和分配，不能把它的全部收益归因于 head finding。',
        'protected/ffn_aware 2.5× 的 attention 全修复随当前输入长度展开；balanced 使用固定 head 配额。N961–2048 范围内的尾 tile、selector 跳变和预算分别验证，实际运行仍检查当前形状。更长输入不在本次准备范围内。',
        '1.25/1.5/2.0 的独立资产和 head 配额已生成，本阶段只运行 2.5× 的归因实验。未据开发结果修改原 72 场景或运行新的最终测试。']
    report=root/'report';report.mkdir(exist_ok=True);write(report/'quality.json',result);(report/'REPORT.md').write_text('\n'.join(lines)+'\n')
    archive=Path(__file__).resolve().parents[2]/'results/repair_redesign_20260917';archive.mkdir(exist_ok=True)
    (archive/(a.archive_label.upper()+'.md')).write_text('\n'.join(lines)+'\n')
    for name,path in [('development_quality',report/'quality.json'),('development_manifest',root/'manifest.json'),('cohort',a.study/'cohort.json'),('candidates',a.study/'candidates/candidates.json')]:
        filename=name if a.archive_label=='development' else a.archive_label+'_'+name
        (archive/(filename+'.json.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    print(json.dumps(dict(complete=True,report=str(archive/(a.archive_label.upper()+'.md')))))


if __name__=='__main__':main()

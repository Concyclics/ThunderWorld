"""Scene-clustered, category-stratified paired inference on a frozen subset."""
import argparse,collections,json,statistics
from pathlib import Path
import numpy as np
from baselines import sha,write

METRICS=('lpips','ssim_mean','relative_l2','psnr_db','rmse')


def average(values):
    return float(np.mean(values)) if all(v is not None for v in values) else None


def stratified_interval(groups,*,replicates=10000,seed=170917,family=14):
    """Each input array has one paired delta per SCENE (seeds already averaged)."""
    if not groups or any(not len(x) or not np.isfinite(x).all() for x in groups.values()):raise ValueError('Finite nonempty scene strata required')
    rng=np.random.default_rng(seed);samples=[]
    for key in sorted(groups):
        x=np.asarray(groups[key],float);samples.append(x[rng.integers(0,len(x),(replicates,len(x)))].mean(1))
    boot=np.mean(samples,axis=0);alpha=.05/family
    return dict(mean_delta=float(np.mean([np.mean(x) for x in groups.values()])),ci95=np.quantile(boot,[.025,.975]).tolist(),
        familywise_ci=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),n_scenes=sum(map(len,groups.values())),n_strata=len(groups))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--generation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    study=json.loads(a.manifest.read_text());progress=json.loads((a.metrics/'progress.json').read_text())
    if not progress['complete'] or progress['identity']['manifest_sha256']!=sha(a.manifest):raise ValueError('Complete fixed subset required')
    scenes=[];files={};costs=[];seed_metrics=[]
    for entry in study['cases']:
        case_id=entry['case']['id']
        for name in study['methods']:
            receipt_path=a.generation/name/case_id/'result.json';receipt=json.loads(receipt_path.read_text())
            if not receipt['complete']:raise ValueError('Incomplete outputs')
            rowcost=dict(name=name,case_id=case_id,benchmark=entry['benchmark'],
                max_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipt['rows']),
                max_reserved_bytes=max(r['peak_reserved_bytes'] for r in receipt['rows']),
                actual_mma_ratios=[r['actual_mma_ratio'] for r in receipt['rows']],receipt_sha256=sha(receipt_path))
            if study['methods'][name]['config'].get('compute_budget'):
                cap=study['methods'][name]['config']['compute_budget']
                if any(x is None or x>cap for x in rowcost['actual_mma_ratios']):raise ValueError('Missing or violated per-scene budget')
            costs.append(rowcost)
            if name=='bf16':continue
            path=a.metrics/name/(case_id+'.json');value=json.loads(path.read_text());files[str(path)]=sha(path)
            if not value['complete'] or len(value['rows'])!=len(entry['seeds']) or {r['seed'] for r in value['rows']}!=set(entry['seeds']):raise ValueError('Incomplete or duplicate scene seeds')
            if value['provenance']['identity']!=progress['identity']:raise ValueError('Mixed metric implementations')
            if value['provenance']['candidate_sha256']!=sha(receipt_path):raise ValueError('Changed generation since metric calculation')
            if value['provenance']['reference_sha256']!=sha(a.generation/'bf16'/case_id/'result.json'):raise ValueError('Changed BF16 reference since metric calculation')
            seed_metrics.append(dict(name=name,case_id=case_id,benchmark=entry['benchmark'],category=entry['category'],
                rows=value['rows'],provenance=value['provenance']))
            scenes.append(dict(name=name,case_id=case_id,benchmark=entry['benchmark'],category=entry['category'],
                metrics={m:average([r['metrics']['video'][m] for r in value['rows']]) for m in METRICS},
                exact_psnr_clips=sum(r['metrics']['video']['psnr_exact'] for r in value['rows'])))
    methods=[n for n in study['methods'] if n!='bf16'];summaries={};contrasts=[]
    for benchmark in ('rbench','paibench'):
        rows=[r for r in scenes if r['benchmark']==benchmark];summaries[benchmark]={}
        for name in methods:
            own=[r for r in rows if r['name']==name];categories=sorted({r['category'] for r in own})
            cat={c:{m:average([r['metrics'][m] for r in own if r['category']==c]) for m in METRICS} for c in categories}
            summaries[benchmark][name]=dict(category_macro={m:average([v[m] for v in cat.values()]) for m in METRICS},
                scene_micro={m:average([r['metrics'][m] for r in own]) for m in METRICS},categories=cat,n_scenes=len(own),n_videos=2*len(own),
                exact_psnr_clips=sum(r['exact_psnr_clips'] for r in own))
            if name=='fp8':continue
            reference={r['case_id']:r for r in rows if r['name']=='fp8'};detail={}
            for metric in METRICS:
                groups=collections.defaultdict(list);deltas=[]
                for row in own:
                    x=row['metrics'][metric];y=reference[row['case_id']]['metrics'][metric]
                    delta=None if x is None or y is None else x-y
                    groups[row['category']].append(delta);deltas.append((row['case_id'],delta))
                if any(d is None for _,d in deltas):
                    detail[metric]=dict(status='infinite_exact_match_psnr',ci95=None);continue
                detail[metric]=stratified_interval(groups,replicates=study['statistics']['replicates'],seed=study['statistics']['bootstrap_seed'],family=study['statistics']['family_size'])
                lower=metric not in ('ssim_mean','psnr_db');worst=sorted(deltas,key=lambda t:t[1],reverse=lower)[:5]
                detail[metric].update(worst_scenes=[dict(case_id=c,delta=d) for c,d in worst],
                    worse_scene_count=sum((d>0 if lower else d<0) for _,d in deltas),higher_is_better=not lower)
            passed=detail['lpips']['familywise_ci'][1]<0 and detail['ssim_mean']['mean_delta']>=0 and detail['relative_l2']['mean_delta']<=0
            contrasts.append(dict(benchmark=benchmark,candidate=name,control='fp8',metrics=detail,exceeds_fp8=passed))
    summaries['combined_equal_datasets']={name:{m:average([summaries[b][name]['category_macro'][m] for b in ('rbench','paibench')]) for m in METRICS} for name in methods}
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(complete=True,manifest_sha256=sha(a.manifest),statistics=study['statistics'],scenes=scenes,seed_metrics=seed_metrics,summaries=summaries,contrasts=contrasts,metric_files=files,
        scope=study['scope'],performance_eligible=False,selection_eligible=False)
    write(a.output/'quality.json',result);write(a.output/'costs.json',dict(complete=True,rows=costs,scope='Per-scene request bounds and allocator peaks; shared timings excluded. Fixed integer quotas retained as prompt length changes.'))
    lines=['# RBench / PAI-Bench 多场景 FP4 一致性测试','',
        '72 个冻结场景，每场景 2 个独立预留种子；九种固定配置，共 1,296 段视频。RBench 9 类各 4 场景；PAI-Bench 7 个文件前缀领域共 36 场景（human 6，其余各 5）。未用这些场景校准或选参。',
        '', 'Edge，192×320，61 帧，35 steps，CFG 6。使用原始 prompt / prompt_en 与统一 negative prompt，均含新鲜 AR prefill。此设置用于相对 BF16 的一致性，不是官方榜单分数、物理正确率或机器人任务成功率。',
        '', 'PSNR = −10 log10(整段 RGB 视频 MSE)，固定 data_range=1；先算每视频，再平均两个种子到场景，再等权平均类别。完全相同时 PSNR 为正无穷，JSON 用 null + exact 标记，不任意截断。']
    for benchmark in ('rbench','paibench'):
        lines+=['',f'## {benchmark}（类别等权）','', '| 配置 | LPIPS ↓ | SSIM ↑ | 相对 L2 ↓ | PSNR dB ↑ |','|---|---:|---:|---:|---:|']
        for name in methods:
            v=summaries[benchmark][name]['category_macro'];psnr=f'{v["psnr_db"]:.3f}' if v['psnr_db'] is not None else '+∞'
            lines.append(f'| {name} | {v["lpips"]:.6f} | {v["ssim_mean"]:.6f} | {v["relative_l2"]:.6f} | {psnr} |')
        lines+=['','| 相对 FP8 | LPIPS 差值 | 场景配对 95% CI | 14 项校正 CI | 通过原质量门槛 |','|---|---:|---|---|---|']
        for row in contrasts:
            if row['benchmark']!=benchmark:continue
            d=row['metrics']['lpips'];lines.append(f'| {row["candidate"]} | {d["mean_delta"]:.6f} | {d["ci95"]} | {d["familywise_ci"]} | {row["exceeds_fp8"]} |')
    lines+=['','## 实际请求预算与显存','', '| 数据集 | 配置 | MMA/C0 最小–最大 | 最大 allocated GiB |', '|---|---|---:|---:|']
    for benchmark in ('rbench','paibench'):
        for name in study['methods']:
            own=[r for r in costs if r['benchmark']==benchmark and r['name']==name]
            ratios=[x for r in own for x in r['actual_mma_ratios'] if x is not None]
            span=f'{min(ratios):.6f}–{max(ratios):.6f}' if ratios else '无预算计划账本'
            lines.append(f'| {benchmark} | {name} | {span} | {max(r["max_allocated_bytes"] for r in own)/2**30:.3f} |')
    lines+=['','## 统计与边界','',
        'Bootstrap 在每一类别内重采样场景，同一场景两个种子始终作为一个配对簇。主检验共两数据集 × 七个 FP4 对 FP8 比较，使用 14 项 Bonferroni 区间；PSNR 是补充指标，没有事后替换 LPIPS 主指标。逐类别均值、场景均值、最差五个场景和退化场景数在 quality.json。',
        '固定整数修复配额直接应用于新提示词长度；每个场景重新发现实际形状并验证预算，因此实际 MMA 比例可能低于标称上限。没有为新测试图像重新校准权重或重排离线支持。',
        '本轮九种主配置不包含 ARCQuant、SharQ、AWQ 和旧优化版 2.5× 的跨场景运行；不能将上一轮单场景排名外推到这两个数据集。低预算配置此前的验证失败仍保留，本轮新场景结果不改写旧验收。',
        '所有运行来自共享 PRO 6000，未测独占速度。本报告是所选子集、固定生成设置下的结论，不是全量基准或跨模型泛化证明。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

"""Post-test report: preserve frozen hypotheses and validation eligibility."""
import argparse,json
from pathlib import Path
from baselines import sha,write


def qualify(selection,gates):
    contrasts={(r['candidate'],r['control']):r for r in gates['contrasts']}
    rows=[]
    for decision in selection['decisions']:
        name=decision['selected'];control=decision['control']
        mechanism=contrasts[name,control];fp8=contrasts[name,'fp8']
        rows.append(dict(**decision,mechanism_test=mechanism,fp8_test=fp8,
            mechanism_accepted=decision['validation_guardrails'] and mechanism['paired_gate_passed'],
            exceeds_fp8=decision['validation_guardrails'] and fp8['paired_gate_passed']))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.workspace
    study=root/'contribution_complete_study_20260917';selection_path=study/'frozen/selection.json'
    paths=dict(selection=selection_path,quality=study/'final_quality.json',gates=study/'final_gates.json')
    values={k:json.loads(path.read_text()) for k,path in paths.items()}
    if any(v.get('complete') is False for v in values.values()):raise ValueError('Complete final results required')
    decisions=qualify(values['selection'],values['gates']);quality=values['quality']
    stores={}
    for kind,file in (('control','contribution_budget_storage_20260917.json'),('marginal','contribution_marginal_budget_storage_20260917.json'),
                      ('head_coordinate','contribution_head_coordinate_storage_20260917.json')):
        storage=json.loads((root/file).read_text())
        for row in storage['rows']:
            if kind!='head_coordinate' and not row['name'].endswith('_grow_mixed'):continue
            stores[f'b{row["budget"]:g}_{kind}']=row
    costs=[]
    for name,entry in stores.items():
        if name not in quality['means']:continue
        metric=quality['means'][name];reference=quality['means'][f'b{entry["budget"]:g}_control']
        guard=metric['ssim_mean']>=reference['ssim_mean'] and metric['relative_l2']<=reference['relative_l2']
        decision=next((d for d in decisions if d['selected']==name),None)
        eligible=guard and (decision is None or decision['validation_guardrails'])
        costs.append(dict(name=name,budget=entry['budget'],mma_ratio=entry['ratio'],metrics=metric,
            asset_payload_bytes=entry['bytes']['payload_bytes'],asset_file_bytes=entry['bytes']['file_bytes'],
            full_w1_control_bytes=entry['bytes']['same_support_full_w1_payload_bytes'],w1_union_density=entry['union_density'],
            compute_ledger={k:v for k,v in entry['compute_ledger'].items() if k!='rows'},mean_guardrails=guard,validation_eligible=eligible,
            objective=[metric['lpips'],entry['ratio'],entry['bytes']['payload_bytes']]))
    eligible=[r for r in costs if r['validation_eligible']]
    for row in costs:
        row['observed_nondominated']=row['validation_eligible'] and not any(
            all(x<=y for x,y in zip(other['objective'],row['objective'])) and any(x<y for x,y in zip(other['objective'],row['objective'])) for other in eligible)
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'acceptance.json',dict(complete=True,sources={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        decisions=decisions,anchor_gates=[r for r in values['gates']['contrasts'] if r['candidate'].startswith('fp4_')],
        performance_eligible=False,scope='Frozen family of ten comparisons. Failed validation guardrails cannot be rescued by final testing. Legacy and external differences are descriptive.'))
    write(a.output/'frontier.json',dict(complete=True,rows=costs,quality_sha256=sha(paths['quality']),
        scope='Observed final LPIPS/MMA/asset frontier among frozen policies and their controls, with validation and SSIM/L2 mean guards. Not latency optimality or a significance claim.'))
    lines=['# 冻结的 Edge I2V 最终测试','',
        '同一 development 场景的 16 个新种子（181001–181016），35 steps、CFG 6、相同模型和生成算子范围。配置由 171001–171008 验证集冻结。这里是对 BF16 输出的一致性，不是跨场景泛化或任务成功率。',
        '全部运行来自共享 PRO 6000；外部适配的执行模式和数值定义见 baseline 清单。不得用本批 wall time 排速度，也不得把离线解码 AWQ 的内存当作 W4 部署内存。','',
        '| 方法 | LPIPS ↓ | SSIM ↑ | 视频相对 L2 ↓ |','|---|---:|---:|---:|']
    for name,m in quality['means'].items():lines.append(f'| {name} | {m["lpips"]:.6f} | {m["ssim_mean"]:.6f} | {m["relative_l2"]:.6f} |')
    lines+=['','## 四档预注册比较','',
        '区间为 LPIPS 配对差值，负值更好；同时报告普通 95% 区间和十项比较 Bonferroni 区间。成立还要求 SSIM/L2 均值不退化，并通过先前验证约束。','',
        '| 预算／候选 | 对照 | 差值 | 95% CI | 多重校正 CI | 全部约束通过 |','|---|---|---:|---|---|---|']
    for row in decisions:
        for key,accepted in (('mechanism_test',row['mechanism_accepted']),('fp8_test',row['exceeds_fp8'])):
            test=row[key];lp=test['metrics']['lpips']
            lines.append(f'| {row["budget"]} / {row["selected"]} | {test["control"]} | {lp["mean_delta"]:.6f} | {lp["ci95"]} | {lp["familywise_ci"]} | {accepted} |')
    lines+=['','最差种子退化、逐指标配对统计、验证约束与原始门槛分别保存在 acceptance.json 和 final_gates.json.gz 中。旧优化版 legacy_2.5 和外部方法未加入这十项确认性假设，均值差异仅作描述，不据此追加显著性主张。',
        '', '## 计算与实际资产','',
        '| 方法 | 请求 MMA / C0 | W1 支持并集 | 资产张量 MiB | 同支持完整 W1 MiB | 观测非支配 |','|---|---:|---:|---:|---:|---|']
    for row in costs:lines.append(f'| {row["name"]} | {row["mma_ratio"]:.8f} | {row["w1_union_density"]:.4f} | {row["asset_payload_bytes"]/2**20:.2f} | {row["full_w1_control_bytes"]/2**20:.2f} | {row["observed_nondominated"]} |')
    measured=json.loads((root/'contribution_runtime_costs_20260917.json').read_text())
    lines+=['', '### 本批请求实测内存','',
        'Allocated / reserved 是各配置 16 个完整请求的最大值。驻留数据包括模型与量化模块；不是只算 W1 的推算值。BF16 的准备权重为 0 表示未额外构建量化资产，其原生权重已计入常驻模型。共享负载不用于推导延迟优势。',
        '', '| 方法 | 常驻模型 GiB | 量化模块准备权重 GiB | 峰值 allocated GiB | 峰值 reserved GiB |', '|---|---:|---:|---:|---:|']
    for row in measured['rows']:
        if 'memory' not in row:continue
        m=row['memory'];lines.append(f'| {row["name"]} | {m["resident_model_state_bytes"]/2**30:.3f} | {m["prepared_weight_bytes"]/2**30:.3f} | {m["peak_allocated_bytes"]/2**30:.3f} | {m["peak_reserved_bytes"]/2**30:.3f} |')
    lines+=['', '计算账本包含 padding、交叉项上界、非 MMA 运算和 packing 字节。资产数字只覆盖生成投影；整模型常驻状态和 Graph 峰值在运行 receipt 中另列。选择性 residual packing 的写入量下降，但完整 scratch 分配尚未消除。',
        '', '## 贡献判定与未完成部分','',
        '- W0/outlier 联合优化：全层局部结果和完整开发轨迹不支持普遍优势。FourOverSix 局部下降未转化为视频改善；必须保留这一负结果。',
        '- 收益分配：以本页冻结的配对比较为准，不能用开发集、单层 NMSE 或 attention-only BF16 投影结果代替联合 FP4 验收。',
        '- 实际工作与存储：同支持压缩、选择性 packing 已验证数值一致；未完成新策略的独占全请求融合/overlap 速度归因。',
        '- AR：相同缓存下三／四项 FP4 预填充的开发输出误差仍明显，不能称 AR 的额外修复是免费或无损。',
        '- 官方 ARCQuant/SharQ 已完成适配参考检查和完整生成；ARC 固定 50% 通道，未完成原方法全模型比例搜索。AWQ 是官方 NVFP4 AWQ-lite W4A16 解码质量参考。',
        '- ViDiT-Q 仅有官方版本锁定和层／时间启发对照；SVDQuant 使用 ModelOpt lite 算子参考，不是 Nunchaku 原生 Cosmos 性能移植。',
        '- 这些结果不能证明全局 Pareto optimal。独占 PRO 6000 / 5090 延迟复测、policy/Nano 独立校准和扩展验收尚未完成。']
    (a.output/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

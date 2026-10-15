"""Archive inspectable evidence and regenerate a deliberately bounded study report."""
import argparse,gzip,json,shutil,statistics
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.workspace
    files={
        'cohort':'contribution_cohort_frozen_20260917.json',
        'mechanism':'contribution_mechanism_all_20260917/mechanism.json',
        'mechanism_summary':'contribution_mechanism_all_20260917/report.json',
        'four_six_quality':'contribution_foursix_20260917/development/quality.json',
        'head_mechanism':'contribution_attention_marginal_20260917.json',
        'attention_development':'contribution_attention_generation_quality_20260917.json',
        'attention_validation':'contribution_attention_validation_20260917/quality.json',
        'attention_gates':'contribution_attention_validation_20260917/gates.json',
        'storage_probe':'contribution_storage_probe_20260917.json',
        'packing_probe':'contribution_packing_probe_20260917.json',
        'runtime_costs':'contribution_runtime_costs_20260917.json',
        'budget_storage':'contribution_budget_storage_20260917.json',
        'marginal_candidates':'contribution_marginal_budget_candidates_20260917/candidates.json',
        'joint_development':'contribution_joint_development_20260917/quality.json',
        'arc_probe':'contribution_arcquant_probe_20260917.json',
        'arc_build':'contribution_arcquant_adapted_20260917/build.json',
        'arc_memcheck':'contribution_arc_memcheck_20260917.json',
        'arc_racecheck':'contribution_arc_racecheck_20260917.json',
        'sharq_probe':'contribution_sharq_adapter_probe_20260917.json',
        'sharq_quality':'contribution_sharq_development_20260917/quality.json',
        'arc_quality':'contribution_arcquant_development_20260917/quality.json',
        'awq_quality':'contribution_awq_development_20260917/quality.json',
        'awq_assets':'contribution_awq_20260917/assets/manifest.json',
        'sage_probe':'contribution_sageattention_v2_20260917/probe.json',
        'sage_build':'contribution_sageattention_v2_20260917/build.json',
        'lowrank':'contribution_lowrank_20260917.json',
        'student_report':'contribution_student_report_20260917.json',
        'head_coordinate':'contribution_head_coordinate_20260917/interventions.json',
        'head_coordinate_development':'contribution_head_coordinate_development_20260917/quality.json',
        'head_coordinate_storage':'contribution_head_coordinate_storage_20260917.json',
        'validation_quality':'contribution_complete_study_20260917/validation_quality.json',
        'frozen_selection':'contribution_complete_study_20260917/frozen/selection.json',
        'frozen_manifest':'contribution_complete_study_20260917/frozen/manifest.json',
        'validation_manifest':'contribution_joint_validation_manifest_20260917.json',
        'marginal_storage':'contribution_marginal_budget_storage_20260917.json',
        'output_gain_outlier':'contribution_output_gain_outlier_all_20260917.json',
        'final_quality':'contribution_complete_study_20260917/final_quality.json',
        'final_gates':'contribution_complete_study_20260917/final_gates.json',
    }
    evidence={};data={}
    for label,relative in files.items():
        src=root/relative
        if not src.exists():
            evidence[label]=dict(status='pending',source=str(src));continue
        value=json.loads(src.read_text())
        if value.get('complete') is False:
            evidence[label]=dict(status='incomplete',source=str(src),sha256=sha(src));continue
        data[label]=value;name=label+'.json.gz'
        with (a.output/name).open('wb') as handle:
            with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as archive:archive.write(src.read_bytes())
        evidence[label]=dict(status='recorded',source=str(src),source_sha256=sha(src),archive=name,archive_sha256=sha(a.output/name))
    shutil.copyfile(root/'contribution_baselines_20260917/sources.lock.json',a.output/'sources.lock.json')
    shutil.copyfile(root/'contribution_arcquant_adapted_20260917/cosmos.patch',a.output/'arcquant-cosmos.patch')
    repo=Path(__file__).resolve().parents[2]
    historical={}
    for label,relative in (('full_overlap','results/pro6000_full_overlap_20260916/summary.json'),
                           ('producer_fusion','results/pro6000_producer_fusion_20260916/summary.json')):
        path=repo/relative;historical[label]=dict(path=relative,sha256=sha(path),status='historical_matched_execution_control',
            scope='Prior strategy and requests; preserved evidence, not a new run or a speed conclusion for new head policies.')
    write(a.output/'historical_execution_controls.json',historical)
    # Portable metadata only: large model, weight-bank and video files stay external.
    samples=[]
    for folder in ('contribution_attention_validation_20260917','contribution_joint_development_20260917',
                   'contribution_sharq_development_20260917','contribution_arcquant_development_20260917','contribution_awq_development_20260917',
                   'contribution_joint_validation_20260917','contribution_head_coordinate_development_20260917','contribution_complete_study_20260917'):
        directory=root/folder
        if not directory.exists():continue
        receipts=[directory/'result.json'] if (directory/'result.json').exists() else sorted(directory.rglob('result.json'))
        for receipt in receipts:
            value=json.loads(receipt.read_text())
            samples.append(dict(receipt=str(receipt),receipt_sha256=sha(receipt),complete=value['complete'],
                configuration_sha256=value.get('configuration_sha256'),admission=value.get('admission'),
                outputs=[dict(seed=row['seed'],file=str(receipt.parent/row['outputs']),
                    sha256=row.get('outputs_sha256'),signal_sha256=row.get('sha256')) for row in value['rows']]))
    write(a.output/'samples.json',dict(samples=samples,scope='Output index; no model or video payload is vendored'))
    write(a.output/'evidence.json',dict(evidence=evidence,implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        performance_eligible=False,all_plan_acceptance_complete=False))
    lines=['# Edge I2V 贡献点前置实验（2026-09-17）','',
        '本页记录局部机制与实现边界；冻结最终测试、每档判定和质量—计算—存储前沿见 FINAL_REPORT.md（若尚未生成，则最终统计未完成）。所有 GPU 测量来自 PRO 6000 共享模式，不是 5090 性能，也不进入速度排行榜。',
        '', '## 1. W0 / 残差表示：保留负结果','',
        '168 个生成线性层，每层 16 个完整 M 教师面板。下面是平均模块目标（面板均值 + 0.25×最差面板），不是视频质量。outlier 实验预留全量 W1，不能当作 1.25×–2.5× 结果。','',
        '| 量化策略 | 三项 | 四项 |','|---|---:|---:|']
    if 'mechanism_summary' in data:
        values={(r['method'],r['terms']):r['mean_module_objective'] for r in data['mechanism_summary']['rows']}
        for method in ('rtn','four_six','magnitude','quant_error','joint_diagonal'):
            if (method,'three') in values:lines.append(f'| {method} | {values[method,"three"]:.8g} | {values[method,"four"]:.8g} |')
    lines+=['', 'FourOverSix 风格尺度选择降低局部误差，但两个开发种子的联合生成平均 LPIPS 从 0.043492 升至 0.050868，SSIM/L2 也退化，因此不晋级该权重资产。',
        '联合对角代理的 outlier 拆分只在四项局部结果中改善，三项反而变差；目前证据不支持“先拆 outlier，再普遍省略交叉项”。该代理不是精确输出收益搜索。',
        '', '## 2. 修复收益分配：均值改善，显著性仍未通过','',
        '同一 attention MMA、当前输入 selector、所有 base blocks 保留。7101 配额曲线提出整数配额，7102 重算完整 attention 组合；局部平均 NMSE 相对均匀配额下降约 60.6%。AR 注意力质量/能量的整 head 排序未得到相同收益。',
        '下面是冻结的 171001–171008 验证结果。两种 attention 策略均使用 **BF16 投影**，只隔离 attention 的贡献；不能作为联合 W4A4 已超过 FP8 的结论。','',
        '| 方法 | LPIPS ↓ | SSIM ↑ | 视频相对 L2 ↓ |','|---|---:|---:|---:|']
    if 'attention_validation' in data:
        for name,m in data['attention_validation']['means'].items():lines.append(f'| {name} | {m["lpips"]:.6f} | {m["ssim_mean"]:.6f} | {m["relative_l2"]:.6f} |')
    if 'attention_gates' in data:
        for row in data['attention_gates']['contrasts']:
            lp=row['metrics']['lpips']
            lines+=['',f'{row["candidate"]} − {row["control"]}：LPIPS 配对差值 {lp["mean_delta"]:.6f}，95% CI {lp["ci95"]}；四比较 Bonferroni 区间 {lp["familywise_ci"]}。门槛通过：**{row["paired_gate_passed"]}**。']
    lines+=['', '配额候选已重新执行完整组合并加入现有请求预算搜索；最终 181001–181016 不提前用于调参。',
        '', '## 3. 实际工作与存储','',
        '50% W1 支持并集的独立探针：残差 code/scale 为 18,432 字节，对照 36,864 字节；M129/M416/M960 的 eager、阶段切换、Graph 当前输入与零输入均逐位一致。',
        '以下为原搜索的混合候选部署资产，尚未通过本轮生成质量门槛。只统计生成投影资产，不是整模型峰值显存。','',
        '| 请求预算 | 实际 MMA 比例 | W1 并集密度 | 资产张量 MiB | 相同支持但保留完整 W1 的 MiB |','|---|---:|---:|---:|---:|']
    if 'budget_storage' in data:
        for row in data['budget_storage']['rows']:
            if 'mixed' not in row['name']:continue
            b=row['bytes'];lines.append(f'| {row["budget"]} | {row["ratio"]:.8f} | {row["union_density"]:.4f} | {b["payload_bytes"]/2**20:.2f} | {b["same_support_full_w1_payload_bytes"]/2**20:.2f} |')
    lines+=['', '选择性 packing 减少 residual 写入，但仍分配完整 residual scratch；残差摘要、选择和所选块的 base 重编码仍有非 MMA 成本。不能把 MMA 减少称为同等延迟减少。',
        '', '## 4. 最近邻 baseline 的实现与边界','',
        '| Baseline | 本轮执行 | 资格边界 |','|---|---|---|',
        '| ARCQuant | 官方内核 + K9216 补丁，完整开发生成 | 固定 50% 通道；未复现模型级比例搜索；eager-only，性能不可比 |',
        '| SharQ | 官方原生内核，完整开发生成 | extra_fusion=False；官方模拟另列，不冒充 native |',
        '| NVFP4-AWQ | 官方 ModelOpt AWQ-lite W4A16，完整开发生成 | 离线解码 BF16 质量参考；不声称 W4 原生存储/速度 |',
        '| Four Over Six | 尺度选择机制复现 + 同样残差 | 非完整官方 Cosmos 内核移植 |',
        '| SageAttention3 | 官方原生内核，48 个真实 attention 面板 | 显式 GQA 复制、K 复制；不是完整生成对照 |',
        '| SVDQuant/Nunchaku | 官方源代码锁定；ModelOpt lite-SVDQuant 算子对照 | 非原始 Nunchaku 性能；BF16 补偿独立列账 |',
        '| ViDiT-Q | 官方源代码锁定 | 层/时间敏感性机制启发，完整方法适配尚未完成 |',
        '', '外部方法没有伪装成同预算。两个开发种子的平均 LPIPS 如下，仅表示这些具体适配，不能推广为论文方法的普遍排名。','',
        '| 适配 | 开发平均 LPIPS |','|---|---:|']
    for label in ('arc_quality','sharq_quality','awq_quality'):
        if label in data:
            for name,m in data[label]['means'].items():lines.append(f'| {name} | {m["lpips"]:.6f} |')
    if 'sage_probe' in data:
        means={name:statistics.mean(r['nmse'][name] for r in data['sage_probe']['rows']) for name in data['sage_probe']['rows'][0]['nmse']}
        lines+=['',f'SageAttention 面板平均 NMSE：{means}。未配平修复计算，不能据此宣称同成本优势。']
    if 'lowrank' in data:
        means={rank:statistics.mean(r['mean_nmse'] for r in data['lowrank']['rows'] if r['rank']==rank) for rank in (0,32,64)}
        lines+=['',f'ModelOpt lite-SVDQuant 六种算子形状的平均 NMSE：{means}。rank32/64 的高精度分支有额外成本，原始 JSON 已列账。']
    lines+=['', '## 验收状态与复测','',
        'CPU 验证命令、各适配构建与执行命令见 `tools/contribution_study/README.md`。原始 JSON 以 gzip 保存，`evidence.json` 记录源文件和归档哈希；`samples.json` 索引完整输出及校验值。模型、视频和未取得重分发许可的上游源码不随报告复制。',
        '', '尚未完成：四档联合方案的完整验证/冻结最终测试、候选自身轨迹追加校准、严格配平的融合/overlap 全请求消融、独占 PRO 6000/5090 速度复测。AR 三/四项归因结果以相应运行 receipt 为准，未完成时不主张 AR 可免费增加预算。',
        '当前不扩展到 policy/Nano，不宣布任何新配置满足“质量超过 FP8”或实际延迟 Pareto optimal。',
        '', '## 已有执行消融实现','',
        '历史 PRO 6000 的 `full_overlap`（串行、QKV、CFG）和 `producer_fusion`（原路径与 producer 融合）作为执行 baseline 保留；索引及 SHA 在 `historical_execution_controls.json`。这些对照对应旧策略，不能把其收益直接外推到新 head 配额，也不与本轮共享质量数据混成正式速度排行榜。']
    if 'output_gain_outlier' in data:
        rows=data['output_gain_outlier']['rows']
        lines+=['', '## 补充：真实输出响应的 outlier 干预','',
            f'{len(rows)} 个算子，每个只允许一个预留 N128/K128 块；全部 16 个完整 M 教师面板。逐组试验后重新评价组合 mask，最终输出 BF16 舍入；独立 FP32 解码参考，不是原生内核逐位 oracle。',
            '', '| 选择依据 | 三项相对 RTN | 四项相对 RTN | 三项改善层数 |', '|---|---:|---:|---:|']
        for method in ('magnitude','quant_error','output_three','output_four'):
            deltas={term:statistics.mean(r['objective'][method][term] for r in rows)/statistics.mean(r['objective']['rtn'][term] for r in rows)-1 for term in ('three','four')}
            wins=sum(r['objective'][method]['three']<r['objective']['rtn']['three'] for r in rows)
            lines.append(f'| {method} | {deltas["three"]:+.3%} | {deltas["four"]:+.3%} | {wins}/{len(rows)} |')
        lines+=['', '这里使用全量 W1 质量锚点且在同一教师校准面板上选择与测量；小幅局部收益不代表独立视频质量改善，也不是四档预算中的已部署方案。组合变差的 mask 保留为负结果，不晋级资产。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if 'final_quality' in data and 'final_gates' in data:
        import subprocess,sys
        subprocess.run([sys.executable,str(Path(__file__).with_name('final_report.py')),
            '--workspace',str(root),'--output',str(a.output)],check=True)


if __name__=='__main__':main()

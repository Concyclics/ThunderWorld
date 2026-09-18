"""Describe the bounded four-candidate study without final-test or speed claims."""
import argparse
import json
from pathlib import Path

from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    paths = [a.study / run / 'report/quality.json' for run in ('two_term', 'ffn_development')]
    attribution, joint = [json.loads(path.read_text()) for path in paths]
    if not all(result['complete'] for result in (attribution, joint)):
        raise ValueError('Completed attribution and all four budget candidates required')
    benchmarks = ('rbench', 'paibench')
    controls = {1.5: 'b1.5_protected_n', 2.: 'b2_protected_n', 2.5: 'b2.5_ffn_aware'}
    lines = ['# 激活重要性、W0/W1 联合调优：本轮结论', '',
             'Edge I2V，192×320、61 帧、35 步、CFG 6，Graph/debug off。16 个校准场景；另 16 个开发场景（RBench 9、PAI-Bench 7）各两种子。与旧 72 场景去重；32 个验证场景和 72 个新最终场景未运行。每类别开发只有一个场景，以下仅为探索性配对均值，不作显著性、泛化或延迟 Pareto 宣称。', '',
             '全部质量指标衡量相对同输入 BF16 的一致性，不能改称官方 benchmark 分数或物理任务成功率。PSNR 使用 data_range=1 的整段视频 RGB MSE，再按视频、场景求均值。', '',
             '## 只修一侧的效果', '',
             '为隔离误差来源，以下四个 FP4 线性方案使用相同的 BF16 attention；其余生成条件相同。这是明确的归因对照，预算候选全部使用 FP4 attention。', '',
             '| 线性修复 | RBench LPIPS ↓ | PAI-Bench LPIPS ↓ |', '|---|---:|---:|']
    for name, label in [('linear_base', '无修复'), ('linear_w_only', '只修权重 W1'), ('linear_a_only', '只修激活 A1'), ('linear_three', '同时修 A1/W1，无交叉项'), ('fp8', '原生 FP8 投影')]:
        values = [attribution['summaries'][b][name]['lpips'] for b in benchmarks]
        lines.append(f'| {label} | {values[0]:.6f} | {values[1]:.6f} |')
    lines += ['', 'A-only 在 PAI 的改善更大；RBench 上两者的 LPIPS 接近，但 A-only 的 L2/PSNR 较差，不能概括为激活在所有指标上都更重要。单侧均未达到 FP8；两侧组合明显更好。该对照支持联合分配修复，不能用孤立项的收益直接相加预测组合。', '',
              '## 固定预算和存储的四组比较', '',
              '每档固定逐算子 A 配额、W1 tile 数、attention head 配额和阶段/CFG 调用次数，只改变 W0 表示、W1 位置及激活选择敏感度。W1 是静态并集压缩存储；在线 A 始终从当前量化残差选择。', '',
              '| 预算 | 配置 | RBench LPIPS ↓ | PAI-Bench LPIPS ↓ | 同预算均值门槛 |', '|---|---|---:|---:|---|']
    decisions = {}
    for cap, control in controls.items():
        candidates = [control] + [f'b{cap:g}_{suffix}' for suffix in ('importance_rtn', 'joint', 'ffn_guarded')]
        eligible = []
        details = []
        for name in candidates:
            values = [joint['summaries'][b][name]['lpips'] for b in benchmarks]
            passed = name != control and all(
                joint['summaries'][b][name]['lpips'] < joint['summaries'][b][control]['lpips']
                and joint['summaries'][b][name]['ssim_mean'] >= joint['summaries'][b][control]['ssim_mean']
                and joint['summaries'][b][name]['relative_l2'] <= joint['summaries'][b][control]['relative_l2']
                for b in benchmarks)
            if passed:
                eligible.append(name)
            status = '对照' if name == control else ('两数据集均值改善' if passed else '未通过')
            lines.append(f'| {cap:g}× | {name} | {values[0]:.6f} | {values[1]:.6f} | {status} |')
            details.append(dict(name=name, exploratory_mean_guard_passed=passed))
        pick = min(eligible, key=lambda name: sum(joint['summaries'][b][name]['lpips'] for b in benchmarks)) if eligible else control
        decisions[str(cap)] = dict(control=control, candidates=details, development_choice=pick,
                                    independent_validation_complete=False, final_quality_qualified=False)
    lines += ['', '均值门槛要求两个数据集分别 LPIPS 下降，且 SSIM/L2 均值不退化；不是置信区间检验。`importance_rtn` 保持基础编码、只改变重要性选择；`joint` 联合搜索缩放/尺度/支持；`ffn_guarded` 固定原 attention 投影，并在完整 up→ReLU²→down 输出上选原/新资产组合。最后一组改变了两个因素，不能单独归因于 FFN 目标。', '',
              '## 2× 的改善和剩余问题', '']
    for b in benchmarks:
        before, after = [joint['summaries'][b][name] for name in ('b2_protected_n', 'b2_joint')]
        base = joint['summaries'][b]['fp8']
        pair = joint['pairs'][f'{b}|b2_joint|b2_protected_n']['lpips']
        lines.append(f'- {b}：LPIPS {before["lpips"]:.6f} → {after["lpips"]:.6f}（下降 {100*(1-after["lpips"]/before["lpips"]):.2f}%）；SSIM {before["ssim_mean"]:.6f} → {after["ssim_mean"]:.6f}；L2 {before["relative_l2"]:.6f} → {after["relative_l2"]:.6f}；PSNR {before["psnr_db"]:.3f} → {after["psnr_db"]:.3f} dB。仍有 {pair["degraded_seeds"]}/{pair["seeds"]} 个视频 LPIPS 退化，最差 +{pair["worst_delta"]:.6f}（{pair["worst_case"]}，seed {pair["worst_seed"]}）。FP8 的 LPIPS 为 {base["lpips"]:.6f}。')
    lines += ['', '## 2.5× 的改善和剩余问题', '']
    for b in benchmarks:
        before, after = [joint['summaries'][b][name] for name in ('b2.5_ffn_aware', 'b2.5_ffn_guarded')]
        pair = joint['pairs'][f'{b}|b2.5_ffn_guarded|b2.5_ffn_aware']['lpips']
        lines.append(f'- {b}：LPIPS {before["lpips"]:.6f} → {after["lpips"]:.6f}（下降 {100*(1-after["lpips"]/before["lpips"]):.2f}%）；SSIM {before["ssim_mean"]:.6f} → {after["ssim_mean"]:.6f}；L2 {before["relative_l2"]:.6f} → {after["relative_l2"]:.6f}；PSNR {before["psnr_db"]:.3f} → {after["psnr_db"]:.3f} dB。仍有 {pair["degraded_seeds"]}/{pair["seeds"]} 个视频 LPIPS 退化，最差 +{pair["worst_delta"]:.6f}（{pair["worst_case"]}，seed {pair["worst_seed"]}）。')
    lines += ['', 'PAI 的 PSNR 平均下降约 0.18 dB，因此不能宣称所有指标都改善。RBench 的 LPIPS 均值略优于 FP8，但相对 L2 仍略差；PAI 与 FP8 仍有明显差距，整体没有达到超过 FP8 的门槛。2× 和 2.5× 偏好的量化范围不同，不能采用全预算通用的“全部冻结/全部重调”规则。']
    lines += ['', '局部线性目标平均改善 15.31%/17.04%/19.91%，完整 FFN 组合目标改善 14.20%/16.79%/19.35%，仍未稳定传递为完整生成改善。数据支持“局部目标不足以保证轨迹”，尚未证明具体是哪一层、哪一种误差传播机制导致失败。', '',
              '## 计算、存储与融合', '',
              '| 预算 | A tile 配额 | W1 并集比例 | W1 MiB | 实际请求 MMA/C0 范围 |', '|---|---:|---:|---:|---:|']
    for cap, control in controls.items():
        name = f'b{cap:g}_joint'
        cost = joint['costs'][name]
        density = {1.5: 4.650297619047619, 2.: 9.375, 2.5: 42.1875}[cap]
        ratios = cost['mma_ratio_range']
        lines.append(f'| {cap:g}× | {100*{1.5:.375,2.:.75,2.5:1.}[cap]:g}% | {density:.4f}% | {cost["offline_cost"]["w1_bytes"]/2**20:.3f} | {ratios[0]:.6f}–{ratios[1]:.6f} |')
    lines += ['', 'W1 字节不是整个模型大小。packing、selector、softmax、分类型 MMA、allocated/reserved 峰值见 FINAL.md 和原始 JSON。12 套策略通过尾块预算枚举；新增 6 套联合资产和 3 套 FFN 混合资产完成精确重载及配额审计。', '',
              '非 identity prescale 会停用原默认 ReLU²/amax 融合。已有 calibrated fusion 的完整输出逐位检查单独归档；不改变主质量实验的执行配置。没有新增 CUDA kernel，没有在共享 GPU 上调优竞速；同 MMA、同 W1 存储不代表同延迟。任意 N/K 稀疏位置也可能改变访存效率。', '',
              '## 下一步应验证的改进', '',
              '1. 将 2× joint、2.5× ffn_guarded 作为下一阶段候选，1.5× 保留原对照，不做逐场景挑选。先做独立验证，不能根据开发均值宣布超过 FP8。',
              '2. 对 QK→softmax→PV 和 FFN 在候选自身轨迹上做同输入 BF16 对照及修复开关干预，再分配 head/层/阶段配额。当前权重贡献分数忽略了通道相关性，当前线性/FFN目标也没有约束跨层传播；这些是待验证原因。',
              '3. 在已经保留的 W1 支持内比较难表示元素拆分，并同时检查交叉项开关。只有完整生成通过后才减少 A tile；规则 N/K tile 支持与融合实现的取舍留给独占延迟测量。', '',
              '实现和复测命令见 DESIGN.md；逐视频结果、失败样本、原始成本与策略均保留。本轮没有训练、梯度、删除 attention base block、复用旧 step 输出或隐藏 BF16 回退。']
    a.archive.mkdir(parents=True, exist_ok=True)
    (a.archive / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    write(a.archive / 'development_selection.json', dict(complete=True, budgets=decisions,
          evidence_sha256={str(path): sha(path) for path in paths}, performance_eligible=False,
          scope='Exploratory equal-dataset mean selection. No independent validation, statistical qualification or optimality claim.'))
    print(json.dumps(decisions))


if __name__ == '__main__':
    main()

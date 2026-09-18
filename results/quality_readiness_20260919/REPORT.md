# Edge / Nano 准确率测试入口资格

本次补齐双模型 I2V 主矩阵的运行缺口，只验证准确率流程；没有运行性能 benchmark，也没有启动最终 72 场景生成。
对应运行目录为 `/home/yufeng/thunderworld/full_quality_matrix_20260919`，离线校准与资产目录为 `/home/yufeng/thunderworld/full_quality_ready_20260919`。

## 补齐的内容

- 新样本 worker 统一使用 `baseline_runtime.make_runner`，保留实际 runtime 开关、Graph/eager 和 ARC/SharQ/AWQ 适配器。历史输出哈希变为可选，新种子不再因缺少历史输出而失败。
- Edge 原有三档数值资产和计划按 SHA 迁移到持久目录，已验证的 `norm` 执行配置保留。
- Nano 独立完成 16 个校准场景、252 个生成投影、四阶段、双 CFG 的校准。捕获 4,032 份完整 M 面板、252 份权重；36 层 SwiGLU 的 gate/up/down 纳入全 FFN 干预和联合选择，不能使用 Edge 的 ReLU²/28 层假设。
- Nano 独立准备 1.5×、2×、2.5× 资产和 head/阶段计划、ARC 全 M 通道顺序，以及官方 ModelOpt AWQ-lite 资产。ARC 增加 K12288 原生模板实例；SharQ 校验全部实际 N/K 形状。
- 增加断点恢复的输出/账本校验、原子 NPZ 写入、错误暂停、CPU 指标 worker、动态双模型报告，以及完整 final 输入的 CPU shape/预算预检。

## 主矩阵与执行定义

每模型 12 种方法：BF16、原生 FP8、普通 W4A4、AWQ 精确参考、ARCQuant 50%、ARCQuant 100%、SharQ、Ours 1.5×/2×/2.5×、全量三项、全量四项。模型 revision、168/252 个投影范围、输入、35 steps、CFG6、61 帧、192×320 和 BF16 AR 均固定。

ARCQuant/SharQ 执行已固定的官方原生算子和 Cosmos 适配，使用 eager；联合 FP4 保留其 Graph 配置。共享环境的耗时和分配器读数均不得进入性能排行榜。

### AWQ 口径调整及负结果

准确率主表的两个模型都使用 `awq_reference`：官方 ModelOpt `NVFP4_AWQ_LITE_CFG`、A16、精确解码权重和 BF16 attention。每层导出与官方量化结果逐位校验。运行时保存解码后的 BF16 权重，明确标记 `storage_comparison_eligible=false` 和 `performance_eligible=false`，不冒充 packed W4 部署，也不替代联合 FP4 路径。

Nano 原生 Marlin 未通过预先固定的 `NMSE <= 1e-4` 算子资格：首层 gate、M129/N12288/K4096 随机输入，相对精确解码参考 NMSE `0.0010945111`。Marlin 将该资产约 6.6205% 的小块尺度清零；按相同清零规则构造诊断参考后，NMSE 为 `0.00000661546`。该诊断解释误差，不替换正确性目标、不放宽门槛。原生 Marlin 保留为后续性能工作，旧 Edge 原生结果不覆盖。

## 验证结果

详见 `readiness.json`、各模型 `qualification.json`、`shape_preflight.json` 和 `smoke/REPORT.md`。开发场景指标只验证流程，不据此选择超参数或宣称优于 FP8/ARCQuant。

- 36 项 CPU 检查通过，覆盖新种子、输入/输出篡改检测、资产持久化、完整恢复、预算与既有 handoff 行为。
- Edge / Nano 各 12/12 方法通过两场景、每场景两种子的完整生成，以及 A→B→A 输出逐位一致检查。共保存 96 份输出，另执行 24 次 roundtrip；88 个非 BF16 视频都完成 LPIPS/SSIM/PSNR/L2/RMSE 评分。
- 两模型的 `b2_joint` 均通过部分恢复实测：独立副本移除一个种子后仅补该种子，补回信号逐位一致，其余输出文件修改时间不变。所有方法的完整恢复也通过校验，并在加载模型前返回。
- Nano ARC 的 4 组形状 × 3 档残差、SharQ 的 4 种实际 N/K 形状通过参考/适配检查；AWQ 的 252 层均完成官方校准导出的逐位核对。
- 新 ARC K12288 实例在 M129/N128、0/50/100% 修复下通过 compute-sanitizer：memcheck **0 errors**，racecheck **0 hazards / 0 warnings**。
- 两模型最终各 72 个输入全部完成 CPU shape 与 issued-tile 预算预检；与实际 smoke warmup 账本核对 tokenization、M 和 CFG 分支尺寸。当前修复计划覆盖本次全部输入。

| 整请求预算 | Edge final 72 场景的上界范围 | Nano final 72 场景的上界范围 |
|---|---:|---:|
| 1.5× | 1.493404–1.499957× | 1.463911–1.467236× |
| 2.0× | 1.927536–1.941912× | 1.943066–1.950771× |
| 2.5× | 2.463768–2.470956× | 2.471497–2.475350× |

以上“通过”指入口、算子、预算和恢复流程的资格，**不代表已完成最终质量评测或超过 FP8**。

预算统计是整 35 步请求的 issued-tile MMA 上界，包含原计划的阶段/CFG、尾块与动态选择上界；不是延迟预测。正式请求仍在自身 warmup 时重新核算，超预算会暂停。

## 下一步准确率运行

[运行说明](../../tools/contribution_study/QUALITY_BENCHMARK.md) 给出完整准备与恢复命令。当前本机直接顺序运行：

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py run \
  --model edge --root /home/yufeng/thunderworld/full_quality_matrix_20260919
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py run \
  --model nano --root /home/yufeng/thunderworld/full_quality_matrix_20260919
```

资格与 final manifest、实际代码和预算预检哈希绑定；改变算法/数值资产需要新研究目录与重新资格检查。每个模型 72 场景 × 2 种子 × 12 方法 = 1,728 次生成，双模型共 3,456 次。相同命令可恢复已完成请求，部分中断只生成缺失种子。

最终种子 `301113–301256`；RBench 36 场景、PAI-Bench 36 场景，使用各自模型同输入同种子的 BF16 参考。报告 LPIPS、SSIM、PSNR、视频相对 L2、RMSE、场景配对 bootstrap 区间和最差退化。主比较为三档对 ARCQuant100，FP8 单独成族，各按 12 项比较校正；保持 SSIM/L2 guard。

这是已约定子集上的 BF16 输出一致性评测，不是两个官方数据集全量或排行榜成绩，也不等价于物理合理性、动作成功率。最终集与本地拟合集的场景 ID、记录的图像 SHA、种子不重叠；未取得远端历史审计，不能声称全球从未使用。

单侧修复配平、其他算子级机制消融、policy/AV 和 5090 性能复测保持原清单的独立范围，不混入主表 3,456 次。本次未据 final 结果调整 Nano 或 Edge。

## 归档

GitHub 保存工具、manifest、三档计划、资产索引/哈希、原始指标 JSON、输出索引、资格与失败诊断。大模型、92 GiB Nano 校准面板、量化张量、生成 NPZ 和原生构建库保存在本机；不将它们打包进 Git。

`readiness.json` 列出每份归档的原始路径与 SHA。`nano/preparation` 保留官方 baseline 构建身份、算子校验、Marlin 负结果与 sanitizer 记录。`edge/smoke` 的 11 个未变方法按资产、代码、输入和输出 SHA 复用本轮前段已完成的资格结果；新 AWQ 精确参考独立运行，来源记录保留在样本 receipt。

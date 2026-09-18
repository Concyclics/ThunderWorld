# 当前版本冻结与双模型完整评测清单

本轮先冻结实验矩阵，不启动生成。实现与证据基于提交
`5d5a9dbec9c03f9f1351c949f0aa274adda25619`，CPU 审计结果见
[inventory.json](inventory.json)。这里的“完整评测”指约定的 **RBench 36 场景 +
PAI-Bench 36 场景子集**，不是两个官方数据集的全部样本，也不是官方排行榜协议。

可以冻结当前算法，但不能将两个模型都标为可直接启动：Edge 已有冻结资产和完整生成
实现；Nano 尚需按自身结构完成相同算法的校准、预算编译及适配检查。新最终样本的统一
运行入口也需要补齐。不得以 Edge 资产替代 Nano 资产，或在测试集上调整修复方案。

## 1. 冻结哪个版本

保留已经验证的离线 W0/W1、prescale、整数修复配额、head/阶段策略，以及默认
`residual_energy` 在线选择。Edge 三档采用 `norm` 执行配置：

```json
{
  "fuse_calibrated": true,
  "fuse_pointwise": true,
  "static_rotary": true,
  "codec_graph": true,
  "fuse_norm": true
}
```

这组执行开关已在 16 个 development 场景 × 2 个种子 × 3 档预算的 **96 次生成**中
通过输出逐位一致检查。它优化当前 native Triton FP4 Tensor Core 路径，不是新 CUTLASS
算法。资格记录及 SHA 已写入 inventory。原 baseline 执行配置保持现有版本，不另行调优。

不升级最近的 outlier 联合资产、represented-gain 选择器或新的 head/step schedule：
现有实验没有证明它们在两个数据集上均优于冻结方案。负结果保留在机制附录。
冻结不等于已经证明全局最优，也不意味着三档均已超过 FP8。

| 模型 | 固定模型版本 | 结构与生成投影范围 | 当前状态 |
|---|---|---|---|
| Cosmos3-Edge | `nvidia/Cosmos3-Edge@a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` | 28 层，16 Q heads / 8 KV heads，ReLU² FFN；168 个生成投影 | 三档资产、外部 baseline 和高预算质量锚点可复用；新 cohort 预检待做 |
| Cosmos3-Nano | `nvidia/Cosmos3-Nano@411f42a8fdfb8c5b2583cb8786e0938f49796eaa` | 36 层，32 Q heads / 8 KV heads，SwiGLU FFN；252 个生成投影 | 模型与通用 runner 已有；当前三档资产、AWQ/ARC 校准和模型专属资格待补 |

Nano 的 K4096/K12288、gate/up packing、GQA quota 与 Graph 形状需重新检查。Edge 的
28 层、16 heads、ReLU² 假设不能直接复制过去。Nano 可以复用算法和校准规则，不能
复制数值资产。Nano 必须在 calibration/development/validation 上完成冻结后才查看
最终质量；历史 Nano 高预算、小样本结果不能替代这次评测。

## 2. 每个模型的主表：10 个方法 + 2 个质量锚点

| 表中名称 | 固定方法 ID | 生成投影/FFN | Attention | 用途 |
|---|---|---|---|---|
| BF16 | `bf16` | 原生 BF16 | BF16 | 每个模型自己的输出参考、速度与显存基线 |
| FP8 | `fp8` | 原生 W8A8；在线 per-token 激活量化 | BF16 | 高精度低位宽对照及独立质量门槛 |
| Plain FP4 | `fp4_base` | 单平面 W4A4，无修复 | FP4，无修复 | 说明直接低精度的质量与成本 |
| NVFP4 AWQ-lite | `awq_marlin` | ModelOpt `NVFP4_AWQ_LITE_CFG`，W4A16，原生 Marlin 融合反量化 | BF16 | 保留高精度激活的权重量化对照 |
| ARCQuant 50% | `arcquant` | 官方内核的 Cosmos 适配，50% 激活残差通道 | BF16 | 中等修复量对照 |
| ARCQuant 100% | `arcquant100` | 同一适配，100% 激活残差通道 | BF16 | 完整激活修复对照 |
| SharQ | `sharq_perf` | 官方内核的 Cosmos 适配，已有 `extra_fusion=True` 路径 | BF16 | 原生 FP4 修复方法对照 |
| Ours 1.5× | `b1.5_protected_n` | 冻结的双侧稀疏修复与资产 | FP4 + 冻结的 QK/PV 修复策略 | 小预算、较小 W1 存储 |
| Ours 2.0× | `b2_joint` | 冻结的双侧稀疏修复与资产 | FP4 + 冻结的 QK/PV 修复策略 | 中间预算 |
| Ours 2.5× | `b2.5_ffn_guarded` | 冻结的 FFN 保护方案与资产 | FP4 + 冻结的 QK/PV 修复策略 | 高质量预算 |
| Full three-term | `fp4_three` | W0A0 + W0Ar + WrA0，全量支持 | 全量三项 FP4 | 检验省去交叉项的损失；质量锚点 |
| Full four-term | `fp4_four` | 三项 + WrAr，全量支持 | 全量四项 FP4 | 完整 FP4 分解的质量锚点 |

主表不重复旧的 `sharq` 未融合执行版；它只用于执行消融。AWQ 明确标注上述校准定义，
不称作 INT4 AWQ，也不把 prescale 搜索称为完整 AWQ。不得使用预先解码的 BF16 权重路径
冒充 Marlin 性能。ARCQuant/SharQ 为官方内核加 Cosmos 适配，不宣称官方原生支持该模型。

ARCQuant/SharQ 当前使用 eager，其他方法主要使用 Graph；这些执行差异必须在主表注明。
按用户要求比较现有可用实现，不再专门优化 baseline，但不能把执行路径差异全部归因于
FP4 算术。ARCQuant、SharQ 也不能一概描述为“反量化后 BF16 GEMM”。

当前 Edge 三档 W1 支持并集 payload 分别为 36,864,000、74,317,824、334,430,208 字节。
这不是模型总大小。1.5/2/2.5 是整请求 MMA 上限，不是固定每层修复比例；实际比值随
输入 shape/padding 逐请求核算。三项/四项锚点按真实成本列账，不强行放进 2.5×。

## 3. 数据、生成次数与统计口径

- 固定现有预留 final cohort：RBench 36 + PAI-Bench 36，**每场景 2 个种子**。
  完整 case、输入 SHA 和种子列表见 [final_cases.json.gz](final_cases.json.gz)。
- 种子范围 `301113–301256`，每个模型、每个方法使用完全相同的 case/seed 配对。
  RBench 为 9 类 × 4 场景；PAI-Bench 为 7 类，每类 5–6 场景。
- 固定 61 帧、192×320、35 steps、CFG 6、负提示词与输入预处理；同模型内使用同一
  生成投影范围。AR 维持 BF16，缓存策略一致，每请求计入新的 AR prefill。
- 原始数据版本：`DAGroup-PKU/RBench@6bdccf349ff5a8f68302428351e94f34ecd62450`；
  `shi-labs/physical-ai-bench-generation@c36b7beb69bd7c4adcedc1431407fce98f9dfe44`。
- 审计了本地 13,272 份 NPZ 文件名，未发现预留最终种子；72 张输入图片均存在并匹配
  SHA；与 calibration/development/validation 的 source ID、已记录的解码像素 SHA
  均无重叠。此次没有重新做语义近重复搜索；本地文件名检查也不证明远端没有运行过。
  声称跨设备完全未使用前，需归档远端运行记录。

| 范围 | 单模型生成数 | 双模型生成数 |
|---|---:|---:|
| 每个方法：72 × 2 | 144 | 288 |
| 10 个主表方法 | 1,440 | **2,880** |
| 再加全量三项、四项 | 1,728 | **3,456** |

这些计数包含生成 BF16 参考，不包含校准、预热、计时重复和独立的机制消融。已有 32 个
development 输出不能计入这次 final 的完成量。先完成 Edge 的 1,728 次，再扩展 Nano，
不得依据 Edge 的 final 结果为同一冻结方法重新搜索 Nano 超参数。

质量全部相对**同模型、同输入、同种子的 BF16 输出**，报告：

- **LPIPS ↓、SSIM ↑、PSNR dB ↑、视频相对 L2 ↓**，同时给出逐种子记录与最差退化。
- RBench、PAI-Bench 分开给表；以类别宏平均为主，合并结果仅作补充。
- 配对统计先在场景内平均两个种子，再在各类别内重采样场景；不能把同一视频的帧数
  当作独立样本数。报告普通配对 95% bootstrap 区间。
- 主比较预先定义为三档预算对 ARCQuant100；3 预算 × 2 模型 × 2 数据集，共 12 个
  比较。联合显著性使用 Bonferroni 校正的区间（每个双侧区间置信度
  `1 - 0.05/12`）；FP8 门槛另成同样大小的一组。至少 100,000 次重采样，固定统计种子。
  AWQ、SharQ、plain FP4 的差值照常完整报告，未预先纳入校正族的比较不作额外显著性主张。
- “质量超过 FP8”仍要求 LPIPS 配对差值上界 < 0，平均 SSIM 不降低、相对 L2 不增加。
  PSNR 不能替代失败的 guard。多预算同时宣称超过时使用上述校正后上界。
- BF16 自身的零误差、SSIM=1、PSNR=∞应明确表示；不要用任意大有限值参与平均。

不根据 final 结果调参、替换种子或隐藏失败任务；失败/OOM/不支持形状也要出现在完整表中。
这是相对 BF16 的生成一致性，不等价于人类偏好、物理合理性评分或动作任务成功率。

## 4. 性能与显存：单独计数，分设备报告

先完成 PRO 6000，5090 使用同一冻结方法与复测包。两台设备各自资格检查、各自出表，
不混合时延，不用共享 GPU 数据更新正式速度榜。吞吐并发仅测速度，按用户要求不验收
并发输出一致性；单路质量评测与其独立。

| 实验 | 要跑的设置 | 输出 |
|---|---|---|
| 单路完整请求 | 两模型 × 主表方法；固定 16 个 development 场景；5 轮，每轮每输入 3 种子，方法随机交错，独立加载并充分预热 | 中位数、P90、videos/s、FPS；三/四项锚点另列实际成本 |
| 吞吐上限 | 两模型 × 方法，固定 RBench/PAI 各 1 输入；并发 1/2/4/8，5 轮 × 3 waves | 总 FPS、videos/s、每请求延迟与显存；OOM/支持限制照实记录 |
| 准备与内存 | 冷加载、离线量化/资产加载、编译/建图、稳定运行分别记录 | 常驻权重、W1 union/索引、scratch、allocated/reserved 峰值 |
| 执行消融 | 冻结数值资产；相同请求下旧执行 vs `norm`，独立 profiler | 量化/packing、GEMM、attention、AR、codec 等组成，探针记录不进入速度榜 |

FPS 统一为 **61 × 完成视频数 / 总秒数**，不是输出视频的播放帧率。
完整请求含 AR prefill、35 步生成、视频解码和 CPU 输出；不含离线准备、NPZ 压缩写盘。
若另报 latent 时间，必须与完整请求分列。冷启动不能用已有编译缓存下的预热时间代替。

当前并发 harness 是独立模型副本，不是真正合并 GEMM batch，也不共享权重。报告必须写
“并发副本数”。连续两次加倍的吞吐收益均不足 5% 可停止继续加倍；这仅代表已测试配置
的饱和趋势，不证明硬件理论上限。资源不足时保留失败，不降精度、缩图或改输入凑 batch。

baseline 保持其既有原生路径；不对模拟执行报告可比性能。三/四项锚点的旧执行开关与
三档 `norm` 不同，未经同样资格检查的耗时不能用于单独归因修复项数量。

正式计时要求 GPU 无其他计算任务、`debug=off`、提前完成编译/建图、支持形状和调优表
资格。现有诊断 worker 仍标为 `performance_eligible=false`，不能只改标签把旧数据变成
正式结果。5090 对 Nano 的冷加载峰值及并发容量必须实测，不预先承诺所有档位都能放下。

## 5. 论文机制附录：与主表分开

为回答“相同预算下双侧是否优于单侧”，补齐以下配平实验。主表的 attention 策略不同，
不能直接用主表差值归因 A/W 修复。

| 配平方法 | 固定条件 | 回答的问题 |
|---|---|---|
| `single_a_act2` | 与以下两项使用同一全量 FP4 ACT2 attention | 只做投影/FFN 激活修复 |
| `single_w_act2` | 同上；全请求成本重新核算 | 只做投影/FFN 权重修复 |
| `joint2_act2` | 同上；相同 2× MMA 口径 | 预算分给双侧是否更好 |
| `joint1.5_act2`，可选 | attention 仍完全相同 | 更低总预算能否优于单侧 2× |

前三项若跑同一 72 场景、两种子、两模型，另加 **864 次**；包括主表与锚点后共
**4,320 次**。第四项另加 288 次。先在开发集确认实现、预算和冻结配置，再进入 final。
ACT2 attention 的 K/V 是当前激活，不是模型权重；这里“单 A/单 W”针对线性层修复。

outlier × 交叉项、均匀 vs head/step 分配、稀疏执行与 W1 实际压缩等原有机制实验继续
保留。尚未通过总体质量门槛的联合 outlier 候选不替换主表算法。

FourOverSix 尺度选择、ViDiT-Q 启发的分配、SageAttention3 算子适配，以及
ModelOpt lite-SVDQuant 算子参考留在机制附录。目前不能列为完整 Cosmos 生成 baseline，
也不把 lite-SVDQuant 参考称作官方 Nunchaku 的完整性能。

Edge policy、Nano policy 和 Nano AV 需要各自的输入、资产与动作/音频指标，不包含在
本轮 3,456 次 I2V 数量内。RBench/PAI 的 I2V 输入不能直接定义动作成功率或配对音频参考。

## 6. 启动顺序与尚需实现的事项

1. **保存精确的 Edge 运行资产。** `norm` manifest 当前引用 `/tmp` 下恢复的资产/plan，
   先迁移到持久目录并核对 SHA、张量字节与原始计划；单独保存外部仓库 commit、补丁和
   构建身份。不要只改路径就认定是同一资产。
2. **接通统一的新样本 worker。** `fp4_handoff/infer.py` 无条件要求历史输出哈希，新
   final 种子会查找失败；`dataset_infer.py` 绕过 `runtime_options` 与 ARC/SharQ
   adapter；`baseline_sweep.quality` 只开放 ARC100/SharQ。统一调用
   `baseline_runtime.make_runner`，保留可选历史哈希校验、预算/资产校验、断点恢复和
   完整 receipt。不存在旧哈希时不能伪造“匹配”。
3. **Edge 在 development 上 smoke test。** 所有方法输出、适配器、当前输入 Graph
   更新、FP4 硬预算、W1 并集与指标检查通过，再锁定最终 runner 修订。final 的 shape
   预检只查支持和成本，不读取质量后改变策略。
4. **运行 Edge final：1,728 次。** 单 GPU、单质量实验进程，可在共享 GPU 做质量；保存
   原始 NPZ/摘要、逐样本 JSON、失败记录与账本。独占条件下另做正式时延和吞吐实验。
5. **Nano 先完成自身校准和资格，再运行同样 1,728 次。** 校准 cohort 与 final 分离，
   同一预算定义，独立权重资产和 per-head quota。任何缺失适配明确列出，不 BF16 回退
   冒充联合 FP4 完成。Nano 的冻结策略遵循已预定算法，不能看 final 后修改。
6. **汇总四张质量表与分设备性能表。** Edge/RBench、Edge/PAI、Nano/RBench、Nano/PAI
   分开；每档给真实 MMA、packing 成本、W1 与峰值显存，保留未通过 FP8 的任务。必要的
   单侧配平消融独立提交，不与主表重复计数。

目前状态：**矩阵与本地输入审计完成；本轮未启动任何最终生成或 GPU benchmark。**
这份 inventory 是准备清单，不是可以直接交给现有旧 worker 的运行 manifest。

## 7. 复核与交付文件

CPU 审计可在当前工作区复跑（输出目录不可覆盖已有快照）：

```bash
python tools/contribution_study/full_benchmark_inventory.py \
  --workspace /home/yufeng/thunderworld \
  --output /tmp/full_benchmark_inventory_recheck
```

- [inventory.json](inventory.json)：模型 revision、投影列表、冻结方法参数、原始配置/资产
  manifest/计划 SHA、qualification、请求数与启动缺口。
- [final_cases.json.gz](final_cases.json.gz)：72 场景的原始最终 case 列表，未复制大模型或图片。
- [edge_external.json](edge_external.json)：已有外部 baseline 构建依赖身份；另参见
  `tools/fp4_handoff/sources.lock.json` 的官方仓库 commit。
- [full_benchmark_inventory.py](../../tools/contribution_study/full_benchmark_inventory.py)：
  只读本地输入、生成准备快照，不启动 GPU。

输入路径仍为本机路径。另一台设备应按数据源和 SHA 恢复输入与资产，不能仅凭一个绝对
路径视为数据已同步。GitHub 同步本次清单、工具和小型审计结果，不上传模型或生成视频。

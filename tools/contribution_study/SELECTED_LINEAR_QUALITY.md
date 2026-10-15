# Selected 双侧覆盖的可达质量比较

问题是“给定计算预算，经过选择的双侧方案能否超过全量单侧”，而不只是均匀 A/W 配额消融。用两套已经冻结的 Edge 资产与线性策略：

| 配置 | 离线表示 | A/W 支持 | 线性预算上限 |
|---|---|---|---:|
| `selected_1p5` | `b1.5_protected_n` | 原 selected 策略 | 1.5 |
| `selected_2p0` | `b2_joint` | 原 selected 策略 | 2.0 |
| `rtn_a_full` | 与 `selected_1p5` 相同 | 全 A、无 W | 2.0 |
| `rtn_w_full` | 与 `selected_1p5` 相同 | 无 A、全 W | 2.0 |
| `joint_a_full` | 与 `selected_2p0` 相同 | 全 A、无 W | 2.0 |
| `joint_w_full` | 与 `selected_2p0` 相同 | 无 A、全 W | 2.0 |

同表示对照逐字节保持 W0、prescale、outer scale 以及原有 W1。单侧对照所需的完整 W1 由原 BF16 权重离线补齐，并验证原有数值表示不变。每个推理实例只保留实际支持并集。激活始终使用当前输入。

所有配置使用上一轮相同的单平面 FP4 QK/PV attention；全部 attention 修复关闭、所有 base blocks 保留。共用 AR、非线性及编解码路径，额外 pointwise/norm/FFN 融合关闭。只比较质量与计算/存储账目，不比较共享 GPU 速度。原 attention 预算不重新分配，实际计算量可能小于标签上限。

RBench 36 + PAI-Bench 36 场景、每场景两个固定种子、35 steps、CFG 6、320×192、61 帧、Graph/debug off。生成 864 个新视频，复用 144 个严格匹配 BF16 参考。场景和种子均来自原 attention-off ablation，原始实验保持不变。

“可达质量”指这些冻结资产和策略中的经验结果，不是对所有单侧校准或所有调度算法的理论上界。不逐场景取最优，不用这 72 场景的输出重新拟合配置。该 cohort 已经观察过固定配额结果，属于后续验证，不能改称全新独立测试。双侧胜过相同表示的单侧才支持覆盖分配的收益；胜过本轮两套表示的所有单侧端点支持更强的经验前沿结论。

指标：同种子 BF16 一致性 LPIPS、SSIM、PSNR、相对 L2。先平均同场景两个种子，再按 benchmark×category 分层 bootstrap 10,000 次；八个预指定 LPIPS 比较提供 Bonferroni 家族 95% 区间。PSNR 和其他指标同时报告，不因某一指标不利而删掉。GPU 指标后端先与独立 CPU/skimage/LPIPS 校验真实样本。

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
export CUDA_VISIBLE_DEVICES=1
export TW_SHARED_CALIBRATION_GPUS=0,1
export TW_SHARED_CALIBRATION_LOCK_SCOPE=device
```

首次准备（在仓库根目录）：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/selected_linear_study.py assets --root /home/yufeng/thunderworld/fp4_selected_linear_20260923
bash tools/fp4_handoff/python.sh tools/contribution_study/selected_linear_study.py prepare --root /home/yufeng/thunderworld/fp4_selected_linear_20260923
```

生成可断点继续；已完成样本先校验哈希再跳过：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/selected_linear_study.py run --root /home/yufeng/thunderworld/fp4_selected_linear_20260923 --gpu 1
```

CPU 导出器可与生成并行；指标只在生成结束、GPU 进程释放后串行运行：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_video_export.py --root /home/yufeng/thunderworld/fp4_selected_linear_20260923/edge --follow
bash tools/fp4_handoff/python.sh tools/contribution_study/selected_linear_finish.py --root /home/yufeng/thunderworld/fp4_selected_linear_20260923 --gpu 1
```

检查 `workflow.json`（生成）、`completion.json`（生成后指标/报告）、`edge/export_progress.json`（视频）。最终报告为 `report/report.md`，原始结果为 `report/results.json`、`per_scene.csv`、`per_seed.csv`，图为 `selected_quality.pdf/png/svg`。原 float32 NPZ、完整生成日志、MP4、FFV1 和 judge 索引都保留。

部署银行中的 `repair_family`、实际 plan 和逐请求 ledger 是单侧/双侧配置的权威定义；单侧 plan 由来源 selected plan 改为全量端点，并非复用原双侧配额。

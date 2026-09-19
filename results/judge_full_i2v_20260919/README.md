# 完整 I2V 场景生成任务：启动记录

本轮是 **RBench 650 + PAI-Bench 1,044 个 I2V 场景的完整覆盖**。
此前 `judge_generation_20260919` 只包含各 36 场景的子集；用户后续明确扩展到全量，
并选择继续使用此前的低分辨率生成设置。

| 生成项 | 本轮配置 |
| --- | --- |
| 模型 | Cosmos3 Edge / Nano，沿用已冻结的本地模型与离线量化资产 |
| 方法 | ARCQuant 50%、ARCQuant 100%、SharQ `extra_fusion=True`、Ours 1.5×、Ours 2× |
| 分辨率（宽×高） | 320×192 |
| 帧数 / 帧率 | 61 帧 / 24 FPS，含条件首帧，约 2.5 秒 |
| 降噪 | 35 步，HostScheduleUniPC，flow shift 3 |
| CFG | 6.0，双分支 |
| 提示词 | 原始 `prompt` / `prompt_en`；不使用扩写版或分辨率/时长模板 |
| 负提示词 | `distorted objects, flickering, blurry, low quality` |
| 种子 | 每场景 2 个，各模型/方法共享配对；已有 301113–301256，新场景 401001–404244 |
| 声音 / debug | I2V 无声音，debug off |
| Attention | 外部三种 baseline 为 BF16；Ours 两档为 FP4，沿用既有策略 |
| 执行模式 | baseline eager；Ours Graph，保持此前已通过资格检查的配置 |

完整矩阵：

| Benchmark | 场景数 | 两模型 × 五方法 × 两种子的视频数 |
| --- | ---: | ---: |
| RBench I2V | 650 | 13,000 |
| PAI-Bench I2V | 1,044 | 20,880 |
| 合计 | 1,694 | 33,880 |

上一轮 1,440 份输出经过输入、配置和字节校验后复用；新增 32,440 份。
GPU0/1 共用 1,360 个新增任务块，每块最多 12 场景，按估算工作量排序，任一卡空闲即可领取。
这不是完成报告：请查看本机 `pool_status.json` 的实时状态。

对全部 1,694 个输入完成 CPU 形状和 MMA 预算核验：

| 模型 | 1.5× 实际 MMA 比值范围 | 2× 实际 MMA 比值范围 |
| --- | ---: | ---: |
| Edge | 1.491209–1.499957 | 1.927536–1.946619 |
| Nano | 1.462786–1.467236 | 1.943066–1.953315 |

保留完整数据集的重复图像条目，并记录各模型曾使用的校准 16、开发 16、验证 32 场景重叠。
因此全量得分不描述为完全未见场景泛化。该运行也不是 Cosmos 仓库的 720p/50 步官方复现规格，
PAI-Bench T2V 不在本次 I2V 范围内；结果不作为正式速度数据。

本机数据：`/home/yufeng/thunderworld/full_i2v_data_20260919`。
生成与日志：`/home/yufeng/thunderworld/judge_full_i2v_20260919`。
每个样本保存 MP4、经 RGB 回读核验的无损 MKV、原始 NPZ、prompt/seed/config、完整生成日志、
预算账本和 judge 索引。复用视频保持原字节及原生成 receipt，出处写入 metadata。
GitHub 仅保存代码、小型清单和启动验收快照；视频、权重、原始 NPZ 留在本机。

复测及续跑命令见 [JUDGE_GENERATION.md](../../tools/contribution_study/JUDGE_GENERATION.md)。

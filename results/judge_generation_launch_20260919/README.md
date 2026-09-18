# Edge / Nano 双 GPU 视频生成启动记录

本目录保存生成任务的冻结清单和启动验收快照。**启动记录不代表 1,440 个视频已经完成**；
持续更新的实际进度在本机 `/home/yufeng/thunderworld/judge_generation_20260919/pool_status.json`。

| 项目 | 冻结设置 |
| --- | --- |
| 模型 | Cosmos3 Edge、Nano，沿用已通过资格检查的本地模型与离线资产 |
| 数据 | 每模型 RBench 36 场景、PAI-Bench 36 场景，每场景 2 个种子 |
| 方法 | ARCQuant 50%、ARCQuant 100%、SharQ `extra_fusion=True`、Ours 1.5×、Ours 2× |
| 生成 | 35 steps、61 帧、192×320、CFG6、24 FPS |
| 种子 | 沿用冻结 final cohort：301113–301256，同场景跨方法及模型对齐 |
| 总数 | 2 模型 × 72 场景 × 2 种子 × 5 方法 = 1,440 个最终视频 |
| 设备 | 本机 RTX PRO 6000 GPU0、GPU1 |
| 调度 | 共享队列，120 块，每块 6 场景/12 视频；按历史估算工作量从重到轻领取 |

两个 worker 都允许执行 Edge 和 Nano，领取下一块取决于空闲状态。
历史共享环境时间只用于排序；不用于正式性能结论。开始最终队列前，每张卡分别复核两个模型
全部五种方法的 4 个历史开发集输出，并验证 A→B→A 当前输入更新。
资格输出不计入最终 1,440 个视频。

启动验收已通过：两张卡合计 20 个模型/方法组合、80 个历史信号逐位匹配、20 次 A→B→A；
完整记录见 `qualification.json`。首批最终样本已实际生成并完成 MP4/MKV 导出，包含帧数、
帧率、无损 RGB 回读、原始输出及日志链接检查。进度数值以 `snapshot.json` 的时间戳为准。
调度、资源锁和导出的 CPU 检查共 13 项通过；运行代码提交为 `e782902`。

每个最终样本保留原始浮点 NPZ、H.264 MP4、逐字节回读验证的 FFV1 RGB MKV、完整 stdout/stderr、
prompt/输入图/种子/方法配置、计算预算账本和输出 SHA。MP4 为有损格式；精确像素评测使用
MKV 或原始 NPZ。全量数据留在本机，GitHub 仅保存代码及小型清单。

本机输出根目录：`/home/yufeng/thunderworld/judge_generation_20260919`。
两个模型的 `judge_index.jsonl` 和 `judge_blind/inputs.jsonl` 随生成更新；全部完成后合并为
根目录 `judge_index.jsonl` 与 `judge_inputs.jsonl`。本轮只生成 judge 输入，不运行 judge。

调度、续跑、日志布局的完整说明见 [JUDGE_GENERATION.md](../../tools/contribution_study/JUDGE_GENERATION.md)。
`manifest.json.gz`、`edge/`、`nano/` 为精确冻结清单；`snapshot.json` 为带时间戳的验收快照。
所有结果均为质量生成数据，不具有正式性能排名资格。

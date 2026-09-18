# 双 GPU judge 视频生成

运行 Edge / Nano × RBench / PAI-Bench × ARCQuant 50%、ARCQuant 100%、SharQ
(`extra_fusion=True`)、Ours 1.5×、Ours 2×。沿用冻结 final 的 36+36 场景、每场景两种子，
共 1,440 份输出。35 steps、61 帧、192×320、CFG6、24 FPS，模型、量化资产及数值策略不变。
这不是官方全量 benchmark，也不在本入口执行 judge 或性能排名。

GPU0、GPU1 共用工作队列，两张卡都可以执行任一模型。默认每块 6 场景/12 视频，共 120 块，
按历史开发样本的请求耗时估算工作量，较重任务优先，空闲 GPU 立即领取下一块。
估算只用于排程，实际每块执行 GPU、开始/结束时间均记录，不把共享环境耗时当作正式性能。
分块会重复模型加载，但避免把全部 Nano 固定给一张卡；尾部最多留下一个任务块。

## 运行

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_generation.py prepare \
  --root /home/yufeng/thunderworld/judge_generation_20260919 \
  --source /home/yufeng/thunderworld/full_quality_matrix_20260919 --gpus 0 1 --chunk-scenes 6
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_generation.py launch \
  --root /home/yufeng/thunderworld/judge_generation_20260919
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_generation.py status \
  --root /home/yufeng/thunderworld/judge_generation_20260919
```

`launch` 启动独立会话的 supervisor、两 GPU worker 和两 CPU 视频 exporter，关闭交互终端
不终止任务。同一个 root 重复 launch 不创建重复活跃进程；中断后可恢复。代码、配置、
编码器和输入 SHA 均固定，禁止在运行中修改这些文件。

每张卡先对两个模型的五种方法复跑开发集两场景、每场景两种子，并做 A→B→A。
全部输出必须逐位匹配上次 GPU0 的资格记录；该卡通过后才能领取最终任务。
资格样本位于 `gpu*/qualification`，不会混入 final 的 1,440 个样本。

本次只改变资源选择/锁和 CPU 导出：`TW_SHARED_CALIBRATION_GPUS=0,1`、
`TW_SHARED_CALIBRATION_LOCK_SCOPE=device`、每进程只设置一张 `CUDA_VISIBLE_DEVICES`。
每卡独占锁防止同卡多实验；共享的全局入口锁仍与原来串行校准进程互斥。
未设置这些环境变量时，历史 GPU0/3、全局串行规则保持不变。

## 输出与日志

- `manifest.json`：完整矩阵、输入/资产/实现指纹、120 个任务及排程估算。
- `queue.json`、`pool_status.json`：每块状态、实际 GPU、尝试次数和总体进度。
- `gpu0/`、`gpu1/`：worker 标准输出、UTC 结构化事件、资格输出与错误栈。
- `tasks/<task>/generation.log`：命令、开始时间和完整 stdout/stderr，续跑追加日志。
- `<model>/final/generation/<method>/<case>/`：指向任务输出的稳定索引；每种子原始
  **float32 NPZ**、信号/文件 SHA、逐请求 receipt、预算账本、配置、完整生成日志。
- `<model>/videos/<benchmark>/<method>/<case>/seed*.mp4`：常用 judge 可读的
  H.264/yuv420p/CRF15 视频，61 帧、24 FPS，不添加文字、水印或插帧。
- 同目录 `.mkv`：FFV1 无损 RGB 视频，对每份视频解码并逐字节校验 uint8 RGB。
  浮点到 RGB 统一用 `round(clip(x,0,1)*255)`；未经取整的浮点结果始终保留在 NPZ。
  MP4 的颜色抽样/压缩不可当作无损输出，要求精确像素的评测用 MKV 或 NPZ。
- 每种子 `.json`：prompt、负提示词、模型/方法、种子、输入图、配置、输出与编码 SHA、
  尺寸、FPS、原始 receipt/日志路径；`.encode.log` 保留完整视频编码日志。
- `<model>/judge_index.jsonl`：随生成更新的完整样本索引。
- `<model>/judge_blind/inputs.jsonl`：只保留 opaque sample ID、benchmark、类别、原始样本 ID、
  prompt、输入图和匿名视频链接，省去模型/方法标签。标签映射保留在完整索引。
- `judge_blind/videos/`：共享的匿名 MP4 链接；`inputs/` 保存原始条件图的核验副本。
- 全部完成后汇总根目录 `judge_index.jsonl`、`judge_inputs.jsonl`；未完成时可直接使用
  两个模型的增量索引，不用等待全部任务。

NPZ、视频、日志都在本机研究目录，GitHub 只同步代码、冻结清单和小型运行记录。
资源不足/OOM、数值不一致、预算超限或视频损坏会暂停并保留进度；不自动降精度、缩图、
换种子或跳过失败样本。新的 source/配置需建立新的研究目录，不能覆盖旧证据。

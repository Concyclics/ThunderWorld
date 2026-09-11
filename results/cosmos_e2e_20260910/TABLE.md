# Cosmos3 Generator 端到端实测

同一图像/提示词，320×192、61 帧、35 步、CFG 6；全部方案共用外层 CUDA Graph。每方案同 seed 至少三次计时，另一个 seed 做完整轨迹对照。包含图像编码、分词、去噪、VAE 解码；不含加载、JIT、权重准备、safety checker、MP4 编码。

| 模型 | 方案 | 总延迟中位数 / 范围 s | Transformer s | PyTorch 峰值 GiB | 相对 BF16 | 相对原始 FP4 | BF16 SSIM，两 seed |
|---|---|---:|---:|---:|---:|---:|---|
| Edge | bf16 | 2.914 / [2.897, 2.918] | 1.795 | 8.22 | 1.000× | — | 1.0000 / 1.0000 |
| Edge | ffn_original | 2.769 / [2.717, 2.770] | 1.673 | 6.86 | 1.052× | — | 0.8909 / 0.8934 |
| Edge | ffn_optimized | 2.559 / [2.539, 2.564] | 1.499 | 6.88 | 1.138× | 1.082× | 0.8909 / 0.8934 |
| Edge | joint_original | 2.771 / [2.752, 2.781] | 1.752 | 6.89 | 1.051× | — | 0.8624 / 0.8552 |
| Edge | joint_optimized | 2.601 / [2.595, 2.635] | 1.571 | 6.89 | 1.120× | 1.066× | 0.8624 / 0.8552 |
| Nano | bf16 | 8.130 / [8.106, 8.311] | 7.037 | 30.25 | 1.000× | — | 1.0000 / 1.0000 |
| Nano | ffn_original | 7.118 / [7.081, 7.211] | 5.949 | 23.05 | 1.142× | — | 0.9274 / 0.8714 |
| Nano | ffn_optimized | 6.109 / [6.098, 6.157] | 5.035 | 23.06 | 1.331× | 1.165× | 0.9274 / 0.8714 |
| Nano | joint_original | 7.367 / [7.341, 7.483] | 6.299 | 23.07 | 1.104× | — | 0.8666 / 0.7958 |
| Nano | joint_optimized | 6.457 / [6.451, 6.485] | 5.378 | 23.07 | 1.259× | 1.141× | 0.8666 / 0.7958 |

倍数大于 1 表示候选更快。PyTorch 显存不含桌面、驱动及其他进程占用。FFN 为模型自身 RTN W4 + full ACT2；joint 另使用 generation attention 75% 双侧修复。没有迁移 Nano layer-0 的 AWQ prescale、校准支持或 head atlas。

| 模型 / 优化方案 | 与原始 FP4 视频逐位一致 | 同 seed 三次输出一致 | offload |
|---|---|---|---|
| Edge / ffn_optimized | True | True | resident |
| Edge / joint_optimized | True | True | resident |
| Nano / ffn_optimized | True | True | resident |
| Nano / joint_optimized | True | True | resident |

原始计时、调用计数、加载一致性、源码哈希及视频路径见 [Edge](edge_result.json) / [Nano](nano_result.json)。SSIM 只衡量与同模型参考的一致性，不代表语义、物理正确性或机器人任务成功率。

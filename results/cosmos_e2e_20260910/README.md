# Cosmos3-Edge / Nano 本机端到端结果

优先阅读 [完整报告](../../docs/COSMOS3_E2E_5090.md) 和 [实测表](TABLE.md)。

- `edge_result.json` / `nano_result.json`：共同 CUDA Graph、35 步、每模型五方案各四次请求。
- `summary.json` / `latency.csv`：中位数、范围、显存、速度倍数及两个 seed 的 BF16 一致性。
- `validation.json`：完整步数、形状、调用数量、输出确定性、优化前后逐位一致和源码验证。
- `models.json`：固定 checkpoint 版本、架构差异及所需下载文件。
- `source_snapshots/`：按 SHA256 保存测量时的源码，包含早于 Graph 接入的 eager 入口版本。
- `edge_eager_result.json`：Edge eager 的完整 35 步对照，单独保留，不混入 Graph 主表。
- `nano_eager_output_validation.json` / `graph_equivalence.json`：Nano eager 的六次输出验证，以及两模型合计 26 次 eager/Graph 视频哈希比对。
- `edge_figures/` / `nano_figures/`：由真实输出数组和计时绘制的图表，未修改视频内容。
- `adapter_tests.log`：四项新增 GPU 适配与 Graph 输入/输出测试。

输入图、视频/数组、权重和依赖源码保留在本机 git 忽略的 `data_local/`、`runs5090/` 和 `models/`。
原始 JSON 的 `video_file` 指向这些本机视频。未包含模型权重。

此试验使用每模型自身未校准 RTN W4 + full ACT2，不沿用 Nano 单层 AWQ prescale 或 head atlas。
优化 FP4 和原始 FP4 视频一致，但与 BF16 仍有量化差异；单图/两个 seed 的视频一致性不是机器人任务成功率。

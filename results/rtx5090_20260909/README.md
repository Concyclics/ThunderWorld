# RTX 5090 本机结果，2026-09-09

阅读 [完整报告](../../docs/RTX5090_RESULTS.md)，直接使用 [推荐执行参数](recommended.json)。

- [TABLE.md](TABLE.md) / [latency.csv](latency.csv)：60 个在线算子的原始 FP4、优化 FP4、BF16 延迟。
- [DIAGNOSTICS.md](DIAGNOSTICS.md)：完整本地 FFN 链、预打包 GEMM 和 M2048 压力测试。
- [summary.json](summary.json)：主表所有轮次、误差、源文件哈希和结果文件位置。
- [validation.json](validation.json) / [validation_logs](validation_logs/)：107 项测试、90 项 step/tail 回归、sanitizer 和 profiler 完成记录。
- [environment.json](environment.json)：实际 GPU、驱动、Python、CUDA 与 profiler 版本。
- [hardware_summary.json](hardware_summary.json)：六组原始/优化 Nsight kernel 计数器，duration_ns 已按 CSV 单位行归一为纳秒。
- [raw](raw/)：测量 JSON、Nsight CSV/details 和执行命令；原始测量的本机路径作为历史记录保留。
- tune_*.json：完整候选搜索及数值筛选；combine_probe.json 保留未采用的 combine 改动实验。
- [input_manifest.json](input_manifest.json) / [weight_preparation.json](weight_preparation.json)：输入及固定 revision 权重重建校验。
- [source_manifest.json](source_manifest.json) / [SHA256SUMS](SHA256SUMS)：本轮源码与本结果目录的校验值。

数据、权重、编译缓存与 .ncu-rep 二进制保留在本机 data_local/、models/、.cache5090/、runs5090/profiles_final/，未纳入此结果目录。
原始内核快照来自提交 9b51f0db23445d7695eb5b7ad9465410aeddd3cd；修改位于 perf/rtx5090-local 工作分支。

本轮保留原有修复预算，测量单算子及本地 layer-0 FFN；没有运行完整视频生成。
修复 attention 多数仍慢于 BF16，不能从 GPU busy 或 pipe active 百分比推导已达到峰值 FLOPs。

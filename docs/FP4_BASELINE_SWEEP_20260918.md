# 冻结 1.5×、ARCQuant 100% 与完整视频吞吐补测

本轮固定同版本 Edge 和已有修复策略，不重新搜索 1.5×，不专项优化 baseline 内核。
质量沿用 16 个 development 场景（RBench 9、PAI-Bench 7），每场景两个种子。
这不是预留 72 个最终场景的实验。新增 ARCQuant 100% 和 SharQ 现成性能模式的完整生成，
并独立重跑 1.5× 的 32 个视频。历史结果通过输入、资产、方法配置和输出哈希核验后复用。

最终原始记录和比较表由 `tools/contribution_study/baseline_sweep_report.py` 汇总到
`results/fp4_baseline_sweep_20260918/REPORT.md`。未完成的吞吐扫描不会生成完成报告。

## 方法与执行边界

- BF16、原生 FP8、普通 W4A4、原生 NVFP4 AWQ-lite / Marlin。
  AWQ 具体采用 ModelOpt 官方 `NVFP4_AWQ_LITE_CFG` 离线权重量化，保持 A16；
  以冻结的 E2M1/E4M3 codes/scales 调用 Marlin，比较结论限定于该实现。
- ARCQuant 50% / 100%：官方固定版本内核，已有 Cosmos 适配；保留 eager 主机 alpha 路径。
- SharQ：官方固定版本 `extra_fusion=True` 的现成残差 epilogue；没有新增 Cosmos RMSNorm 融合。
  由于合并舍入改变，重新测量该模式质量，保留旧 `extra_fusion=False` 结果。
- 我们的冻结 1.5×、2×、2.5× 联合 FP4 attention / 生成投影方案。

本轮三个修复档保持质量实验的 `linear_backend=triton`、`kernel_variant=optimized`，
使用原生 FP4 Tensor Core 及联合 Graph。没有为计时另换 CUTLASS 累加路径；
这些吞吐结果只代表冻结实现，不能称为所有已有后端中的最高性能。

共同条件为 35 步、CFG 6、192×320、61 帧、debug off，AR 缓存仍用 BF16，
每次请求重新做 AR 预填充且包含在计时内。外部线性量化方法保留 BF16 attention。
各方法真实执行成本不同，未将外部方法伪称为相同 MMA 预算。

## 吞吐口径

当前安装的 Cosmos pipeline 会将 prompt 列表截成首个样本，所以不能通过传入列表声称实现了大 batch。
`baseline_batch.py` 的 B 表示并发提交的独立完整请求；每路拥有模型、调度器、Graph、RNG 和输出。
权重按路复制，显存数据不能解释成理想共享权重服务的上限。

每种方法/代表输入按 B=1、2、4、8 递增，固定 5 轮 × 3 波；连续两次翻倍吞吐提升不足 5% 停止，
或到 B8 / 显存准入限制停止。每路 Graph 在独立捕获流上串行准备；正式计时期间不捕获或调参。
ARC/SharQ 官方内核保留默认流；其他路径使用每请求独立执行流。
统一保护 Torch SDPA 的进程级后端开关及主机 dispatch，防止一个线程的 Flash-only 上下文
让另一线程的视频 VAE 找不到合法 attention 内核；不修改已选算法，也不等待 GPU 完成。

按用户后续指令，B>1 只测速度，不以结果一致性作为准入条件。隔离捕获流之前的双路预检曾出现输出不一致；
当前并发版本不作数值验收，因此单请求质量不能直接与并发速度组合成同质量加速结论。
B1 的输出仍核对冻结质量版本。
并发组中保留 35 步/CFG 调用数检查，确保实际执行完整工作负载。

计时从一组请求同时释放起，到完整视频生成、解码和 CPU 输出转换全部完成止。
包含主机输入准备、噪声和 AR 预填充，不包含模型加载、建图、输出文件写入或事后哈希。
报告视频/秒、61 倍帧/秒、请求 P50/P90、allocated/reserved 峰值。
每档另记新增副本的加载、预热、建图及计时前核验时间；离线校准沿用冻结资产。
GPU 每波前后检查外部计算进程；共享资源模式下记录 `performance_eligible=false`，不更新正式排行榜。
观测平台表示现成实现的吞吐平台，不表示 Tensor Core 饱和或硬件峰值。

## 本机复测

工作目录：`/home/yufeng/thunderworld/ThunderWorld-pro6000-review`。

```bash
export PYTHONPATH=packages/fp4-repair/src:experiments/cutlass_fused_repair:tools/contribution_study:/tmp/thunderworld-joint-deps
TW_PY=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
TW_STUDY=/home/yufeng/thunderworld/fp4_baseline_sweep_20260918

# 只有重新建立全套独立证据时，才选择新的空目录执行 prepare。
# 原目录及历史输出不可覆盖。
$TW_PY tools/contribution_study/baseline_sweep.py prepare \
  --previous /home/yufeng/thunderworld/paper_evidence_20260918 \
  --output /tmp/fp4_baseline_sweep_new

# 对已冻结的本轮研究，质量 worker 支持续跑并核验身份。
$TW_PY tools/contribution_study/baseline_sweep.py quality --output "$TW_STUDY" --method arcquant100
$TW_PY tools/contribution_study/baseline_sweep.py quality --output "$TW_STUDY" --method sharq_perf
$TW_PY tools/contribution_study/dataset_infer.py "$TW_STUDY/manifest.json" \
  --method b1.5_protected_n --output "$TW_STUDY/fresh_1p5"
$TW_PY tools/contribution_study/baseline_fresh_verify.py --study "$TW_STUDY"

# 一个 GPU 实验进程；CPU LPIPS 进程请在性能计时前运行完。
CUDA_VISIBLE_DEVICES= $TW_PY tools/contribution_study/dataset_metrics.py \
  "$TW_STUDY/manifest.json" --generation "$TW_STUDY/generation" --output "$TW_STUDY/metrics" --follow
$TW_PY tools/contribution_study/importance_report.py \
  --run "$TW_STUDY" --archive "$TW_STUDY/quality_archive" --label quality

# 新建吞吐输出目录，冻结 worker 哈希；worker 变更后须另建输出，不能接着混跑。
TW_SPEED=/tmp/fp4_throughput_new
$TW_PY tools/contribution_study/baseline_throughput_suite.py prepare --study "$TW_STUDY" --output "$TW_SPEED"
$TW_PY tools/contribution_study/baseline_throughput_suite.py run --output "$TW_SPEED"

# 完成全部 worker 后汇总；不完整的实验会拒绝生成最终报告。
$TW_PY tools/contribution_study/baseline_sweep_report.py \
  --study "$TW_STUDY" --throughput "$TW_SPEED" \
  --archive /tmp/fp4_baseline_sweep_report

# 绘图依赖与推理环境分离，避免改变已测内核的环境。
uv pip install --python "$TW_PY" --target /tmp/thunderworld-plot-deps \
  -r tools/contribution_study/requirements-baseline-plots.txt
MPLCONFIGDIR=/tmp/thunderworld-matplotlib PYTHONPATH=/tmp/thunderworld-plot-deps \
  $TW_PY tools/contribution_study/baseline_sweep_plot.py /tmp/fp4_baseline_sweep_report

# 同一批测量换算 FPS，展示单路/最佳总吞吐及三项质量指标。
$TW_PY tools/contribution_study/baseline_fps_quality.py /tmp/fp4_baseline_sweep_report
MPLCONFIGDIR=/tmp/thunderworld-matplotlib PYTHONPATH=/tmp/thunderworld-plot-deps \
  $TW_PY tools/contribution_study/baseline_sweep_plot.py /tmp/fp4_baseline_sweep_report --throughput-unit fps
```

单独测试一个代表输入可调用 `baseline_batch.py --study ... --method ... --case ... --output NEW.json`。
默认仍为 B=1/2/4/8、5×3 波。`--pilot --rounds 1 --waves 1` 只适合启动检查，不能混入完整结果。
超过 120 秒的单请求由 watchdog 终止该实验进程并保留最后一波记录，避免 CUDA 停滞无限等待。

模型、离线资产、官方源码/扩展版本、数据输入和方法配置均由 manifest 与各 worker 的 receipt 锁定。
更换服务器需映射本地资产路径并重新建图；PRO 6000 结果不能标注成 5090 性能。

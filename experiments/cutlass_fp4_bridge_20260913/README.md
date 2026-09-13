# CUTLASS FP4 接入原型

可以用 CUTLASS 改善当前策略的实现。原型已验证现有离线 W0/W1、在线激活 packing、当前 block mask 和交叉项能够对接，但它尚未接入正式 Cosmos backend，也未完成完整轨迹质量或端到端延迟验收。

## 已验证的路径

RTX PRO 6000 Blackwell 与 RTX 5090 对应 SM120。采用 CUTLASS 的 SM120 block-scaled NVFP4 GEMM，目标指令为 `mma.sync.aligned...block_scale`；不能直接移植依赖 SM100 `tcgen05`/TMEM 的实现。当前 Triton 内核的 PTX 已包含同类 FP4 Tensor Core 指令，因此优化目标是搬运、流水、并行度和补项调度，而非首次启用 Tensor Core。

官方依据：[SM120 NVFP4 GEMM 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)、[CuTe tcgen05 的架构检查](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py)。本机实际编译使用已安装 FlashInfer 附带的 CUTLASS 4.5.0 头文件、CUDA 13.2，固定 M128/N128/K128 launch。

- **无 A/W 修复**：CUTLASS 直接在 epilogue 应用外层 scale，输出 BF16；scale 布局转换时同时计算外层 scale 乘积。
- **有稀疏修复**：CUTLASS 输出未乘外层 scale 的 FP32 base；Triton 从该 FP32 值继续累加 A1W0、A0W1 及可选交叉项，最后只做一次 BF16 转换。W1 保持已有支持并集和索引映射，A 修复读取当前 mask。
- **在线尺度布局转换**：仅重排已有 E4M3 字节，不重新量化。W0 的 CUTLASS scale 布局离线准备。全输入 amax、prescale、量化尺度和修复范围保持原定义。
- **Attention 本体**：保持现有流式实现。本实验中的 Q/K 是线性投影，不是 QK-softmax-PV attention 加速结果。

已安装 vLLM 的 CUTLASS FP4 包装只接受 FP16/BF16 输出，FP32 输出实测报错。本实验用独立 C++ CUTLASS 实例绕过该接口限制，避免“BF16 base 再修复”的中间舍入。仍然存在累加顺序变化，不能仅凭局部相近宣布完整生成质量等价。

## 真实面板诊断结果

使用已冻结的 1.25× I2V/policy 资产与阶段 0、分支 0、layer 0 的原始 full-M 校准输入。每项包含当前激活 packing、在线选择（若有）、scale 布局转换、GEMM 与修复；独立 CUDA Graph，5 轮固定正反顺序，每轮 100 次 replay。固定 launch，不搜索参数。

所有结果标记 `performance_eligible=false`、`debug=timing`、`shared-calibration`。这些是单算子诊断，不是正式性能排行榜、全模型加速或 5090 数据。

| 任务 / M | 算子 | 现有实现 μs | CUTLASS 原型 μs | 现有耗时 / 原型耗时 |
|---|---|---:|---:|---:|
| I2V / 960 | Q projection | 36.86 | 34.84 | 1.058 |
| I2V / 960 | K projection | 37.29 | 38.93 | 0.958 |
| I2V / 960 | FFN up | 61.46 | 47.32 | 1.299 |
| I2V / 960 | FFN down | 93.42 | 55.32 | 1.689 |
| policy / 416 | Q projection | 24.58 | 18.45 | 1.332 |
| policy / 416 | K projection | 32.79 | 34.84 | 0.941 |
| policy / 416 | FFN up | 31.00 | 28.69 | 1.080 |
| policy / 416 | FFN down | 87.80 | 47.44 | 1.851 |

表中四个 FFN 样本在当前冻结策略的该层/阶段均为零修复，使用直接 BF16 epilogue；这些收益不能直接外推到高修复覆盖率 FFN。两个 K projection 均有 2 个 A blocks 与 16 个 W blocks；I2V Q 有 32 个 W blocks，policy Q 为零修复。全量修复及交叉项目前有正确性覆盖，没有对应的正式速度结论。

8 个真实算子中 6 个逐位一致；两个 K projection 相对现有输出的 L2 差异分别为 1.44e-5、2.38e-5。局部 NMSE 几乎相同，但尚未运行完整轨迹门槛。

原始结果位于 [results/i2v.json](results/i2v.json) 和 [results/policy.json](results/policy.json)，包含所有样本、NMSE、实际修复配额和逐项 MMA。M128/K128 固定 launch 保持本次对照的 padded MMA 口径。桥接类拒绝不同 BM、BK 或 Split-K，避免悄悄改变计算账本。

## 成本与下一步

有修复路径需要额外 `4*M*N` 字节 FP32 scratch，以及 `8*M*N` 字节中间结果写入与读取。另有在线 A scale 重排、离线 W0 scale 副本；这些成本不增加 MMA，但会影响实际速度。无修复 fast path 已去除这块 FP32 scratch。

结果支持优先优化大 FFN，尤其 K=9216 的 down projection；尚不支持统一替换所有投影。K projection 的额外布局转换、独立补项和中间读写抵消了 base 加速。这里没有 Nsight stall/occupancy 测量，不能把这些因素的单独贡献定量归因。

推荐的后续实现顺序：

1. 将激活 base packing 与 CUTLASS scale 布局输出融合，避免事后重排。离线权重资产按 backend 准备布局，跨阶段保持同一 W0/W1 支持并集。
2. 对部分修复编写 CUTLASS/CuTe 自定义 mainloop：在同一 CTA 的 FP32 累加器中完成 base 和选中补项，复用 W0，不写 FP32 中间矩阵。对全量 W1 使用规则寻址；不把非选中补项补零后送进 dense GEMM。
3. 根据实际 M/N/K 和修复支持重新设计 tile 与流水；小投影关注 CTA 数量和填充损耗，大 K 关注持续供数。块稀疏修复不是硬件 2:4 稀疏，不套用 2:4 的峰值吞吐。
4. Attention 保持 QK 修复在当前 softmax 前、P/PV 修复在流式 tile 内完成，保留全部 base blocks。将 QK、softmax、P 量化和 PV 拆成通用 GEMM 会引入分数矩阵读写，不应仅看 QK/PV GEMM 的峰值。
5. 所有量化或累加顺序变化先跑完整轨迹门槛。随后在独占设备上比较完整请求，建立包含设备、backend、内核版本、形状和修复配置的调优表；不根据本次共享模式诊断数据直接切换生产默认。

## 编译与验证

在仓库根目录运行。仅构建本地扩展，不自动下载依赖；可用 `--cutlass /path/to/cutlass` 指定头文件目录。不同工具链重新编译，不传递不明来源的二进制。

```bash
export PYTHONPATH=packages/fp4-repair/src:experiments/cutlass_fp4_bridge_20260913:/tmp/thunderworld-joint-deps
PY=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
EXP=experiments/cutlass_fp4_bridge_20260913
$PY "$EXP/build.py" --output /tmp/tw_cutlass_bridge.so
export TW_CUTLASS_LIBRARY=/tmp/tw_cutlass_bridge.so
$PY "$EXP/run_tests.py"

/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 91 \
  "$PY" "$EXP/run_tests.py"
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 91 \
  "$PY" "$EXP/run_tests.py"

ROOT=/home/yufeng/thunderworld/compute_quality_edge_v1
$PY "$EXP/probe.py" --library "$TW_CUTLASS_LIBRARY" \
  --capture "$ROOT/capture" \
  --config "$ROOT/candidates/b1.25_grow_mixed/config.json" \
  --assets "$ROOT/candidates/b1.25_grow_mixed/assets" \
  --output /tmp/tw_cutlass_i2v_new.json
```

`--correctness-only` 关闭诊断计时。GPU 工具串行获取共享实验锁，在 GPU0/3 中选择空闲显存更多的一张；资源不足停止，不改变输入或精度。运行实例只支持串行请求；编译、布局准备在 Graph capture 前完成。

6 个测试覆盖 M129/M416/M1024、scale 字节重排、空/全修复、交叉项、全零输入、共享 W1 映射、动态 mask、Graph 当前输入和输出独立存储。memcheck/racecheck 均零错误，见 [测试输出](results/tests.txt) 和 [sanitizer 输出](results/sanitizer.txt)。源码、构建信息和原始结果作为原型交付；正式推理后端保持原实现。

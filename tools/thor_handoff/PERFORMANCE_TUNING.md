# Thor Edge：已验证的性能版本

本轮仅做速度优化与算子正确性验证。1.5×／2× 最新实现为
`/home/chenhan/thunderworld/variants/cutlass-mainloop-v6`，基础项和稀疏修复共享 CUTLASS
主循环与 FP32 Tensor Memory 累加器；attention 沿用已验证的异步分组实现。
部署与重建见 [FUSED_MAINLOOP_THOR.md](FUSED_MAINLOOP_THOR.md)。
冻结的 1.5×／2× 权重资产与修复计划保持不变，在线选择始终读取当前输入。
v6 与 v5 的累加顺序相同，42 组算子逐位一致，两档各 6 个请求的输出哈希均相同。
v5 相对更早原生实现的累加顺序变化仍需完整质量验收；本轮未做新的视频质量评分。

## 最新五轮完整请求复测

2026-09-23，Edge 320×192、61 帧、35 步、CFG 6；单请求、Transformer Graph、debug off。
五轮随机方法顺序，每种方法每轮两个固定输入各三个种子，共 30 次请求。
包含编码、生成、解码及 CPU 输出，不含加载、建图和文件保存。

| 方法 | 中位秒 | P90 秒 | FPS | 相对 BF16 加速 |
|---|---:|---:|---:|---:|
| BF16 | 7.309 | 7.410 | 8.346 | 1.000× |
| FP8 W8A8＋BF16 attention | 6.813 | 6.913 | 8.954 | 1.073× |
| 普通 W4A4＋FP4 attention v6 | 6.615 | 6.701 | 9.221 | 1.105× |
| 1.5× 修复 v6 | 7.167 | 7.274 | 8.511 | 1.020× |
| 2× 修复 v6 | 7.846 | 8.012 | 7.775 | 0.932× |
| W4A4＋BF16 attention（归因） | 6.302 | 6.403 | 9.680 | 1.160× |

普通 FP4 已快于两条基线。修复版包含既有 norm/RoPE/calibrated pointwise/codec Graph 优化，
基线及普通 FP4 保持原外围配置；修复表是完整实现对比，不是纯精度格式消融。
统一外围开关的额外实验和原始配置见 [STANDARD_COMPARISON.md](STANDARD_COMPARISON.md)。
所有结果是 MAXN/DVFS 下项目内对照，未锁频，不宣称官方峰值或正式性能排行榜。

另补一轮四配置归因、每配置六请求：将上述外围优化全部关闭后，1.5× / 2× 分别为
8.409 / 9.257 秒；同轮完整实现为 7.159 / 7.862 秒，外围优化降低延迟约 14.9% / 15.1%。
两档全部 12 对输出哈希一致、逐请求预算清单一致。该收益属于 norm/RoPE/packing/codec 的组合，
不属于 FP4 算术本身，也没有将单轮结果混入五轮中位数。

同输入 Nsight 诊断：生成投影 GEMM 的 BF16 / FP8 / FP4 累计为 1.911 / 0.871 / 0.685 秒；
FP4 attention 包含 packing/combine 为 0.788 秒，BF16 FlashAttention 为 0.496 秒。
BF16 生成投影仅占 kernel 累计时间约 26.0%；其余包括卷积、布局、归一化和 attention。
微基准同时保留纯 GEMM/在线 packing、相同 buffer/轮换工作集两种口径；不能拿带探针累计值替代请求计时。

最新完整报告：`/home/yufeng/thunderworld/thor_standard_fp4_20260923/measurements/analysis/analysis.md`。
远端原始证据：`comparisons/standard_fp4_20260923/`。下文保留历史实验，不覆盖旧结果。

## v6 主循环融合的历史归因

同轮重测 v5 与 v6，两个输入各三个种子的完整请求结果：

| 策略 | v5 秒 | 主循环融合 v6 秒 | 额外延迟下降 | v6 FPS |
|---|---:|---:|---:|---:|
| 1.5× 修复 | 8.113 | 7.187 | 11.4% | 8.487 |
| 2× 修复 | 9.134 | 7.842 | 14.1% | 7.779 |

每请求投影 GEMM kernel 从 23,520 降至 11,760 次，去除 FP32 中间写回和读取。
allocated 峰值基本不变，分别为 6.145 / 6.176 GiB。memcheck/racecheck 均通过。
原始数据为 `comparisons/cutlass_fused_20260922/`；本地报告为
`/home/yufeng/thunderworld/thor_cutlass_fused_20260923/report/report.md`。
Thor MAXN 未锁频，顺序诊断，不是五轮随机交错的正式速度排行榜。

## v5 与更早版本的历史对照

以下为上一轮两个输入各三个种子的结果，保留原测量口径：

| 策略 | v3 秒 | CUTLASS + attention v5 秒 | 延迟下降 | v5 FPS |
|---|---:|---:|---:|---:|
| 普通 W4A4 | 7.783 | 6.632 | 14.8% | 9.198 |
| 1.5× 修复 | 10.002 | 8.173 | 18.3% | 7.464 |
| 2× 修复 | 10.966 | 9.107 | 17.0% | 6.698 |

主要收益来自 CUTLASS；attention 分组使完整请求再下降约 0.5%–0.9%。
这是未锁频的顺序诊断，尚未证明 Tensor Core 达到峰值。
详细证据为 `/home/yufeng/thunderworld/thor_cutlass_upgrade_20260922/report/report.md`，
原始数据为 `comparisons/cutlass_upgrade_20260922/`。下面保留 v3 及更早实验口径。

2026-09-22 后续对照：V 布局由转置存储改为连续读取的 row-major 布局。
24 项算子检查与原生 v2 逐位一致，memcheck/racecheck 通过；下表每档仍为两个输入各三个种子。

| 策略 | 原生 attention v2 秒 | 连续 V 布局 v3 秒 | 额外延迟下降 | v3 FPS |
|---|---:|---:|---:|---:|
| 普通 W4A4 | 7.986 | 7.783 | 2.5% | 7.838 |
| 1.5× 修复 | 10.465 | 10.002 | 4.4% | 6.099 |
| 2× 修复 | 11.576 | 10.966 | 5.3% | 5.563 |

两档修复各 6 个完整请求的输出哈希与 v2 全部相同，预算和 allocated 峰值不变。
本轮 FP8 为 6.848 秒 / 8.908 FPS，当前 W4A4 v3 仍比 FP8 慢约 13.6%。
同输入 Nsight 分解（带探针，不并入上述中位数）显示：
W4A4 / FP8 投影 GEMM 为 1.814 / 0.870 秒；attention 为 1.087 / 0.496 秒，
其中 FP4 attention 包括 packing 和 combine。FP4 量化相关内核累计 0.504 秒，
没有超过 FP8 量化的 0.553 秒，因此当前主要问题是 GEMM 和 attention 执行效率。

`probe_cutlass_dense.py` 使用相同 FP4 codes/scales 测得原生 CUTLASS GEMM 在
M416/M960 的主要形状上比当前 Triton GEMM 快约 2.7–3.4 倍，五项输出逐位一致。
包含原 packing 与 scale swizzle 后收益缩小，小形状会变慢。此路径尚未接入修复推理，
不能将算子收益当成端到端收益；接入应融合尺度布局转换，并保留修复项的共享 FP32 累加。
接口参考 [vLLM 官方 NVFP4 CUTLASS 实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/cutlass/)。

后续原始记录为 `comparisons/fp4_fp8_attribution_20260922/`，本地报告位于
`/home/yufeng/thunderworld/thor_fp4_fp8_attribution_20260922/report/report.md`。
用 `attribution_report.py --measurements ... --previous-measurements ... --output ...`
可从 JSON/CSV 复算，包括失败、未部署及归因专用路径的边界。

## v2 相对原版的历史对照

同机、同输入的完整请求对照；320×192、61 帧、35 步、CFG 6、flow shift 3，
Graph、debug off。两个输入各三个种子，每版独立充分预热，包含视频解码和输出转换。
MAXN 未锁频；这是调优诊断，不是五轮随机交错的正式排行榜。

| 策略 | 原版中位秒 | 当前中位秒 | 延迟下降 | 当前生成 FPS |
|---|---:|---:|---:|---:|
| 1.5× | 13.321 | 10.465 | 21.4% | 5.829 |
| 2× | 14.080 | 11.576 | 17.8% | 5.269 |

Torch allocated 峰值基本不变，分别约 6.054 / 6.078 GiB。
实际请求计算比率最大值为 1.499956867 / 1.946619217，预算检查通过。
上述时间不包括保存 MP4 文件；FPS 是生成帧数除以请求时间，不是 24 fps 播放帧率。

## 主要改动

旧 Thor attention 的 Q32 `tl.dot_scaled` 在 PTX 中实际使用 BF16 MMA，
没有原生 NVFP4 指令。旧 Q64 也走 tcgen05 f16 路径。原先的 GEMM doctor
确实证明了 GEMM 原生 FP4，但不能证明 attention 的指令类型。

`native_attention.py` 显式使用 Gluon tcgen05 block-scaled NVFP4：

- 计算 K×Qᵀ、V×Pᵀ，使 MMA 的 M=128，N=query tile 64。
  这允许小 query tile 使用原生 FP4，避免直接 Q128 带来的额外 padding。
- 保留全部 base blocks、QK/PV 独立配额、各修复项、softmax、P 在线量化及 split combine。
- 复用原选择器、Q/K/V packing 和 GQA 支持并集；权重量化和 W1 支持不变。
- 使用显式 MMA commit/wait 屏障。不能仅依赖当前 Gluon API 对同步的文字说明。
- 通过独立包副本记录改动和 SHA256，原始研究代码、历史测速、PRO 6000 冻结任务不变。

实现参考 [Triton 官方 block-scaled MMA 教程](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html)。
环境为 SM110 / Triton 3.8.0 / CUDA 13.2；该版本限制在 Thor，不替换 SM120 路径。

24 项数值测试覆盖 M129/M416/M960、GQA、空/全修复、QK/PV 独立配额、四项修复、
显式 mask、全零和 Graph 当前输入更新；所有 repair mask 一致。
另有独立解码参考及 memcheck/racecheck，均通过。
原生 FP4 和旧 BF16-emulated 累加路径不保证逐位一致。这些检查不能替代完整生成质量评估。

## 使用已验证版本

恢复设备后，先检查是否还有实验占用 GPU。实验 GPU 锁位于 `logs/gpu-test.lock`。
所有输出目录使用新的名称，不覆盖已有结果。

```bash
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh

python tools/thor_handoff/tune_speed.py measure \
  --method b1.5_protected_n \
  --variant "$TW_THOR_ROOT/variants/cutlass-mainloop-v6" \
  --cases 0 1 --repeats 3 --output "$TW_THOR_ROOT/results/fused-b15-rerun"

python tools/thor_handoff/tune_speed.py measure \
  --method b2_joint \
  --variant "$TW_THOR_ROOT/variants/cutlass-mainloop-v6" \
  --cases 0 1 --repeats 3 --output "$TW_THOR_ROOT/results/fused-b20-rerun"
```

入口读取并核对 `variant.json` 中每个源文件的 SHA256。
Graph 准备阶段按实际 query tile 编译预算，超限即停止。
原始权重、模型 revision、输入及方法配置来自冻结的
`comparisons/edge_20260922/manifest.json`。

如需从源码重建当前独立包，按 [FUSED_MAINLOOP_THOR.md](FUSED_MAINLOOP_THOR.md)
完成 `qualify_cutlass_thor.py --fused` 和对应 sanitizer，再运行
`create_cutlass_variant.py --fused --name 新目录名`。
入口要求验证结果完整且对应当前源码与二进制 SHA256，不覆盖现有版本。

## 吞吐与未启用实验

`stream_throughput.py` 使用单进程、独立 runner、CUDA stream 和 Graph 存储，
每个 runner 内串行处理请求；仅在所属 stream 的请求边界同步。
它保留完整解码，不做质量评分。不同并发度会增加常驻模型与 scratch。

在同一个 PAI-Bench 输入上，attention 优化版从单路约 5.748 FPS 到双路
6.106 FPS，增加 6.2%；allocated 峰值增加至 12.071 GiB。
因此默认保留单路，不能把更多并发直接等同于更多 Tensor Core 有效吞吐。
四路尚未测试。

以下文件是独立实验，不是当前默认部署实现：

- `tmem_gemm.py`：显式持久 TMEM 累加器，收益较小。
- `prefetch_gemm.py`：寄存器预取版本变慢，没有启用。
- `tma_gemm.py`：恢复连接后收回全部 35 项逐位一致性检查；部分 A 修复配置有收益，
  纯 W 修复等配置反而变慢。memcheck 有 cuFuncSetAttribute 属性 17 的 API 错误，
  尚未通过全部安全检查，禁止据此报告端到端收益。
- row-major packed-V：后续资格检查与请求对照见 `fp4_fp8_attribution_20260922`，
  不混入上面的 v2 历史对照。
- Q128 原型：曾使一个 1.5× 请求达到 1.504764766，已被预算编译器拦截，没有用于推荐结果。

2026-09-22 后续连接检查确认 Thor SSH 已恢复、设备未重启、GPU 无遗留实验任务。
此前中断原因未知。新的归因实验完成并释放 GPU 后，已核对历史九方法速度控制器
PID 1971918 的命令并恢复。恢复记录为 `comparisons/edge_20260922/resumed_after_tuning.json`；
这条历史队列继续使用原冻结实现，不与 v3 结果混合。

## 证据与复算

远端原始记录在 `comparisons/edge_tune_20260922/`。
本机已同步的数据在 `/home/yufeng/thunderworld/thor_edge_tune_20260922/measurements/`。

```bash
python tools/thor_handoff/tuning_report.py \
  --measurements /home/yufeng/thunderworld/thor_edge_tune_20260922/measurements \
  --output /home/yufeng/thunderworld/thor_edge_tune_20260922/report
```

生成报告包含输入配对检查、P90、逐场景数据、预算、显存、验证范围和原始文件摘要。
没有使用共享负载数据，也没有将尚未完成的候选列入收益。
当前结果仍不能宣称已经达到 Thor 的硬件吞吐上限。

# RTX 5090 本机 kernel 优化与复现

本轮以 `docs/rtx5090-kernel-handoff` **分支**的 `tools/kernel_handoff/README.md`
为交接入口。起始提交为 `9b51f0db23445d7695eb5b7ad9465410aeddd3cd`，其内核基线
来自 `9711abb09619a6e5e40be9ba0fb71b1fd63a145b`。工作分支为 `perf/rtx5090-local`。

## 本轮实测结论

在本机 RTX 5090 上完成了 60 个主配置的搜索、独立数值校验和五轮交错对照。
全部主表计时包含当前输入的在线量化；没有通过降低修复预算获取加速。

| 测量范围 | 配置数 | 相对原始 FP4 | 相对 BF16 对照 |
|---|---:|---:|---:|
| FFN 单投影，gate/up/down × M128/512 × 六档 | 36 | 1.13–2.16×，中位 1.60× | 1.78–3.59× |
| Attention，AV/robot × M128/560 × 六档 | 24 | 0.98–1.21×，中位 1.09× | 0.49–1.20× |
| 本地完整 FFN，SiLU(gate) × up → down，M128/512 × 六档 | 12 | 1.34–1.92× | 1.70–3.09× |

60 项没有超过 3% 的性能退化。Attention 收益低于 3% 的候选优先保留原配置，
0.98× 项属于保留原配置的计时波动；不能把所有配置描述为稳定加速。
完整 FFN 使用各方案自己的 gate/up 中间结果输入 down，三步均检查相对 BF16 误差，
没有把固定捕获的 down 输入当作完整 FFN 回放。

实际接入应优先采用优化 FFN。对于本轮多数带修复的 attention 形状，若只看速度，
BF16 对照更快；保留这些 FP4 档位是为了支持相同修复预算下的研究与比较。
本轮没有证明所有算子达到 5090 的硬件上限，也没有测量完整 Cosmos3 推理。

交付入口：

- [60 项完整延迟表](../results/rtx5090_20260909/TABLE.md) 与
  [CSV](../results/rtx5090_20260909/latency.csv)。
- [推荐执行参数](../results/rtx5090_20260909/recommended.json)：按具体输入、M、preset 选择。
- [完整本地 FFN、量化开销及压力诊断](../results/rtx5090_20260909/DIAGNOSTICS.md)。
- [硬件计数器](../results/rtx5090_20260909/hardware_summary.json)、
  [验证记录](../results/rtx5090_20260909/validation.json) 和
  [本机环境](../results/rtx5090_20260909/environment.json)。
- [逐项原始测量](../results/rtx5090_20260909/raw/) 保留全部五轮样本、误差、
  原始/修改后源码哈希及实际修复配额；搜索记录与权重校验回执也在该结果目录。

## 范围与数据

- 输入包 10 个数据文件，64.94 MiB，全部逐文件 SHA256 校验通过。
- FFN：layer 0 gate/up 为 N12288/K4096，down 为 N4096/K12288；每个投影有
  robot steps 0/17/34 各 128 行真实输入。主表 M128/512，六档修复。
- Attention：layer 12、Hq32/Hkv8/D128；AV N2171，robot N845，各含 steps
  0/17/34。主表 M128/560，六档修复，包括非均匀 head-guided 配额。
- 大于 128 的 M 由真实行重复构造，用于测量性能形状，不增加质量样本数。
- 从 Hugging Face 固定 revision `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`
  获取 Cosmos3-Nano 索引和第一片权重；本机离线重建三组 FFN W0/W1，BF16 权重、
  FP4 codes/scales/global scale 的哈希均与交接包一致。
- W4 是 BF16 权重结合包内 prescale 的 RTN 结果，不是完整社区 AWQ checkpoint 的精度评测。

本轮测量单个算子的 CUDA Graph 稳态调用，包含当前输入的在线 prescale、量化/打包，
以及 attention selector、softmax、P 量化；权重准备在计时外。Attention 不计高层
`FP4Attention` graph-cache wrapper 的输入 copy 和输出 clone。没有运行完整视频生成，
不能将算子倍数当作 Cosmos3 端到端加速或完整轨迹质量结论。

FFN 的基础项为 A0W0，选中的 activation 通道块补 A1W0，选中的权重块补 A0W1；
主表不启用 A1W1 交叉项。各档位含义如下：

| FFN preset | activation 修复比例 | weight 修复比例 |
|---|---:|---:|
| base | 0 | 0 |
| a50 | 50% | 0 |
| a75 | 75% | 0 |
| full_act2 | 100% | 0 |
| a75_w6 | 75% | 6.25% |
| a93_w6 | 93.75% | 6.25% |

Attention 的修复同时作用于 QK 和 PV，各自保留完整 dense FP4 基础项：

| Attention preset | 128-token 块预算 | 修复项 |
|---|---|---|
| fp4_base | 0 | 无 |
| full_act2 | 100% | 单侧：Q1K0 / P1V0 |
| balanced | 50% | 双侧一阶：Q1K0+Q0K1 / P1V0+P0V1 |
| quality | 75% | 双侧一阶 |
| full_dual | 100% | 双侧一阶 |
| head_guided | atlas 保护 heads 75%，其余 25% | 双侧一阶 |

配额按块向上取整。例如 quality 在 AV 的实际修复率是 13/17，在 robot 是 6/7；
具体实际预算保存在每个结果 JSON 的 `cost` 或 coverage 字段，不能把 preset 名称
当作精确的逐 token FLOPs 比例。

## 本机环境

WSL2 Linux，NVIDIA GeForce RTX 5090，32GB 显存、170 SM；保持机器原有的 460W
功率上限，不更改时钟或功耗设置。复用现有 Python 3.12.11 / PyTorch 2.11.0+cu130 /
Triton 3.6.0 环境，不修改共享 Python 环境。

系统默认 PATH 曾指向 CUDA 12.0，但本机已有 `/usr/local/cuda-12.9`，包含 Nsight
Compute 2025.2.1。`python5090.sh` 使用该工具目录及独立的本地编译缓存。
Windows 主机开放性能计数器后，Nsight full profiling 已成功。
WSL 的性能计数器开关位于 Windows NVIDIA 控制面板，参见
[NVIDIA 权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM) 和
[WSL profiling 要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)。

本机现有工具已完成编译、profiling 和 sanitizer 检查，本轮无需 Toolkit 更新命令。

## 优化策略

FFN 固定 K128 修复支持、各档 activation/weight 配额和离线残差权重；扫描 M tile、
流水级数与 W0 复用/索引循环。增加可选 `split_k`：沿 K 划分 dense、activation
repair 和 sparse weight repair，写入独立 FP32 partial buffer，再合并、乘外层 scale，
最终舍入一次到 BF16。没有原子累加，也不改变修复块。小 M、长 K 的 down 投影是
这一改动的重点；scratch 大小为 `split_k * M * N * 4` 字节，单独记账。

Attention 扫描 query tile、warps 和 splits，保持 dense base、修复预算、GQA 与
逐 head 选择规则。可选 `transpose_v` 仅改变在线 packed V codes/scales 的物理存放
顺序，量化分组、数值、外层 factor 均不变，并要求与同一执行配置的原布局输出逐位一致。
仅在当前输入 Q/K/V 上量化和选择，不缓存旧 step 激活或输出。

Split 数改变在线 P 量化的归一化/累加顺序。因此每个 attention 执行候选都要通过
对应 split 的独立解码 oracle，并记录相对 BF16 的误差。另加本地筛选条件：相对
原始配置的 BF16 relative L2 不得增加超过 0.002；这不是完整生成质量保证。

尝试过增大 attention combine 的行 tile，未发现主形状稳定收益，保留默认 4 行。
原始 AV M560/quality 的 Nsight 结果显示 streaming kernel 为 255 registers/thread、
约 15.5% achieved occupancy、约 14.9% Tensor pipe elapsed utilization，存在 local
spill 访问；这一算子受寄存器和依赖延迟限制，不能用 `nvidia-smi` 的 GPU utilization
接近 100% 宣称 Tensor Core 算力已打满。

最终三组 full profiling 的对照如下。延迟是 Nsight replay/cache-flush 条件下的
单 kernel 时间，**不是主表的 CUDA Graph 稳态时间**；down 另有 11.84 μs 的
Split-K 合并 kernel，不能省略合并成本来计算完整算子加速。

| 配置 / 主要 kernel | 原始 → 优化 μs | Tensor pipe elapsed | achieved occupancy | 寄存器 / thread |
|---|---:|---:|---:|---:|
| down M128 a75_w6 / GEMM | 134.46 → 40.13 | 8.41% → 28.54% | 8.29% → 12.40% | 118 → 251 |
| gate M512 a75_w6 / GEMM | 202.88 → 129.98 | 24.08% → 37.06% | 31.85% → 15.41% | 120 → 255 |
| AV M560 full_dual / streaming | 232.03 → 215.52 | 13.48% → 14.53% | 15.84% → 15.64% | 255 → 255 |

该 attention 对照的 local load/store 各从 36,864 sectors 降至 0；pack_v 寄存器从
121 降至 96。gate 用更大 tile 和流水减少等待，即使 occupancy 降低也更快，说明
occupancy 不能单独作为优化目标。down 的长 K 分片增加并行度并缩短主循环。
这些结果仍有明显的依赖等待与寄存器约束，不能等同于峰值 FLOPs 利用率。

## 正确性与验证

- 完整 CUDA/接口测试：107 passed。
- 未参与调参的真实 steps 与 M129 尾块：90 个 case 全部通过；包括 FFN steps
  17/34、attention steps 0/34。
- 完整本地 FFN：12 配置 × 3 steps；相对 BF16 的 relative L2 比原始实现最多增加
  约 6.36e-7。
- Compute Sanitizer memcheck：79 passed、12 deselected，0 errors。
- Compute Sanitizer racecheck：同一 79 项，0 hazards、0 errors、0 warnings。
- 仓库既有 evidence 审计：15/15 通过，没有改写历史证据。
- 主表相对独立解码参考的最大 relative L2：FFN 0.001696、attention 0.001666，
  分别低于原有 0.004 / 0.012 阈值。解码参考误差和相对 BF16 的量化误差是两件事，
  两者均单独保存；未把算子数值检查当作视频质量评测。

新增测试覆盖 Split-K 的空分片、不同 M 块的支持、稀疏 W 重叠与交叉修复、非整块
尾部、修改输入后的 graph replay，以及转置 V 的 GQA、逐 head 配额和逐位一致性。

## 复现

从仓库根目录运行。原始源码快照用于同进程交错对照，必须保留：

```bash
mkdir -p data_local/baseline_source
git archive 9b51f0db23445d7695eb5b7ad9465410aeddd3cd |
  tar -x -C data_local/baseline_source

bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/verify_data.py \
  data_local/rtx5090-kernel-inputs-20260909
bash tools/kernel_handoff/python5090.sh -m pytest packages/fp4-repair/tests -q
```

`THUNDERWORLD_PYTHON` 可覆盖脚本使用的 Python 路径。GPU 工作应直接在本机执行，
单 GPU 的各个搜索、计时和 profiler 顺序运行，避免互相干扰。

搜索入口：

```bash
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/tune_ffn.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights --output runs5090/search_ffn.json
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/tune_ffn_split_k.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights \
  --search runs5090/search_ffn.json --output runs5090/search_ffn_split_k.json
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/tune_attention.py \
  --data data_local/rtx5090-kernel-inputs-20260909 --output runs5090/search_attention.json
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/tune_attention_layout.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --search runs5090/search_attention.json --output runs5090/search_attention_layout.json
python tools/kernel_handoff/build_tuning_profile.py \
  --ffn runs5090/search_ffn_split_k.json --attention runs5090/search_attention_layout.json \
  --output runs5090/recommended.json
```

最终比较与未参与调参的真实 step/tail 回归：

```bash
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/run_matrix.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights --suite main \
  --tuning results/rtx5090_20260909/recommended.json --compare-baseline --output runs5090/final_replay
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/run_regressions.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights \
  --tuning results/rtx5090_20260909/recommended.json --output runs5090/regression_replay
python tools/kernel_handoff/summarize5090.py \
  --results runs5090/final_replay --output runs5090/replay_summary

bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/bench_ffn_chain.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights \
  --tuning results/rtx5090_20260909/recommended.json --output runs5090/chain_replay.json
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/diagnostics5090.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights \
  --tuning results/rtx5090_20260909/recommended.json --output runs5090/diagnostics_replay
```

每个最终 case 在同一进程内随机交错测量原始 FP4、优化 FP4、BF16 五轮，保存全部轮次
的计时值与中位数/范围。原始函数从冻结快照加载，不把修改后的 kernel 仅恢复几个参数
冒充原始源码。共享的 FFN activation producer 有逐文件一致性断言；只共享不可变权重。

推荐表严格按 `kind:asset:M:preset` 查找，不推断未测形状。`--inner-only` 可测预打包
FFN GEMM，结果必须与完整在线路径分开。Nsight profile 使用 `--clock-control none`
和 `--nvtx --nvtx-include steady/`；其 cache flush/replay 延迟不替代稳态主表。

复现六组硬件对照与 sanitizer（均顺序运行）：

```bash
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/profile5090.py \
  --data data_local/rtx5090-kernel-inputs-20260909 \
  --weights data_local/rtx5090-prepared-weights \
  --tuning results/rtx5090_20260909/recommended.json --output runs5090/profiles_replay
python tools/kernel_handoff/summarize_profiles.py runs5090/profiles_replay/*_metrics.csv \
  --output runs5090/profiles_replay/summary.json

/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --target-processes all \
  --error-exitcode 1 bash tools/kernel_handoff/python5090.sh -m pytest \
  packages/fp4-repair/tests/test_native_launch.py packages/fp4-repair/tests/test_cuda.py \
  -k 'native_pipeline_graph or transposed_v_layout or heads_graph_and_masks' -q
/usr/local/cuda-12.9/bin/compute-sanitizer --tool racecheck --target-processes all \
  --error-exitcode 1 bash tools/kernel_handoff/python5090.sh -m pytest \
  packages/fp4-repair/tests/test_native_launch.py packages/fp4-repair/tests/test_cuda.py \
  -k 'native_pipeline_graph or transposed_v_layout or heads_graph_and_masks' -q
```

## 推理接入

Attention 的推荐参数对应 `RepairConfig(query_tile=..., splits=..., num_warps=...,
transpose_v=...)`，可与原 preset 通过 `dataclasses.replace` 合并，再交给
`FP4Attention`。head-guided 必须保留原 policy 解析后的 head 配额。

FFN 的推荐参数对应实验 `Native` 的 `bm`、`num_stages`、`num_warps`、`split_k` 和
`interleave`。其中表内 `tile_m` 映射为 `bm`，`indexed_repair` 的反值映射为
`interleave`。此对象用于交接包中的固定 activation 支持及 compact W1；这些收益
不能自动归属于另一个 `FP4Linear` 实现或未接入的新模型。

gate/up 的三步原始激活逐位相同，但两者 prescale 不同；因此不能直接共享一份已打包
activation。本轮保持独立的在线 producer。Attention 的 BF16 对照遵循交接入口，使用
预展开 GQA KV 的 PyTorch Flash Attention；KV 展开在计时外，未宣称这是所有 BF16
实现中的最优速度。

输入包、下载权重、编译缓存和大型 profiler 原始文件留在 git 忽略的本机目录。
可共享的结果应包含完整计时、正确性检查、源码哈希和推荐参数，不包含模型权重。

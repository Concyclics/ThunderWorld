# RTX 5090 kernel 调试交接

**Current Edge/Nano runtime:** [optimization status, repair budgets and receiver
commands](../../docs/OPTIMIZATION_HANDOFF.md). The original input-package workflow
below is a historical operator probe, not the current full-model handoff.

2026-09-09 本机优化已完成：[实测结果与复现](../../docs/RTX5090_RESULTS.md)、
[60 配置延迟表](../../results/rtx5090_20260909/TABLE.md)、
[推荐执行参数](../../results/rtx5090_20260909/recommended.json)。
下文保留原始交接和数据准备流程；优化后的执行参数通过 `--tuning` 显式加载。

2026-09-10 补充：[Cosmos3-Edge / Nano 完整 Generator 对照](../../docs/COSMOS3_E2E_5090.md)。
端到端默认保留 K 累加顺序，不直接迁移单算子 Split-K 参数或 Nano layer-0 校准统计。

这套入口只需要 ThunderWorld 仓库和 **64.94 MiB 的输入/校准数据**。模型由接收端从 Hugging Face 下载；传输包没有 BF16/FP4 权重、残差权重、模型 checkpoint 或编译二进制。

## 同步清单

| 内容 | 获取方式 | 是否必需 |
|---|---|---|
| `Concyclics/ThunderWorld`，分支 `docs/rtx5090-kernel-handoff` | Git clone；保留已知可运行的 commit，另开优化分支 | 必需 |
| `rtx5090-kernel-inputs-20260909/` | 从源服务器 rsync，或复制同名 `.tar.gz` | 真实输入 benchmark 必需 |
| `nvidia/Cosmos3-Nano@411f42a8fdfb8c5b2583cb8786e0938f49796eaa` 的指定分片 | 目标机 `hf download`，约 4.9 GB | FFN 真实权重测试需要；attention 不需要 |
| `Concyclics/RePair4@a3b39e5d04fc0b60ab05ded30f65989786281671` | Git clone，可选 | 仅研究已有 CUTLASS/producer 实现时参考 |
| 完整模型、VAE、tokenizer、Diffusers、完整 2 GB 激活捕获、9.5 GB AWQ checkpoint | 本轮无需同步 | 整模型回归另行准备 |

ThunderWorld 已合并的内核基线是 `9711abb09619a6e5e40be9ba0fb71b1fd63a145b`；本交接分支增加移植入口，不改变该基线的内核。接收后记录 `git rev-parse HEAD`。不要直接运行历史 `experiments/.../study.py` 等入口：它们需要本机完整捕获目录。本交接的 `bench.py` 替代这些路径依赖。

输入包内容：

- 3 个 FFN 文件：layer 0 gate/up/down，在 robot026 steps 0/17/34 捕获的真实 BF16 激活，各 128 行；冻结的 AV 校准排序统计、K64/128/256 权重块分数、少量 AWQ prescale 向量及预期权重校验值。
- 6 个 attention 文件：layer 12，AV/robot 各 3 个 step，真实 Q/K/V，含 GQA 与理解前缀。AV 为 M128/N2171/Hq32/Hkv8/D128；robot 为 M128/N845/Hq32/Hkv8/D128。
- `head_atlas.json`、逐文件长度与 SHA256 的 `manifest.json`。

完整 FFN 的两个 GEMM 形状：gate/up `N12288,K4096`；down `N4096,K12288`。调试更大 M 时脚本重复捕获行，只是性能形状扩展，不是额外精度样本。

## 1. 环境与代码

RTX 5090 与当前源服务器同属 SM120，但显存只有 32 GB，性能参数应重新测量。[NVIDIA 算力表](https://developer.nvidia.com/cuda/gpus)、[5090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

建议 Linux x86_64、Python 3.12，先固定已验证的 PyTorch 2.11.0+cu130 / Triton 3.6.0。CUDA 13.0 对应 R580 或更新的兼容驱动；需要编译 C++/CUDA 扩展和 profiling 时安装 CUDA Toolkit 13.0 及 Nsight Compute/Systems、Compute Sanitizer。[CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)

```bash
git clone --branch docs/rtx5090-kernel-handoff https://github.com/Concyclics/ThunderWorld.git
cd ThunderWorld
git rev-parse HEAD
git switch -c perf/rtx5090-local
python3.12 -m venv .venv5090
source .venv5090/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install triton==3.6.0 'safetensors>=0.5,<1' 'huggingface_hub>=1,<2' 'pytest>=8,<10'
python -m pip install --no-deps -e ./packages/fp4-repair
```

PyTorch 安装命令来自[官方版本页](https://pytorch.org/get-started/previous-versions/)。不复制源服务器 `.venvs/`、`.so`、Triton cache 或 `/tmp` build 产物；源环境部分 Python 包实际位于另一虚拟环境的路径，新机器应重新安装并编译。纯 Triton 路径不需要 RePair4、FlashInfer 或 Diffusers。

下面命令的 GPU 0 假设是目标机已获准使用的 5090；多卡机器先核实物理编号。源服务器的 GPU0/3 限制不用于推断另一台服务器的编号。

```bash
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$PWD/.cache5090/triton"
export TORCH_EXTENSIONS_DIR="$PWD/.cache5090/torch_extensions"
python tools/kernel_handoff/check_env.py --output runs5090/environment.json
```

`check_env.py` 记录 GPU/SM/显存、驱动、框架路径和 profiler 版本，不修改系统时钟或功率。首次 JIT 不计入性能数据。

## 2. 传输必要输入

源服务器目录：`/home/yufeng/thunderworld/transfer/rtx5090-kernel-inputs-20260909/`。可以在目标机执行（替换 SSH 主机名）：

```bash
mkdir -p data_local
rsync -avP USER@SOURCE_HOST:/home/yufeng/thunderworld/transfer/rtx5090-kernel-inputs-20260909/ \
  data_local/rtx5090-kernel-inputs-20260909/
export INPUTS="$PWD/data_local/rtx5090-kernel-inputs-20260909"
python tools/kernel_handoff/verify_data.py "$INPUTS"
```

也可传输压缩包并解压到 `data_local/`。只选择 `kernel-inputs` 包，清单 `data_only=true`。源端已经移除了中间生成的含权重大包。

## 3. 在目标机下载必要模型并离线准备 FFN 权重

当前 layer-0 数据只需要模型索引和第一片 transformer 权重。`hf download --dry-run` 已确认固定 revision 下这两个文件总计约 4.9 GB；无需下载完整约 35 GB 仓库。

```bash
export MODEL_DIR="$PWD/models/Cosmos3-Nano-411f42a"
hf download nvidia/Cosmos3-Nano \
  transformer/diffusion_pytorch_model.safetensors.index.json \
  transformer/diffusion_pytorch_model-00001-of-00007.safetensors \
  --revision 411f42a8fdfb8c5b2583cb8786e0938f49796eaa \
  --local-dir "$MODEL_DIR"
export PREPARED="$PWD/data_local/rtx5090-prepared-weights"
python tools/kernel_handoff/prepare_weights.py \
  --data "$INPUTS" --model "$MODEL_DIR" --output "$PREPARED"
```

若扩展导出的层范围，按 `manifest.json.model_files` 下载所有需要的分片。准备阶段逐矩阵加载，不构造完整模型。三个 FFN 的准备文件约 450 MiB，**在目标机本地生成**，包含 BF16 性能参考和完整 W0/W1 准备平面；最终 Native 对象只持有 W0 和选中的压缩 W1。

`prepare_weights.py` 校验 BF16 张量和重新生成的 codes/scales/global_scale。权重来自同一 BF16 revision，经包内 prescale 和 RTN 量化；这不是社区 packed AWQ 的精度对照。prescale 已是输入包内的小向量，因此无需额外下载完整 AWQ checkpoint。所有 W 量化、残差形成、筛选和打包都在计时外完成。

仅调 attention 时，跳过本节及 `--weights`，直接使用输入包。

## 4. 先验正确性，再运行主要配置

```bash
python -m pytest packages/fp4-repair/tests -q
python tools/kernel_handoff/run_matrix.py \
  --data "$INPUTS" --weights "$PREPARED" \
  --suite quick --output runs5090/quick_v1
python tools/kernel_handoff/run_matrix.py \
  --data "$INPUTS" --weights "$PREPARED" \
  --suite main --output runs5090/baseline_v1
```

- `quick`：7 个 case，三个 FFN A75/W6.25，AV/robot attention 各 balanced/head-guided；1 轮短测仅用于验证入口。
- `main`：FFN 每个形状 M128/512，6 档修复；attention 每种输入 M128/560，6 档修复，共 60 个 case。每 case 5 轮打乱 FP4/BF16 测量次序，保留所有样本。
- `stress`：M2048，适合看大矩阵吞吐上限，与实际小 M 分开报告。
- `--kind attention` 可完全不下载模型；`--dry-run` 打印执行矩阵。
- 每个 case 使用单独进程，整个矩阵顺序执行。失败则停止，并保留日志。每次使用新输出目录，不覆盖基线。

| 子系统 | 主要 preset | 固定语义 |
|---|---|---|
| FFN | `base`, `a50`, `a75`, `full_act2` | A0W0；按比例补 A1W0 |
| FFN | `a75_w6`, `a93_w6` | A75%/93.75%＋离线 W6.25%，默认不加交叉项 |
| Attention | `fp4_base`, `full_act2` | 全 dense FP4；或全量单侧 ACT2 |
| Attention | `balanced`, `quality`, `full_dual` | 双侧一阶修复 50%/75%/100% |
| Attention | `head_guided` | 旧 atlas 指定 heads 75%，其他 25%；用于接口和性能验证，不是已验证精度最优策略 |

Attention 从不删掉 dense 基础连接。修复按 128-token 逻辑块选择；默认用真实当前 Q/K/V 重新选择，head 配额只在计时前准备为常量 tensor。初始 FFN 使用离线固定通道排序，在所有 N 输出块间共享。

## 5. 拆分量化开销和 GEMM 吞吐

```bash
python tools/kernel_handoff/bench.py --kind ffn \
  --data "$INPUTS" --weights "$PREPARED" --asset ffn_l0_down.pt \
  --preset a75_w6 --m 512 --block-k 256 --tile-m 64 \
  --output runs5090/down_a75w6_end_to_end.json
python tools/kernel_handoff/bench.py --kind ffn \
  --data "$INPUTS" --weights "$PREPARED" --asset ffn_l0_down.pt \
  --preset a75_w6 --m 512 --block-k 256 --tile-m 64 --inner-only \
  --output runs5090/down_a75w6_gemm_only.json
python tools/kernel_handoff/bench.py --kind attention \
  --data "$INPUTS" --asset attention_av_step17.pt \
  --preset quality --m 560 --tile-m 32 --splits 2 \
  --output runs5090/attention75.json
```

常规 FFN 路径包括 prescale、amax、在线 activation quant/pack、GEMM；`--inner-only` 提前打包，隔离 GEMM，不能据此宣称端到端加速。两者延迟差受执行与缓存影响，不能当作严格可加的各阶段耗时。

Attention 包含 Q/K/V pack、selector、在线 softmax/P 量化和 FP4 MMA；不计 `FP4Attention` 高层 wrapper 的输入 copy/输出 clone。本入口预先构建 nonuniform head quota tensor，以允许将 selector 本身保留在 CUDA Graph 内。

`--key-tokens 4096` 等可重复 attention K/V，压力测试更大 N；原始 AV/robot N 必须仍作为主表。配置的实际 coverage 向上取整，JSON 会记录实际配额。FFN 可扫描 K64/128/256、M32/64/128；attention 执行 M16/32/64，**不可用 M128**，逻辑修复块仍是 128。

JSON 同时给出基于原问题 FLOPs 的 useful throughput 和含修复乘积的 nominal throughput。二者都不含所有非 GEMM 运算，也没有套用硬件 2:4 稀疏峰值。不要用营销 AI TOPS 当作同口径 dense FP4 Tensor Core 峰值。

## 6. Profiler 与回归

```bash
mkdir -p runs5090/profiles
ncu --set full --nvtx --nvtx-include 'steady/' \
  -o runs5090/profiles/down_a75w6 \
  python tools/kernel_handoff/bench.py --kind ffn \
  --data "$INPUTS" --weights "$PREPARED" --asset ffn_l0_down.pt \
  --preset a75_w6 --m 512 --block-k 256 --tile-m 64 --profile \
  --output runs5090/profiles/down_a75w6.json
nsys profile --trace=cuda,nvtx -o runs5090/profiles/attention75 \
  python tools/kernel_handoff/bench.py --kind attention \
  --data "$INPUTS" --asset attention_av_step17.pt --preset quality \
  --m 560 --profile --output runs5090/profiles/attention75.json
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python tools/kernel_handoff/bench.py --kind ffn \
  --data "$INPUTS" --weights "$PREPARED" --asset ffn_l0_down.pt \
  --preset a75_w6 --m 129 --profile --output runs5090/memcheck_down.json
```

Profiler 只关注 `steady` 范围中的一次当前输入调用，warmup 和 FP4/FP32 oracle 在范围外；Nsight Systems 的完整 trace 中仍可见它们，分析时选择该 NVTX 区段。显存统计包含 BF16 参考、验证/计时工具，不是模型部署峰值。

每次 kernel 修改后保留解码 reference 检查、BF16 误差、tail/GQA/head mask 回归，并在必要时跑 racecheck。`--skip-verify` 只适用于同源码哈希已经通过验证后的 profiling；不得用它绕过数值失败。

## 7. 给接收端 Codex

请阅读同目录 [CODEX_TASK.md](CODEX_TASK.md)，按环境、基线、profiling、单变量优化、回归的顺序推进。源端的成功记录在 [VALIDATION.md](VALIDATION.md)；它证明交接入口在 SM120 源设备上可用，**不是已在 RTX 5090 上实测**。

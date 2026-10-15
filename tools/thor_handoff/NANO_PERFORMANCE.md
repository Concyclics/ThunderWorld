# Thor Nano：固定资产的性能复测

本轮只测 I2V 的请求速度、生成 FPS 和统一内存使用，不新增视频质量结论。
模型为 `nvidia/Cosmos3-Nano`，revision
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`。
Thor 已下载并校验 68 个文件，共 34,986,890,326 bytes。
1.5×、2×、2.5× 三档资产均已恢复，每档 252 个生成投影的张量 hash 与原资产一致。
旧移植版 smoke 已通过；本次另外验证最新 `cutlass-mainloop-v6`。

## 本轮范围

- BF16、原生 FP8 W8A8＋BF16 attention、普通 W4A4＋FP4 attention、1.5×、2×。
- 320×192、61 帧、35 steps、CFG 6、flow shift 3；batch=1、单请求、Transformer Graph、debug off。
- PAI-Bench / RBench 各一个固定 I2V 输入，每输入种子 301017、301018、301031。
- 首轮为五个方法按固定随机顺序各独立加载/预热，合计 30 个计时请求。
  一轮是性能诊断，不等同于 Edge 的五轮统计或全数据集 benchmark。
- Nano 冻结方法的 `runtime_options` 均为空，codec Graph 关闭；AR cache 和 first-frame conditioning 开启。
  不把 Edge 修复版额外 norm/RoPE/codec 融合的收益带入 Nano 表，也不额外调优基线。
- FP4 采用已经编译为 `sm_110a` 的 v6 主循环融合库，保留原量化资产和修复计划。
  实际 padding 后的计算预算在每个请求检查，不用 Edge 的 ledger 代替 Nano。

推理入口复用 `CosmosRunner` 和 `tune_speed.py`，`standard_comparison.py` 只负责冻结配置、
启动独立进程及汇总结果。`fixed-model.json` 从已迁移 Nano package 解析路径，核对模型 receipt、
文件大小、资产恢复凭据和输入摘要；权重不重新校准。

## 形状资格检查

Nano 为 36 层、hidden 4096、FFN 12288、H32/KV8、head dimension 128。
线性检查覆盖 M129/M416/M960，以及 N/K 为 1024/4096/12288 的实际投影形状，
检查空/全/部分 A/W 修复、交叉项、稀疏 packing、当前输入/当前 mask 的 Graph replay。
融合库同时对已验证的 v5 分离执行参考检查逐位一致。

attention 使用 H32/KV8 的 GQA，保留 QK/PV 独立修复、非均匀 head 配额和显式 mask，
并检查 row-major V 与主循环分组不会改变已有 native 输出。
对新增形状运行 memcheck/racecheck，结果和失败日志保存在 `qualification/`。
只有所有资格检查通过后才进入完整请求计时。

## 已完成结果（2026-09-23）

五个方法各六个请求全部完成；以下为首轮中位数。

| 方法 | 完整请求秒数 | 生成 FPS | Torch allocated 峰值 GiB |
|---|---:|---:|---:|
| BF16 | 21.354 | 2.857 | 30.496 |
| FP8 W8A8＋BF16 attention | 14.716 | 4.145 | 24.019 |
| W4A4＋FP4 attention | 15.080 | 4.045 | 21.667 |
| 修复 1.5×＋FP4 attention | 21.170 | 2.881 | 21.876 |
| 修复 2×＋FP4 attention | 24.432 | 2.497 | 22.235 |

普通 FP4 的延迟比 BF16 低 29.4%，比 FP8 高 2.5%。1.5× 与 BF16 基本持平，
0.9% 的差距不足以用这轮诊断证明稳定加速；2× 比 BF16 慢 14.4%。两档修复均未超过 FP8 速度。
这些结果来自相同外围执行开关；不能直接用 Edge 的额外融合收益解释 Nano 差距。

42 个线性算子、18 个稀疏 packing 检查、24 个 attention 用例及四组 memcheck/racecheck
均通过，测量使用的库摘要与资格检查一致。每个修复请求的 17,640 次生成投影在执行清单中
均使用 `cutlass-mainloop-fused`，单次 GEMM launch、无全局 FP32 修复中间结果；
这是准备阶段执行清单，不能代替硬件利用率计数器。
1.5× 的整请求 MMA 比例为 1.4628–1.4672，2× 为 1.9431–1.9533，均满足预算。
每个输入先预热完整请求，同输入三次计时的累计 Graph 数一致；归档数据未保存首个计时请求
前后的 Graph 计数，不能仅凭累计数宣称首个请求内没有建图。

Torch 分配量下降尚不能证明 Thor 整体内存下降。修复版的进程 RSS 最大边界采样约
14.7–14.8 GiB，BF16 约 2.35 GiB；修复版系统可用内存的最小边界采样也更低。
原因尚未隔离，不能把各类内存统计相加或将差值全部归因于权重。
本轮没有 Nano 的逐算子性能 profile，不沿用 Edge 的 GEMM 占比作为 Nano 解释。

## 复测

```bash
source /home/chenhan/thunderworld/src/ThunderWorld/tools/thor_handoff/env.sh
thor_tools="$TW_THOR_ROOT/src/ThunderWorld/tools/thor_handoff"
nano_run="$TW_THOR_ROOT/comparisons/nano_standard_retest"

# Use a fresh output directory. Change --rounds to 5 for the longer protocol.
python "$thor_tools/standard_comparison.py" prepare --model nano --rounds 1 \
  --methods bf16 fp8 fp4 repair15 repair20 --output "$nano_run"

# The first deployment must pass the Nano qualification commands below before run.
python "$thor_tools/standard_comparison.py" run --output "$nano_run"
python "$thor_tools/standard_comparison.py" report --output "$nano_run"
```

首次验证的线性和 attention 命令：

```bash
mkdir -p "$nano_run/qualification"
python "$thor_tools/qualify_cutlass_thor.py" --fused --check-only --model-shapes nano \
  --library "$TW_THOR_ROOT/variants/cutlass-mainloop-v6/cutlass_sm110.so" \
  --reference-library "$TW_THOR_ROOT/variants/cutlass-attention-v5/cutlass_sm110.so" \
  --output "$nano_run/qualification/linear.json"
python "$thor_tools/qualify_native_attention.py" --kernel pipelined --row-major-v \
  --nano-heads --query-tile 64 --output "$nano_run/qualification/attention.json"
```

memcheck/racecheck 对上述命令增加 `--small`，使用新的输出文件，并在 Python 命令前加
`compute-sanitizer --tool memcheck --error-exitcode 99` 或对应 `racecheck`。
内核或编译产物变化后必须重新验证；本轮没有修改内核本身。

## 记录口径

请求时间包含输入准备、编码、35 步生成、视频解码及 CPU 输出转换，排除加载、离线准备、
编译、建图和文件保存。FPS = 61 / 请求秒数，包含条件帧，不是视频播放帧率。

Torch allocated/reserved、进程 RSS/VmHWM 和系统可用内存分列；Thor 使用统一内存，
这些值重叠，不能相加。系统可用内存只是请求边界采样，不称为系统峰值。

MAXN、未锁频，每次请求边界检查无其他计算进程。
所有记录保留 `quality_evaluation=false`、`performance_eligible=false`，不报告硬件饱和或官方峰值。
Edge 与 Nano 对比时应明确方法外围执行开关，不能将不同的融合配置混称为纯模型差异。

本轮远端：`/home/chenhan/thunderworld/comparisons/nano_standard_20260923/`。
本机镜像：`/home/yufeng/thunderworld/thor_nano_speed_20260923/measurements/`。
`driver-status.json` 记录阶段，`qualification/` 为算子和 sanitizer 证据，
`runs/` 与 `logs/` 为完整请求，`summary.json` / `report.md` 为测量汇总。
完成测量并保存资格检查后，可用 CPU 审计冻结配置、已验证内核、预算与实际调用清单：

```bash
python "$thor_tools/verify_nano.py" --campaign "$nano_run"
```

输出 `verification.json`；此审计不新增视频质量主张。

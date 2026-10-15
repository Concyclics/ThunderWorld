# AGX Thor 端侧复测环境

工作目录：`chenhan@thor:/home/chenhan/thunderworld`。代码使用独立副本，
桌面 PRO 6000 上正在运行的冻结 benchmark 及其数值实现保持原状。

**论文 §5.4 当前补测范围（2026-09-23）：** 速度只比较七种 FP4 方案，覆盖 Edge／Nano；
BF16／FP8 继续作为质量和内存参考。请先按
[FP4 速度补测交接](SECTION54_FP4_RETEST.md) 接入缺失方法与 Nano 资产，
配套 [机器可读需求](section54_fp4_retest_spec.json) 固定场景、种子、方法和交付字段。
下面的历史默认命令包含高精度计时，不能直接当成本轮补测入口。

## 环境与架构

- Ubuntu 24.04 / aarch64，Jetson Linux R39.2.0，CUDA Toolkit 13.2。
- 实测 GPU 为 NVIDIA Thor，SM110，20 SM；系统约 122 GiB **统一内存**。
- Torch 2.11.0+cu130、官方 ARM vLLM 0.25.1；其原生 FP8 W8A8 算子已单独测试。
- 主环境保留 Torch 配套的 Triton 3.6；`env.sh` 给直接 Triton JIT 使用独立的
  `compiler/triton-3.8.0`，并指定系统 CUDA 13.2 PTXAS。此组合不代表已验证 `torch.compile`。
- `overlay.py` 仅在独立 ARM 副本添加显式 SM110 opt-in，保存修改前后 SHA256 及 patch。
  不伪造 GPU capability，也不将 SM120 的 CUTLASS 二进制当作 Thor 原生实现。
- Thor 的 tcgen05 支持参见 [Triton 官方变更](https://github.com/triton-lang/triton/pull/9160)。
  `doctor.json/.ptx` 记录实际原生 FP4 指令与独立解码参考的数值检查。

2026-09-22 调优审计补充：`doctor.ptx` 的原生 FP4 证据针对 **GEMM**。
旧 Thor 移植版 Q32 attention 的 `dot_scaled` 被编译器展开为 BF16 MMA，
Q64 旧实现也并非原生 NVFP4；不能据 GEMM doctor 推断整条 attention 路径。
新增 `native_attention.py` 用转置操作数显式发射 tcgen05 NVFP4，并保留完整 packing、
selector、softmax、P 量化和修复项。调优代码使用独立版本目录，历史结果不覆盖。
已验证的配置、端到端数据与中断状态见 [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md)。
Nano 的最新 CUTLASS v6 形状资格检查、固定资产速度测试和内存口径见
[NANO_PERFORMANCE.md](NANO_PERFORMANCE.md)。
后续 CPU 资产释放、紧凑 A1 缓冲和 Edge/Nano 尺寸扩展复测见
[MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md)。

## 固定模型、资产和输入

| 模型 | Hugging Face revision | 原始文件大小 |
|---|---|---:|
| Cosmos3-Edge | `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` | 9,176,128,932 bytes |
| Cosmos3-Nano | `411f42a8fdfb8c5b2583cb8786e0938f49796eaa` | 34,986,890,326 bytes |

`data/model_files.json` 保存本机原始文件逐个 SHA256，下载到 `models/` 后必须全部验证，
才写入各模型的 `thunderworld_verified.json`。不传账号 token，不自动升级模型 revision。

`data/{edge,nano}` 保存当前 1.5×、2×、2.5× 的固定策略、输入图片及紧凑离线资产配方。
重建使用既有 CPU 离线量化工具和逐字节修正，**所有张量 hash 必须与原始资产相同**，
失败时停止，不用重新校准掩盖跨平台差异。`assets/{edge,nano}/restored.json` 是完成凭据。
新设备不重新选参；激活在线修复仍使用当前输入。

输入使用现有 smoke cohort 的 PAI-Bench 和 RBench 各一个场景。完整速度测试可继续加
场景，但不能把这两个输入称为完整 benchmark。保持 **320×192、61 帧、35 steps、CFG 6、
flow shift 3、24 fps 播放帧率、AR cache / first-frame conditioning**。
方法自身已冻结的融合开关在 `study.json` 中保留，`result.json` 记录实际配置。

## 使用

```bash
ssh chenhan@thor
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh

# 查看后台准备进度、模型逐文件校验进度。
cat "$TW_THOR_ROOT/logs/setup-status.json"
cat "$TW_THOR_ROOT/logs/model_download.json"

# 环境与独立数值检查。
python tools/thor_handoff/doctor.py --output "$TW_THOR_ROOT/results/doctor-rerun.json"

# 默认关闭探针；先 prepare、建图，再进行一次完整请求预热。
python tools/thor_handoff/run.py --model edge --method b1.5_protected_n \
  --repeats 5 --output "$TW_THOR_ROOT/results/edge-b15-repeat" --save-outputs
python tools/thor_handoff/run.py --model nano --method bf16 \
  --repeats 5 --output "$TW_THOR_ROOT/results/nano-bf16-repeat"
```

支持的方法参数为 `bf16`、`fp8`、`fp4_base`、`b1.5_protected_n`、`b2_joint`、
`b2.5_ffn_guarded`、`fp4_three`、`fp4_four`。只有完成当前设备验证的结果才能纳入比较。
`--case 0|1` 选择两个固定输入，`--execution eager|graph` 可单独检查执行路径。
一个 GPU 测试锁避免此入口互相重叠；不管理其他用户进程。

初始化/恢复命令（先检查后台 PID，避免重复启动）：

```bash
python tools/thor_handoff/download_models.py --root "$TW_THOR_ROOT" --workers 4
python tools/thor_handoff/continue_setup.py --root "$TW_THOR_ROOT"
```

后台准备程序按 Edge、Nano 顺序等待已校验模型，恢复三档资产，然后串行执行
BF16 / FP8 / 普通 FP4 / 三档修复的完整 35 步 smoke 请求，每项保存 NPZ、输出 hash 和日志。
模型传输、资产恢复、算子测试、完整推理是不同的验收项，不能只看环境安装成功。
失败时查看 `logs/`，已完成的模型和 smoke 结果可复用。

## 测量口径和边界

- 请求耗时与生成 FPS 包含编码、35 步生成、解码及必要输出转换；文件压缩保存单独执行，
  不计入请求。`generated_fps = 61 / request_seconds`，不同于视频播放帧率 24 fps。
- 单独记录模型加载、离线准备、建图、完整请求预热、请求中位数/P90。
- Torch `peak_allocated/reserved`、常驻模型张量、进程 RSS/VmHWM 和系统 MemAvailable
  分列。它们共享统一物理内存，**不能相加**；MemAvailable 仅在请求边界采样，不是系统峰值。
- 当前移植数据统一 `performance_eligible=false`。SM120 调优表不得复用；
  安全 launch 的缺失条目写入 receipt。逻辑修复预算与 SM110 实际发射 MMA 的 padding
  需要进一步核算，不能直接搬用桌面卡 FLOP 效率结论。
- 当前设备为 MAXN。锁频需管理员权限，本轮用户空间设置不改系统功耗/风扇；
  后续独占计时记录功耗模式、频率、温度、后台负载及 Thor 专属调优表。
- 原有 SM120 CUTLASS fused 二进制不能用于 Thor。下面的 baseline 补充分别从官方源代码
  编译真正的 SM110 原生实现；AWQ decoded reference 不作为原生 W4A16 性能结果。
  不支持的路径不做隐式 BF16 回退。

原始环境快照、算子测试 XML、PTX、patch 位于远端 `results/` 与源码根目录。
请结合 `setup-status.json` 查看 **实际完成**的完整模型验证。

## Edge 原生 baseline 速度比较（2026-09-22）

用户已明确本轮 Thor **只测速度，不评估质量**。比较 1.5×、2×、ARCQuant 50% / 100%、
SharQ（官方 `extra_fusion=True`）、NVFP4-AWQ W4A16 Marlin、普通 W4A4、FP8 W8A8、BF16。
共 9 项，使用两个固定输入、每个输入三个相同种子，五轮随机交错，共 270 个计时请求。
这些输入用于速度测量，不能称为全量 RBench/PAI-Bench 质量评估。

`build_baselines.py` 验证原始源文件摘要，在独立副本生成 `thor.patch`，保留编译命令、
CUDA 目标与库摘要。ARCQuant 保留量化器和通道排序，增加 Cosmos K9216 实例，
使用 CUTLASS Sm100 tcgen05 模板和 128×128×256 tile，**实际编译目标为 sm_110a**。
SharQ 使用其现有 SM100 分支、原生稀疏与稠密算子，实际编译目标同为 sm_110a。
模板类型名不等于设备 capability；没有把设备伪装成 SM100。

官方源锁定：ARCQuant `dbf841e123204ee0584006541dea3b4a539b2e1c`、
SharQ `3cc584fe48e32e59c8b4e37323470dcdd606bb11`；其 CUTLASS 子模块版本记录在
`data/baseline_sources.json`。原始副本在 `external/{arcquant,sharq}`，
编译副本在 `external/{arcquant,sharq}-sm110`。

```bash
source tools/thor_handoff/env.sh
# 仅在没有已有 build 副本时运行；不覆盖历史构建。
python tools/thor_handoff/build_baselines.py --method arcquant
python tools/thor_handoff/build_baselines.py --method sharq

# 核心正确性与 Graph/当前输入检查是速度测试的先决条件，不是质量评估。
python tools/thor_handoff/probe_baselines.py --method awq_marlin \
  --output "$TW_THOR_ROOT/comparisons/edge_20260922/awq_marlin-probe.json"
python tools/thor_handoff/probe_baselines.py --method arcquant \
  --output "$TW_THOR_ROOT/comparisons/edge_20260922/arcquant-probe.json"
python tools/thor_handoff/probe_baselines.py --method sharq \
  --output "$TW_THOR_ROOT/comparisons/edge_20260922/sharq-probe.json"

# prepare 只执行一次；run 可跳过已完成且 manifest 相同的轮次。
python tools/thor_handoff/compare_speed.py prepare
python tools/thor_handoff/compare_speed.py run
python tools/thor_handoff/compare_speed.py report
```

修改内核需补做 `compute-sanitizer --tool memcheck` 和 `--tool racecheck`；
当前运行凭据在 `comparisons/edge_20260922/sanitizers.json` 与对应日志。
完整数值检查解码实际打包值，覆盖 M129 / M416 / M960、零输入；
SharQ 参考包含稀疏首项的 BF16 舍入，AWQ 额外检查 Graph replay 的当前输入更新。

计时前先结束迁移验证，方法各自独立加载和充分预热；每个计时请求前后检查其他 CUDA
计算进程。ARCQuant/SharQ 保留官方 eager 执行限制，其他方法使用已冻结 Graph 配置。
结果 JSON 明确记录此差异，不把完整实现速度差异全部归因于 FP4 GEMM。
MAXN 未锁频且尚未调优，所以结果是可复查的 Thor 移植版实测，不作为调优后性能上限。
状态为 `status.json`，原始记录为 `runs/round-*/METHOD/result.json`，
汇总为 `summary.json` 和 `report.md`，完整 stdout/stderr 在 `logs/`。

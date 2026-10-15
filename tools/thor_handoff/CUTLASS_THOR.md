# Thor CUTLASS NVFP4 与流式 attention

本实现只用于 Thor SM110 的速度实验；不修改 PRO 6000 冻结任务的包、模型、权重资产或修复策略。
`cutlass-base-v4` 隔离 GEMM 收益；`cutlass-attention-v5` 增加 attention 操作数复用和异步 MMA 分组。
两者均由 `create_cutlass_variant.py` 生成独立、可校验 SHA256 的包，原安装和历史队列保持原样。
完整结果从原始 JSON 生成，不能用这里的算子检查代替生成质量验收。

## 实现

- `cutlass_sm110.cu`：CUTLASS 4.3 block-scaled NVFP4，CUDA 13.2 编译为 `sm_110a`，
  TMA/warp-specialized mainloop，固定 M128/N128/K128，FP32 累加。尺度布局在权重准备时转换；
  激活使用已有 direct scale packing，一次 packing 同时产生原生修复和 CUTLASS 所需的布局。
- `cutlass_thor_backend.py`：普通 W4A4 使用 CUTLASS 直接输出 BF16；有修复时输出未缩放 FP32 基础项，
  `cutlass_repair_tail.py` 在该结果上累加所有选中的 A/W/交叉 FP4 项，最后统一缩放并转 BF16。
  保留紧凑 W1 支持并集、CSR 映射、当前输入和 mask。无 BF16 GEMM 回退、无在线权重量化。
- 有修复时仍是两个 GEMM kernel，增加 `4*M*N` scratch 和 `8*M*N` 写读流量；这是当前实现的代价，
  不是“完全单内核融合”。运行预算和报告显式列出此开销、尺度副本及 kernel 数。
- 与原 interleaved A repair 相比，先完成 base 再加补项会改变 FP32 累加顺序。
  随机算子测试通过不能保证完整生成质量不变；Thor 本轮只测速度，质量推荐需单独重新验收。
- `pipelined_attention.py`：保留原量化定义、QK/PV 独立配额、GQA 共享支持、softmax、P 量化和全部 base blocks。
  base 与 residual 使用独立 shared-memory / tensor-memory 操作数，每组有序发射多项 MMA，组末一次 commit/wait。
  QK 和 PV 仍在同一 streaming kernel 内，原 selector、packing、combine 接口不变。
- 请求在 Graph 中执行；所有编译、库校验、预热和 capture 均在请求计时之前完成。
  固定 launch 不接受自动调优覆盖，避免执行 tile 与预算账本失配。

## 为什么没有直接启用官方 FP4 attention 包

2026-09-22 检查官方入口：

| 后端 | 检查结果 | 当前选择 |
|---|---|---|
| [FlashInfer NVFP4 attention](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py) | NVFP4 入口限定 SM120/121，带其自己的中心化及校正语义 | Thor SM110 不直接接入；不绕过架构检查 |
| [FlashAttention CuTe](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py) | 有 SM110 架构路径，当前 dtype 入口没有 E2M1 FP4 | 保留本项目的 FP4 attention 路径 |
| 本项目 native FP4 attention | 显式 tcgen05 NVFP4，支持当前动态修复及 Graph | 在此基础上实现操作数复用和异步分组 |

这是 FlashAttention 风格的融合流式计算，不是官方 FlashInfer/FlashAttention 内核。
5090/PRO 6000 的 SM120 可以另行验证官方 NVFP4 attention，但需要重新对齐量化、修复 mask 和质量；
不能直接搬用本轮 SM110 二进制或性能结论。

## 重建及资格检查

先确认 GPU 空闲、历史速度控制器已在任务边界暂停。脚本使用 `logs/gpu-test.lock` 防止本项目互相干扰，
同时检查其他 GPU 进程。输出请使用新目录，保留历史证据。

```bash
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh
thor_run="$TW_THOR_ROOT/comparisons/cutlass_retest"
thor_tools="$TW_THOR_ROOT/src/ThunderWorld/tools/thor_handoff"
mkdir -p "$thor_run"

python "$thor_tools/build_cutlass_sm110.py" --output "$thor_run/cutlass_sm110.so"
python "$thor_tools/qualify_cutlass_thor.py" \
  --library "$thor_run/cutlass_sm110.so" --output "$thor_run/qualified.json"
compute-sanitizer --tool memcheck --error-exitcode 86 python "$thor_tools/qualify_cutlass_thor.py" \
  --library "$thor_run/cutlass_sm110.so" --small --check-only \
  --output "$thor_run/memcheck.json" > "$thor_run/memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode 87 python "$thor_tools/qualify_cutlass_thor.py" \
  --library "$thor_run/cutlass_sm110.so" --small --check-only \
  --output "$thor_run/racecheck.json" > "$thor_run/racecheck.log" 2>&1

python "$thor_tools/qualify_native_attention.py" --kernel pipelined --row-major-v \
  --output "$thor_run/attention.json"
compute-sanitizer --tool memcheck --error-exitcode 86 python "$thor_tools/qualify_native_attention.py" \
  --kernel pipelined --row-major-v --small --output "$thor_run/attention_memcheck.json" \
  > "$thor_run/attention_memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode 87 python "$thor_tools/qualify_native_attention.py" \
  --kernel pipelined --row-major-v --small --output "$thor_run/attention_racecheck.json" \
  > "$thor_run/attention_racecheck.log" 2>&1
# Additional ACT2 coverage and full operator timing, including packing:
python "$thor_tools/qualify_native_attention.py" --kernel pipelined --row-major-v \
  --terms 2 --benchmark --output "$thor_run/attention_act2.json"

python "$thor_tools/create_cutlass_variant.py" \
  --parent "$TW_THOR_ROOT/variants/native-row-major-v3" --name cutlass-retest \
  --library "$thor_run/cutlass_sm110.so" --qualification "$thor_run/qualified.json" \
  --memcheck "$thor_run/memcheck.json" --racecheck "$thor_run/racecheck.json" \
  --attention-qualification "$thor_run/attention.json" \
  --attention-memcheck "$thor_run/attention_memcheck.json" \
  --attention-racecheck "$thor_run/attention_racecheck.json"
```

不传最后三个 attention 参数可创建仅替换 CUTLASS 的对照。构建器验证代码、二进制、数值测试和 sanitizer 记录。
模型、两个输入、1.5×/2× 紧凑 W1 资产及策略来自冻结的 `comparisons/edge_20260922/manifest.json`。

## 完整请求测速与报告

```bash
for method in fp4_base b1.5_protected_n b2_joint; do
  python "$thor_tools/tune_speed.py" measure --method "$method" \
    --variant "$TW_THOR_ROOT/variants/cutlass-retest" \
    --cases 0 1 --repeats 3 --output "$thor_run/requests/$method"
done
```

本轮结果目录为 `comparisons/cutlass_upgrade_20260922/`，分别包含 `cutlass-base-v4` 和 `cutlass-attention-v5`。
本地镜像位于 `/home/yufeng/thunderworld/thor_cutlass_upgrade_20260922/measurements`。

```bash
python tools/thor_handoff/cutlass_report.py \
  --measurements /home/yufeng/thunderworld/thor_cutlass_upgrade_20260922/measurements \
  --previous /home/yufeng/thunderworld/thor_fp4_fp8_attribution_20260922/measurements \
  --output /home/yufeng/thunderworld/thor_cutlass_upgrade_20260922/report
```

报告验证模型/资产/输入/策略/执行条件配对，逐档列出延迟、FPS、allocated/reserved、预算、
额外 FP32 流量和完整请求哈希。历史 FP8 仅作同条件参照；微小差异需要随机交错复测。
这些数据不能说明已达到 Tensor Core 峰值，也不能替代完整质量门槛。

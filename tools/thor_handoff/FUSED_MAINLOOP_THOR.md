# Thor CUTLASS 稀疏修复主循环

`cutlass-mainloop-v6` 将现有 v5 的基础 GEMM 和修复 GEMM 合并为一次 CUTLASS kernel 调用。
仅针对 Thor SM110，保持 M128/N128/K128、冻结的 W0/W1、A/W 支持、35 步和 attention 实现不变。
本轮验证相对 v5 的数值一致性与速度，不重新进行视频质量评分。

## 已完成的端到端对照

两种版本在同一轮独立加载、预热；两个输入各三个固定种子，320×192、61 帧、35 步、CFG 6，
Graph、debug off。时间包含编码、生成、解码和 CPU 输出，不含离线准备、建图和文件保存。

| 修复预算 | v5 中位秒 | 融合 v6 中位秒 | 延迟下降 | v6 P90 秒 | v6 生成 FPS |
|---|---:|---:|---:|---:|---:|
| 1.5× | 8.113 | 7.187 | 11.4% | 7.283 | 8.487 |
| 2× | 9.134 | 7.842 | 14.1% | 7.998 | 7.779 |

两档各 6 个完整请求的输出哈希均与 v5 相同，资产、支持、MMA 预算及 W1 存储均未变化。
每请求投影 GEMM kernel 从 23,520 次降至 11,760 次，去除最大 33.75 MiB 的单算子中间缓冲。
逻辑中间读写减少 244.043 GiB/请求；这不是实测 DRAM 流量。
Torch allocated 峰值为 6.145 / 6.176 GiB，基本不变；reserved 为 7.637 / 7.613 GiB。

42 组算子与 v5 逐位一致，18 组稀疏 packing、6 组 Graph 当前输入/动态 mask 检查通过，
memcheck/racecheck 无报错。结果限于 Thor SM110、未锁频的顺序诊断，未声称达到硬件峰值。
可复算报告见 `/home/yufeng/thunderworld/thor_cutlass_fused_20260923/report/report.md`。

## 执行方式

`sm110_repair_collective.cuh` 扩展 CUTLASS SM100/110 的 block-scaled collective：

1. 基础项沿用原 CUTLASS TMA 主循环。加载 warp 在基础 K 序列后追加选中的 A、W、交叉项。
2. 每项的代码数据通过 TMA 进入同一组流水槽。W0/A0 的已有 CUTLASS 尺度布局继续由 TMA 装载；
   残差 K16 尺度通过 cp.async 写入匹配的共享内存布局，不增加线上全量残差尺度转换。
3. 事务屏障在代码数据和尺度都就绪后发布槽。MMA warp 消费基础项及补项，始终使用同一个 FP32 Tensor Memory accumulator。
4. 仅最终 epilogue 缩放并输出 BF16。省去 v5 的 FP32 M×N 中间写回、读取、分配以及第二个 GEMM kernel。

阶段顺序保持 v5 的“全部基础项 → A 项（K 升序）→ W 项及其交叉项（CSR 顺序）”。
本实现没有把 A 项重新穿插到基础 K 循环中，以保持现有数值顺序。
生产者和消费者均读取当前 mask 的实际块数，兼容能量阈值导致的低于配额或空修复。
W1 仍使用离线紧凑支持并集及 wloc 映射，没有新增完整 W1 副本。

关键文件：

- `cutlass_sm110_fused.cu`：固定 tile 的 C ABI 和 CUTLASS 配置，空修复使用原稠密实例。
- `sm110_repair_collective.cuh`：修复阶段装载、TMA/cp.async 屏障及消费者计数。
- `cutlass_thor_fused_backend.py`：Graph 兼容包装器，只有一个 BF16 输出，无 global FP32 中间结果。
- `qualify_cutlass_thor.py --fused`：与 v5、原 Native 和独立解码参考对照。
- `create_cutlass_variant.py --fused`：只在完整数值、Graph 和 sanitizer 检查通过后生成独立版本。

修复 MMA、padding 和预算定义不变。kernel 次数只统计 GEMM kernel；量化、选择、attention、解码继续正常计入请求时间。
逻辑 global-memory 字节数来自成本账本，不等于实测 DRAM 流量。

## 构建、验证和运行

先确认 GPU 空闲并在任务边界暂停历史测速队列。所有实验使用 GPU 锁及进程检查。
输出使用新目录，保留旧二进制、资格检查及测量记录。

```bash
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh
thor_tools="$TW_THOR_ROOT/src/ThunderWorld/tools/thor_handoff"
thor_run="$TW_THOR_ROOT/comparisons/fused_retest"
thor_old="$TW_THOR_ROOT/comparisons/cutlass_upgrade_20260922"
mkdir -p "$thor_run"

python "$thor_tools/build_cutlass_sm110.py" --fused --output "$thor_run/fused.so"
python "$thor_tools/qualify_cutlass_thor.py" --fused \
  --library "$thor_run/fused.so" --reference-library "$thor_old/cutlass_sm110.so" \
  --output "$thor_run/qualified.json"
compute-sanitizer --tool memcheck --error-exitcode 86 python "$thor_tools/qualify_cutlass_thor.py" \
  --fused --small --check-only --library "$thor_run/fused.so" \
  --reference-library "$thor_old/cutlass_sm110.so" --output "$thor_run/memcheck.json" \
  > "$thor_run/memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode 87 python "$thor_tools/qualify_cutlass_thor.py" \
  --fused --small --check-only --library "$thor_run/fused.so" \
  --reference-library "$thor_old/cutlass_sm110.so" --output "$thor_run/racecheck.json" \
  > "$thor_run/racecheck.log" 2>&1

python "$thor_tools/create_cutlass_variant.py" --fused \
  --parent "$TW_THOR_ROOT/variants/native-row-major-v3" --name fused-retest \
  --library "$thor_run/fused.so" --qualification "$thor_run/qualified.json" \
  --memcheck "$thor_run/memcheck.json" --racecheck "$thor_run/racecheck.json" \
  --attention-qualification "$thor_old/attention.json" \
  --attention-memcheck "$thor_old/attention_memcheck.json" \
  --attention-racecheck "$thor_old/attention_racecheck.json"

for method in b1.5_protected_n b2_joint; do
  python "$thor_tools/tune_speed.py" measure --method "$method" \
    --variant "$TW_THOR_ROOT/variants/cutlass-attention-v5" \
    --cases 0 1 --repeats 3 --output "$thor_run/requests/v5/$method"
  python "$thor_tools/tune_speed.py" measure --method "$method" \
    --variant "$TW_THOR_ROOT/variants/fused-retest" \
    --cases 0 1 --repeats 3 --output "$thor_run/requests/v6/$method"
done

python "$thor_tools/fused_report.py" --measurements "$thor_run/requests" \
  --qualification "$thor_run/qualified.json" --output "$thor_run/report"
```

本轮原始目录为 `comparisons/cutlass_fused_20260922`：根目录保留初始版本，
`tma_scales/` 是改进基础尺度 TMA 装载后的最终资格结果，`requests/` 保存同轮 v5/v6 对照。
最终库为 `cutlass_sm110_fused_v2.so`，部署包中记录完整源码与二进制 SHA256。
本地镜像为 `/home/yufeng/thunderworld/thor_cutlass_fused_20260923`。

报告严格核对模型、输入、资产、策略、MMA 比率及完整请求哈希，列出中位数、P90、FPS、
allocated/reserved、GEMM kernel 次数和去除的中间流量。不会把算子收益乘到端到端结果上。

# Thor 内存优化及 Edge / Nano 尺寸扩展

实现位于 `memory_runtime/`。v7 隔离验证资产释放及紧凑 A1，
独立 `variants/cutlass-memory-v8` 再加入 CPU 上先替换投影的加载顺序优化。
保留 v6、全部量化资产及修复计划，桌面 PRO 6000 的冻结包不改动。

## 改动和保证

1. 离线资产逐层校验、mmap 加载及展开；不同时保留所有层的展开暂存。
2. 全部请求形状预热后调用 `runner.release_host_assets()`，释放 CPU 权重资产和可回收的
   glibc 空闲内存，保留执行所需元数据及 GPU 权重。仅在请求之外执行。
3. 部分 A 修复使用 `[M128 块数 × 配额, 128, K128]` 的紧凑残差缓冲，GPU 当前 mask
   每次构建地址映射。CUTLASS TMA 读取当前选中 tile，K 顺序、补项顺序和 BF16 epilogue 不变。
   不生成新的 selector MMA，不删除 base blocks，不更改 W0/W1。
4. 空/全修复保持原布局。若尾块 padding 加上索引使紧凑表示反而更大，也保留原布局。
5. 先在 CPU 模型中替换生成投影，再将剩余模型搬迁到 GPU，避免完整 BF16 生成权重
   先驻留 GPU 后又被替换。FP8 同样采用此通用加载优化，比较没有专门抬高基线峰值。
6. 释放后的实例仅接受已预热请求模板；新请求模板和资产导出会明确失败，使用新实例重新准备。
   当前模板键包括 prompt、图像路径、尺寸和步数，**不只是 tensor shape**。
   需要多个模板时，应在释放前分别 warmup / warm_request。
   支持已预热形状的 eager/Graph 执行；没有推理中读盘、重新量化或 CPU mask 回读。

内核资格检查：112 组数值对比与 v6 逐位一致，14 组当前输入/选择、空 mask、全零和输出
生命周期 Graph 检查。覆盖 M129、M416、M960、M1920、M2160、M2880、M3840，
Edge / Nano FFN 上下投影和 QKV 维度。两种尾块形状各八组 memcheck / racecheck 通过。
原生 attention 数值实现未变；权重修复的支持并集与已有压缩存储不变。

## 复测设计

每个点是新进程，固定 RBench 场景、种子 `301017 / 301018 / 301031`，35 步、CFG 6、
flow shift 3、batch 1、串行、Graph、debug off。只测速度/内存，不进行视频质量选参。
各方法外围融合开关一致：保留 AR cache / first-frame conditioning，额外融合和 codec Graph 关闭。

| 宽×高 | 帧数 | 视觉 token 数 |
|---|---:|---:|
| 320×192 | 61 | 960 |
| 320×192 | 125 | 1920 |
| 320×192 | 189 | 2880 |
| 480×288 | 61 | 2160 |
| 640×384 | 61 | 3840 |

Edge、Nano 各比较 BF16、原生 FP8＋BF16 attention、W4A4＋FP4 attention、1.5×、2×。
原形状另外比较 v6 和关闭紧凑 A1 的中间版本，隔离缓冲改动；
完整 v7 原形状结果作为 v8 加载顺序优化的独立对照。
合计 60 个配置点、最多 180 个测量请求，另有预热及释放前后一致性检查。

新形状继续检查整请求 MMA 上界。原策略若不再满足预算，保存不可行原因，不减少修复、
抬高预算或更换输出精度掩盖失败。资源不足则保存进度并暂停。

两模型现有 1.5× 策略保存的是各 head 的**整数 attention tile 配额**；扩大序列后仍使用这些
固定配额，因此覆盖率会下降，不能将此曲线解读为相同修复比例或已验证的大尺寸质量。
2× 策略使用全量三项 attention 修复，序列变大后 attention 在总 MMA 中的占比上升，
原计划可能超过 2×。报告逐点列出实际 MMA 比例及预算失败。

## 内存口径

- 请求的 Torch allocated / reserved 峰值，以及加载、建图峰值，分别保存。
- 全过程 Torch 峰值取加载、建图、完整预热、测量请求各区间峰值的最大值。
  loader 变化把搬迁时间从内部 `load_seconds` 移到 `offline_prepare_seconds`，
  冷准备成本应比较外层 `prepare_seconds`，不可单看某个被重新划分的内部阶段。
- 就绪时 RSS 是释放和 GC 后、测量循环前的边界采样。
- 外部驱动每 250 ms 读取进程 RSS / VmHWM 和系统 MemAvailable；保留原始 JSONL。
  `measure` 阶段包含请求和结果记录间隔，采样极值不是严格的整机瞬时峰值。
- 三类指标有重叠，不能相加。降低请求峰值不代表冷启动也有相同收益。
- 旧成本清单的 `residual_allocation_bytes` 对应 v6 完整平面。
  新结果中的 `packing_footprints` 给出每个实际准备算子的紧凑容量和地址映射占用；
  算子缓冲可共享 Graph 池，这些逐算子容量也不能相加为模型驻留内存。
- 外部采样可能影响主机执行；延迟和 FPS 为此次内存诊断的附带数据，不更新正式性能榜。

## 运行

```bash
source /home/chenhan/thunderworld/src/ThunderWorld/tools/thor_handoff/env.sh
mem_tools="$TW_THOR_ROOT/src/ThunderWorld/tools/thor_handoff/memory_runtime"
mem_run="$TW_THOR_ROOT/comparisons/memory_retest"

# 已完成 kernel 资格检查的 v7 为父版本；创建新目录，仅移动一行加载操作。
# 此操作不重编译或改变计算内核。完整请求 hash 对照单独存入结果。
python "$mem_tools/loader_variant.py" --name cutlass-memory-v8
python "$mem_tools/campaign.py" prepare --variant cutlass-memory-v8 --output "$mem_run"
python "$mem_tools/campaign.py" run --output "$mem_run"
python "$mem_tools/campaign.py" report --output "$mem_run"
```

`run` 跳过已完成点和明确的预算不可行点，其他不完整证据保留并要求检查。
新设备或新库先使用 `build.py`、`qualify.py` 和 compute-sanitizer 验证，再由 `seal.py`
固定凭据。每个 manifest 同时绑定模型、策略、资产、源码、库和 worker 的摘要。

已有相同 worker、种子和冻结模型的控制结果，可在 `prepare` 时提供
`--reuse-controls-from /path/to/previous/campaign`。仅复用 v6 控制点，逐项验证摘要；
所有新版本点重新执行，并对已有同条件完整请求验证输出 hash。失败即暂停队列。
`loader-equivalence.json` 保存逐点对照来源及其摘要；同一个点不会被改名冒充新运行。

报告在 CPU 上由原始 JSON 重新生成，不使用共享 GPU 耗时选型：

```bash
python "$mem_tools/analyze.py" --campaign "$mem_run" \
  --previous-campaign /path/to/v7/campaign --output "$mem_run/analysis" --plot
python "$mem_tools/shape_budget_audit.py" --reference /path/to/v7/campaign \
  --campaign "$mem_run" --output "$mem_run/analysis/shape-budget.json"
```

`shape_budget_audit.py` 使用原始实际算子形状和既有 CPU 预算函数，预测固定 prompt 下的
新尺寸成本，并逐项核对实际预热账本（包括被预算门槛拒绝的请求）。预测不代替 launch
之前的实际预算检查。绘图只依赖 NumPy / matplotlib，不应修改推理环境来安装绘图依赖。

常规使用可以在导入包前将已验证 variant 加入 `PYTHONPATH`，使用既有 CosmosRunner：

```python
runner = CosmosRunner(model_path, config, assets=asset_path,
                      ar_cache=True, first_frame=True).prepare()
runner.warmup(case, 35)
runner.warm_request(case, 35)
runner.release_host_assets()  # 准备阶段末尾调用一次
signals, receipt = runner.infer(case, seed, 35)
```

紧凑 A1 是独立可选消融。`worker.py --mode release` 保留 v8 的资产/加载修复，
使用原完整 A1 缓冲；`--mode compact` 再启用紧凑 A1。局部缓冲容量更小不保证 Graph
池峰值或延迟更低，应按最终匹配结果选择，不默认宣称所有配置都有额外收益。

原始 v7 数据：`/home/chenhan/thunderworld/comparisons/memory_20260923/`。
加载优化 v8 数据：`/home/chenhan/thunderworld/comparisons/memory_20260923_v8/`。
本机镜像位于 `/home/yufeng/thunderworld/thor_memory_20260923/`。

内存定义参考 [PyTorch 2.11 CUDA memory management](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)
及 [NVIDIA Jetson tegrastats](https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)。

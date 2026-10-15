# Thor 同条件精度格式速度对照

本工具回答“FP4 的计算优势为什么未完全转化为请求加速”，保持现有量化、修复、模型和输入定义。
它只做速度与算子正确性检查，不替代生成质量评测。使用已完成资格检查的 `cutlass-mainloop-v6`。
以下历史数据和完整归因针对 Edge。Nano 使用同一入口的 `prepare --model nano`，
模型资产单独冻结，并可用 `--rounds 1` 进行首轮诊断；见 [NANO_PERFORMANCE.md](NANO_PERFORMANCE.md)。
`standard_analysis.py` 的 11,760 次投影归因固定于 Edge，不能直接套用到 Nano。

## 三层口径

1. **完整请求**：BF16、FP8、普通 W4A4、1.5×、2×，以及单独的 W4A4＋BF16 attention 归因组。
   同一 Edge revision，320×192、61 帧、35 步、CFG 6、flow shift 3，batch=1、单请求、Transformer Graph、debug off。
   每种配置独立进程准备与预热；五轮固定随机顺序，每轮两个输入各三个相同种子。
   计时包含编码、生成、解码和 CPU 输出；加载、离线准备、建图、保存 MP4 均不混入暖请求。
2. **GPU 构成**：同一个请求另跑 Nsight，包含 CUDA Graph 内部节点。
   kernel 累计时间用于定位工作量，不能替代无探针请求时间，也不代表 SM 或 Tensor Core 利用率。
3. **相同形状算子**：原 cuBLAS BF16、原 vLLM CUTLASS FP8、当前 FP4 及已安装库的 NVFP4 API。
   分离预打包 GEMM 与从当前 BF16 输入开始的在线 packing；权重离线准备。
   区分同 buffer 重放和轮换工作集，记录工作集是否超过两倍 L2、FLOPs、尾块、实际量化定义与解码参考误差。

FP8 固定为 W8A8：权重 per-tensor E4M3、激活在线 per-token E4M3、原生 CUTLASS GEMM、BF16 attention。
不重新调优 BF16 或 FP8。替换为 BF16 attention 的 FP4 组改变了算术，只能用于归因，不能代替联合 FP4 结果。

运行配置必须分别列出：普通 FP4、FP8、BF16 的外围开关一致；冻结修复版额外启用
`fuse_calibrated`、`fuse_pointwise`、`static_rotary`、`codec_graph`、`fuse_norm`。
主表对比现有完整实现，不能将修复版的所有收益归于 FP4 算术。
`runtime_control.py` 在相同资产、预算和 Transformer Graph 下成组关闭上述选项，
对两档修复各运行完整实现/统一外围开关两个配置，随机排序，每配置六请求。
这是一轮机制归因，不混入五轮统计，也不借此修改基线。

所有复测使用同一 MAXN 功耗模式。当前账号没有免密 root，因此没有修改系统时钟。
五轮随机交错降低顺序偏差，保留每轮前后温度/频率记录；结果不标注为锁频硬件上限、MLPerf 或官方峰值跑分。

## 复测命令

GPU 必须空闲；所有 worker 都检查 `gpu-test.lock` 和其他计算进程。
输出目录必须是新目录。中断后 `run` 只跳过通过完整校验的已完成配置，不覆盖失败记录。

```bash
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh
thor_tools="$TW_THOR_ROOT/src/ThunderWorld/tools/thor_handoff"
thor_run="$TW_THOR_ROOT/comparisons/standard_retest"

python "$thor_tools/standard_comparison.py" prepare --output "$thor_run"
python "$thor_tools/standard_comparison.py" run --output "$thor_run"
# Profiling is deliberately performed after all uninstrumented measurements.
python "$thor_tools/standard_comparison.py" profile --output "$thor_run"
python "$thor_tools/precision_probe.py" --output "$thor_run/precision_probe.json"
python "$thor_tools/runtime_control.py" --output "$thor_run/runtime-control"
python "$thor_tools/standard_comparison.py" report --output "$thor_run"
python "$thor_tools/standard_analysis.py" --campaign "$thor_run" \
  --probe "$thor_run/precision_probe.json" \
  --runtime-control "$thor_run/runtime-control/summary.json" --output "$thor_run/analysis"
```

报告核对 model、输入、seed、Graph、debug、资产与生成投影范围；同方法跨轮次还核对源码和配置。
仪器化结果禁止进入延迟汇总。误差区间按“轮”重采样，不能把 30 次重复的两个场景误当作 30 个独立场景。
Nsight 临时目录固定在本 campaign 的 `profiles/tmp/`，避免共享 `/tmp/nvidia` 的所有权冲突；
失败启动证据保留在 `failed-nsight-startup/`，没有混入有效测量。

## 峰值与标准算子测试

[NVIDIA Thor Technical Brief 表 2](https://dam-cdn.nvd.orangelogic.com/AssetLink/v686u306w26ftuk863xjeimilimtx55e.pdf)
区分 FP4 sparse 2070、FP4 dense 1035、FP8 dense 517、FP16 dense 258 TFLOPS。
本项目的软件 tile 修复没有使用 2:4 稀疏 MMA，不能对照 2070 TFLOPS 宣称低利用率或理论速度。
即使某一 GEMM 的稠密 FP4 吞吐是 16 位的 4 倍，r 倍修复计算后也只剩该段的理想 4/r 倍；
这不包含在线量化、尺度处理、选择、softmax、P 量化和其他算子。

微基准采用厂商已安装库的同输入/同形状接口，并记录原始时间；不是完整 CUTLASS profiler kernel 搜索。
如需另测厂商 kernel 峰值，可按 [CUTLASS Profiler 文档](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/profiler.html)
在独立目录为 `110a` 编译选定的 **dense** `e2m1` / `e4m3` / `bf16` kernel，固定 FP32 accumulation、BF16 output、
alpha=1、beta=0、batch_count=1、相同 M/N/K、warmup/iterations、Graph 和 workspace/cache 条件。
不要把 `sptensorop` 的 2:4 测试混入本项目的 dense GEMM 对照，也不要把大方阵结果标注为 Cosmos 请求吞吐。

当前请求的生成投影主要是 M960；M129/M416 测尾块/其他范围，4096/8192 方阵仅作大矩阵参考。
工作集轮换用于观察缓存敏感性，不替代 Nsight Compute 的真实 DRAM/SM 计数器。
此设备 L2 为 32 MiB，已通过 PyTorch 属性和 CUDA runtime 查询交叉验证。
hot 仅表示相同 buffer 重放；8192 方阵一份工作集已大于 L2，不能叫作全缓存命中。
当前算子测试保留 BF16 reduced-precision-reduction 原设置，分别记录量化语义；
独立解码校验检查每形状前 16 行，当前与库 FP4 比较覆盖全部输出。

## 本轮文件

- 远端：`comparisons/standard_fp4_20260923/`
- 本地镜像：`/home/yufeng/thunderworld/thor_standard_fp4_20260923/`
- `manifest.json` 固定顺序、种子、版本；`runs/` 为无探针测量，`profiles/` 为独立诊断。
- `summary.json` / `report.md` 为五轮速度对照，`precision_probe.json` 为两种缓存条件下的算子结果。
- `analysis/analysis.json` / `analysis.md` 给出 GPU 构成、组件预算、配对区间与口径说明。
- `runtime-control/` 为外围执行开关归因；`sources_at_measurement/` 保留五轮和算子诊断使用的原始脚本及 SHA256。
- 五轮完成后增加了 `--common-runtime`；原 manifest 的 worker hash 因此只匹配归档脚本。
  重复旧入口时可设置 `PYTHONPATH="$thor_tools"`，运行归档的 `standard_comparison.py`；新复测使用新目录和新 manifest。
- 原始 `.nsys-rep` / `.sqlite` 留在 Thor，JSON/CSV/日志与 SHA256 已同步本机；本机无需复制大型 trace 才能复算报告。

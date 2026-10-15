# 5090 CUTLASS 复测交接

2026-09-14 本机复测与进一步优化：[5090 报告、直接 scale packing 开关和复现](../../docs/CUTLASS_5090_FOLLOWUP_20260914.md)。

GitHub同步入口：[下载交接包并校验](github_handoff/README.md)。

这是可运行的实验后端。PRO 6000 结果不代表5090性能；N64/N128尚不是已证明最优的内核集合。Attention继续使用现有流式FP4路径，本次CUTLASS融合覆盖生成投影/FFN的GEMM与稀疏修复。

## 包含什么

- `packages/fp4-repair`：统一推理包、CUTLASS/CuTe源代码及测试。
- `recipes/i2v_b125`、`recipes/i2v_b150`：全部168个生成投影的离线校准配方。下载原模型后重建W0/W1，不在推理期间量化权重；重建逐字节核对原资产，失败直接停止。
- `policies/`：冻结的1.25/1.5预算计划，含全部层、四阶段、CFG分支和attention配置。
- `tools/joint_inference`：同条件推理、质量评估、独占GPU调优工具及输入图片。模型、第三方源码、编译缓存和PRO 6000调优表均不打包。
- 可选 `cutlass-5090-panels.tar.gz`：真实M960/M316输入的四阶段算子面板。2026-09-14接收端核对确认：此前M416描述有误，Policy实际张量和manifest均为M316；M416另有正确性测试。解压得到`panels/`；无需下载模型即可调试算子。面板含校准数据，不是最终质量测试集。

先检查`FILES.json`及外层`archives.json`的SHA256。保存所有新测量，勿覆盖历史结果。

## 环境与离线构建

本机验证环境：Linux、CUDA toolkit13.2、SM120、CUTLASS4.5.0（FlashInfer0.6.15.post1随包头文件）、PyTorch2.11.0+cu130、Triton3.6.0。CUTLASS头文件也可通过`--cutlass /path/to/cutlass`显式指定。构建写入源码/头文件/二进制哈希及编译器信息；不在warmup或Graph replay时编译C++。

```bash
python -m pip install -e packages/fp4-repair
python -m thunderworld_fp4.cutlass.build --tile-n 128 --output build/repair_n128.so
python -m thunderworld_fp4.cutlass.build --tile-n 64 --output build/repair_n64.so
export TW_CUTLASS_LIBRARY="$PWD/build/repair_n64.so"
export TW_CUTLASS_OTHER_LIBRARY="$PWD/build/repair_n128.so"
python experiments/cutlass_fused_repair/run_tests.py
compute-sanitizer --tool memcheck --error-exitcode 42 python experiments/cutlass_fused_repair/run_tests.py
compute-sanitizer --tool racecheck --error-exitcode 42 python experiments/cutlass_fused_repair/run_tests.py
```

Python依赖具体版本见`tools/joint_inference/environment-tested.json`；安装包本身不会隐式下载GPU框架。完整Cosmos运行需要该文件中冻结的diffusers提交与其依赖。5090只有GPU0时，自动资源选择会使用GPU0。测试默认获取共享校准锁、要求至少12 GiB空闲显存；OOM停止，不能缩短序列或改变精度继续。

## 算子复测（无需模型）

```bash
python experiments/cutlass_fused_repair/retest.py --capsule panels/i2v_b125 --library build/repair_n128.so --output results_i2v125_n128.json
python experiments/cutlass_fused_repair/retest.py --capsule panels/i2v_b125 --library build/repair_n64.so --output results_i2v125_n64.json
python experiments/cutlass_fused_repair/retest.py --capsule panels/i2v_b150 --library build/repair_n64.so --output results_i2v150_n64.json
python experiments/cutlass_fused_repair/retest.py --capsule panels/policy_b125 --library build/repair_n64.so --output results_policy125_n64.json
```

默认共享诊断；独占时加`--resource-mode exclusive --device 0`，工具会检查其他计算进程。它仍是算子诊断，不进入正式端到端排行榜。计时包含当前输入packing、selector、scale转换和全部修复；同时保留prepacked结果用于归因。报告记录设备、修复支持、编译二进制、实际CTA数量，不能只比较GEMM时间。

可用Nsight检查`RepairCollective`：

```bash
ncu --set full --clock-control none --cache-control none --kernel-name-base demangled --kernel-name 'regex:.*RepairCollective.*' --launch-count 1 --export kernel_diagnostic python experiments/cutlass_fused_repair/probe.py --library build/repair_n64.so --capture panels/i2v_b125/capture --config panels/i2v_b125/config.json --assets panels/i2v_b125/assets --correctness-only --output profiled_panel.json
```

观察Tensor Core active/elapsed两种口径、waves/SM、registers、shared memory、eligible warps及memory stalls。Nsight运行不作延迟对照。

## 完整推理：下载模型后重建离线资产

固定模型：`nvidia/Cosmos3-Edge`，revision `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`。

```bash
hf download nvidia/Cosmos3-Edge --revision a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba --local-dir models/Cosmos3-Edge
python experiments/cutlass_fused_repair/weight_recipe.py rebuild --recipe recipes/i2v_b125 --model models/Cosmos3-Edge --output rebuilt/i2v_b125
python experiments/cutlass_fused_repair/weight_recipe.py rebuild --recipe recipes/i2v_b150 --model models/Cosmos3-Edge --output rebuilt/i2v_b150
python experiments/cutlass_fused_repair/configure.py --policy policies/i2v_b125 --library build/repair_n128.so --small-library build/repair_n64.so --output configs/i2v_b125.json
python -m thunderworld_fp4.cosmos.cli infer --model models/Cosmos3-Edge --config configs/i2v_b125.json --assets rebuilt/i2v_b125 --case tools/joint_inference/cases/robot_observe.json --cohort tools/joint_inference/cohorts/edge_i2v_compute_v1.json --cohort-role trajectory --seeds 7103 7104 --steps 35 --output runs/i2v_b125
```

重建无outlier资产使用原校准编码器；其他outlier方案必须提供显式拆分支持，工具不会猜测。当前配方不需要重新校准，不需要传递全层激活捕获数据。1.5档替换对应路径。`debug=off`、Graph为默认值。默认小grid选择N64的依据是CTA数低于设备SM数，并非共享GPU竞速结论。

## 独占5090调优与正式性能复测

只有独占GPU时使用以下命令。先把配置生成到新路径并指定`--resource-mode exclusive`；两个库同时提供时，`--tune`比较N128/N64，要求当前输入输出逐位一致，再以包含packing/selector的完整算子时间选择，重新测量确认。设备/代码/二进制变化都会使旧表失效。

```bash
python experiments/cutlass_fused_repair/configure.py --policy policies/i2v_b125 --library build/repair_n128.so --small-library build/repair_n64.so --resource-mode exclusive --output configs/i2v_b125_exclusive.json
CUDA_VISIBLE_DEVICES=0 python -m thunderworld_fp4.cosmos.cli warmup --model models/Cosmos3-Edge --config configs/i2v_b125_exclusive.json --assets rebuilt/i2v_b125 --case tools/joint_inference/cases/robot_observe.json --tune --output runs/tuned5090_i2v_b125
CUDA_VISIBLE_DEVICES=0 python -m thunderworld_fp4.cosmos.cli benchmark --model models/Cosmos3-Edge --config configs/i2v_b125_exclusive.json --assets rebuilt/i2v_b125 --case tools/joint_inference/cases/robot_observe.json --tuning runs/tuned5090_i2v_b125/tuning.json --seeds 7103 7104 --output runs/benchmark5090_i2v_b125
```

该命令是冒烟复测，不替代5轮×3种子的正式交错对照。完整报告应另跑原Triton FP4、FP8、BF16，冻结相同模型/输入/35步/CFG/Graph；质量用development/validation选参，最终种子不得用于内核或修复选择。带探针结果、共享负载结果不得进入排行榜。

本机两条development轨迹显示CUTLASS重排累加相对原实现存在小幅质量变化；不能把上述功能验证写成“超过FP8”或“质量等价”。新N64/N128之间另有逐位等价测试。

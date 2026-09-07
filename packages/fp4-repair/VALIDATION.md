# Validation — 2026-09-07

本次验证针对可安装包本身。16 项 CPU 配置测试、13 项 CUDA 测试通过；CUDA memcheck 0 errors，racecheck 0 errors / 0 warnings。CUDA 测试运行在物理 GPU 0，RTX PRO 6000 Blackwell Server Edition，PyTorch 2.11.0+cu130、Triton 3.6.0。

验证覆盖独立 FFN/core/projection 配置、逐 head 配额、现有 atlas 到逐层规则的映射、精确 block mask、128 边界尾块和空 split 防护、稀疏和全量修复、GQA、非连续 sequence stride、跨 stream 图缓存、变化的输入/mask、返回输出的存储独立性。

Attention oracle 解码当前包生成的 Q/K/V FP4，再以独立 PyTorch FP32 计算在线 softmax 和 P 的参考量化。相对 L2 阈值 0.012，涵盖 terms=2/3/4、无修复和部分修复。FFN oracle 解码 A/W，逐 M128/N128/K128 mask 累加残差，阈值 0.004；测试包含 prescale、bias、M 尾块和零输入。两类 oracle 检查计算与所选 FP4 表示的一致性，不代替 BF16 模型质量评估。

```bash
# 仅在获准的 SM120 GPU 上运行
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/fp4-repair/src \
  python -m pytest packages/fp4-repair/tests -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/fp4-repair/src \
  compute-sanitizer --tool memcheck --error-exitcode 1 \
  python packages/fp4-repair/tests/sanitizer_smoke.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/fp4-repair/src \
  compute-sanitizer --tool racecheck --error-exitcode 1 \
  python packages/fp4-repair/tests/sanitizer_smoke.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/fp4-repair/src \
  python packages/fp4-repair/examples/benchmark.py --output /tmp/fp4-benchmark.json
```

Wheel 使用 `pip wheel --no-build-isolation --no-deps` 构建，并在临时目录安装验证；无需导入本地实验路径，所有 preset JSON 随 wheel 分发。配置导入可在未导入 torch 的环境运行。

## 当前性能边界

[原始 benchmark](examples/benchmark_sm120.json) 是共享 GPU 上的一次算子级测量，包含在线激活量化、selector、GEMM 和 attention softmax，采用 `do_bench_cudagraph`；不含离线权重量化，不含 `FP4Attention` 外包装的输入 copy/输出 clone。不是独占设备性能或完整模型延迟。

| Attention，M128/N1024/Hq8/Hkv2/D128 | ms |
|---|---:|
| BF16 Flash SDPA | 0.02338 |
| FP4 base | 0.04439 |
| 双侧 50% | 0.06024 |
| 双侧 75% | 0.06687 |
| 单侧 full ACT2 | 0.04564 |

| Linear，M128/N1024/K1024 | ms |
|---|---:|
| BF16 | 0.01571 |
| FP4 base | 0.02181 |
| ACT2 50% | 0.04047 |
| ACT2 full | 0.02887 |

编译 PTX 中两类 GEMM 均检测到 `mma.sync.aligned` 和 `e2m1`。当前配置接口确实驱动原生 FP4 Tensor Core 计算，但这些小形状上仍未超过 BF16。部分 FFN 修复反而比 full 慢，说明 selector/分支/producer 开销可能抵消减少的乘积；需独占设备、多形状 profiling 才能确定各项归因。不能声称本包已打满 Tensor Core 或获得端到端加速。

下一步性能工作应围绕更大的 GEMM tile、producer 开销、选择计划复用以及已验证 CUTLASS producer 的可移植集成；预算越小不必然越快。当前 `query_tile` 参数不能突破已验证的 SM120 布局边界。

## 历史质量数据的出处

[evidence/historical_act2_vs_dual50.json](evidence/historical_act2_vs_dual50.json) 和 [evidence/historical_consistency.json](evidence/historical_consistency.json) 保留旧实验摘要及源文件 SHA256，作为 preset 选择依据。原始模型、张量和视频不随包分发；这些摘要不构成可独立重跑的整模型资产。它们使用旧 CUTLASS projection 后端，不能归于本次新增 Triton FFN。场景已用于诊断，并非 held-out 测试。

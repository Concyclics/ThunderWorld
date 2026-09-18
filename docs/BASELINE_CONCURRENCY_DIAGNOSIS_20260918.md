# ARCQuant / SharQ 单路吞吐最高的实现诊断

本次只审计已完成测量和当前源码，没有修改 baseline、重新计时或执行 GPU profiler。结论适用于本次固定版本的 Cosmos 适配和并发入口；不能视为算法固有吞吐上限。

数据来自 [throughput.csv](../results/fp4_baseline_sweep_20260918/throughput.csv)。PRO 6000，Edge，35 steps，CFG 6，192×320，每视频 61 帧，包含 AR 预填充、解码及 CPU 输出转换。以下是 RBench 固定输入的诊断结果，performance_eligible=false。B 是独立模型副本的请求并发数，不是合并 GEMM 的 batch；B>1 按约定仅测速度。

| 方法 | B1 输出 FPS | B4 总输出 FPS | 吞吐变化 |
|---|---:|---:|---:|
| ARCQuant 50% | 25.71 | 15.76 | −38.72% |
| ARCQuant 100% | 24.42 | 17.58 | −28.00% |
| SharQ 现成融合模式 | 25.57 | 11.34 | −55.65% |
| NVFP4 AWQ-lite / Marlin | 34.27 | 40.36 | +17.78% |
| 我们的 1.5× | 34.81 | 52.17 | +49.87% |

只有 ARC/SharQ 在本轮已测档位中单路最佳。AWQ-lite、BF16、FP8 均在 B4 达到各自最高测得吞吐。PAI 固定输入有相同趋势。

1. **已确认：ARC/SharQ 使用同一条默认 CUDA stream。** [baseline_runtime.py](../tools/contribution_study/baseline_runtime.py) 的 `default_stream_only` 保留官方内核的 stream 约束；[baseline_batch.py](../tools/contribution_study/baseline_batch.py) 将每个 ARC/SharQ runner 的请求提交到 `torch.cuda.default_stream()`。当前上游量化 kernel 没有传入 stream，CUTLASS 调用也没有传入 PyTorch current stream。其他方法每个请求使用独立 stream。同一 stream 中的工作按序执行，因此这些 CPU 请求线程没有形成对应的跨请求 GPU kernel 并行，也没有增大单个 GEMM 的 M。[NVIDIA CUDA stream 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

2. **已确认：ARC 有逐线性层的 GPU→CPU 标量依赖。** [external.py](../tools/contribution_study/external.py) 的 `ARCLinear.forward` 将当前输入尺度乘积转成 `float(factor*self.factor)`，再传给官方 GEMM。这个 CUDA 标量必须先在 CPU 可用；适配器因此明确拒绝 Transformer Graph capture。该路径包含在线量化和原生 FP4 GEMM，不能归为 W4A16 反量化计算。权重量化在准备阶段完成。[PyTorch CUDA Graph 限制](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)

3. **已确认：SharQ 当前路径不同于 ARC。** 固定版本 `3cc584fe48e32e59c8b4e37323470dcdd606bb11` 的 `model/qLinearLayer.py` 使用 GPU `output_scale` 张量及 `sparse_matmul_tensor_scale`、`matmul_accum_tensor_scale`；没有 ARC 的 Python `float` 回读。当前执行仍是 eager，在线稀疏／残差 packing 在 `kernels/src/bindings.cpp` 中为五个输出缓冲区执行 `torch::zeros`，之后运行稀疏 FP4 GEMM 和稠密 FP4 残差 GEMM。缓存分配不等于每次都执行 cudaMalloc；不能把未测量的分配器耗时当作主因。

4. **存在但未量化的争用来源。** 当前 pybind 绑定没有显式释放 GIL；Pybind 默认不释放 GIL，可能限制多个 Python 线程的提交。测试入口另有所有方法共同使用的 SDPA 主机 dispatch/context 锁，以保护进程级后端标志。独立模型副本也不提供共享权重的真 batch 复用。这些影响与 eager/Graph 路径可能存在交互，尚无时间线或消融来分离它们的占比。[pybind11 GIL 说明](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)

“反量化过重”不能统一解释这组结果：AWQ-lite 使用融合反量化的 Marlin W4A16 路径，B4 吞吐仍提升 17.78%；ARC/SharQ 的上述 GEMM 则直接使用 FP4。反量化的绝对成本与并发时吞吐下降是两个不同的问题。

如果只有相同工作量的 GPU 计算串行化、而没有新增开销，B4 总耗时应接近 B1 的四倍，总吞吐接近持平。SharQ 实测 B1 为 2.386 秒，B4 整组为 21.519 秒，是 B1 的 9.02 倍；这提示多请求执行存在额外开销，不能解读为“单路已打满 Tensor Core”。同流执行本身只说明不能利用跨流重叠，并不能单独解释全部下降。

可支持的主张是：**在当前可运行的固定实现中，我们的 Graph 与多流执行框架有更好的请求并发吞吐。** 不能把全部差距归因于量化数学、反量化、GIL 或 Tensor Core 饱和。若要分配各项耗时，需要另做 B1/B4 的 CPU/CUDA 时间线诊断，分别统计同步等待、kernel 间隙、packing/清零、GEMM、attention、解码及锁等待；带 profiler 的结果单独保存，不替代现有 debug-off 吞吐数据。

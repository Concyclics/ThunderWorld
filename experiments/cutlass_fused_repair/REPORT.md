# PRO 6000 CUTLASS 修复融合验证

实现已接入统一推理包，但尚未证明达到硬件吞吐上限，也未满足“质量超过FP8”的验收。所有本文耗时来自共享GPU诊断；没有更新性能排行榜，没有5090实测，没有将算子倍数当作端到端倍数。

## 实现与验证

- 单CUTLASS GEMM完成基础W4A4、选中A1W0、A0W1及可选交叉项，共享FP32寄存器，最后一次BF16写回；无FP32全局中间矩阵。
- 稠密TMA流水；稀疏修复采用`cp.async`共享内存加载和CuTe `ldmatrix`。消费者有独立修复共享内存，不覆盖生产者的下一tile。
- 支持N128/N64两个编译变体。修复支持仍为128×128；N64只拆分N方向执行，支持N为128倍数时不会增加MMA。
- 无修复独立特化；全W1规则寻址、部分W1共享CSR支持；当前输入的A量化/选择保持不变。
- 离线库、当前设备、形状、修复策略及量化语义共同绑定调优表。只有独占`--tune`可测量选择，且N64/N128必须逐位一致。默认Graph，C++编译在准备前显式完成。
- 173项测试通过；最终两内核memcheck 0 errors，racecheck 0 errors/0 warnings。覆盖M129/M416/M960/M1024、空/全/部分/交叉修复、共享W1、零输入、Graph输入与mask更新，以及profile切换。
- SASS存在`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`，确实执行原生FP4 Tensor Core指令。

Attention的QK/PV/softmax仍由已有流式FP4内核执行；本轮没有把完整attention重写为CUTLASS。激活packing及selector仍是Triton，CUTLASS scale排列转换仍有单独一遍读写，不能称为全部算子已融合。

## 算子诊断

以下为layer0、phase0、branch0，冻结1.25预算。倍数=现有Triton完整算子耗时/CUTLASS完整算子耗时，均包含当前输入packing、选择和scale转换。N128与N64在不同轮测量，表格用于诊断，不用于共享模式下自动选型。

| 输入/算子 | N128 | N64 |
|---|---:|---:|
| I2V M960 Q投影 | 1.060× | 1.052× |
| I2V M960 K投影 | 1.051× | 1.227× |
| I2V M960 FFN up | 1.261× | 1.110× |
| I2V M960 FFN down | 1.695× | 1.266× |
| Policy M416 Q投影 | 1.332× | 1.500× |
| Policy M416 K投影 | 0.970× | 1.147× |
| Policy M416 FFN up | 1.102× | 1.083× |
| Policy M416 FFN down | 1.843× | 2.378× |

这些FFN案例在phase0都是零修复，不能据此宣传“FFN稀疏修复加速2.378倍”。真实修复Q/K及四阶段完整数据见[原始结果](results/summary.json)。N128的1.5预算面板最差约0.903×，仍有回退；不能统一替换后宣称所有算子更快。

第一版直接global→register的融合虽然正确，修复投影只有约0.37–0.40×，失败数据已保留。共享内存改版解决了主要散乱加载开销，但并未消除所有瓶颈。

## 为什么仍没有打满

对M960/N2048/K2048、带W修复的Q投影收集一次Nsight诊断：

| 指标 | N128 | N64 |
|---|---:|---:|
| CTA waves/SM | 0.68 | 1.36 |
| registers/thread | 168 | 168 |
| shared memory/block | 84.992 KB | 82.944 KB |
| Tensor active / active cycles | 59.47% | 38.44% |
| Tensor active / elapsed cycles | 21.10% | 20.50% |
| eligible warps / active cycle | 0.253 | 0.244 |

N64增加了CTA覆盖，但该Q案例利用率和时间未改善。高寄存器/共享内存占用、稀疏补项的同步与加载，以及更多CTA导致的重复读取仍限制吞吐。上述计数器来自共享诊断、未控制时钟/缓存，不能用来宣称峰值效率。原始CSV及摘要保留于results。

下一步优先方向：

1. 保留N128/N64按实际形状和修复预算选型；独占设备上比较完整算子，不能按Tensor active百分比单独选。
2. A1W0尽量插入稠密K主循环，复用已有W0寄存器/共享数据；这也更接近原Triton的累加顺序。当前实现是在dense结束后执行补项，会重新读取W0。
3. 使修复预取与当前tile MMA重叠，降低共享存储和寄存器存活量；任何流水改动重新跑racecheck及完整轨迹。
4. 在激活base packing中直接生成CUTLASS scale布局，减少额外launch/scale流量；weight双布局目前增加84 MiB常驻存储，后续可统一布局。
5. QK/PV继续保持流式softmax设计，单独评估packing/P量化/修复路径。此次线性优化不能解决attention所有开销。

## 完整35步质量与内存

Edge I2V development `robot_observe`，同一模型、输入、35steps、CFG、Graph，种子7103–7104。以下均相对BF16；两种CUTLASS tile之间逐位一致，但CUTLASS与Triton累加顺序不同。

| 配置 | 平均LPIPS↓ | 平均SSIM↑ | 视频relative L2↓ |
|---|---:|---:|---:|
| FP8 | 0.120626 | 0.814926 | 0.231813 |
| Triton 1.25 | 0.184659 | 0.655439 | 0.294614 |
| CUTLASS 1.25 | 0.189465 | 0.647350 | 0.296378 |
| Triton 1.5 | 0.165953 | 0.713952 | 0.266458 |
| CUTLASS 1.5 | 0.167313 | 0.715905 | 0.273915 |

CUTLASS 1.25的LPIPS比原实现高约2.60%；1.5高约0.82%，且L2退化。不能声明质量等价，更不能声明超过FP8。只有两条development轨迹，不作最终配对置信区间或泛化结论；没有使用最终种子选内核。

最终N64/N128联合调度、使用配方重建权重时，两档的两个完整输出均与N128验证版本逐位一致。整请求MMA上界分别为 **1.249995995×** 和 **1.499997073× C0**，包含所有原修复项和padding。

常驻量化投影权重（含scale/索引）分别为973,334,736和1,077,834,368 bytes；较原实现均多88,080,384 bytes（84 MiB），来自额外W0 scale布局。最终Graph预热峰值allocated分别约6.56和6.66 GiB；较历史同配置增加约94和93 MiB，reserved峰值另见完整receipt。没有FP32输出scratch。1.25整请求新增scale读约2.288 GB、写约2.441 GB，已在非MMA成本中单独记录。

暖请求时间保留在raw receipt，均`performance_eligible=false`；不与历史不同负载的请求耗时计算端到端加速倍数。正式延迟、P90、FP8/BF16速度对照留待独占复测。

## 交付

[5090交接说明](README_5090.md)包含离线构建、两tile正确性检查、sanitizer、实际面板、原模型下载、资产重建、统一推理、设备专属调优和复测命令。两档全部168个资产均已从小配方逐字节重建并用于完整推理，无需传输约1.8 GiB量化权重。可选面板包含真实M416/M960四阶段数据，和核心代码/配方分包。

完整请求receipt以`.json.gz`保留；质量JSON、算子原始samples、编译manifest、测试日志和计数器摘要均有SHA256索引。以上结果冻结于24322c0；GitHub分支和Release交接入口见[发布索引](github_handoff/README.md)，本次同步不合并main。

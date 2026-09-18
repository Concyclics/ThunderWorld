# FP4 固定预算方案的进一步优化：PRO 6000

本轮保留原 1.5×、2×、2.5× 的权重资产、修复支持、head/阶段配额和选择器，
接通已有融合与 Graph 执行优化。完整请求吞吐分别提高 **12.5–12.8%、12.3–13.4%、
10.9–11.6%**。三档在 16 个 development 场景、每场景两个种子上的 **96 次完整生成
全部与冻结版本逐位一致**。另一个固定预算的新激活选择器未通过质量门槛，保留为负结果。

推荐新的显式 `norm` 执行配置，继续使用 `residual_energy` 选择器。原 manifest、
历史结果、离线资产和发布版本未被覆盖。这是已测执行方案的改进，尚未证明延迟 Pareto 最优，
也没有新增“质量超过 FP8”的结论。

## 范围与计时

- GPU0：NVIDIA RTX PRO 6000 Blackwell Server Edition，188 SM，96 GB；CUDA 13.0，
  PyTorch 2.11.0+cu130，Triton 3.6.0。不是 5090 结果。
- 同一 Cosmos3-Edge 修订、同一生成投影范围、35 steps、CFG、Graph，`debug=off`。
  激活及修复始终使用当前输入。AR 预填充精度和缓存条件保持原样。
- 单请求串行。计时仅使用两个预先固定的输入：RBench `visual_reasoning_0010` 和
  PAI-Bench `av_660aa819-4a75-45d0-b79a-e6c2f2c321dc`，不代表全部场景的平均延迟。
- 五轮固定随机顺序；每轮每配置独立加载、完整预热，每个输入测三个相同种子。
  FP4 新旧版本共 180 次请求、90 对输出逐位一致，包括第三个计时种子。
- 包含每请求的新 AR prefill、35 轮生成、视频解码和 CPU 输出；不包含模型加载、建图、
  NPZ 压缩写盘与结果哈希。每次计时边界检查 GPU 无其他计算进程。
- FPS 定义为 **61 个输出视频帧 / 完整请求时间**；这是生成吞吐，不是播放帧率。
  如只计排除条件首帧后的 60 帧，所有 FPS 同乘 60/61，比较比例不变。
- 使用固定安全 launch，未完成设备调优表的严格全覆盖资格检查，所有记录保持
  `performance_eligible=false`，不更新正式速度排行榜。独立 profiler 结果不混入计时。

## 实际收益

| 预算 | 固定输入 | 原中位数 ms | 新中位数 ms | 原 FPS | 新 FPS | 吞吐增加 |
|---|---|---:|---:|---:|---:|---:|
| 1.5× | RBench | 1750.88 | 1552.51 | 34.84 | 39.29 | 12.78% |
| 1.5× | PAI-Bench | 1788.16 | 1589.57 | 34.11 | 38.38 | 12.49% |
| 2× | RBench | 1902.03 | 1677.56 | 32.07 | 36.36 | 13.38% |
| 2× | PAI-Bench | 1940.60 | 1727.44 | 31.43 | 35.31 | 12.34% |
| 2.5× | RBench | 2127.21 | 1905.40 | 28.68 | 32.01 | 11.64% |
| 2.5× | PAI-Bench | 2165.02 | 1951.80 | 28.18 | 31.25 | 10.92% |

每请求减少约 0.20–0.22 秒。吞吐增加与延迟降低不是同一个百分比。
[speed.json](speed.json) 保留 P90、逐轮比值和成对逐轮平均比值的 bootstrap 区间；
该区间的估计量不同于上表的合并样本中位数比值，不将两者混写。

峰值 allocated 显存下降 **0.49–0.51 GiB**，峰值 reserved 则增加 **0.29–0.34 GiB**。
后者包含 Graph/分配器保留的空间，因此不能笼统声称所有显存口径都下降。
没有改变离线权重文件大小或 W1 支持并集。

| 预算 | RBench allocated GiB 原→新 | RBench reserved GiB 原→新 |
|---|---:|---:|
| 1.5× | 6.428 → 5.928 | 6.963 → 7.297 |
| 2× | 6.466 → 5.961 | 6.967 → 7.273 |
| 2.5× | 6.707 → 6.214 | 7.133 → 7.424 |

准备开销独立记录。在已有编译缓存下，两种输入 shape 的累计 warmup 中位数分别为
1.5×：9.068→9.830 秒，2×：5.312→6.323 秒，2.5×：6.013→7.008 秒。
独立模型加载约 4.4–4.9 秒，已量化资产的离线准备约 0.13–0.22 秒。
这不是清空 JIT/文件缓存后的冷启动实验；原始阶段记录位于每个请求组的 `preparation`。

### 原配置 BF16 / FP8

按用户要求，没有额外调优或给 baseline 开启新融合选项。另做五轮独立加载的原配置复测：

| 方法 | RBench ms / FPS | PAI-Bench ms / FPS |
|---|---:|---:|
| BF16 | 1673.82 / 36.44 | 1688.52 / 36.13 |
| FP8 W8A8 + BF16 attention | 1470.77 / 41.47 | 1486.50 / 41.04 |
| 新 1.5× | 1552.51 / 39.29 | 1589.57 / 38.38 |
| 新 2× | 1677.56 / 36.36 | 1727.44 / 35.31 |
| 新 2.5× | 1905.40 / 32.01 | 1951.80 / 31.25 |

1.5× 比当前 BF16 实现快约 6–8%，2× 大致持平，2.5× 仍更慢；FP8 在这一组中最快。
baseline 请求的输出也核对了历史 receipt。其计时在 FP4 sweep 后单独执行，没有与
FP4 共同交错，因此这些跨方法比值是诊断结果。新增融合中包含通用优化，不能据此宣称
FP4 算术本身必然比同样优化的 BF16/FP8 更快。[controls.json](controls.json)

## 落地的执行修改

`baseline_runtime.make_runner()` 新增经过验证的 `runtime_options`，旧 manifest 不设置
该字段时保持旧行为，外部 baseline 适配器拒绝未经验证的覆盖。

本轮 `norm` 是以下累计配置，并非只有 normalization：

```json
{
  "fuse_calibrated": true,
  "fuse_pointwise": true,
  "static_rotary": true,
  "codec_graph": true,
  "fuse_norm": true
}
```

它接入带 prescale 补偿的 FFN ReLU-squared/amax 融合、pointwise 融合、请求内 rotary
预计算、使用当前输入且输出独立存储的 codec Graph，以及 normalization 融合。
这些底层实现已存在，本轮消除了冻结推理入口的接入遗漏并完成组合验证。
主线仍使用原 native Triton FP4 Tensor Core GEMM；没有把收益归因于新 CUTLASS 内核。

推荐配置通过 `optimization_presets.py` 导出新 manifest，复用原资产，记录全部质量
receipt 的哈希。具体命令见 [OPTIMIZATION.md](../../tools/fp4_handoff/OPTIMIZATION.md)。

对两个固定输入，新旧实现的 MMA、W1 支持、packing 成本模型均一致。以 RBench 为例：

| 名义预算 | 执行 tile 核算的请求 MMA / C0 | W1 支持并集 payload 字节 |
|---|---:|---:|
| 1.5× | 1.499956867 | 36,864,000 |
| 2× | 1.927536232 | 74,317,824 |
| 2.5× | 2.463768116 | 334,430,208 |

MMA 范围是生成投影 + attention，完整延迟还包括 AR/codec 等工作。
W1 payload 不是模型总大小或峰值显存；packing 流量是实现成本模型，不是硬件带宽计数器。

## 固定预算的质量尝试：保留负结果

新增实验选项 `activation_score=represented_gain`。对当前激活，先在寄存器内计算
实际 E4M3/E2M1 残差表示，再用

`sum(max(r² − (r − decoded_FP4_residual)², 0) × offline_channel_sensitivity)`

给 block 排序。保持 A 的整数 quota、W1 支持、attention、MMA 和资产不变，只有被选中的
残差编码写入内存。相比原选择器，这额外引入全输入残差编码/解码的向量运算，已单独
计入 `non_mma_work`；它不是免费选择。该分数仍是对角敏感度代理，不是完整层输出的真实
边际修复收益。2.5× 的 A 已全量，不需要用它重新排序。

16 个既有 development 场景：9 个 RBench、7 个 PAI-Bench，每场景两个冻结种子，
共 64 次候选生成。以下指标均相对同种子 BF16；**原→候选**：

| 预算 / 数据集 | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | 相对 L2 ↓ |
|---|---:|---:|---:|---:|
| 1.5× / RBench | .182627 → .177407 | .700000 → .707177 | 17.236 → 17.471 | .257774 → .250915 |
| 1.5× / PAI-Bench | .113919 → .123906 | .796471 → .787064 | 23.965 → 23.589 | .150499 → .158111 |
| 2× / RBench | .149061 → .148677 | .756656 → .749366 | 18.671 → 18.686 | .221339 → .222791 |
| 2× / PAI-Bench | .096039 → .104076 | .820029 → .812800 | 25.829 → 25.562 | .134319 → .147136 |

先对每场景的两个种子取平均，再对场景做 20,000 次成对 bootstrap。LPIPS 候选减原方案的
描述性 95% 区间依次为：

- 1.5× / RBench：`[-.016000, +.005566]`；PAI-Bench：`[+.003834, +.017424]`。
- 2× / RBench：`[-.015214, +.017338]`；PAI-Bench：`[-.001545, +.020815]`。

1.5× 在两个数据集分别有 7/18、11/14 个种子 LPIPS 退化；2× 分别有 7/18、8/14 个。
最差单种子 LPIPS 增量分别达到 +.055519、+.043599、+.123463、+.056308。
**两档都拒绝升级选择器**，因此未继续为这个失败质量候选做速度选型。

这说明当前的“可表示残差收益 × 通道敏感度”代理不足以稳定改善完整轨迹。
块之间的误差相关性、阶段敏感性可能是遗漏因素，但本轮没有证明具体因果。
不能因 RBench 的局部平均改善就升级默认方案。所有逐场景结果和失败种子都保留在
[1.5× 数据](b1.5_protected_n_selector.json)、[2× 数据](b2_joint_selector.json)。
这是 development 机制实验，不是 72 场景新一轮验证，也未使用最终保留测试集。

## 还剩多少优化空间

对 2× 的相同 RBench 请求单独采集两个种子的 profiler。只汇总 Chrome trace 的
`cat=kernel, ph=X` 事件，不把 CPU op 与子 kernel 重复计入。

| 类别 | 原累计 kernel ms | 新累计 kernel ms | 新累计 kernel 占比 |
|---|---:|---:|---:|
| FP4 投影/FFN GEMM | 643.8 | 642.6 | 39.4% |
| 线性层 packing、选择及融合 ReLU/amax | 160.3 | 171.6 | 10.5% |
| FP4 attention，含 Q/K/V packing 与 combine | 206.1 | 206.0 | 12.6% |
| 卷积与布局转换 | 366.5 | 366.4 | 22.4% |
| 其余 kernel | 466.5 | 245.6 | 15.0% |

新实现每请求 kernel 数从 **223,309 降为 155,270，减少 30.5%**。
主 GEMM 和 attention 时间基本没变；本轮收益主要来自外围逐元素、复制与执行路径的减少。
融合 ReLU 被归入 packing 类，因此该类别的增加不能单独解释为 packing 退化。
剖析累计 kernel 时间不等于请求关键路径时间，也不表示 Tensor Core 利用率。
[profile.json](profile.json) 保留全部 kernel 名称、时间、计数与 trace 哈希。

后续优先级基于上述实测，而不是承诺额外加速比例：

1. **同质量的 packing → GEMM 工作流。** 合并可证明等价的 amax/summary/pass，缩短
   residual scratch 生命周期，在 prescale/支持条件一致时复用 packing；分 tile 流水必须
   保持全输入 amax 与当前输入语义。先查实际关键路径和寄存器/带宽压力，再决定 overlap。
2. **投影/FFN 主 GEMM 与完整 attention。** 仍有约 39% 和 13% 的累计 kernel 时间。
   针对当前 N/K 与修复支持的 launch、加载/累加路径做设备专属优化，计入 packing/selector。
   并发或真 batch 属于新的吞吐实验，不能用本轮 B1 结果推断其上限。
3. **codec 和剩余点算子。** 卷积/布局约占 22%，仅提升 FP4 GEMM 不会消除它。
   在不改 BF16 质量锚点的条件下减少布局复制、扩展精确融合，并分别保留 latent 与完整解码口径。
4. **质量路线改为组合干预收益。** 保留 head/阶段特征提出候选，但用当前轨迹的同输入
   BF16 输出误差和完整视频 guard 决定替换哪些 tile；联合离线 W0/W1 搜索。
   本轮失败选择器不能作为质量贡献定型。现有 2×、2.5× 还有少量整数预算余量，也须通过
   真实收益实验决定如何用，不能直接假定更多修复必然改善输出。

作为数量级说明，若仅把新实现 GEMM 的累计 kernel 时间缩短 20%，按该剖析占比计算，
总累计 kernel 时间约下降 7.9%；这不是端到端加速预测。继续提升需要组合处理多个瓶颈，
尚无证据说明 Tensor Core 已满载或剩余提升空间已经耗尽。

## 验证与可复测交付

- CPU 检查：34 passed / 18 GPU-only skipped；相关 importance/redesign/schedule 检查另有 10 passed。
- 新选择器 GPU 参考检查：4 passed；覆盖 M129/M416、K2048/K9216、全零输入、独立
  FP4 解码参考、实际选中编码、Graph 更新当前输入。
- compute-sanitizer：memcheck 0 errors；racecheck 0 hazards、0 errors、0 warnings。
- 三档四个执行组合的小范围资格检查 48 次完整请求全相同；完整 development 的
  96 次请求与冻结输出全相同；五轮计时的 90 对请求也全相同。
- `git diff --check` 与新增工具 Python 编译检查通过。

交付入口为 [OPTIMIZATION.md](../../tools/fp4_handoff/OPTIMIZATION.md)。
[measurements.json.gz](measurements.json.gz) 保存 123 份原始 JSON 记录以及 sanitizer 日志，
包括逐请求时间/哈希、协议、计算量账本、逐场景质量及执行配置。
[archive-index.json](archive-index.json) 记录归档哈希。
大型 NPZ 和完整 trace 保留在原始结果目录 `/home/yufeng/thunderworld/fp4_next_20260918`，
没有把模型或视频装入 Git。原始记录内保留输出索引和哈希。

本机可直接使用的新 manifest：
`/home/yufeng/thunderworld/fp4_next_20260918/optimized-study/manifest.json`。
其他设备先按已有 handoff 重建相同离线资产，再重新运行资格检查、导出新 manifest 和性能复测。

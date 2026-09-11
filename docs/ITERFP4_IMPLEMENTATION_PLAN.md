# iterFP4 实施、验证与 demo 计划

日期：2026-09-10。本文是后续工作的验收规格；标为“待实现”的内容不是已运行功能。
总体方法、代码审计与外部来源见 [项目设计](THUNDERWORLD_ITERFP4_DESIGN.md)。

已补充 [全 step 重采样与修复/复用消融](FINDINGS_RESAMPLING_20260910.md)：
两个模型的 layer/head/step 观察、真实 action/AV 路径和研究用 temporal cache 已接通。
随后补充了 [FFN 自动预算原型与硬件验证](AUTO_BUDGET_5090_20260910.md)：
layer/phase/CFG 档位分配、GPU 当前输入支持选择、同预算完整轨迹对照，以及退化时回退均匀预算的质量门控。
这推进了 P3，但还没有完成其独立测试集及 random 等全部验收；bulk-fit 校准和公共部署接口仍待实现。

## 1. 建议推进顺序

先把 **FFN 的修复感知量化＋结构预算**做成两模型可比较的完整方案，再决定 attention 是否进入部署路径。
不把降低质量门槛当作解决当前 attention 比 BF16 慢的方法。

| 顺序 | 实施内容 | 交付与通过条件 |
|---|---|---|
| P0：实验身份 | 固定模型、样本划分、量化公式、mask 语义、source hash；保留当前 RTN pilot | 任一视频/数字能追溯到量化资产、策略、模型 revision 和计时条件 |
| P1：数值参考 | 新增离线 reference 与校准器：absmax、MSE/4-6、修复感知候选；独立 A/W 支持与交叉项 | 合法 NVFP4 scale，解码 oracle，零/极端值测试；真实捕获输入上比较输出误差；不把模拟耗时当 native 性能 |
| P2：FFN 接通 | 导出 W0、稀疏 W1、prescale、A 支持，公共 runtime 加载；接入 Edge/Nano adapter | 不依赖实验目录临时对象；先局部全 FFN、再完整 35-step 自由生成；保持各自非线性 |
| P3：结构预算 | 校准 layer/phase/CFG 和关系收益，编译规则 tile；相同预算做消融 | 未参与选择的轨迹中优于 uniform/random/energy，且计入选择、packing 和索引开销 |
| P4：attention | QK/PV 独立修复预算、模态边界与联合 softmax；量化和 epilogue 优化 | 完整轨迹质量通过，完整算子及 E2E 同时有价值后才设为速度方案 |
| P5：设备与能量 | SM121/SM110 后端、设备实测、能量采样 | 不靠放宽 SM120 guard；通过数值、内存、轨迹和能耗验证再标支持 |

P1 与 P2 是当前最大的实现缺口。已有 5090 tile/流水优化和共同 CUDA Graph 可继续使用；
Split-K 留在实验配置，数值变化必须重新评估完整轨迹。无需为本计划先更新当前可用的 CUDA Toolkit。

## 2. 校准资产与接口

建议新增以下模块，名称为计划，不是现有接口：

```text
tools/iterfp4_calibrate/reference.py     数学参考及合法 scale 编解码
tools/iterfp4_calibrate/fit.py           离线量化/支持交替选择
tools/iterfp4_calibrate/compile.py       语义预算 → 可执行支持与布局
packages/fp4-repair/.../artifact.py      带版本与校验的资产加载
```

资产至少包含：

| 字段组 | 必要信息 |
|---|---|
| 身份 | schema_version、模型 repo/revision、层路径、形状、量化器版本、源码 hash |
| 格式 | E2M1 code 顺序、K16 scale 类型/布局、外层 factor 的广播轴、输出与累加精度 |
| 量化 | 首轮规则/clip 候选、残差规则、prescale 与舍入位置、完整参数 hash |
| 权重 | W0、选定 W1、CSR/连续块索引、实际字节数；BF16 原权重只用于参考/明确 fallback |
| 策略 | FFN SA/SW；attention 的独立 QK/PV 修复项；layer/phase/CFG 与模态区间 |
| 校准 | calibration/dev/test split ID，输入/seed 身份，损失定义与候选选择记录 |
| 执行 | 目标能力、允许 tile、实测 cost table 版本、graph 更新协议、fallback 原因 |

加载时验证模型和拓扑，不允许 Nano 的资产被 Edge 静默接受。
W1 与 W0 必须来自同一变换后的权重；如果 W0 来自外部量化 checkpoint，先恢复其精确码和 scale 再定义残差。
历史 FFN 实验仅借用 AWQ prescale 后重新 RTN，不应被标记为已比较完整 AWQ。

FFN A 支持以 K 块为主，尽量在 N 上共享。若每个 N tile 有不同支持，producer 必须按并集生成 A1，
同时单独记录“生成并集大小”与“实际执行修复 FLOPs”；二者不是同一预算。
W1 稀疏存储可以减少参数占用；A1 只有按选择压缩存储并释放冗余 scratch 后，才有对应的容量收益。

Attention 以原模型合法性 mask 为基础，独立传入 repair descriptor。
部署默认跨 step 只复用描述/支持，重算当前数值。
研究工具另显式测试旧 repair delta / 完整 attention 输出缓存；这属于有误差的算法近似，
按用户要求做完整轨迹质量验证后再决定是否进入部署，不能当作等价 kernel 优化。
新增 phase 策略后，Graph 需要读取更新后的 device buffer 或使用明确有界的变体，测试不能只覆盖固定 mask 重放。

## 3. 最小实验矩阵

### 3.1 分离算法改善与 kernel 改善

第一条轴固定量化码、scale、mask 和累加语义，比较原始/优化实现；第二条轴固定执行框架，比较量化和修复算法。
全部基线使用相同的 Graph/compile、输入、调度器和计时范围。不同 checkpoint 或不同生成步数不得混在同一个加速比中。

| 对照组 | 用途 |
|---|---|
| BF16 与可运行的 FP8 | 质量参考与强性能基线；记录覆盖了哪些模块 |
| 标准 NVFP4 absmax；更强 scale 校准 NVFP4 | 判断改善是否仅来自更好的单轮量化 |
| absmax + full ACT2 | 保留当前 E2E pilot 的可复现锚点 |
| bulk-fit + full ACT2 | 隔离首轮量化改进；“bulk-fit”必须带具体优化定义 |
| 全量三项；含交叉项四项 | 精度/计算上限参考，不能默认三项等价 BF16 |
| 等名义预算 uniform / random / energy / residual-weighted | 判断支持选择本身是否有效；random 使用多个固定种子 |
| 去掉 layer、phase、有向模态、head 的逐项策略消融 | 判断世界模型结构是否是实际贡献 |
| 完整结构策略 | 同预算质量与同质量实际延迟两种 Pareto 对比 |
| ARCQuant / SharQ 等可复现残差基线 | 检验相邻方法；未支持的模块明确标 N/A，不以第三方论文速度作本机数据 |
| 规则 sparse attention / BF16 attention | 评估保留基础连接的成本与收益，不只挑开销大的 sparse 方法 |

先做廉价局部筛选，再对短名单完成全轨迹；不把所有轴做成浪费时间的笛卡尔积。
首轮候选、SA/SW 和 budget 的对比应包含 `A-only / W-only / A+W / cross`，
并在改变首轮 clipping 后重新测交叉项。

### 3.2 输入与模型覆盖

当前数据包适合回归与形状 profiling；重复 M128 输入构造 M512 不提供新质量样本。
必须按来源/场景划分 calibration、development、held-out，避免只换同图 seed 就宣称跨场景泛化。

建议初始开发规模为每模型 12 个不同场景/提示词、3 个 seed，覆盖动作、物体交互、镜头运动和容易出错的控制条件；
这只是开发集建议，不是最终统计充分性保证。冻结策略后再使用未参与调参的测试集，报告分布和区间。
样本数量应随质量差异、失败率和不确定性增加，不能以少量 SSIM 均值作为停止条件。

| 维度 | 最先验证 | 后续扩展 |
|---|---|---|
| 模型 | Edge、Nano 分别校准与评估 | 统计哪些规律可迁移，不共享未经验证的具体参数 |
| 任务 | 当前 I2V；再加 T2V | 模型/接口支持的 AV、action、Reasoner→Generator 联动 |
| 长度与分辨率 | 当前 320×192/61 帧/35 步作为锚点 | 各模型支持的更大分辨率/帧数；OOM 和 offload 单列 |
| 去噪 | 完整候选自己的隐藏状态 | 改变步数/CFG 时用 sigma phase 校准，不能机械复用 step index |
| 性能 | 全驻显存与共同 Graph | 生产 offload/并发路径另测，冷启动和稳态分别报告 |

Edge 使用 ReLU² up/down，Nano 使用 SiLU gate×up/down；GQA 比例也不同。
不同模型、配置的绝对耗时可并列展示，但优化倍率只在各自同条件基线内计算。
I2V 单一视频输出无法验证 action/audio 的方向性预算收益；这些论点必须有相应模态路径和数据。

### 3.3 指标与通过条件

- **局部数值：** FP4 decoded oracle、相对 L2、max error、logit/LSE、projected output；测试极端值、tail、零块、空修复支持、交叉支持与不同累加顺序。
- **轨迹：** teacher-forced 局部评估用于归因；free-running 使用每个候选自己的中间状态和随机种子，记录逐步预测误差与最终输出。
- **生成质量：** paired SSIM/PSNR 仅作一致性；另测视觉/时序质量、prompt/control adherence 和失败样本。音频测同步，action 测轨迹误差/任务成功率。选定并冻结具体评测器版本。
- **性能：** 请求 wall time、transformer/量化/选择/GEMM/softmax/非线性/VAE 时间、吞吐、冷启动、重复样本与区间；短轮数不宣称 p95。
- **内存：** W0/W1/scales/metadata、activation/临时 buffer、Graph pool、KV/其他组件分别统计；同时记录 allocated/reserved 与设备范围。
- **能量：** 在稳定温度下交错跑足够长的 BF16/候选请求窗口，积分功率并除以完成请求数；记录 idle、功率限制、采样率和传感器边界，报告总能量及可选 idle-adjusted 能量。

在 held-out 前按应用定义质量非劣界和任务失败上限，再比较满足约束的最快点。
目前没有用户确认的产品质量阈值，不能把历史 0.90/0.80 sparse SSIM 档位当作新算法的验收标准。
精度、速度、内存与能量都报告失败配置；若 attention 不能通过，部署方案可保留 BF16 attention，并准确报告 FP4 覆盖范围。

## 4. 三个 demo

### Demo A：为什么要按误差修复

**目标：让观众看到“值大”与“值得修复”是两回事。**

展示真实捕获块和可解释的合成块：数值分布、absmax 与拟合 scale 的网格、第一轮残差、选中的补偿块及最终输出误差。
可切换 base / A repair / W repair / first-order / cross；同时显示真实执行乘积预算和表示字节数。
要包含“clipping 后未分配修复导致变差”的反例，以及权重更敏感的块，避免把启发式演示成普遍定理。
数学可视化允许使用高精度 reference，但只有 native kernel 计时才可标 GPU 性能。

状态：**待实现**。可直接复用附件的真实 FFN 输入和现有 decoded oracle；新校准器完成后再产图。

### Demo B：两模型完整生成与预算解释

**目标：同时展示生成结果与为它付出的代价。**

同输入/seed 并排展示 BF16、标准 NVFP4、uniform repair、结构化 iterFP4 四路视频；可切换 Edge/Nano。
附上 layer×phase 修复预算图、模态有向关系图和真正执行的 repair mask；mask 图必须注明基础 attention 仍全部保留。
每路显示延迟、峰值内存、能量（未测则显示未测）、质量指标、quantizer/policy 身份与执行覆盖。
固定质量/固定耗时的 Pareto 图从结果文件生成，不能由理论 FLOPs 推算代替。

已有材料：当前 RTN+ACT2 的两模型视频、[帧图与实测表](COSMOS3_E2E_5090.md)；
可作为“现有实现”演示，不能提前将其标签改成结构化 iterFP4。
状态：**已有 pilot 素材，四路正式算法 demo 待 P1–P3 完成**。

### Demo C：真实端侧任务

**目标：证明在目标设备上少用资源且仍完成任务。**

优先选择实际可获得的 Spark 或 Thor，在相同设备/功率模式上比较 BF16/可运行强基线与候选。
展示持续生成吞吐和 J/request，而非仅显示瞬时 GPU utilization。
若展示具身控制，加入预测轨迹、控制约束和成功率；未执行真实控制时明确标为离线世界模型预测。
基线 OOM 可证明容量可行性，但不能把无限大的时间当作加速比；另给可运行的 offload 基线。

状态：**待目标设备与后端验证**。5090 的结果不能替代 Thor/Spark 实测。

## 5. 文档与发布口径

仓库首页以本次 iterFP4 目标为主，历史 sparse attention 发现保留为证据与对照。
旧文档中的“skip”“dense anchor”“sparse bucket”属于删边方案，不能未经修改直接成为 repair compiler 的执行语义。

当前可写：两模型 FFN pilot 的实测速度/内存收益、已验证 kernel 能力、局部修复收益与失败结果。
当前不可写：接近 BF16 的普遍质量、所有算子打满 FP4、已证明低能耗、已适配所有端侧架构、首次 FP4 残差补偿。
等 P1–P3 的 frozen held-out 结果完成后，再把通过验证的候选升级为推荐部署方案。

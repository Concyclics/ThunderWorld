# ThunderWorld FP4 repair

独立可安装的 SM120 推理组件：权重离线 W4，激活在线量化；FFN、attention core、attention projection 分开控制修复。所有 attention 边保留 dense FP4 基础计算，仅给选中的 block 增加残差乘积。无需训练。

```bash
pip install './packages/fp4-repair[cuda]'
# 已有兼容环境也可 pip install --no-deps ./packages/fp4-repair
pytest packages/fp4-repair/tests -q
```

开发验证环境：PyTorch 2.11.0+cu130、Triton 3.6.0、RTX PRO 6000 Blackwell (SM120)。其他架构主动报错，不静默切换成 BF16。配置模块不依赖 torch；运行内核需 CUDA extra。0.2.0 新增可安装的 `tw-cosmos` 完整请求入口，联合 FFN/投影/attention Graph，默认关闭探针；使用说明和复测命令见 [联合推理文档](../../tools/joint_inference/README.md)。以下底层接口保持兼容。

## 三个独立配置

```python
from thunderworld_fp4 import InferenceConfig, FP4Attention, FP4Linear
cfg = InferenceConfig.from_json('packages/fp4-repair/examples/head_guided.json')
attention = FP4Attention(cfg.attention, graph_cache_size=2)
# FFN 从已校准、离线打包的 W4 创建；QKV/out projection 使用另一份配置。
ffn = FP4Linear(codes, scales, global_scale, prescale=awq_prescale,
                bias=bias, config=cfg.ffn)
projection = FP4Linear(qkv_codes, qkv_scales, qkv_global_scale,
                      prescale=qkv_prescale, config=cfg.attention_projection)
y = ffn(x)
o = attention(q, k, v)
```

配置对象不可变；线性层配置在构造时确定。重新构造模块可切换其静态预算；逐调用动态选择使用 `repair_mask`，无需重复权重量化。Attention 可用 `RepairPolicy` 选择不同层、step、CFG branch 的配置。`examples/*.json` 给出完整三段配置。

| 配置域 | 参数 | 含义 |
|---|---|---|
| `ffn` | `coverage`, `output_budgets` | 对每个 M128/N128 输出块，修复多少 K128 激活通道块；默认全量 ACT2 |
| `attention_projection` | 同 FFN | Q/K/V/out projection 单独配置；与 attention core 无隐式绑定 |
| `attention` | `coverage`, `head_budgets` | 每个 query head、每个 Q128 块选择多少 K128 块 |
| `attention` | `terms=2/3/4` | 单侧 / 双侧一阶 / 包含交叉项；同时应用于 QK 和 PV |
| `attention` | `query_tile=16/32/64`, `splits=1/2/4/8` | 执行 tile 与 split-K；逻辑修复块始终 128×128 |

`head_budgets=[[0,1.0],[1,0.75],[2,0.25]]` 覆盖指定 head 的默认比例；其他 head 使用 `coverage`。各 head 独立按 `ceil(ceil(N/128)*coverage)` 取块，因此名义 50% 在短序列上可能超过 50%。`config.cost(H,N)` 报告实际 padded-tile 比例和乘积次数，不含选块、量化、softmax 等开销。

FFN 每块始终计算 A0W0，被选中时补 A1W0；乘积次数为 `1+ρ`。Attention 每块始终计算 Q0K0 和 P0V0；terms=2 补 Q1K0/P1V0，terms=3 再补 Q0K1/P0V1，terms=4 再补 Q1K1/P1V1；各 GEMM 的乘积次数为 `1+(terms-1)ρ`。这里的 ρ 是实际选择比例，不能直接当作延迟倍率。

## Head finding 与精确 block 控制

```python
from thunderworld_fp4 import config_from_head_scores, FP4Attention, make_repair_mask
# scores 必须是当前模型/层的校准结果；数值越大表示更需要保护。
config = config_from_head_scores(scores, protected_count=4,
                                 coverage=0.25, protected_coverage=0.75)
engine = FP4Attention(config)
y, selected = engine(q, k, v, return_mask=True)

# 或直接使用 head finding 的逐 head block 结果；索引为 (Q128, K128)。
mask = make_repair_mask(q.shape[1], k.shape[1], q.shape[2],
                       {0: [(0, 0), (0, 2)], 1: [(0, 1)]}, device=q.device)
y = engine(q, k, v, repair_mask=mask)
```

未传 mask 时，配额约束下使用当前 Q/K 的近似注意力质量与 V 能量选块。Head finding 控制每个 head 的配额；显式 mask 则直接控制具体块，优先级高于所有配额。mask 是 `[Hq, ceil(M/128), ceil(N/128)]` contiguous CUDA bool；False 只关闭修复。

现有 ThunderWorld head finding atlas 可直接逐层接入：

```python
import json
from thunderworld_fp4 import policy_from_head_atlas, FP4Attention
atlas = json.load(open("results/f4/cosmos3_nano_head_precision_atlas_gpu0.json"))
policy = policy_from_head_atlas(atlas, coverage=.25, protected_coverage=.75)
engine = FP4Attention(policy=policy)
y = engine(q, k, v, layer=layer_index)
```

此 adapter 读取每层 `compiled_q_fp8_heads` 作为受保护 query heads，实际执行仍为 FP4+修复；调用者须确认模型和 head 排序匹配。`examples/compile_head_atlas.py` 可导出逐层参数，便于检查。默认不自动启用旧 atlas。

不要直接把 projection head atlas 当作 attention core 最优评分：历史局部实验中，旧 atlas 的非均匀 75% 分配误差高于均匀 75%。建议用校准集测量增加修复的误差收益，并在独立 prompts/steps 上验证。示例 head 编号仅演示接口，没有代表通用的“重要 head”。

FFN 支持独立的 contiguous CUDA bool mask `[ceil(M/128), N/128, K/128]`：

```python
# 不同输出通道组可修复不同输入激活通道块。
mask = torch.zeros((math.ceil(x.numel()/x.shape[-1]/128),
                    ffn.out_features//128, ffn.in_features//128),
                   device=x.device, dtype=torch.bool)
mask[:, 0, 0] = True
result = ffn(x, repair_mask=mask)
```

默认 FFN selector 依据当前激活的采样能量排列 K128 块，再按每个 N128 输出块的配额选取。它是低成本启发式，不等同于已校准的误差收益。可以用外部校准 mask 替换。

## 跨 step 策略

```python
from thunderworld_fp4 import RepairPolicy, RepairRule, RepairConfig, FP4Attention
policy = RepairPolicy(default=RepairConfig(coverage=.5, terms=3), rules=(
    RepairRule(RepairConfig(coverage=.75, terms=3), steps=(0, 5)),
    RepairRule(RepairConfig(coverage=.75, terms=3), steps=(30, 35)),
))
attention = FP4Attention(policy=policy)
y = attention(q, k, v, step=step_index)
```

规则按顺序首个匹配生效，step 为半开区间；有 layer/step/branch 条件时必须显式提供对应上下文。以上 step 范围是待校准模板。变化率分析可在外部决定 mask 或规则；本实现不自动估计跨 step 变化率，不复用旧激活/输出。重用 mask 只重用选择计划，所有 Q/K/V/P 和 FFN 激活仍重新计算。

## 推荐起点与已有证据

通过 `preset(name)` 获取 attention 配置，FFN/projection 默认建议保持 coverage=1，先隔离 attention 预算的影响。

| Attention preset | coverage / terms | 建议用途 |
|---|---|---|
| `balanced` | 0.50 / 3 | 双侧 50% 起点，配合 FFN full ACT2 |
| `quality` | 0.75 / 3 | 更重视 BF16 一致性的起点 |
| `full_dual` | 1.00 / 3 | 更高预算候选，需要整模型评估 |
| `full_four` | 1.00 / 4 | 交叉项上限对照，计算昂贵 |
| `full_act2` | 1.00 / 2 | 单侧对照，非精度推荐默认值 |
| `fp4_base` | 0.00 / 3 | 无修复 FP4 对照 |

历史实验（2026-09-07，旧 CUTLASS projection + 本 attention kernel 的前身）中，同一 3 组视频相对 BF16 的 SSIM：full 单侧 attention ACT2 **0.6414**，双侧 50% **0.8323**；24 组真实 attention 局部输入的平均相对 L2 为 **0.14751 vs 0.05824**。两者名义乘积次数相同，短序列取整可能增加双侧方案实际配额。另一批同 3 组视频的双侧 75% SSIM 为 **0.8668**。这些是 BF16 一致性指标，不能作为语义准确率。

新包装的 FFN 为独立 Triton 实现，尚未重新完成上述整模型视频实验，不能将旧 CUTLASS 后端的质量、显存或速度归于它。预算降低的 FFN 也未证明保精度，因此默认保留全量 ACT2。历史 attention 修复实现仍慢于 BF16 FlashAttention，原生 FP4 MMA 不代表已打满 Tensor Core，也不代表端到端加速。

## 实现、格式与边界

- Attention 使用原生 `tl.dot_scaled(..., 'e2m1', ...)`，流式在线 FP32 softmax/累加，不物化完整 attention 矩阵。在选中块内部累加修复项；未选中块跳过 P 残差量化。Q/K/V 打包在独立 producer kernels；当前不声称 producer 与 GEMM 已重叠。
- FFN 的在线 prescale + BF16 rounding + FP4 pack 一次产出所有 N tile 复用的激活。A0W0/A1W0 在同一 GEMM kernel/FP32 accumulator 中执行。选块的并集控制残差 producer；仍分配完整双平面 scratch，稀疏预算减少计算并不按比例减少 scratch。W4 不含额外修复权重。
- 离线 packed W：uint8 `[N,K/2]`，低 nibble 在前；未 swizzle E4M3 `[N,K/16]`；正 scalar global_scale；`W=FP4(code)*scale*global_scale`。AWQ prescale 由当前激活在线乘入，权重必须已经离线除以该 prescale。不能直接传 CUTLASS swizzled scales。
- `FP4Linear.from_float(weight, prescale=...)` 是离线 RTN 打包工具，不执行 AWQ 校准。使用 AWQ 应从已校准 checkpoint 传入 codes/scales/global_scale/prescale。
- FFN 输入 BF16 `[...,K]`，N/K 是 128 的倍数。Attention 输入 BF16 `[1,sequence,heads,128]`，支持 GQA 整数比例，仅 batch=1 非因果、无 padding mask 的 attention。两者仅推理；不支持 autograd。
- `FP4Attention` 的 CUDA Graph cache 按 stream、shape、配置与返回 mask 需求隔离，默认最多 2 项。每次复制当前输入并返回独立输出存储；这些成本包含在 wrapper 内。清理/淘汰会同步对应 stream。模型已有外层 graph 时可直接调用函数；非均匀配额需要在外部预制显式 mask。
- 有效输入须有限且幅度在 BF16/FP32 数值范围内。构造线性层时避免 `.to(dtype=...)` 转换 packed buffers 的 FP8 scale 类型，只移动设备。

验证命令与结果见 [VALIDATION.md](VALIDATION.md)。

### Experimental CUTLASS repair fusion (SM120)

`tw-cosmos` accepts `--linear-backend cutlass-fused --cutlass-library PATH.so` and
an optional `--cutlass-small-library N64.so`. Build N128/N64 binaries offline with
`python -m thunderworld_fp4.cutlass.build --tile-n 128 --output PATH.so` (or64).
The backend fuses dense FP4 and selected A/W repair into the same FP32 accumulator
and writes BF16 once. It retains the existing FP4 attention implementation.
Device-bound exclusive tuning compares only bitwise-equivalent tile candidates;
shared calibration never tunes or contributes formal latency results.

This backend changes accumulation order relative to Triton and remains
experimental: complete development outputs show small quality changes, and the
FP8 quality target has not passed. See the
[CUTLASS report and 5090 handoff](../../experiments/cutlass_fused_repair/README.md)
for source builds, verified offline weight reconstruction, measurements and limits.

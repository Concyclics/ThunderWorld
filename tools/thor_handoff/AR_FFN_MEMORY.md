# AR FFN 单独量化与原生 baseline 内存比较

本轮按用户指定仅在 **RTX PRO 6000** 测试。原生成分支预算、资产、head/step schedule 和 attention 不变；新增开关仅覆盖 `layers.*.mlp` 的 AR FFN，保留 AR Q/K/V/O、embedding、视频输入输出头、timestep、norm 的原精度。VAE 始终使用 BF16，主体内存测量前真正释放。

按最新要求，**本轮只补内存压缩效果，不重测质量**；已取消排队中的解码与评分。下面运行生成仅用于记录真实请求显存峰值，不能将这些新配置标注为已完成质量验证。Thor 不运行本轮内存实验。

## 配置与比较边界

| Profile | AR FFN | 其余 AR |
|---|---|---|
| `trim` | BF16 | 原精度 |
| `ar_ffn1` | W4A4 基础项 | 原精度 |
| `ar_ffn8` | 原生 W8A8；在线 per-token 激活量化 | 原精度 |
| `ar_ffn4` | `(A0+A1)(W0+W1)` 四项 FP4 | 原精度 |

扩展范围图和汇总对比中的 FP8 方案固定选择 `ar_ffn8`：DM 与 AR FFN 均使用原生 W8A8 FP8。历史 `fp8/trim` 仅作为尚未扩展 AR FFN 时的对照，不能替换扩展范围的 FP8 主结果。这里的 AR 范围是 FFN；AR Q/K/V/O 与 attention 运算仍保留原精度。

最新直接量化实验 `ar_ffn_direct_memory_20260923` 中，所有 FP4 方法统一使用 `ar_ffn1`，AR FFN 只执行基础项，不保存 W1。原生 DM 配置不变，包括 ARCQuant 50%、SharQ、我方 1.5×/2×。主对比与图由 manifest 的 `comparison_ar_fp4_profile` 决定；历史四项 AR 记录仅作为内存差值对照。该配置未重测质量，不能引用四项 AR 配置的质量结果。

`ar_ffn4` 保存两份 FP4 权重平面及各自尺度，不承诺将 AR FFN 资产缩至单平面的 1/4。W0/W1 在准备时离线生成并缓存，激活量化使用当前 AR 输入。保留两个 CFG 的新鲜 prefill，随后沿用原有 AR cache。

这里统计 GPU 常驻与请求峰值。准备缓存仍依赖原始 checkpoint 启动，可能保存运行时未使用的准备数据；不把“原 checkpoint 加缓存”的磁盘占用当作部署权重大小。

比较 BF16、FP8、普通 W4A4、ARCQuant 50%、SharQ、我方 1.5×/2×。ARCQuant 50% 和 SharQ 保留原始 `trim` 行，并提供 **原生 DM baseline + 公共 AR FFN 适配** 行；后者明确属于扩展适配，不声称是官方默认 AR 实现。外部方法的 eager 和 BF16 attention 保持不变。四项 AR 历史记录单独保留。

1.5×/2× 仍是冻结的 **DM 预算**。额外 AR 工作不藏入原预算，原始 receipt 另存 AR 形状、补项数和实际请求前缀信息。共享负载下不进行性能竞速，不引用诊断耗时计算加速比。

## 可复测命令

在仓库根目录，指定一个新的工作目录。下列命令复用已完成的原范围/四项 AR 记录，补测直接量化的八个新配置。准备只做 CPU 文件操作和源代码复制，不更改冻结的 `packages/fp4-repair`，也不覆盖历史实验。

```bash
python tools/thor_handoff/ar_ffn_prepare.py \
  --root /abs/new-ar-ffn-study \
  --direct-from /home/yufeng/thunderworld/ar_ffn_memory_20260923

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/yufeng/thunderworld/.venvs/cosmos3/bin/python \
  tools/thor_handoff/expand_body_campaign.py \
  --root /abs/new-ar-ffn-study --variant /abs/new-ar-ffn-study/variant \
  --jobs /abs/new-ar-ffn-study/jobs.json

python tools/thor_handoff/ar_ffn_report.py \
  --root /abs/new-ar-ffn-study \
  --prior-breakdown docs/paper/manuscript/evidence/body_memory_breakdown_20260923/breakdown.json \
  --output /abs/new-ar-ffn-study/report

PYTHONPATH=/home/yufeng/thunderworld/body_memory_breakdown_20260923/python-deps \
MPLCONFIGDIR=/tmp/thunderworld-memory-mpl \
/home/yufeng/thunderworld/.venvs/cosmos3/bin/python \
  tools/thor_handoff/ar_ffn_plot.py \
  --input /abs/new-ar-ffn-study/report/memory_breakdown_pro6000_ar_ffn.chart-data.json
```

最后一条改用 `memory_breakdown_pro6000_dm_only.chart-data.json` 可生成补齐 ARC/SharQ 后的原范围对比。两图保留绝对 GiB、单排四项图例，无总标题或底部小字。不能混用两图的权重范围。

`comparison.csv` 只包含当前选定的七种方法；`resident_components.csv` 保留历史及新配置完整账本；`ar_direct_vs_four.csv` 对比同一 DM 方法关闭 AR 修复前后的实际常驻与请求峰值。峰值来自新请求的测量，不从旧峰值减去理论权重大小。

## 证据与验证

- 两个既有场景，每场景种子 `301017/301018/301031`，320×192、61 帧、35 步、CFG 6、flow shift 3。每配置独立进程；只串行运行一个 GPU 实验进程。
- 每请求两个新鲜 AR prefill、70 次 Transformer CFG 调用，无临时 Graph 捕获；移除 VAE 前后 latent 逐位一致。
- `body_inventory.py` 从真实 live CUDA storage 去重统计原生权重、尺度、索引和布局副本，并要求总量与 runner receipt 完全一致。它不把 Graph 或工作区算成模型权重。
- 报告检查模型 identity、BF16 条件 latent hash、VAE 大小、生成投影集合及六个请求身份一致，并要求新增常驻节省恰好来自 AR FFN。
- 直接量化记录额外验证所有 AR prepared shape 都只有一项，AR 常驻中没有 W1 编码或 W1 元数据。
- `expand_body_quality.py` 的独立评分能力保留，本轮未执行；不得将“VAE 隔离前后 latent 相同”当作“量化前后质量相同”。
- 新增范围筛选没有改动数值内核。原生内核沿用已验证版本；范围扩展后的生成质量本轮不作结论。
- 这是两场景的内存测量，不替代完整 RBench/PAI-Bench 质量验收。历史结果保留，新输出和原始 JSON 使用独立目录。

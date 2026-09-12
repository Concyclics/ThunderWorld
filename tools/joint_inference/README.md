# 联合 FP4 Cosmos 推理与复测

`thunderworld-fp4` 0.2.0 将完整请求入口放入可安装包；本目录提供配置、独立实验编排与质量判定。研究入口和历史结果保留。质量判定是相对 BF16 输出的一致性，不是动作任务成功率。

## 环境与入口

验证环境：RTX PRO 6000 Blackwell Server Edition，物理 GPU0/3，PyTorch 2.11.0+cu130、Triton 3.6.0、Diffusers 0.41.0.dev0、vLLM 0.25.1。FP8 基线需要同一 CUDA/PyTorch ABI 的 vLLM `scaled_fp8_quant` 和 `cutlass_scaled_mm`。模型需使用提供 Cosmos3 支持的 Diffusers 版本，不能拿普通旧版代替。实验数字不代表 RTX5090。

```bash
pip install --no-deps ./packages/fp4-repair
# 或在干净兼容 CUDA 环境安装 cuda/cosmos/evaluation extras，再安装匹配的 vLLM。
export CUDA_VISIBLE_DEVICES=0
export MODEL=/path/to/Cosmos3-Edge-a9d944e
tw-cosmos --help
```

CLI 当前要求显式选择物理 GPU0 或 GPU3。推理实例固定设备、按单请求串行运行，不支持并发服务、多 GPU 或准备后 `.to()` 迁移。Python 接口：

```python
from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
import json
config = RunConfig.from_dict(json.load(open('tools/joint_inference/configs/dual.json')))
case = json.load(open('/path/to/case-with-absolute-image.json'))
runner = CosmosRunner('/path/to/model', config).prepare()
runner.warmup(case, steps=35)
signals, stats = runner.infer(case, seed=93001, steps=35)
```

`prepare` 加载模型并离线量化/准备权重；`warmup` 观察真实请求形状，生成所有联合阶段/CFG Graph；`infer` 只执行已预热的请求；`benchmark` 强制 debug off、调优全命中和 GPU 无其他计算进程。未经预热的 task/shape 会报错。改变配置需新建实例；`set_repair_masks()` 清理旧 Graph，要求再次预热。

## 配置语义

| 预设 | FFN/生成投影 | attention core | 用途 |
|---|---|---|---|
| bf16 | BF16 | BF16 | 原生一致性参照 |
| fp8 | 原生 W8A8；W per-tensor，A 在线 per-token | BF16 | 冻结 FP8 对照 |
| base | A0W0 | Q0K0、P0V0 | 普通 W4A4 |
| act2 | A0W0 + A1W0 | 单侧全量 ACT2 | 单侧质量/成本参照 |
| dual | 再加 A0W1 | 双侧一阶全量修复 | 三乘积候选 |
| four | 再加 A1W1 | 包含交叉项的完整修复 | 四乘积质量上限参照 |
| four_dual_attention | 完整四乘积 | 双侧三乘积 | 减少 attention 修复成本 |
| a75_four | A 修复75%、W修复100%，含交叉项 | 完整四乘积 | 缩减激活预算候选 |
| ffn_only | **所有生成投影**使用 ACT2 | BF16 | 历史名称；用于量化投影+BF16 attention 的归因 |

`ffn_only` 包括 FFN 和 Q/K/V/out，不能误称仅量化 FFN 的旧实验。所有基线/候选投影范围相同。`*_legacy` 在同一新入口使用原始 CSR FFN 内核，用于匹配范围的内核消融，不能称作旧历史完整请求的原始测量。

`ffn`、`projection` 各自包含 `activation`、`weight`、`cross`。A 支持是当前输入的 M128/K128 block；W 支持在准备阶段按残差尺度能量选 N128/K128 block。`attention.default` 和规则支持 layer、step、branch、`head_budgets`。逻辑 attention 修复块始终128×128；`query_tile` 只是执行 tile。所有 base blocks 始终计算，False mask 只关闭补项。

```bash
tw-cosmos warmup --model "$MODEL" --config tools/joint_inference/configs/dual.json \
  --case tools/joint_inference/cases/policy_table.json --tune --output /tmp/policy-warm
tw-cosmos benchmark --model "$MODEL" --config tools/joint_inference/configs/dual.json \
  --case tools/joint_inference/cases/policy_table.json --tuning /tmp/policy-warm/tuning.json \
  --steps 35 --seeds 93001 93002 93003 --output /tmp/policy-benchmark
```

可另传 `--attention-policy policy.json`、`--ffn-policy ffn.json --ffn-policy-mode auto50`、`--projection-config linear-spec.json`。FFN policy 兼容现有 `task|branch|phase|layer.part` 分配表，阶段边界固定为0/.2/.5/.8/1。规则并集决定捕获阶段，而不是只按 FFN 四个阶段粗分。

例如下列 attention policy 只控制补项：layer 0、CFG branch 0 的前7步，head 0 全修复、head 1 不修复、其余 head 修复50%的 K128 blocks；其他位置默认25% ACT2。block 由当前 Q/K/V 选择，不沿用旧步输出。这只是配置示例，不是已经达到质量门槛的推荐。每项 rule 包含完整配置，不与 default 合并；第一条匹配的规则生效。

```json
{
  "default": {"coverage": 0.25, "terms": 2},
  "rules": [{
    "layers": [0], "steps": [0, 7], "branches": ["0"],
    "config": {"coverage": 0.5, "terms": 3, "head_budgets": [[0, 1.0], [1, 0.0]]}
  }]
}
```

`--execution graph|eager` 默认 graph。`--debug off|timing|trace` 默认 off。off 不注册诊断 hook、不创建诊断 CUDA Event、不收集轨迹。已知调度表保留 CPU，模型每轮仍接收当前 timestep 张量。域 ID 在每请求第一次 Transformer 调用时检查一次。timing 预分配 Event，记录编码、逐调用 Transformer、解码，结束后读取；trace 额外保存 GPU 摘要、修复 mask 和 NVTX。debug 模式不进入正式性能排行榜。

## 离线资产与校准

```bash
tw-cosmos prepare --model "$MODEL" --config tools/joint_inference/configs/dual.json --output /tmp/prepared
tw-cosmos infer --model "$MODEL" --assets /tmp/prepared/assets \
  --config tools/joint_inference/configs/dual.json --case tools/joint_inference/cases/policy_table.json \
  --seeds 93001 --output /tmp/from-assets
```

资产 manifest 包含模型 config/index 身份、投影范围、每个文件 SHA256。W0 保留，W1 只存最终策略需要的支持并集；不足以支持新增预算时拒绝加载，必须重新准备。CPU 准备时展开，GPU 只保留所选 W1 block。全量 W1 的高预算方案实际接近双份 FP4 权重，不能宣称仍是单份 W4 存储。不同 A/M/cross 共用 W0，相同 W 支持共用 W1；多个不同非零 W 档位目前分别存储对应 GPU residual bank，尚非最小单一 bank。

当前 loader 先载入原始 BF16 pipeline，再替换生成投影；导出资产免去再次量化，但尚未实现直接从打包权重构造整个 pipeline。因此暖机显存不代表启动峰值，也不能据此保证32GB 5090能够加载所有配置。本轮完整模型性能仅在96GB PRO 6000上验证。

```bash
python tools/joint_inference/capture.py --help
python tools/joint_inference/calibrate.py --help
python tools/joint_inference/calibrate_attention.py --help
```

`capture.py` 只接受标为 development 的场景，保留完整 M 的教师输入。`calibrate.py` 接收 full-four 离线资产，对 prescale、实际表示的尺度和激活尺度做无梯度搜索。`calibrate_attention.py` 逐 K128 block 干预并测量各 head 的误差改善，再输出配额候选；它不能证明多块联合修复收益，也不能代替完整轨迹验证。两者输出均须在独立验证集选参后冻结。最终种子不得参与校准。

## 可复现实验

小输入可用 `python tools/optimization_handoff/prepare_inputs.py` 获取；模型直接下载到本机，不随数据包传递。`cases/` 使用相同场景、输入、CFG、分辨率和35步。validation 使用92001–92008；最终 quality 使用91001–91008。这是已有场景的新种子测试，不证明跨场景泛化。

模型仓库与固定 revision 见 [models.json](models.json)，不可用后来更新的 `main` 替代本轮基线。仅需同步代码、`cases`、设备调优表和必要输入；模型与本机离线资产重新生成。更换 GPU 后原调优表会被拒绝，先运行 tuning 再 benchmark。各模型实际 safetensors 的 SHA256 已随报告归档。

```bash
python tools/joint_inference/matrix.py --model "$MODEL" --model-name edge \
  --suite quality --configs bf16 fp8 base act2 dual four --output /tmp/edge-quality
python tools/joint_inference/evaluate.py /tmp/edge-quality/matrix.json \
  --output /tmp/edge-quality/quality.json --lpips-cache /path/to/lpips-cache
python tools/joint_inference/matrix.py --model "$MODEL" --model-name edge \
  --suite tuning --configs bf16 fp8 base dual dual_legacy ffn_only --output /tmp/edge-tuning
python tools/joint_inference/matrix.py --model "$MODEL" --model-name edge \
  --suite performance --configs bf16 fp8 base dual dual_legacy ffn_only \
  --tuning-root /tmp/edge-tuning/profiles --output /tmp/edge-performance
python tools/joint_inference/summarize.py /tmp/edge-performance/matrix.json \
  --quality /tmp/edge-quality/quality.json --output /tmp/edge-performance/summary.json
python tools/joint_inference/matrix.py --model "$MODEL" --model-name edge \
  --suite cold --configs bf16 fp8 base dual --tuning-root /tmp/edge-tuning/profiles --output /tmp/edge-cold
```

Nano 使用 `--model-name nano`；质量矩阵包含 four 与 four_dual_attention，性能矩阵另传 `--case-configs tools/joint_inference/configs/nano_performance_matrix.json`，按独立验证集预先确定各任务候选。默认增加 AV；性能同时运行 latent-only 和波形解码两个口径。性能每配置单独加载/预热，再完整执行一次35步暖机请求，固定随机顺序交错5轮，每轮3个固定种子，边界同步计时。CPU 请求状态/参数构造、图像读取、编码、最终输出转换和解码全部计入请求；结果 JSON/NPZ 写盘和离线质量指标不计入。正式结果必须标记 `complete_request_v2_including_host_setup`；早期排除 CPU setup 的样本不能进入排行榜。冷加载、权重准备、建图/预热单列。

调优身份包含 GPU UUID/型号/SM、驱动、功耗上限、CUDA/PyTorch/Triton、内核源文件哈希，以及实际 shape/修复语义。GPU0 和 GPU3 分开保存；换卡或改内核需重新调优。普通预热可使用有明确 miss 记录的默认 launch，正式 benchmark 不允许遗漏。

`tune_ffn_tiers.py --model "$MODEL" --case tools/joint_inference/cases/policy_table.json --output /tmp/all-tiers` 使用真实请求覆盖 M416 上下投影的全部8种 FFN 修复档位；不把旧 M316 或合成 M416 的性能冒充实际请求测量。

`cold` 记录从父进程启动到子进程完成 Graph 准备的时间，包含解释器/库导入；保留磁盘 kernel cache，不能当作全新环境首次编译时间。`verify_tuned.py` 将冻结调优后的35步输出与原始质量轨迹逐位对照；它只验证等价性，不重新选参。`archive.py` 保留原始数值行、每次加载/准备/显存测量及文件哈希，省去模型权重和视频/音频大文件。

如果正式矩阵被独占检查中断，在同一 GPU 恢复空闲后运行 `python tools/joint_inference/resume.py /path/to/matrix.json`。它保留此前每次尝试，按原有随机顺序重测整个未完成轮次；不能把缺轮或混合设备的数据填入正式报告。

质量检查严格按配对 bootstrap 95% CI：video LPIPS 改善且 SSIM/L2 不退化；policy 还需 action L2 改善且 RMSE 不退化；AV 还需 log-spectral error 改善且 waveform/sound latent L2 不退化。缺少任何指标即失败。没有通过的任务返回 null 推荐。

## 检查和消融

```bash
pytest -q packages/fp4-repair/tests
compute-sanitizer --tool memcheck --error-exitcode 99 python packages/fp4-repair/tests/sanitizer_joint.py
compute-sanitizer --tool racecheck --error-exitcode 99 python packages/fp4-repair/tests/sanitizer_joint.py
python tools/joint_inference/verify_debug.py --help
python tools/joint_inference/ablate.py --help
```

`ablate.py` 固定量化策略和调优表，比较 eager、Graph、timing、trace，以及复现历史逐调用 `.cpu().numpy()` 摘要和 GPU timestep 回读的探针模式，强制输出 hash 一致。带探针数字只用于开销消融。内核改进用同条件 legacy/optimized 的完整请求比较。各阶段收益不能相乘。

常驻量化权重、支持 buffer、准备后 allocated/reserved、建图峰值和请求峰值分别记录。Graph 池包含激活中间量和 scratch；不能把 reserved 当作实际权重占用，或将不同峰值相减解释为精确独占 scratch。

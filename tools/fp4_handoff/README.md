# 冻结 Edge FP4 修复与 ARCQuant / SharQ 交付

交付当前已测的 1.5×、2×、2.5× 推理策略，以及 ARCQuant 50%/100%、SharQ `extra_fusion=True` 的 Cosmos 适配。默认 debug off，保持 35 steps、CFG 6、192×320、61 帧。所有方法使用同一固定版本 Edge，AR 缓存预填充使用 BF16，每次请求重新执行。

本交付分支以 `perf/rtx5090-cutlass-followup` 的 `71b1871` 为基础，承接已有推理框架；没有覆盖较早的 main。历史实测见 [FPS 与质量](../../results/fp4_baseline_sweep_20260918/FPS_QUALITY.md)、[完整结果及置信区间](../../results/fp4_baseline_sweep_20260918/REPORT.md)。

进一步优化保留原资产和修复策略，通过独立 manifest 开启经过输出一致性验证的融合与 Graph 配置。见 [优化运行说明](OPTIMIZATION.md) 和 [PRO 6000 复测报告](../../results/fp4_runtime_next_20260918/REPORT.md)；实验性新选择器未通过质量门槛，默认选择器不变。

| CLI 方法名 | 修复预算 | 生成投影实现 | attention | 执行 |
|---|---:|---|---|---|
| `b1.5_protected_n` | 1.5× | 原生 Triton FP4，冻结离线 W0/W1 | FP4，独立 QK/PV 与 per-head 配额 | Graph |
| `b2_joint` | 2× | 同上 | 同上 | Graph |
| `b2.5_ffn_guarded` | 2.5× | 同上，保留 FFN 修复保护 | 同上 | Graph |
| `arcquant` | 外部方法，非同预算 | 官方 CUTLASS FP4，50% 激活残差通道 | BF16 | eager / 默认流 |
| `arcquant100` | 外部方法，非同预算 | 官方 CUTLASS FP4，100% 激活残差通道 | BF16 | eager / 默认流 |
| `sharq_perf` | 外部方法，非同预算 | 官方稀疏 FP4 + 残差 FP4，现成 epilogue 融合 | BF16 | eager / 默认流 |
| `bf16` | 质量参考 | 原生 BF16 | BF16 | Graph |

预算是全请求生成投影与 attention 的 MMA/C0 上限，不是延迟倍数。四阶段为 `[0,7)、[7,18)、[18,28)、[28,35)`，两个 CFG 分支分别配置。FFN、attention、投影及 per-head 修复以 plan 中的整数 block 支持为准；配置中默认 activation/weight=0 不表示关闭 plan 内修复。

## 文件和必要数据

- [presets/](presets/)：完整配置和原始 plan 的 gzip 副本；解压得到的 plan 字节与原实验一致。
- [sources.lock.json](sources.lock.json)：ARCQuant、SharQ 和 CUTLASS submodule 的固定版本。官方源码由接收端从原仓库下载，本仓库保存适配器、构建脚本及 ARC K9216 小补丁。
- [release.json](release.json)：GitHub Release、文件大小、SHA256 和模型 revision。
- Release 数据包约 **28.4 MiB**：三个档位的校准参数、W1 支持、舍入字节修正，ARC 通道顺序，16 个 development 场景的条件首帧、prompt、种子及原输出哈希。
- 接收端从固定 HF 模型离线重建约 2.65 GiB 的三份部署资产，逐个核对全部 tensor 的 dtype、shape 和字节哈希。权重量化仍只在准备阶段进行。
- 不传原始 BF16 模型、完整 W0/W1 平面、生成视频或约 32 GB 的教师激活面板。模型由接收端下载。

ARC compact orders 与原先“16 个 full-M 教师面板取通道最大值后 stable argsort”完全一致；只是离线保存其结果，不改变通道选择。当前三份资产没有 mandatory outlier removal；recipe v1 遇到此类资产会明确拒绝，避免静默丢失 outlier。

## 接收端安装

Linux、Python 3.12、SM120/120a Blackwell、CUDA Toolkit 13.0，`/usr/local/cuda` 指向该 Toolkit。PRO 6000 和 RTX 5090 均需在本机重新编译扩展、预热并建图。还需要 `git`、`gh`、`cmake`、C++ 编译器及 GitHub/HF 的对应访问权限。仓库与 Release 为私有，使用已登录的 `gh` 下载。

```bash
git clone --branch feat/fp4-repair-baselines-20260918 https://github.com/Concyclics/ThunderWorld.git
cd ThunderWorld
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r tools/fp4_handoff/requirements.txt
python -m pip install --no-deps -e packages/fp4-repair
export TW_PYTHON="$PWD/.venv/bin/python"

bash tools/fp4_handoff/python.sh tools/fp4_handoff/download.py --output data_local/fp4-edge-handoff
bash tools/fp4_handoff/python.sh tools/cosmos_e2e/download.py --model edge --models-dir models
```

`download.py` 验证 Release SHA256、拒绝越界路径/符号链接，并验证包内所有文件。固定模型为 `nvidia/Cosmos3-Edge@a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba`。

```bash
# CPU 离线恢复；完成前不会将资产标记可用。已有输出目录不会被覆盖。
bash tools/fp4_handoff/python.sh tools/fp4_handoff/bundle.py restore \
  --package data_local/fp4-edge-handoff \
  --model models/Cosmos3-Edge-a9d944e --output data_local/fp4-restored

# 获取固定上游版本及 CUTLASS；构建保留官方现成实现。
bash tools/fp4_handoff/python.sh tools/fp4_handoff/baselines.py fetch \
  --package data_local/fp4-edge-handoff --sources data_local/baselines
bash tools/fp4_handoff/python.sh tools/fp4_handoff/baselines.py build \
  --package data_local/fp4-edge-handoff --sources data_local/baselines \
  --arc-build data_local/arcquant-build

# 将可移植路径绑定到本机；生成新的运行 manifest，历史 manifest 不改写。
bash tools/fp4_handoff/python.sh tools/fp4_handoff/bundle.py materialize \
  --package data_local/fp4-edge-handoff --restored data_local/fp4-restored \
  --model models/Cosmos3-Edge-a9d944e \
  --sources data_local/baselines/sources.lock.json --arc-build data_local/arcquant-build \
  --output data_local/fp4-study
```

只运行我们的方法时，`materialize --methods b1.5_protected_n b2_joint b2.5_ffn_guarded bf16` 可省略 `--sources` 和 `--arc-build`，也无需先构建外部基线。

## 正确性、质量和性能复测

```bash
# CPU 配置、资产与传输验证。
CUDA_VISIBLE_DEVICES= bash tools/fp4_handoff/python.sh -m pytest -q \
  tools/fp4_handoff/test_handoff.py packages/fp4-repair/tests/test_compute_budget.py

# GPU0 上逐方法、逐请求运行；两类固定输入、各两个原种子、完整 35 步。
CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/fp4_handoff/smoke.py \
  --study data_local/fp4-study
```

`smoke.json` 记录完整结果与源设备输出哈希是否一致。复现同一 PRO 6000 环境可增加 `--require-source-hash`，任何差异都会失败；跨设备默认记录差异，不将原设备质量套用到新设备。共享模式最多一个实验进程，启动前至少 12 GiB 可用显存；不会更改输入或降低精度以绕过 OOM。

测全部 16 个冻结 development 场景（不是预留 72 场景最终测试）时：

```bash
for method in b1.5_protected_n b2_joint b2.5_ffn_guarded arcquant arcquant100 sharq_perf bf16; do
  CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/fp4_handoff/infer.py \
    --study data_local/fp4-study --method "$method"
done
python -m pip install -r tools/fp4_handoff/requirements-quality.txt
CUDA_VISIBLE_DEVICES= bash tools/fp4_handoff/python.sh tools/contribution_study/dataset_metrics.py \
  data_local/fp4-study/manifest.json --generation data_local/fp4-study/generation \
  --output data_local/fp4-study/metrics
```

指标为相对本机同输入 BF16 的 PSNR、SSIM、LPIPS 与相对 L2。`infer.py` 支持续跑，校验 manifest、输入图像和输出哈希；会校验每请求的实际 MMA 预算。

完成 smoke 后，已有两个吞吐输入的本机质量参考，可运行同一并发扫描：

```bash
# PRO 6000：沿用原实验的 12 GiB reserve 默认值。
bash tools/fp4_handoff/python.sh tools/contribution_study/baseline_throughput_suite.py prepare \
  --study data_local/fp4-study --output data_local/fp4-speed
CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/contribution_study/baseline_throughput_suite.py run \
  --output data_local/fp4-speed

# 24 GiB RTX 5090：另建扫描目录，只改变副本准入的预留显存。
bash tools/fp4_handoff/python.sh tools/contribution_study/baseline_throughput_suite.py prepare \
  --study data_local/fp4-study --output data_local/fp4-speed-5090 --reserve-mib 2048
CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/contribution_study/baseline_throughput_suite.py run \
  --output data_local/fp4-speed-5090
```

每个方法/输入依次测 B=1/2/4/8，每档 5 轮×3 波；显存不足时停止增加副本。B 是独立模型副本的请求并发，不是共享权重或合并 GEMM 的真 batch。B>1 只测速度；B1 核对本机质量输出。输出 FPS = 61 × 视频/秒，包含 AR 预填充、35 步、VAE 解码和 CPU 输出转换，不包含模型加载、建图和写盘。

所有结果保留 `performance_eligible=false`，用于固定实现诊断。原来的 PRO 6000 reserve 默认值不变，5090 参数只调整副本准入；未以此修改内核或获得新的速度结论。阶段性单请求质量不能直接和未验证一致性的并发 FPS 拼成同质量加速。ARC/SharQ 的默认流与 eager 成本要随结果披露；详见 [并发诊断](../../docs/BASELINE_CONCURRENCY_DIAGNOSIS_20260918.md)。

## 实现入口

- 核心可安装包：[`packages/fp4-repair`](../../packages/fp4-repair)，统一 `CosmosRunner.prepare/warmup/infer`、Graph、预算账本与流式 FP4 attention。
- 三档离线配置：[`presets`](presets)；线上 `RepairPlan` 控制 FFN/投影和每层每 head 的 QK/PV 修复。
- 基线适配：[`external.py`](../contribution_study/external.py)、[`baseline_runtime.py`](../contribution_study/baseline_runtime.py)。`precision=bf16` 在外部基线配置中是安装前的调度占位；对应投影实际替换为官方原生 FP4 内核，attention 保持 BF16。
- ARC 构建：[`build_arcquant.py`](../contribution_study/build_arcquant.py)、[`patches`](patches)，只增加 Cosmos K9216 dispatch/instantiation。
- SharQ 构建：[`build_sharq.py`](../contribution_study/build_sharq.py)，保留上游实现，启用已有 `extra_fusion=True`；未移植 Cosmos RMSNorm 融合。
- 发布端导出：`bundle.py export --study ORIGINAL_STUDY --output NEW_PACKAGE`；只读源实验，拒绝覆盖输出。
- 本轮验证回执：[`validation.json`](validation.json)。原始吞吐 worker 保存在结果目录的 `measured_sources/`，可与历史哈希核对。

当前推荐需要按用途选择：1.5× 吞吐优先，2× 质量/吞吐折中，2.5× 质量优先。1.5× 未全面超过 AWQ/ARC100；2.5× 尚未通过超过 FP8 的完整质量门槛。质量和速度结论限定于结果报告中的模型、场景和执行条件。

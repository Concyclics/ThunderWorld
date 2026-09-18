# 双模型质量评测入口

本入口运行已冻结的 Edge / Nano I2V 质量矩阵，没有性能测试模式。主矩阵为 BF16、
FP8、普通 W4A4、NVFP4 AWQ-lite 精确解码参考、ARCQuant 50%/100%、SharQ、三档联合修复
1.5×/2×/2.5×，以及全量三项/四项锚点。每个模型 12 个方法、72 场景、每场景两种子。
质量参考是各自模型同输入同种子的 BF16，不是任务成功率。

## 本机环境与冻结数据

仓库根目录下使用已有 Python 包装器：

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
```

本次运行目录为 `/home/yufeng/thunderworld/full_quality_matrix_20260919`，持久校准与资产目录为
`/home/yufeng/thunderworld/full_quality_ready_20260919`。
输入来自此前冻结的 calibration/development/validation/final cohort；最终种子仍为
301113–301256。模型与图片在仓库外，GitHub 只保存工具、配置和小型证据。

`prepare` 复制并逐文件核对离线资产与预算计划。运行 manifest 不再依赖临时目录的
修复资产。外部 baseline 的源码、构建产物和校准身份仍显式固定，执行类型记录在 manifest。
Nano 独立生成 36 层、32 Q heads、8 KV heads、SwiGLU 的资产及配额，不能使用 Edge 资产。

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py prepare \
  --model edge --root /home/yufeng/thunderworld/full_quality_matrix_20260919 \
  --asset-root /home/yufeng/thunderworld/full_quality_ready_20260919 --awq-mode reference
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py prepare \
  --model nano --root /home/yufeng/thunderworld/full_quality_matrix_20260919 \
  --asset-root /home/yufeng/thunderworld/full_quality_ready_20260919 --awq-mode reference
```

Nano 从原始模型重新准备时，先完成下方 ARC 构建，再使用 `nano_quality_prepare.py`；它按固定次序串行运行
现有全层收益校准、模型结构适配后的 FFN 干预、预算编译、prescale/W0/W1 坐标搜索、
FFN 组合保护、ARC 通道顺序、官方 AWQ 校准与精确编码导出。局部收益不代表最终质量。
完整 M 和实际当前输入均保留；不训练、不在 final 上选参。

```bash
export PYTHONPATH=/home/yufeng/thunderworld/contribution_baselines_20260917/modelopt:/tmp/tw-contribution-deps:/tmp/thunderworld-joint-deps
bash tools/fp4_handoff/python.sh tools/contribution_study/nano_quality_prepare.py \
  --root /home/yufeng/thunderworld/full_quality_ready_20260919/nano --phase all --awq-mode reference
```

该工具使用已冻结的 `nano/cohort.json`。首次在另一工作区使用，需按原 cohort 固定全部
case/seed，仅更新模型与输入路径并核验 SHA。ARC 的新构建需显式包括 K12288；既有
K4096 分派保留，不改变上游量化定义：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/build_arcquant.py \
  --sources /home/yufeng/thunderworld/contribution_baselines_20260917/sources.lock.json \
  --output /home/yufeng/thunderworld/full_quality_ready_20260919/nano/arc_build \
  --pybind11 /tmp/tw-contribution-deps/pybind11/share/cmake/pybind11 \
  --widths 9216 12288
```

不要覆盖已完成的构建目录或旧实验目录。可恢复阶段保持源文件、面板与资产指纹不变。
准备完成后，将 `PYTHONPATH` 恢复为 `/tmp/thunderworld-joint-deps`；生成不依赖 ModelOpt。

## AWQ 的准确率口径

两个模型统一采用官方 ModelOpt `NVFP4_AWQ_LITE_CFG`，关闭激活量化，保存已验证的精确
解码权重。`awq_reference` 是 W4A16 **准确率参考**：量化数值来自官方 AWQ，运行时缓存
解码后的 BF16 权重，其常驻存储和耗时不能代表 packed W4 部署。它不属于联合 FP4 方法。
ARCQuant 与 SharQ 仍执行已验证的官方原生算子及 Cosmos 适配器。

本轮 Nano 的原生 Marlin 资格检查发现，小于 `1/64` 的 E4M3 块尺度可能在其尺度转换中
清零。首层 gate 的 6.62% 块尺度受影响，M129 随机输入上 NMSE 为 `1.09e-3`；按同样
规则清零参考尺度后，NMSE 降为 `6.62e-6`。因此不放宽原生资格阈值，也不把该偏差算作
AWQ 算法的质量。诊断及旧 Edge Marlin 结果保留在资产目录。`--awq-mode marlin` 只用于
单独研究目录，在全部算子资格通过前不得进入主矩阵；原生路径留待后续性能工作。

该诊断可以单独复现（与生成进程顺序运行；输出使用新路径）：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/awq_scale_probe.py \
  --assets /home/yufeng/thunderworld/full_quality_ready_20260919/nano/awq_native \
  --output /tmp/nano_awq_scale_recheck.json
```

## 资格检查与最终生成

两个模型分别检查开发集两个场景、每场景两个种子，并做 A→B→A 请求切换和完成后的
断点恢复。计算 LPIPS、SSIM、PSNR、相对 L2，自动生成与模型/方法数量一致的报告。
这一步不消耗最终种子，也不据此宣称质量超过某个 baseline。

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py qualify \
  --model edge --root /home/yufeng/thunderworld/full_quality_matrix_20260919
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py qualify \
  --model nano --root /home/yufeng/thunderworld/full_quality_matrix_20260919
```

资格通过后，以下命令运行该模型的 1,728 次正式质量生成及 CPU 评分。两个命令**顺序运行**；
不要在两个 GPU 上同时启动校准进程。自动从 GPU0/3 选择空闲显存较多的卡；显存不足或
OOM 保存暂停状态，不改变输入、步数或精度。

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py run \
  --model edge --root /home/yufeng/thunderworld/full_quality_matrix_20260919
bash tools/fp4_handoff/python.sh tools/contribution_study/quality_benchmark.py run \
  --model nano --root /home/yufeng/thunderworld/full_quality_matrix_20260919
```

资格检查与最终 manifest、运行代码哈希绑定；改过数值配置、内核或质量入口后不能沿用旧
资格。代码改动应使用新的资格输出目录，旧结果保持原样。当前 manifest 是新样本质量入口，
`source_output_hashes` 为可选项；只有显式 `--require-source-hash` 才要求每个输入都有历史
输出，缺失不能伪造为匹配。

底层兼容入口：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/dataset_infer.py /path/manifest.json \
  --method arcquant100 --output /path/generation/arcquant100
bash tools/fp4_handoff/python.sh tools/fp4_handoff/infer.py --study /path/study --method b2_joint
```

两者调用同一个质量 worker，接入同一 `baseline_runtime.make_runner`，完整保留
`runtime_options`、AWQ/ARC/SharQ adapter、Graph/eager 配置与计算预算。不会把不支持
Graph 的官方 host-alpha 路径强行捕获，也不会回退到 BF16 冒充 FP4。

## 恢复、报告与成本

- 每个 seed 原子写入 NPZ，保存输出 SHA、当前输入/配置、模型身份和预算账本。
- 恢复时校验已经完成的每个输出；完整方法恢复无需加载模型。改变输入、缺失种子、
  重复种子、损坏文件或改变账本均失败，不跳过。
- 部分完成的请求只生成缺少的种子；原始输出不重算。可使用
  `quality_resume_probe.py --study /path/model/smoke` 在独立副本上验证这一点。
- 结果位于 `<model>/final/generation`、`metrics`、`report/quality.json` 及 `report/REPORT.md`。
  `qualification.json` 只说明入口资格，不说明已跑完 final。
- RBench/PAI 分开报告，先平均场景内种子、再平均类别；使用场景配对、类别分层 bootstrap。
  PSNR 完全相同时 JSON 为 null 加 exact 标记，报告为 +∞，不任意截断。
- 预定义 ARCQuant100 和 FP8 两个独立比较族，每族 3 预算 × 2 模型 × 2 数据集；质量门槛
  保留 LPIPS 校正区间、SSIM/L2 guard。其他对照报告描述性差值，不事后新增显著性主张。
- 所有耗时/分配器峰值均标为共享环境诊断，`performance_eligible=false`。本入口不做并发、
  调优、速度榜或设备间性能比较。

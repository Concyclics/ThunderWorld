# I2V 裁剪和扩大量化实验

这一套脚本创建隔离副本，不改 `packages/fp4-repair`、旧的 Thor v8 或已冻结的质量资产。

## 范围和口径

- `trim` 删除 I2V 不执行的 LM head、动作/音频投影和 modality embedding；入口拒绝 policy/AV 请求，删掉的模块若被调用会报错。BF16 对照应用相同裁剪。
- AR 的 Q/K/V/O 和 FFN 使用现有原生 FP4 内核，可选单项、三项、四项。主实验选四项以先保护条件分支；沿用 BF16 causal attention，CFG 的两个 prefill 每个请求重新执行。
- embedding 采用 K16 E2M1/E4M3、每行 FP32 外尺度。只解码当前 token 的行，无完整 BF16 表缓存。双平面表示 `E0 + E1`，不称为单平面 4-bit 存储。
- 小型输入/输出投影和 timestep 权重采用压缩权重、调用时解码；timestep 的计算保留 FP32。
- VAE 包括实际使用的 **Wan** causal Conv3d 和 Conv2d。保留原 padding、feature cache、tiling；每次只解码当前卷积权重，执行仍是 BF16 cuDNN。这是 **FP4 权重压缩 / W4A16 验证路径**，不是原生 W4A4 卷积加速。禁止 codec Graph，避免私有 Graph 池长期保留逐层反量化工作区。
- norm、bias、rotary 常数等保留原精度，`expansion.protected` 枚举实际未量化参数。不存在隐藏 BF16 原权重副本；离线 CPU staging 在完整预热后释放。
- `1.5×/2×` 仍只标识已冻结的生成分支与 attention 预算；扩展 AR 计算单独列账，不能宣称新全请求仍有相同预算。
- `asset_dir` 是带源权重 hash 的离线缓存，不是一个已支持独立加载的完整模型 checkpoint。首次运行仍读取原 checkpoint；不要将整个原模型目录加缓存后的磁盘占用当作部署权重压缩率。

## 配置

| 名称 | AR | embedding | 辅助矩阵 | VAE |
|---|---|---|---|---|
| original | BF16 | BF16 | 原精度 | BF16 |
| trim | BF16 | BF16 | 原精度 | BF16 |
| ar4_embed2 | FP4 四项 | 双平面 | 原精度 | BF16 |
| transformer2 | FP4 四项 | 双平面 | 双平面 | BF16 |
| full2 | FP4 四项 | 双平面 | 双平面 | 双平面权重，BF16 计算 |
| full1 | FP4 单项 | 单平面 | 单平面 | 单平面权重，BF16 计算 |
| vae1/vae2 | BF16 | BF16 | 原精度 | 单/双平面权重 |

除 original 外均有 I2V 裁剪。`full1` 是压缩上限负对照，必须检查质量，不能默认推荐。

## PRO 6000

使用冻结的本地 study manifest，模型、图片及 deployment 资产路径转换成绝对路径。每个模型需各自的 manifest/资产。禁止使用 Thor 调优表作为 PRO 6000 性能依据。

```bash
python tools/thor_handoff/expand_build.py \
  --parent packages/fp4-repair/src --output /abs/experiment/variant
CUDA_VISIBLE_DEVICES=3 python tools/thor_handoff/expand_qualify.py \
  --variant /abs/experiment/variant --output /abs/experiment/qualification.json
CUDA_VISIBLE_DEVICES=3 python tools/thor_handoff/expand_campaign.py \
  --root /abs/experiment --variant /abs/experiment/variant
TW_LPIPS_CACHE=/abs/experiment/lpips CUDA_VISIBLE_DEVICES=3 \
  python tools/thor_handoff/expand_report.py \
  --root /abs/experiment --output /abs/experiment/report --metrics
```

本轮 pilot 为 RBench/PAI-Bench 各一场景、每场景三个固定种子，320×192、61 帧、35 步、CFG6、flow3。逐帧 LPIPS/SSIM、clip PSNR/L2 使用同条件 BF16 参考。这不是全量质量验收。

## Thor

```bash
source /home/chenhan/thunderworld/src/ThunderWorld/tools/thor_handoff/env.sh
python expand_build.py --parent /home/chenhan/thunderworld/variants/cutlass-memory-v8 \
  --output /home/chenhan/thunderworld/variants/expanded-i2v-v9
python expand_qualify.py --variant /home/chenhan/thunderworld/variants/expanded-i2v-v9 \
  --output /abs/experiment/qualification/kernels.json
compute-sanitizer --tool memcheck --error-exitcode 77 python expand_qualify.py \
  --variant /home/chenhan/thunderworld/variants/expanded-i2v-v9 \
  --output /abs/experiment/qualification/memcheck.json --sanitizer
compute-sanitizer --tool racecheck --error-exitcode 77 python expand_qualify.py \
  --variant /home/chenhan/thunderworld/variants/expanded-i2v-v9 \
  --output /abs/experiment/qualification/racecheck.json --sanitizer
python expand_prepare_thor.py \
  --fixed-root /home/chenhan/thunderworld/comparisons/memory_20260923_v8/campaign/models \
  --variant /home/chenhan/thunderworld/variants/expanded-i2v-v9 --output /abs/experiment
python expand_campaign.py --root /abs/experiment \
  --variant /home/chenhan/thunderworld/variants/expanded-i2v-v9 \
  --jobs /abs/experiment/jobs.json --require-idle
```

Thor 只验证算子正确性、速度和内存，不评完整视频质量；忙时 `--require-idle` 拒绝启动测量。所有本轮耗时为诊断，不进入正式排行榜。每个配置独立进程、充分预热，不在计时区调参或编译。

## 内存与实验完整性

`result.json` 保存源代码/资产 hash、配置、每个视频 NPZ/hash、请求计时和各阶段内存。`resident_model_bytes` 去重计算真实 GPU 存储，含 AR 原生 op 的代码、尺度和索引。

`peak_non_model_remainder_bytes` 包括 Graph 池、输入、输出、激活、临时反量化权重与 workspace，**不能写成“峰值激活”**。独立诊断请求提供编码/解码阶段增长和 AR/去噪区间，并断言探针没有改变输出。正式请求无这些探针。

比较 full2 与 transformer2 才能分离 VAE 的增量代价；比较 ar4_embed2 与 trim 分离 AR/embedding；裁剪收益同样属于 BF16 对照。共享负载、DVFS 或 cuDNN 执行变化下的耗时波动应保留，不据此声称裁剪加速。

## 主表：排除 VAE 的 Cosmos3 主体

`expand_body.py` 和 `expand_body_campaign.py` 提供单独的主体口径：

1. 全部方法使用相同的 BF16 VAE 编码。编码在测量前完成，保留固定图像的条件 latent。
2. 正常路径生成当前配置的参考 latent。随后真正删除 VAE 及其 feature cache，只保留无权重的配置边界，弱引用检查保证 VAE 对象没有被隐藏保留。
3. 验证删除 codec、使用固定 BF16 条件输入前后，35 步输出 latent **逐位一致**。
4. 每个测量请求重新执行两个 CFG 分支的 AR prefill 和全部 35 个 DM 步骤；不重用去噪结果。
5. 报告主体常驻模型、主体 ready allocated、主体 peak allocated/reserved。峰值包含 AR/KV、Graph、输入/输出 latent、激活和 workspace；不是整进程峰值减去 VAE 权重，也不是纯激活内存。

```bash
CUDA_VISIBLE_DEVICES=3 python tools/thor_handoff/expand_body_campaign.py \
  --root /abs/experiment --variant /abs/experiment/variant
```

主体耗时含 AR/DM、scheduler 和输入整理，排除编码、解码与视频输出转换，**不得标为端到端视频延迟/FPS**。完整生成质量主表继续使用 BF16 VAE；VAE 量化是单独消融。

默认主体矩阵的 BF16、FP8、W4A4、1.5×、2×均应用相同裁剪，前五行保留相同的历史生成投影量化范围。额外的 `transformer2` 行扩展 AR/embedding/辅助矩阵，是覆盖范围扩展消融，需要另过质量门槛，不能与历史档位混为一个配置。

主体结果汇总命令：

```bash
python tools/thor_handoff/expand_body_report.py \
  --root /abs/experiment --output /abs/experiment/report
```

独立恢复视频质量（只在 PRO 6000 上执行，Thor 本轮不评完整视频质量）：

```bash
TW_LPIPS_CACHE=/abs/experiment/lpips CUDA_VISIBLE_DEVICES=3 \
  python tools/thor_handoff/expand_body_quality.py --root /abs/experiment
```

此命令读取主体测量保存的 latent，使用原 BF16 VAE 解码，先要求 BF16 控制与完整流程视频逐位相同，再评价候选。旧 latent NPZ 将 BF16 无损存为 FP32；旧格式没有记录原 dtype 时，通过 BF16 golden hash 确认后处理 dtype，所有候选固定同一规则。它不重新执行去噪，也不改写先前内存结果。指标为全帧 LPIPS/SSIM、全视频 PSNR/相对 L2；自比较 PSNR 以 JSON null + `psnr_exact` 表示，不画有限值。

本轮完整口径、主表、VAE 负结果及扩展主体的质量代价见 [实验说明](../../docs/paper/manuscript/I2V_TRIM_AND_BODY_MEMORY_20260923.zh.md)。原始 JSON 压缩存档位于其 evidence 子目录，解压后校验值在 `raw/index.json`。

若已有全流程队列运行，`expand_switch_body.py --root ... --variant ...` 只在本实验 supervisor 没有子进程的等待边界切换；它验证 PID 的命令行，不中断正在生成的 worker，也不操作其他用户进程。

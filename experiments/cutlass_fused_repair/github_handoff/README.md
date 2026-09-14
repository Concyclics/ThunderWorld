# GitHub交接入口

代码分支：[`feat/joint-inference-runtime`](https://github.com/Concyclics/ThunderWorld/tree/feat/joint-inference-runtime)。
交接包：[`cutlass-5090-handoff-20260914` Release](https://github.com/Concyclics/ThunderWorld/releases/tag/cutlass-5090-handoff-20260914)。这是实验性预发布，不表示通过FP8质量或独占性能验收。

仓库为私有仓库，接收机器需要具有读取权限，并通过`gh auth login`认证。然后执行：

```bash
gh repo clone Concyclics/ThunderWorld -- --branch feat/joint-inference-runtime
cd ThunderWorld
python experiments/cutlass_fused_repair/download_github.py --output ../cutlass-5090-download
```

默认下载全部四个附件并核对本目录`release.json`记录的大小和SHA256：

- `cutlass-5090-core.tar.gz`：约13.2 MiB，代码、168个投影的两档离线配方、冻结策略、输入图片及复测说明。
- `cutlass-5090-panels.tar.gz`：约218 MiB，M416/M960真实四阶段算子数据；通过Release交付，不写入普通Git历史。
- `archives.json`、`verification.json`：原始包校验清单及本机解包验证记录。

只需要完整模型推理时，可以给下载命令增加`--core-only`；已有下载可用`--verify-only`再次校验。原模型继续按固定revision从Hugging Face下载。资产、内核和调优表的重建命令见[5090说明](../README_5090.md)。

核心包冻结于源码提交`24322c0`，后续提交只增加GitHub下载入口及发布索引，包内容没有被重写。解压核心包会得到独立目录`ThunderWorld-CUTLASS-5090/`；面板包解压到同一父目录，得到`panels/`，运行算子复测时传入实际的面板路径。

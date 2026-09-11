# Cosmos3-Edge / Nano 端到端对照

[本机实测报告](../../docs/COSMOS3_E2E_5090.md) 与 [10 个正式配置结果](../../results/cosmos_e2e_20260910/TABLE.md)。

比较当前固定 checkpoint 的完整图生视频 Generator：图像 VAE 编码、文本分词、全部去噪步、VAE 解码。
权重加载、离线 W4 打包、JIT/调参和 MP4 编码在计时外；不运行 prompt upsampler、音频和 safety checker。

## 模型差异与接入边界

| 模型 | revision | 层数 | hidden / FFN | 激活 | Hq / Hkv / D |
|---|---|---:|---|---|---|
| Edge | a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba | 28 | 2048 / 9216 | ReLU²，无 gate | 16 / 8 / 128 |
| Nano | 411f42a8fdfb8c5b2583cb8786e0938f49796eaa | 36 | 4096 / 12288 | SiLU(gate) × up | 32 / 8 / 128 |

仅替换每层 `mlp_moe_gen` 中的线性层，以及非因果 generation attention；保持原始非线性、
RoPE、QK norm、文本分支的因果 attention、理解分支 FFN、QKV/out projection 与调度器路径。
每次请求检查替换算子的实际调用次数，避免静默未接入。

此端到端试验的 FFN 使用**当前模型自己的 RTN W4 + full ACT2，identity prescale，无 W1**。
这与原 Nano layer-0 输入包的 AWQ prescale / 静态校准支持不同；不将旧校准统计迁移至 Edge 或其他层。
joint 方案另启用 75% 双侧一阶 attention 修复，所有 head 使用均匀预算，不移植 Nano head atlas。

同模型内，原始与优化 FP4 使用相同离线权重、修复预算、输入和 seed，只改变执行实现。
FFN 在真实首次输入形状上选择 M tile/流水级数，后续层复用形状参数。
`--allow-split-k` 是独立实验开关，默认关闭：小的舍入差异可能经过多层在线 FP4 量化放大。
默认保持 K 累加顺序；attention 只转置 V 的物理打包布局，不改变 splits 或逻辑分组。

## 本机环境

复用已有 PyTorch 2.11.0 / Triton 3.6.0，隔离环境在 `data_local/e2e_venv`。
Diffusers 固定源码提交 `04e724ce518e087301052b65ff81ec35d54f4040`。
本机继承的可选 xformers wheel 与 Torch 不兼容，入口只关闭该可选后端发现，使用
PyTorch Flash SDPA；不修改共享环境。无需更新 CUDA Toolkit。

```bash
/home/chenhan/miniconda3/envs/vllm/bin/python -m venv --system-site-packages data_local/e2e_venv
git clone https://github.com/huggingface/diffusers.git data_local/e2e_src/diffusers
git -C data_local/e2e_src/diffusers checkout 04e724ce518e087301052b65ff81ec35d54f4040
data_local/e2e_venv/bin/python -m pip install -e data_local/e2e_src/diffusers \
  accelerate==1.15.0 av==18.1.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 scikit-image==0.26.0

data_local/e2e_venv/bin/python tools/cosmos_e2e/download.py --model edge
data_local/e2e_venv/bin/python tools/cosmos_e2e/download.py --model nano
```

输入使用 [官方文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3) 的
[robot_153.jpg](https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg)，
本机路径为 `data_local/e2e_inputs/robot_153.jpg`。每份结果记录输入 SHA256。

## 执行与比较

```bash
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh -m pytest tools/cosmos_e2e/test_adapter.py -q
bash tools/kernel_handoff/python5090.sh tools/cosmos_e2e/benchmark.py \
  --model models/Cosmos3-Edge-a9d944e --image data_local/e2e_inputs/robot_153.jpg \
  --steps 35 --frames 61 --height 192 --width 320 --rounds 3 \
  --offload resident --output runs5090/e2e_edge_replay
bash tools/kernel_handoff/python5090.sh tools/cosmos_e2e/benchmark.py \
  --model models/Cosmos3-Nano-411f42a --image data_local/e2e_inputs/robot_153.jpg \
  --steps 35 --frames 61 --height 192 --width 320 --rounds 3 \
  --offload resident --output runs5090/e2e_nano_replay
```

增加 `--cuda-graph` 可在所有 BF16/FP4 方案外使用相同的完整 transformer CUDA Graph。
每个 CFG 分支复制本次输入并返回独立输出，不缓存上一步的激活或预测；在替换权重前清空 graph。
图捕获、warmup 和 launch 搜索不计时，当前输入复制与输出 clone 均计时。
该模式只支持 `--offload resident`，需与 eager 结果分开报告。

本轮正式主表使用 `--cuda-graph`。两模型对应结果都完成后，可汇总和审核：

```bash
python tools/cosmos_e2e/summarize.py \
  --edge runs5090/e2e_edge_graph_final/result.json \
  --nano runs5090/e2e_nano_graph_final/result.json --output results/cosmos_e2e_20260910
python tools/cosmos_e2e/validate.py \
  runs5090/e2e_edge_graph_final/result.json runs5090/e2e_nano_graph_final/result.json \
  --output results/cosmos_e2e_20260910/validation.json
```

单 GPU 顺序执行。显存不足时可显式使用 `--offload model`，但同一模型的所有方案必须采用
同一 offload 策略，并将传输成本计入总时间。不能将含 CPU offload 的 BF16 延迟当作纯 GPU kernel 基线。

默认比较 BF16、FFN 原始/优化 FP4、FFN+attention 原始/优化 FP4。每个方案同一 seed 测三次，
另一个 seed 做完整轨迹输出对照；每种执行模式先用同形状两步预热。完整 35 步结果才用于质量比较，
两步 smoke 仅用于检查接入。当前脚本分方案顺序执行，结果保留样本范围，不把微小差异视为稳定提升。

输出 `result.json` 记录总墙钟时间、每次 transformer 的 CUDA Event 时间、显存、调用计数、
源码哈希、加载信息和实际启动参数，并保存两个 seed 的视频与数组。
PSNR/SSIM 是与同模型 BF16 或原始 FP4 视频的一致性，不是任务成功率或物理正确性评测。
不同模型的参数规模、默认文本模板及 flow schedule 有差异，横向绝对速度不能替代同模型优化倍数。

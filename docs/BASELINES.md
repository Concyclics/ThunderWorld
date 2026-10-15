# Baselines and fair comparisons

All comparisons use the same pinned base version, generation projection scope, input, seed, 35 steps, CFG and output shape. Each manifest records real execution differences. External methods are not forced into a fictitious identical FP4 budget.

| Method ID | Projection implementation | Attention | Interpretation |
|---|---|---|---|
| `bf16` | Native BF16 | BF16 | Same-input/seed reference |
| `fp8` | Native W8A8; per-tensor W, online per-token A via vLLM | BF16 | High-quality quantized reference |
| `fp4_base` | Native NVFP4 W4A4, no repair | FP4 base | Direct low-precision control |
| `b1.5_protected_n` / `b2_joint` | Native FP4 base + selected A/W residual products | Selected FP4 QK/PV repair | Released ThunderWorld profiles |
| `arcquant` / `arcquant100` | Pinned official ARC kernels, 50% / 100% residual channels, Cosmos adapter | BF16 | Native FP4; eager host-alpha API, offline channel permutation |
| `sharq_perf` | Pinned SharQ native sparse / dense residual kernels, existing epilogue flag | BF16 | Native mixed sparse/dense FP4; Cosmos RMSNorm remains outside |
| `awq_reference` | ModelOpt NVFP4 AWQ-lite W4A16, decoded weights cached offline | BF16 | **Quality only**, not native FP4 throughput or packed-storage evidence |

ARCQuant 100% is often labeled ARCQuant (2.0) in the paper. It is an activation residual-channel setting, not an exactly matched whole-request ThunderWorld budget. Both ARCQuant and SharQ use native FP4 paths; it would be incorrect to describe all baselines as BF16 dequantization methods.

## Fetch exact upstream versions

```bash
python tools/public/run.py collect-baselines --work work/edge \
  --names arcquant sharq modelopt fouroversix viditq sageattention nunchaku
```

[The checked-in lock](../configs/public/baselines.lock.json) pins commits. The collector records license/dependency hashes and rejects changed source revisions. It does not execute third-party code or install dependencies. Source trees live under ignored `work/edge/baselines/`; adaptations are separate build outputs / patches.

| Source | Role in this release |
|---|---|
| [ARCQuant](https://github.com/actypedef/ARCQuant) | Official CUDA implementation with Cosmos shape adapter; full-generation baseline |
| [SharQ](https://github.com/actypedef/SharQ) | Official CUDA implementation with Cosmos wrapper; full-generation baseline |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Official NVFP4 AWQ calibration definition; decoded quality reference |
| [Four Over Six](https://github.com/mit-han-lab/fouroversix) | Scale-selection mechanism control with the same residual support |
| [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q) | Inspiration for layer/time sensitivity allocation; mechanism reproduction, not an official Cosmos port |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Native FP4 attention operator probe; not a complete Cosmos baseline |
| [Nunchaku / SVDQuant](https://github.com/nunchux-ai/nunchaku) | Operator-level low-rank compensation control; high-precision compensation accounted separately |

## ARCQuant and SharQ

First finish full-M `capture` as in [CALIBRATION.md](CALIBRATION.md), then:

```bash
python tools/public/run.py reference-capture --model edge --work work/edge
python tools/public/run.py build-baselines --model edge --work work/edge --names arcquant sharq
```

Install CMake, Ninja, pybind11 and a compatible CUDA toolkit first. The builder initializes each pinned CUTLASS submodule. It derives K instantiations from the actual captured shapes; Super's wide-K ARC port partitions packing work across legal CTAs. Build receipts include library hashes and adaptation patches. Run the relevant operator reference checks (`arc_probe.py`, `external_probe.py`, `super_arc_codec.py`) before making new-device performance claims.

The build scripts accept `--nvcc /path/to/nvcc`; otherwise they use `CUDACXX`, `PATH`, then `/usr/local/cuda/bin/nvcc`. Official upstream stream/Graph limitations remain explicit. Do not substitute a Torch simulation when a native extension fails to build.

## NVFP4-AWQ W4A16

Install the collected Model Optimizer source in a compatible calibration environment, following its dependency requirements. Its configuration must expose `NVFP4_AWQ_LITE_CFG`; this is **not INT4 AWQ**.

```bash
python -m pip install -e work/edge/baselines/modelopt
python tools/public/run.py awq --model edge --work work/edge
```

`awq_prepare.py` disables activation/output quantizers **before** calibration, calibrates the weight quantizer and exports its decoded BF16 weights plus prescale. It verifies the exported forward is bitwise equal to the official calibrated forward for each captured panel. The public capture view uses the 16 calibration scenes; this is a fresh calibration protocol, not a claim of identical historical AWQ assets. The build receipt records the installed ModelOpt version and source identity.

`awq_marlin.py` contains a separate native fused-dequantization experiment. Only its qualified packed implementation can support performance statements. The `awq_reference` path above intentionally remains a quality anchor; its decoded tensor sizes and timings cannot be presented as AWQ deployment storage / latency.

## Freeze a comparison

```bash
python tools/public/run.py prepare --model edge --work work/edge \
  --methods bf16 fp8 fp4_base b1.5_protected_n b2_joint \
            arcquant arcquant100 sharq_perf awq_reference
```

Follow [REPRODUCE.md](REPRODUCE.md) for generation, float-output metrics and result export. Every native baseline's dependency and build identity must match its manifest. Shared runs are quality experiments, not official latency comparisons.

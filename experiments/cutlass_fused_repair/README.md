# CUTLASS sparse repair integration

GitHub同步入口：[下载交接包并校验](github_handoff/README.md)。

The packaged experimental backend is `thunderworld_fp4.cutlass`. It implements
`A0W0 + selected(A1)W0 + A0 selected(W1) + optional selected(A1)selected(W1)` in one CUTLASS GEMM launch. The output is BF16 after a single epilogue; all terms share FP32 registers. No FP32 intermediate is written to global memory. W1 remains an offline compact support union. A1 and its mask consume current request inputs.

Dense tiles use CUTLASS SM120 TMA cooperative loading. Repair tiles use consumer-side `cp.async`, separate shared storage, and CuTe `ldmatrix`/native block-scaled FP4 MMA. The separate storage is necessary: a persistent producer may already be loading the next output tile into the base pipeline. Named barriers synchronize all consumer warps without blocking the producer group. Full W1 has regular addressing; partial supports use the existing CSR bank. No-repair calls use a separately instantiated dense path and do not reserve repair shared memory.

Two offline binaries expose M128×N128×K128 and M128×N64×K128. A/W support remains at 128×128; N64 executes the two column halves of each selected W block, preserving the total issued MMA for supported N multiples of128. N64 is not universally faster. With both libraries supplied, a topology heuristic proposes a launch; only exclusive `--tune` compares complete `_compute` calls and admits bitwise-equivalent candidates. The selected launch is frozen before Graph capture. Profile keys include device identity, shape, repair configuration, calibration semantics, source/header and binary hashes. Old Triton launch choices cannot be applied to the CUTLASS backend.

This is not full attention migration: QK/PV/softmax continue using the existing streaming FP4 attention. Online packing/selector remain Triton. The exact E4M3 scale byte permutation and outer-factor multiplication share a kernel, but that scale-layout pass is still separate from activation packing. Its reads/writes are included in the compute ledger. W0 scales in CUTLASS layout are prepared once and shared between phases and tile variants.

## Numerical scope

The CUTLASS path computes dense A0W0 before A1W0; the existing optimized Triton path interleaves A1W0 with dense K iterations. Quantized codes/supports are preserved, but accumulation order changes. Independent decoded reference tests and same-input tests are necessary, yet insufficient for trajectory quality equivalence. The two development seeds measured so far show small differences from Triton; neither1.25 nor1.5 is certified to exceed FP8. Do not silently select BF16 to hide this.

## Reproduction

See [5090 handoff](README_5090.md). `build.py`, `fused.py`, and `run_tests.py` are compatibility/convenience entries around the installed implementation. `probe.py` evaluates actual frozen calibration panels, including packing and selection. `retest.py` verifies portable data hashes and evaluates four phases. These diagnostics never enter the end-to-end leaderboard. `weight_recipe.py` avoids transferring calibrated weight tensors by rebuilding and checking every encoded tensor offline against its original hash.

Raw evidence and the measured limitations are recorded under `results/` and `REPORT.md`. Timing ratios always mean native Triton time divided by CUTLASS time; a ratio below1 is a regression. Final production recommendations still require independent quality validation and exclusive-device end-to-end measurements.

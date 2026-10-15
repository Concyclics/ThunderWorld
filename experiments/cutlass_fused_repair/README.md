# CUTLASS sparse repair integration

GitHub同步入口：[下载交接包并校验](github_handoff/README.md)。

The packaged experimental backend is `thunderworld_fp4.cutlass`. It implements
`A0W0 + selected(A1)W0 + A0 selected(W1) + optional selected(A1)selected(W1)` in one CUTLASS GEMM launch. The output is BF16 after a single epilogue; all terms share FP32 registers. No FP32 intermediate is written to global memory. W1 remains an offline compact support union. A1 and its mask consume current request inputs.

Dense tiles use CUTLASS SM120 TMA cooperative loading. Repair tiles use consumer-side `cp.async`, separate shared storage, and CuTe `ldmatrix`/native block-scaled FP4 MMA. The separate storage is necessary: a persistent producer may already be loading the next output tile into the base pipeline. Named barriers synchronize all consumer warps without blocking the producer group. Full W1 has regular addressing; partial supports use the existing CSR bank. No-repair calls use a separately instantiated dense path and do not reserve repair shared memory.

Two offline binaries expose M128×N128×K128 and M128×N64×K128. A/W support remains at 128×128; N64 executes the two column halves of each selected W block, preserving the total issued MMA for supported N multiples of128. N64 is not universally faster. With both libraries supplied, a topology heuristic proposes a launch; only exclusive `--tune` compares complete `_compute` calls and admits bitwise-equivalent candidates. The selected launch is frozen before Graph capture. Profile keys include device identity, shape, repair configuration, calibration semantics, source/header and binary hashes. Old Triton launch choices cannot be applied to the CUTLASS backend.

This is not full attention migration: QK/PV/softmax continue using the existing streaming FP4 attention. Online packing/selector remain Triton. The default path uses a separate E4M3 scale-layout/outer-factor pass. `--cutlass-direct-pack` folds that pass into the activation producer; both paths are accounted for in the compute ledger. W0 scales in CUTLASS layout are prepared once and shared between phases and tile variants.

`--joint-tuning --tune` searches both packing paths and both supplied CUTLASS
tiles. With zero activation-repair blocks it also considers Triton M32/M64/M128
and two/three stages. Activation-repaired calls stay within CUTLASS because the
native interleaved accumulation order differs. Each joint candidate must match
the anchor bitwise on three inputs; fresh interleaved timing must confirm a
greater than 3% improvement. The saved profile freezes execution choices, not
repair supports or budgets. Runtime requires no online timing or backend search.
Frozen compute plans additionally require candidate M padding to match the
M128 anchor: their strict issued-MMA ratio includes padding in its denominator.
Reducing base padding alone can exceed a nearly saturated ratio even while
reducing absolute work. Repair supports are never removed to admit a launch.
Use `--joint-tuning --tuning PROFILE` with `benchmark` for strict replay. Shared
gate/up packing is incompatible with this search. Profiles also bind the tuning
protocol, engine, and settings source files.

`joint_e2e.py` runs the Edge I2V 1.25/1.5 comparison against BF16 Graph, with
separate tuning, development-trajectory, and five-round performance stages.
`summarize_joint.py` audits full-request timing boundaries, matched inputs,
profile misses, source versions, and repeated output determinism. A local
candidate's bitwise check does not replace full-trajectory validation.
Tuning writes periodic `tuning.partial.json` checkpoints; an unfinished run must
still pass complete warmup and budget validation before its profile is usable.

## Numerical scope

The CUTLASS path computes dense A0W0 before A1W0; the existing optimized Triton path interleaves A1W0 with dense K iterations. Quantized codes/supports are preserved, but accumulation order changes. Independent decoded reference tests and same-input tests are necessary, yet insufficient for trajectory quality equivalence. The two development seeds measured so far show small differences from Triton; neither1.25 nor1.5 is certified to exceed FP8. Do not silently select BF16 to hide this.

## Reproduction

See [5090 handoff](README_5090.md). `build.py`, `fused.py`, and `run_tests.py` are compatibility/convenience entries around the installed implementation. `probe.py` evaluates actual frozen calibration panels, including packing and selection. `retest.py` verifies portable data hashes and evaluates four phases. These diagnostics never enter the end-to-end leaderboard. `weight_recipe.py` avoids transferring calibrated weight tensors by rebuilding and checking every encoded tensor offline against its original hash.

Raw panel evidence and the measured limitations are recorded under `results/` and `REPORT.md`. Panel timing ratios mean native Triton time divided by CUTLASS time; a ratio below1 is a regression. The [joint end-to-end follow-up](../../docs/JOINT_E2E_5090_20260914.md) separately reports BF16/control request time divided by candidate request time. Final production recommendations still require independent quality validation.

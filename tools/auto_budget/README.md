# Automatic repair budgets on RTX 5090

This experiment implements **FFN A/W precision budgets**, retaining BF16
attention after the preceding attention quality screen. It does not claim an
automatic attention policy, a bulk-fit quantizer, or a new hardware backend.

1. Capture a contiguous 64-row BF16 input/output probe for every Generator FFN
   linear at steps 1/11/23/33 of a 35-step trajectory. Record both CFG branches,
   real M/N/K, task, layer and checkpoint revision. Probes are local calibration
   evidence; they are not full-layer or full-trajectory quality measurements.
2. Measure six tiers: base, half A, full A, full W, half A + full W, full A + W.
   Keep the same model-local RTN base/residual planes and no cross term.
3. Allocate by positive local-error improvement per measured incremental
   latency, subject to a **separate per-task extra-FP4-product FLOP ceiling**.
   The greedy algorithm is deterministic, not globally optimal. A 50% ceiling
   permits one extra product on average, matching uniform full ACT2.
4. Tune M tile and pipeline stages for real shapes, validate outputs against
   the original launch, then recompute allocations with those measured costs.
5. During inference, resolve layer/phase/CFG tiers from a frozen table. For
   partial A, one Triton kernel ranks sampled current-input K-block energies and
   emits the mask on the GPU. No Python nonzero, host mask copy or weight
   repacking occurs per step. Selected products still execute NVFP4 Tensor Core
   operations. Outputs, Q/K/V and activations are never reused across steps.

Phases are normalized step ranges `[0,.2), [.2,.5), [.5,.8), [.8,1)` using ceil
boundaries. The tier schedule depends on task/layer/phase/branch; tile supports
also depend on the **current activation**. This is not online closed-loop
quality estimation or an arbitrary GPU-selected per-row compute budget.

The unrestricted allocator is preserved as a negative control. `guard_policy.py`
adds a half-A floor to every unit, then allocates the remaining budget. The
floor was introduced after unrestricted full-trajectory validation failed;
new evaluation seeds are used for the guarded policy. At 25% total repair
budget the floor consumes the whole allowance, so it reduces exactly to
uniform half-A and must not be advertised as a new allocation improvement.

`quality_gate.py` compares guarded schedules against the same-budget uniform
reference across video SSIM/LPIPS, action, audio latents and decoded waveforms.
If any validation seed exceeds the provisional tolerance, the entire task's
schedule falls back to its measured uniform reference. Defaults permit at most
0.02 SSIM loss, 0.01 LPIPS increase, and 5% + 0.001 extra relative/spectral error.
These are configurable engineering tolerances, not absolute quality criteria.
The resulting `policy_validated.json` is validation-selected and still needs
an independent final test; do not report these selection seeds as that test.

`--share-ffn-pack` shares Nano gate/up activation packing. It packs the union of
their current A supports once, preserves each GEMM's own repair mask, tier and
weight factor, and consumes the cache only within the same forward call and
layer for the same input. It does not reuse values across denoising steps.
`bench_shared.py` compares distinct real gate/up weights with alternating-order
CUDA Graph timings. All 36 tier pairs are checked for bitwise equality.

All tiers share one prepared W0/W1 allocation. W1 is retained even when unused
at a step, so a lower compute budget does **not** imply proportionally lower
weight memory. No Split-K is used. All six shape/tier specializations are warmed
before measured requests. Operator timings use CUDA Graph; the initial full
Generator comparisons use the same eager framework and include budget dispatch
and GPU selection overhead. The subsequent execution study below also compares
whole-Transformer Graphs. AV waveform decoding is separate.

Inputs `robot_observe`, `policy_observe`, `av_waterfall`, seed 7101 calibrate;
`robot_table`, `policy_table`, `av_pottery`, seeds 7103/7104 evaluate. The latter
scenes appeared in the earlier research, so they are excluded from this
allocator's calibration but are not completely unseen to the research process.

```bash
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh -m pytest tools/auto_budget/test_budget.py -q
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --cases robot_observe policy_observe --seeds 7101 --steps 35 \
  --capture-budget data_local/new_edge_capture --output runs5090/new_edge_capture
bash tools/kernel_handoff/python5090.sh tools/auto_budget/calibrate.py \
  --model edge --capture data_local/new_edge_capture --output runs5090/new_edge_calibration
bash tools/kernel_handoff/python5090.sh tools/auto_budget/bench.py \
  --model edge --capture data_local/new_edge_capture --output runs5090/new_edge_bench
bash tools/kernel_handoff/python5090.sh tools/auto_budget/retune_policy.py \
  runs5090/new_edge_calibration/policy.json runs5090/new_edge_bench/benchmark.json \
  --output runs5090/new_edge_calibration/policy_tuned.json
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --cases robot_table policy_table --seeds 7103 7104 --steps 35 \
  --modes bf16 ffn_act2 ffn_dual ffn_uniform25 ffn_uniform75 ffn_auto25 ffn_auto50 ffn_auto75 \
  --budget-policy runs5090/new_edge_calibration/policy_tuned.json --output runs5090/new_edge_e2e

# Compile the activation floor using the same calibration evidence.
bash tools/kernel_handoff/python5090.sh tools/auto_budget/guard_policy.py \
  runs5090/new_edge_calibration/policy_tuned.json \
  --output runs5090/new_edge_calibration/policy_guarded.json
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --cases robot_table policy_table --seeds 7203 7204 --steps 35 \
  --modes bf16 ffn_act2 ffn_uniform75 ffn_auto50 ffn_auto75 \
  --budget-policy runs5090/new_edge_calibration/policy_guarded.json --output runs5090/new_edge_guarded
bash tools/kernel_handoff/python5090.sh tools/finding_resample/perceptual.py \
  runs5090/new_edge_guarded --output results/new_edge_guarded
bash tools/kernel_handoff/python5090.sh tools/auto_budget/quality_gate.py \
  runs5090/new_edge_guarded --output runs5090/new_edge_calibration/policy_validated.json
```

For Nano, use the Nano capture/calibration directories, include `av_waterfall`
for calibration and `av_pottery` for evaluation, and run `decode_audio.py`
before the quality gate. Optional `--share-ffn-pack` is numerically verified;
its modest operator gain does not establish an eager end-to-end speedup.

Use `ncu_profile.py BENCHMARK_JSON ... --output FRESH_DIRECTORY` to measure
hardware counters. It explicitly uses CUDA 12.9's Nsight Compute; the older
system-default `/usr/bin/ncu` does not recognize SM120. No toolkit update is
needed. Tensor pipe elapsed-cycle activity and steady-state achieved TFLOP/s
are different metrics. Nsight replay/cache-control durations are not substituted
for the separately measured steady-state operator times.

Parameter audit and concurrency probe: [report](../../docs/PARAMETERS_AND_OVERLAP_5090_20260911.md).
`probe_overlap.py` compares one prepacked GEMM with another independent input's
selector/packer, serially and across two streams. It is not a same-input
pack-to-GEMM pipeline or an end-to-end implementation. `--trace --torch-trace`
records representative Edge M316 a50/act2 CUDA timelines; `analyze_overlap.py`
checks five replay groups per case and plots the median-span replay.

## Whole-Transformer Graph and FFN fusion

[Execution study](../../docs/GRAPH_FUSION_5090_20260911.md): `--phase-graph`
captures each phase/CFG branch during warmup, copies current inputs on every
replay, and updates actual budget counters. `--fuse-ffn` combines BF16 pointwise
operations with the down-projection input's amax reduction, preserving rounding
and the full-input scale. Both flags require a matching budget policy; Graph
currently supports BF16/FFN-only runs without attention sampling or capture.
The instance executes serial requests; clear it before changing weights/tasks.

```bash
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model nano --cases robot_table policy_table av_pottery \
  --seeds 7403 7404 --steps 35 --modes bf16 ffn_act2 ffn_auto50 ffn_auto75 \
  --budget-policy runs5090/auto_budget_nano_calibration/policy_validated.json \
  --phase-graph --fuse-ffn --output runs5090/new_nano_graph_fused
```

For a matched eager control omit both flags; for Graph alone omit `--fuse-ffn`.
Use separate fresh output directories, identical seeds, and the same policy.
For Edge use `--model edge`, its matching policy, and omit `av_pottery`.

`bench_execution.py --model edge|nano --output FRESH_JSON` measures complete
FFNs and real two-slot M-chunk pipelines using the existing tuned launch table.
`chunk_pipeline.py` remains experimental: none of the 120 tested candidates
beats the unsplit operator, so it is not enabled by these runtime flags.
`test_execution.py` checks bitwise equivalence, global scale, tail chunks,
slot reuse, changed-input replay, and phase boundaries.
`bench_chunk_tuning.py --model edge|nano --output FRESH_JSON` follows up on
the closest chunk for each shape/tier with BM32/64/128 and stages2/3, then
remeasures the selected launch against the original with alternating order.
This does not enable chunking in the inference runner.

# ThunderWorld optimization and repair-budget handoff

**Fresh baseline comparison:** [Original and tuned BF16 / FP8 / unrepaired FP4 versus ThunderWorld, including the current FP8 quality gap](ORIGINAL_BASELINE_COMPARISON_5090_20260914.md).

**Latest Edge I2V follow-up (2026-09-14):** [Validated pipeline flags, BF16/FP4 timing, quality gates and reproduction](NEXT_PIPELINE_TUNING_5090_20260914.md). These new measurements cover Edge I2V; the older Edge/Nano audit below retains its original scope.

**Release audit: 2026-09-11 · RTX 5090 / SM120 · Cosmos3-Edge and Cosmos3-Nano**

The compute implementation is ready as an **optimized research baseline**. It is
not a universal peak-performance implementation. Whole-Transformer CUDA Graphs,
shape-tuned native FP4 repair, and FFN pointwise/amax fusion form the supported
execution path. Attention precision allocation and residual calibration remain
research questions. A/W pairing is available as separately measured experimental
paths; those paths are not yet integrated into every budget/Graph combination.

Start with this document rather than the older September 9 input-package guide.
Historical reports retain the implementation state and evidence at their dates.
The [English paper report](../reports/thunderworld_paper_20260911/REPORT.md)
provides the motivation and figures; this document describes executable controls.

## 1. What is actually fast?

All ratios below compare the **same repair configuration** before and after an
execution change. They are not speedups against BF16, and must not be multiplied
across studies. Ranges span tasks/shapes, not confidence intervals.

| Change | Edge | Nano | Measurement and decision |
|---|---:|---:|---|
| Whole-Transformer Graph | 1.721–2.029× | 1.163–1.367× | Archived complete requests, ACT2/auto50/auto75; useful default |
| Graph + FFN fusion | 1.750–2.251× | 1.174–1.372× | Same archived eager reference; small Graph-to-fusion differences are noisy |
| FFN pointwise + amax fusion | 1.034–1.072× | 1.002–1.010× | Archived **complete FFN**, not whole request |
| Direct A pairing, ACT2 | 0.975–1.154× | 0.940–1.100× | **New frozen-choice retiming**, selector + packing + GEMM |
| Direct A pairing, dual | 0.945–1.094× | 0.891–1.070× | New retiming; shape-dependent, retain fallback |
| W pairing, W-only | 0.998–1.375× | 0.943–1.236× | New fixed-support packing + GEMM; numerical caveat below |
| W pairing, dual | 0.850–0.971× | 0.803–0.908× | New retiming: slower on every audited shape |

[Full per-tier speed audit](../results/optimization_handoff_20260911/SPEED_AUDIT.md)
includes half-A, half-A+W, native launch tuning, both layouts, and each Graph task.
The new audit contains **56 A configurations and 30 W configurations**, with five
randomized CUDA Graph timing rounds each. It reuses frozen launch choices rather
than selecting new winners on the audit data. A and W-plane outputs pass bitwise
checks; W-pair passes only the stated local error bound. Repeated captured rows
are timing inputs, not independent quality samples.

The latest regression run passed **126 tests**. Fresh two-step release smoke
runs exercise the committed runtime profiles and Graph/fusion entry point; their
results establish integration only, not 35-step quality or throughput.

Why further optimization is possible:

- Existing Nsight samples show tensor-pipe elapsed-cycle activity of about
  14.2–44.6% and achieved occupancy of 8.2–22.4%. These counters are **not a
  percentage of peak FP4 TFLOP/s**. They do not establish saturation.
- Small M, partial repair branching, input selection/packing, launch overhead,
  and output/nonlinear work have different costs by tier. No one launch wins all.
- Native base/A/W repair already accumulates in one FP32 accumulator inside one
  kernel. Pairing is not a reduction from three separate GEMM launches to one.
- The tested two-slot chunk pipelines all lost to the unsplit operator. Separate
  independent-stream overlap is measurable, but is not a same-input pipeline
  speedup. Do not enable chunking or assume SIMT work is automatically hidden.

## 2. Repair math and budget semantics

Using column-vector notation, the first-order approximation is

`Y = W0 A0 + mW · W1 A0 + mA · W0 A1`.

The code uses row-major linear inputs, so the corresponding products are
`A0 W0ᵀ`, `A0 W1ᵀ`, and `A1 W0ᵀ`. W0/W1 are offline weight planes;
A0/A1 are generated from the **current** activation. The cross term is omitted
in the six-tier budget runtime. Repair masks select residual products; the base
product remains dense.

| Tier | A repair fraction pA | W repair fraction pW | Normalized budget ρ | Ideal product count relative to base | Uniform runtime mode |
|---|---:|---:|---:|---:|---|
| base | 0 | 0 | 0 | 1 | `ffn_base` |
| a50 | 0.5 | 0 | 0.25 | 1.5 | `ffn_uniform25` |
| act2 | 1 | 0 | 0.5 | 2 | `ffn_act2` / `ffn_uniform50` |
| wgt2 | 0 | 1 | 0.5 | 2 | `ffn_wgt2` |
| a50w | 0.5 | 1 | 0.75 | 2.5 | `ffn_uniform75` |
| dual | 1 | 1 | 1 | 3 | `ffn_dual` |

Here `ρ=(pA+pW)/2`; cost is `1+2ρ` ideal FP4 products, **not measured latency**.
`ffn_auto25`, `ffn_auto50`, and `ffn_auto75` allocate these discrete tiers across
layer × denoising phase × CFG branch, under a weighted **per-task extra-FLOP
ceiling**. An auto50 setting is not 50% of attention tiles, nor a millisecond
budget. The guarded auto25 floor consumes its entire allowance and therefore
reduces to uniform half-A.

The policy is a frozen schedule, not an online quality estimator. Partial-A
support is selected from the current activation on the GPU. All tiers retain
the prepared W0/W1 allocation, so reducing compute does not proportionally reduce
resident weight memory. Record `budget_actual`, not just the requested label.

## 3. Adjustable controls and code locations

| Control | Where / supported values | Meaning and constraints |
|---|---|---|
| FFN policy | `--budget-policy`; [dynamic.py](../tools/auto_budget/dynamic.py) | Model-specific JSON; choose the six tiers above |
| Phase boundaries | `dynamic.BOUNDS` and matching policy `phases`; currently 0/.2/.5/.8/1 | Code-level change, not a free JSON knob; runtime rejects mismatches. Ceil boundaries: 7/11/10/7 steps at 35; regenerate allocations and Graphs together |
| Allocation objective | [allocate.py](../tools/auto_budget/allocate.py), calibration/retuning tools | Greedy local-error improvement per measured incremental latency, subject to task FLOP cap; not a global optimum |
| Partial A support | `BudgetLinear.prepare` | Sampled 8×8 energy per K128 block; selected count rounded to integer blocks; do not mutate `na`, masks, and union independently |
| FFN launch | Policy `launches`: BM 32/64/128, stages | Current runtime uses K128 and shape/tier lookup; benchmark and retune on receiver |
| Native experiments | [native.py](../experiments/ffn_block_repair_20260907/native.py) | Warps 4/8, stages 1–4, K64/128/256 have experimental paths; not all exposed by dynamic policy |
| Split-K | Native experimental backend | Keep 1 for accepted path; 2/4/8 change reduction order and can amplify trajectory error |
| Graph | `--phase-graph` | Prewarmed phase/CFG/shape specialization, current-input copy every replay; serial resident instance |
| Pointwise fusion | `--fuse-ffn` | Edge ReLU²+amax; Nano multiply+amax after retained Torch SiLU; preserves BF16 rounding/global scale |
| Nano shared input pack | `--share-ffn-pack` | Gate/up same-forward union support; own masks/scales remain; measured operator gain only about 1.005–1.036× |
| A pairing | [activation wrapper](../tools/activation_repair/run_trajectories.py) | `ffn_act_pair`, `ffn_dual_pair`; shape-specific >1.05× tuning threshold plus native fallback |
| W pairing | [repair wrapper](../tools/repair_design/run_trajectories.py) | `ffn_split_aw_merged`; research-only W-down/A-up-gate assignment |
| Residual calibration | [activation repair tools](../tools/activation_repair/README.md) | Base/residual scale multipliers and residual gain; preserve actual FP8 scale rounding; default gain 1 |

The FP4 block-scale group is **K16**, whereas repair support uses K128 tiles.
The native accumulator is FP32, output BF16. Arbitrarily adding W0+W1 or A0+A1
and requantizing to one FP4 value generally loses the residual. Expanded-K pairing
preserves two represented products and still executes twice the arithmetic on a
repaired tile; its possible gain comes from scheduling and data movement.

### Attention controls are a separate budget

See [RepairConfig](../packages/fp4-repair/src/thunderworld_fp4/config.py) and
[structured mask compiler](../tools/repair_design/structured_attention.py).

| Control | Values / behavior |
|---|---|
| `coverage` | 0–1; per-head quota is `ceil(coverage × ceil(N/128))`; this is tile count, not retained attention mass |
| `head_budgets` | Per-head coverage overrides |
| `terms` | 2: base + Q1K0 / P1V0; 3: also Q0K1 / P0V1; 4: includes cross terms |
| `repair_mask` | Contiguous CUDA bool `[Hq, ceil(M/128), ceil(N/128)]`, shared across batch; overrides coverage; false skips repair only |
| `query_tile` / `num_warps` | Default or 16/32/64; 4/8 warps; retune for shape |
| `splits` | 1/2/4/8; changes reduction grouping and needs quality validation |
| `transpose_v` | Physical packing optimization with verified equivalent outputs |
| `lazy_p` / `fast_p` | Experimental execution/quantization choices; fast reciprocal path needs numerical checks |
| Layer/step/branch rules | `RepairPolicy`, ordered first match; half-open ranges; caller defines branch names |

QK and PV currently share terms and mask; independently calibrated budgets are
not implemented. K/V are dynamic activations here, not offline FFN weights.
Keep joint streaming softmax: separately normalizing repaired and unrepaired
tiles and adding their outputs is incorrect.

AR→DM attention-mass decline and directed modality asymmetry motivate allocation,
but density is not measured repair benefit. The feature-only half-tile candidate
has not established a satisfactory quality default. Keep BF16 attention when
isolating FFN repair improvements.

## 4. Integration status: what can be combined today?

| Path | Entry point | Status |
|---|---|---|
| Six-tier FFN + BF16 attention | `tools/finding_resample/run.py --budget-policy ...` | Implemented and measured on both models |
| Above + Graph + fusion | Same entry point, both flags | Accepted execution baseline; archived complete-request evidence and release smoke |
| Nano shared gate/up pack | Same entry point, `--share-ffn-pack` | Independently verified; do not attribute its local gain to the whole stacked runtime |
| A pair + calibration variants | `tools/activation_repair/run_trajectories.py` | Separate FFN wrapper; not the dynamic budget Graph/fusion integration |
| W merge / structured attention | `tools/repair_design/run_trajectories.py` | Separate research wrapper; inspect actual dispatch receipts |
| Graph + fusion + shared pack + A/W pairing | No fully validated combined path | Requires integration, matched numerical tests, and new complete-request timings |
| Temporal output/delta reuse | Finding experiments only | Not enabled in the accepted baseline; stable masks do not imply reusable outputs |

Graph runs require BF16/FFN-only modes, no attention sampling, capture, or temporal
reuse. Rebuild/clear graphs before changing weights, task, or unsupported shape.
Graph reuse is command replay, never reuse of the previous step's output.

## 5. Reproduce on another machine

### Checkout, environment, and assets

```bash
git clone --branch perf/rtx5090-optimized-handoff https://github.com/Concyclics/ThunderWorld.git
cd ThunderWorld
git rev-parse HEAD
# Needed by original-vs-optimized adapter tests and the historical E2E adapter.
# Export immutable source only; do not copy today's optimized kernels as baseline.
mkdir -p data_local/baseline_source
git archive 9b51f0db23445d7695eb5b7ad9465410aeddd3cd \
  packages/fp4-repair/src/thunderworld_fp4 experiments/ffn_block_repair_20260907/native.py \
  | tar -x -C data_local/baseline_source
python3.12 -m venv data_local/e2e_venv
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
"$THUNDERWORLD_PYTHON" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
"$THUNDERWORLD_PYTHON" -m pip install \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@04e724ce518e087301052b65ff81ec35d54f4040' \
  accelerate==1.15.0 av==18.1.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 \
  scikit-image==0.26.0 transformers==5.9.0 tokenizers==0.22.2 \
  huggingface_hub==1.30.0 safetensors==0.8.0 numpy==2.2.5 pillow==11.1.0 \
  sentencepiece==0.2.0 protobuf==5.29.6 pytest pandas matplotlib
export CUDA_VISIBLE_DEVICES=0
bash tools/kernel_handoff/python5090.sh tools/kernel_handoff/check_env.py --output runs5090/receiver_env.json
bash tools/kernel_handoff/python5090.sh tools/cosmos_e2e/download.py --model edge
bash tools/kernel_handoff/python5090.sh tools/cosmos_e2e/download.py --model nano --with-sound
bash tools/kernel_handoff/python5090.sh tools/optimization_handoff/prepare_inputs.py
```

The measured runtime uses Torch 2.11/cu130, Triton 3.6 and SM120. The commands
describe a new isolated environment; this release did not reinstall the local
toolchain. Model access requires the receiver's own authorized Hugging Face
credentials. `check_env.py` deliberately rejects other GPU architectures: port
and retune kernels rather than assuming the 5090 launch table is portable.
CUDA 12.9 Nsight Compute was used locally; the old system-default profiler cannot
profile SM120. No local toolkit update was needed for this release.

| Model | Fixed revision |
|---|---|
| `nvidia/Cosmos3-Edge` | `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` |
| `nvidia/Cosmos3-Nano` | `411f42a8fdfb8c5b2583cb8786e0938f49796eaa` |

Weights, environments, downloaded assets, raw calibration captures, and payloads
are excluded from Git. `prepare_inputs.py` downloads the public NVIDIA robot
assets, verifies SHA256, and reconstructs the exact robot-table first frame.
The older `rtx5090-kernel-inputs-20260909.tar.gz` is only the original operator
probe package: it does **not** contain all model weights or the newer all-layer
calibration inputs. It is not required for the following full-model entry point.

### Run a matched execution comparison

Run the GPU regression suite after the environment and frozen-source preparation:

```bash
bash tools/kernel_handoff/python5090.sh -m pytest packages/fp4-repair/tests \
  tools/auto_budget/test_budget.py tools/auto_budget/test_execution.py \
  tools/cosmos_e2e/test_adapter.py tools/finding_resample/test_runtime.py -q
```

Compact, committed runtime policies are under
[`tools/optimization_handoff/profiles`](../tools/optimization_handoff/profiles).
They preserve the archived validation-selected schedules and launches. They are
not calibration/refitting inputs. Full policies remain in the evidence archive.

```bash
# Start with this two-step integration check, then use 35 steps for quality.
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --cases policy_table --seeds 8702 --steps 2 \
  --modes bf16 ffn_act2 ffn_auto50 ffn_auto75 \
  --budget-policy tools/optimization_handoff/profiles/edge.json \
  --phase-graph --fuse-ffn --output runs5090/receiver_edge_smoke

# Matched same-policy eager reference.
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --cases robot_table policy_table --seeds 8801 8802 --steps 35 --rounds 3 \
  --modes bf16 ffn_uniform25 ffn_act2 ffn_wgt2 ffn_uniform75 ffn_dual ffn_auto50 ffn_auto75 \
  --budget-policy tools/optimization_handoff/profiles/edge.json \
  --output runs5090/receiver_edge_eager

# Repeat the same command with --phase-graph --fuse-ffn and a NEW output path.
# For Graph-only, add --phase-graph. For Nano, change --model and profile to nano,
# and include av_pottery if sound components are installed.
```

Use fresh output directories, fixed seeds, matching shapes/steps/offload mode and
an idle GPU. Warmup/JIT/graph capture/weight packing are outside measured requests;
report those startup costs separately if deployment requires them. Complete
Generator time includes input encoding and video decoding; audio waveform decode
and MP4 writing are separate. Record both request wall time and Transformer GPU
time. Do not use the separate research wrappers' eager diagnostics as a reliable
service-throughput comparison.

### Refit budgets and explore pairing

1. Follow the capture → calibrate → benchmark → retune → guard → quality-gate
   commands in [auto-budget README](../tools/auto_budget/README.md), using fresh
   directories. Capture BF16 development trajectories on `robot_observe` and
   `policy_observe`, seed 7101, 35 steps; add `av_waterfall` for Nano. Keep final
   evaluation seeds separate from calibration and policy selection.
2. `bench.py` saves shape payloads beside its benchmark JSON. For release retiming
   use directories named `auto_budget_edge_bench` and `auto_budget_nano_bench`
   under a common `--payload-root`. Default is local `runs5090`. These payloads
   contain model data and are intentionally not committed.
3. A pairing: use [benchmark/calibration tools](../tools/activation_repair/README.md)
   and pass `--pair-benchmark NEW_BENCHMARK_JSON --calibration-policy NEW_JSON`
   to the trajectory wrapper. W pairing: pass `--merge-benchmark NEW_JSON` and
   optionally `--attention-prior NEW_NPZ` to the repair wrapper. Defaults reproduce
   the archived 5090 choices; new hardware should retune. The release `retime.json`
   is an audit, **not** a deployment-profile schema.
4. Inspect actual shape/layout receipts. Policy M416 was missing from the earlier
   W tuning cohort (M316); successful fallback is not a merged-kernel quality test.

## 6. Repair research priorities and guardrails

| Priority | Experiment | Required evidence |
|---|---|---|
| 1 | Attribute A-only vs W-only by projection/layer/phase | Same-cost ACT2 vs WGT2, base and dual anchors; video, action and audio separately |
| 2 | Calibrate residuals with full-input global scale | Preserve the actual global factor when using 64-row probes; evaluate quantized-trajectory inputs as well as teacher inputs |
| 3 | Output-aware joint A/W calibration | Fit rounded FP8 scales/gain on development data; independent steps, seeds and tasks for final evaluation |
| 4 | Causal attention-tile repair benefit | Compare feature priors to measured error reduction, separate QK/PV budgets, retain dense base and joint softmax |
| 5 | Integrate exact A pairing into dynamic Graph/fusion | Preserve scale/mask/accumulation semantics; retune whole stack, log fallback and actual hits |
| 6 | Further overlap / buffering | Profile dependencies and registers/shared memory; accept only same-input complete-FFN/request improvements |

Existing negative evidence should guide this work:

- W-pair rearranges FP32 accumulation. Small local error did not guarantee full
  trajectory equivalence: Nano I2V video relative L2 reached about 0.1222 versus
  the original split layout. Keep it experimental even where W-only kernels win.
- Pooled residual-scale/gain calibration selected the original settings for all
  ten projection/tier groups. Finer Edge changes improved a tiny local mean but
  worsened the fresh trajectory screen (I2V SSIM 0.86415→0.86251; action relative
  L2 0.18362→0.18747). Gain 1 remains the baseline, not an unfinished default.
- Feature-only attention allocation and step-output reuse do not yet meet a
  demonstrated general quality guarantee. Attention mass or head stability alone
  is insufficient evidence to skip recomputation.
- The quality gate's default allowances (SSIM −0.02, LPIPS +0.01, relative/spectral
  error ×1.05 +0.001) are provisional selection tolerances. BF16 consistency is
  not task success, physical plausibility, or a perceptual-quality guarantee.

## 7. Evidence and acceptance checklist

| Evidence | Location |
|---|---|
| Fresh speed audit and CSV/JSON | [release results](../results/optimization_handoff_20260911/SPEED_AUDIT.md) |
| Graph/fusion/chunk experiments | [execution report](GRAPH_FUSION_5090_20260911.md) |
| Budget allocation and quality selection | [budget report](AUTO_BUDGET_5090_20260910.md) |
| A pairing and residual calibration | [A study](ACTIVATION_PAIR_AND_CALIBRATION_20260911.md) |
| A/W attribution and W merge | [repair-path study](REPAIR_PATH_STUDY_20260911.md) |
| Scripts and release checks | [handoff tools](../tools/optimization_handoff/README.md) |

Before accepting a receiver change, record commit, source hashes, model revision,
environment, input hashes, actual tier/tile counts, real dispatch hits, peak
memory, matched wall/GPU times and full-trajectory quality. Test changed-input
Graph replay, tails, partial/full supports and fallback shapes. Recompute outputs
for every denoising step. Report model/task failures alongside wins.

Historical source snapshots and their hashes describe historical executions;
they need not equal current source files. Do not rewrite archived measurements
or validation-selected policies to make a new result look like a reproduction.

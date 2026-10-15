# Frozen FP4 optimization experiments

These experiments keep the published 1.5x/2x/2.5x assets, repair plans, full
attention base, 35 denoising steps, CFG, and current-input semantics. Original
presets and outputs are not overwritten. The new manifest `runtime_options`
field explicitly records qualified execution options; an absent field preserves
the old runtime. External baseline adapters reject these overrides.

The PRO 6000 experiment and measured limitations are recorded in
[`results/fp4_runtime_next_20260918/REPORT.md`](../../results/fp4_runtime_next_20260918/REPORT.md).
The execution recommendation is the cumulative `norm` variant. It enables
existing runtime fusion/Graph implementations through the frozen runner; this
is not a new CUTLASS GEMM implementation. The experimental `gain` selector did
not pass the two-dataset development guard and must not replace the default.

`optimize.py` provides execution qualification, complete-request diagnostic
timing, full development generation, and separate profiler traces. It records
source hashes, asset/plan identity, current GPU, output hashes, MMA ledgers,
non-MMA work, peak allocated/reserved memory, and preparation time. The
benchmark worker requires an idle GPU at each request boundary. All results
remain `performance_eligible=false` because these experiments use fixed safe
launches rather than a complete strictly qualified device tuning table.

Execution variants are cumulative:

| Variant | Changes from the frozen runtime |
|---|---|
| `calibrated` | Prescale-aware ReLU-squared/amax FFN fusion |
| `pointwise` | Above, plus existing pointwise fusion |
| `rotary` | Above, plus request-local rotary tables |
| `codec` | Above, plus current-input VAE graphs with owned outputs |
| `norm` | Above, plus existing normalization fusion |

Execution changes must match the frozen complete output bytes before timing.
The main comparison independently reloads each method/variant in each of five
rounds, in a fixed randomized order, with three seeds per input. Two fixed
inputs are measured separately. It is a single-request comparison, not a true
batched GEMM or concurrent service test. A separate 16-scene/two-seed sweep
checks output equivalence; this is development data, not the reserved final set.

```bash
export TW_PYTHON=/path/to/cosmos3/bin/python
export CUDA_VISIBLE_DEVICES=0
study=/path/to/materialized-fp4-study
output=/path/to/new-optimization-results

bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_suite.py qualify \
  --study "$study" --output "$output" --variants frozen calibrated codec norm
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_suite.py benchmark \
  --study "$study" --output "$output" --variants frozen norm
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_suite.py quality \
  --study "$study" --output "$output" --variants norm
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_report.py \
  --study "$output" --output "$output/report"
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_presets.py \
  --study "$study" --qualification "$output" --variant norm \
  --output /path/to/new-materialized-optimized-study
```

The exported study reuses the original offline assets and plans. It contains
the qualified runtime options and hashes of all 16-scene qualification
receipts. Use it with the existing inference and handoff entrypoints. Moving
to a 5090 requires running the same qualification and measurement commands
there; PRO 6000 latency is not a 5090 estimate.

The independent `gain` variant tests a quality hypothesis at unchanged repair
quotas. Its current-input block score is
`sum(max(r*r - (r - decode(FP4(r)))**2, 0) * channel_sensitivity)`.
The actual E2M1/E4M3 representation and calibrated residual scale are used.
Candidate residual codes stay in registers; only selected residual codes are
written. This adds vector quantization/decode work over the entire activation,
which is separately recorded in the request ledger. MMA and W1 storage are
unchanged, but latency is not assumed unchanged. It is a diagonal sensitivity
proxy, not the exact full-layer marginal output gain. Default `residual_energy`
preserves the frozen selector. `represented_gain` is currently restricted to
native Triton FP4 without shared packing; it is an explicit experimental option.

```bash
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_suite.py quality \
  --study "$study" --output "$output/selector" --variants gain \
  --methods b1.5_protected_n b2_joint
CUDA_VISIBLE_DEVICES= bash tools/fp4_handoff/python.sh \
  tools/fp4_handoff/optimization_metrics.py \
  --candidate "$output/selector/quality/b2_joint_gain" \
  --reference-study /path/to/original-fp4-baseline-study \
  --output "$output/selector/b2_metrics.json"
```

Metric comparisons retain PSNR, SSIM, LPIPS, relative L2 and worst-seed changes.
Seeds are averaged within scene for descriptive paired bootstrap intervals.
A selector is not recommended unless both datasets improve mean LPIPS without
SSIM/relative-L2 regression. Passing this exploratory guard is not a final-test
or multi-budget statistical qualification. Failed candidates remain recorded.

Evaluation dependencies are pinned in `requirements-quality.txt`; install them
in the evaluation environment before running metrics. The complete follow-up
can also consume an existing evaluation-only dependency directory without
altering the GPU workers' import paths:

```bash
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_followup.py \
  --study "$study" --reference-study /path/to/original-fp4-baseline-study \
  --output "$output" --evaluation-deps /path/to/evaluation-deps
bash tools/fp4_handoff/python.sh tools/fp4_handoff/optimization_archive.py \
  --study "$output" --reference-study /path/to/original-fp4-baseline-study \
  --output /path/to/compact-evidence
```

The follow-up is serial: sanitizers, execution qualification, selector quality,
CPU metrics, unchanged BF16/FP8 controls, then separate instrumented profiles.
Its controls run after the FP4 sweep, so their ratios are diagnostic; they are
not jointly interleaved formal baseline speedups. The compact archive includes
raw request receipts, output hashes, per-scene metrics, compute ledgers and
sanitizer logs. Large output videos and full traces stay in the result folder.

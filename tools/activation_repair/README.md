# Activation-paired FP4 GEMM and residual calibration

Experimental paths; production defaults unchanged.

- `paired_activation.py`: `[A0 A1] × [W0; W0]` in an expanded K256 scaled dot.
  Gather mode reads existing planes; direct mode reads interleaved packing.
  W0 is aliased without a duplicated buffer. Partial repairs branch to base-only
  K128, avoiding a padded residual multiply on unselected blocks. Optional W
  repair retains the compact native loop, in the same FP32 accumulator. No cross
  term is added. Pairing still requires two products per repaired tile.
- `packing.py`: writes paired layout directly during quantization, with the same
  storage size and launch count as two-plane packing. Optional base/residual
  block-scale multipliers use actual E4M3 scales and E2M1 conversions; residuals
  are recomputed from the rounded base. The shared global factor is retained.
- `benchmark.py`: all existing real model shapes plus M416, four tiers, matched
  launch search, current-input selection included, five randomized graph rounds.
  Repeated probes are latency inputs, not additional quality observations.
- `calibrate.py`: teacher-input output-error calibration on first/middle/last
  layers, pooled per projection. Fit uses development steps 1/11; steps 23/33
  remain held out. A-only changes A quantization; dual compares changing A or W
  quantization. It does not optimize both jointly, runtime search, or per-task
  budgets. These limitations are intentional for an interpretable first screen.
- `calibrate_gain.py`: separately tests post-quantization residual magnitudes,
  using actual rounded FP8 scales. All pooled settings remain at gain=1.
- `fit_granular.py` / `run_granular.py`: fit layer/task/CFG groups on early
  development phases, then evaluate all fit-selected changes on fresh Edge
  trajectories. Held-phase regressions are retained, not filtered away.
- `run_trajectories.py`: fresh-seed FFN-only comparisons with BF16 attention,
  explicit actual-shape kernel receipts and source snapshots. Eager request
  timing is diagnostic; merged layouts require local >1.05 tuning ratio.
- `validate.py`: independent decoded reference, bitwise baseline/direct packing
  checks, partial/full A and A+W, tails and changing-input graph replay.
- `summarize.py`: hash-audited local/trajectory tables and scientific figures.

Run from the repository root with `tools/kernel_handoff/python5090.sh`. Full
models require `THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"`.
Results are in `results/activation_repair_20260911`.

English study: [Activation pairing and calibration](../../docs/ACTIVATION_PAIR_AND_CALIBRATION_20260911.md).

Example full-trajectory comparison (choose a new output path):

```bash
THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python" \
  bash tools/kernel_handoff/python5090.sh tools/activation_repair/run_trajectories.py \
  --model edge --output runs5090/NEW_ACTIVATION_PAIR_RUN \
  --cases robot_table policy_table --seeds 8601 --steps 35 \
  --modes bf16 ffn_act2 ffn_dual ffn_act_pair ffn_dual_pair
```

The wrapper requires the completed benchmark and gain-calibration JSONs in the
results directory. `ffn_act_cal` / `ffn_dual_cal` currently resolve to the original
pooled settings, so duplicate trajectories for those equivalent modes were
omitted. `run_granular.py` adds `ffn_act_granular` for the separate Edge check.

Benchmark sources before the gain extension are preserved under
`results/activation_repair_20260911/benchmark_sources`. Setting gain=1 leaves the
quantized representation unchanged, as checked by validation.

For a new device, `run_trajectories.py --pair-benchmark NEW_BENCHMARK_JSON
--calibration-policy NEW_GAIN_CALIBRATION_JSON ...` selects separate experiment
assets without replacing the archived 5090 evidence. The benchmark must use the
`benchmark.py` schema with `winners`; the release `retime.json` is an audit, not
a deployment profile. Default paths retain the archived reproduction.

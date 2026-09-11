# RTX 5090 execution study — 2026-09-11

[Implementation, conclusions and scope](../../docs/GRAPH_FUSION_5090_20260911.md).

This bundle compares eager, phase-aware whole-Transformer CUDA Graph, and
Graph + numerically equivalent FFN pointwise/amax fusion. All three use the
same frozen repair policy, checkpoint, inputs, seeds, and 35-step schedule.
BF16 controls receive the same Graph execution treatment. Dynamic schedules
retain current-input tile selection; graph replay does not reuse old outputs.

- `paired.csv`: 40 matched triples / 120 complete Generator trajectories.
- `summary.csv`: two-seed aggregates per model, task and precision mode.
  Speed ratios are geometric means of paired ratios; times are arithmetic
  means. Peaks are maximum allocated GiB over the two requests.
- `fusion_ffn.csv`: 20 real-weight full FFN comparisons; five alternating-order
  CUDA Graph timing samples per implementation, medians used for ratios.
- `double_buffer.csv`: 120 M-chunk candidates across 40 shape/tier pairs;
  three alternating-order samples including full-input scale/mask and sync.
- `tuned_double_buffer.csv`: separately retune the closest chunk per shape/tier
  across six M-tile/stage choices (240 trials), then five fresh alternating-order
  comparisons. This is a bounded follow-up, not an exhaustive joint search.
  `num_chunks=1` candidates do not exercise packing/GEMM overlap; the two small
  positive ratios are both in that category. All multi-chunk candidates lose.
- `analysis.json`: exact output/budget equality and graph phase assertions.
- `validation.json`: complete coverage, finite saved arrays, raw output hashes,
  per-trajectory call counts and executed source-snapshot hashes.
- `records/`: original machine-readable results, including all timing samples.
- `inputs/`: matching policies and local benchmark launch/input records.
- `source_snapshots/`: content-addressed Python sources actually recorded by
  the trajectory runs, plus final analysis tools identified in `final_tools.json`.
- `local_raw_inventory.json`: local raw media, NPZ outputs and benchmark input
  identities; large raw outputs and model weights are not duplicated here.
- `logs/`: execution commands/logs, GPU tests and Compute Sanitizer result.
- `SHA256SUMS`: integrity of the complete evidence bundle, excluding itself.

Timing excludes model loading, offline preparation, compilation and Graph
warmup. It includes image encoding, denoising, video decoding and the same
trajectory summaries. AV waveform decoding and AR Reasoner are outside scope.
Each variant runs in a separate sequential process, once per seed; small
end-to-end fusion differences are not proof of a stable service-level gain.
Array equality preserves each mode's existing quality; it does not establish
FP4 quality equal to BF16 or action task success. No power/energy claim is made.

Recompute the CPU analysis from local raw runs:

```bash
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh tools/finding_resample/validate.py \
  runs5090/execution_{edge,nano}_{eager,graph,fused} \
  --output results/execution_20260911/validation.json
bash tools/kernel_handoff/python5090.sh tools/auto_budget/analyze_execution.py \
  --output results/execution_20260911
bash tools/kernel_handoff/python5090.sh tools/auto_budget/archive_execution.py \
  --output results/execution_20260911
cd results/execution_20260911
sha256sum -c SHA256SUMS
```

The [runtime guide](../../tools/auto_budget/README.md#whole-transformer-graph-and-ffn-fusion)
documents fresh GPU runs. `logs/thunderworld_execution_suite.sh` preserves the
formal experiment commands; use fresh output directories when reproducing.

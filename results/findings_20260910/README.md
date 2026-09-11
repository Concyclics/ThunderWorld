# 2026-09-10 findings and ablation evidence

Read [the Chinese research report](../../docs/FINDINGS_RESAMPLING_20260910.md)
for conclusions and measurement boundaries, and
[the tool README](../../tools/finding_resample/README.md) for reproduction.

- `edge/`, `nano/`: whole-trajectory / individual-head CSVs, frozen development
  policies, sampling summaries, layer/step heatmaps and temporal trend figures.
- `ablation/`: paired per-request metrics, full comparison table, prediction
  error traces and fixed-frame panels. CSV values retain model/case/seed scope.
- `records/`: complete run reports plus waveform and LPIPS evaluation records.
- `source_snapshots/`: source bytes named by SHA256, including versions actually
  executed and the final analysis tools. `final_analysis_sources.json` maps the
  final tools to their hashes.
- `local_raw_inventory.json`: absolute paths, sizes and hashes of larger local
  NPZ/video/audio artifacts. These are retained under `runs5090/`, not copied
  into this report directory. The head NPZs contain all steps/layers/heads/roles
  and both CFG branches where applicable.
- `*_validation.json`, `validation_logs/`: full-matrix checks, GPU numeric tests,
  memory checking and dependency/input receipts.
- `SHA256SUMS`: hashes of this reviewable evidence package.

Canonical runs:

| Role | Local directory |
|---|---|
| Edge BF16 observations | `runs5090/findings_edge_sampling` |
| Nano BF16 observations | `runs5090/findings_nano_sampling_complete` |
| Edge full ablation | `runs5090/findings_edge_ablation_v2` |
| Nano full ablation | `runs5090/findings_nano_ablation` |

Nano's canonical sampling report preserves provenance from a partial non-AV
campaign and the completed AV latent-saving campaign. Exact duplicate AV video
hashes were checked before merging. The original interrupted statuses remain
unchanged. Two-step smoke runs and the failed initial Edge ablation (before the
new adapter's contiguous-mask fix) are excluded from scientific comparisons.

All comparisons are complete Generator trajectories. They exclude a separate
AR Reasoner rollout, real robot execution and energy measurement. AV waveform
decode is measured in a separate process. Two seeds and a few related scenes
cannot establish population quality guarantees or stable small latency gains.

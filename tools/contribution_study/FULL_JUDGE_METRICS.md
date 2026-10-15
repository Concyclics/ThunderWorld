# Full RBench / PAI-Bench video consistency evaluation

This scores the existing full Edge and Nano generations against the corresponding
BF16 video for the **same model, scene, and seed**. It does not regenerate videos,
use a real future video as the reference, or estimate task success.

## Frozen scope

- RBench: 650 scenes; PAI-Bench: 1,044 scenes.
- Edge and Nano, with the original two seeds per scene.
- BF16, native FP8 W8A8, ModelOpt AWQ NVFP4 W4A16 decoded quality reference,
  plain W4A4, ARCQuant 50%, ARCQuant 100%, SharQ, ours 1.5x, ours 2.0x.
- 60,984 source videos, including 6,776 BF16 references; **54,208 paired scores**.
- Generation: 320×192, 61 frames, 35 steps, CFG 6, flow shift 3, 24 fps export.
  Source manifests contain the complete prompts, model identities and method assets.
- Existing scene exposures remain recorded. This is a complete benchmark-cohort
  comparison, not a new held-out calibration experiment.

Sources:

```
/home/yufeng/thunderworld/judge_full_i2v_20260919
/home/yufeng/thunderworld/judge_full_baselines_20260922
```

Result root:

```
/home/yufeng/thunderworld/judge_full_metrics_20260923
```

## Metric definitions

All metrics consume original float32 RGB NPZ arrays in [0,1]. They evaluate all
61 frames, including the first conditioned frame. There is no MP4 decoding,
uint8 conversion, resizing, frame subsampling, or per-video range normalization.

- **LPIPS ↓**: official AlexNet LPIPS v0.1 weights, float32, batches of eight;
  average all frame distances. BF16 reference features are cached across methods.
- **SSIM ↑**: the existing `skimage.metrics.structural_similarity` defaults,
  7×7 uniform windows, sample covariance 49/48, K1=0.01, K2=0.03, data range 1.
  Average RGB channels and frames using the original rounding conventions.
  The equivalent GPU implementation uses separable double-accumulation uniform
  filters with float32 intermediate output. It is qualified against skimage.
- **PSNR ↑**: `-10 log10(mean((candidate-reference)^2))`, with float64 accumulation
  over the entire clip and data range 1. Average *clip PSNRs* across seeds/scenes;
  do not replace this with PSNR computed from pooled MSE.
- Relative L2 and RMSE are retained for the pre-existing FP8 quality guardrails.

Exact equality is `psnr_db: null, psnr_exact: true`, representing +infinity.
It is not a missing value and is never silently dropped or capped. BF16 self
comparison appears only as a reference row (LPIPS=0, SSIM=1, PSNR=reference).

## Aggregation and uncertainty

Both summaries are exported for each model and benchmark separately:

1. `category_macro`: average two seeds within a scene, scenes within a category,
   then weight categories equally. This preserves the historical primary summary.
2. `scene_micro`: average two seeds within a scene, then weight scenes equally.

Confidence intervals use the frozen 100,000 bootstrap replicates, seed 190919
(plus the stable group index), resampling **scenes within category**. Both seeds
and all methods remain paired. Multinomial draw counts are exactly the ordinary
with-replacement scene bootstrap. Interval endpoints use empirical inverse-CDF
ranks, which also handle exact-pair +infinity without interpolation NaNs.

Paired primary comparisons against ARCQuant 100% and the separate FP8 gate
preserve the conservative Bonferroni family size 12 from the generation manifest.
Other method contrasts are descriptive/exploratory. A quality gate still requires
lower LPIPS with its CI below zero, no mean SSIM decline, and no mean relative-L2
increase; PSNR does not replace that gate.

## Commands

Run from `/home/yufeng/thunderworld/ThunderWorld-fp4-next-20260918`. Use the project
wrapper and installed metric dependencies:

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

bash tools/fp4_handoff/python.sh tools/contribution_study/full_judge_metrics.py prepare \
  --root /home/yufeng/thunderworld/judge_full_metrics_20260923

bash tools/fp4_handoff/python.sh tools/contribution_study/full_judge_metrics.py qualify \
  --root /home/yufeng/thunderworld/judge_full_metrics_20260923 --gpu 0
bash tools/fp4_handoff/python.sh tools/contribution_study/full_judge_metrics.py qualify \
  --root /home/yufeng/thunderworld/judge_full_metrics_20260923 --gpu 2

bash tools/fp4_handoff/python.sh tools/contribution_study/full_judge_metrics.py run \
  --root /home/yufeng/thunderworld/judge_full_metrics_20260923 \
  --gpus 0 2 --workers-per-gpu 2

bash tools/fp4_handoff/python.sh tools/contribution_study/full_judge_metrics.py status \
  --root /home/yufeng/thunderworld/judge_full_metrics_20260923
```

`prepare` is one-time. It checks exact source pairing, receipt/model/projection
identity, AR-cache mode, original seeds, debug state, and full scene/method coverage.
The original 72 reused scenes are checked against their original receipt hashes.
Each worker verifies both compressed-file and decoded-video SHA256 before scoring.

`qualify` independently checks 64 real pairs spanning both models, benchmarks and
all candidates, plus six synthetic boundary cases, on each requested GPU.
Four real pairs also use the original CPU LPIPS implementation. Qualifications
are bound to the frozen manifest; weights and package versions are recorded.

`run` automatically produces the complete report when all workers finish. Repeat
the same command to resume: every pair is checkpointed atomically, partial scenes
resume from their saved rows, and changed source/metric identities fail closed.
The worker count must remain unchanged on resume. The coordinator holds the
project's shared/global and exclusive/per-device leases and only manages its own
children. It requires at least 12 GiB free on each selected GPU at admission.
GPU OOM or any worker error preserves checkpoints and stops this job's workers.

No generation kernel is changed. Shared-GPU timings here describe **metric job
progress only**, and must not be used as inference speed comparisons.

## Outputs

- `manifest.json`, `input_audit.json`: frozen cohort, code/weight hashes, execution
  types, exact coverage and recorded baseline adapters.
- `qualification_gpu0.json`, `qualification_gpu2.json`: numerical checks.
- `results/<model>/<case>.json`: per-seed, per-method, per-frame metrics and hashes.
- `progress.json`, `workers/slot*.json`, `workers/slot*.log`: resumable live progress.
- `report/per_video.csv`, `report/per_video.jsonl`: complete paired scores and
  original NPZ/MP4 paths; JSONL also includes all frame LPIPS and SSIM values.
- `report/per_scene.csv`, `report/per_category.csv`: aggregation inputs.
- `report/summary.csv`, `report/summary.json`, `report/REPORT.md`: all four
  model/benchmark comparisons, both weighting schemes and confidence intervals.
- `report/paired_comparisons.csv`: paired changes, familywise intervals, original
  quality guardrails and worst-seed LPIPS regressions.
- `report/complete.json`: final coverage and output checksums; absent until the
  full report succeeds. The report command refuses incomplete cohorts.

AWQ results are quality-only decoded ModelOpt NVFP4 W4A16 outputs. ARCQuant and
SharQ rows refer to the recorded Cosmos adaptations. No timing or memory advantage
is inferred from these paths or from this scoring run.

Tests:

```bash
bash tools/fp4_handoff/python.sh -m unittest discover \
  -s tools/contribution_study -p test_full_judge_metrics.py -v
```

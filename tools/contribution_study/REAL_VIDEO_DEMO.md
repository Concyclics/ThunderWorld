# Ten real-video I2V comparisons

This extends `EXPANSION_DEMO.md` with a new paired recorded-video reference.
The completed run is `/home/yufeng/thunderworld/fp4_real_video_demo_20260921`.
It preserves the earlier BF16-consistency experiment in its original directory.

## Frozen scope

- Edge, 10 scenes, one paired seed per scene, all 12 existing methods: BF16,
  native FP8, NVFP4 AWQ-lite decoded W4A16, and 1/3/4/6A/6W/9/12A/12W/16 FP4 products.
- 320×192, 61 frames, 24 FPS, 35 steps, CFG 6, flow shift 3, Graph, debug off.
- Four AgiBot robotics, three OpenDV driving and three Ego/Exo activity clips.
  `real_video_demo_selection.json` fixes the exact IDs and original captions before scoring.
- Dataset: `shi-labs/physical-ai-bench-conditional-generation`, revision
  `3357122d526dfdb7beaa97ccea138694c0cff9b7`.
- Only the first frame and caption condition generation. The source videos are
  adapted to I2V; no official PAI-Bench-C depth/edge/segmentation/blur control
  inputs are used. These are not official PAI-Bench-C scores.
- Primary scores exclude the conditioning frame (frames 1–60); secondary scores
  include all 61 frames. Raw float32 outputs are scored, never the viewing MP4.
- PTS nearest-frame sampling at `t = arange(61)/24`, without time stretching,
  padding or output-dependent matching. Reference resizing and center cropping
  are checked against the actual Cosmos image preprocessing helper.

## Reproduce

Reuse the qualified native expansion implementation and offline weight bank from
`EXPANSION_DEMO.md`. Do not edit a frozen generation manifest or reuse its root
for changed settings. The helper below uses this machine's existing environment;
paths must be adapted when moving the experiment.

```bash
tw() {
  env TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python \
    PYTHONPATH=/tmp/thunderworld-expansion-report-deps:/tmp/thunderworld-joint-deps \
    MPLCONFIGDIR=/tmp/thunderworld-expansion-mpl \
    TW_SHARED_CALIBRATION_GPUS=0,1 TW_SHARED_CALIBRATION_LOCK_SCOPE=device \
    bash tools/fp4_handoff/python.sh "$@"
}
REAL_ROOT=/home/yufeng/thunderworld/fp4_real_video_demo_reproduce
mkdir -p "$REAL_ROOT"
cp tools/contribution_study/real_video_demo_selection.json "$REAL_ROOT/selection.json"
tw -m huggingface_hub.cli.hf download \
  shi-labs/physical-ai-bench-conditional-generation \
  videos/task_0054.mp4 videos/task_0194.mp4 videos/task_0133.mp4 \
  videos/task_0062.mp4 videos/task_0259.mp4 videos/task_0309.mp4 \
  videos/task_0357.mp4 videos/task_0403.mp4 videos/task_0549.mp4 \
  videos/task_0512.mp4 metadata.csv README.md \
  --type dataset --revision 3357122d526dfdb7beaa97ccea138694c0cff9b7 \
  --local-dir "$REAL_ROOT/source" --max-workers 4
tw tools/contribution_study/real_video_reference.py --root "$REAL_ROOT" \
  --base /home/yufeng/thunderworld/fp4_expansion_demo_20260921
tw tools/contribution_study/expansion_experiment.py run --root "$REAL_ROOT" --gpus 0 1
tw tools/contribution_study/real_video_metrics.py --root "$REAL_ROOT"
tw tools/contribution_study/dataset_metrics.py "$REAL_ROOT/edge/final/manifest.json" \
  --generation "$REAL_ROOT/edge/final/generation" \
  --output "$REAL_ROOT/edge/final/metrics_bf16"
tw tools/contribution_study/judge_video_export.py --root "$REAL_ROOT/edge"
tw tools/contribution_study/test_real_video_reference.py
tw tools/contribution_study/real_video_validate.py --root "$REAL_ROOT"
tw tools/contribution_study/real_video_report.py --root "$REAL_ROOT"
```

Metrics/export can alternatively run with `--follow` while generation proceeds.
The quality runner uses the existing per-GPU resource lock. No performance tuning
or latency ranking is performed by this experiment.

## Outputs and interpretation

- `report/report.md`, `results.json`, `per_demo.csv`: all methods, future/all-frame
  means, scene bootstrap intervals, category means and paired baseline changes.
- `report/quality_vs_{terms,planes}.{png,pdf,svg}`: the real BF16 mean is a finite
  measured baseline. FP8 uses long dashes; AWQ-FP4 uses short dashes.
- `report/paired_vs_fp8.*`: paired differences to separate method effects from
  between-scene difficulty. Intervals are exploratory and unadjusted.
- `report/gallery.html`: original aligned video plus all 12 methods, with 10
  synchronized comparison grids. Individual generated MP4 and lossless FFV1
  videos and full generation receipts/logs are retained in `edge/` and `logs/`.
- `validation.json`: 120 request checks; independently recomputed MSE/PSNR/RMSE/L2
  for both frame scopes; framewise SSIM/LPIPS spot checks on three declared clips.
- Supplementary BF16-consistency metrics remain separate from real-video fidelity.

One recorded future is not the only plausible I2V continuation. Small pixel or
perceptual differences on ten scenes do not establish broad generative quality,
physical success, or a statistically validated superiority claim. Increasing FP4
expansion order does not imply monotonically improving real-reference metrics.

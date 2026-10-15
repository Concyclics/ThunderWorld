# Linear repair with fixed base FP4 attention

This experiment answers whether the previously strong “W-side” result came from
linear weight repair or from attention's K/V correction. K and V are current
activations; calling their correction weight-only repair is incorrect.

The user-confirmed intervention disables **all** Q/P and K/V attention repair.
Each FP4 candidate executes one-plane dense QK and PV, keeps every base attention
block, and varies only generation projection/FFN corrections. BF16 remains the
same-model, same-input, same-seed reference. It uses BF16 attention as before.

## Frozen panels and policies

Root: `/home/yufeng/thunderworld/fp4_attention_base_validation_20260922`.

- `matched10`: original ten scenes and seeds, six methods, 60 videos. Regenerated
  BF16/base signal hashes must match the original experiment. This isolates the
  attention intervention against the earlier ten-scene results.
- `expanded72`: RBench 36 + PAI-Bench 36, two seeds each, six methods, 864 videos.
  Cases are selected by a fixed SHA256 order within categories, excluding the
  original ten, nonempty inventoried prior scene roles, and duplicate RGB images.
  Seeds are 92231001–92231144 after auditing historical manifests/output names.
  These scenes are new to this ablation, not necessarily never used by earlier
  whole-benchmark generation jobs.

| Method ID | Linear expression/support | Linear MMA cap |
|---|---|---:|
| `bf16` | Original BF16 reference | Different dtype |
| `fp4_base` | A0 W0ᵀ | 1× |
| `linear_a` | A0 W0ᵀ + A1 W0ᵀ | 2× |
| `linear_w` | A0 W0ᵀ + A0 W1ᵀ | 2× |
| `linear_both_1p5` | Base + A/W each 25% K128 blocks | 1.5× |
| `linear_both_2p0` | Base + A/W each 50% K128 blocks | 2× |

The latter two are fixed support-proxy ablations, not newly calibrated production
schedules. Integer quotas round down. W supports use represented W1 block energy;
A supports use current represented residual energy times fixed W0 channel-block
sensitivity. W0/W1 come from the same frozen parent bank. Quantizer, prescale,
output rounding, current-input semantics, and accumulation paths are preserved.
There is no cross term, selector MMA, step reuse, or output-driven tuning.

All experiments retain 35 steps, CFG 6, 320×192, 61 frames, Graph, debug off, and
the same generation projection scope. AR prefill and nonlinear operations retain
their common BF16 implementation. `compute.json` counts padded executed FP4 MMA
over both CFG branches and all steps, before Graph capture. Because attention
stays at 1×, actual whole controlled-MMA ratios are lower than linear caps.
These are not end-to-end FLOP or latency ratios. Linear residual packing and
full required weight planes remain; this ablation does not establish packing,
storage, or speed savings.

## Run or resume

Run from the repository root. Parent offline weights, models, images and locked
qualification outputs must exist. `prepare` was already executed; do not repeat
it or overwrite the frozen manifests. The driver resumes completed cases and
uses one GPU process at a time.

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
export TW_SHARED_CALIBRATION_GPUS=0,1
export TW_SHARED_CALIBRATION_LOCK_SCOPE=device
ABLATION_ROOT=/home/yufeng/thunderworld/fp4_attention_base_validation_20260922

CUDA_VISIBLE_DEVICES=1 bash tools/fp4_handoff/python.sh \
  tools/contribution_study/attention_base_study.py run \
  --root "$ABLATION_ROOT" --gpu 1
```

CPU-only metrics and video export can follow generation. Run these for each
panel (`matched10`, then `expanded72`) in separate terminals if following live:

```bash
ABLATION_PANEL=expanded72
CUDA_VISIBLE_DEVICES='' bash tools/fp4_handoff/python.sh \
  tools/contribution_study/dataset_metrics.py \
  "$ABLATION_ROOT/$ABLATION_PANEL/edge/final/manifest.json" \
  --generation "$ABLATION_ROOT/$ABLATION_PANEL/edge/final/generation" \
  --output "$ABLATION_ROOT/$ABLATION_PANEL/edge/final/metrics" --follow

CUDA_VISIBLE_DEVICES='' bash tools/fp4_handoff/python.sh \
  tools/contribution_study/judge_video_export.py \
  --root "$ABLATION_ROOT/$ABLATION_PANEL/edge" --follow

CUDA_VISIBLE_DEVICES='' bash tools/fp4_handoff/python.sh \
  tools/contribution_study/attention_base_report.py \
  --root "$ABLATION_ROOT" --panel "$ABLATION_PANEL"
```

Reports require all paired metric files. Re-run the report after exports finish
to include the gallery. Artifacts include raw float32 NPZs, full generation logs,
MMA receipts, per-seed/scene CSVs, lossless FFV1 and MP4 previews, judge indexes,
and a collapsible paired gallery. Fidelity is calculated from raw NPZs.

## Statistical interpretation

Metrics preserve the original definition: clip-global PSNR at range 1, frame-mean
SSIM and AlexNet LPIPS, plus clip relative L2, including the conditioning frame.
Average the two seeds within each expanded scene before averaging scenes or
bootstrapping. The primary expanded panel uses 10,000 paired scene resamples
within benchmark × category. Report 95% intervals and Bonferroni 99% intervals
for the five frozen primary LPIPS contrasts. Subgroup and leave-one-category-out
results are descriptive, not additional unadjusted discovery tests.

The original ten have only one scene per category, making category-stratified
bootstrap degenerate. Its report explicitly uses exploratory benchmark-stratified
intervals instead; it does not support a confirmatory claim. Never pool it with
the 72-scene panel to claim independent replication. No tuning is performed from
either panel. Sparse bilateral policies may lose to a unilateral policy; preserve
that result rather than changing supports after observing the test outputs.

All timings are non-official diagnostics on a shared GPU. The paper's older
attention-on Figure 2 and evidence are preserved separately until a deliberate
revision with the new experiment identity.

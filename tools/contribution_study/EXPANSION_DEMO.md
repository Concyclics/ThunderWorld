# Native FP4 residual expansion quality experiment

The frozen study is `/home/yufeng/thunderworld/fp4_expansion_demo_20260921`.
It contains 10 Edge I2V tasks (five RBench and five PAI-Bench), each with one
new seed, and 12 configurations. Sampling depends only on frozen scene IDs.
All configurations use 320×192, 61 frames, 35 steps, CFG 6 and flow shift 3.

The experiment measures consistency to paired BF16 outputs. It does not measure
physical correctness, task success, inference speed or generalization to unseen
scenes. Metrics use raw decoded float32 RGB videos, before MP4 compression.

| Configuration | W planes | A planes | Products |
|---|---:|---:|---:|
| W4A4 | 1 | 1 | 1 |
| Two-plane, no cross | 2 | 2 | 3 |
| Two-plane, complete | 2 | 2 | 4 |
| Activation-priority rectangle | 2 | 3 | 6 |
| Weight-priority rectangle | 3 | 2 | 6 |
| Three-plane, complete | 3 | 3 | 9 |
| Activation-priority rectangle | 3 | 4 | 12 |
| Weight-priority rectangle | 4 | 3 | 12 |
| Four-plane, complete | 4 | 4 | 16 |

BF16, native FP8 W8A8 and ModelOpt NVFP4 AWQ-lite decoded W4A16 are the other
three configurations. Their generation projection scope is identical; FP8/AWQ
retain the historical BF16 attention definition. AWQ is a quality reference,
not a native packed W4A16 performance implementation.

Every FP4 configuration expands projections, FFN and both QK/PV. For attention,
the A count means Q/P and the W count means current K/V, not offline attention
weights. No base blocks are removed. AR prefill and nonlinearities share the
BF16 path. `W0` is held fixed; no sparse schedule, outlier split or parameter
search is added to this experiment. The 6/12-term rectangles are explicitly
defined alternatives, not a Taylor total-degree truncation.

Each plane uses actual E2M1 codes and E4M3 scales. Residuals are recomputed from
decoded values, with the existing common outer factor and scale floor. GPU
products use native FP4 Tensor Core `dot_scaled`, shared FP32 accumulation and
BF16 output boundaries. The first two packing planes and 1/3/4-term attention
and 3/4-term GEMM match the existing kernels bitwise in qualification tests.
Increasing plane count need not monotonically improve complete video outputs.

The four-plane weight master bank is prepared offline. Each method copies only
the needed prefix to the GPU. Every activation residual comes from the current
request input; Graph replay updates current inputs. The adapter prepares the
unquantized model to install offline assets, then switches both runner and
engine to the recorded actual FP4 configuration before warmup. The method's
`expansion` field is authoritative; use `expansion_experiment.py worker` so the
ordinary baseline worker cannot silently substitute a different path.

## Reproduce on this machine

Use a **fresh** output directory for a new run. Existing studies are immutable
and resume through their worker receipts. From the repository root:

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-expansion-report-deps:/tmp/thunderworld-joint-deps
export TW_SHARED_CALIBRATION_GPUS=0,1
export TW_SHARED_CALIBRATION_LOCK_SCOPE=device
export EXPANSION_RUN=/home/yufeng/thunderworld/fp4_expansion_reproduction

# Isolated plotting dependencies; does not modify the inference environment.
bash tools/fp4_handoff/python.sh -m pip install --target /tmp/thunderworld-expansion-report-deps --no-deps matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 fonttools==4.65.0 kiwisolver==1.5.1 pyparsing==3.3.3

bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_experiment.py prepare --root "$EXPANSION_RUN"
CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_experiment.py prepare-assets --root "$EXPANSION_RUN"
CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_qualify.py --root "$EXPANSION_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_experiment.py freeze --root "$EXPANSION_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_experiment.py run --root "$EXPANSION_RUN" --gpus 0 1

bash tools/fp4_handoff/python.sh tools/contribution_study/dataset_metrics.py "$EXPANSION_RUN/edge/final/manifest.json" --generation "$EXPANSION_RUN/edge/final/generation" --output "$EXPANSION_RUN/edge/final/metrics"
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_video_export.py --root "$EXPANSION_RUN/edge"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_weight_diagnostics.py --root "$EXPANSION_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_report.py --root "$EXPANSION_RUN"
```

`run --methods ... --gpus ...` permits disjoint method queues. Run only one
worker per device for this experiment. Generation failures preserve completed
cases; they do not lower precision, resolution or step count. Methods are
resumable. Kernel/source/asset changes require a new frozen implementation.

The manifest references pinned local models, the complete dataset inventory,
and the existing AWQ calibration assets. `prepare` checks input image hashes
and copies the 10 inputs into the study. Moving the experiment to a new host
requires resolving those source assets first; it is not a self-contained model
download package.

Outputs include individual MP4/FFV1 videos, raw NPZ, request receipts, worker
logs, PSNR/SSIM/LPIPS and relative-L2 metrics, exploratory bootstrap intervals,
PNG/PDF/SVG trend figures and 10 synchronized comparison-grid videos. W4 plane
reconstruction diagnostics sample 64 evenly spaced output rows per projection
while retaining the stored full-matrix quantization factor.

Timings from these mechanism kernels are ineligible for speed claims. P-plane
quantization is recomputed per V plane to bound register pressure, and the
term count does not include packing/non-MMA cost or unchanged model operators.

## Two-term single-side extension

`/home/yufeng/thunderworld/fp4_expansion_two_term_20260922` adds 20 videos to the
same frozen cohort: `fp4_two_a` is W0A0 + W0A1 (W1×A2, mask 3), and `fp4_two_w`
is W0A0 + W1A0 (W2×A1, mask 17). Here W1×A2 denotes one weight plane and two
activation planes, not residual indices. Original BF16 outputs are reused with
explicit reference-manifest hashes and exact input/seed/model/scope matching;
the original twelve-method evidence is unchanged.

Attention uses the same side convention as the higher-order rectangles:
A-only repairs Q/P and W-only repairs current K/V. K/V remain online activations.
The comparison therefore measures a joint projection/FFN/attention policy, not
projection-only weight or activation sensitivity with attention held fixed.

With the environment above, use a fresh extension root:

```bash
export TWO_TERM_RUN=/home/yufeng/thunderworld/fp4_two_term_reproduction
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_two_term.py prepare --root "$TWO_TERM_RUN" --parent /home/yufeng/thunderworld/fp4_expansion_demo_20260921
CUDA_VISIBLE_DEVICES=0 bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_two_term.py qualify --root "$TWO_TERM_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_two_term.py run --root "$TWO_TERM_RUN" --gpus 0 1
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_two_term.py metrics --root "$TWO_TERM_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_video_export.py --root "$TWO_TERM_RUN/edge"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_two_term_report.py --root "$TWO_TERM_RUN"
```

Qualification adds eight tests for asymmetric decoded GEMM/full-attention
references, M129/M416, GQA/tails, zeros and current-input Graph replay. A-only
attention also matches the existing ACT2 kernel bitwise. Sanitizer receipts
cover both asymmetric variants. The combined report retains all 14 methods,
130 paired non-BF16 scores, individual videos and ten synchronized grids. Its
PSNR panels omit BF16 self-reference labels; LPIPS/SSIM retain those references.

## Strict 1.5× / 2× bilateral extension

`/home/yufeng/thunderworld/fp4_expansion_sparse_matched_20260922` adds two
predefined support policies to the **same ten scenes and seeds**, using the
unchanged offline weight bank and parent quantizer. It does not import the
different sixteen-scene, differently calibrated production-policy comparison.

Each operand receives 25% (1.5× cap) or 50% (2× cap) of repair tiles, rounded
down to an integer count per output tile. The base is always dense and no cross
term executes. The cap counts padded native FP4 MMA over projections, FFN, QK
and PV for the complete 35-step request and both CFG branches. It excludes the
unchanged AR/BF16 and non-MMA work. Selectors add no matrix multiplication.

Offline W1 support ranks actual represented residual energy. A support uses
current represented A1 energy times fixed W0 channel-block sensitivity.
Attention QK supports use current operand norm-product proxies; PV uses V0/V1
norm proxies because P is not available before softmax. Stable index order
breaks score ties. There is no fitting to the ten generation outputs, no W0
change, and no step/head policy search. These support proxies must be evaluated,
not assumed to beat either single-side baseline.

For clean arithmetic isolation the new mechanism path still stores both full
weight planes and packs all residual codes. **No storage, packing or speed
advantage is claimed for this experiment.** Shared-load timings are diagnostic.
This is a strict support ablation, not a replacement for the separately
calibrated deployment policies.

```bash
export SPARSE_RUN=/home/yufeng/thunderworld/fp4_sparse_matched_reproduction
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_sparse.py prepare --root "$SPARSE_RUN"
CUDA_VISIBLE_DEVICES=1 bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_sparse.py qualify --root "$SPARSE_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_sparse.py run --root "$SPARSE_RUN" --gpu 1
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_sparse.py metrics --root "$SPARSE_RUN"
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_video_export.py --root "$SPARSE_RUN/edge"
bash tools/fp4_handoff/python.sh tools/contribution_study/expansion_sparse_report.py --root "$SPARSE_RUN"
```

The environment exports from the earlier reproduction section still apply.
Run one worker at a time on the selected GPU. Qualification covers empty/full
and single-side endpoints, partial decoded references, current-input Graph
replay, M129/M416, GQA/tails, zeros, integer quotas, and padded whole-request
accounting including M960. Memcheck/racecheck precede full generation. Budget
verification runs after shape discovery and before graph capture. Original
generation receipts, frozen manifests and raw outputs are never overwritten;
new videos, logs, metrics, paired intervals and negative outcomes are retained.

# Calibration, quantization and repair allocation

The portable commands call the original algorithms under `tools/contribution_study/`; there is one inference runtime. Downloading the frozen models reproduces their exact weights. Recalibrating produces a new experiment whose quality must be measured; it does not automatically reproduce the historical selected asset hashes.

## Data and roles

`configs/public/{edge,nano,super}-calibration.json` records scene IDs, image / decoded-pixel / prompt hashes and seeds. Each model is calibrated independently:

| Role | Scenes | Use |
|---|---:|---|
| calibration | 16 (9 RBench, 7 PAI-Bench-G) | Weight representation and repair proposals; seeds 301001–301016 |
| development | 16 | Trajectory diagnostics |
| validation | 32 | Candidate selection |
| final | 72 | Frozen subset evaluation |

The public full-benchmark evaluation has 650 + 1,044 scenes and overlaps these earlier roles. It is not an unseen-scene test. Do not fit on `final` or `full` results and then describe those same results as held-out selection evidence.

Inputs are fetched from fixed dataset revisions, using prompt lists from NVIDIA/cosmos revision `5b50bba0fb97f4aa07b03d299bc1a22bafa10ade`. The downloader checks every frozen image, decoded RGB image and prompt hash. Prompt JSONs can be read from a local checkout with `--cosmos-root`; images can be reused with `--dataset-root` or the Hub cache.

```bash
python tools/public/data.py --model edge --base-model /path/to/Cosmos3-Edge \
  --output work/edge
```

Use the pinned base revision from [INFERENCE.md](INFERENCE.md). We do not bundle original images or checkpoints in Git. Source weights remain offline inputs; every activation intervention consumes a current full-M input.

## End-to-end calibration workflow

Choose one GPU with enough free memory. Shared mode permits correctness and quality work but refuses formal tuning / performance claims. Use two sufficiently large devices for Super, in DM,AR order.

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/public/run.py calibrate --model edge --work work/edge
python tools/public/run.py capture   --model edge --work work/edge
python tools/public/run.py plan      --model edge --work work/edge --budgets 1.25 1.5 2 2.5
python tools/public/run.py joint     --model edge --work work/edge --budgets 1.25 1.5 2 2.5
```

Append `--dry-run` to print expensive worker commands. Operations run serially. Capture/calibration progress resumes only if input and source hashes match; plans are immutable, so use a new work directory when changing a protocol. OOM or insufficient memory pauses work without reducing precision or changing inputs. Full-layer capture and candidate assets require substantial disk space; panels are shared with AWQ/ARC rather than duplicated.

The stages produce:

1. `calibration/`: all generation projections/FFNs, all four denoising phases × both CFG branches on the 16 scenes; RMS summaries, measured intervention benefits and an RTN residual bank.
2. `capture/`: weights saved once and 16 full-M panels per operator. Each scene contributes one phase/CFG pair, with all eight pairs represented twice. These are **not** 16 × 8 retained matrix panels; the full phase statistics come from stage 1.
3. `controls/`: protected N128/full-K W1 proposals and current-input A quotas under each request cap, plus matched allocation controls. Each candidate includes `plan.json`, `compute.json`, compact `assets/`, and provenance.
4. `priors/`, `joint_calibration/`, `candidates/`: channel-scale and represented residual search at fixed per-operator counts, activation sensitivity and W1 support alternatives. An unchanged control remains feasible in every local search.

For channel scaling `s`, the compensated product is `(A/s)(W*s)ᵀ`; the packer's `prescale` is `1/s`. Search uses actual E4M3/E2M1 encoding, BF16 rounding and residuals recomputed from represented W0. Full-M amax is retained. The objective is equal-dataset mean panel NMSE plus 0.25 times the worst panel. The current public `joint` driver searches the historically implemented scale set in `importance_joint_calibrate.py`; it is not a claim to search every possible scale or reach a global optimum.

## Selection and validation

For example, compare a calibrated candidate with its protected control on development, then validation:

```bash
python tools/public/run.py prepare --model edge --work work/edge \
  --split validation --manifest work/edge/validation.json \
  --candidates work/edge/candidates/candidates.json \
  --methods bf16 fp8 fp4_base b1.5_protected_n b1.5_joint b2_protected_n b2_joint
python tools/public/run.py generate --model edge --work work/edge --manifest work/edge/validation.json --evaluation-dir work/edge/validation-run
python tools/public/run.py metrics  --model edge --work work/edge --manifest work/edge/validation.json --evaluation-dir work/edge/validation-run
python tools/public/run.py report   --model edge --work work/edge --manifest work/edge/validation.json --evaluation-dir work/edge/validation-run
```

Use `--evaluation-dir` to separate generation/metric/report output for each evaluation split; do not mix receipts from different manifests in one `generation/` directory. Local NMSE is a screening objective. Complete generation, paired LPIPS uncertainty and SSIM/L2 guardrails decide whether a candidate improves the chosen control. An FP8 pass is a separate condition, not implied by a lower repair-budget error.

## Research extensions

- `mechanism.py`, `representation.py`, `output_gain_outlier.py`, `outlier_detail*.py`: direct residual, scale selection and hard-to-represent-element interventions. Outlier moves require mandatory W1 support; audit the cross term jointly. These are not enabled in the released profiles.
- `structure_repair.py`, `structure_repair_run.py`: measured role/head/phase priors and matched-compute selection tests. Keep every base block.
- `schedule_*.py`, `head_coordinate.py`, `redesign_ffn_calibrate.py`: phase/head/FFN allocation interventions and combined-output checks.
- `expansion_*.py`: A-only/W-only, three/four-term and higher-order studies. Higher order is not monotonically better.

These older experiment drivers retain their original manifests and historical path defaults. Their functions and `--help` are available for extending a study; use the portable workflow above for a new standard calibration. [Baseline definitions](BASELINES.md) distinguish official execution from numerical controls.

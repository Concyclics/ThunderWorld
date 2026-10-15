# Contribution experiments

These scripts extend `thunderworld_fp4.cosmos.CosmosRunner`, the existing capture,
native kernels, budget compiler and cohort workflow. They do not implement a
second inference engine. All GPU experiments here use `shared-calibration`, a
single-process lock, GPU0/3 admission and a 12 GiB minimum free-memory check.
Request diagnostics are **not eligible for a speed leaderboard**.

## Paper evidence supplement (2026-09-18)

The supplement reuses the frozen 16 calibration / 16 development / 32 validation /
72 final scene cohort. Historical 72-scene results remain separate. New scripts
retain all base attention blocks and the existing Cosmos runtime:

| Script | Frozen comparison / qualification |
|---|---|
| `paper_study.py`, `paper_probe.py` | Full-four asset held fixed; independently remove projection, FFN, QK or PV cross terms; M129/M416/M960 and Graph current-input checks |
| `paper_mot.py`, `paper_mot_report.py` | Same current-student scene × four phases × both CFG branches × 28 layers; pure AR / equal-work DM / mixed boundary repair interventions; current residual-energy control |
| `paper_outlier.py`, `paper_outlier_report.py` | 2× and 2.5× fixed-support native RTN / Four-Six / magnitude / error / bounded output search; identical tensor storage; outlier × cross interaction |
| `paper_external.py` | Official ARCQuant/SharQ adapters on the same development scenes, with their actual execution and calibration limits |
| `awq_marlin.py` | Official ModelOpt NVFP4 AWQ-lite original-scale code extraction; exact frozen BF16 representation; native vLLM Marlin W4A16, independent decode/Graph probes and full generation |
| `evidence_outlier_gate.py` | Complete-video paired development guards; no promotion from local NMSE alone |
| `evidence_validation.py` | Audit unused reserved scenes, freeze the passing development candidate, then run paired scene-clustered validation without fitting |
| `evidence_final.py` | Wait for the frozen validation decision; consume the reserved final scenes only after the matched-control gate passes |
| `evidence_weight_diagnostics.py` | CPU full-weight K16 error-versus-amplitude ranking; tie-aware diagnostic, not an output-benefit or novelty claim |
| `evidence_costs.py`, `evidence_retest.py` | Shape-bound arithmetic ledger; emit exclusive per-device tuning → full-output equality → five shuffled rounds × three requests, including native AWQ |
| `evidence_figures.py`, `evidence_archive.py` | PDF/PNG scientific figures, raw JSON, source snapshots and sample hashes; no model weights/videos in Git |

Use the local archived manifests for exact model revision, case inputs, asset
hashes and commands. For a new workspace, first rebind source paths and generate
a new provenance record; do not edit a frozen running manifest. Example:

```bash
python tools/contribution_study/run.py paper -- prepare \
  --previous /data/workspace/aligned_sides_20260917 \
  --anchors /data/workspace/contribution_multiscene_20260917 \
  --cohort /data/workspace/repair_redesign_20260917/cohort.json \
  --awq /data/workspace/awq_multiscene_20260917 --output /data/paper/cross
python tools/contribution_study/paper_probe.py --output /data/paper/cross/probe.json
python tools/contribution_study/run.py paper -- run --output /data/paper/cross
python tools/contribution_study/paper_pipeline.py --root /data/paper \
  --workspace /data/workspace --archive results/paper_experiments
python tools/contribution_study/evidence_awq_workflow.py --root /data/paper --workspace /data/workspace
python tools/contribution_study/evidence_outlier_gate.py --run /data/paper/outlier_development \
  --output results/paper_experiments/outlier_quality
```

`evidence_awq_workflow.py` waits for the serial mechanism pipeline; it never runs
a second GPU experiment concurrently. Marlin calibration needs the same pinned
ModelOpt installation as the decoded AWQ reference. Native export rejects any
value outside its original FP4 codebook; it does not refit scales or approximate
the old BF16 checkpoint. Native accumulation still requires complete-generation
remeasurement. The inherited AWQ calibration is the historical sixteen-panel
definition, not a new sixteen-scene AWQ calibration; report that limitation.

The 2× Four/Six **control** passed development LPIPS/SSIM/L2 guards on both
datasets; its PAI PSNR decreased and is reported. Outlier candidates and the
2.5× variants failed. The frozen 2× control subsequently completed all 32
reserved validation scenes against equal-budget RTN and FP8/BF16. Mean LPIPS
improved by 3.23% / 4.49%, but both paired confidence intervals crossed zero;
neither dataset passed. The final workflow therefore stopped without consuming
the 72 reserved final scenes. This is not evidence of outlier novelty or a
validated low-budget improvement. Exact decisions remain in the frozen receipts.

```bash
python tools/contribution_study/evidence_validation.py prepare --root /data/paper \
  --workspace /data/workspace --cohort /data/workspace/repair_redesign_20260917/cohort.json \
  --gate results/paper_experiments/outlier_quality/gate.json
python tools/contribution_study/evidence_validation.py run --root /data/paper \
  --after /data/paper/awq_native/orchestrator.json --archive results/paper_experiments
python tools/contribution_study/evidence_final.py --root /data/paper \
  --workspace /data/workspace --cohort /data/workspace/repair_redesign_20260917/cohort.json \
  --archive results/paper_experiments
python tools/contribution_study/evidence_archive.py --root /data/paper \
  --workspace /data/workspace --archive results/paper_experiments
python tools/contribution_study/evidence_retest.py freeze --root /data/paper --output /data/paper/retest.json
python tools/contribution_study/evidence_retest.py emit --manifest /data/paper/retest.json \
  --output /data/pro6000-exclusive --device 0
```

The last command emits commands only. Repeat preparation/qualification into a
new directory on a 5090; neither device inherits the other's timing conclusion.
Formal benchmark workers refuse shared GPUs, probes, missing tuning entries,
changed assets or mismatched frozen full outputs. Use the native AWQ adapter's
weight/scratch counters; the generic external `prepared_weight_bytes` counter
does not include its Parameters. No claim of tensor-core saturation follows
from a compute ratio or from successful native compilation.

## Entry points

From the repository root, with the existing Cosmos environment:

```bash
export PYTHONPATH="$PWD/packages/fp4-repair/src:$PWD/experiments/cutlass_fused_repair:/tmp/thunderworld-joint-deps"
python tools/contribution_study/run.py --help
python tools/contribution_study/run.py collect-baselines --output /data/baselines --lock results/contribution_study_20260917/sources.lock.json
python tools/contribution_study/run.py freeze-cohort --cohort tools/contribution_study/cohort.json --history /data/experiment-history --output /data/cohort.frozen.json
python tools/contribution_study/run.py mechanism --capture /data/capture --assets /data/base-calibration --output /data/mechanism
python tools/contribution_study/run.py report --study /data/mechanism
```

`capture`, `search`, and `evaluate` forward arguments following `--` to the
existing `capture_budget.py`, `search_budget.py`, and `run_compute_cohort.py`.
The latter performs validation, freezes selection, then evaluates final seeds.
Do not send external baseline jobs to that native-only workflow. For external
adapters use `suite.py MANIFEST`, which serializes `external_infer.py` processes.
The archived command manifests show the exact model, case and assets used here;
replace local paths when moving machines and create a new provenance record.

The mixed native/external workflow is also available through the unified entry:

```bash
python tools/contribution_study/run.py evaluate -- external --manifest /data/validation-manifest.json --cohort /data/cohort.frozen.json --baseline-root /data/bf16-fp8-validation --legacy-template /data/legacy-template.json --output /data/complete-study
```

This records validation, freezes one of the two experimental candidates per cap,
runs the untouched final split, and applies ten-comparison correction including
the three/four-term FP8 anchor contrasts. The existing optimized 2.5x candidate
is retained as an additional anchor. A failed quality gate is not promotion.

Additional mechanism operations:

| Script | Operation |
|---|---|
| `attention_mechanism.py` | Full QK–softmax–PV interventions, head priors and integer quota curves |
| `augment_search.py` | Rerun proposed head combinations on both teacher seeds; augment the existing allocator |
| `head_coordinate.py` | Hold every linear support and per-group QK/PV total fixed; at most three head quota coordinate passes |
| `export_representation.py` | Offline RTN / FourOverSix-inspired represented-weight assets |
| `representation_trajectory.py` | Same-policy native trajectory comparison of those weight assets |
| `build_arcquant.py`, `arc_probe.py` | Isolated official ARCQuant build, K9216 adaptation and independent packed-byte reference |
| `external_probe.py` | Official SharQ adapter equality and separately reported upstream simulation differences |
| `awq_prepare.py` | Official ModelOpt AWQ-lite, explicitly disable activation quantization before calibration |
| `awq_multiscene.py`, `awq_multiscene_report.py` | Frozen 72-scene AWQ quality supplement, matched historical controls and paired scene-clustered statistics |
| `sage_probe.py` | Pinned SageAttention3 native build and real attention-panel comparison |
| `lowrank_probe.py` | Official ModelOpt **lite** SVDQuant operator comparison; not Nunchaku kernel performance |
| `storage_probe.py`, `asset_report.py` | Exact support-union storage and eager/Graph current-input verification |
| `packing_probe.py` | Same consumed codes and native output, selected residual writes versus full packing |
| `student_report.py` | Student-current-input BF16 errors, separate from teacher-input proxy errors |
| `quality.py`, `quality_gate.py` | Saved-output metrics, paired bootstrap and family-wise intervals |

Example budget export after augmentation:

```bash
python tools/contribution_study/augment_search.py --search /data/search/search.json --study /data/attention.json --capture /data/capture --output /data/augmented
python tools/joint_inference/export_budget.py /data/augmented/search.json --budgets 1.25 1.5 2 2.5 --output /data/candidates
python tools/contribution_study/asset_report.py /data/candidates/candidates.json --output /data/storage.json
```

## Baseline qualification and fairness

`sources.lock.json` records official URL, exact commit, license/dependency-file
hashes and submodules. `collect-baselines --lock` fetches that exact commit,
including on a new machine. It does not install dependencies or execute source.
Build scripts preserve upstream checkouts and record patches/library hashes.
ARCQuant/SharQ/Nunchaku checkouts without a detected redistribution license are
not vendored into this repository.

Initialize the repositories' pinned submodules before building. SharQ has a
reproducible `build_sharq.py --sources ... --pybind11 ...` entry. ARCQuant uses
`build_arcquant.py`; its patch is saved separately. Both explicitly target
SM120a. Sage's build receives an already pinned CUTLASS checkout, preventing
upstream setup from fetching an unversioned dependency.

- **ARCQuant**: official native quantization/GEMM, Cosmos K9216 dispatch/template
  patch, fixed 50% activation channels ranked from full teacher panels. This is
  not the paper's model-wide proportion search. Host-scalar alpha makes this
  adapter eager-only; quality can be measured, native Graph speed is unqualified.
- **SharQ**: official native kernels with `extra_fusion=False`, shape adapter and
  registered weight buffers. Its official simulation is a distinct numerical
  path; agreement is measured rather than assumed.
- **AWQ**: official `NVFP4_AWQ_LITE_CFG` has its input/output quantizers disabled
  **before** calibration to create W4A16. Offline BF16 decoded-weight export is
  checked bitwise against official forward. It is a quality anchor, not a packed
  W4 deployment/storage or fused-dequantization speed claim.
  The September 18 supplement separately qualifies actual packed W4A16 Marlin
  execution through `awq_marlin.py`: exact original-scale export, all-layer
  decode/Graph probes and fresh complete-video metrics. Packed storage belongs
  to this adapter; formal speed still requires its exclusive retest.
  The multi-scene workflow accepts this adapter through `baseline_adapter` in
  the existing dataset manifest. It checks all-layer asset/model hashes, BF16
  historical output equality and scene-switch Graph replay before generation.
  Run `awq_multiscene.py prepare --previous PREVIOUS_STUDY --assets AWQ_ASSETS
  --output NEW_STUDY`, then `probe` and `run` with the same `--output`. Follow
  `results/awq_multiscene_20260917/PROTOCOL.md` for the frozen comparison protocol.
- **Four Over Six**: actual E4M3/E2M1 scale-choice mechanism reproduction with
  unchanged prescale and identical residual arithmetic. It is not a qualified
  upstream native Cosmos port.
- **ViDiT-Q**: official source collected. Existing layer/phase sensitivity search
  is an inspired mechanism control; it is not a faithful ViDiT-Q reproduction.
- **SageAttention3**: official native API, explicit K copy (upstream mutates K),
  explicit GQA replication. Report this duplicate packing work. No silent BF16
  fallback is allowed for the evaluated D128 path.
- **SVDQuant**: official Nunchaku source collected; operator experiment uses
  NVIDIA ModelOpt's explicitly named *lite* SVDQuant. BF16 low-rank work/storage
  are separate from FP4 work; this is not an original Nunchaku performance result.

External methods retain their real costs. They are not relabeled as having the
same FP4 budget. A-only/W-only controls can saturate below a requested cap; report
actual ratios. They are feasible under the same cap, not equal-compute matches.

## Cohorts and numerical boundaries

7101–7102 teacher panels propose choices; 7103–7104 are development trajectories.
The proposal reserves 171001–171008 for validation and 181001–181016 for final.
`freeze-cohort` checks local executed records/output filenames, shifts both new
splits together on collision and writes an immutable file. It cannot audit a
remote server. Final external runs additionally require a frozen selection whose
configuration digests match exactly. Do not tune on final results.

For outlier removal, each K16 group moves at most one element, only inside
reserved mandatory W1 blocks. Residuals always come from the actually represented
W0. `joint_diagonal` is a diagonal activation-weighted proposal objective, **not**
an exact output-gain oracle; output errors are measured afterward. The original
mechanism study uses full-W1 three/four-term anchors. The September 18 extension
also measures fixed sparse support at 2×/2.5× and complete native generation;
the output search is bounded and is not an exhaustive per-group oracle.

Head tables are a separable proposal. `augment_search.py` reruns each complete
proposed configuration on both full-M teacher inputs before entering the budget
allocator. Native selectors consume the current input and retain all base
attention blocks. CPU staging may expand W1 during preparation, but budgeted
`RepairLinear` retains only the physical CPU/GPU support union afterward.

## Verification

```bash
PYTHONPATH=packages/fp4-repair/src:tools/contribution_study python -m pytest packages/fp4-repair/tests/test_compute_budget.py tools/contribution_study/test_study.py -q
compute-sanitizer --tool memcheck --error-exitcode 99 python tools/contribution_study/arc_probe.py --build /data/arc-build --output /data/arc-memcheck.json
compute-sanitizer --tool racecheck --error-exitcode 99 python tools/contribution_study/arc_probe.py --build /data/arc-build --output /data/arc-racecheck.json
```

Do not infer whole-model VRAM from packed generation-weight payload. Do not infer
request latency from MMA ratios. No shared-GPU result establishes a latency
Pareto frontier, and a local NMSE win does not establish full-generation quality.

`retest.py` emits (and only with `--execute` runs) the exclusive PRO 6000/5090
commands: separate device tuning, five fixed randomized rounds, three seeds per
configuration, debug off and strict tuning lookup. It requires a frozen quality
selection and native jobs with config/plan/asset hashes. Do not execute it while
the GPU is shared. Historical execution controls remain separate evidence.

## Portable frozen policies (no model-weight transfer)

`portable.py export` writes calibration metadata, support plans, exact packed
hashes and sparse byte corrections for CPU/CUDA rounding-boundary differences.
It transfers neither W0 nor W1 planes. `restore` downloads nothing and runs on
CPU: use the separately downloaded **same revision** of Cosmos3-Edge. Every
reconstructed tensor must match the original frozen asset, or restoration stops.
A different Torch encoding behavior cannot silently pass this check. Temporary
full planes exist only during offline reconstruction; deployment assets retain
only the support union. The complete 168-layer, eight-policy reconstruction was
verified in this study. Earlier uncorrected reconstruction was rejected, not used.

```bash
python tools/contribution_study/portable.py export \
  --manifest /data/complete-study/frozen/manifest.json \
  --selection /data/complete-study/frozen/selection.json --output /data/portable
python tools/contribution_study/portable.py restore --package /data/portable \
  --model /models/Cosmos3-Edge-a9d944e --output /data/restored
python tools/contribution_study/retest.py /data/restored/retest.json \
  --output /data/device-retest --device 0
```

The last command only emits commands. Add `--execute` with a new output directory
when that GPU is exclusively available. The 13 native configurations comprise
four frozen candidates, four controls, BF16, FP8 and FP4 base/three/four anchors.
A frozen negative candidate is still a diagnostic control, not a quality-qualified
recommendation. ARCQuant's host-alpha adapter and decoded AWQ are excluded from
this native timing suite. Follow `docs/OPTIMIZATION_HANDOFF.md` for the pinned
Cosmos/Diffusers environment; this package does not replace those dependencies.

After the frozen final workflow completes:

```bash
python tools/contribution_study/run.py report --workspace /data/workspace \
  --output results/contribution_study_20260917
```

`FINAL_REPORT.md`, `acceptance.json` and `frontier.json` preserve validation
eligibility separately from final statistical tests. A candidate failing the
validation SSIM/L2 guards cannot become recommended because of a favorable final
seed set. Raw receipts and outputs are indexed separately; no shared timing is
promoted to a speed result.

`output_gain_outlier.py` additionally tests actual full-output responses for all
16 possible removed positions inside each K16 group of one pre-reserved
N128/K128 block. It reevaluates the complete combined mask with BF16 output
rounding, under both three/four terms. This bounded decoded-reference probe is
explicitly distinct from `joint_diagonal`; it is not a native rounding oracle or
a deployment asset certified under the four request budgets.

The portable retest additionally checks all 16 frozen complete-output signal
hashes **after device tuning and before any benchmark round**. Any mismatch
stops the timing suite and requires full quality requalification; operator-level
tuning probes alone cannot certify an unchanged trajectory. This is an equality
check, not a new search over final-test perceptual metrics.

`runtime_costs.py --study /data/complete-study --output /data/runtime-costs.json`
audits all 16 request ledgers per cap and records measured resident tensors and
allocated/reserved peaks. External baselines have separate logical work counts
(dense and structured-sparse FP4, FP8, BF16); these explicitly lack issued-tile
padding qualification and are not assigned artificial same-budget ratios.

## Joint denoising schedules

`schedule_calibrate.py` measures current-input projection, complete FFN and
complete attention errors in four denoising phases and both CFG branches.
`schedule_plan.py` reallocates A/W/cross repair within the fixed offline W1 bank
under 1.5×, 2× and 2.5× request budgets. QK and PV have separate fractional
per-head quotas, resolved to integer tiles for each actual key length before
Graph capture. The two CFG lengths vary independently: budget verification
enumerates their Cartesian product, not just equal-length pairs.

The workflow, frozen cohort rules and commands are in
[`DESIGN.md`](../../results/joint_schedule_20260917/DESIGN.md), with stage tables
in [`SCHEDULES.md`](../../results/joint_schedule_20260917/SCHEDULES.md).
`schedule_report.py` retains the fixed control if no new candidate improves
LPIPS while preserving mean SSIM and L2 on **both** development datasets.
Only a frozen passing candidate enters the reserved validation scenes.

```bash
PYTHONPATH=packages/fp4-repair/src:tools/contribution_study python -m pytest \
  packages/fp4-repair/tests/test_config.py \
  packages/fp4-repair/tests/test_compute_budget.py \
  tools/contribution_study/test_schedule.py -q
```

`schedule_diagnose.py` reports component/phase/CFG local tradeoffs under a common
mean-NMSE definition. Those local errors are diagnostic, not complete-trajectory
quality evidence. `schedule_archive.py` stores policies, raw calibration and
output receipts without copying model weights or generated videos into Git.

## Single-sided repair with matched FP4 attention

`aligned_sides.py prepare` creates fixed single-A and single-W baselines with
full ACT2 FP4 attention. Their generated projection/FFN/attention MMA is exactly
2× C0 including padded tiles. It also switches the existing 1.5× and 2× joint
policies to the same attention while preserving their linear quotas and assets.
Those attribution variants may use less than their budget caps; costs are not
silently treated as equal. Original joint and BF16-attention results are reused
only after matched-input/configuration and output-hash verification.

`aligned_sides_probe.py` qualifies endpoint numerical equivalence before
`aligned_sides.py run` starts serial shared-GPU generation. The report includes
paired comparisons against both one-sided baselines, current-request ledgers,
LPIPS/SSIM/L2/PSNR, W1 storage and memory. See the
[protocol](../../results/aligned_sides_20260917/PROTOCOL.md) for exact commands
and the development-only evidence boundary.

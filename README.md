# ThunderWorld

**CUTLASS follow-up on RTX 5090 (2026-09-14):** [measured gains, direct scale
packing, full-trajectory equality and remaining bottlenecks](docs/CUTLASS_5090_FOLLOWUP_20260914.md).

**Start here for another-device research:** [optimization status, measured speedups,
repair-budget knobs, and runnable handoff](docs/OPTIMIZATION_HANDOFF.md) (2026-09-11).
Includes versioned Edge/Nano runtime policies and the latest 86-configuration
A/W fusion audit. Graph/fusion is the execution baseline; experimental pairing
and quality-calibration paths have explicit evidence boundaries.

ThunderWorld studies FP4 inference for mixed autoregressive–diffusion (AR–DM)
world models. The current research direction retains complete low-precision base
products and uses additional FP4 products to repair selected errors. Layer,
denoising phase, head, and directed modality structure guide the repair budget.
Cosmos3-Edge and Cosmos3-Nano are the current integration targets.

**Integrated paper report (2026-09-11):** [motivation, contributions, methodology, implementation evidence, and AI-generated architecture figure](reports/thunderworld_paper_20260911/REPORT.md), with [HTML](reports/thunderworld_paper_20260911/REPORT.html) and [PDF](reports/thunderworld_paper_20260911/REPORT.pdf) editions.

**Current design (2026-09-10):** [iterFP4 method, implementation audit, and evidence boundaries](docs/THUNDERWORLD_ITERFP4_DESIGN.md)
and [implementation priorities, evaluation matrix, and demo plan](docs/ITERFP4_IMPLEMENTATION_PLAN.md).
The distribution-aware quantizer and complete structure-driven repair policy are
planned work. The existing end-to-end pilot uses RTN W4 + full activation repair;
it does not yet validate the complete proposed method.

RTX 5090 local kernel optimization (2026-09-09): [results and reproduction](docs/RTX5090_RESULTS.md),
[60-case latency table](results/rtx5090_20260909/TABLE.md), and
[tuned launch settings](results/rtx5090_20260909/recommended.json).
These are operator and local FFN measurements, not full-model generation speedups.

Cosmos3-Edge / Nano end-to-end Generator comparison (2026-09-10):
[measured speed, memory, and video consistency](docs/COSMOS3_E2E_5090.md), with
[reproducible full-pipeline adapters](tools/cosmos_e2e/README.md).

Fresh full-step findings and quality ablations (2026-09-10):
[layer/modality findings, head trajectories, and temporal reuse](docs/FINDINGS_RESAMPLING_20260910.md),
with [sampling and ablation tools](tools/finding_resample/README.md).

Automatic FFN repair budgets and hardware profiling (2026-09-10):
[phase/layer allocation, quality gates, and Tensor Core measurements](docs/AUTO_BUDGET_5090_20260910.md),
with [calibration and execution tools](tools/auto_budget/README.md).

Parameter and overlap audit (2026-09-11):
[available knobs, remaining bottlenecks, and measured SIMT/GEMM concurrency](docs/PARAMETERS_AND_OVERLAP_5090_20260911.md).

CUDA Graph, FFN fusion, and shape-dependent double buffering (2026-09-11):
[implementation, equal-output comparisons, and measured limits](docs/GRAPH_FUSION_5090_20260911.md).

> **Working thesis:** retain dense FP4 base computation, allocate precision repair
> using world-model structure, and validate error over complete trajectories.

The repository also preserves the audited sparse-attention evidence snapshot
as of **2026-08-29**. Those experiments remove attention work; their quality and
speed results must not be attributed to the newer FP4 precision-repair method.

## Historical sparse-attention results (2026-08-29)

| Operating point | Evidence | Joint-QK reduction | 16K attention speedup | Quality screen | Status |
| --- | --- | ---: | ---: | ---: | --- |
| q12, 50% GQA groups sparse | 1 T2V prompt × 3 seeds × 20 steps | 36.68% | 1.21× measured | min SSIM 0.923; min PSNR 25.30 dB | current 0.90 development point |
| q12, 75% GQA groups sparse | 1 T2V prompt × 2 seeds × 20 steps | 55.02% | 1.41× measured | min SSIM 0.856; min PSNR 21.29 dB | provisional 0.80 aggressive point |
| all-head sparse | 1 T2V prompt × 1 seed × 20 steps | 73.36% kernel case | 1.93× measured | best comparable SSIM 0.770 | quality failure |

The 1.21×/1.41×/1.93× numbers are **H200 attention microbenchmarks**, not
complete Cosmos3 generation speedups. With the explicit Amdahl assumptions used
in the analysis—Generator share 0.9 and attention share 0.387—the corresponding
end-to-end estimates are approximately 1.065×, 1.112×, and 1.202×.

The all-sparse kernel is therefore useful as an execution bucket for selected
`layer × stage × relation` cells. Current evidence does **not** support making
every layer, head, and diffusion step sparse.

![ThunderWorld finding evidence overview](figures/finding-evidence-overview.png)

## Historical sparse-attention research components

1. **Relation/head/modality characterization.** Historical directed row-mass
   ratios are 4.09–13.53×; their token-count normalization is qualified in the
   [fresh resampling](docs/FINDINGS_RESAMPLING_20260910.md). 37.7% of Generator head-relation units are
   specialists, 45.7% hybrid, and only 16.6% mixers; action/audio self-relations
   are high-information dense anchors.
2. **Interaction-aware trajectory compiler.** Head-wise measurements are
   compiled into executable GQA buckets. A joint-projected error Gram, complete
   diffusion rollout, and uncertainty bound replace local-error or hand-written
   layer heuristics.
3. **Joint-softmax sparse runtime.** Dense anchors, local/semantic block lists,
   directed relation skips, and cross-step reuse share one online-softmax state,
   so DM←DM and DM←AR preserve their original joint normalization.

## Historical sparse-attention negative results

- Shallow DM←AR hot blocks do not predict deep DM←DM hot blocks after position
  control: residual correlation is -0.065. The two branches must select blocks
  independently.
- Naive K-only semantic reorder does not close the oracle gap and is not a
  default runtime path.
- Hand-written depth relocation fails an equal-QK 20-step screen: SSIM falls
  from 0.906 to 0.861 even though its lookup-table speed estimate improves.
- Reusing prior-step attention outputs reaches only about 0.77–0.80 paired
  SSIM. Reuse descriptors/block lists, but recompute QK/softmax/PV.
- An apparent all-head SSIM 0.827 result used only four diffusion steps and is
  excluded from the comparable 20-step frontier.

## Repository map

- [`docs/THUNDERWORLD_ITERFP4_DESIGN.md`](docs/THUNDERWORLD_ITERFP4_DESIGN.md) —
  current project thesis, code audit, related work, and hardware claim boundaries.
- [`docs/ITERFP4_IMPLEMENTATION_PLAN.md`](docs/ITERFP4_IMPLEMENTATION_PLAN.md) —
  planned implementation, calibration artifacts, experiments, and three demos.
- [`docs/COSMOS3_E2E_5090.md`](docs/COSMOS3_E2E_5090.md) — measured Edge/Nano
  Generator pilot, including quality limitations and negative results.
- [`docs/EXPERIMENTAL_FINDINGS.md`](docs/EXPERIMENTAL_FINDINGS.md) — complete
  historical sparse-attention evidence, definitions, and claim boundaries.
- [`docs/METHOD_AND_IMPLEMENTATION.md`](docs/METHOD_AND_IMPLEMENTATION.md) —
  historical sparse-attention algorithm/runtime design and engineering decisions.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — source inventory,
  regeneration commands, and validation rules.
- [`reports/thunderworld-current-evidence.html`](reports/thunderworld-current-evidence.html)
  — self-contained technical report.
- [`data/evidence_snapshot.json`](data/evidence_snapshot.json) — machine-readable
  reviewed metrics.
- [`data/finding_ledger.csv`](data/finding_ledger.csv) — frozen
  finding→implementation→contribution decisions.
- [`data/source_manifest.json`](data/source_manifest.json) — SHA-256 provenance
  for every private-workspace aggregate used to build this snapshot.

## Reproduce the public snapshot

The validation step has no third-party Python dependency:

```bash
python3 scripts/validate_evidence.py \
  --data-dir data \
  --output data/validation_report.json
```

To rebuild the snapshot from the private experiment workspace:

```bash
python3 scripts/build_evidence_snapshot.py \
  --source-root /path/to/WorldModel \
  --output-dir data

python3 scripts/validate_evidence.py \
  --data-dir data \
  --output data/validation_report.json
```

The public bundle intentionally excludes model weights, datasets, generated
videos, raw attention traces, and multi-gigabyte Parquet files. Their reviewed
aggregate identities and hashes remain in `data/source_manifest.json`.

## Confidence of the historical snapshot

**Share with caveats.** The calculations and experiment-family comparability
pass the included audit, but the latest quality frontier is still dominated by
one T2V prompt. Final VBench, control adherence, AV synchronization, action
error, trajectory error, and task success remain open.

## Configurable FP4 repair runtime

The [standalone FP4 package](packages/fp4-repair/README.md) provides independent FFN, attention core, and projection repair settings, per-head block budgets/masks, a head-atlas adapter, and native SM120 Tensor Core kernels. See [validation and current performance limits](packages/fp4-repair/VALIDATION.md) before use; the new portable FFN has operator validation but no full-model quality claim.

The [FFN block-repair study](docs/FFN_BLOCK_REPAIR_20260907.md) adds K128/K256 activation-channel repair and an offline sparse weight-residual bank, with equal-budget comparisons, full local FFN replay, and native fused kernels. These experimental candidates leave production defaults unchanged.

## RTX 5090 kernel handoff

The [receiver guide](tools/kernel_handoff/README.md) provides a data-only workload bundle, pinned Hugging Face shard downloads, offline weight preparation, and portable FFN/attention benchmarks. See the [Codex task brief](tools/kernel_handoff/CODEX_TASK.md) for the optimization protocol.

# RTX 5090 joint execution: full-request evidence

See the [English report](../../docs/JOINT_E2E_5090_20260914.md) for the measured
result, diagrams, numerical limits, implementation, and reproduction commands.

| Budget | Selected configuration | Median complete request | BF16 speedup |
|---|---|---:|---:|
| 1.25 | `joint125` | 2.1155 s | 1.263× |
| 1.50 | `cutlass150` | 2.1858 s | 1.222× |

BF16 Graph: 2.6716 s. Scope: Cosmos3-Edge I2V, `robot_observe`, 192×320,
61 frames, 35 steps, CFG 6, resident warm execution on this RTX 5090.
Five isolated process rounds × three seeds per configuration; 75 requests total.
Model loading, preparation, tuning, Graph capture, and file serialization are
outside the timer. Host request setup, conditioning, denoising, decoding, and
output conversion are inside it.

- `summary.json`: all 75 samples, medians/P90, paired-round intervals, device and
  source identities, actual dispatch counts, memory, startup costs, and aggregate
  issued-MMA/packing accounting. Dispatch counts count observed module/phase
  records; they are not per-request execution counts.
- `latency.csv`: individual request measurements.
- `quality.json`: two development seeds, LPIPS/SSIM/relative L2 versus BF16, and
  four byte-identical joint/control trajectory pairs. No held-out quality claim.
- `diagnostic.json`: separate event intervals, with output equivalence checked.
  These do not sum exactly to host wall time; do not infer additive percentages.
- `recommendations.json`: selected configuration/profile per fixed budget.
- `e2e_comparison.*`, `paired_rounds.*`: PNG and SVG figures.
- `validation.txt`: final GPU regression output, 218 passed.
- `evidence.tar.xz`: original run receipts and logs, frozen configuration/profile
  files, matrices, full summaries/ledgers, input asset/plan and CUTLASS build
  manifests, and the explicitly rejected padding experiment. Model weights,
  binaries, and generated video NPZ payloads are excluded. Receipts preserve the
  raw signal hashes; payloads remain in the local experiment directory.

The six additional performance budget/seed pairs are also identical, and all
repeated outputs are deterministic. All three performance seeds give distinct
outputs. These equivalence checks compare each optimized FP4 configuration to
its same-budget CUTLASS control, not to BF16.

Validate the published artifacts with `sha256sum -c SHA256SUMS` from this folder.
Execution profiles bind this GPU, source, libraries, and request shapes. Use the
reproduction script to generate new profiles on PRO 6000, Nano, Policy, or a new
input shape; do not transplant the saved 5090 profile.

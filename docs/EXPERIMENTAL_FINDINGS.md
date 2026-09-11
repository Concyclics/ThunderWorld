# ThunderWorld experimental findings

Evidence cutoff: **2026-08-29**. Compilation date: **2026-09-01**.

> These are historical structure and sparse-attention findings. The current
> [iterFP4 design](THUNDERWORLD_ITERFP4_DESIGN.md) reuses them as candidate priors;
> their ability to predict FP4 repair benefit requires separate validation.
> The [2026-09-10 full-step resampling](FINDINGS_RESAMPLING_20260910.md) adds
> Edge/Nano measurements and qualifies the directional normalization below.

## Technical summary

The experiments support a narrower and stronger thesis than “apply sparse
attention to Cosmos3”:

> **Sparsity is local to tower, directed relation, layer, head, and diffusion
> stage; approximation risk only becomes reliable after the candidates interact
> along the complete diffusion trajectory.**

The evidence is sufficient to retain a relation-aware GQA policy and selective
all-sparse kernel buckets. It is not sufficient to claim globally sparse,
quality-neutral production inference.

## Architecture and metric definitions

Using query rows and key columns, the valid Cosmos3 Generator branches are:

| Branch | Semantics | Current role |
| --- | --- | --- |
| AR←AR | causal Reasoner self-attention | dense and cached by default |
| DM←AR | Generator reads fixed Reasoner context | directed relation budgets/skip; AR index reusable across steps |
| DM←DM | bidirectional Generator self-attention | main local/semantic sparse target |

DM←DM and DM←AR are partitions of one softmax over `[K_AR; K_DM]`. Their
separate outputs cannot be independently normalized and added. Every valid
runtime path must keep a common online-softmax state or merge branch LSE values
with `logaddexp`.

The main reported metrics mean:

- **Joint-QK reduction:** removed score entries divided by the original
  DM←DM + DM←AR score count. It excludes FFN, projections, VAE, routing,
  communication, and other non-attention work.
- **Paired SSIM/PSNR:** same-seed candidate versus dense generated output. It is
  a trajectory-fidelity screen, not human preference or task success.
- **Attention speedup:** dense three-way attention latency divided by sparse
  amortized attention latency on H200 NVL random tensors.
- **Estimated E2E speedup:** Amdahl estimate using Generator share 0.9 and
  attention share 0.387. It is not measured full-pipeline latency.

## Evidence families must remain separate

| Experiment family | Scope | What it establishes |
| --- | --- | --- |
| Relation atlas | 6 tasks, 1 sample/seed, 4 steps, 7/36 layers, 32 heads | discovery of directed modality, anchor, and layer patterns |
| Equal-budget candidate campaign | 8 tasks × 24 prompts × 3 seeds, 20/40 steps, all 36 layers × 32 heads | geometry/semantic/hybrid characterization and local-to-final mismatch |
| Timestep trace | 24 T2V prompts × 3 seeds × 20 steps × 36 layers × 32 heads × 2 candidates | local sparsifiability and layout stability by diffusion step |
| Trajectory/GQA route validation | 4 held-out T2V prompts × 3 seeds × 20 steps | non-additive group error and local-to-trajectory failure modes |
| DRSA screen | T2V/I2V/DROID, 1 prompt/task, seed 29, 20 steps | equal-QK direction and depth-policy screening |
| q12 50% quality point | 1 T2V prompt × 3 seeds × 20 steps | latest 0.90 development operating point |
| q12 75% quality point | 1 T2V prompt × 2 seeds × 20 steps | provisional 0.80 aggressive point |
| H200 microbenchmark | random tensors, one attention layer, DM 4K/8K/16K | kernel scaling and upper bound, not model quality |

## Finding 1 — directed cross-modal relations are strongly asymmetric

Four paired relations show 4.09–13.53× forward/reverse attention-mass
asymmetry:

| Relation pair | Ratio | 95% CI | Same-direction consistency |
| --- | ---: | ---: | ---: |
| T2VS sound←vision / vision←sound | 8.14× | [7.14, 9.30] | 94.2% |
| Bridge action←vision / vision←action | 12.72× | [10.10, 16.13] | 95.5% |
| DROID action←vision / vision←action | 13.53× | [10.34, 18.05] | 94.6% |
| I2V clean←noisy / noisy←clean | 4.09× | [3.59, 4.72] | 93.3% |

The two directions in a pair have the same score-entry count, but that alone
does **not** remove token-count effects from mean per-query row mass. The frozen
aggregate does not retain the counts/normalization needed to reconstruct this
distinction. Its reported ratios above must not be read as verified per-edge
density ratios. The 2026-09-10 resampling reports both definitions and finds
substantial attenuation after normalization, especially on Nano. Different
checkpoints/tasks prevent treating that resampling as an exact replication.
A separate 20-step equal-QK I2V counterfactual
reinforces the mechanism: deleting the low-mass direction gives SSIM 0.975,
while deleting the high-mass reverse gives 0.776.

**Implementation:** preserve `(query role, key role)` in the policy key; assign
separate GQA budgets and mandatory anchors to both directions.

**Claim boundary:** the structural bootstrap resamples head-layer units, not 24
independent prompts. The finding is a primary candidate, not yet a final
population estimate.

## Finding 2 — head archetypes are stable enough for offline classification

After removing the Reasoner units, the Generator atlas contains 2,464
`layer × head × query-role` units:

- specialist: 929, or 37.7%;
- hybrid: 1,127, or 45.7%;
- mixer: 408, or 16.6%.

The dominant relation is stable across sampled denoising passes in 94.5% of
units. Cross-task dominant-relation agreement for noisy-vision queries is 99.1%
for T2V↔T2VS, 93.8% for T2V↔I2V, and 90.2% for Bridge↔DROID.

**Implementation:** profile every Q-head, then compile into
`GQA group × relation × stage × archetype` buckets. Split one KV group into two
Q-head buckets only if the measured cost model justifies the extra launch and
descriptor overhead.

**Negative boundary:** a fixed hand-written layer relocation is not safe. At
exactly the same 35.33% joint-QK reduction, generic head selection reaches SSIM
0.906, a hand depth schedule reaches 0.861, and cost relocation reaches 0.858.
Layer is a compiler feature, not a standalone fixed policy.

## Finding 3 — small action/audio relations are dense anchors

| Relation | Attention mass | Score share | Mass / score share |
| --- | ---: | ---: | ---: |
| DROID action←action | 33.54% | 0.188% | 178× |
| Bridge action←action | 35.23% | 0.117% | 302× |
| T2VS sound←sound | 48.89% | 0.366% | 134× |

The inexpensive small relations carry disproportionate information. Sparsifying
them creates little compute benefit and high task risk.

**Implementation:** instruction, special, action, audio, clean-control, and
physical-time-neighbor blocks are hard anchors. The sparse budget targets the
large noisy-vision residual.

## Finding 4 — semantic blockification is required, but naive reorder fails

At 99% attention mass, token-level oracle score saving is 35.4%–43.3% across
the modality tasks. Keeping natural 32-token blocks reduces that opportunity to
11.6%–19.4%. Important tokens are sparse but scattered in geometric order.

The full eight-task equal-budget campaign confirms that selector choice depends
on relation/task:

- clean-control PAI-C tasks favor semantic retrieval over hybrid by about
  1.86–2.46 dB final PSNR;
- DROID forward favors anchor-preserving hybrid over semantic by 3.89 dB;
- DROID inverse action relative-L2 is 0.049 for hybrid versus 0.426 for
  semantic;
- naive semantic reorder does not consistently beat virtual block lists at the
  same active-tile budget.

**Implementation:** use local/physical-time anchors plus
query/head/timestep-conditioned semantic block retrieval. Prefer virtual block
lists; perform physical K/V reorder only when reuse amortizes the copy below 15%
of saved attention time.

## Finding 5 — approximation errors interact across heads and steps

The projected GQA error cancellation ratio has median 0.412: the norm of the
combined error is not the sum of individual group-error norms. Lower local
coverage can even produce worse final quality than a higher-coverage policy.
The eight-task campaign also shows that lower attention-output drift can yield
worse final trajectory PSNR on clean-control tasks.

**Implementation:** candidate selection must be calibrated after joint-softmax
merge and output projection, then evaluated over complete diffusion rollouts.
The compiler should use a cross-candidate error Gram and conservative residual
UCB instead of adding local output drift.

## Finding 6 — timestep adaptation should reuse layouts, not outputs

Across the full 20-step T2V trace:

- strict local safe share grows from 3.82% at step 0 to 12.93% at step 19,
  a 3.39× increase;
- adjacent safe-set Jaccard averages 0.930;
- cross-attention top-anchor Jaccard averages 0.919;
- a four-stage allocation reduces mean local robust error by 11.8% and p95 by
  24.3% at equal sparse-head coverage versus one static mask.

However, previous-step merged-output drift is U-shaped and grows near the final
step. Closed-loop output-cache controls obtain only about 0.77–0.80 SSIM.

**Implementation:** stage-wise budget, low-frequency semantic descriptor refresh,
warm-started clustering, and block-list reuse are retained. Attention-output
reuse is disabled by default.

## Negative result — DM←AR hot blocks do not transfer to DM←DM

Across 45 dense trajectories, raw cross/self correlation is partly explained by
position. After removing the position effect, residual correlation is -0.065.
Transferring cross anchors changes held-out omitted mass by -0.0144 and
transferring cross-derived query budgets changes it by -0.0203; both are worse
than independent selection.

**Decision:** remove cross→self transfer from the primary algorithm. Keep the
negative result as an ablation and let the trajectory compiler model branch
interaction only after each branch selects independently.

## Current quality–compute frontier

| Policy | Scope | Joint-QK reduction | 16K attention | Amdahl E2E | Min SSIM | Min PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q12, 50% GQA sparse | 1 prompt, 3 seeds, 20 steps | 36.68% | 1.211× | 1.065× | 0.923 | 25.30 dB |
| q12, 75% GQA sparse | 1 prompt, 2 seeds, 20 steps | 55.02% | 1.407× | 1.112× | 0.856 | 21.29 dB |
| all-head sparse, best comparable | 1 prompt, 1 seed, 20 steps | 73.36% kernel case | 1.930× | 1.202× | 0.770 | 18.44 dB |

The q16 exact-Taylor output with SSIM 0.827 is not in this frontier because its
saved generation arguments show `num_steps=4`.

The 0.8 mode should therefore mean **75% GQA sparse plus selective all-sparse
layer/stage buckets**, not blanket all-head sparsity. It should also retain a
PSNR≥20 dB screen and task-specific non-inferiority; DROID action/task success
must not be allowed a generic 20% degradation.

## Kernel scaling and end-to-end expectations

For the all-sparse attention bucket on H200 NVL:

| DM tokens | AR=61 | AR=2,048 |
| ---: | ---: | ---: |
| 4,096 | 1.03× | 1.02× |
| 8,192 | 1.32× | 1.32× |
| 16,384 | 1.93× | 1.80× |

The sparse kernel becomes worthwhile as dense DM self approaches the dominant
quadratic cost. Short sequences should fall back to dense FA3 through a measured
lookup table. Additional patterns cannot be added as independent savings when
they delete overlapping score domains: semantic blocks, relation skip, head
skip, and stage budgets must be composed before computing the joint-QK cut.

At approximately 14.7K DM tokens, attention-only acceleration is unlikely to
produce 2× end-to-end speedup because attention is only a fraction of the
complete Transformer and Generator. A credible 2× target requires ultra-long
contexts plus additional complete block-step/FFN reuse, not only sparse QK.

## Frozen contribution map

| Finding | Implementation | Publication role |
| --- | --- | --- |
| Relation/head/modality heterogeneity | directed pattern bank + GQA compiler | characterization contribution |
| Local error fails to predict trajectory risk | projected Gram + rollout/UCB calibration | policy contribution |
| AR/DM share one normalization | fused online-softmax sparse merge | runtime contribution |

Everything else is folded into these three roles or retained as a negative
result. Exact LSE merge is a correctness invariant; it should not be claimed as
a stand-alone algorithm unless a calibrated tail correction materially improves
the final quality–compute Pareto.

## What remains unproven

- 24 independent final prompts per task with paired bootstrap confidence
  intervals have not been run for the latest q12 policy.
- The latest frontier does not yet include I2V, T2VS, PAI-C, or DROID official
  task metrics.
- The H200 kernels use random tensors and are not integrated full-rollout
  wall-clock results.
- Nano→Edge, 20→40 steps, short→long context, and a second GPU have not been
  confirmed for the frozen policy.
- Context-parallel dynamic sparse communication is not implemented; multi-GPU
  execution must fall back to dense.

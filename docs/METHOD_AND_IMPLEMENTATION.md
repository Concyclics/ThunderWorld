# Method and implementation decisions

## Recommended runtime

The retained method compiles measurement-time Q-head information into four
regular execution buckets:

1. dense FA3;
2. 3D local + dense anchors;
3. semantic/hybrid block list + dense anchors;
4. DM←AR directed relation skip.

The policy key is:

```text
checkpoint topology × layer × diffusion stage × KV group × Q-head bucket
× directed relation × archetype × tile budget
```

Every bucket participates in one online-softmax state. For relation partitions
`r`, the merge invariant is:

```text
L = logsumexp_r(L_r)
O = sum_r exp(L_r - L) O_r
```

Independent softmax followed by addition or averaging is invalid.

## Policy compiler

The compiler should follow this sequence:

1. Measure dense-return candidates per Q-head. Model outputs remain exactly
   dense while sparse alternatives record block recall, omitted mass, LSE drift,
   projected output drift, and CountSketch error vectors.
2. Aggregate to GQA execution units. Split one KV group only when the additional
   bucket produces a robust risk improvement and costs less than 15% of the
   attention time it saves.
3. Build the cross-candidate projected error Gram after joint merge and output
   projection.
4. Search 25%, 37.5%, and 50% active-tile budgets using complete trajectory
   risk, hard-anchor constraints, and a measured hardware cost model.
5. Validate shortlisted policies through complete 20-step rollouts; stress the
   top policies at 40 steps.
6. Store a calibration residual UCB and dense-fallback conditions in the frozen
   policy.

Layer depth, task, modality, Q/K fingerprints, centroid margin, prior LSE, mask
stability, and normalized sigma are features. None of them may independently
declare a head safe.

## 0.90 and 0.80 operating modes

### Standard development mode

- current base: 50% GQA groups sparse;
- q12 sampled-mass descriptor with full-moment repair;
- hard anchors always dense;
- dense fallback on high entropy, high LSE drift, low centroid margin, or
  out-of-range sequence/modality;
- current measured 16K attention speedup: 1.21×;
- current three-seed minimum paired SSIM: 0.923.

### Aggressive mode

- current candidate: 75% GQA groups sparse;
- same hard anchors and fallback rules;
- compile selected layer-stage cells into the all-sparse bucket;
- do not make early denoising steps globally all-sparse;
- current measured 16K attention speedup: 1.41×;
- current two-seed minimum paired SSIM: 0.856.

The first selective-all-sparse schedule worth testing is a single kill test:

```text
steps 0–14: 50% or 75% GQA sparse
steps 15–19: all-sparse only for compiler-approved cells
```

The existing “last 10 steps all-sparse” screen reaches only SSIM 0.754 and is
too aggressive. The last five steps are motivated by local safe-share growth,
but remain unvalidated.

## Engineering decisions

### Keep

- GQA-aligned head buckets;
- directed relation descriptors;
- dense action/audio/control/special anchors;
- virtual semantic/local block lists;
- cached AR index and K/V;
- stage-boundary semantic refresh with drift-triggered early refresh;
- one fused online-softmax state and LSE output;
- FA3 fallback selected by measured lookup table;
- 64/128 tile support and packed sequences.

### Rework

- semantic selection must be query/head/timestep conditioned; global K-only
  clustering is insufficient;
- hand depth/cost relocation must be replaced with learned Gram/UCB trajectory
  weights;
- relation skip and semantic blocks must be composed in the same descriptor
  before reporting joint-QK savings;
- full sparse should be an execution bucket, not a global policy.

### Drop or demote

- cross→self hot-block transfer as a primary algorithm;
- blanket DM←AR layer skip as a stand-alone contribution;
- naive per-step semantic reorder;
- direct previous-step attention-output cache;
- independent-softmax merge;
- additive accounting of overlapping sparse patterns;
- context-parallel dynamic sparse speedup claims before implementation.

## Sparse-aware adaptation gate

Training-free routing should be exhausted first. Sparse-aware Q/K/O LoRA is
only justified when safe coverage remains below the target after trajectory
calibration. The intended adaptation freezes Reasoner, VAE, encoders, V
projection, and FFN; trains Generator Q/K/O LoRA plus per-head temperature and
router gates; and keeps every hard anchor.

If adaptation still cannot jointly reach:

- task-level non-inferiority;
- at least 30% Generator joint-QK reduction; and
- at least 22% safe executable coverage,

the project should stop expanding kernel/data investment and publish the
characterization plus negative results without a production acceleration claim.

## Minimal next experiment set

1. Add seed 17 to the q12 75% policy.
2. Run one selective last-five-step all-sparse T2V schedule.
3. Apply the current 50% and 75% policies to one I2V case and one DROID case.
4. Fuse semantic and directed-relation descriptors in one kernel dispatch and
   report routing, descriptor packing, refresh, kernel, and merge separately.
5. Measure complete Generator and end-to-end latency at DM≈14,720 before
   expanding to the full final-test campaign.

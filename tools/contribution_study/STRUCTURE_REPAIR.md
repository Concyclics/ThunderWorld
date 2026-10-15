# Same-budget structure-aware attention repair

This is a direct intervention on the complete QK–softmax–PV output. Attention
imbalance proposes a selector; only measured, paired output error establishes
whether that selector improves repair. No generated-video quality claim is made.

`structure_repair_study.py prepare` freezes three cohorts: Edge I2V, Edge policy,
and Nano audiovisual. Each uses two established scenes and four fresh evaluation
seeds (241101–241104). A descriptive prior is fitted only from the development
scene's existing BF16 observations at seeds 6101–6102. Validation-scene statistics,
FP4 errors and BF16 reference outputs never enter the selector.

The prior preserves query←key direction, query head, layer and four denoising
phases. Its per-key-token density is combined with actual token-role counts inside
each complete 128-token tile. Mixed-role tiles remain indivisible. DM queries
share one softmax across their AR prefix and DM keys; AR←DM is not invented.

Controls are: structure, structure without head conditioning, structure without
step conditioning, eight independent random masks, and a bidirectional sliding
window centered at the query's **global** token index, accounting for the AR
prefix. All controls retain every base block and use the same three-term repair
on selected QK and PV tiles. The same H×K tile count is spent in every M128 row.
This stronger constraint matches actual issued M32/N128 MMA even when the last
query tile is incomplete. Costs are checked using explicit masks, including GQA
support unions. Coverage 25/50/75% corresponds approximately to attention MMA
ratios 1.5/2/2.5, not whole-model request budgets.

All masks are compiled outside the measured operator. The study does not measure
selector runtime; it neither charges the original online energy selector to one
control nor gives another extra teacher GEMMs. The sparse Q/K/V packing path is
shared; GQA support union density and packing bounds are reported separately.

Example (paths point to local, already available model/observation/input assets):

```bash
python tools/contribution_study/structure_repair_study.py prepare \
  --root work/structure_repair --observations work/bf16_observations \
  --models work/models --cases work/case_manifests --gpu 3
python tools/contribution_study/structure_repair_study.py run --root work/structure_repair
python tools/contribution_study/structure_repair_study.py report --root work/structure_repair
```

The observation folder has `edge/` and `nano/`, containing the established
`bf16_CASE_seed6101_r0_heads.npz` and seed6102 files. Case manifests contain
`edge_cases.json` and `nano_cases.json`; the model folder contains pinned Edge
and Nano checkpoints. The runner reuses the existing `finding_resample` entry and
native FP4 package. It does not change the BF16 trajectory or production defaults.

Five evenly spaced layers, four steps and conditional branch 0 are measured at
full M with all keys and query heads. Random repeats are first averaged within a
panel; panels are then averaged within a request. Bootstrap pairs seeds within
each fixed scene. Heads, layers, steps and random masks are **not** independent
statistical samples. The report retains 95% paired intervals plus Bonferroni
intervals for the six primary comparisons per cohort. With only two established
scenes, results are an exploratory mechanism check, not broad scene generalization.

Negative outcomes and losses against either baseline remain in the report.
Shared-GPU timings are excluded from performance claims. Source hashes, the
frozen protocol, full raw per-panel results, per-role errors, and unchanged BF16
generation artifacts are retained for inspection.

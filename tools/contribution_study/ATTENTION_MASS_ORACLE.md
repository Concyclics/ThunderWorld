# Current-input attention-mass oracle

This extends `STRUCTURE_REPAIR.md` without changing its frozen prior, seeds,
native FP4 kernel, or old results. It adds a privileged selector that reads the
current BF16 Q/K. It does **not** observe V, reference outputs, or repair errors.

For every head, compute `P = softmax(Q K^T / sqrt(d))` in FP32 with TF32 off.
Every real query and key participates; query chunking does not change the
softmax domain. Score a 128-by-128 tile by the mean, over its real query rows,
of probability mass summed over its real key columns. GQA maps consecutive
query heads to the corresponding shared KV head. Padding contributes no mass.

Within each M128 row, select exactly `floor(H * ceil(N/128) * coverage)` tiles
across heads and key blocks. This maximizes captured attention mass within the
same per-row quota used by the frozen structure, random, and window policies.
Coverage is 25%, 50%, or 75%. Both QK and PV use that mask with three-term
repair, so attention-only issued FP4 MMA is 1.5x, 2.0x, or 2.5x where the quota
divides exactly. All dense base blocks remain. This is not a whole-model budget.

The oracle is exact for this **attention-mass objective**, not for minimum
attention-output NMSE. High-mass blocks can have little quantization error;
softmax coupling and V also affect the output. Structure may therefore beat
the mass oracle in output error. Such a result must not be described as
beating an error lower bound or attaining globally optimal repair allocation.

The privileged FP32 QK, full softmax, reduction, transfer, and sorting are
additional analysis work, separately recorded in `oracle_selector`. The
matched budget covers FP4 repair execution. Do not present the oracle as a
free deployable selector or use its shared-GPU wall time as a speed baseline.

## Replay and statistical protocol

The extension uses the two completed Edge cohorts: I2V and policy, each with
two fixed scenes and seeds 241101–241104, 35 denoising steps, conditional CFG
branch 0, layers 0/7/14/20/27, and steps 0/7/18/28. That is 16 trajectories and
320 attention panels. Use the existing complete manifests and model revision;
the task-specific generation settings remain unchanged.

Base, full three-term, and structure errors are remeasured. Their masks, shapes,
issued MMA, global NMSE, and per-role NMSE must exactly match the frozen study.
The entire generated video/trajectory/action hashes must also match before
reusing old random/window results. The observer asserts that it never modifies
Q/K/V or the returned BF16 attention output. A failure stops the comparison.

Panel NMSE is `sum((FP4_output - BF16_output)^2) / sum(BF16_output^2)` before the
attention output projection. Average panels within each request, then give
equal weight to requests/scenes. Average random replicates within each panel;
they are not additional experimental samples. Bootstrap paired seeds within
the two fixed scenes with 10,000 replicates. Report ordinary 95% intervals and
Bonferroni familywise intervals for the three budgets within each cohort.
These intervals do not establish generalization to new scenes.

Report absolute NMSE, structure/oracle relative NMSE gap, covered attention
mass, support overlap, per-role and per-scene errors, and GQA KV support union.
The optional fraction of oracle error reduction recovered is
`(E_base - E_structure)/(E_base - E_oracle)`; do not clip values above 100% or
report it when the denominator is nonpositive. A small fraction of the
remaining gap can look like a large percentage of base-error recovery, so
always inspect the absolute and relative remaining errors as well.

## Reproduction

From the research checkout, with its Cosmos environment and dependencies:

```bash
python tools/contribution_study/test_attention_mass_oracle.py
python tools/contribution_study/attention_mass_oracle_study.py prepare \
  --root "$ORACLE_RUN" --reference "$FROZEN_STRUCTURE_RUN" --gpu 0
python tools/contribution_study/attention_mass_oracle_study.py run --root "$ORACLE_RUN"
python tools/contribution_study/attention_mass_oracle_study.py report --root "$ORACLE_RUN" \
  --report-budgets 1.5 2.0
```

`prepare` refuses an existing directory and freezes the reference/source
hashes. `run` is sequential, admits Edge only with at least 12 GiB free, uses
the shared-calibration device lock, and can resume an interrupted run. It
does not update performance leaderboards. The original result directory is
read-only to this experiment. No new native kernel is introduced.

The report defaults to the requested 1.5x and 2.0x comparison. This is a
presentation filter: it preserves the original observations and statistical
corrections. JSON/CSV, English/Chinese reports, and the two LaTeX tables use
the same selected rows. Relative changes for all three comparators are
`100 * (Ours / comparator - 1)`; negative values mean lower output NMSE.

# A/W repair attribution and merged-weight experiments

Research-only code; production defaults and archived experiments are unchanged.

- `analyze_terms.py`: separates complete-trajectory A/W effects from local
  squared-error attribution by FFN projection.
- `merged_weight.py`: full-W repair in one native FP4 GEMM kernel. `plane`
  concatenates weight planes along K and preserves the original accumulation
  order; `pair` interleaves W0/W1 in expanded reduction tiles and aliases A0
  loads. No activation duplication buffer, offline requantization, cross term,
  or external output reduction is used. Arithmetic remains 2K plus A repair.
- `bench_merge.py`: 10 real model shapes × 3 tiers; launch search, numerical
  checks, then five randomized CUDA-Graph timing rounds. Captured 64-row inputs
  are repeated to actual M for latency only; these are not extra quality samples.
- `fit_attention_prior.py`: development-only head/layer/phase directed density
  prior, with CFG branches averaged. It is a descriptive feature prior, not
  calibrated repair benefit or a task-specific allocation model.
- `structured_attention.py`: compile real query/key role layouts (including
  mixed-role/tail tiles) to an exact global padded-tile quota. False entries
  skip residual products only; all base attention connections remain.
- `run_trajectories.py`: process-local integration into the established Generator
  runner. `ffn_split_aw` assigns W-only repair to down and A-only to up/gate;
  `ffn_split_aw_merged` adds locally promising shape-specific merged layouts.
  `attn_struct_a50` uses the frozen feature prior; `attn_energy_a50` uses current
  mass × V-energy with the same total half-tile budget. Both use terms=2 in QK
  and PV, retaining fresh values and joint streaming softmax.
- `validate.py`: independent decoded GEMM reference, tails, graph replay after
  changed inputs, exact mask quotas, and full/empty attention-mask boundaries.

Use the existing full-model environment for trajectory runs:

```bash
THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python" bash tools/kernel_handoff/python5090.sh tools/repair_design/run_trajectories.py --model edge --output runs5090/NEW_RUN --cases robot_table policy_table --seeds 8501 --steps 35 --modes bf16 ffn_act2 ffn_wgt2 ffn_split_aw ffn_split_aw_merged attn_act2 attn_energy_a50 attn_struct_a50
```

The wrapper requires the completed merge benchmark and fitted prior in
`results/repair_design_20260911`. Preparation happens during warmup when possible;
full-request times remain diagnostic eager measurements, not service throughput.
Merged weights coexist with the fallback layout in this screen, so its peak
allocation must not be presented as deployment memory. Waveform decoding stays
separate from Generator timing. Model output consistency is not task success.

Results and interpretation: [English study](../../docs/REPAIR_PATH_STUDY_20260911.md).
`summarize.py` verifies output hashes, checks duplicate
anchors, and produces comparison CSVs and five figures (including the attribution
figure from `analyze_terms.py`). `render_note.py` builds standalone HTML/PDF.

The two-step dispatch audit found policy M=416, absent from the M=316 tuning
cohort. Edge and Nano policy/AV equality checks mostly exercise fallback; only
Nano I2V selects the merged pair in the new 35-step screen. Do not interpret a
selected-launch table as proof of actual execution. Current wrapper receipts
record real shape hits, and source snapshots are stored beside each new run.

For another device or calibration, the wrapper accepts `--merge-benchmark
NEW_BENCHMARK_JSON` and `--attention-prior NEW_PRIOR_NPZ`. Defaults reproduce the
archived experiments; keep new results separate rather than editing history.

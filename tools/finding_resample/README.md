# Finding resampling and temporal-reuse ablations

Research-only tools for two findings: layer-dependent DM←AR mass and directed
cross-modal attention. Production defaults are unchanged. Temporal reuse is an
explicit approximation, not an equivalent kernel optimization.

## Sampling

`run.py --sample` observes the BF16 trajectory without replacing attention.
It samples up to 32 **fixed random query rows per present role**, independently
of the hidden values, at every layer, query head and denoising step. All key
positions are included in each sampled row's softmax. Small roles such as a
16-token action sequence are observed in full. Both CFG branches are recorded
separately; plots currently show the conditional branch. Seeds 6101/6102 are
fresh relative to the prior 5090 pilot.

Roles are read from the actual pipeline sequence metadata: AR context, clean
vision, noisy vision, sound, action, and other generation markers. A robot
picture in I2V does not count as action-token sampling. `policy_*` explicitly
runs the action-generation pathway with the documented Bridge interface.
The observation images do not provide matched Bridge action ground truth;
action consistency is not an absolute accuracy or task-success measurement.
Edge has no sound capability; Nano additionally runs two AV prompts.

The two image sources are the official `robot_153.jpg` and the first frame of
the official `robot_example.zip` input video. They are kept in `data_local/`;
download receipts and image hashes identify them. T2V/AV use waterfall and
pottery prompts. Policy/I2V pairs sharing an image, and T2V/AV pairs sharing a
scene description, must not be treated as independent scene populations.

The per-head features include role mass, entropy, maximum probability, output
norm, adjacent-step Q/output cosine, relative output L2, dominant-role changes,
role-distribution Jensen–Shannon divergence, and support overlap of the top 50%
of key tiles ranked by mass (a tile-count quota, not a 50% mass-coverage target).
This descriptive mass-tile support is **not** the native mass×V-energy repair
selector. Native selector reuse is tested separately in complete generation.

Directional comparisons report both row mass and token-count-adjusted density.
For roles i and j, `density_ratio = (mass_i_to_j / n_j) / (mass_j_to_i / n_i)`.
It equals the total paired attention-mass ratio because the directions contain
the same number of score entries. Raw row-mass ratios alone confound token counts.

```bash
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
export MPLCONFIGDIR=/tmp/thunderworld-mpl
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --sample --seeds 6101 6102 --steps 35 \
  --output runs5090/new_edge_sampling
bash tools/kernel_handoff/python5090.sh tools/finding_resample/analyze.py \
  runs5090/new_edge_sampling --output results/new_findings/edge
```

Use `--model nano` for the other checkpoint. Nano sound requires the pinned
`sound_tokenizer/` component. To avoid WDDM memory spill, the sampling/ablation
process saves the generated sound latent and defers waveform decoding to
`decode_audio.py` after releasing the Transformer. AV Generator timing includes
video decoding but excludes waveform decoding. The separate decoder time is
reported independently, not presented as measured integrated-request latency.
Observation overhead makes sampling wall time unsuitable for performance claims.

## Quality ablations

All ablation runs use the same **eager** framework. This is a diagnostic quality
experiment with end-to-end timings, not a replacement for the earlier shared
CUDA Graph benchmark. Loading, offline weight packing, JIT warmup, NPZ and MP4
serialization are outside measured requests. Per-request prediction sketches
and current-input bookkeeping are included for every mode.

| Mode | Behavior |
|---|---|
| `bf16` | BF16 reference |
| `ffn_base` | Generator FFNs use current model's RTN W4A4, no repair |
| `ffn_act2` | FFN A0W0 + A1W0 |
| `ffn_wgt2` | FFN A0W0 + A0W1 |
| `ffn_dual` | FFN both first-order corrections |
| `ffn_four` | FFN adds A1W1 |
| `attn_base` | FP4 attention without correction |
| `attn_act2` | Full A-side correction in both QK and PV |
| `attn_dual25/50/75/100` | Both first-order corrections at the stated tile quota |
| `attn_four100` | Full four-term QK/PV |
| `mask_reuse2` | Every second step reuses the previous native 50% repair mask; all numerical products are current |
| `delta_reuse2` | Current FP4 base + previous repaired-minus-base output on alternate steps |
| `output_reuse2` | Reuse the whole previous BF16 attention output on alternate steps |
| `*_middle` | Reuse only within steps `[ceil(.2S), floor(.8S))`; boundary steps are freshly computed |
| `output_reuse2_stable` | Reuse selected complete GQA groups in that middle window; recompute other groups |

All attention ablations share `ffn_act2`; all FFN ablations retain BF16
attention. Thus changes can be attributed to their respective modules. Every
variant uses its own intermediate states throughout all 35 steps. The current
quantizer remains absmax RTN, not the future repair-aware bulk-fit algorithm.

Delta reuse is defined at the **normalized attention output** as
`D_t = O_repaired_t - O_base_t`. It includes QK/softmax effects. It is not the
independent reuse of a PV tile with a mismatched normalization denominator.
Refresh steps currently evaluate both base and repaired paths, and that extra
work is charged to latency. No cross-request, layer, or CFG cache sharing is
allowed. Reuse saves the targeted attention work; QKV/output projections and
the rest of the layer are still computed.

The frozen stable-group policy is selected only from `development` cases:
rank middle-step output drift conservatively across query roles and heads in
each GQA group, then retain the most stable 25% of groups globally. Evaluation
cases and new seeds do not choose the support. The window is a hypothesis
motivated by the development traces, not an established quality guarantee.
Evaluation scenes also appear in the descriptive atlas under other seeds, so
this is exploratory validation rather than a blinded, wholly unseen-scene test.

```bash
bash tools/kernel_handoff/python5090.sh -m pytest tools/finding_resample/test_runtime.py -q
bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py \
  --model edge --cases robot_table policy_table --seeds 6103 6104 --steps 35 \
  --modes bf16 ffn_base ffn_act2 ffn_wgt2 ffn_dual ffn_four \
    attn_base attn_act2 attn_dual50 attn_dual100 attn_four100 \
    mask_reuse2 delta_reuse2 delta_reuse2_middle \
    output_reuse2 output_reuse2_middle output_reuse2_stable \
  --policy results/new_findings/edge/stable_group_policy.json \
  --output runs5090/new_edge_ablation
bash tools/kernel_handoff/python5090.sh tools/finding_resample/summarize_ablation.py \
  runs5090/new_edge_ablation --output results/new_findings/ablation
```

Use a fresh output directory for each run; partial/failed attempts are never
included in summaries. Snapshot source files are stored by SHA256. Keep the
full result JSON, NPZ head statistics, per-step output sketches, videos and
paired action/audio arrays. Report individual trajectories and observed ranges;
two seeds do not estimate p95 or population-level non-inferiority.

After the model campaign exits, run `decode_audio.py RUN_DIRECTORY` on Nano
AV results, and `perceptual.py RUN_DIRECTORY ... --output RESULTS_DIRECTORY`
for all-frame LPIPS AlexNet v0.1 and fixed-frame comparison panels. These use
the GPU sequentially. LPIPS takes native-resolution RGB mapped to [-1,1];
lower distance means closer to the paired reference, not higher task accuracy.
Then rerun `summarize_ablation.py` to include decoded-waveform and LPIPS metrics.

`trajectory_error.py` plots error propagation in fixed 256-coordinate prediction
sketches, explicitly separate from full-tensor error. `validate.py` checks the
complete request matrix, finite outputs, hashes, source snapshots and cache age.
`archive.py` copies reviewable records and source snapshots into the results
directory and inventories the larger local NPZ/video/audio files by SHA256.

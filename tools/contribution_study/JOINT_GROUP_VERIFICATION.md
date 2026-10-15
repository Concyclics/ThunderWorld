# Fixed-budget representation and temporal verification

This experiment keeps the frozen `b2_joint` runtime and W1 bank. It does not
promote a new default or consume validation/final cohorts. Runtime kernels are
unchanged; the new search and interventions run offline without gradients.

`joint_group_verify.py` compares the exact source asset with scale-only,
removal-only, and joint proposals. It preserves prescale, activation selection,
W1 support, outer scale and all W0 bytes outside the existing W1 bank. Each K16
group independently considers no removal, its largest magnitude, and the two
largest actual base quantization errors. Scale candidates are `{.75,.875,1,1.5}`
for W0 and `{.875,1,1.125}` for Wr; 1.5 includes a four-over-six scale control.
This bounded search is not an exhaustive sixteen-position oracle.

For three-term inference, let `u=A0+Ar`, `eta=X-u`, and `dW=W-W0-Wr`. The error
is `eta W^T + u dW^T + Ar Wr^T`. The proposal objective uses full within-K16
covariance of `[eta,u,Ar]`, including the cost of omitting the cross term.
Activation codes and the partial repair mask come from the existing native
current-input packer with the complete input M/amax. Unselected residual bytes
are never decoded. Proposal scores omit cross-group correlations and final
BF16 output rounding, so whole proposed operators must pass native evaluation
on all sixteen panels. The exact original always competes; joint also competes
against accepted scale-only encoding. Full video generation is a separate gate.

Outlier blocks become mandatory in every phase. Stored payloads stay sparse;
the new report checks their support, tensor bytes, untouched tensors, and the
complete request's work ledger. The `weight_scale` tuple is source metadata;
the saved per-group codes/scales are authoritative for these experimental
assets. Do not regenerate them solely from that tuple. Re-run this search or
transfer the exact assets with hashes. Existing promoted recipes are unchanged.

`schedule_calibrate.py` retains its original default sampling. New optional
flags select a bounded pilot and sample first/middle/last points within each
phase. `--project-attention` evaluates all attention proposals through the same
original BF16 Wo and records head-isolated perturbations. These are offline
diagnostics, not extra runtime computation. Their head gains are not additive,
and the projected metric is not a full Transformer, CFG, or rollout objective.
`temporal_verify_report.py` measures response drift and head rank stability;
it does not fit or promote a schedule.

Commands for the existing local calibration workspace (change paths as needed):

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export CUDA_VISIBLE_DEVICES=0
# Install requirements-quality.txt, or expose the already pinned metric packages:
export PYTHONPATH=/tmp/thunderworld-joint-deps${PYTHONPATH:+:$PYTHONPATH}
TW_WORKSPACE=/home/yufeng/thunderworld
TW_STUDY="$TW_WORKSPACE/fp4_joint_verify_20260919"
# For a fresh study directory, copy the archived protocol before starting.
# mkdir -p "$TW_STUDY"
# cp results/fp4_joint_verify_20260919/protocol.json "$TW_STUDY/protocol.json"

bash tools/fp4_handoff/python.sh tools/contribution_study/joint_group_verify.py \
  --methods "$TW_WORKSPACE/fp4_baseline_sweep_20260918/manifest.json" \
  --capture "$TW_WORKSPACE/activation_importance_20260917/capture" \
  --cohort "$TW_WORKSPACE/repair_redesign_20260917/cohort.json" \
  --output "$TW_STUDY/all_layers"

bash tools/fp4_handoff/python.sh tools/contribution_study/joint_group_workflow.py \
  --study "$TW_STUDY" --previous "$TW_WORKSPACE/fp4_baseline_sweep_20260918"

bash tools/fp4_handoff/python.sh tools/contribution_study/schedule_calibrate.py \
  --study "$TW_STUDY/temporal" --previous "$TW_WORKSPACE/repair_redesign_20260917" \
  --importance "$TW_WORKSPACE/activation_importance_20260917" --budgets 2 \
  --case-indices 0 6 7 15 --layers 0 7 14 21 27 --steps-per-phase 3 --project-attention

bash tools/fp4_handoff/python.sh tools/contribution_study/joint_group_report.py \
  --study "$TW_STUDY" --output results/fp4_joint_verify_20260919
bash tools/fp4_handoff/python.sh tools/contribution_study/temporal_verify_report.py \
  --calibration "$TW_STUDY/temporal/calibration/b2" \
  --output results/fp4_joint_verify_20260919/temporal
```

The study requires a frozen `protocol.json` (archived with the results), original
calibration panels and source assets; large weights/videos are not embedded in
Git. Resume checks reject changed sources. New source versions need new output
directories. Only one GPU worker runs at a time; CPU metrics may overlap. The
optional `--after-generation PATH/TO/suite.json` waits for serial generation to
finish before acquiring the shared calibration lock. All timings remain
ineligible for performance claims.

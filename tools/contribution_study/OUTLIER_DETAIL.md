# Fixed-support outlier encoding experiment

This is a local mechanism test, not a new generation-quality claim. It tests the
incremental benefit of reserving one K16 element for W1 **over scale search**.
It holds the existing 2x W1 support and activation policy fixed. The current bank
has 63 paid attention-projection operators; FFN and attention QK/PV are outside
this experiment. Four SHA-selected paid N128/K128 blocks per operator are changed.

Controls: original RTN + actual residual; scale-only; forced max removal at fixed
scales; optional max removal + scales; optional any-element removal at fixed
scales; optional any-element removal + scales. All 16 positions compete with no
removal. All coding uses the existing actual E4M3/E2M1 codec and BF16 compensation.
Search/gating see eight panels; eight other panels review the frozen choice.
Both sets are reused development scenes and the original assets used them already.

Run from the repository root with the project's CUDA Python environment:

```bash
export PYTHONPATH=packages/fp4-repair/src:tools/contribution_study
python -m unittest tools/contribution_study/test_outlier_detail.py -v
python tools/contribution_study/outlier_detail_study.py prepare \
  --methods "$METHODS_MANIFEST" --capture "$CAPTURE_DIR" \
  --cohort "$COHORT_FILE" --output "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=3 TW_SHARED_CALIBRATION_GPUS=3 \
TW_SHARED_CALIBRATION_LOCK_SCOPE=device \
python -u tools/contribution_study/outlier_detail_study.py run --output "$OUTPUT_DIR"
python tools/contribution_study/outlier_detail_report.py --root "$OUTPUT_DIR"
```

Optional encoding/historical-control audit (the previous file must be the
original fixed-bank `joint_group_verify.py` progress, not unrelated assets):

```bash
python tools/contribution_study/outlier_detail_audit.py \
  --root "$OUTPUT_DIR" --previous "$PREVIOUS_PROGRESS"
CUDA_VISIBLE_DEVICES=3 TW_SHARED_CALIBRATION_GPUS=3 \
TW_SHARED_CALIBRATION_LOCK_SCOPE=device \
python tools/contribution_study/outlier_detail_boundary_check.py --root "$OUTPUT_DIR"
python tools/contribution_study/outlier_detail_report.py --root "$OUTPUT_DIR"
```

The byte audit distinguishes unchanged retained codes from rare old-asset tie
rounding differences. The sensitivity check restores flagged complete groups to
the scale-only proposal, evaluates affected operators again, and keeps the same
fit-only acceptance rule. It preserves the primary experiment and uses no review
output to identify groups. Both primary and sensitivity results are reported.

Prepare freezes protocol, source hashes, input hashes and sampled blocks before
GPU execution. Run checkpoints each operator and verifies all completed artifact
hashes on resume. `--max-modules 1` can check one operator, then resume without the
limit. It requires 12 GiB free on the selected GPU and one admitted worker.
No timing results are eligible for speed comparisons.

Raw per-group arrays and per-panel native output errors remain available even
when a proposal fails the fit-only whole-operator gate. Report separates 3-term
selection from 4-term diagnostics, sampled weight reconstruction from full-M
operator output, and total effects from the conditional subset with removal.
The native gate can choose an unchanged or scale-only encoding; report counts
these decisions explicitly. Necessary W1 blocks must stay enabled in all stages.

This experiment introduces no new GPU kernel or inference fallback. The original
deployment policy/assets are not changed. Patches are offline diagnostics, with
byte-identical compact/reload checks; no hidden full W1 is added to deployment.

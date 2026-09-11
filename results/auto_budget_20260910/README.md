# Automatic FFN repair budgets on RTX 5090

Interpretation and limits: [experiment report](../../docs/AUTO_BUDGET_5090_20260910.md).
Commands and implementation: [tools](../../tools/auto_budget/README.md).

- `unrestricted/`: initial greedy policies, seeds 7103/7104, 80 complete trajectories.
- `guarded/`: half-A floor, new seeds 7203/7204, 50 complete trajectories.
- `shared/`: shared versus separate Nano gate/up packing, seed 7103, 9 complete trajectories. Exact array hashes are checked against the initial cohort.
- `kernels.csv`, `kernel_summary.json`, `tensor_core_activity.png`: 60 shape/tier measurements, 360 launch-equivalence checks, eight Nsight profiles, and 12 alternating-order packing comparisons.
- `validation.json`, `postprocess_validation.json`: coverage, raw numeric/hash, source-snapshot, LPIPS and waveform checks.
- `records/`: calibration policies/probes metadata, complete experiment JSON, operator timing samples, source run metadata, raw Nsight CSV/reports.
- `validation_logs/`: full experiment, sanitizer and analysis logs.
- `source_snapshots/`: content-addressed Python sources; E2E `result.json` maps the sources actually run. `final_analysis_sources.json` separately maps final analysis tools.
- `local_raw_inventory.json`: locations, sizes and SHA-256 of large local captured tensors/media and original records; tensors/media are not duplicated here.
- `SHA256SUMS`: hashes of archived files.

Video quality is paired with the same model/seed BF16 output. Phase policies use
local 64-row BF16 calibration probes from seed 7101, separate calibration scenes.
Scenes used for evaluation have appeared in preceding research; the process is
not completely blinded. `policy_validated.json` selects/falls back using guarded
validation results and needs a separate final test. Its acceptance means
provisional non-regression against a uniform reference, not absolute quality.

Full Generator timing uses the common eager framework; operator timings use
CUDA Graph replay. Audio waveform decoding is separate. Nsight Tensor pipe
elapsed-cycle activity is not the same metric as percentage of FP4 FLOP peak.
Lower repair compute does not remove the resident W1 plane.

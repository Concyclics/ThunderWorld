# Contributing

Use the root installation and public workflows documented in the README. Open an issue with the model/profile, pinned revision, GPU name/capability, package versions and a minimal input or synthetic shape that reproduces a failure. Do not attach credentials or licensed checkpoints.

For a change, explain the trigger, resulting behavior and validation. Keep unrelated research artifacts and large generated data outside Git. Preserve historical measurements and frozen assets; create new manifests instead of overwriting them.

- Portable CLI / metadata changes: run `tests/public` and CPU budget/protocol tests.
- Runtime changes: test current-input updates, stage/CFG transitions, GQA, tail tiles, masks, zero/full repair and output lifetime in eager and Graph modes.
- Kernel changes: independent decoded references plus Compute Sanitizer memcheck/racecheck on supported hardware.
- Quantization or accumulation changes: complete trajectory quality validation, not just local GEMM error.
- Performance claims: debug off, current-device tuning, exclusive GPU, matched request scope, repeated interleaved trials and raw measurements.

Contributions to project code are under Apache-2.0. Respect the separate licenses listed in THIRD_PARTY_NOTICES.md.

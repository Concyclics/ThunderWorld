# Handoff validation — 2026-09-09

Validated on the source RTX PRO 6000 Blackwell (SM120), physical GPU 0. **No RTX 5090 execution has occurred here.** The receiver must record its own baseline.

- Data-only bundle: 10 files, 68,091,122 bytes plus manifest. Archive: 49,380,641 bytes. No model weight tensors are transferred.
- Receiver offline preparation recreated all three FFN modules from local copies of the pinned HF shard. BF16 weights, packed codes, scales and global-scale tensor hashes all matched the export expectations.
- Seven quick-matrix cases passed independent decoded-FP4 correctness checks: three FFN modules, two attention input distributions, uniform and nonuniform head quotas.
- Two extra cases passed from the clean checkout: FFN M129/K256/M64 GEMM-only and attention M129/M16/splits4/full single-sided ACT2/profile mode.
- Sixteen CPU package configuration tests passed. The unchanged repository evidence audit matched its checked-in receipt. Main-matrix dry run generated 60 commands.
- Preflight and NVTX profile entrypoints ran successfully. This does not claim a full Nsight profiling session or that the target GPU achieves any particular throughput.

Numerical results and code hashes are in [validation.json](validation.json). Short smoke timings are intentionally not presented as 5090 performance estimates. The receiver should use at least five randomized rounds for actual comparisons.

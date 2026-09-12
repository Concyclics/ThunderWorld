# edge i2v compute-constrained FP4 final results

All runs used shared PRO 6000 GPUs, 35 steps, identical inputs and Graph execution with debug off. No latency speedup is claimed.

| Budget | Selected | LPIPS ↓ | Control LPIPS ↓ | SSIM ↑ | Relative L2 ↓ | Same-budget gate | FP8 gate | MMA/C0 (selected/control) | W1 MiB |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| 1 | control_b1_a | 0.406711 | 0.406711 | 0.582606 | 0.373316 | False | False | 1.000000/1.000000 | 0.00 |
| 1.125 | control_joint_b1.125_balanced | 0.365284 | 0.365284 | 0.607019 | 0.351237 | False | False | 1.124698/1.124698 | 48.73 |
| 1.25 | b1.25_grow_mixed | 0.274868 | 0.344413 | 0.692435 | 0.302037 | True | False | 1.249996/1.206295 | 86.06 |
| 1.5 | b1.5_refined | 0.264763 | 0.330790 | 0.707249 | 0.296064 | True | False | 1.499999/1.456295 | 185.59 |

Paired 95% bootstrap intervals, worst-seed degradation, complete cost and memory breakdowns are in report.json.

Observed nondominated configurations: control_b1_a, control_joint_b1.125_balanced, b1.25_grow_mixed, control_joint_b1.25_balanced, b1.5_refined, fp4

Shared GPU timings cannot establish acceleration or latency Pareto optimality.
Packing traffic is an implementation-derived bound; fields ending in upper are not measured savings.
Residual address space is still allocated in full; sparse writes do not imply proportional scratch reduction.
Prepared weight bytes cover replaced projections; resident model state and request peaks are reported separately.
New seeds in existing scenes do not establish cross-scene generalization.

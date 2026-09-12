# edge policy compute-constrained FP4 final results

All runs used shared PRO 6000 GPUs, 35 steps, identical inputs and Graph execution with debug off. No latency speedup is claimed.

| Budget | Selected | LPIPS ↓ | Control LPIPS ↓ | SSIM ↑ | Relative L2 ↓ | Same-budget gate | FP8 gate | MMA/C0 (selected/control) | W1 MiB |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| 1 | control_b1_a | 0.078556 | 0.078556 | 0.892744 | 0.148625 | False | False | 1.000000/1.000000 | 0.00 |
| 1.125 | control_joint_b1.125_balanced | 0.053062 | 0.053062 | 0.921694 | 0.121622 | False | False | 1.123673/1.123673 | 48.73 |
| 1.25 | b1.25_refined | 0.046390 | 0.063643 | 0.934231 | 0.118962 | False | False | 1.247110/1.247970 | 83.23 |
| 1.5 | b1.25_refined | 0.046390 | 0.066048 | 0.934231 | 0.118962 | False | False | 1.247110/1.450890 | 83.23 |

| Budget | Selected action.relative_l2 ↓ | Control ↓ |
|---|---:|---:|
| 1 | 0.45129595 | 0.45129595 |
| 1.125 | 0.42312240 | 0.42312240 |
| 1.25 | 0.39224272 | 0.42134793 |
| 1.5 | 0.39224272 | 0.47114434 |

Paired 95% bootstrap intervals, worst-seed degradation, complete cost and memory breakdowns are in report.json.

Observed nondominated configurations: control_b1_a, control_joint_b1.125_balanced, b1.25_refined, fp4

Shared GPU timings cannot establish acceleration or latency Pareto optimality.
Packing traffic is an implementation-derived bound; fields ending in upper are not measured savings.
Residual address space is still allocated in full; sparse writes do not imply proportional scratch reduction.
Prepared weight bytes cover replaced projections; resident model state and request peaks are reported separately.
New seeds in existing scenes do not establish cross-scene generalization.

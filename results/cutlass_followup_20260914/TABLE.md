# RTX 5090 CUTLASS follow-up measurements

32 operators per captured graph, 9 alternating rounds. Full operator includes selection, packing, scale layout and GEMM. Ranges span 16 shapes/phases per row, not confidence intervals. Native is the fixed BM128/stages2 Triton baseline, not a new best-of-all-Triton search. Actual release tensors are I2V M960 and Policy M316; the older M416 panel description is inaccurate. M416 is covered by correctness tests, not this policy timing table.

| Panel | CUTLASS N | Native / original CUTLASS | Direct-pack speedup range | Median direct-pack speedup | >1.03× cases |
|---|---:|---:|---:|---:|---:|
| i2v_b125 | 128 | 0.963–1.624× | 0.986–1.076× | 1.029× | 8/16 |
| i2v_b125 | 64 | 1.019–1.438× | 0.949–1.103× | 1.015× | 6/16 |
| i2v_b150 | 128 | 0.869–1.593× | 0.989–1.084× | 1.027× | 7/16 |
| i2v_b150 | 64 | 0.748–1.333× | 0.982–1.165× | 1.021× | 5/16 |
| policy_b125 | 128 | 1.001–1.877× | 0.991–1.085× | 1.036× | 9/16 |
| policy_b125 | 64 | 0.973–2.496× | 0.978–1.143× | 1.056× | 10/16 |

All 96 direct-pack outputs match the same CUTLASS baseline bitwise. Both budgets and both development seeds match across complete 35-step trajectories. Pipeline timings are intentionally not promoted to a speed claim. First-attempt host-replay timing remains in records/direct_*.json.

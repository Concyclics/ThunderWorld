# Guarded complete-trajectory comparison

Seeds 7203/7204; two trajectories per row. Gains are versus the same-budget uniform reference; positive quality gains mean closer to BF16. Speedups are geometric means of paired Generator wall-time ratios. This is an exploratory eager measurement, not an independent test of the validation-selected policy.

| Model | Task | Budget | Auto SSIM mean | Worst SSIM gain | Mean LPIPS gain | Speedup / uniform | Speedup / BF16 | Quality gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| edge | i2v | 50% | 0.6286 | -0.0058 | +0.0292 | 1.068× | 0.966× | Retain candidate |
| edge | i2v | 75% | 0.6194 | -0.0856 | -0.0561 | 0.960× | 0.914× | Fallback: ffn_uniform75 |
| edge | policy | 50% | 0.9509 | -0.0145 | -0.0113 | 1.014× | 0.925× | Fallback: ffn_act2 |
| edge | policy | 75% | 0.9760 | -0.0402 | -0.0088 | 1.097× | 1.004× | Fallback: ffn_uniform75 |
| nano | i2v | 50% | 0.8400 | -0.0165 | +0.0035 | 1.022× | 1.224× | Retain candidate |
| nano | i2v | 75% | 0.9558 | -0.0000 | +0.2424 | 0.987× | 1.175× | Retain candidate |
| nano | policy | 50% | 0.9663 | -0.0000 | +0.0059 | 1.049× | 1.608× | Fallback: ffn_act2 |
| nano | policy | 75% | 0.9886 | -0.0046 | -0.0010 | 0.962× | 1.555× | Fallback: ffn_uniform75 |
| nano | av | 50% | 0.8362 | +0.0009 | +0.0414 | 0.977× | 1.243× | Fallback: ffn_act2 |
| nano | av | 75% | 0.8456 | -0.0378 | -0.0055 | 1.019× | 1.198× | Fallback: ffn_uniform75 |

Retained schedules satisfy only the provisional comparison gate. Their absolute BF16 distance may still be substantial. The 25% guarded schedule is exactly uniform half-A and was not rerun as an allocation improvement.

## Action and audio distances

| Model | Mode | Task | Action relative L2 | Audio latent relative L2 | Waveform relative L2 | Log-spectral L1 |
|---|---|---|---:|---:|---:|---:|
| edge | ffn_act2 | policy | 0.1097 | — | — | — |
| edge | ffn_auto50 | policy | 0.3418 | — | — | — |
| edge | ffn_uniform75 | policy | 0.0194 | — | — | — |
| edge | ffn_auto75 | policy | 0.0461 | — | — | — |
| nano | ffn_act2 | policy | 0.1679 | — | — | — |
| nano | ffn_auto50 | policy | 0.2007 | — | — | — |
| nano | ffn_uniform75 | policy | 0.0347 | — | — | — |
| nano | ffn_auto75 | policy | 0.0497 | — | — | — |
| nano | ffn_act2 | av | — | 0.0984 | 0.1172 | 0.4487 |
| nano | ffn_auto50 | av | — | 0.0710 | 0.0864 | 0.4414 |
| nano | ffn_uniform75 | av | — | 0.0819 | 0.0936 | 0.4483 |
| nano | ffn_auto75 | av | — | 0.0494 | 0.0576 | 0.3977 |

All distances are against paired BF16 outputs and are not ground-truth task-success measures. Waveform decoding occurs in a separate process.

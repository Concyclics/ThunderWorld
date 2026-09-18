# edge: smoke quality

2 scenes, 12 methods, 48 generated outputs. 35 steps, debug off.
Metrics measure consistency with this model’s same-seed BF16 reference. Shared-GPU timings are ineligible.
Means average seeds within scenes, then scenes within categories, then categories equally. Exact PSNR is +infinity (JSON null + exact marker).

Development/smoke execution qualification only; no final quality superiority claim.

## paibench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| bf16 reference | 0 | 1 | +∞ | 0 |
| fp8 | 0.023812 | 0.961502 | 34.1422 | 0.046112 |
| fp4_base | 0.163281 | 0.793595 | 25.8046 | 0.119210 |
| awq_reference | 0.022627 | 0.954447 | 34.3664 | 0.044482 |
| arcquant | 0.052081 | 0.914537 | 30.7067 | 0.067792 |
| arcquant100 | 0.047654 | 0.900738 | 29.8075 | 0.086430 |
| sharq_perf | 0.038936 | 0.927720 | 31.4643 | 0.062169 |
| b1.5_protected_n | 0.052808 | 0.913011 | 30.4810 | 0.069635 |
| b2_joint | 0.034014 | 0.945736 | 32.5672 | 0.054980 |
| b2.5_ffn_guarded | 0.015076 | 0.971481 | 35.9931 | 0.037124 |
| fp4_three | 0.014974 | 0.976810 | 36.5779 | 0.035236 |
| fp4_four | 0.013642 | 0.979162 | 37.0037 | 0.033920 |

## rbench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| bf16 reference | 0 | 1 | +∞ | 0 |
| fp8 | 0.141102 | 0.770273 | 18.0351 | 0.256028 |
| fp4_base | 0.321331 | 0.490649 | 13.1418 | 0.451731 |
| awq_reference | 0.211167 | 0.676846 | 14.8711 | 0.373357 |
| arcquant | 0.268828 | 0.598586 | 12.9244 | 0.462359 |
| arcquant100 | 0.255651 | 0.638401 | 13.1650 | 0.449835 |
| sharq_perf | 0.224831 | 0.653762 | 14.8205 | 0.370390 |
| b1.5_protected_n | 0.232836 | 0.637135 | 14.6746 | 0.381959 |
| b2_joint | 0.187920 | 0.694618 | 16.4348 | 0.307395 |
| b2.5_ffn_guarded | 0.164473 | 0.758071 | 17.4009 | 0.287197 |
| fp4_three | 0.133648 | 0.780626 | 18.2712 | 0.251232 |
| fp4_four | 0.069420 | 0.871854 | 21.7400 | 0.166932 |

All paired deltas, ordinary 95% intervals, Bonferroni intervals, worst seeds and quality guards are in quality.json.
Only the predeclared ARCQuant100 and FP8 comparison families can support corrected superiority claims; final results never feed calibration.

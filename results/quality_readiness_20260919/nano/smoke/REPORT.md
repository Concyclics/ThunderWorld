# nano: smoke quality

2 scenes, 12 methods, 48 generated outputs. 35 steps, debug off.
Metrics measure consistency with this model’s same-seed BF16 reference. Shared-GPU timings are ineligible.
Means average seeds within scenes, then scenes within categories, then categories equally. Exact PSNR is +infinity (JSON null + exact marker).

Development/smoke execution qualification only; no final quality superiority claim.

## paibench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| bf16 reference | 0 | 1 | +∞ | 0 |
| fp8 | 0.060836 | 0.873965 | 27.0778 | 0.108210 |
| fp4_base | 0.348481 | 0.566658 | 19.6963 | 0.242334 |
| awq_reference | 0.045261 | 0.905986 | 28.5729 | 0.088349 |
| arcquant | 0.104983 | 0.769618 | 23.9278 | 0.149624 |
| arcquant100 | 0.077757 | 0.833641 | 25.2773 | 0.127462 |
| sharq_perf | 0.097726 | 0.751600 | 23.7865 | 0.151752 |
| b1.5_protected_n | 0.109719 | 0.793731 | 24.4542 | 0.142953 |
| b2_joint | 0.086197 | 0.825679 | 25.2524 | 0.130705 |
| b2.5_ffn_guarded | 0.045279 | 0.906913 | 28.4560 | 0.089648 |
| fp4_three | 0.074830 | 0.824488 | 25.7395 | 0.123866 |
| fp4_four | 0.077469 | 0.812915 | 25.4228 | 0.128636 |

## rbench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| bf16 reference | 0 | 1 | +∞ | 0 |
| fp8 | 0.094598 | 0.817596 | 18.9499 | 0.227331 |
| fp4_base | 0.261004 | 0.589731 | 15.7040 | 0.325323 |
| awq_reference | 0.074440 | 0.853787 | 20.4303 | 0.194372 |
| arcquant | 0.139753 | 0.748986 | 16.3461 | 0.304235 |
| arcquant100 | 0.129366 | 0.768125 | 16.7794 | 0.289227 |
| sharq_perf | 0.120555 | 0.785852 | 18.2548 | 0.250059 |
| b1.5_protected_n | 0.117233 | 0.777786 | 19.1136 | 0.227970 |
| b2_joint | 0.119412 | 0.792784 | 18.0799 | 0.252136 |
| b2.5_ffn_guarded | 0.090972 | 0.822805 | 19.1884 | 0.224684 |
| fp4_three | 0.065551 | 0.861332 | 21.3334 | 0.176847 |
| fp4_four | 0.069199 | 0.856285 | 20.9758 | 0.184274 |

All paired deltas, ordinary 95% intervals, Bonferroni intervals, worst seeds and quality guards are in quality.json.
Only the predeclared ARCQuant100 and FP8 comparison families can support corrected superiority claims; final results never feed calibration.

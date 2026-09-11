# Full-trajectory repair and reuse ablation

All runs use the same eager execution and input within each comparison. Attention ablations share RTN W4 + full ACT2 FFNs. Sampling overhead is excluded by running a separate process. AV waveform decoding is timed separately and excluded from Generator wall time. Numbers are diagnostic eager latency, not the earlier CUDA Graph production benchmark.

| Model | Mode | Requests | Mean / min BF16 SSIM | Mean BF16 LPIPS ↓ | Max action rel L2 | Max audio rel L2 | BF16 / parent speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| edge | bf16 | 4 | 1.0000 / 1.0000 | 0.0000 | — | — | 1.000× / 1.000× |
| edge | ffn_base | 4 | 0.7477 / 0.5255 | 0.2239 | 0.3826 | — | 0.974× / 1.071× |
| edge | ffn_act2 | 4 | 0.7333 / 0.4810 | 0.2565 | 0.1028 | — | 0.909× / 1.000× |
| edge | ffn_wgt2 | 4 | 0.7629 / 0.5498 | 0.2085 | 0.2546 | — | 0.973× / 1.071× |
| edge | ffn_dual | 4 | 0.8338 / 0.6497 | 0.1263 | 0.0366 | — | 0.923× / 1.015× |
| edge | ffn_four | 4 | 0.8736 / 0.7423 | 0.0891 | 0.0344 | — | 1.009× / 1.110× |
| edge | attn_base | 4 | 0.7062 / 0.4570 | 0.2734 | 0.1533 | — | 0.870× / 0.957× |
| edge | attn_act2 | 4 | 0.7038 / 0.4838 | 0.2588 | 0.1475 | — | 0.883× / 0.972× |
| edge | attn_dual50 | 4 | 0.7192 / 0.4716 | 0.2638 | 0.0992 | — | 0.906× / 0.997× |
| edge | attn_dual100 | 4 | 0.7302 / 0.4797 | 0.2571 | 0.1043 | — | 0.892× / 0.981× |
| edge | attn_four100 | 4 | 0.7337 / 0.4793 | 0.2531 | 0.1053 | — | 0.985× / 1.083× |
| edge | mask_reuse2 | 4 | 0.7191 / 0.4698 | 0.2638 | 0.0994 | — | 0.898× / 0.991× |
| edge | delta_reuse2 | 4 | 0.7441 / 0.4648 | 0.2460 | 0.0482 | — | 0.838× / 0.925× |
| edge | delta_reuse2_middle | 4 | 0.7178 / 0.4727 | 0.2641 | 0.1040 | — | 0.860× / 0.949× |
| edge | output_reuse2 | 4 | 0.7278 / 0.4816 | 0.2578 | 0.1509 | — | 0.878× / 0.966× |
| edge | output_reuse2_middle | 4 | 0.7297 / 0.4808 | 0.2589 | 0.1446 | — | 0.868× / 0.955× |
| edge | output_reuse2_stable | 4 | 0.7321 / 0.4810 | 0.2572 | 0.1093 | — | 0.896× / 0.985× |
| nano | bf16 | 6 | 1.0000 / 1.0000 | 0.0000 | — | 0.0000 | 1.000× / 1.000× |
| nano | ffn_base | 6 | 0.8881 / 0.7610 | 0.0639 | 0.1754 | 1.1496 | 1.549× / 1.072× |
| nano | ffn_act2 | 6 | 0.9187 / 0.8062 | 0.0417 | 0.0739 | 0.5770 | 1.444× / 1.000× |
| nano | ffn_wgt2 | 6 | 0.9235 / 0.8706 | 0.0414 | 0.0824 | 0.3927 | 1.397× / 0.967× |
| nano | ffn_dual | 6 | 0.9665 / 0.9292 | 0.0161 | 0.0219 | 0.0489 | 1.262× / 0.874× |
| nano | ffn_four | 6 | 0.9644 / 0.9364 | 0.0169 | 0.1217 | 0.0766 | 1.190× / 0.824× |
| nano | attn_base | 6 | 0.6857 / 0.4941 | 0.2942 | 0.4875 | 3.9974 | 1.464× / 1.014× |
| nano | attn_act2 | 6 | 0.7131 / 0.5117 | 0.2606 | 0.2357 | 4.2400 | 1.426× / 0.987× |
| nano | attn_dual50 | 6 | 0.8790 / 0.7581 | 0.0805 | 0.2649 | 1.5837 | 1.404× / 0.972× |
| nano | attn_dual100 | 6 | 0.9268 / 0.8516 | 0.0392 | 0.0775 | 0.7083 | 1.418× / 0.982× |
| nano | attn_four100 | 6 | 0.9219 / 0.8199 | 0.0438 | 0.0768 | 0.7430 | 1.417× / 0.981× |
| nano | mask_reuse2 | 6 | 0.8662 / 0.6820 | 0.0904 | 0.2684 | 1.7242 | 1.334× / 0.950× |
| nano | delta_reuse2 | 6 | 0.8703 / 0.7302 | 0.0890 | 0.2726 | 2.0993 | 1.397× / 0.995× |
| nano | delta_reuse2_middle | 6 | 0.8757 / 0.7340 | 0.0784 | 0.3418 | 1.8320 | 1.309× / 0.932× |
| nano | output_reuse2 | 6 | 0.8100 / 0.4020 | 0.1631 | 0.1216 | 1.9225 | 1.458× / 1.009× |
| nano | output_reuse2_middle | 6 | 0.9202 / 0.8336 | 0.0412 | 0.0889 | 0.6210 | 1.422× / 0.984× |
| nano | output_reuse2_stable | 6 | 0.9196 / 0.8146 | 0.0412 | 0.0735 | 0.5641 | 1.446× / 1.001× |

Parent = current attn_dual50 for mask/delta reuse; ffn_act2 for output reuse and other modes where available. Per-case values and exact parent labels are in ablation.csv. Action/audio are paired numerical differences, not task success or perceptual quality scores.

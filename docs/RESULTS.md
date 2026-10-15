# Results and evidence boundaries

The tables below are generated from the checked-in source CSVs, not re-entered estimates. They report the completed three-model × two-benchmark evaluation. Super is an extension beyond the Edge/Nano main preprint experiments.

## Protocol

RBench: 650 scenes / 1,300 videos per method and model. PAI-Bench-G: 1,044 scenes / 2,088 videos. Settings: 320×192, 61 stored frames, 35 steps, CFG 6, flow shift 3, playback 24 fps; same inputs/seeds across methods. Full-corpus evaluation overlaps earlier calibration/development scenes. It is not held-out cross-scene generalization.

Pixel metrics compare original float video outputs to paired BF16 generations. Lower LPIPS / relative L2 and higher SSIM / PSNR mean greater consistency, not necessarily greater physical task success. All tables use category-macro aggregation: seeds → scene → category → dataset. Scene-micro numbers are retained in CSV. BF16 self-reference has LPIPS 0, SSIM 1, relative L2 0 and infinite PSNR; no arbitrary finite BF16 PSNR is plotted.

## Pixel consistency

### Edge / RBench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| W4A4 | 0.3001 | 0.5687 | 15.6096 | 0.3169 |
| ARCQuant 50% | 0.1935 | 0.6919 | 17.3226 | 0.2687 |
| ARCQuant 100% | 0.1776 | 0.7214 | 17.7465 | 0.2577 |
| SharQ | 0.1918 | 0.6997 | 17.3063 | 0.2704 |
| AWQ W4A16 reference | 0.1768 | 0.7247 | 17.8659 | 0.2559 |
| ThunderWorld 1.5× | 0.1928 | 0.6908 | 17.3690 | 0.2669 |
| ThunderWorld 2× | 0.1717 | 0.7248 | 18.0990 | 0.2479 |
| FP8 | 0.1188 | 0.7992 | 20.6086 | 0.1920 |

### Edge / PAI-Bench-G

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| W4A4 | 0.2824 | 0.5648 | 17.7578 | 0.3048 |
| ARCQuant 50% | 0.1829 | 0.6775 | 19.9388 | 0.2487 |
| ARCQuant 100% | 0.1670 | 0.7047 | 20.4627 | 0.2390 |
| SharQ | 0.1812 | 0.6794 | 19.8577 | 0.2517 |
| AWQ W4A16 reference | 0.1749 | 0.6964 | 20.1303 | 0.2469 |
| ThunderWorld 1.5× | 0.1834 | 0.6676 | 19.7945 | 0.2502 |
| ThunderWorld 2× | 0.1567 | 0.7157 | 20.8478 | 0.2279 |
| FP8 | 0.0990 | 0.8071 | 23.9571 | 0.1657 |

### Nano / RBench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| W4A4 | 0.3584 | 0.5253 | 16.0177 | 0.3108 |
| ARCQuant 50% | 0.1339 | 0.7592 | 19.8385 | 0.2116 |
| ARCQuant 100% | 0.1250 | 0.7765 | 20.2030 | 0.2042 |
| SharQ | 0.1301 | 0.7677 | 19.9489 | 0.2097 |
| AWQ W4A16 reference | 0.1201 | 0.7881 | 20.4695 | 0.2007 |
| ThunderWorld 1.5× | 0.1381 | 0.7520 | 19.8599 | 0.2105 |
| ThunderWorld 2× | 0.1181 | 0.7848 | 20.6700 | 0.1944 |
| FP8 | 0.0931 | 0.8241 | 22.1309 | 0.1661 |

### Nano / PAI-Bench-G

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| W4A4 | 0.3979 | 0.4849 | 16.7131 | 0.3464 |
| ARCQuant 50% | 0.1408 | 0.7287 | 21.7561 | 0.2096 |
| ARCQuant 100% | 0.1322 | 0.7446 | 22.1487 | 0.2036 |
| SharQ | 0.1346 | 0.7396 | 22.0339 | 0.2048 |
| AWQ W4A16 reference | 0.1214 | 0.7661 | 22.7717 | 0.1928 |
| ThunderWorld 1.5× | 0.1519 | 0.7093 | 21.4246 | 0.2149 |
| ThunderWorld 2× | 0.1219 | 0.7577 | 22.7115 | 0.1906 |
| FP8 | 0.0960 | 0.8042 | 24.2610 | 0.1620 |

### Super / RBench

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| W4A4 | 0.1620 | 0.7062 | 18.6602 | 0.2371 |
| ARCQuant 50% | 0.1043 | 0.8021 | 21.0460 | 0.1882 |
| ARCQuant 100% | 0.0959 | 0.8177 | 21.5454 | 0.1793 |
| SharQ | 0.1036 | 0.8006 | 21.0876 | 0.1866 |
| AWQ W4A16 reference | 0.0956 | 0.8187 | 21.4774 | 0.1805 |
| ThunderWorld 1.5× | 0.1046 | 0.7959 | 21.1081 | 0.1853 |
| ThunderWorld 2× | 0.0884 | 0.8254 | 22.0186 | 0.1693 |
| FP8 | 0.0748 | 0.8492 | 23.1032 | 0.1508 |

### Super / PAI-Bench-G

| Method | LPIPS ↓ | SSIM ↑ | PSNR dB ↑ | Relative L2 ↓ |
|---|---:|---:|---:|---:|
| W4A4 | 0.1956 | 0.6479 | 19.6913 | 0.2541 |
| ARCQuant 50% | 0.1195 | 0.7618 | 22.9016 | 0.1867 |
| ARCQuant 100% | 0.1119 | 0.7766 | 23.3401 | 0.1802 |
| SharQ | 0.1172 | 0.7651 | 22.9473 | 0.1856 |
| AWQ W4A16 reference | 0.1137 | 0.7758 | 23.2348 | 0.1836 |
| ThunderWorld 1.5× | 0.1257 | 0.7488 | 22.5786 | 0.1906 |
| ThunderWorld 2× | 0.1055 | 0.7878 | 23.7037 | 0.1723 |
| FP8 | 0.0885 | 0.8191 | 24.9851 | 0.1509 |

The 2× profile has lower mean LPIPS than the listed native FP4 baselines in all six groups. That is a mean comparison; consult paired intervals before making a significance claim. It does not dominate AWQ on every metric (for example, Nano / PAI-Bench-G), and neither released low-budget profile exceeds FP8 pixel fidelity. The 1.5× profile is a compute/storage trade-off, not the universal quality winner.

ARCQuant and SharQ keep BF16 attention; ThunderWorld and W4A4 use FP4 attention. AWQ here is a decoded ModelOpt NVFP4 W4A16 quality reference, whose cached BF16 weights do not establish deployment memory or speed. External method costs are not artificially forced into the ThunderWorld arithmetic cap.

## Adapted task judge

Judge: Qwen3-VL-8B-Instruct-FP8 through vLLM. RBench scores are in [1,5]; PAI-Bench-G Domain VQA is a fraction in [0,1], shown below as percent. Empty official PAI question sets are excluded from Domain VQA and their supplemental prompt-adherence score is reported separately. The judge is not the official leaderboard, and BF16 is a comparator rather than ground truth.

| Model | Method | RBench score ↑ | PAI Domain VQA % ↑ | RBench scored scenes | PAI scored scenes |
|---|---|---:|---:|---:|---:|
| Edge | BF16 | 2.8649 | 72.31 | 650 | 1036 |
| Edge | FP8 | 2.8658 | 71.91 | 650 | 1036 |
| Edge | W4A4 | 2.0423 | 67.64 | 650 | 1036 |
| Edge | ARCQuant 50% | 2.7792 | 71.46 | 650 | 1036 |
| Edge | ARCQuant 100% | 2.8062 | 72.73 | 650 | 1036 |
| Edge | SharQ | 2.8255 | 71.82 | 650 | 1036 |
| Edge | AWQ W4A16 reference | 2.8173 | 71.84 | 650 | 1036 |
| Edge | ThunderWorld 1.5× | 2.7501 | 71.53 | 650 | 1036 |
| Edge | ThunderWorld 2× | 2.7864 | 72.06 | 650 | 1036 |
| Nano | BF16 | 2.8880 | 74.89 | 650 | 1036 |
| Nano | FP8 | 2.8641 | 75.37 | 650 | 1036 |
| Nano | W4A4 | 1.3982 | 58.97 | 650 | 1036 |
| Nano | ARCQuant 50% | 2.8736 | 74.78 | 650 | 1036 |
| Nano | ARCQuant 100% | 2.9180 | 74.91 | 650 | 1036 |
| Nano | SharQ | 2.8636 | 75.32 | 650 | 1036 |
| Nano | AWQ W4A16 reference | 2.8790 | 75.41 | 650 | 1036 |
| Nano | ThunderWorld 1.5× | 2.8235 | 74.69 | 650 | 1036 |
| Nano | ThunderWorld 2× | 2.8556 | 75.25 | 650 | 1036 |
| Super | BF16 | 2.8978 | 76.73 | 650 | 1036 |
| Super | FP8 | 2.8828 | 76.79 | 650 | 1036 |
| Super | W4A4 | 2.7445 | 75.21 | 650 | 1036 |
| Super | ARCQuant 50% | 2.9049 | 76.80 | 650 | 1036 |
| Super | ARCQuant 100% | 2.9363 | 76.87 | 650 | 1036 |
| Super | SharQ | 2.8690 | 76.05 | 650 | 1036 |
| Super | AWQ W4A16 reference | 2.8580 | 76.13 | 650 | 1036 |
| Super | ThunderWorld 1.5× | 2.8936 | 76.47 | 650 | 1036 |
| Super | ThunderWorld 2× | 2.8700 | 76.16 | 650 | 1036 |

## Inspectable sources

- [Pixel summary](results/pixel_summary.csv): category-macro and scene-micro values, scene/video counts.
- [Pixel paired bootstrap](results/pixel_paired_bootstrap.csv): paired scene uncertainty; preserve its stated comparison family.
- [Judge summary](results/judge_summary.csv), [original paired judge intervals](results/judge_paired_original_intervals.csv), [PAI answer-bias diagnostic](results/pai_answer_bias.csv).
- Release loading validation: [Edge](results/edge-release-validation.json), [Nano](results/nano-release-validation.json), [Super](results/super-release-validation.json). Six complete-request hash checks validate reloading; historical full-corpus quality is a separate experiment.
- [Generated benchmark artifacts](https://huggingface.co/datasets/Concyclics/thunderworld-benchmark), [paper](https://www.preprints.org/manuscript/202610.0689), and [figure identities](assets/provenance.json).

## Throughput and memory

The original Figure 6 / paper throughput convention is **60 newly generated frames per request**. The 61-frame stored clip contains the conditioning frame. Historical utilities that report `61 × videos/s` use a different numerator and must not be silently compared. Latent-only, decoded full request, cold preparation and warm request timings are also separate quantities.

The paper reports 1.17–1.88× throughput over ARCQuant (2.0) for its specified PRO 6000 / Thor cases. This does not assert all baselines are slow, a universal win over BF16/FP8, or a new 5090 measurement. See [PERFORMANCE.md](PERFORMANCE.md). Asset-file bytes, resident weight bytes, scratch, allocated peak and reserved peak must be listed separately; the Hub packages do not contain all unchanged model components.

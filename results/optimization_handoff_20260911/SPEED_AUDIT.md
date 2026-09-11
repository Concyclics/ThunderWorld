# Verified speed ranges

> Ratios compare the same repair math inside each experiment. Ranges span shapes, not confidence intervals. Do not multiply ratios across experiments.

## Current frozen-choice A/W retiming

| Family | Model | Tier | Layout | Shapes | Speedup range | Shapes >1.05× |
|---|---|---|---|---:|---:|---:|
| A | edge | a50 | gather | 6 | 0.770–0.889× | 0 |
| A | edge | a50 | direct | 6 | 0.777–0.889× | 0 |
| A | edge | act2 | gather | 6 | 0.823–0.977× | 0 |
| A | edge | act2 | direct | 6 | 0.975–1.154× | 2 |
| A | edge | a50w | gather | 6 | 0.764–0.983× | 0 |
| A | edge | a50w | direct | 6 | 0.795–0.987× | 0 |
| A | edge | dual | gather | 6 | 0.911–1.034× | 0 |
| A | edge | dual | direct | 6 | 0.945–1.094× | 1 |
| A | nano | a50 | gather | 8 | 0.731–0.985× | 0 |
| A | nano | a50 | direct | 8 | 0.787–1.099× | 1 |
| A | nano | act2 | gather | 8 | 0.811–0.881× | 0 |
| A | nano | act2 | direct | 8 | 0.940–1.100× | 2 |
| A | nano | a50w | gather | 8 | 0.757–0.990× | 0 |
| A | nano | a50w | direct | 8 | 0.788–1.062× | 1 |
| A | nano | dual | gather | 8 | 0.825–1.038× | 0 |
| A | nano | dual | direct | 8 | 0.891–1.070× | 3 |
| W | edge | wgt2 | plane | 4 | 0.909–1.095× | 1 |
| W | edge | wgt2 | pair | 4 | 0.998–1.375× | 2 |
| W | edge | a50w | plane | 4 | 0.916–1.003× | 0 |
| W | edge | a50w | pair | 4 | 0.836–1.063× | 1 |
| W | edge | dual | plane | 4 | 0.880–1.105× | 1 |
| W | edge | dual | pair | 4 | 0.850–0.971× | 0 |
| W | nano | wgt2 | plane | 6 | 0.899–0.991× | 0 |
| W | nano | wgt2 | pair | 6 | 0.943–1.236× | 3 |
| W | nano | a50w | plane | 6 | 0.894–0.953× | 0 |
| W | nano | a50w | pair | 6 | 0.796–0.934× | 0 |
| W | nano | dual | plane | 6 | 0.870–1.037× | 0 |
| W | nano | dual | pair | 6 | 0.803–0.908× | 0 |

A includes selector + packing + GEMM; W repeats the fixed-support packing + GEMM protocol. W-pair numerical/trajectory limitations still apply.

## Archived launch tuning and full-FFN fusion

| Model | Tier | Native launch tuning | Complete FFN fusion |
|---|---|---:|---:|
| edge | a50 | 0.998–1.591× | 1.067–1.072× |
| edge | act2 | 1.001–2.049× | 1.040–1.061× |
| edge | a50w | 0.979–1.553× | 1.041–1.067× |
| edge | dual | 1.024–1.943× | 1.034–1.057× |
| nano | a50 | 0.985–1.262× | 1.006–1.010× |
| nano | act2 | 0.957–1.614× | 1.007–1.010× |
| nano | a50w | 0.946–1.429× | 1.005–1.008× |
| nano | dual | 0.946–1.669× | 1.002–1.007× |

## Archived complete-request Graph comparison

| Model | Mode | Task | Graph / eager speedup | Graph + fusion / eager speedup |
|---|---|---|---:|---:|
| edge | ffn_act2 | robot_table | 1.819× | 1.817× |
| edge | ffn_auto50 | robot_table | 1.721× | 1.750× |
| edge | ffn_auto75 | robot_table | 1.790× | 1.817× |
| edge | ffn_act2 | policy_table | 2.029× | 2.251× |
| edge | ffn_auto50 | policy_table | 1.771× | 1.894× |
| edge | ffn_auto75 | policy_table | 1.912× | 1.941× |
| nano | ffn_act2 | robot_table | 1.217× | 1.213× |
| nano | ffn_auto50 | robot_table | 1.218× | 1.202× |
| nano | ffn_auto75 | robot_table | 1.176× | 1.174× |
| nano | ffn_act2 | policy_table | 1.291× | 1.291× |
| nano | ffn_auto50 | policy_table | 1.367× | 1.372× |
| nano | ffn_auto75 | policy_table | 1.307× | 1.323× |
| nano | ffn_act2 | av_pottery | 1.204× | 1.194× |
| nano | ffn_auto50 | av_pottery | 1.181× | 1.179× |
| nano | ffn_auto75 | av_pottery | 1.163× | 1.174× |

Graph results are archived two-seed experiments, not new release timings. Separate-process small differences are noisy; BF16 controls revealed environmental variation. No cross-stage compounded speedup is claimed.

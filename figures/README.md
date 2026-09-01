# Figure index

These figures are frozen views of the reviewed experiment families. The
machine-readable values and evidence boundaries are in `data/` and
`docs/EXPERIMENTAL_FINDINGS.md`.

| Figure | Scope | Supported use |
| --- | --- | --- |
| `attention-relation-demo.png` | sampled Cosmos3 attention matrices | qualitative relation/archetype illustration |
| `finding-evidence-overview.png` | structural finding map | relation/head/layer overview |
| `anchor-efficiency.png` | action/audio relation atlas | dense-anchor motivation |
| `equal-budget-overview.png` | 8 tasks × 24 prompts × 3 seeds × 20/40 steps | geometry/semantic/hybrid equal-budget comparison |
| `cross-self-negative.png` | 45 dense trajectories | position-controlled cross→self negative result |
| `timestep-patterns.svg` | 24 T2V prompts × 3 seeds × 20 steps | timestep sparsifiability and layout-reuse evidence |
| `trajectory-frontier.png` | 4 held-out T2V prompts × 3 seeds | local-versus-closed-loop policy frontier |

Do not use the structural relation-atlas figures as independent-prompt
confidence evidence. Do not read kernel or end-to-end speed from these figures;
use `data/kernel_scaling.csv` and `data/quality_compute_frontier.csv` instead.

# Standalone figures

The retained AI overview plus 13 new reproducible figures. SVG text remains editable; PDF figures are suitable for paper/slides.

| Figure | Content | Formats |
|---|---|---|
| 1 | AI-generated overall architecture | [PNG](assets/thunderworld_architecture.png) |
| Overview | Motivation, candidate algorithm contributions, and implemented system status. | [SVG](assets/figures/00_research_logic.svg) · [PDF](assets/figures/00_research_logic.pdf) · [PNG](assets/figures/00_research_logic.png) |
| 2 | Logical residual terms and the distinction between executed and omitted terms. | [SVG](assets/figures/02_residual_products.svg) · [PDF](assets/figures/02_residual_products.pdf) · [PNG](assets/figures/02_residual_products.png) |
| 3 | Offline allocation and full-trajectory gates; online phase and current-input selection. | [SVG](assets/figures/03_budget_flow.svg) · [PDF](assets/figures/03_budget_flow.pdf) · [PNG](assets/figures/03_budget_flow.png) |
| 4 | Eager dependencies versus captured execution; geometry is not duration. | [SVG](assets/figures/04_execution_flow.svg) · [PDF](assets/figures/04_execution_flow.pdf) · [PNG](assets/figures/04_execution_flow.png) |
| 5 | Equal trajectory means, query-count-weighted DM mass; all layers, heads and 35 steps, conditional branch. | [SVG](assets/figures/05_depth_mass.svg) · [PDF](assets/figures/05_depth_mass.pdf) · [PNG](assets/figures/05_depth_mass.png) |
| 6 | Dots are whole-trajectory ratios, not confidence intervals; Nano sound curves average layers/heads within a trajectory then two seeds. | [SVG](assets/figures/06_directionality.svg) · [PDF](assets/figures/06_directionality.pdf) · [PNG](assets/figures/06_directionality.png) |
| 7 | Top: all observation trajectories. Bottom: robot_observe, two seeds, middle layer, common heatmap scale. First step has no predecessor and is omitted. | [SVG](assets/figures/07_temporal.svg) · [PDF](assets/figures/07_temporal.pdf) · [PNG](assets/figures/07_temporal.png) |
| 9 | Seeds 6103/6104, BF16 attention. Video mean over two seeds; action/audio maxima. Reference is same-seed BF16. | [SVG](assets/figures/09_quality_ablation.svg) · [PDF](assets/figures/09_quality_ablation.pdf) · [PNG](assets/figures/09_quality_ablation.png) |
| 8 | Exploratory eager requests; worst SSIM and paired geometric-mean speedup against own uncached reference. Different model case sets. | [SVG](assets/figures/08_cache_tradeoff.svg) · [PDF](assets/figures/08_cache_tradeoff.pdf) · [PNG](assets/figures/08_cache_tradeoff.png) |
| 10 | All 15 FP4 configurations from the latest execution cohort. Two seeds, 35 steps, Graph+fusion, common BF16 Graph reference. | [SVG](assets/figures/10_end_to_end.svg) · [PDF](assets/figures/10_end_to_end.pdf) · [PNG](assets/figures/10_end_to_end.png) |
| 11 | Graph speedups versus own eager tier; I2V peak allocated memory with resident W1, not weight-only compression. | [SVG](assets/figures/11_execution_memory.svg) · [PDF](assets/figures/11_execution_memory.pdf) · [PNG](assets/figures/11_execution_memory.png) |
| 12 | Left: idealized r=4 cost model, not measured speedup. Right: measured retuned cases, separating true multichunk execution from one-chunk cases. | [SVG](assets/figures/12_compute_limits.svg) · [PDF](assets/figures/12_compute_limits.pdf) · [PNG](assets/figures/12_compute_limits.png) |
| 13 | Implementation priorities and proposed side-by-side demo; not a screenshot of an existing UI. | [SVG](assets/figures/13_roadmap_demo.svg) · [PDF](assets/figures/13_roadmap_demo.pdf) · [PNG](assets/figures/13_roadmap_demo.png) |

[Plotted data and provenance](data/visual_manifest.json) · [Drawing source](build_figures.py) · [Download all figures](FIGURES.zip)

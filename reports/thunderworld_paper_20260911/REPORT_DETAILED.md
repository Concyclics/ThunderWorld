# ThunderWorld: Dense FP4 Computation and Structured Precision Repair for World Models

**Research Narrative, Methodology, and Implementation Evidence · 2026-09-11**  
Models: Cosmos3-Edge / Cosmos3-Nano. Measured hardware: RTX 5090.  
This report consolidates the design, resampling analysis, quality ablations, automatic budgets, and execution optimizations available as of this date. It supports paper development, implementation planning, and demo design.

## Executive Summary

**ThunderWorld aims to deliver high-quality world-model inference through complete low-precision base computation supplemented by a small, selected set of low-precision error repairs.** The base information paths remain intact. Additional computation is allocated according to repair benefit, while native FP4 Tensor Cores execute the main matrix products. Higher-precision accumulation, necessary nonlinear operations, and sensitive modules are retained.

The research connects three levels: local deployment calls for lower storage and response costs; FP4 residual representations offer adjustable precision; and the layer, generation-phase, and modality structure of world models provides potential signals for allocating repair. The system turns these signals into regular work that hardware can execute efficiently.

The current implementation includes native FP4 FFN repair on both models, automatic layer/phase/CFG budgets, quality gates, shape tuning, whole-Transformer CUDA Graphs, and equivalent activation/amax fusion. Across the latest 120 complete trajectories, the execution changes preserve bitwise-identical outputs and identical actual budgets. For the three main FP4 tiers, Graph execution accelerates Edge by **1.72–2.03×** and Nano by **1.16–1.37×** relative to each tier's own eager path. These are execution gains, not gains attributable to automatic allocation or to FP4 versus BF16. [Execution evidence](../../docs/GRAPH_FUSION_5090_20260911.md)

The most important unfinished algorithmic components are **repair-aware first-pass quantization** and **a mapping from structural features to actual repair benefit**. Automatic budgets do not yet consistently outperform uniform repair; attention repair is outside the recommended path; and energy savings and Thor/Spark performance have not been measured. The current status is therefore a validated core execution system with partial support for the algorithmic hypotheses, while the complete method is still being developed.

## 1. High-Level Motivation: Why This Matters

### 1.1 Deployment value for world models

World models must maintain temporal, physical, and conditioning consistency over multiple generation steps, potentially producing actions or joint audio and video. For embodied intelligence, shorter inference can permit more candidate evaluations before a decision. Lower memory use may accommodate longer contexts or more parallel hypotheses, while lower energy per request may extend operating time. These latter benefits require separate measurements; latency alone does not establish them.

The project should therefore optimize **latency, memory, and energy per complete inference trajectory under task-quality constraints**. Tensor Core utilization is a diagnostic measure, rather than the final research objective.

### 1.2 Native FP4 hardware creates an opportunity, but peak throughput is not application speedup

Jetson Thor targets robotics and physical AI, while DGX Spark supports local model development and execution. Both offer FP4 compute. Their advertised peaks include hardware-sparsity assumptions and cannot directly serve as the utilization denominator for the current dense MMA path. [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/); [DGX Spark specifications and footnotes](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

NVIDIA's current HGX table lists Rubin dense NVFP4 training throughput at 35 PFLOPS and dense BF16 at 4 PFLOPS, a comparable ratio of 8.75. The page still marks these as preliminary specifications. This motivates studying the growing low-/high-precision throughput gap; it does not predict ThunderWorld's speedup, and it must not be combined with the sparse-inference column. Product specifications and footnotes should be rechecked before submission. [HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/)

Practical constraints include quantization errors propagating through generation, the additional matrix products required by repair, and overhead from online quantization, scale statistics, softmax, nonlinear operations, data movement, scheduling, and inefficient matrix shapes. FP32 accumulation is already part of the relevant MMA path. It is inaccurate to attribute the entire shortfall in FP4 speedup to higher-precision accumulation.

### 1.3 Low precision is inexpensive; comprehensive precision recovery is not

The FP4 codebook is coarse. The nonnegative NVFP4 E2M1 values are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. This project uses K16 microblocks with E4M3 scales and an outer factor. The bulk of a distribution and its outliers may not fit the same grid well, and the largest absolute value need not have the largest quantization error or output impact. [Format reference](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

One representation uses approximately `4 + 8/16 = 4.5 bit/element`; two complete FP4 planes use approximately 9 bit/element before padding, outer factors, and indices. Residual repair can therefore still reduce operand storage, but two FP4 planes do not imply fourfold memory compression. Operand compression also differs from the reduction in whole-model peak memory.

A repair budget must answer both **which errors are worth correcting** and **whether the cost of correcting them is justified by the benefit**. This motivates designing the numerical method and execution system together.

## 2. Paper Narrative and Contributions

### 2.1 Proposed central claim

> For mixed AR–DM world models, preserve complete low-precision base computation, translate nonuniform sensitivity across model structure and generation trajectories into precision-repair budgets, and compile these budgets into regular native FP4 computation to improve inference cost under quality constraints.

Suggested paper title: **ThunderWorld: Budgeted Precision Repair for FP4 World-Model Inference**. Until repair-aware quantization is complete, the title and abstract should not claim a completed “near-lossless iterative FP4” method.

The narrative should follow deployment constraints → single-pass FP4 error → full-repair cost → nonuniform repair benefit → structured budgets → hardware execution → complete-trajectory validation. CUDA Graphs and fusion belong to the system contribution and do not substitute for the algorithmic contribution.

### 2.2 Proposed contributions and available evidence

| Contribution | What the paper must establish | Current status and limits |
|---|---|---|
| C1: Repair-aware FP4 representation | Jointly optimize first-pass quantization and a limited residual budget to improve quality at equal cost | Base/residual FP4 and product expansion are implemented. The first pass still uses absmax RTN; bulk-fit and repair-aware calibration remain to be implemented. |
| C2: Structure- and trajectory-informed budgets | Layer, phase, head, and modality relationships predict repair benefit and outperform equal-budget uniform and random allocation | Both findings have been resampled. FFN layer/phase/CFG allocation and quality fallback exist. Head/modality attention policies and consistent gains remain unproven. |
| C3: An efficient, executable repair system | Regular FP4 GEMMs, quantization fusion, and scheduling convert theoretical savings into inference gains | Both models' FFNs, the SM120 backend, shape tuning, Graphs, and equivalent fusion have been tested. Cross-device performance and energy efficiency remain unvalidated. |
| C4: Complete-trajectory evidence and failure analysis | Identify useful approximations, task-specific failures, and the limits of local metrics | Multimodal resampling, repair-term/cache ablations, quality gates, numerical-equivalence checks, and negative results are available. Scene coverage remains limited. |

The complete forms of C1/C2 are the main candidate algorithmic contributions. C3 is already a substantial engineering contribution, and C4 supports the credibility of the method. Novelty still requires comparison with related work; naming a component does not establish priority.

### 2.3 Related work and distinctions to preserve

FP4 residual methods already exist. For example, ARCQuant folds residual channels into the reduction dimension to execute a unified NVFP4 GEMM. ThunderWorld must demonstrate its specific advances in world-model repair budgets, numerical representation, and execution, rather than claim that adding another FP4 representation is itself unique. [ARCQuant](https://arxiv.org/abs/2601.07475)

Sparse attention primarily reduces computation over base connections. ThunderWorld's design preserves the connections permitted by the original model and selects the repair terms. These approaches may be combined, or each may work better at different quality targets and shapes. Existing sparse methods should not all be described as requiring expensive retrieval or being unsuitable for Tensor Cores: Sparse VideoGen includes hardware-oriented layouts and kernels. [Sparse VideoGen](https://arxiv.org/abs/2502.01776)

HPC mixed-precision iterative refinement inspires the combination of inexpensive computation and error correction. Here, however, the method uses successive tensor residual representations and a limited set of product corrections. It is not the same iterative algorithm used to solve `Ax=b`, and its convergence theorems do not transfer directly. [Higham on iterative refinement](https://nhigham.com/2017/07/)

## 3. Architecture Overview

![ThunderWorld architecture: implemented FFN path and research roadmap](assets/thunderworld_architecture.png)

**Figure 1. Preparation, budgeting, and execution in ThunderWorld.** Generated with the built-in image_gen tool; the [generation prompt](assets/architecture_prompt.txt) and [correction prompt](assets/architecture_edit_prompt.txt) are included. Solid paths show the current implementation: prepare RTN FP4 weights and repair policies offline, quantize and select blocks from current activations online, and execute the complete base plus selected repairs with higher-precision accumulation. Dashed paths show research directions. Repair products are logical terms that may be fused within one native GEMM; model code determines FFN projection and layer ordering. Attention, necessary nonlinear operations, and normalization retain higher precision.

Three interfaces are essential: calibration must include actual hardware costs; budgets must map to executable tiles; and Graphs must copy current-step inputs and switch phases correctly, rather than use the capture-time budget for the entire trajectory.

## 4. Methodology: From Residual Representation to Adjustable Precision

### 4.1 Mathematical formulation of iterFP4

Here, `Q` denotes the mathematical value decoded after quantization. It does not imply materializing a complete BF16 operand at runtime. For a linear layer, let `A ∈ R^(M×K)` and `W ∈ R^(K×N)`:

```text
X0 = Qs0(X)
R1 = X − X0
X1 = Qs1(R1)
X ≈ X0 + X1

A ≈ A0 + A1;  W ≈ W0 + W1
Y_hat = A0W0 + (SA ⊙ A1)W0 + A0(SW ⊙ W1)
```

The base term `A0W0` is always executed. `SA/SW` determine the residual supports. The full four-term reference additionally includes `A1W1`. This is a bilinear product expansion, not a quadratic equation. The main GEMM directly consumes FP4 codes/scales with FP32 accumulation and BF16 output. A precise description of “no dequantization” is that the main GEMM does not require materializing complete BF16 operands; scale application and necessary local conversions still occur.

Let `Ar=SA⊙A1`, `Wr=SW⊙W1`, `EA=A−A0−Ar`, and `EW=W−W0−Wr`. In exact arithmetic:

```text
AW − Y_hat = ArWr + EA(W0+Wr) + (A0+Ar)EW + EA EW
```

Error therefore includes omitted support and residual quantization error, in addition to the omitted cross term. Actual execution also introduces accumulation and rounding errors. Deliberate first-pass clipping changes these terms: an earlier observation that the cross term is small cannot simply be reused after changing the quantizer.

### 4.2 Why budgeted repair is necessary

If the executed A/W repair fractions are `ρA/ρW`, the nominal matrix-product cost is approximately `1+ρA+ρW`. Assuming a comparable native FP4/BF16 throughput ratio of `r=4`, the ideal pure-GEMM limits are:

| Repair configuration | Nominal product cost | Ideal speedup limit, r=4 |
|---|---:|---:|
| Base only | 1 | 4× |
| Half A | 1.5 | 2.67× |
| Full A | 2 | 2× |
| Half A + full W | 2.5 | 1.6× |
| Both first-order repairs in full | 3 | 1.33× |

These illustrative limits ignore quantization, bandwidth, and shape efficiency. If only a fraction `p` of execution time is replaceable, a simplified whole-network model is `S ≈ 1 / [(1−p)+p(1+ρA+ρW)/r]`. Actual allocation should use measured latency rather than select tiers from nominal FLOPs alone.

### 4.3 Coordinating the first pass with repair

The intuition of fitting the distribution bulk first and correcting the residual can become an offline search objective: minimize calibration output error under legal scales, hardware tiles, and a fixed budget, accounting for measured latency and storage cost. Comparisons should progress from absmax RTN to single-pass MSE/output-MSE scale search, repair-aware search with fixed support, and alternating quantization/support optimization.

Outliers cannot be ignored unconditionally. If a location receives no repair budget, discarding it in the first pass may worsen the final output. The objective must evaluate the actual `Y_hat`, without assuming that the entire residual is recovered. **This is a proposed method component, not a source of the currently measured gains.** [Numerical design audit](../../docs/THUNDERWORLD_ITERFP4_DESIGN.md)

## 5. World-Model Structure: Validated Findings and Their Role in the Method

### 5.1 Depth prior: DM attention to AR context declines

Using the convention “query ← key/value,” DM←AR means that the Generator reads context from the AR region. The 28 BF16 observation trajectories cover all 35 steps and every layer/head. At most 32 queries are randomly sampled per real role, with statistics computed over all keys. Shallow and deep refer to the first and last quarters of the layers.

| Metric | Edge | Nano |
|---|---:|---:|
| Shallow DM←AR mass | 0.2126 | 0.3869 |
| Deep DM←AR mass | 0.1068 | 0.1235 |
| Mean per-trajectory depth Pearson correlation | −0.3475 | −0.6922 |

This supports depth as a candidate prior, but does not establish a monotonic decrease at every layer or justify removing deep-layer AR information. The experiments do not include full AR Reasoner integration, so not all AR tokens can be interpreted as generated reasoning traces. [Resampling report](../../docs/FINDINGS_RESAMPLING_20260910.md)

### 5.2 Directed modality prior: control for token counts

Let `m(i,j)` be the mean row mass from i-queries to j-keys and `n_j` the number of j tokens. Report both the raw directional ratio and the density ratio: `[m(i,j)/n_j] / [m(j,i)/n_i]`.

| Direction compared with its reverse | Raw row-mass ratio | Length-normalized density ratio |
|---|---:|---:|
| Edge action ← noisy vision | 24.0–29.2× | 1.40–1.61× |
| Nano action ← noisy vision | 14.8–18.8× | 0.925–1.051× |
| Nano sound ← noisy vision | 12.8–16.4× | 0.850–1.095× |

Edge action retains moderate directionality. Nano action/audio are close to symmetric when averaged over complete trajectories, and sound directionality changes with step. The paper can discuss nonuniform attention allocation under real token layouts, but should not interpret the large raw ratios as modality preferences independent of sequence length. [Normalization and phase trends](../../docs/FINDINGS_RESAMPLING_20260910.md)

### 5.3 Temporal prior: stable selection does not imply stable values

For noisy-vision queries, the dominant mass role remains unchanged across adjacent steps about 99.16% of the time in both models. Yet attention-output relative L2 change is 0.1163 for Edge and 0.2190 for Nano. The Jaccard overlap of mass-ranked top-50% tiles is 0.9407 and 0.9123, respectively. These observed supports are not the runtime repair selector's supports.

Changes are generally smaller in the latter half of generation, then rise again in the final steps. Existing ablations of mask, repair-delta, and full-output reuse have not shown reliable net speedup. For Nano AV, alternating-step output caching throughout generation reaches a worst SSIM of 0.4352 against the uncached reference. The current validated execution path therefore computes fresh values at every step.

### 5.4 From correlation to repair benefit

The quantity relevant to allocation is counterfactual repair benefit, `Δloss / Δtime`: at the same trajectory state, change only one support or support group and measure changes in outputs and downstream predictions. Attention mass, token counts, V energy, and head stability should be input features, not substitutes for repair benefit.

FFN repair operates on M/N/K supports. Attention requires separate calibration of QK and PV, query heads/GQA groups, directed modality relationships, and token tiles. These components may share an upper-level budgeting principle but cannot share an unvalidated mask. Changes to QK affect the joint softmax; attention repair cannot be implemented by simply summing independently normalized tiles.

## 6. Implemented Automatic Budgets and Execution System

### 6.1 Budget definition and allocation

The decision unit is “task × CFG branch × denoising phase × linear operator within a layer.” Phases are `[0,.2), [.2,.5), [.5,.8), [.8,1)`, corresponding to 7/11/10/7 of the 35 steps. Candidates are base, half A, full A, full W, half A + full W, and full A + full W.

The 25%/50%/75% tiers cap extra FP4 product FLOPs relative to both first-order repairs in full, allowing an average of 0.5/1/1.5 extra products. They are neither memory fractions nor fractions of total network FLOPs.

The current process collects 672 local units for Edge and 2,160 for Nano, measures errors and real-shape latency, greedily allocates positive error improvement per incremental time, applies a half-A floor, validates quality over complete trajectories, and exports a frozen policy. Online execution looks up the tier for the current layer/phase/CFG branch and selects half-A K128 support on the GPU using current activation energy.

Thus, “dynamic” means phase switching and current-value-dependent block selection. It does not mean invoking a BF16 teacher during generation for closed-loop quality estimation. With the half-A floor, the 25% tier becomes uniform a50 and cannot count as an allocator gain. [Budget implementation](../../docs/AUTO_BUDGET_5090_20260910.md)

### 6.2 Quality gates and current limitations

Local greedy allocation without a minimum A budget previously caused substantial complete-trajectory degradation. The revised procedure uses equal-budget uniform repair as the reference and separately checks video, action, and audio. Of 10 task/budget combinations, only 3 automatic candidates are retained; 7 fall back to uniform repair. The retained candidates are Edge I2V auto50 and Nano I2V auto50/auto75.

Local probes contain only 64 consecutive rows and recompute amax and support, without fully preserving the complete-M scale/selection context. This is an identified calibration-to-inference mismatch. Quality gating is validation-set selection, not a final blind test. Although the latest execution tests use new seeds, they reuse existing scenes and do not establish generalization to unseen scenes.

### 6.3 Actual computation path

| Component | Implemented role | Main entry point |
|---|---|---|
| Native FP4 GEMM | Complete base term, independent A/W repair supports, higher-precision accumulation | [native.py](../../experiments/ffn_block_repair_20260907/native.py) |
| Automatic-budget runtime | GPU block selection from current inputs, layer/phase/CFG tiers, actual FLOP accounting | [dynamic.py](../../tools/auto_budget/dynamic.py) |
| Shape tuning | Search tiles and stages by M/N/K/tier | [auto_budget tools](../../tools/auto_budget/README.md) |
| Whole-Transformer Graph | Separate phase/branch captures, current-input copies, independently owned outputs, updated counters | [phase_graph.py](../../tools/auto_budget/phase_graph.py) |
| Equivalent fusion | Combine Edge ReLU² or Nano BF16 multiplication with down-input amax | [fusion.py](../../tools/auto_budget/fusion.py) |
| Two-model multimodal validation | Complete Generator, arrays/videos, source-code and budget provenance | [run.py](../../tools/finding_resample/run.py) |

Graph execution supports the current BF16/FFN modes. Automatic tiers use up to 4 phases × 2 CFG branches, captured during warmup. Graphs neither reduce mathematical work nor cache the previous step's output. Nano retains the original SiLU implementation. Fusion performs the same BF16 rounding before computing amax to preserve complete-trajectory equivalence.

All tiers share prepared W0/W1, and W1 currently remains resident. Pruning W1 that a single policy never uses, compact residual assets, and a unified checkpoint/API remain unfinished. Reproducible experiment tools do not yet constitute a production deployment SDK.

## 7. Measured Results: Attribution by Evidence Level

### 7.1 Quality contribution of repair terms

The following independent repair-ablation cohort uses seeds 6103/6104. All FFN variants retain BF16 attention. I2V reports mean video SSIM across both seeds; action reports the maximum relative L2 across both policy seeds. [Full ablation and LPIPS](../../results/findings_20260910/ablation/ABLATION.md)

| FFN terms | Edge I2V SSIM ↑ | Nano I2V SSIM ↑ | Edge action L2 ↓ | Nano action L2 ↓ |
|---|---:|---:|---:|---:|
| Base | 0.5337 | 0.9229 | 0.3826 | 0.1754 |
| Base + A repair | 0.5329 | 0.9563 | 0.1028 | 0.0739 |
| Base + W repair | 0.5727 | 0.9221 | 0.2546 | 0.0824 |
| Both first-order repairs | 0.6919 | 0.9603 | 0.0366 | 0.0219 |
| All four terms | 0.7710 | 0.9644 | 0.0344 | 0.1217 |

With both first-order repairs, all-frame LPIPS falls from the base value of 0.4254 to 0.2437 for Edge, and from 0.0474 to 0.0182 for Nano. Nano's worst audio waveform relative L2 falls from 1.1496 for base to 0.0489 with both first-order repairs, but is 0.0766 with all four terms.

These results support useful, task-dependent residual repair and correct the assumption of a universal ordering. A-only is more effective for Nano I2V, but that conclusion does not hold for the current Edge I2V samples. More repair terms do not guarantee improvement in every complete-trajectory metric. Even all four terms do not recover BF16-equivalent quality for Edge, pointing to first-pass quantization and sensitive-module handling as key gaps.

### 7.2 The allocator's independent contribution remains limited

In quality-gate development validation with seeds 7203/7204, the retained candidates achieve 1.068× for Edge I2V auto50, 1.022× for Nano I2V auto50, and 0.987× for Nano I2V auto75 relative to equal-budget uniform repair. Each uses two seeds and one request per seed. Small differences cannot be presented as stable advantages.

Retention only indicates compliance with provisional relative tolerances. For example, Edge auto50 has mean SSIM against BF16 of only 0.6286 and remains slightly slower than BF16. The current allocator contributes executable budgets and multimodal degradation detection, rather than a generally improved quality–speed Pareto frontier. [Quality-gate evidence](../../docs/AUTO_BUDGET_5090_20260910.md)

### 7.3 Latest end-to-end results under a common Graph framework

The following results use frozen `policy_validated.json`, seeds 7403/7404, and 35 steps per trajectory. Candidates use Graph+fusion, and BF16 uses the same Graph framework. Times are means across the two seeds; speedups are geometric means of paired, same-seed ratios. ACT2 and auto75 show the baseline comparison and a higher-budget tier; the source table also includes auto50 and other rows. [All 20 result groups](../../results/execution_20260911/summary.csv)

| Model / task | Tier | BF16 time, s | Candidate time, s | Speedup vs. BF16 | Video SSIM ↑ |
|---|---|---:|---:|---:|---:|
| edge / I2V | act2 | 3.024 | 2.632 | 1.149× | 0.7167 |
| edge / I2V | auto75 | 3.024 | 2.819 | 1.073× | 0.6876 |
| edge / policy | act2 | 1.290 | 1.253 | 1.029× | 0.9199 |
| edge / policy | auto75 | 1.290 | 1.277 | 1.010× | 0.9560 |
| nano / I2V | act2 | 8.586 | 6.657 | 1.290× | 0.9742 |
| nano / I2V | auto75 | 8.586 | 7.070 | 1.215× | 0.9726 |
| nano / policy | act2 | 3.560 | 2.655 | 1.341× | 0.9464 |
| nano / policy | auto75 | 3.560 | 2.713 | 1.312× | 0.9702 |
| nano / AV | act2 | 9.233 | 6.895 | 1.339× | 0.8090 |
| nano / AV | auto75 | 9.233 | 7.467 | 1.237× | 0.8796 |

Video SSIM measures consistency with same-seed BF16, not absolute generation quality, physical accuracy, or task success. Policy also requires separate action validation; AV requires audio and synchronization validation. Some auto75 tasks have fallen back to uniform budgets, so their speedups cannot be attributed to automatic allocation.

Timing includes image encoding, denoising, video decoding, and a consistent trajectory sketch. It excludes model loading, offline quantization, JIT/Graph warmup, and serialization. AV waveform decoding and the AR Reasoner are outside scope. There are only two seeds and previously seen scenes; variants run sequentially in separate processes. These measurements are not service-level latency distributions.

As a negative control, Nano BF16 AV has mean request times of 10.992 / 9.233 s in the Graph / Graph+fusion processes, even though no FFN fusion occurs in this mode. Transformer event times are similar. Differences in other process-level costs can therefore affect small apparent speedups; the complete negative control is retained.

### 7.4 Execution gains and memory

| Optimization | Measured result | Attribution limit |
|---|---|---|
| Real-shape/tier launch tuning | 60 operator cases, 360 candidates with identical outputs; selected small-M Edge/Nano down projections reach 2.05×/1.67× | Operator-level gains, not whole-model speedups |
| Whole-Transformer Graph | FP4 ACT2/auto50/auto75 versus their own eager paths: Edge 1.72–2.03×, Nano 1.16–1.37× | Submission/execution gains; BF16 also benefits |
| Pointwise + amax fusion | 20 complete FFN cases: Edge +3.4–7.2%, Nano +0.2–1.0% | Additional end-to-end gains are not stable |
| Shared Nano gate/up packing | Local two-projection gains of +0.5–3.6%; 9 bitwise-identical trajectories | Not enabled together with this round's fusion; gains must not be multiplied |
| Latest equivalence validation | Identical outputs and budgets across three paths on 120 complete trajectories; 3 new GPU tests pass; memcheck reports 0 errors | Equivalence within the same FP4 tier, not losslessness against BF16 |

Latest I2V peak allocated memory falls from Edge BF16 8.208 GiB to auto75 7.354 GiB, approximately 10.4%, and from Nano 30.228 to 25.860 GiB, approximately 14.4%. These figures use the current budget runtime with resident W1. Earlier pilots retaining only A-repair assets have different memory results and must not be combined with this cohort into one operating point. Graphs add less than 11 MiB to peak allocation for every configuration. [Execution and memory definitions](../../docs/GRAPH_FUSION_5090_20260911.md)

### 7.5 Tensor Cores are not saturated; negative results inform system design

Across eight down-projection profiles, Tensor-pipe elapsed activity is 14.2–44.6% and active-warp occupancy is 8.2–22.4%; some configurations use 254 registers/thread. This activity metric is not utilization of theoretical FLOP/s, and Nsight replay time is not the official steady-state latency. [Hardware evidence](../../docs/AUTO_BUDGET_5090_20260910.md)

Independent SIMT/GEMM probes on separate streams showed local overlap, but a real producer–consumer pipeline with two M-chunk buffers failed to beat the original path in all 120 initial candidates. A further 240 tile/stage configurations were tested. The best genuinely multichunk configuration remained slower than a complete GEMM. Two apparent gains of approximately 1–1.9% came from M316/chunk512, which actually uses only one chunk. Double buffering therefore remains a research tool and is disabled by default.

Local Split-K errors previously amplified during complete generation. The current path preserves K accumulation order. More concurrency, more repair terms, lower local error, or higher utilization of an individual unit cannot substitute for complete-request quality and performance validation.

## 8. Paper Structure and Figures

### 8.1 Draft abstract reflecting current evidence

Local world-model deployment faces both inference-latency and memory constraints, while quantization error and repair overhead limit the potential of native FP4 computation. We investigate ThunderWorld, a framework that preserves complete low-precision base computation and adds FP4 repair for selected errors. On Cosmos3-Edge and Cosmos3-Nano, we reexamine layer, modality, and generation-phase characteristics, implement FFN activation/weight residual repair and a hardware-cost-aware budget prototype, and constrain policies using complete-trajectory multimodal metrics. Shape tuning, phase-specific CUDA Graphs, and equivalent fusion reduce execution overhead; the latest validation across 120 trajectories preserves outputs and budgets within each tier. The experiments also identify modality-length confounding, mismatch between local calibration and complete trajectories, and failure modes of caching and chunked pipelines. Repair-aware first-pass quantization, attention-structure-guided budgets, unseen-scene quality, and energy efficiency require further validation.

This abstract is suitable for an internal progress report or the current system version. A full algorithm paper should revise it after validating C1/C2, rather than submit it with the limitations simply removed.

### 8.2 Suggested organization

| Section | Question to answer | Key figures and tables |
|---|---|---|
| Introduction / Motivation | Why allocate precision instead of applying uniform compensation? | Budget versus ideal cost, deployment constraints |
| Characterization | Which structural and temporal features are reliable? | Depth × step mass, length-normalized directionality, head stability versus output drift |
| Method | How are quantization, repair selection, and budget compilation defined? | Architecture, product expansion, benefit/cost and quality-gate flow |
| System | How is the native FP4 execution advantage preserved? | Tiles/packing, Graph phases, SIMT/GEMM overhead breakdown |
| Evaluation | Is inference faster and less costly at matched quality? | Common-framework Pareto curves, per-task quality, memory and energy |
| Ablation / Limitations | Which designs contribute, and which fail? | RTN/repair-aware, A/W/cross, uniform/random/structure, cache and pipeline failures |

Use the latest end-to-end table with matched inputs and execution frameworks in the main text. Label the original layer-0 package microbenchmarks, early single-scene pilots, and later budget experiments as separate cohorts. Their speedups cannot be multiplied, and trajectory counts cannot simply be added as independent samples.

## 9. Implementation and Demo Priorities

| Priority | Next deliverable | Acceptance criterion |
|---|---|---|
| P0 | Quantization/repair reference and calibration with complete-M context | Preserve real global scales, supports, and rounding; pass an independent numerical reference |
| P1 | Repair-aware first-pass quantization and sensitive-module fallback | Beat RTN and a strong single-pass baseline at equal budget; pass complete-trajectory and multimodal checks |
| P2 | Map structure to repair benefit | Beat uniform, random, mass-only, and energy-only policies; ablate layer/phase/head/modality contributions |
| P3 | Higher-quality attention-repair candidates | Separate QK/PV; preserve joint softmax and validity masks; enter the recommended path only after quality/speed validation |
| P4 | Frozen-policy export and deployable assets | Prune W1 that is never used; unify model revision, scales, budget, and launch formats |
| P5 | Unseen-scene, device, and energy validation | Freeze the algorithm before testing unseen scenes, more seeds, action ground truth, and AV synchronization; record J/request |

Further system work can examine in-kernel prefetching and division of execution work, plus epilogue/packing integration. However, global amax must finish before the current exact scale is usable; fusion must not silently replace it with a local scale. Cross-device backends require actual validation. Removing the SM120 guard does not establish compatibility.

The demo should contain three coordinated areas: fixed inputs/seed/task and quality tier on the left; side-by-side BF16, uniform-repair, and frozen-automatic-policy videos with corresponding actions/audio in the center; and request time, peak allocated memory, actual budget, and phase progress on the right. Clearly label tasks where the automatic policy falls back to uniform repair.

A research view should show layer × step budgets, normalized modality relationships, and failure cases. Cross-step value reuse and double buffering should remain unavailable by default. Only configurations that pass fresh-trajectory validation should enter the recommended menu. Generation/sampling tools and traceable media already exist, but the integrated demo UI, real robot task execution, and energy panel remain unfinished.

## 10. Evidence, Reproducibility, and Claim Boundaries

Pinned model revisions are Edge `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` and Nano `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`. Their Generators currently have 28/36 layers and 16/32 Q-heads, respectively, with 8 KV-heads each. Edge has no sound path. Measurements use the local RTX 5090, Torch 2.11.0+cu130, Triton 3.6, and CUDA 12.9 tools. Complete requests depend on a pinned Diffusers revision.

| Evidence package | Purpose | Interpretation scope |
|---|---|---|
| [Original 5090 operator study](../../results/rtx5090_20260909/validation.json) | 60 main configurations, local FFNs, profiling, numerical regression | Captured weights/inputs and constructed performance shapes; not whole-model quality |
| [All-step findings and ablations](../../results/findings_20260910/README.md) | 28 observation trajectories, 170 ablation trajectories, all-frame LPIPS, 34 audio decodes | Limited scene clusters; action lacks matched ground truth |
| [Automatic budgets](../../results/auto_budget_20260910/README.md) | 139 complete comparisons, quality gates, 60 operators, hardware metrics | Exploratory/validation selection with uniform fallback |
| [Graph / fusion execution](../../results/execution_20260911/README.md) | Three-path equivalence on 120 trajectories, 20 FFNs, chunking and tuning failures | New seeds, existing scenes, requests after warmup; excludes the complete application loop |

Internal evidence links point to original repository records. External specifications and papers provide motivation and positioning; their performance figures are not ThunderWorld results. The AI-generated figure is a conceptual architecture illustration, not a profiling plot. All measured values come from archived experiments. Data extracts, source hashes, generation prompts, and editable Markdown accompany the report so future updates need not rewrite earlier cohorts.

Supported claims are that both models have native FP4 FFN repair and executable budgets; residual benefits vary by model and modality; execution optimizations preserve numerical equivalence on the tested trajectories; and Graphs provide measured request-time gains after warmup. The evidence does not yet support claiming a completed iterFP4 quantization algorithm, universally optimal structural policies, near-BF16 quality across all modules, saturated Tensor Cores, reduced energy use, or equivalent gains on Thor/Spark/Rubin.

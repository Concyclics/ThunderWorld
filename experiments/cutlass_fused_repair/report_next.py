"""Generate the source-backed engineering report and before/after GPU-work chart."""
import json,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs5090/next_tuning_20260914';OUT=ROOT/'results/next_tuning_5090_20260914'
def read(p):return json.loads(p.read_text())
def main():
    summary=read(OUT/'summary.json');rows={r['name']:r for r in summary['rows']}
    before=read(ROOT/'results/ar_cache_5090_20260914/optimized_breakdown.json');after=read(OUT/'breakdown.json')
    components=read(OUT/'component_experiments.json');pilot=read(RUN/'pilot_joint125/result.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cats=[];totals=[]
    for value in (before,after):
        c=Counter()
        for r in value['categories']:c[r['category']]+=r['kernel_ms']
        totals.append(c)
    cats=sorted(set(totals[0])|set(totals[1]),key=lambda c:-totals[0][c])
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained');y=list(range(len(cats)))
    ax.barh([i-.18 for i in y],[totals[0][c] for c in cats],height=.35,label='AR-cached baseline',color='#94a3b8')
    ax.barh([i+.18 for i in y],[totals[1][c] for c in cats],height=.35,label='Tuned exact pipeline',color='#2563eb')
    ax.set_yticks(y,cats);ax.invert_yaxis();ax.set_xlabel('Summed GPU kernel time (ms; separate instrumented requests)');ax.legend();ax.spines[['top','right']].set_visible(False)
    ax.set_title('Remaining work: Cosmos3-Edge / FP4 budget 1.25 / RTX 5090')
    for ext in ('png','svg'):fig.savefig(OUT/f'work_comparison.{ext}',dpi=180,bbox_inches='tight')
    perf='\n'.join(f"| {label} | {rows[name]['median_seconds']:.4f} | {rows[name]['p90_seconds']:.4f} | {rows[name]['peak_allocated_bytes']/2**30:.3f} | {rows[name]['same_precision_speedup']:.3f}× | {rows[name]['fair_bf16_speedup']:.3f}× |" for name,label in [('bf16_ar','BF16 + AR cache'),('bf16_best','BF16 tuned'),('joint125_ar','FP4 1.25 + AR cache'),('joint125_best','FP4 1.25 tuned'),('cutlass150_ar','FP4 1.50 + AR cache'),('cutlass150_best','FP4 1.50 tuned')])
    stage_rows={r['stage']:r for r in after['stages']}
    stages='\n'.join(f"| {s} | {r['kernel_ms']:.2f} | {r['kernels']:,} |" for s,r in stage_rows.items())
    kernels='\n'.join(f"| {c} | {totals[0][c]:.2f} | {totals[1][c]:.2f} |" for c in cats)
    ablation='\n'.join(f"| {name} | {statistics.median(r['wall_seconds'] for r in row['rows']):.4f} | Yes |" for name,row in pilot['variants'].items())
    variants='\n'.join(f"| {r['variant']} | {r['timing']['median_ms']:.2f} | {r['comparisons'][0]['normalized_rmse']*100:.3f}% | {r['comparisons'][0]['max_abs']:.6f} | {'Exact' if all(c['exact'] for c in r['comparisons']) else 'Excluded from exact preset'} |" for r in components['codec'])
    best=rows['joint125_best'];anchor=rows['joint125_ar'];bf16=rows['bf16_best']
    doc=f'''# RTX 5090: next-stage inference tuning, items 1–4

**The exact-output FP4 1.25 pipeline takes {best['median_seconds']:.4f} s/request, {best['same_precision_speedup']:.3f}× faster than the AR-cached pipeline and {best['fair_bf16_speedup']:.3f}× faster than equally optimized BF16.** These are complete resident requests, not GEMM-only timings.

The work covers conditioning, codec graphs/layout, DM norm/activation/RoPE fusion, and static rotary/timestep work. New optimizations are opt-in. Decoder layout changes and timestep broadcasting were tested separately and excluded from the exact-output preset.

## Measured end-to-end result

![Full request latency and memory](../results/next_tuning_5090_20260914/pipeline_comparison.png)

| Configuration | Median s | P90 s | Peak allocated GiB | Speedup vs same precision + AR | Speedup vs equally configured BF16 |
|---|---:|---:|---:|---:|---:|
{perf}

Five randomized rounds, six isolated process configurations, three seeds per process: **90 measured requests**. Cosmos3-Edge, RTX 5090, 192×320, 61 frames, 35 denoising steps, two CFG branches, CFG 6; CPU threads 8, TF32 disabled. Both FP4 compute budgets keep their original calibrated assets, repair plans, and frozen GEMM launches. BF16 receives all applicable shared optimizations; calibrated FP4 packing fusion has no BF16 modules to patch.

Timing includes host request setup, current conditioning, fresh AR prefill for both branches, all denoising steps, decoding and output conversion. Loading, offline preparation, graph capture and saving files are outside resident latency. Every process performs a complete warm request; profiling is disabled during measurement. Source, model, input, environment, asset, repair configuration and observed launch identities are checked. Each configuration has 15 samples.

FP4 1.25 paired round speedup versus its AR baseline: {best['same_precision_paired']['median']:.3f}×, bootstrap 95% interval [{best['same_precision_paired']['ci95'][0]:.3f}, {best['same_precision_paired']['ci95'][1]:.3f}]. The interval uses five round-level ratios and is descriptive, not a broad workload generalization.

## What changed

Items 1–4 follow the priority table in the AR-cache report: conditioning encoder, DM pointwise chains, VAE dispatch, and VAE convolution/layout. Calibrated FFN fusion and rotary-table reuse were also validated.

```mermaid
flowchart LR
  I[Current input image] --> E[Encode first causal frame]
  E --> L[Full latent shape + fresh noise]
  P[Current prompt + positions] --> AR[AR prefill: 2 CFG branches]
  P --> R[Refresh full rotary tables]
  L --> DM[35 steps × 2 DM graph replays]
  AR --> DM
  R --> DM
  DM --> D[Whole causal decoder CUDA Graph]
  D --> O[Current generated video]
```

| Item | Implementation | Numerical contract |
|---|---|---|
| 1. Conditioning | Encode one image frame; fill unused latent slots with zeros before the existing conditioning mask/noise construction | First causal latent unchanged; full noise shape and RNG order preserved. Guarded single-image I2V only |
| 2a. RMSNorm | Fuse cast/square, retain PyTorch FP32 mean, fuse rsqrt/multiply/weight/cast | Preserve reduction order; three kernels instead of the surrounding unfused chain |
| 2b. RoPE | Fuse rotation, two products and addition | Explicit BF16 product rounding and disabled FMA preserve the reference expression |
| 3. Codec graphs | Capture complete causal encode/decode loops; copy current input into static buffers and return owned outputs | Preserve tiling, causal feature-cache reconstruction and exact output. Shape, stream and tiling settings belong to the cache key |
| 4. Codec layout trials | Measured Conv3d channels-last weights and untiled decoding | Faster component trials, but changed raw decoder outputs; excluded from the exact preset |
| Extra: calibrated FFN | Fuse ReLU² with calibrated global-amax partial production; reuse the factor in native/CUTLASS packing | Amax uses BF16(activation × per-channel prescale). Original global factor reduction, A0/A1 quantization, sparse selection and W0/W1 GEMM remain intact |
| Extra: static work | Refresh exact full rotary cos/sin tables once per current request/CFG branch, outside DM graphs | Current position IDs refill stable buffers. No previous-step DM output reuse |

```mermaid
flowchart LR
  X[BF16 DM state] --> N[Fused RMSNorm pointwise stages + original FP32 mean]
  N --> Q[Activation FP4 packing]
  Q --> G[FP4 GEMM + selected repair]
  G --> RO[Fused BF16 RoPE]
  RO --> A[Attention / softmax / selected repair]
  A --> U[FFN up projection]
  U --> F[Fused ReLU² + prescaled amax partials]
  F --> M[Original global factor reduction]
  M --> PK[Calibrated A0/A1 packing]
  PK --> DOWN[FFN down GEMM + selected W1/A1 repair]
```

The initial cumulative pilot below uses two seeds per fresh runner. It established exactness and direction; these timings are **exploratory**, not randomized attribution of each small optimization. `first`, `codec`, `norm`, `pointwise`, and `rotary` add changes in that order. Calibrated FFN fusion was added after this pilot; the formal `tuned` result includes it.

| Cumulative pilot | Median s | Both reference videos exact |
|---|---:|---|
{ablation}

## Correctness and quality scope

- 22 GPU kernel/graph tests passed, including BF16 norm/RoPE/ReLU², nonidentity prescales, zero inputs, and changed-input graph ownership. CPU regression: 50 passed; 87 GPU-dependent cases skipped in that CPU-only invocation.
- All three configurations passed fresh control/candidate comparisons for two seeds, a changed same-length prompt, a changed conditioning image, and restored original input: 30 full qualification requests.
- All 90 formal requests preserve their respective precision/budget output hashes; distinct seeds remain distinct. The optimized profiler request also matches its uninstrumented reference.
- This proves preservation of the previously evaluated FP4 paths on these requests. It does **not** assert that FP4 is bitwise equal to BF16, establish new perceptual quality across a dataset, or generalize the timings to Nano/AV/action/other resolutions.

## Workflow breakdown and remaining cost

![GPU work before and after](../results/next_tuning_5090_20260914/work_comparison.png)

| Kernel category | AR baseline ms | Tuned ms |
|---|---:|---:|
{kernels}

| Tuned stage | Summed kernel ms | Kernel count |
|---|---:|---:|
{stages}

Kernel launches fall from **{before['kernel_count']:,} to {after['kernel_count']:,}**. There are 74 CUDA Graph replays: 70 DM, two fresh AR prefills, one conditioning encoder, and one decoder. The decoder accounts for {stage_rows['video_decode']['kernel_ms']/after['kernel_sum_ms']*100:.1f}% of summed kernel time; FP4 GEMM accounts for {totals[1]['FP4 GEMM incl. fused repair']/after['kernel_sum_ms']*100:.1f}%. FP4 GEMM work remains effectively unchanged; the gains come from the surrounding pipeline. These breakdowns come from separate CUPTI-instrumented requests. Their summed GPU durations are diagnostic, and must not be added to CPU durations or substituted for the uninstrumented latency table. Fusion moves some work between categories: for example, norm finish now includes multiplications/casts formerly counted as generic pointwise work. Category deltas therefore are not standalone operation speedups. Fused calibrated amax is counted with nonlinear activation work; it is not counted again under quantization. FP4 GEMM includes fused repair, and attention includes its internal softmax/repair. The trace does not establish Tensor Core saturation.

BF16/FP32 still serves required work: BF16 codec convolutions and AR prefill, FP32 timestep MLP and normalization reductions, BF16 residual/nonlinear interfaces, scales and higher-precision accumulators. FP4 acceleration applies to eligible dense products and selected correction terms; it does not make normalization, softmax, codec or memory movement FP4 operations.

## Candidates tested but not enabled

The raw encoder moment tensor's first frame is bitwise identical between full 61-frame and one-frame encoding. Component median: {components['first_frame']['full']['median_ms']:.2f} → {components['first_frame']['first']['median_ms']:.2f} ms. These host-synchronized component timings explain the opportunity; full-request measurements above establish the actual gain.

Decoder trials use the real FP4 1.25 latent and a modified latent, five measured repetitions after warmup. Changes below are relative to the original tiled decoder. A difference is not automatically a quality degradation: untiled decoding uses different spatial context. It does need explicit quality evaluation before replacing the reference.

| Decoder trial | Median ms | Original-latent normalized RMSE | Max absolute raw-output difference | Decision |
|---|---:|---:|---:|---|
{variants}

Channels-last here converts **Conv3d weights only**; it is not an end-to-end persistent-layout implementation. Component results do not imply an equal end-to-end speedup, nor measure combination with codec graphs. Raw decoder errors are measured before final output clipping/format conversion; they are not perceptual metrics.

All 35 observed timestep arrays are uniform. Computing one embedding and broadcasting changes BF16 values in **9/35 steps** (maximum absolute difference 0.0001220703125). Changing GEMM shape changes rounding. This candidate is excluded from the exact preset despite reduced component cost. A shape-preserving table precomputed from the full schedule would be another route, but its per-request preparation cost must be included.

## Where further work is worthwhile

1. **Decoder data movement and convolution scheduling.** Preserve layout through padding, causal-cache updates, normalization and upsampling. Fuse halo/cache gathering or use stable causal buffers to reduce concatenation/copy traffic. The tested untiled path is a promising quality-gated alternative; avoid assuming tile removal is an exact implementation change.
2. **Reduce norm-to-quantization traffic.** Current fusion retains the original mean reduction for exactness. A combined normalization/packing producer or a reduction implementation matching the existing summation order may save more memory passes. Global activation scaling still creates a real reduction dependency.
3. **Attention packing and static AR contributions.** Static K work may be reusable, but V quantization tiles can straddle static AR and changing DM tokens; cached scales must not include stale DM values. Preserve per-projection calibration when considering shared Q/K/V packing.
4. **Overlap only independent work.** Quantization → GEMM and FFN up → nonlinear → down are dependencies. Separate streams cannot remove them. Candidate overlap should target independent projections or shape-aware producer/consumer tiles, then measure SM/register/shared-memory/bandwidth competition with counters. This turn does not claim a new overlap speedup or peak Tensor Core utilization.

The large improvements here come from removing unused work and memory/launch overhead. The tested pipeline is faster, but the remaining decoder and packing work means it is not at an established hardware limit.

## Reproduce and hand off

New opt-in flags, installed **before warmup**:

```text
--ar-cache --first-frame --codec-graph --fuse-norm
--fuse-pointwise --static-rotary --fuse-calibrated
```

Use the same flags for BF16 and FP4. Keep `--config`, prepared assets, repair plan and frozen tuning profile fixed. `--fuse-pointwise` currently requires `--ar-cache`; first-frame and rotary reuse are guarded for the runner's I2V flow. Norm fusion targets Edge's Nemotron RMSNorm. Captured instances must be recreated after weight changes; requests are serial. CUDA 12.9 / the existing local runtime were sufficient; no toolkit update was required.

```bash
PYTHONPATH="$PWD/data_local/cutlass_diffusers_20260914/src" \\
THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python" \\
bash tools/kernel_handoff/python5090.sh -m thunderworld_fp4.cosmos benchmark \\
  --model models/Cosmos3-Edge-a9d944e \\
  --config runs5090/joint_e2e_20260914/configs/joint125.json \\
  --tuning runs5090/joint_e2e_20260914/profiles/joint125.json \\
  --assets data_local/cutlass5090_rebuilt_20260914/i2v_b125 \\
  --case tools/joint_inference/cases/robot_observe.json \\
  --steps 35 --seeds 93001 93002 93003 --output runs5090/my_tuned_run \\
  --ar-cache --first-frame --codec-graph --fuse-norm \\
  --fuse-pointwise --static-rotary --fuse-calibrated
```

Implementation: [pipeline shortcuts](../packages/fp4-repair/src/thunderworld_fp4/cosmos/pipeline_tuning.py), [calibrated fusion](../packages/fp4-repair/src/thunderworld_fp4/cosmos/calibrated_fusion.py), [norm fusion](../packages/fp4-repair/src/thunderworld_fp4/cosmos/norm_fusion.py), [codec graphs](../packages/fp4-repair/src/thunderworld_fp4/cosmos/codec_graph.py), [runner](../packages/fp4-repair/src/thunderworld_fp4/cosmos/runner.py).

Evidence: [formal summary](../results/next_tuning_5090_20260914/summary.json), [validation](../results/next_tuning_5090_20260914/validation.json.xz), [full receipts](../results/next_tuning_5090_20260914/receipts.json.xz), [component experiments](../results/next_tuning_5090_20260914/component_experiments.json), [GPU breakdown](../results/next_tuning_5090_20260914/breakdown.json), [compressed trace](../results/next_tuning_5090_20260914/trace.json.xz). Scripts: [qualification](../experiments/cutlass_fused_repair/next_tuning_probe.py), [benchmark](../experiments/cutlass_fused_repair/next_benchmark.py), [component trials](../experiments/cutlass_fused_repair/next_components.py), [strict summary](../experiments/cutlass_fused_repair/summarize_next.py).
'''
    (ROOT/'docs/NEXT_PIPELINE_TUNING_5090_20260914.md').write_text(doc)
if __name__=='__main__':main()

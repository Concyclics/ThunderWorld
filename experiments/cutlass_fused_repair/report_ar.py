"""Write the AR result and whole-pipeline optimization report from verified artifacts."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('evidence',type=Path);a=p.parse_args()
    summary=json.loads((a.run/'performance/summary.json').read_text());assert summary['complete']
    comparison=json.loads((a.evidence/'trace_comparison.json').read_text());assert comparison['complete']
    audit=json.loads((a.run/'audit_joint125/result.json').read_text());assert audit['all_bitwise_equal']
    rows={r['name']:r for r in summary['rows']};names=['bf16_original','bf16_ar','joint125_original','joint125_ar','cutlass150_original','cutlass150_ar']
    labels=['BF16','BF16 + AR cache','FP4 1.25','FP4 1.25 + AR cache','FP4 1.50','FP4 1.50 + AR cache']
    table='\n'.join(f"| {label} | {rows[n]['median_seconds']:.3f} | {rows[n]['p90_seconds']:.3f} | {rows[n]['same_precision_speedup']:.3f}× | {rows[n]['fair_bf16_speedup']:.3f}× | {rows[n]['peak_allocated_bytes']/2**30:.3f} |" for n,label in zip(names,labels))
    timing='\n'.join(f"| {base} | {rows[base+'_ar']['same_precision_paired']['median']:.3f}× | {rows[base+'_ar']['same_precision_paired']['ci95'][0]:.3f}–{rows[base+'_ar']['same_precision_paired']['ci95'][1]:.3f}× |" for base in ('bf16','joint125','cutlass150'))
    cat='\n'.join(f"| {r['category']} | {r['before_ms']:.1f} | {r['after_ms']:.1f} | {r['before_count']:,} → {r['after_count']:,} |" for r in comparison['rows'])
    stages='\n'.join(f"| {r['stage']} | {r['kernel_ms']:.1f} | {r['kernels']:,} |" for r in comparison['after_stages'])
    j=rows['joint125_ar'];old=rows['joint125_original'];ratio=rows['bf16_original']['median_seconds']/j['median_seconds']
    memdelta=(j['peak_allocated_bytes']-old['peak_allocated_bytes'])/2**20
    path=ROOT/'docs/AR_CACHE_AND_PIPELINE_5090_20260914.md'
    doc=f'''# AR caching tested; full inference pipeline and next optimizations

**AR caching is validated for this Cosmos3-Edge I2V workload.** Budget-1.25 FP4 complete-request median changed from **{old['median_seconds']:.3f} s to {j['median_seconds']:.3f} s ({j['same_precision_speedup']:.3f}×)**. Outputs match bit for bit. Peak allocated memory changed by **{memdelta:+.1f} MiB**.

The optimized FP4 path is **{ratio:.3f}×** faster than the original BF16 pipeline. With AR caching applied equally to BF16, the FP4 comparison is **{j['fair_bf16_speedup']:.3f}×**. These are distinct comparisons; the second isolates FP4 more fairly.

## Complete-request measurements

![AR cache latency and memory](../results/ar_cache_5090_20260914/ar_comparison.png)

| Configuration | Median seconds | P90 seconds | Speedup vs same-precision original | Speedup vs equally optimized BF16 | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
{table}

Five isolated process rounds × three seeds per configuration, randomized configuration order (seed 914), **90 measured requests**. Each process loads identical local assets and strict frozen launch profiles, warms the full request, and runs without diagnostic hooks. The measured interval includes host request setup, image conditioning, **fresh AR prefill for both CFG branches**, all 35 denoising steps, VAE decode, and output conversion. It excludes model loading, offline quantization, graph capture, and output-file serialization.

Paired round estimates use each round's median of the three same-seed speed ratios. The 95% intervals bootstrap the five rounds (10,000 resamples); five rounds remain a small sample.

| Precision / budget | Paired median AR-cache speedup | Bootstrap 95% interval |
|---|---:|---:|
{timing}

## Correctness and implementation

- The reference eager audit checked **{audit['counts']['layer_output']:,} AR layer outputs**, **{audit['counts']['k_before_rope']:,} DM-facing keys before RoPE**, and **{audit['counts']['v']:,} value tensors** against branch-specific prefills across all 35 steps. Every comparison was bitwise equal. Full position IDs and text IDs also remained unchanged within each branch. Its video matched the prior frozen graph reference.
- Each of BF16, FP4 budget 1.25, and FP4 budget 1.50 passed a fresh-runner baseline/candidate test on seeds 7103 and 7104, a same-token-length changed prompt (“robot arm” → “human arm”), then the restored original prompt. Changed seeds/prompts produce different outputs; restoring the original reproduces its output.
- All three benchmark seeds (93001–93003) matched between each original/cache pair across every round. Source, model, environment, input, calibrated assets, repair configuration, and selected DM launches were audited.
- AR caching does **not** reduce the FP4 repair budget or substitute an old DM output. It reuses the invariant text-only causal prefix. FP4/BF16 output differences from quantization remain those of the existing configurations; cache equality is not a new claim of BF16-equivalent FP4 quality.

The implementation performs AR prefill once per request and CFG branch using two small CUDA Graphs, writing stable per-layer AR K/V/output buffers. The 70 existing Transformer graphs then run the DM layers against those buffers. Current prefix inputs are copied before each prefill replay; no prompt computation is reused across requests. Cache keys separate branches, shapes, and streams. Unwarmed shapes, non-I2V tasks, or modality tokens inside the AR prefix are rejected. The runner is a serial, resident inference interface; model weights must stay fixed.

Use `--ar-cache` on the Cosmos CLI. It is opt-in. Validation covers **Edge, this I2V input/shape, and the tested prompt changes**; Nano, AV, Policy, dynamic AR generation, and other devices have no new validation here.

## Full pipeline after AR caching

Before resident inference, model loading, prepared FP4 weight/residual loading, weight-layout preparation, launch-profile validation, and CUDA Graph capture run once. These cold-start costs remain in the receipts and are excluded from the request-latency comparison. AR caching adds prefill graph capture during warmup; its graph replay and fresh-input copies are included in every measured request.

```mermaid
flowchart TD
    A["Request setup: image / prompt / seeded noise"] --> B["VAE conditioning encode · BF16 convolution"]
    B --> C["AR prefill · once per request / CFG branch<br/>BF16 AR layers → cached layer K/V"]
    C --> D["35 denoising steps × 2 CFG branches"]
    D --> E["Vision input projection + timestep embedding + position math"]
    E --> F["DM layer: RMSNorm → Q/K/V projection<br/>BF16 stream → activation FP4 packing → FP4 GEMM + repair"]
    F --> G["RoPE + joint AR/DM attention<br/>FP4 QK/PV + selected repair; FP32 softmax statistics"]
    G --> H["Output projection + residual + RMSNorm"]
    H --> I["FFN up GEMM → ReLU² / calibrated quantization<br/>FFN down GEMM + residual"]
    I --> J["Vision output projection → CFG → scheduler update"]
    J --> D
    D --> K["After final step: VAE decode · BF16 convolution"]
    K --> L["Video output conversion / host transfer"]
    C -. cached K/V .-> G
```

The diagram is schematic: quantization happens at each relevant linear/attention boundary. FP4 MMA uses FP32 accumulation; normalization, residual formation, and attention statistics retain higher precision. AR computation has moved to prefill, while VAE convolution and the Transformer input/output adapters remain higher precision.

## Where GPU work remains

![Kernel work before and after AR caching](../results/ar_cache_5090_20260914/ar_kernel_breakdown.png)

| Kernel category | Original ms | AR-cache ms, including prefill | Kernel count |
|---|---:|---:|---:|
{cat}

| Optimized stage | Summed kernel ms | Kernel count |
|---|---:|---:|
{stages}

Total kernel count changed from **{comparison['before_kernel_count']:,} to {comparison['after_kernel_count']:,}**. CUDA Graph launches changed from **{comparison['before_graph_launches']} to {comparison['after_graph_launches']}**, reflecting two fresh AR prefills alongside the 70 DM graph calls. The number of FP4 linear GEMMs remains **11,760**: AR caching leaves DM base/repair dispatch intact.

These are **separate instrumented CUPTI requests**, useful for attribution and work counts, not formal latency ratios or utilization measurements. CPU and GPU times overlap. Category names group generic casts/elementwise operations conservatively; fused kernels are not artificially split. The AR-prefill annotation nests within the Transformer wrapper, and the analyzer assigns each kernel to its innermost containing annotation to avoid double counting.

## Next optimization points, in priority order

| Priority | Pipeline point | Proposed change | What must be preserved / measured |
|---|---|---|---|
| 1 | I2V conditioning encoder | Encode only the first conditioning frame rather than the repeated 61-frame clip whose other encoded latents are discarded | First latent equality, full noise shape/RNG order, masks, final video; generalize separately for multi-frame conditioning |
| 2 | DM RMSNorm, RoPE, residual chains | Fuse casts and pointwise operations; avoid materializing `_rotate_half` concatenations | BF16 rounding and FP32 reduction order; full-trajectory comparison |
| 3 | VAE dispatch | Finish validation of whole-codec CUDA Graphs; earlier component pilot showed a gain | Full-request output and retained graph memory; apply equally to BF16 |
| 4 | VAE convolution/layout | Shape-aware tiling threshold and a sustained convolution-compatible layout | Decoder overlap processes 1.20× latent input area here; quality and peak memory may change when tiling changes |
| 5 | Calibrated FFN activation quantization | Fuse ReLU², nonidentity prescaling, and amax; reduce repeated reads before packing | Existing fusion is skipped for these calibrated prescales; preserve calibration and residual scaling |
| 6 | Remaining per-step setup | Cache static rotary tables; consider computing repeated timestep embeddings once then broadcasting | Preserve exact positions; changing GEMM shape can change rounding and must be tested |
| 7 | Remaining GEMM/attention shapes | Tune buffer stages / tile sizes and overlap only independent work | Profile occupancy and bandwidth counters first; dependency-bound quantization → GEMM cannot simply overlap |

Cached AR K/V still participates in each step’s joint attention concatenation and packing. Reusing packed AR data would require respecting quantization block boundaries, including any block shared with changing DM tokens; this is a separate optimization.

The earlier codec and RMSNorm candidates remain separate, opt-in experiments. **This AR benchmark enables neither of them.** The first-frame encoder, new RoPE fusion, layout changes, and timestep hoisting are proposals rather than implemented/validated gains in this result.

The optimized trace places roughly half of active kernel time in the VAE encoder/decoder together. AR prefill is now a small per-request cost; the remaining DM and codec work dominate.

The next large gains should come from deleting unused conditioning work and avoiding intermediate tensor traffic. Optimizing the already small repair-selector cost has a lower measured priority. No “GPU saturated” or “maximum possible speed” claim follows from this trace.

## Reproduction and evidence

```bash
cd /home/chenhan/ThunderWorldDev/ThunderWorld
export PYTHONPATH="$PWD/data_local/cutlass_diffusers_20260914/src"
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
export MPLCONFIGDIR=/tmp/tw_ar_mpl

bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/ar_probe.py \\
  --mode audit --name joint125 --output runs5090/ar_recheck/audit_joint125
bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/ar_suite.py \\
  --output runs5090/ar_recheck
```

Use fresh output directories. The suite gates formal timing on all three changed-input pilots, then produces the optimized trace. All GPU jobs run sequentially on the local RTX 5090. Existing frozen profiles/assets are reused; no toolkit update is needed.

- [Formal benchmark summary](../results/ar_cache_5090_20260914/summary.json), [trace comparison](../results/ar_cache_5090_20260914/trace_comparison.json), [validation evidence](../results/ar_cache_5090_20260914/validation.json.xz).
- [AR cache implementation](../packages/fp4-repair/src/thunderworld_fp4/cosmos/ar_cache.py), [audit/pilot](../experiments/cutlass_fused_repair/ar_probe.py), [benchmark audit and statistics](../experiments/cutlass_fused_repair/summarize_ar.py).
- [Earlier source-level pipeline analysis](PIPELINE_COST_CAUSES_5090_20260914.md), [original full-request profile](PIPELINE_OVERHEAD_5090_20260914.md).

Workload: Cosmos3-Edge, robot-observe I2V, 192×320, 61 frames, 24 fps, 35 steps, CFG 6, two branches; Torch threads 8, TF32 disabled, pinned Diffusers and local CUDA/CUTLASS artifacts. Quality scope is output equivalence to each configuration's reference, not a held-out perceptual-quality certification or an energy measurement.
'''
    path.write_text(doc);print(path)


if __name__=='__main__':main()

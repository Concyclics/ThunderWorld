"""Archive original/tuned precision controls and create an evidence-backed comparison."""
import hashlib,json,lzma,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from baseline_compare import ROOT,OUT,NAMES,read
from summarize_baselines import LABELS
DEST=ROOT/'results/baseline_compare_5090_20260914'

def main():
    DEST.mkdir(parents=True,exist_ok=True);summary=read(OUT/'summary.json');quality=read(OUT/'quality.json');gate=read(OUT/'qualification_summary.json');matrix=read(OUT/'performance_matrix.json')
    assert all(r['complete'] for r in (summary,quality,gate,matrix));by={r['name']:r for r in summary['rows']}
    raw={}
    for job in gate['jobs']+matrix['jobs']:raw[Path(job['output']).name]=read(Path(job['output'])/'result.json')
    raw['qualification_fp4_cutlass']=read(OUT/'qualification_fp4_cutlass/result.json')
    for name in ('prepare_fp8','prepare_fp4','tune_fp4','tune_fp4_cutlass'):raw[name]=read(OUT/name/'result.json')
    (DEST/'receipts.json.xz').write_bytes(lzma.compress(json.dumps(raw,separators=(',',':')).encode(),preset=3))
    for f in ('summary.json','quality.json','qualification_summary.json','performance_matrix.json','runtime_metadata.json','telemetry_spot_check.json'):shutil.copyfile(OUT/f,DEST/f)
    (DEST/'configs.json').write_text(json.dumps({p.stem:read(p) for p in sorted((OUT/'configs').glob('*.json'))},indent=2)+'\n')
    (DEST/'profiles.json.xz').write_bytes(lzma.compress(json.dumps({p.stem:read(p) for p in sorted((OUT/'profiles').glob('*.json'))},separators=(',',':')).encode(),preset=3))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#94a3b8','#a78bfa','#fdba74','#475569','#7c3aed','#ea580c','#2563eb','#0f766e']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    labels=[LABELS[n].replace(' (no repair)','\n(no repair)') for n in NAMES]
    for ax,key,title,factor in [(axes[0],'median_seconds','Complete request latency (s)',1),(axes[1],'peak_allocated_bytes','Peak allocated memory (GiB)',2**30)]:
        vals=[by[n][key]/factor for n in NAMES];ax.barh(labels,vals,color=colors);ax.invert_yaxis();ax.set_title(title);ax.set_xlim(0,max(vals)*1.17)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.01,i,f'{v:.3f}',va='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('RTX 5090 · Cosmos3-Edge I2V · 61 frames · 192×320 · 35 steps\nOriginal baselines already use transformer CUDA Graphs · 15 requests per configuration')
    for ext in ('png','svg'):fig.savefig(DEST/f'latency_memory.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    controls=NAMES[:6];candidates=['thunder125','thunder150'];ratios=np.array([[by[n]['speedups'][c]['ratio_of_medians'] for c in controls] for n in candidates])
    fig,ax=plt.subplots(figsize=(12,3.2),layout='constrained');im=ax.imshow(ratios,cmap='RdYlGn',vmin=.6,vmax=2.);ax.set_xticks(range(6),[LABELS[n].replace(' (no repair)','\n(no repair)') for n in controls],rotation=15,ha='right');ax.set_yticks(range(2),[LABELS[n] for n in candidates])
    for i in range(2):
        for j in range(6):ax.text(j,i,f'{ratios[i,j]:.2f}×',ha='center',va='center',fontsize=13)
    ax.set_title('ThunderWorld speedup = control median / ThunderWorld median\nAbove 1× is faster; below 1× is slower')
    for ext in ('png','svg'):fig.savefig(DEST/f'speedup_matrix.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for name,color in zip(NAMES,colors):
        family={'bf16_tuned':'bf16_original','fp8_tuned':'fp8_original','fp4_tuned':'fp4_original'}.get(name,name)
        lpips=0. if family=='bf16_original' else quality['means'][family]['lpips']
        ax.scatter(by[name]['median_seconds'],lpips,color=color,s=85,label=LABELS[name],marker='s' if name.startswith('thunder') else 'o')
    ax.set_xlabel('Full-request median seconds (lower is faster)');ax.set_ylabel('Mean LPIPS to paired BF16 (lower is closer)');ax.set_title('Speed / BF16 consistency · two development seeds\nNot a held-out task-quality evaluation');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    for ext in ('png','svg'):fig.savefig(DEST/f'speed_quality.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    table='\n'.join(f"| {LABELS[n]} | {by[n]['median_seconds']:.4f} | {by[n]['p90_seconds']:.4f} | {by[n]['peak_allocated_bytes']/2**30:.3f} |" for n in NAMES)
    speed='\n'.join('| '+LABELS[n]+' | '+' | '.join(f"{by[n]['speedups'][c]['ratio_of_medians']:.3f}×" for c in controls)+' |' for n in candidates)
    qtable='\n'.join(f"| {LABELS[n]} | {quality['means'][n]['lpips']:.4f} | {quality['means'][n]['ssim_mean']:.4f} | {quality['means'][n]['relative_l2']:.4f} |" for n in ('fp8_original','fp4_original','thunder125','thunder150'))
    head=by['thunder125'];best=summary['fp4_backend']
    qfp8=quality['means']['fp8_original']['lpips'];qfp4=quality['means']['fp4_original']['lpips'];qtw=quality['means']['thunder125']['lpips']
    doc=f'''# ThunderWorld versus original BF16, FP8 and FP4 pipelines

**On this RTX 5090 Cosmos3-Edge I2V workload, ThunderWorld budget 1.25 takes {head['median_seconds']:.4f} s/request:** {head['speedups']['bf16_original']['ratio_of_medians']:.3f}× faster than original BF16, {head['speedups']['fp8_original']['ratio_of_medians']:.3f}× faster than original FP8, and {head['speedups']['fp4_original']['ratio_of_medians']:.3f}× faster than original unrepaired FP4.

The comparison below also gives every baseline the shared pipeline optimizations. This distinguishes an improved inference pipeline from the cost/quality tradeoff of adding FP4 repair. Speedup less than 1× means ThunderWorld is slower than that control.

## What “original” means here

| Path | DM FFN and Q/K/V/output projections | DM attention | Shared pipeline changes |
|---|---|---|---|
| Original BF16 | BF16 | BF16 native Flash Attention | Existing transformer CUDA Graphs and host scheduler; recent flags off |
| Original FP8 | Native E4M3 W8A8; weight per tensor, activation per token | BF16 native Flash Attention | Same original pipeline |
| Original FP4 | Ordinary min-max W4A4; A0W0 only, zero repair | Q0K0 and P0V0, zero repair | Same original pipeline; native Triton linear backend |
| Tuned BF16 / FP8 | Same respective precision and weights | Same respective attention | AR prefill cache, first-frame encode, codec graphs, norm/RoPE/activation fusion, rotary reuse |
| Tuned FP4 | Same ordinary FP4 weights and zero repair | Same FP4 zero-repair attention | Same shared changes; separately selected frozen no-repair backend |
| ThunderWorld 1.25 / 1.50 | Calibrated FP4 base plus budgeted A/W repair | Budgeted FP4 attention repair | Validated tuned pipeline and previously frozen repair plans |

All paths quantize the same eligible **168 DM projections** where applicable. AR layers, video codec and other unconverted components retain their existing higher-precision computation. FP8 is therefore a mixed FP8/BF16 baseline; it is not FP8 attention. Plain FP4 has no calibrated repair. These are the repository's baseline implementations, not a benchmark of TensorRT/ModelOpt or every vendor inference engine. Original BF16 is already graph-enabled, not an artificially slow ungraphed control.

## Full-request speed and memory

![Latency and memory](../results/baseline_compare_5090_20260914/latency_memory.png)

| Configuration | Median seconds | P90 seconds | Peak allocated GiB |
|---|---:|---:|---:|
{table}

![Speedup matrix](../results/baseline_compare_5090_20260914/speedup_matrix.png)

| ThunderWorld configuration | vs original BF16 | vs original FP8 | vs original FP4 | vs tuned BF16 | vs tuned FP8 | vs tuned FP4 |
|---|---:|---:|---:|---:|---:|---:|
{speed}

Workload: one Edge I2V prompt/image, 192×320, 61 output frames, 35 steps, two CFG branches, CFG 6. **120 timed complete resident requests**, eight configurations × five randomized isolated process rounds × three seeds (93001–93003). Every process uses a full warm request, strict frozen profiles, debug off, eight CPU threads and TF32 disabled; GPU idle checks bracket requests. Loading, offline quantization, graph capture and file serialization are excluded. Host request setup, image conditioning, fresh AR work, all denoising steps, decoding and output conversion are included. The environment/source/input hashes are identical across configurations. Clocks were not locked or changed; process-to-process variation is retained in the P90 values and paired round confidence intervals in the summary. A single read-only temperature/clock spot check is archived; it is not an energy measurement.

The tuned plain-FP4 backend was selected as **`{best}`** in a separate two-seed development pilot before formal timing. CUTLASS/native full-output equality checks: **{gate['cutlass_exact']}**. Pilot medians: native {gate['native_pilot_median_seconds']:.4f} s, CUTLASS {gate['cutlass_pilot_median_seconds']:.4f} s. CUTLASS is selected only if exact and at least 3% faster in that pilot; the formal data is not used to select its own winner. Both variants preserve zero repair. Frozen profiles and definitions are archived.

The previous tuning report used a separate run matrix. Use the freshly measured controls above for these ratios; do not multiply speedups across reports or attribute all pipeline gains to FP4 arithmetic.

## Output consistency and the repair tradeoff

**This is not yet a quality-matched win over FP8.** On the two development seeds, ThunderWorld 1.25 has LPIPS {qtw:.4f}, versus {qfp4:.4f} for plain FP4 and {qfp8:.4f} for FP8. Calibration plus repair reduces the plain-FP4 LPIPS discrepancy by {(1-qtw/qfp4)*100:.1f}%, but both repaired budgets still have higher LPIPS/relative L2 and lower SSIM than FP8 in this sample. The repair policy needs further quality work before claiming FP8-level consistency.

With shared optimizations enabled on all paths, ThunderWorld 1.25 is {head['speedups']['fp8_tuned']['ratio_of_medians']:.3f}× faster than FP8 and incurs {(head['median_seconds']/by['fp4_tuned']['median_seconds']-1)*100:.1f}% more latency than unrepaired FP4. The original-baseline speedups include substantial improvements that also benefit BF16/FP8/plain FP4. The next quality experiment should adjust calibration/repair allocation toward FP8 consistency, then remeasure this same pipeline comparison.

![Speed versus paired BF16 consistency](../results/baseline_compare_5090_20260914/speed_quality.png)

| Quantized path | LPIPS ↓ | SSIM ↑ | Relative L2 ↓ |
|---|---:|---:|---:|
{qtable}

These metrics average all frames from two **development** seeds (7103/7104) against the same-seed BF16 videos. They measure BF16 consistency, not ground-truth accuracy, perceptual preference or task success. This is not the repository's reserved eight-seed final quality gate. Original and tuned versions of each baseline preserve the same videos, so they share quality metrics. The quality comparison is computed on CPU after performance measurement and never enters the inference timer.

All baseline tuning changes passed full-video bitwise equality on both qualification seeds. All 120 timed requests pass determinism and exact original/tuned output checks within each precision family; different seeds produce distinct outputs. Both ThunderWorld configurations still match their earlier frozen trajectories. Cross-precision output equality is neither expected nor required: unrepaired FP4 does less arithmetic, and the correction paths add work. ThunderWorld also uses calibrated assets, so the quality difference versus ordinary FP4 combines calibration and repair; this comparison does not isolate their individual contributions.

## Reproduce and inspect

Use the existing local runtime; no toolkit/package update was needed. New scripts prepare native FP8 and ordinary min-max FP4 assets, tune/freeze the FP4 backends, validate trajectories, and run the comparison serially on the local GPU:

```bash
# Prefix each command below with the same local environment:
export PYTHONPATH="$PWD/data_local/cutlass_diffusers_20260914/src"
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/baseline_compare_setup.py
bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/baseline_compare.py --stage qualification
bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/baseline_compare.py --stage performance
bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/summarize_baselines.py --stage performance
bash tools/kernel_handoff/python5090.sh experiments/cutlass_fused_repair/summarize_baselines.py --stage quality
```

Scripts keep existing complete run directories on rerun. To collect an independent new cohort, change the experiment output root rather than treating reused files as fresh measurements. Per-job exact commands are in the matrix. Benchmark settings, calibrated ThunderWorld assets and plans are the same as the [pipeline tuning report](NEXT_PIPELINE_TUNING_5090_20260914.md).

Evidence: [summary and paired confidence intervals](../results/baseline_compare_5090_20260914/summary.json), [quality samples](../results/baseline_compare_5090_20260914/quality.json), [full receipts](../results/baseline_compare_5090_20260914/receipts.json.xz), [configuration definitions](../results/baseline_compare_5090_20260914/configs.json), [native FP8 binary/runtime fingerprints](../results/baseline_compare_5090_20260914/runtime_metadata.json), [matrix commands](../results/baseline_compare_5090_20260914/performance_matrix.json), [qualification](../results/baseline_compare_5090_20260914/qualification_summary.json).
'''
    (ROOT/'docs/ORIGINAL_BASELINE_COMPARISON_5090_20260914.md').write_text(doc)
    (DEST/'checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='checksums.json'},indent=2)+'\n')
if __name__=='__main__':main()

"""Paired raw-video evidence and exportable figures for the FP4 expansion demo."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from baselines import sha, write

LABELS = dict(bf16='BF16', fp8='Native FP8', awq_reference='NVFP4 AWQ W4A16',
    fp4_two_a='2 terms (A only)', fp4_two_w='2 terms (W only)',
    fp4_base='1 term (W4A4)', fp4_three='3 terms', fp4_four='4 terms (2x2)',
    fp4_six_a='6 terms (W2 x A3)', fp4_six_w='6 terms (W3 x A2)',
    fp4_nine='9 terms (3x3)', fp4_twelve_a='12 terms (W3 x A4)',
    fp4_twelve_w='12 terms (W4 x A3)', fp4_sixteen='16 terms (4x4)')
METRICS = [('lpips','LPIPS (lower is better)'),('ssim_mean','SSIM (higher is better)'),
           ('psnr_db','PSNR, dB (higher is better)'),('relative_l2','Video relative L2')]
PRIMARY = ['fp4_base','fp4_three','fp4_four','fp4_six_a','fp4_nine','fp4_twelve_a','fp4_sixteen']
SQUARE = ['fp4_base','fp4_four','fp4_nine','fp4_sixteen']


def interval(values, indices):
    v=np.asarray(values,dtype=np.float64)
    return np.quantile(v[indices].mean(1),[.025,.975]).tolist()


def collect(root):
    manifest=root/'edge/final/manifest.json';study=json.loads(manifest.read_text())
    paths=[];rows=[];by_method={}
    for name in study['methods']:
        if name=='bf16':continue
        records=[]
        for entry in study['cases']:
            path=root/'edge/final/metrics'/name/(entry['case']['id']+'.json')
            data=json.loads(path.read_text());paths.append(path)
            if not data['complete'] or data['provenance']['identity']['manifest_sha256']!=sha(manifest):
                raise ValueError('Incomplete or mismatched paired metrics')
            if len(data['rows'])!=1 or data['rows'][0]['seed']!=entry['seeds'][0]:
                raise ValueError('Expected one paired seed per frozen scene')
            row=dict(method=name,label=LABELS[name],case_id=entry['case']['id'],
                benchmark=entry['benchmark'],category=entry['category'],seed=entry['seeds'][0],
                **data['rows'][0]['metrics']['video'])
            # Exact equality is explicitly represented; never fabricate finite PSNR.
            if row['psnr_exact']:raise ValueError('Exact equality requires explicit +infinity plotting support')
            if not all(np.isfinite(row[k]) for k,_ in METRICS):raise ValueError('Nonfinite quality value')
            rows.append(row);records.append(row)
        by_method[name]=records
    rng=np.random.default_rng(study['statistics']['bootstrap_seed'])
    indices=rng.integers(0,len(study['cases']),(study['statistics']['replicates'],len(study['cases'])))
    summary={}
    for name,records in by_method.items():
        spec=study['methods'][name].get('expansion')
        prepared=json.loads((root/'edge/final/generation'/name/'prepared.json').read_text())
        summary[name]=dict(label=LABELS[name],n=len(records),expansion=spec,
            metrics={k:dict(mean=float(np.mean([r[k] for r in records])),
                ci95=interval([r[k] for r in records],indices),
                minimum=min(r[k] for r in records),maximum=max(r[k] for r in records)) for k,_ in METRICS},
            by_benchmark={b:{k:float(np.mean([r[k] for r in records if r['benchmark']==b])) for k,_ in METRICS}
                          for b in ('rbench','paibench')},
            prepared_weight_bytes=prepared['runtime']['prepared_weight_bytes'],
            packed_weight_bytes=(prepared.get('adapter') or {}).get('packed_weight_bytes'),
            performance_eligible=False)
    pairs=[(b,a) for a,b in zip(PRIMARY,PRIMARY[1:])]
    pairs += [('fp4_nine','fp4_four'),('fp4_sixteen','fp4_nine'),
              ('fp4_six_w','fp4_six_a'),('fp4_twelve_w','fp4_twelve_a')]
    pairs += [(n,ref) for n in PRIMARY+['fp4_six_w','fp4_twelve_w'] for ref in ('fp8','awq_reference')]
    pairs += [(n,ref) for n in ('fp4_two_a','fp4_two_w') for ref in ('fp4_base','fp8','awq_reference','fp4_three','fp4_four')]
    pairs += [('fp4_two_w','fp4_two_a')]
    comparisons=[]
    for candidate,reference in pairs:
        if candidate not in by_method or reference not in by_method:continue
        metrics={}
        for k,_ in METRICS:
            d=np.array([a[k]-b[k] for a,b in zip(by_method[candidate],by_method[reference])])
            higher=k in ('ssim_mean','psnr_db')
            metrics[k]=dict(mean_difference=float(d.mean()),ci95=interval(d,indices),
                improved_scenes=int(np.sum(d>0 if higher else d<0)),
                tied_scenes=int(np.sum(d==0)),worst_signed_regression=float(np.max(-d if higher else d)))
        comparisons.append(dict(candidate=candidate,reference=reference,metrics=metrics,inferential=False))
    evidence=dict(complete=True,manifest_sha256=sha(manifest),methods=summary,comparisons=comparisons,
        metric_files={str(p.relative_to(root)):sha(p) for p in paths},source_sha256=sha(__file__),
        interpretation='Paired consistency to BF16 on 10 selected scenes; exploratory confidence intervals '
                       'are unadjusted and are not final significance or task-success claims.')
    return study,rows,evidence


def figures(root, study, rows, evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':200,'pdf.fonttype':42,'svg.fonttype':'none'})
    out=root/'report';out.mkdir(exist_ok=True)
    summary=evidence['methods']
    primary=list(PRIMARY)
    if 'fp4_two_a' in summary:primary.insert(1,'fp4_two_a')
    for kind,names in [('terms',primary),('planes',SQUARE)]:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        for ax,(key,title) in zip(axes,METRICS[:3]):
            x=[summary[n]['expansion']['terms' if kind=='terms' else 'weight_planes'] for n in names]
            y=[summary[n]['metrics'][key]['mean'] for n in names]
            ci=np.array([summary[n]['metrics'][key]['ci95'] for n in names]).T
            ax.errorbar(x,y,yerr=[np.maximum(0,np.array(y)-ci[0]),np.maximum(0,ci[1]-y)],
                color='#145c9e',marker='o',capsize=3,
                label='FP4 expansion (2: A only)' if kind=='terms' and 'fp4_two_a' in summary else 'FP4 expansion')
            if kind=='terms':
                alternatives=['fp4_six_w','fp4_twelve_w']
                if 'fp4_two_w' in summary:alternatives.insert(0,'fp4_two_w')
                for j,n in enumerate(alternatives):
                    item=summary[n];value=item['metrics'][key];v=value['mean'];lo,hi=value['ci95']
                    ax.errorbar(item['expansion']['terms'],v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],
                        color='#7a3e9d',marker='D',capsize=3,linestyle='none',
                        label=('W-priority (2: W only)' if 'fp4_two_w' in summary else 'W-priority rectangle') if not j else None)
            for ref,col in [('fp8','#d25d17'),('awq_reference','#428442')]:
                dash=(0,(8,5)) if ref=='fp8' else (0,(3,2))
                ax.axhline(summary[ref]['metrics'][key]['mean'],color=col,linestyle=dash,label=LABELS[ref])
            if key in ('lpips','ssim_mean'):
                reference=0. if key=='lpips' else 1.
                ax.axhline(reference,color='#30343b',linestyle='-.',linewidth=1.5,
                    label=f'BF16 reference ({reference:g})')
            ax.set_xticks(x);ax.set_xlabel('FP4 product terms per controlled GEMM' if kind=='terms' else 'FP4 planes on each side')
            ax.set_ylabel(title);ax.grid(alpha=.18);ax.legend(fontsize=8,handlelength=3.8)
        fig.suptitle('Edge: 5 RBench + 5 PAI-Bench scenes | paired BF16 consistency | 35 steps\n'
                     'Mean and exploratory 95% scene bootstrap interval; no monotonicity constraint',fontsize=12)
        for suffix in ('png','pdf','svg'):fig.savefig(out/f'quality_vs_{kind}.{suffix}')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,(key,title) in zip(axes,METRICS[:3]):
        for ci,case in enumerate(study['cases']):
            cid=case['case']['id'];data={r['method']:r for r in rows if r['case_id']==cid}
            ax.plot([summary[n]['expansion']['terms'] for n in primary],
                [data[n][key] for n in primary],marker='.',alpha=.7,
                label=case['benchmark']+'/'+case['category'])
        ax.set_xlabel('FP4 terms, A-priority rectangles');ax.set_ylabel(title)
        ax.set_xticks([summary[n]['expansion']['terms'] for n in primary]);ax.grid(alpha=.18)
    axes[-1].legend(fontsize=7,loc='best')
    fig.suptitle('Per-demo trends: no envelope, smoothing, or scene removal')
    for suffix in ('png','pdf'):fig.savefig(out/f'per_demo.{suffix}')
    plt.close(fig)


def report(root):
    study,rows,evidence=collect(root);out=root/'report';out.mkdir(exist_ok=True)
    write(out/'results.json',evidence)
    with (out/'per_demo.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(root,study,rows,evidence)
    text=['# Edge FP4 residual expansion: ten-demo evidence','',
        'All values measure agreement with same-seed BF16 outputs, not reference-video fidelity or task success. '
        'Five RBench and five PAI-Bench scenes were selected by frozen ID hashes before generation. '
        'Each uses one new seed, 320×192, 61 frames, 24 FPS, 35 steps, CFG 6, flow shift 3.','',
        '| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Relative L2 ↓ |',
        '|---|---:|---:|---:|---:|']
    for name,item in evidence['methods'].items():
        v=item['metrics']
        text.append(f"| {item['label']} | {v['psnr_db']['mean']:.4f} | {v['ssim_mean']['mean']:.6f} | {v['lpips']['mean']:.6f} | {v['relative_l2']['mean']:.6f} |")
    text += ['', '![Quality vs terms](quality_vs_terms.png)','![Quality vs planes](quality_vs_planes.png)',
        '', '## Arithmetic and comparison scope','',
        'A plane is one actual E2M1/E4M3 quantized residual. Each later plane is quantized from the error '
        'remaining after decoding all preceding planes. The outer factor is fixed across planes; each K16 '
        'group has its own E4M3 scale with floor 1/512. W0 is identical in every FP4 configuration. '
        'All weight planes are prepared offline, and activation planes always use current inputs.','',
        'Three terms are W0A0 + W0A1 + W1A0. Four, nine and sixteen terms are complete 2×2, 3×3 and 4×4 '
        'products. Six terms use W2×A3 and W3×A2; twelve use W3×A4 and W4×A3. '
        'In attention, A counts the Q/P planes and W counts current K/V planes. QK terms are summed before '
        'softmax; PV has the same term count. All base attention blocks remain.','',
        'Native FP4 dot_scaled products use shared FP32 accumulation and BF16 operator outputs. '
        'Causal AR prefill and nonlinear operations retain the common BF16 path. The native FP8 baseline '
        'uses W8A8 projections with online per-token activation quantization and BF16 attention. '
        'AWQ is the pinned ModelOpt NVFP4 AWQ-lite decoded W4A16 quality adapter with BF16 attention, '
        'not INT4 AWQ and not a packed deployment latency measurement.','',
        'The x-axis counts product terms only in the controlled projections/FFN and QK/PV. It is not '
        'total request FLOPs or measured speed. Repeated on-chip P quantization and additional packing '
        'are outside this multiplier. Shared-calibration timings must not be interpreted as speedups.','',
        '## Paired changes along the primary path','',
        '| Change | ΔLPIPS (95% exploratory CI) | Improved scenes | ΔSSIM | ΔPSNR, dB |',
        '|---|---:|---:|---:|---:|']
    for comparison in evidence['comparisons'][:len(PRIMARY)-1]:
        c=comparison;v=c['metrics'];lo,hi=v['lpips']['ci95']
        text.append(f"| {LABELS[c['reference']]} → {LABELS[c['candidate']]} | {v['lpips']['mean_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {v['lpips']['improved_scenes']}/10 | {v['ssim_mean']['mean_difference']:+.6f} | {v['psnr_db']['mean_difference']:+.4f} |")
    text += ['', '## FP4 term and weight costs', '',
        '| Method | W planes | A planes | Terms per controlled GEMM | Packed projection weights, MiB |',
        '|---|---:|---:|---:|---:|']
    for name,item in evidence['methods'].items():
        spec=item['expansion']
        if spec:
            text.append(f"| {item['label']} | {spec['weight_planes']} | {spec['activation_planes']} | {spec['terms']} | {item['packed_weight_bytes']/2**20:.2f} |")
    text += ['', 'Packed bytes include E2M1 codes, E4M3 scales and matrix factors of the 168 replaced '
        'generation projections. They exclude common AR/VAE/text weights, biases, prescale vectors, '
        'activation scratch and Graph buffers. Each runtime loads only its required weight prefix. '
        'The reusable offline master asset has all four planes; it is not the resident per-method cost.', '']
    weights=root/'weight_reconstruction.json'
    if weights.exists():
        data=json.loads(weights.read_text())
        text += ['## Offline weight reconstruction check','',data['scope'],'',
            '| Cumulative planes | Pooled weight NMSE | Nonzero codes in new plane | Scales at floor |',
            '|---|---:|---:|---:|']
        for r in data['summary']:
            text.append(f"| {r['planes']} | {r['pooled_nmse']:.6g} | {r['nonzero_code_fraction']:.2%} | {r['floor_scale_fraction']:.2%} |")
        text += ['', '[Per-layer reconstruction measurements](../weight_reconstruction.json). '
            'A falling weight reconstruction error does not guarantee monotonic final-video improvement.', '']
    text += ['', '## Limits and artifacts','',
        'Each metric is first computed per complete raw float32 video, then averaged equally across scenes. '
        'PSNR uses RGB clip-global MSE at data range 1; SSIM and LPIPS are averaged across all 61 frames. '
        'The plots show unadjusted paired-scene bootstrap intervals (10,000 replicates). '
        'Ten demos cannot establish broad generalization or the final FP8 quality gate. '
        'Higher plane counts need not improve every scene: the E4M3 scale floor, FP32 accumulation, '
        'BF16 boundaries, streaming softmax and denoising trajectory can create plateaus or regressions.','',
        '- [All demo videos](gallery.html)', '- [Per-demo raw metrics](per_demo.csv)',
        '- [Means, intervals and paired baseline comparisons](results.json)',
        '- [Frozen experiment](../edge/final/manifest.json)',
        '- [Kernel correctness and sanitizer receipts](../qualification/results.json)',
        '- [Implementation and weight lock](../implementation.lock.json)',
        '- [Generation logs](../logs/)', '- [Per-demo curves](per_demo.png)', '']
    (out/'report.md').write_text('\n'.join(text))
    make_grids(root,study)
    gallery(root,study)
    return dict(complete=True,report=str(out/'report.md'),paired_comparisons=len(rows))


def gallery(root,study):
    index=root/'edge/judge_index.jsonl'
    records=[json.loads(s) for s in index.read_text().splitlines()]
    lookup={(r['method'],r['case_id']):r for r in records}
    if len(lookup)!=len(study['cases'])*len(study['methods']):raise ValueError('Video export incomplete')
    import os
    out=root/'report';parts=['<!doctype html><meta charset="utf-8"><title>Edge FP4 expansion demos</title>',
        '<style>body{font:16px system-ui;margin:32px;background:#f5f6fa;color:#172033}h2{margin-top:36px}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
        'figure{margin:0;background:white;padding:12px;border-radius:8px}video{width:100%}'
        'figcaption{margin-bottom:8px}details{margin:12px 0}button{padding:8px 16px;margin:4px}</style>',
        '<h1>Edge: FP4 residual expansion</h1><p>10 fixed scenes, one shared seed per scene; 35 steps, 61 frames. '
        'BF16 is the consistency reference. These are mechanism demos, not a benchmark quality claim.</p>']
    for i,entry in enumerate(study['cases']):
        cid=entry['case']['id'];parts += [f'<section id="scene{i}"><h2>{html.escape(cid)} · seed {entry["seeds"][0]}</h2>',
            '<details><summary>Prompt</summary>'+html.escape(entry['case']['prompt'])+'</details>',
            '<details><summary>All 12 methods in one synchronized video</summary>'
            '<video controls muted loop preload="none" src="grids/'+html.escape(cid,quote=True)+'.mp4"></video></details>',
            f'<button onclick="playScene({i})">Restart and play this scene</button>',
            f'<button onclick="pauseScene({i})">Pause this scene</button><div class="grid">']
        for name in study['methods']:
            video=os.path.relpath(lookup[name,cid]['video_path'],out)
            parts += ['<figure><figcaption>'+html.escape(LABELS[name])+'</figcaption>',
                '<video controls muted loop preload="none" src="'+html.escape(video,quote=True)+'"></video></figure>']
        parts.append('</div></section>')
    parts.append('<script>function pauseScene(i){document.querySelectorAll("#scene"+i+" video").forEach(v=>v.pause())}'
        'function playScene(i){document.querySelectorAll("video").forEach(v=>v.pause());'
        'document.querySelectorAll("#scene"+i+" video").forEach(v=>{v.currentTime=0;v.play()})}</script>')
    (out/'gallery.html').write_text('\n'.join(parts))


def make_grids(root,study):
    from PIL import Image,ImageDraw,ImageFont
    from judge_video_export import encode,frames_uint8
    order=['bf16','fp8','awq_reference','fp4_base','fp4_three','fp4_four','fp4_six_a','fp4_six_w',
           'fp4_nine','fp4_twelve_a','fp4_twelve_w','fp4_sixteen']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    folder=root/'report/grids';folder.mkdir(exist_ok=True)
    for entry in study['cases']:
        cid=entry['case']['id'];target=folder/(cid+'.mp4');meta=folder/(cid+'.json')
        sources={}
        for name in order:
            path=root/'edge/final/generation'/name/cid/'result.json'
            data=json.loads(path.read_text());sources[name]=dict(receipt_sha256=sha(path),row=data['rows'][0])
        identity=dict(manifest_sha256=sha(root/'edge/final/manifest.json'),source_sha256=sha(__file__),sources=sources)
        if meta.exists():
            old=json.loads(meta.read_text())
            if old['identity']!=identity or sha(target)!=old['video']['sha256']:raise ValueError('Changed demo grid')
            continue
        case=entry['case'];h,w,n=case['height'],case['width'],case['frames'];title_h=26
        grid=np.zeros((n,3*(h+title_h),4*w,3),dtype=np.uint8)
        for i,name in enumerate(order):
            y,x=(i//4)*(h+title_h),(i%4)*w
            label=Image.new('RGB',(w,title_h),(24,34,50));draw=ImageDraw.Draw(label)
            draw.text((8,4),LABELS[name],font=font,fill='white');grid[:,y:y+title_h,x:x+w]=np.asarray(label)
            path=root/'edge/final/generation'/name/cid/sources[name]['row']['outputs']
            if sha(path)!=sources[name]['row']['outputs_sha256']:raise ValueError('Changed grid source')
            with np.load(path) as data:frames=frames_uint8(data['video'],case)
            grid[:,y+title_h:y+title_h+h,x:x+w]=frames
        video=encode(grid,target,lossless=False)
        Image.fromarray(grid[n//2]).save(folder/(cid+'.jpg'),quality=95)
        write(meta,dict(complete=True,identity=identity,video=video,scope='Viewing only; metrics use original float32 arrays'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(report(args.root.resolve())))

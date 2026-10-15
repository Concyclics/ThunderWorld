"""Report the frozen I2V experiment against aligned recorded videos, including BF16."""
import argparse
import csv
import hashlib
import html
import inspect
import json
import os
from pathlib import Path

import numpy as np
from baselines import sha, write
from expansion_report import LABELS, METRICS, PRIMARY, SQUARE, interval

CATEGORIES = {'robotics': 'AgiBot robotics', 'driving': 'OpenDV driving',
              'egocentric': 'Ego/Exo activities'}
SCOPES = ('future', 'all_frames')
METHOD_ORDER = ['bf16', 'fp8', 'awq_reference', 'fp4_base', 'fp4_three', 'fp4_four',
                'fp4_six_a', 'fp4_six_w', 'fp4_nine', 'fp4_twelve_a', 'fp4_twelve_w', 'fp4_sixteen']


def collect(root):
    manifest = root / 'edge/final/manifest.json'
    study = json.loads(manifest.read_text())
    reference_path = Path(study['reference_manifest'])
    refs = json.loads(reference_path.read_text())
    if not study['frozen'] or not refs['complete'] or sha(reference_path) != study['reference_manifest_sha256']:
        raise ValueError('Changed or unfinished reference cohort')
    if set(study['methods']) != set(METHOD_ORDER):
        raise ValueError('The report requires all twelve frozen methods')
    cases = study['cases']
    if len(cases) != 10 or len({e['case']['id'] for e in cases}) != 10:
        raise ValueError('Expected ten distinct frozen cases')
    rows, paths, grouped = [], [], {}
    metric_identity = None
    for name in METHOD_ORDER:
        grouped[name] = []
        for entry in cases:
            cid = entry['case']['id']
            path = root / 'edge/final/metrics_real' / name / (cid + '.json')
            data = json.loads(path.read_text())
            receipt = root / 'edge/final/generation' / name / cid / 'result.json'
            p = data['provenance']
            if metric_identity is None:
                metric_identity = p['identity']
            if (not data['complete'] or data['method'] != name or data['case_id'] != cid or
                    p['identity'] != metric_identity or
                    p['identity']['manifest_sha256'] != sha(manifest) or
                    p['identity']['reference_manifest_sha256'] != sha(reference_path) or
                    p['generation_receipt_sha256'] != sha(receipt) or
                    p['aligned_sha256'] != refs['cases'][cid]['aligned_sha256']):
                raise ValueError('Unmatched real-reference metric: ' + str(path))
            if len(data['rows']) != 1 or [data['rows'][0]['seed']] != entry['seeds']:
                raise ValueError('Expected one matched seed per scene')
            base = dict(method=name, label=LABELS[name], case_id=cid, category=entry['category'],
                        seed=entry['seeds'][0])
            item = {**base, **{scope: data['rows'][0][scope] for scope in SCOPES}}
            for scope in SCOPES:
                scores = item[scope]
                if scores['psnr_exact'] or not all(np.isfinite(scores[k]) for k, _ in METRICS):
                    raise ValueError('Nonfinite metric needs explicit infinity handling')
                rows.append({**base, 'scope': scope, **scores})
            grouped[name].append(item)
            paths.append(path)
    rng = np.random.default_rng(study['statistics']['bootstrap_seed'])
    indices = rng.integers(0, len(cases), (study['statistics']['replicates'], len(cases)))
    summary = {}
    for name, records in grouped.items():
        scopes = {}
        for scope in SCOPES:
            scopes[scope] = {
                key: dict(mean=float(np.mean([r[scope][key] for r in records])),
                          ci95=interval([r[scope][key] for r in records], indices),
                          minimum=min(r[scope][key] for r in records),
                          maximum=max(r[scope][key] for r in records)) for key, _ in METRICS}
        summary[name] = dict(label=LABELS[name], n=len(records), scopes=scopes,
            expansion=study['methods'][name].get('expansion'),
            by_category={c: {scope: {k: float(np.mean([r[scope][k] for r in records if r['category'] == c]))
                                    for k, _ in METRICS} for scope in SCOPES} for c in CATEGORIES})
    pairs = [(b, a) for a, b in zip(PRIMARY, PRIMARY[1:])]
    pairs += [('fp4_nine', 'fp4_four'), ('fp4_sixteen', 'fp4_nine'),
              ('fp4_six_w', 'fp4_six_a'), ('fp4_twelve_w', 'fp4_twelve_a')]
    pairs += [(name, ref) for name in METHOD_ORDER[3:] for ref in ('bf16', 'fp8', 'awq_reference')]
    comparisons = []
    for candidate, reference in pairs:
        scopes = {}
        for scope in SCOPES:
            scores = {}
            for key, _ in METRICS:
                delta = np.array([a[scope][key] - b[scope][key]
                                  for a, b in zip(grouped[candidate], grouped[reference])])
                regression = -delta if key in ('ssim_mean', 'psnr_db') else delta
                worst = int(np.argmax(regression))
                scores[key] = dict(mean_difference=float(delta.mean()), ci95=interval(delta, indices),
                    improved_scenes=int(np.sum(regression < 0)), tied_scenes=int(np.sum(delta == 0)),
                    worst_signed_regression=float(regression[worst]), worst_case=cases[worst]['case']['id'])
            scopes[scope] = scores
        comparisons.append(dict(candidate=candidate, reference=reference, scopes=scopes, inferential=False))
    supplementary = {}
    for name in METHOD_ORDER[1:]:
        values = []
        for entry in cases:
            path = root / 'edge/final/metrics_bf16' / name / (entry['case']['id'] + '.json')
            data = json.loads(path.read_text())
            if not data['complete'] or data['provenance']['identity']['manifest_sha256'] != sha(manifest):
                raise ValueError('Incomplete supplementary BF16 consistency metrics')
            if len(data['rows']) != 1 or data['rows'][0]['seed'] != entry['seeds'][0]:
                raise ValueError('Unmatched supplementary seed')
            values.append(data['rows'][0]['metrics']['video'])
            paths.append(path)
        supplementary[name] = {k: float(np.mean([v[k] for v in values])) for k, _ in METRICS}
    evidence = dict(complete=True, manifest_sha256=sha(manifest), reference_manifest_sha256=sha(reference_path),
        metric_identity=metric_identity, methods=summary, comparisons=comparisons,
        supplementary_bf16_consistency_all61=supplementary,
        metric_files={str(p.relative_to(root)): sha(p) for p in paths}, source_sha256=sha(__file__),
        statistics=study['statistics'],
        interpretation='Recorded-video fidelity on ten frozen PAI-Bench-C source videos adapted to I2V. '
            'Primary excludes the conditioning frame. Unadjusted scene-bootstrap intervals are descriptive; '
            'not official PAI-Bench-C scores, task success or a final quality gate.')
    return study, rows, evidence


def figures(root, study, rows, evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 200, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    out = root / 'report'
    summary = evidence['methods']
    for kind, names in [('terms', PRIMARY), ('planes', SQUARE)]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
        for ax, (key, title) in zip(axes, METRICS[:3]):
            x = [summary[n]['expansion']['terms' if kind == 'terms' else 'weight_planes'] for n in names]
            y = np.array([summary[n]['scopes']['future'][key]['mean'] for n in names])
            ci = np.array([summary[n]['scopes']['future'][key]['ci95'] for n in names]).T
            ax.errorbar(x, y, yerr=[np.maximum(0, y-ci[0]), np.maximum(0, ci[1]-y)],
                        color='#145c9e', marker='o', capsize=3, label='FP4 expansion', elinewidth=.8)
            if kind == 'terms':
                for j, name in enumerate(('fp4_six_w', 'fp4_twelve_w')):
                    v = summary[name]['scopes']['future'][key]
                    ax.errorbar(summary[name]['expansion']['terms'], v['mean'],
                        yerr=[[max(0, v['mean']-v['ci95'][0])], [max(0, v['ci95'][1]-v['mean'])]],
                        color='#7a3e9d', marker='D', capsize=3, linestyle='none', elinewidth=.8,
                        label='W-priority rectangle' if j == 0 else None)
            for ref, color, dash in [('fp8', '#d25d17', (0, (8, 5))),
                                     ('awq_reference', '#428442', (0, (3, 2))),
                                     ('bf16', '#30343b', '-.')]:
                ax.axhline(summary[ref]['scopes']['future'][key]['mean'], color=color,
                           linestyle=dash, linewidth=1.5, label=LABELS[ref])
            ax.set_xticks(x)
            ax.set_xlabel('FP4 products per controlled GEMM' if kind == 'terms' else 'FP4 planes on each side')
            ax.set_ylabel(title)
            ax.grid(alpha=.18)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=5, fontsize=9, handlelength=3.8)
        fig.suptitle('Edge: 10 PAI-Bench-C source videos adapted to I2V | real-video reference\n'
                     'Future frames 1–60; mean and descriptive 95% scene-bootstrap interval', fontsize=12)
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(out / f'quality_vs_{kind}.{suffix}')
        plt.close(fig)
    # Pairing removes between-scene difficulty variation from the comparison interval.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    names = METHOD_ORDER[3:]
    comparisons = {c['candidate']: c for c in evidence['comparisons'] if c['reference'] == 'fp8'}
    for ax, (key, title) in zip(axes, METRICS[:3]):
        values = [comparisons[n]['scopes']['future'][key] for n in names]
        y = np.array([v['mean_difference'] for v in values])
        ci = np.array([v['ci95'] for v in values]).T
        ax.errorbar(range(len(names)), y, yerr=[np.maximum(0, y-ci[0]), np.maximum(0, ci[1]-y)],
                    color='#145c9e', fmt='o', capsize=4)
        ax.axhline(0, color='#d25d17', linestyle=(0, (8, 5)), label='Native FP8')
        ax.set_xticks(range(len(names)), ['1', '3', '4', '6A', '6W', '9', '12A', '12W', '16'])
        ax.set_xlabel('FP4 product terms (A/W: extra activation/weight plane)')
        ax.set_ylabel('Δ ' + title)
        ax.grid(alpha=.18)
    fig.suptitle('Paired change versus FP8 on recorded-video fidelity | 10 scenes, future frames only\n'
                 'Unadjusted descriptive 95% intervals; no final significance claim', fontsize=12)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'paired_vs_fp8.{suffix}')
    plt.close(fig)


def render_grid(videos, order, label_map):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    n, h, w, _ = videos[order[0]].shape
    title_h = 26
    grid = np.full((n, 4*(h+title_h), 4*w, 3), 238, dtype=np.uint8)
    for i, name in enumerate(order):
        y, x = (i//4)*(h+title_h), (i%4)*w
        label = Image.new('RGB', (w, title_h), (24, 34, 50))
        ImageDraw.Draw(label).text((8, 4), label_map[name], font=font, fill='white')
        grid[:, y:y+title_h, x:x+w] = np.asarray(label)
        grid[:, y+title_h:y+title_h+h, x:x+w] = videos[name]
    return grid


def media(root, study):
    from PIL import Image
    from judge_video_export import encode, frames_uint8
    refs = json.loads(Path(study['reference_manifest']).read_text())
    index = [json.loads(s) for s in (root / 'edge/judge_index.jsonl').read_text().splitlines()]
    lookup = {(r['method'], r['case_id']): r for r in index}
    if len(lookup) != 120 or len(index) != 120:
        raise ValueError('Missing or duplicate video exports')
    out = root / 'report'
    (out / 'grids').mkdir(exist_ok=True)
    (out / 'references').mkdir(exist_ok=True)
    parts = ['<!doctype html><meta charset="utf-8"><title>Edge: recorded-video comparison</title>',
        '<style>body{font:16px system-ui;margin:32px;background:#f5f6fa;color:#172033}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
        'figure{margin:0;background:white;padding:12px}video{width:100%}'
        'details{margin:12px 0}button{padding:8px 16px;margin:4px}</style>',
        '<h1>Edge: FP4 expansion versus recorded video</h1>'
        '<p>10 frozen PAI-Bench-C source clips adapted to first-frame I2V, 12 methods each. '
        '320×192, 61 frames, 24 FPS, 35 steps; primary metrics exclude frame 0. '
        'These are paired fidelity demos, not official benchmark scores or task success.</p>']
    labels = {'real': 'Real recorded video', **LABELS}
    order = ['real'] + METHOD_ORDER
    renderer_hash = hashlib.sha256(inspect.getsource(render_grid).encode()).hexdigest()
    for i, entry in enumerate(study['cases']):
        cid = entry['case']['id']
        videos, sources = {}, {}
        path = Path(refs['cases'][cid]['aligned_file'])
        if sha(path) != refs['cases'][cid]['aligned_sha256']:
            raise ValueError('Changed reference export')
        with np.load(path) as data:
            videos['real'] = frames_uint8(data['video'], entry['case'])
        real_path = out / 'references' / (cid + '.mp4')
        source = dict(aligned_sha256=sha(path), renderer_sha256=renderer_hash)
        for name in METHOD_ORDER:
            row = lookup[name, cid]
            receipt = Path(row['generation_receipt']) if 'generation_receipt' in row else Path(row['raw_outputs']).parent / 'result.json'
            value = json.loads(receipt.read_text())['rows'][0]
            raw = Path(row['raw_outputs'])
            if sha(raw) != value['outputs_sha256']:
                raise ValueError('Changed generation export')
            with np.load(raw) as data:
                videos[name] = frames_uint8(data['video'], entry['case'])
            sources[name] = dict(receipt_sha256=sha(receipt), output_sha256=sha(raw))
        identity = dict(source=source, sources=sources)
        meta = out / 'grids' / (cid + '.json')
        target = out / 'grids' / (cid + '.mp4')
        if meta.exists():
            old = json.loads(meta.read_text())
            if old['identity'] != identity or sha(target) != old['grid']['sha256'] or sha(real_path) != old['real']['sha256']:
                raise ValueError('Changed comparison gallery')
        else:
            real = encode(videos['real'], real_path, lossless=False)
            grid = render_grid(videos, order, labels)
            encoded = encode(grid, target, lossless=False)
            Image.fromarray(grid[len(grid)//2]).save(target.with_suffix('.jpg'), quality=95)
            write(meta, dict(complete=True, identity=identity, grid=encoded, real=real,
                scope='Viewing only; all metrics use raw float32 signals'))
        print(json.dumps(dict(exported_comparison=cid)), flush=True)
        parts += [f'<section id="scene{i}"><h2>{html.escape(cid)} · {CATEGORIES[entry["category"]]} · seed {entry["seeds"][0]}</h2>',
            '<details><summary>Original caption</summary>' + html.escape(entry['case']['prompt']) + '</details>',
            '<details><summary>Real video + all 12 methods, synchronized</summary>'
            f'<video controls muted loop preload="none" src="grids/{cid}.mp4"></video></details>',
            f'<button onclick="playScene({i})">Restart and play scene</button>'
            f'<button onclick="pauseScene({i})">Pause scene</button><div class="grid">']
        for name in order:
            video = real_path if name == 'real' else Path(lookup[name, cid]['video_path'])
            parts += ['<figure><figcaption>' + html.escape(labels[name]) + '</figcaption>',
                '<video controls muted loop preload="none" src="' + html.escape(os.path.relpath(video, out), quote=True) + '"></video></figure>']
        parts.append('</div></section>')
    parts.append('<script>function pauseScene(i){document.querySelectorAll("#scene"+i+" video").forEach(v=>v.pause())}'
        'function playScene(i){document.querySelectorAll("video").forEach(v=>v.pause());'
        'document.querySelectorAll("#scene"+i+" video").forEach(v=>{v.currentTime=0;v.play()})}</script>')
    (out / 'gallery.html').write_text('\n'.join(parts))


def table(summary, scope):
    lines = ['| Method | PSNR ↑, dB | SSIM ↑ | LPIPS ↓ | Relative L2 ↓ |', '|---|---:|---:|---:|---:|']
    for name in METHOD_ORDER:
        item = summary[name]
        v = item['scopes'][scope]
        lines.append(f"| {item['label']} | {v['psnr_db']['mean']:.4f} | {v['ssim_mean']['mean']:.6f} | {v['lpips']['mean']:.6f} | {v['relative_l2']['mean']:.6f} |")
    return lines


def report(root, skip_media=False):
    study, rows, evidence = collect(root)
    out = root / 'report'
    out.mkdir(exist_ok=True)
    write(out / 'results.json', evidence)
    with (out / 'per_demo.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(root, study, rows, evidence)
    text = ['# Edge FP4 expansion against recorded video', '',
        'This is a new, frozen cohort of ten PAI-Bench-C source videos: four AgiBot robotics, '
        'three OpenDV driving and three Ego/Exo activity clips. Every method receives only the same '
        'first frame and original caption. This I2V adaptation does not use the official C task’s '
        'depth/segmentation/edge/blur controls and is not an official PAI-Bench-C benchmark score.', '',
        'All twelve methods ran the same Edge checkpoint, 320×192, 61 frames at 24 FPS, '
        '35 steps, CFG 6, flow shift 3, Graph execution and debug off. '
        'One paired seed per scene (92222001–92222010); IDs were selected by a frozen hash rule '
        'before generation or scoring. No scene was removed based on its results.', '',
        '## Primary: generated future frames 1–60', '',
        'The reference is the recorded video, so BF16 has measured finite error and finite PSNR. '
        'Frame 0 is excluded because it was provided as input. Each scene contributes equally. '
        'These metrics assess similarity to one recorded continuation; they do not measure the '
        'probability or physical validity of other plausible futures.', '',
        *table(evidence['methods'], 'future'), '',
        '![Quality versus FP4 terms](quality_vs_terms.png)', '',
        '![Quality versus planes](quality_vs_planes.png)', '',
        '## Paired comparison with FP8', '',
        'Intervals resample the same paired scenes (10,000 bootstrap replicates). They are descriptive '
        'and unadjusted for multiple comparisons, not a final superiority claim.', '',
        '| FP4 method | ΔLPIPS [95% CI] | LPIPS improved scenes | ΔSSIM | ΔPSNR, dB |',
        '|---|---:|---:|---:|---:|']
    for c in evidence['comparisons']:
        if c['reference'] != 'fp8':
            continue
        v = c['scopes']['future']
        lo, hi = v['lpips']['ci95']
        text.append(f"| {LABELS[c['candidate']]} | {v['lpips']['mean_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {v['lpips']['improved_scenes']}/10 | {v['ssim_mean']['mean_difference']:+.6f} | {v['psnr_db']['mean_difference']:+.4f} |")
    best = min(METHOD_ORDER[3:], key=lambda n: evidence['methods'][n]['scopes']['future']['lpips']['mean'])
    best_values = evidence['methods'][best]['scopes']['future']
    best_pair = next(c for c in evidence['comparisons'] if c['candidate'] == best and c['reference'] == 'fp8')
    delta = best_pair['scopes']['future']['lpips']
    base_lpips = evidence['methods']['fp4_base']['scopes']['future']['lpips']['mean']
    text += ['', '## What this cohort supports', '',
        f"The lowest observed FP4 mean LPIPS is {LABELS[best]}: {best_values['lpips']['mean']:.6f}, "
        f"a {100*(1-best_values['lpips']['mean']/base_lpips):.2f}% reduction from plain W4A4. "
        'This is a descriptive comparison of all reported candidates, not a held-out selection result.', '',
        f"Its paired LPIPS change versus FP8 is {delta['mean_difference']:+.6f}, with descriptive 95% "
        f"interval [{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}]. "
        + ('The interval crosses zero; this experiment does not establish superiority over FP8.'
           if delta['ci95'][0] <= 0 <= delta['ci95'][1] else
           'This unadjusted interval is exploratory and does not replace the frozen final quality gate.'), '',
        'The complete 2×2, 3×3 and 4×4 expansions do not produce monotonically improving '
        'recorded-video fidelity. Arithmetic restoration and matching a particular recorded future '
        'must remain separate claims; both sets of measurements are retained below.', '',
        '![Paired differences](paired_vs_fp8.png)', '',
        '## Secondary: all 61 frames', '', *table(evidence['methods'], 'all_frames'), '',
        '## Supplementary: consistency to BF16 on this same new cohort', '',
        'This separate diagnostic includes all 61 frames and measures restoration of native model '
        'outputs. It is not combined with the recorded-video metrics above.', '',
        '| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |', '|---|---:|---:|---:|']
    for name, v in evidence['supplementary_bf16_consistency_all61'].items():
        text.append(f"| {LABELS[name]} | {v['psnr_db']:.4f} | {v['ssim_mean']:.6f} | {v['lpips']:.6f} |")
    refs = json.loads(Path(study['reference_manifest']).read_text())
    text += ['', '## Reference alignment and definitions', '',
        'Original video PTS determine the nearest recorded frame at 0, 1/24, …, 60/24 seconds. '
        'There is no time stretching, output-dependent registration, optical-flow warping or padding. '
        'Spatial preprocessing matches the generator: aspect-preserving bilinear antialias resize, '
        'center crop and uint8 rounding. The first-frame transform was checked pixelwise against '
        'Cosmos’s actual conditioning helper for all ten clips.', '',
        '| Scene | Source group | Seed | Max time error, ms | Unique reference frames |',
        '|---|---|---:|---:|---:|']
    for e in study['cases']:
        cid = e['case']['id']; v = refs['cases'][cid]
        text.append(f"| {cid} | {CATEGORIES[e['category']]} | {e['seeds'][0]} | {1000*v['maximum_timestamp_error_seconds']:.3f} | {v['unique_reference_frames']} |")
    text += ['', 'PSNR is −10 log10 of clip-global RGB MSE, data range 1, calculated before averaging '
        'across scenes. SSIM uses channel-wise RGB structural_similarity with data range 1 and '
        'averages frames. LPIPS uses AlexNet v0.1 and averages frames. Metrics use original float32 '
        'generated outputs and decoded real RGB/255, never the lossy viewing MP4 files.', '',
        '## Arithmetic scope and limits', '',
        'All generation projections/FFN and noncausal QK/PV use native FP4 Tensor Core expansion '
        'for the FP4 methods, with current-input activation and Q/K/V planes. AR causal prefill and '
        'nonlinear operations retain the common BF16 path. Weights are prepared offline; W0 is '
        'identical across FP4 configurations. Three terms omit W1A1 from the full 2×2 expansion. '
        '6A/6W mean W2×A3/W3×A2; 12A/12W mean W3×A4/W4×A3. In attention, W counts K/V '
        'planes and A counts Q/P planes. All base attention blocks remain.', '',
        'FP8 is native W8A8 projection with current per-token activation quantization and BF16 '
        'attention. AWQ is the pinned ModelOpt NVFP4 AWQ-lite decoded W4A16 quality adapter, '
        'with BF16 attention; it is not a packed-kernel performance measurement. The x axis '
        'counts products in controlled GEMMs, not end-to-end FLOPs or speed. All runs use '
        'shared-calibration mode on RTX PRO 6000; no latency conclusion follows.', '',
        'More accurate arithmetic need not improve agreement with a single recorded future. '
        'Ten clips and one seed per clip are exploratory evidence, not broad generalization. '
        'Per-category values, paired changes, worst-scene regressions and source hashes are retained '
        'in results.json. The earlier 5 RBench + 5 PAI-Bench-G BF16-consistency experiment is preserved.', '',
        '- [Real video and all generated demos](gallery.html)',
        '- [Per-scene metrics, primary and secondary](per_demo.csv)',
        '- [Full statistics, comparisons and source hashes](results.json)',
        '- [Frozen generation manifest](../edge/final/manifest.json)',
        '- [Reference timestamps and hashes](../references/manifest.json)',
        '- [Independent metric and request validation](../validation.json)',
        '- [Original dataset at pinned revision](https://huggingface.co/datasets/shi-labs/physical-ai-bench-conditional-generation/tree/' + refs['source_revision'] + ')',
        '- [Complete generation logs](../logs/)', '']
    (out / 'report.md').write_text('\n'.join(text))
    if not skip_media:
        media(root, study)
    return dict(complete=True, metric_pairs=120, report=str(out / 'report.md'), media_generated=not skip_media)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--skip-media', action='store_true')
    args = parser.parse_args()
    print(json.dumps(report(args.root.resolve(), args.skip_media)))

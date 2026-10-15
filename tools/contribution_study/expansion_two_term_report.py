"""Combine the two-term extension with unchanged original BF16-consistency evidence."""
import argparse
import copy
import csv
import hashlib
import html
import inspect
import json
import os
from pathlib import Path

import numpy as np
from baselines import sha, write
from expansion_report import LABELS, METRICS, collect, figures, interval
from expansion_two_term import load, VARIANTS

ORDER = ['bf16', 'fp8', 'awq_reference', 'fp4_base', 'fp4_two_a', 'fp4_two_w',
         'fp4_three', 'fp4_four', 'fp4_six_a', 'fp4_six_w', 'fp4_nine',
         'fp4_twelve_a', 'fp4_twelve_w', 'fp4_sixteen']


def combined(root):
    extension, parent, base = load(root)
    frozen = json.loads((parent/'report/results.json').read_text())
    if sha(parent/'report/results.json') != extension['extension']['parent_metrics_sha256']:
        raise ValueError('Parent numerical evidence changed')
    _, old_rows, old = collect(parent)
    if old['methods'] != frozen['methods'] or old['comparisons'] != frozen['comparisons']:
        raise ValueError('Original metrics changed during aggregation')
    _, new_rows, new = collect(root)
    study = copy.deepcopy(base)
    specs = {**base['methods'], **extension['methods']}
    study['methods'] = {name: specs[name] for name in ORDER}
    all_rows = old_rows+new_rows
    lookup = {(r['method'], r['case_id']): r for r in all_rows}
    if len(lookup) != 130 or len(all_rows) != 130:
        raise ValueError('Missing or duplicate paired rows')
    rows = [lookup[name, e['case']['id']] for name in ORDER[1:] for e in study['cases']]
    methods = {**old['methods'], **new['methods']}
    methods = {name: methods[name] for name in ORDER[1:]}
    rng = np.random.default_rng(study['statistics']['bootstrap_seed'])
    indices = rng.integers(0, 10, (study['statistics']['replicates'], 10))
    pairs = [(n, ref) for n in VARIANTS for ref in ('fp4_base', 'fp8', 'awq_reference', 'fp4_three', 'fp4_four')]
    pairs += [('fp4_two_w', 'fp4_two_a')]
    additions = []
    for candidate, reference in pairs:
        metrics = {}
        for key, _ in METRICS:
            delta = np.array([lookup[candidate, e['case']['id']][key]-lookup[reference, e['case']['id']][key]
                              for e in study['cases']])
            regression = -delta if key in ('psnr_db', 'ssim_mean') else delta
            metrics[key] = dict(mean_difference=float(delta.mean()), ci95=interval(delta, indices),
                improved_scenes=int(np.sum(regression < 0)), tied_scenes=int(np.sum(delta == 0)),
                worst_signed_regression=float(regression.max()),
                worst_case=study['cases'][int(regression.argmax())]['case']['id'])
        additions.append(dict(candidate=candidate, reference=reference, metrics=metrics, inferential=False))
    evidence = dict(complete=True, methods=methods, comparisons=old['comparisons']+additions,
        extension_comparisons=additions, source_sha256=sha(__file__),
        plotting_source_sha256=sha(Path(__file__).with_name('expansion_report.py')),
        parent_manifest=dict(file=str(parent/'edge/final/manifest.json'), sha256=sha(parent/'edge/final/manifest.json')),
        extension_manifest=dict(file=str(root/'edge/final/manifest.json'), sha256=sha(root/'edge/final/manifest.json')),
        metric_files={**{str(parent/p): v for p, v in old['metric_files'].items()},
                      **{str(root/p): v for p, v in new['metric_files'].items()}},
        statistics=study['statistics'], performance_eligible=False,
        interpretation='Same ten-scene BF16 consistency experiment; original twelve methods reused unchanged, '
            'two new asymmetric FP4 methods measured against original same-seed BF16 outputs.')
    return study, parent, rows, evidence


def gallery(root, parent, study):
    from PIL import Image
    from judge_video_export import encode, frames_uint8
    from real_video_report import render_grid
    rows = [json.loads(s) for path in (parent/'edge/judge_index.jsonl', root/'edge/judge_index.jsonl')
            for s in path.read_text().splitlines()]
    lookup = {(r['method'], r['case_id']): r for r in rows}
    if len(lookup) != 140 or len(rows) != 140:
        raise ValueError('Missing or duplicate exported videos')
    out = root/'report'
    (out/'grids').mkdir(exist_ok=True)
    page = ['<!doctype html><meta charset="utf-8"><title>FP4 expansion with two-term repairs</title>',
        '<style>body{font:16px system-ui;margin:32px;background:#f5f6fa;color:#172033}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
        'figure{margin:0;background:white;padding:12px}video{width:100%}'
        'details{margin:12px 0}button{padding:8px 16px;margin:4px}</style>',
        '<h1>FP4 expansion: two-term A-only and W-only repairs</h1>'
        '<p>Same five RBench + five PAI-Bench scenes; 320×192, 61 frames, 35 steps. '
        'BF16 is the consistency reference. The original 12 methods are reused unchanged; '
        '20 new videos cover A-only and W-only two-product expansions. '
        'For attention these names mean Q/P-only or K/V-only residual planes.</p>']
    for i, entry in enumerate(study['cases']):
        cid = entry['case']['id']
        sources, videos = {}, {}
        for name in ORDER:
            row = lookup[name, cid]
            meta = json.loads(Path(row['metadata']).read_text())
            source = Path(row['raw_outputs'])
            if sha(source) != meta['provenance']['npz_sha256']:
                raise ValueError('Changed grid source')
            if name in VARIANTS:
                log = Path(meta['generation_log']); target = root/'logs'/(name+'.log')
                if not log.exists():
                    log.symlink_to(target)
                if log.resolve() != target.resolve() or not target.is_file():
                    raise ValueError('Missing full generation log')
            with np.load(source) as data:
                videos[name] = frames_uint8(data['video'], entry['case'])
            sources[name] = dict(file=str(source), sha256=sha(source))
        identity = dict(sources=sources, labels={n: LABELS[n] for n in ORDER},
            renderer_sha256=hashlib.sha256(inspect.getsource(render_grid).encode()).hexdigest())
        target = out/'grids'/(cid+'.mp4'); meta = target.with_suffix('.json')
        if meta.exists():
            previous = json.loads(meta.read_text())
            if previous['identity'] != identity or sha(target) != previous['video']['sha256']:
                raise ValueError('Changed grid identity')
        else:
            grid = render_grid(videos, ORDER, LABELS)
            encoded = encode(grid, target, lossless=False)
            Image.fromarray(grid[len(grid)//2]).save(target.with_suffix('.jpg'), quality=95)
            write(meta, dict(complete=True, identity=identity, video=encoded,
                scope='Viewing only; metrics use raw float32 outputs'))
        print(json.dumps(dict(comparison_grid=cid)), flush=True)
        page += [f'<section id="scene{i}"><h2>{html.escape(cid)} · seed {entry["seeds"][0]}</h2>',
            '<details><summary>Prompt</summary>'+html.escape(entry['case']['prompt'])+'</details>',
            '<details><summary>All 14 methods in one synchronized video</summary>'
            f'<video controls muted loop preload="none" src="grids/{cid}.mp4"></video></details>',
            f'<button onclick="playScene({i})">Restart and play scene</button>'
            f'<button onclick="pauseScene({i})">Pause scene</button><div class="grid">']
        for name in ORDER:
            video = os.path.relpath(lookup[name, cid]['video_path'], out)
            page += ['<figure><figcaption>'+html.escape(LABELS[name])+'</figcaption>',
                '<video controls muted loop preload="none" src="'+html.escape(video, quote=True)+'"></video></figure>']
        page.append('</div></section>')
    page.append('<script>function pauseScene(i){document.querySelectorAll("#scene"+i+" video").forEach(v=>v.pause())}'
        'function playScene(i){document.querySelectorAll("video").forEach(v=>v.pause());'
        'document.querySelectorAll("#scene"+i+" video").forEach(v=>{v.currentTime=0;v.play()})}</script>')
    (out/'gallery.html').write_text('\n'.join(page))
    (out/'all_video_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def report(root, skip_media=False):
    study, parent, rows, evidence = combined(root)
    out = root/'report'; out.mkdir(exist_ok=True)
    write(out/'results.json', evidence)
    with (out/'per_demo.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures(root, study, rows, evidence)
    text = ['# FP4 expansion: two-term single-side repairs', '',
        'The original ten scenes and seeds are unchanged: five RBench and five PAI-Bench, '
        'Edge, 320×192, 61 frames at 24 FPS, 35 steps, CFG 6, flow shift 3, Graph, debug off. '
        'Original BF16, FP8, AWQ and 1/3/4/6/9/12/16-term outputs and scores are reused, '
        'with manifest, request, model, projection-scope and raw-signal hashes checked. '
        'Only 20 additional videos are generated.', '',
        '- A-only: W0A0 + W0A1, one weight plane and two current activation planes.',
        '- W-only: W0A0 + W1A0, two offline weight planes and one activation plane.', '',
        'Both variants retain FP4 attention. A-only repairs Q/P; W-only repairs current K/V. '
        'K/V are online activations, not offline model weights. All base attention blocks remain. '
        'This follows the side naming of the earlier rectangular expansion experiment; '
        'it is not an ablation that changes only projection weights while holding attention fixed.', '',
        '## Paired consistency to the original BF16 outputs', '',
        '| Method | PSNR ↑, dB | SSIM ↑ | LPIPS ↓ | Relative L2 ↓ |',
        '|---|---:|---:|---:|---:|']
    for name, item in evidence['methods'].items():
        v = item['metrics']
        text.append(f"| {LABELS[name]} | {v['psnr_db']['mean']:.4f} | {v['ssim_mean']['mean']:.6f} | {v['lpips']['mean']:.6f} | {v['relative_l2']['mean']:.6f} |")
    text += ['', '![Quality versus FP4 product count](quality_vs_terms.png)', '',
        'Blue includes the A-only two-term point; purple includes the W-only point. '
        'FP8 uses long dashes, AWQ-FP4 short dashes. BF16 reference lines remain for '
        'LPIPS and SSIM; the PSNR panel omits the BF16 self-reference annotation.', '',
        '## Paired changes for the new variants', '',
        'Deltas are candidate minus reference. Scene bootstrap intervals use the same '
        '10 paired scenes and 10,000 resamples; they are exploratory and unadjusted.', '',
        '| Candidate | Reference | ΔLPIPS [95% interval] | Improved scenes | ΔSSIM | ΔPSNR, dB |',
        '|---|---|---:|---:|---:|---:|']
    for c in evidence['extension_comparisons']:
        v = c['metrics']; lo, hi = v['lpips']['ci95']
        text.append(f"| {LABELS[c['candidate']]} | {LABELS[c['reference']]} | {v['lpips']['mean_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {v['lpips']['improved_scenes']}/10 | {v['ssim_mean']['mean_difference']:+.6f} | {v['psnr_db']['mean_difference']:+.4f} |")
    text += ['', '## Stored weight and arithmetic scope', '',
        '| Method | W planes | A planes | Products per controlled GEMM | Packed projection weights, MiB |',
        '|---|---:|---:|---:|---:|']
    for name in ['fp4_base', 'fp4_two_a', 'fp4_two_w', 'fp4_three', 'fp4_four']:
        item = evidence['methods'][name]; spec = item['expansion']
        text.append(f"| {LABELS[name]} | {spec['weight_planes']} | {spec['activation_planes']} | {spec['terms']} | {item['packed_weight_bytes']/2**20:.2f} |")
    text += ['', 'Each runtime loads only its required prefix from the same offline nested weight bank. '
        'W0 and A0 use exactly the same quantization definitions across variants. Packed bytes cover '
        'codes, scales and matrix factors for 168 generation projections, not total model memory. '
        'Two products describe controlled GEMM/QK/PV arithmetic, not whole-request compute or speed. '
        'AR prefill and nonlinearities retain the common BF16 path.', '',
        'The new asymmetric variants passed independent decoded GEMM and full attention references, '
        'M129/M416, GQA/tails, zero inputs and current-input Graph checks (8 tests). A-only attention '
        'also matches the existing ACT2 kernel bitwise. Memcheck and racecheck reported no errors.', '',
        'Metrics use all 61 raw float32 video frames. PSNR is clip-global RGB MSE at data range 1; '
        'SSIM and AlexNet LPIPS v0.1 are frame averages, followed by equal weighting across scenes. '
        'This remains BF16 output consistency, not physical task success or real-video fidelity. '
        'Shared-calibration request timings are ineligible for performance conclusions.', '',
        '- [All 14 methods and synchronized videos](gallery.html)',
        '- [All paired measurements](per_demo.csv)',
        '- [Means, intervals and source hashes](results.json)',
        '- [Qualification logs](../qualification/results.json)',
        '- [New generation manifest](../edge/final/manifest.json)',
        '- [New generation logs](../logs/)',
        '- [Original report]('+os.path.relpath(parent/'report/report.md', out)+')', '']
    (out/'report.md').write_text('\n'.join(text))
    if not skip_media:
        gallery(root, parent, study)
    return dict(complete=True, methods=14, total_videos=140, newly_generated=20, paired_metrics=130,
                report=str(out/'report.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--skip-media', action='store_true')
    args = parser.parse_args()
    print(json.dumps(report(args.root.resolve(), args.skip_media)))

"""Audit and summarize fixed FP4 attention ablations at the paired scene grain."""
import argparse
import collections
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
from baselines import sha, write
from attention_base_study import NAMES, PARENT, SPARSE

METRICS = ('lpips', 'ssim_mean', 'psnr_db', 'relative_l2')
HIGHER = {'ssim_mean', 'psnr_db'}
LABELS = dict(bf16='BF16 reference', fp4_base='Base W4A4',
              linear_a='Linear A only', linear_w='Linear W only',
              linear_both_1p5='Linear A/W each 25% (1.5x linear cap)',
              linear_both_2p0='Linear A/W each 50% (2x linear cap)')


def read(path):
    return json.loads(path.read_text())


def bootstrap_indices(entries, replicates, seed, keys=('benchmark', 'category')):
    """Resample scenes, never individual frames or seed replicates."""
    groups = collections.defaultdict(list)
    for i, entry in enumerate(entries):
        groups[tuple(entry[k] for k in keys)].append(i)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(g, (replicates, len(g)), replace=True)
                           for g in groups.values()], axis=1)


def delta_stats(values, indices, entries, higher=False):
    values = np.asarray(values, dtype=float)
    draws = values[indices].mean(axis=1)
    regression = -values if higher else values
    groups = sorted({(e['benchmark'], e['category']) for e in entries})
    leave_one_out = {}
    for group in groups:
        keep = [(e['benchmark'], e['category']) != group for e in entries]
        if any(keep):
            leave_one_out['/'.join(group)] = float(values[keep].mean())
    return dict(mean_difference=float(values.mean()), median_difference=float(np.median(values)),
                ci95=np.quantile(draws, [.025, .975]).tolist(),
                ci99_bonferroni_five=np.quantile(draws, [.005, .995]).tolist(),
                improved_scenes=int((regression < 0).sum()), tied_scenes=int((regression == 0).sum()),
                worst_scene=entries[int(regression.argmax())]['case']['id'],
                worst_scene_regression=float(regression.max()),
                leave_one_category_out_mean=leave_one_out)


def collect(root, panel):
    folder = root / panel
    manifest = folder / 'edge/final/manifest.json'
    study = read(manifest)
    frozen = read(root / 'study.json')
    if frozen['manifests'][panel] != sha(manifest):
        raise ValueError('Changed frozen manifest')
    from attention_base_runtime import source_identity
    lock = read(root / 'implementation.lock.json')
    if lock['source_sha256'] != source_identity():
        raise ValueError('Changed runtime source')
    for path, digest in study['sources'].items():
        if sha(path) != digest:
            raise ValueError('Changed source manifest: ' + path)
    raw, scenes, audit, costs = [], [], [], collections.defaultdict(list)
    reference_matches = []
    weight_hash = sha(PARENT / 'offline_weights/manifest.json')
    for entry in study['cases']:
        cid = entry['case']['id']
        gold_path = folder / 'edge/final/generation/bf16' / cid / 'result.json'
        gold = read(gold_path)
        c0 = None
        for name in NAMES:
            case_dir = folder / 'edge/final/generation' / name / cid
            receipt_path = case_dir / 'result.json'
            receipt = read(receipt_path)
            if not receipt['complete'] or receipt['identity']['manifest_sha256'] != sha(manifest):
                raise ValueError('Incomplete or different generation')
            for key in ('case', 'image_sha256', 'seeds'):
                if receipt['identity'][key] != gold['identity'][key]:
                    raise ValueError('Unmatched paired input: ' + key)
            for key in ('targets', 'model_identity'):
                if receipt[key] != gold[key]:
                    raise ValueError('Unmatched model/scope')
            if sorted(r['seed'] for r in receipt['rows']) != sorted(entry['seeds']):
                raise ValueError('Missing or duplicated seeds')
            for row in receipt['rows']:
                if (row['steps'], row['calls'], row['debug'], row['execution']) != (35, 70, 'off', 'graph'):
                    raise ValueError('Changed generation protocol')
                if sha(case_dir / row['outputs']) != row['outputs_sha256']:
                    raise ValueError('Corrupt output archive')
                if panel == 'matched10' and name in ('bf16', 'fp4_base'):
                    reference_matches.append(row['source_signal_hashes_match'] is True)
            record = dict(method=name, case_id=cid, receipt_sha256=sha(receipt_path))
            if name == 'bf16':
                audit.append(record)
                continue
            adapter = receipt['adapter']
            if adapter['attention_repair'] is not False or adapter['attention_expansion'] != study['methods'][name]['attention_expansion']:
                raise ValueError('Attention corrections enabled')
            if adapter['source_sha256'] != lock['source_sha256'] or adapter['weight_manifest_sha256'] != weight_hash:
                raise ValueError('Different weight bank or implementation')
            ledger_path = case_dir / 'compute.json'
            ledger = read(ledger_path)
            if not ledger['feasible'] or ledger['attention_ratio'] != 1 or ledger['linear_ratio'] > ledger['linear_budget']:
                raise ValueError('Invalid compute budget')
            if ledger['ratio'] != ledger['total_mma'] / ledger['base_mma']:
                raise ValueError('Incorrect MMA ratio')
            if c0 is not None and c0 != ledger['base_mma']:
                raise ValueError('Different base workload')
            c0 = ledger['base_mma']
            for r in ledger['rows']:
                if r['kind'] == 'attention' and any(r[k] != 0 for k in
                    ('qk_a_mma', 'qk_w_mma', 'pv_a_mma', 'pv_w_mma', 'cross_mma')):
                    raise ValueError('Attention correction MMA detected')
            for r in receipt['rows']:
                if r['compute_sha256'] != sha(ledger_path) or r['actual_mma_ratio'] != ledger['ratio']:
                    raise ValueError('Changed request ledger')
            costs[name].append(dict(case_id=cid, **{k: ledger[k] for k in
                ('base_mma', 'total_mma', 'ratio', 'linear_ratio', 'linear_budget', 'attention_ratio')},
                packed_weight_bytes=adapter['packed_weight_bytes']))
            metric_path = folder / 'edge/final/metrics' / name / (cid + '.json')
            metric = read(metric_path)
            provenance = metric['provenance']
            if not metric['complete'] or provenance['candidate_sha256'] != sha(receipt_path) or provenance['reference_sha256'] != sha(gold_path):
                raise ValueError('Invalid paired metric provenance')
            if provenance['identity']['manifest_sha256'] != sha(manifest):
                raise ValueError('Wrong metric cohort')
            if sorted(r['seed'] for r in metric['rows']) != sorted(entry['seeds']):
                raise ValueError('Missing metric seed')
            values = []
            for r in metric['rows']:
                v = {k: float(r['metrics']['video'][k]) for k in METRICS}
                if not all(np.isfinite(x) for x in v.values()):
                    raise ValueError('Nonfinite quality value')
                values.append(v)
                raw.append(dict(method=name, case_id=cid, benchmark=entry['benchmark'],
                                category=entry['category'], seed=r['seed'], **v))
            scenes.append(dict(method=name, case_id=cid, benchmark=entry['benchmark'],
                               category=entry['category'], seeds=len(values),
                               **{k: float(np.mean([v[k] for v in values])) for k in METRICS}))
            record.update(metric_sha256=sha(metric_path), compute_sha256=sha(ledger_path))
            audit.append(record)
    if reference_matches and not all(reference_matches):
        raise ValueError('Original BF16/base signals are not bitwise reproduced')
    return study, raw, scenes, dict(costs), audit, reference_matches


def old_attention_intervention(study, lookup, indices):
    path = SPARSE / 'report/per_demo.csv'
    with path.open() as stream:
        old = {(r['method'], r['case_id']): r for r in csv.DictReader(stream)}
    values = {}
    for name, previous in [('linear_a', 'fp4_two_a'), ('linear_w', 'fp4_two_w'),
                           ('linear_both_1p5', 'fp4_both_1p5'), ('linear_both_2p0', 'fp4_both_2p0')]:
        values[name] = {}
        for key in METRICS:
            diff = []
            for e in study['cases']:
                cid = e['case']['id']
                before = old[previous, cid]
                if [int(before['seed'])] != e['seeds']:
                    raise ValueError('Old intervention is not seed matched')
                diff.append(lookup[name, cid][key] - float(before[key]))
            values[name][key] = delta_stats(diff, indices, study['cases'], key in HIGHER)
    old_gap, new_gap = [], []
    for e in study['cases']:
        cid = e['case']['id']
        old_gap.append(float(old['fp4_two_a', cid]['lpips']) - float(old['fp4_two_w', cid]['lpips']))
        new_gap.append(lookup['linear_a', cid]['lpips'] - lookup['linear_w', cid]['lpips'])
    return dict(source_csv=str(path), source_sha256=sha(path),
                scope='Whole-trajectory attention intervention; remaining linear policy fixed. Exploratory ten scenes.',
                off_minus_on=values, old_a_minus_w_lpips=float(np.mean(old_gap)),
                new_a_minus_w_lpips=float(np.mean(new_gap)),
                gap_change=delta_stats(np.array(new_gap)-old_gap, indices, study['cases']))


def gallery(folder, study, out):
    index = folder / 'edge/judge_index.jsonl'
    if not index.exists():
        return False
    videos = {(r['method'], r['case_id'], r['seed']): r
              for r in (json.loads(line) for line in index.read_text().splitlines())}
    expected = {(n, e['case']['id'], s) for n in NAMES for e in study['cases'] for s in e['seeds']}
    if not expected <= videos.keys():
        return False
    page = ['<!doctype html><meta charset="utf-8"><title>FP4 without attention repair</title>',
            '<style>body{font:16px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}figure{margin:0}summary{padding:15px;cursor:pointer}</style>',
            '<h1>Fixed base FP4 attention: paired videos</h1>',
            '<p>All FP4 methods disable every Q/P and K/V correction. BF16 is the paired reference. '+
            '1.5x/2x refer to linear MMA caps. MP4 previews are not used to calculate fidelity.</p>']
    for e in study['cases']:
        cid = e['case']['id']
        for seed in e['seeds']:
            page.append('<details><summary>'+html.escape(cid)+f' / seed {seed}</summary><div class="grid">')
            for name in NAMES:
                path = os.path.relpath(videos[name, cid, seed]['video_path'], out)
                page.append('<figure><figcaption>'+html.escape(LABELS[name])+'</figcaption><video controls muted loop preload="none" src="'+html.escape(path, quote=True)+'"></video></figure>')
            page.append('</div></details>')
    (out / 'gallery.html').write_text('\n'.join(page))
    return True


def report(root, panel):
    study, raw, scenes, costs, audit, matches = collect(root, panel)
    lookup = {(r['method'], r['case_id']): r for r in scenes}
    seed_lookup = {(r['method'], r['case_id'], r['seed']): r for r in raw}
    stats = study['statistics']
    # Ten old scenes have one scene per category. Category-stratified resampling
    # is degenerate, so use explicitly exploratory benchmark-stratified intervals.
    keys = ('benchmark',) if panel == 'matched10' else ('benchmark', 'category')
    indices = bootstrap_indices(study['cases'], stats['replicates'], stats['bootstrap_seed'], keys)
    comparisons = []
    for candidate, reference in stats['comparisons']:
        metrics = {}
        for key in METRICS:
            delta = [lookup[candidate, e['case']['id']][key] - lookup[reference, e['case']['id']][key] for e in study['cases']]
            value = delta_stats(delta, indices, study['cases'], key in HIGHER)
            seed_deltas = [(e['case']['id'], s, seed_lookup[candidate, e['case']['id'], s][key] -
                           seed_lookup[reference, e['case']['id'], s][key]) for e in study['cases'] for s in e['seeds']]
            worst = max(seed_deltas, key=lambda x: -x[2] if key in HIGHER else x[2])
            value['worst_seed'] = dict(case_id=worst[0], seed=worst[1], difference=worst[2])
            metrics[key] = value
        guard = all(v['mean_difference'] >= 0 if k in HIGHER else v['mean_difference'] <= 0 for k, v in metrics.items())
        comparisons.append(dict(candidate=candidate, reference=reference, metrics=metrics,
            all_four_mean_guard=guard,
            lpips_familywise_interval_below_zero=panel == 'expanded72' and metrics['lpips']['ci99_bonferroni_five'][1] < 0))
    methods = {}
    breakdown = []
    for name in NAMES[1:]:
        rows = [r for r in scenes if r['method'] == name]
        methods[name] = dict(metrics={k: dict(mean=float(np.mean([r[k] for r in rows])),
                                            median=float(np.median([r[k] for r in rows]))) for k in METRICS},
                             mma_ratio_range=[min(r['ratio'] for r in costs[name]), max(r['ratio'] for r in costs[name])])
        for dimension in ('benchmark', 'category'):
            groups = collections.defaultdict(list)
            for r in rows:
                group = r['benchmark'] if dimension == 'benchmark' else r['benchmark']+'/'+r['category']
                groups[group].append(r)
            for group, subset in groups.items():
                breakdown.append(dict(method=name, dimension=dimension, group=group, scenes=len(subset),
                                      **{k: float(np.mean([r[k] for r in subset])) for k in METRICS}))
    out = root / panel / 'report'
    out.mkdir(exist_ok=True)
    result = dict(complete=True, panel=panel, scenes=len(study['cases']),
                  seeds=sum(len(e['seeds']) for e in study['cases']), methods=methods,
                  comparisons=comparisons, costs=costs, audit=audit, breakdown=breakdown,
                  original_signals_bitwise_match=all(matches) if matches else None,
                  manifest_sha256=sha(root/panel/'edge/final/manifest.json'), source_sha256=sha(__file__),
                  statistics=dict(**stats, executed_strata=list(keys),
                    interval_note='Ten-scene category strata are singletons; intervals stratify by benchmark and are exploratory.' if panel == 'matched10' else 'Frozen category-stratified paired scene bootstrap; 99% intervals adjust the five primary LPIPS contrasts.'),
                  performance_eligible=False, attention_repair=False)
    if panel == 'matched10':
        result['attention_intervention'] = old_attention_intervention(study, lookup, indices)
    result['gallery_complete'] = gallery(root/panel, study, out)
    for filename, rows in [('per_seed.csv', raw), ('per_scene.csv', scenes), ('breakdown.csv', breakdown)]:
        with (out/filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write(out/'results.json', result)
    lines = ['# FP4 attention without repairs: '+panel, '',
        f"Completed {result['scenes']} scenes, {result['seeds']} paired seeds, {result['seeds']*6} generated videos. ",
        'All FP4 candidates use the same dense one-plane attention: no Q/P or K/V repairs and no deleted base blocks. '+
        'Only generation projection and FFN repairs change. W0/W1 and selectors are frozen; no outcome-driven tuning. '+
        '35 steps, 320×192, 61 frames, CFG 6, Graph, debug off. Common BF16 AR prefill/nonlinearities are unchanged.', '',
        '## Quality relative to the same-seed BF16 output', '',
        '| Linear policy | Actual controlled MMA/C0 | LPIPS ↓ | SSIM ↑ | PSNR ↑ (dB) | Relative L2 ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name, method in methods.items():
        v = method['metrics']; lo, hi = method['mma_ratio_range']
        lines.append(f"| {LABELS[name]} | {lo:.6f}–{hi:.6f} | {v['lpips']['mean']:.6f} | {v['ssim_mean']['mean']:.6f} | {v['psnr_db']['mean']:.4f} | {v['relative_l2']['mean']:.6f} |")
    lines += ['', '1.5x/2x are LINEAR-only caps, not whole-request ratios. C0 includes padded FP4 projection/FFN and QK/PV MMA across both CFG branches and all 35 steps; common BF16 AR and non-MMA work are outside this count. '+
              'The sparse ablation still stores full required W planes and packs both activation planes; lower MMA does not establish memory or latency savings.', '',
              '## Paired contrasts', '', result['statistics']['interval_note'], '',
              'Seeds are averaged within each scene before bootstrap. Delta is candidate minus reference. PSNR uses clip-global raw RGB MSE at range 1; LPIPS/SSIM average all 61 frames, including the conditioning frame, matching the original experiment. '+
              'These are BF16 fidelity metrics, not task-success or physical-correctness scores.', '',
              '| Candidate | Reference | ΔLPIPS [95% CI] | 99% CI (five-comparison adjustment) | LPIPS wins | ΔSSIM | ΔPSNR | All four mean guard |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for c in comparisons:
        v = c['metrics']; p = v['lpips']; lo, hi = p['ci95']; a, b = p['ci99_bonferroni_five']
        lines.append(f"| {LABELS[c['candidate']]} | {LABELS[c['reference']]} | {p['mean_difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] | [{a:+.6f}, {b:+.6f}] | {p['improved_scenes']}/{result['scenes']} | {v['ssim_mean']['mean_difference']:+.6f} | {v['psnr_db']['mean_difference']:+.4f} | {c['all_four_mean_guard']} |")
    if panel == 'matched10':
        v = result['attention_intervention']
        lines += ['', '## Matched attention intervention', '',
            'All 20 regenerated BF16/base signal hashes match the original videos bitwise. The old A-side attention repaired Q/P; the old W-side attention repaired current K/V (which are activations).', '',
            f"Original A-minus-W LPIPS gap: {v['old_a_minus_w_lpips']:+.6f}; attention-off gap: {v['new_a_minus_w_lpips']:+.6f}.", '',
            '| Linear policy held fixed | ΔLPIPS (attention off minus on) | ΔSSIM | ΔPSNR |', '|---|---:|---:|---:|']
        for name, metrics in v['off_minus_on'].items():
            lines.append(f"| {LABELS[name]} | {metrics['lpips']['mean_difference']:+.6f} | {metrics['ssim_mean']['mean_difference']:+.6f} | {metrics['psnr_db']['mean_difference']:+.4f} |")
        lines += ['', 'This intervention measures effects through the complete generation trajectory. It does not prove an additive local matrix-error mechanism. The new 72-scene panel is reported separately.']
    lines += ['', '## Scope and artifacts', '',
        'The expanded panel contains 36 scenes from each benchmark, sampled by a fixed hash within category and excluding the original ten, inventoried prior cohort roles, and duplicate decoded images. '+
        'It is new to this ablation, not a claim of new environments or never-used public scenarios. Every failure and seed remains in the CSVs. '+
        'GPU timings are shared-load diagnostics and are not compared as performance results.', '',
        '- [Per-seed metrics](per_seed.csv)', '- [Scene means](per_scene.csv)',
        '- [Dataset/category breakdown](breakdown.csv)', '- [All intervals, worst seeds, leave-one-category-out sensitivity, and cost audit](results.json)',
        '- [Frozen manifest](../edge/final/manifest.json)', '- [Full generation logs](../logs/)',
        '- [Video export index](../edge/judge_index.jsonl)']
    if result['gallery_complete']:
        lines.append('- [Paired video gallery](gallery.html)')
    if (out/'paired_contrasts.pdf').exists():
        lines.append('- [Three-metric paired contrasts](paired_contrasts.pdf)')
    if (out/'attention_on_off.pdf').exists():
        lines.append('- [Matched attention on/off intervention](attention_on_off.pdf)')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return dict(complete=True, panel=panel, report=str(out/'report.md'), gallery_complete=result['gallery_complete'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--panel', choices=('matched10', 'expanded72'), required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root.resolve(), args.panel)))

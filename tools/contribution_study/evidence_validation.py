"""Freeze passing development controls and consume reserved scenes in order.

No fitting occurs here. A validation failure stops the mechanism. Four/Six is
explicitly a nearest-neighbor representation control, not our outlier method.
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from baselines import sha, write
from dataset_report import stratified_interval
from evidence_outlier_gate import FIELDS, DATASETS, paired_rows


def audit(cohort, workspace, role):
    from PIL import Image
    pixels, ids, seeds = set(), set(), set()
    for split, entries in cohort['cases'].items():
        for entry in entries:
            image = Path(entry['case']['image'])
            if sha(image) != entry['image_sha256']:
                raise ValueError('Changed frozen input')
            with Image.open(image) as im:
                im = im.convert('RGB')
                digest = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            identity = (entry['benchmark'], entry['source_id'])
            if digest != entry['pixel_sha256'] or digest in pixels or identity in ids or seeds.intersection(entry['seeds']):
                raise ValueError('Cohort source/pixel/seed overlap')
            pixels.add(digest)
            ids.add(identity)
            seeds.update(entry['seeds'])
    files = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', '*.npz', str(workspace)], capture_output=True, text=True)
    if files.returncode not in (0, 1) or files.stderr:
        raise RuntimeError('Local saved-output history audit failed')
    reserved = cohort['cases'][role]
    names = {r['case']['id'] for r in reserved}
    stems = {f'seed{s}' for r in reserved for s in r['seeds']}
    collisions = [p for p in files.stdout.splitlines() if Path(p).stem in stems or names.intersection(Path(p).parts)]
    if collisions:
        raise ValueError('Reserved inputs already have local outputs: '+str(collisions[:3]))
    return dict(complete=True, role=role, saved_npz_checked=len(files.stdout.splitlines()),
                pixel_source_seed_disjoint=True, scope='Local saved-output audit plus frozen source/decoded-pixel separation; no remote or semantic-near-duplicate guarantee.')


def prepare(a):
    cohort = json.loads(a.cohort.read_text())
    development = a.root/'outlier_development'
    old = json.loads((development/'manifest.json').read_text())
    gate = json.loads(a.gate.read_text())
    if not gate['complete'] or gate['manifest_sha256'] != sha(development/'manifest.json') or not cohort['frozen']:
        raise ValueError('Verified development decision and frozen cohort required')
    chosen = [name for item in gate['selections'].values() for name in item['candidates']]
    if not chosen:
        raise ValueError('No candidate passed development; do not consume validation')
    names = list(dict.fromkeys(['bf16', 'fp8', *[gate['comparisons'][name]['reference'] for name in chosen], *chosen]))
    methods = {name: copy.deepcopy(old['methods'][name]) for name in names}
    contrasts = [dict(candidate=name, reference=ref) for name in chosen
                 for ref in (gate['comparisons'][name]['reference'], 'fp8')]
    selection = dict(frozen=True, candidates=chosen, methods=methods, development_gate_sha256=sha(a.gate),
        cohort_sha256=sha(a.cohort), source_sha256=sha(__file__), primary_comparisons=contrasts,
        statistics=dict(replicates=10000, bootstrap_seed=180918, family_size=len(DATASETS)*len(contrasts)),
        scope='Frozen before any reserved validation output. Four/Six is a representation baseline, not an outlier novelty claim. '
              'No recalibration, support reordering, schedule adjustment or secondary-metric substitution.')
    output = a.root/'validation'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'selection.json').exists():
        if json.loads((output/'selection.json').read_text()) != selection:
            raise ValueError('Changed frozen validation selection')
        return
    checked = audit(cohort, a.workspace, 'validation')
    write(output/'audit.json', checked)
    write(output/'selection.json', selection)
    manifest = copy.deepcopy(old)
    manifest.update(role='validation', cases=cohort['cases']['validation'], methods=methods,
        selection_sha256=sha(output/'selection.json'), statistics=selection['statistics'],
        primary_comparisons=contrasts, selection_eligible=True,
        scope='Reserved 32 scenes, two/category and two seeds/scene. Frozen development choice; no fitting on these outputs.')
    write(output/'manifest.json', manifest)


def report(folder):
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    progress = json.loads((folder/'metrics/progress.json').read_text())
    if not progress['complete'] or progress['identity']['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Complete matched reserved metrics required')
    by_seed, by_scene, raw, costs = {}, {}, [], {}
    for name, spec in manifest['methods'].items():
        ratios, peaks = [], []
        for entry in manifest['cases']:
            case = entry['case']['id']
            receipt_path = folder/'generation'/name/case/'result.json'
            receipt = json.loads(receipt_path.read_text())
            if not receipt['complete'] or {r['seed'] for r in receipt['rows']} != set(entry['seeds']):
                raise ValueError('Missing reserved output')
            for row in receipt['rows']:
                if row['performance_eligible'] or row['steps'] != 35 or row['debug'] != 'off':
                    raise ValueError('Changed execution conditions')
                if row['actual_mma_ratio'] is not None:
                    if row['actual_mma_ratio'] > spec['config']['compute_budget']:
                        raise ValueError('Exceeded request budget')
                    ratios.append(row['actual_mma_ratio'])
                peaks.append(row['peak_allocated_bytes'])
            if name == 'bf16':
                continue
            data = json.loads((folder/'metrics'/name/(case+'.json')).read_text())
            if (not data['complete'] or data['provenance']['identity'] != progress['identity'] or
                data['provenance']['candidate_sha256'] != sha(receipt_path) or
                data['provenance']['reference_sha256'] != sha(folder/'generation/bf16'/case/'result.json') or
                len(data['rows']) != len(entry['seeds']) or {r['seed'] for r in data['rows']} != set(entry['seeds'])):
                raise ValueError('Changed or unpaired reserved metrics')
            raw.append(data)
            key = entry['benchmark'], name
            for row in data['rows']:
                by_seed.setdefault(key, {})[case, row['seed']] = row['metrics']['video']
            by_scene.setdefault(key, {})[case] = dict(category=entry['category'],
                metrics={field: float(np.mean([r['metrics']['video'][field] for r in data['rows']])) for field in FIELDS})
        costs[name] = dict(ratio_range=[min(ratios), max(ratios)] if ratios else None,
                           peak_allocated_bytes=[min(peaks), max(peaks)])
    summaries = {b: {} for b in DATASETS}
    for (benchmark, name), rows in by_scene.items():
        categories = {r['category'] for r in rows.values()}
        summaries[benchmark][name] = {field: float(np.mean([np.mean([r['metrics'][field] for r in rows.values()
            if r['category'] == category]) for category in categories])) for field in FIELDS}
    contrasts = []
    stats = manifest['statistics']
    for pair in manifest['primary_comparisons']:
        name, reference = pair['candidate'], pair['reference']
        for benchmark in DATASETS:
            rows, old = by_scene[benchmark, name], by_scene[benchmark, reference]
            paired = paired_rows(by_seed[benchmark, name], by_seed[benchmark, reference])
            for field in FIELDS:
                groups = collections.defaultdict(list)
                for case, row in rows.items():
                    groups[row['category']].append(row['metrics'][field]-old[case]['metrics'][field])
                if any(len(v) < 2 for v in groups.values()):
                    raise ValueError('At least two scenes/category required for reserved bootstrap')
                paired[field].update(stratified_interval(groups, replicates=stats['replicates'],
                    seed=stats['bootstrap_seed'], family=stats['family_size']))
            passed = (paired['lpips']['familywise_ci'][1] < 0 and paired['ssim_mean']['mean_delta'] >= 0 and
                      paired['relative_l2']['mean_delta'] <= 0)
            contrasts.append(dict(**pair, benchmark=benchmark, metrics=paired, passed=passed))
    eligible = [name for name in dict.fromkeys(r['candidate'] for r in contrasts)
        if all(r['passed'] for r in contrasts if r['candidate'] == name and r['reference'] != 'fp8')]
    result = dict(complete=True, role=manifest['role'], manifest_sha256=sha(manifest_path),
        summaries=summaries, costs=costs, contrasts=contrasts, final_eligible=eligible, raw=raw,
        statistics=stats, performance_eligible=False,
        scope='Scene-clustered, category-stratified paired bootstrap; seeds averaged within scene. '
              'FP8 comparisons are separate from equal-budget representation improvement. '
              'Only independent final data can confirm a selected validation result.')
    dest = folder/'report'
    dest.mkdir(exist_ok=True)
    write(dest/'quality.json', result)
    lines = ['# 冻结表示对照：'+manifest['role'], '', result['scope'], '',
        '| 数据集 | 配置 | LPIPS ↓ | SSIM ↑ | L2 ↓ | PSNR dB ↑ |', '|---|---|---:|---:|---:|---:|']
    for benchmark, rows in summaries.items():
        for name, values in rows.items():
            lines.append('| '+benchmark+' | '+name+' | '+' | '.join(f'{values[f]:.6f}' for f in FIELDS)+' |')
    lines += ['', '| 数据集 | 候选 / 对照 | LPIPS 差值 | 95% CI | 校正 CI | 全门槛通过 |', '|---|---|---:|---|---|---|']
    for row in contrasts:
        v = row['metrics']['lpips']
        lines.append(f"| {row['benchmark']} | {row['candidate']} / {row['reference']} | {v['mean_delta']:+.6f} | {v['ci95']} | {v['familywise_ci']} | {row['passed']} |")
    lines += ['', '进入独立最终测试的候选：'+str(eligible), '', '未通过的候选停止，不继续在这些场景上调参。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def execute(a):
    root = a.root/'validation'
    selection = json.loads((root/'selection.json').read_text())
    if selection['source_sha256'] != sha(__file__):
        raise ValueError('Changed frozen validation implementation')
    while True:
        status = json.loads(a.after.read_text()) if a.after.exists() else {}
        if status.get('paused'):
            raise RuntimeError('Predecessor paused; do not overlap GPU experiments')
        if status.get('complete'):
            break
        time.sleep(5)
    from importance_workflow import run
    run(root/'manifest.json', root)
    result = report(root)
    archive = a.archive/'validation'
    archive.mkdir(parents=True, exist_ok=True)
    for name, file in (('manifest', root/'manifest.json'), ('selection', root/'selection.json'),
                       ('audit', root/'audit.json'), ('quality', root/'report/quality.json')):
        (archive/(name+'.json.gz')).write_bytes(gzip.compress(file.read_bytes(), mtime=0))
    (archive/'REPORT.md').write_bytes((root/'report/REPORT.md').read_bytes())
    write(root/'decision.json', dict(complete=True, final_eligible=result['final_eligible'],
        quality_sha256=sha(root/'report/quality.json'), final_consumed=False,
        scope='Final selection must be frozen before any final generation. No fitting on validation.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'run', 'report'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workspace', type=Path)
    p.add_argument('--cohort', type=Path)
    p.add_argument('--gate', type=Path)
    p.add_argument('--after', type=Path)
    p.add_argument('--archive', type=Path)
    a = p.parse_args()
    if a.operation == 'prepare':
        if not all((a.workspace, a.cohort, a.gate)):
            p.error('prepare needs workspace, cohort and gate')
        prepare(a)
    elif a.operation == 'run':
        if not all((a.after, a.archive)):
            p.error('run needs predecessor and archive')
        execute(a)
    else:
        report(a.root/'validation')


if __name__ == '__main__':
    main()

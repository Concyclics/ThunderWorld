"""Standalone scientific figures from audited attention-off paired reports."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/thunderworld-attention-ablation-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from baselines import sha, write

LABELS = {'linear_a': 'A only', 'linear_w': 'W only',
          'linear_both_1p5': 'Both 25%', 'linear_both_2p0': 'Both 50%'}


def draw(report):
    source = report / 'results.json'
    data = json.loads(source.read_text())
    if not data['complete'] or data['attention_repair']:
        raise ValueError('Require complete audited attention-off results')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    comparisons = data['comparisons']
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.7), sharey=True)
    fig.subplots_adjust(left=.17, right=.985, bottom=.22, top=.77, wspace=.23)
    names = [LABELS[c['candidate']]+' − '+LABELS[c['reference']] for c in comparisons]
    for ax, key, label in zip(axes, ('lpips', 'ssim_mean', 'psnr_db'),
                              ('Δ LPIPS (lower is better)', 'Δ SSIM (higher is better)', 'Δ PSNR, dB (higher is better)')):
        ax.axvline(0, color='#555b63', linestyle='--', linewidth=1, zorder=1)
        for i, comparison in enumerate(comparisons):
            stats = comparison['metrics'][key]
            center = stats['mean_difference']
            lo, hi = stats['ci95']
            if key == 'lpips':
                outer = stats['ci99_bonferroni_five']
                ax.plot(outer, [i, i], color='#b37b31', linewidth=1.2, zorder=2)
            ax.plot([lo, hi], [i, i], color='#25749f', linewidth=3, solid_capstyle='butt', zorder=3)
            ax.plot(center, i, 'o', color='#174b6a', markersize=5, zorder=4)
        ax.set_xlabel(label)
        ax.grid(axis='x', color='#e6e8ea', linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    fig.suptitle(f"Attention repair disabled · {data['scenes']} scenes, {data['seeds']} paired seeds", y=.96, fontsize=13)
    note = 'Mean paired scene difference; thick blue: 95% CI; thin gold (LPIPS): 99% CI for five contrasts.'
    if data['panel'] == 'matched10':
        note += '\nExploratory benchmark-stratified bootstrap; original ten scenes.'
    else:
        note += '\nBenchmark × category stratified bootstrap; two seeds averaged within each scene.'
    fig.text(.5, .84, note, ha='center', va='center', fontsize=9)
    fig.text(.5, .025, 'Both 25% / 50%: fixed linear A/W repair quotas (1.5× / 2× linear caps). All dense FP4 attention base blocks retained.', ha='center', fontsize=8.5)
    paths = []
    for suffix in ('pdf', 'png', 'svg'):
        path = report / ('paired_contrasts.'+suffix)
        fig.savefig(path, dpi=190)
        paths.append(path)
    plt.close(fig)
    if data['panel'] == 'matched10':
        old_path = Path(data['attention_intervention']['source_csv'])
        old_rows = list(csv.DictReader(old_path.open()))
        pairs = [('linear_a', 'fp4_two_a'), ('linear_w', 'fp4_two_w'),
                 ('linear_both_1p5', 'fp4_both_1p5'), ('linear_both_2p0', 'fp4_both_2p0')]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
        fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.77, wspace=.24)
        for ax, key, label in zip(axes, ('lpips', 'ssim_mean', 'psnr_db'), ('LPIPS ↓', 'SSIM ↑', 'PSNR ↑, dB')):
            for i, (new, old) in enumerate(pairs):
                before = np.mean([float(r[key]) for r in old_rows if r['method'] == old])
                after = data['methods'][new]['metrics'][key]['mean']
                ax.plot([i-.13, i+.13], [before, after], color='#92979d', linewidth=1.1)
                ax.plot(i-.13, before, 's', color='#b37b31', label='Attention repair on' if i == 0 else None)
                ax.plot(i+.13, after, 'o', color='#25749f', label='Attention repair off' if i == 0 else None)
            ax.set_xticks(range(4), ['A\nonly', 'W\nonly', 'Both\n25%', 'Both\n50%'])
            ax.set_ylabel(label)
            ax.grid(axis='y', color='#e6e8ea', linewidth=.6)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .9), ncol=2, frameon=False, fontsize=9)
        fig.suptitle('Matched attention intervention · original 10 scenes', y=.99, fontsize=13)
        fig.text(.5, .025, 'Same linear policies and BF16 reference; points are scene means. Old A-side attention repaired Q/P; old W-side repaired K/V.', ha='center', fontsize=8)
        for suffix in ('pdf', 'png', 'svg'):
            path = report / ('attention_on_off.'+suffix)
            fig.savefig(path, dpi=190)
            paths.append(path)
        plt.close(fig)
    write(report/'figure_sources.json', dict(report_sha256=sha(source), script_sha256=sha(__file__),
        statistic_unit='scene; not frame or seed', files={p.name: sha(p) for p in paths}))
    print(json.dumps(dict(figures=[str(p) for p in paths])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    draw(parser.parse_args().report.resolve())

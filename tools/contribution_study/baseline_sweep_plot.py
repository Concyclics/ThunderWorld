"""Standalone plots of completed quality and concurrency evidence; no peak claim."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--throughput-unit', choices=('videos', 'fps'), default='videos')
    a = p.parse_args()
    data = json.loads((a.archive/'summary.json').read_text())
    if not data['complete'] or data['performance_eligible']:
        raise ValueError('Expected a completed diagnostic summary')
    fps = a.throughput_unit == 'fps'
    frame_counts = {}
    if fps:
        manifest = json.loads(gzip.decompress((a.archive/'manifest.json.gz').read_bytes()))
        frame_counts = {e['case']['id']: e['case']['frames'] for e in manifest['cases']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter
    import numpy as np
    labels = [('bf16', 'BF16'), ('fp8', 'FP8'), ('fp4_base', 'W4A4'),
              ('awq_marlin', 'AWQ-lite'), ('arcquant', 'ARC50'), ('arcquant100', 'ARC100'),
              ('sharq_perf', 'SharQ'), ('b1.5_protected_n', 'Ours 1.5x'),
              ('b2_joint', 'Ours 2x'), ('b2.5_ffn_guarded', 'Ours 2.5x')]
    colors = ['#333333', '#888888', '#9467bd', '#8c564b', '#9ecae1', '#2171b5',
              '#e377c2', '#e6550d', '#31a354', '#006d2c']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, dataset in zip(axes, ('rbench', 'paibench')):
        for i, (name, label) in enumerate(labels):
            curve = data['throughput'][dataset+'|'+name]
            rows = curve['rows']
            factor = frame_counts[curve['case_id']] if fps else 1
            ax.plot([r['batch'] for r in rows], [r['median_videos_per_second']*factor for r in rows],
                    label=label, color=colors[i], marker='o', markersize=4,
                    linestyle='-' if name.startswith('b') else '--', linewidth=1.7)
        ax.set(xscale='log', xticks=[1, 2, 4, 8], xlabel='Concurrent complete requests',
               title='RBench representative' if dataset == 'rbench' else 'PAI-Bench representative',
               ylim=(0, None))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Total output frames / second (FPS)' if fps else 'Decoded videos / second (higher is better)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(.5, -.005))
    fig.suptitle('RTX PRO 6000 diagnostics: 35 steps, 61 frames, 192 x 320', y=.995)
    fig.text(.5, .14, 'Fixed implementations. B > 1: speed only. Independent model per request; not merged-M batching.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .16, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(a.archive/(('throughput_fps.' if fps else 'throughput.')+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    if fps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ('rbench', 'paibench')):
        names = labels[1:]
        values = [data['quality'][dataset][name]['lpips'] for name, _ in names]
        ax.bar(np.arange(len(names)), values, color=colors[1:])
        ax.set(xticks=np.arange(len(names)), xticklabels=[label for _, label in names],
               title='RBench: 9 scenes x 2 seeds' if dataset == 'rbench' else 'PAI-Bench: 7 scenes x 2 seeds',
               ylim=(0, max(v for d in data['quality'].values() for m, q in d.items()
                            for v in [q['lpips']])*1.12))
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('LPIPS to matched BF16 output (lower is better)')
    fig.suptitle('Frozen development cohort: single-request quality', y=.995)
    fig.text(.5, -.015, 'Descriptive means, existing scenes; not a final quality qualification or concurrency-quality claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(a.archive/('quality.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

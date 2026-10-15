"""Paper figures: absolute resident memory for targeted generation weights."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('Thor', 'PRO6000'), default='Thor')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    import numpy as np
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.labelsize': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    report = json.loads(a.input.read_text())
    methods = ('bf16', 'fp8', 'fp4_base', 'b1.5_protected_n', 'b2_joint')
    labels = ('BF16', 'FP8', 'W4A4', 'Repair\n1.5×', 'Repair\n2×')
    resident = {(r['model'], r['method']): r for r in report['resident']
                if r['device'] == a.device}
    # Keep the backend's additional scale layout explicit, without double counting.
    parts = (
        ('base_payload_bytes', 'Base weights', '#346C9C'),
        ('base_metadata_bytes', 'Scales & metadata', '#D8B45B'),
        ('extra_scale_layout_bytes', 'Extra scale layout', '#68A39B'),
        ('w1_bytes', 'W1 & indices', '#C9744B'),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))
    fig.subplots_adjust(left=.066, right=.985, bottom=.22, top=.85, wspace=.24)
    handles = [Patch(facecolor=color, label=label) for _, label, color in parts]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995),
               ncol=4, frameon=False, fontsize=10.5, handlelength=1.8,
               columnspacing=2.0)
    plot_data = []
    for ax, model in zip(axes, ('edge', 'nano')):
        rows = []
        for method in methods:
            r = resident[model, method]
            repair = sum(r[k] for k in (
                'w1_codes', 'w1_scales', 'w1_columns', 'w1_locations',
                'w1_extra_offsets',
            ))
            metadata = sum(r[k] for k in (
                'fp8_scale', 'w0_scales', 'prescale', 'sensitivity',
                'outer_w0_factor', 'base_csr_offsets',
            ))
            row = dict(
                device=a.device, model=model, method=method,
                scope='targeted_generation_weights', display_unit='GiB',
                base_payload_bytes=r['base_codes'],
                base_metadata_bytes=metadata,
                extra_scale_layout_bytes=r['extra_w0_scale_layout'],
                w1_bytes=repair,
                target_total_bytes=r['target_resident_bytes'],
            )
            assert all(row[key] >= 0 for key, _, _ in parts)
            assert sum(row[key] for key, _, _ in parts) == row['target_total_bytes']
            rows.append(row)
        plot_data.extend(rows)
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods))
        for key, _, color in parts:
            values = np.array([r[key] for r in rows]) / 2**30
            ax.bar(x, values, .68, bottom=bottom, color=color,
                   edgecolor='white', linewidth=.45, zorder=3)
            bottom += values
        ymax = max(bottom) * 1.13
        for i, value in enumerate(bottom):
            ax.annotate(f'{value:.3f} GiB', (i, value), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=9.5)
        for i in (2, 3, 4):
            payload = rows[i]['base_payload_bytes'] / 2**30
            ax.text(i, payload / 2, f'{payload:.3f}\nGiB', color='white',
                    ha='center', va='center', fontsize=9, linespacing=1.05,
                    fontweight='semibold')
        ax.set_ylim(0, ymax)
        ax.set_xlim(-.6, 4.6)
        ax.set_xticks(x, labels)
        ax.set_xlabel(model.title(), labelpad=9, fontweight='semibold')
        ax.set_ylabel('Target weights (GiB)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.grid(axis='y', color='#E5E7EA', linewidth=.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#9A9A9A')
        ax.tick_params(axis='both', length=3, color='#9A9A9A')
    a.output.mkdir(parents=True, exist_ok=True)
    stem = a.output / ('memory_breakdown_' + a.device.lower())
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=200, facecolor='white')
    plt.close(fig)
    stem.with_suffix('.chart-data.json').write_text(json.dumps(plot_data, indent=2) + '\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()

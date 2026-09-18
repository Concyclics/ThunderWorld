"""Publication-exportable figures from verified paper-study result JSON.

Development means are descriptive. No repeated head/step measurements are
plotted as independent statistical samples, and no shared timing is plotted.
"""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    out = a.archive/'figures'
    out.mkdir(parents=True, exist_ok=True)
    cross = json.loads((a.archive/'cross/cross_interventions.json').read_text())
    mot = json.loads((a.archive/'mot/mot.json').read_text())
    # Use the source study's schema without inventing confidence intervals.
    rows = {r['benchmark']+'|'+r['candidate']+'|'+r['reference']: r['metrics']
            for r in cross['comparisons']}
    labels = [('without_cross_projection', 'Projection'), ('without_cross_ffn', 'FFN'),
              ('without_cross_qk', 'QK'), ('without_cross_pv', 'PV'), ('fp4_three', 'All')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, benchmark, title in zip(axes, ('rbench', 'paibench'), ('RBench', 'PAI-Bench')):
        values = [rows[benchmark+'|'+name+'|fp4_four']['lpips']['mean_delta'] for name, _ in labels]
        colors = ['#b85251' if value > 0 else '#277a78' for value in values]
        ax.bar(range(len(values)), values, color=colors)
        ax.axhline(0, color='#444444', linewidth=.8)
        ax.set_xticks(range(len(values)), [label for _, label in labels])
        ax.set_title(title)
        ax.set_xlabel('Cross term removed (other assets/support fixed)')
    axes[0].set_ylabel('LPIPS change versus full four terms')
    fig.suptitle('Cross terms have different full-generation effects', y=1.02)
    fig.tight_layout()
    save(fig, out/'cross_terms')
    variants = [('rtn', 'RTN'), ('four_six', 'Four/Six'), ('magnitude', 'Magnitude'),
                ('quant_error', 'Quant. error'), ('output_joint', 'Output search')]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, budget in zip(axes, ('2', '2.5')):
        data = json.loads((a.archive/('outlier_b'+budget)/'outlier.json').read_text())
        summary = data['summary']
        base = summary['rtn']['three_objective']
        vals = [(summary[name]['three_objective']/base-1)*100 for name, _ in variants]
        ax.bar(range(5), vals, color=['#687386', '#277a78', '#b85251', '#b85251', '#7567a4'])
        ax.axhline(0, color='#444444', linewidth=.8)
        ax.set_xticks(range(5), [label for _, label in variants], rotation=20, ha='right')
        ax.set_title('Fixed '+budget+'x parent support')
        ax.set_ylim(-17.5, 1.5)
        for i, value in enumerate(vals):
            ax.annotate(f'{value:+.4f}%', (i, value), xytext=(0, -13 if value < -1 else 4),
                        textcoords='offset points', ha='center', fontsize=8)
    axes[0].set_ylabel('Local native output objective change (%)')
    fig.suptitle('Representation controls: calibration error is not video quality', y=1.02)
    fig.tight_layout()
    save(fig, out/'outlier_local')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    interventions = [('qk_zero', 'Drop QK repair'), ('pv_zero', 'Drop PV repair'),
        ('uniform_native_half', 'Native half'), ('current_v_residual_half', 'V-residual half'),
        ('drop_ar_pure', 'Drop pure AR'), ('drop_dm_matched', 'Drop matched DM'),
        ('drop_mixed_boundary', 'Drop boundary')]
    for ax, benchmark, title in zip(axes, ('rbench', 'paibench'), ('RBench', 'PAI-Bench')):
        data = mot['summary'][benchmark]['mean_nmse_delta']
        ax.barh(range(len(interventions)), [data[name] for name, _ in interventions], color='#416c9e')
        ax.set_yticks(range(len(interventions)), [label for _, label in interventions])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Full attention NMSE increase vs. full three terms')
    fig.suptitle('Current-student interventions: 16 scenes, four phases, two CFG branches', y=1.02)
    fig.tight_layout()
    save(fig, out/'mot_interventions')
    qualities = {label: json.loads(gzip.decompress((a.archive/label/'quality.json.gz').read_bytes()))
                 for label in ('external', 'awq_native', 'outlier_quality')}
    if not all(value['complete'] for value in qualities.values()):
        raise ValueError('Complete baseline quality required')
    methods = [('external', 'fp4_base', 'Single FP4'),
               ('awq_native', 'awq_marlin', 'AWQ-lite Marlin'),
               ('external', 'arcquant', 'ARCQuant 50%'),
               ('external', 'sharq', 'SharQ native'),
               ('outlier_quality', 'b2_joint_outlier_four_six', '2x Four/Six control'),
               ('external', 'fp8', 'Native FP8'),
               ('external', 'fp4_three', 'Full three terms'),
               ('external', 'fp4_four', 'Full four terms')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, benchmark, title in zip(axes, ('rbench', 'paibench'), ('RBench', 'PAI-Bench')):
        values = [qualities[source]['summaries'][benchmark][name]['lpips'] for source, name, _ in methods]
        # Verified imported FP8 is an inexpensive check against accidental cohort mixing.
        if len({value['summaries'][benchmark]['fp8']['lpips'] for value in qualities.values()}) != 1:
            raise ValueError('Unmatched baseline comparison cohorts')
        ax.barh(range(len(methods)), values, color=['#687386']*4+['#277a78', '#416c9e', '#7567a4', '#7567a4'])
        ax.set_yticks(range(len(methods)), [label for _, _, label in methods])
        ax.set_title(title)
        ax.set_xlabel('Video LPIPS relative to BF16 (lower is closer)')
        ax.set_xlim(0, .35)
        for i, value in enumerate(values):
            ax.text(value+.004, i, f'{value:.3f}', va='center', fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle('Complete generation: fixed development configurations', y=1.02)
    fig.tight_layout()
    save(fig, out/'baseline_quality')
    caption = (
        '# Figure provenance and interpretation\n\n'
        '- `cross_terms`: 16 development scenes x 2 seeds; candidate minus full four terms. '
        'No error bars: one scene per category cannot support the planned stratified interval. '
        'The same change may have opposite effects on the two datasets; do not sum isolated deltas.\n'
        '- `outlier_local`: all 168 operators x 16 calibration panels, averaged operator objectives. '
        'Identical prescale, A selection, W1 support and tensor payload within each parent. '
        'Output search is bounded to one reserved block per operator, not an exhaustive oracle.\n'
        '- `mot_interventions`: current b2 student inputs, same-scene crossed sampling; first average '
        'layers/phases/CFG within scene, then scenes. All base blocks remain. RBench pure-AR and '
        'matched-DM bars are zero because these requests have no complete AR tile. V-residual '
        'selection has separately recorded diagnostic packing/decoding cost; not a speed baseline.\n'
        '- `baseline_quality`: same 16 development scenes x 2 seeds, 35 steps. Matched frozen '
        'inputs and projection range; external methods retain their own arithmetic and attention '
        'precision. AWQ-lite inherits its historical calibration; ARCQuant uses fixed 50% channels, '
        'not its model-wide proportion search. These are adapter configurations, not a ranking of '
        'each paper\'s best model. Four/Six is a control. No development significance claim; '
        'SSIM/L2/PSNR and independent validation are reported separately.\n'
        '- All figures concern BF16 consistency. They do not establish task success, global Pareto '
        'optimality, independent test significance or shared-GPU acceleration.\n')
    (out/'CAPTIONS.md').write_text(caption)


def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == '__main__':
    main()

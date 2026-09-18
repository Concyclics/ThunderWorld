"""CPU diagnostic: is the worst represented-weight error an amplitude extreme?

Uses frozen base codes, not a new calibration or an output-gain oracle. Ties are
handled as sets. A mismatch supports a ranking distinction, not successful
outlier removal or full-generation improvement.
"""
import argparse
import collections
import json
from pathlib import Path
from baselines import sha, write


def counts(weight, base):
    w, error = weight.float().reshape(-1, 16), (weight.float()-base.float()).abs().reshape(-1, 16)
    peak = error.amax(-1, keepdim=True)
    worst = error == peak
    nonzero = peak[:, 0] > 0
    amplitude = w.abs() == w.abs().amax(-1, keepdim=True)
    extrema = (w == w.amax(-1, keepdim=True)) | (w == w.amin(-1, keepdim=True))
    return dict(groups=len(w), nonzero_error_groups=int(nonzero.sum()),
        worst_error_excludes_absolute_max=int((nonzero & ~(worst & amplitude).any(-1)).sum()),
        worst_error_excludes_signed_min_and_max=int((nonzero & ~(worst & extrema).any(-1)).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'assets', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    capture = json.loads((a.capture/'manifest.json').read_text())
    assets = json.loads((a.assets/'manifest.json').read_text())
    if not capture['complete'] or not assets['complete'] or capture['model_identity'] != assets['model_identity']:
        raise ValueError('Matching complete frozen sources required')
    import torch
    from thunderworld_fp4.cosmos.calibration import decode
    torch.set_num_threads(8)
    rows = []
    with torch.inference_mode():
        for name, entry in assets['modules'].items():
            source = capture['weights'][name]
            wp, ap = a.capture/source['file'], a.assets/entry['file']
            if sha(wp) != source['sha256'] or sha(ap) != entry['sha256']:
                raise ValueError('Changed weight or frozen FP4 representation')
            w = torch.load(wp, map_location='cpu', weights_only=True)
            asset = torch.load(ap, map_location='cpu', weights_only=True)
            transformed = (w.float()/asset['prescale']).bfloat16()
            represented = decode(asset['codes_base'], asset['scales_base'], asset['factor'])
            row = counts(transformed, represented)
            rows.append(dict(name=name, family='ffn' if '.mlp_moe_gen.' in name else 'projection', **row))
    summary = {}
    for family in ('all', 'projection', 'ffn'):
        total = collections.Counter()
        for row in rows:
            if family == 'all' or row['family'] == family:
                total.update({k: v for k, v in row.items() if isinstance(v, int)})
        values = dict(total)
        for field in ('worst_error_excludes_absolute_max', 'worst_error_excludes_signed_min_and_max'):
            values[field+'_fraction'] = values[field]/values['nonzero_error_groups']
        summary[family] = values
    result = dict(complete=True, rows=rows, summary=summary, source_sha256=sha(__file__),
        assets_sha256=sha(a.assets/'manifest.json'), capture_sha256=sha(a.capture/'manifest.json'),
        performance_eligible=False,
        scope='CPU BF16-transformed weights and decoded frozen W0, complete 168-operator inventory. '
              'K16 groups weighted by group count; ties counted as sets, zero-error groups excluded from fractions. '
              'Diagnostic represented-weight error, not activation-weighted output contribution, native kernel quality, '
              'successful outlier removal, or a new quantization novelty claim.')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'weight_ranking.json', result)
    lines = ['# Weight error is not amplitude ranking', '', result['scope'], '',
        '| Family | Nonzero-error groups | Worst error excludes abs-max | Worst error excludes signed min/max |', '|---|---:|---:|---:|']
    for family, row in summary.items():
        lines.append(f"| {family} | {row['nonzero_error_groups']} | {row['worst_error_excludes_absolute_max_fraction']:.2%} | {row['worst_error_excludes_signed_min_and_max_fraction']:.2%} |")
    (a.output/'WEIGHT_RANKING.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

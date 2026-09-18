"""Within-phase repair response/head ranking verification, not schedule fitting."""
import argparse
import collections
import gzip
import json
from pathlib import Path

from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    from scipy.stats import rankdata
    manifest = json.loads((a.calibration/'manifest.json').read_text())
    definitions = json.loads((a.calibration/'definitions.json').read_text())
    if not manifest['complete'] or manifest['definitions_sha256'] != sha(a.calibration/'definitions.json'):
        raise ValueError('Complete immutable temporal pilot required')
    if len(manifest['identity']['sampled_steps']) != 12 or not manifest['identity']['project_attention']:
        raise ValueError('Requires three timepoints and common Wo projection')
    records = []; samples = []; source_hashes = {}

    def correlation(x, y):
        rx, ry = rankdata(x), rankdata(y)
        return None if np.std(rx) == 0 or np.std(ry) == 0 else float(np.corrcoef(rx, ry)[0, 1])

    def top_overlap(x, y):
        left = set(sorted(range(len(x)), key=lambda i: (-x[i], i))[:8])
        right = set(sorted(range(len(y)), key=lambda i: (-y[i], i))[:8])
        return len(left & right)/8

    for case, digest in manifest['cases'].items():
        path = a.calibration/(case+'.json')
        if sha(path) != digest:
            raise ValueError('Changed calibration case')
        source_hashes[case] = digest
        data = json.loads(path.read_text())
        if not data['complete']:
            raise ValueError('Incomplete current trajectory')
        grouped = collections.defaultdict(list)
        for step, groups in data['samples'].items():
            for key, row in groups.items():
                if row['step'] != int(step) or not row['baseline_bitwise_equal']:
                    raise ValueError('Changed current-input/base output')
                grouped[key].append(row)
        for key, values in grouped.items():
            values.sort(key=lambda r: r['step'])
            if len(values) != 3:
                raise ValueError('Incomplete within-phase samples')
            kind, branch, phase, name = key.split('|', 3)
            meta = dict(case=case, dataset=data['entry']['benchmark'], kind=kind, branch=int(branch), phase=int(phase), name=name)
            labels = ('qk50', 'pv50') if kind == 'attention' else ('up_a_high', 'down_a_high') if kind == 'ffn' else ('a_high',)
            for label in labels:
                # Positive = harm from this change. No additivity assumption.
                field = 'projected_losses' if kind == 'attention' else 'losses'
                deltas = [r[field][label]-r[field]['base'] for r in values]
                for i in (1, 2):
                    records.append(dict(meta, label=label, first_step=values[0]['step'], other_step=values[i]['step'],
                        first_delta=deltas[0], other_delta=deltas[i],
                        relative_change=abs(deltas[i]-deltas[0])/max(abs(deltas[i]), abs(deltas[0]), 1e-12),
                        sign_flip=deltas[i]*deltas[0]<0))
            if kind == 'attention':
                for label in ('qk50', 'pv50'):
                    def gains(row, field):
                        return np.asarray(row[field][label])-np.asarray(row[field]['base'])
                    first = gains(values[0], 'projected_head_errors')
                    for i, row in enumerate(values):
                        projected = gains(row, 'projected_head_errors'); raw = gains(row, 'head_errors')
                        samples.append(dict(meta, label=label, step=row['step'],
                            energy_spearman=correlation(row['q_head_energy'], projected),
                            raw_vs_projected_spearman=correlation(raw, projected),
                            raw_vs_projected_top8_overlap=top_overlap(raw, projected),
                            first_vs_current_spearman=correlation(first, projected) if i else None,
                            first_vs_current_top8_overlap=top_overlap(first, projected) if i else None,
                            head_benefits=projected.tolist()))
    summary = {}
    for kind in ('projection', 'ffn', 'attention'):
        rows = [r for r in records if r['kind'] == kind]
        summary[kind] = dict(comparisons=len(rows), median_relative_response_change=float(np.median([r['relative_change'] for r in rows])),
            response_change_gt_25pct=sum(r['relative_change']>.25 for r in rows)/len(rows),
            sign_flip_rate=sum(r['sign_flip'] for r in rows)/len(rows))
    head_summary = {}
    for key in ('energy_spearman', 'raw_vs_projected_spearman', 'raw_vs_projected_top8_overlap',
                'first_vs_current_spearman', 'first_vs_current_top8_overlap'):
        vals = [r[key] for r in samples if r[key] is not None]
        head_summary[key] = dict(mean=float(np.mean(vals)), median=float(np.median(vals)), count=len(vals))
    result = dict(complete=True, calibration_sha256=sha(a.calibration/'manifest.json'), source_sha256=sha(__file__),
        source_hashes=source_hashes, identity=manifest['identity'], response=summary, heads=head_summary,
        records=records, head_records=samples, performance_eligible=False,
        scope='Four calibration scenes, five layers, two CFG branches, first/middle/last in four phases. '
              'Changes measured on current b2 trajectories. Head-isolated projected gains do not add to joint head gains. '
              'No new schedule is selected, and no full-Transformer/CFG/rollout or held-out quality claim.')
    a.output.mkdir(parents=True, exist_ok=True)
    detail_path=a.output/'temporal_records.json.gz'
    detail_path.write_bytes(gzip.compress(json.dumps(dict(records=result.pop('records'),head_records=result.pop('head_records'))).encode(),mtime=0))
    result['detail']=dict(file=detail_path.name,sha256=sha(detail_path))
    write(a.output/'temporal.json', result)
    for case in manifest['cases']:
        (a.output/(case+'.json.gz')).write_bytes(gzip.compress((a.calibration/(case+'.json')).read_bytes(), mtime=0))
    (a.output/'calibration_manifest.json').write_text((a.calibration/'manifest.json').read_text())
    (a.output/'definitions.json.gz').write_bytes(gzip.compress((a.calibration/'definitions.json').read_bytes(), mtime=0))
    print(json.dumps(dict(response=summary, heads=head_summary), indent=2))


if __name__ == '__main__':
    main()

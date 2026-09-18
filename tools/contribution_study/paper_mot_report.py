"""Scene-level summaries of current-input MoT repair interventions."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import spearmanr
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.study/'manifest.json').read_text())
    if not manifest['complete'] or manifest['identity'].get('smoke') or len(manifest['cases']) != 16:
        raise ValueError('Complete crossed scene/phase/CFG sampling required')
    cases, raw = {}, []
    for case, digest in manifest['cases'].items():
        file = a.study/(case+'.json')
        if sha(file) != digest:
            raise ValueError('Changed intervention panel')
        data = json.loads(file.read_text())
        if not data['complete'] or len(data['panels']) != 224:
            raise ValueError('Missing scene x 4 phases x 2 CFG x 28 layers')
        raw.append(data)
        deltas, correlations = collections.defaultdict(list), collections.defaultdict(list)
        roles, changes = [], collections.defaultdict(list)
        for row in data['panels'].values():
            roles.append(row['roles'])
            for label, value in row['errors'].items():
                deltas[label].append(float(np.sum(value)-np.sum(row['errors']['three'])))
            for side in ('qk', 'pv'):
                gain = np.asarray(row['errors'][side+'_zero'])-np.asarray(row['errors']['three'])
                role_gain = np.asarray(row['errors']['drop_ar_pure'])-np.asarray(row['errors']['three'])
                for feature in ('ar_mass', 'ar_density', 'dm_density', 'q_head_energy', 'v_head_rms', 'v_residual_energy'):
                    for target, values in ((side+'_full_repair', gain), ('ar_pure_repair', role_gain)):
                        if target == 'ar_pure_repair' and (side != 'qk' or not row['roles']['pure_ar']):
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            rho = float(spearmanr(row[feature], values).statistic)
                        if np.isfinite(rho):
                            correlations[target+'.'+feature].append(rho)
            for field, value in row['adjacent_step_change'].items():
                changes[field].append(value)
        cases[case] = dict(benchmark=data['entry']['benchmark'], category=data['entry']['category'],
            mean_nmse_delta={k: float(np.mean(v)) for k, v in deltas.items()},
            within_panel_head_correlations={k: dict(mean=float(np.mean(v)), finite_panels=len(v)) for k, v in correlations.items()},
            panel_count=len(roles), no_pure_ar_panels=sum(not r['pure_ar'] for r in roles),
            mean_adjacent_relative_l2={k: float(np.mean(v)) for k, v in changes.items()})
    summary = {}
    for dataset in ('rbench', 'paibench'):
        subset = [r for r in cases.values() if r['benchmark'] == dataset]
        labels = sorted(set().union(*(r['mean_nmse_delta'] for r in subset)))
        summary[dataset] = dict(scenes=len(subset),
            mean_nmse_delta={k: float(np.mean([r['mean_nmse_delta'][k] for r in subset])) for k in labels},
            no_pure_ar_panel_fraction=sum(r['no_pure_ar_panels'] for r in subset)/sum(r['panel_count'] for r in subset))
    a.output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=True, manifest_sha256=sha(a.study/'manifest.json'), source_sha256=sha(__file__),
        cases=cases, summary=summary, performance_eligible=False,
        scope='One seed per scene. Repeated heads/layers/steps are averaged within scene, not independent samples. '
              'Local current-student intervention errors, not video quality, phase policy selection or latency evidence.')
    write(a.output/'mot.json', result)
    (a.output/'mot_panels.json.gz').write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
    lines = ['# Current-input MoT repair interventions', '', result['scope'], '',
        '| Intervention | RBench ΔNMSE | PAI ΔNMSE |', '|---|---:|---:|']
    for label in summary['rbench']['mean_nmse_delta']:
        lines.append('| '+label+' | '+' | '.join(f"{summary[b]['mean_nmse_delta'][label]:+.8g}" for b in ('rbench', 'paibench'))+' |')
    lines += ['', 'Pure AR and matched DM removal have equal issued attention MMA. Mixed boundary removal is reported separately.',
              'The explicit residual-energy selector includes extra diagnostic packing/decoding; it is not a timing-qualified implementation.',
              'Dense FP32 feature QK is separately recorded and must never be hidden in a deployment compute budget.']
    (a.output/'MOT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

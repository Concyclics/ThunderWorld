"""Frozen paper interventions using the existing multi-scene inference workflow.

Development evidence only. This entry never promotes a configuration, consumes
held-out scenes, or qualifies shared-GPU request times as performance results.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path

from baselines import sha, write


FAMILIES = ('projection', 'ffn', 'qk', 'pv')
FIELDS = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')


def cross_variant(four, family):
    if family not in FAMILIES:
        raise ValueError('Unknown cross-term family')
    value = copy.deepcopy(four)
    cfg = value['config']
    if cfg.get('budget_plan') or value.get('assets'):
        raise ValueError('This experiment requires the common full RTN anchor')
    if not all(cfg[key]['cross'] for key in ('ffn', 'projection')):
        raise ValueError('Expected a full four-term anchor')
    attention = cfg['attention']['default']
    if cfg['attention']['rules'] or attention['terms'] != 4 or attention.get('pv_terms') not in (None, 4):
        raise ValueError('Expected four-term QK and PV without overrides')
    if family in ('projection', 'ffn'):
        cfg[family]['cross'] = False
    elif family == 'qk':
        attention.update(terms=3, pv_terms=4)
    else:
        attention['pv_terms'] = 3
    # Upstream receipt hashes describe the parent, not the changed variant.
    value = {key: item for key, item in value.items() if not key.startswith('origin_')}
    value.update(intervention=dict(disabled_cross=family, parent='fp4_four'),
                 scope='Only this cross family is disabled. W0/W1, prescale, all base blocks and first-order repair are unchanged.')
    return value


def prepare(args):
    from thunderworld_fp4.cosmos.settings import RunConfig
    source = json.loads((args.previous / 'manifest.json').read_text())
    anchors = json.loads((args.anchors / 'manifest.json').read_text())
    cohort = json.loads(args.cohort.read_text())
    if not cohort['frozen'] or source['cases'] != cohort['cases']['development']:
        raise ValueError('Use the complete frozen development split, not selected scenes')
    if source['model'] != anchors['model'] or source['model'] != cohort['model']:
        raise ValueError('Changed model identity')
    if source['steps'] != 35 or anchors['steps'] != 35:
        raise ValueError('Keep 35 denoising steps')
    args.output.mkdir(parents=True, exist_ok=False)
    reused = ('bf16', 'fp8', 'fp4_three', 'single_a_act2', 'single_w_act2', 'b1.5_protected_n', 'b2_joint')
    methods = {name: source['methods'][name] for name in reused}
    methods['fp4_four'] = anchors['methods']['fp4_four']
    for family in FAMILIES:
        methods['without_cross_' + family] = cross_variant(methods['fp4_four'], family)
    methods['fp4_base'] = anchors['methods']['fp4_base']
    if args.awq:
        awq = json.loads((args.awq / 'manifest.json').read_text())
        if awq['model'] != source['model']:
            raise ValueError('AWQ model mismatch')
        methods['awq'] = awq['methods']['awq']
    for spec in methods.values():
        config = RunConfig.from_dict(spec['config'])
        if config.debug != 'off' or config.execution != 'graph' or config.resource_mode != 'shared-calibration':
            raise ValueError('Changed shared quality execution')
    pairs = [dict(candidate='without_cross_' + f, reference='fp4_four', family=f) for f in FAMILIES]
    pairs += [dict(candidate='fp4_three', reference='fp4_four', family='all')]
    manifest = dict(schema=1, frozen=True, role='development', model=source['model'],
        cases=source['cases'], methods=methods, steps=35, debug='off', execution='graph',
        resource_mode='shared-calibration', selection_eligible=False, performance_eligible=False,
        cohort_sha256=sha(args.cohort), primary_comparisons=pairs,
        parent_manifest_sha256=sha(args.previous / 'manifest.json'),
        anchor_manifest_sha256=sha(args.anchors / 'manifest.json'),
        imports={name: str(args.previous.resolve()) for name in reused},
        scope='Predeclared module cross-term interventions on all 16 development scenes x 2 seeds. '
              'One scene per category: paired means and worst-seed changes only, no confirmatory CI. '
              'Do not add isolated LPIPS deltas; full-three measures the joint removal. '
              'No final/validation scenes or new policy selection. AWQ is a decoded quality reference.')
    write(args.output / 'manifest.json', manifest)
    write(args.output / 'protocol.json', dict(complete=True, source_sha256=sha(__file__),
        hypotheses={f: 'Removing this cross family trades measured quality for reduced MMA' for f in FAMILIES},
        statistical_unit='scene; average both seeds before dataset/category means',
        historical_results_reused=True, calibration_changes=False, changed_kernels=False,
        withheld_counts={key: len(cohort['cases'][key]) for key in ('validation', 'final')},
        new_videos=(len(methods)-len(reused))*2*len(source['cases'])))
    print(json.dumps(dict(complete=True, methods=list(methods), new_methods=[n for n in methods if n not in reused])))


def execute(args):
    from importance_workflow import import_method, run
    manifest = args.output / 'manifest.json'
    data = json.loads(manifest.read_text())
    probe = json.loads((args.output / 'probe.json').read_text())
    if not probe['complete'] or probe['source_sha256'] != sha(Path(__file__).with_name('paper_probe.py')) or len(probe['rows']) != 12:
        raise ValueError('Qualify all mixed attention configurations before generation')
    for name, origin in data['imports'].items():
        if not (args.output / 'generation' / name / 'progress.json').exists():
            import_method(Path(origin), args.output, name, manifest)
    run(manifest, args.output)


def report(args):
    """Reuse provenance-checked existing reporting, then add paired interventions."""
    import subprocess
    import sys
    import numpy as np
    root = args.output
    archive = args.archive or root / 'report'
    subprocess.run([sys.executable, str(Path(__file__).with_name('importance_report.py')),
                    '--run', str(root), '--archive', str(archive), '--label', 'paper'], check=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    values = {}
    for name in manifest['methods']:
        if name == 'bf16':
            continue
        for entry in manifest['cases']:
            case = entry['case']['id']
            data = json.loads((root / 'metrics' / name / (case + '.json')).read_text())
            if not data['complete'] or {r['seed'] for r in data['rows']} != set(entry['seeds']):
                raise ValueError('Incomplete paired metrics')
            values[name, case] = {r['seed']: r['metrics']['video'] for r in data['rows']}
    comparisons = []
    for pair in manifest['primary_comparisons']:
        for benchmark in ('rbench', 'paibench'):
            metrics = {}
            entries = [e for e in manifest['cases'] if e['benchmark'] == benchmark]
            for field in FIELDS:
                rows = []
                for entry in entries:
                    case = entry['case']['id']
                    for seed in entry['seeds']:
                        delta = values[pair['candidate'], case][seed][field] - values[pair['reference'], case][seed][field]
                        rows.append(dict(case=case, seed=seed, delta=delta))
                sign = 1 if field in ('lpips', 'relative_l2') else -1
                metrics[field] = dict(mean_delta=float(np.mean([r['delta'] for r in rows])),
                    worst_seed=max(rows, key=lambda r: sign*r['delta']),
                    regression_fraction=float(np.mean([sign*r['delta'] > 0 for r in rows])), rows=rows)
            comparisons.append(dict(**pair, benchmark=benchmark, metrics=metrics))
    result = dict(complete=True, comparisons=comparisons, manifest_sha256=sha(root / 'manifest.json'),
        source_sha256=sha(__file__), performance_eligible=False, confirmatory=False,
        scope='Development-only paired contrasts, one scene per category. No artificial within-stratum bootstrap significance.')
    write(archive / 'cross_interventions.json', result)
    lines = ['# Module cross-term interventions', '', result['scope'], '',
        '| Removed cross | Dataset | ΔLPIPS | ΔSSIM | Δrelative L2 | ΔPSNR dB |',
        '|---|---|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append('| ' + ' | '.join([row['family'], row['benchmark']] +
            [f"{row['metrics'][f]['mean_delta']:+.6f}" for f in FIELDS]) + ' |')
    lines += ['', 'Each row holds all other repair families fixed. The all-family row is directly measured, not a sum of the four interventions.',
              'All differences are relative to the same BF16 reference through the corresponding metric; lower LPIPS/L2 and higher SSIM/PSNR are better.',
              'This experiment does not prove equivalence, low-budget feasibility, or request speedup.']
    (archive / 'CROSS_INTERVENTIONS.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'run', 'report'))
    p.add_argument('--output', type=Path, required=True)
    for key in ('previous', 'anchors', 'cohort', 'awq', 'archive'):
        p.add_argument('--' + key, type=Path)
    args = p.parse_args()
    if args.operation == 'prepare' and any(getattr(args, key) is None for key in ('previous', 'anchors', 'cohort')):
        p.error('prepare requires --previous, --anchors and --cohort')
    {'prepare': prepare, 'run': execute, 'report': report}[args.operation](args)


if __name__ == '__main__':
    main()

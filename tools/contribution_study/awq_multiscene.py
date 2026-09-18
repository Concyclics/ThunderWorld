"""Extend the frozen 72-scene cohort with the existing official AWQ quality assets."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from baselines import sha, write
from importance_workflow import import_method, run


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def prepare(args):
    parent = args.previous / 'manifest.json'
    old = json.loads(parent.read_text())
    assets = json.loads((args.assets / 'manifest.json').read_text())
    if not old['frozen'] or len(old['cases']) != 72 or old['steps'] != 35:
        raise ValueError('Expected the frozen 72-scene, 35-step cohort')
    if not assets['complete'] or len(assets['modules']) != 168:
        raise ValueError('Complete all-layer AWQ calibration required')
    if not all(entry['bitwise_export'] and entry['activation_bits'] == 16 and entry['weight_bits'] == [2, 1]
               for entry in assets['modules'].values()):
        raise ValueError('Expected qualified official NVFP4 W4A16 exports')
    for entry in assets['modules'].values():
        if sha(args.assets / entry['file']) != entry['sha256']:
            raise ValueError('Changed AWQ tensor asset')
    model = Path(old['model'])
    if any(sha(model / name) != digest for name, digest in assets['model_identity'].items()):
        raise ValueError('AWQ model mismatch')
    for entry in old['cases']:
        if len(entry['seeds']) != 2 or sha(entry['case']['image']) != entry['image_sha256']:
            raise ValueError('Changed paired cohort or input image')
    value = copy.deepcopy(old)
    methods = value['methods']
    config = copy.deepcopy(methods['bf16']['config'])
    methods['awq'] = dict(config=config, baseline_adapter=dict(method='awq', assets=str(args.assets.resolve()),
        assets_sha256=sha(args.assets / 'manifest.json'), implementation_sha256=sha(HERE / 'external.py')),
        scope='Official ModelOpt NVFP4 AWQ-lite, A quantizers disabled before calibration; '
              'decoded BF16 weights with calibrated current-input prescale, BF16 attention. Quality reference only.',
        performance_eligible=False, storage_comparison_eligible=False)
    contrasts = [dict(candidate=name, reference='awq') for name in methods if name not in ('bf16', 'awq')]
    value.update(parent_manifest_sha256=sha(parent), imports={name: str(args.previous.resolve()) for name in old['methods']},
        primary_comparisons=contrasts, role='fixed_cohort_awq_supplement', selection_eligible=False,
        awq_calibration_manifest_sha256=sha(args.assets / 'manifest.json'),
        scope='AWQ supplement on the unchanged historical 72-scene cohort, two identical seeds per scene. '
              'No calibration, search or selection; not a new unseen final cohort. '
              'Historical controls reused only after matching configuration/input/output hashes. '
              'Shared PRO6000 quality only; AWQ decoded storage and timing are ineligible as W4 deployment results.')
    value['statistics'].update(family_size=2 * len(contrasts), reference='awq',
        hypothesis_scope='Supplementary paired comparisons frozen before new AWQ generation; '
                         'historical candidate results were already observed. Historical FP8 gates are unchanged.')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'manifest.json', value)
    write(args.output / 'assets_audit.json', dict(complete=True, manifest_sha256=sha(args.assets / 'manifest.json'),
        modules=len(assets['modules']), model_identity=assets['model_identity'],
        bitwise_export_all=True, provenance=assets['provenance'], scope=assets['scope']))
    print(json.dumps(dict(complete=True, cases=len(value['cases']), new_videos=144, methods=len(methods))))


def command(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def probe(args):
    root = args.output
    path = root / 'manifest.json'
    study = json.loads(path.read_text())
    result = dict(complete=False, manifest_sha256=sha(path), checks={}, performance_eligible=False)
    write(root / 'probe.json', result)
    for method in ('bf16', 'awq'):
        folder = root / 'probe' / method
        command(HERE / 'dataset_infer.py', path, '--method', method, '--output', folder, '--smoke')
        progress = json.loads((folder / 'progress.json').read_text())
        if not progress.get('roundtrip_bitwise_equal'):
            raise ValueError('A -> different scene -> A changed output')
        if method == 'bf16':
            previous = Path(study['imports']['bf16'])
            for entry in (study['cases'][0], study['cases'][36]):
                case = entry['case']['id']
                new = json.loads((folder / case / 'result.json').read_text())
                old = json.loads((previous / 'generation/bf16' / case / 'result.json').read_text())
                if new['model_identity'] != old['model_identity'] or new['targets'] != old['targets']:
                    raise ValueError('Historical BF16 model/scope changed')
                if {r['seed']: r['sha256'] for r in new['rows']} != {r['seed']: r['sha256'] for r in old['rows']}:
                    raise ValueError('Current BF16 runtime differs from historical references')
        result['checks'][method] = dict(scene_roundtrip_bitwise=True,
            historical_output_bitwise=True if method == 'bf16' else None,
            progress_sha256=sha(folder / 'progress.json'), prepared_sha256=sha(folder / 'prepared.json'))
        write(root / 'probe.json', result)
    result['complete'] = True
    write(root / 'probe.json', result)


def import_metrics(source, root, method):
    """Reuse identical CPU metric outputs, retaining original provenance separately."""
    old_manifest = source / 'manifest.json'
    manifest = root / 'manifest.json'
    study = json.loads(manifest.read_text())
    prior = json.loads((source / 'metrics/progress.json').read_text())
    identity = dict(manifest_sha256=sha(manifest), source_sha256=sha(HERE / 'dataset_metrics.py'),
                    quality_sha256=sha(REPO / 'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'))
    if not prior['complete'] or prior['identity']['manifest_sha256'] != sha(old_manifest):
        raise ValueError('Incomplete or changed historical metrics')
    for key in ('source_sha256', 'quality_sha256'):
        if prior['identity'][key] != identity[key]:
            raise ValueError('Metric semantics changed; recompute metrics instead of reusing')
    for entry in study['cases']:
        case = entry['case']['id']
        original = source / 'metrics' / method / (case + '.json')
        value = json.loads(original.read_text())
        if not value['complete'] or value['provenance']['identity'] != prior['identity']:
            raise ValueError('Changed historical metric receipt')
        for key, name in (('reference_sha256', 'bf16'), ('candidate_sha256', method)):
            if value['provenance'][key] != sha(source / 'generation' / name / case / 'result.json'):
                raise ValueError('Historical metrics do not match generated outputs')
        value['imported_from'] = dict(path=str(original), sha256=sha(original), provenance=value['provenance'])
        value['provenance'] = dict(identity=identity,
            reference_sha256=sha(root / 'generation/bf16' / case / 'result.json'),
            candidate_sha256=sha(root / 'generation' / method / case / 'result.json'))
        target = root / 'metrics' / method / (case + '.json')
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError('Changed imported metric')
        write(target, value)


def execute(args):
    root = args.output
    manifest = root / 'manifest.json'
    value = json.loads(manifest.read_text())
    qualification = json.loads((root / 'probe.json').read_text())
    if not qualification['complete'] or qualification['manifest_sha256'] != sha(manifest):
        raise ValueError('Matched-runtime and scene-switch qualification required')
    for name, source in value['imports'].items():
        import_method(Path(source), root, name, manifest)
    for name, source in value['imports'].items():
        if name != 'bf16':
            import_metrics(Path(source), root, name)
    run(manifest, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'probe', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    {'prepare': prepare, 'probe': probe, 'run': execute}[args.operation](args)


if __name__ == '__main__':
    main()

"""Archive auditable paper evidence without copying models or generated videos."""
import argparse
import gzip
import json
from pathlib import Path

from baselines import sha, write


def compressed(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    return dict(source=str(source.resolve()), source_sha256=sha(source), archived_sha256=sha(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    a = p.parse_args()
    root, archive = a.root.resolve(), a.archive.resolve()
    if not json.loads((root/'pipeline.json').read_text())['complete']:
        raise ValueError('Mechanism pipeline is incomplete')
    if not json.loads((root/'awq_native/orchestrator.json').read_text())['complete']:
        raise ValueError('Native AWQ qualification is incomplete')
    archive.mkdir(parents=True, exist_ok=True)
    copies, studies, samples = {}, {}, []
    labels = ['cross', 'outlier_development', 'external', 'awq_native']
    if (root/'validation/selection.json').exists():
        decision = json.loads((root/'validation/decision.json').read_text())
        if not decision['complete']:
            raise ValueError('Frozen validation is not complete')
        labels.append('validation')
    if (root/'final/manifest.json').exists():
        if not json.loads((root/'final/orchestrator.json').read_text())['complete']:
            raise ValueError('Frozen final run is not complete')
        labels.append('final')
    for label in labels:
        source = root/label
        manifest = json.loads((source/'manifest.json').read_text())
        quality_path = source/'report/quality.json'
        quality = json.loads(quality_path.read_text())
        if not quality['complete'] or quality['manifest_sha256'] != sha(source/'manifest.json'):
            raise ValueError('Changed or incomplete study metrics')
        folder = archive/label
        folder.mkdir(exist_ok=True)
        for name, file in (('manifest', source/'manifest.json'), ('quality', quality_path),
                           ('metric_progress', source/'metrics/progress.json')):
            copies[label+'/'+name] = compressed(file, folder/(name+'.json.gz'))
        if label in ('validation', 'final'):
            lines = ['# Paired worst-seed results', '',
                'Candidate minus reference. Worst direction is largest for LPIPS/L2 and smallest for SSIM/PSNR. '
                'These extrema describe robustness; they are not additional independent significance tests. '
                'All seeds remain in the primary scene-clustered analysis.', '',
                '| Dataset | Candidate / reference | Metric | Mean delta | Worst delta | Degraded seeds | Case | Seed |',
                '|---|---|---|---:|---:|---:|---|---:|']
            for contrast in quality['contrasts']:
                for metric, values in contrast['metrics'].items():
                    lines.append(f"| {contrast['benchmark']} | {contrast['candidate']} / {contrast['reference']} | "
                        f"{metric} | {values['mean_delta']:+.6f} | {values['worst_delta']:+.6f} | "
                        f"{values['degraded_seeds']}/{values['seeds']} | {values['worst_case']} | {values['worst_seed']} |")
            (folder/'WORST_SEEDS.md').write_text('\n'.join(lines)+'\n')
        receipts = []
        new_outputs = reused_outputs = 0
        for method in manifest['methods']:
            progress = json.loads((source/'generation'/method/'progress.json').read_text())
            if not progress['complete']:
                raise ValueError('Incomplete method: '+method)
            for entry in manifest['cases']:
                case = entry['case']['id']
                run = source/'generation'/method/case
                file = run/'result.json'
                result = json.loads(file.read_text())
                if not result['complete'] or {r['seed'] for r in result['rows']} != set(entry['seeds']):
                    raise ValueError('Incomplete scene outputs')
                for row in result['rows']:
                    output = run/row['outputs']
                    if sha(output) != row['outputs_sha256']:
                        raise ValueError('Changed generated video: '+str(output))
                    samples.append(dict(study=label, method=method, case=case, benchmark=entry['benchmark'],
                        category=entry['category'], seed=row['seed'], path=str(output), sha256=row['outputs_sha256'],
                        receipt_sha256=sha(file), performance_eligible=False,
                        reused=bool(progress.get('imported_from'))))
                if progress.get('imported_from'):
                    reused_outputs += len(result['rows'])
                else:
                    new_outputs += len(result['rows'])
                receipt = dict(method=method, case=case, sha256=sha(file), result=result)
                if (run/'compute.json').exists():
                    receipt['compute'] = json.loads((run/'compute.json').read_text())
                    receipt['compute_sha256'] = sha(run/'compute.json')
                receipts.append(receipt)
        packed = json.dumps(receipts, ensure_ascii=False).encode()
        (folder/'receipts.json.gz').write_bytes(gzip.compress(packed, mtime=0))
        studies[label] = dict(cases=len(manifest['cases']), methods=len(manifest['methods']),
            outputs=sum(len(r['result']['rows']) for r in receipts),
            new_outputs=new_outputs, reused_outputs=reused_outputs,
            summaries=quality['summaries'], costs=quality['costs'],
            role=manifest['role'], performance_eligible=False, confirmatory=manifest['role'] in ('validation', 'final'))
    for label, source in {
        'pipeline': root/'pipeline.json', 'native_awq_workflow': root/'awq_native/orchestrator.json',
        'native_awq_operators': root/'awq_native/operators.json',
        'native_awq_assets': root/'awq_native/assets/manifest.json',
        'native_awq_scene_roundtrip': root/'awq_native/smoke/progress.json',
        'awq_scale_memcheck': root/'awq_scale_memcheck.json',
        'awq_scale_racecheck': root/'awq_scale_racecheck.json',
        'awq_sanitizer_completion': root/'awq_sanitizers.json',
        'awq_installed_sources': root/'awq_environment.json',
        'awq_rejected_export': root/'awq_native/rejected_torch_scale_export/orchestrator.json',
        'cross_probe': root/'cross/probe.json', 'mot_manifest': root/'mot/manifest.json',
        'cohort': a.workspace/'repair_redesign_20260917/cohort.json',
        'baseline_sources': a.workspace/'contribution_baselines_20260917/sources.lock.json',
        'exclusive_retest_manifest': root/'retest.json',
        'exclusive_pro6000_commands': root/'retest_pro6000/commands.json',
        'exclusive_5090_commands': root/'retest_5090/commands.json',
    }.items():
        copies[label] = compressed(source, archive/(label+'.json.gz'))
    here = Path(__file__).resolve().parent
    sources = {p.name: sha(p) for pattern in ('paper_*.py', 'evidence_*.py', 'test_*evidence*.py') for p in here.glob(pattern)}
    sources.update({p: sha(here/p) for p in ('awq_marlin.py', 'awq_scales.py', 'awq_scales_probe.py',
                   'test_awq_marlin.py', 'dataset_infer.py', 'test_paper_study.py', 'test_weight_diagnostics.py')})
    for name in sources:
        copies['source/'+name] = compressed(here/name, archive/'source'/(name+'.gz'))
    for label, file in (('validation_decision', root/'validation/decision.json'),
                        ('final_workflow', root/'final/orchestrator.json')):
        if file.exists():
            if not json.loads(file.read_text())['complete']:
                raise ValueError('Pending study decision: '+str(file))
            copies[label] = compressed(file, archive/(label+'.json.gz'))
    remaining = ['Exclusive two-device operator/request timing',
                 'Matched execution fusion/overlap controls for a quality-qualified new policy',
                 'Second model/task and deployment-resolution qualification']
    if 'validation' not in labels:
        remaining.append('Independent validation for development-passing representations')
    elif decision['final_eligible'] and 'final' not in labels:
        remaining.append('Independent final confirmation for validation-passing representations')
    result = dict(complete=True, paper_submission_ready=False, studies=studies, archive=copies,
        source_sha256=sources, performance_eligible=False,
        held_out_validation_consumed='validation' in labels, held_out_final_consumed='final' in labels,
        remaining=remaining,
        scope='New evidence supplement, not a claim that every proposed paper contribution succeeded. '
              'Full-generation samples are indexed by hashes and stay outside Git. '
              'Imported controls are counted as comparisons, never as newly generated experiments.')
    write(archive/'evidence.json', result)
    write(archive/'samples.json', dict(complete=True, rows=samples, scope='No videos or model weights are embedded.'))
    write(archive/'files.json', dict(complete=True, files={str(p.relative_to(archive)):
        dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(archive.rglob('*'))
        if p.is_file() and p != archive/'files.json'},
        scope='Integrity inventory for the completed archive; excludes this self-referential index.'))
    print(json.dumps(dict(complete=True, comparisons=len(samples), archive=str(archive))))


if __name__ == '__main__':
    main()

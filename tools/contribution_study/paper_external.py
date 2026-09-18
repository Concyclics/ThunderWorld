"""Matched development comparisons through the existing official baseline adapters.

Each external request delegates to external_infer.py. Eager ARCQuant and decoded
AWQ are quality comparisons only; neither is admitted to the native speed table.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from baselines import sha, write
from importance_workflow import import_method


def prepare(a):
    old = json.loads((a.previous/'manifest.json').read_text())
    cohort = json.loads(a.cohort.read_text())
    if not cohort['frozen'] or old['cases'] != cohort['cases']['development'] or cohort['model'] != old['model']:
        raise ValueError('Complete frozen development split required')
    a.output.mkdir(parents=True, exist_ok=False)
    original_capture = json.loads((a.capture/'manifest.json').read_text())
    if not original_capture['complete'] or original_capture['identity']['cohort_sha256'] != sha(a.cohort):
        raise ValueError('ARC channel priors must use calibration scenes only')
    rows = []
    for row in original_capture['rows']:
        file = a.capture/row['file']
        if sha(file) != row['sha256']:
            raise ValueError('Corrupt calibration panel')
        rows.append(dict(row, kind='linear', file=str(file.resolve())))
    write(a.output/'arc_capture/manifest.json', dict(complete=True, rows=rows,
        model_identity=original_capture['model_identity'], parent_sha256=sha(a.capture/'manifest.json'),
        scope='Same sixteen calibration scenes; full M, one phase/CFG pair per scene. Not exhaustive phase coverage.'))
    retained = ('bf16', 'fp8', 'fp4_base', 'fp4_three', 'fp4_four', 'awq')
    methods = {name: old['methods'][name] for name in retained}
    for name in ('arcquant', 'sharq'):
        cfg = copy.deepcopy(methods['bf16']['config'])
        cfg['execution'] = 'eager'
        methods[name] = dict(config=cfg, actual_generation_precision=name,
            execution_type='official_native_cosmos_adapter', performance_eligible=False,
            scope='Quality comparison with BF16 attention. Dispatch precision is a placeholder; native external FP4 linears replace it.',
            limitations=['ARC fraction fixed at 50%, not upstream model-wide proportion search', 'Eager execution; no Graph speed qualification']
                         if name == 'arcquant' else ['extra_fusion=False; official native path, not simulation'])
    manifest = dict(schema=1, frozen=True, model=old['model'], cases=old['cases'], methods=methods,
        steps=35, role='development', selection_eligible=False, performance_eligible=False,
        resource_mode='shared-calibration', debug='off', execution='mixed-explicit-quality-only',
        imports={name: str(a.previous.resolve()) for name in retained}, cohort_sha256=sha(a.cohort),
        external=dict(sources=str(a.sources.resolve()), sources_sha256=sha(a.sources),
            arc_build=str(a.arc_build.resolve()), arc_build_sha256=sha(a.arc_build/'build.json'),
            capture=str((a.output/'arc_capture').resolve()), capture_sha256=sha(a.output/'arc_capture/manifest.json')),
        primary_comparisons=[dict(candidate=c, reference=r) for c in ('fp4_three', 'fp4_four') for r in ('arcquant', 'sharq')],
        scope='All methods on the same sixteen development scenes and two seeds. Native FP4 repair uses FP4 attention; external methods retain BF16 attention. '
              'No same-MMA or native speed claim; official adaptation/calibration limits are explicit.')
    write(a.output/'manifest.json', manifest)
    for entry in old['cases']:
        name = entry['case']['id']
        write(a.output/'inputs'/(name+'.json'), entry['case'])
        write(a.output/'seeds'/(name+'.json'), dict(seed_status='frozen', seeds={'trajectory': entry['seeds']},
            parent_cohort_sha256=sha(a.cohort), role='development', case=name,
            scope='Exact scene seeds inherited from the audited frozen development cohort, not a new seed selection.'))


def execute(a):
    root = a.output
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ext = manifest['external']
    for file, digest in ((Path(ext['sources']), ext['sources_sha256']),
                         (Path(ext['arc_build'])/'build.json', ext['arc_build_sha256']),
                         (Path(ext['capture'])/'manifest.json', ext['capture_sha256'])):
        if sha(file) != digest:
            raise ValueError('Changed pinned adapter input')
    for name, source in manifest['imports'].items():
        if not (root/'generation'/name/'progress.json').exists():
            import_method(Path(source), root, name, manifest_path)
    source = Path(__file__).with_name('external_infer.py')
    for name in ('arcquant', 'sharq'):
        identity = dict(manifest_sha256=sha(manifest_path), method=name,
            source_sha256=sha(__file__), external_infer_sha256=sha(source), smoke=False)
        progress = root/'generation'/name/'progress.json'
        state = dict(complete=False, identity=identity, completed=[], performance_eligible=False)
        if progress.exists():
            old = json.loads(progress.read_text())
            if old['identity'] != identity:
                raise ValueError('Changed baseline resume identity')
        for entry in manifest['cases']:
            case = entry['case']['id']
            folder = root/'generation'/name/case
            if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('identity'):
                data = json.loads((folder/'result.json').read_text())
                if data['identity']['manifest_sha256'] != sha(manifest_path) or not data['complete']:
                    raise ValueError('Changed/incomplete imported baseline')
            else:
                if folder.exists():
                    raise RuntimeError('Inspect incomplete external request before resume: '+str(folder))
                cmd = [sys.executable, str(source), '--model', manifest['model'], '--case', str(root/'inputs'/(case+'.json')),
                    '--sources', ext['sources'], '--output', str(folder), '--method', name,
                    '--execution', 'eager', '--cohort', str(root/'seeds'/(case+'.json')), '--split', 'trajectory',
                    '--seeds', *map(str, entry['seeds'])]
                if name == 'arcquant':
                    cmd += ['--arc-build', ext['arc_build'], '--capture', ext['capture'], '--arc-fraction', '.5']
                logfile = root/'logs'/(name+'_'+case+'.log')
                logfile.parent.mkdir(parents=True, exist_ok=True)
                write(logfile.with_suffix('.command.json'), dict(command=cmd))
                with logfile.open('w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                raw_path = folder/'result.json'
                data = json.loads(raw_path.read_text())
                if not data['complete'] or data['case'] != entry['case'] or data['image_sha256'] != entry['image_sha256']:
                    raise ValueError('Unmatched external result')
                raw_path.rename(folder/'external_result.json')
                data['external_receipt_sha256'] = sha(folder/'external_result.json')
                data['identity'] = dict(identity, case=entry['case'], image_sha256=entry['image_sha256'], seeds=entry['seeds'])
                for row in data['rows']:
                    row.update(actual_mma_ratio=None, compute_sha256=None)
                write(raw_path, data)
            for row in data['rows']:
                if sha(folder/row['outputs']) != row['outputs_sha256']:
                    raise ValueError('Changed baseline output')
            state['completed'].append(case)
            write(progress, state)
            print(json.dumps(dict(method=name, completed=len(state['completed']), total=len(manifest['cases']))), flush=True)
        state['complete'] = True
        write(progress, state)
    write(root/'generation/suite.json', dict(complete=True, manifest_sha256=sha(manifest_path), performance_eligible=False))
    subprocess.run([sys.executable, str(Path(__file__).with_name('dataset_metrics.py')), str(manifest_path),
                    '--generation', str(root/'generation'), '--output', str(root/'metrics')], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('importance_report.py')), '--run', str(root),
                    '--archive', str(root/'report'), '--label', 'external'], check=True)
    write(root/'workflow.json', dict(complete=True, manifest_sha256=sha(manifest_path), performance_eligible=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'run'))
    p.add_argument('--output', type=Path, required=True)
    for key in ('previous', 'cohort', 'capture', 'sources', 'arc-build'):
        p.add_argument('--'+key, type=Path)
    a = p.parse_args()
    if a.operation == 'prepare' and any(getattr(a, k) is None for k in ('previous', 'cohort', 'capture', 'sources', 'arc_build')):
        p.error('prepare requires all source paths')
    (prepare if a.operation == 'prepare' else execute)(a)


if __name__ == '__main__':
    main()

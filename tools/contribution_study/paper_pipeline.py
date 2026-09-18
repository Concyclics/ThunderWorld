"""Serial, resumable paper evidence supplement; no tuning on held-out scenes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from baselines import sha, write


def outlier_generation(root, workspace):
    from importance_workflow import import_method, run
    previous = workspace/'joint_schedule_20260917/development'
    old = json.loads((previous/'manifest.json').read_text())
    output = root/'outlier_development'
    output.mkdir(parents=True, exist_ok=True)
    retained = ('bf16', 'fp8', 'fp4_three', 'b2_joint', 'b2.5_ffn_guarded')
    methods = {name: old['methods'][name] for name in retained}
    pairs = []
    for cap in ('2', '2.5'):
        catalog = json.loads((root/('outlier_b'+cap)/'candidates.json').read_text())
        if not catalog['complete']:
            raise ValueError('Incomplete candidate bank')
        selected = {c['name'].rsplit('_outlier_', 1)[-1]: c for c in catalog['candidates']}
        for label in ('rtn', 'four_six', 'output_joint'):
            methods[selected[label]['name']] = selected[label]
        for label in ('four_six', 'output_joint'):
            pairs.append(dict(candidate=selected[label]['name'], reference=selected['rtn']['name'], budget=float(cap)))
    manifest = dict(schema=1, frozen=True, role='development', model=old['model'], cases=old['cases'], methods=methods,
        steps=35, debug='off', execution='graph', resource_mode='shared-calibration', performance_eligible=False,
        selection_eligible=True, primary_comparisons=pairs,
        scope='Fixed-support RTN, FourOverSix and bounded output-selected outlier candidates. '
              'Only development scenes; complete generation must improve both datasets in LPIPS and preserve SSIM/L2 before validation.')
    file = output/'manifest.json'
    if file.exists() and json.loads(file.read_text()) != manifest:
        raise ValueError('Changed frozen generation')
    write(file, manifest)
    for name in retained:
        if not (output/'generation'/name/'progress.json').exists():
            import_method(previous, output, name, file)
    run(file, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    root, workspace = a.root.resolve(), a.workspace.resolve()
    here = Path(__file__).resolve().parent
    cohort = workspace/'repair_redesign_20260917/cohort.json'
    methods = workspace/'joint_schedule_20260917/development/manifest.json'
    capture = workspace/'activation_importance_20260917/capture'
    path = root/'pipeline.json'
    identity = dict(root=str(root), workspace=str(workspace), cohort_sha256=sha(cohort),
        methods_sha256=sha(methods), source_sha256={p.name: sha(p) for p in sorted(here.glob('paper_*.py'))})
    state = dict(complete=False, identity=identity, stages={}, performance_eligible=False)
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Pipeline source or inputs changed during resume')
    root.mkdir(parents=True, exist_ok=True)
    a.archive.mkdir(parents=True, exist_ok=True)
    write(path, state)
    # This running predecessor owns its own GPU admission lock. Never start a
    # second GPU worker while it is active; errors must stop dependent stages.
    while True:
        predecessor = json.loads((root/'cross/workflow.json').read_text())
        if predecessor.get('paused'):
            raise RuntimeError('Cross-term generation paused: '+str(predecessor.get('error')))
        if predecessor['complete']:
            break
        time.sleep(5)

    def stage(name, script, args, receipt):
        if name in state['stages']:
            if sha(receipt) != state['stages'][name]['receipt_sha256']:
                raise ValueError('Changed completed stage: '+name)
            return
        cmd = [sys.executable, str(here/script), *map(str, args)]
        write(root/(name+'.command.json'), dict(command=cmd))
        state['active_stage'] = name
        write(path, state)
        with (root/(name+'.log')).open('a') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        value = json.loads(receipt.read_text())
        if not value.get('complete', value.get('frozen', False)):
            raise ValueError('Incomplete stage result: '+name)
        state['stages'][name] = dict(receipt=str(receipt), receipt_sha256=sha(receipt))
        write(path, state)
        print(json.dumps(dict(stage=name, complete=True)), flush=True)

    try:
        stage('cross_report', 'paper_study.py', ['report', '--output', root/'cross', '--archive', a.archive/'cross'],
              a.archive/'cross/cross_interventions.json')
        stage('mot_probe', 'paper_mot.py', ['--cohort', cohort, '--methods', methods, '--method', 'b2_joint', '--smoke', '--output', root/'mot_smoke'], root/'mot_smoke/manifest.json')
        stage('mot', 'paper_mot.py', ['--cohort', cohort, '--methods', methods, '--method', 'b2_joint', '--output', root/'mot'], root/'mot/manifest.json')
        stage('mot_report', 'paper_mot_report.py', ['--study', root/'mot', '--output', a.archive/'mot'], a.archive/'mot/mot.json')
        for cap, method in (('2', 'b2_joint'), ('2.5', 'b2.5_ffn_guarded')):
            folder = root/('outlier_b'+cap)
            stage('outlier_b'+cap, 'paper_outlier.py', ['--methods', methods, '--method', method,
                '--capture', capture, '--cohort', cohort, '--output', folder], folder/'progress.json')
            stage('outlier_report_b'+cap, 'paper_outlier_report.py', ['--study', folder, '--output', a.archive/('outlier_b'+cap)],
                a.archive/('outlier_b'+cap)/'outlier.json')
        if 'outlier_generation' not in state['stages']:
            state['active_stage'] = 'outlier_generation'
            write(path, state)
            outlier_generation(root, workspace)
            receipt = root/'outlier_development/workflow.json'
            state['stages']['outlier_generation'] = dict(receipt=str(receipt), receipt_sha256=sha(receipt))
            write(path, state)
        stage('outlier_quality', 'importance_report.py', ['--run', root/'outlier_development', '--archive', a.archive/'outlier_quality', '--label', 'outlier'],
              root/'outlier_development/report/quality.json')
        external = root/'external'
        stage('external_prepare', 'paper_external.py', ['prepare', '--output', external, '--previous', root/'cross',
            '--cohort', cohort, '--capture', capture, '--sources', workspace/'contribution_baselines_20260917/sources.lock.json',
            '--arc-build', workspace/'contribution_arcquant_adapted_20260917'], external/'manifest.json')
        stage('external', 'paper_external.py', ['run', '--output', external], external/'workflow.json')
        state.update(complete=True, active_stage=None,
            excluded=['No new held-out validation/final generation before a frozen development decision',
                      'No decoded AWQ speed/storage claim', 'No shared-GPU speed leaderboard'])
        state.pop('paused', None)
        state.pop('error', None)
        write(path, state)
    except Exception as error:
        state.update(paused=True, error=repr(error))
        write(path, state)
        raise


if __name__ == '__main__':
    main()

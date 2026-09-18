"""Run packed AWQ qualification after the serial paper mechanism pipeline."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    a = p.parse_args()
    root, workspace = a.root.resolve(), a.workspace.resolve()
    here = Path(__file__).resolve().parent
    output = root/'awq_native'
    output.mkdir(parents=True, exist_ok=True)
    # workflow.json belongs to importance_workflow's generation executor.
    # Keep orchestration/resume identity in a separate, stable receipt.
    status = output/'orchestrator.json'
    identity = dict(source_sha256=sha(__file__), adapter_sha256=sha(here/'awq_marlin.py'),
        dataset_infer_sha256=sha(here/'dataset_infer.py'), scale_export_sha256=sha(here/'awq_scales.py'))
    state = dict(complete=False, identity=identity, stages={}, performance_eligible=False)
    if status.exists():
        state = json.loads(status.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed native AWQ workflow identity')
    write(status, state)
    while True:
        parent = json.loads((root/'pipeline.json').read_text())
        if parent.get('paused'):
            state.update(paused=True, error='Parent mechanism pipeline paused')
            write(status, state)
            raise RuntimeError(state['error'])
        if parent['complete']:
            break
        time.sleep(5)

    def stage(name, script, args, receipt):
        if name in state['stages']:
            if sha(receipt) != state['stages'][name]:
                raise ValueError('Changed completed stage')
            return
        state['active_stage'] = name
        write(status, state)
        command = [sys.executable, str(here/script), *map(str, args)]
        write(output/(name+'.command.json'), dict(command=command))
        with (output/(name+'.log')).open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not json.loads(receipt.read_text())['complete']:
            raise ValueError('Incomplete native AWQ stage')
        state['stages'][name] = sha(receipt)
        write(status, state)

    try:
        assets = output/'assets'
        stage('export', 'awq_marlin.py', ['export', '--capture', workspace/'compute_quality_edge_v1/capture',
            '--reference', workspace/'contribution_awq_20260917/assets', '--output', assets], assets/'manifest.json')
        stage('operators', 'awq_marlin.py', ['probe', '--assets', assets, '--output', output/'operators.json'], output/'operators.json')
        previous = root/'cross'
        old = json.loads((previous/'manifest.json').read_text())
        retained = ('bf16', 'fp8', 'fp4_base', 'fp4_three', 'fp4_four', 'awq')
        methods = {name: old['methods'][name] for name in retained}
        native = copy.deepcopy(methods['awq'])
        native.update(scope='Same frozen AWQ codes/scales, native official Marlin W4A16. Full quality remeasured for changed accumulation.',
            storage_comparison_eligible=True,
            baseline_adapter=dict(method='marlin_awq', assets=str(assets), assets_sha256=sha(assets/'manifest.json'),
                                  implementation_sha256=sha(here/'awq_marlin.py')))
        methods['awq_marlin'] = native
        manifest = dict(schema=1, frozen=True, model=old['model'], cases=old['cases'], methods=methods,
            steps=35, debug='off', execution='graph', resource_mode='shared-calibration',
            role='development', selection_eligible=False, performance_eligible=False,
            primary_comparisons=[dict(candidate='awq_marlin', reference='awq')],
            parent_manifest_sha256=sha(previous/'manifest.json'),
            scope='Same-input native AWQ generation; no decoded-reference or operator result is substituted for native full-request quality.')
        manifest_path = output/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Changed frozen native AWQ study')
        write(manifest_path, manifest)
        stage('graph_roundtrip', 'dataset_infer.py', [manifest_path, '--method', 'awq_marlin', '--smoke', '--output', output/'smoke'], output/'smoke/progress.json')
        from importance_workflow import import_method, run
        from awq_multiscene import import_metrics
        for name in retained:
            if not (output/'generation'/name/'progress.json').exists():
                import_method(previous, output, name, manifest_path)
            if name != 'bf16':
                import_metrics(previous, output, name)
        if 'generation' not in state['stages']:
            state['active_stage'] = 'generation'
            write(status, state)
            run(manifest_path, output)
            state['stages']['generation'] = sha(output/'generation/suite.json')
            write(status, state)
        elif sha(output/'generation/suite.json') != state['stages']['generation']:
            raise ValueError('Changed completed generation receipt')
        stage('quality', 'importance_report.py', ['--run', output, '--archive', output/'report', '--label', 'native_awq'], output/'report/quality.json')
        state.update(complete=True, active_stage=None, full_generation_measured=True,
            scope='Operator and full native generation qualification only. Exclusive speed tests are separate.')
        write(status, state)
    except Exception as error:
        state.update(paused=True, error=repr(error))
        write(status, state)
        raise


if __name__ == '__main__':
    main()

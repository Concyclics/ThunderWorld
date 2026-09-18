"""Immutable shared-GPU studies with verified reuse of identical baseline evidence."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from baselines import sha, write


def import_method(source, target, name, manifest):
    old = json.loads((source / 'manifest.json').read_text())
    new = json.loads(manifest.read_text())
    if old['methods'][name] != new['methods'][name] or old['cases'] != new['cases']:
        raise ValueError('Cannot reuse changed methods or inputs')
    progress = json.loads((source / 'generation' / name / 'progress.json').read_text())
    if not progress['complete'] or progress['identity']['manifest_sha256'] != sha(source / 'manifest.json'):
        raise ValueError('Incomplete or changed baseline')
    digest = sha(manifest)
    for case in new['cases']:
        src = source / 'generation' / name / case['case']['id']
        dst = target / 'generation' / name / case['case']['id']
        receipt = json.loads((src / 'result.json').read_text())
        if not receipt['complete'] or receipt['identity']['case'] != case['case'] or receipt['identity']['seeds'] != case['seeds']:
            raise ValueError('Changed baseline input')
        dst.mkdir(parents=True, exist_ok=True)
        for row in receipt['rows']:
            file = src / row['outputs']
            if sha(file) != row['outputs_sha256']:
                raise ValueError('Corrupt baseline video')
            if not (dst / file.name).exists():
                os.link(file, dst / file.name)
            elif sha(dst / file.name) != sha(file):
                raise ValueError('Changed imported video')
        if (src / 'compute.json').exists() and not (dst / 'compute.json').exists():
            os.link(src / 'compute.json', dst / 'compute.json')
        receipt['imported_from'] = dict(path=str(src / 'result.json'), sha256=sha(src / 'result.json'))
        receipt['identity']['manifest_sha256'] = digest
        write(dst / 'result.json', receipt)
    progress['identity']['manifest_sha256'] = digest
    progress['imported_from'] = dict(path=str(source / 'generation' / name / 'progress.json'), sha256=sha(source / 'generation' / name / 'progress.json'))
    write(target / 'generation' / name / 'progress.json', progress)


def run(manifest, root):
    here = Path(__file__).resolve().parent
    status = root / 'workflow.json'
    state = dict(complete=False, manifest_sha256=sha(manifest), performance_eligible=False)
    write(status, state)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    with (root / 'metrics.log').open('a') as ml:
        metrics = subprocess.Popen([sys.executable, str(here / 'dataset_metrics.py'), str(manifest), '--generation', str(root / 'generation'), '--output', str(root / 'metrics'), '--follow'], env=env, stdout=ml, stderr=subprocess.STDOUT)
        try:
            with (root / 'generation.log').open('a') as gl:
                subprocess.run([sys.executable, str(here / 'dataset_suite.py'), str(manifest), '--output', str(root / 'generation')], stdout=gl, stderr=subprocess.STDOUT, check=True)
            if metrics.wait() != 0:
                raise RuntimeError('Metric worker failed')
            state['complete'] = True
        except Exception as error:
            if metrics.poll() is None:
                metrics.terminate()
                metrics.wait()
            state.update(paused=True, error=repr(error))
            raise
        finally:
            write(status, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--catalog', type=Path)
    p.add_argument('--output-subdir', default='two_term')
    p.add_argument('--after', type=Path)
    a = p.parse_args()
    if a.after:
        while True:
            status = json.loads(a.after.read_text()) if a.after.exists() else {}
            if status.get('paused'):
                raise RuntimeError('Previous experiment paused')
            if status.get('complete'):
                break
            time.sleep(5)
    from thunderworld_fp4.cosmos.settings import RunConfig, LinearSpec
    old = json.loads((a.previous / 'manifest.json').read_text())
    root = a.study / a.output_subdir
    root.mkdir(parents=True, exist_ok=True)
    retained = ['bf16', 'fp8', 'fp4_three', 'b2.5_marginal', 'b2.5_ffn_aware']
    methods = {name: old['methods'][name] for name in retained if name in old['methods']}
    if a.catalog:
        catalog = json.loads(a.catalog.read_text())
        if not catalog['complete']:
            raise ValueError('Incomplete candidate assets')
        methods.update({c['name']: c for c in catalog['candidates']})
    else:
        for name, af, wf in [('linear_base', 0, 0), ('linear_w_only', 0, 1), ('linear_a_only', 1, 0), ('linear_three', 1, 1)]:
            cfg = RunConfig(resource_mode='shared-calibration', precision='fp4', attention_backend='bf16', ffn=LinearSpec(af, wf), projection=LinearSpec(af, wf), share_pack=False)
            methods[name] = dict(config=cfg.to_dict(), scope='Linear-only attribution: BF16 attention is fixed for all four rows; not a joint FP4 budget candidate.')
    value = copy.deepcopy(old)
    value.update(methods=methods, role='development', selection_eligible=True,
                 previous_manifest_sha256=sha(a.previous / 'manifest.json'),
                 scope='Full-M activation-importance development. No validation or final cohort use. Shared GPU timings are ineligible.')
    if a.catalog:
        value['catalog_sha256'] = sha(a.catalog)
    manifest = root / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != value:
        raise ValueError('Immutable experiment changed')
    write(manifest, value)
    for name in methods:
        if name in old['methods']:
            import_method(a.previous, root, name, manifest)
    run(manifest, root)


if __name__ == '__main__':
    main()

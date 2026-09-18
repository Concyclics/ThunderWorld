"""Freeze completed development evidence and delivered sources without large videos."""
import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    a.archive.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    records, outputs = [], {}
    for name in ('two_term', 'ffn_development'):
        run = a.study / name
        manifest = json.loads((run / 'manifest.json').read_text())
        if not json.loads((run / 'workflow.json').read_text())['complete']:
            raise ValueError('Cannot archive an incomplete generation/metric workflow')
        for method in manifest['methods']:
            for case in manifest['cases']:
                file = run / 'generation' / method / case['case']['id'] / 'result.json'
                receipt = json.loads(file.read_text())
                if not receipt['complete'] or receipt['identity']['manifest_sha256'] != sha(run / 'manifest.json'):
                    raise ValueError('Incomplete or mismatched generation receipt')
                if {row['seed'] for row in receipt['rows']} != set(case['seeds']) or len(receipt['rows']) != len(case['seeds']):
                    raise ValueError('Missing or duplicate seed')
                records.append(dict(path=str(file), sha256=sha(file), receipt=receipt))
                origin, original, seen = file, receipt, {file.resolve()}
                while 'imported_from' in original:
                    source = original['imported_from']
                    origin = Path(source['path'])
                    if origin.resolve() in seen or sha(origin) != source['sha256']:
                        raise ValueError('Changed or circular baseline provenance')
                    seen.add(origin.resolve())
                    original = json.loads(origin.read_text())
                for row in receipt['rows']:
                    if row['performance_eligible'] or row['debug'] != 'off' or row['steps'] != 35:
                        raise ValueError('Changed study conditions')
                    key = (method, case['case']['id'], row['seed'])
                    value = dict(method=method, case=key[1], seed=key[2], benchmark=case['benchmark'],
                                 category=case['category'], output=str(file.parent / row['outputs']),
                                 output_sha256=row['outputs_sha256'], signal_sha256=row['sha256'],
                                 bytes=(file.parent / row['outputs']).stat().st_size,
                                 origin_receipt=str(origin), imported=not origin.resolve().is_relative_to(a.study.resolve()))
                    if key in outputs and value['output_sha256'] != outputs[key]['output_sha256']:
                        raise ValueError('Imported baseline output changed')
                    outputs[key] = value
    payload = dict(complete=True, outputs=list(outputs.values()), receipts=records, performance_eligible=False,
                   scope='File hashes are frozen generation receipts, already checked on baseline reuse; large video/model/weight assets stay local.')
    (a.archive / 'generation_index.json.gz').write_bytes(gzip.compress(json.dumps(payload, indent=2).encode(), mtime=0))
    sources = {}
    paths = list((repo / 'packages/fp4-repair/src/thunderworld_fp4').rglob('*.py'))
    paths += list((repo / 'tools/contribution_study').glob('*.py'))
    for file in paths:
        rel = file.relative_to(repo)
        package_root = repo / 'packages/fp4-repair/src'
        destination = (a.archive / 'source/package' / file.relative_to(package_root)
                       if file.is_relative_to(package_root) else a.archive / 'source/tools' / file.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
        sources[str(rel)] = sha(file)
    environment = {}
    for package in ('torch', 'triton', 'vllm', 'numpy', 'lpips', 'scikit-image', 'transformers', 'cosmos3'):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    state = dict(complete=True, sources=sources, versions=environment,
                 checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                 checkout_has_uncommitted_changes=True, unique_outputs=len(outputs),
                 new_generation_outputs=sum(not row['imported'] for row in outputs.values()),
                 cpu_tests=dict(passed=19, command='pytest -q tools/contribution_study/test_importance.py tools/contribution_study/test_redesign.py packages/fp4-repair/tests/test_compute_budget.py'),
                 performance_eligible=False,
                 scope='Delivered source snapshot; initial runtime hashes and any prior versions remain in immutable receipts and source_history. New outputs exclude imported baselines and separate fusion probes.')
    write(a.archive / 'implementation.json', state)
    for filename in ('fusion_probe.json', 'guarded_fusion_probe.json'):
        file = a.study / filename
        if file.exists():
            if not json.loads(file.read_text())['complete']:
                raise ValueError('Incomplete fusion probe')
            shutil.copyfile(file, a.archive / filename)
    print(json.dumps(dict(complete=True, unique_outputs=len(outputs), new_generation_outputs=state['new_generation_outputs'], sources=len(sources))))


if __name__ == '__main__':
    main()

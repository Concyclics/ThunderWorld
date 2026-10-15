"""Frozen paired development generation for fixed-support representation search."""
import argparse
import copy
import json
from pathlib import Path
from baselines import sha, write
from importance_workflow import import_method, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    a = p.parse_args()
    # The quality worker uses this interpreter. Catch a missing optional metric
    # environment before launching a long generation suite.
    import importlib.util
    for module in ('lpips', 'skimage', 'scipy'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError('Missing quality dependency '+module+'; install tools/fp4_handoff/requirements-quality.txt')
    catalog_path = a.study/'all_layers/candidates.json'
    catalog = json.loads(catalog_path.read_text())
    protocol = json.loads((a.study/'protocol.json').read_text())
    old = json.loads((a.previous/'manifest.json').read_text())
    if not catalog['complete'] or not protocol['frozen'] or old['role'] != 'development':
        raise ValueError('Complete candidates and frozen development-only protocol required')
    control = protocol['weight']['control']
    retained = ('bf16', 'fp8', control)
    methods = {name: old['methods'][name] for name in retained}
    methods.update({row['name']: row for row in catalog['candidates']})
    manifest = copy.deepcopy(old)
    manifest.update(methods=methods, performance_eligible=False, role='development', selection_eligible=True,
        catalog_sha256=sha(catalog_path), protocol_sha256=sha(a.study/'protocol.json'),
        parent_manifest_sha256=sha(a.previous/'manifest.json'),
        scope='Fixed A/W support, three terms, native kernels and exact original asset control; only offline encoding changes. '
              '16 development scenes x two seeds. Validation/final untouched. No latency claims.')
    root = a.study/'development'; root.mkdir(exist_ok=True); path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Frozen generation changed')
    write(path, manifest)
    for name in retained:
        import_method(a.previous, root, name, path)
    run(path, root)


if __name__ == '__main__':
    main()

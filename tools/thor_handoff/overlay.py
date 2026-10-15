"""Opt-in SM110 guards in an isolated ARM checkout; no architecture spoofing."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    if platform.machine() != 'aarch64':
        raise RuntimeError('Apply only on the isolated ARM Thor checkout')
    root = a.root.resolve()
    package = root/'packages/fp4-repair/src/thunderworld_fp4'
    changes = {
        'linear.py': [
            ('torch.cuda.get_device_capability(weight.device) != (12,0)', 'not _supported_device(weight.device)'),
            ('torch.cuda.get_device_capability(x.device) != (12,0)', 'not _supported_device(x.device)')],
        'runtime.py': [('torch.cuda.get_device_capability(q.device) != (12, 0)', 'not _supported_device(q.device)')],
        '_native.py': [('torch.cuda.get_device_capability(wc.device)!=(12,0)', 'not _supported_device(wc.device)')],
    }
    helper = '''# Explicit experimental Thor opt-in, independent of the SM120 qualification.
import os as _thor_os
def _supported_device(device):
    capability = torch.cuda.get_device_capability(device)
    return capability == (12, 0) or (capability == (11, 0) and
        _thor_os.environ.get("THUNDERWORLD_EXPERIMENTAL_SM110") == "1")

'''
    receipt = dict(schema='thor-sm110-opt-in-overlay-v1', files={})
    diff = []
    marker = root/'thor_overlay.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        for rel, entry in old['files'].items():
            if hashlib.sha256((root/rel).read_bytes()).hexdigest() != entry['after']:
                raise RuntimeError('Overlay changed since preparation: '+rel)
        return
    for name, pairs in changes.items():
        path = package/name
        before = path.read_text()
        after = before
        for source, target in pairs:
            if after.count(source) != 1:
                raise RuntimeError('Unexpected source revision: '+name)
            after = after.replace(source, target)
        # Helpers only run after torch has been imported; insert immediately
        # before the first definition/decorator, leaving module docstrings valid.
        positions = [after.find('\ndef '), after.find('\nclass '), after.find('\n@')]
        at = min(x for x in positions if x >= 0)
        after = after[:at]+'\n'+helper+after[at:]
        rel = str(path.relative_to(root))
        receipt['files'][rel] = dict(before=hashlib.sha256(before.encode()).hexdigest(),
                                   after=hashlib.sha256(after.encode()).hexdigest())
        diff.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                       fromfile='a/'+rel, tofile='b/'+rel))
        path.write_text(after)
    (root/'thor_overlay.patch').write_text(''.join(diff))
    marker.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()

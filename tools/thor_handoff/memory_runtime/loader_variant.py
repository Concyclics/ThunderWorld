"""Derive v8 without materializing replaced BF16 weights on the GPU."""
import argparse
import json
from pathlib import Path
import shutil
import time

from build import replace, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--parent', default='cutlass-memory-v7')
    p.add_argument('--name', default='cutlass-memory-v8')
    a = p.parse_args()
    parent = a.root / 'variants' / a.parent
    old = json.loads((parent / 'variant.json').read_text())
    if not old.get('qualified'):
        raise ValueError('Parent kernel qualification is required')
    for name, digest in old['variant_hashes'].items():
        if sha(parent / 'thunderworld_fp4' / name) != digest:
            raise ValueError('Parent source changed: ' + name)
    if sha(parent / 'cutlass_sm110.so') != old['library_sha256']:
        raise ValueError('Parent library changed')
    target = a.root / 'variants' / a.name
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    package = target / 'thunderworld_fp4'
    runner = package / 'cosmos/runner.py'
    text = runner.read_text()
    text = replace(text, "        self.pipe.to('cuda')\n", '')
    text = replace(text, '        self.engine=JointEngine(self.pipe.transformer,self.config,self.table,assets)\n',
        '''        # Replace generation projections before transferring the remaining
        # BF16 modules. Prepared FP4/FP8 buffers are already on CUDA; Module.to
        # retains their dtype and storage. AR and codec preparation follows.
        self.engine=JointEngine(self.pipe.transformer,self.config,self.table,assets)
        self.pipe.to('cuda')
''')
    runner.write_text(text)
    changed = [name for name, digest in old['variant_hashes'].items()
               if sha(package / name) != digest]
    if changed != ['cosmos/runner.py']:
        raise ValueError('Unexpected numerical source change: ' + repr(changed))
    receipt = dict(old)
    receipt.update(name=a.name, created=time.time(),
        parent=dict(path=str(parent), sha256=sha(parent / 'variant.json')),
        deferred_gpu_transfer=True,
        qualification_scope='Unchanged v7 numerical kernels and binary; loader equivalence is checked by matched complete requests in the campaign',
        loader_qualification='requires_matched_full_request_hashes',
        changed_sources=changed,
        variant_hashes={str(path.relative_to(package)):sha(path)
                        for path in sorted(package.rglob('*.py'))},
        loader_builder_sha256=sha(__file__))
    (target / 'variant.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(variant=str(target), unchanged_kernel_sha256=sha(target / 'cutlass_sm110.so'),
                          variant_sha256=sha(target / 'variant.json'))), flush=True)


if __name__ == '__main__':
    main()

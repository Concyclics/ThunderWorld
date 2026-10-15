"""Extend only missing ARC reorder template dispatch in a new SM110 build."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from build_baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    parent = a.root/'external/arcquant-sm110'
    old = json.loads((parent/'build.json').read_text())
    assert old['complete'] and sha(old['library']) == old['library_sha256']
    if a.output.exists():
        raise FileExistsError('Preserve earlier builds: ' + str(a.output))
    shutil.copytree(parent, a.output, symlinks=True, ignore=shutil.ignore_patterns('build', '__pycache__'))
    patches = []
    for name in ['kernels/src/bindings.cpp', 'kernels/src/reorder.cu']:
        f = a.output/name
        before = f.read_text()
        after = before
        if name.endswith('bindings.cpp'):
            for side in ('X', 'W'):
                needle = f'        CASE_REORDER_{side}_16(9216)'
                assert after.count(needle) == 1 and f'CASE_REORDER_{side}_16(12288)' not in after
                after = after.replace(needle, f'        CASE_REORDER_{side}_16(12288)\n' + needle)
        else:
            after += '\n' + ''.join(
                f'template void run_reorder_{side}_bf16_nvfp4<16, 12288>(bf16_t*, int, int16_t*, uint8_t*, sf_t*, int, int);\n'
                for side in ('x', 'w'))
        f.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    (a.output/'section54_shapes.patch').write_text(''.join(patches))
    record = dict(old, complete=False, parent_build_sha256=sha(parent/'build.json'),
                  dispatch_added=[12288], quantizer_modified=False, started=time.time())
    record['patches'] = dict(old['patches'], **{'section54_shapes.patch': sha(a.output/'section54_shapes.patch')})
    commands = [[str(v).replace(str(parent), str(a.output)) for v in c] for c in old['commands']]
    record['commands'] = commands
    record['library'] = str(a.output/'kernels/build/agemm.so')
    write(a.output/'build.json', record)
    try:
        for i, cmd in enumerate(commands):
            with (a.output/f'section54_build_{i}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, TORCH_CUDA_ARCH_LIST='11.0a'), check=True)
        record.update(complete=True, library_sha256=sha(record['library']), finished=time.time())
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        write(a.output/'build.json', record)


if __name__ == '__main__':
    main()

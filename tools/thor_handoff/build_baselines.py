"""Recorded SM110 ports of pinned official ARCQuant and SharQ kernels.

The quantizers are unchanged. CUTLASS's Sm100 template traits select the
tcgen05 implementation; the actual binary target is always sm_110a.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--method', choices=('arcquant', 'sharq'), required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    source = a.root/'external'/a.method
    metadata = json.loads((a.root/'data/baseline_sources.json').read_text())[a.method]
    for name, expected in metadata['files'].items():
        if sha(source/name) != expected:
            raise ValueError('Official source changed: '+name)
    dest = a.root/'external'/(a.method+'-sm110')
    if dest.exists():
        raise FileExistsError('Preserve earlier build evidence: '+str(dest))
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns('third-party'))
    (dest/'third-party').mkdir()
    (dest/'third-party/cutlass').symlink_to(source/'third-party/cutlass', target_is_directory=True)
    patches = []

    def edit(name, transform):
        file = dest/name
        before = file.read_text()
        after = transform(before)
        if before == after:
            raise ValueError('Expected adaptation missing: '+name)
        file.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/'+name, tofile='b/'+name))

    if a.method == 'arcquant':
        edit('kernels/src/nvfp4.cu', lambda s: s.replace('cutlass::arch::Sm120', 'cutlass::arch::Sm100')
             .replace('Shape<_128,_128,_128>', 'Shape<_128,_128,_256>'))
        edit('kernels/CMakeLists.txt', lambda s: s.replace('120a', '110a')
             .replace('CUTLASS_ARCH_SM120_SUPPORTED', 'CUTLASS_ARCH_SM110_SUPPORTED')
             .replace('set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}/../../)', '')
             .replace('set(CUTLASS_ROOT ../third-party/cutlass)', 'set(CUTLASS_ROOT ${CMAKE_CURRENT_LIST_DIR}/../third-party/cutlass)'))
        def dispatch(s):
            for side in ('X','W'):
                needle = f'        CASE_REORDER_{side}_16(2048)'
                assert s.count(needle) == 1
                s = s.replace(needle, f'        CASE_REORDER_{side}_16(9216)\n'+needle)
            return s
        edit('kernels/src/bindings.cpp', dispatch)
        edit('kernels/src/reorder.cu', lambda s: s+'\n'+''.join(
            f'template void run_reorder_{side}_bf16_nvfp4<16, 9216>(bf16_t*, int, int16_t*, uint8_t*, sf_t*, int, int);\n'
            for side in ('x','w')))
        target = 'agemm'
        extra = []
    else:
        # Preserve upstream SM100 dense/sparse dispatch and all quantizers.
        # Only redirect its compilation and labels to the real Thor ISA.
        edit('kernels/CMakeLists.txt', lambda s: s.replace('sm100a', 'sm110a')
             .replace('10.0a','11.0a').replace('compute_100a','compute_110a').replace('sm_100a','sm_110a'))
        edit('kernels/include/sharq_blackwell_arch.h', lambda s: s.replace('"sm100a"','"sm110a"')
             .replace('"build_cmake_sm100a"','"build_cmake_sm110a"'))
        target = 'sharq_ops'
        extra = ['-DSHARQ_CUDA_ARCH=sm110a']
    (dest/'thor.patch').write_text(''.join(patches))
    import pybind11
    env = {**os.environ, 'TORCH_CUDA_ARCH_LIST':'11.0a'}
    commands = [
        ['cmake','-S',str(dest/'kernels'),'-B',str(dest/'kernels/build'),
         '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
         '-DPython3_EXECUTABLE='+sys.executable,'-Dpybind11_DIR='+pybind11.get_cmake_dir(),*extra],
        ['cmake','--build',str(dest/'kernels/build'),'--target',target,'-j',str(a.jobs)]]
    record = dict(complete=False, method=a.method, source_commit=metadata['source_commit'],
        cutlass_commit=metadata['cutlass_commit'], original_source_verified=True,
        architecture='sm_110a', template_traits='Sm100 tcgen05; real SM110 target',
        quantizer_modified=False, patches={'thor.patch':sha(dest/'thor.patch')}, commands=commands,
        performance_eligible=False, started=time.time())
    write(dest/'build.json', record)
    try:
        for i, cmd in enumerate(commands):
            with (dest/f'build_{i}.log').open('w') as stream:
                subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        library = dest/'kernels/build'/(target+'.so')
        record.update(complete=True, library=str(library), library_sha256=sha(library), finished=time.time())
    except Exception as e:
        record['error'] = repr(e)
        raise
    finally:
        write(dest/'build.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

"""Reproduce the official SM120a SharQ build without editing tracked sources."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from baselines import git,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--pybind11',type=Path,required=True)
    a=p.parse_args();entry=json.loads(a.sources.read_text())['sources']['sharq'];source=Path(entry['path'])
    if git(source,'rev-parse','HEAD')!=entry['commit'] or git(source,'status','--porcelain','--untracked-files=no'):
        raise ValueError('Official source changed')
    cutlass=source/'third-party/cutlass';build=source/'kernels/build_cmake_sm120a'
    expected=git(source,'ls-tree','HEAD','third-party/cutlass').split()[2]
    if git(cutlass,'rev-parse','HEAD')!=expected:raise ValueError('Initialize the pinned CUTLASS submodule first')
    build.mkdir(parents=True,exist_ok=True)
    commands=[['cmake','-S',str(source/'kernels'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc','-DPython3_EXECUTABLE='+sys.executable,
        '-Dpybind11_DIR='+str(a.pybind11),'-DSHARQ_CUDA_ARCH=sm120a'],
        ['cmake','--build',str(build),'--target','sharq_ops','-j','4']]
    receipt=dict(complete=False,source_commit=entry['commit'],cutlass_commit=expected,commands=commands,patches=[],architecture='sm_120a')
    write(build/'contribution-build.json',receipt)
    for i,command in enumerate(commands):
        with (build/f'contribution-build-{i}.log').open('w') as log:
            subprocess.run(command,env={**os.environ,'TORCH_CUDA_ARCH_LIST':'12.0a'},stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt.update(complete=True,library_sha256=sha(build/'sharq_ops.so'));write(build/'contribution-build.json',receipt)


if __name__=='__main__':main()

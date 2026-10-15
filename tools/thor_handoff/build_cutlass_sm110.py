"""Offline Thor CUTLASS build. Never compile or tune inside a timed request."""
import argparse
import json
from pathlib import Path
import subprocess
from build_baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutlass',type=Path,default=Path('/home/chenhan/thunderworld/external/arcquant/third-party/cutlass'))
    p.add_argument('--nvcc',default='/usr/local/cuda/bin/nvcc')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fused',action='store_true',help='Compile mainloop-integrated sparse repair')
    a=p.parse_args()
    source=Path(__file__).with_name('cutlass_sm110_fused.cu' if a.fused else 'cutlass_sm110.cu')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[a.nvcc,'-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda',
         '-gencode','arch=compute_110a,code=sm_110a','--shared','-Xcompiler=-fPIC',
         '-I'+str(a.cutlass/'include'),'-I'+str(a.cutlass/'tools/util/include'),str(source),'-o',str(a.output)]
    subprocess.run(cmd,check=True)
    sources={source.name:sha(source)}
    if a.fused:sources['sm110_repair_collective.cuh']=sha(source.with_name('sm110_repair_collective.cuh'))
    write(a.output.with_suffix('.build.json'),dict(schema_version=1,backend='cutlass-sm110-fused-mainloop' if a.fused else 'cutlass-sm110-base-fp32',
        architecture='sm_110a',tile=[128,128,128],command=cmd,sources=sources,
        library_sha256=sha(a.output),cutlass_root=str(a.cutlass),
        cutlass_version=(a.cutlass/'include/cutlass/version.h').read_text(),
        nvcc=subprocess.check_output([a.nvcc,'--version'],text=True)))
    print(json.dumps(dict(library=str(a.output),sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()

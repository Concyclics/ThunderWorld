"""Offline build: python -m thunderworld_fp4.cutlass.build --output PATH.so."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def source_hashes():
    root=Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.suffix in ('.cu','.cuh')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutlass',type=Path,help='CUTLASS root; default: installed FlashInfer headers')
    p.add_argument('--nvcc',default='/usr/local/cuda/bin/nvcc')
    p.add_argument('--tile-n',type=int,choices=[64,128],default=128)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.cutlass
    if root is None:
        spec=importlib.util.find_spec('flashinfer')
        if spec is None:raise ValueError('Pass --cutlass or install FlashInfer separately')
        root=Path(spec.origin).parent/'data/cutlass'
    if not (root/'include/cutlass/cutlass.h').exists():raise FileNotFoundError(root)
    source=Path(__file__).with_name('fused.cu')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    command=[a.nvcc,f'-DTW_TILE_N={a.tile_n}','-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda',
        '-gencode','arch=compute_120a,code=sm_120a','--shared','-Xcompiler=-fPIC',
        '-I'+str(root/'include'),'-I'+str(root/'tools/util/include'),str(source),'-o',str(a.output)]
    subprocess.run(command,check=True)
    metadata=dict(schema_version=1,backend='cutlass-fused-sm120',tile=[128,a.tile_n,128],
        command=command,sources=source_hashes(),
        library_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        nvcc=subprocess.check_output([a.nvcc,'--version'],text=True),
        cutlass_version=(root/'include/cutlass/version.h').read_text())
    a.output.with_suffix('.build.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()

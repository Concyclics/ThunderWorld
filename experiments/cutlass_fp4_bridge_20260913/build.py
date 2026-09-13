"""Build before GPU warmup/capture. No online compilation or dependency download."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cutlass',type=Path,help='CUTLASS root; otherwise use installed FlashInfer headers')
p.add_argument('--nvcc',default='/usr/local/cuda/bin/nvcc')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=a.cutlass
if root is None:
    spec=importlib.util.find_spec('flashinfer')
    if spec is None:raise ValueError('Pass --cutlass or install FlashInfer separately')
    root=Path(spec.origin).parent/'data/cutlass'
if not (root/'include/cutlass/cutlass.h').exists():raise FileNotFoundError(root)
source=Path(__file__).with_name('base.cu')
if a.output.exists():raise FileExistsError(a.output)
command=[a.nvcc,'-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda',
    '-gencode','arch=compute_120a,code=sm_120a','--shared','-Xcompiler=-fPIC',
    '-I'+str(root/'include'),'-I'+str(root/'tools/util/include'),str(source),'-o',str(a.output)]
subprocess.run(command,check=True)
metadata=dict(command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    library_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
    nvcc=subprocess.check_output([a.nvcc,'--version'],text=True),
    cutlass_version=(root/'include/cutlass/version.h').read_text())
a.output.with_suffix('.build.json').write_text(json.dumps(metadata,indent=2)+'\n')

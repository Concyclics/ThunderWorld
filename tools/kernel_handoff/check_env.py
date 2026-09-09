"""Receiver preflight. This does not alter clocks, power limits or driver settings."""
import argparse,json,os,platform,subprocess
from pathlib import Path
import torch,triton

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Explicitly select the GPU authorized on this host')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check driver and cu130 PyTorch installation')
    if torch.cuda.device_count()!=1:raise RuntimeError('Select exactly one GPU for isolated kernel timing')
    cap=torch.cuda.get_device_capability()
    if cap!=(12,0):raise RuntimeError(f'Expected SM120, got {cap}')
    prop=torch.cuda.get_device_properties(0)
    def command(cmd):
        try:return subprocess.run(cmd,capture_output=True,text=True,check=False).stdout.strip()
        except FileNotFoundError:return 'not installed'
    out={'python':platform.python_version(),'torch':torch.__version__,'torch_path':torch.__file__,'cuda_runtime':torch.version.cuda,'triton':triton.__version__,'gpu':prop.name,'compute_capability':cap,'vram_bytes':prop.total_memory,'sm_count':prop.multi_processor_count,'visible_device':os.environ['CUDA_VISIBLE_DEVICES'],
         'nvidia_smi':command(['nvidia-smi']),'nvcc':command(['nvcc','--version']),'ncu':command(['ncu','--version']),'nsys':command(['nsys','--version']),'sanitizer':command(['compute-sanitizer','--version'])}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

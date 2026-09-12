"""Inspect compiled PTX for native FP4 Tensor Core instructions (not a speed test)."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from thunderworld_fp4.cosmos.linears import RepairLinear
from thunderworld_fp4.cosmos.settings import TIERS
from thunderworld_fp4.cosmos.tuning import LaunchTable
from thunderworld_fp4 import RepairConfig,fp4_attention
from thunderworld_fp4._native import gemm
from thunderworld_fp4._kernels import streaming_attention


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:d,record_linear=lambda *args:None)
    source=torch.nn.Linear(384,256,bias=False,device='cuda',dtype=torch.bfloat16)
    op=RepairLinear(source,controller,'test',TIERS['four'])
    op(torch.randn(416,384,device='cuda',dtype=torch.bfloat16))
    q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,257,2,128,device='cuda',dtype=torch.bfloat16)
    fp4_attention(q,k,torch.randn_like(k),RepairConfig(coverage=1,terms=4,reload_q=True))
    torch.cuda.synchronize();result=dict(complete=True,device=controller.table.identity,kernels={})
    for name,kernel in [('ffn',gemm),('attention',streaming_attention)]:
        lines=[]
        for compiled in kernel.device_caches[torch.cuda.current_device()][0].values():
            lines.extend(line.strip() for line in compiled.asm['ptx'].splitlines()
                         if 'mma.' in line and ('mxf4' in line or 'nvf4' in line or 'e2m1' in line))
        if not lines:raise AssertionError('Missing native FP4 MMA instruction: '+name)
        result['kernels'][name]=dict(instruction_forms=sorted(set(line.split()[0] for line in lines)),
                                    static_instruction_sites=len(lines),example=lines[0])
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

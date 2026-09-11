"""Bitwise-checked merge tile probe; synthetic parts, not operator speedups."""
import json
import random
import statistics
from pathlib import Path
import torch
import triton
import triton.testing
import triton.language as tl

@triton.jit
def combine(PARTS,STATS,OUT,M:tl.constexpr,H:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr,ROWS:tl.constexpr=4):
    r=tl.program_id(0)*ROWS+tl.arange(0,ROWS);h=tl.program_id(1)
    # Keep split reduction local to each output element. A cross-warp tl.sum
    # lowering in the first prototype produced shared-memory WAW race reports.
    top=tl.full((ROWS,),-float('inf'),tl.float32)
    for s in tl.static_range(SPLITS):
        mx=tl.load(STATS+((s*H+h)*M+r)*2,r<M,-float('inf'))
        top=tl.maximum(top,mx)
    top=tl.where(r<M,top,0.)
    di=tl.arange(0,128)
    accum=tl.full((ROWS,128),0,tl.float32);total=tl.full((ROWS,),0,tl.float32)
    for s in tl.static_range(SPLITS):
        si=((s*H+h)*M+r)*2
        mx=tl.load(STATS+si,r<M,-float('inf'));den=tl.load(STATS+si+1,r<M,0.)
        alpha=tl.exp(mx-top);total=total+den*alpha
        idx=((s*H+h)*M+r[:,None])*128+di[None,:]
        value=tl.load(PARTS+idx,r[:,None]<M,0.)
        accum=accum+value*alpha[:,None]
    tl.store(OUT+(r[:,None]*H+h)*128+di[None,:],accum/total[:,None],r[:,None]<M)


@torch.inference_mode()
def main():
    torch.manual_seed(5090)
    rows=[]
    for m in [1,17,128,129,560,2048]:
        for splits in [2,4,8]:
            h=32
            parts=torch.randn((splits,h,m,128),device='cuda')
            stats=torch.randn((splits,h,m,2),device='cuda');stats[:,:,:,1].abs_().add_(1)
            outputs={r:torch.empty((1,m,h,128),device='cuda',dtype=torch.bfloat16) for r in [4,8,16,32]}
            def run(r):
                combine[(triton.cdiv(m,r),h)](parts,stats,outputs[r],m,h,splits,splits,r,num_warps=4,enable_fp_fusion=False)
            for r in outputs:run(r)
            for r in outputs:assert torch.equal(outputs[r],outputs[4]),(m,splits,r)
            samples={r:[] for r in outputs};rng=random.Random(5090)
            for _ in range(5):
                order=list(outputs);rng.shuffle(order)
                for r in order:samples[r].append(triton.testing.do_bench_cudagraph(lambda:run(r),rep=20))
            entry={'m':m,'splits':splits,'bitwise_equal':True,'latency_ms':{r:{'median':statistics.median(v),'samples':v} for r,v in samples.items()}}
            rows.append(entry)
            print(m,splits,{r:round(statistics.median(v)*1000,2) for r,v in samples.items()},flush=True)
    Path('runs5090/combine_probe.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()

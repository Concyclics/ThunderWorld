"""Native cost frontier and profiling payloads for actual Generator shapes."""
import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import torch
import triton.testing
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers import Cosmos3OmniTransformer
from dynamic import ROOT,BudgetLinear,TIERS


def bench(fn):return statistics.median(triton.testing.do_bench_cudagraph(fn,rep=10) for _ in range(3))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['edge','nano']);p.add_argument('--capture',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--tier',default='a50')
    p.add_argument('--bm',type=int,default=128);p.add_argument('--stages',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    if a.payload:
        d=torch.load(a.payload,weights_only=True,map_location='cpu')
        source=torch.nn.Linear(d['weight'].shape[1],d['weight'].shape[0],bias=False,dtype=torch.bfloat16)
        source.weight=torch.nn.Parameter(d['weight'],requires_grad=False)
        op=BudgetLinear(source);op.tier=a.tier;op.op.bm=a.bm;op.op.num_stages=a.stages
        x=d['x'].cuda();op(x);torch.cuda.synchronize()
        with torch.cuda.nvtx.range('steady'):op(x)
        torch.cuda.synchronize();return
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.capture/'manifest.json').read_text());assert manifest['complete']
    modelpath=ROOT/'models'/('Cosmos3-Edge-a9d944e' if a.model=='edge' else 'Cosmos3-Nano-411f42a')
    model=Cosmos3OmniTransformer.from_pretrained(str(modelpath/'transformer'),torch_dtype=torch.bfloat16,local_files_only=True)
    selected={}
    for r in manifest['rows']:
        shape=f'{r["m"]}:{r["n"]}:{r["k"]}'
        selected.setdefault(shape,r)
    rows=[];launches={}
    for shape,record in selected.items():
        li,part=record['name'].split('.');src=getattr(model.layers[int(li)].mlp_moe_gen,part)
        op=BudgetLinear(src);data=torch.load(a.capture/record['file'],weights_only=True,map_location='cpu')
        x=data['x'].cuda().repeat(math.ceil(record['m']/len(data['x'])),1)[:record['m']].contiguous()
        weight=src.weight.detach().cuda();gold=torch.nn.functional.linear(x,weight)
        bf16_ms=bench(lambda:torch.nn.functional.linear(x,weight))
        payload=a.output/(shape.replace(':','_')+'.pt')
        torch.save(dict(weight=src.weight.detach().cpu().clone(),x=x.cpu(),metadata=record),payload)
        for tier,(ac,wc) in TIERS.items():
            op.tier=tier;op.op.bm=128;op.op.num_stages=2
            reference=op(x).float();original_ms=bench(lambda:op(x));candidates=[]
            for bm in [32,64,128]:
                for stages in [2,3]:
                    op.op.bm=bm;op.op.num_stages=stages;y=op(x).float()
                    error=float((y-reference).norm()/reference.norm().clamp_min(1e-12))
                    assert torch.isfinite(y).all() and error<.004,(shape,tier,bm,stages,error)
                    candidates.append(dict(bm=bm,num_stages=stages,ms=bench(lambda:op(x)),relative_l2=error))
            winner=min(candidates,key=lambda c:c['ms']);op.op.bm=winner['bm'];op.op.num_stages=winner['num_stages']
            op.prepare(x,tier);packed=op.op.pack(x)
            packed_ms=bench(lambda:op.op.run_packed(x,packed))
            fixed_mask_ms=bench(lambda:op.op(x))
            pack_ms=bench(lambda:op.op.pack(x))
            select_ms=bench(lambda:op.prepare(x,tier)) if 0<ac<1 else 0.
            flops=2*record['m']*record['n']*record['k']*(1+ac+int(wc))
            rec=dict(model=a.model,shape=shape,tier=tier,task=record['task'],source=record['file'],
                original_ms=original_ms,optimized_ms=winner['ms'],bf16_ms=bf16_ms,packed_gemm_ms=packed_ms,
                fixed_mask_ms=fixed_mask_ms,pack_ms=pack_ms,select_ms=select_ms,
                achieved_tflops_operator=flops/winner['ms']/1e9,achieved_tflops_gemm=flops/packed_ms/1e9,
                bf16_relative_l2=float((reference-gold.float()).norm()/gold.float().norm().clamp_min(1e-12)),
                winner=winner,candidates=candidates,payload=str(payload))
            rows.append(rec);launches[shape+'|'+tier]={k:winner[k] for k in ['bm','num_stages']}
            print(a.model,shape,tier,'ms',round(winner['ms'],4),'vs original',round(original_ms/winner['ms'],3),flush=True)
        del op,x,weight,gold;gc.collect();torch.cuda.empty_cache()
    (a.output/'benchmark.json').write_text(json.dumps(dict(complete=True,model=a.model,rows=rows,launches=launches,
        scope='CUDA Graph operator timing; 3x10 ms per candidate; all packing and GPU selection included in optimized_ms; captured rows repeated to actual M for latency only'),indent=2)+'\n')


if __name__=='__main__':main()

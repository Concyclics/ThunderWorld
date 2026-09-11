"""Retune the closest chunk candidate per shape/tier, then remeasure alternately."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
import triton.testing
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers import Cosmos3OmniTransformer
from dynamic import ROOT,BudgetLinear
from chunk_pipeline import ChunkPipeline
from bench_execution import compare


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['edge','nano'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
    previous=json.loads((ROOT/f'runs5090/graph_fusion_{a.model}_bench/result.json').read_text())
    benchmark=json.loads((ROOT/f'runs5090/auto_budget_{a.model}_bench/benchmark.json').read_text())
    mp=ROOT/'models'/('Cosmos3-Edge-a9d944e' if a.model=='edge' else 'Cosmos3-Nano-411f42a')
    model=Cosmos3OmniTransformer.from_pretrained(str(mp/'transformer'),torch_dtype=torch.bfloat16,local_files_only=True)
    mlp=model.layers[0].mlp_moe_gen
    ops={n:BudgetLinear(getattr(mlp,n)) for n in ['up_proj','down_proj']}
    del mlp,model;gc.collect();rows=[]
    for record in previous['pipelines']:
        shape,tier=record['shape'],record['tier'];m,n,k=map(int,shape.split(':'))
        op=ops['up_proj' if n>k else 'down_proj'];op.tier=tier
        payload=next(r['payload'] for r in benchmark['rows'] if r['shape']==shape)
        x=torch.load(payload,weights_only=True,map_location='cpu')['x'].cuda()
        original=benchmark['launches'][shape+'|'+tier]
        def baseline():
            op.op.bm=original['bm'];op.op.num_stages=original['num_stages'];return op(x)
        reference=baseline().clone()
        chunk=int(min(['128','256','512'],key=lambda c:record['times'][c]['ms']))
        pipe=ChunkPipeline(op,x,tier,chunk);trials=[]
        def run(bm,stages):
            op.op.bm=bm;op.op.num_stages=stages;return pipe(x)
        for bm in [32,64,128]:
            for stages in [2,3]:
                assert torch.equal(run(bm,stages),reference),(shape,tier,bm,stages)
                ms=triton.testing.do_bench_cudagraph(lambda:run(bm,stages),rep=10)
                trials.append(dict(bm=bm,num_stages=stages,ms=ms,bitwise_equal=True))
        best=min(trials,key=lambda t:t['ms'])
        times=compare({'original':baseline,'tuned':lambda:run(best['bm'],best['num_stages'])},5)
        speedup=times['original']['ms']/times['tuned']['ms']
        rows.append(dict(shape=shape,tier=tier,chunk=chunk,trials=trials,best=best,times=times,speedup=speedup,bitwise_equal=True))
        print(a.model,shape,tier,'chunk',chunk,'launch',best['bm'],best['num_stages'],'speedup',round(speedup,4),flush=True)
    a.output.write_text(json.dumps(dict(complete=True,model=a.model,rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='For each shape/tier, select closest of three chunks from the earlier study, tune BM32/64/128 x stages2/3, then five fresh alternating-order 15 ms Graph timings. All work included; no E2E claim or exhaustive joint search.'),indent=2)+'\n')


if __name__=='__main__':main()

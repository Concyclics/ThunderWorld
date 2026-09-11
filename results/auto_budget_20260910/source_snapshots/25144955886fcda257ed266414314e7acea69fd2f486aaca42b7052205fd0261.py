"""Paired gate/up microbenchmarks with real distinct Nano projection weights."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import torch
import triton.testing
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers import Cosmos3OmniTransformer
from dynamic import ROOT,BudgetLinear,BudgetController


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    policy=json.loads(a.policy.read_text());b=json.loads(a.benchmark.read_text())
    model=Cosmos3OmniTransformer.from_pretrained(str(ROOT/'models/Cosmos3-Nano-411f42a/transformer'),torch_dtype=torch.bfloat16,local_files_only=True)
    c=BudgetController(SimpleNamespace(call=0),policy,share_pack=False);c.paired_names={'0.gate_proj','0.up_proj'}
    gate=BudgetLinear(model.layers[0].mlp_moe_gen.gate_proj,c,'0.gate_proj')
    up=BudgetLinear(model.layers[0].mlp_moe_gen.up_proj,c,'0.up_proj')
    del model;gc.collect()
    selected={r['shape']:r for r in b['rows'] if r['tier']=='act2' and int(r['shape'].split(':')[1])>int(r['shape'].split(':')[2])}
    rows=[]
    for shape,record in selected.items():
        x=torch.load(record['payload'],weights_only=True,map_location='cpu')['x'].cuda()
        for op in [gate,up]:op.warmed_shapes.add(tuple(x.shape))
        for at,bt in [('a50','a50'),('act2','act2'),('a50','dual'),('base','dual')]:
            c.resolve=lambda name:at if name.endswith('gate_proj') else bt
            def run():return gate(x),up(x)
            c.share_pack=False;reference=tuple(y.clone() for y in run())
            c.share_pack=True;actual=run()
            assert all(torch.equal(x,y) for x,y in zip(actual,reference))
            times={False:[],True:[]}
            for i in range(5):
                for shared in ([False,True] if i%2==0 else [True,False]):
                    c.share_pack=shared;times[shared].append(triton.testing.do_bench_cudagraph(run,rep=20))
            plain=statistics.median(times[False]);shared=statistics.median(times[True])
            rows.append(dict(shape=shape,gate=at,up=bt,plain_ms=plain,shared_ms=shared,speedup=plain/shared,
                samples_plain_ms=times[False],samples_shared_ms=times[True],bitwise_equal=True))
            print(shape,at,bt,'speedup',round(plain/shared,3),flush=True)
    a.output.write_text(json.dumps(dict(complete=True,rows=rows,scope='Five alternating-order CUDA Graph measurements per configuration, 20 ms each; same real distinct gate/up weights, same inputs/tiers, includes packing and GPU selection'),indent=2)+'\n')


if __name__=='__main__':main()

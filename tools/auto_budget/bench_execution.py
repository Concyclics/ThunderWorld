"""Full real-weight FFN fusion and full-input-scale two-slot operator timings."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import torch
import triton.testing
import diffusers.utils.import_utils as imports
imports._xformers_available=False
from diffusers import Cosmos3OmniTransformer
from dynamic import ROOT,BudgetLinear,TIERS
from fusion import activate_with_factor
from chunk_pipeline import ChunkPipeline


def compare(fns,repeats=5):
    times={k:[] for k in fns}
    for i in range(repeats):
        order=list(fns)
        if i%2:order.reverse()
        for k in order:times[k].append(triton.testing.do_bench_cudagraph(fns[k],rep=15))
    return {k:dict(ms=statistics.median(v),samples_ms=v) for k,v in times.items()}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['edge','nano'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
    benchmark=json.loads((ROOT/f'runs5090/auto_budget_{a.model}_bench/benchmark.json').read_text())
    mp=ROOT/'models'/('Cosmos3-Edge-a9d944e' if a.model=='edge' else 'Cosmos3-Nano-411f42a')
    model=Cosmos3OmniTransformer.from_pretrained(str(mp/'transformer'),torch_dtype=torch.bfloat16,local_files_only=True)
    mlp=model.layers[0].mlp_moe_gen;ops={n:BudgetLinear(m) for n,m in mlp.named_children() if isinstance(m,torch.nn.Linear)}
    del mlp,model;gc.collect()
    up_records={r['shape']:r for r in benchmark['rows'] if r['tier']=='base' and int(r['shape'].split(':')[1])>int(r['shape'].split(':')[2])}
    rows=[];pipelines=[]
    for shape,r in up_records.items():
        x=torch.load(r['payload'],weights_only=True,map_location='cpu')['x'].cuda()
        for tier in ['a50','act2','a50w','dual']:
            for op in ops.values():
                op.tier=tier;key=f'{len(x)}:{op.out_features}:{op.in_features}|{tier}'
                launch=benchmark['launches'][key];op.op.bm=launch['bm'];op.op.num_stages=launch['num_stages']
            def chain(fused=False):
                if a.model=='edge':
                    up=ops['up_proj'](x)
                    if fused:z,f=activate_with_factor(up)
                    else:z=torch.relu(up).square()
                else:
                    gate=torch.nn.functional.silu(ops['gate_proj'](x));up=ops['up_proj'](x)
                    if fused:z,f=activate_with_factor(gate,up)
                    else:z=gate*up
                return ops['down_proj'](z,factor=f if fused else None)
            reference=chain();actual=chain(True);assert torch.equal(reference,actual),(shape,tier)
            times=compare({'original':chain,'fused':lambda:chain(True)})
            rows.append(dict(shape=shape,tier=tier,bitwise_equal=True,times=times,speedup=times['original']['ms']/times['fused']['ms']))
            print('FFN',a.model,shape,tier,round(rows[-1]['speedup'],3),flush=True)
        # Full operator comparison at the same shape, down and up, with every
        # stage included. The global scale/mask are computed once for all chunks.
        for part in ['up_proj','down_proj']:
            op=ops[part];keyshape=f'{len(x)}:{op.out_features}:{op.in_features}'
            record=next(r for r in benchmark['rows'] if r['shape']==keyshape)
            z=torch.load(record['payload'],weights_only=True,map_location='cpu')['x'].cuda()
            for tier in ['a50','act2','a50w','dual']:
                op.tier=tier;cfg=benchmark['launches'][keyshape+'|'+tier];op.op.bm=cfg['bm'];op.op.num_stages=cfg['num_stages']
                reference=op(z).clone();fns={'original':lambda:op(z)}
                bank={chunk:ChunkPipeline(op,z,tier,chunk) for chunk in [128,256,512]}
                for chunk,pipe in bank.items():assert torch.equal(pipe(z),reference),(a.model,keyshape,tier,chunk)
                fns.update({str(chunk):(lambda pipe=pipe:pipe(z)) for chunk,pipe in bank.items()})
                times=compare(fns,3);winner=min(times,key=lambda k:times[k]['ms'])
                pipelines.append(dict(shape=keyshape,tier=tier,bitwise_equal=True,times=times,winner=winner,
                    best_speedup=times['original']['ms']/times[winner]['ms']))
                print('Pipeline',a.model,keyshape,tier,winner,round(pipelines[-1]['best_speedup'],3),flush=True)
    a.output.write_text(json.dumps(dict(complete=True,model=a.model,ffn=rows,pipelines=pipelines,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Local layer-0 full FFN uses distinct real projections and each variant own intermediate outputs. Five alternating-order 15 ms Graph timings. Pipeline is full operator with real shape/input, global factor/mask and two reusable slots; three alternating-order timings per chunk size; no E2E speed claim.'),indent=2)+'\n')


if __name__=='__main__':main()

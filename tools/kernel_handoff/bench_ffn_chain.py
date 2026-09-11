"""Layer-0 SiLU(gate)*up -> down using each scheme's own intermediate values."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
import torch
import triton.testing
from bench import ROOT,FFN,load_module,repeat_rows,top_mask
from frozen_baseline import load as load_frozen,share_weights


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ['data','weights','tuning','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    profile=json.loads(a.tuning.read_text())['configs']
    native=load_module('chain_native',ROOT/'experiments/ffn_block_repair_20260907/native.py')
    old_native,_,old_hashes=load_frozen()
    payload={kind:torch.load(a.data/f'ffn_l0_{kind}.pt',weights_only=True) for kind in ['gate','up','down']}
    for x,y in zip(payload['gate']['eval_inputs'],payload['up']['eval_inputs']):assert torch.equal(x,y)
    sources=[Path(__file__),Path(native.__file__),*sorted((ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py'))]
    report={'complete':False,'scope':'Local layer-0 FFN chain; current candidate gate/up outputs feed its own down projection; no diffusion rollout',
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'baseline_source_sha256':old_hashes,'rows':[]}
    for m in [128,512]:
        for mode,(pa,pw) in FFN.items():
            ops={};original={};weights={}
            for kind in ['gate','up','down']:
                asset=f'ffn_l0_{kind}.pt';d=payload[kind];ready=torch.load(a.weights/asset,weights_only=True)
                assert ready['input_sha256']==hashlib.sha256((a.data/asset).read_bytes()).hexdigest()
                cfg=profile[f'ffn:{asset}:{m}:{mode}'];n,k=ready['weight_bf16'].shape
                am=top_mask(d['activation_score'].reshape(-1,128).sum(-1),pa).repeat_interleave(128)[None].repeat(m//128,1).cuda()
                wm=top_mask(d['weight_block_scores']['128'],pw).repeat_interleave(128,0).repeat_interleave(128,1).cuda()
                op=native.Native(ready['codes'].cuda(),ready['scales'].cuda(),ready['global_scale'].cuda(),d['prescale'].cuda(),am,wm,
                                 b=128,bm=cfg['tile_m'],interleave=not cfg['indexed_repair'],num_warps=cfg['num_warps'],
                                 num_stages=cfg['num_stages'],split_k=cfg.get('split_k',1))
                ops[kind]=op;original[kind]=share_weights(op,old_native.Native);weights[kind]=ready['weight_bf16'].cuda()
                del ready,am,wm
            def chain(x,bank):return bank['down'](torch.nn.functional.silu(bank['gate'](x))*bank['up'](x))
            def bf16(x):
                z=torch.nn.functional.silu(torch.nn.functional.linear(x,weights['gate']))*torch.nn.functional.linear(x,weights['up'])
                return torch.nn.functional.linear(z,weights['down'])
            row={'m':m,'preset':mode,'checks':[]}
            for i,step in enumerate([0,17,34]):
                x=repeat_rows(payload['gate']['eval_inputs'][i].cuda(),m,0)
                y=chain(x,ops).float();old=chain(x,original).float();gold=bf16(x).float()
                error=float((y-gold).norm()/gold.norm());old_error=float((old-gold).norm()/gold.norm())
                assert torch.isfinite(y).all() and error<=old_error+.002
                row['checks'].append({'step':step,'bf16_relative_l2':error,'original_bf16_relative_l2':old_error,
                                      'original_output_relative_l2':float((y-old).norm()/old.norm())})
            x=repeat_rows(payload['gate']['eval_inputs'][0].cuda(),m,0)
            torch.cuda.synchronize()
            fns={'fp4':lambda:chain(x,ops),'original_fp4':lambda:chain(x,original),'bf16':lambda:bf16(x)}
            samples={key:[] for key in fns};rng=random.Random(5090)
            for _ in range(5):
                keys=list(fns);rng.shuffle(keys)
                for key in keys:samples[key].append(triton.testing.do_bench_cudagraph(fns[key],rep=40))
            row['latency_ms']={key:dict(median=statistics.median(v),min=min(v),max=max(v),samples=v) for key,v in samples.items()}
            report['rows'].append(row);a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(m,mode,{k:round(v['median']*1000,2) for k,v in row['latency_ms'].items()},flush=True)
    report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

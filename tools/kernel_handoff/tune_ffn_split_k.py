"""Split-K search with fixed K128 supports and FP32 partial reduction."""
import argparse
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path

import torch
import triton.testing
from bench import ROOT, FFN, decode, load_module, repeat_rows, top_mask
from thunderworld_fp4.linear import _pack


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--search',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--rep-ms',type=int,default=20)
    args=p.parse_args()
    previous=json.loads(args.search.read_text());assert previous['complete']
    winners={(r['asset'],r['m'],r['preset']):next(c for c in r['candidates'] if c['key']==r['winner']) for r in previous['rows']}
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    native=load_module('tuning_native',ROOT/'experiments/ffn_block_repair_20260907/native.py')
    report={'complete':False,'gpu':torch.cuda.get_device_name(),'rows':[],
            'source_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'scope':'K128 support fixed; online prescale/amax/pack/GEMM timed; weights offline'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for asset in ['ffn_l0_gate.pt','ffn_l0_up.pt','ffn_l0_down.pt']:
        payload=torch.load(args.data/asset,weights_only=True)
        ready=torch.load(args.weights/asset,weights_only=True)
        assert ready['input_sha256']==hashlib.sha256((args.data/asset).read_bytes()).hexdigest()
        c,s,g=(ready[key].cuda() for key in ['codes','scales','global_scale'])
        d=payload['prescale'].cuda();n,k=ready['weight_bf16'].shape
        wp=decode(c,s)*g
        for m,mode in itertools.product([128,512],FFN):
            pa,pw=FFN[mode]
            x=repeat_rows(payload['eval_inputs'][0].cuda(),m,0)
            am=top_mask(payload['activation_score'].reshape(-1,128).sum(-1),pa).repeat_interleave(128)[None].repeat(m//128,1).cuda()
            wm=top_mask(payload['weight_block_scores']['128'],pw).repeat_interleave(128,0).repeat_interleave(128,1).cuda()
            ac,sc,ag,_=_pack(x,d,True,True,x);ap=decode(ac,sc)[:,:m]*ag
            ref=ap[0]@wp[0].T+(ap[1]*am.repeat_interleave(128,0))@wp[0].T+ap[0]@(wp[1]*wm).T
            del ac,sc,ag,ap
            row={'asset':asset,'m':m,'preset':mode,'block_k':128,'candidates':[]}
            ops={}
            old=winners[asset,m,mode]
            plans=[(old['tile_m'],old['num_stages'],old['interleave'],1)]
            plans += [(bm,stages,True,sk) for bm,stages,sk in itertools.product([32,64,128],[2,3],[2,4,8])]
            for bm,stages,inter,sk in plans:
                key=f'm{bm}_st{stages}_sk{sk}_'+('reuse' if inter else 'indices')
                entry={'key':key,'tile_m':bm,'num_stages':stages,'num_warps':4,'interleave':inter,'split_k':sk}
                try:
                    op=native.Native(c,s,g,d,am,wm,b=128,bm=bm,interleave=inter,num_stages=stages,split_k=sk)
                    y=op(x).float();error=float((y-ref).norm()/ref.norm())
                    entry.update(decoded_reference_relative_l2=error,passed=bool(torch.isfinite(y).all()) and error<.004)
                    if entry['passed']:ops[key]=op
                except Exception as exc:entry.update(passed=False,error=str(exc))
                row['candidates'].append(entry)
            samples={key:[] for key in ops};rng=random.Random(5090)
            for _ in range(args.rounds):
                order=list(ops);rng.shuffle(order)
                for key in order:samples[key].append(triton.testing.do_bench_cudagraph(lambda:ops[key](x),rep=args.rep_ms))
            for entry in row['candidates']:
                if entry['key'] in samples:
                    values=samples[entry['key']]
                    entry['latency_ms']=dict(median=statistics.median(values),min=min(values),max=max(values),samples=values)
            good=[e for e in row['candidates'] if e['passed']]
            row['winner']=min(good,key=lambda e:e['latency_ms']['median'])['key']
            report['rows'].append(row);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(asset,m,mode,row['winner'],round(min(e['latency_ms']['median'] for e in good)*1000,2),'us',flush=True)
            del ops,op,ref,x,y
        del c,s,g,wp
    report['complete']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

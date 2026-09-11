"""V storage experiment: preserve quantized values, change physical byte layout.

Every candidate must be bitwise equal to the already oracle-validated original
layout at the same tile/split/warp settings. Selection runs on current Q/K/V.
"""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path
import torch
import triton.testing
from bench import ROOT, repeat_rows
from thunderworld_fp4 import preset, policy_from_head_atlas
from thunderworld_fp4.runtime import _forward


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--search',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    search=json.loads(a.search.read_text());assert search['complete']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    report={'complete':False,'gpu':torch.cuda.get_device_name(),'rows':[],
            'oracle_search_sha256':hashlib.sha256(a.search.read_bytes()).hexdigest(),
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in (ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py')}}
    for old in search['rows']:
        asset,m,mode=old['asset'],old['m'],old['preset']
        payload=torch.load(a.data/asset,weights_only=True)
        q=repeat_rows(payload['q'].cuda(),m,1);k=payload['k'].cuda();v=payload['v'].cuda()
        cfg=(policy_from_head_atlas(json.loads((a.data/'head_atlas.json').read_text()),coverage=.25,protected_coverage=.75).resolve(layer=payload['layer'])
             if mode=='head_guided' else preset(mode))
        quotas=torch.tensor(cfg.head_counts(q.shape[2],k.shape[1]),device='cuda',dtype=torch.int32)
        old_candidates={c['key']:c for c in old['candidates']}
        plans=[(old_candidates[key],False) for key in dict.fromkeys([old['winner'],'m32_s2_w4'])]
        plans += [(old_candidates[f'm{bm}_s{sp}_w{w}'],True)
                  for (bm,w),sp in itertools.product([(16,4),(32,4),(64,4),(64,8)],[1,2,4,8])]
        row={'asset':asset,'m':m,'preset':mode,'candidates':[]};fns={}
        for old_entry,transpose in plans:
            if not old_entry['passed']:continue
            entry={key:old_entry[key] for key in ['tile_m','splits','num_warps','decoded_reference_relative_l2','bf16_relative_l2']}
            entry.update(key=old_entry['key']+('_transposed' if transpose else '_original'),transpose_v=transpose)
            original=replace(cfg,query_tile=entry['tile_m'],splits=entry['splits'],num_warps=entry['num_warps'],transpose_v=False)
            candidate=replace(original,transpose_v=transpose)
            try:
                ref=_forward(q,k,v,original,counts_tensor=quotas)[0]
                fn=lambda c=candidate:_forward(q,k,v,c,counts_tensor=quotas)[0]
                y=fn();entry['bitwise_equal']=torch.equal(y,ref)
                entry['passed']=old_entry['passed'] and entry['bitwise_equal']
                if entry['passed']:fns[entry['key']]=fn
            except Exception as exc:entry.update(passed=False,error=str(exc))
            row['candidates'].append(entry)
        samples={key:[] for key in fns};rng=random.Random(5090)
        for _ in range(5):
            keys=list(fns);rng.shuffle(keys)
            for key in keys:samples[key].append(triton.testing.do_bench_cudagraph(fns[key],rep=20))
        for entry in row['candidates']:
            if entry['passed']:
                values=samples[entry['key']]
                entry['latency_ms']=dict(median=statistics.median(values),min=min(values),max=max(values),samples=values)
        best=min((e for e in row['candidates'] if e['passed']),key=lambda e:e['latency_ms']['median'])
        row['winner']=best['key'];report['rows'].append(row)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(asset,m,mode,best['key'],round(best['latency_ms']['median']*1000,2),'us',flush=True)
    report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

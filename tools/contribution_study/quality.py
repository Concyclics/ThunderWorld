"""Paired saved-video quality, strictly separate from GPU request timings."""
import argparse,json,statistics
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--run',action='append',required=True,help='name=directory')
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    roots=dict(v.split('=',1) for v in a.run)
    if len(roots)!=len(a.run):raise ValueError('Duplicate method')
    roots={k:Path(v) for k,v in roots.items()}
    reference=json.loads((a.reference/'result.json').read_text())
    for folder in roots.values():
        receipt=json.loads((folder/'result.json').read_text())
        if not receipt['complete']:raise ValueError('Incomplete run')
        for key in ('case','image_sha256','model_identity','targets'):
            if receipt[key]!=reference[key]:raise ValueError('Unmatched '+key)
        for seed in a.seeds:
            row=next(r for r in receipt['rows'] if r['seed']==seed)
            if row['steps']!=35:raise ValueError('Changed steps')
    import torch,lpips
    import hashlib,numpy as np
    def load(root,seed):
        receipt=json.loads((root/'result.json').read_text())
        if not receipt['complete']:raise ValueError('Incomplete generation')
        row=next(r for r in receipt['rows'] if r['seed']==seed)
        with np.load(root/row['outputs']) as archive:signals=dict(archive)
        if {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in signals.items()}!=row['sha256']:
            raise ValueError('Output hash mismatch')
        return signals
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8);torch.hub.set_dir('/tmp/thunderworld-lpips-cache')
    metric=lpips.LPIPS(net='alex',version='0.1').eval().cpu()
    result=dict(complete=False,performance_eligible=False,seeds=a.seeds,rows=[],
        reference_sha256=sha(a.reference/'result.json'),
        receipts={k:sha(v/'result.json') for k,v in roots.items()},
        scope='BF16 agreement, not task success. Development results cannot establish a final quality gate.')
    for seed in a.seeds:
        ref=load(a.reference,seed)
        for name,root in roots.items():
            row=dict(name=name,seed=seed,metrics=compare(load(root,seed),ref,metric))
            result['rows'].append(row);write(a.output,result);print(json.dumps(row),flush=True)
    result['means']={name:{m:statistics.mean(r['metrics']['video'][m] for r in result['rows'] if r['name']==name)
        for m in ('lpips','ssim_mean','relative_l2','rmse')} for name in roots}
    result['complete']=True;write(a.output,result)


if __name__=='__main__':main()

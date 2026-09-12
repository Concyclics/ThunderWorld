"""Measure causal single-block repair benefit, then compile per-head quotas.

All base edges remain present. Each intervention enables one K block for all
query tiles/heads; errors are measured separately, since heads are independent.
This is offline development analysis, never a runtime teacher computation.
"""
import argparse,json,math
from pathlib import Path
import torch
from thunderworld_fp4.config import RepairConfig
from thunderworld_fp4.runtime import _forward


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--coverage',type=float,default=.75)
    a=p.parse_args()
    if not 0<=a.coverage<=1:raise ValueError('Invalid coverage')
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.capture/'manifest.json').read_text())
    if manifest['split']!='development' or not manifest['complete']:raise ValueError('Development capture required')
    records=[];rules=[]
    for row in manifest['rows']:
        if row['kind']!='attention':continue
        data=torch.load(a.capture/row['file'],weights_only=True,map_location='cpu')
        q,k,v,gold=(data[n].cuda() for n in ('q','k','v','y'))
        h,m,n=q.shape[2],q.shape[1],k.shape[1];mb=math.ceil(m/128);nb=math.ceil(n/128)
        mask=torch.zeros((h,mb,nb),device=q.device,dtype=torch.bool)
        cfg=RepairConfig(coverage=0,terms=4,query_tile=32,transpose_v=True)
        base=_forward(q,k,v,cfg,repair_mask=mask)[0].float()
        baseline=(base-gold.float()).square().sum((0,1,3))
        benefits=[]
        for block in range(nb):
            mask.zero_();mask[:,:,block]=True
            y=_forward(q,k,v,cfg,repair_mask=mask)[0].float()
            benefits.append(baseline-(y-gold.float()).square().sum((0,1,3)))
        gains=torch.stack(benefits,1).cpu()
        # Spend at most the requested quota; never propose measured harmful repairs.
        remaining=math.ceil(a.coverage*h*nb)
        order=torch.argsort(gains.flatten(),descending=True,stable=True)
        order=order[gains.flatten()[order]>0][:remaining]
        counts=torch.bincount(order//nb,minlength=h)
        ph=row['phase'];bounds=[0,7,18,28,35]
        budgets=[[i,(int(c)-.25)/nb if c else 0.] for i,c in enumerate(counts)]
        # All measured strata are explicit; untouched layers retain full-four anchor.
        rules.append(dict(layers=[int(row['name'])],steps=bounds[ph:ph+2],branches=[str(row['branch'])],
            config=dict(coverage=0,terms=4,head_budgets=budgets,query_tile=32,transpose_v=True)))
        records.append(dict(**row,single_block_benefit=gains.tolist(),head_counts=counts.tolist()))
    a.output.write_text(json.dumps(dict(default=dict(coverage=1,terms=4,query_tile=32,transpose_v=True),rules=rules),indent=2)+'\n')
    a.output.with_suffix('.evidence.json').write_text(json.dumps(dict(complete=True,records=records,
        scope='Development interventions. Joint mask interactions and complete trajectories still require validation.'),indent=2)+'\n')


if __name__=='__main__':main()

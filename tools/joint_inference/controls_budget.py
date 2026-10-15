"""Uncalibrated controls: uniform A-first, W-first and balanced projection repair.

Attention remains dense base FP4 in these attribution controls. Exact same input,
tiles, scopes and requests are used. Integer rounding and spare budget are reported.
"""
import argparse,json,math
from pathlib import Path
from thunderworld_fp4.cosmos.budget import linear_cost,attention_cost,PHASES
from export_budget import write_candidate
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--joint-attention',action='store_true',help='Additional uniform joint controls; W-only keeps attention base because the runtime has no W-only attention repair')
    p.add_argument('--budgets',type=float,nargs='+',default=[1.,1.125,1.25,1.5])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from thunderworld_fp4.cosmos.budget import BUDGETS
    if len(set(a.budgets))!=len(a.budgets) or any(b not in BUDGETS for b in a.budgets):raise ValueError('Invalid budgets')
    import torch
    capture=json.loads((a.capture/'manifest.json').read_text());manifest=json.loads((a.assets/'manifest.json').read_text())
    search=dict(task=capture['case']['task'],branches=1 if capture['case']['task']=='policy' else 2,model_identity=capture['model_identity'],cohort_sha256=capture['cohort_sha256'],modules={},attention={})
    for name,entry in manifest['modules'].items():
        r=next(r for r in capture['rows'] if r['kind']=='linear' and r['name']==name)
        m,k=r['shapes']['x'];n=r['shapes']['y'][1]
        asset=torch.load(a.assets/entry['file'],weights_only=True,map_location='cpu')
        order=torch.argsort(asset['score'].flatten(),descending=True,stable=True).tolist()
        search['modules'][name]={**entry,'shape':[m,n,k],'support_order':order}
    for r in capture['rows']:
        if r['kind']=='attention':
            q,k=r['shapes']['q'],r['shapes']['k'];search['attention'][r['name']]=dict(shape=[q[1],k[1],q[2],k[2]])
    def allocate(budget,strategy,fraction):
        groups={};base=total=0
        for branch in range(search['branches']):
            for ph in range(4):
                calls=PHASES[ph+1]-PHASES[ph]
                for name,entry in search['modules'].items():
                    m,n,k=entry['shape'];af=fraction if strategy=='a' else fraction/2 if strategy=='balanced' else 0.
                    wf=fraction if strategy=='w' else fraction/2 if strategy=='balanced' else 0.
                    na=math.floor(k//128*af);ids=entry['support_order'][:math.floor(n//128*(k//128)*wf)]
                    cost=linear_cost(m,n,k,a_blocks=na,w_blocks=ids)
                    key=f'linear|{branch}|{ph}|{name}'
                    groups[key]=dict(kind='linear',name=name,branch=branch,phase=ph,a_blocks=na,w_blocks=ids,cross=False)
                    base+=cost['base_mma']*calls;total+=cost['total_mma']*calls
                for name,entry in search['attention'].items():
                    shape=entry['shape'];nt=math.ceil(shape[1]/128)
                    fraction_attn=(fraction if strategy=='a' else fraction/2 if strategy=='balanced' else 0.) if a.joint_attention else 0.
                    total_quota=math.floor(nt*shape[2]*fraction_attn)
                    quota,remainder=divmod(total_quota,shape[2]);terms=2 if strategy=='a' else 3
                    # Even integer quotas, differing by at most one block. The
                    # remainder follows head index, with no calibrated head finding.
                    counts=[quota+(head<remainder) for head in range(shape[2])]
                    setting=dict(qk_counts=counts,pv_counts=counts,qk_terms=terms,pv_terms=terms)
                    cost=attention_cost(*shape,**setting)
                    key=f'attention|{branch}|{ph}|{name}'
                    groups[key]=dict(kind='attention',name=name,branch=branch,phase=ph,**setting)
                    base+=cost['base_mma']*calls;total+=cost['total_mma']*calls
        return dict(assignments=groups,budget=budget,base_mma=base,total_mma=total,ratio=total/base,
                    strategy=strategy,uniform_fraction=fraction,scope='Uncalibrated uniform joint attention/projection allocation' if a.joint_attention else 'Uncalibrated fixed allocation, ordinary FP4 attention')
    candidates=[]
    for budget in a.budgets:
        for strategy in (('a','balanced') if a.joint_attention else ('a','w','balanced')):
            if a.joint_attention and budget==1.:continue  # Identical empty-repair control already exists.
            lo,hi=0.,1.
            for _ in range(18):
                mid=(lo+hi)/2;candidate=allocate(budget,strategy,mid)
                if candidate['ratio']<=budget:lo=mid
                else:hi=mid
            candidate=allocate(budget,strategy,lo);name=f'control_{"joint_" if a.joint_attention else ""}b{budget:g}_{strategy}'
            folder=a.output/name;write_candidate(folder,search,candidate,a.assets)
            candidates.append(dict(name=name,budget=budget,path=str(folder.resolve()),ratio=candidate['ratio']))
    atomic_json(a.output/'controls.json',dict(complete=True,candidates=candidates))


if __name__=='__main__':main()

"""Native head/phase repair interventions, with seed-separated local allocation.

All base attention connections remain. Head priors are tested against actual
attention-output interventions, never interpreted as causal evidence by themselves.
"""
import argparse,json,math
from pathlib import Path
from baselines import sha,write
from head_allocation import allocate_head_quotas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',nargs='+')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();capture=json.loads((a.capture/'manifest.json').read_text())
    if not capture['complete'] or capture['split']!='development':raise ValueError('Development capture required')
    layers=a.layers or sorted({r['name'] for r in capture['rows'] if r['kind']=='attention'},key=int)
    identity=dict(capture_sha256=sha(a.capture/'manifest.json'),source_sha256=sha(__file__),layers=layers)
    result=dict(complete=False,performance_eligible=False,provenance=identity,layers={},
        scope='Native full-attention output interventions; seed7101 proposes head quotas, seed7102 evaluates combinations. '
              'This is local calibration evidence, not independent generation validation. '
              'AR mass accounts for the prepended AR prefix and is normalized by token count.')
    if a.output.exists():
        if not a.resume:raise FileExistsError(a.output)
        result=json.loads(a.output.read_text())
        if result['provenance']!=identity:raise ValueError('Resume provenance mismatch')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        result['resource']=select_shared_device()
        import torch
        from thunderworld_fp4 import fp4_attention,RepairConfig
        from thunderworld_fp4.cosmos.budget import attention_cost
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        write(a.output,result)
        with torch.inference_mode():
            for layer in layers:
                if layer in result['layers']:continue
                rows=[r for r in capture['rows'] if r['kind']=='attention' and r['name']==layer]
                if len(rows)!=16:raise ValueError('Missing phase/branch panels')
                proposals={};interventions=[];comparisons=[]
                for r in sorted(rows,key=lambda r:(r['seed'],r['phase'],r['branch'])):
                    if sha(a.capture/r['file'])!=r['sha256']:raise ValueError('Corrupt panel')
                    panel=torch.load(a.capture/r['file'],weights_only=True)
                    q,k,v=(panel[key].cuda() for key in ('q','k','v'))
                    gold=panel['y'].cuda().float();m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2]
                    nt=math.ceil(n/128);common=dict(query_tile=32,transpose_v=True,sparse_pack=True)
                    configs={
                        'base':RepairConfig(coverage=0.,**common),
                        'qk3':RepairConfig(coverage=0.,head_quotas=(nt,)*h,pv_head_quotas=(0,)*h,**common),
                        'pv3':RepairConfig(coverage=0.,head_quotas=(0,)*h,pv_head_quotas=(nt,)*h,**common),
                        'three':RepairConfig(coverage=1.,terms=3,**common),
                        'four':RepairConfig(coverage=1.,terms=4,**common),
                    }
                    errors={}
                    denominator=gold.square().sum().clamp_min(1e-20)
                    for label,cfg in configs.items():
                        pred=fp4_attention(q,k,v,cfg).float()
                        errors[label]=((pred-gold).square().sum((0,1,3))/denominator).cpu()
                    logits=torch.einsum('bmhd,bnhd->bhmn',q.float(),k.repeat_interleave(h//hk,2).float())/math.sqrt(128)
                    probability=logits.softmax(-1)
                    ar=n-m
                    if ar<=0:raise ValueError('Expected prepended AR prefix')
                    mass=probability[...,:ar].sum(-1).mean((0,2))
                    per_token=mass/ar-((1-mass)/m)
                    energy=q.float().square().mean((0,1,3))
                    gain=errors['base']-errors['three']
                    key=(r['phase'],r['branch'])
                    interventions.append(dict(seed=r['seed'],phase=r['phase'],branch=r['branch'],
                        errors={k:x.tolist() for k,x in errors.items()},ar_tokens=ar,dm_tokens=m,
                        ar_mass=mass.cpu().tolist(),ar_minus_dm_mass_per_token=per_token.cpu().tolist(),
                        q_energy=energy.cpu().tolist()))
                    if r['seed']==7101:
                        proposals[key]={'measured_gain':gain,'ar_mass_per_token':per_token.cpu(),'q_energy':energy.cpu()}
                        table=[errors['base']]
                        for quota in range(1,nt):
                            cfg=RepairConfig(coverage=0.,head_quotas=(quota,)*h,terms=3,**common)
                            pred=fp4_attention(q,k,v,cfg).float()
                            table.append(((pred-gold).square().sum((0,1,3))/denominator).cpu())
                        table.append(errors['three'])
                        table=torch.stack(table,1).tolist()
                        proposals[key]['quota_solution']=allocate_head_quotas(table,h*math.ceil(nt*.5))
                        interventions[-1]['measured_quota_losses']=table
                    elif r['seed']==7102:
                        candidate_configs={'uniform':RepairConfig(coverage=.5,terms=3,**common)}
                        for label,score in proposals[key].items():
                            if label=='quota_solution':continue
                            order=torch.argsort(score,descending=True,stable=True).tolist()
                            quotas=[0]*h
                            # Equal total integer quota, including odd tile counts.
                            remaining=h*math.ceil(nt*.5)
                            for head in order:
                                quotas[head]=min(nt,remaining);remaining-=quotas[head]
                            candidate_configs[label]=RepairConfig(coverage=0.,head_quotas=tuple(quotas),terms=3,**common)
                        candidate_configs['marginal_quota']=RepairConfig(coverage=0.,
                            head_quotas=proposals[key]['quota_solution'],terms=3,**common)
                        costs={}
                        for label,cfg in candidate_configs.items():
                            pred=fp4_attention(q,k,v,cfg).float()
                            cost=attention_cost(m,n,h,hk,bm=32,qk_counts=cfg.head_counts(h,n),
                                pv_counts=cfg.pv_counts(h,n),qk_terms=3,pv_terms=3)
                            costs[label]=cost['total_mma']
                            comparisons.append(dict(seed=r['seed'],phase=r['phase'],branch=r['branch'],method=label,
                                nmse=float((pred-gold).square().sum()/denominator),cost=cost,config=cfg.to_dict()))
                        if len(set(costs.values()))!=1:raise ValueError('Head allocation controls are not MMA matched')
                    del logits,probability,panel,q,k,v,gold,pred
                result['layers'][layer]=dict(interventions=interventions,comparisons=comparisons)
                write(a.output,result);print(json.dumps(dict(layer=layer,complete=True)),flush=True)
            result['complete']=True;write(a.output,result)


if __name__=='__main__':main()

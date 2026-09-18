"""Add measured integer-head combinations to the existing whole-request search.

Head tables only propose quotas. Every proposed combination is reevaluated on
both teacher seeds before its loss is admitted to the existing allocator.
"""
import argparse,json,math,os
from pathlib import Path
from baselines import sha,write
from head_allocation import allocate_head_quotas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('search','study','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    identity={k:sha(getattr(a,k)/'manifest.json' if k=='capture' else getattr(a,k)) for k in ('search','study','capture')}
    identity['source']=sha(__file__)
    a.output.mkdir(parents=True,exist_ok=a.resume);target=a.output/'search.json'
    result=json.loads(a.search.read_text());study=json.loads(a.study.read_text());capture=json.loads((a.capture/'manifest.json').read_text())
    if not all(v['complete'] for v in (result,study,capture)):raise ValueError('Complete inputs required')
    result.update(complete=False,augmentation=identity,augmented_layers=[])
    if target.exists():
        result=json.loads(target.read_text())
        if result['augmentation']!=identity:raise ValueError('Resume provenance mismatch')
    for entry in result['modules'].values():
        src=a.search.parent/entry['file'];dst=a.output/entry['file']
        if sha(src)!=entry['sha256']:raise ValueError('Asset hash mismatch')
        if not dst.exists():os.link(src,dst)
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        result['augmentation_resource']=select_shared_device()
        import torch
        from thunderworld_fp4 import fp4_attention,RepairConfig
        from thunderworld_fp4.cosmos.budget import attention_cost,PHASES
        from thunderworld_fp4.cosmos.calibration import panel_objective
        torch.set_num_threads(8)
        with torch.inference_mode():
            for layer,data in study['layers'].items():
                if layer in result['augmented_layers']:continue
                for intervention in data['interventions']:
                    if intervention['seed']!=7101:continue
                    phase=intervention['phase'];branch=intervention['branch']
                    table=intervention['measured_quota_losses'];nt=len(table[0])-1;h=len(table)
                    key=f'attention|{branch}|{phase}|{layer}'
                    group=result['groups'][key]
                    rows=[r for r in capture['rows'] if r['kind']=='attention' and r['name']==layer
                          and r['phase']==phase and r['branch']==branch]
                    if {r['seed'] for r in rows}!={7101,7102}:raise ValueError('Missing teacher pair')
                    for fraction in (.125,.25,.5):
                        quotas=allocate_head_quotas(table,h*math.ceil(nt*fraction))
                        setting=dict(qk_counts=list(quotas),pv_counts=list(quotas),qk_terms=3,pv_terms=3)
                        if any(all(c[k]==v for k,v in setting.items()) for c in group):continue
                        cfg=RepairConfig(coverage=0.,head_quotas=quotas,terms=3,sparse_pack=True,query_tile=32,transpose_v=True)
                        errors=[];costs=[]
                        for row in rows:
                            if sha(a.capture/row['file'])!=row['sha256']:raise ValueError('Corrupt panel')
                            panel=torch.load(a.capture/row['file'],weights_only=True)
                            q,k,v=(panel[x].cuda() for x in ('q','k','v'));gold=panel['y'].cuda().float()
                            pred=fp4_attention(q,k,v,cfg).float()
                            errors.append(float((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20)))
                            cost=attention_cost(q.shape[1],k.shape[1],q.shape[2],k.shape[2],**setting)
                            costs.append(cost)
                        if costs[0]['total_mma']!=costs[1]['total_mma']:raise ValueError('Teacher padding costs differ')
                        calls=PHASES[phase+1]-PHASES[phase]
                        group.append(dict(id=max(c['id'] for c in group)+1,kind='attention',name=layer,phase=phase,branch=branch,
                            loss=panel_objective(errors)*calls,mean_nmse=sum(errors)/len(errors),
                            mma=cost['total_mma']*calls,base_mma=cost['base_mma']*calls,**setting,
                            proposal='integer_head_marginal',panel_nmse=errors,proposal_seed=7101))
                result['augmented_layers'].append(layer);write(target,result)
                print(json.dumps(dict(layer=layer,complete=True)),flush=True)
            result['complete']=True;write(target,result)


if __name__=='__main__':main()

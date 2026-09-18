"""Head-only coordinate repair: preserve every linear support and QK/PV total.

Unlike a global proxy allocator, this intervention cannot move work between
attention and FFN. Three QK/PV coordinate passes at most; actual combined outputs
on both calibration seeds determine acceptance. All base blocks remain.
"""
import argparse,copy,json,math,sys
from pathlib import Path
from baselines import sha,write
from head_allocation import allocate_head_quotas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','search','controls','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=a.resume)
    capture=json.loads((a.capture/'manifest.json').read_text());search=json.loads(a.search.read_text())
    control=json.loads(a.controls.read_text())
    selected=[row for row in control['candidates'] if row['name'].endswith('_grow_mixed')]
    allocations={str(row['budget']):json.loads((Path(row['path'])/'allocation.json').read_text()) for row in selected}
    identity=dict(capture=sha(a.capture/'manifest.json'),search=sha(a.search),controls=sha(a.controls),source=sha(__file__))
    target=a.output/'interventions.json';result=dict(complete=False,identity=identity,rows={},performance_eligible=False)
    if target.exists():
        result=json.loads(target.read_text())
        if result['identity']!=identity:raise ValueError('Resume inputs changed')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        result['resource']=select_shared_device()
        import torch
        from thunderworld_fp4 import RepairConfig,fp4_attention
        from thunderworld_fp4.cosmos.calibration import panel_objective
        from thunderworld_fp4.cosmos.budget import PHASES,attention_cost
        torch.set_num_threads(8)
        with torch.inference_mode():
            for layer in sorted(search['attention'],key=int):
                for phase in range(4):
                    for branch in range(2):
                        key=f'attention|{branch}|{phase}|{layer}'
                        panels=[]
                        for row in sorted((r for r in capture['rows'] if r['kind']=='attention' and r['name']==layer and r['phase']==phase and r['branch']==branch),key=lambda r:r['seed']):
                            if sha(a.capture/row['file'])!=row['sha256']:raise ValueError('Panel changed')
                            panels.append({k:v.cuda() for k,v in torch.load(a.capture/row['file'],weights_only=True).items()})
                        if len(panels)!=2:raise ValueError('Two teacher seeds required')
                        q,k,v=(panels[0][s] for s in ('q','k','v'));h=q.shape[2];nt=math.ceil(k.shape[1]/128)
                        def evaluate(setting,panel=panels[0]):
                            cfg=RepairConfig(coverage=0.,head_quotas=tuple(setting['qk_counts']),pv_head_quotas=tuple(setting['pv_counts']),
                                terms=setting['qk_terms'],pv_terms=setting['pv_terms'],query_tile=32,transpose_v=True,sparse_pack=True)
                            pred=fp4_attention(panel['q'],panel['k'],panel['v'],cfg).float();gold=panel['y'].float()
                            return ((pred-gold).square().sum((0,1,3))/gold.square().sum().clamp_min(1e-20)).cpu()
                        for budget,allocation in allocations.items():
                            tag=f'{budget}|{key}';old=allocation['assignments'][key]
                            setting={s:old[s] for s in ('qk_counts','pv_counts','qk_terms','pv_terms')}
                            if tag not in result['rows']:
                                proposed=copy.deepcopy(setting);before=[float(evaluate(setting,panel).sum()) for panel in panels];rounds=0
                                for round_id in range(3):
                                    start=copy.deepcopy(proposed)
                                    for side in ('qk_counts','pv_counts'):
                                        total=sum(proposed[side])
                                        if total in (0,h*nt):continue
                                        table=[]
                                        for quota in range(nt+1):
                                            trial={**proposed,side:[quota]*h};table.append(evaluate(trial))
                                        loss=torch.stack(table,1).tolist()
                                        proposed[side]=list(allocate_head_quotas(loss,total))
                                    rounds=round_id+1
                                    if proposed==start:break
                                after=[float(evaluate(proposed,panel).sum()) for panel in panels]
                                accepted=panel_objective(after)<panel_objective(before)
                                final=proposed if accepted else setting
                                cost=attention_cost(q.shape[1],k.shape[1],h,k.shape[2],**final)
                                calls=PHASES[phase+1]-PHASES[phase]
                                if cost['total_mma']*calls!=old['mma']:raise ValueError('Head-only intervention changed MMA')
                                result['rows'][tag]=dict(setting=final,before=before,proposed=after,accepted=accepted,rounds=rounds)
                                write(target,result)
                            row=result['rows'][tag]
                            losses=row['proposed'] if row['accepted'] else row['before']
                            allocation['assignments'][key]=dict(old,**row['setting'])
                            allocation['assignments'][key].update(loss=panel_objective(losses)*(PHASES[phase+1]-PHASES[phase]),
                                mean_nmse=sum(losses)/2,proposal='fixed_total_head_coordinate')
                print(json.dumps(dict(layer=layer,completed=True)),flush=True)
        # Export with the existing asset/support compiler. No new W0 or W1 is fit.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'joint_inference'))
        from export_budget import write_candidate
        candidates=[]
        for budget,allocation in allocations.items():
            allocation['proxy_loss']=sum(row['loss'] for row in allocation['assignments'].values())
            folder=a.output/f'b{float(budget):g}_head_coordinate'
            if not folder.exists():write_candidate(folder,search,allocation,a.search.parent)
            if not json.loads((folder/'assets/manifest.json').read_text())['complete']:raise ValueError('Incomplete export')
            candidates.append(dict(name=folder.name,budget=float(budget),ratio=allocation['ratio'],path=str(folder.resolve())))
        result['complete']=True;write(target,result)
        write(a.output/'candidates.json',dict(complete=True,candidates=candidates,
            scope='Fixed linear supports and per-layer/phase/branch QK/PV total MMA; only head quotas change. Calibration proposals, not generation-qualified.'))


if __name__=='__main__':main()

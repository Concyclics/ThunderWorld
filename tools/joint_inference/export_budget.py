"""Compile measured local search into exact per-request plans and compact offline assets."""
import argparse,hashlib,json
from copy import deepcopy
from pathlib import Path
from thunderworld_fp4.cosmos.allocation import allocate
from thunderworld_fp4.cosmos.budget import RepairPlan,linear_cost,attention_cost,compile_request,PHASES
from thunderworld_fp4.cosmos.settings import RunConfig,LinearSpec
from capture_budget import atomic_json


def write_candidate(folder,search,allocation,source):
    import torch
    torch.set_num_threads(8)
    from thunderworld_fp4.cosmos.assets import expand,compact
    folder.mkdir(parents=True,exist_ok=False)
    plan=dict(schema_version=1,task=search.get('task','i2v'),steps=35,phases=list(PHASES),budget=allocation['budget'],
              model_identity=search['model_identity'],cohort_sha256=search['cohort_sha256'],linears={},attention={})
    rows=[]
    for row in allocation['assignments'].values():
        key=f'{row["branch"]}|{row["phase"]}|{row["name"]}'
        if row['kind']=='linear':
            setting={k:row[k] for k in ('a_blocks','w_blocks','cross')}
            plan['linears'][key]=setting
            shape=search['modules'][row['name']]['shape']
            cost=linear_cost(*shape,**setting)
        else:
            setting={k:row[k] for k in ('qk_counts','pv_counts','qk_terms','pv_terms')}
            plan['attention'][key]=setting
            cost=attention_cost(*search['attention'][row['name']]['shape'],**setting)
        rows.append({**cost,'name':row['name'],'branch':row['branch'],'phase':row['phase']})
    RepairPlan(plan)
    ledger=compile_request(rows,allocation['budget'],branches=search.get('branches',2),expected_modules=list(search['modules']),expected_layers=list(search['attention']))
    atomic_json(folder/'plan.json',plan);atomic_json(folder/'allocation.json',allocation);atomic_json(folder/'compute.json',ledger)
    bank=folder/'assets';bank.mkdir()
    manifest=dict(complete=False,precision='fp4',model_identity=search['model_identity'],modules={},
                  cohort_sha256=search['cohort_sha256'],plan_sha256=RepairPlan(plan).fingerprint,performance_eligible=False)
    selectors={}
    for name,entry in search['modules'].items():
        asset=expand(torch.load(source/entry['file'],weights_only=True,map_location='cpu'),1.)
        selectors[name]=asset.get('selector','activation_energy')
        union=sorted({i for k,r in plan['linears'].items() if k.split('|',2)[-1]==name for i in r['w_blocks']})
        path=bank/entry['file'];torch.save(compact(asset,union),path)
        manifest['modules'][name]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest['complete']=True;atomic_json(bank/'manifest.json',manifest)
    for row in rows:
        if row['kind']=='linear':
            sparse=selectors[row['name']]=='residual_sensitivity' and 0<row['a_blocks']<row['shape'][2]//128
            row['packing']['implementation']='selected_codes' if sparse else 'legacy_full_plane'
            row['packing']['residual_write_bytes']=row['packing']['selected_residual_write_bytes'] if sparse else row['packing']['residual_allocation_bytes']
            if sparse:row['non_mma']['base_reencoded_elements']=row['non_mma']['selected_residual_elements']
        else:row['packing_implementation']='gqa_union_selected_codes'
    ledger=compile_request(rows,allocation['budget'],branches=search.get('branches',2),expected_modules=list(search['modules']),expected_layers=list(search['attention']))
    atomic_json(folder/'compute.json',ledger)
    config=RunConfig(resource_mode='shared-calibration',compute_budget=allocation['budget'],budget_plan=str((folder/'plan.json').resolve()),
                     ffn=LinearSpec(0,0),projection=LinearSpec(0,0),share_pack=False)
    atomic_json(folder/'config.json',config.to_dict())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('search',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();search=json.loads(a.search.read_text())
    if not search['complete']:raise ValueError('Complete all-layer search required')
    a.output.mkdir(parents=True,exist_ok=False);candidates=[]
    for budget in (1.,1.125,1.25,1.5):
        unique=set()
        for direction,preference in (('grow',None),('prune',None),('grow','a'),('grow','w')):
            allocation=allocate(search['groups'],budget,direction=direction,preference=preference)
            signature=tuple((k,c['id']) for k,c in sorted(allocation['assignments'].items()))
            if signature in unique:continue
            unique.add(signature)
            name=f'b{budget:g}_{direction}_{preference or "mixed"}'
            folder=a.output/name;write_candidate(folder,search,allocation,a.search.parent)
            candidates.append(dict(name=name,budget=budget,path=str(folder.resolve()),ratio=allocation['ratio']))
    atomic_json(a.output/'candidates.json',dict(complete=True,candidates=candidates,cohort_sha256=search['cohort_sha256'],
        validation_limit_per_budget=3,scope='Local search proposals; development rollout and independent validation still required'))


if __name__=='__main__':main()

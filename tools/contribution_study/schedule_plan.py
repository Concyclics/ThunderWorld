"""Joint stage/CFG/layer/head allocation with frozen weight representations."""
import argparse
import collections
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from baselines import sha,write
from schedule_common import CONTROLS,KEY_TILES,cost_choice
from schedule_envelope import KEY_PAIRS,compile_envelope
from schedule_allocate import allocate
from thunderworld_fp4.cosmos.budget import PHASES


def objective(values,benchmarks):
    return sum(np.mean([v for v,b in zip(values,benchmarks) if b==dataset]) for dataset in ('rbench','paibench'))/2+.25*max(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','previous'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--budgets',nargs='+',type=float,default=[1.5,2.,2.5]);a=p.parse_args()
    cohort=json.loads((a.previous/'cohort.json').read_text());benchmarks=[c['benchmark'] for c in cohort['cases']['calibration']]
    out=a.study/'candidates';out.mkdir(exist_ok=True,parents=True);catalog=[]
    for cap in a.budgets:
        root=a.study/'calibration'/f'b{cap:g}'
        manifest=json.loads((root/'manifest.json').read_text());definitions=json.loads((root/'definitions.json').read_text())
        if not manifest['complete'] or manifest['identity']['max_cases'] is not None:raise ValueError('Full calibration required')
        if manifest['identity']['cohort_sha256']!=sha(a.previous/'cohort.json') or manifest['definitions_sha256']!=sha(root/'definitions.json'):raise ValueError('Changed calibration')
        controls=definitions['control'];old=json.loads(Path(controls['config']['budget_plan']).read_text());shapes=definitions['shapes']
        samples=[]
        for entry in cohort['cases']['calibration']:
            file=root/(entry['case']['id']+'.json')
            if sha(file)!=manifest['cases'][entry['case']['id']]:raise ValueError('Changed repair interventions')
            samples.append(json.loads(file.read_text())['groups'])
        baselines=[compile_envelope(old,shapes,pair) for pair in KEY_PAIRS]
        caps=np.asarray([r['base_mma']*cap for r in baselines]);base=np.asarray([r['base_mma'] for r in baselines])
        attention_base=np.asarray([sum(r['base_mma']*r['calls'] for r in b['rows'] if r['kind']=='attention') for b in baselines])
        attention_old=np.asarray([sum(r['total_mma']*r['calls'] for r in b['rows'] if r['kind']=='attention') for b in baselines])
        catalog.append(controls)
        for profile,movement,floor in [('phase',.15,.95),('joint',.15,.9),('relaxed',.3,.8)]:
            folder=out/f'b{cap:g}_schedule_{profile}'
            if folder.exists():
                candidate=json.loads((folder/'candidate.json').read_text())
                if candidate['calibration_sha256']!=sha(root/'manifest.json'):raise ValueError('Immutable schedule changed')
                catalog.append(candidate);continue
            groups=collections.defaultdict(list)
            for key,group in definitions['groups'].items():
                if profile=='phase':
                    family=group['name'].split('.')[-1] if group['kind']=='projection' else group['kind']
                    target=f'{group["kind"]}|{group["branch"]}|{group["phase"]}|{family}'
                else:target=key
                groups[target].append(key)
            options={}
            for target,keys in sorted(groups.items()):
                first=definitions['groups'][keys[0]]
                labels=set(first['choices'])
                for key in keys:labels &= set(definitions['groups'][key]['choices'])
                if profile=='phase':labels={s for s in labels if 'heads' not in s}
                rows=[];calls=PHASES[first['phase']+1]-PHASES[first['phase']]
                for label in sorted(labels,key=lambda x:(x!='base',x)):
                    losses=[sum(sample[key]['losses'][label] for key in keys) for sample in samples]
                    costs=[]
                    one_branch={nt:sum(cost_choice(definitions['groups'][key]['kind'],definitions['groups'][key]['choices'][label],definitions['groups'][key]['shapes'],nt)['total_mma'] for key in keys)*calls for nt in KEY_TILES}
                    costs=[one_branch[pair[first['branch']]] for pair in KEY_PAIRS]
                    rows.append(dict(label=label,kind=first['kind'],loss=float(objective(losses,benchmarks))*calls,
                                     costs=costs,members=keys))
                options[target]=rows
            initial=None
            if profile!='phase':
                seed_profile='phase' if profile=='joint' else 'joint'
                seed=json.loads((out/f'b{cap:g}_schedule_{seed_profile}'/'allocation.json').read_text())
                initial={key:row['label'] for row in seed['assignments'].values() for key in row['members']}
            allocation=allocate(options,caps,change_limit=movement*base,initial=initial,
                                attention_floor=attention_base+floor*(attention_old-attention_base),rounds=3)
            value=copy.deepcopy(old)
            changed=collections.Counter()
            for selection in allocation['assignments'].values():
                for key in selection['members']:
                    group=definitions['groups'][key];label=selection['label'];prefix=f'{group["branch"]}|{group["phase"]}|'
                    if group['kind']=='attention':value['attention'][prefix+group['name']]=copy.deepcopy(group['choices'][label]['attention'])
                    else:
                        for name,setting in group['choices'][label].items():value['linears'][prefix+name]=copy.deepcopy(setting)
                    if group['choices'][label]!=group['choices']['base']:changed[group['kind']]+=1
            value['design']=dict(schedule_profile=profile,calibration_sha256=sha(root/'manifest.json'),
                                 source_plan_sha256=controls['plan_sha256'],frozen_w0=True,same_w1_bank_cap=True,
                                 phase_ranges=list(PHASES),group_cost_change_limit=movement,attention_repair_floor=floor,
                                 attention_key_tokens_planned_max=2048,student_current_inputs=True)
            ledgers=[compile_envelope(value,shapes,pair) for pair in KEY_PAIRS]
            folder.mkdir();write(folder/'plan.json',value)
            # Retain the N961 ledger for generic CPU tools; every shape is also
            # verified individually and actual warmup recompiles the request.
            write(folder/'compute.json',ledgers[0]);write(folder/'bounds.json',dict(complete=True,checks=[dict(key_tiles_by_cfg=list(pair),ratio=l['ratio']) for pair,l in zip(KEY_PAIRS,ledgers)]))
            write(folder/'allocation.json',dict(allocation,changed_groups=dict(changed),calibration_sha256=sha(root/'manifest.json'),
                 scope='Local current-input combinations are proposals; no full-trajectory guarantee. Same frozen weight-bank cap; all base attention tiles retained.'))
            import torch
            from thunderworld_fp4.cosmos.assets import expand,compact
            torch.set_num_threads(8)
            source_bank=Path(controls['assets']);am=json.loads((source_bank/'manifest.json').read_text());bank=folder/'assets';bank.mkdir()
            packed_manifest=dict(complete=False,precision='fp4',model_identity=am['model_identity'],modules={});cost=collections.Counter()
            for name,entry in am['modules'].items():
                union=sorted({i for key,row in value['linears'].items() if key.split('|',2)[-1]==name for i in row['w_blocks']})
                file=source_bank/entry['file']
                if sha(file)!=entry['sha256']:raise ValueError('Changed source asset')
                raw=torch.load(file,map_location='cpu',weights_only=True)
                if not set(union).issubset(raw['residual_indices'].tolist()):raise ValueError('Schedule expanded W1 bank')
                target=bank/entry['file']
                if union==raw['residual_indices'].tolist():shutil.copyfile(file,target);asset=raw
                else:asset=compact(expand(raw,union),union);torch.save(asset,target)
                packed_manifest['modules'][name]=dict(file=target.name,sha256=sha(target))
                cost['w1_bytes']+=len(union)*128*128*9//16;cost['file_bytes']+=target.stat().st_size
                cost['tensor_bytes']+=sum(v.numel()*v.element_size() for v in asset.values() if isinstance(v,torch.Tensor))
            if cost['w1_bytes']>controls['cost']['w1_bytes']:raise ValueError('W1 storage cap exceeded')
            packed_manifest['complete']=True;write(bank/'manifest.json',packed_manifest)
            candidate=dict(name=folder.name,budget=cap,config=dict(controls['config'],budget_plan=str((folder/'plan.json').resolve())),
                assets=str(bank.resolve()),assets_sha256=sha(bank/'manifest.json'),plan_sha256=sha(folder/'plan.json'),
                calibration_sha256=sha(root/'manifest.json'),cost=dict(cost),bound_ratio=max(l['ratio'] for l in ledgers))
            write(folder/'candidate.json',candidate);catalog.append(candidate)
            print(json.dumps(dict(name=candidate['name'],ratio=candidate['bound_ratio'],changed=dict(changed),
                                  local_improvement=1-allocation['proxy_loss']/allocation['control_proxy_loss'])),flush=True)
    write(out/'candidates.json',dict(complete=True,candidates=catalog,cohort_sha256=sha(a.previous/'cohort.json')))


if __name__=='__main__':main()

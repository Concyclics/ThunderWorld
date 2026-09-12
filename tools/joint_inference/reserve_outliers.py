"""Reserve an already stored high-gain W1 block in every phase, inside the same MMA cap.

This is a proposal only. Low-ranked tail W1 blocks pay for the reservation; the
subsequent native combination refinement and full trajectories must accept it.
"""
import argparse,json
from pathlib import Path
from copy import deepcopy
from thunderworld_fp4.cosmos.budget import linear_cost,attention_cost,PHASES
from export_budget import write_candidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--search',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();search=json.loads(a.search.read_text());allocation=json.loads((a.candidate/'allocation.json').read_text())
    import torch
    torch.set_num_threads(8)
    assignments=allocation['assignments'];required={};scores={}
    for name,entry in search['modules'].items():
        used=set(i for row in assignments.values() if row['kind']=='linear' and row['name']==name for i in row['w_blocks'])
        if not used:continue
        asset=torch.load(a.search.parent/entry['file'],map_location='cpu',weights_only=True)
        order=entry['support_order'];required[name]=next(i for i in order if i in used)
        scores[name]=asset['score'].flatten().tolist()
    for row in assignments.values():
        if row['kind']=='linear' and row['name'] in required:
            row['w_blocks']=sorted(set(row['w_blocks'])|{required[row['name']]})
    def row_cost(row):
        if row['kind']=='linear':
            c=linear_cost(*search['modules'][row['name']]['shape'],a_blocks=row['a_blocks'],w_blocks=row['w_blocks'],cross=row['cross'])
        else:
            c=attention_cost(*search['attention'][row['name']]['shape'],**{k:row[k] for k in ('qk_counts','pv_counts','qk_terms','pv_terms')})
        return c['total_mma']*(PHASES[row['phase']+1]-PHASES[row['phase']])
    total=sum(row_cost(r) for r in assignments.values());base=allocation['base_mma'];cap=allocation['budget']*base
    trimmed=[]
    while total>cap:
        proposals=[]
        for key,row in assignments.items():
            if row['kind']!='linear' or row['name'] not in required:continue
            ids=[i for i in row['w_blocks'] if i!=required[row['name']]]
            if not ids:continue
            # Remove a tail of the measured nested order, never the mandatory core.
            index=min(ids,key=lambda i:(scores[row['name']][i],-i))
            trial={**row,'w_blocks':[i for i in row['w_blocks'] if i!=index]}
            saved=row_cost(row)-row_cost(trial)
            if saved>0:proposals.append((scores[row['name']][index]/saved,key,index,saved,trial))
        if not proposals:raise ValueError('Cannot fund mandatory reservation from existing W1 supports')
        _,key,index,saved,trial=min(proposals,key=lambda r:(r[0],r[1],r[2]))
        assignments[key]=trial;total-=saved;trimmed.append(dict(scope=key,block=index))
    allocation.update(total_mma=total,ratio=total/base,reserved_w1_core=required,
        trimmed_for_reservation=trimmed,scope='Mandatory W1 reservation proposal; native combination refinement and trajectory acceptance required')
    write_candidate(a.output,search,allocation,a.search.parent)


if __name__=='__main__':main()

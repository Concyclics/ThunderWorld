"""Independent conditional/unconditional key lengths: 9 x 9 shape envelope."""
from itertools import product
from thunderworld_fp4 import RepairConfig
from thunderworld_fp4.cosmos.budget import RepairPlan,linear_cost,attention_cost,compile_request
from schedule_common import KEY_TILES

KEY_PAIRS=tuple(product(KEY_TILES,repeat=2))


def compile_envelope(value,shapes,key_pair):
    rows=[];plan=RepairPlan(value)
    for key,setting in value['linears'].items():
        branch,ph,name=key.split('|',2)
        rows.append(dict(linear_cost(*shapes['linears'][name],**setting),branch=int(branch),phase=int(ph),name=name))
    for key in value['attention']:
        branch,ph,layer=key.split('|',2);m,_,h,hk=shapes['attention'][layer]
        n=max(961,(key_pair[int(branch)]-1)*128+1)
        cfg=plan.attention_config('i2v',int(branch),int(ph),int(layer),RepairConfig())
        rows.append(dict(attention_cost(m,n,h,hk,qk_counts=cfg.head_counts(h,n),pv_counts=cfg.pv_counts(h,n),
             qk_terms=cfg.terms,pv_terms=cfg.pv_terms or cfg.terms),branch=int(branch),phase=int(ph),name=layer))
    return compile_request(rows,value['budget'],expected_modules=list(shapes['linears']),expected_layers=list(shapes['attention']))

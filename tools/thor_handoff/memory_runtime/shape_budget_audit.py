"""CPU forecast from observed original shapes, checked against new-shape ledgers.

This audit never authorizes a launch: the ordinary warmup compiler remains the
budget gate. Prompt token counts are held fixed with resolution templates off.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=Path(__file__).resolve().parents[3]/'packages/fp4-repair/src/thunderworld_fp4/cosmos/budget.py'
    spec=importlib.util.spec_from_file_location('cost_only',source)
    cost=importlib.util.module_from_spec(spec);spec.loader.exec_module(cost)
    shapes=[(320,192,61),(320,192,125),(320,192,189),(480,288,61),(640,384,61)]
    rows=[]
    for model in ('edge','nano'):
        for method in ('b1.5_protected_n','b2_joint'):
            paths=list((a.reference/'runs').glob(f'*-{model}-compact-{method}-320x192x61/result.json'))
            if len(paths)!=1:raise ValueError('Need a unique complete original-shape reference')
            reference=json.loads(paths[0].read_text())
            if not reference['complete']:raise ValueError('Reference is incomplete')
            ledger=reference['rows'][0]['compute_ledger']
            for width,height,frames in shapes:
                m=(width//32)*(height//32)*((frames-1)//4+1)
                total=base=0;by_kind={}
                for before in ledger['rows']:
                    if before['shape'][0]!=960:raise ValueError('Unexpected original DM length')
                    if before['kind']=='linear':
                        _,n,k=before['shape']
                        after=cost.linear_cost(m,n,k,bm=before['launch']['bm'],
                            a_blocks=before['a_blocks'],w_blocks=before['w_blocks'],
                            cross=bool(before['mma_by_dtype']['fp4']['cross']))
                    else:
                        old_m,old_n,h,hk,d=before['shape'];n=m+old_n-old_m
                        counts={key:(before[key] if method=='b1.5_protected_n' else [cost.ceildiv(n,128)]*h)
                                for key in ('qk_counts','pv_counts')}
                        terms={key+'_terms':(4 if before['mma_by_dtype']['fp4'][key+'_cross'] else
                                           3 if before['mma_by_dtype']['fp4'][key+'_w'] else 2)
                               for key in ('qk','pv')}
                        after=cost.attention_cost(m,n,h,hk,bm=before['launch']['bm'],**counts,**terms)
                    calls=before['calls'];base+=after['base_mma']*calls;total+=after['total_mma']*calls
                    subtotal=by_kind.setdefault(before['kind'],dict(base=0,total=0))
                    subtotal['base']+=after['base_mma']*calls;subtotal['total']+=after['total_mma']*calls
                row=dict(model=model,method=method,shape=[width,height,frames],vision_tokens=m,
                    budget=ledger['budget'],base_mma=base,total_mma=total,ratio=total/base,
                    feasible=total/base<=ledger['budget'],by_kind=by_kind,
                    original_receipt_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())
                if [width,height,frames]==[320,192,61]:
                    if (base,total)!=(ledger['base_mma'],ledger['total_mma']):
                        raise ValueError('Forecast failed to reconstruct the observed reference')
                if a.campaign:
                    actual=list((a.campaign/'runs').glob(f'*-{model}-compact-{method}-{width}x{height}x{frames}/result.json'))
                    if actual:
                        data=json.loads(actual[0].read_text())
                        observed=None
                        if data['complete']:
                            current=data['rows'][0]['compute_ledger']
                            observed=(current['base_mma'],current['total_mma'])
                        elif data['status']=='infeasible_budget':
                            match=re.search(r'Compute budget exceeded: (\d+)/(\d+)',data['error'])
                            if not match:raise ValueError('Unrecognized budget rejection')
                            observed=(int(match[2]),int(match[1]))
                        if observed:
                            if observed!=(base,total):raise ValueError('Forecast differs from actual warmup: '+str(row))
                            row['warmup_verified']=True
                            row['observed_status']=data['status']
                rows.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(rows=rows,
        budget_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Analytic forecast, separately matched to actual warmup if present; not a substitute for launch qualification'),indent=2)+'\n')
    for row in rows:
        print(row['model'],row['method'],row['shape'],f"{row['ratio']:.9f}",row['feasible'],
              'verified' if row.get('warmup_verified') else 'forecast')


if __name__=='__main__':main()
